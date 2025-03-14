import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.data_preprocess import preprocess_data
from src.eda import get_eda_insights, get_description
from src.train import train_model_for_equipment, train_model_for_faulty, make_prediction

st.title("Equipment Anomaly Detection - Data Analysis & Prediction")
st.markdown("This is a Streamlit app that shows the equipment anomaly detection dataset analysis. For more information, please refer to readme.pdf.")

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

data = load_data(f"data/raw/equipment_anomaly_data.csv")
get_description(data)


# sidebar
st.sidebar.header("Filter Data")
pressure_range = st.sidebar.slider("Pressure Range", 0.0, 80.0, (0.0, 80.0))
temp_range = st.sidebar.slider("Temperature Range", 0.0, 150.0, (0.0, 150.0))
humidity_range = st.sidebar.slider("Humidity Range", 10.0, 90.0, (10.0, 90.0))
vibration_range = st.sidebar.slider("Vibration Range", -1.0, 5.0, (-1.0, 5.0))

data = data[(data['temperature'] >= temp_range[0]) & (data['temperature'] <= temp_range[1])]
data = data[(data['pressure'] >= pressure_range[0]) & (data['pressure'] <= pressure_range[1])]
data = data[(data['humidity'] >= humidity_range[0]) & (data['humidity'] <= humidity_range[1])]
data = data[(data['vibration'] >= vibration_range[0]) & (data['vibration'] <= vibration_range[1])]



tab1, tab2 = st.tabs(["EDA", "Results Prediction"])

with tab1:
    eda_insights = get_eda_insights(data)
    # '''
    # key_features的作用，但是好像在这个数据集上并不适用
    # 	  1.	降维（减少特征冗余，提高计算效率）
    #     2.	特征工程（标准化、非线性变换）
    #     3.	异常值处理（删除、转换）
    #     4.	模型优化（重点调整类别不平衡、数据增强）
    # '''
    if st.button("Preprocess the data", type="primary"):
        with st.status("Preprocessing the data..."):
            processed_data = preprocess_data(data.copy(), eda_insights)
            if not os.path.exists('./data/processed'):
                os.makedirs('./data/processed')   
            processed_data.to_csv(r'./data/processed/equipment_anomaly_data.csv', index=False)
            st.write("Processed data saved to '../data/processed/equipment_anomaly_data.csv'.")
            
            st.dataframe(processed_data)
            st.write(processed_data.describe())
            st.write("Data preprocessing is completed.")

with tab2:
    st.write("For now only LGBM model is supported.")
    equipment_model = st.selectbox("Select Model for Equipment", ["lgbm"])
    model = st.selectbox("Select Model for Faulty", ["lgbm"])
    st.subheader("Model Prediction Results")
    st.markdown("If the model has been trained, then you can view the prediction results without waiting for the model to train again.")
    if st.button("View Model Results"):
        st.write("### Results for Equipment Prediction")
        with st.spinner("Training the model for equipment..."):
            train_model_for_equipment("./data/processed/equipment_anomaly_data.csv", "./models/equipment", True, equipment_model)
        st.write("### Results for Faulty Prediction")
        with st.spinner("Training the model for fault..."):
            train_model_for_faulty("./data/processed/equipment_anomaly_data.csv", "./models/faulty", True, equipment_model, model)

    st.subheader("Prediction for New Data")
    temperature = st.text_input("Enter Temperature (possible range: [0.0, 150.0])", 70)
    pressure = st.text_input("Enter Pressure (possible range: [0.0, 80.0])", 35)
    humidity = st.text_input("Enter Humidity (possible range: [10.0, 90.0])", 20.26)
    vibration = st.text_input("Enter Vibration (possible range: [-1.0, 5.0])", 3)
    location = st.selectbox("Select Location", ["New York", "San Francisco", "Los Angeles", "Chicago", "Houston"])
    num_data = [temperature, pressure, humidity, vibration]

    if st.button("Predict"):
        with st.spinner("Inferencing..."):
            num_data = list(map(float, num_data))
            input_data = [num_data + [location]]
            input_data = pd.DataFrame(input_data, columns=["temperature", "pressure", "humidity", "vibration", "location"])
            print(input_data.info())
            equipment_predict, is_faulty, fault_prob = make_prediction(input_data.copy(), "./models", equipment_model, model)
            if is_faulty:
                st.write(f"The equipment may be {fault_prob}% faulty. You may want to check the {equipment_predict}.")
            else:
                st.write(f"The equipment may be not faulty. The fault probability is {fault_prob}%. You can check the {equipment_predict} if you want.")