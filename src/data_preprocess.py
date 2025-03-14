import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import streamlit as st
from geopy.geocoders import Nominatim
from imblearn.over_sampling import SMOTENC,SMOTE
import pickle
import os


def get_lat_lon(city):
    geolocator = Nominatim(user_agent="geo_locator")
    try:
        location = geolocator.geocode(city)
        return (location.latitude, location.longitude)
    except:
        return (None, None)

def preprocess_data(data, eda_insights, label_encode=True, verbose=True):
    
    # 1. drop duplicated rows if any
    initial_rows = len(data)
    data.drop_duplicates(inplace=True)
    if verbose:
        st.write(f"Removed {initial_rows - len(data)} duplicate rows.")
    
    # 2. tackle outliers
    if eda_insights['outliers']['vibration']:
        data['vibration'] = data['vibration'].clip(lower=0)  # 负值设为 0
        if verbose:
            st.write("Clipped negative vibration values to 0.")
    # if eda_insights['outliers']['temperature']:
    #     data['temperature'] = data['temperature'].clip(upper=140)  # 高温截断
    #     st.write("Clipped temperature values above 140.")

    # 3. skewness transformation
    if 'vibration' in eda_insights['skewed_features']:
        data['vibration_log'] = np.log1p(data['vibration'])  
        if verbose:
            st.write("Applied log transformation to vibration due to skewness.")

    # 5. add latitude and longitude based on location
    if "latitude" not in data.columns:
        locations = data["location"].unique()
        location_dict = {location: get_lat_lon(location) for location in locations}
        location_df = pd.DataFrame(location_dict).T.reset_index()
        location_df.columns = ["location", "latitude", "longitude"]
        data = data.merge(location_df, on="location", how="left")
    if verbose:
        st.write("Added latitude and longitude based on location.")

    # 6. handle imbalanced data
    if eda_insights['imbalanced']:
        categorical_features = ["equipment", "location"]
        categorical_indices = [data.columns.get_loc(col) for col in categorical_features]
        le = {col: LabelEncoder() for col in categorical_features}
        for col in categorical_features:
            data[col] = le[col].fit_transform(data[col])
        X = data[["temperature", "pressure", "humidity", "vibration", "equipment", "location", "vibration_log", "latitude", "longitude"]]
        y = data["faulty"] 
        smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)
        X_resampled, y_resampled = smote_nc.fit_resample(X, y)
        for col in categorical_features:
            X_resampled[col] = le[col].inverse_transform(X_resampled[col])
        data = pd.concat([X_resampled, y_resampled], axis=1)
        if verbose:
            st.write("Balanced the data using SMOTENC.")

    
    # 7. label encoding
    if label_encode:
        if os.path.exists('./models/utils/le_equipment.pkl'):
            le_equipment = pickle.load(open('./models/utils/le_equipment.pkl', 'rb'))
            data["le_equipment"] = le_equipment.transform(data["equipment"])
        else:
            le_equipment = LabelEncoder()
            data["le_equipment"] = le_equipment.fit_transform(data["equipment"])
            pickle.dump(le_equipment, open('./models/utils/le_equipment.pkl', 'wb'))
        if verbose:
            st.write("Encoded equipment labels for modeling.")

    
    # 8. feature standardization
    features = ['temperature', 'pressure', 'vibration', 'humidity', 'latitude', 'longitude', 'vibration_log']
    if os.path.exists('./models/utils/scaler.pkl'):
        scaler = pickle.load(open('./models/utils/scaler.pkl', 'rb'))
        data[features] = scaler.transform(data[features])
    else:
        scaler = StandardScaler()
        data[features] = scaler.fit_transform(data[features])
        pickle.dump(scaler, open('./models/utils/scaler.pkl', 'wb'))
    if verbose:
        st.write("Standardized numerical features for model compatibility.")

    
    return data

