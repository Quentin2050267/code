import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import streamlit as st
import pickle
import os
from typing import Literal
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data_preprocess import preprocess_data

def get_data_for_equipment(data_path):
    data = pd.read_csv(data_path)
    X = data[["temperature", "pressure", "humidity", "vibration", "vibration_log", "latitude", "longitude"]]
    y = data["le_equipment"]

    with st.container(border=True):
        st.write("Data trained for equipment:")
        st.dataframe(X)
        st.write("Target trained for equipment:")
        st.dataframe(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_data_for_faulty(data_path, model_path, model_name):
    data = pd.read_csv(data_path)
    X = data[["temperature", "pressure", "humidity", "vibration", "vibration_log", "latitude", "longitude"]]
    if os.path.exists(f"{model_path}/{model_name}.pkl"):
        grid = pickle.load(open(f"{model_path}/{model_name}.pkl", "rb"))
    else:
        st.write("Model for equipment has not been trained yet.")
        return
    data["le_equipment_predict"] = grid.predict(X)
    le_equipment = pickle.load(open('./models/utils/le_equipment.pkl', 'rb'))
    data["equipment_predict"] = le_equipment.inverse_transform(data["le_equipment_predict"])
    
    # One-hot encode the equipment_predict column
    if os.path.exists('./models/utils/oh_encoder.pkl'):
        oh_encoder = pickle.load(open('./models/utils/oh_encoder.pkl', 'rb'))
        equipment_predict_encoded = oh_encoder.transform(data[["equipment_predict"]])
    else:
        oh_encoder = OneHotEncoder()
        equipment_predict_encoded = oh_encoder.fit_transform(data[["equipment_predict"]])
        pickle.dump(oh_encoder, open('./models/utils/oh_encoder.pkl', 'wb'))
    data = pd.concat([data, pd.DataFrame(equipment_predict_encoded.toarray(), columns=oh_encoder.get_feature_names_out())], axis=1)
    st.write("One-hot encoded the equipment_predict column.")

    # Standardize the data
    if os.path.exists('./models/utils/oh_scaler.pkl'):
        scaler = pickle.load(open('./models/utils/oh_scaler.pkl', 'rb'))
        data[oh_encoder.get_feature_names_out()] = scaler.transform(data[oh_encoder.get_feature_names_out()])
    else:
        scaler = StandardScaler()
        data[oh_encoder.get_feature_names_out()] = scaler.fit_transform(data[oh_encoder.get_feature_names_out()])
        pickle.dump(scaler, open('./models/utils/oh_scaler.pkl', 'wb'))
    st.write("Standardized the data.")

    X = data.drop(["le_equipment", "faulty", "le_equipment_predict", "equipment_predict", "location", "equipment"], axis=1)
    y = data["faulty"]

    with st.container(border=True):
        st.write("Data trained for faulty:")
        st.dataframe(X)
        st.write("Target trained for faulty:")
        st.dataframe(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model_for_equipment(
        data_path, 
        model_path, 
        use_exisiting_model: bool = False,
        model_name: Literal["lgbm", "rf"] = "lgbm",
        **param_grid):
    X_train, X_test, y_train, y_test = get_data_for_equipment(data_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if os.path.exists(f"{model_path}/{model_name}.pkl") and use_exisiting_model:
        grid = pickle.load(open(f"{model_path}/{model_name}.pkl", "rb"))
    else:
        if model_name == "lgbm":
            if not param_grid:
                param_grid = {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [15, 31, 63],
                    'max_depth': [-1, 5, 10],
                    'min_child_samples': [10, 20, 30],
                    'n_estimators': [100, 300, 500],
                    "objective": ["multiclass"],
                    "num_class": [3]
                }
            model = lgb.LGBMClassifier(verbose=-1)
        elif model_name == "rf":
            if not param_grid:
                param_grid = {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2'],
                }
            model = RandomForestClassifier()
        # ... other models
        grid = GridSearchCV(model, param_grid, cv=5)
        grid.fit(X_train, y_train)
        pickle.dump(grid, open(f"{model_path}/{model_name}.pkl", "wb"))
        grid = pickle.load(open(f"{model_path}/{model_name}.pkl", "rb"))

    y_pred = grid.predict(X_test)
    st.write("Model training for equipment is completed.")
    with st.container(border=True):
        st.write(f"Model has been saved to '{model_path}/{model_name}.pkl'.")                       
        st.write(f"Model: {model_name}")
        st.write(f"Best parameters: {grid.best_params_}")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")

        le_equipment = pickle.load(open('./models/utils/le_equipment.pkl', 'rb'))
        labels = le_equipment.classes_
        st.write("Label projection:")
        st.write({i: label for i, label in enumerate(labels)})
       
        st.write("Classification Report:")
        st.code(classification_report(y_test, y_pred), language="plaintext")
        print(classification_report(y_test, y_pred))

        st.write("Confusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        df_conf_matrix = pd.DataFrame(conf_matrix, index=labels, columns=labels)
        df_conf_matrix_reset = df_conf_matrix.reset_index().melt(id_vars="index")
        st.vega_lite_chart(df_conf_matrix_reset, {
            'width': 300,
            'height': 300,
            'mark': 'rect',
            'encoding': {
                'x': {'field': 'variable', 'type': 'ordinal', 'title': 'Predicted Label'},
                'y': {'field': 'index', 'type': 'ordinal', 'title': 'True Label'},
                'color': {'field': 'value', 'type': 'quantitative', 'scale': {'scheme': 'reds'}},
                'tooltip': [{'field': 'value', 'type': 'quantitative'}]
            }
        })
        
        st.write("Feature Importance:")
        feature_importance = pd.DataFrame(grid.best_estimator_.feature_importances_, index=X_train.columns, columns=["importance"])
        print(feature_importance)
        st.bar_chart(feature_importance)

def train_model_for_faulty(
        data_path,
        model_path,
        use_exisiting_model: bool = False,
        equipment_model_name: Literal["lgbm", "rf"] = "lgbm",
        model_name: Literal["lgbm", "rf"] = "lgbm",
        **param_grid):
    X_train, X_test, y_train, y_test = get_data_for_faulty(data_path, './models/equipment', equipment_model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if os.path.exists(f"{model_path}/{model_name}.pkl") and use_exisiting_model:
        grid = pickle.load(open(f"{model_path}/{model_name}.pkl", "rb"))
    else:
        if model_name == "lgbm":
            if not param_grid:
                param_grid = {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [15, 31, 63],
                    'max_depth': [-1, 5, 10],
                    'min_child_samples': [10, 20, 30],
                    'n_estimators': [100, 300, 500],
                    "objective": ["binary"],
                }
            model = lgb.LGBMClassifier(verbose=-1)
        elif model_name == "rf":
            if not param_grid:
                param_grid = {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2'],
                }
            model = RandomForestClassifier()
        # ... other models
        grid = GridSearchCV(model, param_grid, cv=5)
        grid.fit(X_train, y_train)
        pickle.dump(grid, open(f"{model_path}/{model_name}.pkl", "wb"))
        grid = pickle.load(open(f"{model_path}/{model_name}.pkl", "rb"))
    y_pred = grid.predict(X_test)
    st.write("Model training for equipment is completed.")

    with st.container(border=True):
        st.write(f"Model has been saved to '{model_path}/{model_name}.pkl'.")
        st.write(f"Model: {model_name}")
        st.write(f"Best parameters: {grid.best_params_}")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")

        st.write("Classification Report:")
        st.code(classification_report(y_test, y_pred), language="plaintext")

        st.write("Confusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)
        df_conf_matrix = pd.DataFrame(conf_matrix, index=["Normal", "Faulty"], columns=["Normal", "Faulty"])
        df_conf_matrix_reset = df_conf_matrix.reset_index().melt(id_vars="index")
        st.vega_lite_chart(df_conf_matrix_reset, {
            'width': 300,
            'height': 300,
            'mark': 'rect',
            'encoding': {
                'x': {'field': 'variable', 'type': 'ordinal', 'title': 'Predicted Label'},
                'y': {'field': 'index', 'type': 'ordinal', 'title': 'True Label'},
                'color': {'field': 'value', 'type': 'quantitative', 'scale': {'scheme': 'reds'}},
                'tooltip': [{'field': 'value', 'type': 'quantitative'}]
            }
        })



        st.write("Feature Importance:")
        feature_importance = pd.DataFrame(grid.best_estimator_.feature_importances_, index=X_train.columns, columns=["importance"])
        print(feature_importance)
        st.bar_chart(feature_importance)
        st.caption("Equipments do no help in predicting faulty status. 😢")

        st.write("### ROC Curve")
        scores = grid.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, scores[:, 1])
        roc_data = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
        diagonal_data = pd.DataFrame({"False Positive Rate": [0, 1], "True Positive Rate": [0, 1]})
        st.vega_lite_chart({
            "layer": [
                {
                    "mark": {"type": "line", "color": "blue"},
                    "encoding": {
                        "x": {"field": "False Positive Rate", "type": "quantitative"},
                        "y": {"field": "True Positive Rate", "type": "quantitative"}
                    },
                    "data": {"values": roc_data.to_dict(orient="records")}
                },
                {
                    "mark": {"type": "line", "color": "red", "strokeDash": [5, 5]},
                    "encoding": {
                        "x": {"field": "False Positive Rate", "type": "quantitative"},
                        "y": {"field": "True Positive Rate", "type": "quantitative"}
                    },
                    "data": {"values": diagonal_data.to_dict(orient="records")}
                }
            ]
        })

        # AUC Score
        auc_score = roc_auc_score(y_test, scores[:, 1])
        st.write(f"AUC Score: {auc_score}, almost perfect model.")


def make_prediction(input_data, model_path, equipment_model_name, model_name):
    grid_equipment = pickle.load(open(f"{model_path}/equipment/{equipment_model_name}.pkl", "rb"))
    grid_faulty = pickle.load(open(f"{model_path}/faulty/{model_name}.pkl", "rb"))
    insight = {
        'outliers': {'vibration': False},
        'skewed_features': ['vibration'],
        'imbalanced': False,
        # 'key_features': ['vibration'] 
    }
    st.write(input_data)
    data = preprocess_data(input_data, insight, label_encode=False, verbose=False)
    data.drop(["location"], axis=1, inplace=True)
    data["le_equipment_predict"] = grid_equipment.predict(data)
    le_equipment = pickle.load(open('./models/utils/le_equipment.pkl', 'rb'))
    data["equipment_predict"] = le_equipment.inverse_transform(data["le_equipment_predict"])
    oh_encoder = pickle.load(open('./models/utils/oh_encoder.pkl', 'rb'))
    equipment_predict_encoded = oh_encoder.transform(data[["equipment_predict"]])
    data = pd.concat([data, pd.DataFrame(equipment_predict_encoded.toarray(), columns=oh_encoder.get_feature_names_out())], axis=1)
    scaler = pickle.load(open('./models/utils/oh_scaler.pkl', 'rb'))
    data[oh_encoder.get_feature_names_out()] = scaler.transform(data[oh_encoder.get_feature_names_out()])
    X = data.drop(["le_equipment_predict", "equipment_predict"], axis=1)
    y = grid_faulty.predict(X)
    y_prob = grid_faulty.predict_proba(X)
    return data["equipment_predict"].values[0], y, y_prob[0][1]
