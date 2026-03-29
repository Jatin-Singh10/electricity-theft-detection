import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load saved files
model = load_model("electricity_theft_dnn.keras")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Electricity Theft Detection")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "CONS_NO" in data.columns:
        data = data.drop("CONS_NO", axis=1)

    if "FLAG" in data.columns:
        data = data.drop("FLAG", axis=1)

    data = data[feature_columns]

    data_imputed = imputer.transform(data)
    data_scaled = scaler.transform(data_imputed)

    probs = model.predict(data_scaled).flatten()
    preds = (probs > 0.5).astype(int)

    data["Prediction"] = preds
    data["Probability"] = probs

    st.write(data.head())
