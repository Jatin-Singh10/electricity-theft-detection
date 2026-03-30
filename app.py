import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="wide"
)

# =========================
# LOAD SAVED FILES
# =========================
@st.cache_resource
def load_artifacts():
    model = load_model("electricity_theft_dnn.keras")
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, imputer, scaler, feature_columns

model, imputer, scaler, feature_columns = load_artifacts()

# =========================
# TITLE
# =========================
st.title("Electricity Theft Detection")
st.write("Upload a CSV file to detect suspicious electricity consumption patterns.")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # -------------------------
        # READ FILE
        # -------------------------
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head())

        # -------------------------
        # CLEAN COMMON TEXT NULLS
        # -------------------------
        data = data.replace(["None", "none", "NULL", "null", "NaN", "nan", ""], np.nan)

        # -------------------------
        # STORE OPTIONAL ORIGINAL ID
        # -------------------------
        original_cons_no = None
        if "CONS_NO" in data.columns:
            original_cons_no = data["CONS_NO"].copy()

        # -------------------------
        # REMOVE NON-FEATURE COLUMNS
        # -------------------------
        if "CONS_NO" in data.columns:
            data = data.drop("CONS_NO", axis=1)

        if "FLAG" in data.columns:
            data = data.drop("FLAG", axis=1)

        # -------------------------
        # ADD MISSING COLUMNS IF ANY
        # -------------------------
        missing_cols = [col for col in feature_columns if col not in data.columns]
        extra_cols = [col for col in data.columns if col not in feature_columns]

        if len(missing_cols) > 0:
            st.warning(
                f"{len(missing_cols)} expected columns are missing in uploaded file. "
                f"They will be added as missing values and handled by the imputer."
            )
            for col in missing_cols:
                data[col] = np.nan

        if len(extra_cols) > 0:
            st.info(
                f"{len(extra_cols)} extra columns found in uploaded file. "
                f"They will be ignored."
            )

        # -------------------------
        # KEEP EXACT TRAINING ORDER
        # -------------------------
        data = data[feature_columns]

        # -------------------------
        # CONVERT TO NUMERIC
        # -------------------------
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        # -------------------------
        # SHOW MISSING VALUE SUMMARY
        # -------------------------
        total_missing = data.isna().sum().sum()
        st.write(f"Total missing values before imputation: **{int(total_missing)}**")

        # -------------------------
        # PREPROCESS
        # -------------------------
        data_imputed = imputer.transform(data)
        data_scaled = scaler.transform(data_imputed)

        # -------------------------
        # PREDICT
        # -------------------------
        probabilities = model.predict(data_scaled).flatten()
        predictions = (probabilities > 0.5).astype(int)

        # -------------------------
        # RESULT TABLE
        # -------------------------
        result_df = pd.DataFrame()

        if original_cons_no is not None:
            result_df["CONS_NO"] = original_cons_no

        result_df["Theft_Probability"] = probabilities
        result_df["Prediction"] = predictions
        result_df["Prediction_Label"] = result_df["Prediction"].map({
            0: "Normal",
            1: "Suspicious / Theft"
        })

        st.success("Prediction completed successfully.")

        st.subheader("Prediction Results")
        st.dataframe(result_df.head(20))

        # -------------------------
        # SUMMARY METRICS
        # -------------------------
        normal_count = int((predictions == 0).sum())
        theft_count = int((predictions == 1).sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(result_df))
        col2.metric("Normal Consumers", normal_count)
        col3.metric("Suspicious Consumers", theft_count)

        # -------------------------
        # DOWNLOAD BUTTON
        # -------------------------
        csv_output = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction Results",
            data=csv_output,
            file_name="electricity_theft_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

else:
    st.info("Please upload a CSV file to start prediction.")
    
