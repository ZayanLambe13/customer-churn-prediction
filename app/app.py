import streamlit as st
import pandas as pd
import joblib

# Load model
pipeline = joblib.load("models/churn_pipeline.pkl")

churn_prob = pipeline.predict_proba(input_df)[0][1]


st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn based on their details.")

# User Inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Feature Engineering
tenure_group = "0-12" if tenure <= 12 else "12-24" if tenure <= 24 else "24-48" if tenure <= 48 else "48+"
avg_monthly_spend = total_charges / tenure if tenure > 0 else 0

# Create input dataframe
input_data = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "avg_monthly_spend": [avg_monthly_spend],
    "tenure_group_12-24": [1 if tenure_group == "12-24" else 0],
    "tenure_group_24-48": [1 if tenure_group == "24-48" else 0],
    "tenure_group_48+": [1 if tenure_group == "48+" else 0],
    "Contract_One year": [1 if contract == "One year" else 0],
    "Contract_Two year": [1 if contract == "Two year" else 0],
    "InternetService_Fiber optic": [1 if internet_service == "Fiber optic" else 0],
    "InternetService_No": [1 if internet_service == "No" else 0]
})

# Align columns
expected_cols = scaler.feature_names_in_
input_data = input_data.reindex(columns=expected_cols, fill_value=0)

# Prediction
if st.button("Predict Churn"):
    scaled_input = scaler.transform(input_data)
    churn_prob = model.predict_proba(scaled_input)[0][1]

    if churn_prob > 0.5:
        st.error(f"⚠️ High Churn Risk ({churn_prob:.2%})")
    else:
        st.success(f"✅ Low Churn Risk ({churn_prob:.2%})")
