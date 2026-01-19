import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# Load pipeline
pipeline = joblib.load("../models/churn_pipeline.pkl")

st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn based on their details.")

# User Inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Create RAW input dataframe (NO MANUAL ENCODING)
input_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service
}])

# Prediction
if st.button("Predict Churn"):
    churn_prob = pipeline.predict_proba(input_df)[0][1]

    st.write(f"### 🔍 Churn Probability: **{churn_prob:.2%}**")

    if churn_prob > 0.5:
        st.error("⚠️ High Churn Risk")
    else:
        st.success("✅ Low Churn Risk")
