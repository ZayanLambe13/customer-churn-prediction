import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

pipeline = joblib.load("models/churn_pipeline.pkl")

st.title("📉 Customer Churn Prediction App")
st.write("Predict customer churn using subscription and service details.")

# ---------------- BASIC INFO ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

tenure = st.slider("Tenure (months)", 0, 72, 12)

# ---------------- SERVICES ----------------
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# ---------------- CONTRACT & BILLING ----------------
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, value=1000.0)

# ---------------- INPUT DF ----------------
input_df = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless],
    "PaymentMethod": [payment],
    "MonthlyCharges": [monthly],
    "TotalCharges": [total]
})

# ---------------- PREDICTION ----------------
if st.button("Predict Churn"):
    prob = pipeline.predict_proba(input_df)[0][1]

    st.metric("Churn Probability", f"{prob:.2%}")

    if prob > 0.5:
        st.error("⚠️ High Churn Risk")
    else:
        st.success("✅ Low Churn Risk")
