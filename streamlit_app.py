
import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# FastAPI backend URL — replace with your actual Render URL
API_URL = "https://customer-churn-prediction-2-vph2.onrender.com/predict"

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict the probability of churn.")

# Customer inputs
gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])


tenure = st.number_input(
    "Tenure (months)",
    min_value=0.0,
    max_value=100.0,
    value=12.0
)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# Feature engineering
if tenure > 0:
    AverageMonthlyCharges = TotalCharges / tenure
else:
    AverageMonthlyCharges = 0.0

MonthlyCharges_Tenure = MonthlyCharges * tenure

if tenure <= 12:
    TenureGroup = "New"
elif tenure <= 24:
    TenureGroup = "Short-term"
elif tenure <= 48:
    TenureGroup = "Medium-term"
else:
    TenureGroup = "Long-term"

# Prediction button
if st.button("Predict Churn"):

    input_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "AverageMonthlyCharges": AverageMonthlyCharges,
        "MonthlyCharges_Tenure": MonthlyCharges_Tenure,
        "TenureGroup": TenureGroup
    }

    try:
        with st.spinner("Predicting customer churn..."):
            response = requests.post(
                API_URL,
                json=input_data,
                timeout=120
            )

        if response.ok:
            result = response.json()

            # These response keys must match your FastAPI /predict response.
            prediction = result.get("prediction")
            probability = result.get("churn_probability")

            if prediction is None or probability is None:
                st.error(
                    "The API response does not contain the expected "
                    "'prediction' and 'churn_probability' fields."
                )
                st.json(result)

            else:
                if str(prediction).upper() in ["1", "YES", "CHURN"]:
                    st.error("⚠️ Customer is likely to CHURN")
                else:
                    st.success("✅ Customer is likely to NOT CHURN")

                st.metric(
                    "Churn Probability",
                    f"{float(probability) * 100:.2f}%"
                )

        else:
            st.error(f"API error ({response.status_code})")
            st.code(response.text)

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the prediction API: {e}")

