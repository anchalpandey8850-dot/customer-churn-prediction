
import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing objects
model = joblib.load("churn_logistic_regression_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
feature_selector = joblib.load("feature_selector.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

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

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

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

    input_data = pd.DataFrame([{
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
    }])

    try:
        transformed_data = preprocessor.transform(input_data)
        selected_data = feature_selector.transform(transformed_data)

        prediction = model.predict(selected_data)[0]
        probability = model.predict_proba(selected_data)[0][1]

        if prediction == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to NOT CHURN")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
