
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Customer Churn Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "churn_logistic_regression_model.pkl"))
preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
feature_selector = joblib.load(os.path.join(BASE_DIR, "feature_selector.pkl"))


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        data = customer.model_dump()

        tenure = data["tenure"]
        data["AverageMonthlyCharges"] = (
            data["TotalCharges"] / tenure if tenure > 0 else 0.0
        )
        data["MonthlyCharges_Tenure"] = data["MonthlyCharges"] * tenure

        if tenure <= 12:
            data["TenureGroup"] = "New"
        elif tenure <= 24:
            data["TenureGroup"] = "Short-term"
        elif tenure <= 48:
            data["TenureGroup"] = "Medium-term"
        else:
            data["TenureGroup"] = "Long-term"

        input_df = pd.DataFrame([data])
        transformed = preprocessor.transform(input_df)
        selected = feature_selector.transform(transformed)

        prediction = int(model.predict(selected)[0])
        probability = float(model.predict_proba(selected)[0][1])

        return {
            "prediction": prediction,
            "result": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": round(probability * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
