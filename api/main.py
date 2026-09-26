# File: api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List
import os

# ===== INITIALIZE APP =====
app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability using machine learning",
    version="1.0.0"
)

# ===== LOAD MODEL & SCALER =====
print("Loading model and scaler...")

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'production_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
ENCODERS_PATH = os.path.join(BASE_DIR, 'models', 'label_encoders.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoders = joblib.load(ENCODERS_PATH)

print(" Model loaded successfully")

# ===== REQUEST/RESPONSE MODELS =====

class CustomerData(BaseModel):
    """Input data for a single customer"""
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
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
    MultipleLines: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 24,
                "MonthlyCharges": 65.5,
                "TotalCharges": 1570.0,
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Electronic check",
                "MultipleLines": "No"
            }
        }

class ChurnPredictionResponse(BaseModel):
    """Output prediction"""
    churn_probability: float
    prediction: str
    confidence: float
    risk_level: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    version: str

FEATURE_ORDER = list(scaler.feature_names_in_)  # exact training column order

def build_features(customer: CustomerData) -> pd.DataFrame:
    """Reproduce src/feature_engineering.py: label-encode, order columns, scale."""
    row = customer.dict()
    for col, encoder in label_encoders.items():
        row[col + '_encoded'] = int(encoder.transform([row[col]])[0])
    X = pd.DataFrame([[row[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return pd.DataFrame(scaler.transform(X), columns=FEATURE_ORDER)

# ===== ENDPOINTS =====

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint - verify the API is running
    """
    return {
        "status": "healthy",
        "model": type(model).__name__,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(customer: CustomerData):
    try:
        X_scaled = build_features(customer)
        
        churn_prob = float(model.predict_proba(X_scaled)[0][1])
        
        return {
            "churn_probability": round(churn_prob, 4),
            "prediction": "Will Churn" if churn_prob > 0.5 else "Will Stay",
            "confidence": round(max(churn_prob, 1 - churn_prob), 4),
            "risk_level": "HIGH" if churn_prob > 0.7 else "MEDIUM" if churn_prob > 0.4 else "LOW"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/predict-batch")
def predict_batch(customers: List[CustomerData]):
    """
    Predict churn for multiple customers at once.
    """
    predictions = []
    
    for customer in customers:
        customer_scaled = build_features(customer)
        
        # Make prediction
        prob = float(model.predict_proba(customer_scaled)[0][1])
        predictions.append({
            "churn_probability": round(prob, 4),
            "prediction": "Will Churn" if prob > 0.5 else "Will Stay"
        })
    
    return {"predictions": predictions}


@app.get("/debug/features")
def get_features():
    """Debug endpoint - shows expected features"""
    return {
        "scaler_features": scaler.get_feature_names_out().tolist() if hasattr(scaler, 'get_feature_names_out') else "Unknown",
        "model_n_features": model.n_features_in_
    }

# ===== RUN LOCALLY =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)