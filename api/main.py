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
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_tuned.pkl')
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

# ===== ENDPOINTS =====

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint - verify the API is running
    """
    return {
        "status": "healthy",
        "model": "Random Forest (Tuned)",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(customer: CustomerData):
    try:
        customer_dict = customer.dict()
        customer_df = pd.DataFrame([customer_dict])
        
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                           'InternetService', 'OnlineSecurity', 'OnlineBackup',
                           'DeviceProtection', 'TechSupport', 'StreamingTV', 
                           'StreamingMovies', 'Contract', 'PaperlessBilling',
                           'PaymentMethod', 'MultipleLines']
        
        for col in categorical_cols:
            if col in label_encoders:
                customer_df[col + '_encoded'] = label_encoders[col].transform(customer_df[[col]]).ravel()
        
        expected_features = [
            "SeniorCitizen", "tenure", "MonthlyCharges",
            "customerID_encoded",
            "gender_encoded", "Partner_encoded", "Dependents_encoded", 
            "PhoneService_encoded", "MultipleLines_encoded", "InternetService_encoded",
            "OnlineSecurity_encoded", "OnlineBackup_encoded", "DeviceProtection_encoded",
            "TechSupport_encoded", "StreamingTV_encoded", "StreamingMovies_encoded",
            "Contract_encoded", "PaperlessBilling_encoded", "PaymentMethod_encoded",
            "TotalCharges_encoded"
        ]
        
        feature_values = []
        for feat in expected_features:
            if feat == "customerID_encoded":
                feature_values.append(0)  # Default customerID
            elif feat == "TotalCharges_encoded":
                feature_values.append(customer_df["TotalCharges"].values[0])
            elif feat in customer_df.columns:
                feature_values.append(customer_df[feat].values[0])
            else:
                feature_values.append(0)
        
        X_input = pd.DataFrame([feature_values], columns=expected_features)
        X_scaled = scaler.transform(X_input)
        
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
        customer_dict = customer.dict()
        customer_df = pd.DataFrame([customer_dict])
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in customer_df.columns:
                customer_df[col + '_encoded'] = encoder.transform(customer_df[col])
        
        # Drop original categorical columns
        categorical_cols = [col for col in customer_df.columns if col in label_encoders.keys()]
        customer_df = customer_df.drop(columns=categorical_cols)
        
        # Scale features
        customer_scaled = pd.DataFrame(
            scaler.transform(customer_df),
            columns=customer_df.columns
        )
        
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