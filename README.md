# Churn Prediction Pipeline 

**Live API:** https://churn-prediction-pipeline-9fum.onrender.com/docs

## Overview
An end-to-end machine learning pipeline that predicts customer churn with **85.8% accuracy**, deployed as a production REST API.

## Architecture
Raw Data → EDA → Feature Engineering → ML Models → FastAPI → Render Cloud

## Key Business Insights
- 🔴 New customers (0-6 months) churn at **60%** vs established customers at **<10%**
- 🔴 Month-to-month contracts churn at **42%** vs 2-year contracts at **3%**
- 🔴 Higher monthly charges slightly increase churn risk
- ✅ **Contract type** is the #1 predictor of churn (SHAP value: 0.97)

## Models Compared

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 80.0% | 0.8406 |
| Random Forest (Tuned) | 85.8% | 0.8751 ✅ |
| XGBoost | ~83% | 0.8381 |

**Selected:** Random Forest — best AUC, validated with SHAP explainability.

## API Usage

### Health Check
```bash
curl https://churn-prediction-pipeline-9fum.onrender.com/health
```

### Predict Churn
```bash
curl -X POST "https://churn-prediction-pipeline-9fum.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Response
```json
{
  "churn_probability": 0.211,
  "prediction": "Will Stay",
  "confidence": 0.789,
  "risk_level": "LOW"
}
```

## Project Structure
churn-prediction-pipeline/
├── data/
│ ├── raw_data/telco_customers.csv
│ └── processed/
├── notebooks/01_exploratory_analysis.ipynb
├── figures/ (11 visualizations)
├── src/
│ ├── feature_engineering.py
│ ├── train_baseline.py
│ ├── train_advanced.py
│ ├── tune_hyperparameters.py
│ └── train_xgboost.py
├── models/
│ ├── production_model.pkl
│ ├── scaler.pkl
│ ├── label_encoders.pkl
│ └── shap_explainer.pkl
├── api/main.py
├── render.yaml
└── requirements.txt

## Tech Stack
- **Data:** Python, Pandas, NumPy
- **ML:** Scikit-learn, XGBoost, SHAP
- **API:** FastAPI, Uvicorn
- **Deployment:** Render (Free tier)
- **Version Control:** Git, GitHub

## How to Run Locally
```bash
git clone https://github.com/BhumikaB1/churn-prediction-pipeline.git
cd churn-prediction-pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd api && python main.py
# Visit http://localhost:8000/docs
```

## Interactive API Docs
Visit: https://churn-prediction-pipeline-9fum.onrender.com/docs

---
Built by Bhumika | B.Tech CSE (Data Science) | github.com/BhumikaB1
