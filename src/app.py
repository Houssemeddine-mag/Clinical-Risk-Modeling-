# Save this locally as app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(title="Clinical Risk Stratification API", version="1.0")

# Load model artifacts (path is relative to this file, not the working directory)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "final_catboost_clinical_model.pkl"
bundle = joblib.load(MODEL_PATH)
model = bundle['model']
expected_features = bundle['features']

# Input Schema Validation
class PatientData(BaseModel):
    age: float = Field(..., example=65.0)
    anaemia: int = Field(..., example=0)
    creatinine_phosphokinase: int = Field(..., example=250)
    diabetes: int = Field(..., example=0)
    ejection_fraction: float = Field(..., example=30.0)
    high_blood_pressure: int = Field(..., example=1)
    platelets: float = Field(..., example=250000.0)
    serum_creatinine: float = Field(..., example=1.8)
    serum_sodium: float = Field(..., example=135.0)
    sex: int = Field(..., example=1)
    smoking: int = Field(..., example=0)

@app.get("/")
def root():
    return {
        "service": "Clinical Risk Stratification API",
        "version": "1.0",
        "docs": "/docs",
        "endpoint": "POST /api/v1/patient-risk"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "features": len(expected_features)}

@app.post("/api/v1/patient-risk")
def predict_risk(patient: PatientData):
    try:
        # 1. Convert input to DataFrame
        data = patient.dict()
        df_in = pd.DataFrame([data])
        
        # 2. Automated Feature Engineering Pipeline
        df_in['renal_cardiac_ratio'] = df_in['serum_creatinine'] / df_in['ejection_fraction']
        df_in['age_hypertension'] = df_in['age'] * df_in['high_blood_pressure']
        df_in['creatinine_sodium_ratio'] = df_in['serum_creatinine'] / df_in['serum_sodium']
        
        # Ensure column order matches training exactly
        df_in = df_in[expected_features]
        
        # 3. Predict Probability
        risk_score = float(model.predict_proba(df_in)[0, 1])
        
        # 4. Classify Risk Category
        if risk_score >= 0.60:
            level = "HIGH"
        elif risk_score >= 0.35:
            level = "MODERATE"
        else:
            level = "LOW"
            
        return {
            "risk_score": round(risk_score, 4),
            "risk_level": level,
            "key_metrics": {
                "renal_cardiac_ratio": round(float(df_in['renal_cardiac_ratio'].iloc[0]), 4),
                "serum_creatinine": float(patient.serum_creatinine),
                "ejection_fraction": float(patient.ejection_fraction)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))