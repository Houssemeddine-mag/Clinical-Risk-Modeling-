# Clinical Risk Stratification - Technical Documentation

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Data Pipeline](#data-pipeline)
- [Model Details](#model-details)
- [API Implementation](#api-implementation)
- [Feature Engineering Deep Dive](#feature-engineering-deep-dive)
- [Model Artifacts](#model-artifacts)
- [Deployment Guide](#deployment-guide)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Architecture Overview

The system follows a simple ML service architecture:

```plaintext
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Clinical Data   │────▶│  FastAPI      │────▶│  Trained Model  │
│  (JSON Input)    │     │  Validation   │     │  (joblib/pkl)   │
└─────────────────┘     └──────────────┘     └─────────────────┘
                              │                        │
                              ▼                        ▼
                        ┌──────────────┐     ┌─────────────────┐
                        │  Feature      │     │  Risk Score     │
                        │  Engineering  │     │  + Category     │
                        └──────────────┘     └─────────────────┘
```

### Components

1. **Data Layer**: Raw CSV files in `data/raw/`, processed data in `data/processed/`
2. **Model Layer**: Serialized models in `models/` directory
3. **API Layer**: FastAPI applications in `src/` (App00.py baseline, app.py improved)
4. **Analysis Layer**: Jupyter notebooks for experimentation
5. **Reporting Layer**: PDF reports with detailed analysis

## Data Pipeline

### Primary Dataset: Heart Failure Clinical Records

**File**: `data/raw/heart_failure_clinical_records_dataset.csv`

```python
# Schema
Columns: age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
         high_blood_pressure, platelets, serum_creatinine, serum_sodium,
         sex, smoking, time, DEATH_EVENT
Shape: (299, 13)
Target: DEATH_EVENT (binary classification)
```

### Data Loading and Preprocessing

The notebooks handle:
- Missing value checks (dataset is generally clean)
- Exploratory data analysis
- Feature correlation analysis
- Train/test splits
- Model training and validation

### Alternative Dataset

The `scripts/download_data.py` script can download and merge the UCI Heart Disease dataset (4 sources) into a combined dataset with 920 records. Note: this is a different dataset structure than the primary heart failure dataset used by the APIs.

## Model Details

### Baseline: LightGBM

**Implementation**: `src/App00.py`

```python
# Model loading
MODEL_PATH = Path(__file__).resolve().parent / "models" / "clinical_lgbm_model_baseline.pkl"
bundle = joblib.load(MODEL_PATH)
model = bundle['model']
expected_features = bundle['features']
```

**Characteristics**:
- Uses only raw features (11 features)
- No feature engineering
- Fast inference
- Good baseline performance

### Improved: CatBoost

**Implementation**: `src/app.py`

```python
# Model loading
MODEL_PATH = Path(__file__).resolve().parent / "models" / "final_catboost_clinical_model.pkl"
bundle = joblib.load(MODEL_PATH)
model = bundle['model']
expected_features = bundle['features']
```

**Characteristics**:
- Uses engineered features (14 features total)
- CatBoost handles categorical features well
- Better performance through domain-driven feature engineering
- More clinically interpretable ratios

## API Implementation

### Common Patterns

Both APIs follow the same structure:

#### 1. Input Validation with Pydantic

```python
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
```

**Validation**: FastAPI automatically validates types and returns 422 errors for invalid inputs.

#### 2. Prediction Endpoint

**Route**: `POST /api/v1/patient-risk`

**Flow**:
1. Receive and validate `PatientData`
2. Convert to pandas DataFrame
3. Apply feature engineering (improved model only)
4. Reorder columns to match `expected_features`
5. Get probability via `model.predict_proba()`
6. Classify into risk level based on thresholds
7. Return structured response

#### 3. Risk Classification

```python
if risk_score >= 0.60:
    level = "HIGH"
elif risk_score >= 0.35:
    level = "MODERATE"
else:
    level = "LOW"
```

These thresholds are consistent across both APIs.

### API Differences

| Aspect | Baseline (App00.py) | Improved (app.py) |
| --- | --- | --- |
| Model | LightGBM | CatBoost |
| Features | 11 raw only | 11 raw + 3 engineered |
| Feature Engineering | None | Applied in endpoint |
| Port (default) | 8001 | 8000 |
| Response key_metrics | Includes serum_creatinine, ejection_fraction, serum_sodium, age | Includes renal_cardiac_ratio, serum_creatinine, ejection_fraction |
| Pydantic API | Uses `patient.dict()` | Uses `patient.model_dump()` with fallback |

## Feature Engineering Deep Dive

### Rationale

Feature engineering is based on clinical knowledge about heart failure:

#### 1. Renal-Cardiac Ratio
```python
df_in['renal_cardiac_ratio'] = df_in['serum_creatinine'] / df_in['ejection_fraction']
```

- **Clinical Significance**: Cardiorenal syndrome is common in heart failure
- **Interpretation**: Higher ratio indicates worse renal function relative to cardiac output
- **Normal Range**: Varies, but elevated values correlate with worse prognosis

#### 2. Age-Hypertension Interaction
```python
df_in['age_hypertension'] = df_in['age'] * df_in['high_blood_pressure']
```

- **Clinical Significance**: Hypertension risk compounds with age
- **Interpretation**: Captures cumulative vascular damage
- **Use Case**: Non-linear interaction not easily captured by linear models

#### 3. Creatinine-Sodium Ratio
```python
df_in['creatinine_sodium_ratio'] = df_in['serum_creatinine'] / df_in['serrom_sodium']  # Note: typo in variable name context
```

- **Clinical Significance**: Reflects dilutional hyponatremia and renal dysfunction
- **Interpretation**: Important prognostic marker in acute and chronic heart failure
- **Note**: In code it's `serum_creatinine / serum_sodium`

### Feature Order

The improved model expects features in a specific order stored in `bundle['features']`. The API ensures correct ordering:
```python
df_in = df_in[expected_features]
```

This is critical for model compatibility.

## Model Artifacts

### File Structure

```python
# Bundle structure (typical)
{
    'model': <trained_model_object>,
    'features': [list_of_feature_names_in_order]
}
```

### Loading

Both models use `joblib.load()` for deserialization. This is compatible with:
- scikit-learn models
- LightGBM models (via joblib)
- CatBoost models (via joblib)

### Model Files

| File | Size | Model Type | Features |
| --- | --- | --- | --- |
| `clinical_lgbm_model_baseline.pkl` | ~varies | LightGBM | 11 |
| `final_catboost_clinical_model.pkl` | ~varies | CatBoost | 14 |

## Deployment Guide

### Local Development

1. **Clone/Navigate to project**
```bash
cd "Clinical risk"
```

2. **Install dependencies**
```bash
pip install fastapi uvicorn pydantic joblib pandas numpy lightgbm catboost scikit-learn
```

3. **Run baseline API**
```bash
cd src
python App00.py
# or
uvicorn App00:app --reload --host 127.0.0.1 --port 8001
```

4. **Run improved API**
```bash
cd src
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### Production Considerations

- **ASGI Server**: Use Gunicorn with Uvicorn workers for production
- **Port Binding**: Bind to 0.0.0.0 for external access
- **Environment Variables**: Consider making paths and thresholds configurable
- **Model Versioning**: Track model versions in the API response
- **Monitoring**: Add logging, metrics, and health checks
- **Security**: Implement authentication/authorization if exposed publicly
- **CORS**: Configure CORS if accessed from web frontends

### Docker Deployment (Example)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
WORKDIR /app/src

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Testing

### Manual Testing via Swagger

1. Navigate to `/docs` endpoint
2. Expand `POST /api/v1/patient-risk`
3. Click "Try it out"
4. Fill in patient data
5. Execute and review response

### Example Test Cases

#### High Risk Patient
```json
{
  "age": 75.0,
  "anaemia": 1,
  "creatinine_phosphokinase": 582,
  "diabetes": 0,
  "ejection_fraction": 20.0,
  "high_blood_pressure": 1,
  "platelets": 265000.0,
  "serum_creatinine": 1.9,
  "serum_sodium": 130.0,
  "sex": 1,
  "smoking": 0
}
```

#### Low Risk Patient
```json
{
  "age": 50.0,
  "anaemia": 0,
  "creatinine_phosphokinase": 100,
  "diabetes": 0,
  "ejection_fraction": 55.0,
  "high_blood_pressure": 0,
  "platelets": 300000.0,
  "serum_creatinine": 0.8,
  "serum_sodium": 140.0,
  "sex": 0,
  "smoking": 0
}
```

### API Health Check

```bash
curl http://127.0.0.1:8000/health
```

Expected response:
```json
{"status": "ok", "model": "final_catboost_clinical_model.pkl", "features": 14}
```

## Troubleshooting

### Common Issues

#### 1. Model file not found
**Error**: `FileNotFoundError: [Errno 2] No such file or directory: .../models/...pkl`

**Solution**: Ensure you're running from the correct directory. The code uses paths relative to `__file__` in `src/`, so running from `src/` is fine. Also verify `models/` directory exists with `.pkl` files.

#### 2. Feature mismatch
**Error**: `KeyError` when selecting features

**Solution**: Ensure input features match exactly what the model expects. The improved API adds engineered features automatically - don't add them manually.

#### 3. Port already in use
**Error**: `Address already in use`

**Solution**: Use a different port or kill the process using the port.
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

#### 4. Pydantic validation errors
**Error**: 422 Unprocessable Entity

**Solution**: Check that all required fields are provided with correct types. Binary fields should be integers (0 or 1), numeric fields can be int or float.

### Debugging Tips

- Check the model bundle structure: `bundle.keys()`
- Verify feature names: `bundle['features']`
- Test with the exact examples from `/docs`
- Use `--reload` flag for development to auto-restart on changes

## References

### Datasets
- [Heart Failure Clinical Records Dataset - UCI](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
- [Heart Disease Dataset - UCI](https://archive.ics.uci.edu/ml/datasets/heart+disease)

### Libraries
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [scikit-learn Documentation](https://scikit-learn.org/)

### Clinical Context
- Heart failure mortality prediction literature
- Cardiorenal syndrome in heart failure
- Electrolyte imbalances as prognostic markers

---

**Last Updated**: 2025  
**Project Type**: Machine Learning / Clinical Decision Support  
**Status**: Research/Development
