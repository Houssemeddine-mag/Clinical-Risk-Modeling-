# Clinical Risk Stratification: Heart Failure Mortality Prediction

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Classification-green.svg)]()

## Abstract

This study explores the application of machine learning for early mortality risk stratification in patients with heart failure. Using routinely collected clinical indicators from the UCI Heart Failure Clinical Records dataset, we developed and evaluated interpretable classification models to predict the likelihood of death during follow-up. The research demonstrates that incorporating clinically-motivated feature engineering, alongside standard predictors, improves risk discrimination while maintaining model interpretability for potential clinical use.

## Objectives

- **Predict mortality risk**: Develop models capable of accurately predicting death events in heart failure patients using clinical features.
- **Enable risk stratification**: Classify patients into clinically actionable risk categories (Low, Moderate, High).
- **Prioritize interpretability**: Ensure model decisions are explainable and aligned with established clinical knowledge.
- **Evaluate feature engineering**: Assess whether domain-driven derived features improve predictive performance over baseline approaches.

## Dataset

This work uses the [Heart Failure Clinical Records Dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) from the UC Irvine Machine Learning Repository.

| Attribute | Details |
| --- | --- |
| **Cohort** | 299 patients with heart failure |
| **Time Period** | Follow-up period recorded per patient (days) |
| **Target Variable** | `DEATH_EVENT` (0 = survived, 1 = deceased during follow-up) |
| **Features** | 12 clinical predictors including demographics, comorbidities, laboratory values, and cardiac function measurements |

**Key Predictors:** Age, anaemia, creatinine phosphokinase, diabetes, ejection fraction, high blood pressure, platelets, serum creatinine, serum sodium, sex, smoking status, and follow-up time.

## Methodology

### Study Design
This is a supervised binary classification problem framed as mortality prediction in heart failure. Models were developed using a structured experimental pipeline encompassing data exploration, baseline modeling, feature engineering, and model evaluation.

### Modeling Approach
Two complementary strategies were investigated:

| Approach | Description | Rationale |
| --- | --- | --- |
| **Baseline** | Models trained exclusively on original clinical features. | Establishes a performance benchmark using readily available data with minimal preprocessing. |
| **Enhanced (Feature-Engineered)** | Models augmented with clinically-motivated derived features. | Captures known pathophysiological relationships in heart failure, including cardiorenal dysfunction, electrolyte imbalances, and compounding comorbidities. |

### Risk Stratification
Predicted probabilities were converted into three actionable risk tiers to facilitate clinical interpretation:

| Risk Level | Threshold | Clinical Interpretation |
| --- | --- | --- |
| **Low** | Probability < 0.35 | Lower relative risk of mortality during follow-up |
| **Moderate** | 0.35 ≤ Probability < 0.60 | Intermediate risk requiring closer monitoring |
| **High** | Probability ≥ 0.60 | Elevated risk warranting heightened clinical attention |

### Evaluation & Explainability
Model performance and interpretability were assessed through comprehensive evaluation. Feature importance and explainability techniques were employed to validate that model behavior aligns with established clinical understanding of heart failure prognosis.

## Key Findings

- **Domain knowledge matters**: Clinically-motivated feature engineering improved risk discrimination compared to using raw features alone.
- **Interpretability preserved**: The enhanced models maintained clinical plausibility, with key predictors aligning with established heart failure risk factors.
- **Actionable stratification**: The three-tier risk categorization provides a practical framework for stratifying patients based on predicted mortality risk.
- **Robust benchmarking**: Systematic comparison between baseline and enhanced approaches quantifies the value of feature engineering in this clinical context.

Detailed performance metrics, confusion matrices, ROC curves, and explainability analyses are presented in the full reports.

## Repository Structure

```plaintext
Clinical risk/
├── data/           # Datasets (raw and processed)
├── models/         # Trained model artifacts from experiments
├── notebooks/      # Exploratory data analysis, model development, and evaluation
├── reports/        # Research reports and analysis (EN/FR)
├── scripts/        # Utilities for data acquisition
└── src/            # Demonstration code and supporting implementations
```

## Reports

| Report | Language | Description |
| --- | --- | --- |
| `Clinical Risk Modeling and Explainability: Mortality Prediction in Heart Failure 01.pdf` | English | Feature importance, model interpretability, and explainability analysis. |
| `Clinical Risk Modeling - Phase 02B Improvement Report.pdf` | English | Comparative evaluation between baseline and feature-engineered models. |
| `Rapport Global - Prédiction du Risque Clinique et Explainabilité.pdf` | French | Comprehensive synthesis of methodology, results, and conclusions. |

## Getting Started

For implementation details, API usage, environment setup, and reproducibility instructions, see **[documentation.md](documentation.md)**.

## References

- Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*, 20(1), 16. https://doi.org/10.1186/s12911-020-1023-5
- UCI Machine Learning Repository: Heart Failure Clinical Records Dataset. https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
