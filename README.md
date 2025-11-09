# Predicting EMI — Intelligent EMI Eligibility & Affordability System

## Overview
**Predicting EMI** is a machine learning platform designed to predict:
- Whether an applicant is **Eligible**, **Not Eligible**, or **High Risk** for EMI.
- The **Maximum Affordable EMI** amount based on their financial profile.

The project leverages **MLflow** for experiment tracking and **Streamlit** for an interactive user dashboard.

---

## Tech Stack
- Python (Scikit-learn, XGBoost, Pandas, NumPy)
- MLflow for experiment management
- Streamlit for user interface

---

## Model Overview

### Classification Models
- Logistic Regression (baseline)
- Random Forest Classifier
- XGBoost Classifier

### Regression Models
- Linear Regression (baseline)
- Random Forest Regressor
- XGBoost Regressor

---

## Performance Summary

### Classification
| Metric | Score |
|---------|--------|
| Accuracy | 95.9% |
| Precision | 0.88 |
| Recall | 0.72 |
| F1-score | 0.75 |
| ROC-AUC | 0.987 |

### Regression
| Metric | Score |
|---------|--------|
| RMSE | ₹872 |
| MAE | ₹515 |
| R² | 0.987 |
| MAPE | 23.23% |


---

## Run Locally

### Clone the repository
```bash
git clone https://github.com/paldp20/Predicting_EMI.git
cd Predicting-EMI
```

### Set up environment
```bash
python -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt
```

### Launch MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Run Streamlit App
```bash
stramlit run app.py
```




