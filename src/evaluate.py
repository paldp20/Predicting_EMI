import joblib
import pandas as pd
from src.utils import prepare_xy, transform_with_scaler, classification_report_all, regression_report_all

# Paths
test_csv = r"C:\Users\a_sur\Documents\Labmentix_Project_4_EMIPredict\dataset\test_featured.csv"
scaler_path = "artifacts/scaler.joblib"
best_clf_path = "models/best_classifier.joblib"
best_reg_path = "models/best_regressor.joblib"

# Load data
test_df = pd.read_csv(test_csv)

# Prepare data
X_test, y_test_class, y_test_reg = prepare_xy(test_df, "emi_eligibility", "max_monthly_emi")

# Transform features
scaler = joblib.load(scaler_path)
X_test_scaled = transform_with_scaler(scaler, X_test)

# Classification Evaluation
clf = joblib.load(best_clf_path)
y_pred_class = clf.predict(X_test_scaled)
try:
    y_proba = clf.predict_proba(X_test_scaled)
except Exception:
    y_proba = None

metrics_class = classification_report_all(y_test_class, y_pred_class, y_proba=y_proba)
print("\nFinal Classification Model Performance on Test Set:")
print(metrics_class)

# Regression Evaluation
reg = joblib.load(best_reg_path)
y_pred_reg = reg.predict(X_test_scaled)

metrics_reg = regression_report_all(y_test_reg, y_pred_reg)
print("\nFinal Regression Model Performance on Test Set:")
print(metrics_reg)
