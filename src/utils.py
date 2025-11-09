import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             mean_squared_error, mean_absolute_error, r2_score)

def load_split_csvs(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df

def prepare_xy(df, target_class="emi_eligibility", target_reg="max_monthly_emi", drop_cols=None):
    """
    Returns X, y_class, y_reg for a dataframe.
    drop_cols: list of cols to drop from features (targets included)
    """
    if drop_cols is None:
        drop_cols = [target_class, target_reg]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    y_class = df[target_class] if target_class in df.columns else None
    y_reg = df[target_reg] if target_reg in df.columns else None
    return X, y_class, y_reg

def fit_scaler(X_train, scaler_path=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if scaler_path:
        joblib.dump(scaler, scaler_path)
    return scaler, X_train_scaled

def transform_with_scaler(scaler, X):
    return scaler.transform(X)

# Classification metrics (multiclass)
def classification_report_all(y_true, y_pred, y_proba=None, average="macro"):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    if y_proba is not None:
        try:
            # multiclass roc auc (one-vs-rest)
            metrics['roc_auc'] = roc_auc_score(pd.get_dummies(y_true), y_proba, average=average)
        except Exception:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
    return metrics

# Regression metrics
def regression_report_all(y_true, y_pred):
    res = {}
    res['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    res['mae'] = mean_absolute_error(y_true, y_pred)
    res['r2'] = r2_score(y_true, y_pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    nonzero = (np.abs(y_true) > 0)
    if nonzero.sum() == 0:
        res['mape'] = None
    else:
        res['mape'] = (np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))) * 100
    return res
