import os
import argparse
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV

from src.utils import (
    load_split_csvs, prepare_xy, fit_scaler, transform_with_scaler,
    classification_report_all, regression_report_all
)

os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

def train_classification(X_train, y_train, X_val, y_val, experiment_name="EMIPredict_Classification"):
    mlflow.set_experiment(experiment_name)
    best_score = -np.inf
    best_model_info = None

    # Models to train (baseline + ensembles + xgboost). small hyperparam sets for speed.
    models = {
        "logistic": (LogisticRegression(max_iter=1000), {"C":[1.0]}),
        "random_forest": (RandomForestClassifier(random_state=42, n_jobs=-1),
                          {"n_estimators":[100], "max_depth":[None]}),
        "xgboost": (XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_jobs=4),
                    {"n_estimators":[100], "max_depth":[6]})
    }

    for name, (estimator, param_grid) in models.items():
        with mlflow.start_run(run_name=name):
            # GridSearch for RF and XGB (light)
            if param_grid:
                gs = GridSearchCV(estimator, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
                params = gs.best_params_
            else:
                estimator.fit(X_train, y_train)
                model = estimator
                params = {}

            # Predict on validation set
            y_pred = model.predict(X_val)
            # probabilities (if supported)
            y_proba = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val)
            else:
                # try decision function fallback for probabilistic score
                try:
                    scores = model.decision_function(X_val)
                    # convert to proba-like via softmax
                    exp = np.exp(scores)
                    y_proba = exp / np.sum(exp, axis=1, keepdims=True)
                except Exception:
                    y_proba = None

            metrics = classification_report_all(y_val, y_pred, y_proba=y_proba)
            print(f"Model {name} val metrics: {metrics}")

            # Log to MLflow
            mlflow.log_params(params)
            mlflow.log_param("model_name", name)
            mlflow.log_metrics(metrics)

            # save model artifact
            model_path = f"models/{name}_model.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="models")

            # store best by accuracy (or f1) — here using f1
            if metrics.get("f1", 0) > best_score:
                best_score = metrics.get("f1", 0)
                best_model_info = {"name": name, "model_path": model_path, "metrics": metrics, "estimator": model}

    return best_model_info

def train_regression(X_train, y_train, X_val, y_val, experiment_name="EMIPredict_Regression"):
    mlflow.set_experiment(experiment_name)
    best_rmse = np.inf
    best_model_info = None

    models = {
        "linear": (LinearRegression(), {}),
        "random_forest": (RandomForestRegressor(random_state=42, n_jobs=-1), {"n_estimators":[100]}),
        "xgboost": (XGBRegressor(n_jobs=4), {"n_estimators":[100], "max_depth":[6]})
    }

    for name, (estimator, param_grid) in models.items():
        with mlflow.start_run(run_name=name):
            if param_grid:
                gs = GridSearchCV(estimator, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
                params = gs.best_params_
            else:
                estimator.fit(X_train, y_train)
                model = estimator
                params = {}

            y_pred = model.predict(X_val)
            metrics = regression_report_all(y_val, y_pred)
            print(f"Regressor {name} val metrics: {metrics}")

            mlflow.log_params(params)
            mlflow.log_param("model_name", name)
            mlflow.log_metrics(metrics)

            model_path = f"models/{name}_regressor.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path, artifact_path="models")

            if metrics['rmse'] < best_rmse:
                best_rmse = metrics['rmse']
                best_model_info = {"name": name, "model_path": model_path, "metrics": metrics, "estimator": model}

    return best_model_info

def main(args):
    # load processed featured csvs
    train_df, val_df, test_df = load_split_csvs(
        args.train_csv, args.val_csv, args.test_csv
    )

    # Prepare X, y
    X_train, y_train_class, y_train_reg = prepare_xy(train_df, args.target_class, args.target_reg)
    X_val, y_val_class, y_val_reg = prepare_xy(val_df, args.target_class, args.target_reg)
    X_test, y_test_class, y_test_reg = prepare_xy(test_df, args.target_class, args.target_reg)

    # Fit scaler on train then transform each split
    scaler, X_train_scaled = fit_scaler(X_train, scaler_path="artifacts/scaler.joblib")
    X_val_scaled = transform_with_scaler(scaler, X_val)
    X_test_scaled = transform_with_scaler(scaler, X_test)

    # Classification training
    best_clf = train_classification(X_train_scaled, y_train_class.values, X_val_scaled, y_val_class.values)

    # Regression training
    best_reg = train_regression(X_train_scaled, y_train_reg.values, X_val_scaled, y_val_reg.values)

    print("BEST MODELS:")
    print("Best classifier:", best_clf)
    print("Best regressor:", best_reg)

    # Save best models to a canonical path for deployment
    if best_clf:
        joblib.dump(best_clf['estimator'], "models/best_classifier.joblib")
    if best_reg:
        joblib.dump(best_reg['estimator'], "models/best_regressor.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=r"C:\Users\a_sur\Documents\Labmentix_Project_4_EMIPredict\dataset\train_featured.csv")
    parser.add_argument("--val_csv", default=r"C:\Users\a_sur\Documents\Labmentix_Project_4_EMIPredict\dataset\val_featured.csv")
    parser.add_argument("--test_csv", default=r"C:\Users\a_sur\Documents\Labmentix_Project_4_EMIPredict\dataset\test_featured.csv")
    parser.add_argument("--target_class", default="emi_eligibility")
    parser.add_argument("--target_reg", default="max_monthly_emi")
    args = parser.parse_args()
    main(args)
