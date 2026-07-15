"""
Rising Waters — flood prediction model training.

Trains Decision Tree, Random Forest, K-Nearest Neighbours, and XGBoost
classifiers on historical weather data, compares them on held-out test
data, and saves the best-performing model + scaler + metadata for the
Flask app to load.

Usage:
    python train_model.py
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

DATA_PATH = "data/flood_dataset.csv"
MODEL_DIR = "model"
FEATURES = [
    "annual_rainfall_mm",
    "seasonal_rainfall_mm",
    "cloud_visibility_km",
    "humidity_percent",
    "temperature_c",
    "days_continuous_rainfall",
    "river_discharge_cumecs",
]
TARGET = "flood_occurred"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate(name, model, X_test, y_test, scaled=False, scaler=None):
    X_eval = scaler.transform(X_test) if scaled else X_test
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall": round(recall_score(y_test, y_pred) * 100, 2),
        "f1_score": round(f1_score(y_test, y_pred) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, y_proba) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"\n{name}")
    print(f"  Accuracy : {metrics['accuracy']}%")
    print(f"  Precision: {metrics['precision']}%")
    print(f"  Recall   : {metrics['recall']}%")
    print(f"  F1 score : {metrics['f1_score']}%")
    print(f"  ROC AUC  : {metrics['roc_auc']}%")
    return metrics


def main():
    X_train, X_test, y_train, y_test = load_data()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    fitted_models = {}

    # 1. Decision Tree
    dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y_train)
    results["Decision Tree"] = evaluate("Decision Tree", dt, X_test, y_test)
    fitted_models["Decision Tree"] = (dt, False)

    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results["Random Forest"] = evaluate("Random Forest", rf, X_test, y_test)
    fitted_models["Random Forest"] = (rf, False)

    # 3. K-Nearest Neighbours (needs scaled features)
    knn = KNeighborsClassifier(n_neighbors=15, weights="distance")
    knn.fit(X_train_scaled, y_train)
    results["KNN"] = evaluate("KNN", knn, X_test, y_test, scaled=True, scaler=scaler)
    fitted_models["KNN"] = (knn, True)

    # 4. XGBoost (falls back to sklearn's GradientBoostingClassifier if the
    #    xgboost package isn't installed in this environment)
    if HAS_XGBOOST:
        xgb = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        xgb_name = "XGBoost"
    else:
        print("\n[!] xgboost not installed in this environment — training "
              "sklearn's GradientBoostingClassifier as a drop-in stand-in. "
              "Run `pip install xgboost` and re-run this script to train "
              "real XGBoost.")
        xgb = GradientBoostingClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.85, random_state=42,
        )
        xgb_name = "XGBoost (fallback: GradientBoosting)"
    xgb.fit(X_train, y_train)
    results[xgb_name] = evaluate(xgb_name, xgb, X_test, y_test)
    fitted_models[xgb_name] = (xgb, False)

    # Pick the best model by accuracy
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model, best_needs_scaling = fitted_models[best_name]

    print(f"\n{'=' * 50}")
    print(f"Best model: {best_name} ({results[best_name]['accuracy']}% accuracy)")
    print(f"{'=' * 50}")

    joblib.dump(best_model, f"{MODEL_DIR}/flood_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    metadata = {
        "best_model": best_name,
        "needs_scaling": best_needs_scaling,
        "features": FEATURES,
        "results": results,
        "xgboost_available": HAS_XGBOOST,
    }
    with open(f"{MODEL_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved: {MODEL_DIR}/flood_model.pkl, scaler.pkl, metadata.json")


if __name__ == "__main__":
    main()
