"""
Customer Churn Prediction — End-to-End ML Pipeline
Author: Aleena Anam
GitHub: github.com/anam-aleena
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# ─── CONFIG ──────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_DIR = "models"
REPORT_DIR = "reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ─── 1. DATA GENERATION (synthetic — mirrors real Telco churn datasets) ──────

def generate_churn_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generates a realistic synthetic telecom churn dataset.
    Features mirror IBM Telco Churn dataset structure.
    """
    np.random.seed(RANDOM_STATE)

    tenure         = np.random.randint(1, 72, n_samples)
    monthly_charges = np.random.uniform(18, 118, n_samples)
    total_charges  = tenure * monthly_charges + np.random.normal(0, 50, n_samples)
    contract       = np.random.choice(["Month-to-month", "One year", "Two year"],
                                      n_samples, p=[0.55, 0.25, 0.20])
    internet       = np.random.choice(["DSL", "Fiber optic", "No"],
                                      n_samples, p=[0.35, 0.44, 0.21])
    payment        = np.random.choice(
                        ["Electronic check", "Mailed check",
                         "Bank transfer", "Credit card"],
                        n_samples, p=[0.34, 0.23, 0.22, 0.21])
    tech_support   = np.random.choice(["Yes", "No", "No internet service"],
                                      n_samples, p=[0.29, 0.50, 0.21])
    senior         = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    dependents     = np.random.choice(["Yes", "No"], n_samples, p=[0.30, 0.70])
    num_services   = np.random.randint(1, 8, n_samples)

    # Churn probability — realistic business logic
    churn_prob = (
        0.35 * (contract == "Month-to-month").astype(float)
        + 0.20 * (internet == "Fiber optic").astype(float)
        + 0.15 * (tech_support == "No").astype(float)
        + 0.10 * (payment == "Electronic check").astype(float)
        - 0.25 * (np.clip(tenure, 0, 24) / 24)
        - 0.05 * (num_services / 7)
        + 0.08 * senior
        + np.random.normal(0, 0.05, n_samples)
    )
    churn = (churn_prob > np.percentile(churn_prob, 73)).astype(int)

    df = pd.DataFrame({
        "tenure":           tenure,
        "MonthlyCharges":   monthly_charges.round(2),
        "TotalCharges":     np.clip(total_charges, 0, None).round(2),
        "Contract":         contract,
        "InternetService":  internet,
        "PaymentMethod":    payment,
        "TechSupport":      tech_support,
        "SeniorCitizen":    senior,
        "Dependents":       dependents,
        "NumServices":      num_services,
        "Churn":            churn
    })

    df.to_csv("data/churn_data.csv", index=False)
    print(f"[DATA] Generated {n_samples} records. Churn rate: {churn.mean():.1%}")
    return df


# ─── 2. PREPROCESSING ────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Encodes categoricals, scales numerics, splits into train/test.
    Returns X_train, X_test, y_train, y_test, feature_names.
    """
    df = df.copy()

    cat_cols = ["Contract", "InternetService", "PaymentMethod",
                "TechSupport", "Dependents"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    print(f"[PREP] Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_names


# ─── 3. MODEL TRAINING & COMPARISON ─────────────────────────────────────────

def train_models(X_train, y_train):
    """
    Trains Logistic Regression, Random Forest, Gradient Boosting, XGBoost.
    Uses 5-fold cross-validation to select best model.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                      random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                           max_depth=4, random_state=RANDOM_STATE),
        "XGBoost":             XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                                             use_label_encoder=False, eval_metric="logloss",
                                             random_state=RANDOM_STATE, n_jobs=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    print("\n[TRAIN] Cross-validation results (5-fold ROC-AUC):")
    print("-" * 50)
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        cv_results[name] = scores.mean()
        print(f"  {name:<25} AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(cv_results, key=cv_results.get)
    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    print(f"\n[TRAIN] Best model: {best_name} (AUC: {cv_results[best_name]:.4f})")
    return models, best_model, best_name, cv_results


# ─── 4. EVALUATION ───────────────────────────────────────────────────────────

def evaluate(models, best_model, best_name, X_train, X_test, y_train, y_test, feature_names):
    """
    Evaluates best model. Prints metrics. Saves confusion matrix + feature importance plots.
    """
    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print(f"\n[EVAL] {best_name} — Test Set Performance")
    print("=" * 50)
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"], ax=ax)
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # Feature Importance (tree models only)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh([feature_names[i] for i in indices],
                       importances[indices], color="#2196F3", edgecolor="white")
        ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{REPORT_DIR}/feature_importance.png", dpi=150)
        plt.close()
        print(f"[EVAL] Plots saved to /{REPORT_DIR}/")

    return {
        "model": best_name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }


# ─── 5. INFERENCE ────────────────────────────────────────────────────────────

def predict_single(customer: dict) -> dict:
    """
    Loads saved model + scaler. Predicts churn for a single customer dict.
    Example usage:
        result = predict_single({
            "tenure": 3, "MonthlyCharges": 85.0, "TotalCharges": 255.0,
            "Contract": 0, "InternetService": 1, "PaymentMethod": 0,
            "TechSupport": 1, "SeniorCitizen": 0, "Dependents": 1, "NumServices": 2
        })
    """
    model  = joblib.load(f"{MODEL_DIR}/best_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

    X = pd.DataFrame([customer])
    X_scaled = scaler.transform(X)
    pred  = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]

    return {
        "churn_prediction": "CHURN" if pred == 1 else "RETAINED",
        "churn_probability": round(float(proba), 4),
        "risk_level": "HIGH" if proba > 0.7 else "MEDIUM" if proba > 0.4 else "LOW"
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CUSTOMER CHURN PREDICTION — ML PIPELINE")
    print("  Author: Aleena Anam | github.com/anam-aleena")
    print("=" * 60)

    df = generate_churn_data(5000)
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)
    models, best_model, best_name, cv_results = train_models(X_train, y_train)
    metrics = evaluate(models, best_model, best_name,
                       X_train, X_test, y_train, y_test, feature_names)

    print("\n[INFERENCE] Sample single-customer prediction:")
    result = predict_single({
        "tenure": 2, "MonthlyCharges": 95.0, "TotalCharges": 190.0,
        "Contract": 0, "InternetService": 1, "PaymentMethod": 0,
        "TechSupport": 1, "SeniorCitizen": 1, "Dependents": 1, "NumServices": 2
    })
    print(f"  {result}")
    print("\n[DONE] Pipeline complete. Models saved to /models/. Reports in /reports/.")
