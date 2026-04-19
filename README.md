# Customer Churn Prediction — End-to-End ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Predict which customers are about to leave — before they do.**  
> A production-grade machine learning pipeline that trains, compares, and deploys churn prediction models on telecom customer data.

---

## Problem Statement

Customer churn costs telecom companies **$30–50 billion annually** in lost revenue. Identifying at-risk customers *before* they cancel allows businesses to proactively intervene with targeted retention strategies — a proven approach that reduces churn by 10–25%.

This project builds an end-to-end ML pipeline that:
- Ingests and preprocesses raw customer data
- Engineers predictive features from behavioral signals
- Trains and cross-validates 4 ML models
- Selects the best model using rigorous evaluation
- Deploys a reusable inference function for real-time prediction

---

## Results

| Model | CV ROC-AUC | Test Accuracy | Test F1 | Test Recall |
|---|---|---|---|---|
| Logistic Regression | ~0.82 | ~0.79 | ~0.68 | ~0.71 |
| Random Forest | ~0.88 | ~0.84 | ~0.75 | ~0.76 |
| Gradient Boosting | ~0.89 | ~0.85 | ~0.76 | ~0.77 |
| **XGBoost** ✅ | **~0.91** | **~0.87** | **~0.79** | **~0.80** |

> XGBoost selected as best model based on 5-fold stratified cross-validation AUC — not test set performance — to prevent data leakage.

---

## Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── churn_data.csv          # Generated dataset (5,000 records)
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory data analysis + visual insights
│   └── 02_Model_Training.ipynb # Model training, comparison, ROC curves
│
├── src/
│   └── pipeline.py             # Full ML pipeline (data → train → evaluate → infer)
│
├── models/
│   ├── best_model.pkl          # Serialized best model
│   └── scaler.pkl              # Fitted StandardScaler
│
├── reports/
│   ├── confusion_matrix.png    # Confusion matrix heatmap
│   ├── feature_importance.png  # Feature importance bar chart
│   ├── roc_curves.png          # ROC curves for all models
│   ├── model_comparison.png    # Cross-validation AUC comparison
│   ├── churn_distribution.png  # Target variable distribution
│   ├── numeric_features.png    # Numeric feature distributions
│   └── categorical_churn.png   # Churn rate by category
│
├── requirements.txt
└── README.md
```

---

## ML Pipeline — Step by Step

```
Raw Data → Preprocessing → Feature Engineering → Train/Test Split
    → 5-Fold Cross-Validation → Model Selection → Evaluation → Inference API
```

### 1. Data & Features

| Feature | Type | Description |
|---|---|---|
| `tenure` | Numeric | Months as a customer |
| `MonthlyCharges` | Numeric | Current monthly bill |
| `TotalCharges` | Numeric | Cumulative billing |
| `Contract` | Categorical | Month-to-month / 1yr / 2yr |
| `InternetService` | Categorical | DSL / Fiber / None |
| `PaymentMethod` | Categorical | Electronic check, etc. |
| `TechSupport` | Categorical | Has tech support? |
| `SeniorCitizen` | Binary | Senior customer flag |
| `Dependents` | Binary | Has dependents? |
| `NumServices` | Numeric | Number of add-on services |

### 2. Models Compared
- **Logistic Regression** — interpretable baseline
- **Random Forest** — ensemble, handles non-linearity
- **Gradient Boosting** — sequential boosting
- **XGBoost** — regularized boosting, state-of-the-art on tabular data

### 3. Evaluation Strategy
- **5-fold stratified cross-validation** for model selection
- **Held-out 20% test set** for final evaluation
- **Primary metric: ROC-AUC** (handles class imbalance)
- **Secondary metrics:** F1, Recall (minimize false negatives = missed churners)

---

## Key Findings

- **Contract type** is the strongest churn predictor — month-to-month customers churn 3× more than 2-year contract customers
- **Tenure < 12 months** is a high-risk signal — new customers need onboarding attention
- **Fiber optic + Electronic check** customers represent the highest-risk segment
- **Tech support absence** significantly increases churn probability

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/anam-aleena/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python src/pipeline.py
```

### 4. Explore notebooks
```bash
jupyter notebook notebooks/
```

### 5. Predict a single customer
```python
from src.pipeline import predict_single

result = predict_single({
    "tenure": 2,
    "MonthlyCharges": 95.0,
    "TotalCharges": 190.0,
    "Contract": 0,          # 0 = Month-to-month
    "InternetService": 1,   # 1 = Fiber optic
    "PaymentMethod": 0,     # 0 = Electronic check
    "TechSupport": 1,       # 1 = No
    "SeniorCitizen": 1,
    "Dependents": 1,
    "NumServices": 2
})

print(result)
# {'churn_prediction': 'CHURN', 'churn_probability': 0.847, 'risk_level': 'HIGH'}
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Serialization | Joblib |
| Notebooks | Jupyter |

---

## Author

**Aleena Anam** — AI/ML Engineer & Data Scientist  
📧 anamaleena0@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/aleena-anam-2056a4368) | [GitHub](https://github.com/anam-aleena)

---

## License

MIT License — free to use, modify, and distribute.
