# ML Canvas

**Objective**

- Predict which telecom customers will churn (cancel service) so the retention team can intervene proactively.
- Binary classification: Churn = Yes / No.

**Success metric**

- Primary: PR-AUC >= 0.60.
- Secondaty: ROC-AUC >= 0.85.
- Tertiary: F1 >= 0.70 on the minority class (churners).
- Business KPI: reduce churn rate by 15% through targeted retention.

**Data**

- [Telco Customer Churn dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
  - Size: 7,043 rows x 21 columns.
- Features:
  - demographics (gender, SeniorCitizen, Partner, Dependents)
  - account info (tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges)
  - services (InternetService, OnlineSecurity, TechSupport, StreamingTV, etc.).
- Target: Churn (Yes/No).

**Features**

- 19 input features after dropping customerID.
- Mix of numerical (tenure, MonthlyCharges, TotalCharges) and categorical (Contract, InternetService, PaymentMethod, etc.).

**Model**

- Baseline: Dummy Classifier and Logistic Regression.
- Primary: PyTorch MLP (Multi-Layer Perceptron).

**Constraints**

- Must run on CPU (no GPU requirement).
- API response time < 200ms.
- Model artifact must be serializable and versionable.

**Risks**

- Class imbalance (~26.5% churn).
- Small dataset (7K rows).
- Risk of overfitting with complex models.
- Possible data leakage if preprocessing isn't handled correctly.