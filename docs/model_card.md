# Model Card — Churn Prediction MLP

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | `mlp_churn_v1` |
| **Architecture** | Multi-Layer Perceptron (64 → 32 → 1) with ReLU activations and dropout (0.2) |
| **Framework** | PyTorch 2.11.0 |
| **Training date** | 2026-04-21 |
| **Version** | 1.0.0 |
| **Contact** | Thiago Melo (RM 372447) |

## Intended Use

- **Primary use**: Predict customer churn probability for a telecommunications company
- **Target users**: Retention team analysts reviewing customer risk scores
- **Out of scope**: This model should NOT be used for:
  - Automated customer termination decisions without human review
  - Pricing or credit decisions (not designed for fairness in financial contexts)
  - Domains outside telecommunications (e.g., SaaS, banking)

## Training Data

| Field | Value |
|-------|-------|
| **Dataset** | IBM Telco Customer Churn |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Size** | 7,043 customers |
| **Target** | `Churn` (binary: Yes/No) |
| **Class balance** | ~26.5% churn, ~73.5% no churn |
| **Split** | 70% train / 15% validation / 15% test (stratified) |

## Performance Metrics

| Metric | **MLP (chosen)** | Gradient Boosting | Random Forest | Logistic Regression |
|--------|-----------------:|------------------:|--------------:|--------------------:|
| Accuracy | 0.7736 | 0.7878 | 0.7693 | 0.7381 |
| Precision | 0.5604 | 0.6238 | 0.5505 | 0.5043 |
| Recall | 0.6818 | 0.5053 | 0.7139 | 0.7834 |
| F1-score | 0.6152 | 0.5583 | 0.6217 | 0.6136 |
| ROC-AUC | 0.8401 | 0.8330 | 0.8361 | 0.8415 |
| PR-AUC | 0.6312 | 0.6348 | 0.6482 | 0.6326 |

**Threshold selection**: Default 0.5. In production, the retention team may prefer a lower threshold (e.g., 0.3) to catch more at-risk customers at the cost of more false positives (unnecessary retention offers). This is a business decision, not a model decision.

## Limitations

- **Small dataset**: 7,043 samples is small by modern standards. Model may not generalize to much larger customer bases.
- **Static snapshot**: Data represents a single point in time. Customer behavior patterns change (concept drift).
- **Missing features**: No behavioral data (app usage, call logs, browsing patterns). No sentiment data (support ticket tone). These could significantly improve predictions.
- **No temporal features**: No sequence modeling — doesn't capture trends like "monthly charges increasing over 3 months."
- **Geographic bias**: Dataset is from a single (unnamed) telco. Patterns may not transfer to other regions, cultures, or market structures.

## Biases

- **Senior citizens** (16% of dataset): Small sample size means performance on this subgroup may be less reliable. Monitor recall separately.
- **Gender**: Model includes gender as a feature. Verify that churn predictions don't systematically differ by gender in ways that could lead to discriminatory retention offers.
- **Contract type**: Month-to-month customers churn more. The model may learn to over-index on this, essentially just flagging contract type rather than learning deeper patterns.
- **Payment method**: Electronic check users churn more. This could be a proxy for socioeconomic factors — be cautious about using this for targeted interventions.

## Ethical Considerations

- **Human oversight**: Predictions should inform, not replace, human judgment. A retention agent should review the customer's full context before acting.
- **Transparency**: If a customer asks why they received a retention offer, the team should be able to explain the key factors (contract type, tenure, charges).
- **Data privacy**: Model uses demographic data (gender, senior status). Ensure compliance with local data protection regulations.
- **Feedback loops**: If the model drives retention actions that change customer behavior, the model's training data becomes stale. Plan for periodic retraining.

## Maintenance

| Field | Value |
|-------|-------|
| **Retraining cadence** | Quarterly (recommended) |
| **Drift monitoring** | PSI on top 5 features monthly |
| **Performance review** | Monthly comparison of predicted vs. actual churn rates |
| **Owner** | Thiago Melo |