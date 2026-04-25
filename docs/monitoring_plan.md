# Monitoring Plan — Churn Prediction API

## Service-Level Objectives (SLOs)

| Metric | Target | Alert threshold |
|--------|--------|-----------------|
| Availability | 99.5% uptime | < 99% over 1 hour |
| Latency (P95) | < 100ms | > 150ms over 5 min |
| Latency (P99) | < 200ms | > 300ms over 5 min |
| Error rate | < 1% | > 2% over 5 min |
| Prediction drift | PSI < 0.1 | PSI > 0.2 on any feature |

## Metrics Collected

### Application Metrics (Prometheus)

| Metric | Type | Description |
|--------|------|-------------|
| `churn_api_requests_total` | Counter | Total requests by method, endpoint, status code |
| `churn_api_request_duration_seconds` | Histogram | Request latency distribution |
| `churn_api_prediction_value` | Histogram | Distribution of predicted churn probabilities |
| `churn_api_prediction_class_total` | Counter | Count of churn vs. no_churn predictions |
| `churn_api_model_info` | Info | Model version, architecture, framework |

### Infrastructure Metrics

| Metric | Source | Alert condition |
|--------|--------|-----------------|
| CPU usage | Docker stats | > 80% sustained |
| Memory usage | Docker stats | > 85% sustained |
| Container restarts | Docker | > 2 restarts in 10 min |

## Drift Detection Strategy

| Check | Frequency | Method | Action if triggered |
|-------|-----------|--------|---------------------|
| Feature distribution (tenure, charges) | Weekly | PSI | Investigate → retrain if PSI > 0.2 |
| Prediction distribution | Daily | KS test | Investigate → check for upstream data issues |
| Actual vs predicted churn rate | Monthly | Comparison | Retrain if gap > 5 percentage points |
| Model performance (F1, AUC) | Monthly | Holdout eval | Retrain if F1 drops > 10% from baseline |

## Alerting

| Severity | Condition | Response |
|----------|-----------|----------|
| **P0 (Critical)** | API down, 0 successful predictions in 5 min | Page on-call, failover to baseline model |
| **P1 (High)** | Error rate > 5%, P99 > 500ms | Investigate within 1 hour |
| **P2 (Medium)** | PSI > 0.2 on any feature | Investigate within 1 week, plan retraining |
| **P3 (Low)** | Churn rate prediction shift > 3pp | Monitor, review at next monthly check |

## Dashboard Panels (Grafana)

1. Request Rate (req/s) — time series, by endpoint and status code
2. Latency Percentiles (p50, p95, p99) — time series
3. Prediction Distribution — histogram of churn probabilities
4. Prediction Classes — pie chart (churn vs no_churn)
5. Error Rate (%) — stat panel with color thresholds
6. Total Requests — stat panel
7. Average Latency — stat panel
8. Model Info — table with version, architecture, framework

## Concepts

### Data Drift vs Concept Drift

Models degrade over time. Understanding WHY they degrade is the first step to fixing it.

- **Data Drift**: The input distribution P(X) changes over time. Example: model trained on customers aged 30-60, but new customers skew younger (18-25). The relationship between features and churn hasn't changed, but the model sees unfamiliar patterns. It's like a doctor trained on adult patients suddenly treating teenagers — the medicine still works the same way, but the symptoms look different.

- **Concept Drift**: The relationship P(Y|X) itself changes. Example: a pandemic changes customer behavior — what used to predict churn no longer does. The model's learned patterns become stale. It's like the rules of the game changing — the doctor now needs to learn entirely new medical knowledge.

Detection methods:

- **PSI (Population Stability Index)**: Measures how much a feature's distribution has shifted. Formula: PSI = Σ(actual% - expected%) × ln(actual%/expected%). Interpretation: PSI < 0.1 = no significant shift, 0.1-0.2 = moderate shift (investigate), > 0.2 = significant shift (retrain).
- **Kolmogorov-Smirnov test**: Non-parametric statistical test that compares two distributions. Returns a p-value — if p < 0.05, the distributions are significantly different.
- **Prediction distribution monitoring**: Track the average prediction score over time — sudden shifts indicate something changed. If last week the average churn probability was 0.28 and this week it's 0.45, something is wrong with the inputs or the world changed.

### Monitoring Percentiles

We don't use average latency alone — it hides outliers. Consider this scenario:

- 99 requests take 20ms each
- 1 request takes 2000ms
- Average = 39.8ms — looks great!
- But that 1 user waited 2 full seconds

Percentiles tell the real story:

- **P50 (median)**: The typical user experience. 50% of requests are faster than this.
- **P95**: 95% of requests are faster than this. This is usually the SLA target ("95% of requests complete in under 100ms").
- **P99**: The worst 1% experience. Critical for customer satisfaction — 1 in 100 users hits this latency.
- **P99.9**: Used for high-traffic systems where even 0.1% of users is thousands of people.

Example: avg=50ms looks great, but P99=2000ms means 1 in 100 users waits 2 seconds. At 10,000 requests/day, that's 100 frustrated users daily.