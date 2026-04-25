# Deployment Architecture

We chose **Real-time (API)** because the retention team needs instant risk scores when reviewing customer accounts. They open a customer profile, the system calls our API, and the churn probability appears in under 200ms.

| Strategy | How it works | When to use | Latency | Example |
|----------|-------------|-------------|---------|---------|
| **Batch** | Scheduled job processes accumulated data | Latency acceptable, large volume | Hours | Nightly churn scores for all customers |
| **Real-time (API)** | REST API responds to individual requests | Instant decisions needed | Milliseconds | Fraud detection on each transaction |
| **Edge** | Model runs on device (mobile, IoT) | Offline needed, privacy critical | Instant | Medical diagnosis app without internet |
| **Streaming** | Process continuous event stream (Kafka) | High-volume, low-latency events | Seconds | Anomaly detection on sensor data |
