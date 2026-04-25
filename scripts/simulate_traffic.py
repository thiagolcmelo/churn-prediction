#!/usr/bin/env python3
"""
Traffic simulator for the Churn Prediction API.
Sends realistic requests over a configurable duration with varying patterns
to create interesting monitoring visualizations in Grafana.

Usage:
    python scripts/simulate_traffic.py                    # 10 min, 3 req/s
    python scripts/simulate_traffic.py --duration 300     # 5 min
    python scripts/simulate_traffic.py --duration 600 --rps 5  # 10 min, 5 req/s
"""

import argparse
import random
import sys
import time
from datetime import datetime
from typing import Any

import requests

API_URL = "http://localhost:8000"

# Realistic customer profiles based on Telco Churn dataset distributions
CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
YES_NO = ["Yes", "No"]
GENDERS = ["Male", "Female"]


def generate_customer() -> dict[str, Any]:
    """Generate a realistic customer profile using dataset distributions."""
    tenure = random.choices(
        [random.randint(0, 12), random.randint(13, 36), random.randint(37, 72)],
        weights=[0.4, 0.3, 0.3],
    )[0]
    monthly = round(random.gauss(64.0, 30.0), 2)
    monthly = max(18.0, min(monthly, 120.0))
    total = round(monthly * tenure + random.gauss(0, 50), 2)
    total = max(0.0, total)

    return {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": random.choices(CONTRACTS, weights=[0.55, 0.21, 0.24])[0],
        "InternetService": random.choices(
            INTERNET_SERVICES, weights=[0.34, 0.44, 0.22]
        )[0],
        "TechSupport": random.choice(YES_NO),
        "OnlineSecurity": random.choice(YES_NO),
        "gender": random.choice(GENDERS),
        "SeniorCitizen": random.choices([0, 1], weights=[0.84, 0.16])[0],
        "Partner": random.choice(YES_NO),
        "Dependents": random.choices(YES_NO, weights=[0.70, 0.30])[0],
        "PhoneService": random.choices(YES_NO, weights=[0.90, 0.10])[0],
        "MultipleLines": random.choice(YES_NO),
        "OnlineBackup": random.choice(YES_NO),
        "DeviceProtection": random.choice(YES_NO),
        "StreamingTV": random.choice(YES_NO),
        "StreamingMovies": random.choice(YES_NO),
        "PaperlessBilling": random.choices(YES_NO, weights=[0.60, 0.40])[0],
        "PaymentMethod": random.choices(
            PAYMENT_METHODS, weights=[0.34, 0.23, 0.22, 0.21]
        )[0],
    }


def generate_bad_request() -> Any:
    """Generate an intentionally malformed request to test error handling."""
    bad_types = [
        {},  # empty body
        {"tenure": -5, "MonthlyCharges": 70.0},  # negative tenure + missing fields
        {"tenure": "abc", "MonthlyCharges": "xyz"},  # wrong types
    ]
    return random.choice(bad_types)


def get_rps_for_phase(elapsed: float, duration: float, base_rps: float) -> float:
    """
    Vary request rate over time to create interesting Grafana patterns:
    Phase 1 (0-20%):   Ramp up from 0.5 to base_rps
    Phase 2 (20-50%):  Steady at base_rps
    Phase 3 (50-65%):  Traffic spike at 3x base_rps
    Phase 4 (65-80%):  Back to base_rps
    Phase 5 (80-100%): Cool down to 0.5
    """
    progress = elapsed / duration
    if progress < 0.20:
        return 0.5 + (base_rps - 0.5) * (progress / 0.20)
    elif progress < 0.50:
        return base_rps
    elif progress < 0.65:
        return base_rps * 3.0
    elif progress < 0.80:
        return base_rps
    else:
        return base_rps * (1 - (progress - 0.80) / 0.20) + 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate API traffic")
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Duration in seconds (default: 600 = 10 min)",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=3.0,
        help="Base requests per second (default: 3)",
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.05,
        help="Fraction of bad requests (default: 0.05 = 5%%)",
    )
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print("Churn API Traffic Simulator")
    print(f"{'=' * 60}")
    print(f"Target:     {API_URL}")
    print(f"Duration:   {args.duration}s ({args.duration / 60:.1f} min)")
    print(f"Base RPS:   {args.rps}")
    print(f"Error rate: {args.error_rate * 100:.0f}%")
    print(f"{'=' * 60}")

    # Verify API is running
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"API health: {r.json()}")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {API_URL}/health — {e}")
        print("Make sure the API is running (docker compose up or make run)")
        sys.exit(1)

    start_time = time.time()
    total_requests = 0
    total_errors = 0
    total_churn = 0
    latencies = []

    print(f"\nStarted at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'─' * 60}")

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                break

            current_rps = get_rps_for_phase(elapsed, args.duration, args.rps)
            sleep_time = 1.0 / current_rps if current_rps > 0 else 1.0

            # Decide if this is a bad request
            is_bad = random.random() < args.error_rate

            try:
                req_start = time.time()
                if is_bad:
                    payload = generate_bad_request()
                    r = requests.post(
                        f"{API_URL}/predict",
                        json=payload,
                        timeout=10,
                    )
                else:
                    customer = generate_customer()
                    r = requests.post(
                        f"{API_URL}/predict",
                        json=customer,
                        timeout=10,
                    )

                req_duration = (time.time() - req_start) * 1000  # ms
                latencies.append(req_duration)
                total_requests += 1

                if r.status_code == 200:
                    result = r.json()
                    if result.get("churn_prediction"):
                        total_churn += 1
                elif r.status_code in (400, 422):
                    total_errors += 1
                else:
                    total_errors += 1

            except requests.exceptions.RequestException:
                total_errors += 1
                total_requests += 1

            # Progress report every 50 requests
            if total_requests % 50 == 0 and total_requests > 0:
                avg_lat = sum(latencies[-50:]) / min(50, len(latencies))
                phase_progress = elapsed / args.duration * 100
                churn_rate = total_churn / max(1, total_requests - total_errors) * 100
                print(
                    f"  [{phase_progress:5.1f}%] "
                    f"Sent: {total_requests:>5} | "
                    f"Errors: {total_errors:>3} | "
                    f"Avg latency: {avg_lat:>6.1f}ms | "
                    f"RPS: {current_rps:.1f} | "
                    f"Churn rate: {churn_rate:.1f}%"
                )

            # Add jitter to sleep time for more realistic traffic
            jittered_sleep = sleep_time * random.uniform(0.5, 1.5)
            time.sleep(jittered_sleep)

    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user after {time.time() - start_time:.0f}s")

    # Final summary
    elapsed_total = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("Simulation Complete")
    print(f"{'=' * 60}")
    print(f"Duration:       {elapsed_total:.0f}s ({elapsed_total / 60:.1f} min)")
    print(f"Total requests: {total_requests}")
    print(
        f"Errors:         {total_errors} ({total_errors / max(1, total_requests) * 100:.1f}%)"
    )
    print(
        f"Churn preds:    {total_churn} ({total_churn / max(1, total_requests - total_errors) * 100:.1f}%)"
    )
    if latencies:
        latencies.sort()
        print(f"Latency p50:    {latencies[len(latencies) // 2]:.1f}ms")
        print(f"Latency p95:    {latencies[int(len(latencies) * 0.95)]:.1f}ms")
        print(f"Latency p99:    {latencies[int(len(latencies) * 0.99)]:.1f}ms")
    print(f"{'=' * 60}")
    print("\nCheck Grafana at http://localhost:3000 to see the results!")


if __name__ == "__main__":
    main()
