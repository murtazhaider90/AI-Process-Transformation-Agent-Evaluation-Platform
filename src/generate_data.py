from __future__ import annotations
import numpy as np
import pandas as pd

TOPICS = ["sla_delay", "rework", "handoffs", "automation", "data_quality", "incident"]


def generate_dataset(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    topics = rng.choice(TOPICS, n, p=[0.22, 0.18, 0.16, 0.14, 0.16, 0.14])
    base_cycle = rng.gamma(3.2, 2.0, n) + 3
    handoffs = rng.poisson(2.2, n) + 1
    rework = np.clip(rng.beta(1.5, 9, n), 0, 1)
    errors = rng.poisson(0.7, n)
    missing = np.clip(rng.beta(1.2, 18, n), 0, 1)
    volume = rng.integers(20, 500, n)

    # Inject a controlled set of operational anomalies for evaluation.
    anomaly_idx = rng.choice(n, size=max(60, n // 10), replace=False)
    labels = np.zeros(n, dtype=int)
    labels[anomaly_idx] = 1

    for i in anomaly_idx:
        topic = topics[i]
        if topic == "sla_delay":
            base_cycle[i] *= rng.uniform(2.3, 4.0)
        elif topic == "rework":
            rework[i] = rng.uniform(0.42, 0.80)
        elif topic == "handoffs":
            handoffs[i] += rng.integers(6, 12)
        elif topic == "automation":
            errors[i] += rng.integers(4, 9)
            base_cycle[i] *= rng.uniform(1.5, 2.2)
        elif topic == "data_quality":
            missing[i] = rng.uniform(0.25, 0.65)
        else:
            errors[i] += rng.integers(6, 12)

    notes = [
        {
            "sla_delay": "Case ageing and queue congestion are driving missed service targets.",
            "rework": "Repeated corrections are reducing first-pass quality and extending completion time.",
            "handoffs": "Multiple ownership changes and approvals are creating avoidable process friction.",
            "automation": "Manual repetitive steps and exception handling are limiting throughput.",
            "data_quality": "Missing and inconsistent source fields are reducing confidence in downstream analysis.",
            "incident": "Recurring operational incidents require structured root-cause investigation and remediation.",
        }[t]
        for t in topics
    ]

    return pd.DataFrame(
        {
            "case_id": [f"CASE-{i:04d}" for i in range(n)],
            "topic": topics,
            "cycle_time_hours": np.round(base_cycle, 2),
            "handoffs": handoffs,
            "rework_rate": np.round(rework, 3),
            "error_count": errors,
            "missing_data_rate": np.round(missing, 3),
            "monthly_volume": volume,
            "case_note": notes,
            "is_injected_anomaly": labels,
        }
    )
