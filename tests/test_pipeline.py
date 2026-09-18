import json
from pathlib import Path

from src.generate_data import generate_dataset
from src.pipeline import TransformationPipeline

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_produces_evaluable_output():
    policies = json.loads((ROOT / "data" / "policy_library.json").read_text())
    df = generate_dataset(n=300, seed=7)
    p = TransformationPipeline(policies, contamination=0.10)
    all_cases, flagged = p.run(df)
    metrics = p.evaluate(all_cases, flagged)
    assert len(flagged) > 0
    assert 0 <= metrics["anomaly_precision"] <= 1
    assert 0 <= metrics["retrieval_top1_topic_accuracy"] <= 1
    assert flagged["recommendation"].str.len().min() > 20
