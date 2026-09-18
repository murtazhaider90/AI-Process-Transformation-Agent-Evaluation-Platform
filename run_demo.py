from __future__ import annotations
import json
from pathlib import Path

from src.generate_data import generate_dataset
from src.pipeline import TransformationPipeline, write_management_brief

ROOT = Path(__file__).resolve().parent


def main():
    with open(ROOT / "data" / "policy_library.json", "r", encoding="utf-8") as f:
        policies = json.load(f)

    df = generate_dataset(n=800, seed=42)
    pipeline = TransformationPipeline(policies, contamination=0.10)
    all_cases, flagged = pipeline.run(df)
    metrics = pipeline.evaluate(all_cases, flagged)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    all_cases.to_csv(out_dir / "all_cases.csv", index=False)
    flagged.to_csv(out_dir / "flagged_cases.csv", index=False)
    with open(out_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_management_brief(flagged, metrics, out_dir / "management_brief.md")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
