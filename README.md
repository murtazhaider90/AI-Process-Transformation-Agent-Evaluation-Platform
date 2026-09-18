# AI Process Transformation & Agent Evaluation Platform

A compact AI-engineering portfolio project that turns operational data and process-policy text into evidence-backed transformation recommendations, then evaluates the pipeline instead of treating automated outputs as automatically trustworthy.

The project is designed around an internal-consulting use case: identify operational problems, retrieve relevant controls or process guidance, recommend an intervention, and produce stakeholder-ready output with measurable evaluation.

## What it does

1. Generates a labelled synthetic operational dataset with realistic process metrics and injected failure modes.
2. Detects unusual process cases using an `IsolationForest`.
3. Retrieves relevant process/policy evidence using TF-IDF similarity.
4. Routes flagged cases through lightweight specialist components for analytics, evidence retrieval and recommendation.
5. Produces a management brief prioritising high-impact issues and suggested actions.
6. Evaluates anomaly precision/recall/F1, evidence-retrieval accuracy and evidence coverage on known labels.

The project intentionally uses synthetic data so it is fully reproducible and contains no confidential or personal information.

## Demo results

Using the default reproducible benchmark of 800 operational cases:

| Metric | Result |
| --- | ---: |
| Anomaly precision | 0.825 |
| Anomaly recall | 0.825 |
| Anomaly F1 | 0.825 |
| Top-1 evidence topic accuracy | 1.000 |
| Evidence coverage | 1.000 |
| Cases prioritised for review | 80 / 800 |

These numbers are benchmark results on deliberately generated synthetic data and should not be interpreted as production performance.

## Why it matters

Enterprise AI and internal consulting need more than a model prediction. Useful systems have to connect data analysis, evidence, recommendations, evaluation and stakeholder communication.

This project demonstrates that end-to-end loop:

```text
Operational data
      |
      v
Anomaly detection
      |
      v
Evidence retrieval
      |
      v
Recommendation
      |
      v
Evaluation + management brief
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_demo.py
pytest
```

Generated outputs are written to `outputs/`.

## Stack

- Python
- pandas / NumPy
- scikit-learn
- Isolation Forest
- TF-IDF retrieval + cosine similarity
- pytest

## Project structure

```text
run_demo.py
src/
  __init__.py
  generate_data.py
  agents.py
  pipeline.py
data/
  policy_library.json
outputs/
  evaluation.json
  management_brief.md
tests/
  test_pipeline.py
```

## Design notes

The current version deliberately keeps the workflow local and deterministic: anomaly detection uses scikit-learn, evidence retrieval uses TF-IDF, and recommendations are mapped to transparent process-improvement actions. This makes the evaluation behaviour inspectable and reproducible. A natural extension would be to replace or augment the recommendation stage with an LLM while preserving the same evaluation and evidence-grounding framework.
