from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from .agents import ProcessAnalyticsAgent, EvidenceRetrievalAgent, RecommendationAgent


class TransformationPipeline:
    def __init__(self, policy_library, contamination: float = 0.10):
        self.analytics = ProcessAnalyticsAgent(contamination=contamination)
        self.retriever = EvidenceRetrievalAgent(policy_library)
        self.recommender = RecommendationAgent()

    def run(self, df: pd.DataFrame):
        result = self.analytics.analyse(df)
        out = df.copy()
        out["anomaly_score"] = result.scores
        out["predicted_anomaly"] = result.predicted

        flagged = out[out["predicted_anomaly"] == 1].copy().sort_values(
            "anomaly_score", ascending=False
        )
        evidence_topics = []
        evidence_titles = []
        recommendations = []
        for _, row in flagged.iterrows():
            query = f"{row['topic']} {row['case_note']}"
            evidence = self.retriever.retrieve(query, top_k=2)
            evidence_topics.append(evidence[0]["topic"])
            evidence_titles.append(evidence[0]["title"])
            recommendations.append(self.recommender.recommend(row, evidence))

        flagged["retrieved_topic"] = evidence_topics
        flagged["evidence_title"] = evidence_titles
        flagged["recommendation"] = recommendations
        return out, flagged

    @staticmethod
    def evaluate(all_cases: pd.DataFrame, flagged: pd.DataFrame):
        y_true = all_cases["is_injected_anomaly"].astype(int)
        y_pred = all_cases["predicted_anomaly"].astype(int)
        retrieval_acc = (
            float((flagged["topic"] == flagged["retrieved_topic"]).mean())
            if len(flagged)
            else 0.0
        )
        evidence_coverage = (
            float(flagged["evidence_title"].notna().mean()) if len(flagged) else 0.0
        )
        return {
            "anomaly_precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "anomaly_recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "anomaly_f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "retrieval_top1_topic_accuracy": retrieval_acc,
            "evidence_coverage": evidence_coverage,
            "flagged_cases": int(len(flagged)),
            "total_cases": int(len(all_cases)),
        }


def write_management_brief(flagged: pd.DataFrame, metrics: dict, path: Path, top_n: int = 8):
    lines = [
        "# AI Process Transformation - Management Brief",
        "",
        "## Executive summary",
        f"The pipeline analysed {metrics['total_cases']} operational cases and prioritised {metrics['flagged_cases']} for review.",
        f"On the labelled synthetic benchmark, anomaly detection achieved precision {metrics['anomaly_precision']:.2f}, recall {metrics['anomaly_recall']:.2f}, and F1 {metrics['anomaly_f1']:.2f}. Evidence retrieval matched the known issue topic in {metrics['retrieval_top1_topic_accuracy']:.0%} of flagged cases.",
        "",
        "## Highest-priority cases",
        "",
    ]
    for _, row in flagged.head(top_n).iterrows():
        lines.extend(
            [
                f"### {row['case_id']} - {row['topic']}",
                f"- Anomaly score: {row['anomaly_score']:.3f}",
                f"- Evidence: {row['evidence_title']}",
                f"- Recommendation: {row['recommendation']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
