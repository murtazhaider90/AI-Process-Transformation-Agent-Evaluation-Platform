from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FEATURES = ["cycle_time_hours", "handoffs", "rework_rate", "error_count", "missing_data_rate"]


@dataclass
class AnomalyResult:
    scores: np.ndarray
    predicted: np.ndarray


class ProcessAnalyticsAgent:
    def __init__(self, contamination: float = 0.10, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=random_state,
        )

    def analyse(self, df: pd.DataFrame) -> AnomalyResult:
        X = df[FEATURES].astype(float)
        pred_raw = self.model.fit_predict(X)
        predicted = (pred_raw == -1).astype(int)
        # Larger is more anomalous for easy ranking.
        scores = -self.model.score_samples(X)
        return AnomalyResult(scores=scores, predicted=predicted)


class EvidenceRetrievalAgent:
    def __init__(self, library: List[Dict[str, str]]):
        self.library = library
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(
            [f"{x['title']} {x['text']}" for x in library]
        )

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, str]]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        idx = np.argsort(sims)[::-1][:top_k]
        return [dict(self.library[i], score=float(sims[i])) for i in idx]


class RecommendationAgent:
    ACTIONS = {
        "sla_delay": "Map queue ageing and bottlenecks, introduce ageing-based prioritisation, and test whether ownership changes reduce SLA breaches.",
        "rework": "Trace repeat defects to the earliest controllable step, add targeted validation, and monitor first-pass yield after the change.",
        "handoffs": "Reduce duplicate approvals and ownership changes; define a simpler accountable process and measure cycle-time impact.",
        "automation": "Automate repetitive rules-based steps with explicit exception handling, audit logging and post-deployment monitoring.",
        "data_quality": "Add source-level validation, field ownership and exception reporting before downstream analytics consume the data.",
        "incident": "Run structured root-cause analysis, assign remediation owners and verify recurrence falls after corrective action.",
    }

    def recommend(self, row: pd.Series, evidence: List[Dict[str, str]]) -> str:
        topic = row["topic"]
        evidence_title = evidence[0]["title"] if evidence else "process-control evidence"
        return f"{self.ACTIONS[topic]} Evidence basis: {evidence_title}."
