"""Evaluation runner: turns fitted models into a comparable metrics table.

Protocol
--------
For each evaluable user we build a request from the interactions that precede
the held-out window, ask the model for ``max(k_values)`` destinations, and score
that ranking against the held-out destinations.

Two details matter for the numbers to mean anything:

**Context is treated as known at request time.** The month, trip duration and
budget of the held-out trip are passed to the model. This is not leakage of the
label: in the product the traveller states "two weeks in September on a mid
budget" *before* asking where to go. The destination -- the thing being
predicted -- is never revealed. Every model receives identical context, so the
comparison stays fair, and ``use_target_context=False`` reproduces the stricter
no-context setting.

**Already-visited destinations are excluded** from every model's output, since
recommending a place the traveller has just come from is not a valid answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.dataset import Split, TravelDataset
from src.evaluation.metrics import aggregate, evaluate_user
from src.models.base import BaseRecommender, RecommendationRequest
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Metrics plus the raw recommendations that produced them."""

    model_name: str
    stage: str
    metrics: Dict[str, float]
    recommendations: Dict[str, List[str]] = field(default_factory=dict)
    n_users: int = 0

    def row(self, k_values: Sequence[int]) -> Dict[str, object]:
        """Flatten into one row of the comparison table."""
        row: Dict[str, object] = {"model": self.model_name, "users": self.n_users}
        for k in k_values:
            for metric in ("precision", "recall", "f1", "map", "mrr", "ndcg"):
                key = f"{metric}@{k}"
                row[key] = self.metrics.get(key, float("nan"))
        return row


def build_requests(
    dataset: TravelDataset,
    split: Split,
    stage: str,
    *,
    use_target_context: bool = True,
) -> Dict[str, RecommendationRequest]:
    """Build one request per evaluable user for the given stage."""
    history_frame = split.history_for(stage)
    target_frame = split.targets_for(stage)

    histories = dataset.user_histories(history_frame)
    # The earliest held-out trip supplies the request context (see module docs).
    first_targets = (
        target_frame.sort_values(["user_id", "trip_index"])
        .groupby("user_id", sort=False)
        .first()
    )

    requests: Dict[str, RecommendationRequest] = {}
    for user_id in target_frame["user_id"].unique():
        history = histories.get(user_id, [])
        if not history:
            # No visible history: this user cannot be evaluated as a warm user.
            continue
        request = RecommendationRequest(
            history=history[:-1],
            current_destination=history[-1],
            user_id=str(user_id),
        )
        if use_target_context and user_id in first_targets.index:
            target = first_targets.loc[user_id]
            request.month = int(target["month"])
            request.trip_duration_days = int(target["trip_duration_days"])
            request.budget = str(target["budget"])
        requests[str(user_id)] = request
    return requests


def evaluate_model(
    model: BaseRecommender,
    dataset: TravelDataset,
    split: Split,
    *,
    stage: str = "test",
    k_values: Sequence[int] = (5, 10, 20),
    use_target_context: bool = True,
    max_users: Optional[int] = None,
    seed: int = 42,
) -> EvaluationResult:
    """Evaluate one fitted model and return metrics plus recommendations."""
    target_frame = split.targets_for(stage)
    ground_truth = (
        target_frame.groupby("user_id")["destination_id"].apply(set).to_dict()
    )
    requests = build_requests(dataset, split, stage, use_target_context=use_target_context)

    user_ids = sorted(requests)
    if max_users is not None and len(user_ids) > max_users:
        # Deterministic subsample keeps repeated runs comparable.
        rng = np.random.default_rng(seed)
        user_ids = sorted(rng.choice(user_ids, size=max_users, replace=False).tolist())

    top_n = max(k_values)
    per_user: List[Dict[str, float]] = []
    recommendations: Dict[str, List[str]] = {}

    for user_id in user_ids:
        ranked = model.recommend_ids(requests[user_id], k=top_n)
        recommendations[user_id] = ranked
        per_user.append(evaluate_user(ranked, ground_truth.get(user_id, set()), k_values))

    metrics = aggregate(per_user)
    LOGGER.info(
        "%-22s %s ndcg@10=%.4f recall@10=%.4f (%d users)",
        model.name,
        stage,
        metrics.get("ndcg@10", float("nan")),
        metrics.get("recall@10", float("nan")),
        len(user_ids),
    )
    return EvaluationResult(
        model_name=model.name,
        stage=stage,
        metrics=metrics,
        recommendations=recommendations,
        n_users=len(user_ids),
    )


def comparison_table(
    results: Sequence[EvaluationResult], k_values: Sequence[int] = (5, 10, 20)
) -> pd.DataFrame:
    """Assemble evaluation results into the model comparison table."""
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame([result.row(k_values) for result in results])
    sort_key = f"ndcg@{k_values[len(k_values) // 2]}"
    if sort_key in frame.columns:
        frame = frame.sort_values(sort_key, ascending=False)
    return frame.reset_index(drop=True)
