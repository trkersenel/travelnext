"""Ranking metrics for top-K recommendation.

All metrics take a ranked list of recommended destination ids and a set of
relevant (held-out) destination ids, and use binary relevance. They are written
from their textbook definitions and unit-tested against hand-computed values in
``tests/test_metrics.py`` -- no metric implementation is taken on trust.

Conventions
-----------
* ``recommended`` is ordered best-first and must not contain duplicates.
* An empty relevant set yields ``nan`` for every metric, so that users with no
  ground truth are excluded from means rather than silently scoring 0.
* ``Recall@K`` uses the full relevant set as denominator, so it is capped below
  1.0 when ``len(relevant) > k``. ``MAP@K`` normalises by ``min(len(relevant), k)``,
  which is the standard convention and keeps a perfect ranking at 1.0.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np

METRIC_NAMES: tuple[str, ...] = ("precision", "recall", "f1", "map", "mrr", "ndcg")


def _hits(recommended: Sequence[str], relevant: Set[str], k: int) -> np.ndarray:
    """Binary relevance vector for the top ``k`` recommendations."""
    top_k = recommended[:k]
    return np.fromiter((1.0 if item in relevant else 0.0 for item in top_k), dtype="float64", count=len(top_k))


def precision_at_k(recommended: Sequence[str], relevant: Set[str], k: int) -> float:
    """Fraction of the top-K recommendations that are relevant."""
    if not relevant or k <= 0:
        return float("nan")
    return float(_hits(recommended, relevant, k).sum() / k)


def recall_at_k(recommended: Sequence[str], relevant: Set[str], k: int) -> float:
    """Fraction of the relevant items that appear in the top K."""
    if not relevant or k <= 0:
        return float("nan")
    return float(_hits(recommended, relevant, k).sum() / len(relevant))


def f1_at_k(recommended: Sequence[str], relevant: Set[str], k: int) -> float:
    """Harmonic mean of precision@K and recall@K."""
    precision = precision_at_k(recommended, relevant, k)
    recall = recall_at_k(recommended, relevant, k)
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        return 0.0 if not math.isnan(precision) else float("nan")
    return float(2.0 * precision * recall / (precision + recall))


def average_precision_at_k(recommended: Sequence[str], relevant: Set[str], k: int) -> float:
    """Average precision, normalised by ``min(len(relevant), k)``."""
    if not relevant or k <= 0:
        return float("nan")
    hits = _hits(recommended, relevant, k)
    if hits.sum() == 0:
        return 0.0
    positions = np.arange(1, len(hits) + 1, dtype="float64")
    running_precision = np.cumsum(hits) / positions
    return float((running_precision * hits).sum() / min(len(relevant), k))


def reciprocal_rank(recommended: Sequence[str], relevant: Set[str], k: int | None = None) -> float:
    """Reciprocal of the rank of the first relevant item (0 if none in top K)."""
    if not relevant:
        return float("nan")
    limit = len(recommended) if k is None else min(k, len(recommended))
    for position in range(limit):
        if recommended[position] in relevant:
            return 1.0 / (position + 1)
    return 0.0


def ndcg_at_k(recommended: Sequence[str], relevant: Set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    if not relevant or k <= 0:
        return float("nan")
    hits = _hits(recommended, relevant, k)
    discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
    dcg = float((hits * discounts).sum())
    ideal_hits = min(len(relevant), k)
    idcg = float((1.0 / np.log2(np.arange(2, ideal_hits + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_user(
    recommended: Sequence[str],
    relevant: Iterable[str],
    k_values: Sequence[int],
) -> Dict[str, float]:
    """Compute every metric at every K for a single user.

    Returns a flat mapping such as ``{"precision@10": 0.2, "mrr@10": 0.5, ...}``.
    """
    relevant_set = set(relevant)
    scores: Dict[str, float] = {}
    for k in k_values:
        scores[f"precision@{k}"] = precision_at_k(recommended, relevant_set, k)
        scores[f"recall@{k}"] = recall_at_k(recommended, relevant_set, k)
        scores[f"f1@{k}"] = f1_at_k(recommended, relevant_set, k)
        scores[f"map@{k}"] = average_precision_at_k(recommended, relevant_set, k)
        scores[f"mrr@{k}"] = reciprocal_rank(recommended, relevant_set, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(recommended, relevant_set, k)
    return scores


def aggregate(per_user_scores: List[Dict[str, float]]) -> Dict[str, float]:
    """Average per-user metric dictionaries, ignoring NaN entries."""
    if not per_user_scores:
        return {}
    keys = sorted({key for scores in per_user_scores for key in scores})
    aggregated: Dict[str, float] = {}
    for key in keys:
        values = np.array(
            [scores.get(key, float("nan")) for scores in per_user_scores], dtype="float64"
        )
        finite = values[~np.isnan(values)]
        aggregated[key] = float(finite.mean()) if finite.size else float("nan")
    aggregated["n_users_evaluated"] = float(len(per_user_scores))
    return aggregated
