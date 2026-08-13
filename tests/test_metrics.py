"""Metric tests against values computed by hand.

Every number in these assertions was derived on paper from the metric's
definition. If an implementation is quietly wrong, the reported model
comparison is worthless, so these are the highest-value tests in the project.
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import (
    aggregate,
    average_precision_at_k,
    evaluate_user,
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Ranked list with relevant items at positions 1 and 3 (1-indexed).
RECOMMENDED = ["a", "b", "c", "d", "e"]
RELEVANT = {"a", "c", "z"}  # "z" is relevant but never recommended


def test_precision_at_k() -> None:
    assert precision_at_k(RECOMMENDED, RELEVANT, 1) == pytest.approx(1.0)
    assert precision_at_k(RECOMMENDED, RELEVANT, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(2 / 5)


def test_recall_at_k_uses_full_relevant_set() -> None:
    # 3 relevant items exist; 2 are retrieved in the top 5.
    assert recall_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RECOMMENDED, RELEVANT, 1) == pytest.approx(1 / 3)


def test_f1_is_harmonic_mean() -> None:
    precision = precision_at_k(RECOMMENDED, RELEVANT, 5)
    recall = recall_at_k(RECOMMENDED, RELEVANT, 5)
    expected = 2 * precision * recall / (precision + recall)
    assert f1_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(expected)


def test_average_precision_at_k() -> None:
    # Hits at ranks 1 and 3 -> precisions 1/1 and 2/3.
    # Normaliser is min(|relevant|=3, k=5) = 3.
    expected = (1.0 + 2 / 3) / 3
    assert average_precision_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(expected)


def test_average_precision_perfect_ranking_is_one() -> None:
    assert average_precision_at_k(["a", "b"], {"a", "b"}, 5) == pytest.approx(1.0)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RECOMMENDED, RELEVANT, 5) == pytest.approx(1.0)
    assert reciprocal_rank(["x", "y", "a"], {"a"}, 5) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y"], {"a"}, 5) == pytest.approx(0.0)


def test_ndcg_at_k() -> None:
    # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5
    # IDCG (2 achievable hits within k=5, 3 relevant -> min(3,5)=3 ideal slots)
    #      = 1/log2(2) + 1/log2(3) + 1/log2(4)
    dcg = 1.0 + 1.0 / math.log2(4)
    idcg = 1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    assert ndcg_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)


def test_no_relevant_items_yields_nan() -> None:
    scores = evaluate_user(RECOMMENDED, [], [5])
    assert all(math.isnan(value) for value in scores.values())


def test_k_larger_than_recommendation_list() -> None:
    # Asking for 20 results when only 5 exist must not crash or index past end.
    scores = evaluate_user(RECOMMENDED, RELEVANT, [20])
    assert scores["precision@20"] == pytest.approx(2 / 20)
    assert scores["recall@20"] == pytest.approx(2 / 3)


def test_empty_recommendations_score_zero() -> None:
    scores = evaluate_user([], RELEVANT, [10])
    assert scores["precision@10"] == pytest.approx(0.0)
    assert scores["recall@10"] == pytest.approx(0.0)
    assert scores["ndcg@10"] == pytest.approx(0.0)
    assert scores["mrr@10"] == pytest.approx(0.0)


def test_aggregate_ignores_nan_users() -> None:
    aggregated = aggregate(
        [
            {"precision@5": 1.0},
            {"precision@5": float("nan")},
            {"precision@5": 0.0},
        ]
    )
    assert aggregated["precision@5"] == pytest.approx(0.5)
    assert aggregated["n_users_evaluated"] == 3.0


def test_aggregate_empty_input() -> None:
    assert aggregate([]) == {}
