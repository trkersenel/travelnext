"""Tests for the evaluation runner, the ranker and the beyond-accuracy analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.beyond_accuracy import analyse_recommendations, gini_coefficient, most_recommended
from src.candidate_generation.generator import CandidateGenerator, candidate_recall
from src.data.dataset import TravelDataset
from src.evaluation.evaluate import build_requests, comparison_table, evaluate_model
from src.models.collaborative import ItemItemCFRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.context import ContextScorer
from src.models.hybrid import HybridRecommender
from src.models.popularity import PopularityRecommender
from src.models.tune_hybrid import tune_weights
from src.ranking.features import RankingFeatureBuilder, build_training_data
from src.ranking.ltr import LearningToRankRecommender, requests_from_interactions


@pytest.fixture(scope="module")
def components(dataset: TravelDataset, split):
    popularity = PopularityRecommender().fit(dataset, split.train)
    content = ContentBasedRecommender().fit(dataset, split.train)
    collaborative = ItemItemCFRecommender().fit(dataset, split.train)
    return popularity, content, collaborative


@pytest.fixture(scope="module")
def generator(dataset: TravelDataset, components):
    popularity, content, collaborative = components
    return CandidateGenerator(dataset, content, collaborative, popularity, n_candidates=20)


@pytest.fixture(scope="module")
def builder(dataset: TravelDataset, components):
    popularity, content, collaborative = components
    return RankingFeatureBuilder(
        dataset, content, collaborative, popularity, ContextScorer(dataset)
    )


# ------------------------------------------------------------------ requests
def test_build_requests_excludes_target_from_history(dataset: TravelDataset, split) -> None:
    requests = build_requests(dataset, split, "test")
    truth = split.test.groupby("user_id")["destination_id"].apply(set).to_dict()
    for user_id, request in list(requests.items())[:25]:
        assert request.visited().isdisjoint(truth.get(user_id, set()))


def test_build_requests_attaches_context(dataset: TravelDataset, split) -> None:
    request = next(iter(build_requests(dataset, split, "test").values()))
    assert request.month is not None
    assert request.budget is not None


def test_build_requests_without_context(dataset: TravelDataset, split) -> None:
    request = next(iter(build_requests(dataset, split, "test", use_target_context=False).values()))
    assert request.month is None


# ---------------------------------------------------------------- evaluation
def test_evaluate_model_produces_all_metrics(dataset: TravelDataset, split, components) -> None:
    popularity = components[0]
    result = evaluate_model(popularity, dataset, split, stage="test", k_values=[5, 10], max_users=40)
    for key in ("precision@5", "recall@10", "map@10", "mrr@10", "ndcg@10", "f1@5"):
        assert key in result.metrics
        assert not np.isnan(result.metrics[key])
    assert result.n_users > 0


def test_metrics_are_within_valid_ranges(dataset: TravelDataset, split, components) -> None:
    result = evaluate_model(components[0], dataset, split, k_values=[10], max_users=40)
    for key, value in result.metrics.items():
        if key == "n_users_evaluated":
            continue
        assert 0.0 <= value <= 1.0, f"{key}={value}"


def test_evaluation_is_deterministic(dataset: TravelDataset, split, components) -> None:
    first = evaluate_model(components[0], dataset, split, k_values=[10], max_users=30, seed=1)
    second = evaluate_model(components[0], dataset, split, k_values=[10], max_users=30, seed=1)
    assert first.metrics["ndcg@10"] == pytest.approx(second.metrics["ndcg@10"])


def test_comparison_table_shape(dataset: TravelDataset, split, components) -> None:
    popularity, content, _ = components
    results = [
        evaluate_model(model, dataset, split, k_values=[5, 10], max_users=30)
        for model in (popularity, content)
    ]
    table = comparison_table(results, [5, 10])
    assert len(table) == 2
    assert {"model", "precision@5", "ndcg@10"} <= set(table.columns)


def test_comparison_table_handles_no_results() -> None:
    assert comparison_table([]).empty


# --------------------------------------------------------- candidate recall
def test_candidate_recall_is_a_fraction(dataset: TravelDataset, split, generator) -> None:
    requests = build_requests(dataset, split, "test")
    truth = split.test.groupby("user_id")["destination_id"].apply(set).to_dict()
    sample = dict(list(requests.items())[:40])
    recall = candidate_recall(generator, sample, truth)
    assert 0.0 <= recall <= 1.0


# ------------------------------------------------------------------- ranker
def test_requests_from_interactions_uses_last_trip_as_target(
    dataset: TravelDataset, split
) -> None:
    pairs = requests_from_interactions(dataset, split.train)
    assert pairs
    for request, target in pairs[:20]:
        assert target not in request.visited()


def test_training_data_has_one_positive_per_group(
    dataset: TravelDataset, builder, generator, split
) -> None:
    pairs = requests_from_interactions(dataset, split.train)[:40]
    features, labels, groups = build_training_data(
        dataset, builder, generator, pairs, negatives_per_positive=8, seed=3
    )
    assert features.shape[0] == labels.shape[0] == groups.sum()
    offset = 0
    for size in groups:
        assert labels[offset : offset + size].sum() == 1
        offset += size


def test_feature_matrix_is_finite(dataset: TravelDataset, builder, generator) -> None:
    from src.models.base import RecommendationRequest

    request = RecommendationRequest(
        history=["amsterdam-nl"], current_destination="berlin-de", month=6, budget="budget"
    )
    matrix = builder.build(request, generator.generate(request))
    assert np.isfinite(matrix).all()
    assert matrix.shape[1] == len(builder.feature_names)


def test_feature_builder_handles_cold_start(dataset: TravelDataset, builder, generator) -> None:
    from src.models.base import RecommendationRequest

    request = RecommendationRequest()
    matrix = builder.build(request, generator.generate(request))
    assert np.isfinite(matrix).all()


def test_ranker_trains_and_ranks(dataset: TravelDataset, builder, generator, split) -> None:
    pairs = requests_from_interactions(dataset, split.train)
    ranker = LearningToRankRecommender(
        generator, builder, params={"n_estimators": 30, "num_leaves": 7}, negatives_per_positive=8
    )
    ranker.fit_ranker(dataset, pairs)

    from src.models.base import RecommendationRequest

    results = ranker.rank_candidates(
        RecommendationRequest(history=["amsterdam-nl"], current_destination="berlin-de"), k=5
    )
    assert len(results) == 5
    assert [item.rank for item in results] == [1, 2, 3, 4, 5]
    # Feature values travel with the result so explanations can use them.
    assert results[0].components


def test_ranker_feature_importance(dataset: TravelDataset, builder, generator, split) -> None:
    pairs = requests_from_interactions(dataset, split.train)
    ranker = LearningToRankRecommender(
        generator, builder, params={"n_estimators": 25, "num_leaves": 7}, negatives_per_positive=8
    )
    ranker.fit_ranker(dataset, pairs)
    importance = ranker.feature_importance()
    assert set(importance["feature"]) == set(builder.feature_names)
    assert importance["gain_share"].sum() == pytest.approx(1.0, abs=1e-6)


def test_ranker_excludes_visited(dataset: TravelDataset, builder, generator, split) -> None:
    from src.models.base import RecommendationRequest

    pairs = requests_from_interactions(dataset, split.train)
    ranker = LearningToRankRecommender(
        generator, builder, params={"n_estimators": 25, "num_leaves": 7}, negatives_per_positive=8
    )
    ranker.fit_ranker(dataset, pairs)
    history = ["amsterdam-nl", "berlin-de"]
    ranked = ranker.recommend_ids(RecommendationRequest(history=history), k=10)
    assert set(ranked).isdisjoint(set(history))


def test_ranker_save_and_load(tmp_path, dataset: TravelDataset, builder, generator, split) -> None:
    pairs = requests_from_interactions(dataset, split.train)
    ranker = LearningToRankRecommender(
        generator, builder, params={"n_estimators": 20, "num_leaves": 7}, negatives_per_positive=8
    )
    ranker.fit_ranker(dataset, pairs)
    path = tmp_path / "ranker.joblib"
    ranker.save(path)

    reloaded = LearningToRankRecommender(generator, builder)
    reloaded.dataset = dataset
    reloaded.load(path)

    from src.models.base import RecommendationRequest

    request = RecommendationRequest(history=["amsterdam-nl"])
    assert ranker.recommend_ids(request, k=5) == reloaded.recommend_ids(request, k=5)


def test_ranker_fit_via_base_interface_is_rejected(dataset: TravelDataset, builder, generator, split) -> None:
    ranker = LearningToRankRecommender(generator, builder)
    with pytest.raises(NotImplementedError):
        ranker.fit(dataset, split.train)


# ---------------------------------------------------------------- tuning
def test_tune_weights_returns_valid_weights(dataset: TravelDataset, split, components) -> None:
    popularity, content, collaborative = components
    hybrid = HybridRecommender(
        content=content, collaborative=collaborative, popularity=popularity
    ).fit(dataset, split.train)
    best, table = tune_weights(hybrid, dataset, split, k=10, step=0.5, max_users=40)
    total = best.content + best.collaborative + best.popularity + best.context
    assert total == pytest.approx(1.0, abs=1e-6)
    assert not table.empty
    assert table["ndcg@10"].iloc[0] >= table["ndcg@10"].iloc[-1]


# ------------------------------------------------------- beyond accuracy
def test_gini_of_uniform_exposure_is_zero() -> None:
    assert gini_coefficient(np.ones(50)) == pytest.approx(0.0, abs=1e-9)


def test_gini_of_single_item_is_high() -> None:
    counts = np.zeros(100)
    counts[0] = 100
    assert gini_coefficient(counts) > 0.95


def test_gini_of_empty_exposure() -> None:
    assert gini_coefficient(np.zeros(10)) == 0.0


def test_beyond_accuracy_report_ranges(dataset: TravelDataset, split, components) -> None:
    result = evaluate_model(components[0], dataset, split, k_values=[10], max_users=50)
    report = analyse_recommendations(dataset, result.recommendations, model_name="popularity", k=10)
    assert 0.0 < report.catalog_coverage <= 1.0
    assert 0.0 <= report.gini <= 1.0
    assert 0.0 <= report.mean_popularity_percentile <= 1.0
    assert report.geographic_spread_km > 0


def test_popularity_baseline_has_low_coverage(dataset: TravelDataset, split, components) -> None:
    """A model that ranks everyone identically should touch little of the catalog."""
    popularity_result = evaluate_model(components[0], dataset, split, k_values=[10], max_users=60)
    content_result = evaluate_model(components[1], dataset, split, k_values=[10], max_users=60)

    popularity_report = analyse_recommendations(
        dataset, popularity_result.recommendations, model_name="popularity", k=10
    )
    content_report = analyse_recommendations(
        dataset, content_result.recommendations, model_name="content", k=10
    )
    assert popularity_report.catalog_coverage <= content_report.catalog_coverage


def test_most_recommended_lists_cities(dataset: TravelDataset, split, components) -> None:
    result = evaluate_model(components[0], dataset, split, k_values=[10], max_users=40)
    frame = most_recommended(dataset, result.recommendations, k=10, top=5)
    assert len(frame) <= 5
    assert {"city", "country", "times_recommended"} <= set(frame.columns)


def test_analyse_empty_recommendations(dataset: TravelDataset) -> None:
    report = analyse_recommendations(dataset, {}, model_name="none", k=10)
    assert report.catalog_coverage == 0.0
    assert report.n_unique_recommended == 0
