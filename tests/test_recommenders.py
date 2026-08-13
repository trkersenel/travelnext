"""Behavioural tests for every recommender.

The emphasis is on the contracts that must hold for the system to be usable in
production: no crash on cold start, no crash on unknown ids, never recommending
somewhere the traveller has already been, and sane behaviour when K exceeds the
number of available destinations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.candidate_generation.generator import CandidateGenerator
from src.data.dataset import TravelDataset
from src.models.base import RecommendationRequest, normalise_scores
from src.models.cold_start import ColdStartRecommender, filter_known_destinations, is_cold_start
from src.models.collaborative import ItemItemCFRecommender, MatrixFactorizationRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.context import ContextScorer
from src.models.hybrid import HybridRecommender, HybridWeights
from src.models.next_destination import NextDestinationRecommender, build_transition_matrix
from src.models.popularity import PopularityRecommender


@pytest.fixture(scope="module")
def fitted_models(dataset: TravelDataset, split):
    """All component models fitted on the training split."""
    train = split.train
    popularity = PopularityRecommender().fit(dataset, train)
    content = ContentBasedRecommender().fit(dataset, train)
    collaborative = ItemItemCFRecommender().fit(dataset, train)
    hybrid = HybridRecommender(
        content=content, collaborative=collaborative, popularity=popularity
    ).fit(dataset, train)
    next_destination = NextDestinationRecommender(
        content=content, collaborative=collaborative, popularity=popularity
    ).fit(dataset, train)
    cold = ColdStartRecommender(popularity, content, ContextScorer(dataset)).fit(dataset, train)
    return {
        "popularity": popularity,
        "content": content,
        "collaborative": collaborative,
        "hybrid": hybrid,
        "next_destination": next_destination,
        "cold_start": cold,
    }


ALL_MODELS = ["popularity", "content", "collaborative", "hybrid", "next_destination", "cold_start"]


# --------------------------------------------------------------- basic shape
@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_score_vector_covers_catalog(fitted_models, dataset, model_name) -> None:
    model = fitted_models[model_name]
    request = RecommendationRequest(history=["amsterdam-nl"], current_destination="berlin-de")
    scores = model.score(request)
    assert scores.shape == (dataset.n_destinations,)
    assert np.isfinite(scores).all() or model_name == "learning_to_rank"


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_recommend_returns_requested_count(fitted_models, model_name) -> None:
    model = fitted_models[model_name]
    request = RecommendationRequest(history=["amsterdam-nl", "berlin-de"])
    results = model.recommend(request, k=5)
    assert len(results) == 5
    assert [item.rank for item in results] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_scores_are_monotonically_ranked(fitted_models, model_name) -> None:
    model = fitted_models[model_name]
    results = model.recommend(RecommendationRequest(history=["prague-cz"]), k=8)
    scores = [item.score for item in results]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------ filtering
@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_visited_destinations_are_never_recommended(fitted_models, model_name) -> None:
    history = ["amsterdam-nl", "berlin-de", "prague-cz"]
    request = RecommendationRequest(history=history, current_destination="vienna-at")
    results = fitted_models[model_name].recommend(request, k=15)
    returned = {item.destination_id for item in results}
    assert returned.isdisjoint(set(history) | {"vienna-at"})


def test_explicit_exclusions_are_respected(fitted_models) -> None:
    request = RecommendationRequest(history=["amsterdam-nl"], exclude=["rotterdam-nl", "utrecht-nl"])
    results = fitted_models["hybrid"].recommend(request, k=10)
    returned = {item.destination_id for item in results}
    assert "rotterdam-nl" not in returned and "utrecht-nl" not in returned


def test_candidate_restriction_limits_output(fitted_models, dataset) -> None:
    allowed = ["rome-it", "vienna-at", "budapest-hu"]
    results = fitted_models["hybrid"].recommend(
        RecommendationRequest(history=["prague-cz"]), k=10, candidate_ids=allowed
    )
    assert {item.destination_id for item in results} <= set(allowed)


# ---------------------------------------------------------------- edge cases
@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_k_larger_than_catalog_returns_everything_available(
    fitted_models, dataset, model_name
) -> None:
    history = ["amsterdam-nl"]
    results = fitted_models[model_name].recommend(
        RecommendationRequest(history=history), k=dataset.n_destinations + 50
    )
    # Everything except the single visited destination.
    assert len(results) == dataset.n_destinations - 1


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_cold_start_user_does_not_crash(fitted_models, model_name) -> None:
    results = fitted_models[model_name].recommend(RecommendationRequest(), k=10)
    assert len(results) == 10


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_unknown_destination_in_history_is_ignored(fitted_models, model_name) -> None:
    request = RecommendationRequest(history=["atlantis-xx", "amsterdam-nl"])
    results = fitted_models[model_name].recommend(request, k=5)
    assert len(results) == 5


def test_unknown_only_history_behaves_as_cold_start(fitted_models, dataset) -> None:
    request = RecommendationRequest(history=["atlantis-xx", "el-dorado-xx"])
    assert is_cold_start(dataset, request)
    results = fitted_models["cold_start"].recommend(request, k=5)
    assert len(results) == 5


def test_filter_known_destinations_separates_ids(dataset) -> None:
    known, unknown = filter_known_destinations(dataset, ["amsterdam-nl", "atlantis-xx"])
    assert known == ["amsterdam-nl"]
    assert unknown == ["atlantis-xx"]


def test_k_of_zero_returns_empty(fitted_models) -> None:
    assert fitted_models["hybrid"].recommend(RecommendationRequest(), k=0) == []


def test_using_a_model_before_fitting_raises(dataset) -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        PopularityRecommender().recommend(RecommendationRequest(), k=5)


# ------------------------------------------------------------- model content
def test_popularity_uses_training_interactions_not_the_proxy(dataset, split) -> None:
    model = PopularityRecommender().fit(dataset, split.train)
    observed = split.train["destination_id"].value_counts()
    top_observed = observed.index[0]
    ranked = model.recommend_ids(RecommendationRequest(), k=3)
    assert top_observed in ranked


def test_popularity_is_identical_for_every_user(fitted_models) -> None:
    model = fitted_models["popularity"]
    first = model.recommend_ids(RecommendationRequest(history=["rome-it"]), k=5)
    second = model.recommend_ids(RecommendationRequest(history=["tokyo-jp"]), k=5)
    # Only the excluded history differs, so the shared entries must agree.
    assert set(first) & set(second)


def test_content_model_finds_similar_cities(fitted_models) -> None:
    similar = dict(fitted_models["content"].similar_to("amsterdam-nl", k=6))
    # The Dutch/Belgian culture cluster in the fixture should surface.
    assert set(similar) & {"rotterdam-nl", "utrecht-nl", "antwerp-be", "brussels-be"}


def test_content_model_handles_interests_without_history(fitted_models) -> None:
    beach_request = RecommendationRequest(interests=["beaches"])
    ranked = fitted_models["content"].recommend_ids(beach_request, k=6)
    assert {"barcelona-es", "valencia-es", "lisbon-pt", "sydney-au"} & set(ranked)


def test_collaborative_returns_zero_scores_without_history(fitted_models, dataset) -> None:
    scores = fitted_models["collaborative"].score(RecommendationRequest())
    assert np.allclose(scores, 0.0)


def test_matrix_factorization_fits_and_ranks(dataset, split) -> None:
    model = MatrixFactorizationRecommender(n_components=8).fit(dataset, split.train)
    results = model.recommend(RecommendationRequest(history=["berlin-de"]), k=5)
    assert len(results) == 5


# ------------------------------------------------------------------- hybrid
def test_hybrid_weights_normalise_to_one() -> None:
    weights = HybridWeights(content=2.0, collaborative=2.0, popularity=1.0, context=0.0)
    normalised = weights.normalised()
    total = (
        normalised.content
        + normalised.collaborative
        + normalised.popularity
        + normalised.context
    )
    assert total == pytest.approx(1.0)


def test_all_zero_hybrid_weights_fall_back_to_popularity() -> None:
    normalised = HybridWeights(0.0, 0.0, 0.0, 0.0).normalised()
    assert normalised.popularity == pytest.approx(1.0)


def test_hybrid_with_only_popularity_weight_matches_popularity(dataset, split) -> None:
    popularity = PopularityRecommender().fit(dataset, split.train)
    hybrid = HybridRecommender(
        popularity=popularity, weights=HybridWeights(0.0, 0.0, 1.0, 0.0)
    ).fit(dataset, split.train)
    request = RecommendationRequest(history=["amsterdam-nl"])
    assert hybrid.recommend_ids(request, k=5) == popularity.recommend_ids(request, k=5)


def test_hybrid_component_scores_are_normalised(fitted_models) -> None:
    components = fitted_models["hybrid"].component_scores(
        RecommendationRequest(history=["amsterdam-nl", "berlin-de"])
    )
    for name, values in components.items():
        assert values.min() >= 0.0 - 1e-9, name
        assert values.max() <= 1.0 + 1e-9, name


# -------------------------------------------------------- next destination
def test_transition_matrix_counts_consecutive_trips_only(dataset) -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b", "b"],
            "destination_id": [
                "amsterdam-nl",
                "berlin-de",
                "prague-cz",
                "rome-it",
                "amsterdam-nl",
            ],
            "trip_index": [0, 1, 2, 0, 1],
        }
    )
    matrix = build_transition_matrix(dataset, interactions)
    amsterdam = dataset.index_of["amsterdam-nl"]
    berlin = dataset.index_of["berlin-de"]
    prague = dataset.index_of["prague-cz"]
    rome = dataset.index_of["rome-it"]

    assert matrix[amsterdam, berlin] == 1.0
    assert matrix[berlin, prague] == 1.0
    assert matrix[rome, amsterdam] == 1.0
    # Non-adjacent within the same user must not be counted.
    assert matrix[amsterdam, prague] == 0.0


def test_next_destination_prefers_nearby_places(fitted_models) -> None:
    ranked = fitted_models["next_destination"].recommend_ids(
        RecommendationRequest(current_destination="amsterdam-nl"), k=6
    )
    nearby = {"rotterdam-nl", "utrecht-nl", "antwerp-be", "brussels-be"}
    assert nearby & set(ranked), f"expected a neighbour of Amsterdam in {ranked}"


def test_next_destination_without_origin_falls_back(fitted_models) -> None:
    results = fitted_models["next_destination"].recommend(RecommendationRequest(), k=5)
    assert len(results) == 5


def test_recommend_next_helper(fitted_models) -> None:
    pairs = fitted_models["next_destination"].recommend_next("berlin-de", k=4)
    assert len(pairs) == 4
    assert all(isinstance(destination, str) for destination, _ in pairs)


# ------------------------------------------------------------------ context
def test_season_scores_differ_by_month(dataset) -> None:
    scorer = ContextScorer(dataset)
    january = scorer.season_scores(1)
    july = scorer.season_scores(7)
    assert not np.allclose(january, july)


def test_budget_fit_peaks_at_matching_band(dataset) -> None:
    scorer = ContextScorer(dataset)
    budget_fit = scorer.budget_scores("budget")
    expensive_fit = scorer.budget_scores("expensive")
    cheapest = int(np.argmin(dataset.destinations["cost_percentile"].to_numpy()))
    assert budget_fit[cheapest] > expensive_fit[cheapest]


def test_missing_context_is_neutral(dataset) -> None:
    scorer = ContextScorer(dataset)
    assert np.allclose(scorer.season_scores(None), 0.5)
    assert np.allclose(scorer.budget_scores(None), 0.5)


def test_short_trips_penalise_distance_more_than_long_trips(dataset) -> None:
    scorer = ContextScorer(dataset)
    _, short = scorer.proximity_scores("amsterdam-nl", 2)
    _, long_trip = scorer.proximity_scores("amsterdam-nl", 21)
    tokyo = dataset.index_of["tokyo-jp"]
    assert long_trip[tokyo] > short[tokyo]


# --------------------------------------------------------------- normalising
def test_normalise_scores_maps_to_unit_interval() -> None:
    scaled = normalise_scores(np.array([-5.0, 0.0, 5.0]))
    assert scaled.min() == pytest.approx(0.0)
    assert scaled.max() == pytest.approx(1.0)


def test_normalise_constant_vector_is_zero() -> None:
    assert np.allclose(normalise_scores(np.array([3.0, 3.0, 3.0])), 0.0)


def test_normalise_handles_infinities() -> None:
    scaled = normalise_scores(np.array([-np.inf, 1.0, 2.0]))
    assert np.isfinite(scaled).all()


# ------------------------------------------------------ candidate generation
def test_candidate_generator_respects_budget(fitted_models, dataset) -> None:
    generator = CandidateGenerator(
        dataset,
        fitted_models["content"],
        fitted_models["collaborative"],
        fitted_models["popularity"],
        n_candidates=10,
    )
    candidates = generator.generate(RecommendationRequest(history=["amsterdam-nl"]))
    assert len(candidates) <= 10


def test_candidate_generator_excludes_visited(fitted_models, dataset) -> None:
    generator = CandidateGenerator(
        dataset,
        fitted_models["content"],
        fitted_models["collaborative"],
        fitted_models["popularity"],
        n_candidates=50,
    )
    request = RecommendationRequest(history=["amsterdam-nl", "berlin-de"])
    assert set(generator.candidate_ids(request)).isdisjoint({"amsterdam-nl", "berlin-de"})


def test_candidate_generator_works_for_cold_start(fitted_models, dataset) -> None:
    generator = CandidateGenerator(
        dataset,
        fitted_models["content"],
        fitted_models["collaborative"],
        fitted_models["popularity"],
        n_candidates=30,
    )
    assert len(generator.candidate_ids(RecommendationRequest())) > 0


def test_candidate_sources_are_recorded(fitted_models, dataset) -> None:
    generator = CandidateGenerator(
        dataset,
        fitted_models["content"],
        fitted_models["collaborative"],
        fitted_models["popularity"],
        n_candidates=40,
    )
    candidates = generator.generate(RecommendationRequest(current_destination="amsterdam-nl"))
    flags = [candidates.source_flags(int(i)) for i in candidates.indices]
    assert any(flag["from_geographic"] == 1.0 for flag in flags)
    assert any(flag["from_popularity"] == 1.0 for flag in flags)
