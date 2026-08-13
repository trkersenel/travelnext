"""Tests for the explanation layer.

The property that matters is *groundedness*: a reason may only appear when the
number behind it clears its threshold. These tests therefore check both
directions -- that a reason shows up when the evidence exists, and that it
stays away when it does not.
"""

from __future__ import annotations

import pytest

from src.data.dataset import TravelDataset
from src.explainability.explain import Explainer, Reason
from src.models.base import RecommendationRequest
from src.models.collaborative import ItemItemCFRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.context import ContextScorer


@pytest.fixture(scope="module")
def explainer(dataset: TravelDataset, split) -> Explainer:
    content = ContentBasedRecommender().fit(dataset, split.train)
    collaborative = ItemItemCFRecommender().fit(dataset, split.train)
    return Explainer(dataset, content, collaborative, ContextScorer(dataset))


def test_explanations_are_produced_for_a_warm_user(explainer: Explainer) -> None:
    request = RecommendationRequest(
        history=["amsterdam-nl", "berlin-de"],
        current_destination="prague-cz",
        month=9,
        budget="mid-range",
        trip_duration_days=5,
    )
    reasons = explainer.explain(request, "vienna-at")
    assert reasons, "a warm request with rich context should produce reasons"
    assert all(isinstance(reason, Reason) for reason in reasons)


def test_every_reason_carries_evidence(explainer: Explainer) -> None:
    request = RecommendationRequest(
        history=["amsterdam-nl"], current_destination="berlin-de", month=7, budget="budget"
    )
    for reason in explainer.explain(request, "copenhagen-dk"):
        assert reason.evidence, f"{reason.kind} has no supporting evidence"
        assert 0.0 <= reason.strength <= 1.0


def test_reasons_are_ordered_by_strength(explainer: Explainer) -> None:
    request = RecommendationRequest(history=["amsterdam-nl", "rotterdam-nl"], month=6)
    strengths = [reason.strength for reason in explainer.explain(request, "utrecht-nl")]
    assert strengths == sorted(strengths, reverse=True)


def test_max_reasons_is_respected(explainer: Explainer) -> None:
    request = RecommendationRequest(
        history=["amsterdam-nl", "berlin-de"], month=9, budget="mid-range"
    )
    assert len(explainer.explain(request, "vienna-at", max_reasons=2)) <= 2


def test_no_season_reason_without_a_month(explainer: Explainer) -> None:
    request = RecommendationRequest(history=["amsterdam-nl"])
    kinds = {reason.kind for reason in explainer.explain(request, "vienna-at")}
    assert "season" not in kinds


def test_no_budget_reason_without_a_budget(explainer: Explainer) -> None:
    request = RecommendationRequest(history=["amsterdam-nl"])
    kinds = {reason.kind for reason in explainer.explain(request, "vienna-at")}
    assert "budget" not in kinds


def test_geography_reason_appears_for_a_near_neighbour(explainer: Explainer) -> None:
    request = RecommendationRequest(current_destination="amsterdam-nl", trip_duration_days=3)
    kinds = {reason.kind for reason in explainer.explain(request, "rotterdam-nl")}
    assert "geography" in kinds


def test_no_geography_reason_for_a_distant_destination(explainer: Explainer) -> None:
    request = RecommendationRequest(current_destination="amsterdam-nl")
    kinds = {reason.kind for reason in explainer.explain(request, "sydney-au")}
    assert "geography" not in kinds


def test_geography_reason_reports_a_plausible_distance(explainer: Explainer) -> None:
    request = RecommendationRequest(current_destination="amsterdam-nl")
    reasons = [r for r in explainer.explain(request, "rotterdam-nl") if r.kind == "geography"]
    assert reasons
    distance = reasons[0].evidence["distance_km"]
    assert 40.0 < float(distance) < 80.0


def test_content_similarity_reason_names_a_visited_place(explainer: Explainer) -> None:
    request = RecommendationRequest(history=["amsterdam-nl", "sydney-au"])
    reasons = [
        r for r in explainer.explain(request, "rotterdam-nl") if r.kind == "content_similarity"
    ]
    if reasons:
        assert reasons[0].evidence["most_similar_visited"] in {"amsterdam-nl", "sydney-au"}
        assert "Amsterdam" in reasons[0].text or "Sydney" in reasons[0].text


def test_unvisited_reason_only_with_history(explainer: Explainer) -> None:
    with_history = {r.kind for r in explainer.explain(
        RecommendationRequest(history=["amsterdam-nl"]), "vienna-at"
    )}
    without_history = {r.kind for r in explainer.explain(RecommendationRequest(), "vienna-at")}
    assert "unvisited" in with_history
    assert "unvisited" not in without_history


def test_unknown_destination_yields_no_reasons(explainer: Explainer) -> None:
    assert explainer.explain(RecommendationRequest(history=["amsterdam-nl"]), "atlantis-xx") == []


def test_cold_start_request_does_not_crash(explainer: Explainer) -> None:
    reasons = explainer.explain(RecommendationRequest(month=8, budget="budget"), "barcelona-es")
    assert isinstance(reasons, list)


def test_explain_batch_covers_every_destination(explainer: Explainer) -> None:
    request = RecommendationRequest(history=["amsterdam-nl"], month=5)
    ids = ["vienna-at", "rome-it", "tokyo-jp"]
    explained = explainer.explain_batch(request, ids)
    assert set(explained) == set(ids)


def test_reason_serialises_to_dict(explainer: Explainer) -> None:
    request = RecommendationRequest(history=["amsterdam-nl"], month=7, budget="budget")
    reasons = explainer.explain(request, "rotterdam-nl")
    assert reasons
    payload = reasons[0].as_dict()
    assert set(payload) == {"kind", "text", "strength", "evidence"}
    assert isinstance(payload["text"], str) and payload["text"]


def test_season_reason_quotes_the_real_temperature(
    explainer: Explainer, dataset: TravelDataset
) -> None:
    request = RecommendationRequest(history=["amsterdam-nl"], month=7)
    reasons = [r for r in explainer.explain(request, "rome-it") if r.kind == "season"]
    if reasons:
        index = dataset.index_of["rome-it"]
        expected = float(dataset.destinations.at[index, "temp_m07"])
        assert reasons[0].evidence["mean_temp_c"] == pytest.approx(expected, abs=0.1)
