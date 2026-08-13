"""API endpoint tests.

The FastAPI dependency is overridden with a service built from the in-memory
fixture dataset, so these tests exercise the real routing, validation and
serialisation code without requiring an ingested dataset on disk.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app, service_dependency
from src.data.dataset import TravelDataset
from src.service import RecommendationService


@pytest.fixture(scope="module")
def client(dataset: TravelDataset, split):
    """A test client wired to a fixture-backed service."""
    service = RecommendationService(dataset=dataset, split=split)
    app.dependency_overrides[service_dependency] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ------------------------------------------------------------------- health
def test_health_reports_dataset_size(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["n_destinations"] > 0
    assert payload["interactions_are_synthetic"] is True
    assert "hybrid" in payload["available_models"]


def test_health_declares_data_sources(client) -> None:
    sources = client.get("/health").json()["data_sources"]
    assert "OpenStreetMap" in sources["attributes"]
    assert "SYNTHETIC" in sources["interactions"]


# ------------------------------------------------------------------ catalog
def test_list_destinations_paginates(client) -> None:
    response = client.get("/destinations", params={"limit": 5})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["destinations"]) == 5
    assert payload["total"] >= 5


def test_list_destinations_search_filter(client) -> None:
    payload = client.get("/destinations", params={"search": "amsterdam"}).json()
    assert payload["total"] >= 1
    assert any(d["city"] == "Amsterdam" for d in payload["destinations"])


def test_list_destinations_country_filter(client) -> None:
    payload = client.get("/destinations", params={"country_code": "NL"}).json()
    assert all(d["country_code"] == "NL" for d in payload["destinations"])


def test_list_destinations_rejects_bad_limit(client) -> None:
    assert client.get("/destinations", params={"limit": 0}).status_code == 422


def test_destination_detail(client) -> None:
    response = client.get("/destination/amsterdam-nl")
    assert response.status_code == 200
    payload = response.json()
    assert payload["city"] == "Amsterdam"
    assert len(payload["monthly_climate"]) == 12
    assert payload["similar_destinations"]


def test_unknown_destination_returns_404(client) -> None:
    response = client.get("/destination/atlantis-xx")
    assert response.status_code == 404
    assert "Unknown destination" in response.json()["detail"]


# ---------------------------------------------------------------- recommend
def test_recommend_with_history(client) -> None:
    response = client.post(
        "/recommend",
        json={
            "history": ["amsterdam-nl", "berlin-de"],
            "current_destination": "prague-cz",
            "month": 9,
            "trip_duration_days": 5,
            "budget": "mid-range",
            "k": 5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["recommendations"]) == 5
    assert payload["cold_start"] is False
    ranks = [item["rank"] for item in payload["recommendations"]]
    assert ranks == [1, 2, 3, 4, 5]


def test_recommend_excludes_visited(client) -> None:
    history = ["amsterdam-nl", "berlin-de", "prague-cz"]
    payload = client.post("/recommend", json={"history": history, "k": 10}).json()
    returned = {item["destination_id"] for item in payload["recommendations"]}
    assert returned.isdisjoint(set(history))


def test_recommend_returns_explanations(client) -> None:
    payload = client.post(
        "/recommend",
        json={"history": ["amsterdam-nl"], "month": 7, "budget": "budget", "k": 3},
    ).json()
    first = payload["recommendations"][0]
    assert first["reasons"]
    assert first["reason_details"][0]["evidence"]


def test_recommend_can_disable_explanations(client) -> None:
    payload = client.post(
        "/recommend", json={"history": ["amsterdam-nl"], "k": 3, "explain": False}
    ).json()
    assert payload["recommendations"][0]["reasons"] == []


def test_recommend_empty_history_is_cold_start(client) -> None:
    payload = client.post("/recommend", json={"history": [], "k": 5}).json()
    assert payload["cold_start"] is True
    assert payload["model"] == "cold_start"
    assert len(payload["recommendations"]) == 5


def test_recommend_reports_unknown_destinations(client) -> None:
    payload = client.post(
        "/recommend", json={"history": ["atlantis-xx", "amsterdam-nl"], "k": 3}
    ).json()
    assert payload["unknown_destinations"] == ["atlantis-xx"]
    assert len(payload["recommendations"]) == 3


def test_recommend_rejects_invalid_budget(client) -> None:
    response = client.post("/recommend", json={"history": [], "budget": "free"})
    assert response.status_code == 422


def test_recommend_rejects_invalid_month(client) -> None:
    assert client.post("/recommend", json={"history": [], "month": 13}).status_code == 422


def test_recommend_rejects_k_above_limit(client) -> None:
    assert client.post("/recommend", json={"history": [], "k": 500}).status_code == 422


def test_recommend_unknown_model_returns_400(client) -> None:
    response = client.post(
        "/recommend", json={"history": ["amsterdam-nl"], "model": "telepathy"}
    )
    assert response.status_code == 400
    assert "Unknown model" in response.json()["detail"]


@pytest.mark.parametrize(
    "model", ["hybrid", "content", "collaborative", "popularity", "next_destination"]
)
def test_every_model_serves_recommendations(client, model) -> None:
    payload = client.post(
        "/recommend", json={"history": ["amsterdam-nl", "berlin-de"], "model": model, "k": 4}
    ).json()
    assert len(payload["recommendations"]) == 4


# ----------------------------------------------------------- next endpoint
def test_recommend_next(client) -> None:
    response = client.get("/recommend/next/amsterdam-nl", params={"k": 5})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["recommendations"]) == 5
    assert "amsterdam-nl" not in {i["destination_id"] for i in payload["recommendations"]}


def test_recommend_next_prefers_neighbours(client) -> None:
    payload = client.get("/recommend/next/amsterdam-nl", params={"k": 6}).json()
    returned = {item["destination_id"] for item in payload["recommendations"]}
    assert returned & {"rotterdam-nl", "utrecht-nl", "antwerp-be", "brussels-be"}


def test_recommend_next_unknown_destination_404(client) -> None:
    assert client.get("/recommend/next/atlantis-xx").status_code == 404


def test_recommend_next_accepts_context(client) -> None:
    response = client.get(
        "/recommend/next/berlin-de",
        params={"k": 3, "month": 6, "trip_duration_days": 4, "budget": "budget"},
    )
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 3


# -------------------------------------------------------------------- meta
def test_models_endpoint(client) -> None:
    payload = client.get("/models").json()
    assert payload["default"] == "hybrid"
    assert set(payload["hybrid_weights"]) == {
        "content", "collaborative", "popularity", "context"
    }
