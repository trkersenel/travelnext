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


# ------------------------------------------------------------------ profile
def test_profile_infers_traits_from_history(client) -> None:
    payload = client.post(
        "/profile", json={"history": ["amsterdam-nl", "berlin-de", "prague-cz"]}
    ).json()
    assert payload["n_visited"] == 3
    assert payload["traits"], "a three-city history should yield at least one trait"
    assert len(payload["visited"]) == 3
    for trait in payload["traits"]:
        # Traits are only reported when the history leans above the norm.
        assert trait["deviation"] > 0
        assert trait["label"]


def test_profile_distinguishes_different_histories(client) -> None:
    """The whole point of standardising: two histories must differ."""
    city = client.post(
        "/profile", json={"history": ["amsterdam-nl", "berlin-de", "prague-cz"]}
    ).json()
    coast = client.post(
        "/profile", json={"history": ["barcelona-es", "valencia-es", "lisbon-pt"]}
    ).json()
    assert [t["category"] for t in city["traits"]] != [t["category"] for t in coast["traits"]]


def test_profile_empty_history(client) -> None:
    payload = client.post("/profile", json={"history": []}).json()
    assert payload["n_visited"] == 0
    assert payload["traits"] == []


def test_profile_ignores_unknown_ids(client) -> None:
    payload = client.post(
        "/profile", json={"history": ["atlantis-xx", "amsterdam-nl"]}
    ).json()
    assert payload["n_visited"] == 1


def test_profile_reports_region_and_cost(client) -> None:
    payload = client.post("/profile", json={"history": ["amsterdam-nl", "berlin-de"]}).json()
    assert payload["region"]
    assert payload["cost_band"] in {"budget", "mid-range", "expensive"}


# --------------------------------------------------------------- interests
@pytest.mark.parametrize(
    "interest,expected",
    [
        ("history", "heritage"),
        ("local_life", "walkability"),
        ("photography", "outdoor"),
        ("outdoor", "nature"),
    ],
)
def test_onboarding_interests_map_to_measured_categories(
    dataset, split, interest, expected
) -> None:
    """Product-facing interest names must resolve to real OSM categories."""
    service = RecommendationService(dataset=dataset, split=split)
    assert expected in service.resolve_interests([interest])


def test_unknown_interest_is_dropped(dataset, split) -> None:
    service = RecommendationService(dataset=dataset, split=split)
    assert service.resolve_interests(["teleportation"]) == []


def test_interests_change_the_ranking(client) -> None:
    beaches = client.post(
        "/recommend", json={"history": [], "interests": ["beaches"], "k": 5}
    ).json()
    museums = client.post(
        "/recommend", json={"history": [], "interests": ["museums", "history"], "k": 5}
    ).json()
    assert [r["destination_id"] for r in beaches["recommendations"]] != [
        r["destination_id"] for r in museums["recommendations"]
    ]


# ----------------------------------------------------------------- countries
def test_country_catalog_covers_the_world(client) -> None:
    payload = client.get("/countries").json()
    # Natural Earth 1:110m, minus the two entries without an ISO code.
    assert payload["total"] == 175
    codes = {c["country_code"] for c in payload["countries"]}
    assert {"NL", "DE", "TR", "JP", "US", "BR", "ZA"} <= codes


def test_country_catalog_marks_visited(client) -> None:
    payload = client.get("/countries", params={"visited": "NL,DE"}).json()
    assert payload["visited_count"] == 2
    visited = {c["country_code"] for c in payload["countries"] if c["visited"]}
    assert visited == {"NL", "DE"}


def test_country_catalog_ignores_unknown_visited_codes(client) -> None:
    payload = client.get("/countries", params={"visited": "NL,ZZ,,QQ"}).json()
    assert payload["visited_count"] == 1


def test_country_catalog_reports_city_coverage(client) -> None:
    """The map spans more countries than the recommender has cities for."""
    payload = client.get("/countries").json()
    with_cities = [c for c in payload["countries"] if c["cities_in_catalog"] > 0]
    without = [c for c in payload["countries"] if c["cities_in_catalog"] == 0]
    assert with_cities and without
    assert len(with_cities) < payload["total"]


def test_country_catalog_has_continents(client) -> None:
    payload = client.get("/countries").json()
    lookup = {c["country_code"]: c for c in payload["countries"]}
    assert lookup["NL"]["continent"] == "Europe"
    assert lookup["JP"]["continent"] == "Asia"


# --------------------------------------------------- per-user country store
def test_me_countries_requires_sign_in(client) -> None:
    assert client.get("/me/countries").status_code == 401
    assert client.post("/me/countries/NL").status_code == 401


def test_country_store_round_trip(tmp_path) -> None:
    from src.storage import TravellerStore

    store = TravellerStore(tmp_path / "t.db")
    store.upsert_user({"sub": "u1", "email": "a@b.c", "name": "T", "picture": ""})

    assert store.get_countries("u1") == []
    assert store.add_country("u1", "nl") == ["NL"]          # normalised
    assert store.add_country("u1", "NL") == ["NL"]          # idempotent
    assert set(store.add_country("u1", "DE")) == {"NL", "DE"}
    assert store.remove_country("u1", "DE") == ["NL"]
    assert set(store.set_countries("u1", ["TR", "JP", "TR"])) == {"TR", "JP"}


def test_deleting_a_user_removes_their_countries(tmp_path) -> None:
    from src.storage import TravellerStore

    store = TravellerStore(tmp_path / "t.db")
    store.upsert_user({"sub": "u1", "email": "", "name": "", "picture": ""})
    store.add_country("u1", "NL")
    store.delete_user("u1")
    assert store.get_countries("u1") == []


def test_country_code_normalisation() -> None:
    from src.data.countries import normalise

    assert normalise("nl") == "NL"
    assert normalise(" de ") == "DE"
    assert normalise("ZZ") == ""
    assert normalise("") == ""
