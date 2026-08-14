"""Tests for optional Google Sign-In and the traveller store.

The OAuth handshake itself is not exercised against Google — that would need a
network round trip and real credentials. What is tested is everything the
project actually controls: that the feature degrades safely when unconfigured,
that the redirect target cannot be abused, that protected routes reject
anonymous callers, and that the store round-trips data correctly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app, service_dependency
from src.auth import AuthSettings, load_auth_settings, safe_next_path, user_from_claims
from src.data.dataset import TravelDataset
from src.service import RecommendationService
from src.storage import TravellerStore


@pytest.fixture(scope="module")
def client(dataset: TravelDataset, split):
    service = RecommendationService(dataset=dataset, split=split)
    app.dependency_overrides[service_dependency] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ------------------------------------------------------------- settings
def test_auth_disabled_without_credentials(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    assert load_auth_settings().enabled is False


def test_auth_enabled_with_credentials(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "id.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret")
    settings = load_auth_settings()
    assert settings.enabled is True
    # A session key must always exist, generated if not supplied.
    assert settings.session_secret


def test_https_only_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("TRAVELNEXT_HTTPS", raising=False)
    assert load_auth_settings().https_only is False
    monkeypatch.setenv("TRAVELNEXT_HTTPS", "1")
    assert load_auth_settings().https_only is True


# --------------------------------------------------------- open redirect
@pytest.mark.parametrize(
    "candidate",
    [
        "https://evil.example/steal",
        "//evil.example",
        "http://evil.example",
        r"\\evil.example",
        None,
        "",
    ],
)
def test_offsite_redirects_are_refused(candidate) -> None:
    """A freshly authenticated user must not be bounced to another origin."""
    assert safe_next_path(candidate) == "/"


@pytest.mark.parametrize("candidate", ["/", "/trips", "/recommend?k=5"])
def test_same_origin_redirects_are_allowed(candidate) -> None:
    assert safe_next_path(candidate) == candidate


# ---------------------------------------------------------------- claims
def test_user_from_claims_keeps_identity_only() -> None:
    user = user_from_claims(
        {
            "sub": "1234",
            "email": "traveller@example.com",
            "name": "Traveller",
            "picture": "https://example.com/p.jpg",
            "email_verified": True,
            # Anything else Google sends is deliberately dropped.
            "access_token": "should-not-be-kept",
            "hd": "example.com",
        }
    )
    assert set(user) == {"sub", "email", "name", "picture", "email_verified"}
    assert "access_token" not in user


def test_user_from_claims_falls_back_to_a_display_name() -> None:
    assert user_from_claims({"sub": "1"})["name"] == "Traveller"


# ------------------------------------------------------------- endpoints
def test_auth_config_reports_disabled(client) -> None:
    payload = client.get("/auth/config").json()
    assert payload["enabled"] is False
    assert payload["user"] is None


def test_health_reports_login_state(client) -> None:
    assert client.get("/health").json()["login_enabled"] is False


def test_login_route_404s_when_unconfigured(client) -> None:
    assert client.get("/auth/login", follow_redirects=False).status_code == 404


def test_protected_routes_require_a_session(client) -> None:
    assert client.get("/me/trips").status_code == 401
    assert client.put("/me/trips", json={"history": []}).status_code == 401
    assert client.get("/me/preferences").status_code == 401
    assert client.delete("/me").status_code == 401


def test_logout_is_always_safe(client) -> None:
    assert client.post("/auth/logout").json() == {"signed_out": True}


# ----------------------------------------------------------------- store
@pytest.fixture()
def store(tmp_path) -> TravellerStore:
    return TravellerStore(tmp_path / "travellers.db")


def test_store_upserts_a_user(store: TravellerStore) -> None:
    store.upsert_user({"sub": "u1", "email": "a@b.c", "name": "A", "picture": ""})
    store.upsert_user({"sub": "u1", "email": "new@b.c", "name": "A2", "picture": ""})
    user = store.get_user("u1")
    assert user["email"] == "new@b.c"
    assert user["name"] == "A2"


def test_store_round_trips_trips_in_order(store: TravellerStore) -> None:
    store.upsert_user({"sub": "u1", "email": "", "name": "", "picture": ""})
    history = ["amsterdam-nl", "berlin-de", "prague-cz"]
    assert store.set_trips("u1", history) == history
    assert store.get_trips("u1") == history


def test_store_deduplicates_trips(store: TravellerStore) -> None:
    store.upsert_user({"sub": "u1", "email": "", "name": "", "picture": ""})
    stored = store.set_trips("u1", ["a", "b", "a"])
    assert stored == ["a", "b"]


def test_store_replaces_rather_than_appends(store: TravellerStore) -> None:
    store.upsert_user({"sub": "u1", "email": "", "name": "", "picture": ""})
    store.set_trips("u1", ["a", "b", "c"])
    store.set_trips("u1", ["c", "a"])
    assert store.get_trips("u1") == ["c", "a"]


def test_store_isolates_users(store: TravellerStore) -> None:
    for sub in ("u1", "u2"):
        store.upsert_user({"sub": sub, "email": "", "name": "", "picture": ""})
    store.set_trips("u1", ["amsterdam-nl"])
    store.set_trips("u2", ["tokyo-jp"])
    assert store.get_trips("u1") == ["amsterdam-nl"]
    assert store.get_trips("u2") == ["tokyo-jp"]


def test_store_round_trips_preferences(store: TravellerStore) -> None:
    store.upsert_user({"sub": "u1", "email": "", "name": "", "picture": ""})
    saved = store.set_preferences(
        "u1", interests=["museums", "history"], duration_days=6, budget="mid-range"
    )
    assert saved["interests"] == ["museums", "history"]
    assert saved["duration_days"] == 6
    assert saved["budget"] == "mid-range"


def test_preferences_default_when_absent(store: TravellerStore) -> None:
    assert store.get_preferences("nobody") == {
        "interests": [],
        "duration_days": None,
        "budget": None,
    }


def test_deleting_a_user_removes_their_trips(store: TravellerStore) -> None:
    store.upsert_user({"sub": "u1", "email": "", "name": "", "picture": ""})
    store.set_trips("u1", ["amsterdam-nl"])
    store.set_preferences("u1", interests=["food"])
    store.delete_user("u1")
    assert store.get_user("u1") is None
    assert store.get_trips("u1") == []
    assert store.get_preferences("u1")["interests"] == []
