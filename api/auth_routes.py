"""Authentication and per-user routes.

Mounted by ``api.main``. Every route degrades safely when Google Sign-In is not
configured: ``/auth/config`` reports ``enabled: false`` and the login routes
return 404, so the front-end can present the anonymous flow instead of a button
that cannot work.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.auth import safe_next_path, user_from_claims
from src.service import RecommendationService
from src.storage import TravellerStore
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

router = APIRouter(tags=["auth"])

SESSION_USER_KEY = "user"


# --------------------------------------------------------------- schemas
class SessionUser(BaseModel):
    """The signed-in traveller, as exposed to the front-end."""

    sub: str
    email: str = ""
    name: str = ""
    picture: str = ""


class AuthConfig(BaseModel):
    """Whether this deployment can offer Google Sign-In."""

    enabled: bool = Field(description="False when no Google credentials are configured.")
    provider: str = "google"
    user: Optional[SessionUser] = None


class TripsPayload(BaseModel):
    """A traveller's stored history."""

    history: List[str] = Field(default_factory=list)


class PreferencesPayload(BaseModel):
    """A traveller's stored trip preferences."""

    interests: List[str] = Field(default_factory=list)
    duration_days: Optional[int] = Field(default=None, ge=1, le=365)
    budget: Optional[str] = None


# ------------------------------------------------------------ dependencies
def get_oauth(request: Request):
    oauth = getattr(request.app.state, "oauth", None)
    if oauth is None:
        raise HTTPException(status_code=404, detail="Google Sign-In is not configured")
    return oauth


def get_store(request: Request) -> TravellerStore:
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Traveller store unavailable")
    return store


def current_user(request: Request) -> Dict[str, Any]:
    """Return the signed-in user or raise 401."""
    user = request.session.get(SESSION_USER_KEY)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in")
    return user


# ------------------------------------------------------------------ routes
@router.get("/auth/config", response_model=AuthConfig)
def auth_config(request: Request) -> AuthConfig:
    """Report whether sign-in is available, and who is signed in."""
    enabled = getattr(request.app.state, "oauth", None) is not None
    session_user = request.session.get(SESSION_USER_KEY)
    return AuthConfig(
        enabled=enabled,
        user=SessionUser(**session_user) if session_user else None,
    )


@router.get("/auth/login")
async def login(
    request: Request,
    next: str = Query("/", description="Same-origin path to return to."),
    oauth=Depends(get_oauth),
):
    """Begin the Google authorization-code flow."""
    # Stored in the signed session cookie and validated on the way back, which
    # is what prevents an attacker from replaying someone else's callback.
    request.session["next"] = safe_next_path(next)
    redirect_uri = str(request.url_for("auth_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request, oauth=Depends(get_oauth)) -> RedirectResponse:
    """Complete the flow: verify the ID token and start a session."""
    try:
        # Authlib validates state, exchanges the code, then verifies the ID
        # token against Google's JWKS (signature, issuer, audience, nonce).
        token = await oauth.google.authorize_access_token(request)
    except Exception as exc:  # noqa: BLE001 - surface any OAuth failure as 401
        LOGGER.warning("OAuth callback rejected: %s", type(exc).__name__)
        raise HTTPException(status_code=401, detail="Sign-in failed or was cancelled")

    claims = token.get("userinfo") or {}
    if not claims.get("sub"):
        raise HTTPException(status_code=401, detail="Google returned no identity claim")

    user = user_from_claims(claims)
    # The OAuth tokens themselves are not needed beyond this point: the app
    # only ever acts as the user inside its own database, never against Google.
    request.session[SESSION_USER_KEY] = user

    store: TravellerStore = request.app.state.store
    store.upsert_user(user)

    destination = safe_next_path(request.session.pop("next", "/"))
    return RedirectResponse(url=destination, status_code=303)


@router.post("/auth/logout")
def logout(request: Request) -> Dict[str, bool]:
    """Clear the session cookie."""
    request.session.clear()
    return {"signed_out": True}


@router.get("/me/trips", response_model=TripsPayload)
def read_trips(
    user: Dict[str, Any] = Depends(current_user),
    store: TravellerStore = Depends(get_store),
) -> TripsPayload:
    """The signed-in traveller's stored history."""
    return TripsPayload(history=store.get_trips(user["sub"]))


@router.put("/me/trips", response_model=TripsPayload)
def write_trips(
    payload: TripsPayload,
    request: Request,
    user: Dict[str, Any] = Depends(current_user),
    store: TravellerStore = Depends(get_store),
) -> TripsPayload:
    """Replace the stored history, ignoring ids outside the catalog."""
    service: RecommendationService = request.app.state.service_getter()
    known = [d for d in payload.history if d in service.dataset.index_of]
    return TripsPayload(history=store.set_trips(user["sub"], known))


@router.get("/me/preferences", response_model=PreferencesPayload)
def read_preferences(
    user: Dict[str, Any] = Depends(current_user),
    store: TravellerStore = Depends(get_store),
) -> PreferencesPayload:
    """The signed-in traveller's stored preferences."""
    return PreferencesPayload(**store.get_preferences(user["sub"]))


@router.put("/me/preferences", response_model=PreferencesPayload)
def write_preferences(
    payload: PreferencesPayload,
    user: Dict[str, Any] = Depends(current_user),
    store: TravellerStore = Depends(get_store),
) -> PreferencesPayload:
    """Store trip preferences for the signed-in traveller."""
    return PreferencesPayload(
        **store.set_preferences(
            user["sub"],
            interests=payload.interests,
            duration_days=payload.duration_days,
            budget=payload.budget,
        )
    )


@router.delete("/me")
def delete_account(
    request: Request,
    user: Dict[str, Any] = Depends(current_user),
    store: TravellerStore = Depends(get_store),
) -> Dict[str, bool]:
    """Delete the traveller's account and all stored data."""
    store.delete_user(user["sub"])
    request.session.clear()
    return {"deleted": True}
