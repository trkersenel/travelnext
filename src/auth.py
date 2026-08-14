"""Google Sign-In (OAuth 2.0 / OpenID Connect) for TravelNext.

DESIGN CONSTRAINT
-----------------
This project's central promise is that it runs with no account and no API key.
Google Sign-In necessarily breaks that for the person deploying it -- it needs
a Google Cloud project and a client secret -- so it is **strictly optional**.
When ``GOOGLE_CLIENT_ID`` and ``GOOGLE_CLIENT_SECRET`` are absent the feature
disables itself, the API reports ``login_enabled: false``, and the product
continues to work exactly as before. Nothing in the recommendation engine
depends on a signed-in user.

Google's OAuth endpoints are themselves free of charge; the cost is the account
and the credential handling, not money.

SECURITY NOTES
--------------
* Uses the Authorization Code flow with OpenID Connect. Authlib fetches
  Google's JWKS and verifies the ID token signature, issuer, audience and
  nonce -- we never trust unverified token claims.
* ``state`` and ``nonce`` live in a signed, HttpOnly session cookie, which is
  what stops CSRF and token-replay on the callback.
* Access and refresh tokens are deliberately **not** persisted. The app only
  needs identity, so it keeps the subject id, email and display name and
  discards the tokens once the ID token has been verified.
* Secrets are read from the environment and never logged.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

GOOGLE_METADATA_URL = "https://accounts.google.com/.well-known/openid-configuration"

# Identity only. We ask for nothing that would let the app act on the user's
# behalf, which keeps the consent screen honest and the blast radius small.
GOOGLE_SCOPES = "openid email profile"


@dataclass(frozen=True)
class AuthSettings:
    """OAuth configuration, loaded from the environment."""

    client_id: str
    client_secret: str
    session_secret: str
    redirect_path: str = "/auth/callback"
    # Session cookies are only sent over HTTPS when this is true. It defaults
    # to false so local development over http://localhost works; set
    # TRAVELNEXT_HTTPS=1 in any real deployment.
    https_only: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.client_id and self.client_secret)


def load_auth_settings() -> AuthSettings:
    """Read OAuth settings from the environment.

    A missing client id or secret is not an error: it simply means sign-in is
    switched off for this deployment.
    """
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    session_secret = os.environ.get("TRAVELNEXT_SESSION_SECRET", "").strip()

    if client_id and client_secret and not session_secret:
        # A random per-process key keeps cookies signed correctly for this run,
        # but every restart invalidates existing sessions -- fine for local use,
        # not for a deployment, hence the warning.
        session_secret = secrets.token_urlsafe(48)
        LOGGER.warning(
            "TRAVELNEXT_SESSION_SECRET is not set; generated an ephemeral key. "
            "Sessions will not survive a restart. Set it explicitly to deploy."
        )

    return AuthSettings(
        client_id=client_id,
        client_secret=client_secret,
        session_secret=session_secret or secrets.token_urlsafe(48),
        https_only=os.environ.get("TRAVELNEXT_HTTPS", "").strip() in {"1", "true", "yes"},
    )


def build_oauth_client(settings: AuthSettings):
    """Create the Authlib Google client, or ``None`` when sign-in is disabled."""
    if not settings.enabled:
        return None

    from authlib.integrations.starlette_client import OAuth

    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.client_id,
        client_secret=settings.client_secret,
        server_metadata_url=GOOGLE_METADATA_URL,
        client_kwargs={"scope": GOOGLE_SCOPES},
    )
    LOGGER.info("Google Sign-In enabled")
    return oauth


def user_from_claims(claims: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce verified ID-token claims to the fields the product uses.

    Only identity is retained. ``sub`` is the stable Google account id and is
    what user records key on; the email may change, so it is display data only.
    """
    return {
        "sub": str(claims.get("sub", "")),
        "email": str(claims.get("email", "")),
        "name": str(claims.get("name") or claims.get("given_name") or "Traveller"),
        "picture": str(claims.get("picture", "")),
        "email_verified": bool(claims.get("email_verified", False)),
    }


def safe_next_path(candidate: Optional[str]) -> str:
    """Return a same-origin redirect target, defaulting to the app root.

    Only root-relative single-slash paths are allowed. This blocks the open
    redirect where ``?next=https://evil.example`` or ``//evil.example`` would
    otherwise bounce a freshly authenticated user off-site.
    """
    if not candidate:
        return "/"
    if not candidate.startswith("/") or candidate.startswith("//"):
        return "/"
    return candidate
