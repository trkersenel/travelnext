"""FastAPI application for Waygo.

    uvicorn api.main:app --reload

All endpoints are backed by :class:`src.service.RecommendationService`, so the
API and the Streamlit UI always agree. Models are fitted on the first request
and cached for the life of the process.

No endpoint requires an API key, and none of them call an external service at
request time: everything is served from the local processed dataset.
"""

from __future__ import annotations

from typing import List, Optional

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from api.schemas import (
    DestinationListResponse,
    DestinationSummary,
    HealthResponse,
    ProfileRequest,
    ProfileResponse,
    RecommendRequest,
    RecommendResponse,
)
from api import auth_routes
from src.auth import build_oauth_client, load_auth_settings
from src.config import load_config
from src.service import RecommendationService, get_service
from src.storage import TravellerStore
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

DATA_SOURCES = {
    "destinations": "GeoNames cities15000 (CC BY 4.0)",
    "attributes": "OpenStreetMap via Overpass API (ODbL 1.0)",
    "climate": "Open-Meteo historical archive (CC BY 4.0)",
    "popularity": "Wikimedia Pageviews API (CC0) - proxy for interest, not arrivals",
    "cost": "World Bank GNI per capita PPP (CC BY 4.0) - country-level proxy",
    "text": "English Wikipedia REST API (CC BY-SA 4.0)",
    "interactions": "SYNTHETIC - generated, not real travellers",
}

app = FastAPI(
    title="Waygo API",
    version="0.1.0",
    description=(
        "An explainable travel recommendation engine built entirely on free, "
        "openly licensed data. Destination data is real; user interactions are "
        "synthetic and are labelled as such."
    ),
)

# The Streamlit UI runs on a different port, so allow local cross-origin calls.
# Credentials are NOT allowed cross-origin: the session cookie must only ever
# be sent to this origin, which is what a wildcard origin would undermine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------- auth
# Google Sign-In is optional. Without credentials the middleware and routes
# still load (so /auth/config can answer "disabled"), but no login is offered
# and the product works exactly as it does anonymously.
AUTH_SETTINGS = load_auth_settings()
app.add_middleware(
    SessionMiddleware,
    secret_key=AUTH_SETTINGS.session_secret,
    session_cookie="travelnext_session",
    same_site="lax",          # survives the OAuth redirect back from Google
    https_only=AUTH_SETTINGS.https_only,
    max_age=60 * 60 * 24 * 30,
)
app.state.oauth = build_oauth_client(AUTH_SETTINGS)
app.state.store = TravellerStore(load_config().path("data_processed") / "travellers.db")
app.state.service_getter = get_service
app.include_router(auth_routes.router)

if not AUTH_SETTINGS.enabled:
    LOGGER.info(
        "Google Sign-In disabled (set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET "
        "to enable it). The app runs anonymously."
    )


# ---------------------------------------------------------------- frontend
# The web UI is plain HTML/CSS/JS served from the same origin, so it needs no
# build step, no bundler and no second process. Mounted before the route
# definitions so /static never collides with an API path.
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the TravelNext web app."""
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Web frontend not installed")
    return FileResponse(page)


def service_dependency() -> RecommendationService:
    """Provide the shared service, converting startup failures into 503s."""
    try:
        return get_service()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Dataset not built. Run `python -m src.data.build_destinations` and "
                f"`python -m src.data.build_dataset` first. ({exc})"
            ),
        ) from exc


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(service: RecommendationService = Depends(service_dependency)) -> HealthResponse:
    """Service status, dataset size and data attribution."""
    return HealthResponse(
        status="ok",
        n_destinations=service.dataset.n_destinations,
        n_countries=int(service.dataset.destinations["country_code"].nunique()),
        n_users=service.dataset.n_users,
        n_interactions=len(service.dataset.interactions),
        interactions_are_synthetic=True,
        ranker_available=service.ranker is not None,
        login_enabled=app.state.oauth is not None,
        available_models=service.available_models(),
        data_sources=DATA_SOURCES,
    )


@app.get("/destinations", response_model=DestinationListResponse, tags=["catalog"])
def list_destinations(
    service: RecommendationService = Depends(service_dependency),
    search: Optional[str] = Query(None, description="Case-insensitive city or country match."),
    country_code: Optional[str] = Query(None, min_length=2, max_length=2),
    continent: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> DestinationListResponse:
    """List catalog destinations with optional filtering and pagination."""
    frame = service.destinations_frame()

    if search:
        needle = search.strip().lower()
        matches = frame["city"].str.lower().str.contains(needle, na=False) | frame[
            "country"
        ].str.lower().str.contains(needle, na=False)
        frame = frame[matches]
    if country_code:
        frame = frame[frame["country_code"] == country_code.upper()]
    if continent:
        frame = frame[frame["continent"].str.lower() == continent.strip().lower()]

    total = len(frame)
    page = frame.sort_values("popularity_score", ascending=False).iloc[offset : offset + limit]

    return DestinationListResponse(
        total=total,
        limit=limit,
        offset=offset,
        destinations=[
            DestinationSummary(
                destination_id=str(row.destination_id),
                city=str(row.city),
                country=str(row.country),
                country_code=str(row.country_code),
                continent=str(row.continent),
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                population=int(row.population),
                cost_category=str(row.cost_category),
                popularity_percentile=round(float(row.popularity_score), 4),
                image_url=str(getattr(row, "image_url", "") or ""),
                image_url_hd=str(getattr(row, "image_url_hd", "") or ""),
                image_url_md=str(getattr(row, "image_url_md", "") or ""),
                image_width=int(getattr(row, "image_width", 0) or 0),
                image_width_md=int(getattr(row, "image_width_md", 0) or 0),
                image_width_hd=int(getattr(row, "image_width_hd", 0) or 0),
            )
            for row in page.itertuples(index=False)
        ],
    )


@app.get("/destination/{destination_id}", tags=["catalog"])
def get_destination(
    destination_id: str, service: RecommendationService = Depends(service_dependency)
) -> dict:
    """Full record for one destination, including similar destinations."""
    try:
        return service.destination_detail(destination_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown destination: {destination_id}")


@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(
    request: RecommendRequest, service: RecommendationService = Depends(service_dependency)
) -> RecommendResponse:
    """Recommend destinations for a traveller.

    An empty history is valid and routes to the cold-start path. Unknown
    destination ids are reported back in ``unknown_destinations`` and ignored
    rather than causing an error.
    """
    try:
        payload = service.recommend(
            history=request.history,
            current_destination=request.current_destination,
            month=request.month,
            trip_duration_days=request.trip_duration_days,
            budget=request.budget,
            interests=request.interests,
            model=request.model,
            k=request.k,
            explain=request.explain,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RecommendResponse(**payload)


@app.get("/recommend/next/{destination_id}", response_model=RecommendResponse, tags=["recommend"])
def recommend_next(
    destination_id: str,
    service: RecommendationService = Depends(service_dependency),
    k: int = Query(10, ge=1, le=50),
    month: Optional[int] = Query(None, ge=1, le=12),
    trip_duration_days: Optional[int] = Query(None, ge=1, le=365),
    budget: Optional[str] = Query(None),
    explain: bool = Query(True),
) -> RecommendResponse:
    """Answer "where should I go after <destination_id>?"."""
    try:
        payload = service.recommend_next(
            destination_id,
            k=k,
            month=month,
            trip_duration_days=trip_duration_days,
            budget=budget,
            explain=explain,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown destination: {destination_id}")
    return RecommendResponse(**payload)


@app.post("/profile", response_model=ProfileResponse, tags=["recommend"])
def travel_profile(
    request: ProfileRequest, service: RecommendationService = Depends(service_dependency)
) -> ProfileResponse:
    """Infer a travel profile from a visit history.

    Traits are the categories where the traveller's history sits *above the
    catalog norm*, so the profile describes this traveller rather than
    restating that large cities have many museums.
    """
    return ProfileResponse(**service.travel_profile(request.history))


@app.get("/models", tags=["meta"])
def list_models(service: RecommendationService = Depends(service_dependency)) -> dict:
    """Which recommendation models this deployment can serve."""
    return {
        "available_models": service.available_models(),
        "default": "hybrid",
        "hybrid_weights": service.hybrid.weights.as_dict(),
        "ranker_available": service.ranker is not None,
    }


def main() -> None:
    """Run the API with uvicorn using the configured host and port."""
    import uvicorn

    config = load_config()
    uvicorn.run(
        "api.main:app",
        host=str(config.get("api.host", "0.0.0.0")),
        port=int(config.get("api.port", 8000)),
        reload=False,
    )


if __name__ == "__main__":
    main()
