"""Pydantic request and response models for the TravelNext API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

BUDGETS = {"budget", "mid-range", "expensive"}


class RecommendRequest(BaseModel):
    """Body of ``POST /recommend``."""

    history: List[str] = Field(
        default_factory=list,
        description="Destination ids the traveller has already visited, oldest first.",
        examples=[["amsterdam-nl", "berlin-de", "prague-cz"]],
    )
    current_destination: Optional[str] = Field(
        default=None, description="Where the traveller is now, or their most recent trip."
    )
    month: Optional[int] = Field(default=None, ge=1, le=12, description="Travel month, 1-12.")
    trip_duration_days: Optional[int] = Field(
        default=None, ge=1, le=365, description="Planned trip length in days."
    )
    budget: Optional[str] = Field(
        default=None, description="One of: budget, mid-range, expensive."
    )
    interests: List[str] = Field(
        default_factory=list,
        description="Interest categories, e.g. museums, nightlife, nature.",
    )
    model: str = Field(default="hybrid", description="Which recommender to use.")
    k: int = Field(default=10, ge=1, le=50, description="How many results to return.")
    explain: bool = Field(default=True, description="Attach explanations to each result.")

    @field_validator("budget")
    @classmethod
    def _check_budget(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalised = value.strip().lower()
        if normalised not in BUDGETS:
            raise ValueError(f"budget must be one of {sorted(BUDGETS)}")
        return normalised


class ReasonDetail(BaseModel):
    """One explanation with the evidence supporting it."""

    kind: str
    text: str
    strength: float
    evidence: Dict[str, Any] = Field(default_factory=dict)


class RecommendationItem(BaseModel):
    """A single ranked destination."""

    destination_id: str
    destination: str
    city: str
    country: str
    country_code: str
    continent: str
    latitude: float
    longitude: float
    score: float
    rank: int
    cost_category: str
    popularity_percentile: float
    image_url: str = Field(default="", description="Wikimedia Commons thumbnail (CC/PD licensed).")
    image_page: str = Field(default="", description="Wikipedia article the image came from.")
    reasons: List[str] = Field(default_factory=list)
    reason_details: List[ReasonDetail] = Field(default_factory=list)
    attributes: Dict[str, float] = Field(default_factory=dict)


class RecommendResponse(BaseModel):
    """Response of ``POST /recommend`` and ``GET /recommend/next/{id}``."""

    model: str = Field(description="The model that actually produced the ranking.")
    requested_model: str
    cold_start: bool = Field(
        description="True when no usable history was supplied and the cold-start path was used."
    )
    k: int
    unknown_destinations: List[str] = Field(
        default_factory=list,
        description="Ids in the request that are not in the catalog; these were ignored.",
    )
    recommendations: List[RecommendationItem]


class DestinationSummary(BaseModel):
    """A catalog entry as returned by ``GET /destinations``."""

    destination_id: str
    city: str
    country: str
    country_code: str
    continent: str
    latitude: float
    longitude: float
    population: int
    cost_category: str
    popularity_percentile: float
    image_url: str = Field(default="", description="Wikimedia Commons thumbnail (CC/PD licensed).")


class DestinationListResponse(BaseModel):
    """Paginated destination listing."""

    total: int
    limit: int
    offset: int
    destinations: List[DestinationSummary]


class HealthResponse(BaseModel):
    """Service health, dataset size and data attribution."""

    status: str
    n_destinations: int
    n_countries: int
    n_users: int
    n_interactions: int
    interactions_are_synthetic: bool
    ranker_available: bool
    available_models: List[str]
    data_sources: Dict[str, str]


class ProfileRequest(BaseModel):
    """Body of ``POST /profile``."""

    history: List[str] = Field(
        default_factory=list, description="Destination ids the traveller has visited."
    )


class TraitItem(BaseModel):
    """One inferred trait, with the deviation that justifies it."""

    category: str
    label: str
    deviation: float = Field(
        description="Standard deviations above the catalog norm across the visited set."
    )


class VisitedItem(BaseModel):
    """A destination in the traveller's history."""

    destination_id: str
    city: str
    country: str
    country_code: str
    image_url: str = ""
    latitude: float
    longitude: float


class ProfileResponse(BaseModel):
    """Inferred travel profile. Every field is derived from measured data."""

    n_visited: int
    traits: List[TraitItem] = Field(default_factory=list)
    region: str = ""
    continent: str = ""
    cost_band: str = ""
    walkable: bool = False
    visited: List[VisitedItem] = Field(default_factory=list)
