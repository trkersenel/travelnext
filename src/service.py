"""The recommendation service shared by the API and the Streamlit UI.

Both front-ends need exactly the same behaviour: load the artefacts once, pick
the right model, handle cold start and unknown ids, attach explanations, and
return enriched destination records. Putting that here means the UI and the API
can never drift apart in what they recommend or how they justify it.

Loading is lazy and cached: the first request builds the models from the
processed parquet files (a few seconds at this catalog size), and every
subsequent request reuses them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.candidate_generation.generator import CandidateGenerator
from src.config import Config, load_config
from src.data.dataset import Split, TravelDataset, load_split
from src.explainability.explain import Explainer, Reason
from src.models.base import BaseRecommender, RecommendationRequest
from src.models.cold_start import ColdStartRecommender, filter_known_destinations, is_cold_start
from src.models.collaborative import ItemItemCFRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.context import ContextScorer
from src.models.hybrid import HybridRecommender, weights_from_config
from src.models.next_destination import NextDestinationRecommender
from src.models.popularity import PopularityRecommender
from src.preprocessing.features import PROFILE_CATEGORIES
from src.ranking.features import RankingFeatureBuilder
from src.ranking.ltr import LearningToRankRecommender
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

RANKER_FILE = "ranker.joblib"


@dataclass
class Recommendation:
    """One enriched, explained recommendation ready for display or JSON."""

    destination_id: str
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
    image_url: str = ""
    image_page: str = ""
    reasons: List[Reason] = field(default_factory=list)
    attributes: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "destination_id": self.destination_id,
            "destination": self.city,
            "city": self.city,
            "country": self.country,
            "country_code": self.country_code,
            "continent": self.continent,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "score": round(float(self.score), 4),
            "rank": self.rank,
            "cost_category": self.cost_category,
            "popularity_percentile": round(float(self.popularity_percentile), 4),
            "image_url": self.image_url,
            "image_page": self.image_page,
            "reasons": [reason.text for reason in self.reasons],
            "reason_details": [reason.as_dict() for reason in self.reasons],
            "attributes": {k: round(float(v), 3) for k, v in self.attributes.items()},
        }


class RecommendationService:
    """Fits (or loads) every model once and serves ranked, explained results."""

    # Attribute categories surfaced with each recommendation.
    DISPLAY_ATTRIBUTES = (
        "museums",
        "culture",
        "heritage",
        "architecture",
        "nightlife",
        "food",
        "nature",
        "beaches",
        "outdoor",
        "family",
        "walkability",
    )

    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        dataset: Optional[TravelDataset] = None,
        split: Optional[Split] = None,
    ) -> None:
        self.config = config or load_config()
        if dataset is not None and split is not None:
            # Injecting an in-memory dataset lets the tests exercise the real
            # service (and therefore the real API) without needing the ingested
            # parquet files to exist.
            self.dataset, self.split = dataset, split
        else:
            self.dataset, self.split = load_split(self.config)
        train = self.split.train

        LOGGER.info("Fitting service models on %d training interactions", len(train))
        self.popularity = PopularityRecommender().fit(self.dataset, train)
        self.content = ContentBasedRecommender(
            tfidf_max_features=int(self.config.get("models.content.tfidf_max_features", 20000)),
            text_weight=float(self.config.get("models.content.text_weight", 0.45)),
            attribute_weight=float(self.config.get("models.content.attribute_weight", 0.55)),
        ).fit(self.dataset, train)
        self.collaborative = ItemItemCFRecommender(
            shrinkage=float(self.config.get("models.collaborative.shrinkage", 10.0)),
            top_k_neighbours=int(self.config.get("models.collaborative.top_k_neighbours", 200)),
        ).fit(self.dataset, train)
        self.hybrid = HybridRecommender(
            content=self.content,
            collaborative=self.collaborative,
            popularity=self.popularity,
            # Serving weights, not the validation-optimal ones -- see
            # weights_from_config() and configs/config.yaml for the reasoning.
            weights=weights_from_config(self.config, serving=True),
        ).fit(self.dataset, train)
        self.next_destination = NextDestinationRecommender(
            content=self.content, collaborative=self.collaborative, popularity=self.popularity
        ).fit(self.dataset, train)
        self.context = ContextScorer(self.dataset)
        self.cold_start = ColdStartRecommender(
            self.popularity, self.content, self.context
        ).fit(self.dataset, train)

        self.generator = CandidateGenerator(
            self.dataset,
            content=self.content,
            collaborative=self.collaborative,
            popularity=self.popularity,
            n_candidates=int(self.config.get("candidate_generation.n_candidates", 150)),
            per_source=dict(self.config.get("candidate_generation.per_source", {}) or {}),
        )
        self.feature_builder = RankingFeatureBuilder(
            self.dataset,
            content=self.content,
            collaborative=self.collaborative,
            popularity=self.popularity,
            context=self.context,
        )
        self.ranker = self._load_ranker()
        self.explainer = Explainer(self.dataset, self.content, self.collaborative, self.context)
        LOGGER.info("Service ready: %d destinations", self.dataset.n_destinations)

    # ---------------------------------------------------------------- setup
    def _load_ranker(self) -> Optional[LearningToRankRecommender]:
        """Load the trained ranker if one has been produced."""
        path = self.config.path("models") / RANKER_FILE
        if not path.exists():
            LOGGER.warning(
                "%s not found; the learning_to_rank model will be unavailable until "
                "`python -m src.evaluation.run_experiments` has been run.",
                path,
            )
            return None
        ranker = LearningToRankRecommender(self.generator, self.feature_builder)
        ranker.dataset = self.dataset
        return ranker.load(path)

    def available_models(self) -> List[str]:
        """Model names accepted by :meth:`recommend`."""
        names = ["hybrid", "content", "collaborative", "popularity", "next_destination"]
        if self.ranker is not None:
            names.append("learning_to_rank")
        return names

    def _pick_model(self, name: str, request: RecommendationRequest) -> BaseRecommender:
        """Resolve a model name, falling back to cold start when appropriate."""
        if is_cold_start(self.dataset, request):
            # Personalised models have nothing to work with; say so in the logs
            # rather than silently returning a popularity list under their name.
            LOGGER.info("Cold-start request: routing to the cold_start model")
            return self.cold_start

        lookup: Dict[str, Optional[BaseRecommender]] = {
            "hybrid": self.hybrid,
            "content": self.content,
            "collaborative": self.collaborative,
            "popularity": self.popularity,
            "next_destination": self.next_destination,
            "learning_to_rank": self.ranker,
        }
        model = lookup.get(name)
        if model is None:
            if name == "learning_to_rank":
                LOGGER.warning("Ranker unavailable; falling back to hybrid")
                return self.hybrid
            raise ValueError(
                f"Unknown model {name!r}. Available: {', '.join(self.available_models())}"
            )
        return model

    # ------------------------------------------------------------- serving
    def recommend(
        self,
        *,
        history: Sequence[str] = (),
        current_destination: Optional[str] = None,
        month: Optional[int] = None,
        trip_duration_days: Optional[int] = None,
        budget: Optional[str] = None,
        interests: Sequence[str] = (),
        model: str = "hybrid",
        k: int = 10,
        explain: bool = True,
    ) -> Dict[str, Any]:
        """Produce ranked, explained recommendations for one request."""
        max_k = int(self.config.get("api.max_k", 50))
        k = max(1, min(int(k), max_k))

        known_history, unknown_history = filter_known_destinations(self.dataset, list(history))
        unknown: List[str] = list(unknown_history)
        if current_destination and current_destination not in self.dataset.index_of:
            unknown.append(current_destination)
            current_destination = None

        request = RecommendationRequest(
            history=known_history,
            current_destination=current_destination,
            month=month,
            trip_duration_days=trip_duration_days,
            budget=budget,
            # Onboarding interests use product names ("Local life"); the models
            # only know measured OSM categories. Translate at the boundary.
            interests=self.resolve_interests(interests),
        )

        selected = self._pick_model(model, request)
        scored = selected.recommend(request, k=k)

        recommendations = [
            self._enrich(item.destination_id, item.score, item.rank, request, explain)
            for item in scored
        ]
        return {
            "model": selected.name,
            "requested_model": model,
            "cold_start": is_cold_start(self.dataset, request),
            "k": k,
            "unknown_destinations": unknown,
            "recommendations": [r.as_dict() for r in recommendations],
        }

    def recommend_next(
        self,
        destination_id: str,
        *,
        k: int = 10,
        month: Optional[int] = None,
        trip_duration_days: Optional[int] = None,
        budget: Optional[str] = None,
        explain: bool = True,
    ) -> Dict[str, Any]:
        """Answer "where should I go after <destination>?"."""
        if destination_id not in self.dataset.index_of:
            raise KeyError(destination_id)
        return self.recommend(
            current_destination=destination_id,
            month=month,
            trip_duration_days=trip_duration_days,
            budget=budget,
            model="next_destination",
            k=k,
            explain=explain,
        )

    def _enrich(
        self,
        destination_id: str,
        score: float,
        rank: int,
        request: RecommendationRequest,
        explain: bool,
    ) -> Recommendation:
        """Attach catalog metadata, attributes and explanations to a result."""
        index = self.dataset.index_of[destination_id]
        row = self.dataset.destinations.iloc[index]

        attributes: Dict[str, float] = {}
        for category in self.DISPLAY_ATTRIBUTES:
            value = row.get(f"score_{category}")
            if value is not None and not pd.isna(value):
                attributes[category] = float(value)

        return Recommendation(
            destination_id=destination_id,
            city=str(row["city"]),
            country=str(row["country"]),
            country_code=str(row["country_code"]),
            continent=str(row["continent"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            score=float(score),
            rank=rank,
            cost_category=str(row["cost_category"]),
            popularity_percentile=float(row["popularity_score"]),
            image_url=str(row.get("image_url", "") or ""),
            image_page=str(row.get("image_page", "") or ""),
            reasons=self.explainer.explain(request, destination_id) if explain else [],
            attributes=attributes,
        )

    # ------------------------------------------------------------- profile
    # Human labels for the measured OSM categories, plus the onboarding
    # interests that map onto them. Several UI interests share an underlying
    # measurement -- "Photography" and "Outdoor activities" both rest on
    # viewpoints and peaks -- and that is stated here rather than papered over
    # with an invented feature.
    TRAIT_LABELS: Dict[str, str] = {
        "museums": "Museums and galleries",
        "culture": "Culture and performance",
        "heritage": "History and heritage",
        "architecture": "Architecture",
        "nightlife": "Nightlife",
        "food": "Food and cafés",
        "nature": "Parks and green space",
        "beaches": "Beaches",
        "outdoor": "Outdoors and viewpoints",
        "family": "Family attractions",
        "shopping": "Markets and shopping",
        "walkability": "Walkable streets",
    }

    INTEREST_MAP: Dict[str, tuple] = {
        "architecture": ("architecture",),
        "culture": ("culture",),
        "food": ("food",),
        "nightlife": ("nightlife",),
        "museums": ("museums",),
        "nature": ("nature",),
        "history": ("heritage",),
        "beaches": ("beaches",),
        "shopping": ("shopping",),
        "local_life": ("walkability", "food"),
        "photography": ("outdoor", "architecture"),
        "outdoor": ("outdoor", "nature"),
    }

    def resolve_interests(self, interests: Sequence[str]) -> List[str]:
        """Map onboarding interest names onto measured OSM categories."""
        resolved: List[str] = []
        for interest in interests:
            for category in self.INTEREST_MAP.get(str(interest).lower(), ()):
                if category not in resolved:
                    resolved.append(category)
        return resolved

    def travel_profile(self, history: Sequence[str]) -> Dict[str, Any]:
        """Infer a travel profile from the destinations a traveller has visited.

        The traits are the categories where the visited set deviates *above the
        catalog norm*, not simply the categories that score highly. That
        distinction matters: every major European city sits in the top decile
        of museums, culture, nightlife and food, so raw percentiles describe
        "a big city" rather than this traveller. Standardising first is what
        separates a museum-and-heritage history (Amsterdam, Berlin, Prague,
        Paris) from a coastal one (Barcelona, Lisbon, Nice).
        """
        known, _ = filter_known_destinations(self.dataset, list(history))
        indices = self.dataset.indices(known)
        if indices.size == 0:
            return {"n_visited": 0, "traits": [], "visited": []}

        frame = self.dataset.destinations
        columns = [f"profile_{name}" for name in PROFILE_CATEGORIES]
        profiles = frame[columns].astype("float64")
        profiles = profiles.fillna(profiles.mean())
        spread = profiles.std().replace(0.0, 1.0)
        standardised = (profiles - profiles.mean()) / spread

        deviation = standardised.iloc[indices].mean().sort_values(ascending=False)
        traits = [
            {
                "category": name.replace("profile_", ""),
                "label": self.TRAIT_LABELS.get(name.replace("profile_", ""), name),
                "deviation": round(float(value), 3),
            }
            # Only report a trait when the history genuinely leans that way.
            for name, value in deviation.items()
            if value > 0.15
        ][:5]

        visited_rows = frame.iloc[indices]
        regions = visited_rows["region"].value_counts()
        continents = visited_rows["continent"].value_counts()
        cost_bands = visited_rows["cost_category"].value_counts()

        return {
            "n_visited": int(indices.size),
            "traits": traits,
            "region": str(regions.index[0]) if len(regions) else "",
            "continent": str(continents.index[0]) if len(continents) else "",
            "cost_band": str(cost_bands.index[0]) if len(cost_bands) else "",
            "walkable": bool(
                float(visited_rows["score_walkability"].mean(skipna=True) or 0) > 0.7
            ),
            "visited": [
                {
                    "destination_id": str(row["destination_id"]),
                    "city": str(row["city"]),
                    "country": str(row["country"]),
                    "country_code": str(row["country_code"]),
                    "image_url": str(row.get("image_url", "") or ""),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                }
                for _, row in visited_rows.iterrows()
            ],
        }

    # ------------------------------------------------------------- catalog
    def destinations_frame(self) -> pd.DataFrame:
        """The destination catalog, for listing endpoints and the UI."""
        return self.dataset.destinations

    def destination_detail(self, destination_id: str) -> Dict[str, Any]:
        """Full record for one destination, including similar destinations."""
        index = self.dataset.index_of.get(destination_id)
        if index is None:
            raise KeyError(destination_id)
        row = self.dataset.destinations.iloc[index]

        similar = [
            {
                "destination_id": other_id,
                "city": str(self.dataset.destinations.at[self.dataset.index_of[other_id], "city"]),
                "similarity": round(float(value), 4),
            }
            for other_id, value in self.content.similar_to(destination_id, k=8)
        ]

        return {
            "destination_id": destination_id,
            "city": str(row["city"]),
            "country": str(row["country"]),
            "country_code": str(row["country_code"]),
            "continent": str(row["continent"]),
            "region": str(row["region"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "population": int(row["population"]),
            "cost_category": str(row["cost_category"]),
            "cost_data_available": bool(row["cost_available"]),
            "popularity_percentile": round(float(row["popularity_score"]), 4),
            "osm_data_available": bool(row["poi_available"]),
            "climate_data_available": bool(row["climate_available"]),
            "attributes": {
                category: (
                    None
                    if pd.isna(row.get(f"score_{category}"))
                    else round(float(row[f"score_{category}"]), 3)
                )
                for category in self.DISPLAY_ATTRIBUTES
            },
            "monthly_climate": [
                {
                    "month": month,
                    "mean_temp_c": (
                        None
                        if pd.isna(row.get(f"temp_m{month:02d}"))
                        else round(float(row[f"temp_m{month:02d}"]), 1)
                    ),
                    "season_score": round(float(row[f"season_score_m{month:02d}"]), 3),
                }
                for month in range(1, 13)
            ],
            "summary": str(row.get("summary", ""))[:600],
            "image_url": str(row.get("image_url", "") or ""),
            "image_page": str(row.get("image_page", "") or ""),
            "wiki_title": str(row.get("wiki_title", "")),
            "similar_destinations": similar,
        }


@lru_cache(maxsize=1)
def get_service() -> RecommendationService:
    """Process-wide singleton, so models are fitted once per process."""
    return RecommendationService()
