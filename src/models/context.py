"""Context-aware scoring: season, budget, trip duration and geography.

These signals are deliberately kept separate from the collaborative and content
models so they can be reused three ways: as a re-ranking term in the hybrid, as
features for the learning-to-rank model, and as material for explanations.

Every value is derived from measured destination data (Open-Meteo climate
normals, World Bank cost proxy, great-circle distance). None of it depends on
live prices, flight availability or any paid API, so the system behaves
identically offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.dataset import TravelDataset
from src.models.base import RecommendationRequest
from src.utils.geo import distance_decay, haversine_km

# Budget labels mapped to their target position on the cost percentile scale.
BUDGET_TARGETS: Dict[str, float] = {"budget": 1 / 6, "mid-range": 0.5, "expensive": 5 / 6}

# A short trip realistically means staying closer to home; a long one opens up
# the map. These are the 1/e distances of the proximity kernel, in kilometres.
_SHORT_TRIP_SCALE_KM = 600.0
_LONG_TRIP_SCALE_KM = 6000.0
_SHORT_TRIP_DAYS = 3.0
_LONG_TRIP_DAYS = 14.0


@dataclass
class ContextScores:
    """Per-destination context signals for one request."""

    season: np.ndarray
    budget: np.ndarray
    distance_km: np.ndarray
    proximity: np.ndarray

    def combined(
        self,
        *,
        season_weight: float = 1.0,
        budget_weight: float = 1.0,
        proximity_weight: float = 1.0,
    ) -> np.ndarray:
        """Weighted mean of the individual context signals, in [0, 1]."""
        total_weight = season_weight + budget_weight + proximity_weight
        if total_weight <= 0:
            return np.zeros_like(self.season)
        return (
            season_weight * self.season
            + budget_weight * self.budget
            + proximity_weight * self.proximity
        ) / total_weight


class ContextScorer:
    """Computes context signals for a request against the whole catalog."""

    def __init__(self, dataset: TravelDataset) -> None:
        self.dataset = dataset
        frame = dataset.destinations
        self._season = frame[[f"season_score_m{m:02d}" for m in range(1, 13)]].to_numpy(
            dtype="float64"
        )
        self._cost = frame["cost_percentile"].to_numpy(dtype="float64")
        self._latitude = frame["latitude"].to_numpy(dtype="float64")
        self._longitude = frame["longitude"].to_numpy(dtype="float64")

    def season_scores(self, month: Optional[int]) -> np.ndarray:
        """Climate comfort for the requested month; neutral when unspecified."""
        if month is None:
            return np.full(len(self._cost), 0.5)
        index = int(np.clip(month, 1, 12)) - 1
        return self._season[:, index]

    def budget_scores(self, budget: Optional[str]) -> np.ndarray:
        """Closeness of each destination's cost band to the requested budget."""
        if not budget:
            return np.full(len(self._cost), 0.5)
        target = BUDGET_TARGETS.get(str(budget).lower())
        if target is None:
            return np.full(len(self._cost), 0.5)
        # 1 - distance on the percentile scale, so an exact band match is 1.0.
        return 1.0 - np.abs(self._cost - target)

    def distance_from(self, origin_id: Optional[str]) -> np.ndarray:
        """Great-circle distance from ``origin_id`` to every destination."""
        if not origin_id:
            return np.full(len(self._cost), np.nan)
        index = self.dataset.index_of.get(origin_id)
        if index is None:
            return np.full(len(self._cost), np.nan)
        return haversine_km(
            self._latitude[index], self._longitude[index], self._latitude, self._longitude
        )

    def proximity_scores(
        self, origin_id: Optional[str], trip_duration_days: Optional[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(distance_km, proximity_score)`` for the request.

        The decay scale is interpolated from trip duration: a 2-day break
        favours neighbours, a 3-week trip barely penalises distance at all.
        """
        distances = self.distance_from(origin_id)
        if np.all(np.isnan(distances)):
            return distances, np.full(len(self._cost), 0.5)

        if trip_duration_days is None:
            scale = 2000.0
        else:
            days = float(np.clip(trip_duration_days, _SHORT_TRIP_DAYS, _LONG_TRIP_DAYS))
            fraction = (days - _SHORT_TRIP_DAYS) / (_LONG_TRIP_DAYS - _SHORT_TRIP_DAYS)
            scale = _SHORT_TRIP_SCALE_KM + fraction * (_LONG_TRIP_SCALE_KM - _SHORT_TRIP_SCALE_KM)

        proximity = distance_decay(np.nan_to_num(distances, nan=0.0), scale)
        return distances, proximity

    def score(self, request: RecommendationRequest) -> ContextScores:
        """Compute every context signal for ``request``."""
        origin = request.current_destination
        if origin is None and request.history:
            # Fall back to the most recent trip as the point of departure.
            origin = request.history[-1]
        distances, proximity = self.proximity_scores(origin, request.trip_duration_days)
        return ContextScores(
            season=self.season_scores(request.month),
            budget=self.budget_scores(request.budget),
            distance_km=distances,
            proximity=proximity,
        )
