"""Cold-start handling for new users and new destinations.

Three distinct situations must never crash and never return an empty list:

**New user, no history.** Nothing personalised can be inferred from behaviour,
so the system falls back to what it does know: stated interests, a preferred
region if the user picked one, the travel month, the budget, and popularity as
the final prior. This is the onboarding path.

**New destination, no interactions.** Collaborative filtering has no signal for
it, but every content feature (OSM attribute profile, Wikipedia text, climate,
cost) exists from the moment the destination is ingested, so the content model
ranks it normally. This is exactly why the hybrid keeps a content component
rather than relying on collaborative signal alone.

**Unknown identifiers.** An unrecognised destination id in a history is
skipped rather than raising, so a stale bookmark or a typo degrades the
recommendation quality slightly instead of returning a 500.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.dataset import TravelDataset
from src.models.base import BaseRecommender, RecommendationRequest, normalise_scores
from src.models.content_based import ContentBasedRecommender
from src.models.context import ContextScorer
from src.models.popularity import PopularityRecommender
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


class ColdStartRecommender(BaseRecommender):
    """Ranks destinations for a traveller with no usable history."""

    name = "cold_start"

    def __init__(
        self,
        popularity: PopularityRecommender,
        content: ContentBasedRecommender,
        context: Optional[ContextScorer] = None,
        *,
        interest_weight: float = 0.45,
        popularity_weight: float = 0.30,
        context_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.popularity = popularity
        self.content = content
        self.context = context
        self.interest_weight = float(interest_weight)
        self.popularity_weight = float(popularity_weight)
        self.context_weight = float(context_weight)

    def _fit(self, dataset: TravelDataset, train_interactions: pd.DataFrame) -> None:
        for component in (self.popularity, self.content):
            if not component._fitted:
                component.fit(dataset, train_interactions)
        if self.context is None:
            self.context = ContextScorer(dataset)

    def region_scores(
        self, continents: Sequence[str], countries: Sequence[str]
    ) -> np.ndarray:
        """Binary preference for onboarding-selected continents or countries."""
        dataset = self._require_fitted()
        frame = dataset.destinations
        if not continents and not countries:
            return np.zeros(dataset.n_destinations)
        mask = np.zeros(dataset.n_destinations, dtype="float64")
        if continents:
            mask += frame["continent"].isin(list(continents)).to_numpy(dtype="float64")
        if countries:
            upper = [c.upper() for c in countries]
            mask += frame["country_code"].isin(upper).to_numpy(dtype="float64")
        return np.clip(mask, 0.0, 1.0)

    def score(self, request: RecommendationRequest) -> np.ndarray:
        dataset = self._require_fitted()
        popularity_scores = normalise_scores(self.popularity.score(request))

        # ``interests`` drives the content model's onboarding path; with no
        # interests selected this is a zero vector and popularity dominates,
        # which is the correct behaviour for a genuinely unknown traveller.
        interest_scores = normalise_scores(self.content.score(request))

        context_scores = (
            normalise_scores(self.context.score(request).combined())
            if self.context is not None
            else np.zeros(dataset.n_destinations)
        )

        return (
            self.interest_weight * interest_scores
            + self.popularity_weight * popularity_scores
            + self.context_weight * context_scores
        )


def is_cold_start(dataset: TravelDataset, request: RecommendationRequest) -> bool:
    """True when a request carries no history the catalog recognises.

    A history made entirely of unknown destination ids counts as cold, since
    none of it can be turned into a profile.
    """
    known = dataset.indices(list(request.visited()))
    return known.size == 0


def filter_known_destinations(
    dataset: TravelDataset, destination_ids: Sequence[str]
) -> tuple[List[str], List[str]]:
    """Split ids into ``(known, unknown)`` without raising on the unknown ones."""
    known = [d for d in destination_ids if d in dataset.index_of]
    unknown = [d for d in destination_ids if d not in dataset.index_of]
    if unknown:
        LOGGER.info("Ignoring %d unknown destination id(s): %s", len(unknown), unknown[:5])
    return known, unknown
