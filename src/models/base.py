"""Shared recommender interface.

Every model exposes the same two-part contract:

``fit(dataset, train_interactions)``
    Learn from *training interactions only*. Nothing in a model may read the
    validation or test partitions.

``score(request)``
    Return a dense score vector over the entire catalog, aligned with
    ``dataset.destination_ids``.

Returning a full score vector rather than a ranked list is what lets the hybrid
model blend components and the learning-to-rank stage consume them as features,
without any model needing to know about the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.dataset import TravelDataset


@dataclass
class RecommendationRequest:
    """Everything a model may condition on when ranking destinations.

    All fields are optional: a request with nothing set is the cold-start case
    and every model must still return a usable ranking.
    """

    history: List[str] = field(default_factory=list)
    current_destination: Optional[str] = None
    month: Optional[int] = None
    trip_duration_days: Optional[int] = None
    budget: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    # Destinations to keep out of the results regardless of score.
    exclude: List[str] = field(default_factory=list)

    def visited(self) -> set[str]:
        """Destinations the traveller has already been to."""
        seen = set(self.history)
        if self.current_destination:
            seen.add(self.current_destination)
        return seen

    def is_cold_start(self) -> bool:
        """True when there is no history to personalise from."""
        return not self.history and not self.current_destination


@dataclass
class ScoredDestination:
    """One ranked recommendation with the component scores behind it."""

    destination_id: str
    score: float
    rank: int
    components: Dict[str, float] = field(default_factory=dict)


class BaseRecommender:
    """Base class providing masking, ranking and cold-start plumbing."""

    name: str = "base"

    def __init__(self) -> None:
        self.dataset: Optional[TravelDataset] = None
        self._fitted = False

    # ------------------------------------------------------------ fitting
    def fit(self, dataset: TravelDataset, train_interactions: pd.DataFrame) -> "BaseRecommender":
        """Fit the model on training interactions. Subclasses override ``_fit``."""
        self.dataset = dataset
        self._fit(dataset, train_interactions)
        self._fitted = True
        return self

    def _fit(self, dataset: TravelDataset, train_interactions: pd.DataFrame) -> None:
        """Model-specific fitting. Default is a no-op."""

    def _require_fitted(self) -> TravelDataset:
        if not self._fitted or self.dataset is None:
            raise RuntimeError(f"{self.name} recommender used before fit()")
        return self.dataset

    # ------------------------------------------------------------ scoring
    def score(self, request: RecommendationRequest) -> np.ndarray:
        """Return a score for every destination in catalog order."""
        raise NotImplementedError

    def recommend(
        self,
        request: RecommendationRequest,
        k: int = 10,
        *,
        candidate_ids: Optional[Sequence[str]] = None,
    ) -> List[ScoredDestination]:
        """Rank destinations and return the top ``k``.

        Already-visited and explicitly excluded destinations are removed before
        ranking. ``k`` larger than the number of available destinations simply
        returns everything available rather than raising.
        """
        dataset = self._require_fitted()
        scores = np.asarray(self.score(request), dtype="float64")
        if scores.shape[0] != dataset.n_destinations:
            raise ValueError(
                f"{self.name} returned {scores.shape[0]} scores for "
                f"{dataset.n_destinations} destinations"
            )

        mask = np.ones(dataset.n_destinations, dtype=bool)
        blocked = request.visited() | set(request.exclude)
        if blocked:
            blocked_indices = dataset.indices(sorted(blocked))
            if blocked_indices.size:
                mask[blocked_indices] = False
        if candidate_ids is not None:
            allowed = np.zeros(dataset.n_destinations, dtype=bool)
            allowed_indices = dataset.indices(list(candidate_ids))
            if allowed_indices.size:
                allowed[allowed_indices] = True
            mask &= allowed

        available = np.flatnonzero(mask)
        if available.size == 0:
            return []

        k = max(0, min(int(k), available.size))
        if k == 0:
            return []

        available_scores = scores[available]
        # argpartition then sort: O(n) selection instead of a full sort.
        top_unsorted = np.argpartition(-available_scores, k - 1)[:k]
        order = top_unsorted[np.argsort(-available_scores[top_unsorted], kind="stable")]
        chosen = available[order]

        return [
            ScoredDestination(
                destination_id=dataset.destination_ids[int(index)],
                score=float(scores[int(index)]),
                rank=position + 1,
            )
            for position, index in enumerate(chosen)
        ]

    def recommend_ids(self, request: RecommendationRequest, k: int = 10) -> List[str]:
        """Convenience wrapper returning just the ranked destination ids."""
        return [item.destination_id for item in self.recommend(request, k)]


def normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Rescale scores to [0, 1] for comparable blending across models.

    Uses min-max over finite values. A constant vector maps to all zeros, which
    correctly makes that component contribute nothing to a hybrid blend.
    """
    values = np.asarray(scores, dtype="float64")
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    lowest = values[finite].min()
    highest = values[finite].max()
    if highest <= lowest:
        return np.zeros_like(values)
    scaled = (values - lowest) / (highest - lowest)
    return np.clip(np.where(finite, scaled, 0.0), 0.0, 1.0)
