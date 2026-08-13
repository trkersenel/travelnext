"""The "where should I go after X?" recommender.

This is the project's headline mode, and it differs from the general
recommender in one important way: it is anchored on a *single* origin rather
than on a whole history. That makes a signal available which the other models
do not use -- the **transition matrix** of consecutive trips.

Co-visitation (used by item-item CF) asks "which destinations appear in the
same traveller's history?". Transitions ask the sharper question "which
destination did travellers go to *immediately after* this one?", which is
exactly the question being posed. Amsterdam -> Rotterdam is a strong
transition; Amsterdam -> Sydney may be a co-visit but is rarely a next step.

The final score blends five signals, all configurable:

    transition  observed next-trip frequency after the origin
    content     attribute and text similarity to the origin
    collaborative co-visitation similarity
    geographic  proximity, since a next trip is often a nearby one
    popularity  a prior that keeps the tail sane

As everywhere in this project, the transition statistics here are learned from
synthetic interactions and demonstrate the mechanism rather than real travel
patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from src.data.dataset import TravelDataset
from src.models.base import BaseRecommender, RecommendationRequest, normalise_scores
from src.models.collaborative import ItemItemCFRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.popularity import PopularityRecommender
from src.utils.geo import distance_decay, pairwise_distance_matrix
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class NextDestinationWeights:
    """Blend weights for the next-destination score."""

    transition: float = 0.30
    content: float = 0.25
    collaborative: float = 0.20
    geographic: float = 0.15
    popularity: float = 0.10

    def normalised(self) -> "NextDestinationWeights":
        total = (
            self.transition
            + self.content
            + self.collaborative
            + self.geographic
            + self.popularity
        )
        if total <= 0:
            return NextDestinationWeights(0.0, 0.0, 0.0, 0.0, 1.0)
        return NextDestinationWeights(
            transition=self.transition / total,
            content=self.content / total,
            collaborative=self.collaborative / total,
            geographic=self.geographic / total,
            popularity=self.popularity / total,
        )


class NextDestinationRecommender(BaseRecommender):
    """Recommends the next destination given a single current destination."""

    name = "next_destination"

    def __init__(
        self,
        content: ContentBasedRecommender,
        collaborative: ItemItemCFRecommender,
        popularity: PopularityRecommender,
        *,
        weights: Optional[NextDestinationWeights] = None,
        geo_scale_km: float = 900.0,
        transition_smoothing: float = 1.0,
    ) -> None:
        super().__init__()
        self.content = content
        self.collaborative = collaborative
        self.popularity = popularity
        self.weights = weights or NextDestinationWeights()
        self.geo_scale_km = float(geo_scale_km)
        self.transition_smoothing = float(transition_smoothing)
        self.transitions: Optional[sparse.csr_matrix] = None
        self._distances: Optional[np.ndarray] = None

    def _fit(self, dataset: TravelDataset, train_interactions: pd.DataFrame) -> None:
        for component in (self.content, self.collaborative, self.popularity):
            if not component._fitted:
                component.fit(dataset, train_interactions)

        self.transitions = build_transition_matrix(dataset, train_interactions)
        self._distances = pairwise_distance_matrix(
            dataset.destinations["latitude"].to_numpy(),
            dataset.destinations["longitude"].to_numpy(),
        )
        LOGGER.info(
            "Next-destination model: %d observed transitions", int(self.transitions.nnz)
        )

    def transition_scores(self, origin_index: int) -> np.ndarray:
        """Normalised distribution over destinations visited after ``origin``."""
        dataset = self._require_fitted()
        if self.transitions is None:
            return np.zeros(dataset.n_destinations)
        row = np.asarray(self.transitions[origin_index].todense()).ravel()
        total = row.sum()
        if total <= 0:
            # Never observed as an origin: no transition evidence exists, and
            # saying so honestly lets the other signals carry the ranking.
            return np.zeros(dataset.n_destinations)
        return row / total

    def score(self, request: RecommendationRequest) -> np.ndarray:
        dataset = self._require_fitted()
        origin = request.current_destination or (request.history[-1] if request.history else None)
        origin_index = dataset.index_of.get(origin) if origin else None
        if origin_index is None:
            # No anchor: fall back to popularity, which is the honest answer to
            # "where next?" when we do not know where you are.
            return normalise_scores(self.popularity.score(request))

        anchored = RecommendationRequest(
            history=list(request.history),
            current_destination=origin,
            month=request.month,
            trip_duration_days=request.trip_duration_days,
            budget=request.budget,
            interests=list(request.interests),
        )

        weights = self.weights.normalised()
        geographic = (
            distance_decay(self._distances[origin_index], self.geo_scale_km)
            if self._distances is not None
            else np.zeros(dataset.n_destinations)
        )

        return (
            weights.transition * normalise_scores(self.transition_scores(origin_index))
            + weights.content * normalise_scores(self.content.score(anchored))
            + weights.collaborative * normalise_scores(self.collaborative.score(anchored))
            + weights.geographic * normalise_scores(geographic)
            + weights.popularity * normalise_scores(self.popularity.score(anchored))
        )

    def component_scores(self, origin: str) -> Dict[str, np.ndarray]:
        """Per-signal score vectors for one origin, used by explanations."""
        dataset = self._require_fitted()
        origin_index = dataset.index_of.get(origin)
        if origin_index is None:
            return {}
        anchored = RecommendationRequest(current_destination=origin)
        return {
            "transition": normalise_scores(self.transition_scores(origin_index)),
            "content": normalise_scores(self.content.score(anchored)),
            "collaborative": normalise_scores(self.collaborative.score(anchored)),
            "geographic": normalise_scores(
                distance_decay(self._distances[origin_index], self.geo_scale_km)
                if self._distances is not None
                else np.zeros(dataset.n_destinations)
            ),
            "popularity": normalise_scores(self.popularity.score(anchored)),
        }

    def recommend_next(self, origin: str, k: int = 10, **context) -> List[Tuple[str, float]]:
        """Convenience API: ``recommend_next("amsterdam-nl", k=5)``."""
        request = RecommendationRequest(current_destination=origin, **context)
        return [(item.destination_id, item.score) for item in self.recommend(request, k)]


def build_transition_matrix(
    dataset: TravelDataset, interactions: pd.DataFrame
) -> sparse.csr_matrix:
    """Count consecutive (previous -> next) destination pairs per traveller.

    Only pairs that are adjacent in one user's ordered history are counted, so
    this measures sequence, not mere co-occurrence.
    """
    ordered = interactions.sort_values(["user_id", "trip_index"])
    indices = ordered["destination_id"].map(dataset.index_of)
    users = ordered["user_id"].to_numpy()

    previous_index = indices.shift(1)
    same_user = pd.Series(users).eq(pd.Series(users).shift(1)).to_numpy()

    valid = same_user & indices.notna().to_numpy() & previous_index.notna().to_numpy()
    rows = previous_index[valid].astype("int64").to_numpy()
    columns = indices[valid].astype("int64").to_numpy()

    matrix = sparse.csr_matrix(
        (np.ones(rows.size, dtype="float32"), (rows, columns)),
        shape=(dataset.n_destinations, dataset.n_destinations),
    )
    matrix.sum_duplicates()
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return matrix
