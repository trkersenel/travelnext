"""Popularity baseline.

The point of this model is to be the bar every other model must clear. In
recommender systems a well-implemented popularity baseline is deceptively
strong, and reporting a fancy model that fails to beat it is the single most
common way published results mislead.

Popularity is counted from **training interactions**, not from the Wikipedia
pageview proxy. Using the external proxy would make this a content model in
disguise and would not answer the question "can a model beat simply
recommending what most travellers picked?".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset import TravelDataset
from src.models.base import BaseRecommender, RecommendationRequest


class PopularityRecommender(BaseRecommender):
    """Ranks every traveller identically, by observed training frequency."""

    name = "popularity"

    def __init__(self, smoothing: float = 1.0) -> None:
        super().__init__()
        # Additive smoothing keeps never-visited destinations rankable rather
        # than tied at exactly zero.
        self.smoothing = float(smoothing)
        self.counts: np.ndarray = np.zeros(0)
        self.scores_: np.ndarray = np.zeros(0)

    def _fit(self, dataset: TravelDataset, train_interactions: pd.DataFrame) -> None:
        counts = np.zeros(dataset.n_destinations, dtype="float64")
        observed = train_interactions["destination_id"].value_counts()
        for destination_id, count in observed.items():
            index = dataset.index_of.get(str(destination_id))
            if index is not None:
                counts[index] = float(count)
        self.counts = counts
        total = counts.sum() + self.smoothing * dataset.n_destinations
        self.scores_ = (counts + self.smoothing) / total

    def score(self, request: RecommendationRequest) -> np.ndarray:
        self._require_fitted()
        return self.scores_.copy()
