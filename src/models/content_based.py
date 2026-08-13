"""Content-based recommender over measured destination attributes.

A destination is represented by two blocks:

* a **text block** -- TF-IDF over the Wikipedia lead extract plus structured
  metadata, capturing character words like "canal", "baroque" or "beach";
* an **attribute block** -- the L2-normalised OpenStreetMap profile shares,
  capturing measured composition (how museum-heavy, how nightlife-heavy).

A user profile is the recency-weighted mean of the destinations they visited,
computed separately in each block. Candidate scores are the cosine similarity
between profile and destination in each block, combined with configurable
weights.

Because the representation is built entirely from destination content, this
model handles a brand-new destination with no interaction history (item cold
start) and a brand-new user with a single visit (user cold start) equally well.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.data.dataset import TravelDataset
from src.models.base import BaseRecommender, RecommendationRequest
from src.preprocessing.features import PROFILE_CATEGORIES, profile_matrix
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """Cosine similarity between a user preference profile and destinations."""

    name = "content"

    def __init__(
        self,
        *,
        tfidf_max_features: int = 20000,
        tfidf_ngram_max: int = 2,
        text_weight: float = 0.45,
        attribute_weight: float = 0.55,
        recency_decay: float = 0.85,
    ) -> None:
        super().__init__()
        self.tfidf_max_features = int(tfidf_max_features)
        self.tfidf_ngram_max = int(tfidf_ngram_max)
        self.text_weight = float(text_weight)
        self.attribute_weight = float(attribute_weight)
        self.recency_decay = float(recency_decay)

        self.vectoriser: Optional[TfidfVectorizer] = None
        self.text_matrix = None  # scipy sparse, L2-normalised rows
        self.attribute_matrix: np.ndarray = np.zeros((0, 0))

    def _fit(self, dataset: TravelDataset, train_interactions: pd.DataFrame) -> None:
        # Note: this model reads only destination content, never interactions,
        # so there is nothing here that could leak held-out trips.
        documents = dataset.destinations["text_blob"].fillna("").astype(str).tolist()
        self.vectoriser = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, self.tfidf_ngram_max),
            stop_words="english",
            min_df=2,
            sublinear_tf=True,
        )
        self.text_matrix = normalize(self.vectoriser.fit_transform(documents))
        self.attribute_matrix = profile_matrix(dataset.destinations, PROFILE_CATEGORIES)
        LOGGER.info(
            "Content model: %d docs, %d TF-IDF terms, %d attribute dimensions",
            len(documents),
            self.text_matrix.shape[1],
            self.attribute_matrix.shape[1],
        )

    def _recency_weights(self, count: int) -> np.ndarray:
        """Exponentially decaying weights, heaviest on the most recent trip."""
        if count <= 0:
            return np.zeros(0)
        # history[-1] is the latest trip -> weight 1.0.
        exponents = np.arange(count - 1, -1, -1, dtype="float64")
        weights = self.recency_decay**exponents
        return weights / weights.sum()

    def build_profile(self, history: Sequence[str]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return the (text, attribute) profile vectors for a visit history."""
        dataset = self._require_fitted()
        indices = dataset.indices(list(history))
        if indices.size == 0:
            return None, None
        weights = self._recency_weights(indices.size)

        text_profile = np.asarray(
            self.text_matrix[indices].multiply(weights[:, None]).sum(axis=0)
        ).ravel()
        attribute_profile = (self.attribute_matrix[indices] * weights[:, None]).sum(axis=0)
        return text_profile, attribute_profile

    def score(self, request: RecommendationRequest) -> np.ndarray:
        dataset = self._require_fitted()
        history = list(request.history)
        if request.current_destination:
            history.append(request.current_destination)

        text_profile, attribute_profile = self.build_profile(history)
        if text_profile is None or attribute_profile is None:
            # Cold start: fall back to the interest vector if the user gave one,
            # otherwise return a flat score and let the caller blend popularity.
            return self._interest_scores(request.interests)

        text_scores = np.asarray(self.text_matrix @ text_profile).ravel()
        attribute_scores = self.attribute_matrix @ attribute_profile

        combined = self.text_weight * text_scores + self.attribute_weight * attribute_scores
        if request.interests:
            # Stated interests nudge the profile without overriding history.
            combined = combined + 0.5 * self._interest_scores(request.interests)
        return combined

    def _interest_scores(self, interests: Sequence[str]) -> np.ndarray:
        """Score destinations against explicitly selected interest categories.

        This is the onboarding path for a user with no travel history at all.
        """
        dataset = self._require_fitted()
        valid = [i for i in interests if i in PROFILE_CATEGORIES]
        if not valid:
            return np.zeros(dataset.n_destinations)
        columns = [f"score_{name}" for name in valid]
        frame = dataset.destinations[columns].astype("float64")
        return frame.fillna(frame.mean(axis=0)).mean(axis=1).to_numpy()

    def similar_to(self, destination_id: str, k: int = 10) -> List[tuple[str, float]]:
        """Return the ``k`` destinations most similar to ``destination_id``.

        Powers item-to-item explanations ("similar to Amsterdam") and the
        content branch of candidate generation.
        """
        dataset = self._require_fitted()
        index = dataset.index_of.get(destination_id)
        if index is None:
            return []
        text_scores = np.asarray(self.text_matrix @ self.text_matrix[index].T.toarray()).ravel()
        attribute_scores = self.attribute_matrix @ self.attribute_matrix[index]
        combined = self.text_weight * text_scores + self.attribute_weight * attribute_scores
        combined[index] = -np.inf
        top = np.argsort(-combined)[: max(0, k)]
        return [(dataset.destination_ids[int(i)], float(combined[int(i)])) for i in top]
