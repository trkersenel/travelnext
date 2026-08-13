"""Feature engineering for the learning-to-rank stage.

For each ``(request, candidate destination)`` pair we assemble a feature vector
from signals that already exist elsewhere in the system: the component
recommender scores, the context signals, similarity to the traveller's history,
geographic relationships and the destination's own measured attributes.

Two rules govern what is allowed in here:

1. **Every feature must be computable at request time.** Nothing may depend on
   the held-out trip, on future interactions, or on the label in any form.
2. **Every feature must come from the free data sources.** No feature requires
   a paid API, a live price feed or a key.

The same builder is used for training and for serving, which is what keeps the
two consistent -- training/serving skew in the feature code is the most common
way a ranker silently degrades in production.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.candidate_generation.generator import CandidateSet
from src.data.dataset import TravelDataset
from src.models.base import RecommendationRequest
from src.models.collaborative import ItemItemCFRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.context import ContextScorer
from src.models.popularity import PopularityRecommender
from src.preprocessing.features import PROFILE_CATEGORIES
from src.utils.geo import haversine_km
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

CANDIDATE_SOURCES: Tuple[str, ...] = ("content", "collaborative", "geographic", "popularity")


class RankingFeatureBuilder:
    """Builds the learning-to-rank design matrix for candidate shortlists."""

    def __init__(
        self,
        dataset: TravelDataset,
        content: ContentBasedRecommender,
        collaborative: ItemItemCFRecommender,
        popularity: PopularityRecommender,
        context: Optional[ContextScorer] = None,
    ) -> None:
        self.dataset = dataset
        self.content = content
        self.collaborative = collaborative
        self.popularity = popularity
        self.context = context or ContextScorer(dataset)

        frame = dataset.destinations
        self._latitude = frame["latitude"].to_numpy(dtype="float64")
        self._longitude = frame["longitude"].to_numpy(dtype="float64")
        self._popularity_percentile = frame["popularity_score"].to_numpy(dtype="float64")
        self._cost_percentile = frame["cost_percentile"].to_numpy(dtype="float64")
        self._population_log = frame["population_log"].to_numpy(dtype="float64")
        self._seasonality = frame["seasonality"].to_numpy(dtype="float64")
        self._country = frame["country_code"].to_numpy()
        self._region = frame["region"].to_numpy()
        self._continent = frame["continent"].to_numpy()

        profile_columns = [f"profile_{name}" for name in PROFILE_CATEGORIES]
        profiles = frame[profile_columns].astype("float64")
        self._profiles = profiles.fillna(profiles.mean(axis=0)).to_numpy()

    @property
    def feature_names(self) -> List[str]:
        """Names of the produced features, in column order."""
        return [
            # component recommender scores
            "content_score",
            "collaborative_score",
            "popularity_score",
            "context_score",
            # destination properties
            "popularity_percentile",
            "cost_percentile",
            "population_log",
            "seasonality",
            # context fit
            "season_fit",
            "budget_fit",
            "proximity",
            "distance_km_log",
            # similarity to the traveller's history
            "max_content_similarity",
            "mean_content_similarity",
            "max_cf_similarity",
            "profile_cosine",
            # geographic relationship to the origin
            "same_country",
            "same_region",
            "same_continent",
            # request-level descriptors
            "history_length",
            "trip_duration_days",
            "month_sin",
            "month_cos",
            # candidate provenance
            *[f"from_{source}" for source in CANDIDATE_SOURCES],
        ]

    def build(
        self,
        request: RecommendationRequest,
        candidates: CandidateSet,
    ) -> np.ndarray:
        """Return the ``(n_candidates, n_features)`` matrix for one request."""
        indices = candidates.indices
        if indices.size == 0:
            return np.zeros((0, len(self.feature_names)), dtype="float32")

        content_scores = self.content.score(request)
        collaborative_scores = self.collaborative.score(request)
        popularity_scores = self.popularity.score(request)
        context_scores = self.context.score(request)

        history = list(request.history)
        if request.current_destination:
            history.append(request.current_destination)
        history_indices = self.dataset.indices(history)

        # --- similarity of each candidate to the visited set ---------------
        if history_indices.size:
            text = self.content.text_matrix
            similarity_block = np.asarray(
                (text[indices] @ text[history_indices].T).todense(), dtype="float64"
            )
            max_content = similarity_block.max(axis=1)
            mean_content = similarity_block.mean(axis=1)

            cf_similarity = self.collaborative.similarity
            if cf_similarity is not None:
                cf_block = np.asarray(
                    cf_similarity[indices][:, history_indices].todense(), dtype="float64"
                )
                max_cf = cf_block.max(axis=1) if cf_block.size else np.zeros(indices.size)
            else:
                max_cf = np.zeros(indices.size)

            user_profile = self._profiles[history_indices].mean(axis=0)
            profile_norm = np.linalg.norm(user_profile)
            candidate_profiles = self._profiles[indices]
            candidate_norms = np.linalg.norm(candidate_profiles, axis=1)
            denominator = candidate_norms * profile_norm
            profile_cosine = np.divide(
                candidate_profiles @ user_profile,
                denominator,
                out=np.zeros(indices.size),
                where=denominator > 0,
            )
        else:
            max_content = np.zeros(indices.size)
            mean_content = np.zeros(indices.size)
            max_cf = np.zeros(indices.size)
            profile_cosine = np.zeros(indices.size)

        # --- geographic relationship to the origin -------------------------
        origin = request.current_destination or (history[-1] if history else None)
        origin_index = self.dataset.index_of.get(origin) if origin else None
        if origin_index is not None:
            distances = haversine_km(
                self._latitude[origin_index],
                self._longitude[origin_index],
                self._latitude[indices],
                self._longitude[indices],
            )
            same_country = (self._country[indices] == self._country[origin_index]).astype("float64")
            same_region = (self._region[indices] == self._region[origin_index]).astype("float64")
            same_continent = (
                self._continent[indices] == self._continent[origin_index]
            ).astype("float64")
        else:
            distances = np.full(indices.size, np.nan)
            same_country = np.zeros(indices.size)
            same_region = np.zeros(indices.size)
            same_continent = np.zeros(indices.size)

        # Unknown distance becomes the median rather than 0, so "no origin"
        # does not masquerade as "right next door".
        distance_log = np.log1p(np.nan_to_num(distances, nan=float(np.nanmedian(distances)) if np.isfinite(distances).any() else 3000.0))

        month = request.month
        month_angle = 0.0 if month is None else 2.0 * np.pi * (int(month) - 1) / 12.0
        month_sin = np.full(indices.size, np.sin(month_angle) if month else 0.0)
        month_cos = np.full(indices.size, np.cos(month_angle) if month else 0.0)

        source_columns = [
            np.array(
                [1.0 if source in candidates.sources.get(int(i), set()) else 0.0 for i in indices]
            )
            for source in CANDIDATE_SOURCES
        ]

        columns = [
            content_scores[indices],
            collaborative_scores[indices],
            popularity_scores[indices],
            context_scores.combined()[indices],
            self._popularity_percentile[indices],
            self._cost_percentile[indices],
            self._population_log[indices],
            self._seasonality[indices],
            context_scores.season[indices],
            context_scores.budget[indices],
            context_scores.proximity[indices],
            distance_log,
            max_content,
            mean_content,
            max_cf,
            profile_cosine,
            same_country,
            same_region,
            same_continent,
            np.full(indices.size, float(len(history))),
            np.full(indices.size, float(request.trip_duration_days or 0)),
            month_sin,
            month_cos,
            *source_columns,
        ]
        matrix = np.column_stack(columns).astype("float32")
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def build_training_data(
    dataset: TravelDataset,
    builder: RankingFeatureBuilder,
    generator,
    training_requests: Sequence[Tuple[RecommendationRequest, str]],
    *,
    negatives_per_positive: int = 30,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble ``(X, y, group_sizes)`` for LambdaRank training.

    Each training example is one traveller's shortlist, labelled 1 for the
    destination they actually chose next and 0 otherwise. Negatives are
    subsampled from the shortlist to keep groups small and balanced; the
    positive is force-inserted when candidate generation missed it, so the
    ranker always sees a target to rank.
    """
    rng = np.random.default_rng(seed)
    feature_blocks: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    groups: List[int] = []
    missed_positives = 0

    for request, target_id in training_requests:
        target_index = dataset.index_of.get(target_id)
        if target_index is None:
            continue

        candidates = generator.generate(request)
        candidate_indices = candidates.indices
        if target_index not in set(candidate_indices.tolist()):
            missed_positives += 1
            candidate_indices = np.append(candidate_indices, target_index)
            candidates.sources.setdefault(int(target_index), set())

        negatives = candidate_indices[candidate_indices != target_index]
        if negatives.size > negatives_per_positive:
            negatives = rng.choice(negatives, size=negatives_per_positive, replace=False)

        group_indices = np.append(negatives, target_index)
        subset = CandidateSet(indices=group_indices, sources=candidates.sources)

        block = builder.build(request, subset)
        if block.shape[0] == 0:
            continue
        label = (group_indices == target_index).astype("int32")

        feature_blocks.append(block)
        labels.append(label)
        groups.append(len(group_indices))

    if not feature_blocks:
        raise ValueError("No training groups could be built for the ranker")

    LOGGER.info(
        "LTR training data: %d groups, %d rows, %d positives missed by candidate generation",
        len(groups),
        sum(groups),
        missed_positives,
    )
    return (
        np.vstack(feature_blocks),
        np.concatenate(labels),
        np.array(groups, dtype="int64"),
    )
