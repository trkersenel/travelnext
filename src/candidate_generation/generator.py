"""Stage 1 of the two-stage architecture: candidate generation.

Scoring every destination with an expensive ranker is wasteful, and at a larger
catalog it would be infeasible. Instead four cheap, complementary retrievers
each nominate a shortlist, the union is de-duplicated, and only that shortlist
(typically 100-200 destinations) reaches the learning-to-rank stage.

The four sources are chosen to fail in different ways, so their union has much
better recall than any one of them:

``content``       destinations resembling what the traveller already liked
``collaborative`` destinations co-visited by similar travellers
``geographic``    destinations near the current one (trip chaining)
``popularity``    a safety net that guarantees a non-empty, sane shortlist

Recall of the candidate set is measurable and is reported by
``src/evaluation/run_experiments.py``: if stage 1 drops the correct answer, no
amount of ranking skill in stage 2 can recover it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from src.data.dataset import TravelDataset
from src.models.base import RecommendationRequest
from src.models.collaborative import ItemItemCFRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.popularity import PopularityRecommender
from src.utils.geo import haversine_km
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class CandidateSet:
    """Candidate destination indices plus which retrievers nominated each."""

    indices: np.ndarray
    sources: Dict[int, Set[str]] = field(default_factory=dict)

    def source_flags(self, index: int) -> Dict[str, float]:
        """Binary indicators of which retrievers nominated ``index``."""
        nominated = self.sources.get(int(index), set())
        return {
            f"from_{source}": 1.0 if source in nominated else 0.0
            for source in ("content", "collaborative", "geographic", "popularity")
        }

    def __len__(self) -> int:
        return int(self.indices.size)


class CandidateGenerator:
    """Unions several cheap retrievers into one shortlist per request."""

    def __init__(
        self,
        dataset: TravelDataset,
        content: ContentBasedRecommender,
        collaborative: ItemItemCFRecommender,
        popularity: PopularityRecommender,
        *,
        n_candidates: int = 150,
        per_source: Optional[Dict[str, int]] = None,
    ) -> None:
        self.dataset = dataset
        self.content = content
        self.collaborative = collaborative
        self.popularity = popularity
        self.n_candidates = int(n_candidates)
        self.per_source = per_source or {
            "content": 80,
            "collaborative": 80,
            "geographic": 40,
            "popularity": 40,
        }
        self._latitude = dataset.destinations["latitude"].to_numpy(dtype="float64")
        self._longitude = dataset.destinations["longitude"].to_numpy(dtype="float64")

    @staticmethod
    def _top_indices(scores: np.ndarray, k: int, blocked: np.ndarray) -> np.ndarray:
        """Return the indices of the ``k`` highest scores, skipping ``blocked``."""
        if k <= 0:
            return np.zeros(0, dtype="int64")
        usable = scores.astype("float64").copy()
        usable[blocked] = -np.inf
        finite = np.isfinite(usable)
        if not finite.any():
            return np.zeros(0, dtype="int64")
        k = min(k, int(finite.sum()))
        top = np.argpartition(-usable, k - 1)[:k]
        return top[np.argsort(-usable[top], kind="stable")]

    def generate(self, request: RecommendationRequest) -> CandidateSet:
        """Build the candidate shortlist for one request."""
        n = self.dataset.n_destinations
        blocked = np.zeros(n, dtype=bool)
        blocked_ids = request.visited() | set(request.exclude)
        if blocked_ids:
            indices = self.dataset.indices(sorted(blocked_ids))
            if indices.size:
                blocked[indices] = True

        sources: Dict[int, Set[str]] = {}

        def add(indices: np.ndarray, source: str) -> None:
            for index in indices:
                sources.setdefault(int(index), set()).add(source)

        add(
            self._top_indices(self.content.score(request), self.per_source["content"], blocked),
            "content",
        )
        add(
            self._top_indices(
                self.collaborative.score(request), self.per_source["collaborative"], blocked
            ),
            "collaborative",
        )

        origin = request.current_destination or (request.history[-1] if request.history else None)
        origin_index = self.dataset.index_of.get(origin) if origin else None
        if origin_index is not None:
            distances = haversine_km(
                self._latitude[origin_index],
                self._longitude[origin_index],
                self._latitude,
                self._longitude,
            )
            add(
                self._top_indices(-distances, self.per_source["geographic"], blocked),
                "geographic",
            )

        # Always last, and always present: guarantees the shortlist is usable
        # even for a brand-new user where every other retriever returns nothing.
        add(
            self._top_indices(
                self.popularity.score(request), self.per_source["popularity"], blocked
            ),
            "popularity",
        )

        candidate_indices = np.array(sorted(sources), dtype="int64")
        if candidate_indices.size > self.n_candidates:
            # Trim by how many retrievers agreed, then by popularity, so the
            # cut keeps consensus picks rather than an arbitrary slice.
            popularity_scores = self.popularity.score(request)
            agreement = np.array([len(sources[int(i)]) for i in candidate_indices])
            order = np.lexsort((-popularity_scores[candidate_indices], -agreement))
            candidate_indices = candidate_indices[order][: self.n_candidates]
            candidate_indices.sort()

        return CandidateSet(indices=candidate_indices, sources=sources)

    def candidate_ids(self, request: RecommendationRequest) -> List[str]:
        """Candidate shortlist as destination ids."""
        return self.dataset.ids(self.generate(request).indices)


def candidate_recall(
    generator: CandidateGenerator,
    requests: Dict[str, RecommendationRequest],
    ground_truth: Dict[str, Set[str]],
) -> float:
    """Fraction of held-out destinations that survive candidate generation.

    This is the ceiling on the two-stage system's recall, and is reported
    alongside the model metrics so a ranking result is never mistaken for a
    retrieval failure.
    """
    hits, total = 0, 0
    for user_id, request in requests.items():
        relevant = ground_truth.get(user_id, set())
        if not relevant:
            continue
        candidates = set(generator.candidate_ids(request))
        hits += len(relevant & candidates)
        total += len(relevant)
    return hits / total if total else float("nan")
