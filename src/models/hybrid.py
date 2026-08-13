"""Hybrid recommender combining content, collaborative, popularity and context.

    hybrid = alpha * content + beta * collaborative + gamma * popularity
             + delta * context

Each component is converted to a **percentile rank** before blending. Min-max
was tried first and is wrong here: the raw scales are not merely different
(cosine similarity, shrunk co-visitation counts, a probability), they have
different *shapes*. Content scores are dense and clustered; item-item CF is
sparse with a single sharp peak. Under min-max a nominal 0.4/0.4 split let the
CF peak win outright, which is how "Mobile, Alabama" came top for a traveller
returning from Amsterdam, Berlin and Prague. Under rank normalisation a weight
of 0.4 really is 40% of the decision.

The weights are NOT assumed to be good. ``src/models/tune_hybrid.py`` searches
them against the validation split; the defaults in ``configs/config.yaml`` are
only a starting point.

When a user has no history the collaborative component contributes nothing (it
returns zeros), so the blend degrades gracefully into content plus popularity
rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.dataset import TravelDataset
from src.models.base import BaseRecommender, RecommendationRequest, normalise_scores
from src.models.collaborative import ItemItemCFRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.context import ContextScorer
from src.models.popularity import PopularityRecommender
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class HybridWeights:
    """Blend weights for the hybrid score."""

    content: float = 0.4
    collaborative: float = 0.4
    popularity: float = 0.2
    context: float = 0.0

    def normalised(self) -> "HybridWeights":
        """Return weights rescaled to sum to 1 (all-zero maps to popularity)."""
        total = self.content + self.collaborative + self.popularity + self.context
        if total <= 0:
            return HybridWeights(0.0, 0.0, 1.0, 0.0)
        return HybridWeights(
            content=self.content / total,
            collaborative=self.collaborative / total,
            popularity=self.popularity / total,
            context=self.context / total,
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "content": self.content,
            "collaborative": self.collaborative,
            "popularity": self.popularity,
            "context": self.context,
        }


class HybridRecommender(BaseRecommender):
    """Weighted blend of the component recommenders."""

    name = "hybrid"

    def __init__(
        self,
        content: Optional[ContentBasedRecommender] = None,
        collaborative: Optional[ItemItemCFRecommender] = None,
        popularity: Optional[PopularityRecommender] = None,
        weights: Optional[HybridWeights] = None,
        normalisation: str = "rank",
    ) -> None:
        super().__init__()
        self.normalisation = normalisation
        self.content = content or ContentBasedRecommender()
        self.collaborative = collaborative or ItemItemCFRecommender()
        self.popularity = popularity or PopularityRecommender()
        self.weights = weights or HybridWeights()
        self.context_scorer: Optional[ContextScorer] = None

    def _fit(self, dataset: TravelDataset, train_interactions: pd.DataFrame) -> None:
        # Components may already be fitted (the evaluation runner shares them
        # to avoid refitting the same model four times); fit only what is not.
        for component in (self.content, self.collaborative, self.popularity):
            if not component._fitted:
                component.fit(dataset, train_interactions)
        self.context_scorer = ContextScorer(dataset)

    def component_scores(self, request: RecommendationRequest) -> Dict[str, np.ndarray]:
        """Return each component's normalised score vector for ``request``.

        Exposed publicly because both the explanation layer and the
        learning-to-rank feature builder need exactly these numbers.
        """
        dataset = self._require_fitted()
        # Rank normalisation, not min-max: see normalise_scores() for why
        # min-max lets a sparse, spiky component dominate a blend that is
        # nominally even.
        scores = {
            "content": normalise_scores(self.content.score(request), self.normalisation),
            "collaborative": normalise_scores(
                self.collaborative.score(request), self.normalisation
            ),
            "popularity": normalise_scores(self.popularity.score(request), self.normalisation),
        }
        if self.context_scorer is not None:
            context = self.context_scorer.score(request)
            scores["context"] = normalise_scores(context.combined(), self.normalisation)
        else:
            scores["context"] = np.zeros(dataset.n_destinations)
        return scores

    def score(self, request: RecommendationRequest) -> np.ndarray:
        components = self.component_scores(request)
        weights = self.weights.normalised()
        return (
            weights.content * components["content"]
            + weights.collaborative * components["collaborative"]
            + weights.popularity * components["popularity"]
            + weights.context * components["context"]
        )


def weights_from_config(config, *, serving: bool = False) -> HybridWeights:
    """Build :class:`HybridWeights` from a loaded configuration.

    ``serving=True`` returns the weights used to answer real requests, which
    intentionally differ from the experiment defaults. See the extended comment
    on ``models.hybrid.serving`` in ``configs/config.yaml``: tuning against
    synthetic labels selects a pure-collaborative blend that demonstrably
    recommends the wrong cities, so the served blend keeps real, measured
    content and context signal. Documented product judgement, not a metric
    claim.
    """
    if serving:
        return HybridWeights(
            content=float(config.get("models.hybrid.serving.content", 0.35)),
            collaborative=float(config.get("models.hybrid.serving.collaborative", 0.25)),
            popularity=float(config.get("models.hybrid.serving.popularity", 0.10)),
            context=float(config.get("models.hybrid.serving.context", 0.30)),
        )
    return HybridWeights(
        content=float(config.get("models.hybrid.alpha_content", 0.4)),
        collaborative=float(config.get("models.hybrid.beta_collaborative", 0.4)),
        popularity=float(config.get("models.hybrid.gamma_popularity", 0.2)),
        context=float(config.get("models.hybrid.delta_context", 0.0)),
    )
