"""Stage 2 of the two-stage architecture: LightGBM learning-to-rank.

Uses LambdaRank (``objective="lambdarank"``) optimising NDCG, which is the
standard choice for top-K ranking with binary relevance and trains in seconds
on a laptop at this data size.

The ranker never scores the whole catalog. It re-ranks the shortlist produced
by candidate generation, which is what makes the architecture cheap enough to
serve from a small FastAPI process.

Feature importances are retained after training and surfaced in the
explanation layer, so the reasons shown to a user are grounded in what the
model actually used rather than in a plausible-sounding story.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.candidate_generation.generator import CandidateGenerator
from src.data.dataset import TravelDataset
from src.models.base import BaseRecommender, RecommendationRequest, ScoredDestination
from src.ranking.features import RankingFeatureBuilder, build_training_data
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


class LearningToRankRecommender(BaseRecommender):
    """Candidate generation followed by a LightGBM LambdaRank re-ranker."""

    name = "learning_to_rank"

    def __init__(
        self,
        generator: CandidateGenerator,
        builder: RankingFeatureBuilder,
        *,
        params: Optional[Dict[str, object]] = None,
        early_stopping_rounds: int = 50,
        negatives_per_positive: int = 30,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.builder = builder
        self.params = dict(params or {})
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.negatives_per_positive = int(negatives_per_positive)
        self.seed = int(seed)
        self.model: Optional[lgb.LGBMRanker] = None

    # ------------------------------------------------------------- training
    def fit_ranker(
        self,
        dataset: TravelDataset,
        training_requests: Sequence[Tuple[RecommendationRequest, str]],
        validation_requests: Optional[Sequence[Tuple[RecommendationRequest, str]]] = None,
    ) -> "LearningToRankRecommender":
        """Train the ranker on ``(request, chosen destination)`` pairs."""
        self.dataset = dataset
        features, labels, groups = build_training_data(
            dataset,
            self.builder,
            self.generator,
            training_requests,
            negatives_per_positive=self.negatives_per_positive,
            seed=self.seed,
        )

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "n_jobs": 4,
            "verbose": -1,
            "random_state": self.seed,
        }
        params.update(self.params)
        # LightGBM's sklearn API spells L2 regularisation "reg_lambda"; accept
        # the native "lambda_l2" spelling from the config without failing.
        if "lambda_l2" in params:
            params["reg_lambda"] = params.pop("lambda_l2")

        self.model = lgb.LGBMRanker(**params)
        fit_kwargs: Dict[str, object] = {}
        if validation_requests:
            validation_features, validation_labels, validation_groups = build_training_data(
                dataset,
                self.builder,
                self.generator,
                validation_requests,
                negatives_per_positive=self.negatives_per_positive,
                seed=self.seed + 1,
            )
            fit_kwargs["eval_set"] = [(validation_features, validation_labels)]
            fit_kwargs["eval_group"] = [validation_groups]
            fit_kwargs["eval_at"] = [5, 10, 20]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        self.model.fit(features, labels, group=groups, **fit_kwargs)
        self._fitted = True
        LOGGER.info(
            "Trained LightGBM ranker: %d trees, %d features",
            self.model.n_estimators_ if hasattr(self.model, "n_estimators_") else params["n_estimators"],
            features.shape[1],
        )
        return self

    def _fit(self, dataset: TravelDataset, train_interactions: pd.DataFrame) -> None:
        """The ranker needs (request, target) pairs, so use :meth:`fit_ranker`."""
        raise NotImplementedError(
            "LearningToRankRecommender is trained with fit_ranker(), which needs "
            "request/target pairs rather than a raw interaction frame."
        )

    # -------------------------------------------------------------- serving
    def score(self, request: RecommendationRequest) -> np.ndarray:
        """Score the full catalog, with non-candidates pushed to the bottom.

        Keeping the ``BaseRecommender`` contract means the ranker plugs into
        the same evaluation and explanation code as every other model.
        """
        dataset = self._require_fitted()
        scores = np.full(dataset.n_destinations, -np.inf)
        candidates = self.generator.generate(request)
        if len(candidates) == 0 or self.model is None:
            return np.zeros(dataset.n_destinations)
        features = self.builder.build(request, candidates)
        scores[candidates.indices] = self.model.predict(features)
        return scores

    def rank_candidates(
        self, request: RecommendationRequest, k: int = 10
    ) -> List[ScoredDestination]:
        """Re-rank the shortlist and return the top ``k`` with their features."""
        dataset = self._require_fitted()
        candidates = self.generator.generate(request)
        if len(candidates) == 0 or self.model is None:
            return []
        features = self.builder.build(request, candidates)
        predictions = self.model.predict(features)

        order = np.argsort(-predictions)[: max(0, k)]
        results: List[ScoredDestination] = []
        for position, row in enumerate(order):
            index = int(candidates.indices[int(row)])
            results.append(
                ScoredDestination(
                    destination_id=dataset.destination_ids[index],
                    score=float(predictions[int(row)]),
                    rank=position + 1,
                    components=dict(
                        zip(self.builder.feature_names, features[int(row)].astype(float))
                    ),
                )
            )
        return results

    # ------------------------------------------------------- introspection
    def feature_importance(self) -> pd.DataFrame:
        """Gain-based feature importance, most important first."""
        if self.model is None:
            return pd.DataFrame(columns=["feature", "gain"])
        gains = self.model.booster_.feature_importance(importance_type="gain")
        frame = pd.DataFrame({"feature": self.builder.feature_names, "gain": gains})
        frame["gain_share"] = frame["gain"] / max(frame["gain"].sum(), 1e-9)
        return frame.sort_values("gain", ascending=False).reset_index(drop=True)

    def save(self, path: Path) -> None:
        """Persist the trained booster (not the data or the retrievers)."""
        if self.model is None:
            raise RuntimeError("Cannot save an untrained ranker")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_names": self.builder.feature_names}, path
        )
        LOGGER.info("Saved ranker to %s", path)

    def load(self, path: Path) -> "LearningToRankRecommender":
        """Load a previously trained booster."""
        payload = joblib.load(path)
        self.model = payload["model"]
        self._fitted = True
        return self


def requests_from_interactions(
    dataset: TravelDataset,
    interactions: pd.DataFrame,
    *,
    min_history: int = 1,
    all_prefixes: bool = True,
) -> List[Tuple[RecommendationRequest, str]]:
    """Turn interaction histories into ``(request, next destination)`` pairs.

    With ``all_prefixes`` (the default) every position in a user's history
    becomes a training example: predict trip *i* from trips ``0..i-1``. A user
    with 8 training trips yields 7 examples instead of 1, which multiplies the
    training set several-fold. This matters -- one example per user gave the
    27-feature ranker only ~4k groups, far too few to beat the collaborative
    score it consumes as a feature.

    This introduces no leakage: every request is built from trips that strictly
    precede its target, and only one partition is ever passed in, so no request
    can see a trip from a later partition.

    Set ``all_prefixes=False`` to keep the older behaviour of using only each
    user's final trip, which mirrors the evaluation protocol exactly.
    """
    ordered = interactions.sort_values(["user_id", "trip_index"])
    pairs: List[Tuple[RecommendationRequest, str]] = []

    for user_id, group in ordered.groupby("user_id", sort=False):
        destinations = group["destination_id"].tolist()
        if len(destinations) < min_history + 1:
            continue

        # Target positions, counted from the end of the visible history.
        positions = range(min_history, len(destinations)) if all_prefixes else [len(destinations) - 1]
        for position in positions:
            target_row = group.iloc[position]
            history = destinations[:position]
            pairs.append(
                (
                    RecommendationRequest(
                        history=history[:-1],
                        current_destination=history[-1],
                        month=int(target_row["month"]),
                        trip_duration_days=int(target_row["trip_duration_days"]),
                        budget=str(target_row["budget"]),
                        user_id=str(user_id),
                    ),
                    str(target_row["destination_id"]),
                )
            )
    return pairs
