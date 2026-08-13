"""Tune the hybrid blend weights on the validation split.

The brief is explicit that the default weights must not be assumed optimal, so
they are searched rather than asserted. The search is a coarse simplex grid
over the four component weights, scored by NDCG@10 on the *validation*
partition only -- the test partition is never touched here, which is what keeps
the final reported numbers honest.

The grid is small on purpose: with four components and a 0.1 step there are a
few hundred normalised combinations, each of which only needs a fast re-scoring
of cached component vectors. Component scores are computed once per user and
reused across every weight combination, which turns an otherwise expensive
search into a few seconds of arithmetic.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.dataset import Split, TravelDataset
from src.evaluation.evaluate import build_requests
from src.evaluation.metrics import ndcg_at_k
from src.models.hybrid import HybridRecommender, HybridWeights
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _weight_grid(step: float = 0.1) -> List[HybridWeights]:
    """All normalised (content, collaborative, popularity, context) weightings."""
    ticks = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    combinations: List[HybridWeights] = []
    for content, collaborative, popularity in product(ticks, repeat=3):
        remainder = 1.0 - (content + collaborative + popularity)
        if remainder < -1e-9 or remainder > 1.0 + 1e-9:
            continue
        combinations.append(
            HybridWeights(
                content=float(content),
                collaborative=float(collaborative),
                popularity=float(popularity),
                context=float(max(remainder, 0.0)),
            )
        )
    return combinations


def tune_weights(
    hybrid: HybridRecommender,
    dataset: TravelDataset,
    split: Split,
    *,
    k: int = 10,
    step: float = 0.1,
    max_users: int = 600,
    seed: int = 42,
) -> Tuple[HybridWeights, pd.DataFrame]:
    """Search blend weights against validation NDCG@K.

    Returns the best weights and the full search table, so the tuning result is
    inspectable rather than a magic constant.
    """
    requests = build_requests(dataset, split, "validation")
    ground_truth = (
        split.validation.groupby("user_id")["destination_id"].apply(set).to_dict()
    )

    user_ids = sorted(requests)
    if len(user_ids) > max_users:
        rng = np.random.default_rng(seed)
        user_ids = sorted(rng.choice(user_ids, size=max_users, replace=False).tolist())
    LOGGER.info("Tuning hybrid weights on %d validation users", len(user_ids))

    # Component scores do not depend on the weights, so compute them once.
    cached: List[Tuple[Dict[str, np.ndarray], np.ndarray, set]] = []
    for user_id in user_ids:
        request = requests[user_id]
        components = hybrid.component_scores(request)
        blocked = dataset.indices(sorted(request.visited() | set(request.exclude)))
        cached.append((components, blocked, ground_truth.get(user_id, set())))

    grid = _weight_grid(step)
    LOGGER.info("Evaluating %d weight combinations", len(grid))

    records: List[Dict[str, float]] = []
    for weights in grid:
        normalised = weights.normalised()
        scores: List[float] = []
        for components, blocked, relevant in cached:
            if not relevant:
                continue
            blended = (
                normalised.content * components["content"]
                + normalised.collaborative * components["collaborative"]
                + normalised.popularity * components["popularity"]
                + normalised.context * components["context"]
            )
            if blocked.size:
                blended = blended.copy()
                blended[blocked] = -np.inf
            top = np.argpartition(-blended, k - 1)[:k]
            top = top[np.argsort(-blended[top], kind="stable")]
            scores.append(ndcg_at_k(dataset.ids(top), relevant, k))
        records.append(
            {
                **weights.as_dict(),
                f"ndcg@{k}": float(np.mean(scores)) if scores else float("nan"),
            }
        )

    table = pd.DataFrame(records).sort_values(f"ndcg@{k}", ascending=False).reset_index(drop=True)
    best_row = table.iloc[0]
    best = HybridWeights(
        content=float(best_row["content"]),
        collaborative=float(best_row["collaborative"]),
        popularity=float(best_row["popularity"]),
        context=float(best_row["context"]),
    )
    LOGGER.info(
        "Best hybrid weights: content=%.2f collaborative=%.2f popularity=%.2f context=%.2f "
        "(validation ndcg@%d=%.4f)",
        best.content,
        best.collaborative,
        best.popularity,
        best.context,
        k,
        float(best_row[f"ndcg@{k}"]),
    )
    return best, table
