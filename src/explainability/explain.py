"""Explanations grounded in the values the models actually computed.

Every reason returned here is produced by reading a number the system already
calculated -- a cosine similarity, a co-visitation score, a climate normal, a
cost percentile, a great-circle distance -- and is emitted only when that
number clears a stated threshold. Nothing is generated from a template because
it "sounds right", and a destination that has no strong signal simply receives
fewer reasons rather than an invented one.

Each :class:`Reason` carries the evidence behind it, so the API can return the
supporting numbers alongside the sentence and a reader can check the claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.dataset import TravelDataset
from src.models.base import RecommendationRequest
from src.models.collaborative import ItemItemCFRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.context import BUDGET_TARGETS, ContextScorer
from src.preprocessing.features import PROFILE_CATEGORIES
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

# Human-readable labels for the measured OSM categories.
CATEGORY_LABELS: Dict[str, str] = {
    "museums": "museums and galleries",
    "culture": "theatres and cultural venues",
    "heritage": "historic monuments",
    "architecture": "notable architecture",
    "nightlife": "bars and nightlife",
    "food": "restaurants and cafés",
    "nature": "parks and green space",
    "beaches": "beaches",
    "outdoor": "outdoor and viewpoints",
    "family": "family attractions",
    "shopping": "markets and shopping",
    "walkability": "pedestrian streets",
}

# Thresholds a signal must clear before it is worth stating out loud. These are
# deliberately conservative: a weak signal produces no sentence at all.
_MIN_CONTENT_SIMILARITY = 0.12
_MIN_CF_SIMILARITY = 0.02
_MIN_ATTRIBUTE_SCORE = 0.65
_MIN_SEASON_SCORE = 0.55
_MIN_BUDGET_FIT = 0.80
_NEAR_DISTANCE_KM = 800.0
_POPULAR_PERCENTILE = 0.85
_HIDDEN_GEM_PERCENTILE = 0.45


@dataclass
class Reason:
    """One explanation with the evidence that justifies it."""

    kind: str
    text: str
    strength: float
    evidence: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "text": self.text,
            "strength": round(float(self.strength), 4),
            "evidence": self.evidence,
        }


class Explainer:
    """Produces evidence-backed reasons for a recommended destination."""

    def __init__(
        self,
        dataset: TravelDataset,
        content: ContentBasedRecommender,
        collaborative: ItemItemCFRecommender,
        context: Optional[ContextScorer] = None,
    ) -> None:
        self.dataset = dataset
        self.content = content
        self.collaborative = collaborative
        self.context = context or ContextScorer(dataset)
        self._frame = dataset.destinations

    def _city_label(self, destination_id: str) -> str:
        index = self.dataset.index_of.get(destination_id)
        if index is None:
            return destination_id
        return str(self._frame.at[index, "city"])

    def _most_similar_visited(
        self, destination_index: int, history: Sequence[str]
    ) -> Tuple[Optional[str], float]:
        """The visited destination most content-similar to the candidate."""
        history_indices = self.dataset.indices(list(history))
        if history_indices.size == 0 or self.content.text_matrix is None:
            return None, 0.0
        text = self.content.text_matrix
        text_similarity = np.asarray(
            (text[destination_index] @ text[history_indices].T).todense()
        ).ravel()
        attribute_similarity = (
            self.content.attribute_matrix[history_indices]
            @ self.content.attribute_matrix[destination_index]
        )
        combined = (
            self.content.text_weight * text_similarity
            + self.content.attribute_weight * attribute_similarity
        )
        best = int(np.argmax(combined))
        return self.dataset.destination_ids[int(history_indices[best])], float(combined[best])

    def _strongest_cf_link(
        self, destination_index: int, history: Sequence[str]
    ) -> Tuple[Optional[str], float]:
        """The visited destination most co-visited with the candidate."""
        similarity = self.collaborative.similarity
        history_indices = self.dataset.indices(list(history))
        if similarity is None or history_indices.size == 0:
            return None, 0.0
        row = np.asarray(similarity[destination_index, history_indices].todense()).ravel()
        if row.size == 0:
            return None, 0.0
        best = int(np.argmax(row))
        return self.dataset.destination_ids[int(history_indices[best])], float(row[best])

    def _shared_strengths(self, destination_index: int, history: Sequence[str]) -> List[str]:
        """Attribute categories that are strong here *and* in the user's past."""
        history_indices = self.dataset.indices(list(history))
        if history_indices.size == 0:
            return []
        matches: List[Tuple[str, float]] = []
        for category in PROFILE_CATEGORIES:
            column = f"score_{category}"
            candidate_score = self._frame.at[destination_index, column]
            if pd.isna(candidate_score) or float(candidate_score) < _MIN_ATTRIBUTE_SCORE:
                continue
            history_scores = self._frame.loc[history_indices, column].astype("float64")
            if history_scores.notna().any() and float(history_scores.mean()) >= _MIN_ATTRIBUTE_SCORE:
                matches.append((category, float(candidate_score)))
        matches.sort(key=lambda item: -item[1])
        return [category for category, _ in matches[:3]]

    def explain(
        self,
        request: RecommendationRequest,
        destination_id: str,
        *,
        max_reasons: int = 6,
    ) -> List[Reason]:
        """Return the ranked reasons supporting one recommendation."""
        index = self.dataset.index_of.get(destination_id)
        if index is None:
            return []

        history = list(request.history)
        if request.current_destination:
            history.append(request.current_destination)
        reasons: List[Reason] = []

        # --- 1. content similarity to a specific past trip -----------------
        similar_id, similarity = self._most_similar_visited(index, history)
        if similar_id and similarity >= _MIN_CONTENT_SIMILARITY:
            reasons.append(
                Reason(
                    kind="content_similarity",
                    text=f"Similar in character to {self._city_label(similar_id)}, which you visited",
                    strength=min(similarity / 0.6, 1.0),
                    evidence={"most_similar_visited": similar_id, "similarity": round(similarity, 4)},
                )
            )

        # --- 2. shared measured strengths ----------------------------------
        shared = self._shared_strengths(index, history)
        if shared:
            labels = [CATEGORY_LABELS.get(category, category) for category in shared]
            joined = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + f" and {labels[-1]}"
            reasons.append(
                Reason(
                    kind="attribute_match",
                    text=f"Strong on {joined}, matching your previous destinations",
                    strength=0.8,
                    evidence={
                        "categories": shared,
                        "scores": {
                            category: round(float(self._frame.at[index, f"score_{category}"]), 3)
                            for category in shared
                        },
                    },
                )
            )

        # --- 3. collaborative signal ---------------------------------------
        cf_id, cf_score = self._strongest_cf_link(index, history)
        if cf_id and cf_score >= _MIN_CF_SIMILARITY:
            reasons.append(
                Reason(
                    kind="collaborative",
                    text=(
                        f"Travellers who went to {self._city_label(cf_id)} often visit here too"
                    ),
                    strength=min(cf_score / 0.2, 1.0),
                    evidence={"co_visited_with": cf_id, "cf_similarity": round(cf_score, 4)},
                )
            )

        # --- 4. season fit --------------------------------------------------
        if request.month:
            season_score = float(self._frame.at[index, f"season_score_m{int(request.month):02d}"])
            temperature = self._frame.at[index, f"temp_m{int(request.month):02d}"]
            if season_score >= _MIN_SEASON_SCORE and pd.notna(temperature):
                month_name = pd.Timestamp(2024, int(request.month), 1).strftime("%B")
                reasons.append(
                    Reason(
                        kind="season",
                        text=(
                            f"Good conditions in {month_name} "
                            f"(average {float(temperature):.0f}°C)"
                        ),
                        strength=season_score,
                        evidence={
                            "month": int(request.month),
                            "season_score": round(season_score, 3),
                            "mean_temp_c": round(float(temperature), 1),
                        },
                    )
                )

        # --- 5. budget fit --------------------------------------------------
        if request.budget:
            budget_fit = float(self.context.budget_scores(request.budget)[index])
            if budget_fit >= _MIN_BUDGET_FIT:
                reasons.append(
                    Reason(
                        kind="budget",
                        text=f"Fits your {request.budget} budget",
                        strength=budget_fit,
                        evidence={
                            "cost_category": str(self._frame.at[index, "cost_category"]),
                            "budget_fit": round(budget_fit, 3),
                            "cost_data_available": bool(self._frame.at[index, "cost_available"]),
                        },
                    )
                )

        # --- 6. geographic convenience --------------------------------------
        origin = request.current_destination or (history[-1] if history else None)
        if origin:
            distances = self.context.distance_from(origin)
            distance = float(distances[index]) if np.isfinite(distances[index]) else None
            if distance is not None and distance <= _NEAR_DISTANCE_KM:
                duration_note = ""
                if request.trip_duration_days and request.trip_duration_days <= 5:
                    duration_note = f", which suits a {request.trip_duration_days}-day trip"
                reasons.append(
                    Reason(
                        kind="geography",
                        text=(
                            f"Only {distance:,.0f} km from {self._city_label(origin)}"
                            f"{duration_note}"
                        ),
                        strength=float(np.exp(-distance / _NEAR_DISTANCE_KM)),
                        evidence={"origin": origin, "distance_km": round(distance, 1)},
                    )
                )

        # --- 7. popularity framing -------------------------------------------
        popularity = float(self._frame.at[index, "popularity_score"])
        if popularity >= _POPULAR_PERCENTILE:
            reasons.append(
                Reason(
                    kind="popularity",
                    text="A widely searched destination",
                    strength=popularity,
                    evidence={"popularity_percentile": round(popularity, 3)},
                )
            )
        elif popularity <= _HIDDEN_GEM_PERCENTILE and len(reasons) >= 2:
            # Only frame something as a lesser-known find when other, stronger
            # reasons already justify recommending it.
            reasons.append(
                Reason(
                    kind="novelty",
                    text="A lesser-known option compared with the usual choices",
                    strength=1.0 - popularity,
                    evidence={"popularity_percentile": round(popularity, 3)},
                )
            )

        # --- 8. always true, and worth saying --------------------------------
        if history:
            reasons.append(
                Reason(
                    kind="unvisited",
                    text="You have not been here yet",
                    strength=0.3,
                    evidence={"history_length": len(history)},
                )
            )

        reasons.sort(key=lambda reason: -reason.strength)
        return reasons[:max_reasons]

    def explain_batch(
        self,
        request: RecommendationRequest,
        destination_ids: Sequence[str],
        *,
        max_reasons: int = 6,
    ) -> Dict[str, List[Reason]]:
        """Explain several recommendations for the same request."""
        return {
            destination_id: self.explain(request, destination_id, max_reasons=max_reasons)
            for destination_id in destination_ids
        }
