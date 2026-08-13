"""Popularity bias, catalog coverage and diversity analysis.

Accuracy metrics alone hide the failure mode this project cares most about: a
model that scores well by recommending Paris, London and Rome to everyone. The
measures here answer three separate questions.

**Is it just recommending famous places?**
    ``mean_popularity_percentile`` and ``share_from_top_decile`` compare the
    popularity of what a model recommends against the catalog as a whole.

**How much of the catalog does it ever use?**
    ``catalog_coverage`` is the fraction of destinations that appear in at
    least one user's list; ``gini`` measures how unevenly exposure is spread
    across the destinations that do appear.

**Is any single list varied?**
    ``intra_list_diversity`` (attribute dissimilarity), ``geographic_spread``
    (mean pairwise distance) and ``country_entropy`` describe one user's top-K,
    averaged over users.

High diversity is not automatically good. A model can maximise it by
recommending unrelated places nobody wants, so these numbers are always read
next to the accuracy table rather than instead of it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.data.dataset import TravelDataset
from src.preprocessing.features import PROFILE_CATEGORIES, profile_matrix
from src.utils.geo import haversine_km
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class BeyondAccuracyReport:
    """Beyond-accuracy measurements for one model's recommendations."""

    model: str
    k: int
    catalog_coverage: float
    n_unique_recommended: int
    gini: float
    mean_popularity_percentile: float
    share_from_top_decile: float
    novelty: float
    intra_list_diversity: float
    geographic_spread_km: float
    country_entropy: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def gini_coefficient(counts: np.ndarray) -> float:
    """Gini coefficient of an exposure distribution (0 = even, 1 = one item).

    Computed over the whole catalog including never-recommended destinations,
    because "this model never shows 90% of the catalog" is exactly the
    concentration we want the number to capture.
    """
    values = np.sort(np.asarray(counts, dtype="float64"))
    total = values.sum()
    if total <= 0:
        return 0.0
    n = values.size
    index = np.arange(1, n + 1)
    return float((2.0 * (index * values).sum()) / (n * total) - (n + 1.0) / n)


def _entropy(labels: Sequence[str]) -> float:
    """Shannon entropy (bits) of a categorical sequence."""
    if not len(labels):
        return 0.0
    _, counts = np.unique(np.asarray(labels), return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def analyse_recommendations(
    dataset: TravelDataset,
    recommendations: Dict[str, List[str]],
    *,
    model_name: str,
    k: int = 10,
) -> BeyondAccuracyReport:
    """Compute every beyond-accuracy measure for one model's output."""
    frame = dataset.destinations
    popularity = frame["popularity_score"].to_numpy(dtype="float64")
    latitude = frame["latitude"].to_numpy(dtype="float64")
    longitude = frame["longitude"].to_numpy(dtype="float64")
    countries = frame["country_code"].to_numpy()
    profiles = profile_matrix(frame, PROFILE_CATEGORIES)

    exposure = np.zeros(dataset.n_destinations, dtype="float64")
    popularity_values: List[float] = []
    top_decile_threshold = float(np.quantile(popularity, 0.9))
    top_decile_hits = 0
    total_recommended = 0

    diversity_values: List[float] = []
    spread_values: List[float] = []
    entropy_values: List[float] = []

    for ranked in recommendations.values():
        top_k = ranked[:k]
        if not top_k:
            continue
        indices = dataset.indices(top_k)
        if indices.size == 0:
            continue

        exposure[indices] += 1.0
        popularity_values.extend(popularity[indices].tolist())
        top_decile_hits += int((popularity[indices] >= top_decile_threshold).sum())
        total_recommended += indices.size

        # --- intra-list attribute diversity ---------------------------------
        if indices.size > 1:
            block = profiles[indices]
            similarity = block @ block.T
            upper = similarity[np.triu_indices(indices.size, k=1)]
            diversity_values.append(float(1.0 - upper.mean()))

            distances = haversine_km(
                latitude[indices][:, None],
                longitude[indices][:, None],
                latitude[indices][None, :],
                longitude[indices][None, :],
            )
            spread_values.append(float(distances[np.triu_indices(indices.size, k=1)].mean()))
            entropy_values.append(_entropy(countries[indices]))

    n_unique = int((exposure > 0).sum())
    # Novelty: self-information of the recommended items. Recommending a
    # rarely-viewed destination carries more information than another Paris.
    popularity_array = np.array(popularity_values, dtype="float64")
    novelty = float(np.mean(-np.log2(np.clip(popularity_array, 1e-6, 1.0)))) if popularity_array.size else float("nan")

    report = BeyondAccuracyReport(
        model=model_name,
        k=k,
        catalog_coverage=n_unique / dataset.n_destinations if dataset.n_destinations else 0.0,
        n_unique_recommended=n_unique,
        gini=gini_coefficient(exposure),
        mean_popularity_percentile=float(popularity_array.mean()) if popularity_array.size else float("nan"),
        share_from_top_decile=top_decile_hits / total_recommended if total_recommended else float("nan"),
        novelty=novelty,
        intra_list_diversity=float(np.mean(diversity_values)) if diversity_values else float("nan"),
        geographic_spread_km=float(np.mean(spread_values)) if spread_values else float("nan"),
        country_entropy=float(np.mean(entropy_values)) if entropy_values else float("nan"),
    )
    LOGGER.info(
        "%-22s coverage=%.3f gini=%.3f pop_pct=%.3f top10%%=%.3f diversity=%.3f",
        model_name,
        report.catalog_coverage,
        report.gini,
        report.mean_popularity_percentile,
        report.share_from_top_decile,
        report.intra_list_diversity,
    )
    return report


def most_recommended(
    dataset: TravelDataset, recommendations: Dict[str, List[str]], *, k: int = 10, top: int = 15
) -> pd.DataFrame:
    """The destinations a model shows most often, for eyeballing bias."""
    counter: Dict[str, int] = {}
    for ranked in recommendations.values():
        for destination_id in ranked[:k]:
            counter[destination_id] = counter.get(destination_id, 0) + 1
    if not counter:
        return pd.DataFrame(columns=["destination_id", "city", "country", "times_recommended"])

    frame = pd.DataFrame(
        sorted(counter.items(), key=lambda item: -item[1])[:top],
        columns=["destination_id", "times_recommended"],
    )
    lookup = dataset.destinations.set_index("destination_id")[["city", "country"]]
    frame = frame.join(lookup, on="destination_id")
    frame["share_of_users"] = frame["times_recommended"] / max(len(recommendations), 1)
    return frame


def compare_reports(reports: Sequence[BeyondAccuracyReport]) -> pd.DataFrame:
    """Assemble beyond-accuracy reports into a comparison table."""
    if not reports:
        return pd.DataFrame()
    return pd.DataFrame([report.as_dict() for report in reports])
