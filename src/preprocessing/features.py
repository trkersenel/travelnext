"""Turn raw ingested data into the modelling-ready destination table.

The important design decision here is how raw OpenStreetMap POI counts become
travel-characteristic features, because that is where a naive implementation
would quietly invent data.

Two complementary representations are produced for every category:

``profile_<category>``
    The category's *share* of the city's total tourism-relevant POIs. This
    answers "what kind of place is this?" and is largely immune to OSM mapping
    density, because it is a ratio computed inside a single city. It is what
    the content-based similarity model uses.

``score_<category>``
    The global percentile of ``log1p(count)``. This answers "how much of this
    does the city have in absolute terms?" and is what the UI shows and what
    interest filters match against. It *is* affected by OSM coverage bias, so
    it is reported as a percentile rather than a raw magnitude and the bias is
    documented in the README.

    ``score_beaches`` is the exception: it is derived from mapped beach *area*,
    because a count measures how finely a coastline was split into polygons
    rather than how much beach exists. See ``add_attribute_features``.

Nothing here fabricates a value: every feature is a documented transformation
of a measured quantity, and cities without data keep an explicit missing flag.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from src.data.sources.overpass import CATEGORY_NAMES
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

MONTHS: tuple[int, ...] = tuple(range(1, 13))

# Categories that describe what a destination *is like*. ``tourism_infra`` is
# excluded from the profile because hotel density measures capacity, not
# character, and would dominate the share vector for resort cities.
PROFILE_CATEGORIES: tuple[str, ...] = tuple(c for c in CATEGORY_NAMES if c != "tourism_infra")

# Comfort model constants. Chosen to place the optimum at mild sightseeing
# weather; documented as a heuristic index over measured values, not as data.
_IDEAL_TEMP_C = 21.0
_TEMP_TOLERANCE_C = 8.0
_PRECIP_HALF_POINT_MM = 100.0


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return ``frame[name]`` as a float Series, or all-NaN if it is absent.

    ``frame.get(name)`` returns ``None`` for a missing column, which then
    silently degrades into a scalar under ``pd.to_numeric``. That happens for
    real when an upstream fetch fails wholesale (no climate columns at all), so
    every optional column is read through here.
    """
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce")


def percentile_rank(series: pd.Series) -> pd.Series:
    """Return the 0-1 percentile rank of ``series`` (ties averaged)."""
    valid = series.notna()
    ranked = pd.Series(np.nan, index=series.index, dtype="float64")
    if valid.sum() == 0:
        return ranked.fillna(0.0)
    ranked.loc[valid] = series[valid].rank(pct=True, method="average")
    return ranked.fillna(0.0)


def add_popularity_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the popularity proxy score and percentile.

    ``popularity_score`` is the percentile rank of log monthly pageviews. The
    log compresses the extreme head (Paris receives orders of magnitude more
    views than Ghent) so that popularity does not swamp every other signal.
    """
    frame = frame.copy()
    # Prefer the median monthly views: it is robust to the automated-traffic
    # bursts that inflate the mean for a minority of articles. Falls back to
    # the mean for datasets built before the median was recorded.
    monthly = _numeric_column(frame, "pageviews_median")
    if monthly.isna().all():
        monthly = _numeric_column(frame, "pageviews_monthly_mean")
    monthly = monthly.fillna(0.0)
    frame["pageviews_log"] = np.log1p(monthly)
    frame["popularity_score"] = percentile_rank(frame["pageviews_log"])
    frame["population_log"] = np.log1p(_numeric_column(frame, "population").fillna(0.0))
    return frame


def add_attribute_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive ``profile_*`` shares and ``score_*`` percentiles from POI counts."""
    frame = frame.copy()
    count_columns = [f"poi_{name}" for name in PROFILE_CATEGORIES]
    counts = frame[count_columns].astype("float64")

    # --- profile shares -------------------------------------------------
    # log1p before taking shares stops a single huge category (restaurants)
    # from crushing every other dimension to near zero.
    logged = np.log1p(counts)
    totals = logged.sum(axis=1)
    for name in PROFILE_CATEGORIES:
        column = logged[f"poi_{name}"]
        frame[f"profile_{name}"] = np.where(totals > 0, column / totals.replace(0, np.nan), 0.0)
    frame[[f"profile_{n}" for n in PROFILE_CATEGORIES]] = frame[
        [f"profile_{n}" for n in PROFILE_CATEGORIES]
    ].fillna(0.0)

    # --- absolute scores ------------------------------------------------
    for name in CATEGORY_NAMES:
        column = np.log1p(_numeric_column(frame, f"poi_{name}").fillna(0.0))
        frame[f"score_{name}"] = percentile_rank(column)

    # Beaches are scored by mapped AREA rather than feature count. A count
    # measures how finely a coastline happens to be split into polygons, not
    # how much beach there is: Oslo maps 56 small urban and fjord beaches and
    # Barcelona 8 large ones, so counting ranked Oslo higher while Barcelona
    # actually has ~14x the beach area. This is the one category where the
    # obvious measurement is the wrong one.
    beach_area = _numeric_column(frame, "beach_area_m2")
    if beach_area.notna().any() and float(beach_area.fillna(0.0).max()) > 0:
        frame["score_beaches"] = percentile_rank(np.log1p(beach_area.fillna(0.0)))

    # Cities with no OSM data must not look like "a city with zero museums".
    missing = ~frame["poi_available"].astype(bool)
    if missing.any():
        LOGGER.warning("%d destinations lack OSM attribute data", int(missing.sum()))
        for name in PROFILE_CATEGORIES:
            frame.loc[missing, f"profile_{name}"] = np.nan
        for name in CATEGORY_NAMES:
            frame.loc[missing, f"score_{name}"] = np.nan
    return frame


def add_cost_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the cost percentile and a three-way budget band.

    Country-level GNI per capita (PPP) is a documented proxy, not a measured
    travel cost. Destinations in countries the World Bank does not cover fall
    back to the median band with an explicit flag.
    """
    frame = frame.copy()
    value = _numeric_column(frame, "cost_proxy_value")
    frame["cost_available"] = value.notna()
    frame["cost_percentile"] = percentile_rank(np.log1p(value))
    frame.loc[~frame["cost_available"], "cost_percentile"] = 0.5

    frame["cost_category"] = pd.cut(
        frame["cost_percentile"],
        bins=[-0.001, 1 / 3, 2 / 3, 1.001],
        labels=["budget", "mid-range", "expensive"],
    ).astype(str)
    return frame


def add_climate_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert monthly normals into a 0-1 comfort index per month.

    ``season_score_mMM`` combines a Gaussian preference around mild temperature
    with a penalty for heavy rainfall. It is a transparent heuristic over real
    measurements; the raw ``temp_mMM`` / ``precip_mMM`` columns are retained so
    any consumer can apply a different definition of "good weather".
    """
    frame = frame.copy()
    has_climate = pd.Series(False, index=frame.index)

    for month in MONTHS:
        temp_col, precip_col = f"temp_m{month:02d}", f"precip_m{month:02d}"
        temp = _numeric_column(frame, temp_col)
        precip = _numeric_column(frame, precip_col)

        comfort = np.exp(-((temp - _IDEAL_TEMP_C) ** 2) / (2 * _TEMP_TOLERANCE_C**2))
        rain_penalty = 1.0 / (1.0 + precip.fillna(0.0) / _PRECIP_HALF_POINT_MM)
        score = comfort * rain_penalty

        # No climate data -> neutral 0.5, so month choice neither helps nor
        # hurts a destination we know nothing about.
        frame[f"season_score_m{month:02d}"] = score.fillna(0.5).clip(0.0, 1.0)
        has_climate |= temp.notna()

    frame["climate_available"] = has_climate
    season_columns = [f"season_score_m{m:02d}" for m in MONTHS]
    frame["season_score_mean"] = frame[season_columns].mean(axis=1)
    # How strongly seasonal a destination is: useful for explanations.
    frame["seasonality"] = frame[season_columns].std(axis=1)
    return frame


def build_text_blob(frame: pd.DataFrame) -> pd.Series:
    """Compose the text document used by the TF-IDF content model.

    Combines the Wikipedia lead extract with structured metadata and the
    destination's strongest measured characteristics, so that text similarity
    reinforces rather than contradicts the numeric profile.
    """
    profile_columns = [f"profile_{name}" for name in PROFILE_CATEGORIES]
    profiles = frame[profile_columns].fillna(0.0)

    def top_traits(row: pd.Series) -> str:
        if row.sum() <= 0:
            return ""
        ordered = row.sort_values(ascending=False)
        # Repeat the leading trait so TF-IDF weights it more heavily.
        traits = [name.replace("profile_", "") for name in ordered.index[:4]]
        return " ".join([traits[0]] * 2 + traits[1:])

    trait_text = profiles.apply(top_traits, axis=1)
    return (
        frame["summary"].fillna("").astype(str)
        + " "
        + frame["city"].astype(str)
        + " "
        + frame["country"].astype(str)
        + " "
        + frame["region"].astype(str)
        + " "
        + frame["continent"].astype(str)
        + " "
        + trait_text
    ).str.strip()


def build_destination_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature-engineering pass over the raw destination table."""
    LOGGER.info("Engineering features for %d destinations", len(raw))
    frame = add_popularity_features(raw)
    frame = add_attribute_features(frame)
    frame = add_cost_features(frame)
    frame = add_climate_features(frame)
    frame["text_blob"] = build_text_blob(frame)
    frame = frame.reset_index(drop=True)

    missing_text = int((frame["text_blob"].str.len() < 20).sum())
    if missing_text:
        LOGGER.warning("%d destinations have very short text descriptions", missing_text)
    return frame


def profile_matrix(
    frame: pd.DataFrame,
    categories: Sequence[str] = PROFILE_CATEGORIES,
    *,
    standardise: bool = True,
) -> np.ndarray:
    """Return the attribute-profile matrix used for similarity work.

    ``standardise`` (default) z-scores each category across destinations before
    the rows are L2-normalised. This is not a cosmetic step -- without it the
    model barely discriminates at all.

    Every city devotes a similar *share* of its POIs to each category: the
    column means run 0.08-0.13 while the standard deviations are around 0.02.
    Cosine similarity over the raw shares is therefore dominated by that shared
    component, and measured on this catalog it gave a mean pairwise similarity
    of 0.953 (minimum 0.628) -- i.e. all 400 destinations looked ~95% alike, so
    "similar to Amsterdam" carried almost no information.

    Z-scoring first makes the comparison about how a city *deviates* from the
    typical profile, which is what "this is a museum city" actually means.

    Destinations without OSM data receive the mean profile, which maps to the
    origin after standardisation: they stay rankable but assert nothing, which
    is the honest representation of "we have no attribute data for this place".
    """
    columns: List[str] = [f"profile_{name}" for name in categories]
    matrix = frame[columns].astype("float64")
    matrix = matrix.fillna(matrix.mean(axis=0))
    values = matrix.to_numpy()

    if standardise:
        spread = values.std(axis=0)
        spread[spread == 0] = 1.0
        values = (values - values.mean(axis=0)) / spread

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms
