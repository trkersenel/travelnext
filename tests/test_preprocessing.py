"""Tests for feature engineering and the train/validation/test split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import TravelDataset, assert_no_leakage, split_interactions
from src.data.regions import continent_of, lookup, region_of
from src.data.synthetic_interactions import GeneratorSettings, generate_interactions
from src.preprocessing.features import (
    MONTHS,
    PROFILE_CATEGORIES,
    add_climate_features,
    add_cost_features,
    build_destination_features,
    percentile_rank,
    profile_matrix,
)
from src.utils.geo import distance_decay, haversine_km


# ------------------------------------------------------------------- regions
def test_region_lookup_known_countries() -> None:
    assert lookup("NL") == ("Europe", "Western Europe")
    assert continent_of("JP") == "Asia"
    assert region_of("BR") == "South America"


def test_region_lookup_is_case_insensitive() -> None:
    assert continent_of("nl") == "Europe"


def test_unknown_country_does_not_raise() -> None:
    assert lookup("ZZ") == ("Unknown", "Unknown")
    assert lookup("") == ("Unknown", "Unknown")


# ---------------------------------------------------------------- geo helpers
def test_haversine_known_distance() -> None:
    # Amsterdam to Rotterdam is roughly 57 km.
    distance = float(haversine_km(52.37, 4.89, 51.92, 4.48))
    assert 50.0 < distance < 65.0


def test_haversine_zero_for_same_point() -> None:
    assert float(haversine_km(10.0, 20.0, 10.0, 20.0)) == pytest.approx(0.0)


def test_distance_decay_is_monotonic() -> None:
    scores = distance_decay(np.array([0.0, 500.0, 5000.0]), 700.0)
    assert scores[0] > scores[1] > scores[2]
    assert scores[0] == pytest.approx(1.0)


# --------------------------------------------------------------- percentiles
def test_percentile_rank_bounds() -> None:
    ranked = percentile_rank(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert ranked.min() > 0.0 and ranked.max() == pytest.approx(1.0)


def test_percentile_rank_handles_all_nan() -> None:
    ranked = percentile_rank(pd.Series([np.nan, np.nan]))
    assert (ranked == 0.0).all()


# ------------------------------------------------------------------ features
def test_engineered_columns_exist(destinations: pd.DataFrame) -> None:
    for category in PROFILE_CATEGORIES:
        assert f"profile_{category}" in destinations.columns
        assert f"score_{category}" in destinations.columns
    for month in MONTHS:
        assert f"season_score_m{month:02d}" in destinations.columns
    assert {"popularity_score", "cost_percentile", "cost_category", "text_blob"} <= set(
        destinations.columns
    )


def test_profile_shares_sum_to_one(destinations: pd.DataFrame) -> None:
    columns = [f"profile_{c}" for c in PROFILE_CATEGORIES]
    totals = destinations[columns].sum(axis=1)
    assert np.allclose(totals, 1.0, atol=1e-6)


def test_scores_are_within_unit_interval(destinations: pd.DataFrame) -> None:
    for category in PROFILE_CATEGORIES:
        values = destinations[f"score_{category}"].dropna()
        assert values.between(0.0, 1.0).all()


def test_profile_matrix_rows_are_unit_norm(destinations: pd.DataFrame) -> None:
    matrix = profile_matrix(destinations)
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0)


def test_missing_osm_data_becomes_nan_not_zero() -> None:
    """A destination with no OSM data must not look like it has nothing."""
    frame = pd.DataFrame(
        {
            "destination_id": ["a", "b"],
            "city": ["A", "B"],
            "country": ["X", "Y"],
            "country_code": ["NL", "DE"],
            "continent": ["Europe", "Europe"],
            "region": ["Western Europe", "Western Europe"],
            "summary": ["a city", "b city"],
            "population": [100000, 200000],
            "pageviews_monthly_mean": [100.0, 200.0],
            "poi_available": [True, False],
            "cost_proxy_value": [30000.0, 40000.0],
            **{f"poi_{c}": [10, 0] for c in PROFILE_CATEGORIES},
            "poi_tourism_infra": [5, 0],
        }
    )
    engineered = build_destination_features(frame)
    assert pd.isna(engineered.loc[1, "profile_museums"])
    assert pd.notna(engineered.loc[0, "profile_museums"])


def test_cost_category_bands(destinations: pd.DataFrame) -> None:
    assert set(destinations["cost_category"]) <= {"budget", "mid-range", "expensive"}


def test_missing_cost_falls_back_to_median_band() -> None:
    frame = pd.DataFrame({"cost_proxy_value": [np.nan, 1000.0, 50000.0]})
    result = add_cost_features(frame)
    assert result.loc[0, "cost_percentile"] == pytest.approx(0.5)
    assert not bool(result.loc[0, "cost_available"])


def test_missing_climate_is_neutral() -> None:
    frame = pd.DataFrame({f"temp_m{m:02d}": [np.nan] for m in MONTHS})
    for month in MONTHS:
        frame[f"precip_m{month:02d}"] = np.nan
    result = add_climate_features(frame)
    assert result.loc[0, "season_score_m07"] == pytest.approx(0.5)
    assert not bool(result.loc[0, "climate_available"])


def test_season_scores_track_temperature(destinations: pd.DataFrame) -> None:
    """A northern-hemisphere city should score better in July than January."""
    row = destinations[destinations["city"] == "Amsterdam"].iloc[0]
    assert row["season_score_m07"] > row["season_score_m01"]


def test_southern_hemisphere_seasons_are_inverted(destinations: pd.DataFrame) -> None:
    row = destinations[destinations["city"] == "Sydney"].iloc[0]
    assert row["season_score_m01"] > row["season_score_m07"]


# --------------------------------------------------------------------- split
def test_split_sizes_and_order(dataset: TravelDataset) -> None:
    split = split_interactions(
        dataset.interactions, val_holdout=1, test_holdout=1, min_train_trips=2
    )
    assert len(split.train) > 0 and len(split.validation) > 0 and len(split.test) > 0
    total = len(split.train) + len(split.validation) + len(split.test)
    assert total == len(dataset.interactions)


def test_test_trips_are_the_most_recent(dataset: TravelDataset) -> None:
    split = split_interactions(
        dataset.interactions, val_holdout=1, test_holdout=1, min_train_trips=2
    )
    for user_id, group in split.test.groupby("user_id"):
        user_trips = dataset.interactions[dataset.interactions["user_id"] == user_id]
        assert group["trip_index"].max() == user_trips["trip_index"].max()


def test_train_trips_precede_validation_and_test(dataset: TravelDataset) -> None:
    split = split_interactions(
        dataset.interactions, val_holdout=1, test_holdout=1, min_train_trips=2
    )
    for user_id, test_group in split.test.groupby("user_id"):
        train_group = split.train[split.train["user_id"] == user_id]
        if train_group.empty:
            continue
        assert train_group["trip_index"].max() < test_group["trip_index"].min()


def test_no_leakage_between_partitions(dataset: TravelDataset) -> None:
    split = split_interactions(
        dataset.interactions, val_holdout=1, test_holdout=1, min_train_trips=2
    )
    assert_no_leakage(split)


def test_leakage_check_catches_a_duplicated_pair(dataset: TravelDataset) -> None:
    split = split_interactions(
        dataset.interactions, val_holdout=1, test_holdout=1, min_train_trips=2
    )
    # Inject the same (user, destination) into train to prove the guard works.
    poisoned = split.test.iloc[[0]].copy()
    split.train = pd.concat([split.train, poisoned], ignore_index=True)
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(split)


def test_users_with_too_few_trips_are_not_evaluated() -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["short", "short", "long", "long", "long", "long", "long"],
            "destination_id": ["a"] * 2 + ["b"] * 5,
            "trip_index": [0, 1, 0, 1, 2, 3, 4],
        }
    )
    split = split_interactions(interactions, val_holdout=1, test_holdout=1, min_train_trips=2)
    assert "short" not in set(split.test["user_id"])
    assert "long" in set(split.test["user_id"])


def test_history_and_targets_for_stages(dataset: TravelDataset) -> None:
    split = split_interactions(
        dataset.interactions, val_holdout=1, test_holdout=1, min_train_trips=2
    )
    assert len(split.history_for("test")) == len(split.train) + len(split.validation)
    assert split.targets_for("validation") is split.validation
    with pytest.raises(ValueError):
        split.history_for("nonsense")


# ------------------------------------------------------------ synthetic data
def test_generator_is_reproducible(destinations: pd.DataFrame) -> None:
    settings = GeneratorSettings(n_users=40, min_trips=3, max_trips=6, seed=99)
    first, _ = generate_interactions(destinations, settings)
    second, _ = generate_interactions(destinations, settings)
    pd.testing.assert_frame_equal(first, second)


def test_generator_never_repeats_a_destination_for_one_user(
    interactions: pd.DataFrame,
) -> None:
    duplicated = interactions.groupby("user_id")["destination_id"].apply(
        lambda series: series.duplicated().any()
    )
    assert not duplicated.any()


def test_generator_uses_all_three_mechanisms(interactions: pd.DataFrame) -> None:
    assert set(interactions["mechanism"]) == {"preference", "popularity", "geographic"}


def test_generator_respects_trip_bounds(destinations: pd.DataFrame) -> None:
    settings = GeneratorSettings(n_users=50, min_trips=3, max_trips=5, seed=5)
    frame, users = generate_interactions(destinations, settings)
    counts = frame.groupby("user_id").size()
    assert counts.min() >= 1 and counts.max() <= 5


def test_generator_rejects_tiny_catalogs(destinations: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least 20 destinations"):
        generate_interactions(destinations.head(5), GeneratorSettings(n_users=5))


# ------------------------------------------------------------------ dataset
def test_dataset_rejects_duplicate_destination_ids(destinations: pd.DataFrame) -> None:
    duplicated = pd.concat([destinations, destinations.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        TravelDataset(duplicated, pd.DataFrame({"user_id": [], "destination_id": []}))


def test_dataset_rejects_unknown_destination_in_interactions(
    destinations: pd.DataFrame,
) -> None:
    bad = pd.DataFrame({"user_id": ["u1"], "destination_id": ["nowhere-xx"], "trip_index": [0]})
    with pytest.raises(ValueError, match="unknown destinations"):
        TravelDataset(destinations, bad)


def test_index_roundtrip(dataset: TravelDataset) -> None:
    ids = ["amsterdam-nl", "tokyo-jp"]
    assert dataset.ids(dataset.indices(ids)) == ids


def test_indices_skip_unknown_ids(dataset: TravelDataset) -> None:
    assert dataset.indices(["amsterdam-nl", "atlantis-xx"]).size == 1
