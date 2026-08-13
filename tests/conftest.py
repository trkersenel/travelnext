"""Shared test fixtures.

The fixtures build a small dataset **in memory** by running the real feature
pipeline over a hand-made raw frame. Nothing here touches the network or the
processed parquet files, so the suite runs on a clean checkout in seconds and
its results do not depend on when the data was last ingested.

Building the fixture through ``build_destination_features`` rather than
hard-coding feature columns means the tests break if the feature contract
changes, which is exactly what they should do.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import TravelDataset, split_interactions
from src.data.synthetic_interactions import GeneratorSettings, generate_interactions
from src.data.sources.overpass import CATEGORY_NAMES
from src.preprocessing.features import build_destination_features

# A miniature catalog spread over three continents, with deliberately different
# attribute profiles so similarity and diversity assertions are meaningful.
_CITIES = [
    # (city, country, iso, lat, lon, population, profile emphasis)
    ("Amsterdam", "Netherlands", "NL", 52.37, 4.89, 900_000, "culture"),
    ("Rotterdam", "Netherlands", "NL", 51.92, 4.48, 650_000, "culture"),
    ("Utrecht", "Netherlands", "NL", 52.09, 5.12, 360_000, "culture"),
    ("Antwerp", "Belgium", "BE", 51.22, 4.40, 530_000, "culture"),
    ("Brussels", "Belgium", "BE", 50.85, 4.35, 1_200_000, "culture"),
    ("Berlin", "Germany", "DE", 52.52, 13.40, 3_600_000, "nightlife"),
    ("Hamburg", "Germany", "DE", 53.55, 9.99, 1_800_000, "nightlife"),
    ("Prague", "Czechia", "CZ", 50.08, 14.44, 1_300_000, "heritage"),
    ("Vienna", "Austria", "AT", 48.21, 16.37, 1_900_000, "heritage"),
    ("Budapest", "Hungary", "HU", 47.50, 19.04, 1_750_000, "heritage"),
    ("Copenhagen", "Denmark", "DK", 55.68, 12.57, 640_000, "culture"),
    ("Barcelona", "Spain", "ES", 41.39, 2.17, 1_600_000, "beaches"),
    ("Valencia", "Spain", "ES", 39.47, -0.38, 800_000, "beaches"),
    ("Lisbon", "Portugal", "PT", 38.72, -9.14, 550_000, "beaches"),
    ("Rome", "Italy", "IT", 41.90, 12.50, 2_800_000, "heritage"),
    ("Naples", "Italy", "IT", 40.85, 14.27, 950_000, "food"),
    ("Bangkok", "Thailand", "TH", 13.75, 100.50, 8_300_000, "food"),
    ("Tokyo", "Japan", "JP", 35.69, 139.69, 9_700_000, "food"),
    ("Kyoto", "Japan", "JP", 35.01, 135.77, 1_460_000, "heritage"),
    ("Sydney", "Australia", "AU", -33.87, 151.21, 5_300_000, "beaches"),
    ("Cape Town", "South Africa", "ZA", -33.92, 18.42, 4_600_000, "nature"),
    ("Nairobi", "Kenya", "KE", -1.29, 36.82, 4_400_000, "nature"),
    ("Denver", "United States", "US", 39.74, -104.98, 715_000, "outdoor"),
    ("Boston", "United States", "US", 42.36, -71.06, 675_000, "museums"),
    ("Montreal", "Canada", "CA", 45.50, -73.57, 1_760_000, "museums"),
]

# Base POI counts per emphasis, so a "culture" city really does have more
# museums than a "beaches" city. Values are plausible magnitudes, and the test
# suite only relies on their relative ordering.
_EMPHASIS_BOOST = {
    "culture": {"museums": 120, "culture": 400, "architecture": 180},
    "heritage": {"heritage": 300, "architecture": 220, "museums": 90},
    "nightlife": {"nightlife": 500, "food": 900},
    "food": {"food": 1500, "shopping": 90},
    "beaches": {"beaches": 40, "outdoor": 200, "food": 700},
    "nature": {"nature": 600, "outdoor": 350},
    "outdoor": {"outdoor": 400, "nature": 380},
    "museums": {"museums": 200, "culture": 250},
}

# Plausible baseline count ranges per category for a city with no particular
# emphasis. Beaches are near zero for inland cities, which matters: a uniform
# baseline would make landlocked cities outrank coastal ones on beaches and
# would mask a real bug in interest-based scoring.
_BASELINE_RANGES = {
    "beaches": (0, 2),
    "museums": (5, 30),
    "culture": (10, 60),
    "heritage": (5, 40),
    "architecture": (10, 50),
    "nightlife": (20, 90),
    "food": (80, 300),
    "nature": (20, 90),
    "outdoor": (10, 50),
    "family": (10, 60),
    "shopping": (5, 25),
    "walkability": (20, 120),
    "tourism_infra": (15, 80),
}


def _raw_destinations() -> pd.DataFrame:
    """Build a raw destination frame in the shape the real ingestion produces."""
    rng = np.random.default_rng(7)
    records: List[dict] = []

    for position, (city, country, iso, lat, lon, population, emphasis) in enumerate(_CITIES):
        record = {
            "destination_id": f"{city.lower().replace(' ', '-')}-{iso.lower()}",
            "city": city,
            "country": country,
            "country_code": iso,
            "latitude": lat,
            "longitude": lon,
            "population": population,
            "wiki_title": city,
            "summary": f"{city} is a city in {country} known for its {emphasis} and its character.",
            "pageviews_total": float(200_000 * (len(_CITIES) - position) + 5_000),
            "pageviews_monthly_mean": float(8_000 * (len(_CITIES) - position) + 200),
            "poi_available": True,
            "cost_proxy_value": float(15_000 + 2_000 * (position % 12)),
            "cost_proxy_year": 2023,
            "continent": "",
            "region": "",
        }
        # Baseline counts plus the emphasis boost.
        for category in CATEGORY_NAMES:
            low, high = _BASELINE_RANGES.get(category, (20, 80))
            record[f"poi_{category}"] = int(rng.integers(low, high + 1))
        for category, value in _EMPHASIS_BOOST[emphasis].items():
            record[f"poi_{category}"] = value

        # Northern-hemisphere seasonality, southern cities inverted.
        southern = lat < 0
        for month in range(1, 13):
            phase = (month - 7) if not southern else (month - 1)
            record[f"temp_m{month:02d}"] = float(14.0 + 9.0 * np.cos(np.pi * phase / 6.0))
            record[f"precip_m{month:02d}"] = float(40.0 + 20.0 * np.sin(np.pi * month / 6.0))
        records.append(record)

    frame = pd.DataFrame(records)
    # ``build_destination_features`` expects continent/region to be filled by
    # the ingestion step; mirror that here.
    from src.data import regions

    frame["continent"] = frame["country_code"].map(regions.continent_of)
    frame["region"] = frame["country_code"].map(regions.region_of)
    return frame


@pytest.fixture(scope="session")
def destinations() -> pd.DataFrame:
    """A small destination catalog with the full engineered feature set."""
    return build_destination_features(_raw_destinations())


@pytest.fixture(scope="session")
def interactions(destinations: pd.DataFrame) -> pd.DataFrame:
    """Synthetic interactions over the fixture catalog."""
    frame, _ = generate_interactions(
        destinations,
        GeneratorSettings(n_users=180, min_trips=4, max_trips=9, seed=13),
    )
    return frame


@pytest.fixture(scope="session")
def dataset(destinations: pd.DataFrame, interactions: pd.DataFrame) -> TravelDataset:
    """A ready-to-use :class:`TravelDataset` for model tests."""
    return TravelDataset(destinations, interactions)


@pytest.fixture(scope="session")
def split(dataset: TravelDataset):
    """The standard leave-last-n-out split over the fixture interactions."""
    return split_interactions(
        dataset.interactions, val_holdout=1, test_holdout=1, min_train_trips=2
    )
