"""Turn the raw ingested tables into the modelling dataset.

Run with::

    python -m src.data.build_dataset

Reads ``data/processed/destinations_raw.parquet`` (produced by
``src.data.build_destinations``), engineers features, generates the synthetic
interaction data and writes:

* ``data/processed/destinations.parquet``    -- real data only
* ``data/processed/interactions.parquet``    -- SYNTHETIC
* ``data/processed/synthetic_users.parquet`` -- SYNTHETIC
* ``data/processed/DATA_PROVENANCE.md``      -- which file came from where

Real and synthetic data are written to separate files, and the provenance note
is regenerated on every run, so the boundary cannot drift out of date.
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from src.config import Config, load_config
from src.data.dataset import DESTINATIONS_FILE, INTERACTIONS_FILE, USERS_FILE
from src.data.synthetic_interactions import generate_interactions, settings_from_config
from src.preprocessing.features import build_destination_features
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

RAW_DESTINATIONS_FILE = "destinations_raw.parquet"
PROVENANCE_FILE = "DATA_PROVENANCE.md"

_PROVENANCE_TEMPLATE = """# Data provenance

Generated automatically by `src/data/build_dataset.py` on {today}.

## Real data (from free, openly licensed public sources)

`{destinations_file}` -- {n_destinations} destinations in {n_countries} countries.

| Column group | Source | Licence |
|---|---|---|
| `city`, `country`, `latitude`, `longitude`, `population` | GeoNames `cities15000` | CC BY 4.0 |
| `wiki_title`, `summary` | English Wikipedia REST API | CC BY-SA 4.0 |
| `pageviews_*`, `popularity_score` | Wikimedia Pageviews API | CC0 |
| `poi_*`, `profile_*`, `score_*` | OpenStreetMap via Overpass API | ODbL 1.0 |
| `temp_m*`, `precip_m*`, `season_score_m*` | Open-Meteo historical archive | CC BY 4.0 |
| `cost_proxy_*`, `cost_percentile`, `cost_category` | World Bank Indicators API | CC BY 4.0 |
| `continent`, `region` | UN M49 classification (bundled) | Public domain |

## Synthetic data (generated, NOT real travellers)

`{interactions_file}` -- {n_interactions} interactions from {n_users} generated users.
`{users_file}` -- the latent attributes used to generate them.

These rows are produced by `src/data/synthetic_interactions.py`. They do not
represent any real person's travel history. Metrics computed on them measure
whether the pipeline recovers the structure the generator injected, not
real-world recommendation accuracy.

## Documented proxies and limitations

* **Popularity** is Wikipedia pageviews, i.e. online interest, not visitor
  arrivals. Cities with large diasporas or recent news coverage rank higher
  than tourism alone would justify.
* **Cost** is country-level GNI per capita (PPP). Every city in one country
  shares a cost band, and tourist-specific pricing is not reflected.
* **Attributes** are OpenStreetMap POI counts. OSM mapping density varies by
  region, so absolute `score_*` values overstate well-mapped areas; the
  `profile_*` shares are computed within a city and are far less affected.
* **Coverage**: {poi_coverage:.1f}% of destinations have OSM attribute data and
  {climate_coverage:.1f}% have climate normals.
"""


def build(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Engineer features, generate interactions and write all outputs."""
    processed = config.path("data_processed")
    raw_path = processed / RAW_DESTINATIONS_FILE
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Run `python -m src.data.build_destinations` first."
        )

    raw = pd.read_parquet(raw_path)
    LOGGER.info("Loaded %d raw destinations", len(raw))

    destinations = build_destination_features(raw)
    destinations.to_parquet(processed / DESTINATIONS_FILE, index=False)
    LOGGER.info("Wrote %s", DESTINATIONS_FILE)

    settings = settings_from_config(config)
    interactions, users = generate_interactions(destinations, settings)
    interactions.to_parquet(processed / INTERACTIONS_FILE, index=False)
    users.to_parquet(processed / USERS_FILE, index=False)
    LOGGER.info("Wrote %s and %s", INTERACTIONS_FILE, USERS_FILE)

    provenance = _PROVENANCE_TEMPLATE.format(
        today=date.today().isoformat(),
        destinations_file=DESTINATIONS_FILE,
        interactions_file=INTERACTIONS_FILE,
        users_file=USERS_FILE,
        n_destinations=len(destinations),
        n_countries=int(destinations["country_code"].nunique()),
        n_interactions=len(interactions),
        n_users=len(users),
        poi_coverage=100.0 * float(destinations["poi_available"].mean()),
        climate_coverage=100.0 * float(destinations["climate_available"].mean()),
    )
    (processed / PROVENANCE_FILE).write_text(provenance, encoding="utf-8")
    LOGGER.info("Wrote %s", PROVENANCE_FILE)

    return destinations, interactions, users


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TravelNext modelling dataset")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    destinations, interactions, users = build(config)
    print(
        f"destinations={len(destinations)} interactions={len(interactions)} "
        f"users={len(users)} (interactions are SYNTHETIC)"
    )


if __name__ == "__main__":
    main()
