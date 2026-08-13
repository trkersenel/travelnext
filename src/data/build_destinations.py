"""Build the unified destination dataset from free public sources.

Run with::

    python -m src.data.build_destinations

Every stage caches its HTTP responses under ``data/raw/cache``, so the script
is safe to interrupt and re-run: a second run performs no network requests at
all. The output is ``data/processed/destinations.parquet``.

Pipeline
--------
1. GeoNames      -> city catalog (name, country, coords, population)
2. Wikipedia     -> article match per city, verified by coordinates
3. Pageviews     -> popularity proxy, used to rank the catalog down to size
4. Overpass/OSM  -> POI counts per travel category
5. Open-Meteo    -> monthly climate normals
6. World Bank    -> country-level cost proxy
7. Wikipedia     -> lead extracts for text features
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import List

import pandas as pd

from src.config import Config, load_config
from src.data import regions
from src.data.sources import (
    climate,
    geonames,
    overpass,
    pageviews,
    wiki_resolve,
    wikipedia,
    worldbank,
)
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Return a lowercase ASCII slug for ``value``."""
    normalised = unicodedata.normalize("NFKD", value or "")
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii").lower()
    return _SLUG_STRIP.sub("-", ascii_only).strip("-") or "unknown"


def assign_destination_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable, unique, human-readable ``destination_id`` per city."""
    base = frame["city"].map(slugify) + "-" + frame["country_code"].str.lower()
    # Disambiguate genuine collisions (e.g. two "Springfield-us") with a suffix.
    counts = base.groupby(base).cumcount()
    frame = frame.copy()
    frame["destination_id"] = [
        slug if index == 0 else f"{slug}-{index + 1}" for slug, index in zip(base, counts)
    ]
    return frame


def _cache_dir(config: Config) -> Path:
    path = config.path("data_raw") / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build(config: Config, *, limit: int | None = None, skip_overpass: bool = False) -> pd.DataFrame:
    """Execute the full ingestion pipeline and return the destination frame."""
    cache_dir = _cache_dir(config)
    user_agent = str(config.get("ingest.user_agent", "TravelNext/0.1"))
    timeout = int(config.get("ingest.request_timeout_s", 60))
    interim = config.path("data_interim")

    # ---------------------------------------------------------- 1. catalog
    LOGGER.info("[1/7] GeoNames city catalog")
    catalog = geonames.load_cities(
        config.path("data_raw"),
        min_population=int(config.get("ingest.min_population", 100_000)),
        user_agent=user_agent,
        timeout=max(timeout, 120),
    )

    # ---------------------------------------------------- 2. wiki matching
    LOGGER.info("[2/7] Matching cities to Wikipedia articles")
    catalog = wiki_resolve.resolve_wiki_titles(
        catalog, cache_dir, user_agent=user_agent, timeout=timeout, max_workers=3
    )
    # An unmatched city has no popularity signal and no text features, so it
    # cannot be scored or explained. Dropping it is honest; keeping it with
    # zeros would silently pollute every model.
    before = len(catalog)
    catalog = catalog[catalog["wiki_resolved"]].reset_index(drop=True)
    LOGGER.info("Kept %d/%d cities with a verified Wikipedia article", len(catalog), before)
    catalog = assign_destination_ids(catalog)
    catalog.to_parquet(interim / "geonames_cities.parquet", index=False)

    # ------------------------------------------------------- 3. popularity
    LOGGER.info("[3/7] Wikipedia pageviews (popularity proxy)")
    views = pageviews.fetch_pageviews(
        catalog["wiki_title"],
        cache_dir,
        months=int(config.get("ingest.pageviews.months", 24)),
        user_agent=user_agent,
        timeout=timeout,
        retries=int(config.get("ingest.pageviews.retries", 2)),
        max_workers=int(config.get("ingest.pageviews.max_workers", 8)),
    )
    catalog = catalog.merge(views, on="wiki_title", how="left")
    view_columns = [
        "pageviews_total",
        "pageviews_monthly_mean",
        "pageviews_median",
        "pageviews_max",
    ]
    catalog[view_columns] = catalog[view_columns].fillna(0.0)
    catalog["pageviews_anomaly_ratio"] = catalog["pageviews_anomaly_ratio"].fillna(1.0)
    catalog["pageviews_available"] = catalog["pageviews_available"].fillna(False).astype(bool)
    unavailable = int((~catalog["pageviews_available"]).sum())
    if unavailable:
        # These have no trustworthy popularity value, so ranking them by it
        # would be meaningless. Excluding them is preferable to letting a
        # throttled request masquerade as an unpopular city.
        LOGGER.warning("Dropping %d cities whose pageviews could not be fetched", unavailable)
        catalog = catalog[catalog["pageviews_available"]].reset_index(drop=True)

    # Rank the world's cities by online interest and keep the top slice. This
    # is an objective, reproducible filter for "places travellers look up",
    # and it keeps small-but-touristic cities (Bruges, Salzburg) that a plain
    # population cut-off would discard.
    catalog_size = int(config.get("ingest.catalog_size", 800))
    if limit is not None:
        catalog_size = min(catalog_size, limit)
    # Ranked by the MEDIAN monthly views, not the total. Bot traffic arrives in
    # short bursts, and ranking by total let two cities with near-identical
    # million-view spikes (Utrecht, Durres) outrank London.
    catalog = (
        catalog.sort_values("pageviews_median", ascending=False)
        .head(catalog_size)
        .reset_index(drop=True)
    )
    LOGGER.info("Catalog reduced to %d destinations", len(catalog))
    catalog.to_parquet(interim / "catalog_ranked.parquet", index=False)

    # -------------------------------------------------------- 3. OSM POIs
    if skip_overpass:
        LOGGER.warning("[4/7] Skipping Overpass (--skip-overpass)")
        poi = pd.DataFrame({"destination_id": catalog["destination_id"]})
        for name in overpass.CATEGORY_NAMES:
            poi[f"poi_{name}"] = 0
        poi["poi_available"] = False
    else:
        LOGGER.info("[4/7] OpenStreetMap POI counts via Overpass")
        poi = overpass.fetch_poi_table(
            catalog[["destination_id", "latitude", "longitude"]],
            cache_dir,
            endpoints=list(config.get("ingest.overpass.endpoints", [])),
            radius_m=int(config.get("ingest.overpass.radius_m", 5000)),
            timeout_s=int(config.get("ingest.overpass.timeout_s", 90)),
            retries=int(config.get("ingest.overpass.retries", 3)),
            sleep_between_s=float(config.get("ingest.overpass.sleep_between_s", 1.5)),
            user_agent=user_agent,
        )
    poi.to_parquet(interim / "osm_pois.parquet", index=False)

    # --------------------------------------------------------- 4. climate
    LOGGER.info("[5/7] Open-Meteo climate normals")
    climate_table = climate.fetch_climate_table(
        catalog[["destination_id", "latitude", "longitude"]],
        cache_dir,
        start_date=str(config.get("ingest.climate.start_date", "2020-01-01")),
        end_date=str(config.get("ingest.climate.end_date", "2024-12-31")),
        user_agent=user_agent,
        timeout=timeout,
        max_workers=int(config.get("ingest.climate.max_workers", 4)),
    )
    climate_table.to_parquet(interim / "climate.parquet", index=False)

    # ------------------------------------------------------------ 5. cost
    LOGGER.info("[6/7] World Bank cost proxy")
    cost = worldbank.fetch_country_cost_proxy(
        cache_dir,
        indicator=str(config.get("ingest.worldbank.indicator", "NY.GNP.PCAP.PP.CD")),
        user_agent=user_agent,
        timeout=timeout,
    )
    cost.to_parquet(interim / "cost_proxy.parquet", index=False)

    # ------------------------------------------------------------ 6. text
    LOGGER.info("[7/7] Wikipedia summaries")
    summaries = wikipedia.fetch_summaries(
        catalog["wiki_title"],
        cache_dir,
        user_agent=user_agent,
        timeout=timeout,
        max_workers=int(config.get("ingest.pageviews.max_workers", 8)),
    )
    summaries.to_parquet(interim / "summaries.parquet", index=False)

    # ----------------------------------------------------------- assemble
    frame = (
        catalog.merge(poi, on="destination_id", how="left")
        .merge(climate_table, on="destination_id", how="left")
        .merge(cost, on="country_code", how="left")
        .merge(summaries, on="wiki_title", how="left")
    )
    frame["continent"] = frame["country_code"].map(regions.continent_of)
    frame["region"] = frame["country_code"].map(regions.region_of)
    frame["summary"] = frame["summary"].fillna("")
    frame["poi_available"] = frame["poi_available"].fillna(False).astype(bool)

    poi_columns: List[str] = [f"poi_{name}" for name in overpass.CATEGORY_NAMES]
    frame[poi_columns] = frame[poi_columns].fillna(0).astype("int64")

    output = config.path("data_processed") / "destinations_raw.parquet"
    frame.to_parquet(output, index=False)
    LOGGER.info("Wrote %d destinations -> %s", len(frame), output)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TravelNext destination dataset")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Cap the catalog size (for testing)")
    parser.add_argument(
        "--skip-overpass",
        action="store_true",
        help="Skip the slow OSM stage (produces zeroed POI columns)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    frame = build(config, limit=args.limit, skip_overpass=args.skip_overpass)
    print(f"Destinations: {len(frame)}  Countries: {frame['country_code'].nunique()}")


if __name__ == "__main__":
    main()
