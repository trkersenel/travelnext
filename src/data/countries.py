"""The country catalog behind the "countries visited" world map.

Source: Natural Earth 1:110m Admin 0 boundaries, which are in the **public
domain**. A slimmed copy lives at ``web/countries.geojson`` (properties reduced
to an ISO code and a name, coordinates rounded to two decimals), which takes it
from 819 KB to 169 KB — small enough to ship to the browser on first paint.

Why this list rather than the destination catalog: the recommender only knows
400 cities across ~107 countries, but a traveller may well have been to a
country Waygo has no city for. Marking a country visited must not be limited by
what the recommender happens to cover, so the map's own 175 countries are the
authority here. The two are joined only where they overlap.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from src.data import regions
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEOJSON_PATH = PROJECT_ROOT / "web" / "countries.geojson"


@lru_cache(maxsize=1)
def load_countries() -> List[Dict[str, Any]]:
    """Return every mappable country, sorted by name.

    Each entry carries the ISO alpha-2 code the map keys on, the display name,
    and the continent from the bundled UN M49 mapping. Returns an empty list if
    the GeoJSON is missing, so the API degrades to "no map" rather than 500.
    """
    if not GEOJSON_PATH.exists():
        LOGGER.warning("%s not found; the world map will be unavailable", GEOJSON_PATH)
        return []

    with GEOJSON_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    countries: List[Dict[str, Any]] = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        iso = str(properties.get("iso", "")).upper()
        if len(iso) != 2:
            continue
        countries.append(
            {
                "country_code": iso,
                "name": str(properties.get("name", iso)),
                "continent": regions.continent_of(iso),
                "region": regions.region_of(iso),
            }
        )

    countries.sort(key=lambda entry: entry["name"])
    LOGGER.info("Country catalog: %d mappable countries", len(countries))
    return countries


@lru_cache(maxsize=1)
def country_names() -> Dict[str, str]:
    """ISO alpha-2 to display name."""
    return {entry["country_code"]: entry["name"] for entry in load_countries()}


def valid_country_codes() -> frozenset[str]:
    """Codes the map can actually colour in."""
    return frozenset(country_names())


def normalise(code: str) -> str:
    """Upper-case an ISO code, returning '' when it is not mappable."""
    candidate = (code or "").strip().upper()
    return candidate if candidate in valid_country_codes() else ""
