"""Destination attributes derived from OpenStreetMap POI density.

Source: the Overpass API (a free, community-run query service over OSM data).
OSM data is licensed under the Open Database License (ODbL 1.0); attribution is
given in the README and in the API's ``/health`` payload.

WHY POI COUNTS
--------------
The brief forbids inventing attribute values. Counting the actual museums,
bars, parks and beaches mapped within a radius of each city centre gives a
measurable, reproducible and auditable signal for every travel characteristic
we expose. The numbers are raw counts here; normalisation into 0-1 scores
happens in ``src/preprocessing``.

KNOWN BIAS
----------
OSM coverage is uneven: Western Europe is mapped far more densely than much of
Asia and Africa, so raw counts overstate European cities. We therefore rank
cities *within their own region* when normalising, and document the residual
bias rather than papering over it.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from src.utils.http import HttpCache
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

# Each entry maps a feature name to the OSM selectors counted for it.
# Selectors are intentionally narrow and cheap: `out count` avoids serialising
# geometry, but very broad selectors still make the endpoint do real work.
POI_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "museums": ('["tourism"="museum"]', '["tourism"="gallery"]'),
    "culture": (
        '["amenity"="theatre"]',
        '["amenity"="arts_centre"]',
        '["tourism"="artwork"]',
        '["amenity"="cinema"]',
    ),
    "heritage": (
        '["historic"="monument"]',
        '["historic"="memorial"]',
        '["historic"="castle"]',
        '["historic"="ruins"]',
        '["historic"="archaeological_site"]',
    ),
    "architecture": (
        '["building"="cathedral"]',
        '["building"="church"]',
        '["historic"="building"]',
        '["tourism"="attraction"]',
    ),
    "nightlife": (
        '["amenity"="bar"]',
        '["amenity"="pub"]',
        '["amenity"="nightclub"]',
    ),
    "food": ('["amenity"="restaurant"]', '["amenity"="cafe"]'),
    "nature": (
        '["leisure"="park"]',
        '["leisure"="nature_reserve"]',
        '["leisure"="garden"]',
    ),
    "beaches": ('["natural"="beach"]', '["leisure"="beach_resort"]'),
    "outdoor": (
        '["tourism"="viewpoint"]',
        '["natural"="peak"]',
        '["leisure"="sports_centre"]',
    ),
    "family": (
        '["tourism"="zoo"]',
        '["tourism"="theme_park"]',
        '["tourism"="aquarium"]',
        '["leisure"="playground"]',
    ),
    "shopping": (
        '["shop"="mall"]',
        '["shop"="department_store"]',
        '["amenity"="marketplace"]',
    ),
    "walkability": ('["highway"="pedestrian"]', '["highway"="living_street"]'),
    "tourism_infra": (
        '["tourism"="hotel"]',
        '["tourism"="hostel"]',
        '["tourism"="guest_house"]',
    ),
}

CATEGORY_NAMES: Tuple[str, ...] = tuple(POI_CATEGORIES)

# How long to back off when the mirror gives us no explicit slot time, and the
# ceiling on any single wait so one unhealthy mirror cannot stall a whole run.
_DEFAULT_COOLDOWN_S = 30.0
_MAX_COOLDOWN_S = 120.0


def build_query(lat: float, lon: float, radius_m: int, timeout_s: int) -> str:
    """Build one Overpass QL query returning a count per POI category.

    Each category emits its own ``out count;`` statement, so a single HTTP
    request yields all categories in declaration order.
    """
    lines = [f"[out:json][timeout:{timeout_s}];"]
    for selectors in POI_CATEGORIES.values():
        union = "".join(f"nwr(around:{radius_m},{lat:.5f},{lon:.5f}){sel};" for sel in selectors)
        lines.append(f"({union});out count;")
    return "\n".join(lines)


def seconds_until_slot(endpoint: str, *, user_agent: str, session=None) -> float:
    """Ask an Overpass mirror when our next query slot frees up.

    Overpass rejects over-quota queries with HTTP 429 or 504 within a few
    seconds. Without this the client cannot tell a rejection from a genuine
    timeout, and retrying immediately (or with a longer server-side timeout)
    just burns the next slot too. ``/api/status`` reports the wait explicitly,
    so we sleep exactly as long as the server asks and no longer.
    """
    status_url = endpoint.replace("/api/interpreter", "/api/status")
    http = session or requests
    try:
        response = http.get(status_url, headers={"User-Agent": user_agent}, timeout=30)
        response.raise_for_status()
        text = response.text
    except requests.RequestException:
        return _DEFAULT_COOLDOWN_S

    if "slots available" in text.lower():
        return 0.0
    waits = [int(match) for match in re.findall(r"in (\d+) seconds", text)]
    if not waits:
        return 0.0 if "Slot available after" not in text else _DEFAULT_COOLDOWN_S
    # Wait for the soonest slot, plus a small margin.
    return float(min(waits)) + 2.0


def _parse_counts(payload: dict) -> Optional[List[int]]:
    """Extract the ordered list of counts from an Overpass response."""
    elements = payload.get("elements", [])
    counts = [
        int(element.get("tags", {}).get("total", 0))
        for element in elements
        if element.get("type") == "count"
    ]
    if len(counts) != len(POI_CATEGORIES):
        return None
    return counts


def fetch_city_pois(
    lat: float,
    lon: float,
    *,
    cache: HttpCache,
    cache_key: str,
    endpoints: Sequence[str],
    radius_m: int = 5000,
    timeout_s: int = 90,
    retries: int = 3,
    user_agent: str = "TravelNext/0.1",
    session: Optional[requests.Session] = None,
) -> Optional[Dict[str, int]]:
    """Return POI counts per category for one city, or ``None`` on failure.

    Rotates across the configured mirrors so a single busy endpoint does not
    stall the run.
    """
    cached = cache.get(cache_key)
    if cached is not None:
        return {name: int(cached.get(name, 0)) for name in CATEGORY_NAMES}

    http = session or requests
    query = build_query(lat, lon, radius_m, timeout_s)
    for attempt in range(retries):
        endpoint = endpoints[attempt % len(endpoints)]
        try:
            response = http.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": user_agent},
                timeout=timeout_s + 30,
            )
            if response.status_code in (429, 504) or response.status_code >= 500:
                # Almost always "you are over quota", not a slow query: these
                # come back in seconds. Ask the server when we may return and
                # wait exactly that long.
                wait = seconds_until_slot(endpoint, user_agent=user_agent, session=http)
                LOGGER.debug(
                    "Overpass %s on %s; waiting %.0fs for a slot",
                    response.status_code,
                    cache_key,
                    wait,
                )
                time.sleep(min(wait, _MAX_COOLDOWN_S))
                continue
            response.raise_for_status()
            counts = _parse_counts(response.json())
            if counts is None:
                time.sleep(_DEFAULT_COOLDOWN_S)
                continue
            result = dict(zip(CATEGORY_NAMES, counts))
            cache.set(cache_key, result)
            return result
        except (requests.RequestException, ValueError):
            time.sleep(_DEFAULT_COOLDOWN_S)

    LOGGER.warning("Overpass failed for %s (%.4f, %.4f)", cache_key, lat, lon)
    return None


def fetch_poi_table(
    cities: pd.DataFrame,
    cache_dir,
    *,
    endpoints: Sequence[str],
    radius_m: int = 5000,
    timeout_s: int = 90,
    retries: int = 3,
    sleep_between_s: float = 1.5,
    user_agent: str = "TravelNext/0.1",
    id_column: str = "destination_id",
    progress_every: int = 25,
) -> pd.DataFrame:
    """Fetch POI counts for every city in ``cities``.

    Concurrency is deliberately capped at *one in-flight request per mirror*:
    Overpass is a donated public service, and its own scheduler allows only a
    small number of slots per client. Each worker owns one endpoint and sleeps
    between its own requests. Results are cached per city, so an interrupted
    run resumes for free and re-runs cost nothing.

    Cities whose query never succeeded are returned with ``poi_available=False``
    and zero counts, letting downstream code decide how to treat them.
    """
    cache = HttpCache(cache_dir, "overpass")
    work: "queue.Queue[Tuple[str, float, float]]" = queue.Queue()
    for city in cities.itertuples(index=False):
        work.put((str(getattr(city, id_column)), float(city.latitude), float(city.longitude)))

    total = work.qsize()
    results: Dict[str, Optional[Dict[str, int]]] = {}
    lock = threading.Lock()
    state = {"done": 0, "failed": 0}

    def worker(endpoint: str) -> None:
        """Drain the queue using a single dedicated Overpass mirror."""
        with requests.Session() as session:
            while True:
                try:
                    key, lat, lon = work.get_nowait()
                except queue.Empty:
                    return
                try:
                    was_cached = cache.get(key) is not None
                    counts = fetch_city_pois(
                        lat,
                        lon,
                        cache=cache,
                        cache_key=key,
                        endpoints=(endpoint,),
                        radius_m=radius_m,
                        timeout_s=timeout_s,
                        retries=retries,
                        user_agent=user_agent,
                        session=session,
                    )
                    with lock:
                        results[key] = counts
                        state["done"] += 1
                        if counts is None:
                            state["failed"] += 1
                        if state["done"] % progress_every == 0:
                            LOGGER.info(
                                "Overpass progress: %d/%d (%d failed)",
                                state["done"],
                                total,
                                state["failed"],
                            )
                    if not was_cached:
                        # Only throttle when we actually hit the network.
                        time.sleep(sleep_between_s)
                finally:
                    work.task_done()

    threads = [
        threading.Thread(target=worker, args=(endpoint,), daemon=True)
        for endpoint in (endpoints or ("https://overpass-api.de/api/interpreter",))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows: List[Dict[str, object]] = []
    for city in cities.itertuples(index=False):
        key = str(getattr(city, id_column))
        counts = results.get(key)
        row: Dict[str, object] = {id_column: key}
        if counts is None:
            row.update({f"poi_{name}": 0 for name in CATEGORY_NAMES})
            row["poi_available"] = False
        else:
            row.update({f"poi_{name}": int(value) for name, value in counts.items()})
            row["poi_available"] = True
        rows.append(row)

    LOGGER.info("Overpass complete: %d cities, %d failures", len(rows), state["failed"])
    return pd.DataFrame(rows)
