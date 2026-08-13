"""Country cost proxy from World Bank open data.

Source: the World Bank Indicators API (https://api.worldbank.org/v2). The data
is published under CC BY 4.0 and the API is free and key-less.

TRANSPARENCY NOTE
-----------------
This is a *proxy*, not a travel-cost dataset. We use GNI per capita at
purchasing power parity as a stand-in for how expensive a destination is for a
visitor. It is country-level, so Munich and Leipzig receive the same cost band,
and it ignores tourist-specific pricing. A per-city cost-of-living dataset with
a permissive licence does not exist for free, so rather than inventing numbers
we use this documented approximation and label it ``cost_proxy_*`` throughout.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.utils.http import HttpCache, request_json
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

API_URL = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"


def fetch_country_cost_proxy(
    cache_dir,
    *,
    indicator: str = "NY.GNP.PCAP.PP.CD",
    user_agent: str = "TravelNext/0.1",
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch the most recent available indicator value per country.

    Returns a frame with ``country_code`` (ISO alpha-2), ``cost_proxy_value``
    and ``cost_proxy_year``.
    """
    cache = HttpCache(cache_dir, "worldbank")
    rows: List[Dict[str, object]] = []
    page = 1

    while True:
        payload = request_json(
            API_URL.format(indicator=indicator),
            params={
                "format": "json",
                "per_page": 1000,
                "page": page,
                # A 10-year window guarantees a recent non-null observation for
                # nearly every country.
                "date": "2015:2024",
            },
            cache=cache,
            cache_key=f"wb:{indicator}:{page}",
            user_agent=user_agent,
            timeout=timeout,
            retries=2,
        )
        if not payload or len(payload) < 2 or not payload[1]:
            break

        for entry in payload[1]:
            value = entry.get("value")
            iso2 = (entry.get("country", {}) or {}).get("id")
            if value is None or not iso2:
                continue
            rows.append(
                {
                    "country_code": str(iso2).upper(),
                    "cost_proxy_value": float(value),
                    "cost_proxy_year": int(entry.get("date", 0)),
                }
            )

        meta = payload[0]
        if page >= int(meta.get("pages", 1)):
            break
        page += 1

    if not rows:
        LOGGER.warning("World Bank returned no cost data; cost features will be missing")
        return pd.DataFrame(columns=["country_code", "cost_proxy_value", "cost_proxy_year"])

    frame = pd.DataFrame(rows)
    # Keep the most recent observation per country.
    frame = frame.sort_values("cost_proxy_year", ascending=False).drop_duplicates(
        subset=["country_code"], keep="first"
    )
    # The API mixes aggregates (EU, World) in with countries; ISO alpha-2 codes
    # for real countries are what we join on, so aggregates simply never match.
    LOGGER.info("World Bank cost proxy: %d countries", len(frame))
    return frame.reset_index(drop=True)
