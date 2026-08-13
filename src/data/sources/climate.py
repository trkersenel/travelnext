"""Monthly climate normals from the Open-Meteo historical archive.

Source: https://archive-api.open-meteo.com — free for non-commercial use, no
API key, no registration. The underlying reanalysis data (ERA5) is published by
Copernicus under a licence permitting free use with attribution; Open-Meteo
distributes its API responses under CC BY 4.0.

We aggregate several years of daily observations into per-month means, giving a
real seasonality signal (used for "is September a good time to visit?") rather
than a guessed climate label.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import pandas as pd
import requests

from src.utils.http import HttpCache, RateLimiter, is_failed, request_json
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Open-Meteo's free tier throttles bursts (measured: 4 concurrent workers draw
# HTTP 429). Five years of daily data per city is a heavy query, so we keep the
# aggregate rate low and let the disk cache make re-runs free.
_RATE_LIMITER = RateLimiter(requests_per_second=2.0)
MONTH_COLUMNS = [f"{stat}_m{month:02d}" for stat in ("temp", "precip") for month in range(1, 13)]


def _fetch_one(
    destination_id: str,
    lat: float,
    lon: float,
    *,
    start_date: str,
    end_date: str,
    cache: HttpCache,
    user_agent: str,
    timeout: int,
    session: requests.Session,
) -> Optional[Dict[str, float]]:
    """Fetch daily history for one city and reduce it to monthly normals."""
    payload = request_json(
        ARCHIVE_URL,
        params={
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "UTC",
        },
        cache=cache,
        cache_key=f"climate:{destination_id}:{start_date}:{end_date}",
        user_agent=user_agent,
        timeout=timeout,
        retries=4,
        backoff_s=4.0,
        session=session,
        rate_limiter=_RATE_LIMITER,
    )
    if is_failed(payload):
        # Distinguish "could not ask" from "no data": the caller retries the
        # former rather than baking a throttled request in as missing climate.
        return None
    daily = (payload or {}).get("daily") or {}
    times = daily.get("time") or []
    temps = daily.get("temperature_2m_mean") or []
    precip = daily.get("precipitation_sum") or []
    if not times or len(times) != len(temps):
        return None

    frame = pd.DataFrame(
        {
            "month": pd.to_datetime(pd.Series(times)).dt.month,
            "temp": pd.to_numeric(pd.Series(temps), errors="coerce"),
            "precip": pd.to_numeric(pd.Series(precip), errors="coerce")
            if len(precip) == len(times)
            else pd.Series([float("nan")] * len(times)),
        }
    )
    grouped = frame.groupby("month").agg(temp=("temp", "mean"), precip=("precip", "mean"))

    record: Dict[str, float] = {"destination_id": destination_id}
    for month in range(1, 13):
        if month in grouped.index:
            record[f"temp_m{month:02d}"] = float(grouped.loc[month, "temp"])
            precip_value = grouped.loc[month, "precip"]
            # Mean daily precipitation -> approximate monthly total (mm).
            record[f"precip_m{month:02d}"] = (
                float(precip_value) * 30.0 if pd.notna(precip_value) else float("nan")
            )
        else:
            record[f"temp_m{month:02d}"] = float("nan")
            record[f"precip_m{month:02d}"] = float("nan")
    return record


def fetch_climate_table(
    cities: pd.DataFrame,
    cache_dir,
    *,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    user_agent: str = "TravelNext/0.1",
    timeout: int = 60,
    max_workers: int = 4,
    id_column: str = "destination_id",
) -> pd.DataFrame:
    """Fetch monthly temperature and precipitation normals for every city.

    Cities the archive cannot serve are returned with NaN columns; downstream
    code treats a missing climate profile as "no seasonal preference" instead
    of failing.
    """
    cache = HttpCache(cache_dir, "climate")
    rows = list(cities.itertuples(index=False))

    def run_pass(subset, workers: int) -> List[Optional[Dict[str, float]]]:
        collected: List[Optional[Dict[str, float]]] = []
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _fetch_one,
                        str(getattr(city, id_column)),
                        float(city.latitude),
                        float(city.longitude),
                        start_date=start_date,
                        end_date=end_date,
                        cache=cache,
                        user_agent=user_agent,
                        timeout=timeout,
                        session=session,
                    )
                    for city in subset
                ]
                for index, future in enumerate(futures, start=1):
                    collected.append(future.result())
                    if index % 100 == 0:
                        LOGGER.info("Climate progress: %d/%d", index, len(futures))
        return collected

    records = run_pass(rows, max_workers)

    # Retry the throttled cities serially so a burst of 429s does not turn into
    # permanently missing climate data.
    failed = [city for city, record in zip(rows, records) if record is None]
    if failed:
        LOGGER.warning("Retrying %d failed climate requests serially", len(failed))
        retried = run_pass(failed, 1)
        records = [r for r in records if r is not None] + [r for r in retried if r is not None]

    ids = [str(getattr(city, id_column)) for city in rows]
    resolved = [record for record in records if record is not None]
    frame = pd.DataFrame(resolved) if resolved else pd.DataFrame(columns=["destination_id"])
    frame = pd.DataFrame({id_column: ids}).merge(frame, on=id_column, how="left")
    LOGGER.info("Climate resolved for %d/%d cities", len(resolved), len(ids))
    return frame
