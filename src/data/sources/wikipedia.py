"""Short article summaries from the Wikipedia REST API.

Source: https://en.wikipedia.org/api/rest_v1 — free, key-less. Article text is
licensed CC BY-SA 4.0; we store only the short lead extract per city and use it
purely as input to a TF-IDF vectoriser, never redisplaying it verbatim in bulk.

The summary is what lets the content-based model reason about *character*
("canal", "baroque", "nightlife", "beach") in a way POI counts alone cannot.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List
from urllib.parse import quote

import pandas as pd
import requests

from src.utils.http import HttpCache, RateLimiter, is_missing, request_json
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
# The REST API tolerates more than action=api.php, but we still cap the
# aggregate rate so a wide thread pool cannot trigger throttling.
_RATE_LIMITER = RateLimiter(requests_per_second=10.0)


def _fetch_one(
    title: str,
    *,
    cache: HttpCache,
    user_agent: str,
    timeout: int,
    session: requests.Session,
) -> Dict[str, str]:
    """Fetch the lead extract for a single article title."""
    payload = request_json(
        SUMMARY_URL.format(title=quote(title.replace(" ", "_"), safe="")),
        cache=cache,
        cache_key=f"summary:{title}",
        user_agent=user_agent,
        timeout=timeout,
        retries=3,
        backoff_s=3.0,
        session=session,
        rate_limiter=_RATE_LIMITER,
    )
    if not payload or is_missing(payload):
        return {"wiki_title": title, "summary": "", "image_url": "", "image_page": ""}

    # The same response carries a Commons image. Using it keeps destination
    # photography free, key-less and correctly licensed -- a stock-photo API
    # would need an account and would break the project's zero-cost guarantee,
    # and these pictures are of the actual destination rather than a generic
    # "travel" image.
    thumbnail = payload.get("thumbnail") or {}
    original = payload.get("originalimage") or {}
    return {
        "wiki_title": title,
        "summary": str(payload.get("extract", "") or ""),
        "image_url": str(thumbnail.get("source", "") or ""),
        "image_page": str(
            (payload.get("content_urls", {}).get("desktop", {}) or {}).get("page", "") or ""
        ),
        "image_width": int(original.get("width", 0) or 0),
    }


def fetch_summaries(
    titles: Iterable[str],
    cache_dir,
    *,
    user_agent: str = "TravelNext/0.1",
    timeout: int = 30,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Fetch lead extracts for many articles. Missing articles yield ``""``."""
    cache = HttpCache(cache_dir, "wiki_summary")
    unique_titles: List[str] = list(dict.fromkeys(t for t in titles if t))
    LOGGER.info("Fetching Wikipedia summaries for %d articles", len(unique_titles))

    rows: List[Dict[str, str]] = []
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    _fetch_one,
                    title,
                    cache=cache,
                    user_agent=user_agent,
                    timeout=timeout,
                    session=session,
                )
                for title in unique_titles
            ]
            for index, future in enumerate(futures, start=1):
                rows.append(future.result())
                if index % 200 == 0:
                    LOGGER.info("  summary progress: %d/%d", index, len(unique_titles))

    frame = pd.DataFrame(rows)
    non_empty = int((frame["summary"].str.len() > 0).sum())
    with_image = int((frame["image_url"].str.len() > 0).sum())
    LOGGER.info("Summaries resolved for %d/%d articles", non_empty, len(frame))
    LOGGER.info("Commons images found for %d/%d articles", with_image, len(frame))
    return frame
