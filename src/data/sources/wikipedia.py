"""Short article summaries from the Wikipedia REST API.

Source: https://en.wikipedia.org/api/rest_v1 — free, key-less. Article text is
licensed CC BY-SA 4.0; we store only the short lead extract per city and use it
purely as input to a TF-IDF vectoriser, never redisplaying it verbatim in bulk.

The summary is what lets the content-based model reason about *character*
("canal", "baroque", "nightlife", "beach") in a way POI counts alone cannot.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List
from urllib.parse import quote

import pandas as pd
import requests

from src.utils.http import HttpCache, RateLimiter, is_missing, request_json
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
ACTION_API = "https://en.wikipedia.org/w/api.php"
# The action API renders a thumbnail at any requested width and accepts 50
# titles per request. Rewriting the width inside an existing thumbnail URL
# looks tempting but returns HTTP 400 -- the size has to be asked for.
IMAGE_BATCH_SIZE = 50
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


_THUMB_WIDTH_RE = re.compile(r"/(\d+)px-")


def _delivered_width(url: str, reported: int) -> int:
    """Return the width of the file the URL actually serves.

    ``thumbnail.width`` from the API is the width that was *requested*, not the
    one delivered: asking for 400px returns a URL ending ``/500px-...`` and a
    file that really is 500px wide (verified by decoding the JPEG header).
    Trusting the reported value would put a false descriptor in ``srcset`` and
    have the browser choose the wrong candidate, so the URL wins.
    """
    match = _THUMB_WIDTH_RE.search(url or "")
    return int(match.group(1)) if match else int(reported or 0)


def fetch_page_images(
    titles: Iterable[str],
    cache_dir,
    *,
    sizes: tuple[int, int, int] = (400, 1000, 1600),
    user_agent: str = "TravelNext/0.1",
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch three widths of the lead image for each article.

    The product displays these photographs at very different sizes, and one
    file cannot serve all of them well:

    * ~500px  - list rows and search suggestions
    * ~1024px - recommendation cards (353px wide, so ~700px on a retina panel)
    * ~1920px - the login hero and the explanation drawer

    The middle size matters more than it looks. With only 500px and 1920px
    available the browser correctly rejects the small file as too soft for a
    retina card and pulls the 1920px one instead -- about 530KB per card, or
    4.7MB for a nine-card screen. The 1024px candidate cuts that by roughly
    three quarters with no visible difference.

    The originals are 4000-5000px wide, so none of these is an upscale.
    """
    cache = HttpCache(cache_dir, "page_images")
    unique_titles: List[str] = list(dict.fromkeys(t for t in titles if t))
    small_width, medium_width, large_width = sizes

    def fetch_size(width: int) -> Dict[str, dict]:
        found: Dict[str, dict] = {}
        chunks = [
            unique_titles[start : start + IMAGE_BATCH_SIZE]
            for start in range(0, len(unique_titles), IMAGE_BATCH_SIZE)
        ]
        with requests.Session() as session:
            for index, chunk in enumerate(chunks, start=1):
                payload = request_json(
                    ACTION_API,
                    params={
                        "action": "query",
                        "format": "json",
                        "prop": "pageimages",
                        "piprop": "thumbnail",
                        "pithumbsize": width,
                        "pilimit": "max",
                        "titles": "|".join(chunk),
                        "redirects": 1,
                    },
                    cache=cache,
                    cache_key=f"pageimages:{width}:" + "|".join(chunk),
                    user_agent=user_agent,
                    timeout=timeout,
                    retries=3,
                    backoff_s=4.0,
                    session=session,
                    rate_limiter=_RATE_LIMITER,
                )
                query = (payload or {}).get("query", {}) or {}
                # Replay redirects/normalisation so a requested title can be
                # traced to the page the API actually answered with.
                alias: Dict[str, str] = {}
                for step in ("normalized", "redirects"):
                    for entry in query.get(step, []) or []:
                        alias[entry.get("from", "")] = entry.get("to", "")

                pages = {
                    page.get("title"): page
                    for page in (query.get("pages", {}) or {}).values()
                    if page.get("title")
                }
                for requested in chunk:
                    resolved = requested
                    seen = set()
                    while resolved in alias and resolved not in seen:
                        seen.add(resolved)
                        resolved = alias[resolved]
                    page = pages.get(resolved)
                    thumbnail = (page or {}).get("thumbnail") or {}
                    if thumbnail.get("source"):
                        found[requested] = thumbnail
                if index % 4 == 0:
                    LOGGER.info("  images @%dpx: %d/%d batches", width, index, len(chunks))
        return found

    LOGGER.info("Fetching page images for %d articles", len(unique_titles))
    small = fetch_size(small_width)
    medium = fetch_size(medium_width)
    large = fetch_size(large_width)

    rows = [
        {
            "wiki_title": title,
            "image_url": str(small.get(title, {}).get("source", "") or ""),
            "image_width": _delivered_width(
                str(small.get(title, {}).get("source", "") or ""),
                int(small.get(title, {}).get("width", 0) or 0),
            ),
            "image_url_md": str(medium.get(title, {}).get("source", "") or ""),
            "image_width_md": _delivered_width(
                str(medium.get(title, {}).get("source", "") or ""),
                int(medium.get(title, {}).get("width", 0) or 0),
            ),
            "image_url_hd": str(large.get(title, {}).get("source", "") or ""),
            "image_width_hd": _delivered_width(
                str(large.get(title, {}).get("source", "") or ""),
                int(large.get(title, {}).get("width", 0) or 0),
            ),
            "image_height_hd": int(large.get(title, {}).get("height", 0) or 0),
        }
        for title in unique_titles
    ]
    frame = pd.DataFrame(rows)
    LOGGER.info(
        "Images: %d small, %d high-resolution (of %d articles)",
        int((frame["image_url"].str.len() > 0).sum()),
        int((frame["image_url_hd"].str.len() > 0).sum()),
        len(frame),
    )
    return frame
