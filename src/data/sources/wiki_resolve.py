"""Match GeoNames cities to English Wikipedia articles.

The Wikipedia article title is the join key for two other free data sources
(pageview popularity and the lead-extract text features), so getting this match
right matters more than it first appears.

Matches are *verified geographically*: the MediaWiki API returns each article's
own coordinates, and a candidate is accepted only when it sits within a small
radius of the GeoNames city centre. That rejects the classic failure modes --
"Cambridge" resolving to the English one when we asked about the American one,
a city name that is also a band or a film, or a redirect that lands on a
region article -- without any manual curation.

API: https://en.wikipedia.org/w/api.php (free, no key, no registration).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from src.utils.http import HttpCache, RateLimiter, request_json
from src.utils.geo import haversine_km
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

API_URL = "https://en.wikipedia.org/w/api.php"
# The MediaWiki API accepts up to 50 titles per request for anonymous clients.
BATCH_SIZE = 50
# A city article's coordinates should sit close to the GeoNames city centre.
# 35km tolerates large metros (Greater London, Istanbul) without letting a
# neighbouring city through.
MAX_MATCH_DISTANCE_KM = 35.0
# Aggregate cap across all worker threads. The MediaWiki action API returns
# HTTP 429 for bursty anonymous traffic; ~2 req/s stays comfortably inside it
# while still resolving 6000 cities in a couple of minutes (50 per request).
_RATE_LIMITER = RateLimiter(requests_per_second=2.0)


def _extract_pages(payload: dict) -> Tuple[Dict[str, str], Dict[str, dict]]:
    """Return (requested-title -> final-title, final-title -> page record).

    MediaWiki reports title normalisation and redirects as separate lists; we
    replay both so a requested title can be traced to the page it landed on.
    """
    query = (payload or {}).get("query", {})
    mapping: Dict[str, str] = {}
    for step in ("normalized", "redirects"):
        for entry in query.get(step, []) or []:
            mapping[entry.get("from", "")] = entry.get("to", "")

    def resolve(title: str) -> str:
        seen = set()
        current = title
        while current in mapping and current not in seen:
            seen.add(current)
            current = mapping[current]
        return current

    pages: Dict[str, dict] = {}
    for page in (query.get("pages", {}) or {}).values():
        title = page.get("title")
        if title and "missing" not in page:
            pages[title] = page

    return {"__resolve__": resolve}, pages  # type: ignore[return-value]


def _page_coordinates(page: dict) -> Tuple[Optional[float], Optional[float]]:
    """Return the primary coordinates of a page record, if any."""
    coords = page.get("coordinates") or []
    if not coords:
        return None, None
    first = coords[0]
    return first.get("lat"), first.get("lon")


def _fetch_batch(
    titles: Sequence[str],
    *,
    cache: HttpCache,
    user_agent: str,
    timeout: int,
    session: requests.Session,
) -> Tuple[Dict[str, str], Dict[str, dict]]:
    """Fetch coordinates and page props for up to ``BATCH_SIZE`` titles."""
    payload = request_json(
        API_URL,
        params={
            "action": "query",
            "format": "json",
            "prop": "coordinates|pageprops",
            "titles": "|".join(titles),
            "redirects": 1,
            "ppprop": "wikibase_item|disambiguation",
            # Without this the API attaches coordinates to only the first 10
            # pages of the batch and silently omits them from the rest, which
            # looks exactly like "these cities have no article".
            "colimit": "max",
        },
        cache=cache,
        cache_key="wtitles:" + "|".join(titles),
        user_agent=user_agent,
        timeout=timeout,
        retries=4,
        backoff_s=4.0,
        session=session,
        rate_limiter=_RATE_LIMITER,
    )
    return _extract_pages(payload or {})


def _search_candidates(
    query_text: str,
    *,
    cache: HttpCache,
    user_agent: str,
    timeout: int,
    session: requests.Session,
    limit: int = 4,
) -> List[str]:
    """Return candidate article titles from the Wikipedia search index."""
    payload = request_json(
        API_URL,
        params={
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query_text,
            "srlimit": limit,
            "srnamespace": 0,
        },
        cache=cache,
        cache_key=f"wsearch:{query_text}",
        user_agent=user_agent,
        timeout=timeout,
        retries=4,
        backoff_s=4.0,
        session=session,
        rate_limiter=_RATE_LIMITER,
    )
    results = ((payload or {}).get("query", {}) or {}).get("search", []) or []
    return [entry["title"] for entry in results if entry.get("title")]


def title_variants(city: str, city_ascii: str, country: str, admin1: str) -> List[str]:
    """Candidate article titles for a city, in decreasing order of likelihood.

    English Wikipedia disambiguates non-unique settlement names with a comma
    suffix ("Springfield, Illinois"; "Marabá, Pará"; "Cordoba, Spain"). Trying
    those patterns directly is far cheaper than the search endpoint: variants
    for 50 cities fit into a single batched title lookup, whereas search costs
    one throttled request per city.
    """
    candidates: List[str] = []
    for name in dict.fromkeys(part for part in (city, city_ascii) if part):
        candidates.append(name)
        if admin1 and admin1 != name:
            candidates.append(f"{name}, {admin1}")
        if country:
            candidates.append(f"{name}, {country}")
    return list(dict.fromkeys(candidates))


def _accept(
    lat: float, lon: float, page: dict, max_distance_km: float
) -> Tuple[bool, Optional[float]]:
    """Decide whether ``page`` is geographically the city at (lat, lon)."""
    if (page.get("pageprops") or {}).get("disambiguation") is not None:
        return False, None
    page_lat, page_lon = _page_coordinates(page)
    if page_lat is None or page_lon is None:
        return False, None
    distance = float(haversine_km(lat, lon, float(page_lat), float(page_lon)))
    return distance <= max_distance_km, distance


def resolve_wiki_titles(
    cities: pd.DataFrame,
    cache_dir,
    *,
    user_agent: str = "TravelNext/0.1",
    timeout: int = 60,
    max_workers: int = 6,
    max_distance_km: float = MAX_MATCH_DISTANCE_KM,
    use_search_fallback: bool = False,
) -> pd.DataFrame:
    """Resolve a Wikipedia article for each city, verified by coordinates.

    Resolution proceeds in cheap-to-expensive passes:

    1. the plain city name, batched 50 titles per request;
    2. disambiguated variants ("Springfield, Illinois"), also batched;
    3. optionally the search index, one throttled request per city.

    Passes 1 and 2 cost roughly one request per 50 candidate titles, so the
    whole catalog resolves in a few minutes. Pass 3 is off by default because
    ``list=search`` is rate-limited far more aggressively and adds little once
    the variants have been tried.

    Returns one row per input city with ``wiki_title`` (empty when unresolved),
    ``wiki_match_distance_km`` and ``wiki_resolved``.
    """
    cache = HttpCache(cache_dir, "wiki_titles")
    frame = cities.reset_index(drop=True)
    resolved: Dict[int, Dict[str, object]] = {}

    def lookup_titles(titles: Sequence[str]) -> Dict[str, dict]:
        """Batch-resolve titles to page records, following redirects."""
        pages: Dict[str, dict] = {}
        chunks = [
            list(titles[start : start + BATCH_SIZE]) for start in range(0, len(titles), BATCH_SIZE)
        ]

        def run(chunk: List[str]) -> Tuple[Dict[str, str], Dict[str, dict]]:
            with requests.Session() as session:
                return _fetch_batch(
                    chunk, cache=cache, user_agent=user_agent, timeout=timeout, session=session
                )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for done, (resolver, batch_pages) in enumerate(pool.map(run, chunks), start=1):
                resolve_fn = resolver["__resolve__"]  # type: ignore[index]
                # Record the requested spelling too, so callers can look up by
                # the title they asked for rather than the redirect target.
                for requested in chunks[done - 1]:
                    page = batch_pages.get(resolve_fn(requested))
                    if page is not None:
                        pages[requested] = page
                pages.update(batch_pages)
                if done % 25 == 0:
                    LOGGER.info("  lookup: %d/%d batches", done, len(chunks))
        return pages

    def claim(index: int, page: dict) -> bool:
        """Accept ``page`` for city ``index`` if it validates geographically."""
        ok, distance = _accept(
            float(frame.at[index, "latitude"]),
            float(frame.at[index, "longitude"]),
            page,
            max_distance_km,
        )
        if not ok:
            return False
        resolved[index] = {
            "wiki_title": page["title"],
            "wiki_match_distance_km": distance,
            "wikidata_id": (page.get("pageprops") or {}).get("wikibase_item", ""),
        }
        return True

    # ------------------------------------------------------- pass 1: names
    LOGGER.info("Resolving %d cities to Wikipedia articles", len(frame))
    plain_names = list(dict.fromkeys(str(frame.at[i, "city"]) for i in range(len(frame))))
    pages = lookup_titles(plain_names)
    for index in range(len(frame)):
        page = pages.get(str(frame.at[index, "city"]))
        if page is not None:
            claim(index, page)
    LOGGER.info("Pass 1 matched %d/%d cities", len(resolved), len(frame))

    # --------------------------------------------------- pass 2: variants
    unresolved = [i for i in range(len(frame)) if i not in resolved]
    if unresolved:
        LOGGER.info("Pass 2: disambiguated variants for %d cities", len(unresolved))
        variants_by_city: Dict[int, List[str]] = {
            index: title_variants(
                str(frame.at[index, "city"]),
                str(frame.at[index, "city_ascii"]) if "city_ascii" in frame.columns else "",
                str(frame.at[index, "country"]),
                str(frame.at[index, "admin1"]) if "admin1" in frame.columns else "",
            )[1:]  # the plain name was already tried in pass 1
            for index in unresolved
        }
        all_variants = list(dict.fromkeys(v for group in variants_by_city.values() for v in group))
        pages = lookup_titles(all_variants)
        for index in unresolved:
            for variant in variants_by_city[index]:
                page = pages.get(variant)
                if page is not None and claim(index, page):
                    break
        LOGGER.info("After pass 2: %d/%d cities matched", len(resolved), len(frame))

    # ----------------------------------------------------- pass 3: search
    unresolved = [i for i in range(len(frame)) if i not in resolved]
    if use_search_fallback and unresolved:
        LOGGER.info("Pass 3: search fallback for %d cities", len(unresolved))

        def run_search(index: int) -> Tuple[int, Optional[dict]]:
            city = str(frame.at[index, "city"])
            country = str(frame.at[index, "country"])
            with requests.Session() as session:
                candidates = _search_candidates(
                    f"{city} {country}",
                    cache=cache,
                    user_agent=user_agent,
                    timeout=timeout,
                    session=session,
                )
                if not candidates:
                    return index, None
                _, candidate_pages = _fetch_batch(
                    candidates, cache=cache, user_agent=user_agent, timeout=timeout, session=session
                )
            best_page, best_distance = None, float("inf")
            for title in candidates:
                page = candidate_pages.get(title)
                if page is None:
                    continue
                ok, distance = _accept(
                    float(frame.at[index, "latitude"]),
                    float(frame.at[index, "longitude"]),
                    page,
                    max_distance_km,
                )
                if ok and distance is not None and distance < best_distance:
                    best_page, best_distance = page, distance
            return index, best_page

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for done, (index, page) in enumerate(pool.map(run_search, unresolved), start=1):
                if page is not None:
                    claim(index, page)
                if done % 250 == 0:
                    LOGGER.info("  pass 3: %d/%d searched", done, len(unresolved))

    # ------------------------------------------------------------- assemble
    frame = frame.copy()
    frame["wiki_title"] = [str(resolved.get(i, {}).get("wiki_title", "")) for i in range(len(frame))]
    frame["wiki_match_distance_km"] = [
        float(resolved.get(i, {}).get("wiki_match_distance_km", np.nan))  # type: ignore[arg-type]
        if i in resolved
        else np.nan
        for i in range(len(frame))
    ]
    frame["wikidata_id"] = [str(resolved.get(i, {}).get("wikidata_id", "")) for i in range(len(frame))]
    frame["wiki_resolved"] = frame["wiki_title"].str.len() > 0

    # Two GeoNames entries can map to one article (a city and its suburb, or
    # duplicate gazetteer records). The article is our join key downstream, so
    # keep the closest, most populous claimant.
    duplicated = frame["wiki_resolved"] & frame.duplicated(subset=["wiki_title"], keep=False)
    if duplicated.any():
        LOGGER.info("Deduplicating %d cities sharing a Wikipedia article", int(duplicated.sum()))
        frame = frame.sort_values(
            ["wiki_title", "wiki_match_distance_km", "population"], ascending=[True, True, False]
        )
        keep = ~(frame["wiki_resolved"] & frame.duplicated(subset=["wiki_title"], keep="first"))
        frame.loc[~keep, ["wiki_title", "wikidata_id"]] = ""
        frame.loc[~keep, "wiki_match_distance_km"] = np.nan
        frame["wiki_resolved"] = frame["wiki_title"].str.len() > 0
        frame = frame.sort_index()

    LOGGER.info("Resolved %d/%d cities to Wikipedia", int(frame["wiki_resolved"].sum()), len(frame))
    return frame.reset_index(drop=True)
