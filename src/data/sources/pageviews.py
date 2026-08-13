"""Wikipedia pageview counts, used as the destination popularity proxy.

Source: the Wikimedia REST Pageviews API. It is free, requires no key, and the
underlying counts are released under CC0.

WHAT THIS IS AND IS NOT
-----------------------
Pageviews measure *online interest in an article*, not visitor arrivals. A city
with a large diaspora, a famous football club or recent news coverage will rank
higher than its tourist numbers justify. We use it because no free, global,
per-city tourist-arrival dataset with a permissive licence exists. It is
labelled ``popularity_proxy`` everywhere in the codebase and the limitation is
documented in the README rather than hidden.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import statistics
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote

import pandas as pd
import requests

from src.utils.http import HttpCache, RateLimiter, is_failed, is_missing, request_json
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

# Measured empirically: 10 req/s draws HTTP 429 from this endpoint even with a
# descriptive User-Agent. 4 req/s runs clean and still covers ~5000 articles in
# roughly 20 minutes, which happens once and is then cached forever.
_RATE_LIMITER = RateLimiter(requests_per_second=4.0)

_BASE = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{title}/monthly/{start}/{end}"
)


def _window(months: int) -> tuple[str, str]:
    """Return the (start, end) API timestamps for a trailing month window.

    The window covers exactly ``months`` **complete** calendar months, ending
    with the last day of the previous month. Both boundaries matter: ending on
    the *first* of a month would append a one-day bucket that looks like a
    catastrophic traffic collapse and drags every average down.
    """
    today = date.today()
    # Last day of the previous month.
    end = date(today.year, today.month, 1) - timedelta(days=1)

    year, month = end.year, end.month - (months - 1)
    while month <= 0:
        month += 12
        year -= 1
    start = date(year, month, 1)
    return start.strftime("%Y%m%d00"), end.strftime("%Y%m%d00")


def _fetch_one(
    title: str,
    start: str,
    end: str,
    *,
    cache: HttpCache,
    user_agent: str,
    timeout: int,
    retries: int,
    session: requests.Session,
) -> Dict[str, float]:
    """Fetch total and monthly-mean views for a single article title."""
    url = _BASE.format(title=quote(title.replace(" ", "_"), safe=""), start=start, end=end)
    payload = request_json(
        url,
        cache=cache,
        cache_key=f"pv:{title}:{start}:{end}",
        user_agent=user_agent,
        timeout=timeout,
        retries=retries,
        backoff_s=3.0,
        session=session,
        rate_limiter=_RATE_LIMITER,
    )
    if is_failed(payload):
        # Could not ask (rate limited / timeout). Distinct from "no such
        # article": treating it as zero views would quietly delete a real city
        # from the catalog, since the catalog is ranked by this number.
        return {
            "wiki_title": title,
            "pageviews_total": 0.0,
            "pageviews_median": 0.0,
            "pageviews_max": 0.0,
            "pageviews_months": 0,
            "pageviews_available": False,
        }
    if not payload or is_missing(payload):
        return {
            "wiki_title": title,
            "pageviews_total": 0.0,
            "pageviews_median": 0.0,
            "pageviews_max": 0.0,
            "pageviews_months": 0,
            "pageviews_available": True,
        }

    views = [float(item.get("views", 0)) for item in payload.get("items", [])]
    if not views:
        return {
            "wiki_title": title,
            "pageviews_total": 0.0,
            "pageviews_median": 0.0,
            "pageviews_max": 0.0,
            "pageviews_months": 0,
            "pageviews_available": True,
        }
    return {
        "wiki_title": title,
        "pageviews_total": float(sum(views)),
        # The median is the headline statistic: automated traffic arrives as
        # short, enormous bursts, and a mean lets one bot month dominate two
        # years of real interest. See ``fetch_pageviews`` for the diagnostic.
        "pageviews_median": float(statistics.median(views)),
        "pageviews_max": float(max(views)),
        "pageviews_months": len(views),
        "pageviews_available": True,
    }


def fetch_pageviews(
    titles: Iterable[str],
    cache_dir,
    *,
    months: int = 24,
    user_agent: str = "TravelNext/0.1",
    timeout: int = 30,
    retries: int = 2,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Fetch trailing-window pageviews for many article titles concurrently.

    Returns a frame with ``wiki_title``, ``pageviews_total``,
    ``pageviews_months`` and ``pageviews_monthly_mean``. Articles that could not
    be resolved get zero views rather than being dropped, so the caller keeps
    full control over filtering.
    """
    cache = HttpCache(cache_dir, "pageviews")
    start, end = _window(months)
    unique_titles: List[str] = list(dict.fromkeys(t for t in titles if t))
    LOGGER.info("Fetching pageviews for %d articles (%s..%s)", len(unique_titles), start, end)

    def run_pass(titles_to_fetch: List[str], workers: int) -> List[Dict[str, float]]:
        collected: List[Dict[str, float]] = []
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _fetch_one,
                        title,
                        start,
                        end,
                        cache=cache,
                        user_agent=user_agent,
                        timeout=timeout,
                        retries=retries,
                        session=session,
                    )
                    for title in titles_to_fetch
                ]
                for index, future in enumerate(futures, start=1):
                    collected.append(future.result())
                    if index % 500 == 0:
                        LOGGER.info(
                            "  pageviews progress: %d/%d", index, len(titles_to_fetch)
                        )
        return collected

    results = run_pass(unique_titles, max_workers)

    # Anything that failed for transport reasons is retried single-threaded, so
    # a burst of throttling does not permanently zero out real destinations.
    by_title = {row["wiki_title"]: row for row in results}
    failed = [t for t, row in by_title.items() if not row.get("pageviews_available", True)]
    if failed:
        LOGGER.warning("Retrying %d rate-limited pageview requests serially", len(failed))
        for row in run_pass(failed, 1):
            by_title[row["wiki_title"]] = row
        still_failing = [t for t, row in by_title.items() if not row.get("pageviews_available", True)]
        if still_failing:
            LOGGER.error(
                "%d titles still unresolved after retry; they are flagged "
                "pageviews_available=False rather than counted as zero",
                len(still_failing),
            )

    frame = pd.DataFrame(list(by_title.values()))
    frame["pageviews_monthly_mean"] = frame.apply(
        lambda row: row["pageviews_total"] / row["pageviews_months"]
        if row["pageviews_months"] > 0
        else 0.0,
        axis=1,
    )
    # Data-quality diagnostic. Organic traffic is fairly stable month to month,
    # so mean ~= median. A large ratio means short bursts of automated traffic
    # that the API's `agent=user` filter did not catch -- observed for real:
    # two unrelated cities showed near-identical million-view spikes.
    frame["pageviews_anomaly_ratio"] = frame.apply(
        lambda row: row["pageviews_monthly_mean"] / row["pageviews_median"]
        if row["pageviews_median"] > 0
        else 1.0,
        axis=1,
    )
    resolved = int((frame["pageviews_total"] > 0).sum())
    suspicious = int((frame["pageviews_anomaly_ratio"] > 1.5).sum())
    LOGGER.info("Pageviews resolved for %d/%d articles", resolved, len(frame))
    LOGGER.info(
        "%d articles show burst-like traffic (mean/median > 1.5); the median is "
        "used as the popularity statistic to limit their influence",
        suspicious,
    )
    return frame
