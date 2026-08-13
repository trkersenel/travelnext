"""Polite, cached HTTP access to the free public APIs TravelNext depends on.

Design rules enforced here:

* every response is cached on disk, so a re-run costs zero requests;
* failures degrade to ``None`` rather than raising, so one bad city cannot
  abort a multi-hour ingestion run;
* a descriptive User-Agent is always sent, as the Wikimedia and Overpass
  usage policies require.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import requests

from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


class RateLimiter:
    """A thread-safe minimum-interval limiter shared across worker threads.

    The Wikipedia API rejects bursty anonymous traffic with HTTP 429. Capping
    the *aggregate* request rate keeps us inside the published etiquette even
    when several threads are resolving cities at once.
    """

    def __init__(self, requests_per_second: float) -> None:
        self._min_interval = 1.0 / max(requests_per_second, 1e-6)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        """Block until this thread is allowed to issue its request."""
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait <= 0:
                self._next_allowed = now + self._min_interval
                wait = 0.0
            else:
                self._next_allowed += self._min_interval
        if wait > 0:
            time.sleep(wait)


class HttpCache:
    """A tiny JSON-on-disk cache keyed by a hash of the request."""

    def __init__(self, cache_dir: Path, namespace: str) -> None:
        self.dir = Path(cache_dir) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.dir / f"{digest}.json"

    def get(self, key: str) -> Optional[Any]:
        """Return the cached payload for ``key``, or ``None`` on a miss."""
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            # A truncated cache file should not be fatal; treat it as a miss.
            LOGGER.warning("Discarding unreadable cache entry: %s", path.name)
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``, writing atomically."""
        path = self._key_path(key)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle)
        tmp.replace(path)


def request_json(
    url: str,
    *,
    cache: Optional[HttpCache] = None,
    cache_key: Optional[str] = None,
    method: str = "GET",
    params: Optional[Mapping[str, Any]] = None,
    data: Optional[Mapping[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    user_agent: str = "TravelNext/0.1",
    timeout: int = 60,
    retries: int = 2,
    backoff_s: float = 2.0,
    session: Optional[requests.Session] = None,
    rate_limiter: Optional[RateLimiter] = None,
) -> Optional[Any]:
    """Fetch and decode a JSON endpoint, with caching and bounded retries.

    Returns ``None`` when the resource is genuinely unavailable (HTTP 404) or
    when every retry failed. Callers are expected to handle missing data.
    """
    key = cache_key or f"{method}:{url}:{json.dumps(params or {}, sort_keys=True)}:{json.dumps(data or {}, sort_keys=True)}"
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    request_headers = {"User-Agent": user_agent, "Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    http = session or requests

    last_error: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            if rate_limiter is not None:
                rate_limiter.acquire()
            response = http.request(
                method,
                url,
                params=params,
                data=data,
                headers=request_headers,
                timeout=timeout,
            )
            if response.status_code == 404:
                # Genuinely absent (e.g. no Wikipedia article). Cache the miss
                # so we do not ask again on the next run.
                if cache is not None:
                    cache.set(key, {"__missing__": True})
                return None
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                # Server-side pressure: back off progressively.
                time.sleep(backoff_s * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json()
            if cache is not None:
                cache.set(key, payload)
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_error = type(exc).__name__
            if attempt < retries:
                time.sleep(backoff_s * (attempt + 1))

    LOGGER.warning("Request failed after %d attempts (%s): %s", retries + 1, last_error, url)
    # Deliberately NOT cached: a transient failure must not be baked in as a
    # permanent answer. Returning a distinct marker (rather than None) lets
    # callers tell "this resource does not exist" apart from "we could not ask",
    # which otherwise silently turns rate limiting into missing data.
    return {"__failed__": True}


def is_missing(payload: Any) -> bool:
    """True when ``payload`` is a cached 'resource does not exist' marker."""
    return isinstance(payload, dict) and payload.get("__missing__") is True


def is_failed(payload: Any) -> bool:
    """True when the request could not be completed (timeout, 429, 5xx)."""
    return isinstance(payload, dict) and payload.get("__failed__") is True
