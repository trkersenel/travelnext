"""Geographic helpers (no external geo dependencies, no paid map APIs)."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """Great-circle distance in kilometres. Broadcasts like NumPy arrays."""
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    inner = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def pairwise_distance_matrix(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Full N x N great-circle distance matrix in kilometres.

    At the catalog sizes TravelNext uses (under ~2000 destinations) this is a
    few tens of megabytes in float32 and comfortably fits the 8GB target.
    """
    lat = np.asarray(latitudes, dtype="float64")
    lon = np.asarray(longitudes, dtype="float64")
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :]).astype("float32")


def distance_decay(distance_km: np.ndarray, scale_km: float = 700.0) -> np.ndarray:
    """Map distance to a 0-1 proximity score with exponential decay.

    ``scale_km`` is the distance at which the score falls to 1/e. Used both to
    generate realistic trip chaining and as a ranking feature.
    """
    return np.exp(-np.asarray(distance_km, dtype="float64") / max(scale_km, 1e-6))
