"""Configuration loading for TravelNext.

The whole pipeline is driven by ``configs/config.yaml``. Nothing in this module
requires network access, credentials or environment variables: the project is
designed to run offline once the data snapshot exists.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

# Repository root = parent of the ``src`` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class Config:
    """Dict-backed configuration with dotted-path lookup.

    Example
    -------
    >>> cfg = Config({"a": {"b": 1}})
    >>> cfg.get("a.b")
    1
    >>> cfg.get("a.missing", default=7)
    7
    """

    def __init__(self, data: Dict[str, Any], root: Path = PROJECT_ROOT) -> None:
        self._data = data
        self.root = root

    def get(self, path: str, default: Any = None) -> Any:
        """Return the value at a dotted ``path`` or ``default`` if absent."""
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        """Return the value at ``path`` or raise if it is missing."""
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise KeyError(f"Missing required config key: {path!r}")
        return value

    def path(self, path_key: str) -> Path:
        """Resolve a ``paths.*`` config entry to an absolute directory path.

        The directory is created if it does not exist, so callers can write to
        it without additional ceremony.
        """
        raw = self.require(f"paths.{path_key}")
        resolved = self.root / raw
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    @property
    def seed(self) -> int:
        """Global random seed used everywhere for reproducibility."""
        return int(self.get("project.random_seed", 42))

    def as_dict(self) -> Dict[str, Any]:
        """Return a copy of the raw configuration mapping."""
        return dict(self._data)


@lru_cache(maxsize=4)
def load_config(config_path: str | os.PathLike[str] | None = None) -> Config:
    """Load and cache the YAML configuration.

    Parameters
    ----------
    config_path:
        Optional override. Defaults to ``configs/config.yaml``, or the value of
        the ``TRAVELNEXT_CONFIG`` environment variable when set.
    """
    if config_path is None:
        config_path = os.environ.get("TRAVELNEXT_CONFIG", DEFAULT_CONFIG_PATH)
    resolved = Path(config_path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return Config(data)
