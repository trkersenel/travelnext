"""City catalog from the GeoNames gazetteer.

Source: https://download.geonames.org/export/dump/ — the ``cities15000`` dump
(every settlement above 15,000 inhabitants) and ``countryInfo.txt``. Both are
free downloads released under CC BY 4.0.

GeoNames replaced an earlier Wikidata-SPARQL implementation here: the public
SPARQL endpoint returned HTTP 504 for any query covering large cities, which
silently truncated the catalog to small towns. A static, versioned download is
both faster and far more reliable, and it works offline once cached.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

CITIES_URL = "https://download.geonames.org/export/dump/cities15000.zip"
COUNTRY_INFO_URL = "https://download.geonames.org/export/dump/countryInfo.txt"
ADMIN1_URL = "https://download.geonames.org/export/dump/admin1CodesASCII.txt"

# Column layout of the GeoNames dump (documented in its readme.txt).
_COLUMNS = [
    "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
    "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
    "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
    "dem", "timezone", "modification_date",
]

# Capitals and administrative seats first: used to break ties between two
# same-named cities, and as a mild notability signal.
_CAPITAL_CODES = frozenset({"PPLC", "PPLA"})


def _download(url: str, destination: Path, *, user_agent: str, timeout: int) -> Path:
    """Download ``url`` to ``destination`` unless it already exists."""
    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info("Using cached download: %s", destination.name)
        return destination
    LOGGER.info("Downloading %s", url)
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return destination


def load_country_names(
    raw_dir, *, user_agent: str = "TravelNext/0.1", timeout: int = 60
) -> Dict[str, str]:
    """Return a mapping of ISO alpha-2 country code to English country name."""
    path = _download(
        COUNTRY_INFO_URL, Path(raw_dir) / "countryInfo.txt", user_agent=user_agent, timeout=timeout
    )
    names: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) > 4:
                names[fields[0].strip().upper()] = fields[4].strip()
    LOGGER.info("Loaded %d country names", len(names))
    return names


def load_admin1_names(
    raw_dir, *, user_agent: str = "TravelNext/0.1", timeout: int = 60
) -> Dict[str, str]:
    """Return a mapping of ``"<country>.<admin1>"`` to first-level region name.

    Used to build Wikipedia title variants such as "Springfield, Illinois",
    which is how English Wikipedia disambiguates most non-unique city names.
    """
    path = _download(
        ADMIN1_URL, Path(raw_dir) / "admin1CodesASCII.txt", user_agent=user_agent, timeout=timeout
    )
    names: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                names[fields[0].strip()] = fields[1].strip()
    LOGGER.info("Loaded %d admin1 region names", len(names))
    return names


def load_cities(
    raw_dir,
    *,
    min_population: int = 100_000,
    user_agent: str = "TravelNext/0.1",
    timeout: int = 120,
) -> pd.DataFrame:
    """Load every populated place above ``min_population`` from GeoNames.

    Returns columns: ``geonameid, city, city_ascii, country_code, country,
    latitude, longitude, population, feature_code, is_capital``.
    """
    archive = _download(
        CITIES_URL, Path(raw_dir) / "cities15000.zip", user_agent=user_agent, timeout=timeout
    )
    with zipfile.ZipFile(archive) as zf:
        with zf.open("cities15000.txt") as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8")
            reader = csv.reader(text, delimiter="\t", quoting=csv.QUOTE_NONE)
            records = []
            for fields in reader:
                if len(fields) < len(_COLUMNS):
                    continue
                try:
                    population = int(fields[14])
                except ValueError:
                    continue
                if population < min_population or fields[6] != "P":
                    continue
                records.append(
                    {
                        "geonameid": int(fields[0]),
                        "city": fields[1].strip(),
                        "city_ascii": fields[2].strip(),
                        "country_code": fields[8].strip().upper(),
                        "latitude": float(fields[4]),
                        "longitude": float(fields[5]),
                        "population": population,
                        "feature_code": fields[7].strip(),
                        "admin1_code": fields[10].strip(),
                    }
                )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("GeoNames dump produced no cities; the download may be corrupt.")

    frame["is_capital"] = frame["feature_code"].isin(_CAPITAL_CODES)
    country_names = load_country_names(raw_dir, user_agent=user_agent, timeout=timeout)
    frame["country"] = frame["country_code"].map(country_names).fillna(frame["country_code"])

    admin1_names = load_admin1_names(raw_dir, user_agent=user_agent, timeout=timeout)
    frame["admin1"] = (
        (frame["country_code"] + "." + frame["admin1_code"].astype(str))
        .map(admin1_names)
        .fillna("")
    )
    frame = frame[frame["country_code"].str.len() == 2].reset_index(drop=True)

    LOGGER.info(
        "GeoNames: %d cities >= %d inhabitants across %d countries",
        len(frame),
        min_population,
        frame["country_code"].nunique(),
    )
    return frame
