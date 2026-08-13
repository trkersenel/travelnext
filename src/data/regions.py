"""Static ISO-3166 alpha-2 to continent / sub-region mapping.

Based on the UN M49 standard geographic classification, which is published by
the United Nations Statistics Division and is not subject to copyright. It is
bundled as code rather than fetched so the pipeline has one less network
dependency and behaves identically offline.
"""

from __future__ import annotations

from typing import Dict, Tuple

# region name -> (continent, iso alpha-2 codes)
_REGION_DEFINITIONS: Tuple[Tuple[str, str, str], ...] = (
    ("Northern Europe", "Europe", "DK FI IS NO SE EE LV LT IE GB AX FO GG IM JE SJ"),
    ("Western Europe", "Europe", "AT BE FR DE LI LU MC NL CH"),
    ("Southern Europe", "Europe", "AL AD BA HR GI GR VA IT MT ME MK PT SM RS SI ES"),
    ("Eastern Europe", "Europe", "BY BG CZ HU MD PL RO RU SK UA"),
    ("Northern Africa", "Africa", "DZ EG LY MA SD TN EH"),
    ("Western Africa", "Africa", "BJ BF CV CI GM GH GN GW LR ML MR NE NG SH SN SL TG"),
    ("Middle Africa", "Africa", "AO CM CF TD CG CD GQ GA ST"),
    ("Eastern Africa", "Africa", "BI KM DJ ER ET KE MG MW MU YT MZ RE RW SC SO SS TZ UG ZM ZW"),
    ("Southern Africa", "Africa", "BW SZ LS NA ZA"),
    ("Northern America", "North America", "BM CA GL PM US"),
    ("Central America", "North America", "BZ CR SV GT HN MX NI PA"),
    ("Caribbean", "North America", "AI AG AW BS BB VG KY CU CW DM DO GD GP HT JM MQ MS PR BL KN LC MF VC SX TT TC VI"),
    ("South America", "South America", "AR BO BR CL CO EC FK GF GY PY PE SR UY VE"),
    ("Eastern Asia", "Asia", "CN HK JP KP KR MO MN TW"),
    ("South-eastern Asia", "Asia", "BN KH ID LA MY MM PH SG TH TL VN"),
    ("Southern Asia", "Asia", "AF BD BT IN IR MV NP PK LK"),
    ("Central Asia", "Asia", "KZ KG TJ TM UZ"),
    ("Western Asia", "Asia", "AM AZ BH CY GE IQ IL JO KW LB OM QA SA PS SY TR AE YE"),
    ("Australia and New Zealand", "Oceania", "AU NZ NF CX CC"),
    ("Melanesia", "Oceania", "FJ NC PG SB VU"),
    ("Micronesia", "Oceania", "GU KI MH FM NR MP PW"),
    ("Polynesia", "Oceania", "AS CK PF NU PN WS TK TO TV WF"),
    ("Antarctica", "Antarctica", "AQ BV GS HM TF"),
)

_COUNTRY_TO_REGION: Dict[str, Tuple[str, str]] = {
    code: (continent, region)
    for region, continent, codes in _REGION_DEFINITIONS
    for code in codes.split()
}

UNKNOWN = ("Unknown", "Unknown")


def lookup(country_code: str) -> Tuple[str, str]:
    """Return ``(continent, region)`` for an ISO alpha-2 code.

    Unrecognised or malformed codes map to ``("Unknown", "Unknown")`` rather
    than raising, so an unexpected country never breaks the pipeline.
    """
    if not country_code:
        return UNKNOWN
    return _COUNTRY_TO_REGION.get(country_code.strip().upper(), UNKNOWN)


def continent_of(country_code: str) -> str:
    """Return the continent name for an ISO alpha-2 code."""
    return lookup(country_code)[0]


def region_of(country_code: str) -> str:
    """Return the UN M49 sub-region name for an ISO alpha-2 code."""
    return lookup(country_code)[1]


def known_country_codes() -> frozenset[str]:
    """All ISO alpha-2 codes covered by the mapping."""
    return frozenset(_COUNTRY_TO_REGION)
