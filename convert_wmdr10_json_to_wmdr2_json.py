#!/usr/bin/env python3
"""Convert simplified WMDR 1.0 JSON records to WMDR2 JSON.

The XML -> WMDR1 JSON conversion is deliberately out of scope here.  This
module consumes the information-preserving WMDR1 JSON representation produced
by ``convert_wmdr10_xml_to_wmdr10_json.py`` and performs the semantic mapping
to the current WMDR2 model.

The WMDR2 representation follows the current official ``wmo-im/wmdr2`` schema
terminology where applicable:

* Facility ``properties.observations`` contain Observation objects.
* Observation ``configurations`` contain Configuration objects.
* Controlled vocabulary values use OGC API Records Concept objects and retain
  the complete source URI as ``Concept.id``.
* Observation programme affiliations use ``programAffiliation``, optional
  ``reportingStatus`` and optional ``dates``.
* Configuration uses ``instrumentSerialNumber`` and the structured
  ``verticalDistance`` object.

The development model still keeps reusable record-local contact, instrument and
schedule catalogues.  In particular, ``Configuration.instrument`` is a
record-local reference to ``properties.instruments[].id``; the full instrument
catalogue object is not duplicated in every Configuration.

The converter never invents missing observational metadata merely to satisfy a
schema.  Where the source has no required value, the resulting record remains
explicitly incomplete and validation can report the source deficiency.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

try:
    import yaml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    yaml = None

VERSION = "0.3.3-dev"
ANNOUNCE_WRITES = True
DEFAULT_PATTERN = "*.json"
OUTPUT_SUFFIX = ".json"
CANONICAL_SCHEDULE_START_DATE = "0001-01-01"
WMDR2_CORE_CONF = "http://wigos.wmo.int/spec/wmdr/2/conf/core"
SOURCE_TEMPORAL_KEYS = {"beginPosition", "endPosition", "validFrom", "validTo"}
_EXPLICIT_NULL = object()

EMPTY_DISCOVERY_POLICY: Dict[str, Dict[str, List[str]]] = {
    "facility": {"keywords": [], "links": []},
    "observation": {"keywords": [], "links": []},
    "configuration": {"keywords": [], "links": []},
}
DEFAULT_DISCOVERY_POLICY: Dict[str, Dict[str, List[str]]] = {
    "facility": {"keywords": ["identifier", "name"], "links": ["onlineResource"]},
    "observation": {"keywords": [], "links": []},
    "configuration": {"keywords": [], "links": []},
}
DISCOVERY_POLICY: Dict[str, Dict[str, List[str]]] = copy.deepcopy(DEFAULT_DISCOVERY_POLICY)
CODE_LIST_LABELS: Dict[str, Dict[str, str]] = {}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strip_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _non_empty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if _non_empty(value) or isinstance(value, bool):
            return value
    return None


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _uniq_scalars(items: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen: set[str] = set()
    for item in items:
        if item in (None, "", [], {}):
            continue
        key = _stable_json(item) if isinstance(item, (dict, list)) else str(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _uniq_dicts(items: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _clean_none(dict(item))
        if not isinstance(cleaned, dict) or not cleaned:
            continue
        marker = _stable_json(cleaned)
        if marker not in seen:
            seen.add(marker)
            out.append(cast(Dict[str, Any], cleaned))
    return out


def _remove_prefix(value: str, prefix: str) -> str:
    """Return value without prefix, compatible with Python < 3.9."""
    return value[len(prefix):] if value.startswith(prefix) else value


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-") or "value"


def _sanitize_id(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^A-Za-z0-9._:/#-]+", "-", text)
    return text.strip("-") or "record"


def _clean_none(value: Any, *, _path: Tuple[str, ...] = ()) -> Any:
    """Remove empty mapping members while preserving meaningful list nulls.

    JSON null can be meaningful inside positional arrays.  Empty method lists
    in temporal geometry are also positional and therefore retained.
    """
    def preserve_empty_list(path: Tuple[str, ...]) -> bool:
        return len(path) >= 2 and path[-2:] in {
            ("temporalGeometry", "methods"),
            ("methods", "methods"),
        }

    if value is _EXPLICIT_NULL:
        return _EXPLICIT_NULL
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            if child is _EXPLICIT_NULL:
                out[key] = _EXPLICIT_NULL
                continue
            cleaned = _clean_none(child, _path=_path + (str(key),))
            if cleaned not in (None, "", [], {}):
                out[key] = cleaned
        return out
    if isinstance(value, list):
        cleaned_list = [_clean_none(child, _path=_path) for child in value]
        return [
            child
            for child in cleaned_list
            if child not in ("", {}) and (child != [] or preserve_empty_list(_path))
        ]
    return value


def _restore_explicit_nulls(value: Any) -> Any:
    if value is _EXPLICIT_NULL:
        return None
    if isinstance(value, dict):
        return {key: _restore_explicit_nulls(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_restore_explicit_nulls(child) for child in value]
    return value


# ---------------------------------------------------------------------------
# Time and identifiers
# ---------------------------------------------------------------------------


def _normalize_time_resolution(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    lower = text.lower()
    aliases = {
        "day": "P1D", "daily": "P1D", "d": "P1D",
        "hour": "PT1H", "hourly": "PT1H", "h": "PT1H",
        "minute": "PT1M", "minutes": "PT1M", "min": "PT1M",
        "second": "PT1S", "seconds": "PT1S", "sec": "PT1S",
    }
    if lower in aliases:
        return aliases[lower]
    for pattern, template in (
        (r"(\d+)\s*(d|day|days)", "P{}D"),
        (r"(\d+)\s*(h|hour|hours)", "PT{}H"),
        (r"(\d+)\s*(m|min|minute|minutes)", "PT{}M"),
        (r"(\d+)\s*(s|sec|second|seconds)", "PT{}S"),
    ):
        match = re.fullmatch(pattern, lower)
        if match:
            return template.format(match.group(1))
    return text


def _normalize_date_value(value: Any) -> Optional[str]:
    if value in (None, "", "None"):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text == "..":
        return ".."
    match = re.match(r"^(\d{4})(\d{2})(\d{2})(?:_|$)", text)
    if match:
        return "-".join(match.groups())
    if re.fullmatch(r"\d{8}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}Z", text):
        return text[:-1]
    if re.match(r"^\d{4}-\d{2}-\d{2}T", text):
        return text[:10]
    return text


def _normalize_record_datetime(value: Any) -> Optional[str]:
    if value in (None, "", "None"):
        return None
    text = str(value).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}T", text):
        return text if text.endswith("Z") else f"{text}Z"
    date = _normalize_date_value(text)
    if date and date != ".." and re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return f"{date}T00:00:00Z"
    return None


def _time_interval(start: Any, end: Any = None, *, resolution: Any = None) -> Optional[Dict[str, Any]]:
    first = _normalize_date_value(start)
    last = _normalize_date_value(end) or ".."
    if first is None and last == "..":
        return None
    result: Dict[str, Any] = {"interval": [first or "..", last]}
    normalized_resolution = _normalize_time_resolution(resolution)
    if normalized_resolution:
        result["resolution"] = normalized_resolution
    return result


def _dates(start: Any, end: Any = None) -> Optional[List[str]]:
    """Return official WMDR2 ``dates`` without inventing a validity period."""
    first = _normalize_date_value(start)
    last = _normalize_date_value(end)
    if first is None and last is None:
        return None
    if first is None:
        return ["..", last or ".."]
    if last is None:
        return [first, ".."]
    return [first, last]


def _extract_interval(obj: Mapping[str, Any]) -> Tuple[Any, Any]:
    time_obj = obj.get("time")
    if isinstance(time_obj, Mapping):
        interval = time_obj.get("interval")
        if isinstance(interval, list) and interval:
            return (
                interval[0] if len(interval) > 0 else None,
                interval[1] if len(interval) > 1 else None,
            )
        return _first_non_empty(time_obj.get("date"), time_obj.get("timestamp")), None
    dates_obj = obj.get("dates")
    if isinstance(dates_obj, list) and dates_obj:
        return (
            dates_obj[0] if len(dates_obj) > 0 else None,
            dates_obj[1] if len(dates_obj) > 1 else None,
        )
    return (
        _first_non_empty(
            obj.get("validFrom"), obj.get("date"), obj.get("beginPosition"),
            obj.get("begin"), obj.get("from"), obj.get("start"), obj.get("dateEstablished"),
        ),
        _first_non_empty(
            obj.get("validTo"), obj.get("endPosition"), obj.get("end"),
            obj.get("stop"), obj.get("dateClosed"),
        ),
    )


def _entry_date(item: Any, fallback: str = "..") -> str:
    if isinstance(item, Mapping):
        start, _ = _extract_interval(item)
        return _normalize_date_value(start) or fallback
    return fallback


WSI_PATTERN = r"(0|1|2|3)-([1-9]\d*)-([0-9]+)-([A-Za-z0-9._-]+)"


def _is_valid_wsi(value: Any) -> bool:
    return bool(re.fullmatch(WSI_PATTERN, str(value or "").strip()))


def _normalize_single_facility_wsi(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    while True:
        lower = text.lower()
        for prefix in ("wsi:", "wigos:", "facility:", "record:", "station:", "id:"):
            if lower.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        else:
            break
    if "/" in text and re.search(r"\d+-\d+-\d+-[A-Za-z0-9._-]+$", text):
        text = text.rstrip("/").rsplit("/", 1)[-1]
    return text


def _facility_wsi_values(value: Any) -> List[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, Mapping):
        values: List[str] = []
        for key in ("identifier", "wigosStationIdentifier", "wigosIdentifier", "wsi", "id", "value", "text", "#text"):
            values.extend(_facility_wsi_values(value.get(key)))
        return cast(List[str], _uniq_scalars(values))
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_facility_wsi_values(item))
        return cast(List[str], _uniq_scalars(values))
    out: List[str] = []
    for candidate in re.split(r"\s*[,;]\s*", str(value).strip()):
        normalized = _normalize_single_facility_wsi(candidate)
        if _is_valid_wsi(normalized):
            out.append(normalized)
    return cast(List[str], _uniq_scalars(out))


def _normalize_facility_wsi(value: Any) -> str:
    values = _facility_wsi_values(value)
    return values[0] if values else _normalize_single_facility_wsi(value)


# ---------------------------------------------------------------------------
# Controlled values / OGC Concept
# ---------------------------------------------------------------------------


def _is_unknown_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if text.startswith(("http://", "https://")):
        text = text.rstrip("/#").rsplit("/", 1)[-1]
    match = re.fullmatch(r"\(([^()]+)\)", text)
    if match:
        text = match.group(1)
    return text.lower() in {"unknown", "none", "null", "nil"}


def _normalize_code_value(value: Any) -> Any:
    """Extract a controlled value while retaining an absolute URI unchanged."""
    if isinstance(value, Mapping):
        if "id" in value and isinstance(value.get("id"), (str, int)):
            value = value.get("id")
        else:
            value = _first_non_empty(
                value.get("href"), value.get("url"), value.get("value"),
                value.get("#text"), value.get("text"),
            )
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, str):
        return value
    text = value.strip().strip("<>")
    if not text:
        return None
    return "unknown" if _is_unknown_token(text) else text


def _last_segment(value: Any) -> Optional[str]:
    normalized = _normalize_code_value(value)
    if not isinstance(normalized, str) or not normalized:
        return None
    raw = normalized.rstrip("/#")
    if "/" in raw:
        return raw.rsplit("/", 1)[-1]
    if "#" in raw:
        return raw.rsplit("#", 1)[-1]
    return raw


def _explicit_nil(value: Any) -> bool:
    if isinstance(value, Mapping):
        if _non_empty(value.get("nilReason")) or _non_empty(value.get("@nilReason")):
            return True
        if value.get("nil") is True or value.get("@nil") is True:
            return True
        # stage 1 normalizes the ordinary textual nil variants to "unknown"
        nested = _first_non_empty(value.get("value"), value.get("#text"), value.get("text"))
        return _is_unknown_token(nested)
    return _is_unknown_token(value)


def _unwrap_controlled(value: Any, *nested_keys: str) -> Any:
    current = value
    if isinstance(current, Mapping) and nested_keys:
        nested = _first_non_empty(
            *(current.get(key) for key in nested_keys),
            current.get("href"), current.get("url"), current.get("value"),
            current.get("#text"), current.get("text"),
        )
        if nested not in (None, "", [], {}):
            current = nested
    return current


def _concept(
    value: Any,
    *nested_keys: str,
    allow_null: bool = False,
    base_uri: Optional[str] = None,
) -> Any:
    """Return an OGC Concept object, retaining a complete URI when supplied.

    ``base_uri`` is used only when the source contains a compact notation and
    the codelist is unambiguous from the target property (for example WMO
    ``unit`` or ``TerritoryName``).  Existing absolute identifiers are never
    rewritten.
    """
    if _explicit_nil(value):
        return _EXPLICIT_NULL if allow_null else None
    value = _unwrap_controlled(value, *nested_keys)
    if _explicit_nil(value):
        return _EXPLICIT_NULL if allow_null else None
    normalized = _normalize_code_value(value)
    if normalized in (None, "", [], {}) or _is_unknown_token(normalized):
        return _EXPLICIT_NULL if allow_null else None
    if isinstance(normalized, int):
        normalized = str(normalized)
    if isinstance(normalized, str):
        identifier = normalized
        if base_uri and not identifier.startswith(("http://", "https://")):
            identifier = f"{base_uri.rstrip('/#')}/{identifier.lstrip('/#')}"
        return {"id": identifier}
    return None


def _wmo_region_concept(value: Any) -> Any:
    """Return a WMO Region Concept using the canonical HTTPS identifier.

    WMDR1 source records commonly contain the historical HTTP form. WMDR2-devt
    constrains WMO Region identifiers to the canonical HTTPS codelist URI.
    Explicit unknown/nil values are omitted rather than serialized as null.
    """
    concept = _concept(
        value,
        base_uri="https://codes.wmo.int/wmdr/WMORegion",
    )
    if not isinstance(concept, Mapping):
        return None

    identifier = concept.get("id")
    if isinstance(identifier, str):
        old_prefix = "http://codes.wmo.int/wmdr/WMORegion/"
        if identifier.startswith(old_prefix):
            concept = dict(concept)
            concept["id"] = (
                "https://codes.wmo.int/wmdr/WMORegion/"
                + identifier[len(old_prefix):]
            )
    return concept


def _concepts(
    value: Any,
    *nested_keys: str,
    base_uri: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        concept = _concept(item, *nested_keys, base_uri=base_uri)
        if isinstance(concept, Mapping):
            out.append(dict(concept))
    return _uniq_dicts(out)


# Compatibility helpers retained for callers/tests that imported them.
def _compact_wmdr_code_value(value: Any) -> Any:
    return _normalize_code_value(value)


def _compact_wmdr_code_values(value: Any, *nested_keys: str) -> List[Any]:
    return [item["id"] for item in _concepts(value, *nested_keys)]


def _first_compact_wmdr_code_value(value: Any, *nested_keys: str) -> Any:
    values = _compact_wmdr_code_values(value, *nested_keys)
    return values[0] if values else None


def _optional_controlled_value(value: Any, *nested_keys: str) -> Any:
    concept = _concept(value, *nested_keys)
    return concept


def _normalize_code_or_nil_reason(value: Any) -> Any:
    # Historical helper name. Current official WMDR2 uses Concept-or-null,
    # not a WMDR-specific {nilReason: ...} object. Keep the internal sentinel
    # private and expose JSON-compatible null to direct callers.
    normalized = _concept(value, allow_null=True)
    return None if normalized is _EXPLICIT_NULL else normalized


def _required_controlled_array(value: Any, *nested_keys: str) -> Any:
    concepts = _concepts(value, *nested_keys)
    return concepts or None


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return None


def _quantity(value: Any, uom: Any = None) -> Optional[Dict[str, Any]]:
    """Compatibility helper for source WMDR1 quantity objects."""
    raw_value = value
    raw_uom = uom
    if isinstance(value, Mapping):
        raw_value = _first_non_empty(value.get("value"), value.get("#text"), value.get("text"))
        raw_uom = _first_non_empty(value.get("uom"), value.get("unit"), value.get("@uom"), raw_uom)
    if raw_value in (None, "", [], {}):
        return None
    if isinstance(raw_value, str):
        try:
            raw_value = float(raw_value.strip())
        except ValueError:
            raw_value = raw_value.strip()
    result: Dict[str, Any] = {"value": raw_value}
    if raw_uom not in (None, "", [], {}):
        result["uom"] = raw_uom
    return result


def _finalize_wmdr2_value(value: Any, *, key: Optional[str] = None) -> Any:
    """Compatibility finalizer that preserves Concept identifiers."""
    if isinstance(value, dict):
        return {child_key: _finalize_wmdr2_value(child, key=child_key) for child_key, child in value.items()}
    if isinstance(value, list):
        return [_finalize_wmdr2_value(child, key=key) for child in value]
    return value


def _normalize_program_affiliations(value: Any) -> List[Dict[str, Any]]:
    """Compatibility alias for source Facility programme normalization."""
    return _facility_program_affiliations(value)


# ---------------------------------------------------------------------------
# Geometry and observation title helpers
# ---------------------------------------------------------------------------


def _parse_pos_lon_lat_z(raw: Any) -> Optional[List[Any]]:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        coords = raw.get("coordinates")
        if isinstance(coords, list) and len(coords) >= 2:
            return coords
        for key in ("geoLocation", "pos", "value", "text", "geometry", "position"):
            child = raw.get(key)
            nested = _parse_pos_lon_lat_z(child)
            if nested:
                return nested
        return None
    if not isinstance(raw, str):
        return None
    numbers: List[float] = []
    for item in raw.replace(",", " ").split():
        try:
            numbers.append(float(item))
        except ValueError:
            pass
    if len(numbers) < 2:
        return None
    lat, lon = numbers[0], numbers[1]
    coords: List[Any] = [lon, lat]
    if len(numbers) >= 3:
        z = numbers[2]
        coords.append(int(round(z)) if abs(z - round(z)) < 1e-9 else z)
    return coords


def _geopositioning_methods(item: Any) -> List[Dict[str, Any]]:
    if not isinstance(item, Mapping):
        return []
    return _concepts(
        item.get("geopositioningMethod"),
        base_uri="http://codes.wmo.int/wmdr/GeopositioningMethod",
    )


def _facility_temporal_geometry_entries(source: Mapping[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for key in ("geospatialLocation", "geospatialLocationHistory", "geometryHistory", "historicalLocation"):
        for item in _as_list(source.get(key)):
            coords = _parse_pos_lon_lat_z(item)
            if coords is None:
                continue
            entry: Dict[str, Any] = {"coordinates": coords, "date": _entry_date(item)}
            methods = _geopositioning_methods(item)
            if methods:
                entry["methods"] = methods
            entries.append(entry)
    entries.sort(key=lambda entry: (entry.get("date") == "..", str(entry.get("date")), _stable_json(entry.get("coordinates"))))
    return _uniq_dicts(entries)


def _facility_geometry_from_entries(entries: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    for entry in reversed(list(entries)):
        coords = entry.get("coordinates")
        if isinstance(coords, list) and len(coords) >= 2:
            return {"type": "Point", "coordinates": coords}
    return None


def _temporal_geometry_extension(entries: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(entries) < 2 and not any(entry.get("methods") for entry in entries):
        return None
    if not entries:
        return None
    result: Dict[str, Any] = {
        "type": "MovingPoint",
        "coordinates": [entry["coordinates"] for entry in entries],
        "dates": [entry.get("date") or ".." for entry in entries],
    }
    if any(entry.get("methods") for entry in entries):
        result["methods"] = [entry.get("methods", []) for entry in entries]
    return result


def _extract_code_list_ref(value: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    normalized = _normalize_code_value(value)
    if not isinstance(normalized, str):
        return None, None, None
    if normalized.startswith(("http://", "https://")):
        parts = [part for part in normalized.rstrip("/#").split("/") if part]
        return normalized, (parts[-2].lstrip("_") if len(parts) >= 2 else None), (parts[-1].lstrip("_") if parts else None)
    return None, None, normalized.lstrip("_")


def _observed_domain_from_observed_variable(value: Any) -> Optional[str]:
    _, register, _ = _extract_code_list_ref(value)
    notation = {
        "ObservedVariableAtmosphere": "atmosphere",
        "ObservedVariableCryosphere": "cryosphere",
        "ObservedVariableHydrology": "hydrological",
        "ObservedVariableHydrological": "hydrological",
        "ObservedVariableOcean": "ocean",
        "ObservedVariableSolidEarth": "solidEarth",
        "ObservedVariableSpace": "space",
        "ObservedVariableOuterSpace": "space",
        "ObservedVariableTerrestrial": "terrestrial",
        "ObservedVariableEarth": "solidEarth",
    }.get(register or "")
    return f"http://codes.wmo.int/wmdr/Domain/{notation}" if notation else None


def _lookup_code_list_label(domain: Optional[str], code: Optional[str]) -> Optional[str]:
    if not domain or not code:
        return None
    return CODE_LIST_LABELS.get(domain, {}).get(code.lstrip("_"))


def _format_observation_title(value: Any, geometry_type: Any = None) -> Optional[str]:
    _, register, code = _extract_code_list_ref(value)
    if not code:
        return None
    label = _lookup_code_list_label(register, code)
    domain_uri = _observed_domain_from_observed_variable(value)
    domain = _last_segment(domain_uri) if domain_uri else None
    geometry = _last_segment(geometry_type)
    parts: List[str] = []
    if domain:
        parts.append(f"domain: {domain}")
    if geometry:
        parts.append(f"geometry: {geometry}")
    parts.append(f"variable: {code}" + (f" {label}" if label else ""))
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Contact and link helpers
# ---------------------------------------------------------------------------


def _normalize_role(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        for key in ("codeListValue", "@codeListValue", "role", "value", "#text", "text", "name", "href", "url"):
            role = _normalize_role(value.get(key))
            if role:
                return role
        return None
    text = _strip_text(value)
    if not text:
        return None
    if text.endswith(("#CI_RoleCode", "/CI_RoleCode", "#RoleCode", "/RoleCode")):
        return None
    return (_last_segment(text) or text).lstrip("_") or None


def _normalize_roles(value: Any) -> List[str]:
    return sorted({role for item in _as_list(value) if (role := _normalize_role(item))})


def _normalize_phone_value(value: str) -> str:
    text = value.strip()
    if text.startswith("+"):
        text = re.sub(r"\(0\)", "", text)
        text = re.sub(r"\(0", "(", text)
        digits = re.sub(r"\D", "", text)
        return f"+{digits}" if digits else value
    digits = re.sub(r"\D", "", text)
    if digits.startswith("00") and len(digits) > 4:
        return "+" + digits[2:]
    return re.sub(r"\s+", "", text)


def _normalize_link(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, str):
        href = value.strip()
        if not href.startswith(("http://", "https://")):
            return None
        return {"href": href, "rel": "about", "type": "text/html"}
    obj = _as_mapping(value)
    href = _strip_text(_first_non_empty(obj.get("href"), obj.get("url"), obj.get("linkage"), obj.get("value")))
    if not href:
        return None
    result: Dict[str, Any] = {"href": href}
    for key in ("rel", "type", "title", "hreflang"):
        text = _strip_text(obj.get(key))
        if text:
            result[key] = text
    result.setdefault("rel", "about")
    result.setdefault("type", "text/html")
    return result


def _normalize_ogc_contact(raw: Any) -> Optional[Dict[str, Any]]:
    payload = _as_mapping(raw)
    if not payload:
        text = _strip_text(raw)
        if not text:
            return None
        return {"emails": [{"value": _remove_prefix(text, "mailto:")}]} if "@" in text else {"organization": text}

    # XML-derived responsibleParty can contain the actual party in a nested
    # same-concept wrapper while validity/id live on the outer occurrence.
    nested_party = _as_mapping(payload.get("responsibleParty"))
    if nested_party:
        merged_payload = dict(nested_party)
        for key, value in payload.items():
            if key != "responsibleParty" and key not in SOURCE_TEMPORAL_KEYS and value not in (None, "", [], {}):
                merged_payload.setdefault(key, value)
        payload = merged_payload

    contact: Dict[str, Any] = {}
    aliases = {"name": "individualName", "organization": "organisationName"}
    for key in ("identifier", "name", "position", "organization", "hoursOfService", "contactInstructions"):
        text = _strip_text(_first_non_empty(payload.get(key), payload.get(aliases.get(key, ""))))
        if text:
            contact[key] = text

    info = _as_mapping(payload.get("contactInfo"))
    if "contactInstructions" not in contact:
        instructions = _strip_text(info.get("contactInstructions"))
        if instructions:
            contact["contactInstructions"] = instructions
    address_info = _as_mapping(info.get("address"))

    emails: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    email_sources: List[Any] = []
    for key in ("emails", "email", "electronicMailAddress", "mail", "mailAddress"):
        email_sources.extend(_as_list(payload.get(key)))
    email_sources.extend(_as_list(address_info.get("electronicMailAddress")))
    for item in email_sources:
        item_obj = _as_mapping(item)
        text = _strip_text(_first_non_empty(item_obj.get("value"), item_obj.get("email"), item_obj.get("address"), item_obj.get("#text"), item))
        if not text:
            continue
        if text.startswith(("http://", "https://")):
            link = _normalize_link(text)
            if link:
                links.append(link)
        else:
            email: Dict[str, Any] = {"value": _remove_prefix(text, "mailto:")}
            roles = _normalize_roles(item_obj.get("roles") or item_obj.get("role"))
            if roles:
                email["roles"] = roles
            emails.append(email)
    if emails:
        contact["emails"] = _uniq_dicts(emails)

    phones: List[Dict[str, Any]] = []
    phone_info = _as_mapping(info.get("phone"))
    phone_sources: List[Any] = []
    for key in ("phones", "phone", "telephone", "voice", "facsimile"):
        phone_sources.extend(_as_list(payload.get(key)))
    for key in ("voice", "facsimile", "phone", "phones"):
        phone_sources.extend(_as_list(phone_info.get(key)))
    for item in phone_sources:
        item_obj = _as_mapping(item)
        text = _strip_text(_first_non_empty(item_obj.get("value"), item_obj.get("phone"), item_obj.get("number"), item_obj.get("#text"), item))
        if text:
            phone: Dict[str, Any] = {"value": _normalize_phone_value(text)}
            roles = _normalize_roles(item_obj.get("roles") or item_obj.get("role"))
            if roles:
                phone["roles"] = roles
            phones.append(phone)
    if phones:
        contact["phones"] = _uniq_dicts(phones)

    address: Dict[str, Any] = {}
    delivery = _first_non_empty(address_info.get("deliveryPoint"), address_info.get("deliveryPoints"), address_info.get("street"))
    if delivery:
        address["deliveryPoint"] = [str(item).strip() for item in _as_list(delivery) if str(item).strip()]
    for key in ("city", "administrativeArea", "postalCode", "country"):
        text = _strip_text(address_info.get(key))
        if text:
            address[key] = text
    address_roles = _normalize_roles(address_info.get("roles") or address_info.get("role"))
    if address_roles:
        address["roles"] = address_roles
    if address:
        contact["addresses"] = [address]

    for key in ("links", "link", "onlineResource", "url", "href"):
        for item in _as_list(payload.get(key)):
            link = _normalize_link(item)
            if link:
                links.append(link)
    online = _as_mapping(info.get("onlineResource"))
    if online:
        link = _normalize_link(online)
        if link:
            links.append(link)
    if links:
        contact["links"] = _uniq_dicts(links)

    roles = _normalize_roles(payload.get("roles") or payload.get("role"))
    if roles:
        contact["roles"] = roles
    return contact or None


def _contact_identifier(contact: Mapping[str, Any]) -> str:
    existing = _strip_text(contact.get("identifier"))
    if existing:
        return existing
    for item in _as_list(contact.get("emails")):
        value = _strip_text(_as_mapping(item).get("value"))
        if value:
            return f"contact:{value.lower()}"
    base = _first_non_empty(contact.get("organization"), contact.get("name"), contact.get("position"), "contact")
    digest = hashlib.sha1(_stable_json(contact).encode("utf-8")).hexdigest()[:10]
    return f"contact:{_slug(base)}-{digest}"


def _merge_contact(existing: Mapping[str, Any], new: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(existing)
    for key, value in new.items():
        if key in {"emails", "phones", "addresses", "links"}:
            result[key] = _uniq_dicts([*_as_list(result.get(key)), *_as_list(value)])
        elif key == "roles":
            result[key] = sorted(set(_normalize_roles(result.get(key)) + _normalize_roles(value)))
        elif key not in result or result[key] in (None, "", [], {}):
            result[key] = value
    return result


def _register_contact(registry: Dict[str, Dict[str, Any]], raw: Any) -> Optional[str]:
    contact = _normalize_ogc_contact(raw)
    if not contact:
        return None
    identifier = _contact_identifier(contact)
    contact["identifier"] = identifier
    contact.pop("roles", None)
    registry[identifier] = _merge_contact(registry.get(identifier, {}), contact)
    return identifier


def _contact_assignments(
    value: Any,
    registry: Dict[str, Dict[str, Any]],
    fallback_role: Any = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        payload = _as_mapping(item)
        ref = _strip_text(payload.get("contact")) if payload else None

        normalized_contact = _normalize_ogc_contact(item)
        normalized_roles = (
            _normalize_roles(normalized_contact.get("roles"))
            if isinstance(normalized_contact, Mapping)
            else []
        )
        if not normalized_roles:
            normalized_roles = _normalize_roles(payload.get("roles") or payload.get("role"))
        if not normalized_roles and fallback_role is not None:
            normalized_roles = _normalize_roles(fallback_role)

        if not ref:
            ref = _register_contact(registry, item)
        if ref and normalized_roles:
            out.append({"contact": ref, "roles": normalized_roles})
    return _uniq_dicts(out)


def _collect_discovery_values(entity_type: str, source: Mapping[str, Any], bucket: str) -> List[Any]:
    values: List[Any] = []
    policy = DISCOVERY_POLICY.get(entity_type, {})
    for key in policy.get(bucket, []):
        values.extend(_as_list(source.get(key)))
    return values


def _keywords_from_values(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for raw in values:
        for item in _as_list(raw):
            text = _strip_text(_last_segment(item) or item)
            if text and not _is_unknown_token(text):
                out.append(text.replace("_", " "))
    return cast(List[str], _uniq_scalars(out))


def _extract_links(source: Mapping[str, Any], entity_type: str) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    policy = DISCOVERY_POLICY.get(entity_type, {})
    for key in policy.get("links", []):
        for item in _as_list(source.get(key)):
            link = _normalize_link(item)
            if link:
                links.append(link)
    return _uniq_dicts(links)


# ---------------------------------------------------------------------------
# Instrument, environment, programme affiliation and vertical distance
# ---------------------------------------------------------------------------


def _unwrap_named_source_object(value: Any, *names: str) -> Any:
    current = value
    allowed = {name.lower() for name in names}
    while isinstance(current, Mapping) and len(current) == 1:
        key, child = next(iter(current.items()))
        if str(key).lower() not in allowed:
            break
        current = child
    return current


def _first_mapping(value: Any, *wrapper_names: str) -> Mapping[str, Any]:
    for item in _as_list(value):
        unwrapped = _unwrap_named_source_object(item, *wrapper_names)
        if isinstance(unwrapped, Mapping):
            return unwrapped
    return {}


def _equipment_from_deployment(src: Mapping[str, Any]) -> Mapping[str, Any]:
    return _first_mapping(
        _first_non_empty(src.get("deployedEquipment"), src.get("equipment")),
        "deployedEquipment", "Equipment", "equipment",
    )


def _instrument_key(src: Mapping[str, Any]) -> Optional[str]:
    manufacturer = _strip_text(src.get("manufacturer"))
    model = _strip_text(src.get("model"))
    if manufacturer and _is_unknown_token(manufacturer):
        manufacturer = None
    if model and _is_unknown_token(model):
        model = None
    if not manufacturer and not model:
        return None
    # Context-local id: the property already establishes that this is an Instrument.
    return "-".join(part for part in (_slug(manufacturer) if manufacturer else "", _slug(model) if model else "") if part)


def _instrument_from_source(src: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    identifier = _instrument_key(src)
    if not identifier:
        return None
    result: Dict[str, Any] = {"id": identifier}
    for key in ("manufacturer", "model", "description"):
        text = _strip_text(src.get(key))
        if text and not _is_unknown_token(text):
            result[key] = text
    methods: List[Dict[str, Any]] = []
    for key in ("observingMethods", "observableMethods"):
        methods.extend(_concepts(src.get(key)))
    if not methods and src.get("observingMethod") not in (None, "", [], {}):
        methods.extend(_concepts(src.get("observingMethod")))
    if methods:
        result["observingMethods"] = _uniq_dicts(methods)
    if _non_empty(src.get("verticalRange")):
        result["verticalRange"] = src.get("verticalRange")
    return result


def _merge_instrument(existing: Mapping[str, Any], new: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(existing)
    for key, value in new.items():
        if key not in result or result[key] in (None, "", [], {}):
            result[key] = value
        elif isinstance(result.get(key), list) or isinstance(value, list):
            if all(isinstance(item, Mapping) for item in [*_as_list(result.get(key)), *_as_list(value)]):
                result[key] = _uniq_dicts([dict(item) for item in [*_as_list(result.get(key)), *_as_list(value)] if isinstance(item, Mapping)])
            else:
                result[key] = _uniq_scalars([*_as_list(result.get(key)), *_as_list(value)])
    return result


def _normalize_territories(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            raw = _first_non_empty(
                item.get("territory"), item.get("territoryName"), item.get("name"),
                item.get("identifier"), item.get("value"), item.get("href"), item.get("url"),
            )
            territory = _concept(raw, allow_null=True, base_uri="http://codes.wmo.int/wmdr/TerritoryName")
            if territory is None:
                continue
            occurrence: Dict[str, Any] = {"territory": territory}
            start, end = _extract_interval(item)
            dates = _dates(start, end)
            if dates:
                occurrence["dates"] = dates
            out.append(occurrence)
        else:
            territory = _concept(item, allow_null=True, base_uri="http://codes.wmo.int/wmdr/TerritoryName")
            if territory is not None:
                out.append({"territory": territory})
    return _uniq_dicts(out)


def _environment_from_facility(facility: Mapping[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    def append(raw: Any, field: str, value: Any) -> None:
        if value in (None, "", [], {}):
            return
        occurrence: Dict[str, Any] = {field: value}
        if isinstance(raw, Mapping):
            start, end = _extract_interval(raw)
            time = _time_interval(start, end)
            if time:
                occurrence["time"] = time
        entries.append(occurrence)

    for source_key in ("climateZone", "surfaceRoughness"):
        for raw in _as_list(facility.get(source_key)):
            append(raw, source_key, _concept(raw, source_key))

    for raw in _as_list(facility.get("surfaceCover")):
        raw_obj = _as_mapping(raw)
        value = _concept(raw, "surfaceCover")
        scheme = _concept(_first_non_empty(raw_obj.get("surfaceCoverClassification"), raw_obj.get("scheme")))
        if value and scheme:
            append(raw, "surfaceCover", {"value": value, "scheme": scheme})

    for raw in _as_list(facility.get("topographyBathymetry")):
        obj = _as_mapping(raw)
        topo: Dict[str, Any] = {}
        for key in ("localTopography", "relativeElevation", "topographicContext", "altitudeOrDepth"):
            concept = _concept(obj.get(key), key)
            if concept:
                topo[key] = concept
        if topo:
            append(raw, "topographyBathymetry", topo)

    existing = facility.get("environment")
    for raw in _as_list(existing):
        if isinstance(raw, Mapping):
            entries.append(dict(raw))

    # Merge explicitly time-identical entries; keep untimed occurrences separate.
    merged: Dict[str, Dict[str, Any]] = {}
    untimed = 0
    for entry in entries:
        if "time" in entry:
            key = _stable_json(entry["time"])
        else:
            untimed += 1
            key = f"__untimed_{untimed}"
        merged.setdefault(key, {}).update(entry)
    return _uniq_dicts(merged.values())


def _facility_program_affiliations(value: Any) -> List[Dict[str, Any]]:
    """Normalize source Facility programme data for mapping/enrichment only."""
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if not isinstance(item, Mapping):
            for concept in _concepts(item):
                out.append({"programAffiliation": concept})
            continue
        programs = _concepts(
            _first_non_empty(item.get("programAffiliation"), item.get("program"), item.get("href"), item.get("value")),
            base_uri="http://codes.wmo.int/wmdr/ProgramAffiliation",
        )
        for program in programs:
            base: Dict[str, Any] = {"programAffiliation": program}
            for key in ("programSpecificFacilityId", "programSpecificFacilityTitle"):
                text = _strip_text(item.get(key))
                if text:
                    base[key] = text
            base_start, base_end = _extract_interval(item)
            base_dates = _dates(base_start, base_end)
            statuses = _as_list(item.get("reportingStatus"))
            if not statuses:
                if base_dates:
                    base["dates"] = base_dates
                out.append(base)
                continue
            for status_item in statuses:
                status_obj = _as_mapping(status_item)
                raw_status = _first_non_empty(
                    status_obj.get("reportingStatus"), status_obj.get("status"),
                    status_obj.get("href"), status_obj.get("value"), status_item,
                )
                occurrence = dict(base)
                status = _concept(raw_status, "reportingStatus", allow_null=True, base_uri="http://codes.wmo.int/wmdr/ReportingStatus")
                if status is not None:
                    occurrence["reportingStatus"] = status
                status_start, status_end = _extract_interval(status_obj)
                dates = _dates(status_start, status_end) or base_dates
                if dates:
                    occurrence["dates"] = dates
                out.append(occurrence)
    return _uniq_dicts(out)


def _programme_index(facility: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    raw = _first_non_empty(facility.get("programAffiliations"), facility.get("programAffiliation"), facility.get("programs"))
    index: Dict[str, List[Dict[str, Any]]] = {}
    for occurrence in _facility_program_affiliations(raw):
        concept = _as_mapping(occurrence.get("programAffiliation"))
        identifier = concept.get("id")
        if isinstance(identifier, (str, int)):
            index.setdefault(str(identifier), []).append(occurrence)
    return index


def _external_ids_and_titles(facility: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], List[str]]:
    raw = _first_non_empty(facility.get("programAffiliations"), facility.get("programAffiliation"), facility.get("programs"))
    external_ids: List[Dict[str, str]] = []
    titles: List[str] = []
    for occurrence in _facility_program_affiliations(raw):
        program_id = _as_mapping(occurrence.get("programAffiliation")).get("id")
        scheme = _last_segment(program_id) if program_id is not None else None
        value = _strip_text(occurrence.get("programSpecificFacilityId"))
        if value:
            external: Dict[str, str] = {"value": value}
            if scheme:
                external["scheme"] = scheme
            external_ids.append(external)
        title = _strip_text(occurrence.get("programSpecificFacilityTitle"))
        if title:
            titles.append(title)
    return _uniq_dicts(external_ids), cast(List[str], _uniq_scalars(titles))


def _observation_program_affiliations(obs: Mapping[str, Any], programme_index: Mapping[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    raw_programmes = _first_non_empty(obs.get("programAffiliations"), obs.get("programAffiliation"), obs.get("programs"))
    for program in _concepts(
        raw_programmes,
        "programAffiliation",
        "program",
        base_uri="http://codes.wmo.int/wmdr/ProgramAffiliation",
    ):
        identifier = str(program.get("id"))
        matches = programme_index.get(identifier, [])
        if not matches:
            out.append({"programAffiliation": program})
            continue
        for match in matches:
            occurrence: Dict[str, Any] = {"programAffiliation": program}
            if "reportingStatus" in match:
                occurrence["reportingStatus"] = match["reportingStatus"]
            # Dates are preservation-only: emit only when the WMDR1 source
            # actually carried the validity information.
            if "dates" in match:
                occurrence["dates"] = match["dates"]
            out.append(occurrence)
    return _uniq_dicts(out)


def _vertical_distance(merged_src: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    raw = _first_non_empty(
        merged_src.get("verticalDistance"),
        merged_src.get("verticalDistanceFromReferenceSurface"),
        merged_src.get("heightAboveLocalReferenceSurface"),
        merged_src.get("heightAboveReferenceSurface"),
    )
    if raw in (None, "", [], {}):
        return None

    distances: List[float] = []
    raw_unit: Any = None
    if isinstance(raw, Mapping) and isinstance(raw.get("distances"), list):
        for value in raw.get("distances", []):
            try:
                distances.append(float(value))
            except (TypeError, ValueError):
                pass
        raw_unit = _first_non_empty(raw.get("unit"), raw.get("uom"), raw.get("@uom"))
    else:
        raw_values = _as_list(raw)
        for item in raw_values:
            item_obj = _as_mapping(item)
            value = _first_non_empty(item_obj.get("value"), item_obj.get("#text"), item_obj.get("text"), item)
            try:
                distances.append(float(value))
            except (TypeError, ValueError):
                pass
            raw_unit = _first_non_empty(raw_unit, item_obj.get("uom"), item_obj.get("unit"), item_obj.get("@uom"))

    raw_unit = _first_non_empty(
        raw_unit,
        merged_src.get("verticalDistanceFromReferenceSurfaceUom"),
        merged_src.get("heightAboveLocalReferenceSurfaceUom"),
        merged_src.get("uom"),
    )
    reference = _first_non_empty(merged_src.get("referenceSurface"), merged_src.get("localReferenceSurface"))
    if not distances:
        return None

    result: Dict[str, Any] = {"distances": distances}
    unit = _concept(raw_unit, allow_null=True, base_uri="http://codes.wmo.int/wmdr/unit")
    ref = _concept(reference, "referenceSurface", "localReferenceSurface", allow_null=True, base_uri="http://codes.wmo.int/wmdr/ReferenceSurfaceType")
    if unit is not None:
        result["unit"] = unit
    if ref is not None:
        result["referenceSurface"] = ref
    return result


# ---------------------------------------------------------------------------
# Schedules and procedures
# ---------------------------------------------------------------------------


def _normalize_diurnal_time(value: Any) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?(?::(\d{1,2}))?Z?", text)
    if not match:
        return text
    h = min(max(int(match.group(1)), 0), 23)
    m = min(max(int(match.group(2) or 0), 0), 59)
    s = min(max(int(match.group(3) or 0), 0), 59)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_diurnal_seconds(value: Any) -> Optional[int]:
    text = _normalize_diurnal_time(value)
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2})", text)
    if not match:
        return None
    h, m, s = map(int, match.groups())
    return h * 3600 + m * 60 + s


def _iso_duration_from_seconds(seconds: int) -> Optional[str]:
    if seconds <= 0:
        return None
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days and not (hours or minutes or seconds):
        return f"P{days}D"
    time = "".join(part for part in (f"{hours}H" if hours else "", f"{minutes}M" if minutes else "", f"{seconds}S" if seconds else "")) or "0S"
    return f"P{days}DT{time}" if days else f"PT{time}"


def _coverage_time_seconds(source: Mapping[str, Any], prefix: str) -> Optional[int]:
    direct = _first_non_empty(source.get(f"{prefix}Time"), source.get(f"{prefix}ClockTime"))
    parsed = _parse_diurnal_seconds(direct) if direct is not None else None
    if parsed is not None:
        return parsed
    hour = _first_non_empty(source.get(f"{prefix}Hour"), source.get(f"{prefix}Hours"))
    if hour is None:
        return None
    minute = _first_non_empty(source.get(f"{prefix}Minute"), source.get(f"{prefix}Minutes"), 0)
    second = _first_non_empty(source.get(f"{prefix}Second"), source.get(f"{prefix}Seconds"), 0)
    return _parse_diurnal_seconds(f"{hour}:{minute}:{second}")


def _diurnal_coverage_fields(*sources: Mapping[str, Any]) -> Dict[str, str]:
    """Build reusable-schedule start/duration from WMDR diurnal coverage."""
    for source in sources:
        if not source:
            continue
        start = _coverage_time_seconds(source, "start")
        end = _coverage_time_seconds(source, "end")
        if start is None and end is None:
            continue
        if start is None:
            start = 0
        h, rem = divmod(start, 3600)
        m, sec = divmod(rem, 60)
        result = {"start": f"{CANONICAL_SCHEDULE_START_DATE}T{h:02d}:{m:02d}:{sec:02d}"}
        if end is not None:
            if end <= start:
                end += 86400
            duration = _iso_duration_from_seconds(end - start)
            if duration:
                result["duration"] = duration
        return result
    return {}


def _schedule_uid(schedule: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in schedule.items() if key not in {"uid", "id"}}
    return "schedule_" + hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()[:10]


def _normalize_schedule_object(raw: Any, *, kind: str = "shared") -> Optional[Dict[str, Any]]:
    if raw in (None, "", [], {}):
        return None
    schedule = dict(raw) if isinstance(raw, Mapping) else {"frequency": raw}
    legacy_id = _strip_text(schedule.pop("id", None))
    uid = _strip_text(schedule.get("uid")) or legacy_id
    if uid and uid.startswith("schedule:"):
        uid = "schedule_" + _slug(_remove_prefix(uid, "schedule:"))
    schedule.setdefault("@type", "Event")
    schedule.setdefault("start", CANONICAL_SCHEDULE_START_DATE)

    sampling = _first_non_empty(
        schedule.pop("samplingFrequency", None), schedule.pop("temporalSamplingInterval", None),
        schedule.get("wmo.int:samplingFrequency"),
    )
    if sampling:
        schedule["wmo.int:samplingFrequency"] = _normalize_time_resolution(sampling)
    aggregation = _first_non_empty(
        schedule.pop("aggregationInterval", None), schedule.get("wmo.int:aggregationInterval"),
    )
    if aggregation:
        schedule["wmo.int:aggregationInterval"] = _normalize_time_resolution(aggregation)
    diurnal = _first_non_empty(schedule.pop("diurnalBaseTime", None), schedule.get("wmo.int:diurnalBaseTime"))
    if diurnal:
        schedule["wmo.int:diurnalBaseTime"] = _normalize_diurnal_time(diurnal)
    frequency = schedule.pop("frequency", None)
    if frequency:
        target = "wmo.int:aggregationInterval" if kind == "reporting" else "wmo.int:samplingFrequency"
        schedule.setdefault(target, _normalize_time_resolution(frequency))

    if not any(_non_empty(schedule.get(key)) for key in ("duration", "recurrenceRules", "wmo.int:samplingFrequency", "wmo.int:aggregationInterval")):
        return None
    schedule["uid"] = uid or _schedule_uid(schedule)
    return cast(Dict[str, Any], _clean_none(schedule))


def _schedule_from_source(src: Mapping[str, Any], *, kind: str) -> Optional[Dict[str, Any]]:
    coverage = _as_mapping(src.get("coverage"))
    sampling = _as_mapping(src.get("sampling"))
    reporting = _as_mapping(src.get("reporting"))
    reporting_coverage = _as_mapping(reporting.get("coverage"))

    explicit = _first_non_empty(
        src.get("observingSchedule") if kind == "observing" else src.get("reportingSchedule"),
        src.get("observingSchedules") if kind == "observing" else src.get("reportingSchedules"),
        sampling.get("observingSchedule"), sampling.get("schedule"),
        reporting.get("reportingSchedule"), reporting.get("schedule"),
        coverage.get("schedule"), reporting_coverage.get("schedule"), src.get("schedule"),
    )
    if isinstance(explicit, list):
        explicit = next((item for item in explicit if isinstance(item, Mapping)), None) or (explicit[0] if explicit else None)
    schedule: Dict[str, Any] = dict(explicit) if isinstance(explicit, Mapping) else ({"frequency": explicit} if explicit else {})

    sampling_interval = _first_non_empty(
        src.get("temporalSamplingInterval"), sampling.get("temporalSamplingInterval"),
        coverage.get("temporalSamplingInterval"),
    )
    aggregation_interval = _first_non_empty(
        reporting.get("aggregationInterval"), src.get("aggregationInterval"),
        coverage.get("aggregationInterval"), reporting_coverage.get("aggregationInterval"),
    )
    if sampling_interval:
        schedule.setdefault("wmo.int:samplingFrequency", sampling_interval)
    if aggregation_interval and kind == "reporting":
        schedule.setdefault("wmo.int:aggregationInterval", aggregation_interval)

    diurnal = _first_non_empty(
        reporting.get("diurnalBaseTime"), src.get("diurnalBaseTime"),
        coverage.get("diurnalBaseTime"), reporting_coverage.get("diurnalBaseTime"),
    )
    if diurnal:
        schedule.setdefault("wmo.int:diurnalBaseTime", diurnal)

    for key, value in _diurnal_coverage_fields(coverage, reporting_coverage, src, reporting).items():
        schedule.setdefault(key, value)

    if not schedule:
        duration = _first_non_empty(
            src.get("duration"), coverage.get("duration"),
            reporting.get("duration"), reporting_coverage.get("duration"),
        )
        if duration:
            schedule["duration"] = duration

    return _normalize_schedule_object(schedule, kind=kind) if schedule else None


def _register_schedule(schedule: Optional[Mapping[str, Any]], registry: Dict[str, Dict[str, Any]], *, kind: str) -> Optional[str]:
    normalized = _normalize_schedule_object(schedule, kind=kind) if schedule else None
    if not normalized:
        return None
    uid = cast(str, normalized["uid"])
    registry[uid] = normalized
    return uid


def _observing_procedure_from_source(src: Mapping[str, Any], schedule_registry: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    schedule = _schedule_from_source(src, kind="observing")
    uid = _register_schedule(schedule, schedule_registry, kind="observing")
    if not uid:
        return None
    result: Dict[str, Any] = {"observingSchedules": [uid]}
    start, end = _extract_interval(src)
    time = _time_interval(start, end)
    if time:
        result["time"] = time
    strategy = _first_non_empty(src.get("strategy"), _as_mapping(src.get("sampling")).get("samplingStrategy"), src.get("samplingStrategy"))
    concept = _concept(
        strategy,
        base_uri="http://codes.wmo.int/wmdr/SamplingStrategy",
    )
    if concept:
        result["strategy"] = concept
    return result


def _reporting_procedure_from_source(
    src: Mapping[str, Any],
    contact_registry: Dict[str, Dict[str, Any]],
    schedule_registry: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    reporting = _as_mapping(src.get("reporting"))
    procedure = _as_mapping(src.get("reportingProcedure"))
    merged: Dict[str, Any] = {**reporting, **procedure}
    for key in (
        "internationalExchange", "dataFormat", "dataPolicy", "levelOfData", "numberOfObservationsInReportingInterval",
        "referenceDatum", "referenceTimeSource", "spatialReportingInterval", "strategy", "timeliness",
        "timeStampMeaning", "uom", "temporalReportingInterval", "temporalAggregate", "reportingSchedule",
        "contact", "contacts", "responsibleParty", "links",
    ):
        if key in src and key not in merged:
            merged[key] = src[key]
    if not merged:
        return None

    result: Dict[str, Any] = {}
    exchange = merged.get("internationalExchange")
    if isinstance(exchange, str):
        lower = exchange.strip().lower()
        exchange = True if lower in {"true", "1", "yes"} else False if lower in {"false", "0", "no"} else None
    if isinstance(exchange, bool):
        result["internationalExchange"] = exchange

    reporting_bases = {
        "dataPolicy": "http://codes.wmo.int/wmdr/DataPolicy",
        "levelOfData": "http://codes.wmo.int/wmdr/LevelOfData",
        "uom": "http://codes.wmo.int/wmdr/unit",
    }
    for key in ("dataPolicy", "levelOfData", "referenceDatum", "strategy", "timeStampMeaning", "uom"):
        raw = merged.get(key)
        if key == "dataPolicy" and isinstance(raw, Mapping):
            raw = _first_non_empty(raw.get("dataPolicy"), raw)
        concept = _concept(
            raw,
            key,
            allow_null=(key == "dataPolicy"),
            base_uri=reporting_bases.get(key),
        )
        if concept is not None:
            result[key] = concept

    array_bases = {
        "dataFormat": "http://codes.wmo.int/wmdr/DataFormat",
        "referenceTimeSource": None,
    }
    for key in ("dataFormat", "referenceTimeSource"):
        concepts = _concepts(merged.get(key), key, base_uri=array_bases.get(key))
        if concepts:
            result[key] = concepts

    for key in ("numberOfObservationsInReportingInterval", "spatialReportingInterval", "timeliness"):
        if _non_empty(merged.get(key)):
            result[key] = merged[key]
    for key in ("temporalReportingInterval", "temporalAggregate"):
        if _non_empty(merged.get(key)):
            result[key] = _normalize_time_resolution(merged[key])

    schedule_source = dict(src)
    schedule_source["reporting"] = merged
    uid = _register_schedule(_schedule_from_source(schedule_source, kind="reporting"), schedule_registry, kind="reporting")
    if uid:
        result["reportingSchedules"] = [uid]

    assignments: List[Dict[str, Any]] = []
    for key in ("contact", "contacts", "responsibleParty"):
        assignments.extend(_contact_assignments(merged.get(key), contact_registry, "responsibleParty" if key == "responsibleParty" else None))
    if assignments:
        result["contactAssignments"] = _uniq_dicts(assignments)
    raw_links = _as_list(merged.get("links"))
    links = [link for item in raw_links if (link := _normalize_link(item))]
    if links:
        result["links"] = _uniq_dicts(links)
    return cast(Optional[Dict[str, Any]], _clean_none(result))


# ---------------------------------------------------------------------------
# Configuration and Observation mapping
# ---------------------------------------------------------------------------


def _status_history_entries(src: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    raw = _first_non_empty(src.get("instrumentOperatingStatus"), src.get("operatingStatus"))
    entries: List[Mapping[str, Any]] = []
    for item in _as_list(raw):
        unwrapped = _unwrap_named_source_object(item, "instrumentOperatingStatus", "operatingStatus")
        if not isinstance(unwrapped, Mapping):
            continue
        start, end = _extract_interval(unwrapped)
        if start not in (None, "", [], {}) or end not in (None, "", [], {}):
            entries.append(unwrapped)
    return entries


def _configuration_source_variants(src: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    history = _status_history_entries(src)
    if len(history) <= 1:
        return [src]
    out: List[Mapping[str, Any]] = []
    for item in history:
        variant = dict(src)
        for key in ("instrumentOperatingStatus", "operatingStatus", "validFrom", "validTo", "beginPosition", "endPosition", "date"):
            variant.pop(key, None)
        status = _first_non_empty(item.get("instrumentOperatingStatus"), item.get("operatingStatus"), item.get("href"), item.get("value"))
        variant["instrumentOperatingStatus"] = status
        start, end = _extract_interval(item)
        time = _time_interval(start, end)
        if time:
            variant["time"] = time
        out.append(variant)
    return out


def _configuration_id(src: Mapping[str, Any], *, ordinal: int, parent_id: str, split_ordinal: int = 0) -> str:
    raw = _first_non_empty(src.get("id"), src.get("uid"), src.get("identifier"))
    if raw not in (None, "", [], {}):
        base = _sanitize_id(raw)
        if base.startswith("configuration:"):
            base = _remove_prefix(base, "configuration:")
        if base.startswith("observingConfiguration:"):
            base = _remove_prefix(base, "observingConfiguration:")
        return f"{base}-{split_ordinal}" if split_ordinal else base
    payload = {key: value for key, value in src.items() if key not in {"responsibleParty", "contact", "contacts"}}
    digest = hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()[:10]
    return f"{_slug(parent_id)}-{ordinal + 1}-{digest}"


def _configuration_from_source(
    src: Mapping[str, Any],
    instrument_registry: Dict[str, Dict[str, Any]],
    contact_registry: Dict[str, Dict[str, Any]],
    *,
    configuration_id: str,
) -> Dict[str, Any]:
    equipment = _equipment_from_deployment(src)
    merged: Dict[str, Any] = dict(equipment)
    for key, value in src.items():
        if key != "deployedEquipment" and value not in (None, "", [], {}):
            merged[key] = value

    result: Dict[str, Any] = {"id": configuration_id}
    start, end = _extract_interval(src)
    time = _time_interval(start, end)
    if time:
        result["time"] = time

    configuration_codes = (
        ("observingMethod", ("observingMethod",), True, None),
        ("sourceOfObservation", ("sourceOfObservation",), True, "http://codes.wmo.int/wmdr/SourceOfObservation"),
        ("operatingStatus", ("operatingStatus", "instrumentOperatingStatus"), False, "http://codes.wmo.int/wmdr/InstrumentOperatingStatus"),
        ("exposure", ("exposure",), False, "http://codes.wmo.int/wmdr/Exposure"),
    )
    for target, source_keys, allow_null, base_uri in configuration_codes:
        raw = _first_non_empty(*(merged.get(key) for key in source_keys))
        if raw not in (None, "", [], {}):
            concept = _concept(raw, *source_keys, allow_null=allow_null, base_uri=base_uri)
            if concept is not None:
                result[target] = concept

    serial = _strip_text(_first_non_empty(merged.get("instrumentSerialNumber"), merged.get("serialNumber")))
    if serial:
        result["instrumentSerialNumber"] = serial
    for key in ("relativeLocation", "description"):
        text = _strip_text(merged.get(key))
        if text:
            result[key] = text

    vertical = _vertical_distance(merged)
    if vertical:
        result["verticalDistance"] = vertical

    locations = _facility_temporal_geometry_entries(merged)
    geometry = _facility_geometry_from_entries(locations)
    if geometry:
        result["geometry"] = geometry

    instrument = _instrument_from_source(merged)
    if instrument:
        instrument_id = cast(str, instrument["id"])
        instrument_registry[instrument_id] = _merge_instrument(instrument_registry.get(instrument_id, {}), instrument)
        result["instrument"] = instrument_id

    assignments: List[Dict[str, Any]] = []
    for key in ("contact", "contacts", "responsibleParty", "operator", "maintainer"):
        fallback = key if key not in {"contact", "contacts"} else None
        assignments.extend(_contact_assignments(merged.get(key), contact_registry, fallback))
    if assignments:
        result["contactAssignments"] = _uniq_dicts(assignments)

    links = _extract_links(merged, "configuration")
    if links:
        result["links"] = links
    return cast(Dict[str, Any], _clean_none(result))


# Backward-compatible alias for older tests/importers.
def _observing_configuration_from_source(
    src: Mapping[str, Any],
    instrument_registry: Dict[str, Dict[str, Any]],
    contact_registry: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return _configuration_from_source(
        src, instrument_registry, contact_registry,
        configuration_id=_configuration_id(src, ordinal=0, parent_id="observation"),
    )


def _observed_feature(obs: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    raw = _first_non_empty(obs.get("observedFeature"), obs.get("observedDomain"), obs.get("observationDomain"))
    result: Dict[str, Any] = {}
    if isinstance(raw, Mapping):
        domain = _concept(
            _first_non_empty(raw.get("domain"), raw.get("value"), raw.get("href")),
            base_uri="http://codes.wmo.int/wmdr/Domain",
        )
        if domain:
            result["domain"] = domain
        domain_feature = _concept(_first_non_empty(raw.get("domainFeature"), raw.get("observedFeatureDomainFeature")))
        if domain_feature:
            result["domainFeature"] = domain_feature
        feature_name = _strip_text(_first_non_empty(raw.get("featureName"), raw.get("observedFeatureName")))
        if feature_name:
            result["featureName"] = feature_name
    elif raw not in (None, "", [], {}):
        domain = _concept(raw, base_uri="http://codes.wmo.int/wmdr/Domain")
        if domain:
            result["domain"] = domain
    if "domain" not in result:
        derived = _observed_domain_from_observed_variable(_first_non_empty(obs.get("observedProperty"), obs.get("observedVariable")))
        if derived:
            result["domain"] = {"id": derived}
    return result or None


# Historical helper name retained.
def _observed_domain_object(obs: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    return _observed_feature(obs)


def _observation_id(obs: Mapping[str, Any], index: int, observed_property: Any, observed_geometry: Any) -> str:
    raw = _first_non_empty(obs.get("id"), obs.get("uid"))
    if raw not in (None, "", [], {}):
        identifier = _sanitize_id(raw)
        for prefix in ("observationSeries:", "observation:"):
            if identifier.startswith(prefix):
                identifier = _remove_prefix(identifier, prefix)
        return identifier
    variable = _last_segment(observed_property) or str(index + 1)
    geometry = _last_segment(observed_geometry)
    return f"{variable}-{_slug(geometry)}" if geometry else variable


def _observation_from_source(
    obs: Mapping[str, Any],
    index: int,
    deployments: Sequence[Any],
    programme_index: Mapping[str, List[Dict[str, Any]]],
    instrument_registry: Dict[str, Dict[str, Any]],
    contact_registry: Dict[str, Dict[str, Any]],
    schedule_registry: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    observed_property = _first_non_empty(obs.get("observedProperty"), obs.get("observedVariable"))
    observed_geometry = _first_non_empty(obs.get("observedGeometry"), obs.get("type"))
    identifier = _observation_id(obs, index, observed_property, observed_geometry)
    result: Dict[str, Any] = {"id": identifier}

    title = _strip_text(obs.get("title")) or _format_observation_title(observed_property, observed_geometry)
    if title:
        result["title"] = title
    description = _strip_text(obs.get("description"))
    if description:
        result["description"] = description

    property_concept = _concept(observed_property)
    geometry_concept = _concept(
        observed_geometry,
        base_uri="http://codes.wmo.int/wmdr/Geometry",
    )
    if property_concept:
        result["observedProperty"] = property_concept
    if geometry_concept:
        result["observedGeometry"] = geometry_concept
    feature = _observed_feature(obs)
    if feature:
        result["observedFeature"] = feature

    local_deployments = _as_list(_first_non_empty(obs.get("deployments"), obs.get("deployment"), obs.get("deploymentRefs")))
    application_values: List[Dict[str, Any]] = []
    application_values.extend(_concepts(obs.get("applicationAreas") or obs.get("applicationArea"), "applicationArea", base_uri="http://codes.wmo.int/wmdr/ApplicationArea"))
    for dep in local_deployments:
        application_values.extend(_concepts(_as_mapping(dep).get("applicationArea"), "applicationArea", base_uri="http://codes.wmo.int/wmdr/ApplicationArea"))
    if application_values:
        result["applicationAreas"] = _uniq_dicts(application_values)

    representativeness = _concept(obs.get("representativeness"), "representativeness", base_uri="http://codes.wmo.int/wmdr/Representativeness")
    if representativeness:
        result["representativeness"] = representativeness

    affiliations = _observation_program_affiliations(obs, programme_index)
    if affiliations:
        result["programAffiliations"] = affiliations

    raw_configs = _as_list(_first_non_empty(obs.get("configurations"), obs.get("observingConfigurations")))
    config_sources: List[Mapping[str, Any]] = []
    if raw_configs:
        config_sources = [_as_mapping(item) for item in raw_configs if _as_mapping(item)]
    elif local_deployments:
        config_sources = [_as_mapping(item) for item in local_deployments if _as_mapping(item)]
    elif deployments:
        refs = {str(item) for item in _as_list(obs.get("deploymentRefs") or obs.get("deployment")) if not isinstance(item, Mapping)}
        for dep in deployments:
            dep_obj = _as_mapping(dep)
            dep_id = str(_first_non_empty(dep_obj.get("id"), dep_obj.get("identifier"), dep_obj.get("uid"), ""))
            if not refs or dep_id in refs:
                config_sources.append(dep_obj)
    else:
        config_sources = [obs]

    configurations: List[Dict[str, Any]] = []
    cfg_ordinal = 0
    for source in config_sources:
        variants = _configuration_source_variants(source)
        for split_index, variant in enumerate(variants):
            cfg_id = _configuration_id(
                variant,
                ordinal=cfg_ordinal,
                parent_id=identifier,
                split_ordinal=(split_index + 1) if len(variants) > 1 else 0,
            )
            configurations.append(
                _configuration_from_source(
                    variant, instrument_registry, contact_registry,
                    configuration_id=cfg_id,
                )
            )
            cfg_ordinal += 1
    if configurations:
        result["configurations"] = _uniq_dicts(configurations)

    reporting_sources: List[Mapping[str, Any]] = []
    for item in _as_list(obs.get("dataGeneration")):
        if _as_mapping(item):
            reporting_sources.append(_as_mapping(item))
    for dep in local_deployments:
        for item in _as_list(_as_mapping(dep).get("dataGeneration")):
            if _as_mapping(item):
                reporting_sources.append(_as_mapping(item))
    if not reporting_sources:
        reporting_sources = [obs]

    observing_procedures: List[Dict[str, Any]] = []
    reporting_procedures: List[Dict[str, Any]] = []
    for source in reporting_sources:
        observing = _observing_procedure_from_source(source, schedule_registry)
        reporting = _reporting_procedure_from_source(source, contact_registry, schedule_registry)
        if observing:
            observing_procedures.append(observing)
        if reporting:
            reporting_procedures.append(reporting)
    if observing_procedures:
        result["observingProcedures"] = _uniq_dicts(observing_procedures)
    if reporting_procedures:
        result["reportingProcedures"] = _uniq_dicts(reporting_procedures)

    assignments: List[Dict[str, Any]] = []
    for key in ("contact", "contacts", "responsibleParty", "operator"):
        assignments.extend(_contact_assignments(obs.get(key), contact_registry, key if key not in {"contact", "contacts"} else None))
    metadata = _as_mapping(obs.get("metadata"))
    assignments.extend(_contact_assignments(metadata.get("contact"), contact_registry))
    if assignments:
        result["contactAssignments"] = _uniq_dicts(assignments)

    links = _extract_links(obs, "observation")
    if links:
        result["links"] = links
    keywords = _keywords_from_values(_collect_discovery_values("observation", obs, "keywords"))
    if keywords:
        result["keywords"] = keywords
    return cast(Dict[str, Any], _clean_none(result))


# Backward-compatible alias; output is now an Observation, not ObservationSeries.
def _observation_series_from_source(
    obs: Mapping[str, Any],
    index: int,
    deployments: Sequence[Any],
    instrument_registry: Dict[str, Dict[str, Any]],
    contact_registry: Dict[str, Dict[str, Any]],
    schedule_registry: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return _observation_from_source(obs, index, deployments, {}, instrument_registry, contact_registry, schedule_registry)


# ---------------------------------------------------------------------------
# Facility construction
# ---------------------------------------------------------------------------


def _split_source(source: Any) -> Tuple[Dict[str, Any], Dict[str, Any], List[Any], List[Any]]:
    if not isinstance(source, Mapping):
        return {}, {}, [], []
    if source.get("type") == "Feature" and isinstance(source.get("properties"), Mapping):
        props = _as_dict(source.get("properties"))
        return props, {}, _as_list(props.get("observations") or props.get("observationSeries")), _as_list(props.get("deployments"))
    facility = _as_dict(source.get("facility"))
    header = _as_dict(source.get("header"))
    observations = _as_list(_first_non_empty(source.get("observations"), source.get("observationSeries"), source.get("observation")))
    deployments = _as_list(_first_non_empty(source.get("deployments"), source.get("deployment"), source.get("configurations"), source.get("observingConfigurations")))
    if not facility:
        domain_keys = {"observedVariable", "observedProperty", "sourceOfObservation", "manufacturer", "serialNumber", "fileDateTime"}
        if not any(key in source for key in domain_keys):
            facility = dict(source)
    return facility, header, observations, deployments


def _facility_identifier(facility: Mapping[str, Any], header: Mapping[str, Any]) -> str:
    for raw in (
        facility.get("identifier"), facility.get("wigosStationIdentifier"), facility.get("wigosIdentifier"),
        facility.get("wsi"), facility.get("id"), header.get("wigosStationIdentifier"), header.get("identifier"), header.get("id"),
    ):
        values = _facility_wsi_values(raw)
        if values:
            return values[0]
    return _normalize_facility_wsi(_first_non_empty(facility.get("identifier"), facility.get("id"), header.get("identifier"), header.get("id")))


def _title_values(value: Any) -> List[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, Mapping):
        values: List[str] = []
        for key in ("title", "name", "value", "text", "#text"):
            values.extend(_title_values(value.get(key)))
        return cast(List[str], _uniq_scalars(values))
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_title_values(item))
        return cast(List[str], _uniq_scalars(values))
    text = _strip_text(str(value))
    return [text] if text else []


def _description_text(value: Any) -> Optional[str]:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, Mapping):
        return _strip_text(_first_non_empty(value.get("description"), value.get("value"), value.get("text"), value.get("#text"), value.get("remarks")))
    if isinstance(value, list):
        parts = [text for item in value if (text := _description_text(item))]
        return "\n\n".join(cast(List[str], _uniq_scalars(parts))) if parts else None
    return _strip_text(str(value))


def _record_timestamps(header: Mapping[str, Any], *, source_name: Optional[str] = None) -> Dict[str, str]:
    created = _normalize_record_datetime(_first_non_empty(
        header.get("created"), header.get("dateCreated"), header.get("creationDate"),
        header.get("fileDateTime"), header.get("dateStamp"), source_name,
    ))
    updated = _normalize_record_datetime(_first_non_empty(
        header.get("updated"), header.get("dateUpdated"), header.get("updateDate"),
        header.get("modified"), header.get("fileDateTime"), header.get("dateStamp"), created,
    ))
    result: Dict[str, str] = {}
    if created:
        result["created"] = created
    if updated:
        result["updated"] = updated
    return result


def _collect_root_contacts(facility: Mapping[str, Any], header: Mapping[str, Any], registry: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    assignments: List[Dict[str, Any]] = []
    for key in ("contact", "contacts", "responsibleParty"):
        assignments.extend(_contact_assignments(facility.get(key), registry))
    assignments.extend(_contact_assignments(header.get("recordOwner"), registry, "owner"))
    assignments.extend(_contact_assignments(facility.get("owner"), registry, "owner"))
    assignments.extend(_contact_assignments(facility.get("operator"), registry, "operator"))
    return _uniq_dicts(assignments)


def build_facility_feature(
    source: Optional[Any] = None,
    *,
    facility: Optional[Mapping[str, Any]] = None,
    header: Optional[Mapping[str, Any]] = None,
    observations: Optional[Sequence[Any]] = None,
    deployments: Optional[Sequence[Any]] = None,
    source_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one WMDR2 Facility GeoJSON Feature from WMDR1 JSON."""
    if isinstance(source, Mapping) and source.get("type") == "Feature":
        return cast(Dict[str, Any], normalize_wmdr2_record(copy.deepcopy(dict(source))))

    src_facility, src_header, src_observations, src_deployments = _split_source(source) if source is not None else ({}, {}, [], [])
    facility_obj = dict(facility or src_facility)
    header_obj = dict(header or src_header)
    observation_items = list(observations if observations is not None else src_observations)
    deployment_items = list(deployments if deployments is not None else src_deployments)

    wsi = _facility_identifier(facility_obj, header_obj)
    geometry_entries = _facility_temporal_geometry_entries(facility_obj)
    geometry = _facility_geometry_from_entries(geometry_entries)
    contact_registry: Dict[str, Dict[str, Any]] = {}
    instrument_registry: Dict[str, Dict[str, Any]] = {}
    schedule_registry: Dict[str, Dict[str, Any]] = {}

    titles = _uniq_scalars(_title_values(facility_obj.get("name")) + _title_values(facility_obj.get("title")))
    primary_title = str(titles[0]) if titles else wsi
    additional_titles = [str(title) for title in titles[1:] if title != primary_title]
    external_ids, programme_titles = _external_ids_and_titles(facility_obj)
    additional_titles = cast(List[str], _uniq_scalars([*additional_titles, *programme_titles]))

    properties: Dict[str, Any] = {
        "type": "facility",
        "title": primary_title,
        **_record_timestamps(header_obj, source_name=source_name),
    }
    if additional_titles:
        properties["additionalTitles"] = additional_titles
    if external_ids:
        properties["externalIds"] = external_ids
    description = _description_text(facility_obj.get("description"))
    if description:
        properties["description"] = description

    if _non_empty(facility_obj.get("facilityType")):
        properties["facilityType"] = _concept(facility_obj.get("facilityType"), allow_null=True, base_uri="http://codes.wmo.int/wmdr/FacilityType")
    if _non_empty(facility_obj.get("wmoRegion")):
        properties["wmoRegion"] = _wmo_region_concept(
            facility_obj.get("wmoRegion")
        )

    territory = _normalize_territories(_first_non_empty(facility_obj.get("territories"), facility_obj.get("territory")))
    if territory:
        properties["territories"] = territory

    wsi_candidates: List[str] = []
    for raw in (
        facility_obj.get("identifier"), facility_obj.get("wigosStationIdentifier"), facility_obj.get("wigosIdentifier"),
        facility_obj.get("wsi"), facility_obj.get("id"), header_obj.get("wigosStationIdentifier"), header_obj.get("identifier"), header_obj.get("id"),
    ):
        wsi_candidates.extend(_facility_wsi_values(raw))
    additional_ids = [candidate for candidate in _uniq_scalars(wsi_candidates) if candidate != wsi]
    if additional_ids:
        properties["additionalIds"] = additional_ids

    environment = _environment_from_facility(facility_obj)
    if environment:
        properties["environment"] = environment

    root_assignments = _collect_root_contacts(facility_obj, header_obj, contact_registry)
    if root_assignments:
        properties["contactAssignments"] = root_assignments

    programme_index = _programme_index(facility_obj)
    converted_observations: List[Dict[str, Any]] = []
    for index, raw in enumerate(observation_items):
        obs = _as_mapping(raw)
        if not obs:
            continue
        converted_observations.append(
            _observation_from_source(
                obs, index, deployment_items, programme_index,
                instrument_registry, contact_registry, schedule_registry,
            )
        )
    if converted_observations:
        properties["observations"] = _uniq_dicts(converted_observations)

    # Register unreferenced deployment instrument types without serial numbers.
    for raw in deployment_items:
        dep = _as_mapping(raw)
        instrument = _instrument_from_source(dep) or _instrument_from_source(_equipment_from_deployment(dep))
        if instrument:
            identifier = cast(str, instrument["id"])
            instrument_registry[identifier] = _merge_instrument(instrument_registry.get(identifier, {}), instrument)

    if instrument_registry:
        properties["instruments"] = sorted(instrument_registry.values(), key=lambda item: str(item.get("id")))
    if schedule_registry:
        properties["schedules"] = sorted(schedule_registry.values(), key=lambda item: str(item.get("uid")))
    if contact_registry:
        properties["contacts"] = sorted(contact_registry.values(), key=lambda item: str(item.get("identifier")))

    keywords = _keywords_from_values(_collect_discovery_values("facility", facility_obj, "keywords"))
    if keywords:
        properties["keywords"] = keywords

    feature: Dict[str, Any] = {
        "type": "Feature",
        "id": wsi,
        "conformsTo": [WMDR2_CORE_CONF],
        "geometry": geometry,
        "properties": properties,
    }
    start, end = _extract_interval(facility_obj)
    time = _time_interval(start, end, resolution="P1D")
    if time:
        feature["time"] = time
    temporal_geometry = _temporal_geometry_extension(geometry_entries)
    if temporal_geometry:
        feature["temporalGeometry"] = temporal_geometry
    links = _extract_links(facility_obj, "facility")
    if links:
        feature["links"] = links

    return cast(Dict[str, Any], _restore_explicit_nulls(_clean_none(feature)))


# ---------------------------------------------------------------------------
# Normalization of already generated WMDR2 features
# ---------------------------------------------------------------------------


def _conceptize_existing(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping) and "id" in value:
        return dict(value)
    return _concept(value, allow_null=True)


def _normalize_existing_configuration(cfg: Mapping[str, Any], ordinal: int, parent_id: str) -> Dict[str, Any]:
    result = copy.deepcopy(dict(cfg))
    if "serialNumber" in result and "instrumentSerialNumber" not in result:
        result["instrumentSerialNumber"] = result.pop("serialNumber")
    if "id" not in result:
        result["id"] = _configuration_id(result, ordinal=ordinal, parent_id=parent_id)
    for key in ("observingMethod", "operatingStatus", "sourceOfObservation", "exposure"):
        if key in result and not (isinstance(result[key], Mapping) and "id" in result[key]):
            result[key] = _conceptize_existing(result[key])

    if "verticalDistance" not in result:
        legacy_distance = result.pop("verticalDistanceFromReferenceSurface", None)
        legacy_reference = result.pop("referenceSurface", None)
        if legacy_distance not in (None, "", [], {}):
            source = dict(result)
            source["verticalDistanceFromReferenceSurface"] = legacy_distance
            if legacy_reference is not None:
                source["referenceSurface"] = legacy_reference
            vertical = _vertical_distance(source)
            if vertical:
                result["verticalDistance"] = vertical
    return cast(Dict[str, Any], _clean_none(result))


def _normalize_existing_observation(obs: Mapping[str, Any], index: int) -> Dict[str, Any]:
    result = copy.deepcopy(dict(obs))
    if "id" in result and isinstance(result["id"], str):
        result["id"] = _remove_prefix(_remove_prefix(result["id"], "observationSeries:"), "observation:")
    elif "id" not in result:
        result["id"] = _observation_id(result, index, result.get("observedProperty"), result.get("observedGeometry"))
    for key in ("observedProperty", "observedGeometry", "representativeness"):
        if key in result and not (isinstance(result[key], Mapping) and "id" in result[key]):
            result[key] = _conceptize_existing(result[key])
    feature = result.get("observedFeature")
    if isinstance(feature, Mapping):
        feature_copy = dict(feature)
        for key in ("domain", "domainFeature"):
            if key in feature_copy and not (isinstance(feature_copy[key], Mapping) and "id" in feature_copy[key]):
                feature_copy[key] = _conceptize_existing(feature_copy[key])
        result["observedFeature"] = feature_copy

    old_affiliations = result.get("programAffiliations")
    if isinstance(old_affiliations, list) and old_affiliations and not isinstance(old_affiliations[0], Mapping):
        result["programAffiliations"] = [{"programAffiliation": _concept(item)} for item in old_affiliations if _concept(item)]
    elif isinstance(old_affiliations, list):
        normalized_affiliations: List[Dict[str, Any]] = []
        for item in old_affiliations:
            obj = _as_mapping(item)
            if not obj:
                continue
            affiliation = dict(obj)
            if "program" in affiliation and "programAffiliation" not in affiliation:
                affiliation["programAffiliation"] = affiliation.pop("program")
            if "programAffiliation" in affiliation and not (isinstance(affiliation["programAffiliation"], Mapping) and "id" in affiliation["programAffiliation"]):
                affiliation["programAffiliation"] = _conceptize_existing(affiliation["programAffiliation"])
            if "reportingStatus" in affiliation and not (isinstance(affiliation["reportingStatus"], Mapping) and "id" in affiliation["reportingStatus"]):
                affiliation["reportingStatus"] = _conceptize_existing(affiliation["reportingStatus"])
            if "time" in affiliation and "dates" not in affiliation:
                start, end = _extract_interval(affiliation)
                dates = _dates(start, end)
                affiliation.pop("time", None)
                if dates:
                    affiliation["dates"] = dates
            normalized_affiliations.append(cast(Dict[str, Any], _clean_none(affiliation)))
        result["programAffiliations"] = normalized_affiliations

    configs = _first_non_empty(result.pop("observingConfigurations", None), result.get("configurations"))
    if configs is not None:
        result["configurations"] = [
            _normalize_existing_configuration(_as_mapping(item), cfg_index, str(result.get("id", index + 1)))
            for cfg_index, item in enumerate(_as_list(configs)) if _as_mapping(item)
        ]
    result.pop("time", None)  # Observation temporal extent is derived.
    return cast(Dict[str, Any], _clean_none(result))


def normalize_wmdr2_record(record: Any) -> Any:
    """Normalize an existing transitional/development WMDR2 Feature."""
    if not isinstance(record, dict):
        return record
    result = copy.deepcopy(record)
    if "id" in result:
        result["id"] = _normalize_facility_wsi(result["id"])
    props = _as_dict(result.get("properties"))
    if not props:
        return result

    if "territory" in props and "territories" not in props:
        props["territories"] = _normalize_territories(props.pop("territory"))
    if "facilityType" in props and not (
        isinstance(props["facilityType"], Mapping)
        and "id" in props["facilityType"]
    ):
        props["facilityType"] = _conceptize_existing(props["facilityType"])
    if "wmoRegion" in props:
        props["wmoRegion"] = _wmo_region_concept(props["wmoRegion"])

    old_observations = _first_non_empty(props.pop("observationSeries", None), props.get("observations"))
    if old_observations is not None:
        props["observations"] = [
            _normalize_existing_observation(_as_mapping(item), index)
            for index, item in enumerate(_as_list(old_observations)) if _as_mapping(item)
        ]

    # Old devt facility-level programme affiliations are no longer serialized.
    props.pop("programAffiliations", None)
    props.pop("programAffiliation", None)

    # OGC Records links are Feature-level, not Facility-properties members.
    prop_links = props.pop("links", None)
    if "links" not in result and prop_links:
        result["links"] = prop_links

    result["properties"] = props
    remaining = _find_source_temporal_keys(result)
    if remaining:
        raise ValueError("WMDR2 record still contains source temporal key(s): " + ", ".join(remaining[:20]))
    return cast(Dict[str, Any], _restore_explicit_nulls(_clean_none(result)))


def _find_source_temporal_keys(value: Any, *, path: Tuple[str, ...] = ()) -> List[str]:
    found: List[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = path + (str(key),)
            if key in SOURCE_TEMPORAL_KEYS:
                found.append("/" + "/".join(child_path))
            found.extend(_find_source_temporal_keys(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_source_temporal_keys(child, path=path + (str(index),)))
    return found


# Compatibility no-op: procedure/schedule structure is canonical at construction.
def _normalize_procedure_schedule_structure(record: Any) -> None:
    return None


def _normalize_facility_additional_ids(record: Any) -> Any:
    if not isinstance(record, dict):
        return record
    props = _as_mapping(record.get("properties"))
    if not isinstance(props, Mapping):
        return record
    candidates = _facility_wsi_values(props.get("additionalIds"))
    primary = _normalize_facility_wsi(record.get("id"))
    clean = [item for item in _uniq_scalars(candidates) if item != primary]
    if clean:
        cast(Dict[str, Any], record["properties"])["additionalIds"] = clean
    return record


# ---------------------------------------------------------------------------
# File/config orchestration
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit(f"Cannot read config file {path}: PyYAML is not installed.")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file {path} must contain a top-level YAML mapping.")
    return cast(Dict[str, Any], data)


def _discover_config_path(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit is not None:
        path = explicit.expanduser()
        return path if path.is_absolute() else Path.cwd() / path
    starts = [Path.cwd(), Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for start in starts:
        for folder in (start, *start.parents):
            for name in ("config.yaml", "config.yml"):
                candidate = folder / name
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate.is_file():
                    return candidate
    return None


def _cfg_section(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    section = cfg.get("convert_wmdr10_json_to_wmdr2_json")
    if isinstance(section, dict):
        return section
    alternate = cfg.get("convert_wmdr10_json_to_wmdr2_geojson")
    return alternate if isinstance(alternate, dict) else {}


def _cfg_first(section: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = section.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


def _resolve_path(value: Any, *, base_dir: Optional[Path], from_config: bool) -> Optional[Path]:
    if value in (None, "", [], {}):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if from_config and base_dir is not None:
        return base_dir / path
    return Path.cwd() / path


def _normalize_discovery_policy(section: Mapping[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    raw = section.get("discovery")
    if not isinstance(raw, Mapping):
        return copy.deepcopy(DEFAULT_DISCOVERY_POLICY)
    policy = copy.deepcopy(EMPTY_DISCOVERY_POLICY)
    aliases = {"observingConfiguration": "configuration"}
    for raw_name, cfg in raw.items():
        entity = aliases.get(str(raw_name), str(raw_name))
        if entity not in policy or not isinstance(cfg, Mapping):
            continue
        for bucket in ("keywords", "links"):
            values = cfg.get(bucket)
            if isinstance(values, list):
                policy[entity][bucket] = [str(value).strip() for value in values if _strip_text(str(value))]
    return policy


def _load_code_list_labels(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            domain = _first_non_empty(row.get("domain"), row.get("codeList"), row.get("list"))
            code = _first_non_empty(row.get("code"), row.get("identifier"), row.get("id"), row.get("notation"))
            label = _first_non_empty(row.get("label"), row.get("prefLabel"), row.get("title"))
            if all(isinstance(value, str) for value in (domain, code, label)):
                CODE_LIST_LABELS.setdefault(cast(str, domain), {})[cast(str, code).lstrip("_")] = cast(str, label)


def _detect_kind(path: Path, payload: Any) -> str:
    stem = path.stem.lower()
    if stem.endswith("_facility"):
        return "facility"
    if stem.endswith("_header"):
        return "header"
    if stem.endswith("_observations"):
        return "observations"
    if stem.endswith("_deployments"):
        return "deployments"
    if isinstance(payload, Mapping):
        if payload.get("type") == "Feature" and isinstance(payload.get("properties"), Mapping):
            return "feature"
        if any(key in payload for key in ("facility", "observations", "observationSeries", "deployments", "header")):
            return "full"
        if any(key in payload for key in ("observedVariable", "observedProperty", "resultTime")):
            return "observations"
        if any(key in payload for key in ("sourceOfObservation", "deployedEquipment", "manufacturer", "serialNumber", "referenceSurface")):
            return "deployments"
    return "unknown"


def _part_group_key(path: Path) -> str:
    for suffix in ("_header", "_facility", "_observations", "_deployments"):
        if path.stem.lower().endswith(suffix):
            return path.stem[:-len(suffix)]
    return path.stem


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any], *, announce: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if announce and ANNOUNCE_WRITES:
        print(f"wrote {path}")


def convert_record(record: Any, *, source_name: Optional[str] = None) -> Dict[str, Any]:
    return build_facility_feature(record, source_name=source_name)


def convert_payload(payload: Any, *, source_name: Optional[str] = None) -> Dict[str, Any]:
    return convert_record(payload, source_name=source_name)


def convert_wmdr10_json_to_wmdr2_json(payload: Any) -> Dict[str, Any]:
    return convert_record(payload)


def convert_file(input_path: Path, output_path: Path) -> Path:
    _write_json(output_path, convert_record(_load_json(input_path), source_name=input_path.name))
    return output_path


def _iter_json_files(root: Path, *, pattern: str, recursive: bool) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".json" else []
    walker = root.rglob if recursive else root.glob
    return sorted(path for path in walker(pattern) if path.is_file() and path.suffix.lower() == ".json")


def _write_group_output(group: str, parts: Mapping[str, Any], output_root: Path) -> Path:
    result = build_facility_feature(
        None,
        facility=_as_mapping(parts.get("facility")),
        header=_as_mapping(parts.get("header")),
        observations=_as_list(parts.get("observations")),
        deployments=_as_list(parts.get("deployments")),
        source_name=f"{group}.json",
    )
    output = output_root / f"{group}{OUTPUT_SUFFIX}"
    _write_json(output, result)
    return output


def convert_path(input_path: Path, output_path: Path, *, pattern: str = DEFAULT_PATTERN, recursive: bool = True, verbose: bool = False) -> List[Path]:
    files = _iter_json_files(input_path, pattern=pattern, recursive=recursive)
    if input_path.is_file():
        target = output_path if output_path.suffix.lower() == ".json" else output_path / input_path.name
        return [convert_file(input_path, target)]

    output_path.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, Dict[str, Any]] = {}
    standalone: List[Path] = []
    for path in files:
        payload = _load_json(path)
        kind = _detect_kind(path, payload)
        if kind in {"header", "facility", "observations", "deployments"} and path.stem.lower().endswith(("_header", "_facility", "_observations", "_deployments")):
            grouped.setdefault(_part_group_key(path), {})[kind] = payload
        else:
            standalone.append(path)

    written: List[Path] = []
    for path in standalone:
        target = output_path / path.relative_to(input_path)
        written.append(convert_file(path, target.with_suffix(".json")))
        if verbose:
            print(f"converted {path} -> {target}")
    for group, parts in sorted(grouped.items()):
        written.append(_write_group_output(group, parts, output_path))
    return written


def _collect_catalogue_items(record_paths: Iterable[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    contacts: List[Mapping[str, Any]] = []
    instruments: List[Mapping[str, Any]] = []
    for path in record_paths:
        try:
            props = _as_mapping(_as_mapping(_load_json(path)).get("properties"))
        except Exception:
            continue
        contacts.extend(_as_mapping(item) for item in _as_list(props.get("contacts")) if _as_mapping(item))
        instruments.extend(_as_mapping(item) for item in _as_list(props.get("instruments")) if _as_mapping(item))
    return _uniq_dicts(contacts), _uniq_dicts(instruments)


def _remove_inline_catalogue_items(record_paths: Iterable[Path]) -> None:
    for path in record_paths:
        try:
            record = _load_json(path)
        except Exception:
            continue
        props = _as_dict(_as_mapping(record).get("properties"))
        props.pop("contacts", None)
        props.pop("instruments", None)
        record["properties"] = props
        _write_json(path, record)


def _write_catalogue_outputs(record_paths: Iterable[Path], *, contacts_path: Optional[Path], instruments_path: Optional[Path], remove_inline: bool = True) -> None:
    contacts, instruments = _collect_catalogue_items(record_paths)
    if contacts_path:
        _write_json(contacts_path, {"contacts": contacts})
    if instruments_path:
        _write_json(instruments_path, {"instruments": instruments})
    if remove_inline:
        _remove_inline_catalogue_items(record_paths)


def _load_code_list_labels_from_config(
    section: Mapping[str, Any],
    *,
    base_dir: Optional[Path],
    cli_path: Optional[Path],
) -> None:
    if cli_path is not None:
        _load_code_list_labels(cli_path)
        return
    simple = _cfg_first(section, "code_list_labels", "codeListLabelsCsv")
    simple_path = _resolve_path(simple, base_dir=base_dir, from_config=True)
    if simple_path is not None:
        _load_code_list_labels(simple_path)
    nested = _as_mapping(section.get("codeListLabels"))
    for item in _as_list(nested.get("files")):
        path = _resolve_path(_as_mapping(item).get("path"), base_dir=base_dir, from_config=True)
        if path is not None:
            _load_code_list_labels(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert simplified WMDR1 JSON to current WMDR2 JSON.")
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--input", dest="input_opt")
    parser.add_argument("--output", dest="output_opt")
    parser.add_argument("--source", dest="source_opt")
    parser.add_argument("--target", dest="target_opt")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--code-list-labels", type=Path)
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_path = _discover_config_path(args.config)
    section: Dict[str, Any] = {}
    config_dir: Optional[Path] = None
    if config_path:
        config_dir = config_path.parent
        section = _cfg_section(_load_config(config_path))
    elif not any((args.input_opt, args.source_opt, args.input, args.output_opt, args.target_opt, args.output)):
        parser.error("missing config.yaml; run from the repository root or provide input and output paths")

    global DISCOVERY_POLICY
    DISCOVERY_POLICY = _normalize_discovery_policy(section)
    _load_code_list_labels_from_config(
        section,
        base_dir=config_dir,
        cli_path=args.code_list_labels,
    )

    cli_input = args.input_opt or args.source_opt or args.input
    cli_output = args.output_opt or args.target_opt or args.output
    config_input = _cfg_first(section, "source", "input", "input_path")
    config_output = _cfg_first(section, "target", "output", "output_path")
    input_path = _resolve_path(cli_input or config_input, base_dir=config_dir, from_config=cli_input is None)
    output_path = _resolve_path(cli_output or config_output, base_dir=config_dir, from_config=cli_output is None)
    if input_path is None or output_path is None:
        parser.error("missing input/output path")

    pattern = args.pattern or str(_cfg_first(section, "pattern") or DEFAULT_PATTERN)
    recursive = False if args.no_recursive else bool(section.get("recursive", True))
    written = convert_path(
        input_path,
        output_path,
        pattern=pattern,
        recursive=recursive,
        verbose=args.verbose,
    )

    catalogues = _as_mapping(section.get("catalogues"))
    if catalogues.get("enabled"):
        records_path = _resolve_path(_cfg_first(catalogues, "records_path", "recordsPath"), base_dir=config_dir, from_config=True) or output_path
        catalogue_records = (
            written
            if records_path.resolve() == output_path.resolve()
            else convert_path(
                input_path,
                records_path,
                pattern=pattern,
                recursive=recursive,
                verbose=args.verbose,
            )
        )
        contacts_path = _resolve_path(_cfg_first(catalogues, "contacts_path", "contactsPath"), base_dir=config_dir, from_config=True)
        instruments_path = _resolve_path(_cfg_first(catalogues, "instruments_path", "instrumentsPath"), base_dir=config_dir, from_config=True)
        _write_catalogue_outputs(catalogue_records, contacts_path=contacts_path, instruments_path=instruments_path, remove_inline=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
