#!/usr/bin/env python3
"""Convert simplified WMDR 1.0 JSON records into WMDR2 v0.3.1 JSON.

This script is both a command-line converter and an importable module.  It reads
one simplified WMDR10 JSON file, an already generated WMDR2 Feature, or a
directory of JSON files.  Directory conversion supports grouped input files with
suffixes such as ``_header.json``, ``_facility.json``, ``_observations.json``
and ``_deployments.json``.

Configuration is the default execution mode for repository use.  Running
``python convert_wmdr10_json_to_wmdr2_json.py`` without positional arguments
loads ``config.yaml`` or ``config.yml`` from the current directory or one of its
parents.  The converter reads the
``convert_wmdr10_json_to_wmdr2_json`` section and uses ``source`` and ``target``
as the default input and output paths.  The aliases ``input``/``input_path`` and
``output``/``output_path`` are also accepted for compatibility.  Paths from the
configuration file are resolved relative to the configuration file location.
Command-line arguments remain available for ad-hoc conversion and override the
configuration.  PyYAML is required for the normal no-argument configuration
workflow.  Every file write is reported on stdout, so normal runs show that
records and catalogue files are actually being produced.

The public API is:

* ``build_facility_feature(...)`` builds one WMDR2 GeoJSON Feature;
* ``normalize_wmdr2_record(...)`` normalizes an already produced WMDR2 record;
* ``convert_record(...)`` converts one in-memory object;
* ``convert_file(...)`` converts one file;
* ``convert_path(...)`` converts a file or directory;
* ``main(...)`` implements the command-line interface.

The v0.3.1 output conventions implemented here are:

* Feature ``id`` is the bare WSI only, without any namespace prefix.
* Reusable contacts are OGC Contact objects in ``properties.contacts``.
* Context-specific contact-role use is represented as ``contactAssignments``.
* Contact e-mails and phones are OGC-style objects with a required ``value``.
* ``time.resolution`` is an ISO 8601 duration such as ``P1D``.
* ``beginPosition``/``endPosition`` and ``validFrom``/``validTo`` are normalized to OGC Records-style ``time.interval``.
* ``observingLocation`` is not emitted as a JSON wrapper.  Its useful members are
  promoted into the surrounding observing configuration object.
* ``reportingProcedures`` implement the v0.3.1 UML ReportingProcedure
  attributes and are not time-bound.  Reporting cadence is represented by
  reusable schedules referenced through ``reportingSchedules``; source
  ``temporalReportingInterval`` values become schedule
  ``wmo.int:aggregationInterval``.  ``duration`` is reserved for diurnal
  coverage windows derived from coverage start/end times.
* ``observingProcedures`` are time-bound and reference reusable schedules
  through ``observingSchedules``.

The converter avoids inventing values when the source metadata is absent.
Unknown or nil WMDR10 fields are preserved as ``{"nilReason": ...}`` where
possible, rather than replaced with guessed values.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

try:
    import yaml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    yaml = None

VERSION = "0.3.1"
ANNOUNCE_WRITES = True
DEFAULT_PATTERN = "*.json"
OUTPUT_SUFFIX = ".json"
CANONICAL_SCHEDULE_START_DATE = "0001-01-01"
OGC_RECORD_CORE_CONF = "http://www.opengis.net/spec/ogcapi-records-1/1.0/conf/record-core"
WMDR2_CORE_CONF = "http://wigos.wmo.int/spec/wmdr/2/conf/core"
_NULL_SENTINEL = "__WMDR2_NULL__"
SOURCE_TEMPORAL_KEYS = {"beginPosition", "endPosition", "validFrom", "validTo"}

EMPTY_DISCOVERY_POLICY: Dict[str, Dict[str, List[str]]] = {
    "facility": {"keywords": [], "links": []},
    "observation": {"keywords": [], "links": []},
    "observingConfiguration": {"keywords": [], "links": []},
    "observingLocation": {"keywords": [], "links": []},
    "deployment": {"keywords": [], "links": []},
}

DEFAULT_DISCOVERY_POLICY: Dict[str, Dict[str, List[str]]] = {
    "facility": {"keywords": ["identifier", "name"], "links": ["onlineResource"]},
    "observation": {"keywords": [], "links": []},
    # Observation configurations are operational/technical history objects, not
    # discovery records.  They may carry links when configured, but they do not
    # emit keywords in v0.3.1.
    "observingConfiguration": {"keywords": [], "links": []},
    # Section aliases accepted for discovery-policy settings from older config files.
    "observingLocation": {"keywords": [], "links": []},
    "deployment": {"keywords": [], "links": []},
}

DISCOVERY_POLICY: Dict[str, Dict[str, List[str]]] = copy.deepcopy(DEFAULT_DISCOVERY_POLICY)
CODE_LIST_LABELS: Dict[str, Dict[str, str]] = {}

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return *value* as a mapping, or an empty mapping.

    This gives static analysis a non-optional mapping before ``.get`` is used.
    WMDR10 source fields are frequently absent, ``None`` or scalar values.
    """
    if isinstance(value, Mapping):
        return value
    return {}


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


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


def _uniq_dicts(items: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _clean_none(dict(item))
        if not isinstance(cleaned, dict) or not cleaned:
            continue
        payload = _stable_json(cleaned)
        if payload not in seen:
            seen.add(payload)
            out.append(cast(Dict[str, Any], cleaned))
    return out


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


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "value"


def _sanitize_id(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^A-Za-z0-9._:/#-]+", "-", text)
    return text.strip("-") or "record"


def _clean_none(obj: Any, *, _path: Tuple[str, ...] = ()) -> Any:
    """Remove empty members while preserving explicit JSON null values.

    JSON ``null`` can be meaningful in WMDR2 two-value arrays, so callers that
    need to preserve nulls should use ``_preserve_nulls`` before this function
    and ``_restore_null_sentinel`` afterwards.
    """

    def preserve_empty_list(path: Tuple[str, ...]) -> bool:
        return len(path) >= 2 and path[-2:] in {
            ("temporalGeometry", "methods"),
            ("methods", "methods"),
        }

    if isinstance(obj, dict):
        cleaned = {k: _clean_none(v, _path=_path + (str(k),)) for k, v in obj.items()}
        return {k: v for k, v in cleaned.items() if v not in (None, "", [], {})}
    if isinstance(obj, list):
        cleaned_list = [_clean_none(v, _path=_path) for v in obj]
        return [v for v in cleaned_list if v not in ("", {}) and (v != [] or preserve_empty_list(_path))]
    return obj


def _preserve_nulls(obj: Any) -> Any:
    if obj is None:
        return _NULL_SENTINEL
    if isinstance(obj, dict):
        return {k: _preserve_nulls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_preserve_nulls(v) for v in obj]
    return obj


def _restore_null_sentinel(obj: Any) -> Any:
    if obj == _NULL_SENTINEL:
        return None
    if isinstance(obj, dict):
        return {k: _restore_null_sentinel(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_null_sentinel(v) for v in obj]
    return obj

# ---------------------------------------------------------------------------
# Date/time, codes and geometry
# ---------------------------------------------------------------------------


def _normalize_time_resolution(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    lower = text.lower()
    mapping = {
        "day": "P1D",
        "daily": "P1D",
        "d": "P1D",
        "hour": "PT1H",
        "hourly": "PT1H",
        "h": "PT1H",
        "minute": "PT1M",
        "min": "PT1M",
        "minutes": "PT1M",
        "second": "PT1S",
        "sec": "PT1S",
        "seconds": "PT1S",
    }
    if lower in mapping:
        return mapping[lower]
    match = re.fullmatch(r"(\d+)\s*(d|day|days)", lower)
    if match:
        return f"P{match.group(1)}D"
    match = re.fullmatch(r"(\d+)\s*(h|hour|hours)", lower)
    if match:
        return f"PT{match.group(1)}H"
    match = re.fullmatch(r"(\d+)\s*(m|min|minute|minutes)", lower)
    if match:
        return f"PT{match.group(1)}M"
    match = re.fullmatch(r"(\d+)\s*(s|sec|second|seconds)", lower)
    if match:
        return f"PT{match.group(1)}S"
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
        y, m, d = match.groups()
        return f"{y}-{m}-{d}"
    if re.fullmatch(r"\d{8}", text):
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}Z", text):
        return text[:-1]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    if re.match(r"^\d{4}-\d{2}-\d{2}T", text):
        return text[:10]
    return text


def _normalize_record_datetime(value: Any) -> Optional[str]:
    if value in (None, "", "None"):
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}T", text):
        return text if text.endswith("Z") else f"{text}Z"
    date = _normalize_date_value(text)
    if not date or date == "..":
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return f"{date}T00:00:00Z"
    return None


def _time_interval(start: Any, end: Any = None, *, resolution: Any = None) -> Optional[Dict[str, Any]]:
    s = _normalize_date_value(start)
    e = _normalize_date_value(end) or ".."
    if s is None and e == "..":
        return None
    out: Dict[str, Any] = {"interval": [s or "..", e]}
    normalized_resolution = _normalize_time_resolution(resolution)
    if normalized_resolution:
        out["resolution"] = normalized_resolution
    return out


def _extract_interval(obj: Mapping[str, Any]) -> Tuple[Any, Any]:
    time_obj = obj.get("time")
    if isinstance(time_obj, Mapping):
        interval = time_obj.get("interval")
        if isinstance(interval, list) and interval:
            return (interval[0] if len(interval) > 0 else None, interval[1] if len(interval) > 1 else None)
        return _first_non_empty(time_obj.get("date"), time_obj.get("timestamp")), None
    return (
        _first_non_empty(
            obj.get("validFrom"),
            obj.get("date"),
            obj.get("beginPosition"),
            obj.get("begin"),
            obj.get("from"),
            obj.get("start"),
            obj.get("dateEstablished"),
        ),
        _first_non_empty(obj.get("validTo"), obj.get("endPosition"), obj.get("end"), obj.get("stop"), obj.get("dateClosed")),
    )


def _entry_date(item: Any, fallback: str = "..") -> str:
    if isinstance(item, Mapping):
        return _normalize_date_value(
            _first_non_empty(
                item.get("validFrom"),
                item.get("date"),
                item.get("beginPosition"),
                item.get("begin"),
                item.get("from"),
                item.get("start"),
            )
        ) or fallback
    return fallback


def _is_unknown_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if text.startswith(("http://", "https://")):
        text = text.rstrip("/#").rsplit("/", 1)[-1]
    match = re.fullmatch(r"\(([^()]+)\)", text)
    if match:
        text = match.group(1)
    return text.strip().lower() in {"unknown", "none", "null", "nil"}


def _last_segment(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().strip("<>").rstrip("/#")
    match = re.fullmatch(r"\(([^()]+)\)", raw)
    if match:
        raw = match.group(1).strip()
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    elif "#" in raw:
        raw = raw.rsplit("#", 1)[-1]
    if _is_unknown_token(raw):
        return "unknown"
    return raw


def _normalize_code_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        value = _first_non_empty(value.get("href"), value.get("url"), value.get("value"), value.get("#text"), value.get("text"))
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    segment = _last_segment(text) or text
    if _is_unknown_token(segment):
        return "unknown"
    if re.fullmatch(r"[+-]?\d+", segment):
        try:
            return int(segment)
        except Exception:
            return segment
    return segment


def _compact_wmdr_code_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        value = _first_non_empty(value.get("href"), value.get("url"), value.get("value"), value.get("#text"), value.get("text"))
    if isinstance(value, str) and value.strip().startswith(("http://codes.wmo.int/wmdr/", "https://codes.wmo.int/wmdr/")):
        return _normalize_code_value(value)
    return value


def _nil_reason(reason: Any = "unknown") -> Dict[str, str]:
    normalized = _normalize_code_value(reason)
    text = str(normalized).strip() if _non_empty(normalized) else "unknown"
    return {"nilReason": text}


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


def _normalize_diurnal_time(value: Any) -> str:
    """Normalize WMDR diurnal base times to ``HH:MM:SS`` when possible.

    Some WMDR10 examples contain compact times such as ``7`` or ``7:5``.
    JSCalendar extension values in WMDR2 should be stable and comparable, so
    recognized numeric clock values are zero-padded and clamped to the valid
    24-hour range.  Non-clock values are returned unchanged.
    """
    text = str(value).strip()
    match = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?(?::(\d{1,2}))?Z?", text)
    if not match:
        return text
    hour = min(max(int(match.group(1)), 0), 23)
    minute = min(max(int(match.group(2) or 0), 0), 59)
    second = min(max(int(match.group(3) or 0), 0), 59)
    return f"{hour:02d}:{minute:02d}:{second:02d}"



def _parse_diurnal_seconds(value: Any) -> Optional[int]:
    """Return seconds after midnight for a compact WMDR clock value."""
    if value in (None, "", [], {}):
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?(?::(\d{1,2}))?Z?", text)
    if not match:
        return None
    hour = min(max(int(match.group(1)), 0), 23)
    minute = min(max(int(match.group(2) or 0), 0), 59)
    second = min(max(int(match.group(3) or 0), 0), 59)
    return hour * 3600 + minute * 60 + second


def _format_diurnal_seconds(seconds: int) -> str:
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 60
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _iso_duration_from_seconds(seconds: int) -> Optional[str]:
    """Return an ISO 8601 duration for a positive number of seconds."""
    if seconds <= 0:
        return None
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days and not (hours or minutes or seconds):
        return f"P{days}D"
    parts = []
    if hours:
        parts.append(f"{hours}H")
    if minutes:
        parts.append(f"{minutes}M")
    if seconds:
        parts.append(f"{seconds}S")
    if days:
        return f"P{days}DT{''.join(parts) or '0S'}"
    return f"PT{''.join(parts) or '0S'}"


def _coverage_time_seconds(source: Mapping[str, Any], prefix: str) -> Optional[int]:
    """Extract a start/end clock from WMDR coverage fields."""
    direct = _first_non_empty(source.get(f"{prefix}Time"), source.get(f"{prefix}ClockTime"))
    parsed = _parse_diurnal_seconds(direct)
    if parsed is not None:
        return parsed
    hour = _first_non_empty(source.get(f"{prefix}Hour"), source.get(f"{prefix}Hours"))
    minute = _first_non_empty(source.get(f"{prefix}Minute"), source.get(f"{prefix}Minutes"), 0)
    second = _first_non_empty(source.get(f"{prefix}Second"), source.get(f"{prefix}Seconds"), 0)
    if hour in (None, "", [], {}):
        return None
    return _parse_diurnal_seconds(f"{hour}:{minute}:{second}")


def _diurnal_coverage_fields(*sources: Mapping[str, Any]) -> Dict[str, str]:
    """Build JSCalendar start/duration from WMDR diurnal coverage fields.

    The real-world validity period is kept on the procedure instance.  The
    reusable schedule uses the conventional dummy date and records only the
    within-day coverage window, e.g. ``0001-01-01T06:00:00`` plus ``PT12H``.
    """
    for source in sources:
        if not source:
            continue
        start_seconds = _coverage_time_seconds(source, "start")
        end_seconds = _coverage_time_seconds(source, "end")
        if start_seconds is None and end_seconds is None:
            continue
        if start_seconds is None:
            start_seconds = 0
        fields: Dict[str, str] = {"start": f"{CANONICAL_SCHEDULE_START_DATE}T{_format_diurnal_seconds(start_seconds)}"}
        if end_seconds is not None:
            adjusted_end = end_seconds
            if adjusted_end <= start_seconds:
                adjusted_end += 24 * 3600
            duration = _iso_duration_from_seconds(adjusted_end - start_seconds)
            if duration:
                fields["duration"] = duration
        return fields
    return {}

def _normalize_code_or_nil_reason(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("nilReason", "@nilReason"):
            if _non_empty(value.get(key)):
                return _nil_reason(value.get(key))
        if _parse_bool(value.get("nil")) is True or _parse_bool(value.get("@nil")) is True:
            return _nil_reason("unknown")
    normalized = _normalize_code_value(value)
    if normalized == "unknown" or _is_unknown_token(normalized):
        return _nil_reason("unknown")
    if _non_empty(normalized) or isinstance(normalized, bool):
        return normalized
    return None


WSI_PATTERN = r"(0|1|2|3)-([1-9]\d*)-([0-9]+)-([A-Za-z0-9._-]+)"


def _is_valid_wsi(value: Any) -> bool:
    return bool(re.fullmatch(WSI_PATTERN, str(value or "").strip()))


def _normalize_single_facility_wsi(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    while True:
        lowered = text.lower()
        changed = False
        for prefix in ("wsi:", "wigos:", "facility:", "record:", "station:", "id:"):
            if lowered.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
                break
        if not changed:
            break
    if "/" in text and re.search(r"\d+-\d+-\d+-[A-Za-z0-9._-]+$", text):
        text = text.rstrip("/").rsplit("/", 1)[-1]
    return text


def _facility_wsi_values(value: Any) -> List[str]:
    """Return syntactically valid WSI values from WMDR10 identifier shapes."""
    values: List[str] = []
    if value in (None, "", [], {}):
        return values
    if isinstance(value, Mapping):
        for key in ("identifier", "wigosStationIdentifier", "wigosIdentifier", "wsi", "id", "value", "text", "#text"):
            values.extend(_facility_wsi_values(value.get(key)))
        return _uniq_scalars(values)
    if isinstance(value, list):
        for item in value:
            values.extend(_facility_wsi_values(item))
        return _uniq_scalars(values)
    text = str(value).strip()
    for candidate in re.split(r"\s*[,;]\s*", text):
        normalized = _normalize_single_facility_wsi(candidate)
        if _is_valid_wsi(normalized):
            values.append(normalized)
    return _uniq_scalars(values)


def _normalize_facility_wsi(value: Any) -> str:
    # Some legacy XML-derived records concatenate multiple WIGOS identifiers
    # into one string.  The WMDR2 Feature id must be a single WSI, so keep the
    # first syntactically valid WSI.
    values = _facility_wsi_values(value)
    if values:
        return values[0]
    return _normalize_single_facility_wsi(value)


def _parse_pos_lon_lat_z(raw: Any) -> Optional[List[Any]]:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        coords_any = raw.get("coordinates")
        if isinstance(coords_any, list) and len(coords_any) >= 2:
            return coords_any
        for key in ("geoLocation", "pos", "value", "text", "geometry", "position"):
            val = raw.get(key)
            if isinstance(val, str):
                raw = val
                break
            if isinstance(val, Mapping):
                nested = _parse_pos_lon_lat_z(val)
                if nested:
                    return nested
    if not isinstance(raw, str):
        return None
    nums: List[float] = []
    for item in raw.replace(",", " ").split():
        try:
            nums.append(float(item))
        except Exception:
            pass
    if len(nums) < 2:
        return None
    # WMDR10 pos is generally lat lon z; GeoJSON is lon lat z.
    lat, lon = nums[0], nums[1]
    coords: List[Any] = [lon, lat]
    if len(nums) >= 3:
        z = nums[2]
        coords.append(int(round(z)) if abs(z - round(z)) < 1e-9 else z)
    return coords


def _geopositioning_methods(item: Any) -> List[str]:
    if not isinstance(item, Mapping):
        return []
    methods: List[str] = []
    for value in _as_list(item.get("geopositioningMethod")):
        compact = _compact_wmdr_code_value(value)
        if isinstance(compact, str) and compact.strip():
            methods.append(compact.strip())
    return sorted(dict.fromkeys(methods))


def _facility_temporal_geometry_entries(source: Mapping[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Tuple[str, str, Dict[str, Any]]] = []

    def add(item: Any) -> None:
        coords = _parse_pos_lon_lat_z(item)
        if coords is None:
            return
        date = _entry_date(item)
        entry: Dict[str, Any] = {"coordinates": coords, "date": date}
        methods = _geopositioning_methods(item)
        if methods:
            entry["methods"] = methods
        entries.append((date, _stable_json(coords), entry))

    for item in _as_list(source.get("geospatialLocation") or source.get("geometry")):
        add(item)
    for item in _as_list(source.get("geospatialLocationHistory") or source.get("geometryHistory") or source.get("historicalLocation")):
        add(item)

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for _, _, entry in sorted(entries, key=lambda row: (row[0] == "..", row[0], row[1])):
        marker = _stable_json(entry)
        if marker not in seen:
            seen.add(marker)
            out.append(entry)
    return out


def _temporal_geometry_extension(entries: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    coordinates: List[Any] = []
    dates: List[str] = []
    methods: List[List[str]] = []
    has_methods = False
    for entry in entries:
        coords = entry.get("coordinates")
        if not isinstance(coords, list):
            continue
        coordinates.append(coords)
        dates.append(str(entry.get("date") or entry.get("validFrom") or ".."))
        raw_methods = entry.get("methods")
        entry_methods = [m for m in raw_methods if isinstance(m, str) and m] if isinstance(raw_methods, list) else []
        if entry_methods:
            has_methods = True
        methods.append(entry_methods)
    if not coordinates:
        return None
    if len(coordinates) == 1 and not has_methods:
        return None
    out: Dict[str, Any] = {"type": "MovingPoint", "coordinates": coordinates, "dates": dates}
    if has_methods:
        out["methods"] = methods
    return out


def _facility_geometry_from_entries(entries: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    for entry in reversed(list(entries)):
        coords = entry.get("coordinates")
        if isinstance(coords, list) and len(coords) >= 2:
            return {"type": "Point", "coordinates": coords}
    return None


def _point_geometry_from_entry(entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    coords = entry.get("coordinates")
    if isinstance(coords, list) and len(coords) >= 2:
        return {"type": "Point", "coordinates": coords}
    return None


def _extract_code_list_ref(value: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if isinstance(value, Mapping):
        value = _first_non_empty(value.get("href"), value.get("url"), value.get("value"), value.get("#text"))
    if not isinstance(value, str):
        return None, None, None
    text = value.strip().strip("<>")
    if not text:
        return None, None, None
    if text.startswith(("http://", "https://")):
        parts = [p for p in text.rstrip("/#").split("/") if p]
        if len(parts) < 2:
            return text, None, None
        return text, parts[-2].lstrip("_"), parts[-1].lstrip("_")
    return None, None, text.lstrip("_")


def _observed_domain_from_observed_variable(value: Any) -> Optional[str]:
    _, domain, _ = _extract_code_list_ref(value)
    if not domain or not domain.startswith("ObservedVariable"):
        return None
    domain_name = domain.removeprefix("ObservedVariable").strip()
    if not domain_name:
        return None
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", domain_name).lower()


def _lookup_code_list_label(domain: Optional[str], code: Optional[str]) -> Optional[str]:
    if not domain or not code:
        return None
    return CODE_LIST_LABELS.get(domain, {}).get(code.lstrip("_"))


def _format_observation_title(value: Any, geometry_type: Any = None) -> Optional[str]:
    _, domain, code = _extract_code_list_ref(value)
    if not code:
        compact_value = _compact_wmdr_code_value(value)
        code = str(compact_value) if compact_value not in (None, "") else None
    if not code:
        return None
    label = _lookup_code_list_label(domain, code)
    domain_value = _observed_domain_from_observed_variable(value)
    geometry_value = _compact_wmdr_code_value(geometry_type)
    parts: List[str] = []
    if isinstance(domain_value, str) and domain_value:
        parts.append(f"domain: {domain_value}")
    if geometry_value not in (None, "", [], {}):
        parts.append(f"geometry: {geometry_value}")
    variable_text = f"variable: {code}"
    if label:
        variable_text = f"{variable_text} {label}"
    parts.append(variable_text)
    return "; ".join(parts)

# ---------------------------------------------------------------------------
# File/config helpers
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit(f"Cannot read config file {path}: PyYAML is not installed.")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise SystemExit(f"Cannot read config file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Config file {path} must contain a top-level YAML mapping.")
    return cast(Dict[str, Any], data)


def _walk_up_for_config(start: Path) -> List[Path]:
    base = start if start.is_dir() else start.parent
    candidates: List[Path] = []
    for folder in (base, *base.parents):
        candidates.extend([folder / "config.yaml", folder / "config.yml"])
    return candidates


def _discover_config_path(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit is not None:
        p = explicit.expanduser()
        return p if p.is_absolute() else Path.cwd() / p
    seen: set[Path] = set()
    for candidate in [*_walk_up_for_config(Path.cwd()), *_walk_up_for_config(Path(__file__).resolve().parent)]:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate.absolute()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _cfg_section(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    section = cfg.get("convert_wmdr10_json_to_wmdr2_json")
    if isinstance(section, dict):
        return section
    alternate = cfg.get("convert_wmdr10_json_to_wmdr2_geojson")
    return alternate if isinstance(alternate, dict) else {}


def _format_loaded_config_hint(config_path: Optional[Path], section: Mapping[str, Any]) -> str:
    if config_path is None:
        return "No config file found; using CLI arguments only."
    keys = sorted(section.keys()) if section else []
    return f"Using config: {config_path} ({', '.join(keys) if keys else 'no converter section keys'})"

def _cfg_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _cfg_first(section: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = section.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


def _resolve_cli_or_config_path(value: Any, *, base_dir: Optional[Path], from_config: bool) -> Optional[Path]:
    if value in (None, "", [], {}):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if from_config and base_dir is not None:
        return base_dir / path
    return Path.cwd() / path


def _resolve_config_path(value: Any, *, base_dir: Optional[Path]) -> Optional[Path]:
    return _resolve_cli_or_config_path(value, base_dir=base_dir, from_config=True)


def _normalize_discovery_policy(section: Mapping[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    raw = section.get("discovery")
    if not isinstance(raw, dict):
        return copy.deepcopy(DEFAULT_DISCOVERY_POLICY)
    policy = copy.deepcopy(EMPTY_DISCOVERY_POLICY)
    for entity in ("facility", "observation", "observingConfiguration", "observingLocation", "deployment"):
        entity_cfg = raw.get(entity)
        if not isinstance(entity_cfg, dict):
            continue
        for bucket in ("keywords", "links"):
            values = entity_cfg.get(bucket)
            if isinstance(values, list):
                policy[entity][bucket] = [str(v).strip() for v in values if isinstance(v, str) and str(v).strip()]
    if policy.get("deployment") and not policy.get("observingConfiguration"):
        policy["observingConfiguration"] = copy.deepcopy(policy["deployment"])
    if policy.get("observingLocation") and not policy.get("observingConfiguration"):
        policy["observingConfiguration"] = copy.deepcopy(policy["observingLocation"])
    return policy


def _iter_json_files(root: Path, *, pattern: str = DEFAULT_PATTERN, recursive: bool = True) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".json" else []
    if not root.is_dir():
        return []
    walker = root.rglob if recursive else root.glob
    return sorted(p for p in walker(pattern) if p.is_file() and p.suffix.lower() == ".json")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any], *, announce: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if announce and ANNOUNCE_WRITES:
        print(f"wrote {path}")


def _detect_kind(path: Path, payload: Any) -> str:
    stem = path.stem.lower()
    if stem.endswith("_facility"):
        return "facility"
    if stem.endswith("_header"):
        return "header"
    if stem.endswith("_observations"):
        return "observationSeries"
    if stem.endswith("_deployments"):
        return "deployments"
    if isinstance(payload, dict):
        if payload.get("type") == "Feature" and isinstance(payload.get("properties"), dict):
            return "feature"
        if any(k in payload for k in ("facility", "observationSeries", "observations", "deployments", "header")):
            return "full"
        if any(k in payload for k in ("observedVariable", "observedProperty", "resultTime")):
            return "observationSeries"
        if any(k in payload for k in ("sourceOfObservation", "manufacturer", "serialNumber", "referenceSurface", "observingLocation")):
            return "deployments"
        if any(k in payload for k in ("fileDateTime", "recordOwner", "dateStamp")):
            return "header"
        if any(k in payload for k in ("identifier", "wigosStationIdentifier", "name", "geospatialLocation")):
            return "facility"
    if isinstance(payload, list):
        first = next((x for x in payload if isinstance(x, dict)), None)
        if not first:
            return "unknown"
        if any(k in first for k in ("observedVariable", "observedProperty", "resultTime")):
            return "observationSeries"
        if any(k in first for k in ("sourceOfObservation", "manufacturer", "serialNumber", "referenceSurface", "observingLocation")):
            return "deployments"
    return "unknown"


def _part_group_key(path: Path) -> str:
    stem = path.stem
    for suffix in ("_header", "_facility", "_observations", "_deployments"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem

# ---------------------------------------------------------------------------
# Discovery, contacts and quantities
# ---------------------------------------------------------------------------


def _keywords_from_values(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        if raw is None:
            continue
        candidate = _normalize_code_value(_last_segment(raw) or raw) if isinstance(raw, str) else raw
        if not isinstance(candidate, str):
            continue
        candidate = candidate.replace("_", " ").strip()
        if not candidate or _is_unknown_token(candidate):
            continue
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _extract_scalar_values(value: Any) -> List[Any]:
    out: List[Any] = []
    if isinstance(value, Mapping):
        for key in ("href", "url", "value", "#text"):
            child = value.get(key)
            if _non_empty(child):
                out.append(child)
        if not out:
            for nested in value.values():
                out.extend(_extract_scalar_values(nested))
    elif isinstance(value, list):
        for item in value:
            out.extend(_extract_scalar_values(item))
    else:
        out.append(value)
    return out


def _collect_discovery_values(entity_type: str, source: Mapping[str, Any], bucket: str) -> List[Any]:
    values: List[Any] = []
    policy = DISCOVERY_POLICY.get(entity_type) or DISCOVERY_POLICY.get("observingConfiguration", {})
    for key in policy.get(bucket, []):
        for item in _as_list(source.get(key)):
            values.extend(_extract_scalar_values(item) if isinstance(item, Mapping) else [item])
    return values


def _about_link(href: str, *, title: Optional[str] = None, media_type: str = "text/html") -> Dict[str, Any]:
    link: Dict[str, Any] = {"href": href, "rel": "about", "type": media_type}
    if title:
        link["title"] = title
    return link


def _extract_links(source: Mapping[str, Any], entity_type: str) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    policy = DISCOVERY_POLICY.get(entity_type) or DISCOVERY_POLICY.get("observingConfiguration", {})
    for key in policy.get("links", []):
        for item in _as_list(source.get(key)):
            href: Optional[str] = None
            title: Optional[str] = None
            media_type = "text/html"
            if isinstance(item, str) and item.strip():
                href = item.strip()
            elif isinstance(item, Mapping):
                raw_href = _first_non_empty(item.get("url"), item.get("href"), item.get("linkage"), item.get("value"))
                if isinstance(raw_href, str) and raw_href.strip():
                    href = raw_href.strip()
                raw_title = item.get("title")
                if isinstance(raw_title, str):
                    title = raw_title.strip()
                raw_type = item.get("type")
                if isinstance(raw_type, str):
                    media_type = raw_type.strip()
            if href and href.startswith(("http://", "https://")):
                links.append(_about_link(href, title=title, media_type=media_type))
    return _uniq_dicts(links)


def _is_role_codelist_reference(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().strip("<>")
    if not text:
        return False
    segment = _last_segment(text) or text
    return segment in {"CI_RoleCode", "RoleCode"} or text.endswith("#CI_RoleCode") or text.endswith("#RoleCode")


def _normalize_role(value: Any) -> Optional[str]:
    if isinstance(value, Mapping):
        for key in ("codeListValue", "@codeListValue", "role", "value", "#text", "text", "name"):
            candidate = value.get(key)
            if _is_role_codelist_reference(candidate):
                continue
            role = _normalize_role(candidate)
            if role:
                return role
        for key in ("href", "url", "codeList", "@codeList"):
            candidate = value.get(key)
            if _is_role_codelist_reference(candidate):
                continue
            role = _normalize_role(candidate)
            if role:
                return role
        return None
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if _is_role_codelist_reference(text):
        return None
    text = text.strip("<>").rstrip("/#")
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    text = text.lstrip("_")
    if not text or _is_unknown_token(text):
        return None
    return text


def _normalize_roles(value: Any) -> List[str]:
    roles: List[str] = []
    for item in _as_list(value):
        role = _normalize_role(item)
        if role:
            roles.append(role)
    return sorted(dict.fromkeys(roles))


def _normalize_ogc_email(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, Mapping):
        raw_value = _first_non_empty(value.get("value"), value.get("email"), value.get("address"), value.get("href"), value.get("url"), value.get("#text"), value.get("text"))
        text = _strip_text(raw_value)
        if not text:
            return None
        if text.startswith("mailto:"):
            text = text.removeprefix("mailto:")
        email: Dict[str, Any] = {"value": text}
        roles = _normalize_roles(value.get("roles") or value.get("role"))
        if roles:
            email["roles"] = roles
        return email
    text = _strip_text(value)
    if not text:
        return None
    if text.startswith("mailto:"):
        text = text.removeprefix("mailto:")
    return {"value": text}


def _normalize_phone_value(value: str) -> str:
    text = value.strip()
    # If the source already provides an international number, remove common
    # presentation punctuation while preserving the leading '+'.  Also drop an
    # international trunk marker such as '(0)', which is not part of E.164.
    if text.startswith("+"):
        text = re.sub(r"\(0\)", "", text)
        text = re.sub(r"\(0", "(", text)
        digits = re.sub(r"\D", "", text)
        if digits:
            return "+" + digits

    digits_only = re.sub(r"\D", "", text)
    # ``00`` is a conventional international call prefix in many WMDR10
    # records.  Converting this prefix to ``+`` is a format normalization, not
    # an inferred country code.  Local-only numbers and bare digit strings are
    # preserved so that the strict OGC Contact schema can flag/comment them.
    if digits_only.startswith("00") and len(digits_only) > 4:
        candidate = digits_only[2:]
        if candidate and candidate[0] != "0":
            return "+" + candidate

    # Without a country code marker there is not enough information to safely
    # normalize to E.164.  Preserve the source value except for whitespace.
    return re.sub(r"\s+", "", text)


def _normalize_ogc_phone(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, Mapping):
        raw_value = _first_non_empty(value.get("value"), value.get("phone"), value.get("number"), value.get("voice"), value.get("facsimile"), value.get("#text"), value.get("text"))
        text = _strip_text(raw_value)
        if not text:
            return None
        phone: Dict[str, Any] = {"value": _normalize_phone_value(text)}
        roles = _normalize_roles(value.get("roles") or value.get("role"))
        if roles:
            phone["roles"] = roles
        return phone
    text = _strip_text(value)
    if not text:
        return None
    return {"value": _normalize_phone_value(text)}


def _normalize_ogc_address(value: Any) -> Optional[Dict[str, Any]]:
    address_obj = _as_mapping(value)
    if not address_obj:
        text = _strip_text(value)
        return {"deliveryPoint": [text]} if text else None
    address: Dict[str, Any] = {}
    delivery_points: List[str] = []
    raw_delivery = _first_non_empty(address_obj.get("deliveryPoint"), address_obj.get("deliveryPoints"), address_obj.get("street"))
    for item in _as_list(raw_delivery):
        text = _strip_text(item)
        if text:
            delivery_points.append(text)
    if delivery_points:
        address["deliveryPoint"] = sorted(dict.fromkeys(delivery_points))
    for key in ("city", "administrativeArea", "postalCode", "country"):
        text = _strip_text(address_obj.get(key))
        if text:
            address[key] = text
    roles = _normalize_roles(address_obj.get("roles") or address_obj.get("role"))
    if roles:
        address["roles"] = roles
    return address or None


def _normalize_link(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, str):
        href = value.strip()
        return _about_link(href) if href.startswith(("http://", "https://")) else None
    obj = _as_mapping(value)
    if not obj:
        return None
    href = _strip_text(_first_non_empty(obj.get("href"), obj.get("url"), obj.get("linkage"), obj.get("value")))
    if not href:
        return None
    link: Dict[str, Any] = {"href": href}
    for key in ("rel", "type", "title", "hreflang"):
        text = _strip_text(obj.get(key))
        if text:
            link[key] = text
    if "rel" not in link:
        link["rel"] = "about"
    if "type" not in link:
        link["type"] = "text/html"
    return link


def _normalize_ogc_contact(raw: Any) -> Optional[Dict[str, Any]]:
    payload = _as_mapping(raw)
    if not payload:
        text = _strip_text(raw)
        if not text:
            return None
        if "@" in text:
            return {"emails": [{"value": text.removeprefix("mailto:")}]}
        return {"organization": text}

    contact: Dict[str, Any] = {}
    for key in ("identifier", "name", "position", "organization", "hoursOfService", "contactInstructions"):
        text = _strip_text(
            _first_non_empty(
                payload.get(key),
                payload.get("individualName") if key == "name" else None,
                payload.get("organisationName") if key == "organization" else None,
            )
        )
        if text:
            contact[key] = text

    emails: List[Dict[str, Any]] = []
    for key in ("emails", "email", "electronicMailAddress", "mail", "mailAddress"):
        for item in _as_list(payload.get(key)):
            email = _normalize_ogc_email(item)
            if email:
                emails.append(email)
    info = _as_mapping(payload.get("contactInfo"))
    address_obj = _as_mapping(info.get("address"))
    for item in _as_list(address_obj.get("electronicMailAddress")):
        email = _normalize_ogc_email(item)
        if email:
            emails.append(email)
    if emails:
        contact["emails"] = _uniq_dicts(emails)

    phones: List[Dict[str, Any]] = []
    for key in ("phones", "phone", "telephone", "voice", "facsimile"):
        for item in _as_list(payload.get(key)):
            phone = _normalize_ogc_phone(item)
            if phone:
                phones.append(phone)
    phone_obj = _as_mapping(info.get("phone"))
    for key in ("voice", "facsimile", "phone", "phones"):
        for item in _as_list(phone_obj.get(key)):
            phone = _normalize_ogc_phone(item)
            if phone:
                phones.append(phone)
    if phones:
        contact["phones"] = _uniq_dicts(phones)

    addresses: List[Dict[str, Any]] = []
    for key in ("addresses", "address"):
        for item in _as_list(payload.get(key)):
            address = _normalize_ogc_address(item)
            if address:
                addresses.append(address)
    if address_obj:
        address = _normalize_ogc_address(address_obj)
        if address:
            addresses.append(address)
    if addresses:
        contact["addresses"] = _uniq_dicts(addresses)

    links: List[Dict[str, Any]] = []
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
    identifier = _strip_text(contact.get("identifier"))
    if identifier:
        return identifier
    emails = contact.get("emails")
    if isinstance(emails, list):
        for item in emails:
            item_obj = _as_mapping(item)
            value = _strip_text(item_obj.get("value"))
            if value:
                return f"contact:{value.lower()}"
    base = _first_non_empty(contact.get("organization"), contact.get("name"), contact.get("position"), _stable_json(contact))
    digest = hashlib.sha1(_stable_json(contact).encode("utf-8")).hexdigest()[:10]
    return f"contact:{_slug(base)}-{digest}"


def _merge_contact(existing: Mapping[str, Any], new_contact: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(existing)
    for key, value in new_contact.items():
        if key == "identifier":
            merged[key] = value
        elif key in {"emails", "phones", "addresses", "links"}:
            merged[key] = _uniq_dicts([*_as_list(merged.get(key)), *_as_list(value)])
        elif key == "roles":
            merged[key] = sorted(dict.fromkeys([*(_normalize_roles(merged.get(key))), *(_normalize_roles(value))]))
        elif key not in merged or merged.get(key) in (None, "", [], {}):
            merged[key] = value
    return merged


def _register_contact(registry: Dict[str, Dict[str, Any]], raw_contact: Any) -> Optional[str]:
    contact = _normalize_ogc_contact(raw_contact)
    if not contact:
        return None
    identifier = _contact_identifier(contact)
    contact["identifier"] = identifier
    # WMDR roles are contextual contactAssignments.  The reusable OGC Contact
    # registry must describe the party, not the party-in-this-context.
    contact.pop("roles", None)
    if identifier in registry:
        registry[identifier] = _merge_contact(registry[identifier], contact)
    else:
        registry[identifier] = contact
    return identifier


def _assignment_from_contact(raw: Any, registry: Dict[str, Dict[str, Any]], fallback_roles: Any = None) -> Optional[Dict[str, Any]]:
    if isinstance(raw, Mapping) and isinstance(raw.get("contact"), str):
        contact_ref = _strip_text(raw.get("contact"))
        roles = _normalize_roles(raw.get("roles") or raw.get("role") or fallback_roles)
        if contact_ref and roles:
            return {"contact": contact_ref, "roles": roles}
    contact_id = _register_contact(registry, raw)
    if not contact_id:
        return None
    payload = _as_mapping(raw)
    roles = _normalize_roles(payload.get("roles") or payload.get("role") or fallback_roles)
    if not roles:
        return None
    return {"contact": contact_id, "roles": roles}


def _extract_contact_assignments_from_field(value: Any, registry: Dict[str, Dict[str, Any]], fallback_roles: Any = None) -> List[Dict[str, Any]]:
    assignments: List[Dict[str, Any]] = []
    for item in _as_list(value):
        assignment = _assignment_from_contact(item, registry, fallback_roles=fallback_roles)
        if assignment:
            assignments.append(assignment)
    return _uniq_dicts(assignments)


def _quantity(value: Any, uom: Any = None) -> Optional[Dict[str, Any]]:
    raw_value = value
    raw_uom = uom
    if isinstance(value, Mapping):
        raw_value = _first_non_empty(value.get("value"), value.get("#text"), value.get("text"))
        raw_uom = _first_non_empty(value.get("uom"), value.get("unit"), value.get("@uom"), raw_uom)
    if raw_value in (None, "", [], {}):
        return None
    if isinstance(raw_value, str):
        text_value = raw_value.strip()
        try:
            raw_value = float(text_value)
        except ValueError:
            raw_value = text_value
    out: Dict[str, Any] = {"value": raw_value}
    if raw_uom not in (None, "", [], {}):
        out["uom"] = raw_uom
    return out


def _record_timestamps(header: Mapping[str, Any], *, source_name: Optional[str] = None) -> Dict[str, str]:
    created = _normalize_record_datetime(
        _first_non_empty(header.get("created"), header.get("dateCreated"), header.get("creationDate"), header.get("fileDateTime"), header.get("dateStamp"), source_name)
    )
    updated = _normalize_record_datetime(
        _first_non_empty(header.get("updated"), header.get("dateUpdated"), header.get("updateDate"), header.get("modified"), header.get("fileDateTime"), header.get("dateStamp"), created)
    )
    out: Dict[str, str] = {}
    if created:
        out["created"] = created
    if updated:
        out["updated"] = updated
    return out

# ---------------------------------------------------------------------------
# WMDR10 -> WMDR2 construction
# ---------------------------------------------------------------------------


def _split_source(source: Any) -> Tuple[Dict[str, Any], Dict[str, Any], List[Any], List[Any]]:
    if not isinstance(source, Mapping):
        return {}, {}, [], []
    if source.get("type") == "Feature" and isinstance(source.get("properties"), Mapping):
        props = _as_dict(source.get("properties"))
        return props, {}, _as_list(props.get("observationSeries") or props.get("observations")), _as_list(props.get("deployments"))
    facility = _as_dict(source.get("facility"))
    header = _as_dict(source.get("header"))
    observations = _as_list(_first_non_empty(source.get("observationSeries"), source.get("observations"), source.get("observation")))
    deployments = _as_list(_first_non_empty(source.get("deployments"), source.get("deployment"), source.get("observingConfigurations")))
    if not facility:
        kind_keys = {"observedVariable", "observedProperty", "resultTime", "sourceOfObservation", "manufacturer", "serialNumber", "fileDateTime"}
        if not any(key in source for key in kind_keys):
            facility = dict(source)
    return facility, header, observations, deployments


def _facility_identifier(facility: Mapping[str, Any], header: Mapping[str, Any]) -> str:
    raw_values = (
        facility.get("identifier"),
        facility.get("wigosStationIdentifier"),
        facility.get("wigosIdentifier"),
        facility.get("wsi"),
        facility.get("id"),
        header.get("wigosStationIdentifier"),
        header.get("identifier"),
        header.get("id"),
    )
    for raw in raw_values:
        values = _facility_wsi_values(raw)
        if values:
            return values[0]
    return _normalize_facility_wsi(_first_non_empty(*raw_values))




def _title_values(value: Any) -> List[str]:
    """Return ordered facility title/name strings from WMDR10 title shapes."""
    values: List[str] = []
    if value in (None, "", [], {}):
        return values
    if isinstance(value, Mapping):
        for key in ("title", "name", "value", "text", "#text"):
            values.extend(_title_values(value.get(key)))
        return _uniq_scalars(values)
    if isinstance(value, list):
        for item in value:
            values.extend(_title_values(item))
        return _uniq_scalars(values)
    text = _strip_text(value)
    return [text] if text else []


def _title_text(value: Any) -> Optional[str]:
    """Return the primary OGC Records title string from WMDR10 title/name shapes."""
    values = _title_values(value)
    return values[0] if values else None

def _description_text(value: Any) -> Optional[str]:
    """Return an OGC Records description string from WMDR10 description shapes.

    WMDR10/XML-derived data may carry descriptions as objects with their own
    validity metadata, or as lists of such objects.  WMDR2 currently uses the
    OGC Records ``description`` member, which is a string.  Preserve only the
    recorded text here; do not invent a temporal description model.
    """
    if value in (None, "", [], {}):
        return None
    if isinstance(value, Mapping):
        text = _strip_text(
            _first_non_empty(
                value.get("description"),
                value.get("value"),
                value.get("text"),
                value.get("#text"),
                value.get("remarks"),
                value.get("remark"),
            )
        )
        return text
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            text = _description_text(item)
            if text and text not in parts:
                parts.append(text)
        return "\n\n".join(parts) if parts else None
    return _strip_text(value)


def _has_explicit_time_period(value: Mapping[str, Any]) -> bool:
    """Return True if a temporal object carries an explicit period anchor."""
    time_obj = _as_mapping(value.get("time"))
    if isinstance(time_obj, Mapping) and _as_list(time_obj.get("interval")):
        return True
    return any(value.get(key) not in (None, "", [], {}) for key in ("beginPosition", "endPosition", "validFrom", "validTo"))


def _normalize_program_affiliations(value: Any) -> List[Any]:
    """Normalize program affiliations without inventing missing validity.

    In v0.3.1 program affiliations are temporal objects in the public schema.
    If the source only provides a bare program code, there is no recorded
    validity period to publish, so the affiliation is not emitted.  This keeps
    the generated WMDR2 record valid without fabricating ``time``.
    """
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            program = _first_non_empty(
                item.get("program"),
                item.get("programAffiliation"),
                item.get("name"),
                item.get("identifier"),
                item.get("value"),
                item.get("href"),
                item.get("url"),
            )
            compact = _compact_wmdr_code_value(program)
            if compact in (None, "", [], {}):
                continue
            if not _has_explicit_time_period(item):
                continue
            payload = dict(item)
            payload.pop("name", None)
            payload.pop("identifier", None)
            payload.pop("value", None)
            payload.pop("href", None)
            payload.pop("url", None)
            payload["program"] = compact
            out.append(cast(Dict[str, Any], _normalize_node(payload, {})))
        else:
            # A scalar program affiliation has no recorded validity period.
            # Do not emit an invalid temporal object and do not invent time.
            continue
    return _uniq_dicts(out)



def _normalize_territories(value: Any) -> List[Dict[str, Any]]:
    """Normalize WMDR10 territory values to the v0.3.1 temporal array shape.

    The schema models ``properties.territory`` as an array of temporal objects.
    WMDR10/XML-derived JSON often contains a single object with ``territoryName``
    plus ``beginPosition``/``endPosition``.  Convert only records with an
    explicit temporal anchor; do not invent time for scalar territory values.
    """
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            raw_territory = _first_non_empty(
                item.get("territory"),
                item.get("territoryName"),
                item.get("name"),
                item.get("identifier"),
                item.get("value"),
                item.get("href"),
                item.get("url"),
                item.get("#text"),
                item.get("text"),
            )
            territory = _compact_wmdr_code_value(raw_territory)
            if territory in (None, "", [], {}):
                continue
            if not _has_explicit_time_period(item):
                continue
            payload: Dict[str, Any] = {"territory": territory}
            start, end = _extract_interval(item)
            interval = _time_interval(start, end)
            if interval:
                payload["time"] = interval
            if "time" in payload:
                out.append(payload)
        else:
            # A scalar territory value has no recorded validity period.
            continue
    return _uniq_dicts(out)


def _environment_from_facility(facility: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    env_src = _as_mapping(facility.get("environment"))
    env: Dict[str, Any] = dict(env_src)
    for key in (
        "temporalClimateZone",
        "temporalSurfaceCover",
        "temporalPopulation",
        "temporalPopulationDensities",
        "temporalSurfaceRoughness",
        "localTopography",
        "relativeElevation",
        "topographicContext",
        "altitudeOrDepth",
    ):
        value = facility.get(key)
        if value not in (None, "", [], {}):
            env[key] = value
    topo = _as_mapping(facility.get("temporalTopographyBathymetry"))
    for key, value in topo.items():
        if value not in (None, "", [], {}):
            env[key] = value
    return cast(Optional[Dict[str, Any]], _clean_none(env)) if env else None


def _instrument_key(src: Mapping[str, Any]) -> Optional[str]:
    manufacturer = _strip_text(src.get("manufacturer"))
    model = _strip_text(src.get("model"))
    if not manufacturer and not model:
        return None
    return f"instrument:{_slug(manufacturer or 'unknown')}-{_slug(model or 'unknown')}"


def _instrument_from_source(src: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    identifier = _instrument_key(src)
    if not identifier:
        return None
    instrument: Dict[str, Any] = {"id": identifier}
    for key in ("manufacturer", "model", "description"):
        text = _strip_text(src.get(key))
        if text:
            instrument[key] = text
    # Catalogue item intentionally excludes serial number; serial-numbered
    # instances belong to an observing configuration, not to the catalogue.
    for key in ("observableProperties", "observableVariables", "observableGeometry", "observingMethods", "verticalRange"):
        value = src.get(key)
        if value not in (None, "", [], {}):
            instrument[key] = value
    return instrument


def _merge_instrument(existing: Mapping[str, Any], new: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(existing)
    for key, value in new.items():
        if key not in out or out.get(key) in (None, "", [], {}):
            out[key] = value
        elif isinstance(out.get(key), list) or isinstance(value, list):
            out[key] = _uniq_scalars([*_as_list(out.get(key)), *_as_list(value)])
    return out


def _observing_configuration_from_source(src: Mapping[str, Any], instrument_registry: Dict[str, Dict[str, Any]], contact_registry: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}

    start, end = _extract_interval(src)
    interval = _time_interval(start, end)
    if interval:
        cfg["time"] = interval

    # Promote observingLocation members into the configuration itself.
    loc = _as_mapping(src.get("observingLocation"))
    merged_src: Dict[str, Any] = dict(loc)
    for key, value in src.items():
        if key != "observingLocation" and key not in merged_src:
            merged_src[key] = value

    for key in ("observingMethod", "operatingStatus", "sourceOfObservation", "exposure", "referenceSurface"):
        if key in merged_src:
            cfg[key] = _normalize_code_or_nil_reason(merged_src.get(key))

    for key in ("serialNumber", "location", "relativeLocation", "configuration", "description"):
        text = _strip_text(merged_src.get(key))
        if text:
            cfg[key] = text

    vertical = _quantity(
        _first_non_empty(merged_src.get("verticalDistanceFromReferenceSurface"), merged_src.get("heightAboveLocalReferenceSurface"), merged_src.get("heightAboveReferenceSurface")),
        _first_non_empty(merged_src.get("verticalDistanceFromReferenceSurfaceUom"), merged_src.get("heightAboveLocalReferenceSurfaceUom"), merged_src.get("uom")),
    )
    if vertical:
        cfg["verticalDistanceFromReferenceSurface"] = vertical

    temporal_entries = _facility_temporal_geometry_entries(merged_src)
    if temporal_entries:
        # The configuration already has ``time`` as its lifecycle anchor.  The
        # location member is therefore simply ``geometry``; do not emit a
        # second, configuration-level ``temporalGeometry`` wrapper.  If source
        # history contains several positions, the latest known position is used
        # for this configuration.
        geometry = _point_geometry_from_entry(temporal_entries[-1])
        if geometry:
            cfg["geometry"] = geometry

    instrument = _instrument_from_source(merged_src)
    if instrument:
        instrument_id = cast(str, instrument["id"])
        if instrument_id in instrument_registry:
            instrument_registry[instrument_id] = _merge_instrument(instrument_registry[instrument_id], instrument)
        else:
            instrument_registry[instrument_id] = instrument
        cfg["instrument"] = instrument_id

    contact_assignments: List[Dict[str, Any]] = []
    for key in ("contacts", "contact", "responsibleParty", "operator", "maintainer"):
        fallback = key if key not in {"contacts", "contact"} else None
        contact_assignments.extend(_extract_contact_assignments_from_field(merged_src.get(key), contact_registry, fallback_roles=fallback))
    if contact_assignments:
        cfg["contactAssignments"] = _uniq_dicts(contact_assignments)

    links = _extract_links(merged_src, "observingConfiguration")
    if links:
        cfg["links"] = links

    return cast(Dict[str, Any], _clean_none(cfg))


def _observed_domain_object(obs: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    raw = obs.get("observedDomain") or obs.get("observationDomain")
    if isinstance(raw, Mapping):
        domain = _first_non_empty(raw.get("domain"), raw.get("value"), raw.get("#text"), raw.get("text"))
        out: Dict[str, Any] = {}
        if domain not in (None, "", [], {}):
            out["domain"] = _compact_wmdr_code_value(domain)
        for key in ("domainFeature", "featureName", "observedFeatureDomainFeature", "observedFeatureName"):
            value = raw.get(key)
            if value not in (None, "", [], {}):
                out_key = "domainFeature" if "Feature" in key else "featureName"
                out[out_key] = value
        return out or None
    if raw not in (None, "", [], {}):
        return {"domain": _compact_wmdr_code_value(raw)}
    observed_property = _first_non_empty(obs.get("observedProperty"), obs.get("observedVariable"))
    domain = _observed_domain_from_observed_variable(observed_property)
    return {"domain": domain} if domain else None



def _schedule_uid(schedule: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in schedule.items() if key not in {"uid", "id"}}
    digest = hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()[:10]
    return f"schedule_{digest}"


def _normalize_schedule_object(raw: Any, *, kind: str = "shared") -> Optional[Dict[str, Any]]:
    """Return a reusable JSCalendar-like schedule object.

    WMDR2 v0.3.1 keeps one shared root-level schedule registry.  The same
    schedule can be referenced from ``observingProcedures`` and
    ``reportingProcedures`` when the observing and reporting rhythm is the same.
    The reference context supplies the role; the schedule itself therefore has
    no ``scheduleType`` discriminator.
    """
    if raw in (None, "", [], {}):
        return None

    if isinstance(raw, Mapping):
        schedule: Dict[str, Any] = dict(raw)
    else:
        # A bare scalar is interpreted in the calling context: sampling
        # frequency for observing, aggregation interval for reporting.
        schedule = {"frequency": raw}

    legacy_id = _strip_text(schedule.pop("id", None))
    uid = _strip_text(schedule.get("uid")) or legacy_id
    if uid and uid.startswith("schedule:"):
        uid = "schedule_" + _slug(uid.removeprefix("schedule:"))

    if "@type" not in schedule:
        schedule["@type"] = "Event"
    if "start" not in schedule and "startDate" in schedule:
        schedule["start"] = schedule.pop("startDate")
    if "start" not in schedule:
        schedule["start"] = CANONICAL_SCHEDULE_START_DATE

    sampling = _first_non_empty(
        schedule.pop("samplingFrequency", None),
        schedule.pop("temporalSamplingInterval", None),
        schedule.get("wmo.int:samplingFrequency"),
    )
    if sampling not in (None, "", [], {}):
        schedule["wmo.int:samplingFrequency"] = _normalize_time_resolution(sampling)

    aggregation = _first_non_empty(
        schedule.pop("aggregationInterval", None),
        schedule.pop("temporalReportingInterval", None),
        schedule.pop("temporalAggregate", None),
        schedule.get("wmo.int:aggregationInterval"),
    )
    if aggregation not in (None, "", [], {}):
        schedule["wmo.int:aggregationInterval"] = _normalize_time_resolution(aggregation)

    duration = _first_non_empty(schedule.get("duration"), schedule.pop("coverageDuration", None), schedule.pop("diurnalDuration", None))
    if duration not in (None, "", [], {}):
        schedule["duration"] = _normalize_time_resolution(duration)

    diurnal = _first_non_empty(
        schedule.pop("diurnalBaseTime", None),
        schedule.get("wmo.int:diurnalBaseTime"),
    )
    if diurnal not in (None, "", [], {}):
        schedule["wmo.int:diurnalBaseTime"] = _normalize_diurnal_time(str(diurnal))

    frequency = schedule.pop("frequency", None)
    if frequency not in (None, "", [], {}):
        if kind == "reporting" and "wmo.int:aggregationInterval" not in schedule:
            schedule["wmo.int:aggregationInterval"] = _normalize_time_resolution(frequency)
        elif "wmo.int:samplingFrequency" not in schedule:
            schedule["wmo.int:samplingFrequency"] = _normalize_time_resolution(frequency)

    if not uid:
        uid = _schedule_uid(schedule)
    schedule["uid"] = uid
    return cast(Dict[str, Any], _clean_none(schedule))

def _register_schedule(schedule: Optional[Mapping[str, Any]], registry: Dict[str, Dict[str, Any]], *, kind: str) -> Optional[str]:
    if not schedule:
        return None
    normalized = _normalize_schedule_object(schedule, kind=kind)
    if not normalized:
        return None
    uid = _strip_text(normalized.get("uid"))
    if not uid:
        return None
    existing = registry.get(uid)
    if existing:
        registry[uid] = _merge_instrument(existing, normalized)
    else:
        registry[uid] = normalized
    return uid


def _schedule_from_source(src: Mapping[str, Any], *, kind: str) -> Optional[Dict[str, Any]]:
    """Build a reusable shared schedule from observing/reporting source metadata.

    The source context controls which explicit schedule object is preferred, but
    when schedule information is inferred from WMDR10 data-generation/coverage
    fields both observing and reporting procedures intentionally receive the
    same normalized schedule object.  This allows the same root-level schedule
    ``uid`` to be reused by ``observingSchedules`` and ``reportingSchedules``.
    """
    coverage = _as_mapping(src.get("coverage"))
    sampling = _as_mapping(src.get("sampling"))
    reporting_src = _as_mapping(src.get("reporting"))
    reporting_coverage = _as_mapping(reporting_src.get("coverage"))

    explicit_keys = (
        ("observingSchedule", "observingSchedules")
        if kind == "observing"
        else ("reportingSchedule", "reportingSchedules")
    )
    raw = _first_non_empty(
        *(src.get(key) for key in explicit_keys),
        *(sampling.get(key) for key in ("observingSchedule", "schedule")),
        *(reporting_src.get(key) for key in ("reportingSchedule", "reportingSchedules", "schedule")),
        *(coverage.get(key) for key in ("observingSchedule", "reportingSchedule", "schedule")),
        *(reporting_coverage.get(key) for key in ("reportingSchedule", "schedule")),
        src.get("schedule"),
    )
    if raw not in (None, "", [], {}):
        return _normalize_schedule_object(raw, kind=kind)

    sampling_interval = _first_non_empty(
        src.get("temporalSamplingInterval"),
        sampling.get("temporalSamplingInterval"),
        coverage.get("temporalSamplingInterval"),
    )
    aggregation_interval = _first_non_empty(
        reporting_src.get("temporalReportingInterval"),
        reporting_src.get("temporalAggregate"),
        reporting_src.get("aggregationInterval"),
        src.get("temporalReportingInterval"),
        src.get("temporalAggregate"),
        src.get("aggregationInterval"),
        coverage.get("temporalReportingInterval"),
        coverage.get("temporalAggregate"),
        coverage.get("aggregationInterval"),
        reporting_coverage.get("temporalReportingInterval"),
        reporting_coverage.get("temporalAggregate"),
        reporting_coverage.get("aggregationInterval"),
    )
    diurnal = _first_non_empty(
        reporting_src.get("diurnalBaseTime"),
        src.get("diurnalBaseTime"),
        coverage.get("diurnalBaseTime"),
        reporting_coverage.get("diurnalBaseTime"),
    )

    payload: Dict[str, Any] = {}
    if sampling_interval not in (None, "", [], {}):
        payload["wmo.int:samplingFrequency"] = sampling_interval
    if aggregation_interval not in (None, "", [], {}):
        payload["wmo.int:aggregationInterval"] = aggregation_interval
    if diurnal not in (None, "", [], {}):
        payload["wmo.int:diurnalBaseTime"] = diurnal

    coverage_fields = _diurnal_coverage_fields(coverage, reporting_coverage, src, reporting_src)
    payload.update(coverage_fields)

    if not payload:
        # Keep an explicit event/window duration only when it is supplied as a
        # duration, not as a reporting interval.  Reporting intervals are mapped
        # to wmo.int:aggregationInterval above.
        duration = _first_non_empty(src.get("duration"), coverage.get("duration"), reporting_src.get("duration"), reporting_coverage.get("duration"))
        if duration not in (None, "", [], {}):
            payload["duration"] = duration

    if not payload:
        return None
    return _normalize_schedule_object(payload, kind=kind)

def _observing_procedure_from_source(src: Mapping[str, Any], schedule_registry: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    schedule = _schedule_from_source(src, kind="observing")
    if not schedule:
        return None
    uid = _register_schedule(schedule, schedule_registry, kind="observing")
    if not uid:
        return None
    proc: Dict[str, Any] = {"observingSchedules": [uid]}
    start, end = _extract_interval(src)
    interval = _time_interval(start, end)
    if interval:
        proc["time"] = interval
    strategy = _first_non_empty(src.get("strategy"), _as_mapping(src.get("sampling")).get("samplingStrategy"), src.get("samplingStrategy"))
    if strategy not in (None, "", [], {}):
        proc["strategy"] = _compact_wmdr_code_value(strategy)
    return cast(Dict[str, Any], _clean_none(proc))



def _compact_reporting_value(value: Any) -> Any:
    """Compact a reporting-procedure scalar or code-list object."""
    if isinstance(value, Mapping) and "dataPolicy" in value:
        return _compact_wmdr_code_value(value.get("dataPolicy"))
    return _compact_wmdr_code_value(value)


def _compact_reporting_values(value: Any) -> Optional[List[Any]]:
    values = []
    for item in _as_list(value):
        compacted = _compact_reporting_value(item)
        if compacted not in (None, "", [], {}):
            values.append(compacted)
    return _uniq_scalars(values) or None

def _reporting_procedure_from_source(src: Mapping[str, Any], contact_registry: Dict[str, Dict[str, Any]], schedule_registry: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize a UML ReportingProcedure.

    ReportingProcedure is not time-bound in the v0.3.1 UML model.  Temporal
    reporting cadence from WMDR10 (for example ``temporalReportingInterval``)
    is represented by a reusable ReportingSchedule referenced through
    ``reportingSchedules``.
    """
    relevant_keys = {
        "reporting",
        "reportingProcedure",
        "internationalExchange",
        "dataFormat",
        "dataPolicy",
        "levelOfData",
        "links",
        "numberOfObservationsInReportingInterval",
        "referenceDatum",
        "referenceTimeSource",
        "spatialReportingInterval",
        "strategy",
        "timeliness",
        "timeStampMeaning",
        "uom",
        "reportingInterval",
        "temporalReportingInterval",
        "temporalAggregate",
        "duration",
        "reportingSchedule",
        "reportingSchedules",
        "coverage",
        "contact",
        "contacts",
        "responsibleParty",
    }
    if not any(k in src for k in relevant_keys):
        return None

    reporting_src = _as_mapping(src.get("reporting"))
    procedure_src = _as_mapping(src.get("reportingProcedure"))
    merged: Dict[str, Any] = {**reporting_src, **procedure_src}
    for key in relevant_keys:
        if key in src and key not in merged:
            merged[key] = src[key]

    proc: Dict[str, Any] = {}

    if "internationalExchange" in merged:
        parsed_bool = _parse_bool(merged.get("internationalExchange"))
        if parsed_bool is not None:
            proc["internationalExchange"] = parsed_bool

    scalar_keys = (
        "dataPolicy",
        "levelOfData",
        "numberOfObservationsInReportingInterval",
        "referenceDatum",
        "spatialReportingInterval",
        "strategy",
        "timeliness",
        "timeStampMeaning",
        "uom",
    )
    for key in scalar_keys:
        value = merged.get(key)
        if value not in (None, "", [], {}):
            proc[key] = _compact_reporting_value(value)

    for key in ("dataFormat", "referenceTimeSource"):
        values = _compact_reporting_values(merged.get(key))
        if values:
            proc[key] = values

    links = _extract_links(merged, "reportingProcedure")
    if not links:
        raw_links = merged.get("links")
        if isinstance(raw_links, list):
            links = [dict(item) for item in raw_links if isinstance(item, Mapping)]
    if links:
        proc["links"] = _uniq_dicts(links)

    # Build the schedule from the complete data-generation source, not only
    # from the flattened reporting-procedure attributes.  This preserves the
    # intended reuse where observing and reporting procedures point to the same
    # schedule object when the same data-generation block provides sampling,
    # aggregation and diurnal coverage information.
    schedule_source: Dict[str, Any] = dict(src)
    if reporting_src or procedure_src:
        schedule_source["reporting"] = {**reporting_src, **procedure_src}
    schedule = _schedule_from_source(schedule_source, kind="reporting")
    if schedule:
        uid = _register_schedule(schedule, schedule_registry, kind="reporting")
        if uid:
            proc["reportingSchedules"] = [uid]

    assignments: List[Dict[str, Any]] = []
    for key in ("contacts", "contact", "responsibleParty"):
        fallback = key if key == "responsibleParty" else None
        assignments.extend(_extract_contact_assignments_from_field(merged.get(key), contact_registry, fallback_roles=fallback))
    if assignments:
        proc["contactAssignments"] = _uniq_dicts(assignments)

    return cast(Optional[Dict[str, Any]], _clean_none(proc))

def _observation_series_from_source(
    obs: Mapping[str, Any],
    index: int,
    deployments: Sequence[Any],
    instrument_registry: Dict[str, Dict[str, Any]],
    contact_registry: Dict[str, Dict[str, Any]],
    schedule_registry: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    observed_property = _first_non_empty(obs.get("observedProperty"), obs.get("observedVariable"))
    observed_geometry = _first_non_empty(obs.get("observedGeometry"), obs.get("observedGeometryType"), obs.get("geometryType"))
    obs_id_base = _first_non_empty(obs.get("id"), obs.get("uid"))
    if obs_id_base not in (None, "", [], {}):
        series_id = _sanitize_id(obs_id_base)
    elif observed_property not in (None, "", [], {}):
        series_id = f"observationSeries:{_compact_wmdr_code_value(observed_property)}"
    else:
        series_id = f"observationSeries:{index + 1}"
    series: Dict[str, Any] = {"id": series_id}

    title = _strip_text(obs.get("title")) or _format_observation_title(observed_property, observed_geometry)
    if title:
        series["title"] = title
    description = _strip_text(obs.get("description"))
    if description:
        series["description"] = description
    if observed_property not in (None, "", [], {}):
        series["observedProperty"] = _compact_wmdr_code_value(observed_property)
    domain = _observed_domain_object(obs)
    if domain:
        series["observedDomain"] = domain
    if observed_geometry not in (None, "", [], {}):
        series["observedGeometry"] = _compact_wmdr_code_value(observed_geometry)

    start, end = _extract_interval(obs)
    interval = _time_interval(start, end)
    if interval:
        series["time"] = interval

    configs: List[Dict[str, Any]] = []
    raw_configs = _as_list(obs.get("observingConfigurations"))
    local_deployments = _as_list(_first_non_empty(obs.get("deployments"), obs.get("deployment"), obs.get("deploymentRefs")))

    def append_config(src_obj: Mapping[str, Any]) -> None:
        cfg = _observing_configuration_from_source(src_obj, instrument_registry, contact_registry)
        if cfg:
            configs.append(cfg)

    if raw_configs:
        for cfg_src in raw_configs:
            append_config(_as_mapping(cfg_src))
    elif local_deployments:
        for dep in local_deployments:
            append_config(_as_mapping(dep))
    elif deployments:
        deployment_refs = {str(x) for x in _as_list(obs.get("deploymentRefs") or obs.get("deployment")) if x not in (None, "")}
        for dep in deployments:
            dep_obj = _as_mapping(dep)
            dep_id = str(_first_non_empty(dep_obj.get("id"), dep_obj.get("identifier"), dep_obj.get("uid"), ""))
            if deployment_refs and dep_id not in deployment_refs:
                continue
            append_config(dep_obj)
    else:
        append_config(obs)

    if configs:
        series["observingConfigurations"] = _uniq_dicts(configs)

    reporting_sources: List[Mapping[str, Any]] = []
    for item in _as_list(obs.get("dataGeneration")):
        item_obj = _as_mapping(item)
        if item_obj:
            reporting_sources.append(item_obj)
    for dep in local_deployments:
        dep_obj = _as_mapping(dep)
        for item in _as_list(dep_obj.get("dataGeneration")):
            item_obj = _as_mapping(item)
            if item_obj:
                reporting_sources.append(item_obj)
    if not reporting_sources:
        reporting_sources = [obs]

    observing_items: List[Dict[str, Any]] = []
    reporting_items: List[Dict[str, Any]] = []
    for reporting_source in reporting_sources:
        observing = _observing_procedure_from_source(reporting_source, schedule_registry)
        if observing:
            observing_items.append(observing)
        reporting = _reporting_procedure_from_source(reporting_source, contact_registry, schedule_registry)
        if reporting:
            reporting_items.append(reporting)
    if observing_items:
        series["observingProcedures"] = _uniq_dicts(observing_items)
    if reporting_items:
        series["reportingProcedures"] = _uniq_dicts(reporting_items)

    assignments: List[Dict[str, Any]] = []
    for key in ("contacts", "contact", "responsibleParty", "operator"):
        fallback = key if key not in {"contacts", "contact"} else None
        assignments.extend(_extract_contact_assignments_from_field(obs.get(key), contact_registry, fallback_roles=fallback))
    if assignments:
        series["contactAssignments"] = _uniq_dicts(assignments)

    links = _extract_links(obs, "observation")
    if links:
        series["links"] = links
    keywords = _keywords_from_values(_collect_discovery_values("observation", obs, "keywords"))
    if keywords:
        series["keywords"] = keywords

    return cast(Dict[str, Any], _clean_none(series))


def _collect_root_contacts(facility: Mapping[str, Any], header: Mapping[str, Any], registry: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    assignments: List[Dict[str, Any]] = []
    for key in ("contacts", "contact", "responsibleParty"):
        assignments.extend(_extract_contact_assignments_from_field(facility.get(key), registry))
    # Record owner is a contextual facility-level role when present.
    assignments.extend(_extract_contact_assignments_from_field(header.get("recordOwner"), registry, fallback_roles="owner"))
    assignments.extend(_extract_contact_assignments_from_field(facility.get("owner"), registry, fallback_roles="owner"))
    assignments.extend(_extract_contact_assignments_from_field(facility.get("operator"), registry, fallback_roles="operator"))
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
    """Build one WMDR2 v0.3.1 facility Feature.

    ``source`` may be a combined object with ``facility``, ``header``,
    ``observations``/``observationSeries`` and ``deployments`` members.  The
    keyword arguments are useful when callers have already split WMDR10 input
    into separate part files.
    """
    if isinstance(source, Mapping) and source.get("type") == "Feature":
        return cast(Dict[str, Any], normalize_wmdr2_record(dict(source)))

    src_facility, src_header, src_observations, src_deployments = _split_source(source) if source is not None else ({}, {}, [], [])
    facility_obj = dict(facility or src_facility)
    header_obj = dict(header or src_header)
    observation_items = list(observations if observations is not None else src_observations)
    deployment_items = list(deployments if deployments is not None else src_deployments)

    wsi = _facility_identifier(facility_obj, header_obj)
    temporal_entries = _facility_temporal_geometry_entries(facility_obj)
    geometry = _facility_geometry_from_entries(temporal_entries)

    contact_registry: Dict[str, Dict[str, Any]] = {}
    instrument_registry: Dict[str, Dict[str, Any]] = {}
    schedule_registry: Dict[str, Dict[str, Any]] = {}

    facility_titles = _uniq_scalars(_title_values(facility_obj.get("name")) + _title_values(facility_obj.get("title")))
    primary_title = facility_titles[0] if facility_titles else wsi
    additional_titles = [title for title in facility_titles[1:] if title != primary_title]

    properties: Dict[str, Any] = {
        "type": "facility",
        "title": primary_title,
        **_record_timestamps(header_obj, source_name=source_name),
    }
    if additional_titles:
        properties["additionalTitles"] = additional_titles
    description = _description_text(facility_obj.get("description"))
    if description:
        properties["description"] = description
    if _non_empty(facility_obj.get("facilityType")):
        properties["facilityType"] = _compact_wmdr_code_value(facility_obj.get("facilityType"))
    territory = _normalize_territories(facility_obj.get("territory"))
    if territory:
        properties["territory"] = territory
    if _non_empty(facility_obj.get("wmoRegion")):
        properties["wmoRegion"] = facility_obj.get("wmoRegion")

    wsi_candidates: List[str] = []
    for raw_identifier in (
        facility_obj.get("identifier"),
        facility_obj.get("wigosStationIdentifier"),
        facility_obj.get("wigosIdentifier"),
        facility_obj.get("wsi"),
        facility_obj.get("id"),
        header_obj.get("wigosStationIdentifier"),
        header_obj.get("identifier"),
        header_obj.get("id"),
    ):
        wsi_candidates.extend(_facility_wsi_values(raw_identifier))
    additional_ids = [candidate for candidate in _uniq_scalars(wsi_candidates) if candidate != wsi]
    if additional_ids:
        properties["additionalIds"] = additional_ids

    start, end = _extract_interval(facility_obj)
    time_obj = _time_interval(start, end, resolution="P1D")
    if time_obj:
        properties["time"] = time_obj

    temporal_geometry = _temporal_geometry_extension(temporal_entries)
    if temporal_geometry:
        properties["temporalGeometry"] = temporal_geometry

    program_affiliations = _normalize_program_affiliations(
        _first_non_empty(facility_obj.get("programAffiliations"), facility_obj.get("programAffiliation"), facility_obj.get("programs"))
    )
    if program_affiliations:
        properties["programAffiliations"] = program_affiliations

    env = _environment_from_facility(facility_obj)
    if env:
        properties["environment"] = env

    contact_assignments = _collect_root_contacts(facility_obj, header_obj, contact_registry)
    if contact_assignments:
        properties["contactAssignments"] = contact_assignments

    observation_series: List[Dict[str, Any]] = []
    for index, obs_raw in enumerate(observation_items):
        obs_obj = _as_mapping(obs_raw)
        if not obs_obj:
            continue
        series = _observation_series_from_source(obs_obj, index, deployment_items, instrument_registry, contact_registry, schedule_registry)
        if series:
            observation_series.append(series)
    if observation_series:
        properties["observationSeries"] = observation_series
    if schedule_registry:
        properties["schedules"] = sorted(schedule_registry.values(), key=lambda item: str(item.get("uid")))

    # Register instruments from deployments even if no observation referred to
    # them, but keep the catalogue generic: no serial-number instance values.
    for dep_raw in deployment_items:
        dep_obj = _as_mapping(dep_raw)
        instrument = _instrument_from_source(dep_obj)
        if not instrument:
            loc_obj = _as_mapping(dep_obj.get("observingLocation"))
            instrument = _instrument_from_source(loc_obj)
        if instrument:
            instrument_id = cast(str, instrument["id"])
            if instrument_id in instrument_registry:
                instrument_registry[instrument_id] = _merge_instrument(instrument_registry[instrument_id], instrument)
            else:
                instrument_registry[instrument_id] = instrument

    if instrument_registry:
        properties["instruments"] = sorted(instrument_registry.values(), key=lambda item: str(item.get("id")))
    if contact_registry:
        properties["contacts"] = sorted(contact_registry.values(), key=lambda item: str(item.get("identifier")))

    keywords = _keywords_from_values(_collect_discovery_values("facility", facility_obj, "keywords"))
    if keywords:
        properties["keywords"] = keywords
    links = _extract_links(facility_obj, "facility")
    if links:
        properties["links"] = links

    feature: Dict[str, Any] = {
        "type": "Feature",
        "id": wsi,
        "conformsTo": [WMDR2_CORE_CONF],
        "geometry": geometry,
        "properties": properties,
    }
    if "time" in properties:
        feature["time"] = properties.pop("time")
    if "temporalGeometry" in properties:
        feature["temporalGeometry"] = properties.pop("temporalGeometry")
    return cast(Dict[str, Any], normalize_wmdr2_record(feature))

# ---------------------------------------------------------------------------
# v0.3.1 normalizer for generated or existing WMDR2 records
# ---------------------------------------------------------------------------


def _normalize_time_members(payload: Dict[str, Any]) -> None:
    """Normalize WMDR10/source temporal anchors to OGC Records ``time``.

    Public WMDR2 JSON must not expose WMDR10 ``beginPosition``/``endPosition``
    or the transitional v0.3.0 ``validFrom``/``validTo`` fields.  Any such
    source anchors found on an object are converted to ``time.interval`` and
    then removed from the object.
    """
    time_obj = payload.get("time")
    if isinstance(time_obj, Mapping):
        normalized = dict(time_obj)
        if "resolution" in normalized:
            normalized["resolution"] = _normalize_time_resolution(normalized.get("resolution"))
        payload["time"] = normalized
    elif "time" in payload and payload.get("time") in (None, "", [], {}):
        payload.pop("time", None)

    has_source_temporal = any(key in payload for key in SOURCE_TEMPORAL_KEYS)
    if has_source_temporal:
        start = _first_non_empty(payload.get("validFrom"), payload.get("beginPosition"))
        end = _first_non_empty(payload.get("validTo"), payload.get("endPosition"))
        interval = _time_interval(start, end, resolution=payload.get("resolution"))
        for key in SOURCE_TEMPORAL_KEYS:
            payload.pop(key, None)
        payload.pop("resolution", None)
        if interval:
            existing = payload.get("time")
            if isinstance(existing, Mapping):
                merged = dict(interval)
                merged.update(existing)
                payload["time"] = merged
            else:
                payload["time"] = interval


def _promote_observing_location(payload: Dict[str, Any]) -> None:
    loc = payload.pop("observingLocation", None)
    loc_obj = _as_mapping(loc)
    if not loc_obj:
        return
    for key, value in loc_obj.items():
        if key in SOURCE_TEMPORAL_KEYS:
            continue
        if key not in payload or payload.get(key) in (None, "", [], {}):
            payload[key] = value
        elif isinstance(payload.get(key), list) or isinstance(value, list):
            payload[key] = _uniq_scalars([*_as_list(payload.get(key)), *_as_list(value)])
    loc_start, loc_end = _extract_interval(loc_obj)
    # Use the former wrapper's time only when the surrounding object has no
    # own temporal anchor.  If both exist, the surrounding
    # ObservingConfiguration remains the lifecycle anchor.
    if "time" not in payload and not any(key in payload for key in SOURCE_TEMPORAL_KEYS):
        interval = _time_interval(loc_start, loc_end)
        if interval:
            payload["time"] = interval


def _geometry_from_temporal_geometry(value: Any) -> Optional[Dict[str, Any]]:
    """Return the latest point geometry from a legacy temporalGeometry object."""
    obj = _as_mapping(value)
    coordinates = obj.get("coordinates")
    if isinstance(coordinates, list) and coordinates:
        latest = coordinates[-1]
        if isinstance(latest, list):
            try:
                numbers = [float(item) for item in latest[:3]]
            except (TypeError, ValueError):
                return None
            if len(numbers) >= 2:
                return {"type": "Point", "coordinates": numbers}
    if obj.get("type") == "Point" and isinstance(obj.get("coordinates"), list):
        return cast(Dict[str, Any], obj)
    return None


def _normalize_contact_arrays(payload: Dict[str, Any], registry: Dict[str, Dict[str, Any]], *, is_root_properties: bool = False) -> None:
    assignments: List[Dict[str, Any]] = []

    existing_assignments = payload.pop("contactReferences", None)
    for item in _as_list(existing_assignments):
        assignment = _assignment_from_contact(item, registry)
        if assignment:
            assignments.append(assignment)

    existing_contact_roles = payload.get("contactRoles")
    if existing_contact_roles is not None:
        payload.pop("contactRoles", None)
        for item in _as_list(existing_contact_roles):
            assignment = _assignment_from_contact(item, registry)
            if assignment:
                assignments.append(assignment)

    existing_contact_assignments = payload.get("contactAssignments")
    if existing_contact_assignments is not None:
        payload.pop("contactAssignments", None)
        for item in _as_list(existing_contact_assignments):
            assignment = _assignment_from_contact(item, registry)
            if assignment:
                assignments.append(assignment)

    raw_contacts = payload.get("contacts")
    if raw_contacts is not None:
        # At root properties, contacts is the reusable OGC Contact registry.  In
        # nested WMDR objects, contacts is interpreted as source/contextual party
        # information and becomes contactAssignments.
        payload.pop("contacts", None)
        for item in _as_list(raw_contacts):
            contact = _normalize_ogc_contact(item)
            if not contact:
                continue
            identifier = _contact_identifier(contact)
            contact["identifier"] = identifier
            if identifier in registry:
                registry[identifier] = _merge_contact(registry[identifier], contact)
            else:
                registry[identifier] = contact
            roles = _normalize_roles(contact.get("roles"))
            if not is_root_properties and roles:
                assignments.append({"contact": identifier, "roles": roles})

    if assignments:
        payload["contactAssignments"] = _uniq_dicts(assignments)


def _normalize_node(node: Any, registry: Dict[str, Dict[str, Any]], *, is_root: bool = False, is_root_properties: bool = False) -> Any:
    if isinstance(node, list):
        return [_normalize_node(item, registry) for item in node]
    if not isinstance(node, dict):
        return node

    payload: Dict[str, Any] = {}
    for key, value in node.items():
        # The root GeoJSON Feature.properties object needs special handling so
        # that its reusable ``contacts`` registry is not first interpreted as a
        # nested contextual contact list.
        if is_root and key == "properties":
            payload[key] = value
        elif key in {"beginPosition", "endPosition", "validFrom", "validTo", "contactReferences", "contactRoles", "contacts", "contactAssignments", "observingLocation"}:
            payload[key] = value
        else:
            payload[key] = _normalize_node(value, registry)

    if is_root and "id" in payload:
        payload["id"] = _normalize_facility_wsi(payload.get("id"))

    if not is_root and "temporalGeometry" in payload:
        temporal_geometry = payload.pop("temporalGeometry", None)
        if "geometry" not in payload or payload.get("geometry") in (None, "", [], {}):
            geometry = _geometry_from_temporal_geometry(temporal_geometry)
            if geometry:
                payload["geometry"] = geometry

    if (
        "keywords" in payload
        and "observingMethod" in payload
        and any(
            key in payload
            for key in (
                "time",
                "geometry",
                "instrument",
                "referenceSurface",
                "verticalDistanceFromReferenceSurface",
                "sourceOfObservation",
                "operatingStatus",
                "exposure",
                "serialNumber",
            )
        )
    ):
        payload.pop("keywords", None)

    _promote_observing_location(payload)
    _normalize_time_members(payload)
    _normalize_contact_arrays(payload, registry, is_root_properties=is_root_properties)

    # Normalize email/phone object shape if a reusable contact object is seen.
    if any(key in payload for key in ("emails", "phones", "addresses")) and any(key in payload for key in ("identifier", "organization", "name")):
        contact = _normalize_ogc_contact(payload)
        if contact:
            payload = contact

    props = payload.get("properties")
    if isinstance(props, dict):
        payload["properties"] = _normalize_node(props, registry, is_root_properties=True)

    if is_root_properties:
        name_values = _title_values(payload.pop("name", None))
        title_values = _title_values(payload.get("title"))
        existing_additional_titles = _title_values(payload.get("additionalTitles"))
        facility_titles = _uniq_scalars(name_values + title_values)
        if facility_titles:
            primary_title = facility_titles[0]
            payload["title"] = primary_title
            additional_titles = [
                title
                for title in _uniq_scalars(facility_titles[1:] + existing_additional_titles)
                if title != primary_title
            ]
            if additional_titles:
                payload["additionalTitles"] = additional_titles
            else:
                payload.pop("additionalTitles", None)
        elif "title" in payload:
            payload.pop("title", None)
        payload["contacts"] = sorted(registry.values(), key=lambda item: str(item.get("identifier")))

    return payload


def _finalize_wmdr2_value(value: Any, *, key: Optional[str] = None) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"interval"} and value.get("interval") == ["..", ".."]:
            return None
        return {
            child_key: (child_value if child_key in {"href", "url"} else _finalize_wmdr2_value(child_value, key=child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_finalize_wmdr2_value(item, key=key) for item in value]
    if isinstance(value, str) and key not in {"href", "url", "id", "identifier", "contact", "instrument", "reportingSchedules", "observingSchedules"}:
        return _compact_wmdr_code_value(value)
    return value


def _find_source_temporal_keys(value: Any, *, path: Tuple[str, ...] = ()) -> List[str]:
    """Return paths where source/transitional temporal keys remain."""
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



def _normalize_procedure_schedule_structure(record: Any) -> None:
    """Apply v0.3.1 procedure/schedule placement rules in-place.

    * ``reportingProcedures`` are not time-bound in the v0.3 UML model, so
      they must not carry ``time``.
    * ``observingProcedures`` are time-bound and reference reusable root-level
      schedules using ``observingSchedules``.
    * ``reportingProcedures`` reference reusable root-level schedules using
      ``reportingSchedules``.
    """
    if not isinstance(record, dict):
        return
    props = _as_mapping(record.get("properties"))
    if not isinstance(props, dict):
        return

    root_schedules: Dict[str, Dict[str, Any]] = {}
    for item in _as_list(props.get("schedules")):
        item_obj = _as_mapping(item)
        schedule = _normalize_schedule_object(item, kind="shared")
        if schedule:
            uid = _strip_text(schedule.get("uid"))
            if uid:
                root_schedules[uid] = schedule

    for series in _as_list(props.get("observationSeries")):
        series_obj = _as_mapping(series)
        if not isinstance(series_obj, dict):
            continue

        for item in _as_list(series_obj.pop("schedules", None)):
            schedule = _normalize_schedule_object(item, kind="observing")
            if schedule:
                uid = _strip_text(schedule.get("uid"))
                if uid:
                    root_schedules[uid] = schedule

        for proc in _as_list(series_obj.get("observingProcedures")):
            proc_obj = _as_mapping(proc)
            if not isinstance(proc_obj, dict):
                continue
            refs: List[str] = [str(ref) for ref in _as_list(proc_obj.get("observingSchedules")) if ref not in (None, "") and not isinstance(ref, Mapping)]
            for item in _as_list(proc_obj.pop("schedules", None)):
                schedule = _normalize_schedule_object(item, kind="observing")
                if schedule:
                    uid = _strip_text(schedule.get("uid"))
                    if uid:
                        root_schedules[uid] = schedule
                        refs.append(uid)
            if refs:
                proc_obj["observingSchedules"] = _uniq_scalars(refs)

        for proc in _as_list(series_obj.get("reportingProcedures")):
            proc_obj = _as_mapping(proc)
            if not isinstance(proc_obj, dict):
                continue
            proc_obj.pop("time", None)
            legacy_interval = _first_non_empty(
                proc_obj.pop("temporalReportingInterval", None),
                proc_obj.pop("temporalAggregate", None),
                proc_obj.pop("wmo.int:aggregationInterval", None),
            )
            refs: List[str] = [str(ref) for ref in _as_list(proc_obj.get("reportingSchedules")) if ref not in (None, "") and not isinstance(ref, Mapping)]
            if legacy_interval not in (None, "", [], {}):
                schedule = _normalize_schedule_object({"wmo.int:aggregationInterval": legacy_interval}, kind="reporting")
                if schedule:
                    uid = _strip_text(schedule.get("uid"))
                    if uid:
                        root_schedules[uid] = schedule
                        refs.append(uid)
            for item in _as_list(proc_obj.pop("schedules", None)):
                schedule = _normalize_schedule_object(item, kind="reporting")
                if schedule:
                    uid = _strip_text(schedule.get("uid"))
                    if uid:
                        root_schedules[uid] = schedule
                        refs.append(uid)
            # Also accept schedule objects accidentally embedded directly in reportingSchedules.
            for item in _as_list(proc_obj.get("reportingSchedules")):
                if isinstance(item, Mapping):
                    schedule = _normalize_schedule_object(item, kind="reporting")
                    if schedule:
                        uid = _strip_text(schedule.get("uid"))
                        if uid:
                            root_schedules[uid] = schedule
                            refs.append(uid)
            if refs:
                proc_obj["reportingSchedules"] = _uniq_scalars(refs)

    if root_schedules:
        props["schedules"] = sorted(root_schedules.values(), key=lambda item: str(item.get("uid")))
    elif "schedules" in props:
        props.pop("schedules", None)


def _normalize_facility_additional_ids(record: Any) -> Any:
    if not isinstance(record, dict):
        return record
    props = record.get("properties")
    if not isinstance(props, dict):
        return record
    primary_id = _normalize_facility_wsi(record.get("id")) if record.get("id") is not None else None
    candidates: List[str] = []
    candidates.extend(_facility_wsi_values(props.get("additionalIds")))
    legacy_identifiers = props.pop("identifiers", None)
    candidates.extend(_facility_wsi_values(legacy_identifiers))
    additional_ids = [
        candidate
        for candidate in _uniq_scalars(candidates)
        if _is_valid_wsi(candidate) and candidate != primary_id
    ]
    if additional_ids:
        props["additionalIds"] = additional_ids
    else:
        props.pop("additionalIds", None)
    return record


def normalize_wmdr2_record(record: Any) -> Any:
    """Normalize a WMDR2 record to v0.3.1 JSON conventions."""
    if not isinstance(record, dict):
        return record
    registry: Dict[str, Dict[str, Any]] = {}
    normalized = _normalize_node(copy.deepcopy(record), registry, is_root=True)
    _normalize_procedure_schedule_structure(normalized)
    finalized = _finalize_wmdr2_value(normalized)
    preserved = _preserve_nulls(finalized)
    cleaned = _clean_none(preserved)
    restored = _restore_null_sentinel(cleaned)
    restored = _normalize_facility_additional_ids(restored)
    remaining_source_temporal_keys = _find_source_temporal_keys(restored)
    if remaining_source_temporal_keys:
        raise ValueError(
            "WMDR2 v0.3.1 record still contains source temporal key(s): "
            + ", ".join(remaining_source_temporal_keys[:20])
        )
    return restored

# ---------------------------------------------------------------------------
# Conversion orchestration
# ---------------------------------------------------------------------------


def convert_record(record: Any, *, source_name: Optional[str] = None) -> Dict[str, Any]:
    return build_facility_feature(record, source_name=source_name)


def convert_file(input_path: Path, output_path: Path) -> Path:
    payload = _load_json(input_path)
    result = convert_record(payload, source_name=input_path.name)
    _write_json(output_path, result)
    return output_path


def _write_group_output(group: str, parts: Mapping[str, Any], output_root: Path) -> Path:
    facility = _as_mapping(parts.get("facility"))
    header = _as_mapping(parts.get("header"))
    observations = _as_list(parts.get("observationSeries"))
    deployments = _as_list(parts.get("deployments"))
    result = build_facility_feature(
        None,
        facility=facility,
        header=header,
        observations=observations,
        deployments=deployments,
        source_name=f"{group}.json",
    )
    out_path = output_root / f"{group}{OUTPUT_SUFFIX}"
    _write_json(out_path, result)
    return out_path


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
        if kind in {"header", "facility", "observationSeries", "deployments"} and path.stem.lower().endswith(("_header", "_facility", "_observations", "_deployments")):
            key = _part_group_key(path)
            grouped.setdefault(key, {})[kind] = payload
        else:
            standalone.append(path)

    written: List[Path] = []
    for path in standalone:
        rel = path.relative_to(input_path)
        target = output_path / rel
        if target.suffix.lower() != ".json":
            target = target.with_suffix(".json")
        written.append(convert_file(path, target))
        if verbose:
            print(f"converted {path} -> {target}")

    for group, parts in sorted(grouped.items()):
        target = _write_group_output(group, parts, output_path)
        written.append(target)
        if verbose:
            print(f"converted grouped parts {group} -> {target}")
    return written


def _load_code_list_labels(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            domain = _first_non_empty(row.get("domain"), row.get("codeList"), row.get("list"))
            code = _first_non_empty(row.get("code"), row.get("identifier"), row.get("id"), row.get("notation"))
            label = _first_non_empty(row.get("label"), row.get("prefLabel"), row.get("title"))
            if isinstance(domain, str) and isinstance(code, str) and isinstance(label, str):
                CODE_LIST_LABELS.setdefault(domain, {})[code.lstrip("_")] = label


def _load_code_list_labels_from_config(section: Mapping[str, Any], *, base_dir: Optional[Path], cli_path: Optional[Path]) -> None:
    if cli_path is not None:
        _load_code_list_labels(cli_path)
        return

    simple = _cfg_first(section, "code_list_labels", "codeListLabelsCsv")
    simple_path = _resolve_config_path(simple, base_dir=base_dir)
    if simple_path is not None:
        _load_code_list_labels(simple_path)

    # Historical config.yaml files may use:
    #
    # codeListLabels:
    #   files:
    #     - path: wmdr_observed_variable_labels.csv
    #
    # Load all such files if present.
    code_list_labels = _cfg_mapping(section.get("codeListLabels"))
    files = code_list_labels.get("files")
    for item in _as_list(files):
        item_map = _as_mapping(item)
        label_path = _resolve_config_path(item_map.get("path"), base_dir=base_dir)
        if label_path is not None:
            _load_code_list_labels(label_path)


def _collect_catalogue_items_from_records(record_paths: Iterable[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    contacts: List[Mapping[str, Any]] = []
    instruments: List[Mapping[str, Any]] = []
    for record_path in record_paths:
        try:
            record = _load_json(record_path)
        except Exception:
            continue
        props = _as_mapping(_as_mapping(record).get("properties"))
        for contact in _as_list(props.get("contacts")):
            contact_obj = _as_mapping(contact)
            if contact_obj:
                contacts.append(contact_obj)
        for instrument in _as_list(props.get("instruments")):
            instrument_obj = _as_mapping(instrument)
            if instrument_obj:
                instruments.append(instrument_obj)
    return _uniq_dicts(contacts), _uniq_dicts(instruments)


def _remove_inline_catalogue_items(record_paths: Iterable[Path]) -> None:
    for record_path in record_paths:
        try:
            record = _load_json(record_path)
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        props = _as_mapping(record.get("properties"))
        if not isinstance(props, Mapping):
            continue
        new_props = dict(props)
        new_props.pop("contacts", None)
        new_props.pop("instruments", None)
        record["properties"] = new_props
        _write_json(record_path, record)


def _write_catalogue_outputs(
    record_paths: Iterable[Path],
    *,
    contacts_path: Optional[Path],
    instruments_path: Optional[Path],
    remove_inline: bool = True,
) -> None:
    contacts, instruments = _collect_catalogue_items_from_records(record_paths)
    if contacts_path is not None:
        _write_json(contacts_path, {"contacts": contacts})
    if instruments_path is not None:
        _write_json(instruments_path, {"instruments": instruments})
    if remove_inline:
        _remove_inline_catalogue_items(record_paths)



def convert_payload(payload: Any, *, source_name: Optional[str] = None) -> Dict[str, Any]:
    """Backward-compatible public alias for in-memory conversion."""
    return convert_record(payload, source_name=source_name)


def convert_wmdr10_json_to_wmdr2_json(payload: Any) -> Dict[str, Any]:
    """Backward-compatible public alias used by older tests and notebooks."""
    return convert_record(payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert simplified WMDR10 JSON to WMDR2 v0.3.1 JSON. "
            "With no positional arguments, paths are read from config.yaml."
        )
    )
    parser.add_argument("input", nargs="?", help="Input JSON file or directory; overrides config source/input.")
    parser.add_argument("output", nargs="?", help="Output JSON file or directory; overrides config target/output.")
    parser.add_argument("--input", dest="input_opt", help="Input JSON file or directory; overrides config source/input.")
    parser.add_argument("--output", dest="output_opt", help="Output JSON file or directory; overrides config target/output.")
    # Backward-compatible aliases used by the XML converter and older E2E tests.
    parser.add_argument("--source", dest="source_opt", help="Alias for --input; overrides config source/input.")
    parser.add_argument("--target", dest="target_opt", help="Alias for --output; overrides config target/output.")
    parser.add_argument("--config", type=Path, help="Config file path. Defaults to config.yaml/config.yml found from the repo root.")
    parser.add_argument("--pattern", default=None, help="Input glob pattern for directory conversion.")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse into input subdirectories.")
    parser.add_argument("--code-list-labels", type=Path, help="Optional CSV with code-list labels; overrides config code-list label files.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_path = _discover_config_path(args.config)
    config_dir: Optional[Path] = config_path.parent if config_path is not None else None
    section: Dict[str, Any] = {}
    if config_path is not None:
        section = _cfg_section(_load_config(config_path))
        if args.verbose:
            print(_format_loaded_config_hint(config_path, section))
    elif not (args.input_opt or args.source_opt or args.input or args.output_opt or args.target_opt or args.output):
        parser.error("missing config.yaml; run from the repository root or pass --config, input and output paths")

    global DISCOVERY_POLICY
    DISCOVERY_POLICY = _normalize_discovery_policy(section)

    _load_code_list_labels_from_config(section, base_dir=config_dir, cli_path=args.code_list_labels)

    config_input = _cfg_first(section, "source", "input", "input_path")
    config_output = _cfg_first(section, "target", "output", "output_path")
    cli_input = args.input_opt or args.source_opt or args.input
    cli_output = args.output_opt or args.target_opt or args.output

    input_path = _resolve_cli_or_config_path(cli_input or config_input, base_dir=config_dir, from_config=cli_input is None)
    output_path = _resolve_cli_or_config_path(cli_output or config_output, base_dir=config_dir, from_config=cli_output is None)
    if input_path is None:
        parser.error("missing input path; set convert_wmdr10_json_to_wmdr2_json.source in config.yaml or pass an input path")
    if output_path is None:
        parser.error("missing output path; set convert_wmdr10_json_to_wmdr2_json.target in config.yaml or pass an output path")

    pattern = args.pattern or str(_cfg_first(section, "pattern") or DEFAULT_PATTERN)
    recursive_value = section.get("recursive")
    recursive = False if args.no_recursive else bool(True if recursive_value is None else recursive_value)

    written = convert_path(input_path, output_path, pattern=pattern, recursive=recursive, verbose=args.verbose)

    catalogues = _cfg_mapping(section.get("catalogues"))
    catalogues_enabled = bool(catalogues.get("enabled"))
    if catalogues_enabled:
        records_path = _resolve_config_path(_cfg_first(catalogues, "records_path", "recordsPath"), base_dir=config_dir)
        if records_path is None:
            records_path = output_path

        if records_path.resolve() == output_path.resolve():
            catalogue_records = written
        else:
            catalogue_records = convert_path(input_path, records_path, pattern=pattern, recursive=recursive, verbose=args.verbose)

        contacts_path = _resolve_config_path(_cfg_first(catalogues, "contacts_path", "contactsPath"), base_dir=config_dir)
        instruments_path = _resolve_config_path(_cfg_first(catalogues, "instruments_path", "instrumentsPath"), base_dir=config_dir)
        _write_catalogue_outputs(
            catalogue_records,
            contacts_path=contacts_path,
            instruments_path=instruments_path,
            remove_inline=True,
        )
        if args.verbose:
            if contacts_path is not None:
                print(f"wrote contacts catalogue -> {contacts_path}")
            if instruments_path is not None:
                print(f"wrote instruments catalogue -> {instruments_path}")

    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
