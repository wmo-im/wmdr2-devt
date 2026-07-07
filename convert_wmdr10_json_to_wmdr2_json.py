#!/usr/bin/env python3
"""Convert simplified WMDR 1.0 JSON records into WMDR2 v0.3.0 JSON.

The v0.3.0 target represented here follows the EA model under
``resources/ea/wmdr2``:

* ``Deployment`` is no longer a JSON/model class. The former deployment payload
  is folded into ``ObservingConfiguration.observingLocation`` and, where useful,
  into ``sourceOfObservation`` / instrument catalogue metadata.
* ``ObservingConfiguration.validFrom`` is the history anchor for each
  observation-series configuration.
* ``ObservingProcedure.validFrom`` is retained for procedure/schedule history.
* ``ReportingProcedure`` is the combined reporting/procedure object. It is
  emitted inline under each observation series as ``reportingProcedures`` and may carry
  ``spatialReportingInterval`` and one or more ``reportingSchedules``.
* JSCalendar-like reporting schedules support the WMO extension property
  ``wmo.int:diurnalBaseTime``.

The module deliberately keeps the public helper names used by the v0.2.x tests
and local workflows, but the semantics of deployment-related helpers now map to
``ObservingLocation`` rather than to a reusable ``Deployment`` class.
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
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

OGC_RECORD_CORE_CONF = "http://www.opengis.net/spec/ogcapi-records-1/1.0/conf/record-core"
WMDR2_CORE_CONF = "http://wigos.wmo.int/spec/wmdr/2/conf/core"
DEFAULT_PATTERN = "*.json"
OUTPUT_SUFFIX = ".json"
CANONICAL_SCHEDULE_START_DATE = "0001-01-01"
_NULL_SENTINEL = "__WMDR2_NULL__"

EMPTY_DISCOVERY_POLICY: Dict[str, Dict[str, List[str]]] = {
    "facility": {"keywords": [], "links": []},
    "observation": {"keywords": [], "links": []},
    "observingLocation": {"keywords": [], "links": []},
    # legacy spelling retained as an accepted configuration key
    "deployment": {"keywords": [], "links": []},
}
DEFAULT_DISCOVERY_POLICY: Dict[str, Dict[str, List[str]]] = {
    "facility": {"keywords": ["identifier", "name"], "links": ["onlineResource"]},
    "observation": {"keywords": [], "links": []},
    "observingLocation": {
        "keywords": ["manufacturer", "model", "serialNumber", "sourceOfObservation", "observingMethod"],
        "links": [],
    },
    "deployment": {
        "keywords": ["manufacturer", "model", "serialNumber", "sourceOfObservation", "observingMethod"],
        "links": [],
    },
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


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _non_empty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if _non_empty(value) or isinstance(value, bool):
            return value
    return None


def _clean_none(obj: Any, *, _path: Tuple[str, ...] = ()) -> Any:
    """Remove empty members while preserving positional empty lists for methods."""

    def preserve_empty_list(path: Tuple[str, ...]) -> bool:
        return len(path) >= 2 and path[-2:] in {
            ("temporalGeometry", "methods"),
            ("methods", "methods"),
        }

    if isinstance(obj, dict):
        cleaned = {k: _clean_none(v, _path=_path + (k,)) for k, v in obj.items()}
        return {k: v for k, v in cleaned.items() if v not in (None, "", [], {})}
    if isinstance(obj, list):
        cleaned = [_clean_none(v, _path=_path) for v in obj]
        return [v for v in cleaned if v not in ("", {}) and (v != [] or preserve_empty_list(_path))]
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


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _uniq_dicts(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        payload = _stable_json(item)
        if payload not in seen:
            seen.add(payload)
            out.append(item)
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


def _sanitize_id(text: Any) -> str:
    raw = str(text).strip()
    raw = re.sub(r"\s+", "-", raw)
    raw = re.sub(r"[^A-Za-z0-9._:/#-]+", "-", raw)
    return raw.strip("-") or "record"


def _slug(text: Any) -> str:
    raw = str(text).strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "-", raw)
    raw = re.sub(r"-{2,}", "-", raw).strip("-")
    return raw or "value"


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
    if isinstance(value, dict):
        value = _first_non_empty(
            value.get("href"), value.get("url"), value.get("value"), value.get("#text"), value.get("text")
        )
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


def _nil_reason(reason: Any = "unknown") -> Dict[str, str]:
    normalized = _normalize_code_value(reason)
    text = str(normalized).strip() if _non_empty(normalized) else "unknown"
    return {"nilReason": text}


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes", "y"}:
            return True
        if low in {"false", "0", "no", "n"}:
            return False
    return None


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


def _compact_wmdr_code_value(value: Any) -> Any:
    if isinstance(value, dict):
        value = _first_non_empty(
            value.get("href"), value.get("url"), value.get("value"), value.get("#text"), value.get("text")
        )
    if isinstance(value, str) and value.strip().startswith(("http://codes.wmo.int/wmdr/", "https://codes.wmo.int/wmdr/")):
        return _normalize_code_value(value)
    return value


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
    if isinstance(value, str) and key not in {"href", "url"}:
        return _compact_wmdr_code_value(value)
    return value


def _drop_source_metadata(obj: Any) -> Any:
    metadata_keys = {"@gml:id", "gml:id", "@id", "@xmlns", "xmlns", "schemaLocation"}
    if isinstance(obj, dict):
        return {k: _drop_source_metadata(v) for k, v in obj.items() if k not in metadata_keys}
    if isinstance(obj, list):
        return [_drop_source_metadata(v) for v in obj]
    return obj


def _normalize_display_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        parts = [_normalize_display_text(v) for v in value]
        text = "; ".join(p for p in parts if p)
        return text or None
    if isinstance(value, dict):
        value = _first_non_empty(value.get("description"), value.get("value"), value.get("#text"), value.get("text"))
    text = str(value).strip()
    if not text:
        return None
    if _is_unknown_token(text):
        return "unknown"
    return re.sub(r"\s+", " ", text)


def _normalize_description_value(value: Any) -> Optional[str]:
    if isinstance(value, list):
        parts = [_normalize_description_value(v) for v in value]
        return "\n\n".join(p for p in parts if p) or None
    if isinstance(value, dict):
        cleaned = _drop_source_metadata(value)
        text = _first_non_empty(cleaned.get("description"), cleaned.get("value"), cleaned.get("#text"), cleaned.get("text"))
        return _normalize_display_text(text) if text else _normalize_display_text(cleaned)
    return _normalize_display_text(value)


def _extract_scalar_values(value: Any) -> List[Any]:
    out: List[Any] = []
    if isinstance(value, dict):
        for key in ("href", "url", "value", "#text"):
            if _non_empty(value.get(key)):
                out.append(value[key])
        if not out:
            for nested in value.values():
                out.extend(_extract_scalar_values(nested))
    elif isinstance(value, list):
        for item in value:
            out.extend(_extract_scalar_values(item))
    else:
        out.append(value)
    return out


# ---------------------------------------------------------------------------
# Date/time, geometry and code-list helpers
# ---------------------------------------------------------------------------


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


def _time_interval(start: Any, end: Any, *, resolution: Optional[str] = None) -> Optional[Dict[str, Any]]:
    s = _normalize_date_value(start)
    e = _normalize_date_value(end) or ".."
    if s is None and e == "..":
        return None
    out: Dict[str, Any] = {"interval": [s or "..", e]}
    if resolution:
        out["resolution"] = resolution
    return out


def _extract_interval(obj: Mapping[str, Any]) -> Tuple[Any, Any]:
    time_obj = obj.get("time")
    if isinstance(time_obj, dict):
        interval = time_obj.get("interval")
        if isinstance(interval, list) and interval:
            return (interval[0] if len(interval) > 0 else None, interval[1] if len(interval) > 1 else None)
        return _first_non_empty(time_obj.get("date"), time_obj.get("timestamp")), None
    return (
        _first_non_empty(obj.get("validFrom"), obj.get("date"), obj.get("beginPosition"), obj.get("begin"), obj.get("start"), obj.get("dateEstablished")),
        _first_non_empty(obj.get("endPosition"), obj.get("end"), obj.get("stop"), obj.get("dateClosed")),
    )


def _entry_date(item: Any, fallback: str = "..") -> str:
    if isinstance(item, dict):
        return _normalize_date_value(_first_non_empty(item.get("validFrom"), item.get("date"), item.get("beginPosition"), item.get("begin"), item.get("from"), item.get("start"))) or fallback
    return fallback


def _valid_from(item: Any, fallback: str = "..") -> str:
    return _entry_date(item, fallback)


def _temporal_begin_date(item: Any) -> str:
    return _entry_date(item, "..")


def _parse_pos_lon_lat_z(raw: Any) -> Optional[List[Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        coords_any: Any = raw.get("coordinates")
        if isinstance(coords_any, list) and len(coords_any) >= 2:
            return coords_any
        for key in ("geoLocation", "pos", "value", "text", "geometry", "position"):
            val = raw.get(key)
            if isinstance(val, str):
                raw = val
                break
            if isinstance(val, dict):
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
    lat, lon = nums[0], nums[1]
    coords: List[Any] = [lon, lat]
    if len(nums) >= 3:
        z = nums[2]
        coords.append(int(round(z)) if abs(z - round(z)) < 1e-9 else z)
    return coords


def _geopositioning_methods(item: Any) -> List[str]:
    if not isinstance(item, dict):
        return []
    raw = item.get("geopositioningMethod")
    methods: List[str] = []
    for value in _as_list(raw):
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
        date = _temporal_begin_date(item)
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


def _temporal_geometry_extension(entries: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    coordinates: List[Any] = []
    dates: List[str] = []
    methods: List[List[str]] = []
    has_methods = False
    for entry in entries:
        if "coordinates" not in entry:
            continue
        coordinates.append(entry["coordinates"])
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


def _facility_geometry_from_entries(entries: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
    if isinstance(value, dict):
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
# Config / file handling
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
    return data


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
    legacy = cfg.get("convert_wmdr10_json_to_wmdr2_geojson")
    return legacy if isinstance(legacy, dict) else {}


def _format_loaded_config_hint(config_path: Optional[Path], section: Mapping[str, Any]) -> str:
    if config_path is None:
        return "No config file found; using CLI arguments only."
    keys = sorted(section.keys()) if section else []
    return f"Using config: {config_path} ({', '.join(keys) if keys else 'no converter section keys'})"


def _normalize_discovery_policy(section: Mapping[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    raw = section.get("discovery")
    if not isinstance(raw, dict):
        return copy.deepcopy(DEFAULT_DISCOVERY_POLICY)
    policy = copy.deepcopy(EMPTY_DISCOVERY_POLICY)
    # accept both old 'deployment' and new 'observingLocation'
    for entity in ("facility", "observation", "deployment", "observingLocation"):
        entity_cfg = raw.get(entity)
        if not isinstance(entity_cfg, dict):
            continue
        for bucket in ("keywords", "links"):
            values = entity_cfg.get(bucket)
            if isinstance(values, list):
                policy[entity][bucket] = [str(v).strip() for v in values if isinstance(v, str) and str(v).strip()]
    if policy.get("deployment") and not policy.get("observingLocation"):
        policy["observingLocation"] = copy.deepcopy(policy["deployment"])
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
        if any(k in payload for k in ("sourceOfObservation", "manufacturer", "serialNumber", "referenceSurface")):
            return "deployments"
        if any(k in payload for k in ("fileDateTime", "recordOwner", "dateStamp")):
            return "header"
        if any(k in payload for k in ("identifier", "name", "geospatialLocation")):
            return "facility"
    if isinstance(payload, list):
        first = next((x for x in payload if isinstance(x, dict)), None)
        if not first:
            return "unknown"
        if any(k in first for k in ("observedVariable", "observedProperty", "resultTime")):
            return "observationSeries"
        if any(k in first for k in ("sourceOfObservation", "manufacturer", "serialNumber", "referenceSurface")):
            return "deployments"
    return "unknown"


def _part_group_key(path: Path) -> str:
    stem = path.stem
    for suffix in ("_header", "_facility", "_observations", "_deployments"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


# ---------------------------------------------------------------------------
# Discovery, contacts, quantities, and status helpers
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


def _collect_discovery_values(entity_type: str, source: Mapping[str, Any], bucket: str) -> List[Any]:
    values: List[Any] = []
    # old callers may pass deployment; new policy uses observingLocation
    policy = DISCOVERY_POLICY.get(entity_type) or DISCOVERY_POLICY.get("observingLocation", {})
    for key in policy.get(bucket, []):
        for item in _as_list(source.get(key)):
            values.extend(_extract_scalar_values(item) if isinstance(item, dict) else [item])
    return values


def _about_link(href: str, *, title: Optional[str] = None, media_type: str = "text/html") -> Dict[str, Any]:
    link: Dict[str, Any] = {"href": href, "rel": "about", "type": media_type}
    if title:
        link["title"] = title
    return link


def _extract_links(source: Mapping[str, Any], entity_type: str) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    policy = DISCOVERY_POLICY.get(entity_type) or DISCOVERY_POLICY.get("observingLocation", {})
    for key in policy.get("links", []):
        for item in _as_list(source.get(key)):
            href: Optional[str] = None
            title: Optional[str] = None
            media_type = "text/html"
            if isinstance(item, str) and item.strip():
                href = item.strip()
            elif isinstance(item, dict):
                raw_href = _first_non_empty(item.get("url"), item.get("href"), item.get("linkage"), item.get("value"))
                if isinstance(raw_href, str) and raw_href.strip():
                    href = raw_href.strip()
                if isinstance(item.get("title"), str):
                    title = item["title"].strip()
                if isinstance(item.get("type"), str):
                    media_type = item["type"].strip()
            if href and href.startswith(("http://", "https://")):
                links.append(_about_link(href, title=title, media_type=media_type))
    return _uniq_dicts(links)


def _is_role_codelist_reference(value: Any) -> bool:
    """Return True for ISO/GMX role codelist references, not role values."""
    if not isinstance(value, str):
        return False
    text = value.strip().strip("<>")
    if not text:
        return False
    segment = _last_segment(text) or text
    return segment in {"CI_RoleCode", "RoleCode"} or text.endswith("#CI_RoleCode") or text.endswith("#RoleCode")


def _normalize_role(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        # Prefer the selected codelist value over the codelist URI/reference.
        # XML parsers commonly preserve both as @codeList and @codeListValue;
        # only the latter is the concrete ISO role.
        for key in ("codeListValue", "@codeListValue", "role", "value", "#text", "text", "name"):
            candidate = value.get(key)
            if _is_role_codelist_reference(candidate):
                continue
            role = _normalize_role(candidate)
            if role:
                return role
        # Treat href/url/codeList as role values only when they are not merely
        # links to the generic CI_RoleCode codelist.
        for key in ("href", "url", "codeList", "@codeList"):
            candidate = value.get(key)
            if _is_role_codelist_reference(candidate):
                continue
            role = _normalize_role(candidate)
            if role:
                return role
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().strip("<>")
    if _is_role_codelist_reference(text):
        return None
    segment = _last_segment(text) or text
    if _is_role_codelist_reference(segment) or _is_unknown_token(segment):
        return None
    return segment


def _normalize_roles(value: Any) -> List[str]:
    roles: List[str] = []
    for item in _as_list(value):
        role = _normalize_role(item)
        if role and role not in roles:
            roles.append(role)
    return roles


def _sanitize_contact_roles(contact: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop generic CI_RoleCode references and retain only concrete roles."""
    sanitized: Dict[str, Any] = dict(contact)
    roles = _normalize_roles(sanitized.get("roles") or sanitized.get("role"))
    sanitized.pop("role", None)
    if roles:
        sanitized["roles"] = roles
    else:
        sanitized.pop("roles", None)
    return sanitized


def _sanitize_contacts(contacts: Any) -> Optional[List[Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    for contact in _as_list(contacts):
        if isinstance(contact, Mapping):
            out.append(_sanitize_contact_roles(contact))
    return _uniq_dicts(_clean_none(out)) or None


def _normalize_contact(raw: Any) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not isinstance(raw, dict):
        return None, None
    payload = raw.get("responsibleParty") if isinstance(raw.get("responsibleParty"), dict) else raw
    if not isinstance(payload, dict):
        return None, None
    contact: Dict[str, Any] = {}
    identifier = _first_non_empty(payload.get("@id"), payload.get("@gml:id"), raw.get("@id"), raw.get("@gml:id"))
    if isinstance(identifier, str):
        contact["identifier"] = identifier
    for src_key, dst_key in (
        ("individualName", "name"), ("name", "name"), ("title", "name"), ("positionName", "position"),
        ("position", "position"), ("organisationName", "organization"), ("organizationName", "organization"), ("organization", "organization"),
    ):
        if dst_key in contact:
            continue
        value = _normalize_display_text(payload.get(src_key))
        if value:
            contact[dst_key] = value
    info = _as_dict(payload.get("contactInfo"))
    phone_obj = _as_dict(info.get("phone"))
    phones = [str(v).strip() for v in _as_list(phone_obj.get("voice")) if isinstance(v, str) and v.strip()]
    if phones:
        contact["phones"] = _uniq_scalars(phones)
    address_obj = _as_dict(info.get("address"))
    emails = [str(e).strip() for e in _as_list(address_obj.get("electronicMailAddress")) if isinstance(e, str) and "@" in e]
    if emails:
        contact["emails"] = _uniq_scalars(emails)
    direct_emails = [str(e).strip() for e in _as_list(payload.get("emails") or payload.get("email")) if isinstance(e, str) and "@" in e]
    if direct_emails:
        contact["emails"] = _uniq_scalars(_as_list(contact.get("emails")) + direct_emails)
    direct_phones = [str(v).strip() for v in _as_list(payload.get("phones") or payload.get("phone")) if isinstance(v, str) and v.strip()]
    if direct_phones:
        contact["phones"] = _uniq_scalars(_as_list(contact.get("phones")) + direct_phones)
    address: Dict[str, Any] = {}
    delivery_points = [dp.strip() for dp in _as_list(address_obj.get("deliveryPoint")) if isinstance(dp, str) and dp.strip()]
    if delivery_points:
        address["deliveryPoint"] = delivery_points
    for src_key in ("city", "administrativeArea", "postalCode", "country"):
        value = _normalize_display_text(address_obj.get(src_key))
        if value:
            address[src_key] = value
    if address:
        contact["addresses"] = [address]
    direct_addresses = [a for a in _as_list(payload.get("addresses") or payload.get("address")) if isinstance(a, dict)]
    if direct_addresses:
        existing_addresses = _as_list(contact.get("addresses"))
        contact["addresses"] = _uniq_dicts(existing_addresses + [dict(a) for a in direct_addresses])
    online = _as_dict(info.get("onlineResource"))
    href = online.get("url") or online.get("href")
    if isinstance(href, str) and href.strip():
        contact["links"] = [_about_link(href.strip())]
    direct_links = [dict(link) for link in _as_list(payload.get("links")) if isinstance(link, dict)]
    if direct_links:
        contact["links"] = _uniq_dicts(_as_list(contact.get("links")) + direct_links)
    roles = _normalize_roles(_first_non_empty(payload.get("role"), raw.get("role"), payload.get("roles"), raw.get("roles")))
    if roles:
        contact["roles"] = roles
    contact = _sanitize_contact_roles(contact)
    if not any(k in contact for k in ("name", "organization", "emails", "phones", "addresses", "links")):
        return None, None
    extension = dict(contact)
    valid_start, valid_end = _extract_interval(raw)
    valid_time = _time_interval(valid_start, valid_end)
    if valid_time:
        extension["validTime"] = valid_time
    return _clean_none(contact), _clean_none(extension)


def _collect_contacts(*groups: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ogc: List[Dict[str, Any]] = []
    temporal: List[Dict[str, Any]] = []
    for group in groups:
        for item in _as_list(group):
            a, b = _normalize_contact(item)
            if a:
                ogc.append(a)
            if b:
                temporal.append(b)
    return _uniq_dicts(ogc), _uniq_dicts(temporal)


def _normalize_official_status(value: Any) -> Optional[str]:
    parsed = _parse_bool(value)
    if parsed is True:
        return "primary"
    if parsed is False:
        return "additional"
    compact = _compact_wmdr_code_value(value)
    if isinstance(compact, str):
        text = compact.strip()
        if not text or _is_unknown_token(text):
            return None
        return text
    if compact in (None, "", [], {}):
        return None
    return str(compact)


def _normalize_quantity_value(value: Any) -> Any:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return text
    return value


def _quantity_from_source(source: Any) -> Optional[Dict[str, Any]]:
    uom = None
    value = source
    if isinstance(source, dict):
        value = _first_non_empty(source.get("#text"), source.get("text"), source.get("value"))
        uom = _first_non_empty(source.get("@uom"), source.get("uom"), source.get("unit"))
    parsed = _normalize_quantity_value(value)
    if parsed in (None, "", [], {}):
        return None
    out: Dict[str, Any] = {"value": parsed}
    if _non_empty(uom):
        out["uom"] = _compact_wmdr_code_value(uom) if isinstance(uom, str) else uom
    return _clean_none(out)


def _normalize_vertical_distance_from_reference_surface(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    source = _first_non_empty(
        raw.get("heightAboveLocalReferenceSurface"),
        raw.get("verticalDistanceFromReferenceSurface"),
        raw.get("distanceFromReferenceSurface"),
    )
    if source in (None, "", [], {}):
        return None
    if isinstance(source, list):
        for item in source:
            q = _quantity_from_source(item)
            if q:
                return q
        return None
    return _quantity_from_source(source)


# ---------------------------------------------------------------------------
# Facility history and environment
# ---------------------------------------------------------------------------


def _entry_value(item: Any, preferred_key: str) -> Any:
    if isinstance(item, dict):
        if preferred_key in item:
            return item.get(preferred_key)
        rest = {k: v for k, v in item.items() if k not in {"date", "validFrom", "beginPosition", "endPosition", "from", "start", "to", "end", "begin"}}
        if len(rest) == 1:
            return next(iter(rest.values()))
        return rest or None
    return item


def _temporal_object_from_item(item: Any, *, output_key: str, value_keys: Sequence[str], date_key: str = "validFrom") -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        raw_value = _first_non_empty(*(item.get(k) for k in value_keys))
        value = _normalize_code_value(raw_value) if isinstance(raw_value, (str, dict)) else raw_value
        if value in (None, "", [], {}) or _is_unknown_token(value):
            return None
        return {date_key: _entry_date(item), output_key: value}
    value = _normalize_code_value(item) if isinstance(item, str) else item
    if value in (None, "", [], {}) or _is_unknown_token(value):
        return None
    return {date_key: "..", output_key: value}


def _normalize_temporal_values(value: Any, *, output_key: str, value_keys: Sequence[str], date_key: str = "validFrom") -> Optional[List[Dict[str, Any]]]:
    records = [_temporal_object_from_item(item, output_key=output_key, value_keys=value_keys, date_key=date_key) for item in _as_list(value)]
    return _uniq_dicts(_clean_none([r for r in records if r])) or None


def _normalize_temporal_climate_zone(value: Any) -> Optional[List[Dict[str, Any]]]:
    return _normalize_temporal_values(value, output_key="climateZone", value_keys=("climateZone", "value", "href"))


def _normalize_temporal_surface_cover(value: Any) -> Optional[List[Dict[str, Any]]]:
    records = _normalize_temporal_values(value, output_key="surfaceCover", value_keys=("surfaceCover", "value", "href")) or []
    for record, item in zip(records, _as_list(value)):
        if isinstance(item, Mapping):
            classification = _first_non_empty(item.get("surfaceClassification"), item.get("classification"))
            if isinstance(classification, Mapping):
                classification = _first_non_empty(classification.get("href"), classification.get("value"), classification.get("surfaceClassification"))
            if _non_empty(classification):
                record["surfaceClassification"] = _compact_wmdr_code_value(classification) if isinstance(classification, str) else classification
    return _uniq_dicts(_clean_none(records)) or None


def _parse_nullable_number(value: Any) -> Optional[float]:
    if value in (None, "", [], {}) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or _is_unknown_token(text):
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _parse_two_value_number_array(value: Any) -> Optional[List[Optional[float]]]:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, list):
        values = [_parse_nullable_number(v) for v in value[:2]]
    elif isinstance(value, str):
        parts = [p for p in re.split(r"[,\s]+", value.strip()) if p]
        values = [_parse_nullable_number(p) for p in parts[:2]]
    else:
        values = [_parse_nullable_number(value)]
    while len(values) < 2:
        values.append(None)
    values = values[:2]
    if all(v is None for v in values):
        return None
    return values


def _parse_two_value_perimeter_array(value: Any) -> List[Optional[float]]:
    parsed = _parse_two_value_number_array(value) or [10.0, 50.0]
    if parsed[0] is None:
        parsed[0] = 10.0
    if parsed[1] is None:
        parsed[1] = 50.0
    return parsed


def _normalize_temporal_population(value: Any) -> Optional[List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    for item in _as_list(value):
        valid_from = _entry_date(item, "..")
        if isinstance(item, Mapping):
            population = _parse_two_value_number_array(_first_non_empty(item.get("population"), item.get("value"), item.get("href")))
            perimeter = _parse_two_value_perimeter_array(_first_non_empty(item.get("perimeter_km"), item.get("perimeterKm"), item.get("perimeter")))
        else:
            population = _parse_two_value_number_array(item)
            perimeter = [10.0, 50.0]
        if population is not None:
            records.append({"validFrom": valid_from, "population": population, "perimeter_km": perimeter})
    return _uniq_dicts(_clean_none(records)) or None


def _normalize_temporal_surface_roughness(value: Any) -> Optional[List[Dict[str, Any]]]:
    return _normalize_temporal_values(value, output_key="surfaceRoughness", value_keys=("surfaceRoughness", "roughness", "value", "href"))


def _first_operating_status_value(raw: Mapping[str, Any]) -> Any:
    raw_status = _first_non_empty(raw.get("operatingStatus"), raw.get("instrumentOperatingStatus"), raw.get("status"))
    for item in _normalize_operating_status(raw_status):
        if _non_empty(item.get("operatingStatus")):
            return item.get("operatingStatus")
    if isinstance(raw_status, Mapping):
        raw_status = _first_non_empty(raw_status.get("operatingStatus"), raw_status.get("instrumentOperatingStatus"), raw_status.get("status"), raw_status.get("value"), raw_status.get("href"))
    return _compact_wmdr_code_value(raw_status) if isinstance(raw_status, str) else raw_status


def _normalize_operating_status(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            raw = _first_non_empty(item.get("operatingStatus"), item.get("instrumentOperatingStatus"), item.get("status"), item.get("value"), item.get("href"))
            date = _entry_date(item)
        else:
            raw = item
            date = ".."
        val = _compact_wmdr_code_value(raw) if isinstance(raw, str) else raw
        if _non_empty(val):
            out.append({"validFrom": date, "operatingStatus": val})
    return _uniq_dicts(_clean_none(out))


def _normalize_temporal_instrument_operating_status(value: Any) -> Optional[List[Dict[str, Any]]]:
    records = []
    for item in _normalize_operating_status(value if isinstance(value, Mapping) else {"instrumentOperatingStatus": value}):
        if "operatingStatus" in item:
            records.append({"validFrom": item.get("validFrom"), "instrumentOperatingStatus": item.get("operatingStatus")})
    return _uniq_dicts(_clean_none(records)) or None


def _normalize_historical_official_status(value: Any, *, fallback_date: str = "..") -> Optional[List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            raw = _first_non_empty(item.get("officialStatus"), item.get("status"), item.get("value"), item.get("href"))
            date = _entry_date(item, fallback=fallback_date)
        else:
            raw = item
            date = fallback_date
        status = _normalize_official_status(raw)
        if status:
            # The XMI currently names this history anchor 'date'.
            records.append({"date": date, "officialStatus": status})
    return _uniq_dicts(_clean_none(records)) or None


def _normalize_temporal_official_status(value: Any, *, fallback_date: str = "..", default_unknown: bool = False) -> Optional[List[Dict[str, Any]]]:
    if value is None and default_unknown:
        return [{"date": fallback_date, "officialStatus": "unknown"}]
    return _normalize_historical_official_status(value if isinstance(value, Mapping) else {"officialStatus": value}, fallback_date=fallback_date)


def _normalize_program_affiliations(value: Any) -> Optional[List[Any]]:
    values: List[Any] = []
    for item in _as_list(value):
        raw = _first_non_empty(item.get("programAffiliation"), item.get("program"), item.get("href"), item.get("value")) if isinstance(item, Mapping) else item
        if _non_empty(raw):
            values.append(_compact_wmdr_code_value(raw) if isinstance(raw, str) else raw)
    return _uniq_scalars(values) or None


def _program_affiliation_values(value: Any) -> Optional[List[Any]]:
    return _normalize_program_affiliations(value)


def _normalize_historical_program_affiliation(value: Any, *, fallback_date: str) -> Optional[List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []

    def add_record(item: Mapping[str, Any], reporting_status: Any = None, *, status_date: Any = None) -> None:
        record: Dict[str, Any] = {"validFrom": _normalize_date_value(status_date) or _entry_date(item, fallback=fallback_date)}
        programs = _normalize_program_affiliations(_first_non_empty(item.get("programAffiliation"), item.get("program"), item.get("href"), item.get("value")))
        if programs:
            record["program"] = programs[0] if len(programs) == 1 else programs
        program_id = _first_non_empty(item.get("programSpecificFacilityId"), item.get("programSpecificFacilityIds"))
        if _non_empty(program_id):
            record["programSpecificFacilityId"] = program_id
        status = reporting_status if reporting_status is not None else _first_non_empty(item.get("reportingStatus"), item.get("declaredReportingStatus"))
        if _non_empty(status):
            record["reportingStatus"] = _compact_wmdr_code_value(status) if isinstance(status, str) else status
        if len(record) > 1:
            records.append(record)

    for item in _as_list(value):
        if isinstance(item, Mapping):
            statuses = _as_list(_first_non_empty(item.get("reportingStatus"), item.get("declaredReportingStatus")))
            dict_statuses = [s for s in statuses if isinstance(s, Mapping)]
            if dict_statuses:
                for status_item in dict_statuses:
                    status_value = _first_non_empty(status_item.get("reportingStatus"), status_item.get("declaredReportingStatus"), status_item.get("status"), status_item.get("value"), status_item.get("href"))
                    add_record(item, status_value, status_date=_entry_date(status_item, fallback=_entry_date(item, fallback=fallback_date)))
            else:
                add_record(item)
        elif _non_empty(item):
            records.append({"validFrom": fallback_date, "program": _compact_wmdr_code_value(item) if isinstance(item, str) else item})
    return _uniq_dicts(_clean_none(records)) or None


def _normalize_temporal_program_affiliation(value: Any, *, fallback_date: str = "..") -> Optional[List[Dict[str, Any]]]:
    return _normalize_historical_program_affiliation(value, fallback_date=fallback_date)


def _normalize_historical_territory(value: Any, *, fallback_date: str) -> Optional[List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            territory = _first_non_empty(item.get("territory"), item.get("territoryName"), item.get("href"), item.get("value"))
            date = _entry_date(item, fallback=fallback_date)
        else:
            territory = item
            date = fallback_date
        if _non_empty(territory):
            records.append({"validFrom": date, "territory": _compact_wmdr_code_value(territory) if isinstance(territory, str) else territory})
    return _uniq_dicts(_clean_none(records)) or None


def _normalize_population_value(value: Any) -> Any:
    if isinstance(value, dict):
        raw_population = _first_non_empty(value.get("population"), value.get("value"), value.get("href"))
        population = _parse_two_value_number_array(raw_population)
        if population is None:
            return raw_population
        raw_perimeter = _first_non_empty(value.get("perimeter_km"), value.get("perimeterKm"), value.get("perimeter"))
        perimeter = _parse_two_value_number_array(raw_perimeter) or [10.0, 50.0]
        return {"population": population, "perimeter_km": perimeter}
    parsed = _parse_two_value_number_array(value)
    return parsed if parsed is not None else value


def _topography_object(source: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}
    container = _as_dict(source.get("topographyBathymetry") or source.get("topography") or source.get("bathymetry"))
    merged: Dict[str, Any] = dict(container)
    for key in ("localTopography", "relativeElevation", "topographicContext", "altitudeOrDepth"):
        if key in source and key not in merged:
            merged[key] = source[key]
    for key in ("localTopography", "relativeElevation", "topographicContext", "altitudeOrDepth"):
        value = merged.get(key)
        if isinstance(value, Mapping):
            value = _first_non_empty(value.get("value"), value.get("href"), value.get("#text"), value.get("text"))
        if _non_empty(value) or isinstance(value, (int, float, bool)):
            out[key] = _compact_wmdr_code_value(value) if isinstance(value, str) else value
    return _clean_none(out) or None


def _normalize_historical_environment(facility: Mapping[str, Any], *, fallback_date: str) -> Optional[List[Dict[str, Any]]]:
    env_sources: List[Mapping[str, Any]] = []
    if isinstance(facility.get("environment"), dict):
        env_sources.append(facility["environment"])
    env_sources.append(facility)
    by_date: Dict[str, Dict[str, Any]] = {}

    def record(date: str) -> Dict[str, Any]:
        return by_date.setdefault(date, {"validFrom": date})

    def add(date: str, key: str, value: Any) -> None:
        if _non_empty(value) or isinstance(value, bool):
            record(date)[key] = value

    def add_history(source_value: Any, key: str, value_keys: Sequence[str], *, transform: Optional[Any] = None) -> None:
        for item in _as_list(source_value):
            date = _entry_date(item, fallback=fallback_date)
            raw_value = _first_non_empty(*(item.get(k) for k in value_keys)) if isinstance(item, dict) else item
            if raw_value in (None, "", [], {}):
                continue
            value = transform(item if key == "population" and isinstance(item, dict) else raw_value) if transform else (_compact_wmdr_code_value(raw_value) if isinstance(raw_value, str) else raw_value)
            if key == "population" and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    add(date, sub_key, sub_value)
            else:
                add(date, key, value)

    for source in env_sources:
        for name in ("temporalClimateZone", "historicalClimateZone", "climateZone"):
            add_history(source.get(name), "climateZone", ("climateZone", "value", "href"))
        for name in ("temporalPopulation", "historicalPopulation", "population"):
            add_history(source.get(name), "population", ("population", "value", "href"), transform=_normalize_population_value)
        for name in ("temporalSurfaceCover", "historicalSurfaceCover", "surfaceCover"):
            add_history(source.get(name), "surfaceCover", ("surfaceCover", "value", "href"))
        for name in ("temporalSurfaceRoughness", "historicalSurfaceRoughness", "surfaceRoughness"):
            add_history(source.get(name), "surfaceRoughness", ("surfaceRoughness", "roughness", "value", "href"))
        for item in _as_list(source.get("environment")):
            if isinstance(item, dict):
                date = _entry_date(item, fallback=fallback_date)
                r = record(date)
                for key, value in item.items():
                    if key in {"date", "validFrom"}:
                        continue
                    if _non_empty(value) or isinstance(value, bool):
                        r[key] = value
        topo = _topography_object(source)
        if topo:
            add(_entry_date(source, fallback=fallback_date), "topographyBathymetry", topo)
    return _uniq_dicts(_clean_none([by_date[k] for k in sorted(by_date)])) or None


def _normalize_environment(facility: Mapping[str, Any]) -> Optional[List[Dict[str, Any]]]:
    start, _ = _extract_interval(facility)
    return _normalize_historical_environment(facility, fallback_date=_normalize_date_value(start) or "..")


def _normalize_temporal_data_policy(value: Any, *, fallback_date: str = "..") -> Optional[List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if not isinstance(item, Mapping):
            if _non_empty(item):
                records.append({"validFrom": fallback_date, "dataPolicy": _compact_wmdr_code_value(item)})
            continue
        raw_policy = _first_non_empty(item.get("dataPolicy"), item.get("policy"), item.get("href"), item.get("value"))
        if not _non_empty(raw_policy):
            continue
        record: Dict[str, Any] = {
            "validFrom": _entry_date(item, fallback=fallback_date),
            "dataPolicy": _compact_wmdr_code_value(raw_policy) if isinstance(raw_policy, str) else raw_policy,
        }
        if _non_empty(item.get("attribution")):
            record["attribution"] = _preserve_nulls(item.get("attribution"))
        records.append(record)
    return _uniq_dicts(_clean_none(records)) or None


def _facility_set_refs(value: Any) -> Optional[List[str]]:
    refs: List[str] = []
    for item in _as_list(value):
        raw = _first_non_empty(item.get("facilitySet"), item.get("facilitySets"), item.get("href"), item.get("value"), item.get("identifier"), item.get("id")) if isinstance(item, dict) else item
        if _non_empty(raw):
            refs.append(f"facilitySet:{_sanitize_id(_compact_wmdr_code_value(raw))}")
    return _uniq_scalars(refs) or None


def facility_set_catalog_entry(value: Any, *, description: Optional[str] = None) -> Dict[str, Any]:
    code = _sanitize_id(_compact_wmdr_code_value(value))
    entry: Dict[str, Any] = {"uid": f"facilitySet:{code}", "title": code}
    if _non_empty(description):
        entry["description"] = str(description)
    return entry


def facility_set_catalog(values: Any) -> Dict[str, Any]:
    entries = [facility_set_catalog_entry(ref.removeprefix("facilitySet:")) for ref in (_facility_set_refs(values) or [])]
    return {"facilitySets": entries}


def _copy_known_facility_properties(facility: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("facilityType", "wmoRegion", "surfaceCoverClassification", "timeZone", "regionOfOrigin"):
        value = facility.get(key)
        if _non_empty(value):
            out[key] = _compact_wmdr_code_value(value) if isinstance(value, str) else value
    facility_sets = _facility_set_refs(facility.get("facilitySet") or facility.get("facilitySets"))
    if facility_sets:
        out["facilitySets"] = facility_sets
    return out


# ---------------------------------------------------------------------------
# Instruments, observing locations, schedules, reporting
# ---------------------------------------------------------------------------


def _legacy_deployment_source_identifier(raw: Mapping[str, Any], *, index: int, facility_id: str) -> str:
    raw_id = _first_non_empty(raw.get("identifier"), raw.get("id"), raw.get("@gml:id"), raw.get("@id"), raw.get("uuid"))
    if raw_id:
        return _sanitize_id(raw_id)
    seed = _stable_json(_clean_none(dict(raw)))
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return _sanitize_id(f"{facility_id}:observing-location:{index}:{digest}")


# legacy name retained; no Deployment class is emitted in v0.3.0
def _deployment_source_identifier(raw: Mapping[str, Any], *, index: int, facility_id: str) -> str:
    return _legacy_deployment_source_identifier(raw, index=index, facility_id=facility_id)


def _normalize_instrument_observed_property(raw: Mapping[str, Any]) -> Optional[List[Any]]:
    value = _first_non_empty(raw.get("observedProperty"), raw.get("observedVariable"), raw.get("observableVariable"))
    values: List[Any] = []
    for item in _as_list(value):
        raw_value = _first_non_empty(item.get("observedProperty"), item.get("observedVariable"), item.get("href"), item.get("value"), item.get("description"), item.get("label"), item.get("name")) if isinstance(item, dict) else item
        compact = _compact_wmdr_code_value(raw_value) if isinstance(raw_value, (str, dict)) else raw_value
        if _non_empty(compact) or isinstance(compact, (int, float, bool)):
            values.append(compact)
    return _uniq_scalars(values) or None


def _normalize_instrument_observed_geometry(raw: Mapping[str, Any]) -> Optional[Any]:
    value = _first_non_empty(raw.get("observedGeometry"), raw.get("observedGeometryType"), raw.get("observableGeometry"))
    compact = _compact_wmdr_code_value(value) if isinstance(value, (str, dict)) else value
    return compact if _non_empty(compact) else None


def _normalize_vertical_range(raw: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    source = _first_non_empty(raw.get("verticalRange"), raw.get("observableVerticalRange"))
    if isinstance(source, Mapping):
        minimum = _normalize_quantity_value(_first_non_empty(source.get("min"), source.get("minimum"), source.get("lower")))
        maximum = _normalize_quantity_value(_first_non_empty(source.get("max"), source.get("maximum"), source.get("upper")))
    else:
        minimum = _normalize_quantity_value(_first_non_empty(raw.get("verticalRangeMin"), raw.get("verticalRangeMinimum"), raw.get("minimumVerticalRange")))
        maximum = _normalize_quantity_value(_first_non_empty(raw.get("verticalRangeMax"), raw.get("verticalRangeMaximum"), raw.get("maximumVerticalRange")))
    if minimum is None or maximum is None:
        return None
    try:
        return {"min": float(minimum), "max": float(maximum)}
    except Exception:
        return None


def _instrument_source_values(raw: Mapping[str, Any]) -> Tuple[Any, Any]:
    instrument = _as_dict(raw.get("instrument") or raw.get("equipment"))
    return (
        _first_non_empty(raw.get("manufacturer"), instrument.get("manufacturer"), instrument.get("make")),
        _first_non_empty(raw.get("model"), instrument.get("model"), instrument.get("type")),
    )


def _is_substantive_instrument_value(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, str) and _is_unknown_token(value):
        return False
    return True


def _instrument_record_id(raw: Mapping[str, Any], *, facility_id: str) -> Optional[str]:
    manufacturer, model = _instrument_source_values(raw)
    vertical_range = _normalize_vertical_range(raw)
    if not any(_is_substantive_instrument_value(v) for v in (manufacturer, model, vertical_range)):
        return None
    parts = [_sanitize_id(str(v)) for v in (manufacturer, model) if _is_substantive_instrument_value(v)]
    if parts:
        return f"instrument:{'--'.join(parts)}"
    seed = _stable_json({"verticalRange": vertical_range})
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    return f"instrument:vertical-range--{digest}"


def _instrument_ref_for_deployment(raw: Mapping[str, Any], *, facility_id: str) -> Optional[str]:
    return _instrument_record_id(raw, facility_id=facility_id)


def _scalar_reference(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item
    return None


def _observing_method_candidates(raw: Mapping[str, Any]) -> List[Any]:
    instrument = _as_dict(raw.get("instrument") or raw.get("equipment"))
    return [instrument.get("observingMethod"), instrument.get("observingMethods"), instrument.get("method"), raw.get("observingMethod"), raw.get("observingMethods"), raw.get("method"), raw.get("observingMethodDetails")]


def _normalize_observing_method_values(raw: Mapping[str, Any]) -> List[Any]:
    values: List[Any] = []
    for candidate in _observing_method_candidates(raw):
        for item in _as_list(candidate):
            value = _normalize_code_or_nil_reason(item)
            if value is not None:
                values.append(value)
    return _uniq_dicts(values)  # type: ignore[arg-type]


def _normalize_instrument_observing_methods(raw: Mapping[str, Any]) -> Optional[List[Any]]:
    values = [v for v in _normalize_observing_method_values(raw) if not (isinstance(v, Mapping) and "nilReason" in v)]
    return values or None


def _normalize_instrument(raw: Mapping[str, Any], *, facility_id: str) -> Optional[Dict[str, Any]]:
    instrument_id = _instrument_record_id(raw, facility_id=facility_id)
    if not instrument_id:
        return None
    manufacturer, model = _instrument_source_values(raw)
    return _clean_none({
        "uid": instrument_id,
        "manufacturer": manufacturer if _is_substantive_instrument_value(manufacturer) else None,
        "model": model if _is_substantive_instrument_value(model) else None,
        "observingMethods": _normalize_instrument_observing_methods(raw),
        "verticalRange": _normalize_vertical_range(raw),
    })


def _merge_catalogue_instrument(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if not _non_empty(value):
            continue
        if key not in merged or not _non_empty(merged.get(key)):
            merged[key] = value
        elif isinstance(merged[key], list):
            merged[key] = _uniq_scalars([*merged[key], *_as_list(value)])
        elif isinstance(merged[key], dict) and isinstance(value, Mapping):
            nested = dict(merged[key])
            for nk, nv in value.items():
                if nk not in nested or not _non_empty(nested.get(nk)):
                    nested[nk] = nv
            merged[key] = nested
    return _clean_none(merged)


def _normalize_instruments(deployments: Sequence[Mapping[str, Any]], *, facility_id: str) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for dep in deployments:
        instrument = _normalize_instrument(dep, facility_id=facility_id)
        if not instrument:
            continue
        iid = instrument.get("uid")
        if isinstance(iid, str):
            by_id[iid] = _merge_catalogue_instrument(by_id.get(iid, {}), instrument)
    return list(by_id.values())


def _observing_location_from_source(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    entries = _facility_temporal_geometry_entries(raw)
    geom = _facility_geometry_from_entries(entries)
    if not geom:
        coords = _parse_pos_lon_lat_z(_first_non_empty(raw.get("geospatialLocation"), raw.get("geometry"), raw.get("geoLocation")))
        if coords:
            geom = {"type": "Point", "coordinates": coords}
    out: Dict[str, Any] = {
        "geometry": geom,
        "referenceSurface": _compact_wmdr_code_value(_first_non_empty(raw.get("referenceSurface"), raw.get("localReferenceSurface"))),
        "relativeLocation": _normalize_display_text(_first_non_empty(raw.get("relativeLocation"), raw.get("location"))),
        "verticalDistanceFromReferenceSurface": _normalize_vertical_distance_from_reference_surface(raw),
    }
    return _clean_none(out) or None


# legacy public helper name: returns a list containing one ObservingLocation-like object
def _normalize_deployment(raw: Mapping[str, Any], *, index: int, facility_id: str, schedule_registry: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    loc = _observing_location_from_source(raw)
    if not loc:
        return []
    return [loc]


# legacy public helper name; no deployment id is emitted in v0.3.0
def _deployment_record_id(raw: Mapping[str, Any], *, index: int, facility_id: str) -> str:
    return f"observingLocation:{_deployment_source_identifier(raw, index=index, facility_id=facility_id)}"


def _normalize_diurnal_time(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        value = _first_non_empty(value.get("value"), value.get("#text"), value.get("href"))
    if value in (None, "", [], {}):
        return None
    text = str(value).strip().rstrip("Z")
    match = re.fullmatch(r"(?P<h>\d{1,2})(?::(?P<m>\d{1,2}))?(?::(?P<s>\d{1,2}))?", text)
    if not match:
        return text
    h = max(0, min(23, int(match.group("h"))))
    m = max(0, min(59, int(match.group("m") or 0)))
    s = max(0, min(59, int(match.group("s") or 0)))
    return f"{h:02d}:{m:02d}:{s:02d}"


def _schedule_start_datetime(raw: Mapping[str, Any]) -> str:
    def as_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    hour = max(0, min(23, as_int(_first_non_empty(raw.get("startHour"), raw.get("hour")), 0)))
    minute = max(0, min(59, as_int(_first_non_empty(raw.get("startMinute"), raw.get("minute")), 0)))
    return f"{CANONICAL_SCHEDULE_START_DATE}T{hour:02d}:{minute:02d}:00"


def _iso_duration(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        value = _first_non_empty(value.get("value"), value.get("#text"), value.get("href"))
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if re.match(r"^P", text) else None


def _schedule_uid_from_event(event_without_uid: Mapping[str, Any]) -> str:
    digest = hashlib.sha1(_stable_json(event_without_uid).encode("utf-8")).hexdigest()[:12]
    return f"schedule_{digest}"


def _flatten_schedule_candidates(value: Any, *, context: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    context = context or {}
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        merged: Dict[str, Any] = dict(context)
        merged.update(item)
        for nested_key in ("coverage", "Coverage", "sampling", "Sampling", "reporting"):
            nested = item.get(nested_key)
            if isinstance(nested, dict):
                n = dict(merged)
                n.update(nested)
                out.append(n)
        if any(k in item for k in ("temporalSamplingInterval", "samplingInterval", "sampleInterval", "temporalAggregate", "temporalReportingInterval", "reportingInterval", "internationalExchange", "uom", "dataPolicy", "levelOfData", "timeliness", "startHour", "diurnalBaseTime", "wmo.int:diurnalBaseTime", "spatialReportingInterval")):
            out.append(merged)
    return out


def _recurrence_rule_from_interval(interval: Optional[str]) -> Dict[str, Any]:
    rule: Dict[str, Any] = {"@type": "RecurrenceRule", "frequency": "daily"}
    if not interval:
        return rule
    if m := re.fullmatch(r"PT(\d+)M", interval):
        rule = {"@type": "RecurrenceRule", "frequency": "minutely"}
        if int(m.group(1)) != 1:
            rule["interval"] = int(m.group(1))
    elif m := re.fullmatch(r"PT(\d+)H", interval):
        rule = {"@type": "RecurrenceRule", "frequency": "hourly"}
        if int(m.group(1)) != 1:
            rule["interval"] = int(m.group(1))
    elif m := re.fullmatch(r"PT(\d+)S", interval):
        rule = {"@type": "RecurrenceRule", "frequency": "secondly"}
        if int(m.group(1)) != 1:
            rule["interval"] = int(m.group(1))
    return rule


def _jscalendar_observing_schedule(raw: Mapping[str, Any], *, time_zone: str = "UTC") -> Optional[Dict[str, Any]]:
    reporting_any: Any = raw.get("reporting")
    reporting: Mapping[str, Any] = cast(Mapping[str, Any], reporting_any) if isinstance(reporting_any, Mapping) else {}
    interval = _iso_duration(_first_non_empty(raw.get("temporalSamplingInterval"), raw.get("samplingInterval"), raw.get("sampleInterval"), raw.get("interval"), raw.get("temporalAggregate"), reporting.get("temporalAggregate"), reporting.get("temporalReportingInterval")))
    has_window = any(_non_empty(raw.get(k)) for k in ("startMonth", "endMonth", "startWeekday", "endWeekday", "startHour", "endHour", "startMinute", "endMinute"))
    if not interval and not has_window:
        return None
    event_without_uid: Dict[str, Any] = {
        "@type": "Event",
        "start": _schedule_start_datetime(raw),
        "timeZone": time_zone,
        "duration": interval or "P1D",
        "recurrenceRules": [_recurrence_rule_from_interval(interval)],
    }
    sampling_frequency = _iso_duration(_first_non_empty(raw.get("temporalSamplingInterval"), raw.get("samplingInterval"), raw.get("sampleInterval"), raw.get("interval")))
    if sampling_frequency:
        event_without_uid["wmo.int:samplingFrequency"] = sampling_frequency
    aggregation_interval = _iso_duration(_first_non_empty(raw.get("temporalAggregate"), raw.get("temporalReportingInterval"), raw.get("reportingInterval"), reporting.get("temporalAggregate"), reporting.get("temporalReportingInterval"), reporting.get("reportingInterval")))
    if aggregation_interval:
        event_without_uid["wmo.int:aggregationInterval"] = aggregation_interval
    diurnal = _normalize_diurnal_time(_first_non_empty(raw.get("wmo.int:diurnalBaseTime"), raw.get("diurnalBaseTime"), reporting.get("wmo.int:diurnalBaseTime"), reporting.get("diurnalBaseTime")))
    if diurnal:
        event_without_uid["wmo.int:diurnalBaseTime"] = diurnal
    event = dict(event_without_uid)
    event["uid"] = _schedule_uid_from_event(event_without_uid)
    return _clean_none(event)


def _jscalendar_reporting_schedule(raw: Mapping[str, Any], *, time_zone: str = "UTC") -> Optional[Dict[str, Any]]:
    # For now reporting schedules use the same JSCalendar subset as observing schedules.
    return _jscalendar_observing_schedule(raw, time_zone=time_zone)


def _strip_misspelled_wmi_extensions(value: Any) -> Any:
    """Remove the historic misspelled ``wmi.int`` JSCalendar extension aliases."""
    if isinstance(value, dict):
        return {k: _strip_misspelled_wmi_extensions(v) for k, v in value.items() if not k.startswith("wmi.int:")}
    if isinstance(value, list):
        return [_strip_misspelled_wmi_extensions(item) for item in value]
    return value


def _register_observing_schedule_refs(value: Any, *, schedule_registry: Dict[str, Dict[str, Any]], time_zone: str = "UTC") -> Optional[List[str]]:
    """Legacy helper retained for tests/local users.

    v0.3.0 still stores reusable JSCalendar schedule objects, while observing
    procedures hold references to them. This helper deduplicates schedules by
    uid and returns those uid references.
    """
    refs: List[str] = []
    for group in _as_list(value):
        for item in _as_list(group):
            if not isinstance(item, Mapping):
                continue
            schedule = _jscalendar_observing_schedule(item, time_zone=time_zone)
            if not schedule:
                continue
            refs.append(_schedule_registry_add(schedule_registry, schedule))
    return _uniq_scalars(refs) or None


def _deployment_temporal_serial_number(value: Any) -> Optional[List[Dict[str, Any]]]:
    """Legacy helper retained after Deployment removal.

    Serial numbers now describe the observing configuration/source instance, not
    a reusable Deployment class. The helper returns v0.3.0 temporal objects.
    """
    records: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            serial = _first_non_empty(item.get("serialNumber"), item.get("serial"), item.get("value"), item.get("#text"))
            date = _entry_date(item)
        else:
            serial = item
            date = ".."
        if _non_empty(serial):
            records.append({"validFrom": date, "serialNumber": serial})
    return _uniq_dicts(_clean_none(records)) or None


def _schedule_registry_add(schedule_registry: Dict[str, Dict[str, Any]], schedule: Dict[str, Any]) -> str:
    uid = schedule.get("uid")
    if not isinstance(uid, str) or not uid:
        uid = _schedule_uid_from_event({k: v for k, v in schedule.items() if k != "uid"})
        schedule["uid"] = uid
    schedule_registry[uid] = schedule
    return uid


def _normalize_observing_procedures(raw: Mapping[str, Any], *, schedule_registry: Dict[str, Dict[str, Any]], fallback_date: str = "..", time_zone: str = "UTC") -> Optional[List[Dict[str, Any]]]:
    candidates = []
    for key in ("coverage", "sampling", "observingProcedure", "observingProcedures"):
        candidates.extend(_flatten_schedule_candidates(raw.get(key), context=raw))
    if not candidates:
        candidates = _flatten_schedule_candidates(raw, context={})
    records: List[Dict[str, Any]] = []
    for candidate in candidates:
        schedule = _jscalendar_observing_schedule(candidate, time_zone=time_zone)
        if not schedule:
            continue
        uid = _schedule_registry_add(schedule_registry, schedule)
        strategy = _compact_wmdr_code_value(_first_non_empty(candidate.get("strategy"), candidate.get("observingStrategy"), raw.get("strategy"), raw.get("observingStrategy"), "continuous"))
        records.append({"validFrom": _entry_date(candidate, fallback=fallback_date), "strategy": strategy, "observingSchedules": [uid]})
    return _uniq_dicts(_clean_none(records)) or None


def _normalize_reporting_scalar(value: Any) -> Any:
    """Return a compact scalar for reporting fields that may arrive as WMDR10 wrappers.

    Some XML-derived WMDR10 JSON payloads wrap reporting attributes such as
    referenceDatum in nested objects, e.g. {"VerticalDatum": {"remarks":
    "mean sea level"}}.  WMDR2 v0.3.0 keeps these fields as compact scalar
    values in ReportingProcedure, so unwrap the most useful scalar rather than
    leaking source wrapper objects into the output.
    """
    if value in (None, "", [], {}):
        return None
    if isinstance(value, Mapping):
        for key in ("href", "url", "value", "#text", "text", "code", "remarks", "description", "name", "title", "id", "identifier"):
            if _non_empty(value.get(key)) or isinstance(value.get(key), (int, float, bool)):
                return _normalize_reporting_scalar(value.get(key))
        if len(value) == 1:
            return _normalize_reporting_scalar(next(iter(value.values())))
        for nested in value.values():
            scalar = _normalize_reporting_scalar(nested)
            if _non_empty(scalar) or isinstance(scalar, (int, float, bool)):
                return scalar
        return None
    if isinstance(value, str):
        compact = _compact_wmdr_code_value(value)
        return compact if _non_empty(compact) else None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else str(value)
    return str(value)


def _normalize_observation_reporting(raw_value: Any, reporting_registry: Optional[Dict[str, Dict[str, Any]]] = None, *, schedule_registry: Optional[Dict[str, Dict[str, Any]]] = None, fallback_date: str = "..", time_zone: str = "UTC") -> Optional[List[Dict[str, Any]]]:
    """Normalize v0.3.0 inline ReportingProcedure objects.

    ``reporting_registry`` is accepted for backward compatibility but is not
    populated anymore because v0.3.0 no longer emits reusable
    ``properties.reporting`` definitions.
    """
    records: List[Dict[str, Any]] = []
    for item in _as_list(raw_value):
        if not isinstance(item, Mapping):
            continue
        reporting_any: Any = item.get("reporting")
        reporting: Mapping[str, Any] = cast(Mapping[str, Any], reporting_any) if isinstance(reporting_any, Mapping) else {}
        merged: Dict[str, Any] = dict(item)
        merged.update(dict(reporting))
        international = _parse_bool(merged.get("internationalExchange"))
        if international is None and _non_empty(merged.get("internationalExchange")):
            international = bool(merged.get("internationalExchange"))
        schedule = _jscalendar_reporting_schedule(merged, time_zone=time_zone)
        record: Dict[str, Any] = {
            "internationalExchange": international,
            "uom": _compact_wmdr_code_value(merged.get("uom")) if isinstance(merged.get("uom"), str) else merged.get("uom"),
            "temporalReportingInterval": _iso_duration(_first_non_empty(merged.get("temporalReportingInterval"), merged.get("reportingInterval"), merged.get("temporalAggregate"))),
            "spatialReportingInterval": _normalize_reporting_scalar(merged.get("spatialReportingInterval")),
            "timeliness": _iso_duration(merged.get("timeliness")) or _normalize_reporting_scalar(merged.get("timeliness")),
            "dataPolicy": _normalize_reporting_scalar(merged.get("dataPolicy")),
            "levelOfData": _normalize_reporting_scalar(merged.get("levelOfData")),
            "numberOfObservationsInReportingInterval": _normalize_reporting_scalar(merged.get("numberOfObservationsInReportingInterval")),
            "referenceDatum": _normalize_reporting_scalar(merged.get("referenceDatum")),
            "referenceTimeSource": _normalize_reporting_scalar(merged.get("referenceTimeSource")),
            "timeStampMeaning": _normalize_reporting_scalar(merged.get("timeStampMeaning")),
            "strategy": _compact_wmdr_code_value(_first_non_empty(merged.get("strategy"), merged.get("reportingStrategy"), "routine")),
            "links": _extract_links(merged, "observation"),
        }
        if schedule:
            if schedule_registry is not None:
                record["reportingSchedules"] = [_schedule_registry_add(schedule_registry, schedule)]
            else:
                record["reportingSchedules"] = [schedule]
        record = _clean_none(record)
        if record:
            records.append(record)
    return _uniq_dicts(records) or None


def _normalize_observation_series_observing_configurations(raw: Mapping[str, Any], deployment_sources: Sequence[Mapping[str, Any]], *, facility_id: str, fallback_date: str = "..") -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    raw_method_values = _normalize_observing_method_values(raw)
    for dep_index, source in enumerate(deployment_sources, start=1):
        if not isinstance(source, Mapping):
            continue
        method_values = _normalize_observing_method_values(source) or raw_method_values or [_nil_reason("unknown")]
        observing_location = _observing_location_from_source(source)
        source_of_observation = _compact_wmdr_code_value(_first_non_empty(source.get("sourceOfObservation"), source.get("instrumentType"), source.get("source")))
        instrument_ref = _instrument_ref_for_deployment(source, facility_id=facility_id)
        for value in method_values:
            record = {
                "validFrom": _entry_date(source, fallback=fallback_date),
                "observingLocation": observing_location,
                "observingMethod": value,
                "operatingStatus": _first_operating_status_value(source) or _nil_reason("unknown"),
                "sourceOfObservation": source_of_observation or _nil_reason("unknown"),
                "instrument": instrument_ref,
                "serialNumber": _first_non_empty(source.get("serialNumber"), source.get("serial")),
                "exposure": _compact_wmdr_code_value(source.get("exposure")) if isinstance(source.get("exposure"), str) else source.get("exposure"),
            }
            records.append(_clean_none(record))
    if not records:
        method_values = raw_method_values or [_nil_reason("unknown")]
        for value in method_values:
            records.append(_clean_none({
                "validFrom": fallback_date,
                "observingMethod": value,
                "operatingStatus": _nil_reason("unknown"),
                "sourceOfObservation": _nil_reason("unknown"),
            }))
    return _uniq_dicts(_clean_none(records))


def _observed_property_value(raw: Mapping[str, Any]) -> Any:
    value = _first_non_empty(raw.get("observedProperty"), raw.get("observedVariable"), raw.get("observableVariable"))
    return _compact_wmdr_code_value(value) if isinstance(value, (str, dict)) else value


def _observed_feature_value(raw: Mapping[str, Any], observed_property: Any) -> Dict[str, Any]:
    observed_feature_any: Any = raw.get("observedFeature")
    observed_domain_any: Any = raw.get("observedDomain")
    observed_feature: Mapping[str, Any] = cast(Mapping[str, Any], observed_feature_any) if isinstance(observed_feature_any, Mapping) else {}
    observed_domain: Mapping[str, Any] = cast(Mapping[str, Any], observed_domain_any) if isinstance(observed_domain_any, Mapping) else {}
    inferred_domain = _observed_domain_from_observed_variable(_first_non_empty(raw.get("observedVariable"), raw.get("observedProperty")))
    domain_raw = _first_non_empty(observed_feature.get("domain"), observed_domain.get("domain"), raw.get("observedDomain"), inferred_domain)
    if isinstance(domain_raw, Mapping):
        domain_raw = _first_non_empty(domain_raw.get("domain"), domain_raw.get("href"), domain_raw.get("value"), inferred_domain)
    out = {
        "domain": _compact_wmdr_code_value(domain_raw) if isinstance(domain_raw, str) else domain_raw,
        "domainFeature": _first_non_empty(observed_feature.get("domainFeature"), observed_domain.get("domainFeature"), raw.get("domainFeature")),
        "featureName": _first_non_empty(observed_feature.get("featureName"), observed_domain.get("featureName"), raw.get("featureName")),
    }
    return _clean_none(out) or {"domain": "unknown"}



def _observation_data_generation_sources(raw: Mapping[str, Any], deployment_sources: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Collect data-generation/procedure/reporting sources from observation and former deployments."""
    sources: List[Mapping[str, Any]] = []
    for value in (raw.get("dataGeneration"), raw.get("dataGenerations")):
        for item in _as_list(value):
            if isinstance(item, Mapping):
                sources.append(cast(Mapping[str, Any], item))
    for source in deployment_sources:
        for value in (source.get("dataGeneration"), source.get("dataGenerations")):
            for item in _as_list(value):
                if isinstance(item, Mapping):
                    sources.append(cast(Mapping[str, Any], item))
    return sources


def _normalize_observing_procedure_history(raw: Mapping[str, Any], deployment_sources: Sequence[Mapping[str, Any]], *, schedule_registry: Dict[str, Dict[str, Any]], fallback_date: str = "..", time_zone: str = "UTC") -> Optional[List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    data_sources = _observation_data_generation_sources(raw, deployment_sources)
    if data_sources:
        for item in data_sources:
            records.extend(_normalize_observing_procedures(item, schedule_registry=schedule_registry, fallback_date=_entry_date(item, fallback=fallback_date), time_zone=time_zone) or [])
    else:
        records.extend(_normalize_observing_procedures(raw, schedule_registry=schedule_registry, fallback_date=fallback_date, time_zone=time_zone) or [])
    return _uniq_dicts(_clean_none(records)) or None


def _normalize_observation_reporting_history(raw: Mapping[str, Any], deployment_sources: Sequence[Mapping[str, Any]], *, reporting_registry: Optional[Dict[str, Dict[str, Any]]] = None, schedule_registry: Optional[Dict[str, Dict[str, Any]]] = None, fallback_date: str = "..", time_zone: str = "UTC") -> Optional[List[Dict[str, Any]]]:
    values: List[Mapping[str, Any]] = []
    data_sources = _observation_data_generation_sources(raw, deployment_sources)
    if data_sources:
        values.extend(data_sources)
    else:
        for item in _as_list(_first_non_empty(raw.get("reporting"), raw.get("reportingProcedures"), raw.get("coverage"), raw.get("sampling"))):
            if isinstance(item, Mapping):
                values.append(cast(Mapping[str, Any], item))
    if not values:
        return None
    return _normalize_observation_reporting(values, reporting_registry=reporting_registry, schedule_registry=schedule_registry, fallback_date=fallback_date, time_zone=time_zone)


def _normalize_observation_official_status(raw: Mapping[str, Any], deployment_sources: Sequence[Mapping[str, Any]], *, fallback_date: str = "..") -> Optional[List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    if _non_empty(raw.get("officialStatus")) or isinstance(raw.get("officialStatus"), bool):
        records.extend(_normalize_temporal_official_status(raw.get("officialStatus"), fallback_date=fallback_date) or [])
    for source in deployment_sources:
        if _non_empty(source.get("officialStatus")) or isinstance(source.get("officialStatus"), bool):
            records.extend(_normalize_temporal_official_status(source.get("officialStatus"), fallback_date=_entry_date(source, fallback=fallback_date)) or [])
    return _uniq_dicts(_clean_none(records)) or None

def _normalize_observation(raw: Mapping[str, Any], *, index: int, facility_id: str, schedule_registry: Optional[Dict[str, Dict[str, Any]]] = None, reporting_registry: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    schedule_registry = schedule_registry if schedule_registry is not None else {}
    reporting_registry = reporting_registry if reporting_registry is not None else {}
    observed_property = _observed_property_value(raw)
    fallback_date = _entry_date(raw, "..")
    deployment_sources: List[Mapping[str, Any]] = [d for d in _as_list(raw.get("deployments") or raw.get("deployment") or raw.get("observingLocations") or raw.get("observingLocation")) if isinstance(d, Mapping)]
    if not deployment_sources and isinstance(raw.get("observingConfigurations"), list):
        # Accept already-converted legacy features and migrate their configuration shape.
        for oc in raw.get("observingConfigurations", []):
            if isinstance(oc, Mapping):
                observing_location_any: Any = oc.get("observingLocation")
                if isinstance(observing_location_any, Mapping):
                    deployment_sources.append(cast(Mapping[str, Any], observing_location_any))
    geometry = _first_non_empty(raw.get("observedGeometry"), raw.get("observedGeometryType"), raw.get("type"))
    digest_source = _stable_json(_clean_none({"observedProperty": observed_property, "index": index, "date": fallback_date, "deploymentCount": len(deployment_sources)}))
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    out: Dict[str, Any] = {
        "uid": _first_non_empty(raw.get("uid"), raw.get("id"), f"observationSeries:{_sanitize_id(observed_property or index)}-{digest}"),
        "title": _first_non_empty(raw.get("title"), _format_observation_title(_first_non_empty(raw.get("observedVariable"), raw.get("observedProperty")), geometry)),
        "observedProperty": observed_property,
        "observedFeature": _observed_feature_value(raw, observed_property),
        "observedGeometry": _compact_wmdr_code_value(geometry) if isinstance(geometry, str) else geometry,
        "programAffiliations": _program_affiliation_values(raw.get("programAffiliation") or raw.get("programAffiliations")),
        "observingConfigurations": _normalize_observation_series_observing_configurations(raw, deployment_sources, facility_id=facility_id, fallback_date=fallback_date),
        "observingProcedures": _normalize_observing_procedure_history(raw, deployment_sources, schedule_registry=schedule_registry, fallback_date=fallback_date),
        "reportingProcedures": _normalize_observation_reporting_history(raw, deployment_sources, reporting_registry=reporting_registry, schedule_registry=schedule_registry, fallback_date=fallback_date),
        "officialStatus": _normalize_observation_official_status(raw, deployment_sources, fallback_date=fallback_date),
        "keywords": _keywords_from_values(_collect_discovery_values("observation", raw, "keywords")),
        "links": _extract_links(raw, "observation"),
    }
    return _clean_none(out)


def _normalize_observation_series(raw: Mapping[str, Any], *, index: int, facility_id: str, schedule_registry: Optional[Dict[str, Dict[str, Any]]] = None, reporting_registry: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    return _normalize_observation(raw, index=index, facility_id=facility_id, schedule_registry=schedule_registry, reporting_registry=reporting_registry)


def _all_observation_deployment_sources(observations: Sequence[Mapping[str, Any]], deployments: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    out.extend(d for d in deployments if isinstance(d, Mapping))
    for obs in observations:
        for dep in _as_list(obs.get("deployments") or obs.get("deployment") or obs.get("observingLocations") or obs.get("observingLocation")):
            if isinstance(dep, Mapping):
                out.append(dep)
    return out


# ---------------------------------------------------------------------------
# Facility feature construction and legacy feature migration
# ---------------------------------------------------------------------------


def _transform_legacy_feature_to_v030(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Migrate a v0.2.x-like Feature to v0.3.0 shape."""
    result: Dict[str, Any] = copy.deepcopy(dict(record))
    props_any: Any = result.setdefault("properties", {})
    if not isinstance(props_any, dict):
        return dict(record)
    props: Dict[str, Any] = props_any
    conforms = [c for c in _as_list(result.get("conformsTo")) if c != OGC_RECORD_CORE_CONF]
    if WMDR2_CORE_CONF not in conforms:
        conforms.append(WMDR2_CORE_CONF)
    result["conformsTo"] = conforms
    sanitized_contacts = _sanitize_contacts(props.get("contacts"))
    if sanitized_contacts:
        props["contacts"] = sanitized_contacts
    else:
        props.pop("contacts", None)
    if "schedules" in props:
        props["schedules"] = _strip_misspelled_wmi_extensions(props.get("schedules"))
    deployment_by_id: Dict[str, Mapping[str, Any]] = {}
    for dep in _as_list(props.get("deployments")):
        if isinstance(dep, Mapping) and isinstance(dep.get("id"), str):
            deployment_by_id[dep["id"]] = dep
    props.pop("deployments", None)
    props.pop("reporting", None)
    for old_key in ("environment", "programAffiliation", "territory"):
        value = props.get(old_key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "date" in item and "validFrom" not in item and old_key != "officialStatus":
                    item["validFrom"] = item.pop("date")
    for obs in _as_list(props.get("observationSeries")):
        if not isinstance(obs, dict):
            continue
        for proc in _as_list(obs.get("observingProcedures")):
            if isinstance(proc, dict) and "date" in proc and "validFrom" not in proc:
                proc["validFrom"] = proc.pop("date")
        for cfg in _as_list(obs.get("observingConfigurations")):
            if not isinstance(cfg, dict):
                continue
            if "date" in cfg and "validFrom" not in cfg:
                cfg["validFrom"] = cfg.pop("date")
            dep_ref = cfg.pop("deployment", None)
            dep = deployment_by_id.get(dep_ref) if isinstance(dep_ref, str) else None
            if dep and "observingLocation" not in cfg:
                loc = _observing_location_from_source(dep)
                if loc:
                    cfg["observingLocation"] = loc
                src = _first_non_empty(dep.get("sourceOfObservation"), dep.get("instrument"), dep.get("instrumentRef"))
                if src and "sourceOfObservation" not in cfg:
                    cfg["sourceOfObservation"] = src
            if cfg.get("observingMethod") == "unknown":
                cfg["observingMethod"] = _nil_reason("unknown")
            cfg.setdefault("operatingStatus", _nil_reason("unknown"))
            cfg.setdefault("sourceOfObservation", _nil_reason("unknown"))
    return _clean_none(result)


def build_facility_feature(
    facility: Mapping[str, Any],
    observation_series: Sequence[Mapping[str, Any]],
    deployments: Sequence[Mapping[str, Any]],
    header: Mapping[str, Any],
    *,
    source_name: Optional[str] = None,
) -> Dict[str, Any]:
    facility_id_raw = _first_non_empty(facility.get("identifier"), facility.get("wigosStationIdentifier"), source_name, "unknown")
    facility_id_text = str(facility_id_raw).removeprefix("facility:").removeprefix("wsi:")
    facility_id = _sanitize_id(facility_id_text)
    root_id = f"wsi:{facility_id}"
    start, end = _extract_interval(facility)
    fallback_date = _normalize_date_value(start) or ".."
    temporal_entries = _facility_temporal_geometry_entries(facility)
    geometry = _facility_geometry_from_entries(temporal_entries) or {"type": "Point", "coordinates": [0.0, 0.0]}
    temporal_geometry = _temporal_geometry_extension(temporal_entries)

    schedule_registry: Dict[str, Dict[str, Any]] = {}
    reporting_registry: Dict[str, Dict[str, Any]] = {}
    observations: List[Dict[str, Any]] = []
    for i, obs in enumerate(observation_series, start=1):
        if not isinstance(obs, Mapping):
            continue
        obs_payload: Mapping[str, Any] = obs
        if deployments and not any(k in obs for k in ("deployments", "deployment", "observingLocations", "observingLocation")):
            merged_obs = dict(obs)
            merged_obs["deployments"] = list(deployments)
            obs_payload = merged_obs
        observations.append(_normalize_observation_series(obs_payload, index=i, facility_id=facility_id, schedule_registry=schedule_registry, reporting_registry=reporting_registry))
    all_deployment_sources = _all_observation_deployment_sources(observation_series, deployments)
    instruments = _normalize_instruments(all_deployment_sources, facility_id=facility_id)
    contacts, _temporal_contacts = _collect_contacts(
        facility.get("contact"),
        facility.get("contacts"),
        header.get("contact") if isinstance(header, Mapping) else None,
        header.get("contacts") if isinstance(header, Mapping) else None,
        header.get("recordOwner") if isinstance(header, Mapping) else None,
    )

    properties: Dict[str, Any] = {
        "type": "facility",
        "title": _normalize_display_text(_first_non_empty(facility.get("name"), facility.get("title"), facility_id)),
        "description": _normalize_description_value(_first_non_empty(facility.get("description"), facility.get("abstract"))),
        **_record_timestamps(header, source_name=source_name),
        **_copy_known_facility_properties(facility),
        "contacts": contacts,
        "keywords": _keywords_from_values(_collect_discovery_values("facility", facility, "keywords")),
        "links": _extract_links(facility, "facility"),
        "environment": _normalize_historical_environment(facility, fallback_date=fallback_date),
        "programAffiliations": _normalize_historical_program_affiliation(facility.get("programAffiliation") or facility.get("programAffiliations"), fallback_date=fallback_date),
        "territory": _normalize_historical_territory(facility.get("territory") or facility.get("territories"), fallback_date=fallback_date),
        "observationSeries": observations,
        "instruments": instruments,
        "schedules": list(schedule_registry.values()) or None,
        "officialStatus": _normalize_temporal_official_status(facility.get("officialStatus"), fallback_date=fallback_date),
        "dataPolicy": _normalize_temporal_data_policy(facility.get("dataPolicy"), fallback_date=fallback_date),
    }
    if not properties.get("keywords"):
        properties.pop("keywords", None)
    record: Dict[str, Any] = {
        "type": "Feature",
        "id": root_id,
        "geometry": geometry,
        "temporalGeometry": temporal_geometry,
        "time": _time_interval(start or fallback_date, end, resolution="day"),
        "conformsTo": [WMDR2_CORE_CONF],
        "properties": properties,
    }
    return _strip_misspelled_wmi_extensions(_restore_null_sentinel(_finalize_wmdr2_value(_clean_none(record))))


def _payload_parts(payload: Mapping[str, Any]) -> Tuple[Mapping[str, Any], List[Mapping[str, Any]], List[Mapping[str, Any]], Mapping[str, Any]]:
    header_any: Any = payload.get("header")
    facility_any: Any = payload.get("facility")
    header: Mapping[str, Any] = cast(Mapping[str, Any], header_any) if isinstance(header_any, Mapping) else {}
    facility: Mapping[str, Any] = cast(Mapping[str, Any], facility_any) if isinstance(facility_any, Mapping) else {}
    observations_raw: Any = _first_non_empty(payload.get("observationSeries"), payload.get("observations"), [])
    deployments_raw: Any = payload.get("deployments") or []
    observations: List[Mapping[str, Any]] = [cast(Mapping[str, Any], x) for x in _as_list(observations_raw) if isinstance(x, Mapping)]
    deployments: List[Mapping[str, Any]] = [cast(Mapping[str, Any], x) for x in _as_list(deployments_raw) if isinstance(x, Mapping)]
    if not facility and any(k in payload for k in ("identifier", "name", "geospatialLocation")):
        facility = payload
    if isinstance(facility, Mapping) and any(k in payload for k in ("contact", "contacts", "recordOwner")):
        merged_facility: Dict[str, Any] = dict(facility)
        if "contact" not in merged_facility and "contact" in payload:
            merged_facility["contact"] = payload.get("contact")
        if "contacts" not in merged_facility and "contacts" in payload:
            merged_facility["contacts"] = payload.get("contacts")
        if "contacts" not in merged_facility and "recordOwner" in payload:
            merged_facility["contacts"] = payload.get("recordOwner")
        facility = merged_facility
    return facility, observations, deployments, header


def convert_payload(payload: Any, *, source_name: Optional[str] = None) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("WMDR10 JSON payload must be a mapping")
    facility, observations, deployments, header = _payload_parts(payload)
    if not facility:
        facility = {"identifier": source_name or "unknown", "name": source_name or "unknown"}
    return build_facility_feature(facility, observations, deployments, header, source_name=source_name)



def convert_wmdr10_json_to_wmdr2_json(payload: Any, *, source_name: Optional[str] = None) -> Dict[str, Any]:
    """Backward-compatible public alias for convert_payload."""
    return convert_payload(payload, source_name=source_name)

def convert_group(group: Mapping[str, Any], *, source_name: Optional[str] = None) -> Dict[str, Any]:
    return convert_payload(group, source_name=source_name)


def convert_record(payload: Any, *, source_name: Optional[str] = None) -> Dict[str, Any]:
    return convert_payload(payload, source_name=source_name)


def convert_file(source: Path | str, target_dir: Path | str) -> Path:
    source_path = Path(source)
    target_path = Path(target_dir)
    payload = _load_json(source_path)
    converted = convert_payload(payload, source_name=source_path.stem)
    output = target_path / f"{source_path.stem}{OUTPUT_SUFFIX}"
    _write_json(output, converted)
    return output


def _load_code_list_labels(config: Mapping[str, Any], *, base_dir: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    base_dir = base_dir or Path.cwd()
    labels: Dict[str, Dict[str, str]] = {}
    raw = config.get("codeListLabels") if isinstance(config, Mapping) else None
    files = raw.get("files") if isinstance(raw, Mapping) else None
    for file_name in _as_list(files):
        path = Path(str(file_name))
        if not path.is_absolute():
            path = base_dir / path
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                domain = (row.get("domain") or "").strip()
                code = (row.get("code") or "").strip().lstrip("_")
                label = (row.get("label") or row.get("title") or "").strip()
                if domain and code and label:
                    labels.setdefault(domain, {})[code] = label
    return labels



def _contact_identifier(contact: Mapping[str, Any]) -> str:
    """Return stable contact catalogue uid, preserving e-mail addresses."""
    for email in _as_list(contact.get("emails") or contact.get("email")):
        if isinstance(email, str) and "@" in email:
            return f"contact:{email.strip().lower()}"
    raw_name = _normalize_display_text(contact.get("name")) or ""
    raw_org = _normalize_display_text(contact.get("organization")) or ""
    seed = "--".join(part for part in (_slug(raw_name), _slug(raw_org)) if part)
    if not seed:
        seed = _slug(_stable_json(contact))
    return f"contact:{seed}"


def _catalogue_inline_contact(contact: Mapping[str, Any], uid: str) -> Dict[str, Any]:
    inline: Dict[str, Any] = {"uid": uid}
    for key in ("name", "organization", "roles"):
        value = contact.get(key)
        if _non_empty(value):
            inline[key] = value
    inline["links"] = [
        {
            "rel": "about",
            "href": f"../catalogues/contacts.json#{uid}",
            "type": "application/json",
        }
    ]
    return _clean_none(inline)

def _run_catalogue_post_processing(target: Path, catalogues: Mapping[str, Any]) -> None:
    if not catalogues.get("enabled"):
        return
    if "source" in catalogues:
        raise SystemExit("catalogues.source is obsolete; use catalogues.records_path, contacts_path and instruments_path")
    records_path = Path(str(catalogues.get("records_path") or target / "catalogue_representation"))
    contacts_path = Path(str(catalogues.get("contacts_path") or target / "catalogues" / "contacts.json"))
    instruments_path = Path(str(catalogues.get("instruments_path") or target / "catalogues" / "instruments.json"))
    contacts: Dict[str, Dict[str, Any]] = {}
    instruments: Dict[str, Dict[str, Any]] = {}
    for path in sorted(target.glob("*.json")):
        record = _load_json(path)
        props = record.get("properties") if isinstance(record, Mapping) else None
        if not isinstance(props, dict):
            continue
        rewritten = copy.deepcopy(record)
        rewritten_props = rewritten.get("properties", {})
        # Contact catalogue: full details go to contacts.json; the catalogue
        # representation keeps a compact actionable link plus human-readable
        # name/organization/roles.  Embedded full records remain untouched.
        inline_contacts: List[Dict[str, Any]] = []
        for c in _as_list(props.get("contacts")):
            if not isinstance(c, Mapping):
                continue
            entry = _sanitize_contact_roles(c)
            cid = _contact_identifier(entry)
            entry["uid"] = cid
            entry.pop("id", None)
            entry.pop("identifier", None)
            contacts[cid] = entry
            inline_contacts.append(_catalogue_inline_contact(entry, cid))
        if inline_contacts:
            rewritten_props["contacts"] = inline_contacts
        for inst in _as_list(props.get("instruments")):
            if not isinstance(inst, Mapping):
                continue
            inst_uid = inst.get("uid") or inst.get("id")
            if not isinstance(inst_uid, str):
                continue
            entry = dict(inst)
            entry["uid"] = inst_uid
            entry.pop("id", None)
            instruments[inst_uid] = entry
        rewritten_props.pop("instruments", None)
        _write_json(records_path / path.name, rewritten)
    if contacts:
        _write_json(contacts_path, {"contacts": sorted(contacts.values(), key=lambda x: x.get("uid", ""))})
    if instruments:
        _write_json(instruments_path, {"instruments": sorted(instruments.values(), key=lambda x: x.get("uid", ""))})


def _convert_source_to_target(source: Path, target: Path, *, pattern: str = DEFAULT_PATTERN, recursive: bool = True) -> int:
    files = _iter_json_files(source, pattern=pattern, recursive=recursive)
    count = 0
    for path in files:
        if path.is_relative_to(target) if hasattr(path, "is_relative_to") else False:
            continue
        rel = path.relative_to(source) if source.is_dir() else Path(path.name)
        out = target / rel.with_suffix(OUTPUT_SUFFIX)
        payload = _load_json(path)
        converted = convert_payload(payload, source_name=path.stem)
        _write_json(out, converted)
        count += 1
    return count


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert WMDR10 JSON to WMDR2 v0.3.0 JSON")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args(argv)

    config_path = _discover_config_path(args.config)
    cfg: Dict[str, Any] = _load_config(config_path) if config_path else {}
    section = _cfg_section(cfg)
    print(_format_loaded_config_hint(config_path, section))

    global DISCOVERY_POLICY, CODE_LIST_LABELS
    DISCOVERY_POLICY = _normalize_discovery_policy(section)
    if config_path:
        CODE_LIST_LABELS = _load_code_list_labels(section, base_dir=config_path.parent)

    catalogues = section.get("catalogues") if isinstance(section.get("catalogues"), Mapping) else {}
    if isinstance(catalogues, Mapping) and "source" in catalogues:
        raise SystemExit("catalogues.source is obsolete; use catalogues.records_path, contacts_path and instruments_path")

    source = args.source or (Path(str(section["source"])) if section.get("source") else None)
    target = args.target or (Path(str(section["target"])) if section.get("target") else None)
    if source is None or target is None:
        raise SystemExit("source and target are required, either by CLI or config")
    pattern = args.pattern or str(section.get("pattern") or DEFAULT_PATTERN)
    recursive = not args.no_recursive and bool(section.get("recursive", True))
    count = _convert_source_to_target(source, target, pattern=pattern, recursive=recursive)
    _run_catalogue_post_processing(target, catalogues if isinstance(catalogues, Mapping) else {})
    print(f"Converted {count} WMDR10 JSON record(s) to {target}")


if __name__ == "__main__":  # pragma: no cover
    main()
