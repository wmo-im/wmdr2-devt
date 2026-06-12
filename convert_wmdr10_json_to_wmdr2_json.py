#!/usr/bin/env python3
"""
convert_wmdr10_json_to_wmdr2_json.py

Convert simplified WMDR 1.0 JSON records into a facility-centric WMDR2 JSON
Feature.  This replacement implements the June 2026 temporal-history update:

* root temporalGeometry remains the only aligned-array temporal object;
* every other temporal* member is emitted as an array of objects;
* environmental histories are grouped under properties.environment;
* temporalTopographyBathymetry is removed and its members are promoted to
  first-level environment properties;
* facilitySet is replaced by facilitySets references;
* externalIds is no longer emitted;
* observation-level program affiliations are emitted as programAffiliations list[str].
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
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
    "deployment": {"keywords": [], "links": []},
}
DEFAULT_DISCOVERY_POLICY: Dict[str, Dict[str, List[str]]] = {
    "facility": {
        "keywords": ["identifier", "name"],
        "links": ["onlineResource"],
    },
    "observation": {
        "keywords": [],
        "links": [],
    },
    "deployment": {
        "keywords": [
            "manufacturer",
            "model",
            "serialNumber",
            "sourceOfObservation",
            "observingMethod",
        ],
        "links": [],
    },
}
DISCOVERY_POLICY: Dict[str, Dict[str, List[str]]] = copy.deepcopy(DEFAULT_DISCOVERY_POLICY)
CODE_LIST_LABELS: Dict[str, Dict[str, str]] = {}

# ---------------------------------------------------------------------------
# Config / file handling
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit(
            f"Cannot read config file {path}: PyYAML is not installed. "
            "Install pyyaml or pass --source/--target explicitly."
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - defensive CLI path
        raise SystemExit(f"Cannot read config file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Config file {path} must contain a top-level YAML mapping.")
    return data


def _walk_up_for_config(start: Path) -> List[Path]:
    base = start if start.is_dir() else start.parent
    candidates: List[Path] = []
    for folder in (base, *base.parents):
        candidates.append(folder / "config.yaml")
        candidates.append(folder / "config.yml")
    return candidates


def _discover_config_path(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit is not None:
        explicit_path = explicit.expanduser()
        return explicit_path if explicit_path.is_absolute() else (Path.cwd() / explicit_path)

    candidates: List[Path] = []
    candidates.extend(_walk_up_for_config(Path.cwd()))
    candidates.extend(_walk_up_for_config(Path(__file__).resolve().parent))

    seen: set[Path] = set()
    for candidate in candidates:
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


def _cfg_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    section = cfg.get("convert_wmdr10_json_to_wmdr2_json")
    if isinstance(section, dict):
        return section
    legacy = cfg.get("convert_wmdr10_json_to_wmdr2_geojson")
    return legacy if isinstance(legacy, dict) else {}


def _format_loaded_config_hint(config_path: Optional[Path], section: Dict[str, Any]) -> str:
    if config_path is None:
        return "No config file found; using CLI arguments only."
    keys = sorted(section.keys()) if section else []
    key_text = ", ".join(keys) if keys else "no converter section keys"
    return f"Using config: {config_path} ({key_text})"


def _normalize_discovery_policy(section: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """Return keyword/link extraction policy from the converter config section.

    If no ``discovery`` block is configured, keep the built-in defaults for a
    convenient command-line conversion.  As soon as a ``discovery`` block is
    present, treat it as authoritative: omitted buckets and empty lists disable
    extraction for that entity/bucket.  This makes a config such as
    ``discovery: {facility: {keywords: []}}`` actually suppress facility
    keywords instead of silently falling back to defaults.
    """
    raw = section.get("discovery")
    if not isinstance(raw, dict):
        return copy.deepcopy(DEFAULT_DISCOVERY_POLICY)

    policy = copy.deepcopy(EMPTY_DISCOVERY_POLICY)
    for entity in ("facility", "observation", "deployment"):
        entity_cfg = raw.get(entity)
        if not isinstance(entity_cfg, dict):
            continue
        for bucket in ("keywords", "links"):
            values = entity_cfg.get(bucket)
            if isinstance(values, list):
                policy[entity][bucket] = [
                    str(v).strip() for v in values if isinstance(v, str) and str(v).strip()
                ]
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    if isinstance(payload, dict):
        if any(k in payload for k in ("facility", "observations", "deployments", "header")):
            return "full"
        if any(k in payload for k in ("observedVariable", "observedProperty", "resultTime")):
            return "observations"
        if any(k in payload for k in ("sourceOfObservation", "manufacturer", "serialNumber")):
            return "deployments"
        if any(k in payload for k in ("fileDateTime", "recordOwner")):
            return "header"
        if any(k in payload for k in ("identifier", "name", "geospatialLocation")):
            return "facility"
    if isinstance(payload, list):
        first = next((x for x in payload if isinstance(x, dict)), None)
        if not first:
            return "unknown"
        if any(k in first for k in ("observedVariable", "observedProperty", "resultTime")):
            return "observations"
        if any(k in first for k in ("sourceOfObservation", "manufacturer", "serialNumber")):
            return "deployments"
    return "unknown"


def _part_group_key(path: Path) -> str:
    stem = path.stem
    for suffix in ("_header", "_facility", "_observations", "_deployments"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


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
        if _non_empty(value):
            return value
    return None


def _clean_none(obj: Any, *, _path: Tuple[str, ...] = ()) -> Any:
    """Remove empty object members, but preserve nulls inside arrays.

    ``temporalGeometry.methods`` is an aligned array whose items are lists of
    geopositioning-method terms. Empty inner lists are meaningful there: they
    mean that no method is declared for the corresponding coordinate/date.
    """

    def preserve_empty_list(path: Tuple[str, ...]) -> bool:
        return len(path) >= 2 and path[-2:] == ("temporalGeometry", "methods")

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
        return {key: _preserve_nulls(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_preserve_nulls(value) for value in obj]
    return obj


def _restore_null_sentinel(obj: Any) -> Any:
    if obj == _NULL_SENTINEL:
        return None
    if isinstance(obj, dict):
        return {key: _restore_null_sentinel(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_restore_null_sentinel(value) for value in obj]
    return obj


def _sanitize_id(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^A-Za-z0-9._:/#-]+", "-", text)
    return text.strip("-") or "record"


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "value"


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
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith(("http://codes.wmo.int/wmdr/", "https://codes.wmo.int/wmdr/")):
        return _normalize_code_value(text)
    return value


def _finalize_wmdr2_value(value: Any, *, key: Optional[str] = None) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"interval"}:
            interval = value.get("interval")
            if interval == ["..", ".."] or interval == "unknown":
                return None
        return {
            child_key: (
                child_value
                if child_key in {"href", "url"}
                else _finalize_wmdr2_value(child_value, key=child_key)
            )
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
        return [_drop_source_metadata(item) for item in obj]
    return obj


def _normalize_display_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    text = value.strip()
    if not text:
        return None
    if _is_unknown_token(text):
        return "unknown"
    return re.sub(r"\s+", " ", text)


def _normalize_description_value(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = _drop_source_metadata(value)
        text = _first_non_empty(
            cleaned.get("description"),
            cleaned.get("value"),
            cleaned.get("#text"),
            cleaned.get("text"),
        )
        if text:
            return _normalize_display_text(text)
        return _clean_none(cleaned)
    return _normalize_display_text(value)


def _uniq_dicts(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        payload = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        if payload in seen:
            continue
        seen.add(payload)
        out.append(item)
    return out


def _uniq_scalars(items: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen: set[str] = set()
    for item in items:
        if item in (None, "", [], {}):
            continue
        key = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str) if isinstance(item, (dict, list)) else str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


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
    match = re.match(r"^(\d{4})(\d{2})(\d{2})(?:_|$)", text)
    if match:
        y, m, d = match.groups()
        return f"{y}-{m}-{d}T00:00:00Z"
    if re.fullmatch(r"\d{8}", text):
        return f"{text[0:4]}-{text[4:6]}-{text[6:8]}T00:00:00Z"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return f"{text}T00:00:00Z"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}Z", text):
        return f"{text[:-1]}T00:00:00Z"
    return text


def _record_timestamps(header: Dict[str, Any], *, source_name: Optional[str] = None) -> Dict[str, str]:
    created = _normalize_record_datetime(
        _first_non_empty(
            header.get("created"),
            header.get("dateCreated"),
            header.get("creationDate"),
            header.get("fileDateTime"),
            header.get("dateStamp"),
            source_name,
        )
    )
    updated = _normalize_record_datetime(
        _first_non_empty(
            header.get("updated"),
            header.get("dateUpdated"),
            header.get("updateDate"),
            header.get("modified"),
            header.get("fileDateTime"),
            header.get("dateStamp"),
            created,
        )
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


def _extract_interval(obj: Dict[str, Any]) -> Tuple[Any, Any]:
    time_obj = obj.get("time")
    if isinstance(time_obj, dict):
        interval = time_obj.get("interval")
        if isinstance(interval, list) and interval:
            start = interval[0] if len(interval) > 0 else None
            end = interval[1] if len(interval) > 1 else None
            return start, end
        return time_obj.get("date") or time_obj.get("timestamp"), None
    return (
        _first_non_empty(obj.get("beginPosition"), obj.get("begin"), obj.get("start"), obj.get("dateEstablished")),
        _first_non_empty(obj.get("endPosition"), obj.get("end"), obj.get("stop"), obj.get("dateClosed")),
    )


def _temporal_begin_date(item: Any) -> str:
    if isinstance(item, dict):
        return (
            _normalize_date_value(
                _first_non_empty(
                    item.get("beginPosition"), item.get("begin"), item.get("start"), item.get("date")
                )
            )
            or ".."
        )
    return ".."


def _parse_pos_lon_lat_z(raw: Any) -> Optional[List[Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        coords_value = raw.get("coordinates")
        if isinstance(coords_value, list) and len(coords_value) >= 2:
            return coords_value
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
            continue
    if len(nums) < 2:
        return None
    lat, lon = nums[0], nums[1]
    coords: List[Any] = [lon, lat]
    if len(nums) >= 3:
        z = nums[2]
        coords.append(int(round(z)) if abs(z - round(z)) < 1e-9 else z)
    return coords


def _geopositioning_methods(item: Any) -> List[str]:
    """Return compact WMDR geopositioning method terms for a location item."""
    if not isinstance(item, dict):
        return []
    raw = item.get("geopositioningMethod")
    if raw in (None, "", [], {}):
        return []

    values = raw if isinstance(raw, list) else [raw]
    methods: List[str] = []
    for value in values:
        if isinstance(value, dict):
            value = _first_non_empty(value.get("href"), value.get("value"), value.get("#text"), value.get("text"))
        if not isinstance(value, str):
            continue
        compact = _compact_wmdr_code_value(value)
        if isinstance(compact, str) and compact.strip():
            methods.append(compact.strip())
    return sorted(dict.fromkeys(methods))


def _facility_temporal_geometry_entries(facility: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        entries.append((date, json.dumps(coords, sort_keys=True), entry))

    for item in _as_list(facility.get("geospatialLocation") or facility.get("geometry")):
        add(item)
    for item in _as_list(facility.get("geospatialLocationHistory") or facility.get("geometryHistory")):
        add(item)

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for _, _, entry in sorted(entries, key=lambda row: (row[0] == "..", row[0], row[1])):
        marker = json.dumps(entry, sort_keys=True)
        if marker in seen:
            continue
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
        raw_date: object = entry.get("date")
        if isinstance(raw_date, str):
            dates.append(raw_date)
        else:
            dates.append("..")
        raw_methods = entry.get("methods")
        entry_methods = raw_methods if isinstance(raw_methods, list) else []
        method_terms = [method for method in entry_methods if isinstance(method, str) and method.strip()]
        if method_terms:
            has_methods = True
        methods.append(method_terms)
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
        coordinates = entry.get("coordinates")
        if isinstance(coordinates, list) and len(coordinates) >= 2:
            return {"type": "Point", "coordinates": coordinates}
    return None


def _extract_code_list_ref(value: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not isinstance(value, str):
        return None, None, None
    text = value.strip().strip("<>")
    if not text:
        return None, None, None
    if text.startswith(("http://", "https://")):
        parts = [part for part in text.rstrip("/#").split("/") if part]
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
# Discovery, contact and link helpers
# ---------------------------------------------------------------------------


def _keywords_from_values(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        if raw is None:
            continue
        if isinstance(raw, str):
            candidate = _normalize_code_value(_last_segment(raw) or raw)
            if not isinstance(candidate, str):
                continue
            candidate = candidate.replace("_", " ").strip()
            if not candidate or _is_unknown_token(candidate):
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
    return out


def _collect_discovery_values(entity_type: str, source: Dict[str, Any], bucket: str) -> List[Any]:
    values: List[Any] = []
    for key in DISCOVERY_POLICY.get(entity_type, {}).get(bucket, []):
        for item in _as_list(source.get(key)):
            if isinstance(item, dict):
                values.extend(_extract_scalar_values(item))
            else:
                values.append(item)
    return values


def _about_link(href: str, *, title: Optional[str] = None, media_type: str = "text/html") -> Dict[str, Any]:
    link: Dict[str, Any] = {"href": href, "rel": "about", "type": media_type}
    if title:
        link["title"] = title
    return link


def _extract_links(source: Dict[str, Any], entity_type: str) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for key in DISCOVERY_POLICY.get(entity_type, {}).get("links", []):
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
                raw_title = item.get("title")
                if isinstance(raw_title, str) and raw_title.strip():
                    title = raw_title.strip()
                raw_media_type = item.get("type")
                if isinstance(raw_media_type, str) and raw_media_type.strip():
                    media_type = raw_media_type.strip()
            if href and href.startswith(("http://", "https://")):
                links.append(_about_link(href, title=title, media_type=media_type))
    return _uniq_dicts(links)


def _normalize_role(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        for key in ("role", "codeListValue", "value", "href", "url", "#text"):
            role = _normalize_role(value.get(key))
            if role:
                return role
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().strip("<>")
    segment = _last_segment(text) or text
    if segment in {"CI_RoleCode", "RoleCode"}:
        return None
    if text.endswith(("#CI_RoleCode", "/CI_RoleCode", "#RoleCode", "/RoleCode")):
        return None
    if _is_unknown_token(segment):
        return None
    return segment


def _normalize_roles(value: Any) -> List[str]:
    roles: List[str] = []
    for item in _as_list(value):
        role = _normalize_role(item)
        if role and role not in roles:
            roles.append(role)
    return roles


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
        ("individualName", "name"),
        ("name", "name"),
        ("title", "name"),
        ("positionName", "position"),
        ("position", "position"),
        ("organisationName", "organization"),
        ("organizationName", "organization"),
        ("organization", "organization"),
    ):
        if dst_key in contact:
            continue
        value = payload.get(src_key)
        if isinstance(value, str) and value.strip():
            contact[dst_key] = value.strip()
    info = _as_dict(payload.get("contactInfo"))
    phone_obj = _as_dict(info.get("phone"))
    phones = [
        {"value": voice.strip()}
        for voice in _as_list(phone_obj.get("voice"))
        if isinstance(voice, str) and voice.strip()
    ]
    if phones:
        contact["phones"] = phones
    address_obj = _as_dict(info.get("address"))
    emails = [
        {"value": email.strip()}
        for email in _as_list(address_obj.get("electronicMailAddress"))
        if isinstance(email, str) and "@" in email
    ]
    if emails:
        contact["emails"] = emails
    address: Dict[str, Any] = {}
    delivery_points = [dp for dp in _as_list(address_obj.get("deliveryPoint")) if isinstance(dp, str) and dp.strip()]
    if delivery_points:
        address["deliveryPoint"] = delivery_points
    for src_key in ("city", "administrativeArea", "postalCode", "country"):
        value = address_obj.get(src_key)
        if isinstance(value, str) and value.strip():
            address[src_key] = value.strip()
    if address:
        contact["addresses"] = [address]
    online = _as_dict(info.get("onlineResource"))
    href = online.get("url") or online.get("href")
    if isinstance(href, str) and href.strip():
        contact["links"] = [_about_link(href.strip())]
    roles = _normalize_roles(_first_non_empty(payload.get("role"), raw.get("role")))
    if roles:
        contact["roles"] = roles
    if not any(key in contact for key in ("name", "organization", "emails", "phones", "addresses", "links")):
        return None, None
    extension = dict(contact)
    valid_start, valid_end = _extract_interval(raw)
    valid_time = _time_interval(valid_start, valid_end)
    if valid_time:
        extension["validTime"] = valid_time
    return _clean_none(contact), _clean_none(extension)


def _collect_contacts(*groups: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ogc_contacts: List[Dict[str, Any]] = []
    temporal_contacts: List[Dict[str, Any]] = []
    for group in groups:
        for item in _as_list(group):
            ogc_contact, temporal_contact = _normalize_contact(item)
            if ogc_contact:
                ogc_contacts.append(ogc_contact)
            if temporal_contact:
                temporal_contacts.append(temporal_contact)
    return _uniq_dicts(ogc_contacts), _uniq_dicts(temporal_contacts)


# ---------------------------------------------------------------------------
# WMDR-specific temporal histories
# ---------------------------------------------------------------------------


def _temporal_object_from_item(
    item: Any,
    *,
    output_key: str,
    value_keys: Sequence[str],
    allow_structured_fallback: bool = False,
) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        value = _normalize_code_value(item)
        if value in (None, "") or _is_unknown_token(value):
            return None
        return {output_key: value, "date": ".."}
    if not isinstance(item, dict):
        if _non_empty(item):
            return {output_key: item, "date": ".."}
        return None
    cleaned = _drop_source_metadata(item)
    if not isinstance(cleaned, dict):
        return None
    raw_value = _first_non_empty(*(cleaned.get(key) for key in value_keys))
    if raw_value is None and allow_structured_fallback:
        raw_value = {
            key: val
            for key, val in cleaned.items()
            if key not in {"beginPosition", "endPosition", "begin", "end", "start", "date"}
        }
    value = _normalize_code_value(raw_value) if isinstance(raw_value, str) else raw_value
    if value in (None, "", [], {}) or _is_unknown_token(value):
        return None
    return {output_key: value, "date": _temporal_begin_date(item)}


def _normalize_temporal_values(
    value: Any,
    *,
    output_key: str,
    value_keys: Sequence[str],
    allow_structured_fallback: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        record = _temporal_object_from_item(
            item,
            output_key=output_key,
            value_keys=value_keys,
            allow_structured_fallback=allow_structured_fallback,
        )
        if record:
            out.append(record)
    out = _uniq_dicts(_clean_none(out))
    return out or None


def _normalize_temporal_territory(value: Any) -> Optional[List[Dict[str, Any]]]:
    return _normalize_temporal_values(
        value,
        output_key="territory",
        value_keys=("territoryName", "territory", "value", "href"),
    )


def _normalize_temporal_climate_zone(value: Any) -> Optional[List[Dict[str, Any]]]:
    return _normalize_temporal_values(value, output_key="climateZone", value_keys=("climateZone", "value", "href"))


def _normalize_temporal_surface_cover(value: Any) -> Optional[List[Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        record = _temporal_object_from_item(item, output_key="surfaceCover", value_keys=("surfaceCover", "value", "href"))
        if not record:
            continue
        if isinstance(item, dict):
            for extra_key in ("surfaceCoverClassification", "surfaceClassification"):
                extra = _first_non_empty(_as_dict(item.get(extra_key)).get("href"), item.get(extra_key))
                if _non_empty(extra):
                    record[extra_key] = _normalize_code_value(extra)
        out.append(record)
    out = _uniq_dicts(_clean_none(out))
    return out or None


def _parse_population_density_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        nums: List[float] = []
        for item in value:
            try:
                nums.append(float(item))
            except Exception:
                return value
        if len(nums) == 1:
            return nums[0]
        return nums
    if isinstance(value, str):
        parts = [part for part in re.split(r"[,\s]+", value.strip()) if part]
        nums: List[float] = []
        for part in parts:
            try:
                nums.append(float(part))
            except Exception:
                return value
        if len(nums) == 1:
            return nums[0]
        if nums:
            return nums
    return value


def _normalize_temporal_population_densities(value: Any) -> Optional[List[Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        record = _temporal_object_from_item(
            item,
            output_key="populationDensity",
            value_keys=("populationDensity", "density", "value"),
            allow_structured_fallback=False,
        )
        if record:
            record["populationDensity"] = _parse_population_density_value(record["populationDensity"])
            out.append(record)
    out = _uniq_dicts(_clean_none(out))
    return out or None


def _normalize_temporal_population(value: Any) -> Optional[List[Dict[str, Any]]]:
    return _normalize_temporal_values(
        value,
        output_key="population",
        value_keys=("population", "value", "href"),
        allow_structured_fallback=True,
    )


def _normalize_temporal_surface_roughness(value: Any) -> Optional[List[Dict[str, Any]]]:
    return _normalize_temporal_values(
        value,
        output_key="surfaceRoughness",
        value_keys=("surfaceRoughness", "roughness", "value", "href"),
    )


def _topography_items(facility: Dict[str, Any], target_key: str) -> List[Any]:
    out: List[Any] = []
    out.extend(_as_list(facility.get(target_key)))
    for container_key in ("topographyBathymetry", "topography", "bathymetry"):
        for container in _as_list(facility.get(container_key)):
            if isinstance(container, dict) and _non_empty(container.get(target_key)):
                source = dict(container)
                value = source.get(target_key)
                if isinstance(value, dict):
                    merged = dict(value)
                    for temporal_key in ("beginPosition", "endPosition", "begin", "end", "start", "date"):
                        if temporal_key in source and temporal_key not in merged:
                            merged[temporal_key] = source[temporal_key]
                    out.append(merged)
                else:
                    out.append({target_key: value, "beginPosition": source.get("beginPosition"), "date": source.get("date")})
    return out


def _normalize_environment(facility: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    environment: Dict[str, Any] = {}

    temporal_climate_zone = _normalize_temporal_climate_zone(facility.get("climateZone"))
    if temporal_climate_zone:
        environment["temporalClimateZone"] = temporal_climate_zone

    temporal_surface_cover = _normalize_temporal_surface_cover(facility.get("surfaceCover"))
    if temporal_surface_cover:
        environment["temporalSurfaceCover"] = temporal_surface_cover

    temporal_population = _normalize_temporal_population(facility.get("population"))
    if temporal_population:
        environment["temporalPopulation"] = temporal_population

    temporal_population_densities = _normalize_temporal_population_densities(
        _first_non_empty(facility.get("populationDensity"), facility.get("populationDensities"), facility.get("demography"))
    )
    if temporal_population_densities:
        environment["temporalPopulationDensities"] = temporal_population_densities

    temporal_surface_roughness = _normalize_temporal_surface_roughness(
        _first_non_empty(facility.get("surfaceRoughness"), facility.get("roughness"))
    )
    if temporal_surface_roughness:
        environment["temporalSurfaceRoughness"] = temporal_surface_roughness

    promoted = {
        "localTopography": "temporalLocalTopography",
        "relativeElevation": "temporalRelativeElevation",
        "topographicContext": "temporalTopographicContext",
        "altitudeOrDepth": "temporalAltitudeOrDepth",
    }
    for source_key, output_key in promoted.items():
        values = _topography_items(facility, source_key)
        temporal_values = _normalize_temporal_values(values, output_key=source_key, value_keys=(source_key, "value", "href"))
        if temporal_values:
            environment[output_key] = temporal_values

    return _clean_none(environment) or None


def _program_affiliation_values(item: Any) -> List[str]:
    if isinstance(item, str):
        value = _normalize_code_value(item)
        return [value] if isinstance(value, str) and value and not _is_unknown_token(value) else []
    if not isinstance(item, dict):
        return []
    values = [
        _normalize_code_value(raw)
        for raw in _as_list(item.get("programAffiliation") or item.get("href") or item.get("value"))
        if isinstance(raw, str) and raw.strip() and not _is_unknown_token(raw)
    ]
    return [value for value in values if isinstance(value, str) and value]


def _normalize_program_affiliations(value: Any) -> Optional[List[str]]:
    """Normalize non-temporal observation-level program affiliations.

    Facility-level program affiliation remains temporal because it can carry
    reporting-status and program-specific facility metadata. Observation-level
    affiliation is a compact set of code values only.
    """
    out: List[str] = []
    seen: set[str] = set()
    for item in _as_list(value):
        for affiliation in _program_affiliation_values(item):
            if affiliation in seen:
                continue
            seen.add(affiliation)
            out.append(affiliation)
    return out or None


def _reporting_status_events(value: Any, *, fallback_date: str = "..") -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for item in _as_list(value):
        if isinstance(item, str):
            status = _normalize_code_value(item)
            if isinstance(status, str) and status and not _is_unknown_token(status):
                rows.append((fallback_date, status))
            continue
        if not isinstance(item, dict):
            continue
        status = _normalize_code_value(
            _first_non_empty(item.get("reportingStatus"), item.get("instrumentOperatingStatus"), item.get("value"), item.get("href"))
        )
        if isinstance(status, str) and status and not _is_unknown_token(status):
            rows.append((_temporal_begin_date(item) or fallback_date, status))
    return rows


def _normalize_temporal_program_affiliation(value: Any) -> Optional[List[Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        affiliations = _program_affiliation_values(item)
        if not affiliations:
            continue
        item_begin = _temporal_begin_date(item) if isinstance(item, dict) else ".."
        status_events = _reporting_status_events(item.get("reportingStatus"), fallback_date=item_begin) if isinstance(item, dict) else []
        psf_id = None
        psf_title = None
        if isinstance(item, dict):
            psf_id = _first_non_empty(item.get("programSpecificFacilityId"), item.get("programSpecificIdentifier"))
            psf_title = _first_non_empty(
                item.get("programSpecificFacilityTitle"),
                item.get("programSpecificFacilityName"),
                item.get("programSpecificTitle"),
            )
        for affiliation in affiliations:
            if status_events:
                for begin, status in status_events:
                    record: Dict[str, Any] = {"programAffiliation": affiliation, "reportingStatus": status, "date": begin}
                    if psf_id:
                        record["programSpecificFacilityId"] = str(psf_id)
                    if psf_title:
                        record["programSpecificFacilityTitle"] = str(psf_title)
                    out.append(record)
            else:
                record = {"programAffiliation": affiliation, "date": item_begin}
                if psf_id:
                    record["programSpecificFacilityId"] = str(psf_id)
                if psf_title:
                    record["programSpecificFacilityTitle"] = str(psf_title)
                out.append(record)
    out = _uniq_dicts(_clean_none(sorted(out, key=lambda row: (row.get("date") == "..", str(row.get("date")), str(row.get("programAffiliation"))))))
    return out or None


def _normalize_reporting_status_timeline(value: Any) -> Optional[List[Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, str):
            status = _normalize_code_value(item)
            if isinstance(status, str) and status and not _is_unknown_token(status):
                out.append({"reportingStatus": status, "date": ".."})
            continue
        if not isinstance(item, dict):
            continue
        status = _normalize_code_value(
            _first_non_empty(item.get("reportingStatus"), item.get("instrumentOperatingStatus"), item.get("value"), item.get("href"))
        )
        if isinstance(status, str) and status and not _is_unknown_token(status):
            out.append({"reportingStatus": status, "date": _temporal_begin_date(item)})
    out = _uniq_dicts(_clean_none(out))
    return out or None


def _normalize_temporal_instrument_operating_status(value: Any) -> Optional[List[Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, str):
            status = _normalize_code_value(item)
            if isinstance(status, str) and status and not _is_unknown_token(status):
                out.append({"instrumentOperatingStatus": status, "date": ".."})
            continue
        if not isinstance(item, dict):
            continue
        status = _normalize_code_value(
            _first_non_empty(item.get("instrumentOperatingStatus"), item.get("status"), item.get("value"), item.get("href"))
        )
        if isinstance(status, str) and status and not _is_unknown_token(status):
            out.append({"instrumentOperatingStatus": status, "date": _temporal_begin_date(item)})
    out = _uniq_dicts(_clean_none(out))
    return out or None


def _normalize_temporal_data_policy(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, str):
            policy = _normalize_code_value(item)
            if isinstance(policy, str) and not _is_unknown_token(policy):
                out.append({"dataPolicy": policy, "date": ".."})
            continue
        if not isinstance(item, dict):
            continue
        record = _drop_source_metadata(item)
        if not isinstance(record, dict):
            continue
        policy = _first_non_empty(record.get("dataPolicy"), record.get("policy"), record.get("value"), record.get("href"))
        if isinstance(policy, str):
            record["dataPolicy"] = _normalize_code_value(policy)
        for key in ("policy", "value", "href", "beginPosition", "endPosition", "begin", "end", "start"):
            if key != "dataPolicy":
                record.pop(key, None)
        record["date"] = _temporal_begin_date(item)
        if record:
            out.append(record)
    return _uniq_dicts(_clean_none(out))


# ---------------------------------------------------------------------------
# Facility sets
# ---------------------------------------------------------------------------


def _facility_set_title(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        title = _first_non_empty(value.get("title"), value.get("name"), value.get("label"), value.get("facilitySet"), value.get("value"), value.get("href"))
        return str(_normalize_code_value(title)) if _non_empty(title) else None
    if _non_empty(value):
        normalized = _normalize_code_value(value) if isinstance(value, str) else value
        return str(normalized) if _non_empty(normalized) else None
    return None


def _facility_set_id(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        raw = _first_non_empty(value.get("id"), value.get("identifier"), value.get("href"), value.get("facilitySet"), value.get("value"), value.get("title"), value.get("name"))
    else:
        raw = value
    if not _non_empty(raw):
        return None
    compact = _normalize_code_value(raw) if isinstance(raw, str) else raw
    text = str(compact).strip()
    if not text:
        return None
    if text.startswith("facilitySet:"):
        return text
    return f"facilitySet:{_sanitize_id(text)}"


def _facility_set_refs(value: Any) -> Optional[List[str]]:
    refs: List[str] = []
    for item in _as_list(value):
        ref = _facility_set_id(item)
        if ref:
            refs.append(ref)
    refs = _uniq_scalars(refs)
    return refs or None


def facility_set_catalog_entry(value: Any, *, description: str = "") -> Optional[Dict[str, Any]]:
    """Return a facility-set catalogue object for a source facilitySet value."""
    ref = _facility_set_id(value)
    title = _facility_set_title(value)
    if not ref or not title:
        return None
    return _clean_none({"id": ref, "title": title, "description": description})


def facility_set_catalog(values: Iterable[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Build a catalogue payload matching schemas/wmdr2-facility-sets.schema.json."""
    entries = [entry for value in values for entry in [facility_set_catalog_entry(value)] if entry]
    return {"facilitySets": _uniq_dicts(entries)}


# ---------------------------------------------------------------------------
# Instruments, deployments, schedules and observations
# ---------------------------------------------------------------------------


def _deployment_source_identifier(raw: Dict[str, Any], *, index: int, facility_id: str) -> str:
    dep_id = _first_non_empty(
        raw.get("identifier"),
        raw.get("@gml:id"),
        raw.get("@id"),
        raw.get("id"),
        raw.get("serialNumber"),
        f"{facility_id}:deployment:{index}",
    )
    return _sanitize_id(str(dep_id))


def _deployment_record_id(raw: Dict[str, Any], *, index: int, facility_id: str) -> str:
    return f"deployment:{_deployment_source_identifier(raw, index=index, facility_id=facility_id)}"


def _is_substantive_instrument_value(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, str) and _is_unknown_token(value):
        return False
    return True


def _instrument_source_values(raw: Dict[str, Any]) -> Tuple[Any, Any]:
    return raw.get("manufacturer"), raw.get("model")


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _normalize_vertical_range(raw: Dict[str, Any]) -> Optional[Dict[str, float]]:
    vertical_range = raw.get("verticalRange")
    if isinstance(vertical_range, dict):
        raw_min = _first_non_empty(
            vertical_range.get("min"),
            vertical_range.get("minimum"),
            vertical_range.get("lower"),
            vertical_range.get("lowerLimit"),
        )
        raw_max = _first_non_empty(
            vertical_range.get("max"),
            vertical_range.get("maximum"),
            vertical_range.get("upper"),
            vertical_range.get("upperLimit"),
        )
    elif isinstance(vertical_range, (list, tuple)) and len(vertical_range) >= 2:
        raw_min, raw_max = vertical_range[0], vertical_range[1]
    else:
        raw_min = _first_non_empty(
            raw.get("verticalRangeMin"),
            raw.get("verticalRangeMinimum"),
            raw.get("minimumVerticalRange"),
            raw.get("lowerVerticalRange"),
            raw.get("verticalRangeLowerLimit"),
        )
        raw_max = _first_non_empty(
            raw.get("verticalRangeMax"),
            raw.get("verticalRangeMaximum"),
            raw.get("maximumVerticalRange"),
            raw.get("upperVerticalRange"),
            raw.get("verticalRangeUpperLimit"),
        )
    min_value = _to_float(raw_min)
    max_value = _to_float(raw_max)
    if min_value is None or max_value is None:
        return None
    return {"min": min_value, "max": max_value}


def _normalize_observable_variables(raw: Dict[str, Any]) -> Optional[List[Any]]:
    raw_value = _first_non_empty(
        raw.get("observableVariables"),
        raw.get("observableVariable"),
        raw.get("instrumentObservableVariables"),
        raw.get("instrumentObservableVariable"),
    )
    values: List[Any] = []
    for item in _as_list(raw_value):
        candidate: Any
        if isinstance(item, dict):
            candidate = _first_non_empty(
                item.get("observableVariable"),
                item.get("observedVariable"),
                item.get("observedProperty"),
                item.get("variable"),
                item.get("description"),
                item.get("value"),
                item.get("href"),
                item.get("#text"),
            )
        else:
            candidate = item
        if not _is_substantive_instrument_value(candidate):
            continue
        normalized = _compact_wmdr_code_value(candidate) if isinstance(candidate, str) else candidate
        if _is_substantive_instrument_value(normalized):
            values.append(normalized)
    return _uniq_scalars(values) or None


def _normalize_observable_geometry(raw: Dict[str, Any]) -> Optional[str]:
    raw_value = _first_non_empty(
        raw.get("observableGeometry"),
        raw.get("instrumentObservableGeometry"),
        raw.get("observableGeometryType"),
        raw.get("instrumentObservableGeometryType"),
    )
    if isinstance(raw_value, dict):
        raw_value = _first_non_empty(
            raw_value.get("observableGeometry"),
            raw_value.get("geometry"),
            raw_value.get("geometryType"),
            raw_value.get("value"),
            raw_value.get("href"),
            raw_value.get("#text"),
        )
    if not _is_substantive_instrument_value(raw_value):
        return None
    normalized = _compact_wmdr_code_value(raw_value) if isinstance(raw_value, str) else raw_value
    if not _is_substantive_instrument_value(normalized):
        return None
    return str(normalized)


def _deployment_has_instrument(raw: Dict[str, Any]) -> bool:
    manufacturer, model = _instrument_source_values(raw)
    return any(_is_substantive_instrument_value(value) for value in (manufacturer, model)) or (
        _normalize_vertical_range(raw) is not None
        or _normalize_observable_variables(raw) is not None
        or _normalize_observable_geometry(raw) is not None
    )


def _instrument_record_id(raw: Dict[str, Any], *, facility_id: str) -> Optional[str]:
    if not _deployment_has_instrument(raw):
        return None
    manufacturer, model = _instrument_source_values(raw)
    vertical_range = _normalize_vertical_range(raw)
    observable_variables = _normalize_observable_variables(raw)
    observable_geometry = _normalize_observable_geometry(raw)
    seed_parts = [
        facility_id,
        str(manufacturer or ""),
        str(model or ""),
        json.dumps(vertical_range, sort_keys=True),
        json.dumps(observable_variables, sort_keys=True),
        str(observable_geometry or ""),
    ]
    digest = hashlib.sha1("|".join(seed_parts).encode("utf-8")).hexdigest()[:12]
    return f"instrument:{digest}"


def _instrument_refs_for_deployment(raw: Dict[str, Any], *, facility_id: str) -> List[str]:
    instrument_id = _instrument_record_id(raw, facility_id=facility_id)
    return [instrument_id] if instrument_id else []


def _deployment_serial_numbers(raw: Dict[str, Any]) -> Optional[Dict[str, List[Any]]]:
    serial_number = raw.get("serialNumber")
    if not _is_substantive_instrument_value(serial_number):
        return None
    start, _ = _extract_interval(raw)
    begin = _normalize_date_value(start) or ".."
    return {"serialNumber": [serial_number], "dates": [begin]}


def _normalize_instrument(raw: Dict[str, Any], *, facility_id: str) -> Optional[Dict[str, Any]]:
    instrument_id = _instrument_record_id(raw, facility_id=facility_id)
    if not instrument_id:
        return None
    manufacturer, model = _instrument_source_values(raw)
    title = _normalize_description_value(
        _first_non_empty(raw.get("instrumentTitle"), raw.get("equipmentTitle"), raw.get("title"), raw.get("name"))
    )
    description = _normalize_description_value(
        _first_non_empty(raw.get("instrumentDescription"), raw.get("equipmentDescription"))
    )
    return _clean_none(
        {
            "id": instrument_id,
            "title": title,
            "description": description,
            "manufacturer": manufacturer if _is_substantive_instrument_value(manufacturer) else None,
            "model": model if _is_substantive_instrument_value(model) else None,
            "verticalRange": _normalize_vertical_range(raw),
            "observableVariables": _normalize_observable_variables(raw),
            "observableGeometry": _normalize_observable_geometry(raw),
        }
    )


def _normalize_instruments(deployments: Sequence[Dict[str, Any]], *, facility_id: str) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for dep in deployments:
        if not isinstance(dep, dict):
            continue
        instrument = _normalize_instrument(dep, facility_id=facility_id)
        if not instrument:
            continue
        instrument_id = instrument.get("id")
        if isinstance(instrument_id, str) and instrument_id not in by_id:
            by_id[instrument_id] = instrument
    return list(by_id.values())


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
    return None


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


def _schedule_start_datetime(raw: Dict[str, Any]) -> str:
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


def _schedule_uid_from_event(event_without_uid: Dict[str, Any]) -> str:
    seed = json.dumps(event_without_uid, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"schedule_{digest}"


def _flatten_schedule_candidates(value: Any, *, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    context = context or {}
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        merged = dict(context)
        merged.update(item)
        coverage = item.get("coverage") or item.get("Coverage")
        if isinstance(coverage, dict):
            cov = dict(merged)
            cov.update(coverage)
            out.append(cov)
        sampling = item.get("sampling") or item.get("Sampling")
        if isinstance(sampling, dict):
            samp = dict(merged)
            samp.update(sampling)
            out.append(samp)
        if any(k in item for k in ("temporalSamplingInterval", "samplingInterval", "temporalAggregate", "startHour", "diurnalBaseTime")):
            out.append(merged)
    return out


def _jscalendar_observing_schedule(raw: Dict[str, Any], *, time_zone: str = "UTC") -> Optional[Dict[str, Any]]:
    interval = _iso_duration(
        _first_non_empty(
            raw.get("temporalSamplingInterval"),
            raw.get("samplingInterval"),
            raw.get("sampleInterval"),
            raw.get("interval"),
            raw.get("temporalAggregate"),
        )
    )
    has_window = any(_non_empty(raw.get(k)) for k in ("startMonth", "endMonth", "startWeekday", "endWeekday", "startHour", "endHour", "startMinute", "endMinute"))
    if not interval and not has_window:
        return None
    rule: Dict[str, Any] = {"@type": "RecurrenceRule", "frequency": "daily"}
    if interval:
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
    event_without_uid: Dict[str, Any] = {
        "@type": "Event",
        "start": _schedule_start_datetime(raw),
        "timeZone": time_zone,
        "duration": interval or "P1D",
        "recurrenceRules": [rule],
    }
    aggregation: Dict[str, Any] = {}
    raw_mapping: Dict[str, Any] = dict(raw)
    reporting_value: object = raw_mapping.get("reporting")
    reporting: Dict[str, Any]
    if isinstance(reporting_value, dict):
        reporting = reporting_value
    else:
        reporting = {}
    temporal_aggregate = _iso_duration(
        _first_non_empty(
            raw_mapping.get("temporalAggregate"),
            reporting.get("temporalAggregate"),
            reporting.get("temporalReportingInterval"),
        )
    )
    if temporal_aggregate:
        aggregation["temporalAggregate"] = temporal_aggregate
    diurnal = _normalize_diurnal_time(raw.get("diurnalBaseTime"))
    if diurnal:
        aggregation["diurnalBaseTime"] = diurnal
    if aggregation:
        event_without_uid["wmo.int:aggregation"] = aggregation
    event = dict(event_without_uid)
    event["uid"] = _schedule_uid_from_event(event_without_uid)
    return event


def _register_observing_schedule_refs(
    groups: Sequence[Any],
    *,
    schedule_registry: Dict[str, Dict[str, Any]],
    time_zone: str = "UTC",
) -> Optional[List[Dict[str, Any]]]:
    refs: List[Dict[str, Any]] = []
    for group in groups:
        for candidate in _flatten_schedule_candidates(group):
            event = _jscalendar_observing_schedule(candidate, time_zone=time_zone)
            if not event:
                continue
            uid = str(event["uid"])
            schedule_registry.setdefault(uid, event)
            refs.append({"observingSchedule": uid, "date": _temporal_begin_date(candidate)})
    refs = _uniq_dicts(_clean_none(refs))
    return refs or None


def _normalize_observation_reporting(*sources: Any) -> Optional[Dict[str, List[Any]]]:
    records: List[Dict[str, Any]] = []
    dates: List[str] = []
    for source in sources:
        for candidate in _flatten_schedule_candidates(source):
            reporting = candidate.get("reporting") if isinstance(candidate.get("reporting"), dict) else candidate
            if not isinstance(reporting, dict):
                continue
            record: Dict[str, Any] = {}
            for source_key, target_key in (
                ("internationalExchange", "internationalExchange"),
                ("temporalReportingInterval", "temporalAggregate"),
                ("temporalAggregate", "temporalAggregate"),
                ("uom", "uom"),
                ("dataPolicy", "dataPolicy"),
                ("levelOfData", "levelOfData"),
                ("referenceDatum", "referenceDatum"),
                ("timeliness", "timeliness"),
            ):
                if source_key not in reporting:
                    continue
                value = reporting.get(source_key)
                if source_key == "internationalExchange":
                    parsed = _parse_bool(value)
                    value = parsed if parsed is not None else value
                if _non_empty(value) or isinstance(value, bool):
                    record[target_key] = _preserve_nulls(value)
            if record:
                records.append(record)
                dates.append(_temporal_begin_date(candidate))
    if not records:
        return None
    deduped_records: List[Dict[str, Any]] = []
    deduped_dates: List[str] = []
    seen: set[str] = set()
    for record, date in zip(records, dates):
        marker = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
        if marker in seen:
            continue
        seen.add(marker)
        deduped_records.append(record)
        deduped_dates.append(date)
    preferred_order = ["internationalExchange", "temporalAggregate", "uom", "dataPolicy", "levelOfData", "referenceDatum"]
    extra_keys = sorted(key for record in deduped_records for key in record if key not in preferred_order and key != "timeliness")
    reporting_out: Dict[str, Any] = {}
    for key in preferred_order + extra_keys:
        if any(key in record for record in deduped_records):
            reporting_out[key] = [record.get(key) for record in deduped_records]
    if any("timeliness" in record for record in deduped_records):
        reporting_out["temporalTimeliness"] = [
            {"timeliness": record.get("timeliness"), "date": date}
            for record, date in zip(deduped_records, deduped_dates)
            if "timeliness" in record
        ]
    return reporting_out or None


def _flatten_deployments_from_observations(observations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deployments: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for obs in observations:
        for dep in _as_list(obs.get("deployments")):
            if not isinstance(dep, dict):
                continue
            dep_id = str(_first_non_empty(dep.get("@gml:id"), dep.get("@id"), dep.get("serialNumber"), dep.get("model"), id(dep)))
            if dep_id in seen:
                continue
            seen.add(dep_id)
            deployments.append(dep)
    return deployments


def _normalize_deployment(
    raw: Dict[str, Any],
    *,
    index: int,
    facility_id: str,
    schedule_registry: Dict[str, Dict[str, Any]],
    time_zone: str = "UTC",
) -> Dict[str, Any]:
    record_id = _deployment_record_id(raw, index=index, facility_id=facility_id)
    start, end = _extract_interval(raw)
    contacts, _ = _collect_contacts(raw.get("contact"), raw.get("contacts"), raw.get("responsibleParty"))
    observing_schedule = _register_observing_schedule_refs(
        [raw.get("dataGeneration"), raw.get("coverage"), raw.get("sampling"), raw.get("observingSchedule")],
        schedule_registry=schedule_registry,
        time_zone=time_zone,
    )
    payload: Dict[str, Any] = {
        "id": record_id,
        "time": _time_interval(start, end),
        "description": _normalize_description_value(raw.get("description")),
        "sourceOfObservation": raw.get("sourceOfObservation"),
        "observingMethod": raw.get("observingMethod"),
        "instrument": _instrument_refs_for_deployment(raw, facility_id=facility_id),
        "serialNumbers": _deployment_serial_numbers(raw),
        "exposure": raw.get("exposure"),
        "representativeness": raw.get("representativeness"),
        "localReferenceSurface": raw.get("localReferenceSurface"),
        "verticalDistanceFromReferenceSurface": _first_non_empty(
            raw.get("verticalDistanceFromReferenceSurface"),
            raw.get("distanceFromReferenceSurface"),
        ),
        "temporalInstrumentOperatingStatus": _normalize_temporal_instrument_operating_status(raw.get("instrumentOperatingStatus")),
        "contacts": contacts,
        "keywords": _keywords_from_values(_collect_discovery_values("deployment", raw, "keywords")),
        "links": _extract_links(raw, "deployment"),
    }
    cleaned = _clean_none(payload)
    if observing_schedule:
        cleaned["temporalObservingSchedule"] = observing_schedule
    return cleaned


def _derive_observation_time(observation: Dict[str, Any], deployments: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    intervals: List[List[Any]] = []
    for dep in deployments:
        start, end = _extract_interval(dep)
        interval = _time_interval(start, end)
        if interval and isinstance(interval.get("interval"), list):
            intervals.append(interval["interval"])
    if intervals:
        starts = [item[0] for item in intervals if item[0] != ".."]
        has_open = any(item[1] == ".." for item in intervals)
        ends = [item[1] for item in intervals if item[1] != ".."]
        return {"interval": [min(starts) if starts else "..", ".." if has_open else (max(ends) if ends else "..")]}
    return _time_interval(observation.get("beginPosition"), observation.get("endPosition"))


def _normalize_observation(raw: Dict[str, Any], *, index: int, facility_id: str) -> Dict[str, Any]:
    embedded_deployments = [item for item in _as_list(raw.get("deployments")) if isinstance(item, dict)]
    observed_variable = raw.get("observedVariable") or raw.get("observedProperty")
    explicit_obs_id = _first_non_empty(raw.get("identifier"), raw.get("@gml:id"), raw.get("@id"), raw.get("id"))
    obs_id = _first_non_empty(explicit_obs_id, _compact_wmdr_code_value(observed_variable), f"{facility_id}:observation:{index}")
    source_id = _sanitize_id(str(obs_id))
    observed_geometry_type = raw.get("observedGeometryType") or raw.get("geometryType") or raw.get("type")
    title = _first_non_empty(
        _format_observation_title(observed_variable, observed_geometry_type),
        raw.get("title"),
        raw.get("name"),
        f"Observation {index}",
    )
    contacts, _ = _collect_contacts(raw.get("contact"), raw.get("contacts"), raw.get("responsibleParty"))
    reporting_sources: List[Any] = [raw.get("dataGeneration"), raw.get("reportingSchedule")]
    reporting_sources.extend(dep.get("dataGeneration") or dep.get("reportingSchedule") for dep in embedded_deployments)
    deployment_refs = [
        _deployment_record_id(dep, index=dep_index, facility_id=facility_id)
        for dep_index, dep in enumerate(embedded_deployments, start=1)
    ]
    payload: Dict[str, Any] = {
        "id": f"observation:{source_id}",
        "title": title,
        "time": _derive_observation_time(raw, embedded_deployments),
        "observedVariable": observed_variable,
        "observedGeometryType": observed_geometry_type,
        "observedDomain": _observed_domain_from_observed_variable(observed_variable),
        "programAffiliations": _normalize_program_affiliations(raw.get("programAffiliation")),
        "contacts": contacts,
        "temporalDataPolicy": _normalize_temporal_data_policy(
            raw.get("temporalDataPolicy") or raw.get("dataPolicy") or raw.get("dataPolicyHistory")
        ),
        "deployments": deployment_refs,
        "keywords": _keywords_from_values(_collect_discovery_values("observation", raw, "keywords")),
        "links": _extract_links(raw, "observation"),
    }
    cleaned = _clean_none(payload)
    reporting = _normalize_observation_reporting(*reporting_sources)
    if reporting:
        cleaned["reporting"] = reporting
    return cleaned


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def _facility_identifier(facility: Dict[str, Any], header: Optional[Dict[str, Any]] = None) -> str:
    header = header or {}
    raw = _first_non_empty(
        facility.get("identifier"),
        facility.get("wigosStationIdentifier"),
        facility.get("wigosIdentifier"),
        facility.get("id"),
        facility.get("@gml:id"),
        header.get("identifier"),
        header.get("id"),
        facility.get("name"),
        "facility",
    )
    return str(raw)


def _facility_title(facility: Dict[str, Any]) -> str:
    return str(_first_non_empty(facility.get("name"), facility.get("title"), facility.get("identifier"), "facility"))


def _copy_known_facility_properties(facility: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("facilityType", "wmoRegion", "surfaceCoverClassification", "timeZone", "regionOfOrigin"):
        if _non_empty(facility.get(key)):
            out[key] = facility[key]
    facility_sets = _facility_set_refs(facility.get("facilitySet") or facility.get("facilitySets"))
    if facility_sets:
        out["facilitySets"] = facility_sets
    temporal_territory = _normalize_temporal_territory(_first_non_empty(facility.get("territory"), facility.get("territoryName")))
    if temporal_territory:
        out["temporalTerritory"] = temporal_territory
    environment = _normalize_environment(facility)
    if environment:
        out["environment"] = environment
    return out


def _facility_time(
    facility: Dict[str, Any],
    observations: Sequence[Dict[str, Any]],
    deployments: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    del observations, deployments
    return _time_interval(facility.get("dateEstablished"), facility.get("dateClosed")) or {"interval": ["..", ".."]}


def _facility_properties(
    facility: Dict[str, Any],
    observations: Sequence[Dict[str, Any]],
    deployments: Sequence[Dict[str, Any]],
    header: Optional[Dict[str, Any]] = None,
    source_name: Optional[str] = None,
) -> Dict[str, Any]:
    header = header or {}
    facility_id = _facility_identifier(facility, header)
    contacts, _ = _collect_contacts(
        facility.get("contact"),
        facility.get("contacts"),
        facility.get("responsibleParty"),
        header.get("recordOwner"),
    )
    keywords = _keywords_from_values(_collect_discovery_values("facility", facility, "keywords"))
    schedule_registry: Dict[str, Dict[str, Any]] = {}
    facility_time_zone = str(_first_non_empty(facility.get("timeZone"), "UTC"))

    all_deployments: List[Dict[str, Any]] = [dep for dep in deployments if isinstance(dep, dict)]
    all_deployments.extend(_flatten_deployments_from_observations([obs for obs in observations if isinstance(obs, dict)]))

    normalized_deployments_by_id: Dict[str, Dict[str, Any]] = {}
    for index, dep in enumerate(all_deployments, start=1):
        normalized = _normalize_deployment(
            dep,
            index=index,
            facility_id=facility_id,
            schedule_registry=schedule_registry,
            time_zone=facility_time_zone,
        )
        dep_id = normalized.get("id")
        if isinstance(dep_id, str) and dep_id not in normalized_deployments_by_id:
            normalized_deployments_by_id[dep_id] = normalized
    normalized_deployments = list(normalized_deployments_by_id.values())
    normalized_instruments = _normalize_instruments(all_deployments, facility_id=facility_id)
    normalized_observations = [
        _normalize_observation(obs, index=index, facility_id=facility_id)
        for index, obs in enumerate(observations, start=1)
        if isinstance(obs, dict)
    ]

    props: Dict[str, Any] = {
        "type": "facility",
        "title": _facility_title(facility),
        **_record_timestamps(header, source_name=source_name),
        "description": _first_non_empty(
            _normalize_description_value(facility.get("description")),
            _normalize_description_value(facility.get("information")),
            f"WMDR2 facility record for {_facility_title(facility)}",
        ),
        "contacts": contacts,
        "keywords": keywords,
        "links": _extract_links(facility, "facility"),
        **_copy_known_facility_properties(facility),
        "temporalProgramAffiliation": _normalize_temporal_program_affiliation(facility.get("programAffiliation")),
        "temporalReportingStatus": _normalize_reporting_status_timeline(facility.get("reportingStatus")),
        "schedules": list(schedule_registry.values()),
        "observations": normalized_observations,
        "deployments": normalized_deployments,
        "instruments": normalized_instruments,
    }
    return _clean_none(props)


def build_facility_feature(
    facility: Dict[str, Any],
    observations: Optional[Sequence[Dict[str, Any]]] = None,
    deployments: Optional[Sequence[Dict[str, Any]]] = None,
    header: Optional[Dict[str, Any]] = None,
    source_name: Optional[str] = None,
) -> Dict[str, Any]:
    observations = observations or []
    deployments = deployments or []
    header = header or {}
    facility_id = _facility_identifier(facility, header)
    temporal_geometry_entries = _facility_temporal_geometry_entries(facility)
    feature: Dict[str, Any] = {
        "type": "Feature",
        "id": f"facility:{_sanitize_id(facility_id)}",
        "geometry": _facility_geometry_from_entries(temporal_geometry_entries),
        "temporalGeometry": _temporal_geometry_extension(temporal_geometry_entries),
        "time": _facility_time(facility, observations, deployments),
        "conformsTo": [WMDR2_CORE_CONF],
        "properties": _facility_properties(facility, observations, deployments, header, source_name=source_name),
    }
    return _restore_null_sentinel(_finalize_wmdr2_value(_clean_none(feature)))


def convert_payload(payload: Any, *, source_name: str = "record") -> Dict[str, Any]:
    if isinstance(payload, dict) and any(k in payload for k in ("facility", "observations", "deployments", "header")):
        header = _as_dict(payload.get("header"))
        facility = _as_dict(payload.get("facility"))
        observations = [item for item in _as_list(payload.get("observations")) if isinstance(item, dict)]
        deployments = [item for item in _as_list(payload.get("deployments")) if isinstance(item, dict)]
        if not facility:
            facility = {"identifier": source_name, "name": source_name}
        return build_facility_feature(facility, observations, deployments, header, source_name=source_name)
    if isinstance(payload, dict):
        return build_facility_feature(payload, [], [], {}, source_name=source_name)
    raise ValueError(f"Cannot convert {source_name}: unsupported JSON payload shape {type(payload).__name__}")


def convert_group(parts: Dict[str, Any], *, source_name: str) -> Dict[str, Any]:
    header = _as_dict(parts.get("header"))
    facility = _as_dict(parts.get("facility"))
    observations = [item for item in _as_list(parts.get("observations")) if isinstance(item, dict)]
    deployments = [item for item in _as_list(parts.get("deployments")) if isinstance(item, dict)]
    if not facility:
        facility = {"identifier": source_name, "name": source_name}
    return build_facility_feature(facility, observations, deployments, header, source_name=source_name)


def _load_code_list_labels(section: Dict[str, Any], *, base_dir: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    labels: Dict[str, Dict[str, str]] = {}
    raw = section.get("codeListLabels") or section.get("code_list_labels") or {}
    if not isinstance(raw, dict):
        return labels
    inline = raw.get("inline")
    if isinstance(inline, dict):
        for domain, mapping in inline.items():
            if not isinstance(mapping, dict):
                continue
            target = labels.setdefault(str(domain), {})
            for code, label in mapping.items():
                if code and label:
                    target[str(code).strip().lstrip("_")] = str(label).strip()
    files = raw.get("files", [])
    if isinstance(files, (str, Path)):
        files = [files]
    if not isinstance(files, list):
        return labels
    for item in files:
        path_text: Optional[str] = None
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            path_text = item["path"]
        elif isinstance(item, (str, Path)):
            path_text = str(item)
        if not path_text:
            continue
        path = Path(path_text)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                uri = (row.get("uri") or row.get("@id") or row.get("id") or "").strip().strip("<>")
                domain = (row.get("domain") or "").strip()
                code = (row.get("notation") or row.get("skos:notation") or row.get("code") or "").strip().strip("'").lstrip("_")
                label = (row.get("label") or row.get("rdfs:label") or row.get("name") or "").strip()
                if uri and (not domain or not code):
                    _, uri_domain, uri_code = _extract_code_list_ref(uri)
                    domain = domain or (uri_domain or "")
                    code = code or (uri_code or "")
                if domain and code and label:
                    labels.setdefault(domain, {})[code] = label
    return labels


def convert_file(
    input_path: Path,
    target_dir: Path,
    *,
    discovery_policy: Optional[Dict[str, Dict[str, List[str]]]] = None,
    code_list_labels: Optional[Dict[str, Dict[str, str]]] = None,
) -> Path:
    global DISCOVERY_POLICY, CODE_LIST_LABELS
    if discovery_policy is not None:
        DISCOVERY_POLICY = copy.deepcopy(discovery_policy)
    if code_list_labels is not None:
        CODE_LIST_LABELS = copy.deepcopy(code_list_labels)
    payload = _load_json(input_path)
    record = convert_payload(payload, source_name=input_path.stem)
    out = target_dir / f"{input_path.stem}{OUTPUT_SUFFIX}"
    _write_json(out, record)
    return out


def convert_tree(
    source: Path,
    target: Path,
    *,
    pattern: str = DEFAULT_PATTERN,
    recursive: bool = True,
    discovery_policy: Optional[Dict[str, Dict[str, List[str]]]] = None,
    code_list_labels: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Path]:
    global DISCOVERY_POLICY, CODE_LIST_LABELS
    if discovery_policy is not None:
        DISCOVERY_POLICY = copy.deepcopy(discovery_policy)
    if code_list_labels is not None:
        CODE_LIST_LABELS = copy.deepcopy(code_list_labels)
    files = _iter_json_files(source, pattern=pattern, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No JSON files found under {source!s}")
    target.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, Dict[str, Any]] = {}
    full_files: List[Path] = []
    for path in files:
        payload = _load_json(path)
        kind = _detect_kind(path, payload)
        if kind in {"header", "facility", "observations", "deployments"} and path.stem.lower().endswith(
            ("_header", "_facility", "_observations", "_deployments")
        ):
            groups.setdefault(_part_group_key(path), {})[kind] = payload
        else:
            full_files.append(path)

    written: List[Path] = []
    for path in full_files:
        written.append(convert_file(path, target, discovery_policy=DISCOVERY_POLICY, code_list_labels=CODE_LIST_LABELS))
    for group_key, parts in sorted(groups.items()):
        record = convert_group(parts, source_name=group_key)
        out = target / f"{group_key}{OUTPUT_SUFFIX}"
        _write_json(out, record)
        written.append(out)
    return sorted(written)


# ---------------------------------------------------------------------------
# Optional catalogue post-processing
# ---------------------------------------------------------------------------


def _config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    parsed = _parse_bool(value)
    return bool(parsed) if parsed is not None else False


def _resolve_config_path_value(value: Any, *, base_dir: Path) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    path = path.expanduser()
    return path if path.is_absolute() else base_dir / path


def _catalogue_paths_from_config(
    section: Dict[str, Any],
    *,
    base_dir: Path,
    target: Path,
    pattern: str,
    recursive: bool,
) -> Optional[Any]:
    catalogues = section.get("catalogues")
    if not isinstance(catalogues, dict) or not _config_bool(catalogues.get("enabled")):
        return None
    if "source" in catalogues:
        raise SystemExit("catalogues.source is obsolete; catalogue input is always the converter target.")

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from convert_wmdr2_json_to_catalogue_version import CataloguePaths

    default_records_path = target / "catalogue_representation"
    default_contacts_path = target / "catalogues" / "contacts.json"
    default_instruments_path = target / "catalogues" / "instruments.json"

    records_path = _resolve_config_path_value(
        catalogues.get("records_path") or default_records_path,
        base_dir=base_dir,
    )
    contacts_path = _resolve_config_path_value(
        catalogues.get("contacts_path") or default_contacts_path,
        base_dir=base_dir,
    )
    instruments_path = _resolve_config_path_value(
        catalogues.get("instruments_path") or default_instruments_path,
        base_dir=base_dir,
    )

    return CataloguePaths(
        source=target,
        records_path=records_path,
        contacts_path=contacts_path,
        instruments_path=instruments_path,
        pattern=pattern,
        recursive=recursive,
    )


def _run_catalogue_post_processing(written: Sequence[Path], catalogue_paths: Any) -> List[Path]:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from convert_wmdr2_json_to_catalogue_version import convert_catalogue_files

    return convert_catalogue_files(written, catalogue_paths)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert simplified WMDR1 JSON to facility-centric WMDR2 core JSON Features."
    )
    parser.add_argument("--config", type=Path, help="Optional YAML config file.")
    parser.add_argument("--source", type=Path, help="Input JSON file or directory.")
    parser.add_argument("--target", type=Path, help="Output directory for generated .json records.")
    parser.add_argument("--pattern", default=None, help="Input glob pattern, default from config or *.json.")
    parser.add_argument("--recursive", action="store_true", default=None, help="Search source recursively.")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive", help="Do not search source recursively.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config_path = _discover_config_path(args.config)
    cfg = _load_config(config_path) if config_path else {}
    section = _cfg_section(cfg)
    base_dir = config_path.parent if config_path else Path.cwd()

    source_raw = args.source or section.get("source")
    target_raw = args.target or section.get("target")
    if not source_raw:
        loaded = _format_loaded_config_hint(config_path, section)
        available_sections = ", ".join(sorted(cfg.keys())) if cfg else "none"
        raise SystemExit(
            "Missing source. Set convert_wmdr10_json_to_wmdr2_json.source "
            "in config.yaml or pass --source.\n"
            f"{loaded}\nAvailable top-level config sections: {available_sections}"
        )
    if not target_raw:
        loaded = _format_loaded_config_hint(config_path, section)
        available_sections = ", ".join(sorted(cfg.keys())) if cfg else "none"
        raise SystemExit(
            "Missing target. Set convert_wmdr10_json_to_wmdr2_json.target "
            "in config.yaml or pass --target.\n"
            f"{loaded}\nAvailable top-level config sections: {available_sections}"
        )

    source = Path(source_raw).expanduser()
    target = Path(target_raw).expanduser()
    if not source.is_absolute():
        source = base_dir / source
    if not target.is_absolute():
        target = base_dir / target
    pattern = args.pattern or section.get("pattern") or DEFAULT_PATTERN
    recursive = args.recursive if args.recursive is not None else bool(section.get("recursive", True))
    discovery_policy = _normalize_discovery_policy(section)
    code_list_labels = _load_code_list_labels(section, base_dir=base_dir)

    print(_format_loaded_config_hint(config_path, section))
    print(f"Source: {source}")
    print(f"Target: {target}")
    print(f"Pattern: {pattern}; recursive={recursive}")
    written = convert_tree(
        source,
        target,
        pattern=str(pattern),
        recursive=recursive,
        discovery_policy=discovery_policy,
        code_list_labels=code_list_labels,
    )
    print(f"Wrote {len(written)} WMDR2 JSON file(s) to {target}")

    catalogue_paths = _catalogue_paths_from_config(
        section,
        base_dir=base_dir,
        target=target,
        pattern=str(pattern),
        recursive=recursive,
    )
    if catalogue_paths is not None:
        externalized = _run_catalogue_post_processing(written, catalogue_paths)
        print(f"Wrote {len(externalized)} catalogue-based WMDR2 JSON file(s) to {catalogue_paths.records_path}")
        print(f"Wrote contacts catalogue to {catalogue_paths.contacts_path}")
        print(f"Wrote instruments catalogue to {catalogue_paths.instruments_path}")


if __name__ == "__main__":
    main()
