#!/usr/bin/env python3
"""Convert simplified WMDR 1.0 JSON records into facility-centric WMDR2 JSON.

WMDR v0.2.2 implementation notes
---------------------------------

* ``temporalGeometry`` remains the trajectory-style aligned-array object.
* Ordinary history is represented as arrays of objects with singular ``date``.
* ``historicalEnvironment`` is emitted directly under ``properties``.
* ``observedFeature`` uses ``domain``, ``domainFeature`` and ``featureName``.
* ``historicalDeployments`` and ``historicalReporting`` are nested under each
  observation because they are observation-specific dated states.
* Reusable JSCalendar schedule objects are emitted under ``properties.schedules``;
  observations refer to them through ``observingSchedules`` entries of the form
  ``{"date": "YYYY-MM-DD", "schedule": "schedule_..."}``.
* Root-level ``properties.deployments`` is no longer emitted.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    "facility": {"keywords": ["identifier", "name"], "links": ["onlineResource"]},
    "observation": {"keywords": [], "links": []},
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
    """Remove empty dictionary members and empty list members.

    Empty inner lists are preserved for ``temporalGeometry.methods`` because
    they are positional companions to ``coordinates`` and ``dates``.
    """

    def preserve_empty_list(path: Tuple[str, ...]) -> bool:
        return len(path) >= 2 and path[-2:] == ("temporalGeometry", "methods")

    if isinstance(obj, dict):
        cleaned = {key: _clean_none(value, _path=_path + (key,)) for key, value in obj.items()}
        return {key: value for key, value in cleaned.items() if value not in (None, "", [], {})}
    if isinstance(obj, list):
        cleaned = [_clean_none(value, _path=_path) for value in obj]
        return [value for value in cleaned if value not in ("", {}) and (value != [] or preserve_empty_list(_path))]
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
    if isinstance(value, dict):
        value = _first_non_empty(value.get("href"), value.get("url"), value.get("value"), value.get("#text"), value.get("text"))
    if isinstance(value, str) and value.strip().startswith(("http://codes.wmo.int/wmdr/", "https://codes.wmo.int/wmdr/")):
        return _normalize_code_value(value)
    return value


def _finalize_wmdr2_value(value: Any, *, key: Optional[str] = None) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"interval"} and value.get("interval") == ["..", ".."]:
            return None
        return {
            child_key: (
                child_value if child_key in {"href", "url"} else _finalize_wmdr2_value(child_value, key=child_key)
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
        return {key: _drop_source_metadata(value) for key, value in obj.items() if key not in metadata_keys}
    if isinstance(obj, list):
        return [_drop_source_metadata(item) for item in obj]
    return obj


def _normalize_display_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if _is_unknown_token(text):
        return "unknown"
    return re.sub(r"\s+", " ", text)


def _normalize_description_value(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = _drop_source_metadata(value)
        text = _first_non_empty(cleaned.get("description"), cleaned.get("value"), cleaned.get("#text"), cleaned.get("text"))
        if text:
            return _normalize_display_text(text)
        return _clean_none(cleaned)
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
    # Preserve explicit date-times. _normalize_date_value intentionally reduces
    # temporal history markers to date resolution, but record timestamps are
    # OGC Record timestamps and should keep their time component.
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


def _extract_interval(obj: Mapping[str, Any]) -> Tuple[Any, Any]:
    time_obj = obj.get("time")
    if isinstance(time_obj, dict):
        interval = time_obj.get("interval")
        if isinstance(interval, list) and interval:
            start = interval[0] if len(interval) > 0 else None
            end = interval[1] if len(interval) > 1 else None
            return start, end
        return _first_non_empty(time_obj.get("date"), time_obj.get("timestamp")), None
    return (
        _first_non_empty(obj.get("beginPosition"), obj.get("begin"), obj.get("start"), obj.get("dateEstablished"), obj.get("date")),
        _first_non_empty(obj.get("endPosition"), obj.get("end"), obj.get("stop"), obj.get("dateClosed")),
    )


def _entry_date(item: Any, fallback: str = "..") -> str:
    if isinstance(item, dict):
        return (
            _normalize_date_value(
                _first_non_empty(item.get("date"), item.get("beginPosition"), item.get("begin"), item.get("from"), item.get("start"))
            )
            or fallback
        )
    return fallback


def _temporal_begin_date(item: Any) -> str:
    return _entry_date(item, "..")


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
    # WMDR/GML pos is lat lon [height]; GeoJSON is lon lat [height].
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
    if raw in (None, "", [], {}):
        return []
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
        entries.append((date, json.dumps(coords, sort_keys=True), entry))

    for item in _as_list(source.get("geospatialLocation") or source.get("geometry")):
        add(item)
    for item in _as_list(source.get("geospatialLocationHistory") or source.get("geometryHistory") or source.get("historicalLocation")):
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
        dates.append(str(entry.get("date") or ".."))
        raw_methods = entry.get("methods")
        entry_methods = [method for method in raw_methods if isinstance(method, str)] if isinstance(raw_methods, list) else []
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
        coordinates = entry.get("coordinates")
        if isinstance(coordinates, list) and len(coordinates) >= 2:
            return {"type": "Point", "coordinates": coordinates}
    return None


def _point_geometry_from_entry(entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    coordinates = entry.get("coordinates")
    if isinstance(coordinates, list) and len(coordinates) >= 2:
        return {"type": "Point", "coordinates": coordinates}
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
# Config / file handling
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit(
            f"Cannot read config file {path}: PyYAML is not installed. Install pyyaml or pass --source/--target explicitly."
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
    key_text = ", ".join(keys) if keys else "no converter section keys"
    return f"Using config: {config_path} ({key_text})"


def _normalize_discovery_policy(section: Mapping[str, Any]) -> Dict[str, Dict[str, List[str]]]:
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
                policy[entity][bucket] = [str(v).strip() for v in values if isinstance(v, str) and str(v).strip()]
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
# Discovery, contacts and links
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
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _collect_discovery_values(entity_type: str, source: Mapping[str, Any], bucket: str) -> List[Any]:
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


def _extract_links(source: Mapping[str, Any], entity_type: str) -> List[Dict[str, Any]]:
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
    phones = [{"value": voice.strip()} for voice in _as_list(phone_obj.get("voice")) if isinstance(voice, str) and voice.strip()]
    if phones:
        contact["phones"] = phones
    address_obj = _as_dict(info.get("address"))
    emails = [
        {"value": email.strip()} for email in _as_list(address_obj.get("electronicMailAddress")) if isinstance(email, str) and "@" in email
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
# Quantity, booleans and status helpers
# ---------------------------------------------------------------------------


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


def _normalize_official_status(value: Any) -> Optional[str]:
    """Normalize WMDR10 officialStatus to WMDR v0.2.2 observation status.

    The WMDR10 XML uses a boolean officialStatus.  The agreed mapping is:
    true -> primary, false -> additional.  String values are compacted but not
    otherwise remapped, except common boolean strings.
    """

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
    value: Any = source
    if isinstance(source, dict):
        value = _first_non_empty(source.get("#text"), source.get("text"), source.get("value"))
        uom = _first_non_empty(source.get("@uom"), source.get("uom"), source.get("unit"))
    parsed_value = _normalize_quantity_value(value)
    if parsed_value in (None, "", [], {}):
        return None
    out: Dict[str, Any] = {"value": parsed_value}
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
            quantity = _quantity_from_source(item)
            if quantity:
                return quantity
        return None
    return _quantity_from_source(source)


# ---------------------------------------------------------------------------
# Facility history and environment
# ---------------------------------------------------------------------------


def _entry_value(item: Any, preferred_key: str) -> Any:
    if isinstance(item, dict):
        if preferred_key in item:
            return item.get(preferred_key)
        rest = {
            key: value
            for key, value in item.items()
            if key not in {"date", "beginPosition", "endPosition", "from", "start", "to", "end", "begin"}
        }
        if len(rest) == 1:
            return next(iter(rest.values()))
        return rest or None
    return item


def _temporal_object_from_item(item: Any, *, output_key: str, value_keys: Sequence[str]) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        raw_value = _first_non_empty(*(item.get(key) for key in value_keys))
        value = _normalize_code_value(raw_value) if isinstance(raw_value, str) else raw_value
        if value in (None, "", [], {}) or _is_unknown_token(value):
            return None
        return {"date": _entry_date(item), output_key: value}
    value = _normalize_code_value(item) if isinstance(item, str) else item
    if value in (None, "", [], {}) or _is_unknown_token(value):
        return None
    return {"date": "..", output_key: value}


def _normalize_temporal_values(value: Any, *, output_key: str, value_keys: Sequence[str]) -> Optional[List[Dict[str, Any]]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        record = _temporal_object_from_item(item, output_key=output_key, value_keys=value_keys)
        if record:
            out.append(record)
    return _uniq_dicts(_clean_none(out)) or None


def _normalize_temporal_climate_zone(value: Any) -> Optional[List[Dict[str, Any]]]:
    return _normalize_temporal_values(value, output_key="climateZone", value_keys=("climateZone", "value", "href"))


def _normalize_temporal_surface_cover(value: Any) -> Optional[List[Dict[str, Any]]]:
    records = _normalize_temporal_values(value, output_key="surfaceCover", value_keys=("surfaceCover", "value", "href")) or []
    source_items = _as_list(value)
    for record, item in zip(records, source_items):
        if isinstance(item, Mapping):
            classification = _first_non_empty(item.get("surfaceClassification"), item.get("classification"))
            if isinstance(classification, Mapping):
                classification = _first_non_empty(classification.get("href"), classification.get("value"), classification.get("surfaceClassification"))
            if _non_empty(classification):
                record["surfaceClassification"] = _compact_wmdr_code_value(classification) if isinstance(classification, str) else classification
    return _uniq_dicts(_clean_none(records)) or None


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
        date = _entry_date(item, fallback="..")
        if isinstance(item, Mapping):
            population = _parse_two_value_number_array(_first_non_empty(item.get("population"), item.get("value"), item.get("href")))
            perimeter = _parse_two_value_perimeter_array(_first_non_empty(item.get("perimeter_km"), item.get("perimeterKm"), item.get("perimeter")))
        else:
            population = _parse_two_value_number_array(item)
            perimeter = [10.0, 50.0]
        if population is not None:
            records.append({"population": population, "perimeter_km": perimeter, "dates": [date, ".."]})
    return _uniq_dicts(_clean_none(records)) or None


def _normalize_temporal_surface_roughness(value: Any) -> Optional[List[Dict[str, Any]]]:
    return _normalize_temporal_values(value, output_key="surfaceRoughness", value_keys=("surfaceRoughness", "roughness", "value", "href"))


def _normalize_temporal_instrument_operating_status(value: Any) -> Optional[List[Dict[str, Any]]]:
    records = []
    for item in _normalize_operating_status(value if isinstance(value, Mapping) else {"instrumentOperatingStatus": value}):
        if "operatingStatus" in item:
            records.append({"date": item.get("date"), "instrumentOperatingStatus": item.get("operatingStatus")})
    return _uniq_dicts(_clean_none(records)) or None


def _normalize_temporal_official_status(value: Any, *, fallback_date: str = "..", default_unknown: bool = False) -> Optional[List[Dict[str, Any]]]:
    if value is None and default_unknown:
        return [{"date": fallback_date, "officialStatus": "unknown"}]
    return _normalize_historical_official_status(value if isinstance(value, Mapping) else {"officialStatus": value}, fallback_date=fallback_date)


def _normalize_program_affiliations(value: Any) -> Optional[List[Any]]:
    values: List[Any] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            raw = _first_non_empty(item.get("programAffiliation"), item.get("program"), item.get("href"), item.get("value"))
        else:
            raw = item
        if _non_empty(raw):
            values.append(_compact_wmdr_code_value(raw) if isinstance(raw, str) else raw)
    return _uniq_scalars(values) or None


def _program_affiliation_values(value: Any) -> Optional[List[Any]]:
    return _normalize_program_affiliations(value)


def _normalize_historical_program_affiliation(value: Any, *, fallback_date: str) -> Optional[List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []

    def add_record(item: Mapping[str, Any], reporting_status: Any = None, *, status_date: Any = None) -> None:
        record: Dict[str, Any] = {"date": _entry_date(item, fallback=fallback_date) if status_date is None else (_normalize_date_value(status_date) or fallback_date)}
        program = _first_non_empty(item.get("programAffiliation"), item.get("programAffiliations"), item.get("program"), item.get("href"), item.get("value"))
        if _non_empty(program):
            record["programAffiliation"] = _compact_wmdr_code_value(program) if isinstance(program, str) else program
        program_id = _first_non_empty(item.get("programSpecificFacilityId"), item.get("programSpecificFacilityIds"))
        if _non_empty(program_id):
            record["programSpecificFacilityId"] = program_id
        program_title = _first_non_empty(item.get("programSpecificFacilityTitle"), item.get("programSpecificFacilityTitles"))
        if _non_empty(program_title):
            record["programSpecificFacilityTitle"] = program_title
        status = reporting_status if reporting_status is not None else _first_non_empty(item.get("reportingStatus"), item.get("declaredReportingStatus"))
        if _non_empty(status):
            record["reportingStatus"] = _compact_wmdr_code_value(status) if isinstance(status, str) else status
        if len(record) > 1:
            records.append(record)

    for item in _as_list(value):
        if isinstance(item, Mapping):
            statuses = _as_list(_first_non_empty(item.get("reportingStatus"), item.get("declaredReportingStatus")))
            dict_statuses = [status for status in statuses if isinstance(status, Mapping)]
            if dict_statuses:
                for status_item in dict_statuses:
                    status_value = _first_non_empty(status_item.get("reportingStatus"), status_item.get("declaredReportingStatus"), status_item.get("status"), status_item.get("value"), status_item.get("href"))
                    add_record(item, status_value, status_date=_entry_date(status_item, fallback=_entry_date(item, fallback=fallback_date)))
            else:
                add_record(item)
        elif _non_empty(item):
            records.append({"date": fallback_date, "programAffiliation": _compact_wmdr_code_value(item) if isinstance(item, str) else item})
    return _uniq_dicts(_clean_none(records)) or None


def _normalize_temporal_program_affiliation(value: Any, *, fallback_date: str = "..") -> Optional[List[Dict[str, Any]]]:
    return _normalize_historical_program_affiliation(value, fallback_date=fallback_date)

def _normalize_historical_territory(value: Any, *, fallback_date: str) -> Optional[List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            territory = _first_non_empty(item.get("territory"), item.get("territoryName"), item.get("href"), item.get("value"))
            date = _entry_date(item, fallback=fallback_date)
        else:
            territory = item
            date = fallback_date
        if _non_empty(territory):
            records.append({"date": date, "territory": _compact_wmdr_code_value(territory) if isinstance(territory, str) else territory})
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
        values = [_parse_nullable_number(item) for item in value[:2]]
    elif isinstance(value, str):
        parts = [part for part in re.split(r"[,\s]+", value.strip()) if part]
        values = [_parse_nullable_number(part) for part in parts[:2]]
    else:
        values = [_parse_nullable_number(value)]
    while len(values) < 2:
        values.append(None)
    values = values[:2]
    if all(item is None for item in values):
        return None
    return values


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

    def add(date: str, key: str, value: Any) -> None:
        if not (_non_empty(value) or isinstance(value, bool)):
            return
        record = by_date.setdefault(date, {"date": date})
        record[key] = value

    def add_history(source_value: Any, key: str, value_keys: Sequence[str], *, transform: Optional[Any] = None) -> None:
        for item in _as_list(source_value):
            date = _entry_date(item, fallback=fallback_date)
            if isinstance(item, dict):
                raw_value = _first_non_empty(*(item.get(k) for k in value_keys))
                if raw_value is None and key in item:
                    raw_value = item.get(key)
            else:
                raw_value = item
            if raw_value in (None, "", [], {}):
                continue
            if transform is not None:
                transform_input = item if key == "population" and isinstance(item, dict) else raw_value
                value = transform(transform_input)
            else:
                value = _compact_wmdr_code_value(raw_value) if isinstance(raw_value, str) else raw_value
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
        for name in ("historicalEnvironment",):
            for item in _as_list(source.get(name)):
                if isinstance(item, dict):
                    date = _entry_date(item, fallback=fallback_date)
                    record = by_date.setdefault(date, {"date": date})
                    for key, value in item.items():
                        if key == "date":
                            continue
                        if _non_empty(value) or isinstance(value, bool):
                            record[key] = value
        topo = _topography_object(source)
        if topo:
            date = _entry_date(source, fallback=fallback_date)
            add(date, "topographyBathymetry", topo)

    return _uniq_dicts(_clean_none([by_date[key] for key in sorted(by_date)])) or None


def _normalize_environment(facility: Mapping[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Compatibility helper returning the v0.2.2 historicalEnvironment array.

    Older local tests called ``_normalize_environment`` and expected an
    ``environment`` wrapper. The v0.2.2 model has no wrapper; the returned
    value is meant to be assigned directly to ``properties.historicalEnvironment``.
    """

    start, _ = _extract_interval(facility)
    return _normalize_historical_environment(facility, fallback_date=_normalize_date_value(start) or "..")


def _normalize_temporal_data_policy(value: Any, *, fallback_date: str = "..") -> Optional[List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if not isinstance(item, Mapping):
            if _non_empty(item):
                records.append({"date": fallback_date, "dataPolicy": _compact_wmdr_code_value(item)})
            continue
        raw_policy = _first_non_empty(item.get("dataPolicy"), item.get("policy"), item.get("href"), item.get("value"))
        if not _non_empty(raw_policy):
            continue
        record: Dict[str, Any] = {
            "date": _entry_date(item, fallback=fallback_date),
            "dataPolicy": _compact_wmdr_code_value(raw_policy) if isinstance(raw_policy, str) else raw_policy,
        }
        if _non_empty(item.get("attribution")):
            record["attribution"] = _preserve_nulls(item.get("attribution"))
        records.append(record)
    return _uniq_dicts(_clean_none(records)) or None


def _facility_set_refs(value: Any) -> Optional[List[str]]:
    refs: List[str] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            raw = _first_non_empty(
                item.get("facilitySet"),
                item.get("facilitySets"),
                item.get("href"),
                item.get("value"),
                item.get("identifier"),
                item.get("id"),
            )
        else:
            raw = item
        if not _non_empty(raw):
            continue
        refs.append(f"facilitySet:{_sanitize_id(_compact_wmdr_code_value(raw))}")
    return _uniq_scalars(refs) or None


def facility_set_catalog_entry(value: Any, *, description: Optional[str] = None) -> Dict[str, Any]:
    code = _sanitize_id(_compact_wmdr_code_value(value))
    entry: Dict[str, Any] = {"id": f"facilitySet:{code}", "title": code}
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
            out[key] = value
    facility_sets = _facility_set_refs(facility.get("facilitySet") or facility.get("facilitySets"))
    if facility_sets:
        out["facilitySets"] = facility_sets
    return out


# ---------------------------------------------------------------------------
# Instruments, schedules, reporting and deployments
# ---------------------------------------------------------------------------


def _deployment_source_identifier(raw: Mapping[str, Any], *, index: int, facility_id: str) -> str:
    raw_id = _first_non_empty(raw.get("identifier"), raw.get("id"), raw.get("@gml:id"), raw.get("@id"), raw.get("uuid"))
    if raw_id:
        return _sanitize_id(raw_id)
    seed = json.dumps(_clean_none(dict(raw)), sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return _sanitize_id(f"{facility_id}:deployment:{index}:{digest}")



def _normalize_instrument_observed_property(raw: Mapping[str, Any]) -> Optional[List[Any]]:
    """Normalize instrument observedProperty values.

    WMDR observed-variable URIs are compacted to numeric code values, while
    local/free-text variables are preserved as strings.
    """

    value = _first_non_empty(raw.get("observedProperty"), raw.get("observedVariable"), raw.get("observableVariable"))
    values: List[Any] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            raw_value = _first_non_empty(
                item.get("observedProperty"),
                item.get("observedVariable"),
                item.get("href"),
                item.get("value"),
                item.get("description"),
                item.get("label"),
                item.get("name"),
            )
        else:
            raw_value = item
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
    return {"min": float(minimum), "max": float(maximum)}

def _instrument_source_values(raw: Mapping[str, Any]) -> Tuple[Any, Any]:
    instrument = _as_dict(raw.get("instrument") or raw.get("equipment"))
    manufacturer = _first_non_empty(raw.get("manufacturer"), instrument.get("manufacturer"), instrument.get("make"))
    model = _first_non_empty(raw.get("model"), instrument.get("model"), instrument.get("type"))
    return manufacturer, model


def _instrument_record_id(raw: Mapping[str, Any], *, facility_id: str) -> Optional[str]:
    instrument = _as_dict(raw.get("instrument") or raw.get("equipment"))
    raw_id = _first_non_empty(
        instrument.get("id"),
        instrument.get("identifier"),
        instrument.get("@gml:id"),
        raw.get("instrument"),
        raw.get("equipment"),
    )
    if isinstance(raw_id, str) and raw_id.startswith("instrument:"):
        return raw_id
    if isinstance(raw_id, str) and raw_id.strip() and not raw_id.strip().startswith(("http://", "https://")):
        return f"instrument:{_sanitize_id(raw_id)}"
    manufacturer, model = _instrument_source_values(raw)
    serial = _first_serial_number(raw)
    observed_property = _normalize_instrument_observed_property(raw)
    observed_geometry = _normalize_instrument_observed_geometry(raw)
    vertical_range = _normalize_vertical_range(raw)
    if not any(_is_substantive_instrument_value(value) for value in (manufacturer, model, serial, observed_property, observed_geometry, vertical_range)):
        return None
    seed = "|".join(str(_first_non_empty(value, "")) for value in (facility_id, manufacturer, model, serial, observed_property, observed_geometry, vertical_range))
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"instrument:{_sanitize_id(facility_id)}:{digest}"


def _instrument_ref_for_deployment(raw: Mapping[str, Any], *, facility_id: str) -> Optional[str]:
    return _instrument_record_id(raw, facility_id=facility_id)


def _scalar_reference(value: Any) -> Optional[str]:
    """Return one scalar identifier/reference from a scalar-or-list value.

    WMDR2 v0.2.x reusable deployment records use a single instrument
    reference, not a one-element list.  This guard keeps the base converter
    aligned with the catalogue externalizer even if an upstream helper or
    legacy input provides a list-shaped reference.
    """

    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item
    return None


def _is_substantive_instrument_value(value: Any) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, str) and _is_unknown_token(value):
        return False
    return True


def _normalize_instrument(raw: Mapping[str, Any], *, facility_id: str) -> Optional[Dict[str, Any]]:
    instrument_id = _instrument_record_id(raw, facility_id=facility_id)
    if not instrument_id:
        return None
    instrument = _as_dict(raw.get("instrument") or raw.get("equipment"))
    manufacturer, model = _instrument_source_values(raw)
    title = _normalize_description_value(_first_non_empty(instrument.get("title"), raw.get("instrumentTitle"), raw.get("equipmentTitle"), raw.get("title"), raw.get("name")))
    description = _normalize_description_value(_first_non_empty(instrument.get("description"), raw.get("instrumentDescription"), raw.get("equipmentDescription")))
    return _clean_none(
        {
            "id": instrument_id,
            "title": title,
            "description": description,
            "manufacturer": manufacturer if _is_substantive_instrument_value(manufacturer) else None,
            "model": model if _is_substantive_instrument_value(model) else None,
            "serialNumber": _first_serial_number(raw),
            "observedProperty": _normalize_instrument_observed_property(raw),
            "observedGeometry": _normalize_instrument_observed_geometry(raw),
            "verticalRange": _normalize_vertical_range(raw),
        }
    )


def _normalize_instruments(deployments: Sequence[Mapping[str, Any]], *, facility_id: str) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for dep in deployments:
        instrument = _normalize_instrument(dep, facility_id=facility_id)
        if not instrument:
            continue
        instrument_id = instrument.get("id")
        if isinstance(instrument_id, str) and instrument_id not in by_id:
            by_id[instrument_id] = instrument
    return list(by_id.values())


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
    seed = json.dumps(event_without_uid, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"schedule_{digest}"


def _flatten_schedule_candidates(value: Any, *, context: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    context = context or {}
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        merged: Dict[str, Any] = dict(context)
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
        if any(
            k in item
            for k in (
                "temporalSamplingInterval",
                "samplingInterval",
                "sampleInterval",
                "temporalAggregate",
                "temporalReportingInterval",
                "reportingInterval",
                "internationalExchange",
                "uom",
                "dataPolicy",
                "levelOfData",
                "timeliness",
                "startHour",
                "diurnalBaseTime",
                "reporting",
            )
        ):
            out.append(merged)
    return out


def _jscalendar_observing_schedule(raw: Mapping[str, Any], *, time_zone: str = "UTC") -> Optional[Dict[str, Any]]:
    reporting_value = raw.get("reporting")
    reporting = reporting_value if isinstance(reporting_value, dict) else {}
    interval = _iso_duration(
        _first_non_empty(
            raw.get("temporalSamplingInterval"),
            raw.get("samplingInterval"),
            raw.get("sampleInterval"),
            raw.get("interval"),
            raw.get("temporalAggregate"),
            reporting.get("temporalAggregate"),
            reporting.get("temporalReportingInterval"),
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
    temporal_aggregate = _iso_duration(_first_non_empty(raw.get("temporalAggregate"), reporting.get("temporalAggregate"), reporting.get("temporalReportingInterval")))
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
            refs.append({"date": _temporal_begin_date(candidate), "schedule": uid})
    return _uniq_dicts(_clean_none(refs)) or None


def _normalize_reporting_value(source_key: str, value: Any) -> Any:
    if source_key == "internationalExchange":
        parsed = _parse_bool(value)
        return parsed if parsed is not None else value
    if source_key == "dataPolicy":
        if isinstance(value, Mapping):
            policy = _first_non_empty(value.get("dataPolicy"), value.get("policy"), value.get("href"), value.get("value"))
            out: Dict[str, Any] = {}
            if _non_empty(policy):
                out["dataPolicy"] = _compact_wmdr_code_value(policy) if isinstance(policy, str) else policy
            if "attribution" in value:
                out["attribution"] = _preserve_nulls(value.get("attribution"))
            return _clean_none(out)
        return _compact_wmdr_code_value(value) if isinstance(value, str) else value
    if source_key in {"uom", "levelOfData", "referenceDatum", "referenceTimeSource", "timeStampMeaning", "dataFormat", "timeliness", "aggregation"}:
        return _compact_wmdr_code_value(value) if isinstance(value, str) else value
    return value


REPORTING_DEFINITION_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("internationalExchange", "internationalExchange"),
    ("temporalReportingInterval", "temporalAggregate"),
    ("temporalAggregate", "temporalAggregate"),
    ("aggregation", "aggregation"),
    ("dataPolicy", "dataPolicy"),
    ("levelOfData", "levelOfData"),
    ("referenceDatum", "referenceDatum"),
    ("referenceTimeSource", "referenceTimeSource"),
    ("timeStampMeaning", "timeStampMeaning"),
    ("numberOfObservationsInReportingInterval", "numberOfObservationsInReportingInterval"),
    ("dataFormat", "dataFormat"),
    ("timeliness", "timeliness"),
)


def _reporting_definition_id(definition: Mapping[str, Any]) -> str:
    seed = json.dumps(_restore_null_sentinel(_clean_none(dict(definition))), sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"reporting:{digest}"


def _reporting_parts(candidate: Mapping[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Split one WMDR10 reporting source into reusable and historical parts.

    The reusable definition is stored once under ``properties.reporting``.  The
    historical part remains attached to an observation and carries the date, a
    reference to that reusable definition, and observation-specific items such as
    unit of measure and links.
    """

    reporting_value = candidate.get("reporting")
    reporting = reporting_value if isinstance(reporting_value, dict) else candidate
    if not isinstance(reporting, dict):
        return None

    definition: Dict[str, Any] = {}
    for source_key, target_key in REPORTING_DEFINITION_FIELDS:
        if source_key not in reporting:
            continue
        value = _normalize_reporting_value(source_key, reporting.get(source_key))
        if _non_empty(value) or isinstance(value, bool):
            definition[target_key] = _preserve_nulls(value)

    historical: Dict[str, Any] = {"date": _temporal_begin_date(candidate)}
    if "uom" in reporting:
        uom = _normalize_reporting_value("uom", reporting.get("uom"))
        if _non_empty(uom) or uom is None:
            historical["uom"] = _preserve_nulls(uom)
    links = _extract_links(reporting, "observation")
    if links:
        historical["links"] = links

    definition = _restore_null_sentinel(_clean_none(definition))
    historical = _restore_null_sentinel(_clean_none(historical))
    if not definition and len(historical) <= 1:
        return None
    return definition, historical


def _register_reporting_refs(
    groups: Sequence[Any],
    *,
    reporting_registry: Dict[str, Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    refs: List[Dict[str, Any]] = []
    for group in groups:
        for candidate in _flatten_schedule_candidates(group):
            parts = _reporting_parts(candidate)
            if parts is None:
                continue
            definition, historical = parts
            if definition:
                reporting_id = _reporting_definition_id(definition)
                reporting_registry.setdefault(reporting_id, {"id": reporting_id, **definition})
                historical["reporting"] = reporting_id
            if len(historical) > 1:
                refs.append(historical)
    return _restore_null_sentinel(_uniq_dicts(_clean_none(refs))) or None


def _normalize_observation_reporting(
    *sources: Any,
    reporting_registry: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Normalize reporting history.

    When a registry is supplied, reusable reporting definitions are registered
    there and the returned historical records contain references.  Without a
    registry, a local temporary registry is used so helper-level callers see the
    same v0.2.3-style structure.
    """

    registry = reporting_registry if reporting_registry is not None else {}
    return _register_reporting_refs(list(sources), reporting_registry=registry)


def _first_serial_number(raw: Mapping[str, Any]) -> Optional[str]:
    source = raw.get("serialNumber")
    for item in _as_list(source):
        if isinstance(item, dict):
            value = _first_non_empty(item.get("serialNumber"), item.get("value"), item.get("#text"), item.get("text"))
        else:
            value = item
        if _non_empty(value):
            return str(value)
    return None


def _deployment_temporal_serial_number(raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(raw.get("serialNumber")):
        if isinstance(item, dict):
            value = _first_non_empty(item.get("serialNumber"), item.get("value"), item.get("#text"), item.get("text"))
            date = _entry_date(item, fallback=_entry_date(raw))
        else:
            value = item
            date = _entry_date(raw)
        if _non_empty(value):
            out.append({"date": date, "serialNumber": str(value)})
    return _uniq_dicts(_clean_none(out))


def _normalize_operating_status(raw: Mapping[str, Any]) -> List[Dict[str, Any]]:
    source = _first_non_empty(raw.get("instrumentOperatingStatus"), raw.get("operatingStatus"), raw.get("status"))
    records: List[Dict[str, Any]] = []
    for item in _as_list(source):
        if isinstance(item, dict):
            value = _first_non_empty(item.get("operatingStatus"), item.get("instrumentOperatingStatus"), item.get("status"), item.get("value"), item.get("href"))
            date = _entry_date(item, fallback=_entry_date(raw))
        else:
            value = item
            date = _entry_date(raw)
        if _non_empty(value):
            records.append({"date": date, "operatingStatus": _compact_wmdr_code_value(value) if isinstance(value, str) else value})
    return _uniq_dicts(_clean_none(records))


def _deployment_official_status_source(raw: Mapping[str, Any]) -> Any:
    reporting_items = raw.get("dataGeneration") or raw.get("reportingSchedule")
    for item in _as_list(reporting_items):
        if not isinstance(item, dict):
            continue
        reporting = _as_dict(item.get("reporting"))
        value = _first_non_empty(reporting.get("officialStatus"), item.get("officialStatus"), reporting.get("declaredStatus"), item.get("declaredStatus"))
        if value is not None:
            return value
    return _first_non_empty(raw.get("officialStatus"), raw.get("declaredStatus"))


def _normalize_historical_official_status(*sources: Mapping[str, Any], fallback_date: str = "..") -> Optional[List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    for source in sources:
        explicit = source.get("historicalOfficialStatus")
        for item in _as_list(explicit):
            if isinstance(item, dict):
                status = _normalize_official_status(_first_non_empty(item.get("officialStatus"), item.get("status"), item.get("value")))
                if status:
                    records.append({"date": _entry_date(item, fallback=fallback_date), "officialStatus": status})
        for value_key in ("officialStatus", "declaredStatus"):
            if value_key in source:
                status = _normalize_official_status(source.get(value_key))
                if status:
                    records.append({"date": _entry_date(source, fallback=fallback_date), "officialStatus": status})
        dep_status = _normalize_official_status(_deployment_official_status_source(source))
        if dep_status:
            records.append({"date": _entry_date(source, fallback=fallback_date), "officialStatus": dep_status})
    return _uniq_dicts(_clean_none(records)) or None


def _deployment_record_id(raw: Mapping[str, Any], *, index: int, facility_id: str) -> str:
    """Return the reusable deployment/instrument-instance identifier."""

    source_identifier = _deployment_source_identifier(raw, index=index, facility_id=facility_id)
    if source_identifier.startswith("deployment:"):
        return source_identifier
    return f"deployment:{source_identifier}"


def _normalize_deployment_record(
    raw: Mapping[str, Any],
    *,
    index: int,
    facility_id: str,
) -> Optional[Dict[str, Any]]:
    """Normalize a reusable deployment / instrument-instance record.

    The dated observation-to-deployment relationship is represented separately
    in ``historicalDeployment`` objects.  This reusable object holds the stable
    instrument-instance identity and references the reusable instrument record.
    """

    instrument_ref = _scalar_reference(_instrument_ref_for_deployment(raw, facility_id=facility_id))
    serial_number = _first_serial_number(raw)
    deployment_id = _deployment_record_id(raw, index=index, facility_id=facility_id)
    record = _clean_none(
        {
            "id": deployment_id,
            "instrument": instrument_ref,
            "serialNumber": serial_number,
            "links": _extract_links(raw, "deployment"),
        }
    )
    # Keep a deployment if it has at least one substantive instance property.
    if any(key in record for key in ("instrument", "serialNumber", "links")):
        return record
    return {"id": deployment_id}


def _normalize_deployments(
    deployments: Sequence[Mapping[str, Any]],
    *,
    facility_id: str,
) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, dep in enumerate(deployments, start=1):
        record = _normalize_deployment_record(dep, index=index, facility_id=facility_id)
        if not record:
            continue
        dep_id = record.get("id")
        if isinstance(dep_id, str) and dep_id not in by_id:
            by_id[dep_id] = record
    return list(by_id.values())


def _normalize_deployment(
    raw: Mapping[str, Any],
    *,
    index: int,
    facility_id: str,
    schedule_registry: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Normalize dated observation-to-deployment history entries.

    ``historicalDeployment`` now references the reusable deployment record, in
    the same way that ``historicalReporting`` references a reusable reporting
    definition.
    """

    del schedule_registry
    start, _ = _extract_interval(raw)
    fallback_date = _normalize_date_value(start) or ".."
    source_identifier = _deployment_source_identifier(raw, index=index, facility_id=facility_id)
    deployment_id = _deployment_record_id(raw, index=index, facility_id=facility_id)
    base: Dict[str, Any] = _clean_none(
        {
            "id": f"historicalDeployment:{source_identifier}",
            "deployment": deployment_id,
            "exposure": raw.get("exposure"),
            "links": _extract_links(raw, "deployment"),
        }
    )
    by_date: Dict[str, Dict[str, Any]] = {}

    def add(date: Any, **values: Any) -> None:
        normalized_date = _normalize_date_value(date) or fallback_date
        record = by_date.setdefault(normalized_date, {"date": normalized_date})
        for key, value in values.items():
            if _non_empty(value) or isinstance(value, bool):
                record[key] = value

    has_state = False
    for item in _normalize_operating_status(raw):
        has_state = True
        add(item.get("date"), operatingStatus=item.get("operatingStatus"))
    for entry in _facility_temporal_geometry_entries(raw):
        geometry = _point_geometry_from_entry(entry)
        if geometry:
            has_state = True
            add(entry.get("date"), geometry=geometry)

    # Always create at least one dated observation-to-deployment relation.
    # Do not create a redundant fallback entry when all relevant state already
    # has explicit dated records, but do anchor exposure/links if present.
    if not has_state or any(key in base for key in ("exposure", "links")):
        add(fallback_date)

    out: List[Dict[str, Any]] = []
    for date in sorted(by_date, key=lambda value: (value == "..", value)):
        record = dict(base)
        record.update(by_date[date])
        out.append(_clean_none(record))
    return _uniq_dicts(out)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _domain_object(raw: Mapping[str, Any], observed_property: Any) -> Optional[Dict[str, Any]]:
    """Return v0.2.2 observedFeature object.

    The output key is ``domain``.  ``domainName`` is accepted only as legacy
    input from earlier intermediate JSON, never emitted.
    """

    domain_source = _first_non_empty(raw.get("observedFeature"), raw.get("domain"), raw.get("observedDomain"))
    domain = _observed_domain_from_observed_variable(observed_property)
    domain_feature = None
    feature_name = None
    if isinstance(domain_source, dict):
        domain = _first_non_empty(domain_source.get("domain"), domain_source.get("domainName"), domain_source.get("value"), domain_source.get("href"), domain)
        domain_feature = _first_non_empty(domain_source.get("domainFeature"), domain_source.get("feature"))
        feature_name = domain_source.get("featureName")
    elif _non_empty(domain_source) and domain is None:
        domain = domain_source
    domain_feature = _first_non_empty(domain_feature, raw.get("domainFeature"), raw.get("observedDomainFeature"))
    feature_name = _first_non_empty(feature_name, raw.get("featureName"), raw.get("observedDomainFeatureName"))
    out: Dict[str, Any] = {}
    if _non_empty(domain):
        out["domain"] = _normalize_code_value(domain) if isinstance(domain, str) else domain
    if _non_empty(domain_feature):
        out["domainFeature"] = str(domain_feature)
    if _non_empty(feature_name):
        out["featureName"] = str(feature_name)
    return _clean_none(out) or None


# Backwards-compatible private alias for tests or local tooling.
def _observed_domain_object(raw: Mapping[str, Any], observed_property: Any) -> Optional[Dict[str, Any]]:
    return _domain_object(raw, observed_property)


def _derive_observation_time(observation: Mapping[str, Any], deployments: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
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


def _first_from_dicts(items: Sequence[Mapping[str, Any]], *keys: str) -> Any:
    for item in items:
        for key in keys:
            value = item.get(key)
            if _non_empty(value) or isinstance(value, bool):
                return value
    return None


def _first_vertical_distance_from_sources(raw: Mapping[str, Any], deployments: Sequence[Mapping[str, Any]]) -> Any:
    value = _normalize_vertical_distance_from_reference_surface(raw)
    if _non_empty(value):
        return value
    for dep in deployments:
        value = _normalize_vertical_distance_from_reference_surface(dep)
        if _non_empty(value):
            return value
    return None


def _deployment_identity(raw: Mapping[str, Any]) -> str:
    return str(_first_non_empty(raw.get("identifier"), raw.get("id"), raw.get("@gml:id"), raw.get("@id"), raw.get("uuid"), raw.get("serialNumber"), id(raw)))


def _flatten_deployments_from_observations(observations: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    deployments: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for obs in observations:
        for dep in _as_list(obs.get("deployments")):
            if not isinstance(dep, dict):
                continue
            dep_id = _deployment_identity(dep)
            if dep_id in seen:
                continue
            seen.add(dep_id)
            deployments.append(dep)
    return deployments


def _deployment_lookup(deployments: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    lookup: Dict[str, Mapping[str, Any]] = {}
    for dep in deployments:
        for key in ("identifier", "id", "@gml:id", "@id", "uuid"):
            value = dep.get(key)
            if _non_empty(value):
                lookup[str(value)] = dep
                lookup[_sanitize_id(value)] = dep
    return lookup


def _observation_deployment_sources(
    obs: Mapping[str, Any],
    *,
    all_deployments: Sequence[Mapping[str, Any]],
    all_observations_count: int,
) -> List[Mapping[str, Any]]:
    embedded: List[Mapping[str, Any]] = [item for item in _as_list(obs.get("deployments")) if isinstance(item, dict)]
    if embedded:
        return embedded
    refs = [item for item in _as_list(obs.get("deployments")) if isinstance(item, str)]
    if refs:
        lookup = _deployment_lookup(all_deployments)
        found = [lookup[ref] for ref in refs if ref in lookup]
        if found:
            return found
    if all_observations_count == 1:
        return list(all_deployments)
    return []


def _normalize_observation(
    raw: Mapping[str, Any],
    *,
    index: int,
    facility_id: str,
    schedule_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    reporting_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    deployment_sources: Optional[Sequence[Mapping[str, Any]]] = None,
    time_zone: str = "UTC",
) -> Dict[str, Any]:
    if schedule_registry is None:
        schedule_registry = {}
    if reporting_registry is None:
        reporting_registry = {}
    if deployment_sources is None:
        deployment_sources = [item for item in _as_list(raw.get("deployments")) if isinstance(item, dict)]
    observed_property = raw.get("observedProperty") or raw.get("observedVariable")
    explicit_obs_id = _first_non_empty(raw.get("identifier"), raw.get("@gml:id"), raw.get("@id"), raw.get("id"))
    obs_id = _first_non_empty(explicit_obs_id, _compact_wmdr_code_value(observed_property), f"{facility_id}:observation:{index}")
    source_id = _sanitize_id(str(obs_id))
    observed_geometry = _first_non_empty(raw.get("observedGeometry"), raw.get("observedGeometryType"), raw.get("geometryType"), raw.get("type"))
    title = _first_non_empty(_format_observation_title(observed_property, observed_geometry), raw.get("title"), raw.get("name"), f"Observation {index}")
    contacts, _ = _collect_contacts(raw.get("contact"), raw.get("contacts"), raw.get("responsibleParty"))

    reporting_sources: List[Any] = [raw.get("dataGeneration"), raw.get("reportingSchedule"), raw.get("reporting")]
    reporting_sources.extend(dep.get("dataGeneration") or dep.get("reportingSchedule") for dep in deployment_sources)

    observing_schedule_sources: List[Any] = [raw.get("dataGeneration"), raw.get("coverage"), raw.get("sampling"), raw.get("observingSchedule")]
    for dep in deployment_sources:
        observing_schedule_sources.extend([dep.get("dataGeneration"), dep.get("coverage"), dep.get("sampling"), dep.get("observingSchedule")])

    historical_deployments: List[Dict[str, Any]] = []
    for dep_index, dep in enumerate(deployment_sources, start=1):
        historical_deployments.extend(_normalize_deployment(dep, index=dep_index, facility_id=facility_id))

    reference_surface = _first_non_empty(raw.get("referenceSurface"), raw.get("localReferenceSurface"), _first_from_dicts(deployment_sources, "referenceSurface", "localReferenceSurface"))
    fallback_date = _entry_date(raw)
    official_sources: List[Mapping[str, Any]] = [raw, *deployment_sources]

    payload: Dict[str, Any] = {
        "id": f"observations:{source_id}",
        "title": title,
        "time": _derive_observation_time(raw, deployment_sources),
        "applicationArea": raw.get("applicationArea"),
        "observedProperty": observed_property,
        "observedGeometry": observed_geometry,
        "observedFeature": _domain_object(raw, observed_property),
        "programAffiliations": _normalize_program_affiliations(raw.get("programAffiliation")),
        "contacts": contacts,
        "sourceOfObservation": _first_non_empty(raw.get("sourceOfObservation"), _first_from_dicts(deployment_sources, "sourceOfObservation")),
        "referenceSurface": reference_surface,
        "representativeness": _first_non_empty(raw.get("representativeness"), _first_from_dicts(deployment_sources, "representativeness")),
        "verticalDistanceFromReferenceSurface": _first_vertical_distance_from_sources(raw, deployment_sources),
        "historicalOfficialStatus": _normalize_historical_official_status(*official_sources, fallback_date=fallback_date),
        "observingSchedules": _register_observing_schedule_refs(observing_schedule_sources, schedule_registry=schedule_registry, time_zone=time_zone),
        "historicalDeployments": historical_deployments,
        "historicalReporting": _preserve_nulls(_normalize_observation_reporting(*reporting_sources, reporting_registry=reporting_registry)),
        "keywords": _keywords_from_values(_collect_discovery_values("observation", raw, "keywords")),
        "links": _extract_links(raw, "observation"),
    }
    return _restore_null_sentinel(_clean_none(payload))


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def _facility_identifier(facility: Mapping[str, Any], header: Optional[Mapping[str, Any]] = None) -> str:
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


def _facility_title(facility: Mapping[str, Any]) -> str:
    return str(_first_non_empty(facility.get("name"), facility.get("title"), facility.get("identifier"), "facility"))


def _facility_time(facility: Mapping[str, Any], observations: Sequence[Mapping[str, Any]], deployments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    del observations, deployments
    return _time_interval(facility.get("dateEstablished"), facility.get("dateClosed")) or {"interval": ["..", ".."]}


def _facility_properties(
    facility: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    deployments: Sequence[Mapping[str, Any]],
    header: Optional[Mapping[str, Any]] = None,
    source_name: Optional[str] = None,
) -> Dict[str, Any]:
    header = header or {}
    facility_id = _facility_identifier(facility, header)
    contacts, _ = _collect_contacts(facility.get("contact"), facility.get("contacts"), facility.get("responsibleParty"), header.get("recordOwner"))
    keywords = _keywords_from_values(_collect_discovery_values("facility", facility, "keywords"))
    schedule_registry: Dict[str, Dict[str, Any]] = {}
    reporting_registry: Dict[str, Dict[str, Any]] = {}
    facility_time_zone = str(_first_non_empty(facility.get("timeZone"), "UTC"))
    facility_start, _ = _extract_interval(facility)
    fallback_date = _normalize_date_value(facility_start) or ".."

    raw_observations: List[Mapping[str, Any]] = [obs for obs in observations if isinstance(obs, dict)]
    all_deployments: List[Mapping[str, Any]] = [dep for dep in deployments if isinstance(dep, dict)]
    all_deployments.extend(_flatten_deployments_from_observations(raw_observations))
    normalized_instruments = _normalize_instruments(all_deployments, facility_id=facility_id)
    normalized_deployments = _normalize_deployments(all_deployments, facility_id=facility_id)

    normalized_observations: List[Dict[str, Any]] = []
    for index, obs in enumerate(raw_observations, start=1):
        deployment_sources = _observation_deployment_sources(
            obs,
            all_deployments=all_deployments,
            all_observations_count=len(raw_observations),
        )
        normalized_observations.append(
            _normalize_observation(
                obs,
                index=index,
                facility_id=facility_id,
                schedule_registry=schedule_registry,
                reporting_registry=reporting_registry,
                deployment_sources=deployment_sources,
                time_zone=facility_time_zone,
            )
        )

    # If there are top-level deployments but no observations, still harvest reusable
    # schedules and instruments.  Deployment records themselves are not emitted at
    # root level in v0.2.2.
    if not raw_observations:
        for dep in all_deployments:
            _register_observing_schedule_refs(
                [dep.get("dataGeneration"), dep.get("coverage"), dep.get("sampling"), dep.get("observingSchedule")],
                schedule_registry=schedule_registry,
                time_zone=facility_time_zone,
            )
            _normalize_observation_reporting(dep.get("dataGeneration"), dep.get("reportingSchedule"), dep.get("reporting"), reporting_registry=reporting_registry)

    copied = _copy_known_facility_properties(facility)
    historical_program_affiliation = _normalize_historical_program_affiliation(facility.get("programAffiliation"), fallback_date=fallback_date)
    historical_territory = _normalize_historical_territory(_first_non_empty(facility.get("territory"), facility.get("territoryName")), fallback_date=fallback_date)
    historical_environment = _normalize_historical_environment(facility, fallback_date=fallback_date)

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
        **copied,
        "historicalProgramAffiliation": historical_program_affiliation,
        "historicalTerritory": historical_territory,
        "historicalEnvironment": historical_environment,
        "schedules": list(schedule_registry.values()),
        "reporting": list(reporting_registry.values()),
        "deployments": normalized_deployments,
        "observations": normalized_observations,
        "instruments": normalized_instruments,
    }
    return _clean_none(props)


def build_facility_feature(
    facility: Mapping[str, Any],
    observations: Optional[Sequence[Mapping[str, Any]]] = None,
    deployments: Optional[Sequence[Mapping[str, Any]]] = None,
    header: Optional[Mapping[str, Any]] = None,
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


def convert_group(parts: Mapping[str, Any], *, source_name: str) -> Dict[str, Any]:
    header = _as_dict(parts.get("header"))
    facility = _as_dict(parts.get("facility"))
    observations = [item for item in _as_list(parts.get("observations")) if isinstance(item, dict)]
    deployments = [item for item in _as_list(parts.get("deployments")) if isinstance(item, dict)]
    if not facility:
        facility = {"identifier": source_name, "name": source_name}
    return build_facility_feature(facility, observations, deployments, header, source_name=source_name)


# ---------------------------------------------------------------------------
# Optional code-list labels
# ---------------------------------------------------------------------------


def _load_code_list_labels(section: Mapping[str, Any], *, base_dir: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
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


# ---------------------------------------------------------------------------
# File/tree conversion and optional catalogue post-processing
# ---------------------------------------------------------------------------


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
        if kind in {"header", "facility", "observations", "deployments"} and path.stem.lower().endswith(("_header", "_facility", "_observations", "_deployments")):
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
    section: Mapping[str, Any],
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
    records_path = _resolve_config_path_value(catalogues.get("records_path") or default_records_path, base_dir=base_dir)
    contacts_path = _resolve_config_path_value(catalogues.get("contacts_path") or default_contacts_path, base_dir=base_dir)
    instruments_path = _resolve_config_path_value(catalogues.get("instruments_path") or default_instruments_path, base_dir=base_dir)
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
    parser = argparse.ArgumentParser(description="Convert simplified WMDR1 JSON to facility-centric WMDR2 core JSON Features.")
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
        raise SystemExit(f"Missing source. {loaded} Available top-level config sections: {available_sections}")
    if not target_raw:
        raise SystemExit("Missing target. Pass --target or configure convert_wmdr10_json_to_wmdr2_json.target.")

    source = _resolve_config_path_value(source_raw, base_dir=base_dir)
    target = _resolve_config_path_value(target_raw, base_dir=base_dir)
    pattern = str(args.pattern or section.get("pattern") or DEFAULT_PATTERN)
    recursive = bool(section.get("recursive", True)) if args.recursive is None else bool(args.recursive)
    discovery_policy = _normalize_discovery_policy(section)
    code_list_labels = _load_code_list_labels(section, base_dir=base_dir)
    written = convert_tree(
        source,
        target,
        pattern=pattern,
        recursive=recursive,
        discovery_policy=discovery_policy,
        code_list_labels=code_list_labels,
    )
    catalogue_paths = _catalogue_paths_from_config(section, base_dir=base_dir, target=target, pattern=pattern, recursive=recursive)
    if catalogue_paths is not None:
        written = _run_catalogue_post_processing(written, catalogue_paths)
    for path in written:
        print(path)


if __name__ == "__main__":  # pragma: no cover
    main()
