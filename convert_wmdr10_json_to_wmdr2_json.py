#!/usr/bin/env python3
"""
convert_wmdr10_json_to_wmdr2_json.py

Convert simplified WMDR 1.0 JSON records into a WMDR2 core JSON
representation.

Important naming note
---------------------
The generated output files are ``.json`` records.

Design choices
--------------
- Output is a single facility-centric GeoJSON-like JSON Feature per WMDR1
  input record.
- The Feature root remains compatible with the OGC API - Records / GeoJSON
  envelope: ``type``, ``id``, ``geometry``, ``time``, ``conformsTo`` and
  ``properties``.
- WMDR2 core elements are first-class members of ``Feature.properties``.
  There is deliberately no ``properties.wmdr2`` wrapper.
- Observations and deployments are embedded under the facility record as
  ``properties.observations`` and ``properties.deployments``; instruments are
  represented once under ``properties.instruments`` and referenced by deployments.
- ``keywords`` are retained as lightweight discovery text. Controlled-vocabulary
  WMDR concepts are emitted as explicit compact code values, not as full
  code-list URLs and not as OGC Records ``themes``.
- The converter remains defensive because the simplified WMDR1 JSON shape has
  varied across iterations of the XML-to-JSON simplifier.

Example
-------
python convert_wmdr10_json_to_wmdr2_json.py \
    --source resources/wmdr10_json_examples \
    --target resources/wmdr2_json_examples

python convert_wmdr10_json_to_wmdr2_json.py --config config.yaml

Example config.yaml section
---------------------------
convert_wmdr10_json_to_wmdr2_json:
  source: resources/wmdr10_json_examples
  target: resources/wmdr2_json_examples
  pattern: "*.json"
  recursive: true

The legacy config section name ``convert_wmdr10_json_to_wmdr2_geojson`` is also
accepted for now, to ease migration of older local configs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


OGC_RECORD_CORE_CONF = "http://www.opengis.net/spec/ogcapi-records-1/1.0/conf/record-core"
WMDR2_CORE_CONF = "https://schemas.wmo.int/wmdr/2.0/core/facility-record"
DEFAULT_PATTERN = "*.json"
OUTPUT_SUFFIX = ".json"

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
    """Load a YAML config file.

    A present config file should never be ignored silently: if PyYAML is not
    installed, the file cannot be parsed, or the top-level object is not a
    mapping, fail clearly so the user knows why config values were not used.
    """
    if yaml is None:
        raise SystemExit(
            f"Cannot read config file {path}: PyYAML is not installed. "
            "Install pyyaml or pass --source/--target explicitly."
        )

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise SystemExit(f"Cannot read config file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"Config file {path} must contain a top-level YAML mapping.")

    return data


def _walk_up_for_config(start: Path) -> List[Path]:
    """Return config.yaml/config.yml candidates from ``start`` and its parents."""
    base = start if start.is_dir() else start.parent
    candidates: List[Path] = []
    for folder in (base, *base.parents):
        candidates.append(folder / "config.yaml")
        candidates.append(folder / "config.yml")
    return candidates


def _discover_config_path(explicit: Optional[Path] = None) -> Optional[Path]:
    """Return the config path to use for CLI execution.

    Resolution order:
    1. Explicit ``--config`` path.
    2. ``config.yaml`` / ``config.yml`` in the current working directory.
    3. ``config.yaml`` / ``config.yml`` in parent directories of the current
       working directory.
    4. ``config.yaml`` / ``config.yml`` next to this script.
    5. ``config.yaml`` / ``config.yml`` in parent directories of this script.

    The parent-directory search makes the CLI robust when VS Code, pytest, or a
    shell launches the script from a subdirectory instead of the repository root.
    """
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


def _format_loaded_config_hint(config_path: Optional[Path], section: Dict[str, Any]) -> str:
    """Return a concise CLI hint describing the loaded configuration."""
    if config_path is None:
        return "No config file found; using CLI arguments only."
    keys = sorted(section.keys()) if section else []
    key_text = ", ".join(keys) if keys else "no converter section keys"
    return f"Using config: {config_path} ({key_text})"

def _cfg_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return the converter config section.

    The new section name is preferred, but the old name is retained so existing
    configs keep working during the refactor.
    """
    section = cfg.get("convert_wmdr10_json_to_wmdr2_json")
    if isinstance(section, dict):
        return section
    section = cfg.get("convert_wmdr10_json_to_wmdr2_geojson")
    return section if isinstance(section, dict) else {}


def _normalize_discovery_policy(section: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    policy = copy.deepcopy(DEFAULT_DISCOVERY_POLICY)
    raw = section.get("discovery")
    if not isinstance(raw, dict):
        return policy

    for entity in ("facility", "observation", "deployment"):
        entity_cfg = raw.get(entity)
        if not isinstance(entity_cfg, dict):
            continue
        for bucket in ("keywords", "links"):
            values = entity_cfg.get(bucket)
            if isinstance(values, list):
                policy[entity][bucket] = [
                    str(v).strip()
                    for v in values
                    if isinstance(v, str) and str(v).strip()
                ]
    return policy


def _iter_json_files(root: Path, *, pattern: str = DEFAULT_PATTERN, recursive: bool = True) -> List[Path]:
    """Return JSON files under ``root``.

    Output files in the target directory should not be fed back as input. The
    caller is responsible for choosing separate source/target directories or a
    sufficiently restrictive input pattern.
    """
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
    """Detect whether an input file is a full record or a part file."""
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
        if "facility" in payload or "observations" in payload or "header" in payload:
            return "full"
        if "observedVariable" in payload or "observedProperty" in payload or "resultTime" in payload:
            return "observations"
        if "sourceOfObservation" in payload or "manufacturer" in payload or "serialNumber" in payload:
            return "deployments"
        if "fileDateTime" in payload and "recordOwner" in payload:
            return "header"
        if "identifier" in payload or "name" in payload or "geospatialLocation" in payload:
            return "facility"

    if isinstance(payload, list):
        first = next((x for x in payload if isinstance(x, dict)), None)
        if not first:
            return "unknown"
        if "observedVariable" in first or "observedProperty" in first or "resultTime" in first:
            return "observations"
        if "sourceOfObservation" in first or "manufacturer" in first or "serialNumber" in first:
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


def _clean_none(obj: Any) -> Any:
    """Recursively remove empty values while preserving list nulls.

    JSON null values are removed from object properties, but retained inside
    arrays because several WMDR2 compact structures use aligned arrays where
    ``null`` is a meaningful placeholder for a missing value at that index.
    """
    if isinstance(obj, dict):
        cleaned = {k: _clean_none(v) for k, v in obj.items()}
        return {k: v for k, v in cleaned.items() if v not in (None, "", [], {})}
    if isinstance(obj, list):
        cleaned = [_clean_none(v) for v in obj]
        return [v for v in cleaned if v not in ("", [], {})]
    return obj


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "record"


def _sanitize_id(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^A-Za-z0-9._:/#-]+", "-", text)
    return text.strip("-") or "record"


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


def _simplify_unknown_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    normalized = re.sub(r"[()]", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return None
    tokens = [token.lower() for token in normalized.split()]
    if tokens and all(token in {"unknown", "none", "null", "nil"} for token in tokens):
        return "unknown"
    return text


def _normalize_code_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = _simplify_unknown_text(value)
    if not isinstance(text, str):
        return text
    text = text.strip()
    if not text:
        return None
    match = re.fullmatch(r"\(([^()]+)\)", text)
    if match:
        text = match.group(1).strip()
    if _is_unknown_token(text):
        return "unknown"
    return text


def _normalize_display_text(value: Any) -> Optional[str]:
    """Normalize human-facing text while retaining meaningful values."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    text = value.strip()
    if not text:
        return None
    normalized = re.sub(r"\((unknown|null|none|nil)\)", "unknown", text, flags=re.IGNORECASE)
    normalized = re.sub(r"\bunknown\s*/\s*unknown\b", "unknown", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bunknown\s+unknown\b", "unknown", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return "unknown" if _is_unknown_token(normalized) else normalized


def _drop_source_metadata(obj: Any) -> Any:
    """Remove XML/GML source bookkeeping that should not survive in WMDR2 JSON."""
    metadata_keys = {"@gml:id", "gml:id", "@id", "@xmlns", "xmlns", "schemaLocation"}
    if isinstance(obj, dict):
        return {
            key: _drop_source_metadata(value)
            for key, value in obj.items()
            if key not in metadata_keys
        }
    if isinstance(obj, list):
        return [_drop_source_metadata(item) for item in obj]
    return obj


def _normalize_description_value(value: Any) -> Any:
    """Normalize descriptions to plain human-readable text when possible.

    WMDR1 descriptions may arrive as history objects with XML/GML bookkeeping
    fields and validity timestamps. For the WMDR2 core record, the ``description``
    property itself should be a simple text field. Temporal provenance of the
    description can be modelled separately later if needed, but it should not
    make ``description`` an object.
    """
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


def _humanize_identifier(value: Any) -> Optional[str]:
    text = _last_segment(value) if isinstance(value, str) else None
    if not text:
        return None
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if text else None


def _compact_display_values(values: Iterable[Any]) -> List[str]:
    """Return unique display values, dropping ``unknown`` when real values exist."""
    known: List[str] = []
    unknown_seen = False
    seen: set[str] = set()
    for raw in values:
        normalized = _normalize_display_text(raw)
        if not normalized:
            continue
        key = normalized.lower()
        if key == "unknown":
            unknown_seen = True
            continue
        if key in seen:
            continue
        seen.add(key)
        known.append(normalized)
    if known:
        return known
    return ["unknown"] if unknown_seen else []


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


def _uri_parent(value: str) -> str:
    raw = value.rstrip("/#")
    if "/" in raw:
        return raw.rsplit("/", 1)[0]
    if "#" in raw:
        return raw.rsplit("#", 1)[0]
    return raw


def _normalize_open_end(value: Any) -> str:
    if value in (None, "", "None"):
        return ".."
    text = str(value).strip()
    if not text:
        return ".."
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}Z", text):
        return text[:-1]
    return text


def _normalize_time_value(value: Any) -> Optional[str]:
    if value in (None, "", "None"):
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}Z", text):
        return text[:-1]
    return text


def _normalize_record_datetime(value: Any) -> Optional[str]:
    """Normalize record created/updated timestamps to an ISO-like UTC string."""
    if value in (None, "", "None"):
        return None
    text = str(value).strip()
    if not text:
        return None

    # Filename/source-name pattern used by legacy examples, e.g.
    # 20200304_0-20000-0-06494.
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
    """Derive OGC-record created/updated fields from header metadata."""
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
    s = _normalize_time_value(start)
    e = _normalize_open_end(end)
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


def _summarize_intervals(intervals: Sequence[Sequence[Any]]) -> Optional[Dict[str, Any]]:
    starts = [str(item[0]) for item in intervals if len(item) >= 1 and isinstance(item[0], str) and item[0] != ".."]
    ends = [str(item[1]) for item in intervals if len(item) >= 2 and isinstance(item[1], str) and item[1] != ".."]
    has_open = any(len(item) >= 2 and item[1] == ".." for item in intervals)
    if not starts and not ends and not has_open:
        return None
    return {"interval": [min(starts) if starts else "..", ".." if has_open else (max(ends) if ends else "..")]}


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


def _parse_quantity(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, (int, float)):
        return {"value": value}
    if isinstance(value, dict):
        raw = value.get("#text") or value.get("value")
        if raw in (None, ""):
            return None
        try:
            num: Any = int(raw) if str(raw).isdigit() else float(raw)
        except Exception:
            num = raw
        out: Dict[str, Any] = {"value": num}
        if value.get("@uom"):
            out["uom"] = value.get("@uom")
        elif value.get("uom"):
            out["uom"] = value.get("uom")
        return out
    return None


def _parse_pos_lon_lat_z(raw: Any) -> Optional[List[Any]]:
    """Parse a WMDR position into GeoJSON coordinate order [lon, lat, z?]."""
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

    parts = raw.replace(",", " ").split()
    nums: List[float] = []
    for item in parts:
        try:
            nums.append(float(item))
        except Exception:
            continue
    if len(nums) < 2:
        return None

    # WMDR/GML pos is usually lat lon [z]. GeoJSON is lon lat [z].
    lat, lon = nums[0], nums[1]
    coords: List[Any] = [lon, lat]
    if len(nums) >= 3:
        z = nums[2]
        coords.append(int(round(z)) if abs(z - round(z)) < 1e-9 else z)
    return coords


def _point_from_pos(raw: Any) -> Optional[Dict[str, Any]]:
    coords = _parse_pos_lon_lat_z(raw)
    if coords is None:
        return None
    return {"type": "Point", "coordinates": coords}


def _uniq_dicts(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        payload = json.dumps(item, sort_keys=True, ensure_ascii=False)
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
        key = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Code-list labels and discovery helpers
# ---------------------------------------------------------------------------


def _extract_code_list_ref(value: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(uri, domain, code)`` for a WMO code-list URI or code-like value."""
    if not isinstance(value, str):
        return None, None, None
    text = value.strip().strip("<>")
    if not text:
        return None, None, None
    if text.startswith(("http://", "https://")):
        parts = [part for part in text.rstrip("/#").split("/") if part]
        if len(parts) < 2:
            return text, None, None
        domain = parts[-2].lstrip("_")
        code = parts[-1].lstrip("_")
        return text, domain, code
    return None, None, text.lstrip("_")




def _observed_domain_from_observed_variable(value: Any) -> Optional[str]:
    """Derive the compact WMDR observed-domain code from an observed variable.

    Legacy WMDR1 XML encodes the observed domain in the observed-variable
    code-list name, for example::

        http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/179

    WMDR2 exposes this as the compact domain code::

        atmosphere
    """
    _, domain, _ = _extract_code_list_ref(value)
    if not domain or not domain.startswith("ObservedVariable"):
        return None
    domain_name = domain.removeprefix("ObservedVariable").strip()
    if not domain_name:
        return None
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", domain_name).lower()


def _deployment_source_identifier(raw: Dict[str, Any], *, index: int, facility_id: str) -> str:
    """Return the stable source identifier used for a deployment."""
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
    """Return the WMDR2 deployment id used by observations for references."""
    return f"deployment:{_deployment_source_identifier(raw, index=index, facility_id=facility_id)}"


def _compact_wmdr_code_value(value: Any) -> Any:
    """Return a compact WMDR code-list value.

    Full WMO code-list URLs are an interchange convenience in WMDR1. WMDR2 core
    keeps only the actual code value and relies on schemas/validators to know
    the applicable code list from the property context.

    Examples:
        http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006 -> 12006
        https://codes.wmo.int/wmdr/Domain/atmosphere -> "atmosphere"
    """
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text.startswith(("http://codes.wmo.int/wmdr/", "https://codes.wmo.int/wmdr/")):
        return value

    code = text.rstrip("/#").rsplit("/", 1)[-1].lstrip("_")
    if not code:
        return value
    if re.fullmatch(r"[+-]?\d+", code):
        try:
            return int(code)
        except Exception:
            return code
    return code


def _finalize_wmdr2_value(value: Any, *, key: Optional[str] = None) -> Any:
    """Finalize WMDR2 output values.

    This pass is intentionally applied only at the final output boundary. It
    keeps converter internals close to the legacy WMDR1 input while enforcing
    the current WMDR2 core output conventions:

    - compact WMO code-list URLs to their actual code values;
    - preserve real links such as ``href`` values;
    - collapse unknown-only interval structures to JSON null;
    - collapse singleton unknown schedule lists to JSON null.
    """
    if isinstance(value, dict):
        if set(value.keys()) == {"interval"}:
            interval = value.get("interval")
            if interval == ["..", ".."] or interval == "unknown":
                return None

        out: Dict[str, Any] = {}
        for child_key, child_value in value.items():
            if child_key in {"href", "url"}:
                out[child_key] = child_value
            else:
                out[child_key] = _finalize_wmdr2_value(child_value, key=child_key)
        return out

    if isinstance(value, list):
        if len(value) == 1:
            first = value[0]
            if isinstance(first, dict) and set(first.keys()) == {"interval"} and first.get("interval") == "unknown":
                return None
        return [_finalize_wmdr2_value(item, key=key) for item in value]

    if isinstance(value, str) and key not in {"href", "url"}:
        return _compact_wmdr_code_value(value)
    return value


def _is_substantive_instrument_value(value: Any) -> bool:
    """Return true when a manufacturer/model value identifies equipment."""
    if value in (None, "", [], {}):
        return False
    if isinstance(value, str) and _is_unknown_token(value):
        return False
    return True


def _instrument_source_values(raw: Dict[str, Any]) -> Tuple[Any, Any]:
    """Return manufacturer and model values from a deployment."""
    return raw.get("manufacturer"), raw.get("model")


def _deployment_serial_number(raw: Dict[str, Any]) -> Any:
    """Return the deployment-level serial-number value, if available."""
    return raw.get("serialNumber")


def _deployment_has_instrument(raw: Dict[str, Any]) -> bool:
    manufacturer, model = _instrument_source_values(raw)
    return any(_is_substantive_instrument_value(value) for value in (manufacturer, model))


def _instrument_record_id(raw: Dict[str, Any], *, facility_id: str) -> Optional[str]:
    """Return the stable WMDR2 instrument id referenced by deployments.

    Instruments are reusable catalog entries. Serial-number histories remain on
    deployments because serial numbers can change with deployment context and
    should not make two otherwise identical manufacturer/model catalog entries
    become separate instruments.
    """
    if not _deployment_has_instrument(raw):
        return None

    manufacturer, model = _instrument_source_values(raw)
    seed_parts = [
        facility_id,
        str(manufacturer or ""),
        str(model or ""),
    ]
    digest = hashlib.sha1("|".join(seed_parts).encode("utf-8")).hexdigest()[:12]
    return f"instrument:{digest}"


def _instrument_refs_for_deployment(raw: Dict[str, Any], *, facility_id: str) -> List[str]:
    instrument_id = _instrument_record_id(raw, facility_id=facility_id)
    return [instrument_id] if instrument_id else []


def _deployment_serial_numbers(raw: Dict[str, Any]) -> Optional[Dict[str, List[Any]]]:
    """Return deployment-level serial-number history.

    The WMDR2 instrument object is a reusable catalog entry. The serial number
    remains on the deployment, represented as aligned arrays so later changes
    can be added without changing the structure.
    """
    serial_number = _deployment_serial_number(raw)
    if not _is_substantive_instrument_value(serial_number):
        return None

    start, _ = _extract_interval(raw)
    begin = _normalize_time_value(start) or ".."
    return {
        "serialNumber": [serial_number],
        "datetimes": [begin],
    }


def _normalize_instrument(raw: Dict[str, Any], *, facility_id: str) -> Optional[Dict[str, Any]]:
    """Normalize deployment manufacturer/model data into an instrument catalog entry."""
    instrument_id = _instrument_record_id(raw, facility_id=facility_id)
    if not instrument_id:
        return None

    manufacturer, model = _instrument_source_values(raw)
    payload: Dict[str, Any] = {
        "id": instrument_id,
        "manufacturer": manufacturer if _is_substantive_instrument_value(manufacturer) else None,
        "model": model if _is_substantive_instrument_value(model) else None,
    }
    return _clean_none(payload)


def _normalize_instruments(deployments: Sequence[Dict[str, Any]], *, facility_id: str) -> List[Dict[str, Any]]:
    """Return de-duplicated reusable instruments derived from deployment equipment fields."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for dep in deployments:
        if not isinstance(dep, dict):
            continue
        instrument = _normalize_instrument(dep, facility_id=facility_id)
        if not instrument:
            continue
        instrument_id = instrument.get("id")
        if not isinstance(instrument_id, str):
            continue
        if instrument_id not in by_id:
            by_id[instrument_id] = instrument

    return list(by_id.values())


def _load_code_list_labels(section: Dict[str, Any], *, base_dir: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """Load optional code-list labels used for human-readable titles."""
    labels: Dict[str, Dict[str, str]] = {}
    raw = section.get("codeListLabels") or section.get("code_list_labels") or {}
    if not isinstance(raw, dict):
        return labels

    inline = raw.get("inline")
    if isinstance(inline, dict):
        for domain, mapping in inline.items():
            if not isinstance(mapping, dict):
                continue
            domain_key = str(domain).strip()
            if not domain_key:
                continue
            target = labels.setdefault(domain_key, {})
            for code, label in mapping.items():
                code_key = str(code).strip().lstrip("_")
                label_text = str(label).strip() if label is not None else ""
                if code_key and label_text:
                    target[code_key] = label_text

    files = raw.get("files", [])
    if isinstance(files, (str, Path)):
        files = [files]
    if not isinstance(files, list):
        return labels

    for item in files:
        path_text: Optional[str] = None
        if isinstance(item, dict):
            raw_path = item.get("path")
            if isinstance(raw_path, str):
                path_text = raw_path
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


def _lookup_code_list_label(domain: Optional[str], code: Optional[str]) -> Optional[str]:
    if not domain or not code:
        return None
    return CODE_LIST_LABELS.get(domain, {}).get(code.lstrip("_"))


def _display_domain_name(domain: Optional[str]) -> Optional[str]:
    if not domain:
        return None
    text = re.sub(r"^ObservedVariable", "", domain)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or domain


def _format_observation_title(value: Any, geometry_type: Any = None) -> Optional[str]:
    """Build an observation title from compact domain, geometry and variable info.

    Target form::

        domain: atmosphere; geometry: point; variable: 12006 Horizontal wind speed ...
    """
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


def _about_link(href: str, *, title: Optional[str] = None, media_type: str = "text/html") -> Dict[str, Any]:
    link: Dict[str, Any] = {"href": href, "rel": "about", "type": media_type}
    if title:
        link["title"] = title
    return link


def _canonical_type_link(href: str) -> Dict[str, Any]:
    return {"href": href, "rel": "type"}


def _collect_discovery_values(entity_type: str, source: Dict[str, Any], bucket: str) -> List[Any]:
    values: List[Any] = []
    for key in DISCOVERY_POLICY.get(entity_type, {}).get(bucket, []):
        for item in _as_list(source.get(key)):
            if isinstance(item, dict):
                values.extend(_extract_scalar_values(item))
            else:
                values.append(item)
    return values


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
# Contact normalization
# ---------------------------------------------------------------------------


def _normalize_role(value: Any) -> Optional[str]:
    """Normalize a contact role to a specific role code.

    Generic ISO code-list references such as ``gmxCodelists.xml#CI_RoleCode``
    identify the vocabulary, not the role assignment. Those are deliberately
    dropped until the source XML/JSON provides a specific role code, e.g.
    ``owner``, ``pointOfContact`` or ``custodian``.
    """
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

    info: Dict[str, Any] = _as_dict(payload.get("contactInfo"))
    phone_obj: Dict[str, Any] = _as_dict(info.get("phone"))
    phones = [
        {"value": voice.strip()}
        for voice in _as_list(phone_obj.get("voice"))
        if isinstance(voice, str) and voice.strip()
    ]
    if phones:
        contact["phones"] = phones

    address_obj: Dict[str, Any] = _as_dict(info.get("address"))
    emails = [
        {"value": email.strip()}
        for email in _as_list(address_obj.get("electronicMailAddress"))
        if isinstance(email, str) and "@" in email
    ]
    if emails:
        contact["emails"] = emails

    address: Dict[str, Any] = {}
    delivery_points = [
        dp for dp in _as_list(address_obj.get("deliveryPoint"))
        if isinstance(dp, str) and dp.strip()
    ]
    if delivery_points:
        address["deliveryPoint"] = delivery_points
    for src_key in ("city", "administrativeArea", "postalCode", "country"):
        value = address_obj.get(src_key)
        if isinstance(value, str) and value.strip():
            address[src_key] = value.strip()
    if address:
        contact["addresses"] = [address]

    links = []
    online: Dict[str, Any] = _as_dict(info.get("onlineResource"))
    href = online.get("url") or online.get("href")
    if isinstance(href, str) and href.strip():
        links.append(_about_link(href.strip()))
    if links:
        contact["links"] = links

    instructions = info.get("contactInstructions")
    if isinstance(instructions, str) and instructions.strip():
        contact["contactInstructions"] = instructions.strip()

    roles = _normalize_roles(_first_non_empty(payload.get("role"), raw.get("role")))
    if roles:
        contact["roles"] = roles

    if not any(key in contact for key in ("name", "organization", "emails", "phones", "addresses", "links")):
        return None, None

    valid_start, valid_end = _extract_interval(raw)
    extension = dict(contact)
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
# WMDR-specific normalization
# ---------------------------------------------------------------------------


def _normalize_reporting_status(value: Any) -> List[Dict[str, Any]]:
    statuses: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, str):
            normalized = _normalize_code_value(item)
            if isinstance(normalized, str) and not _is_unknown_token(normalized):
                statuses.append({"value": normalized})
            continue
        if not isinstance(item, dict):
            continue
        status_val = _first_non_empty(item.get("reportingStatus"), item.get("instrumentOperatingStatus"), item.get("value"))
        record: Dict[str, Any] = {}
        if isinstance(status_val, str):
            record["value"] = _normalize_code_value(status_val)
        if isinstance(item.get("@gml:id"), str):
            record["id"] = item.get("@gml:id")
        elif isinstance(item.get("id"), str):
            record["id"] = item.get("id")
        valid_time = _time_interval(item.get("beginPosition"), item.get("endPosition"))
        if valid_time:
            record["time"] = valid_time
        if record:
            statuses.append(record)
    return _uniq_dicts(_clean_none(statuses))


def _normalize_program_affiliation(value: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, str):
            normalized = _normalize_code_value(item)
            if isinstance(normalized, str) and not _is_unknown_token(normalized):
                out.append({"programAffiliation": [normalized]})
            continue
        if not isinstance(item, dict):
            continue

        record: Dict[str, Any] = {}
        affiliations = [
            _normalize_code_value(raw)
            for raw in _as_list(item.get("programAffiliation") or item.get("href") or item.get("value"))
            if isinstance(raw, str) and raw.strip() and not _is_unknown_token(raw)
        ]
        if affiliations:
            record["programAffiliation"] = affiliations

        psfi = item.get("programSpecificFacilityId")
        if isinstance(psfi, str) and psfi.strip():
            record["programSpecificFacilityId"] = psfi.strip()

        valid_time = _time_interval(item.get("beginPosition"), item.get("endPosition"))
        if valid_time:
            record["time"] = valid_time

        statuses = _normalize_reporting_status(item.get("reportingStatus"))
        if statuses:
            record["reportingStatus"] = statuses

        if record:
            out.append(record)
    return _uniq_dicts(_clean_none(out))


def _normalize_reporting_status_timeline(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize reporting-status history into aligned value/begin-date arrays.

    Each reporting status becomes valid at the datetime in the same position
    and remains valid until the next datetime entry. End dates from WMDR1 are
    therefore not carried into WMDR2.
    """
    rows: List[Tuple[str, str]] = []

    for item in _as_list(value):
        if isinstance(item, str):
            status = _normalize_code_value(item)
            if isinstance(status, str) and status and not _is_unknown_token(status):
                rows.append(("..", status))
            continue

        if not isinstance(item, dict):
            continue

        status = _normalize_code_value(
            _first_non_empty(
                item.get("reportingStatus"),
                item.get("instrumentOperatingStatus"),
                item.get("value"),
                item.get("href"),
            )
        )
        if not isinstance(status, str) or not status or _is_unknown_token(status):
            continue

        rows.append((_temporal_begin_datetime(item), status))

    if not rows:
        return None

    # Sort known begin dates chronologically. Unknown begin dates are retained
    # after known dates so that explicit temporal history appears first.
    rows = sorted(rows, key=lambda row: (row[0] == "..", row[0]))

    statuses: List[str] = []
    datetimes: List[str] = []
    seen: set[Tuple[str, str]] = set()
    for begin, status in rows:
        marker = (begin, status)
        if marker in seen:
            continue
        seen.add(marker)
        datetimes.append(begin)
        statuses.append(status)

    return _clean_none({"reportingStatus": statuses, "datetimes": datetimes})


def _program_affiliation_values(item: Any) -> List[str]:
    """Return normalized program-affiliation URI values for one source item."""
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


def _reporting_status_events(value: Any, *, fallback_datetime: str = "..") -> List[Tuple[str, str]]:
    """Return ``(begin_datetime, reporting_status)`` events.

    These rows are used to build aligned WMDR2 arrays. Repeated reporting
    statuses are allowed when they occur at different begin datetimes.
    """
    rows: List[Tuple[str, str]] = []

    for item in _as_list(value):
        if isinstance(item, str):
            status = _normalize_code_value(item)
            if isinstance(status, str) and status and not _is_unknown_token(status):
                rows.append((fallback_datetime, status))
            continue

        if not isinstance(item, dict):
            continue

        status = _normalize_code_value(
            _first_non_empty(
                item.get("reportingStatus"),
                item.get("instrumentOperatingStatus"),
                item.get("value"),
                item.get("href"),
            )
        )
        if not isinstance(status, str) or not status or _is_unknown_token(status):
            continue

        rows.append((_temporal_begin_datetime(item) or fallback_datetime, status))

    return rows


def _normalize_temporal_program_affiliation(value: Any) -> Optional[Dict[str, List[Any]]]:
    """Normalize temporal program affiliations as aligned arrays.

    Each array index represents one temporal program-affiliation state:
    ``programAffiliation[i]``, ``reportingStatus[i]`` and ``datetimes[i]``
    belong together. A history of reporting status for the same program is
    represented by repeating the same ``programAffiliation`` value at another
    index with a different reporting status and datetime.

    Example:
        {
          "programAffiliation": [
            "http://codes.wmo.int/wmdr/ProgramAffiliation/GOSGeneral",
            "http://codes.wmo.int/wmdr/ProgramAffiliation/GOSGeneral"
          ],
          "reportingStatus": [
            "http://codes.wmo.int/wmdr/ReportingStatus/operational",
            "http://codes.wmo.int/wmdr/ReportingStatus/closed"
          ],
          "datetimes": [
            "2000-08-17T00:00:00Z",
            "2025-05-28T00:00:00Z"
          ]
        }
    """
    rows: List[Tuple[str, str, Optional[str]]] = []

    for item in _as_list(value):
        affiliations = _program_affiliation_values(item)
        if not affiliations:
            continue

        item_begin = _temporal_begin_datetime(item) if isinstance(item, dict) else ".."
        status_events = (
            _reporting_status_events(item.get("reportingStatus"), fallback_datetime=item_begin)
            if isinstance(item, dict)
            else []
        )

        for affiliation in affiliations:
            if status_events:
                for begin, status in status_events:
                    rows.append((begin, affiliation, status))
            else:
                rows.append((item_begin, affiliation, None))

    if not rows:
        return None

    # Sort known begin dates chronologically. Unknown begin dates are retained
    # after known dates. Do not group by program: repeated program values are
    # intentional and represent reporting-status history for that program.
    rows = sorted(rows, key=lambda row: (row[0] == "..", row[0], row[1], row[2] or ""))

    program_affiliation: List[str] = []
    reporting_status: List[Any] = []
    datetimes: List[str] = []
    seen: set[Tuple[str, str, Optional[str]]] = set()
    has_reporting_status = any(status is not None for _, _, status in rows)

    for begin, affiliation, status in rows:
        marker = (begin, affiliation, status)
        if marker in seen:
            continue
        seen.add(marker)
        program_affiliation.append(affiliation)
        datetimes.append(begin)
        if has_reporting_status:
            # Preserve array alignment when a rare entry lacks status.
            # ``null`` means no reporting status was present in the source for
            # this program-affiliation event.
            reporting_status.append(status)

    out: Dict[str, List[Any]] = {
        "programAffiliation": program_affiliation,
        "datetimes": datetimes,
    }
    if has_reporting_status:
        out["reportingStatus"] = reporting_status

    return out


def _normalize_simple_timed_value(value: Any, *, value_key: str) -> Optional[Dict[str, Any]]:
    if isinstance(value, str):
        actual = _normalize_code_value(value)
        if actual in (None, ""):
            return None
        return {"value": actual}
    if not isinstance(value, dict):
        return None

    actual = _normalize_code_value(_first_non_empty(value.get(value_key), value.get("value"), value.get("href")))
    if actual in (None, ""):
        return None

    out: Dict[str, Any] = {"value": actual}
    if isinstance(value.get("@gml:id"), str):
        out["id"] = value.get("@gml:id")
    elif isinstance(value.get("id"), str):
        out["id"] = value.get("id")

    valid_time = _time_interval(value.get("beginPosition"), value.get("endPosition"))
    if valid_time:
        out["time"] = valid_time

    for extra_key in (
        "surfaceCoverClassification",
        "localTopography",
        "relativeElevation",
        "topographicContext",
        "altitudeOrDepth",
    ):
        if _non_empty(value.get(extra_key)):
            out[extra_key] = value.get(extra_key)
    return _clean_none(out)


def _normalize_temporal_geometry(current: Any, history: Any = None) -> List[Dict[str, Any]]:
    """Collect geospatial history entries in chronological order.

    WMDR2 temporal geometry uses aligned ``coordinates`` and begin-date
    ``datetimes`` arrays. Each coordinate becomes valid at the datetime in the
    same position and remains valid until the next entry. End dates from WMDR1
    are not carried into WMDR2.
    """
    entries: List[Tuple[Optional[str], str, Dict[str, Any]]] = []

    def add_entry(item: Any) -> None:
        if isinstance(item, str):
            coords = _parse_pos_lon_lat_z(item)
            if coords is None:
                return
            entries.append((None, json.dumps(coords, sort_keys=True), {"coordinates": coords, "datetime": ".."}))
            return

        if not isinstance(item, dict):
            return

        coords = _parse_pos_lon_lat_z(
            item.get("geometry")
            or item.get("geoLocation")
            or item.get("geospatialLocation")
            or item.get("pos")
            or item
        )
        if coords is None:
            return

        begin = _temporal_begin_datetime(item)
        entries.append(
            (
                None if begin == ".." else begin,
                json.dumps(coords, sort_keys=True),
                {"coordinates": coords, "datetime": begin},
            )
        )

    for item in _as_list(current):
        add_entry(item)
    for item in _as_list(history):
        add_entry(item)

    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for _, _, entry in sorted(entries, key=lambda item: (item[0] is None, item[0] or "", item[1])):
        marker = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(entry)
    return deduped


def _temporal_geometry_extension(entries: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a MovingPoint-style temporal geometry if a history exists."""
    if len(entries) <= 1:
        return None

    coordinates: List[Any] = []
    datetimes: List[str] = []
    for entry in entries:
        if "coordinates" not in entry:
            continue
        coordinates.append(entry["coordinates"])
        dt = entry.get("datetime")
        datetimes.append(dt if isinstance(dt, str) and dt else "..")

    if len(coordinates) <= 1:
        return None

    return {"type": "MovingPoint", "coordinates": coordinates, "datetimes": datetimes}


def _normalize_temporal_observing_schedule(value: Any) -> List[Dict[str, Any]]:
    """Normalize temporal observing schedules.

    Observing-schedule entries are not referenceable WMDR2 entities, so source
    XML ids are deliberately not carried over. A generic ``interval`` fallback
    is retained for now because observing schedules may otherwise consist only
    of time-window fields or a schedule placeholder.
    """
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        record: Dict[str, Any] = {}

        interval = _first_non_empty(
            item.get("interval"),
            item.get("reportingInterval"),
            item.get("temporalResolution"),
            item.get("period"),
        )
        if isinstance(interval, dict):
            interval = _first_non_empty(interval.get("value"), interval.get("#text"), interval.get("href"))
        record["interval"] = str(interval).strip() if _non_empty(interval) else "unknown"

        for key in (
            "startMonth",
            "endMonth",
            "startWeekday",
            "endWeekday",
            "startHour",
            "endHour",
            "startMinute",
            "endMinute",
            "diurnalBaseTime",
        ):
            if _non_empty(item.get(key)):
                record[key] = item[key]

        if record:
            out.append(record)
    return _uniq_dicts(_clean_none(out))


def _normalize_temporal_reporting_schedule(value: Any) -> List[Dict[str, Any]]:
    """Normalize temporal reporting schedules.

    This mirrors observing schedules but retains explicit reporting fields when
    they contain meaningful content. Current tests expect ``interval: unknown``
    for an id-only reporting object.
    """
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        record: Dict[str, Any] = {}
        source_id = _first_non_empty(item.get("@gml:id"), item.get("@id"), item.get("id"))
        if isinstance(source_id, str) and source_id.strip():
            record["id"] = source_id.strip()

        interval = _first_non_empty(
            item.get("reportingInterval"),
            item.get("interval"),
            item.get("temporalResolution"),
            item.get("period"),
        )
        if isinstance(interval, dict):
            interval = _first_non_empty(interval.get("value"), interval.get("#text"), interval.get("href"))
        record["interval"] = str(interval).strip() if _non_empty(interval) else "unknown"

        reporting = item.get("reporting")
        if isinstance(reporting, dict):
            parsed_reporting = {
                key: (_parse_bool(val) if _parse_bool(val) is not None else val)
                for key, val in reporting.items()
                if _non_empty(val)
            }
            # For the WMDR2 observation-level reporting object, even a single
            # internationalExchange flag is substantive and must be retained.
            if parsed_reporting:
                record["reporting"] = parsed_reporting

        if record:
            out.append(record)
    return _uniq_dicts(_clean_none(out))


def _derive_observation_time(observation: Dict[str, Any], deployments: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    intervals: List[List[Any]] = []
    for dep in deployments:
        for candidate in (dep, _as_dict(dep.get("temporalExtent")), _as_dict(dep.get("time"))):
            start, end = _extract_interval(candidate)
            interval = _time_interval(start, end)
            if interval and isinstance(interval.get("interval"), list):
                intervals.append(interval["interval"])
                break
    if intervals:
        return _summarize_intervals(intervals)
    return _time_interval(observation.get("beginPosition"), observation.get("endPosition"))


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


def _observation_description(observation: Dict[str, Any], deployments: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Build a concise human-readable observation description."""
    parts: List[str] = []
    _, _, code = _extract_code_list_ref(observation.get("observedProperty"))
    if code:
        parts.append(f"Observed variable {code}")
    elif _non_empty(observation.get("observedProperty")):
        parts.append(f"Observed variable {_last_segment(observation.get('observedProperty')) or observation.get('observedProperty')}")

    geom_type = _normalize_display_text(observation.get("type") or observation.get("geometryType"))
    if geom_type:
        parts.append(f"geometry type {geom_type}")

    procedure_candidates: List[Any] = []
    for dep in deployments:
        procedure_candidates.extend(
            [
                dep.get("observingMethod"),
                dep.get("procedure"),
                dep.get("manufacturer"),
                dep.get("model"),
            ]
        )
    procedures = _compact_display_values(_humanize_identifier(item) or item for item in procedure_candidates)
    if procedures:
        parts.append(f"deployment procedure {' / '.join(procedures)}")

    return "; ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def _external_id(value: Any, scheme: str) -> Optional[Dict[str, str]]:
    if not isinstance(value, str) or not value.strip():
        return None
    return {"scheme": scheme, "value": value.strip()}


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


def _extract_links(source: Dict[str, Any], entity_type: str) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []

    for key in DISCOVERY_POLICY.get(entity_type, {}).get("links", []):
        for item in _as_list(source.get(key)):
            href: Optional[str] = None
            title: Optional[str] = None
            media_type: str = "text/html"

            if isinstance(item, str):
                candidate_href = item.strip()
                if candidate_href:
                    href = candidate_href

            elif isinstance(item, dict):
                raw_href = _first_non_empty(
                    item.get("url"),
                    item.get("href"),
                    item.get("linkage"),
                    item.get("value"),
                )
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


def _temporal_begin_datetime(item: Any) -> str:
    """Return the begin datetime for a temporal facility value.

    WMDR2 temporal facility descriptors use two aligned arrays, for example
    ``climateZone`` and ``datetimes``. Each datetime is interpreted as the
    begin date for the corresponding value; the value remains valid until the
    next datetime entry. Unknown begin dates are represented as ``".."``.
    """
    if isinstance(item, dict):
        return _normalize_time_value(
            _first_non_empty(
                item.get("beginPosition"),
                item.get("begin"),
                item.get("start"),
                item.get("date"),
            )
        ) or ".."
    return ".."


def _normalize_temporal_territory(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize WMDR1 territory history into aligned value/begin-date arrays."""
    territories: List[Any] = []
    datetimes: List[str] = []

    for item in _as_list(value):
        if isinstance(item, str):
            territory = _normalize_code_value(item)
            if territory and not _is_unknown_token(territory):
                territories.append(territory)
                datetimes.append("..")
            continue

        if not isinstance(item, dict):
            continue

        territory = _normalize_code_value(
            _first_non_empty(
                item.get("territoryName"),
                item.get("territory"),
                item.get("value"),
                item.get("href"),
            )
        )
        if not territory or _is_unknown_token(territory):
            continue

        territories.append(territory)
        datetimes.append(_temporal_begin_datetime(item))

    if not territories:
        return None

    # Preserve order and alignment. Each datetime is the begin date for the
    # territory at the same index; the value holds until the next entry.
    return _clean_none({"territory": territories, "datetimes": datetimes})


def _normalize_temporal_facility_values(
    value: Any,
    *,
    output_key: str,
    value_keys: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Normalize time-varying facility descriptors into value/begin-date arrays.

    WMDR1 JSON can contain structures such as ``climateZone`` or
    ``surfaceCover`` with ``beginPosition`` / ``endPosition`` fields. WMDR2
    exposes these as explicit temporal facility properties, for example::

        {
          "climateZone": ["..."],
          "datetimes": ["2016-04-28T00:00:00Z"]
        }

    Each datetime is interpreted as the begin date for the value at the same
    index. The value remains valid until the next datetime entry. This avoids
    carrying WMDR1/XML validity fields directly inside the value.
    """
    values: List[Any] = []
    datetimes: List[str] = []

    for item in _as_list(value):
        if isinstance(item, str):
            normalized = _normalize_code_value(item)
            if normalized and not _is_unknown_token(normalized):
                values.append(normalized)
                datetimes.append("..")
            continue

        if not isinstance(item, dict):
            continue

        cleaned = _drop_source_metadata(item)
        if not isinstance(cleaned, dict):
            continue

        raw_value = _first_non_empty(*(cleaned.get(key) for key in value_keys))
        normalized_value = _normalize_code_value(raw_value)
        if not normalized_value or _is_unknown_token(normalized_value):
            continue

        values.append(normalized_value)
        datetimes.append(_temporal_begin_datetime(item))

    if not values:
        return None

    return _clean_none({output_key: values, "datetimes": datetimes})


def _normalize_temporal_climate_zone(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize climate-zone history into the WMDR2 temporal structure."""
    return _normalize_temporal_facility_values(
        value,
        output_key="climateZone",
        value_keys=("climateZone", "value", "href"),
    )


def _normalize_temporal_surface_cover(value: Any) -> Optional[Dict[str, Any]]:
    """Normalize surface-cover history into the WMDR2 temporal structure."""
    return _normalize_temporal_facility_values(
        value,
        output_key="surfaceCover",
        value_keys=("surfaceCover", "value", "href"),
    )


def _copy_known_facility_properties(facility: Dict[str, Any]) -> Dict[str, Any]:
    """Copy and normalize known WMDR2 core facility properties.

    Time-varying facility descriptors are exposed as explicit ``temporal*``
    properties instead of carrying WMDR1/XML ``beginPosition`` / ``endPosition``
    fields inside the raw value.
    """
    out: Dict[str, Any] = {}

    scalar_or_structured_keys = [
        "facilitySet",
        "facilityType",
        "wmoRegion",
        "surfaceCoverClassification",
        "localTopography",
        "relativeElevation",
        "topographicContext",
        "altitudeOrDepth",
        "timeZone",
        "regionOfOrigin",
        # dateEstablished/dateClosed are consumed by the root Feature ``time``
        # member and are intentionally not repeated under properties.
    ]
    for key in scalar_or_structured_keys:
        if _non_empty(facility.get(key)):
            out[key] = facility[key]

    temporal_territory = _normalize_temporal_territory(
        _first_non_empty(facility.get("territory"), facility.get("territoryName"))
    )
    if temporal_territory:
        out["temporalTerritory"] = temporal_territory

    temporal_climate_zone = _normalize_temporal_climate_zone(facility.get("climateZone"))
    if temporal_climate_zone:
        out["temporalClimateZone"] = temporal_climate_zone

    temporal_surface_cover = _normalize_temporal_surface_cover(facility.get("surfaceCover"))
    if temporal_surface_cover:
        out["temporalSurfaceCover"] = temporal_surface_cover

    return out


def _normalize_deployment(raw: Dict[str, Any], *, index: int, facility_id: str) -> Dict[str, Any]:
    """Normalize a deployment for the facility-level deployments collection.

    Deployments are now represented once under ``properties.deployments`` and
    observations only reference them by id. Reporting schedules are not stored
    here because international reporting belongs to the observation in the
    WMDR2 core model.
    """
    record_id = _deployment_record_id(raw, index=index, facility_id=facility_id)

    # Deployments are referenceable WMDR2 entities and therefore keep an ``id``.
    # They intentionally do not carry a generic discovery ``title``; display
    # labels can be derived by clients from manufacturer/model/serialNumber
    # where those values are available.
    start, end = _extract_interval(raw)
    time = _time_interval(start, end)

    observing_schedule = _normalize_temporal_observing_schedule(raw.get("dataGeneration") or raw.get("observingSchedule"))
    contacts, _ = _collect_contacts(raw.get("contact"), raw.get("contacts"), raw.get("responsibleParty"))

    payload: Dict[str, Any] = {
        "id": record_id,
        "time": time,
        "description": _normalize_description_value(raw.get("description")),
        "sourceOfObservation": raw.get("sourceOfObservation"),
        "observingMethod": raw.get("observingMethod"),
        "instrument": _instrument_refs_for_deployment(raw, facility_id=facility_id),
        "serialNumbers": _deployment_serial_numbers(raw),
        "exposure": raw.get("exposure"),
        "representativeness": raw.get("representativeness"),
        "localReferenceSurface": raw.get("localReferenceSurface"),
        "instrumentOperatingStatus": _normalize_reporting_status(raw.get("instrumentOperatingStatus")),
        "contacts": contacts,
        "temporalObservingSchedule": observing_schedule,
        "keywords": _keywords_from_values(_collect_discovery_values("deployment", raw, "keywords")),
        "links": _extract_links(raw, "deployment"),
    }
    return _clean_none(payload)




def _normalize_observation_reporting(*sources: Any) -> Optional[Dict[str, List[Any]]]:
    """Normalize observation-level reporting into aligned property arrays.

    WMDR1 often nests reporting information inside deployment/data-generation
    structures. In the WMDR2 facility-centric model, reporting belongs to the
    observation. The output uses parallel arrays; values at the same index
    describe the same reporting configuration. Missing values are represented
    as JSON null to preserve array alignment.

    Example:
        {
          "internationalExchange": [true, false],
          "temporalReportingInterval": ["PT1H", "PT10M"],
          "uom": [null, "http://codes.wmo.int/wmdr/unit/mm"]
        }
    """
    records: List[Dict[str, Any]] = []

    for source in sources:
        for item in _normalize_temporal_reporting_schedule(source):
            reporting = _as_dict(item.get("reporting"))
            record: Dict[str, Any] = {}

            for key, value in reporting.items():
                if key == "uom" and _is_unknown_token(value):
                    continue
                if _non_empty(value) or isinstance(value, bool):
                    record[key] = value

            if record:
                records.append(record)

    records = _uniq_dicts(records)
    if not records:
        return None

    preferred_order = [
        "internationalExchange",
        "temporalReportingInterval",
        "uom",
        "referenceDatum",
    ]
    extra_keys = sorted(
        key
        for record in records
        for key in record
        if key not in preferred_order
    )

    reporting: Dict[str, List[Any]] = {}
    for key in preferred_order + extra_keys:
        if not any(key in record for record in records):
            continue
        reporting[key] = [record.get(key) for record in records]

    return reporting or None


def _normalize_temporal_data_policy(value: Any) -> List[Dict[str, Any]]:
    """Normalize optional observation-level temporal data-policy entries."""
    out: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if isinstance(item, str):
            policy = _normalize_code_value(item)
            if isinstance(policy, str) and not _is_unknown_token(policy):
                out.append({"dataPolicy": policy})
            continue
        if not isinstance(item, dict):
            continue

        record = _drop_source_metadata(item)
        if not isinstance(record, dict):
            continue

        policy = _first_non_empty(record.get("dataPolicy"), record.get("policy"), record.get("value"), record.get("href"))
        if isinstance(policy, str):
            record["dataPolicy"] = _normalize_code_value(policy)
            for key in ("policy", "value", "href"):
                if key != "dataPolicy":
                    record.pop(key, None)

        valid_time = _time_interval(item.get("beginPosition"), item.get("endPosition"))
        if valid_time:
            record["time"] = valid_time
            record.pop("beginPosition", None)
            record.pop("endPosition", None)

        if record:
            out.append(record)
    return _uniq_dicts(_clean_none(out))

def _normalize_observation(raw: Dict[str, Any], *, index: int, facility_id: str) -> Dict[str, Any]:
    """Normalize one observation embedded in a facility-centric record."""
    embedded_deployments = [item for item in _as_list(raw.get("deployments")) if isinstance(item, dict)]
    observed_variable = raw.get("observedVariable") or raw.get("observedProperty")
    explicit_obs_id = _first_non_empty(
        raw.get("identifier"),
        raw.get("@gml:id"),
        raw.get("@id"),
        raw.get("id"),
    )
    obs_id = _first_non_empty(
        explicit_obs_id,
        _compact_wmdr_code_value(observed_variable),
        f"{facility_id}:observation:{index}",
    )
    source_id = _sanitize_id(str(obs_id))

    observed_geometry_type = raw.get("observedGeometryType") or raw.get("geometryType") or raw.get("type")
    title = _first_non_empty(
        _format_observation_title(observed_variable, observed_geometry_type),
        raw.get("title"),
        raw.get("name"),
        f"Observation {index}",
    )
    time = _derive_observation_time(raw, embedded_deployments)

    contacts, _ = _collect_contacts(raw.get("contact"), raw.get("contacts"), raw.get("responsibleParty"))
    observing_schedule = _normalize_temporal_observing_schedule(raw.get("dataGeneration") or raw.get("observingSchedule"))

    reporting_sources: List[Any] = [raw.get("dataGeneration"), raw.get("reportingSchedule")]
    reporting_sources.extend(dep.get("dataGeneration") or dep.get("reportingSchedule") for dep in embedded_deployments)

    deployment_refs = [
        _deployment_record_id(dep, index=dep_index, facility_id=facility_id)
        for dep_index, dep in enumerate(embedded_deployments, start=1)
    ]

    payload: Dict[str, Any] = {
        "id": f"observation:{source_id}",
        "title": title,
        "description": None,
        "time": time,
        "observedVariable": observed_variable,
        "observedGeometryType": observed_geometry_type,
        "observedDomain": _observed_domain_from_observed_variable(observed_variable),
        "programAffiliation": _normalize_program_affiliation(raw.get("programAffiliation")),
        "contacts": contacts,
        "temporalObservingSchedule": observing_schedule,
        "temporalDataPolicy": _normalize_temporal_data_policy(raw.get("temporalDataPolicy") or raw.get("dataPolicy") or raw.get("dataPolicyHistory")),
        "deployments": deployment_refs,
        "keywords": _keywords_from_values(_collect_discovery_values("observation", raw, "keywords")),
        "links": _extract_links(raw, "observation"),
    }

    cleaned = _clean_none(payload)
    cleaned["description"] = None
    reporting = _normalize_observation_reporting(*reporting_sources)
    if reporting:
        cleaned["reporting"] = reporting
    return cleaned


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

    keywords = _keywords_from_values(
        [facility_id, facility.get("name")] + _collect_discovery_values("facility", facility, "keywords")
    )
    normalized_observations = [
        _normalize_observation(obs, index=index, facility_id=facility_id)
        for index, obs in enumerate(observations, start=1)
        if isinstance(obs, dict)
    ]

    all_deployments: List[Dict[str, Any]] = [dep for dep in deployments if isinstance(dep, dict)]
    all_deployments.extend(_flatten_deployments_from_observations([obs for obs in observations if isinstance(obs, dict)]))

    normalized_deployments_by_id: Dict[str, Dict[str, Any]] = {}
    for index, dep in enumerate(all_deployments, start=1):
        normalized = _normalize_deployment(dep, index=index, facility_id=facility_id)
        dep_id = normalized.get("id")
        if isinstance(dep_id, str) and dep_id not in normalized_deployments_by_id:
            normalized_deployments_by_id[dep_id] = normalized
    normalized_deployments = list(normalized_deployments_by_id.values())
    normalized_instruments = _normalize_instruments(all_deployments, facility_id=facility_id)

    temporal_geometry_entries = _normalize_temporal_geometry(
        facility.get("geospatialLocation") or facility.get("geometry"),
        facility.get("geospatialLocationHistory") or facility.get("geometryHistory"),
    )
    temporal_geometry = _temporal_geometry_extension(temporal_geometry_entries)

    known_facility_properties = _copy_known_facility_properties(facility)

    props: Dict[str, Any] = {
        "type": "facility",
        "title": _facility_title(facility),
        **_record_timestamps(header, source_name=source_name),
        "description": _first_non_empty(
            _normalize_description_value(facility.get("description")),
            _normalize_description_value(facility.get("information")),
            f"WMDR2 facility record for {_facility_title(facility)}",
        ),
        "externalIds": _uniq_dicts(
            item
            for item in [
                _external_id(facility.get("identifier"), "WMO:WIGOS"),
                _external_id(facility.get("wigosStationIdentifier"), "WMO:WIGOS"),
                _external_id(facility.get("wigosIdentifier"), "WMO:WIGOS"),
                _external_id(header.get("identifier"), "WMDR"),
            ]
            if item
        ),
        "contacts": contacts,
        "keywords": keywords,
        "links": _extract_links(facility, "facility"),
        **known_facility_properties,
        "temporalGeometry": temporal_geometry,
        "temporalProgramAffiliation": _normalize_temporal_program_affiliation(facility.get("programAffiliation")),
        "temporalReportingStatus": _normalize_reporting_status_timeline(facility.get("reportingStatus")),
        "observations": normalized_observations,
        "deployments": normalized_deployments,
        "instruments": normalized_instruments,
    }
    return _clean_none(props)


def _facility_geometry(facility: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Prefer current geometry/geospatialLocation; fall back to first history item.
    for candidate in (
        facility.get("geometry"),
        facility.get("geospatialLocation"),
        next(iter(_as_list(facility.get("geospatialLocationHistory"))), None),
        next(iter(_as_list(facility.get("geometryHistory"))), None),
    ):
        point = _point_from_pos(candidate)
        if point:
            return point
    return None


def _facility_time(
    facility: Dict[str, Any],
    observations: Sequence[Dict[str, Any]],
    deployments: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the facility lifecycle interval.

    The root Feature ``time`` describes the temporal extent of the facility
    resource itself. It must therefore be derived only from facility lifecycle
    fields, not from observation or deployment validity intervals.

    If WMDR1 does not provide explicit ``dateEstablished`` / ``dateClosed``
    values, keep the facility temporal extent explicit but unknown using the
    OGC Records open interval marker ``..``.

    ``observations`` and ``deployments`` are accepted for backward-compatible
    call sites, but are deliberately unused.
    """
    del observations, deployments

    return _time_interval(
        facility.get("dateEstablished"),
        facility.get("dateClosed"),
    ) or {"interval": ["..", ".."]}


def build_facility_feature(
    facility: Dict[str, Any],
    observations: Optional[Sequence[Dict[str, Any]]] = None,
    deployments: Optional[Sequence[Dict[str, Any]]] = None,
    header: Optional[Dict[str, Any]] = None,
    source_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one facility-centric WMDR2 core Feature."""
    observations = observations or []
    deployments = deployments or []
    header = header or {}
    facility_id = _facility_identifier(facility, header)

    feature: Dict[str, Any] = {
        "type": "Feature",
        "id": f"facility:{_sanitize_id(facility_id)}",
        "geometry": _facility_geometry(facility),
        "time": _facility_time(facility, observations, deployments),
        "conformsTo": [OGC_RECORD_CORE_CONF, WMDR2_CORE_CONF],
        "properties": _facility_properties(facility, observations, deployments, header, source_name=source_name),
    }
    record = _finalize_wmdr2_value(_clean_none(feature))
    properties = record.get("properties") if isinstance(record, dict) else None
    observations_out = properties.get("observations") if isinstance(properties, dict) else None
    if isinstance(observations_out, list):
        for observation in observations_out:
            if isinstance(observation, dict):
                observation["description"] = None
    return record


def convert_payload(payload: Any, *, source_name: str = "record") -> Dict[str, Any]:
    """Convert a loaded WMDR1 JSON payload into one WMDR2 core Feature."""
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
    """Convert a group of part-file payloads into one facility-centric Feature."""
    header = _as_dict(parts.get("header"))
    facility = _as_dict(parts.get("facility"))
    observations = [item for item in _as_list(parts.get("observations")) if isinstance(item, dict)]
    deployments = [item for item in _as_list(parts.get("deployments")) if isinstance(item, dict)]
    if not facility:
        facility = {"identifier": source_name, "name": source_name}
    return build_facility_feature(facility, observations, deployments, header, source_name=source_name)


def convert_file(
    input_path: Path,
    target_dir: Path,
    *,
    discovery_policy: Optional[Dict[str, Dict[str, List[str]]]] = None,
    code_list_labels: Optional[Dict[str, Dict[str, str]]] = None,
) -> Path:
    """Convert one full WMDR1 JSON file and write a ``.json`` record."""
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
    """Convert all matching JSON files under ``source``.

    Full files are converted individually. Files ending in ``_header.json``,
    ``_facility.json``, ``_observations.json`` and ``_deployments.json`` are
    grouped by their shared stem prefix and converted to a single record.
    """
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
            f"{loaded}\n"
            f"Available top-level config sections: {available_sections}"
        )
    if not target_raw:
        loaded = _format_loaded_config_hint(config_path, section)
        available_sections = ", ".join(sorted(cfg.keys())) if cfg else "none"
        raise SystemExit(
            "Missing target. Set convert_wmdr10_json_to_wmdr2_json.target "
            "in config.yaml or pass --target.\n"
            f"{loaded}\n"
            f"Available top-level config sections: {available_sections}"
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


if __name__ == "__main__":
    main()
