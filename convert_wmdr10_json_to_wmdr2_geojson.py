#!/usr/bin/env python3
"""
convert_wmdr10_json_to_wmdr2_geojson_simplified.py

Convert WMDR 1.0 JSON exports derived from WMDR 1.0 XML into a WMDR2-oriented
GeoJSON representation that remains compliant with OGC API - Records - Part 1.

Design choices
--------------
- Output is a GeoJSON FeatureCollection.
- Each feature is still a valid OGC API Records GeoJSON record:
  - Feature.id
  - Feature.geometry
  - Feature.time
  - Feature.conformsTo
  - Feature.properties.{type,title,description,keywords,themes,externalIds,contacts}
- WMDR2-core semantics are carried as an extension under `properties.wmdr2`.
- The converter emits top-level record features for:
  - facility
  - observation
  - deployment
- Associated WMDR2 classes such as contact, temporalGeometry, procedure and
  temporalReportingSchedule are embedded in `properties.wmdr2`.

Input shapes supported
----------------------
- Full JSON:
    {"header": {...}, "facility": {...}, "observations": [...]}
- Part files:
    *_header.json
    *_facility.json
    *_observations.json
    *_deployments.json
- Observation payloads may contain embedded deployments.
- Deployment payloads may be top-level arrays or objects.

The converter is intentionally defensive. WMDR 1.0 JSON generated from
different simplification steps may differ slightly in naming and nesting.

Example
-------
python convert_wmdr10_json_to_wmdr2_geojson.py \
    --source resources/wmdr10_json_examples \
    --target resources/wmdr2_geojson_examples

python convert_wmdr10_json_to_wmdr2_geojson.py \
    --config config.yaml

Example config.yaml section
---------------------------
convert_wmdr10_json_to_wmdr2_geojson:
  source: resources/wmdr10_json_examples
  target: resources/wmdr2_geojson_examples
  pattern: "*.json"
  recursive: true
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


OGC_RECORD_CORE_CONF = "http://www.opengis.net/spec/ogcapi-records-1/1.0/conf/record-core"
DEFAULT_PATTERN = "*.json"
DEFAULT_DISCOVERY_POLICY: Dict[str, Dict[str, List[str]]] = {
    "facility": {
        "keywords": ["identifier", "name"],
        "themes": [
            "facilitySet",
            "facilityType",
            "wmoRegion",
            "territoryName",
            "climateZone",
            "surfaceCover",
            "surfaceCoverClassification",
            "localTopography",
            "relativeElevation",
            "topographicContext",
            "altitudeOrDepth",
            "programAffiliation",
            "reportingStatus",
        ],
        "links": ["onlineResource"],
    },
    "observation": {
        "keywords": [],
        "themes": ["programAffiliation"],
        "links": [],
    },
    "deployment": {
        "keywords": ["manufacturer", "model", "serialNumber", "sourceOfObservation", "observingMethod"],
        "themes": [
            "sourceOfObservation",
            "observingMethod",
            "exposure",
            "representativeness",
            "localReferenceSurface",
            "instrumentOperatingStatus",
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
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _cfg_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
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
        for bucket in ("keywords", "themes", "links"):
            values = entity_cfg.get(bucket)
            if isinstance(values, list):
                policy[entity][bucket] = [str(v).strip() for v in values if isinstance(v, str) and str(v).strip()]
    return policy




def _extract_code_list_ref(value: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(uri, domain, code)`` for a WMO code-list URI or code-like value.

    Examples
    --------
    ``http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/179`` becomes
    ``("http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/179", "ObservedVariableAtmosphere", "179")``.
    """
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


def _load_code_list_labels(section: Dict[str, Any], *, base_dir: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """Load optional code-list labels used for human-readable titles.

    The preferred CSV columns are ``uri, domain, notation, label``. The loader
    also understands the WMO Codes Registry CSV columns ``@id``,
    ``skos:notation`` and ``rdfs:label``. This keeps label resolution offline
    and reproducible once a registry snapshot has been placed in the repo.
    """
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


def _humanize_identifier(value: Any) -> Optional[str]:
    text = _last_segment(value) if isinstance(value, str) else None
    if not text:
        return None
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if text else None


def _display_domain_name(domain: Optional[str]) -> Optional[str]:
    if not domain:
        return None
    text = re.sub(r"^ObservedVariable", "", domain)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or domain


def _format_observation_title(value: Any) -> Optional[str]:
    """Build a discovery title from an observed-property code-list URI."""
    _, domain, code = _extract_code_list_ref(value)
    if not code:
        return None
    label = _lookup_code_list_label(domain, code)
    domain_label = _display_domain_name(domain)
    prefix = f"variable {code}"
    if label:
        prefix = f"{prefix}: {label}"
    if domain_label:
        return f"{prefix}; domain: {domain_label}"
    return prefix

def _iter_json_files(root: Path, *, pattern: str = DEFAULT_PATTERN, recursive: bool = True) -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".json" else []
    if not root.is_dir():
        return []
    walker = root.rglob if recursive else root.glob
    return sorted(p for p in walker(pattern) if p.is_file() and p.suffix.lower() == ".json")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
        if "facility" in payload or "observations" in payload or "header" in payload:
            return "full"
        if "observedProperty" in payload or "resultTime" in payload:
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
        if "observedProperty" in first or "resultTime" in first:
            return "observations"
        if "sourceOfObservation" in first or "manufacturer" in first or "serialNumber" in first:
            return "deployments"

    return "unknown"


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
    if isinstance(value, dict):
        return value
    return {}


def _non_empty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if _non_empty(value):
            return value
    return None


def _clean_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        cleaned = {k: _clean_none(v) for k, v in obj.items()}
        return {k: v for k, v in cleaned.items() if v not in (None, "", [], {})}
    if isinstance(obj, list):
        cleaned = [_clean_none(v) for v in obj]
        return [v for v in cleaned if v not in (None, "", [], {})]
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
        tail = text.rstrip("/#").rsplit("/", 1)[-1]
        text = tail
    if re.fullmatch(r"\(([^()]+)\)", text):
        text = re.fullmatch(r"\(([^()]+)\)", text).group(1)  # type: ignore[union-attr]
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
    simplified = _simplify_unknown_text(text)
    if isinstance(simplified, str):
        text = simplified.strip()
    if _is_unknown_token(text):
        return "unknown"
    return text


def _normalize_display_text(value: Any) -> Optional[str]:
    """Normalize human-facing text while keeping non-empty meaningful values."""
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


def _compact_display_values(values: Iterable[Any]) -> List[str]:
    """Return unique display values, dropping unknown when real values exist."""
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


def _summarize_intervals(intervals: Sequence[Sequence[Any]]) -> Optional[Dict[str, Any]]:
    starts = [str(item[0]) for item in intervals if len(item) >= 1 and isinstance(item[0], str) and item[0] != ".."]
    ends = [str(item[1]) for item in intervals if len(item) >= 2 and isinstance(item[1], str) and item[1] != ".."]
    has_open = any(len(item) >= 2 and item[1] == ".." for item in intervals)
    if not starts and not ends and not has_open:
        return None
    return {"interval": [min(starts) if starts else "..", ".." if has_open else (max(ends) if ends else "..")] }


def _derive_observation_time(observation: Dict[str, Any], deployments: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    intervals: List[List[Any]] = []
    for dep in deployments:
        candidates = [dep, _as_dict(dep.get("temporalExtent")), _as_dict(dep.get("time"))]
        for candidate in candidates:
            start, end = _extract_interval(candidate)
            interval = _time_interval(start, end)
            if interval and isinstance(interval.get("interval"), list):
                intervals.append(interval["interval"])
                break
    if intervals:
        return _summarize_intervals(intervals)
    return _time_interval(observation.get("beginPosition"), observation.get("endPosition"))


def _last_segment(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().rstrip("/#")
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


def _time_interval(start: Any, end: Any, *, resolution: Optional[str] = None) -> Optional[Dict[str, Any]]:
    s = _normalize_time_value(start)
    e = _normalize_open_end(end)
    if s is None and e == "..":
        return None
    interval = [s or "..", e]
    out: Dict[str, Any] = {"interval": interval}
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
        raw = value.get("#text")
        if raw in (None, ""):
            return None
        try:
            num: Any = int(raw) if str(raw).isdigit() else float(raw)
        except Exception:
            num = raw
        out: Dict[str, Any] = {"value": num}
        if value.get("@uom"):
            out["uom"] = value.get("@uom")
        return out
    return None


def _parse_pos_lon_lat_z(raw: Any) -> Optional[List[Any]]:
    if raw is None:
        return None

    if isinstance(raw, dict):
        coords_value = raw.get("coordinates")
        if isinstance(coords_value, list) and len(coords_value) >= 2:
            return coords_value  # already GeoJSON style

    if isinstance(raw, dict):
        for key in ("geoLocation", "pos", "value", "text", "geometry"):
            val = raw.get(key)
            if isinstance(val, str):
                raw = val
                break

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


# ---------------------------------------------------------------------------
# OGC API Records helpers
# ---------------------------------------------------------------------------

def _external_id(value: Any, scheme: str) -> Optional[Dict[str, str]]:
    if not isinstance(value, str) or not value.strip():
        return None
    return {"scheme": scheme, "value": value.strip()}


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



def _themes_from_uris(values: Iterable[Any]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for raw in values:
        if not isinstance(raw, str) or not raw.startswith(("http://", "https://")):
            continue
        if _is_unknown_token(raw):
            continue
        concept_id = _last_segment(raw)
        if not concept_id or _is_unknown_token(concept_id):
            continue
        scheme = _uri_parent(raw)
        grouped.setdefault(scheme, []).append({"id": concept_id, "url": raw})

    themes: List[Dict[str, Any]] = []
    for scheme, concepts in grouped.items():
        themes.append({"scheme": scheme, "concepts": _uniq_dicts(concepts)})
    return _uniq_dicts(themes)

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


# ---------------------------------------------------------------------------
# Contact normalization
# ---------------------------------------------------------------------------

def _normalize_role(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    return _last_segment(value) or value.strip()


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

    name = _first_non_empty(payload.get("individualName"), payload.get("name"), payload.get("title"))
    if isinstance(name, str):
        contact["name"] = name

    position = _first_non_empty(payload.get("positionName"), payload.get("position"))
    if isinstance(position, str):
        contact["position"] = position

    organization = _first_non_empty(payload.get("organisationName"), payload.get("organizationName"), payload.get("organization"))
    if isinstance(organization, str):
        contact["organization"] = organization

    info: Dict[str, Any] = _as_dict(payload.get("contactInfo"))

    phone_obj: Dict[str, Any] = _as_dict(info.get("phone"))
    voices = _as_list(phone_obj.get("voice"))
    phones = []
    for voice in voices:
        if isinstance(voice, str) and voice.strip():
            phones.append({"value": voice.strip()})
    if phones:
        contact["phones"] = phones

    address_obj: Dict[str, Any] = _as_dict(info.get("address"))
    emails = []
    for email in _as_list(address_obj.get("electronicMailAddress")):
        if isinstance(email, str) and "@" in email:
            emails.append({"value": email.strip()})
    if emails:
        contact["emails"] = emails

    addresses = []
    if isinstance(address_obj, dict):
        address: Dict[str, Any] = {}
        delivery_points = [dp for dp in _as_list(address_obj.get("deliveryPoint")) if isinstance(dp, str) and dp.strip()]
        if delivery_points:
            address["deliveryPoint"] = delivery_points
        for src_key, dst_key in (
            ("city", "city"),
            ("administrativeArea", "administrativeArea"),
            ("postalCode", "postalCode"),
            ("country", "country"),
        ):
            value = address_obj.get(src_key)
            if isinstance(value, str) and value.strip():
                address[dst_key] = value.strip()
        if address:
            addresses.append(address)
    if addresses:
        contact["addresses"] = addresses

    links = []
    online: Dict[str, Any] = _as_dict(info.get("onlineResource"))
    href = online.get("url")
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
    extended_contacts: List[Dict[str, Any]] = []

    for group in groups:
        for item in _as_list(group):
            ogc_contact, ext_contact = _normalize_contact(item)
            if ogc_contact:
                ogc_contacts.append(ogc_contact)
            if ext_contact:
                extended_contacts.append(ext_contact)

    return _uniq_dicts(ogc_contacts), _uniq_dicts(extended_contacts)


# ---------------------------------------------------------------------------
# WMDR-specific normalization
# ---------------------------------------------------------------------------

def _normalize_reporting_status(value: Any) -> List[Dict[str, Any]]:
    statuses: List[Dict[str, Any]] = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        status_val = _first_non_empty(item.get("reportingStatus"), item.get("instrumentOperatingStatus"))
        record: Dict[str, Any] = {}
        if isinstance(status_val, str):
            record["value"] = status_val
        if isinstance(item.get("@gml:id"), str):
            record["id"] = item.get("@gml:id")
        valid_time = _time_interval(item.get("beginPosition"), item.get("endPosition"))
        if valid_time:
            record["time"] = valid_time
        if record:
            statuses.append(record)
    return _uniq_dicts(statuses)


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
            normalized
            for raw in _as_list(item.get("programAffiliation"))
            for normalized in [raw if isinstance(raw, str) else None]
            if isinstance(normalized, str) and normalized.strip() and not _is_unknown_token(normalized)
        ]
        if affiliations:
            record["programAffiliation"] = [_normalize_code_value(x) for x in affiliations]
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
    return _uniq_dicts(out)


def _normalize_simple_timed_value(value: Any, *, value_key: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    actual = _normalize_code_value(value.get(value_key))
    if actual in (None, ""):
        return None
    out: Dict[str, Any] = {"value": actual}
    if isinstance(value.get("@gml:id"), str):
        out["id"] = value.get("@gml:id")
    valid_time = _time_interval(value.get("beginPosition"), value.get("endPosition"))
    if valid_time:
        out["time"] = valid_time
    for extra_key in ("surfaceCoverClassification", "localTopography", "relativeElevation", "topographicContext", "altitudeOrDepth"):
        if _non_empty(value.get(extra_key)):
            out[extra_key] = value.get(extra_key)
    return _clean_none(out)


def _normalize_temporal_geometry(current: Any, history: Any = None) -> List[Dict[str, Any]]:
    """Collect geospatial history entries in chronological order.

    Returns a list of dictionaries with the keys:
    - coordinates: GeoJSON-style [lon, lat, z?]
    - datetimes: [begin, end] using ".." for open bounds
    - id: optional source identifier

    The stage-1 XML -> JSON converter now normalizes facility geospatialLocation
    histories into a clean chronological list. Stage 2 uses that shape directly to
    build WMDR2 temporalGeometry as a MovingPoint when more than one location is
    present.
    """

    entries: List[Tuple[Optional[str], str, Dict[str, Any]]] = []

    def add_entry(item: Any) -> None:
        if isinstance(item, str):
            coords = _parse_pos_lon_lat_z(item)
            if coords is None:
                return
            payload = {"coordinates": coords, "datetimes": ["..", ".."]}
            entries.append((None, json.dumps(coords, sort_keys=True), payload))
            return

        if not isinstance(item, dict):
            return

        coords = _parse_pos_lon_lat_z(item.get("geometry") or item.get("geoLocation") or item.get("pos") or item)
        if coords is None:
            return

        start, end = _extract_interval(item)
        interval_dict = _time_interval(start, end)
        interval = interval_dict["interval"] if interval_dict else ["..", ".."]
        start_key = interval[0]

        payload: Dict[str, Any] = {
            "coordinates": coords,
            "datetimes": interval,
        }
        source_id = _first_non_empty(item.get("@gml:id"), item.get("@id"), item.get("id"))
        if isinstance(source_id, str) and source_id.strip():
            payload["id"] = source_id.strip()

        entries.append((None if start_key == ".." else str(start_key), json.dumps(coords, sort_keys=True), payload))

    for item in _as_list(current):
        add_entry(item)
    for item in _as_list(history):
        add_entry(item)

    if not entries:
        return []

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
    if len(entries) <= 1:
        return None

    out: Dict[str, Any] = {
        "type": "MovingPoint",
        "coordinates": [entry["coordinates"] for entry in entries if isinstance(entry.get("coordinates"), list)],
        "datetimes": [entry["datetimes"] for entry in entries if isinstance(entry.get("datetimes"), list)],
    }
    ids = []
    for entry in entries:
        entry_id = entry.get("id")
        if isinstance(entry_id, str) and entry_id.strip():
            ids.append(entry_id)
    if ids:
        out["id"] = ids if len(ids) > 1 else ids[0]
    return _clean_none(out)


def _current_geometry_from_temporal_geometry(temporal_geometry: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(temporal_geometry, list) or not temporal_geometry:
        return None

    def rank(entry: Dict[str, Any]) -> Tuple[int, str, str]:
        interval = entry.get("datetimes")
        if isinstance(interval, list) and len(interval) == 2:
            start = str(interval[0])
            end = str(interval[1])
            if end == "..":
                return (0, start, "")
            return (1, end, start)
        return (2, "", "")

    latest = sorted(temporal_geometry, key=rank)[0]
    coords = latest.get("coordinates")
    if not isinstance(coords, list) or len(coords) < 2:
        return None
    return {"type": "Point", "coordinates": coords}


def _time_from_temporal_geometry(entries: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(entries, list) or not entries:
        return None

    intervals: List[List[Any]] = []
    for entry in entries:
        raw_interval = entry.get("datetimes")
        if isinstance(raw_interval, list) and len(raw_interval) == 2:
            intervals.append(raw_interval)
    if not intervals:
        return None

    starts = [str(interval[0]) for interval in intervals if isinstance(interval[0], str) and interval[0] != ".."]
    ends = [str(interval[1]) for interval in intervals if isinstance(interval[1], str) and interval[1] != ".."]
    start = min(starts) if starts else ".."
    end = ".." if any(isinstance(interval[1], str) and interval[1] == ".." for interval in intervals) else (max(ends) if ends else "..")
    return {"interval": [start, end]}


def _normalize_schedule_coverage(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
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
            if _non_empty(value.get(key)):
                out[key] = value.get(key)
        return out or None
    return None


def _normalize_data_policy(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    out: Dict[str, Any] = {}
    if isinstance(value.get("dataPolicy"), str):
        out["dataPolicy"] = value.get("dataPolicy")
    if isinstance(value.get("levelOfData"), str):
        out["levelOfData"] = value.get("levelOfData")
    attribution = value.get("attribution")
    if isinstance(attribution, dict):
        originator = attribution.get("originator")
        if originator is not None:
            contact, ext = _normalize_contact(originator)
            out["attribution"] = ext or contact or originator
    return out or None


def _normalize_sampling(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    if value.get("Sampling") is None and len(value) == 1:
        return None
    out: Dict[str, Any] = {}
    for key in ("samplingProcedure", "samplingStrategy", "temporalSamplingInterval", "samplingTimePeriod"):
        if _non_empty(value.get(key)):
            out[key] = value.get(key)
    return out or None


def _normalize_processing(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    if value.get("Processing") is None and len(value) == 1:
        return None
    out: Dict[str, Any] = {}
    for key in ("aggregationPeriod",):
        if _non_empty(value.get(key)):
            out[key] = value.get(key)
    return out or None


def _normalize_schedule_entry(data_generation: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(data_generation, dict):
        return None
    out: Dict[str, Any] = {}
    if isinstance(data_generation.get("@gml:id"), str):
        out["id"] = data_generation.get("@gml:id")

    sampling = _normalize_sampling(data_generation.get("sampling"))
    if sampling:
        out["sampling"] = sampling

    processing = _normalize_processing(data_generation.get("processing"))
    if processing:
        out["processing"] = processing

    reporting = data_generation.get("reporting") if isinstance(data_generation.get("reporting"), dict) else None
    if reporting:
        reporting_obj: Dict[str, Any] = {}
        for key in (
            "internationalExchange",
            "uom",
            "temporalReportingInterval",
            "timeStampMeaning",
            "numberOfObservationsInReportingInterval",
        ):
            if _non_empty(reporting.get(key)):
                reporting_obj[key] = reporting.get(key)

        interval = _time_interval(reporting.get("beginPosition"), reporting.get("endPosition"))
        if interval:
            reporting_obj["validTime"] = interval

        coverage = _normalize_schedule_coverage(reporting.get("coverage"))
        if coverage:
            reporting_obj["coverage"] = coverage

        data_policy = _normalize_data_policy(reporting.get("dataPolicy"))
        if data_policy:
            reporting_obj["dataPolicy"] = data_policy

        if reporting_obj:
            out["reporting"] = reporting_obj

    return out or None


def _normalize_temporal_reporting_schedule(data_generation: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(data_generation):
        entry = _normalize_schedule_entry(item)
        if not entry:
            continue
        report = entry.get("reporting")
        if not isinstance(report, dict):
            continue
        schedule_obj: Dict[str, Any] = {}
        if entry.get("id"):
            schedule_obj["id"] = entry["id"]
        if _non_empty(report.get("temporalReportingInterval")):
            schedule_obj["interval"] = report["temporalReportingInterval"]
        elif entry.get("id"):
            schedule_obj["interval"] = "unknown"
        coverage = report.get("coverage")
        if coverage:
            schedule_obj["schedule"] = coverage
        valid_time = report.get("validTime")
        if valid_time:
            schedule_obj["time"] = valid_time
        if schedule_obj:
            out.append(schedule_obj)
    return _uniq_dicts(out)


def _normalize_temporal_observing_schedule(data_generation: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(data_generation):
        entry = _normalize_schedule_entry(item)
        if not entry:
            continue
        sampling = entry.get("sampling")
        reporting = entry.get("reporting")
        obj: Dict[str, Any] = {}
        if entry.get("id"):
            obj["id"] = entry["id"]
        if isinstance(sampling, dict):
            if _non_empty(sampling.get("temporalSamplingInterval")):
                obj["interval"] = sampling["temporalSamplingInterval"]
            schedule = reporting.get("coverage") if isinstance(reporting, dict) else None
            if schedule:
                obj["schedule"] = schedule
        if entry.get("id") and not _non_empty(obj.get("interval")):
            obj["interval"] = "unknown"
        if obj:
            out.append(obj)
    return _uniq_dicts(out)


def _derive_observation_level_reporting(deployments: Sequence[Dict[str, Any]]) -> Tuple[Optional[bool], List[Dict[str, Any]]]:
    current_flags: List[bool] = []
    schedules: List[Dict[str, Any]] = []

    for dep in deployments:
        for dg in _as_list(dep.get("dataGeneration")):
            if not isinstance(dg, dict):
                continue
            report: Dict[str, Any] = _as_dict(dg.get("reporting"))
            flag = _parse_bool(report.get("internationalExchange"))
            if flag is not None:
                current_flags.append(flag)
            interval = report.get("temporalReportingInterval")
            coverage = _normalize_schedule_coverage(report.get("coverage"))
            if interval or coverage:
                entry: Dict[str, Any] = {}
                if interval:
                    entry["interval"] = interval
                if coverage:
                    entry["schedule"] = coverage
                valid_time = _time_interval(report.get("beginPosition"), report.get("endPosition"))
                if valid_time:
                    entry["time"] = valid_time
                if entry:
                    schedules.append(entry)

    inferred_flag = None
    if current_flags:
        if all(current_flags):
            inferred_flag = True
        elif not any(current_flags):
            inferred_flag = False

    return inferred_flag, _uniq_dicts(schedules)


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _record_template(record_id: str, record_type: str) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "id": record_id,
        "geometry": None,
        "time": None,
        "conformsTo": [OGC_RECORD_CORE_CONF],
        "properties": {
            "type": record_type,
        },
    }


def _facility_record_id(facility: Dict[str, Any]) -> str:
    raw = _first_non_empty(facility.get("identifier"), facility.get("@id"), facility.get("@gml:id"), facility.get("name"))
    return f"facility:{_sanitize_id(str(raw))}"


def _observation_record_id(observation: Dict[str, Any], index: int) -> str:
    raw = _first_non_empty(
        observation.get("@gml:id"),
        observation.get("@id"),
        observation.get("identifier"),
        observation.get("observedProperty"),
        f"observation-{index + 1}",
    )
    return f"observation:{_sanitize_id(str(raw))}"


def _deployment_record_id(deployment: Dict[str, Any], index: int) -> str:
    raw = _first_non_empty(
        deployment.get("@gml:id"),
        deployment.get("@id"),
        deployment.get("identifier"),
        deployment.get("serialNumber"),
        deployment.get("model"),
        f"deployment-{index + 1}",
    )
    return f"deployment:{_sanitize_id(str(raw))}"


def _record_owner_contact(header: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(header, dict):
        return None
    return header.get("recordOwner")


def _header_created_updated(header: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(header, dict):
        return None, None
    value = header.get("fileDateTime")
    if isinstance(value, str) and value.strip():
        return value.strip(), value.strip()
    return None, None


def _facility_description_text(facility: Dict[str, Any]) -> Optional[str]:
    desc = facility.get("description")
    if isinstance(desc, str):
        return desc
    if isinstance(desc, dict):
        inner = _first_non_empty(desc.get("description"), desc.get("value"), desc.get("text"))
        if isinstance(inner, str):
            return inner
    return None


def _facility_geometry_and_extension(
    facility: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    temporal_entries = _normalize_temporal_geometry(
        facility.get("geospatialLocation"),
        _first_non_empty(
            facility.get("geospatialLocationHistory"),
            facility.get("geometryHistory"),
            facility.get("temporalGeometry"),
        ),
    )
    geometry = _current_geometry_from_temporal_geometry(temporal_entries)
    temporal_geometry = _temporal_geometry_extension(temporal_entries)
    return geometry, temporal_geometry, temporal_entries


def _collect_facility_formal_uris(facility: Dict[str, Any]) -> List[str]:
    uris: List[str] = []
    for key in ("facilitySet", "facilityType", "wmoRegion"):
        value = facility.get(key)
        if isinstance(value, str):
            uris.append(value)
    territory = facility.get("territory")
    if isinstance(territory, dict) and isinstance(territory.get("territoryName"), str):
        uris.append(territory["territoryName"])
    climate = facility.get("climateZone")
    if isinstance(climate, dict) and isinstance(climate.get("climateZone"), str):
        uris.append(climate["climateZone"])
    surface = facility.get("surfaceCover")
    if isinstance(surface, dict):
        for key in ("surfaceCover", "surfaceCoverClassification"):
            if isinstance(surface.get(key), str):
                uris.append(surface[key])
    topo = facility.get("topographyBathymetry")
    if isinstance(topo, dict):
        for key in ("localTopography", "relativeElevation", "topographicContext", "altitudeOrDepth"):
            if isinstance(topo.get(key), str):
                uris.append(topo[key])
    for entry in _as_list(facility.get("programAffiliation")):
        if isinstance(entry, dict):
            for uri in _as_list(entry.get("programAffiliation")):
                if isinstance(uri, str):
                    uris.append(uri)
            for status in _as_list(entry.get("reportingStatus")):
                if isinstance(status, dict):
                    for key in ("reportingStatus",):
                        if isinstance(status.get(key), str):
                            uris.append(status[key])
    return uris



def _policy_list(entity: str, bucket: str) -> List[str]:
    entity_cfg = DISCOVERY_POLICY.get(entity, {})
    values = entity_cfg.get(bucket, [])
    return [value for value in values if isinstance(value, str) and value]


def _facility_discovery_values(facility: Dict[str, Any], token: str) -> List[Any]:
    mapping: Dict[str, List[Any]] = {
        "identifier": [facility.get("identifier")],
        "name": [facility.get("name"), facility.get("facilityName")],
        "facilitySet": [facility.get("facilitySet")],
        "facilityType": [facility.get("facilityType")],
        "wmoRegion": [facility.get("wmoRegion")],
        "territoryName": [_as_dict(facility.get("territory")).get("territoryName")],
        "climateZone": [_as_dict(facility.get("climateZone")).get("climateZone")],
        "surfaceCover": [_as_dict(facility.get("surfaceCover")).get("surfaceCover")],
        "surfaceCoverClassification": [_as_dict(facility.get("surfaceCover")).get("surfaceCoverClassification")],
        "localTopography": [_as_dict(facility.get("topographyBathymetry")).get("localTopography")],
        "relativeElevation": [_as_dict(facility.get("topographyBathymetry")).get("relativeElevation")],
        "topographicContext": [_as_dict(facility.get("topographyBathymetry")).get("topographicContext")],
        "altitudeOrDepth": [_as_dict(facility.get("topographyBathymetry")).get("altitudeOrDepth")],
        "onlineResource": [_as_dict(facility.get("onlineResource")).get("url")],
        "programAffiliation": [
            uri
            for entry in _as_list(facility.get("programAffiliation")) if isinstance(entry, dict)
            for uri in _as_list(entry.get("programAffiliation"))
        ],
        "reportingStatus": [
            status.get("reportingStatus")
            for entry in _as_list(facility.get("programAffiliation")) if isinstance(entry, dict)
            for status in _as_list(entry.get("reportingStatus")) if isinstance(status, dict)
        ],
    }
    return mapping.get(token, [])


def _observation_discovery_values(observation: Dict[str, Any], deployments: Sequence[Dict[str, Any]], token: str) -> List[Any]:
    mapping: Dict[str, List[Any]] = {
        "programAffiliation": list(_as_list(observation.get("programAffiliation"))),
    }
    return mapping.get(token, [])


def _deployment_discovery_values(deployment: Dict[str, Any], token: str) -> List[Any]:
    if token == "instrumentOperatingStatus":
        status = _as_dict(deployment.get("instrumentOperatingStatus")).get("instrumentOperatingStatus")
        return [status]
    mapping: Dict[str, List[Any]] = {
        "manufacturer": [deployment.get("manufacturer")],
        "model": [deployment.get("model")],
        "serialNumber": [deployment.get("serialNumber")],
        "sourceOfObservation": [deployment.get("sourceOfObservation")],
        "observingMethod": [deployment.get("observingMethod")],
        "exposure": [deployment.get("exposure")],
        "representativeness": [deployment.get("representativeness")],
        "localReferenceSurface": [deployment.get("localReferenceSurface")],
    }
    return mapping.get(token, [])


def _discovery_keywords(entity: str, source: Dict[str, Any], *, deployments: Optional[Sequence[Dict[str, Any]]] = None) -> List[str]:
    raw_values: List[Any] = []
    for token in _policy_list(entity, "keywords"):
        if entity == "facility":
            raw_values.extend(_facility_discovery_values(source, token))
        elif entity == "observation":
            raw_values.extend(_observation_discovery_values(source, deployments or [], token))
        else:
            raw_values.extend(_deployment_discovery_values(source, token))
    return _keywords_from_values(raw_values)


def _discovery_themes(entity: str, source: Dict[str, Any], *, deployments: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    raw_values: List[Any] = []
    for token in _policy_list(entity, "themes"):
        if entity == "facility":
            raw_values.extend(_facility_discovery_values(source, token))
        elif entity == "observation":
            raw_values.extend(_observation_discovery_values(source, deployments or [], token))
        else:
            raw_values.extend(_deployment_discovery_values(source, token))
    return _themes_from_uris(raw_values)


def _discovery_links(entity: str, source: Dict[str, Any]) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    for token in _policy_list(entity, "links"):
        if entity == "facility" and token == "onlineResource":
            url = _as_dict(source.get("onlineResource")).get("url")
            if isinstance(url, str) and url.strip() and not _is_unknown_token(url):
                links.append(_about_link(url.strip(), title="Facility online resource"))
    return _uniq_dicts(links)


def _build_facility_feature(facility: Dict[str, Any], header: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    record_id = _facility_record_id(facility)
    feature = _record_template(record_id, "facility")

    created, updated = _header_created_updated(header)
    if created:
        feature["properties"]["created"] = created
    if updated:
        feature["properties"]["updated"] = updated

    identifier = facility.get("identifier")
    name = _first_non_empty(facility.get("name"), identifier, facility.get("facilityName"))
    if isinstance(name, str):
        feature["properties"]["title"] = name

    description = _facility_description_text(facility)
    if description:
        feature["properties"]["description"] = description

    geometry, temporal_geometry, temporal_entries = _facility_geometry_and_extension(facility)
    feature["geometry"] = geometry
    feature["time"] = _time_interval(facility.get("dateEstablished"), facility.get("dateClosed")) or _time_from_temporal_geometry(temporal_entries)

    ogc_contacts, _ = _collect_contacts(facility.get("responsibleParty"))
    if ogc_contacts:
        feature["properties"]["contacts"] = ogc_contacts

    external_ids = _uniq_dicts(
        [item for item in [_external_id(identifier, "WMO:WIGOS")] if item]
    )
    if external_ids:
        feature["properties"]["externalIds"] = external_ids

    keywords = _discovery_keywords("facility", facility)
    if keywords:
        feature["properties"]["keywords"] = keywords

    themes = _discovery_themes("facility", facility)
    if themes:
        feature["properties"]["themes"] = themes

    links = _discovery_links("facility", facility)
    if links:
        feature["links"] = links

    feature["properties"]["wmdr2"] = _clean_none(
        {
            "facilitySet": _normalize_code_value(facility.get("facilitySet")),
            "additionalWsi": [x for x in _as_list(facility.get("additionalWSI")) if isinstance(x, str)],
            "temporalProgramAffiliation": _normalize_program_affiliation(facility.get("programAffiliation")),
            "climateZone": _normalize_simple_timed_value(facility.get("climateZone"), value_key="climateZone"),
            "surfaceCover": _normalize_simple_timed_value(facility.get("surfaceCover"), value_key="surfaceCover"),
            "topographyBathymetry": _normalize_simple_timed_value(facility.get("topographyBathymetry"), value_key="localTopography"),
            "temporalGeometry": temporal_geometry,
            "observationIds": [],
            "deploymentIds": [],
        }
    )

    return _clean_none(feature)

def _collect_observation_formal_uris(observation: Dict[str, Any], deployments: Sequence[Dict[str, Any]]) -> List[str]:
    uris: List[str] = []
    for key in ("facility", "observedProperty", "type"):
        value = observation.get(key)
        if isinstance(value, str):
            uris.append(value)
    for uri in _as_list(observation.get("programAffiliation")):
        if isinstance(uri, str):
            uris.append(uri)
    for dep in deployments:
        for key in ("sourceOfObservation", "observingMethod"):
            value = dep.get(key)
            if isinstance(value, str):
                uris.append(value)
    return uris


def _observation_description(observation: Dict[str, Any], deployments: Sequence[Dict[str, Any]]) -> Optional[str]:
    observed = _last_segment(observation.get("observedProperty"))
    geom_type = _last_segment(observation.get("type"))
    manufacturers = _compact_display_values(dep.get("manufacturer") for dep in deployments if isinstance(dep, dict))
    models = _compact_display_values(dep.get("model") for dep in deployments if isinstance(dep, dict))
    methods = _compact_display_values(_humanize_identifier(dep.get("observingMethod")) for dep in deployments if isinstance(dep, dict))
    bits: List[str] = []
    if observed:
        bits.append(f"Observed property {observed}")
    if geom_type:
        bits.append(f"geometry type {geom_type}")
    proc_sources: List[str] = []
    proc_sources.extend(manufacturers)
    proc_sources.extend(models)
    proc_sources.extend(methods)
    proc_parts = _compact_display_values(proc_sources)
    if proc_parts:
        proc = " / ".join(proc_parts)
        proc = _normalize_display_text(proc) or proc
        bits.append(f"deployment procedure {proc}")
    description = "; ".join(bits) if bits else None
    return _normalize_display_text(description) if description else None



def _build_observation_feature(
    observation: Dict[str, Any],
    header: Optional[Dict[str, Any]],
    *,
    facility_record_id: Optional[str],
    facility_geometry: Optional[Dict[str, Any]],
    index: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    record_id = _observation_record_id(observation, index)
    feature = _record_template(record_id, "observation")

    created, updated = _header_created_updated(header)
    if created:
        feature["properties"]["created"] = created
    if updated:
        feature["properties"]["updated"] = updated

    deployments = [dep for dep in _as_list(observation.get("deployments")) if isinstance(dep, dict)]

    observed_property = observation.get("observedProperty")
    title = _format_observation_title(observed_property) or _last_segment(observed_property) or f"Observation {index + 1}"
    feature["properties"]["title"] = title

    description = _observation_description(observation, deployments)
    if description:
        feature["properties"]["description"] = description

    feature["geometry"] = facility_geometry
    feature["time"] = _derive_observation_time(observation, deployments)

    metadata_contact = observation.get("metadata", {}).get("contact") if isinstance(observation.get("metadata"), dict) else None
    ogc_contacts, _ = _collect_contacts(metadata_contact)
    if ogc_contacts:
        feature["properties"]["contacts"] = ogc_contacts

    keywords = _discovery_keywords("observation", observation, deployments=deployments)
    if keywords:
        feature["properties"]["keywords"] = keywords

    themes = _discovery_themes("observation", observation, deployments=deployments)
    if themes:
        feature["properties"]["themes"] = themes

    inferred_exchange, reporting_schedule = _derive_observation_level_reporting(deployments)
    deployment_ids = [_deployment_record_id(dep, dep_index) for dep_index, dep in enumerate(deployments)]

    feature["properties"]["wmdr2"] = _clean_none(
        {
            "facilityId": facility_record_id,
            "deploymentIds": deployment_ids,
            "observationType": _normalize_code_value(observation.get("type")),
            "observedProperty": _normalize_code_value(observed_property),
            "observedVariableCoordinates": observation.get("observedVariableCoordinates"),
            "temporalProgramAffiliation": _normalize_program_affiliation(observation.get("programAffiliation")),
            "internationalExchange": inferred_exchange,
            "internationalReportingSchedule": reporting_schedule,
            "result": observation.get("result"),
        }
    )

    return _clean_none(feature), deployments

def _collect_deployment_formal_uris(deployment: Dict[str, Any]) -> List[str]:
    uris: List[str] = []
    for key in (
        "sourceOfObservation",
        "exposure",
        "representativeness",
        "localReferenceSurface",
        "facility",
        "observedProperty",
        "type",
        "observingMethod",
    ):
        value = deployment.get(key)
        if isinstance(value, str):
            uris.append(value)

    status = deployment.get("instrumentOperatingStatus")
    if isinstance(status, dict):
        for key in ("instrumentOperatingStatus",):
            if isinstance(status.get(key), str):
                uris.append(status[key])

    for dg in _as_list(deployment.get("dataGeneration")):
        if not isinstance(dg, dict):
            continue
        reporting: Dict[str, Any] = _as_dict(dg.get("reporting"))
        for key in ("uom", "timeStampMeaning"):
            value = reporting.get(key)
            if isinstance(value, str):
                uris.append(value)
        data_policy = reporting.get("dataPolicy")
        if isinstance(data_policy, dict):
            for key in ("dataPolicy", "levelOfData"):
                value = data_policy.get(key)
                if isinstance(value, str):
                    uris.append(value)
        sampling = dg.get("sampling")
        if isinstance(sampling, dict):
            for key in ("samplingProcedure", "samplingStrategy"):
                value = sampling.get(key)
                if isinstance(value, str):
                    uris.append(value)
    return uris



def _build_deployment_feature(
    deployment: Dict[str, Any],
    header: Optional[Dict[str, Any]],
    *,
    facility_record_id: Optional[str],
    observation_record_id: Optional[str],
    facility_geometry: Optional[Dict[str, Any]],
    index: int,
) -> Dict[str, Any]:
    record_id = _deployment_record_id(deployment, index)
    feature = _record_template(record_id, "deployment")

    created, updated = _header_created_updated(header)
    if created:
        feature["properties"]["created"] = created
    if updated:
        feature["properties"]["updated"] = updated

    title_bits = [
        deployment.get("manufacturer") if isinstance(deployment.get("manufacturer"), str) else None,
        deployment.get("model") if isinstance(deployment.get("model"), str) else None,
        f"SN {deployment.get('serialNumber')}" if isinstance(deployment.get("serialNumber"), str) else None,
    ]
    raw_title = " ".join(bit for bit in title_bits if bit) or _last_segment(deployment.get("observedProperty")) or f"Deployment {index + 1}"
    normalized_title = _normalize_display_text(raw_title) or _normalize_code_value(raw_title)
    feature["properties"]["title"] = normalized_title if isinstance(normalized_title, str) else raw_title

    description_bits = [
        _normalize_display_text(_last_segment(deployment.get("sourceOfObservation"))),
        _normalize_display_text(deployment.get("configuration") if isinstance(deployment.get("configuration"), str) else None),
        _normalize_display_text(_humanize_identifier(deployment.get("observingMethod"))),
    ]
    description = "; ".join(bit for bit in description_bits if bit)
    if description:
        feature["properties"]["description"] = description

    deployment_temporal_entries = _normalize_temporal_geometry(
        _first_non_empty(
            deployment.get("temporalGeometry"),
            deployment.get("geospatialLocation"),
            deployment.get("geometry"),
        ),
        deployment.get("geometryHistory"),
    )
    deployment_geometry = _current_geometry_from_temporal_geometry(deployment_temporal_entries) or facility_geometry
    deployment_temporal_geometry = _temporal_geometry_extension(deployment_temporal_entries)

    feature["geometry"] = deployment_geometry
    feature["time"] = _time_interval(deployment.get("beginPosition"), deployment.get("endPosition")) or _time_from_temporal_geometry(deployment_temporal_entries)

    ogc_contacts, _ = _collect_contacts(deployment.get("responsibleParty"))
    if ogc_contacts:
        feature["properties"]["contacts"] = ogc_contacts

    keywords = _discovery_keywords("deployment", deployment)
    if keywords:
        feature["properties"]["keywords"] = keywords

    themes = _discovery_themes("deployment", deployment)
    if themes:
        feature["properties"]["themes"] = themes

    procedure = _clean_none(
        {
            "observingMethod": _normalize_code_value(deployment.get("observingMethod")),
            "instrumentManufacturer": deployment.get("manufacturer"),
            "instrumentModel": deployment.get("model"),
            "instrumentSerialNumber": deployment.get("serialNumber"),
        }
    )

    reporting_status = _normalize_reporting_status([deployment.get("instrumentOperatingStatus")] if deployment.get("instrumentOperatingStatus") is not None else [])

    feature["properties"]["wmdr2"] = _clean_none(
        {
            "facilityId": facility_record_id,
            "observationId": observation_record_id,
            "observedProperty": _normalize_code_value(deployment.get("observedProperty")),
            "deploymentType": _normalize_code_value(deployment.get("type")),
            "temporalReportingStatus": reporting_status,
            "sourceOfObservation": _normalize_code_value(deployment.get("sourceOfObservation")),
            "exposure": _normalize_code_value(deployment.get("exposure")),
            "representativeness": _normalize_code_value(deployment.get("representativeness")),
            "configuration": deployment.get("configuration"),
            "heightAboveLocalReferenceSurface": _parse_quantity(deployment.get("heightAboveLocalReferenceSurface")),
            "localReferenceSurface": _normalize_code_value(deployment.get("localReferenceSurface")),
            "temporalGeometry": deployment_temporal_geometry,
            "procedure": procedure,
            "temporalReportingSchedule": _normalize_temporal_reporting_schedule(deployment.get("dataGeneration")),
            "temporalObservingSchedule": _normalize_temporal_observing_schedule(deployment.get("dataGeneration")),
            "temporalDataPolicy": [
                item
                for item in (
                    _normalize_data_policy((dg.get("reporting") or {}).get("dataPolicy"))
                    for dg in _as_list(deployment.get("dataGeneration"))
                    if isinstance(dg, dict)
                )
                if item
            ],
        }
    )

    return _clean_none(feature)

def _feature_collection(features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features)}


# ---------------------------------------------------------------------------
# Conversion orchestration
# ---------------------------------------------------------------------------

def _split_payload(kind: str, payload: Any) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    header: Optional[Dict[str, Any]] = None
    facility: Optional[Dict[str, Any]] = None
    observations: List[Dict[str, Any]] = []
    deployments: List[Dict[str, Any]] = []

    if kind == "full" and isinstance(payload, dict):
        header = payload.get("header") if isinstance(payload.get("header"), dict) else None
        facility = payload.get("facility") if isinstance(payload.get("facility"), dict) else None
        obs_raw = payload.get("observations")
        observations = [x for x in _as_list(obs_raw) if isinstance(x, dict)]
        deployments = _flatten_deployments_from_observations(observations)
        return header, facility, observations, deployments

    if kind == "facility" and isinstance(payload, dict):
        facility = payload
        return header, facility, observations, deployments

    if kind == "header" and isinstance(payload, dict):
        header = payload
        return header, facility, observations, deployments

    if kind == "observations":
        if isinstance(payload, dict) and "observations" in payload:
            observations = [x for x in _as_list(payload.get("observations")) if isinstance(x, dict)]
            header = payload.get("header") if isinstance(payload.get("header"), dict) else None
            facility = payload.get("facility") if isinstance(payload.get("facility"), dict) else None
        elif isinstance(payload, list):
            observations = [x for x in payload if isinstance(x, dict)]
        elif isinstance(payload, dict):
            observations = [payload]
        deployments = _flatten_deployments_from_observations(observations)
        return header, facility, observations, deployments

    if kind == "deployments":
        if isinstance(payload, dict) and "deployments" in payload:
            deployments = [x for x in _as_list(payload.get("deployments")) if isinstance(x, dict)]
            header = payload.get("header") if isinstance(payload.get("header"), dict) else None
            facility = payload.get("facility") if isinstance(payload.get("facility"), dict) else None
        elif isinstance(payload, list):
            deployments = [x for x in payload if isinstance(x, dict)]
        elif isinstance(payload, dict):
            deployments = [payload]
        return header, facility, observations, deployments

    if isinstance(payload, dict):
        header = payload.get("header") if isinstance(payload.get("header"), dict) else None
        facility = payload.get("facility") if isinstance(payload.get("facility"), dict) else None
        observations = [x for x in _as_list(payload.get("observations")) if isinstance(x, dict)]
        deployments = _flatten_deployments_from_observations(observations)

    return header, facility, observations, deployments


def convert_payload(
    payload: Any,
    *,
    source_name: str = "input.json",
    discovery_policy: Optional[Dict[str, Dict[str, List[str]]]] = None,
    code_list_labels: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    global DISCOVERY_POLICY, CODE_LIST_LABELS
    previous_policy = copy.deepcopy(DISCOVERY_POLICY)
    previous_labels = copy.deepcopy(CODE_LIST_LABELS)
    if discovery_policy is not None:
        DISCOVERY_POLICY = copy.deepcopy(discovery_policy)
    if code_list_labels is not None:
        CODE_LIST_LABELS = copy.deepcopy(code_list_labels)
    kind = _detect_kind(Path(source_name), payload)
    header, facility, observations, deployments = _split_payload(kind, payload)

    features: List[Dict[str, Any]] = []

    facility_feature: Optional[Dict[str, Any]] = None
    facility_record_id: Optional[str] = None
    facility_geometry: Optional[Dict[str, Any]] = None

    if facility is not None:
        facility_feature = _build_facility_feature(facility, header)
        facility_record_id = facility_feature["id"]
        facility_geometry = facility_feature.get("geometry")
        features.append(facility_feature)

    observation_features: List[Dict[str, Any]] = []
    deployment_counter = 0

    for obs_index, observation in enumerate(observations):
        obs_feature, embedded_deployments = _build_observation_feature(
            observation,
            header,
            facility_record_id=facility_record_id,
            facility_geometry=facility_geometry,
            index=obs_index,
        )
        observation_features.append(obs_feature)
        features.append(obs_feature)

        for dep in embedded_deployments:
            dep_feature = _build_deployment_feature(
                dep,
                header,
                facility_record_id=facility_record_id,
                observation_record_id=obs_feature["id"],
                facility_geometry=facility_geometry,
                index=deployment_counter,
            )
            deployment_counter += 1
            features.append(dep_feature)

    if not observations and deployments:
        for dep_index, deployment in enumerate(deployments):
            dep_feature = _build_deployment_feature(
                deployment,
                header,
                facility_record_id=facility_record_id,
                observation_record_id=None,
                facility_geometry=facility_geometry,
                index=dep_index,
            )
            features.append(dep_feature)

    if facility_feature is not None:
        facility_wmdr2 = _as_dict(facility_feature.get("properties", {}).get("wmdr2"))
        if facility_wmdr2:
            facility_wmdr2["observationIds"] = [feature["id"] for feature in observation_features]
            facility_wmdr2["deploymentIds"] = [
                feature["id"] for feature in features if feature.get("properties", {}).get("type") == "deployment"
            ]
            facility_feature["properties"]["wmdr2"] = _clean_none(facility_wmdr2)

    collection = _feature_collection(features)
    DISCOVERY_POLICY = previous_policy
    CODE_LIST_LABELS = previous_labels
    return collection


def convert_file(
    inp: Path,
    out_dir: Path,
    *,
    discovery_policy: Optional[Dict[str, Dict[str, List[str]]]] = None,
    code_list_labels: Optional[Dict[str, Dict[str, str]]] = None,
) -> Path:
    payload = _load_json(inp)
    collection = convert_payload(
        payload,
        source_name=inp.name,
        discovery_policy=discovery_policy,
        code_list_labels=code_list_labels,
    )
    rel_name = inp.with_suffix(".geojson").name
    out_path = out_dir / rel_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(collection, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert WMDR 1.0 JSON exports to WMDR2-oriented GeoJSON records compliant with OGC API Records Part 1."
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--source", type=Path, default=None, help="Input JSON file or directory.")
    parser.add_argument("--target", type=Path, default=None, help="Output directory.")
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN, help="Input file glob pattern.")
    parser.add_argument("--recursive", action="store_true", help="Scan recursively (default).")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Do not scan recursively.")
    parser.set_defaults(recursive=True)
    args = parser.parse_args()

    cfg = _cfg_section(_load_config(args.config))
    discovery_policy = _normalize_discovery_policy(cfg)
    code_list_labels = _load_code_list_labels(cfg, base_dir=args.config.parent)
    source = args.source or Path(str(cfg.get("source", "")))
    target = args.target or Path(str(cfg.get("target", "")))
    pattern = args.pattern or str(cfg.get("pattern", DEFAULT_PATTERN))
    recursive = bool(cfg.get("recursive", args.recursive)) if "recursive" in cfg else args.recursive

    if not str(source):
        raise SystemExit("Missing source. Set convert_wmdr10_json_to_wmdr2_geojson.source in config.yaml or pass --source.")
    if not str(target):
        raise SystemExit("Missing target. Set convert_wmdr10_json_to_wmdr2_geojson.target in config.yaml or pass --target.")

    files = _iter_json_files(source, pattern=pattern, recursive=recursive)
    if not files:
        raise SystemExit(f"No JSON files found under {source!s}")

    target.mkdir(parents=True, exist_ok=True)
    written = []
    for inp in files:
        out = convert_file(inp, target, discovery_policy=discovery_policy, code_list_labels=code_list_labels)
        written.append(out)

    print(f"Wrote {len(written)} WMDR2 GeoJSON file(s) to {target}")


if __name__ == "__main__":
    main()
