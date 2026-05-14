#!/usr/bin/env python3
"""
convert_wmdr10_json_to_wmdr2_geojson.py

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


def _last_segment(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.rstrip("/#")
    if "/" in raw:
        return raw.rsplit("/", 1)[-1]
    if "#" in raw:
        return raw.rsplit("#", 1)[-1]
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

    if isinstance(raw, dict) and isinstance(raw.get("coordinates"), list):
        coords = raw.get("coordinates")
        if len(coords) >= 2:
            return coords  # already GeoJSON style

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
        scheme = _uri_parent(raw)
        concept_id = _last_segment(raw)
        if not concept_id:
            continue
        grouped.setdefault(scheme, []).append({"id": concept_id, "url": raw})
    themes = [{"scheme": scheme, "concepts": concepts} for scheme, concepts in grouped.items()]
    return _uniq_dicts(themes)


def _keywords_from_values(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values:
        if raw is None:
            continue
        if isinstance(raw, str):
            candidate = _last_segment(raw) or raw
            candidate = candidate.replace("_", " ").strip()
            if not candidate:
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

    info = payload.get("contactInfo") if isinstance(payload.get("contactInfo"), dict) else {}
    if not isinstance(info, dict):
        info = {}

    phone_obj = info.get("phone") if isinstance(info.get("phone"), dict) else {}
    voices = _as_list(phone_obj.get("voice"))
    phones = []
    for voice in voices:
        if isinstance(voice, str) and voice.strip():
            phones.append({"value": voice.strip()})
    if phones:
        contact["phones"] = phones

    address_obj = info.get("address") if isinstance(info.get("address"), dict) else {}
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
    online = info.get("onlineResource") if isinstance(info.get("onlineResource"), dict) else {}
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
        if not isinstance(item, dict):
            continue
        record: Dict[str, Any] = {}
        affiliations = [x for x in _as_list(item.get("programAffiliation")) if isinstance(x, str) and x.strip()]
        if affiliations:
            record["programAffiliation"] = affiliations
        psfi = item.get("programSpecificFacilityId")
        if isinstance(psfi, str) and psfi.strip():
            record["programSpecificFacilityId"] = psfi.strip()
        statuses = _normalize_reporting_status(item.get("reportingStatus"))
        if statuses:
            record["reportingStatus"] = statuses
        if record:
            out.append(record)
    return _uniq_dicts(out)


def _normalize_simple_timed_value(value: Any, *, value_key: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    actual = value.get(value_key)
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


def _normalize_temporal_geometry(current: Any, history: Any = None) -> Optional[Dict[str, Any]]:
    entries: List[Tuple[Optional[str], str, Dict[str, Any]]] = []

    def add_entry(item: Any) -> None:
        if not isinstance(item, dict):
            return
        geometry = _point_from_pos(item.get("geometry") or item.get("geoLocation") or item.get("pos") or item)
        if geometry is None:
            return
        start, end = _extract_interval(item)
        interval = _time_interval(start, end)
        if interval is None:
            interval = {"interval": ["..", ".."]}
        start_key = interval["interval"][0]
        entries.append((None if start_key == ".." else str(start_key), json.dumps(geometry, sort_keys=True), {"time": interval, "geometry": geometry}))

    add_entry(current)
    for item in _as_list(history):
        add_entry(item)

    if not entries:
        return None

    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for _, payload_key, entry in sorted(entries, key=lambda item: (item[0] is None, item[0] or "", item[1])):
        marker = json.dumps(entry, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(entry)

    datetimes = [entry["time"]["interval"] for entry in deduped]
    geometries = [entry["geometry"] for entry in deduped]
    out = {"datetime": datetimes, "geometry": geometries}
    return out


def _current_geometry_from_temporal_geometry(temporal_geometry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(temporal_geometry, dict):
        return None
    geometries = temporal_geometry.get("geometry")
    datetimes = temporal_geometry.get("datetime")
    if not isinstance(geometries, list) or not geometries:
        return None
    if not isinstance(datetimes, list) or len(datetimes) != len(geometries):
        return geometries[-1] if isinstance(geometries[-1], dict) else None

    def rank(interval: Any) -> Tuple[int, str]:
        if isinstance(interval, list) and len(interval) == 2:
            end = interval[1]
            if end == "..":
                return (0, "")
            return (1, str(end))
        return (2, "")

    paired = list(zip(datetimes, geometries))
    paired.sort(key=lambda pair: rank(pair[0]))
    return paired[0][1] if paired else None


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
            report = dg.get("reporting") if isinstance(dg.get("reporting"), dict) else {}
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


def _facility_geometry_and_extension(facility: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    temporal_geometry = _normalize_temporal_geometry(
        facility.get("geospatialLocation"),
        _first_non_empty(
            facility.get("geospatialLocationHistory"),
            facility.get("geometryHistory"),
            facility.get("temporalGeometry"),
        ),
    )
    geometry = _current_geometry_from_temporal_geometry(temporal_geometry)
    return geometry, temporal_geometry


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

    geometry, temporal_geometry = _facility_geometry_and_extension(facility)
    feature["geometry"] = geometry

    feature["time"] = _time_interval(facility.get("dateEstablished"), facility.get("dateClosed"))

    ogc_contacts, ext_contacts = _collect_contacts(facility.get("responsibleParty"), _record_owner_contact(header))
    if ogc_contacts:
        feature["properties"]["contacts"] = ogc_contacts

    external_ids = _uniq_dicts(
        [
            item
            for item in [
                _external_id(identifier, "WMO:WIGOS"),
                _external_id(record_id, "WMDR2:record"),
            ]
            if item
        ]
    )
    if external_ids:
        feature["properties"]["externalIds"] = external_ids

    keywords = _keywords_from_values(
        [
            identifier,
            facility.get("name"),
            facility.get("facilityType"),
            facility.get("facilitySet"),
            facility.get("wmoRegion"),
            facility.get("onlineResource", {}).get("url") if isinstance(facility.get("onlineResource"), dict) else None,
            *(entry.get("programAffiliation") for entry in _as_list(facility.get("programAffiliation")) if isinstance(entry, dict)),
        ]
    )
    if keywords:
        feature["properties"]["keywords"] = keywords

    formal_uris = _collect_facility_formal_uris(facility)
    themes = _themes_from_uris(formal_uris)
    if themes:
        feature["properties"]["themes"] = themes

    links: List[Dict[str, Any]] = []
    if isinstance(facility.get("facilityType"), str):
        links.append(_canonical_type_link(facility["facilityType"]))
    online_resource = facility.get("onlineResource")
    if isinstance(online_resource, dict) and isinstance(online_resource.get("url"), str):
        links.append(_about_link(online_resource["url"], title="Facility online resource"))
    if links:
        feature["links"] = _uniq_dicts(links)

    facility_ext: Dict[str, Any] = {
        "class": "facility",
        "id": record_id,
        "identifier": identifier,
        "name": facility.get("name"),
        "temporalPeriod": _time_interval(facility.get("dateEstablished"), facility.get("dateClosed")),
        "description": description,
        "temporalGeometry": temporal_geometry,
        "contact": ext_contacts,
        "facilitySet": facility.get("facilitySet"),
        "facilityTypes": [x for x in _as_list(facility.get("facilityType")) if isinstance(x, str)],
        "additionalWsi": [x for x in _as_list(facility.get("additionalWSI")) if isinstance(x, str)],
        "territory": _normalize_simple_timed_value(facility.get("territory"), value_key="territoryName"),
        "wmoRegion": facility.get("wmoRegion"),
        "programAffiliation": _normalize_program_affiliation(facility.get("programAffiliation")),
        "climateZone": _normalize_simple_timed_value(facility.get("climateZone"), value_key="climateZone"),
        "surfaceCover": _normalize_simple_timed_value(facility.get("surfaceCover"), value_key="surfaceCover"),
        "topographyBathymetry": _normalize_simple_timed_value(facility.get("topographyBathymetry"), value_key="localTopography"),
    }
    feature["properties"]["wmdr2"] = _clean_none({"facility": facility_ext})

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
    manufacturers = [dep.get("manufacturer") for dep in deployments if isinstance(dep.get("manufacturer"), str)]
    models = [dep.get("model") for dep in deployments if isinstance(dep.get("model"), str)]
    bits = []
    if observed:
        bits.append(f"Observed property {observed}")
    if geom_type:
        bits.append(f"geometry type {geom_type}")
    if manufacturers or models:
        proc = " / ".join([x for x in [", ".join(sorted(set(manufacturers))) if manufacturers else None,
                                      ", ".join(sorted(set(models))) if models else None] if x])
        if proc:
            bits.append(f"deployment procedure {proc}")
    return "; ".join(bits) if bits else None


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
    title = _last_segment(observed_property) or f"Observation {index + 1}"
    feature["properties"]["title"] = title

    description = _observation_description(observation, deployments)
    if description:
        feature["properties"]["description"] = description

    feature["geometry"] = facility_geometry
    feature["time"] = _time_interval(observation.get("beginPosition"), observation.get("endPosition"))

    record_owner = _record_owner_contact(header)
    metadata_contact = observation.get("metadata", {}).get("contact") if isinstance(observation.get("metadata"), dict) else None
    ogc_contacts, ext_contacts = _collect_contacts(metadata_contact, record_owner)
    if ogc_contacts:
        feature["properties"]["contacts"] = ogc_contacts

    external_ids = _uniq_dicts(
        [
            item
            for item in [
                _external_id(str(_first_non_empty(observation.get("@gml:id"), observation.get("@id"), record_id)), "WMDR1:gml"),
                _external_id(record_id, "WMDR2:record"),
                _external_id(observation.get("facility"), "WMO:facility-uri"),
                _external_id(observed_property, "WMO:observed-property-uri"),
            ]
            if item
        ]
    )
    if external_ids:
        feature["properties"]["externalIds"] = external_ids

    observation_program_affiliations = observation.get("programAffiliation") if isinstance(observation.get("programAffiliation"), list) else []
    keywords = _keywords_from_values(
        [
            observed_property,
            observation.get("type"),
            *observation_program_affiliations,
            *(dep.get("manufacturer") for dep in deployments if isinstance(dep, dict)),
            *(dep.get("model") for dep in deployments if isinstance(dep, dict)),
        ]
    )
    if keywords:
        feature["properties"]["keywords"] = keywords

    formal_uris = _collect_observation_formal_uris(observation, deployments)
    themes = _themes_from_uris(formal_uris)
    if themes:
        feature["properties"]["themes"] = themes

    links: List[Dict[str, Any]] = []
    if isinstance(observed_property, str):
        links.append(_canonical_type_link(observed_property))
    if links:
        feature["links"] = _uniq_dicts(links)

    inferred_exchange, reporting_schedule = _derive_observation_level_reporting(deployments)
    related_records = []
    if facility_record_id:
        related_records.append(facility_record_id)

    deployment_ids = [_deployment_record_id(dep, dep_index) for dep_index, dep in enumerate(deployments)]
    related_records.extend(deployment_ids)

    obs_ext: Dict[str, Any] = {
        "class": "observation",
        "id": record_id,
        "facility": observation.get("facility"),
        "relatedRecords": related_records,
        "observationType": observation.get("type"),
        "observedProperty": observed_property,
        "observedVariableCoordinates": observation.get("observedVariableCoordinates"),
        "temporalExtent": _time_interval(observation.get("beginPosition"), observation.get("endPosition")),
        "contact": ext_contacts,
        "programAffiliation": [x for x in _as_list(observation.get("programAffiliation")) if isinstance(x, str)],
        "internationalExchange": inferred_exchange,
        "internationalReportingSchedule": reporting_schedule,
        "result": observation.get("result"),
    }
    feature["properties"]["wmdr2"] = _clean_none({"observation": obs_ext})

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
        reporting = dg.get("reporting") if isinstance(dg.get("reporting"), dict) else {}
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
    title = " ".join(bit for bit in title_bits if bit) or _last_segment(deployment.get("observedProperty")) or f"Deployment {index + 1}"
    feature["properties"]["title"] = title

    description_bits = [
        _last_segment(deployment.get("sourceOfObservation")),
        deployment.get("configuration") if isinstance(deployment.get("configuration"), str) else None,
        _last_segment(deployment.get("observingMethod")),
    ]
    description = "; ".join(bit for bit in description_bits if bit)
    if description:
        feature["properties"]["description"] = description

    feature["geometry"] = facility_geometry
    feature["time"] = _time_interval(deployment.get("beginPosition"), deployment.get("endPosition"))

    ogc_contacts, ext_contacts = _collect_contacts(deployment.get("responsibleParty"), _record_owner_contact(header))
    if ogc_contacts:
        feature["properties"]["contacts"] = ogc_contacts

    external_ids = _uniq_dicts(
        [
            item
            for item in [
                _external_id(str(_first_non_empty(deployment.get("@gml:id"), deployment.get("@id"), record_id)), "WMDR1:gml"),
                _external_id(record_id, "WMDR2:record"),
                _external_id(deployment.get("facility"), "WMO:facility-uri"),
                _external_id(deployment.get("observedProperty"), "WMO:observed-property-uri"),
                _external_id(deployment.get("serialNumber"), "instrument:serialNumber"),
            ]
            if item
        ]
    )
    if external_ids:
        feature["properties"]["externalIds"] = external_ids

    keywords = _keywords_from_values(
        [
            deployment.get("manufacturer"),
            deployment.get("model"),
            deployment.get("serialNumber"),
            deployment.get("sourceOfObservation"),
            deployment.get("observingMethod"),
            deployment.get("localReferenceSurface"),
            deployment.get("observedProperty"),
        ]
    )
    if keywords:
        feature["properties"]["keywords"] = keywords

    formal_uris = _collect_deployment_formal_uris(deployment)
    themes = _themes_from_uris(formal_uris)
    if themes:
        feature["properties"]["themes"] = themes

    links: List[Dict[str, Any]] = []
    obs_method = deployment.get("observingMethod")
    if isinstance(obs_method, str):
        links.append(_canonical_type_link(obs_method))
    if links:
        feature["links"] = _uniq_dicts(links)

    procedure = _clean_none(
        {
            "observingMethod": deployment.get("observingMethod"),
            "instrumentManufacturer": deployment.get("manufacturer"),
            "instrumentModel": deployment.get("model"),
            "instrumentSerialNumber": deployment.get("serialNumber"),
        }
    )

    vertical_geometry = _clean_none(
        {
            "heightAboveLocalReferenceSurface": _parse_quantity(deployment.get("heightAboveLocalReferenceSurface")),
            "localReferenceSurface": deployment.get("localReferenceSurface"),
        }
    )

    reporting_status = _normalize_reporting_status([deployment.get("instrumentOperatingStatus")] if deployment.get("instrumentOperatingStatus") is not None else [])

    wmdr2_dep = _clean_none(
        {
            "class": "deployment",
            "id": record_id,
            "relatedRecords": [x for x in [facility_record_id, observation_record_id] if x],
            "facility": deployment.get("facility"),
            "observedProperty": deployment.get("observedProperty"),
            "deploymentType": deployment.get("type"),
            "temporalExtent": _time_interval(deployment.get("beginPosition"), deployment.get("endPosition")),
            "temporalReportingStatus": reporting_status,
            "sourceOfObservation": deployment.get("sourceOfObservation"),
            "exposure": deployment.get("exposure"),
            "representativeness": deployment.get("representativeness"),
            "configuration": deployment.get("configuration"),
            "heightAboveLocalReferenceSurface": _parse_quantity(deployment.get("heightAboveLocalReferenceSurface")),
            "localReferenceSurface": deployment.get("localReferenceSurface"),
            "temporalGeometry": vertical_geometry,
            "procedure": procedure,
            "contact": ext_contacts,
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

    feature["properties"]["wmdr2"] = {"deployment": wmdr2_dep}
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


def convert_payload(payload: Any, *, source_name: str = "input.json") -> Dict[str, Any]:
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

    return _feature_collection(features)


def convert_file(inp: Path, out_dir: Path) -> Path:
    payload = _load_json(inp)
    collection = convert_payload(payload, source_name=inp.name)
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
        out = convert_file(inp, target)
        written.append(out)

    print(f"Wrote {len(written)} WMDR2 GeoJSON file(s) to {target}")


if __name__ == "__main__":
    main()
