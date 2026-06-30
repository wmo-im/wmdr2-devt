#!/usr/bin/env python3
"""
convert_wmdr2_json_to_wmdr2_gc_dar.py

Convert WMDR2 v0.2.5 full facility records to a compact GC-DAR record.

GC-DAR means Global Catalogue Discovery, Access and Retrieval.  The output is
not an editing representation.  It is a derived OGC Records-friendly current
state projection with enough provenance information to show where it came from
and enough access/retrieval links to fetch the full WMDR2 record.

Typical workflow:

    XML -> WMDR10 -> WMDR2 full -> WMDR2 catalogue
                              -> WMDR2 GC-DAR

The converter accepts a single JSON file or a directory of JSON files and writes
one derived JSON Feature per input record.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - optional CLI convenience
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

DEFAULT_PATTERN = "*.json"
WMDR2_CORE_CONF = "http://wigos.wmo.int/spec/wmdr/2/conf/core"
WMDR2_GC_DAR_CONF = "http://wigos.wmo.int/spec/wmdr/2/conf/gc-dar"
JSON_MEDIA_TYPE = "application/json"
GEOJSON_MEDIA_TYPE = "application/geo+json"
OPEN_END_VALUES = {"..", "", None}


@dataclass(frozen=True)
class GcDarPaths:
    source: Path
    target: Path
    pattern: str = DEFAULT_PATTERN
    recursive: bool = True
    full_record_href_template: str | None = None
    dar_record_href_template: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _flatten(value: Any) -> list[Any]:
    out: list[Any] = []
    for item in _as_list(value):
        if isinstance(item, list):
            out.extend(_flatten(item))
        else:
            out.append(item)
    return out


def _clean_empty(obj: Any) -> Any:
    """Remove empty values while preserving meaningful false/zero values."""
    if isinstance(obj, dict):
        cleaned = {key: _clean_empty(value) for key, value in obj.items()}
        return {key: value for key, value in cleaned.items() if value not in (None, "", [], {})}
    if isinstance(obj, list):
        cleaned = [_clean_empty(item) for item in obj]
        return [item for item in cleaned if item not in (None, "", [], {})]
    return obj


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _content_hash(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _dedupe(items: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for item in items:
        if item in (None, "", [], {}):
            continue
        marker = _stable_json(item)
        if marker not in seen:
            seen.add(marker)
            out.append(item)
    return out


def _format_template(template: str | None, *, record_id: str, filename: str) -> str | None:
    if not template:
        return None
    plain_id = record_id.removeprefix("facility:")
    return template.format(record_id=record_id, plain_id=plain_id, filename=filename)


def _iter_json_files(root: Path, *, pattern: str, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".json" else []
    if not root.is_dir():
        return []
    walker = root.rglob if recursive else root.glob
    return sorted(path for path in walker(pattern) if path.is_file() and path.suffix.lower() == ".json")


def _relative_output_path(source_file: Path, source_root: Path) -> Path:
    if source_root.is_file():
        return Path(source_file.name)
    try:
        return source_file.relative_to(source_root)
    except ValueError:
        return Path(source_file.name)


def _date_key(value: Any) -> str:
    if value in OPEN_END_VALUES:
        return "9999-99-99"
    return str(value)


def _latest_dict(items: Iterable[Any], *, fallback_key: str | None = None) -> dict[str, Any] | None:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        date = item.get("date")
        if date in OPEN_END_VALUES and fallback_key:
            date = item.get(fallback_key)
        candidates.append((_date_key(date), item))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]


def _latest_value(items: Iterable[Any], key: str) -> Any | None:
    candidate = _latest_dict(items)
    if candidate is None:
        return None
    return candidate.get(key)


def _interval(series: dict[str, Any]) -> list[Any]:
    time = series.get("time")
    if not isinstance(time, dict):
        return []
    interval = time.get("interval")
    return interval if isinstance(interval, list) else []


def _is_current_series(series: dict[str, Any]) -> bool:
    interval = _interval(series)
    if len(interval) < 2:
        return True
    return interval[1] in OPEN_END_VALUES


def _series_sort_key(series: dict[str, Any]) -> str:
    interval = _interval(series)
    end = interval[1] if len(interval) > 1 else None
    start = interval[0] if interval else series.get("date")
    return _date_key(end if end not in OPEN_END_VALUES else start)


def _current_or_latest_series(series_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the current state: open-ended series, or latest closed series if none are current."""
    current = [series for series in series_items if _is_current_series(series)]
    if current:
        return current
    if not series_items:
        return []
    return [sorted(series_items, key=_series_sort_key)[-1]]


def _programme_rows(items: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, list):
            rows.extend(_programme_rows(item))
        elif isinstance(item, dict):
            programmes = _flatten(item.get("programAffiliation"))
            if not programmes and item.get("program"):
                programmes = _flatten(item.get("program"))
            for programme in programmes:
                rows.append(
                    _clean_empty(
                        {
                            "date": item.get("date"),
                            "programAffiliation": programme,
                            "reportingStatus": item.get("reportingStatus"),
                        }
                    )
                )
        elif item not in (None, "", [], {}):
            rows.append({"programAffiliation": item})
    latest_by_programme: dict[str, dict[str, Any]] = {}
    for row in rows:
        programme = row.get("programAffiliation")
        if not programme:
            continue
        key = str(programme)
        previous = latest_by_programme.get(key)
        if previous is None or _date_key(row.get("date")) >= _date_key(previous.get("date")):
            latest_by_programme[key] = row
    return list(latest_by_programme.values())


def _program_affiliations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return latest program affiliations as explicit program/status objects.

    This avoids aligned arrays whose semantics depend on positional coupling.
    The object shape is intentionally extensible for future GC-DAR facets such as
    program-specific identifiers or affiliation dates.
    """
    affiliations: list[dict[str, Any]] = []
    for row in rows:
        programme = row.get("programAffiliation")
        if programme in (None, "", [], {}):
            continue
        affiliations.append(
            _clean_empty(
                {
                    "program": programme,
                    "reportingStatus": row.get("reportingStatus", "unknown"),
                    "date": row.get("date"),
                }
            )
        )
    return affiliations


def _reporting_definitions(properties: dict[str, Any]) -> list[dict[str, Any]]:
    values = properties.get("reporting")
    if not isinstance(values, list):
        values = properties.get("reportingDefinitions")
    return [item for item in _as_list(values) if isinstance(item, dict)]


def _reporting_map(properties: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for reporting in _reporting_definitions(properties):
        reporting_id = reporting.get("id")
        if reporting_id:
            out[str(reporting_id)] = reporting
    return out


def _deployment_map(properties: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for deployment in _as_list(properties.get("deployments")):
        if isinstance(deployment, dict) and deployment.get("id"):
            out[str(deployment["id"])] = deployment
    return out


def _instrument_map(properties: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for instrument in _as_list(properties.get("instruments")):
        if isinstance(instrument, dict) and instrument.get("id"):
            out[str(instrument["id"])] = instrument
    return out


def _latest_official_status(series: dict[str, Any]) -> Any | None:
    values = _as_list(series.get("officialStatus"))
    if not values:
        return None
    if len(values) == 1 and not isinstance(values[0], dict):
        return values[0]
    return _latest_value(values, "officialStatus")


def _latest_reporting(series: dict[str, Any], reporting_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    series_reporting = [item for item in _as_list(series.get("reporting")) if isinstance(item, dict)]
    latest = _latest_dict(series_reporting)
    if latest is None:
        return {}
    reporting_id = latest.get("reporting")
    definition = reporting_by_id.get(str(reporting_id)) if reporting_id else None
    merged: dict[str, Any] = {}
    if definition:
        merged.update(definition)
    merged.update(latest)
    # The id/reference is an internal full-record indirection; GC-DAR resolves it.
    merged.pop("id", None)
    merged.pop("reporting", None)
    return _clean_empty(merged)


def _latest_observing_configuration(
    series: dict[str, Any], deployment_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    configs = [item for item in _as_list(series.get("observingConfigurations")) if isinstance(item, dict)]
    latest = _latest_dict(configs)
    if latest is None:
        return {}
    deployment_id = latest.get("deployment")
    deployment = deployment_by_id.get(str(deployment_id)) if deployment_id else None
    instrument = deployment.get("instrument") if isinstance(deployment, dict) else None
    return _clean_empty(
        {
            "date": latest.get("date"),
            "deployment": deployment_id,
            "instrument": instrument,
            "observingMethod": latest.get("observingMethod"),
        }
    )


def _series_summary(
    series: dict[str, Any],
    *,
    reporting_by_id: dict[str, dict[str, Any]],
    deployment_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    observed_feature_raw = series.get("observedFeature")
    observed_feature: dict[str, Any] = observed_feature_raw if isinstance(observed_feature_raw, dict) else {}
    reporting = _latest_reporting(series, reporting_by_id)
    observing_configuration = _latest_observing_configuration(series, deployment_by_id)
    return _clean_empty(
        {
            "id": series.get("id"),
            "title": series.get("title"),
            "time": series.get("time"),
            "current": _is_current_series(series),
            "observedProperty": series.get("observedProperty"),
            "observedGeometry": series.get("observedGeometry"),
            "observedFeature": {
                "domain": observed_feature.get("domain"),
                "domainFeature": observed_feature.get("domainFeature"),
                "featureName": observed_feature.get("featureName"),
            },
            "programAffiliation": _dedupe(_flatten(series.get("programAffiliation"))),
            "applicationArea": _dedupe(_flatten(series.get("applicationArea"))),
            "sourceOfObservation": series.get("sourceOfObservation"),
            "officialStatus": _latest_official_status(series),
            "deployment": observing_configuration.get("deployment"),
            "instrument": observing_configuration.get("instrument"),
            "observingMethod": observing_configuration.get("observingMethod"),
            "observingConfigurationDate": observing_configuration.get("date"),
            "uom": reporting.get("uom"),
            "internationalExchange": reporting.get("internationalExchange"),
            "temporalAggregate": reporting.get("temporalAggregate"),
            "levelOfData": reporting.get("levelOfData"),
            "dataPolicy": reporting.get("dataPolicy"),
            "timeliness": reporting.get("timeliness"),
            "reportingDate": reporting.get("date"),
            "reportingStrategy": reporting.get("strategy"),
        }
    )


def _deployment_summary(deployment: dict[str, Any]) -> dict[str, Any]:
    return _clean_empty(
        {
            "id": deployment.get("id"),
            "date": deployment.get("date"),
            "instrument": deployment.get("instrument"),
            "operatingStatus": deployment.get("operatingStatus"),
            "exposure": deployment.get("exposure"),
            "geometry": deployment.get("geometry"),
        }
    )


def _instrument_summary(instrument: dict[str, Any]) -> dict[str, Any]:
    return _clean_empty(
        {
            "id": instrument.get("id"),
            "title": instrument.get("title"),
            "manufacturer": instrument.get("manufacturer"),
            "model": instrument.get("model"),
            "observingMethods": instrument.get("observingMethods"),
            "observedProperty": instrument.get("observedProperty"),
            "observedGeometry": instrument.get("observedGeometry"),
        }
    )


def _make_links(
    *,
    record_id: str,
    filename: str,
    existing_links: Iterable[Any],
    full_record_href_template: str | None,
    dar_record_href_template: str | None,
) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    dar_href = _format_template(dar_record_href_template, record_id=record_id, filename=filename)
    full_href = _format_template(full_record_href_template, record_id=record_id, filename=filename)
    if dar_href:
        links.append({"rel": "self", "href": dar_href, "type": GEOJSON_MEDIA_TYPE, "profile": WMDR2_GC_DAR_CONF})
    if full_href:
        links.append(
            {
                "rel": "alternate",
                "href": full_href,
                "type": JSON_MEDIA_TYPE,
                "profile": WMDR2_CORE_CONF,
                "title": "Full WMDR2 record",
            }
        )
    for link in existing_links:
        if isinstance(link, dict):
            links.append(copy.deepcopy(link))
    return _dedupe(links)  # type: ignore[return-value]


def _source_links(properties: dict[str, Any], record: dict[str, Any]) -> list[Any]:
    links: list[Any] = []
    links.extend(_as_list(record.get("links")))
    links.extend(_as_list(properties.get("links")))
    return links


def convert_record_to_gc_dar(
    record: dict[str, Any],
    *,
    source_filename: str = "record.json",
    derived_at: str | None = None,
    full_record_href_template: str | None = None,
    dar_record_href_template: str | None = None,
) -> dict[str, Any]:
    """Return a compact GC-DAR discovery/provenance/retrieval Feature."""
    if record.get("type") != "Feature":
        raise ValueError("Input is not a WMDR2 Feature record.")
    record_id = str(record.get("id") or "").strip()
    if not record_id:
        raise ValueError("Input record has no id.")
    properties = record.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Input record has no object-valued properties member.")

    all_series = [item for item in _as_list(properties.get("observationSeries")) if isinstance(item, dict)]
    current_series = _current_or_latest_series(all_series)
    deployment_by_id = _deployment_map(properties)
    instrument_by_id = _instrument_map(properties)
    reporting_by_id = _reporting_map(properties)

    observation_summaries = [
        _series_summary(series, reporting_by_id=reporting_by_id, deployment_by_id=deployment_by_id)
        for series in current_series
    ]

    current_deployment_ids = _dedupe(series.get("deployment") for series in observation_summaries)
    current_instrument_ids = _dedupe(series.get("instrument") for series in observation_summaries)
    current_deployments = [
        _deployment_summary(deployment_by_id[deployment_id])
        for deployment_id in current_deployment_ids
        if isinstance(deployment_id, str) and deployment_id in deployment_by_id
    ]
    current_instruments = [
        _instrument_summary(instrument_by_id[instrument_id])
        for instrument_id in current_instrument_ids
        if isinstance(instrument_id, str) and instrument_id in instrument_by_id
    ]

    observed_features: list[dict[str, Any]] = []
    for item in current_series:
        observed_feature = _as_dict(item.get("observedFeature"))
        if observed_feature:
            observed_features.append(observed_feature)

    source_summary: dict[str, Any] = _as_dict(properties.get("summary"))

    territory_items = _as_list(properties.get("territory"))
    latest_territory = _latest_value(territory_items, "territory") if territory_items else None
    if latest_territory is None:
        flat_territory = _flatten(properties.get("territory"))
        latest_territory = flat_territory[-1] if flat_territory else None
    if latest_territory is None:
        latest_territory = source_summary.get("latestTerritory") or source_summary.get("territory")
        if isinstance(latest_territory, list):
            latest_territory = _flatten(latest_territory)[-1] if _flatten(latest_territory) else None

    programme_rows = _programme_rows(_as_list(properties.get("programAffiliation")))
    if not programme_rows:
        programs_obj = _as_dict(source_summary.get("programs"))
        affiliations = _flatten(programs_obj.get("programAffiliation"))
        statuses = _flatten(programs_obj.get("reportingStatus"))
        dates = _flatten(programs_obj.get("date"))
        if programs_obj:
            programme_rows = [
                _clean_empty(
                    {
                        "programAffiliation": affiliation,
                        "reportingStatus": statuses[index] if index < len(statuses) else None,
                        "date": dates[index] if index < len(dates) else None,
                    }
                )
                for index, affiliation in enumerate(affiliations)
            ]
    if not programme_rows:
        source_program_affiliation = _as_list(source_summary.get("programAffiliation"))
        if any(isinstance(item, dict) for item in source_program_affiliation):
            # Current GC-DAR shape: summary.programAffiliation is an array of objects
            # such as {"program": "GAWregional", "reportingStatus": "operational"}.
            programme_rows = _programme_rows(source_program_affiliation)
        else:
            # Legacy fallback: independent summary arrays from the first GC-DAR draft.
            affiliations = _dedupe(_flatten(source_program_affiliation))
            statuses = _flatten(source_summary.get("reportingStatus"))
            programme_rows = [
                _clean_empty(
                    {
                        "programAffiliation": affiliation,
                        "reportingStatus": statuses[index] if index < len(statuses) else None,
                    }
                )
                for index, affiliation in enumerate(affiliations)
            ]

    discovery_facets = _clean_empty(
        {
            "facilitySets": _as_list(properties.get("facilitySets")),
            "territory": latest_territory,
            "wmoRegion": properties.get("wmoRegion"),
            "facilityType": properties.get("facilityType"),
            "programAffiliation": _program_affiliations(programme_rows),
            "observedProperty": _dedupe(series.get("observedProperty") for series in current_series),
            "observedGeometry": _dedupe(series.get("observedGeometry") for series in current_series),
            "observedFeatureDomain": _dedupe(
                feature.get("domain") for feature in observed_features if isinstance(feature, dict)
            ),
            "observedFeatureDomainFeature": _dedupe(
                feature.get("domainFeature") for feature in observed_features if isinstance(feature, dict)
            ),
            "observedFeatureName": _dedupe(
                feature.get("featureName") for feature in observed_features if isinstance(feature, dict)
            ),
            "applicationArea": _dedupe(area for series in current_series for area in _flatten(series.get("applicationArea"))),
            "officialStatus": _dedupe(series.get("officialStatus") for series in observation_summaries),
            "observationSeriesCount": len(observation_summaries),
        }
    )

    filename = Path(source_filename).name
    links = _make_links(
        record_id=record_id,
        filename=filename,
        existing_links=_source_links(properties, record),
        full_record_href_template=full_record_href_template,
        dar_record_href_template=dar_record_href_template,
    )
    full_href = _format_template(full_record_href_template, record_id=record_id, filename=filename)
    dar_href = _format_template(dar_record_href_template, record_id=record_id, filename=filename)

    dar = {
        "type": "Feature",
        "id": record_id,
        "geometry": copy.deepcopy(record.get("geometry")),
        "time": copy.deepcopy(record.get("time")),
        "conformsTo": [WMDR2_GC_DAR_CONF],
        "properties": _clean_empty(
            {
                "type": "facility",
                "profile": "wmdr2-gc-dar",
                "modelVersion": "0.2.5",
                "sourceProfile": "wmdr2-full",
                "title": properties.get("title"),
                "description": properties.get("description"),
                "created": properties.get("created"),
                "updated": properties.get("updated"),
                "keywords": _as_list(properties.get("keywords")),
                "contacts": copy.deepcopy(properties.get("contacts")),
                **discovery_facets,
                "observationSeries": observation_summaries,
                "deployments": current_deployments,
                "instruments": current_instruments,
                "provenance": {
                    "derivedFrom": record_id,
                    "sourceProfile": "wmdr2-full",
                    "sourceConformsTo": copy.deepcopy(record.get("conformsTo")),
                    "sourceContentHash": _content_hash(record),
                    "derivedAt": derived_at or _utc_now(),
                    "sourceFile": filename,
                },
                "retrieval": {
                    "fullRecord": {
                        "href": full_href,
                        "type": JSON_MEDIA_TYPE,
                        "profile": WMDR2_CORE_CONF,
                        "title": "Full WMDR2 record",
                    }
                    if full_href
                    else None,
                    "darRecord": {
                        "href": dar_href,
                        "type": GEOJSON_MEDIA_TYPE,
                        "profile": WMDR2_GC_DAR_CONF,
                        "title": "GC-DAR discovery record",
                    }
                    if dar_href
                    else None,
                },
            }
        ),
        "links": links,
    }
    return _clean_empty(dar)


def convert_files(paths: GcDarPaths) -> list[Path]:
    written: list[Path] = []
    for source_file in _iter_json_files(paths.source, pattern=paths.pattern, recursive=paths.recursive):
        payload = json.loads(source_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        dar = convert_record_to_gc_dar(
            payload,
            source_filename=source_file.name,
            full_record_href_template=paths.full_record_href_template,
            dar_record_href_template=paths.dar_record_href_template,
        )
        rel = _relative_output_path(source_file, paths.source)
        target_file = paths.target / rel
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(json.dumps(dar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(target_file)
    return written


def _paths_from_config(config_path: Path) -> GcDarPaths:
    if yaml is None:
        raise SystemExit("PyYAML is required when using --config.")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = (
        cfg.get("convert_wmdr2_json_to_wmdr2_gc_dar")
        or cfg.get("convert_wmdr2_json_to_gc_dar")
        or cfg.get("convert_wmdr2_full_to_gc_dar")
        or {}
    )
    if not isinstance(section, dict):
        raise SystemExit("Missing convert_wmdr2_json_to_wmdr2_gc_dar config section.")
    base = config_path.parent
    source = Path(str(section.get("source") or "results/wmdr2_json_examples"))
    target = Path(str(section.get("target") or "results/wmdr2_json_examples/gc_dar"))

    def abs_path(path: Path) -> Path:
        return path if path.is_absolute() else base / path

    return GcDarPaths(
        source=abs_path(source.expanduser()),
        target=abs_path(target.expanduser()),
        pattern=str(section.get("pattern") or DEFAULT_PATTERN),
        recursive=bool(section.get("recursive", True)),
        full_record_href_template=section.get("full_record_href_template"),
        dar_record_href_template=section.get("dar_record_href_template"),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert WMDR2 full records to GC-DAR discovery records.")
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    parser.add_argument("--source", type=Path, help="Source full WMDR2 record file or directory.")
    parser.add_argument("--target", type=Path, help="Target GC-DAR output directory.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument(
        "--full-record-href-template",
        help="Template for full-record links. Fields: {record_id}, {plain_id}, {filename}.",
    )
    parser.add_argument(
        "--dar-record-href-template",
        help="Template for DAR self links. Fields: {record_id}, {plain_id}, {filename}.",
    )
    args = parser.parse_args(argv)

    if args.source and args.target:
        paths = GcDarPaths(
            source=args.source,
            target=args.target,
            pattern=args.pattern,
            recursive=not args.no_recursive,
            full_record_href_template=args.full_record_href_template,
            dar_record_href_template=args.dar_record_href_template,
        )
    elif args.config.exists():
        paths = _paths_from_config(args.config)
    else:
        raise SystemExit("Pass --source and --target, or provide config.yaml with convert_wmdr2_json_to_wmdr2_gc_dar.")

    written = convert_files(paths)
    print(f"Wrote {len(written)} GC-DAR record(s) to {paths.target}")


if __name__ == "__main__":
    main()
