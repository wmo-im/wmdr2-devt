#!/usr/bin/env python3
"""Create compact WMDR2 GC-DAR discovery records from full WMDR2 records."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json


@dataclass(frozen=True)
class GcDarPaths:
    source: Path
    target: Path


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: v for k, v in ((k, _clean(v)) for k, v in obj.items()) if v not in (None, "", [], {})}
    if isinstance(obj, list):
        return [v for v in (_clean(v) for v in obj) if v not in (None, "", [], {})]
    return obj


def _plain_id(record_id: str) -> str:
    return record_id.split(":", 1)[1] if ":" in record_id else record_id


def _source_hash(record: Mapping[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _date_from_interval(series: Mapping[str, Any]) -> str | None:
    interval = series.get("time", {}).get("interval") if isinstance(series.get("time"), Mapping) else None
    if isinstance(interval, list) and interval:
        return str(interval[0])
    return None


def _is_current(series: Mapping[str, Any]) -> bool:
    interval = series.get("time", {}).get("interval") if isinstance(series.get("time"), Mapping) else None
    return not (isinstance(interval, list) and len(interval) > 1 and interval[1] not in (None, "", ".."))


def _latest(items: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return items[-1] if items else None


def _program_affiliations(props: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in _as_list(props.get("programAffiliations") or props.get("programAffiliation")):
        if not isinstance(item, Mapping):
            continue
        program = item.get("program") or item.get("programAffiliation")
        if isinstance(program, list) and len(program) == 1:
            program = program[0]
        record = {"program": program, "reportingStatus": item.get("reportingStatus"), "date": item.get("validFrom") or item.get("date")}
        out.append(_clean(record))
    return out


def _territory(props: Mapping[str, Any]) -> Any:
    items = [i for i in _as_list(props.get("territory")) if isinstance(i, Mapping)]
    item = _latest(items)
    return item.get("territory") if item else None


def _reporting_index(props: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for item in _as_list(props.get("reporting")):
        if isinstance(item, Mapping) and isinstance(item.get("id"), str):
            index[item["id"]] = item
    return index


def _instrument_index(props: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for item in _as_list(props.get("instruments")):
        if not isinstance(item, Mapping):
            continue
        uid = item.get("uid") or item.get("id")
        if isinstance(uid, str):
            index[uid] = item
    return index


def _series_summary(series: Mapping[str, Any], *, reporting_defs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    if not _is_current(series):
        return None
    cfgs = [c for c in _as_list(series.get("observingConfigurations")) if isinstance(c, Mapping)]
    cfg = _latest(cfgs) or {}
    reps = [r for r in _as_list(series.get("reportingProcedures") or series.get("reporting")) if isinstance(r, Mapping)]
    rep = dict(_latest(reps) or {})
    ref = rep.get("reporting")
    if isinstance(ref, str) and ref in reporting_defs:
        rep = {**reporting_defs[ref], **rep}
    observed_feature = series.get("observedFeature") if isinstance(series.get("observedFeature"), Mapping) else {}
    out = {
        "uid": series.get("uid") or series.get("id"),
        "title": series.get("title"),
        "current": True,
        "observedProperty": series.get("observedProperty"),
        "observedGeometry": series.get("observedGeometry"),
        "observedFeatureDomain": observed_feature.get("domain") if isinstance(observed_feature, Mapping) else None,
        "observedFeatureDomainFeature": observed_feature.get("domainFeature") if isinstance(observed_feature, Mapping) else None,
        "observedFeatureName": observed_feature.get("featureName") if isinstance(observed_feature, Mapping) else None,
        "programAffiliations": series.get("programAffiliations") or series.get("programAffiliation"),
        "sourceOfObservation": cfg.get("sourceOfObservation") or series.get("sourceOfObservation"),
        "observingMethod": cfg.get("observingMethod"),
        "instrument": cfg.get("instrument"),
        "serialNumber": cfg.get("serialNumber"),
        "uom": rep.get("uom"),
        "internationalExchange": rep.get("internationalExchange"),
        "temporalReportingInterval": rep.get("temporalReportingInterval") or rep.get("temporalAggregate"),
        "levelOfData": rep.get("levelOfData"),
    }
    return _clean(out)


def convert_record_to_gc_dar(record: Mapping[str, Any], *, source_filename: str | None = None, derived_at: str | None = None, full_record_href_template: str | None = None, dar_record_href_template: str | None = None) -> dict[str, Any]:
    rid = str(record.get("id") or "record:unknown")
    props = record.get("properties") if isinstance(record.get("properties"), Mapping) else {}
    props = props if isinstance(props, Mapping) else {}
    reporting_defs = _reporting_index(props)
    series_summaries = [s for s in (_series_summary(s, reporting_defs=reporting_defs) for s in _as_list(props.get("observationSeries"))) if s]
    observed_properties = sorted({s.get("observedProperty") for s in series_summaries if s.get("observedProperty") is not None}, key=str)
    observed_domains = sorted({s.get("observedFeatureDomain") for s in series_summaries if s.get("observedFeatureDomain")}, key=str)
    plain = _plain_id(rid)
    derived = derived_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    out = {
        "type": "Feature",
        "id": rid,
        "geometry": record.get("geometry"),
        "time": record.get("time"),
        "conformsTo": ["http://wigos.wmo.int/spec/wmdr/2/conf/gc-dar"],
        "properties": {
            "type": "facility",
            "profile": "wmdr2-gc-dar",
            "title": props.get("title"),
            "description": props.get("description"),
            "keywords": props.get("keywords"),
            "territory": _territory(props),
            "programAffiliations": _program_affiliations(props),
            "observedProperty": observed_properties,
            "observedFeatureDomain": observed_domains,
            "observationSeries": series_summaries,
            "provenance": {"derivedAt": derived, "sourceFilename": source_filename, "sourceContentHash": _source_hash(record)},
            "retrieval": {
                "fullRecord": {"href": full_record_href_template.format(plain_id=plain, id=rid) if full_record_href_template else None},
                "darRecord": {"href": dar_record_href_template.format(plain_id=plain, id=rid) if dar_record_href_template else None},
            },
        },
    }
    return _clean(out)


def _iter_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*.json") if p.is_file())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def convert_files(paths: GcDarPaths) -> list[Path]:
    written: list[Path] = []
    for src in _iter_files(paths.source):
        record = json.loads(src.read_text(encoding="utf-8"))
        rel = src.relative_to(paths.source) if paths.source.is_dir() else Path(src.name)
        out = paths.target / rel
        _write_json(out, convert_record_to_gc_dar(record, source_filename=str(rel)))
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args(argv)
    written = convert_files(GcDarPaths(source=Path(args.source), target=Path(args.target)))
    print(f"Wrote {len(written)} GC-DAR record(s) to {args.target}")


if __name__ == "__main__":
    main()
