#!/usr/bin/env python3
"""convert_wmdr10_json_to_records_part1.py

Convert WMDR10 lean JSON exports (full record or parts) into OGC API - Records - Part 1 Core
GeoJSON encodings (Record(s) as GeoJSON Features in a FeatureCollection).

Key behavior (as requested):
- Reads config.yaml first (default: alongside this script) to get:
    convert_wmdr10_json_to_records_part1:
      source: <file-or-directory>
      target: <directory>
      mapping: <combined mapping CSV>   # preferred
      # or mapping_configs: { facility: ..., observations: ..., deployments: ... }  # legacy support
- Walks the source folder (recursive by default) and writes ONE GeoJSON file per JSON input file.
- Output file names are preserved: <input filename>.json -> <input filename>.geojson (same stem, same suffixes).

Notes:
- A full WMDR JSON file produces a single FeatureCollection containing multiple record Features
  (facility + observations + deployments, when present).
- Part files (e.g., *_facility.json, *_observations.json, *_deployments.json, *_header.json) each produce
  a FeatureCollection for that part, in a single output file with the same base name.

"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# -----------------------------
# Mapping model + loader
# -----------------------------

@dataclass(frozen=True)
class MapRow:
    part: str                   # facility | observations | deployments
    src: str                    # wmdr10-json
    dst: str                    # records-part1
    cardinality: str = ""
    link_relation: str = ""     # records-part1-link-relation
    link_group: str = ""        # records-part1-link-group
    codelist: str = ""
    description: str = ""


def _load_mapping_combined(path: Path) -> Dict[str, List[MapRow]]:
    """Load a combined mapping file (wmdr10_vs_records_part1.csv) into groups by part."""
    by_part: Dict[str, List[MapRow]] = {"facility": [], "observations": [], "deployments": []}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for raw in r:
            src = (raw.get("wmdr10-json") or "").strip()
            dst = (raw.get("records-part1") or "").strip()
            if not src or not dst:
                continue
            part = (raw.get("wmdr10-part") or "").strip() or _infer_part_from_src(src)
            if part not in by_part:
                # keep unknown parts out of conversion, but don't crash
                continue
            by_part[part].append(
                MapRow(
                    part=part,
                    src=src,
                    dst=dst,
                    cardinality=(raw.get("records-part1-cardinality") or "").strip(),
                    link_relation=(raw.get("records-part1-link-relation") or "").strip(),
                    link_group=(raw.get("records-part1-link-group") or "").strip(),
                    codelist=(raw.get("wmdr10-codelist") or "").strip(),
                    description=(raw.get("description") or "").strip(),
                )
            )
    return by_part


def _load_mapping_simple(path: Path, part: str) -> List[MapRow]:
    """Load a legacy part mapping file (no wmdr10-part column)."""
    rows: List[MapRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for raw in r:
            src = (raw.get("wmdr10-json") or "").strip()
            dst = (raw.get("records-part1") or "").strip()
            if not src or not dst:
                continue
            rows.append(
                MapRow(
                    part=part,
                    src=src,
                    dst=dst,
                    cardinality=(raw.get("records-part1-cardinality") or "").strip(),
                    link_relation=(raw.get("records-part1-link-relation") or "").strip(),
                    link_group=(raw.get("records-part1-link-group") or "").strip(),
                    codelist=(raw.get("wmdr10-codelist") or "").strip(),
                    description=(raw.get("description") or "").strip(),
                )
            )
    return rows


def _infer_part_from_src(src: str) -> str:
    s = src.strip()
    if s.startswith("facility.") or s.startswith("header."):
        return "facility"
    if s.startswith("observations"):
        return "observations"
    if s.startswith("deployments"):
        return "deployments"
    return "facility"


# -----------------------------
# Config
# -----------------------------

def _load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config.yaml (pip install pyyaml).")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sec = cfg.get("convert_wmdr10_json_to_records_part1")
    return sec if isinstance(sec, dict) else {}


# -----------------------------
# Small helpers
# -----------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iter_json_files(root: Path, recursive: bool = True, pattern: str = "*.json") -> List[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".json" else []
    if not root.is_dir():
        return []
    if recursive:
        return sorted([p for p in root.rglob(pattern) if p.is_file() and p.suffix.lower() == ".json"])
    return sorted([p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() == ".json"])


def _parse_geo_pos(raw: Any) -> Optional[Dict[str, Any]]:
    """Parse WMDR geoLocation (often a space-separated 'lat lon [z]') into GeoJSON Point."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        for k in ("pos", "value", "text", "geoLocation"):
            v = raw.get(k)
            if isinstance(v, str):
                raw = v
                break
    if not isinstance(raw, str):
        return None
    parts = raw.replace(",", " ").split()
    nums: List[float] = []
    for p in parts:
        try:
            nums.append(float(p))
        except Exception:
            continue
    if len(nums) < 2:
        return None
    lat, lon = nums[0], nums[1]
    return {"type": "Point", "coordinates": [lon, lat]}


def _extract_indexed(obj: Any, path: str) -> List[Tuple[Optional[Tuple[int, ...]], Any]]:
    """Extract nested values from dict/list with tokens, supporting [*] wildcards."""
    tokens = [t for t in path.split(".") if t]
    results: List[Tuple[Optional[Tuple[int, ...]], Any]] = [(None, obj)]

    for tok in tokens:
        nxt: List[Tuple[Optional[Tuple[int, ...]], Any]] = []

        if tok.endswith("[*]"):
            key = tok[:-3]
            for idx_tuple, cur in results:
                child = cur.get(key) if isinstance(cur, dict) else None
                if isinstance(child, list):
                    for i, item in enumerate(child):
                        new_idx = (i,) if idx_tuple is None else (idx_tuple + (i,))
                        nxt.append((new_idx, item))
                elif child is not None:
                    new_idx = (0,) if idx_tuple is None else (idx_tuple + (0,))
                    nxt.append((new_idx, child))
            results = nxt
            continue

        for idx_tuple, cur in results:
            if isinstance(cur, dict) and tok in cur:
                nxt.append((idx_tuple, cur[tok]))
        results = nxt

    return results


def _ensure_list_len(lst: List[Any], idx: int) -> None:
    while len(lst) <= idx:
        lst.append(None)


def _set_nested(root: Dict[str, Any], dst_path: str, val: Any, idx_tuple: Optional[Tuple[int, ...]]) -> None:
    """Set value in a nested dict/list using dst_path with [*] slots, using idx_tuple for indexing."""
    tokens = [t for t in dst_path.split(".") if t]
    cur: Any = root
    idx_pos = 0

    for i, tok in enumerate(tokens):
        last = i == len(tokens) - 1

        if tok.endswith("[*]"):
            key = tok[:-3]
            if not isinstance(cur, dict):
                return
            if key not in cur or not isinstance(cur[key], list):
                cur[key] = []
            lst: List[Any] = cur[key]

            use_idx = 0
            if idx_tuple is not None and idx_pos < len(idx_tuple):
                use_idx = idx_tuple[idx_pos]
            idx_pos += 1

            _ensure_list_len(lst, use_idx)
            if last:
                lst[use_idx] = val
                return
            if lst[use_idx] is None or not isinstance(lst[use_idx], dict):
                lst[use_idx] = {}
            cur = lst[use_idx]
            continue

        if not isinstance(cur, dict):
            return
        if last:
            cur[tok] = val
            return
        if tok not in cur or not isinstance(cur[tok], dict):
            cur[tok] = {}
        cur = cur[tok]


def _strip_prefix(src: str, prefix: str) -> str:
    return src[len(prefix):] if src.startswith(prefix) else src


# -----------------------------
# Mapping application
# -----------------------------

def _record_template(record_type: str) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "id": None,
        "geometry": None,
        "properties": {"type": record_type},
        "links": [],
    }


def _apply_mapping_to_record(
    src_obj: Dict[str, Any],
    mapping_rows: List[MapRow],
    *,
    part: str,
    header: Optional[Dict[str, Any]] = None,
    record_type: str = "dataset",
    src_prefix_to_strip: str = "",
    default_geometry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rec = _record_template(record_type)

    # collect links by (group, idx_tuple)
    link_indexed: Dict[Tuple[str, Tuple[int, ...]], Dict[str, Any]] = {}
    link_appended: List[Dict[str, Any]] = []

    for row in mapping_rows:
        if row.part != part:
            continue

        src_path = row.src

        # header mappings
        use_obj: Any = src_obj
        if src_path.startswith("header."):
            if header is None:
                continue
            use_obj = header
            src_path = _strip_prefix(src_path, "header.")

        if src_prefix_to_strip:
            src_path = _strip_prefix(src_path, src_prefix_to_strip)

        for idx_tuple, val in _extract_indexed(use_obj, src_path):
            # geometry special case
            if row.dst == "geometry":
                geom = _parse_geo_pos(val)
                if geom is not None:
                    rec["geometry"] = geom
                continue

            # link fields are routed into links[*].<field>
            if row.dst.startswith("links[*]."):
                field = row.dst.split(".", 1)[1]
                if idx_tuple is not None:
                    key = (row.link_group or row.link_relation or "link", idx_tuple)
                    link = link_indexed.setdefault(key, {})
                    if field == "href":
                        link["href"] = val
                        link["rel"] = row.link_relation or link.get("rel") or "related"
                    else:
                        link[field] = val
                        link.setdefault("rel", row.link_relation or "related")
                else:
                    # append mode
                    if field == "href":
                        link_appended.append({"href": val, "rel": row.link_relation or "related"})
                    else:
                        if not link_appended:
                            link_appended.append({"rel": row.link_relation or "related"})
                        link_appended[-1][field] = val
                        link_appended[-1].setdefault("rel", row.link_relation or "related")
                continue

            _set_nested(rec, row.dst, val, idx_tuple)

    # finalize links
    links: List[Dict[str, Any]] = []
    for (_g, _idx), link in sorted(link_indexed.items(), key=lambda x: (x[0][0], x[0][1])):
        if "href" in link:
            links.append(link)
    for link in link_appended:
        if "href" in link:
            links.append(link)
    rec["links"] = links

    # fallback geometry from WMDR keys
    if rec.get("geometry") is None:
        for k in ("geospatialLocation", "geoSpatialLocation"):
            loc = src_obj.get(k)
            if isinstance(loc, dict) and "geoLocation" in loc:
                geom = _parse_geo_pos(loc.get("geoLocation"))
                if geom is not None:
                    rec["geometry"] = geom
                    break
    if rec.get("geometry") is None and default_geometry is not None:
        rec["geometry"] = default_geometry

    # id fallback
    if rec.get("id") in (None, "", {}):
        for k in ("wigosStationIdentifier", "gawId", "facilityId", "@gml:id", "id"):
            v = src_obj.get(k)
            if isinstance(v, str) and v:
                rec["id"] = v
                break

    # Ensure records response fields are not wildly non-compliant
    rec.setdefault("type", "Feature")
    rec["properties"].setdefault("type", record_type)

    # helpful part marker
    rec["properties"].setdefault("wmdr:part", part)

    return rec


def _as_feature_collection(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": features,
        "timeStamp": _now_utc_iso(),
        "numberReturned": len(features),
    }


# -----------------------------
# Input classification
# -----------------------------

def _classify_by_filename(p: Path) -> Optional[str]:
    name = p.stem.lower()
    if name.endswith("_facility"):
        return "facility"
    if "_observations" in name:
        return "observations"
    if "_deployments" in name:
        return "deployments"
    if name.endswith("_header"):
        return "header"
    return None


def _classify_by_payload(payload: Any) -> str:
    if isinstance(payload, dict) and ("facility" in payload or "observations" in payload or "header" in payload):
        return "full"
    if isinstance(payload, dict):
        if "observedProperty" in payload or "resultTime" in payload or "procedure" in payload:
            return "observations"
        if "serialNumber" in payload or "manufacturer" in payload or "model" in payload:
            return "deployments"
        # header-only often has fileIdentifier / recordOwner / fileDateTime
        if "recordOwner" in payload or "fileIdentifier" in payload or "fileDateTime" in payload:
            return "header"
        return "facility"
    if isinstance(payload, list):
        # guess by first dict
        first = next((x for x in payload if isinstance(x, dict)), None)
        if first is None:
            return "unknown"
        if "observedProperty" in first or "resultTime" in first:
            return "observations"
        if "serialNumber" in first or "manufacturer" in first:
            return "deployments"
        return "observations"
    return "unknown"


def _flatten_deployments_from_observations(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deps: List[Dict[str, Any]] = []
    for o in observations:
        dep_raw = o.get("deployments")
        if isinstance(dep_raw, list):
            deps.extend([d for d in dep_raw if isinstance(d, dict)])
    return deps


# -----------------------------
# Conversion for a single file
# -----------------------------

def _convert_file(
    inp: Path,
    payload: Any,
    mapping: Dict[str, List[MapRow]],
    *,
    record_type_facility: str = "dataset",
    record_type_observation: str = "dataset",
    record_type_deployment: str = "dataset",
) -> Dict[str, Any]:
    """Return a FeatureCollection for this input file."""
    file_hint = _classify_by_filename(inp)
    kind = file_hint or _classify_by_payload(payload)

    header: Optional[Dict[str, Any]] = None
    facility: Optional[Dict[str, Any]] = None
    observations: List[Dict[str, Any]] = []
    deployments: List[Dict[str, Any]] = []

    if kind == "full" and isinstance(payload, dict):
        header = payload.get("header") if isinstance(payload.get("header"), dict) else None
        facility = payload.get("facility") if isinstance(payload.get("facility"), dict) else None
        obs_raw = payload.get("observations")
        observations = [o for o in obs_raw if isinstance(o, dict)] if isinstance(obs_raw, list) else []
        deployments = _flatten_deployments_from_observations(observations)

    elif kind == "facility" and isinstance(payload, dict):
        facility = payload

    elif kind == "header" and isinstance(payload, dict):
        header = payload

    elif kind == "observations":
        if isinstance(payload, dict) and "observations" in payload:
            obs_raw = payload.get("observations")
            observations = [o for o in obs_raw if isinstance(o, dict)] if isinstance(obs_raw, list) else []
            header = payload.get("header") if isinstance(payload.get("header"), dict) else None
            facility = payload.get("facility") if isinstance(payload.get("facility"), dict) else None
        elif isinstance(payload, list):
            observations = [o for o in payload if isinstance(o, dict)]
        elif isinstance(payload, dict):
            observations = [payload]
        deployments = _flatten_deployments_from_observations(observations)

    elif kind == "deployments":
        if isinstance(payload, dict) and "deployments" in payload:
            dep_raw = payload.get("deployments")
            deployments = [d for d in dep_raw if isinstance(d, dict)] if isinstance(dep_raw, list) else []
            header = payload.get("header") if isinstance(payload.get("header"), dict) else None
            facility = payload.get("facility") if isinstance(payload.get("facility"), dict) else None
        elif isinstance(payload, list):
            deployments = [d for d in payload if isinstance(d, dict)]
        elif isinstance(payload, dict):
            deployments = [payload]

    else:
        # best-effort: try as full dict
        if isinstance(payload, dict):
            header = payload.get("header") if isinstance(payload.get("header"), dict) else None
            facility = payload.get("facility") if isinstance(payload.get("facility"), dict) else None
            obs_raw = payload.get("observations")
            observations = [o for o in obs_raw if isinstance(o, dict)] if isinstance(obs_raw, list) else []
            deployments = _flatten_deployments_from_observations(observations)

    features: List[Dict[str, Any]] = []

    # facility geometry for fallback
    facility_geom: Optional[Dict[str, Any]] = None

    if facility is not None:
        rec = _apply_mapping_to_record(
            facility,
            mapping["facility"],
            part="facility",
            header=header,
            record_type=record_type_facility,
            src_prefix_to_strip="facility.",
        )
        if not rec["properties"].get("title"):
            name = facility.get("facilityName") or facility.get("name") or facility.get("gawId")
            rec["properties"]["title"] = name if isinstance(name, str) and name else "Facility"
        facility_geom = rec.get("geometry")
        features.append(rec)

    # header-only input -> create a record only from header rows (no facility object)
    if facility is None and header is not None and kind == "header":
        dummy: Dict[str, Any] = {}
        rec = _apply_mapping_to_record(
            dummy,
            mapping["facility"],
            part="facility",
            header=header,
            record_type=record_type_facility,
            src_prefix_to_strip="facility.",
        )
        rec["properties"].setdefault("title", "Header")
        features.append(rec)

    if observations:
        for o in observations:
            rec = _apply_mapping_to_record(
                o,
                mapping["observations"],
                part="observations",
                header=header,
                record_type=record_type_observation,
                src_prefix_to_strip="observations[*].",
                default_geometry=facility_geom,
            )
            if not rec["properties"].get("title"):
                op = o.get("observedProperty")
                title = op.rsplit("/", 1)[-1] if isinstance(op, str) and op else "Observation"
                rec["properties"]["title"] = title
            features.append(rec)

    if deployments:
        for d in deployments:
            rec = _apply_mapping_to_record(
                d,
                mapping["deployments"],
                part="deployments",
                header=header,
                record_type=record_type_deployment,
                src_prefix_to_strip="deployments[*].",
                default_geometry=facility_geom,
            )
            if not rec["properties"].get("title"):
                man = d.get("manufacturer")
                mod = d.get("model")
                ser = d.get("serialNumber")
                bits = [b for b in [man, mod, f"SN {ser}" if ser else None] if isinstance(b, str) and b]
                rec["properties"]["title"] = " ".join(bits) if bits else "Deployment"
            features.append(rec)

    return _as_feature_collection(features)


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert WMDR10 lean JSON to OGC Records Part 1 GeoJSON, preserving filenames.")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to config.yaml (default: script folder/config.yaml)",
    )
    ap.add_argument("--source", type=Path, default=None, help="Override config source (file or directory).")
    ap.add_argument("--target", type=Path, default=None, help="Override config target directory.")
    ap.add_argument("--mapping", type=Path, default=None, help="Override combined mapping CSV path.")
    ap.add_argument("--pattern", type=str, default="*.json", help="Glob pattern for JSON files (default: *.json)")
    ap.add_argument("--recursive", action="store_true", help="Scan source directory recursively (default).")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false", help="Scan only top-level.")
    ap.set_defaults(recursive=True)
    args = ap.parse_args()

    cfg = _cfg_section(_load_config(args.config))

    source = args.source or Path(cfg.get("source", ""))
    target = args.target or Path(cfg.get("target", ""))

    if not str(source):
        raise SystemExit("Missing source. Set convert_wmdr10_json_to_records_part1.source in config.yaml or pass --source.")
    if not str(target):
        raise SystemExit("Missing target. Set convert_wmdr10_json_to_records_part1.target in config.yaml or pass --target.")

    # Resolve mapping
    mapping_path = args.mapping
    if mapping_path is None:
        mapping_path = cfg.get("mapping")
        if mapping_path:
            mapping_path = Path(str(mapping_path))
        else:
            raise SystemExit("Missing mapping. Provide convert_wmdr10_json_to_records_part1.mapping or mapping_configs.* in config.yaml.")
    if "mapping" not in locals():
        # combined mapping path case
        mp = Path(mapping_path)  # type: ignore[arg-type]
        if not mp.is_absolute():
            mp = (args.config.parent / mp).resolve()
        if not mp.exists():
            raise SystemExit(f"Mapping file not found: {mp}")
        mapping = _load_mapping_combined(mp)

    source = source if source.is_absolute() else (args.config.parent / source).resolve()
    target = target if target.is_absolute() else (args.config.parent / target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    json_files = _iter_json_files(source, recursive=args.recursive, pattern=args.pattern)
    if not json_files:
        raise SystemExit(f"No JSON files found under: {source}")

    ok = 0
    failed: List[Tuple[Path, str]] = []

    # record type defaults (could be extended via config later)
    rt_fac = "dataset"
    rt_obs = "dataset"
    rt_dep = "dataset"

    for inp in json_files:
        try:
            payload = json.loads(inp.read_text(encoding="utf-8"))

            # Preserve relative path structure
            rel = inp.relative_to(source) if source.is_dir() else inp.name
            if isinstance(rel, Path):
                out_path = (target / rel).with_suffix(".geojson")
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = (target / inp.name).with_suffix(".geojson")
                out_path.parent.mkdir(parents=True, exist_ok=True)

            fc = _convert_file(
                inp,
                payload,
                mapping,
                record_type_facility=rt_fac,
                record_type_observation=rt_obs,
                record_type_deployment=rt_dep,
            )

            out_path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote: {out_path}")
            ok += 1

        except Exception as e:
            failed.append((inp, str(e)))
            print(f"ERROR: {inp}: {e}")

    print(f"Done. Converted {ok}/{len(json_files)} file(s).")
    if failed:
        print("Failures:")
        for p, msg in failed:
            print(f"  - {p}: {msg}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
