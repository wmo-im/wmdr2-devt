#!/usr/bin/env python3
"""convert_wmdr10_json_to_records_part1.py

Convert WMDR10 lean JSON exports (full record or parts) into OGC API - Records - Part 1 Core
GeoJSON encodings (Record(s) as GeoJSON Feature / FeatureCollection).

OGC API - Records - Part 1 examples show:
- A record is a GeoJSON Feature containing: id, type, geometry, (optional) time, (optional) conformsTo, properties, links
- A records response container is a FeatureCollection with features[] and (optional) links, timeStamp, numberReturned, ...

See Listing 21 (FeatureCollection container) and Listing 22 (single Record as Feature). (OGC 20-004r1)
https://docs.ogc.org/is/20-004r1/20-004r1.html

This script is intentionally conservative:
- It always emits GeoJSON FeatureCollection(s).
- It ensures each record has: type="Feature", id, geometry (or null), properties{}, links[].
- It uses the mapping CSVs to copy WMDR values into Records fields where it is a direct mapping.
- For complex WMDR sub-structures (e.g., dataGeneration objects), it preserves the raw object in properties as extensions.

Usage examples
--------------
# Convert full WMDR JSON into 3 outputs in same folder:
python convert_wmdr10_json_to_records_part1.py /path/to/20250504_0-20008-0-NRB.json

# Convert only observations:
python convert_wmdr10_json_to_records_part1.py /path/to/_observations.json --parts observations --out /tmp/outdir

# Provide custom mapping files:
python convert_wmdr10_json_to_records_part1.py input.json \
  --mapping-facility wmdr10_facility_vs_records_part1_enriched_v2.csv \
  --mapping-observations wmdr10_observations_vs_records_part1_enriched_v3.csv \
  --mapping-deployments wmdr10_deployments_vs_records_part1_enriched.csv

Notes
-----
- The mapping CSVs are expected to have (at least) these columns:
  wmdr10-json, records-part1, description
  Optionally: records-part1-link-relation

- This converter supports limited wildcard handling:
  * one [*] wildcard per path is supported for index-aligned list building.

- Link objects:
  * Rows targeting links[*].href/.title/.type are used to build link objects.
  * If a mapping row provides records-part1-link-relation, it is used as the link.rel value.

"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

CONFORMS_TO_RECORD_CORE = "http://www.opengis.net/spec/ogcapi-records-1/1.0/conf/record-core"

_seg_pat = re.compile(r"^(?P<key>.+?)\[(?P<idx>\*|\d+)\]$")


@dataclass(frozen=True)
class MapRow:
    src: str
    dst: str
    desc: str = ""
    rel: str = ""
    group: str = ""


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_mapping(path: Path) -> List[MapRow]:
    df = pd.read_csv(path)
    for col in ["wmdr10-json", "records-part1"]:
        if col not in df.columns:
            raise ValueError(f"Mapping file {path} missing required column '{col}'. Found: {list(df.columns)}")
    desc_col = "description" if "description" in df.columns else None
    rel_col = "records-part1-link-relation" if "records-part1-link-relation" in df.columns else None
    group_col = "records-part1-link-group" if "records-part1-link-group" in df.columns else None

    out: List[MapRow] = []
    for _, r in df.iterrows():
        src = str(r["wmdr10-json"]).strip()
        dst = str(r["records-part1"]).strip()
        if not src or not dst or src.lower() == "nan" or dst.lower() == "nan":
            continue
        desc = str(r[desc_col]).strip() if desc_col else ""
        rel = str(r[rel_col]).strip() if rel_col and str(r[rel_col]).lower() != "nan" else ""
        group = str(r[group_col]).strip() if group_col and str(r[group_col]).lower() != "nan" else ""
        out.append(MapRow(src=src, dst=dst, desc=desc, rel=rel, group=group))
    return out


def _strip_prefix(src_path: str, part: str) -> str:
    """Normalize mapping 'wmdr10-json' paths to be relative to the object being converted."""
    src_path = src_path.strip()
    if part == "facility" and src_path.startswith("facility."):
        return src_path[len("facility.") :]
    if part == "header" and src_path.startswith("header."):
        return src_path[len("header.") :]
    if part == "observations" and src_path.startswith("observations[*]."):
        return src_path[len("observations[*].") :]
    if part == "deployments" and src_path.startswith("deployments[*]."):
        return src_path[len("deployments[*].") :]
    return src_path


def _extract_indexed(obj: Any, path: str) -> List[Tuple[Optional[Tuple[int, ...]], Any]]:
    """Extract values following a dotted path; returns (index_tuple, value).

    Supports one or more [*] wildcards. For each wildcard, an index is appended to the index_tuple.

    If no wildcard used, index_tuple is None.
    """
    if not path:
        return []

    parts = path.split(".")
    results: List[Tuple[Optional[Tuple[int, ...]], Any]] = [(None, obj)]

    for part in parts:
        m = _seg_pat.match(part)
        next_results: List[Tuple[Optional[Tuple[int, ...]], Any]] = []

        if m:
            key = m.group("key")
            idx = m.group("idx")
            for idx_tuple, cur in results:
                child = None
                if isinstance(cur, dict):
                    child = cur.get(key)
                elif isinstance(cur, list):
                    collected = []
                    for it in cur:
                        if isinstance(it, dict) and key in it:
                            collected.append(it[key])
                    child = collected if collected else None

                if idx == "*":
                    if isinstance(child, list):
                        for i, item in enumerate(child):
                            new_idx = (i,) if idx_tuple is None else (idx_tuple + (i,))
                            next_results.append((new_idx, item))
                    elif child is not None:
                        # Treat scalar/dict as a single-element list at index 0 (helps harmonize shapes)
                        new_idx = (0,) if idx_tuple is None else (idx_tuple + (0,))
                        next_results.append((new_idx, child))
                else:
                    try:
                        i = int(idx)
                    except ValueError:
                        continue
                    if isinstance(child, list) and 0 <= i < len(child):
                        new_idx = (i,) if idx_tuple is None else (idx_tuple + (i,))
                        next_results.append((new_idx, child[i]))
        else:
            key = part
            for idx_tuple, cur in results:
                if isinstance(cur, dict) and key in cur:
                    next_results.append((idx_tuple, cur[key]))
                elif isinstance(cur, list):
                    for it in cur:
                        if isinstance(it, dict) and key in it:
                            next_results.append((idx_tuple, it[key]))

        results = next_results
        if not results:
            break

    cleaned: List[Tuple[Optional[Tuple[int, ...]], Any]] = []
    for idx_tuple, v in results:
        if v is None:
            continue
        cleaned.append((idx_tuple, v))
    return cleaned


def _ensure_list_len(lst: List[Any], n: int) -> None:
    while len(lst) < n:
        lst.append({})


def _parse_geo_pos(pos: Any) -> Optional[Dict[str, Any]]:
    """Parse WMDR geoLocation strings (usually "lat lon [alt]") into GeoJSON Point."""
    if pos is None:
        return None
    if isinstance(pos, dict) and "geoLocation" in pos:
        pos = pos.get("geoLocation")
    if not isinstance(pos, str):
        return None
    s = pos.strip()
    if not s:
        return None
    parts = [p for p in s.split() if p]
    if len(parts) < 2:
        return None
    try:
        a = float(parts[0]); b = float(parts[1])
        c = float(parts[2]) if len(parts) >= 3 else None
    except ValueError:
        return None

    # WMDR commonly uses lat lon [alt]; GeoJSON needs lon lat.
    lat, lon, alt = a, b, c
    if abs(lat) > 90 and abs(lon) <= 90:
        lon, lat = a, b

    coords: List[float] = [lon, lat]
    if alt is not None:
        coords.append(alt)
    return {"type": "Point", "coordinates": coords}


def _set_nested(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur: Any = d
    for k in keys[:-1]:
        if not isinstance(cur, dict):
            return
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    if isinstance(cur, dict):
        cur[keys[-1]] = value


def _get_or_create_nested(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    cur = d
    for k in keys:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    return cur


def _apply_mapping_to_record(
    src_obj: Dict[str, Any],
    mapping: List[MapRow],
    *,
    part: str,
    header: Optional[Dict[str, Any]] = None,
    record_type: str = "dataset",
    default_geometry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a Records Part 1 record (GeoJSON Feature) from src_obj using mapping rows."""
    record: Dict[str, Any] = {
        "type": "Feature",
        "id": None,
        "geometry": None,
        "properties": {},
        "links": [],
        "conformsTo": [CONFORMS_TO_RECORD_CORE],
    }

    record["properties"]["type"] = record_type

    # header-based defaults
    if header:
        file_dt = header.get("fileDateTime") or header.get("dateStamp") or header.get("fileDatetime")
        if isinstance(file_dt, str) and file_dt:
            record["properties"].setdefault("created", file_dt)
            record["properties"].setdefault("updated", file_dt)

    # Link collection:
    # - If source has wildcards, we can index-align links (e.g., onLine[*])
    # - Otherwise, use records-part1-link-group to avoid collisions when multiple WMDR fields map to links[*]
    indexed_links: Dict[Tuple[str, Tuple[int, ...]], Dict[str, Any]] = {}
    appended_links_by_group: Dict[str, List[Dict[str, Any]]] = {}

    for row in mapping:
        src_path = _strip_prefix(row.src, part)
        dst_path = row.dst.strip()

        extracted = _extract_indexed(src_obj, src_path)
        if not extracted:
            continue

        for idx_tuple, val in extracted:
            # geometry
            if dst_path == "geometry":
                geom = _parse_geo_pos(val)
                if geom is not None:
                    record["geometry"] = geom
                continue

            # id
            if dst_path == "id":
                if record.get("id") is None and isinstance(val, str) and val:
                    record["id"] = val
                continue

            # time interval
            if dst_path.startswith("time.interval"):
                m = _seg_pat.match(dst_path.replace("time.", ""))
                if m and m.group("key") == "interval":
                    try:
                        i = int(m.group("idx"))
                    except ValueError:
                        continue
                    record.setdefault("time", {}).setdefault("interval", ["..", ".."])
                    if isinstance(val, str) and val:
                        record["time"]["interval"][i] = val
                    elif val is None:
                        record["time"]["interval"][i] = ".."
                continue

            # links
            if dst_path.startswith("links[*]."):
                field = dst_path.split(".", 1)[1].replace("[*].", "")
                group = row.group or row.rel or "link"

                if idx_tuple is not None:
                    key = (group, idx_tuple)
                    link = indexed_links.setdefault(key, {})
                    if field == "href":
                        link["href"] = val
                        link["rel"] = row.rel or link.get("rel") or "related"
                    else:
                        link[field] = val
                else:
                    grp_list = appended_links_by_group.setdefault(group, [])

                    if field == "href":
                        grp_list.append({"href": val, "rel": row.rel or "related"})
                    else:
                        # attach to the latest link of that group if possible, otherwise stage a placeholder
                        if not grp_list:
                            grp_list.append({})
                        grp_list[-1][field] = val
                        # ensure rel exists if later becomes a valid link
                        grp_list[-1].setdefault("rel", row.rel or "related")
                continue

            # generic with wildcard support
            parts = dst_path.split(".")
            wildcard_pos = None
            for j, p in enumerate(parts):
                mm = _seg_pat.match(p)
                if mm and mm.group("idx") == "*":
                    wildcard_pos = j
                    break

            if wildcard_pos is None:
                _set_nested(record, parts, val)
                continue

            pre = parts[:wildcard_pos]
            seg = parts[wildcard_pos]
            mm = _seg_pat.match(seg)
            assert mm is not None
            list_key = mm.group("key")
            post = parts[wildcard_pos + 1 :]

            parent = _get_or_create_nested(record, pre)
            arr = parent.get(list_key)
            if not isinstance(arr, list):
                arr = []
                parent[list_key] = arr

            if idx_tuple is not None:
                i = idx_tuple[-1]
                _ensure_list_len(arr, i + 1)
                if not isinstance(arr[i], dict):
                    arr[i] = {}
                tgt = arr[i]
                if post == ["href"] and isinstance(val, str):
                    tgt["href"] = val
                elif len(post) == 1:
                    tgt[post[0]] = val
                else:
                    _set_nested(tgt, post, val)
            else:
                if post == []:
                    arr.append(val)
                elif post == ["href"]:
                    arr.append({"href": val})
                else:
                    obj = {}
                    _set_nested(obj, post, val)
                    arr.append(obj)

    # Finalize links:
    # 1) indexed links (sorted by group then index)
    for (group, idx), link in sorted(indexed_links.items(), key=lambda x: (x[0][0], x[0][1])):
        if "href" in link:
            record["links"].append(link)

    # 2) appended links (preserve group order by appearance in mapping iteration)
    for group, links in appended_links_by_group.items():
        for link in links:
            if isinstance(link, dict) and "href" in link:
                record["links"].append(link)

    # Geometry fallback from common WMDR keys (useful when mapping CSV uses a different key spelling)
    if record.get("geometry") is None:
        for k in ("geospatialLocation", "geoSpatialLocation"):
            if isinstance(src_obj.get(k), dict) and "geoLocation" in src_obj.get(k):
                geom = _parse_geo_pos(src_obj[k].get("geoLocation"))
                if geom is not None:
                    record["geometry"] = geom
                    break

    # mandatory members
    if record["id"] is None:
        record["id"] = f"urn:uuid:{abs(hash(json.dumps(src_obj, sort_keys=True))) % (10**12)}"
    if "geometry" not in record:
        record["geometry"] = None
    if "links" not in record:
        record["links"] = []

    # optional geometry fallback (e.g., apply facility geometry to deployments without own location)
    if record.get("geometry") is None and default_geometry is not None:
        record["geometry"] = default_geometry

    if "time" in record and isinstance(record["time"], dict):
        interval = record["time"].get("interval")
        if isinstance(interval, list):
            record["time"]["interval"] = [(x if x is not None else "..") for x in interval]

    return record


def _as_feature_collection(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "timeStamp": _now_utc_iso(),
        "numberReturned": len(features),
        "numberMatched": len(features),
        "features": features,
        "links": [],
    }


def _detect_input_kind(payload: Any) -> str:
    """Return one of {"full", "facility", "observations", "deployments"}."""
    if isinstance(payload, dict):
        if "facility" in payload and "observations" in payload:
            return "full"
        if "facilityName" in payload or "wigosStationIdentifier" in payload or "observingFacility" in payload:
            return "facility"
        if "observedProperty" in payload and "deployments" in payload:
            return "observations"
        if "@gml:id" in payload and ("beginPosition" in payload or "observingMethod" in payload):
            return "deployments"
    if isinstance(payload, list):
        if not payload:
            return "observations"
        first = payload[0]
        if isinstance(first, dict) and "observedProperty" in first:
            return "observations"
        if isinstance(first, dict) and "@gml:id" in first and ("beginPosition" in first or "observingMethod" in first):
            return "deployments"
    return "full"


def _flatten_deployments_from_observations(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for o in observations:
        ds = o.get("deployments") or []
        if isinstance(ds, list):
            out.extend([d for d in ds if isinstance(d, dict)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert WMDR10 lean JSON to OGC API Records Part 1 GeoJSON.")
    ap.add_argument("input", type=Path, help="Input WMDR10 lean JSON file (full or part).")
    ap.add_argument("--out", type=Path, default=None, help="Output directory (default: same as input).")

    ap.add_argument("--parts", choices=["auto", "facility", "observations", "deployments", "all"], default="auto",
                    help="Which part(s) to generate. 'auto' detects from input; 'all' generates facility+observations+deployments when possible.")

    ap.add_argument("--mapping-facility", type=Path, default=Path("mappings/wmdr10_facility_vs_records_part1.csv"))
    ap.add_argument("--mapping-observations", type=Path, default=Path("mappings/wmdr10_observations_vs_records_part1.csv"))
    ap.add_argument("--mapping-deployments", type=Path, default=Path("mappings/wmdr10_deployments_vs_records_part1.csv"))

    ap.add_argument("--record-type-facility", default="dataset", help="records properties.type value for facility records")
    ap.add_argument("--record-type-observation", default="dataset", help="records properties.type value for observation records")
    ap.add_argument("--record-type-deployment", default="dataset", help="records properties.type value for deployment records")

    args = ap.parse_args()

    inp = args.input
    out_dir = args.out or inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(inp.read_text(encoding="utf-8"))
    detected = _detect_input_kind(payload)
    kind = detected if args.parts == "auto" else args.parts
    if kind == "all":
        kind = "full"

    script_dir = Path(__file__).resolve().parent

    def resolve_map(p: Path) -> Path:
        if p.exists():
            return p
        q = script_dir / p
        if q.exists():
            return q
        r = inp.parent / p
        if r.exists():
            return r
        raise FileNotFoundError(f"Mapping file not found: {p} (also tried {q} and {r})")

    map_fac = _load_mapping(resolve_map(args.mapping_facility))
    map_obs = _load_mapping(resolve_map(args.mapping_observations))
    map_dep = _load_mapping(resolve_map(args.mapping_deployments))

    header = None
    facility = None
    observations: List[Dict[str, Any]] = []
    deployments: List[Dict[str, Any]] = []

    if isinstance(payload, dict) and kind == "full":
        header = payload.get("header") if isinstance(payload.get("header"), dict) else None
        facility = payload.get("facility") if isinstance(payload.get("facility"), dict) else None
        observations = payload.get("observations") if isinstance(payload.get("observations"), list) else []
        deployments = _flatten_deployments_from_observations(observations)
    elif kind == "facility":
        facility = payload if isinstance(payload, dict) else None
    elif kind == "observations":
        if isinstance(payload, dict) and "observations" in payload:
            observations = payload.get("observations") if isinstance(payload.get("observations"), list) else []
            header = payload.get("header") if isinstance(payload.get("header"), dict) else None
            facility = payload.get("facility") if isinstance(payload.get("facility"), dict) else None
        elif isinstance(payload, list):
            observations = payload
        elif isinstance(payload, dict):
            observations = [payload]
        deployments = _flatten_deployments_from_observations(observations)
    elif kind == "deployments":
        if isinstance(payload, dict) and "deployments" in payload:
            deployments = payload.get("deployments") if isinstance(payload.get("deployments"), list) else []
            header = payload.get("header") if isinstance(payload.get("header"), dict) else None
            facility = payload.get("facility") if isinstance(payload.get("facility"), dict) else None
        elif isinstance(payload, list):
            deployments = payload
        elif isinstance(payload, dict):
            deployments = [payload]
    else:
        # fallback: treat as full
        if isinstance(payload, dict):
            header = payload.get("header") if isinstance(payload.get("header"), dict) else None
            facility = payload.get("facility") if isinstance(payload.get("facility"), dict) else None
            observations = payload.get("observations") if isinstance(payload.get("observations"), list) else []
            deployments = _flatten_deployments_from_observations(observations)

    outputs: List[Tuple[str, Dict[str, Any]]] = []

    facility_geometry = None

    if facility is not None and (args.parts in ("auto", "all") or kind in ("full", "facility")):
        rec = _apply_mapping_to_record(facility, map_fac, part="facility", header=header, record_type=args.record_type_facility)
        facility_geometry = rec.get("geometry")
        if not rec["properties"].get("title"):
            name = facility.get("facilityName") or facility.get("name") or facility.get("gawId")
            rec["properties"]["title"] = name if isinstance(name, str) and name else "Facility"
        outputs.append(("facility", _as_feature_collection([rec])))

    if observations and (args.parts in ("auto", "all") or kind in ("full", "observations")):
        feats: List[Dict[str, Any]] = []
        for o in observations:
            if not isinstance(o, dict):
                continue
            rec = _apply_mapping_to_record(o, map_obs, part="observations", header=header, record_type=args.record_type_observation, default_geometry=facility_geometry)
            if not rec["properties"].get("title"):
                op = o.get("observedProperty")
                fac = None
                if isinstance(o.get("facility"), dict):
                    fac = o["facility"].get("wigosStationIdentifier") or o["facility"].get("facilityName")
                title = "Observation"
                if isinstance(op, str) and op:
                    title = op.rsplit("/", 1)[-1]
                if fac:
                    title = f"{title} @ {fac}"
                rec["properties"]["title"] = title
            if "deployments" in o:
                rec["properties"].setdefault("wmdr:deployments", o.get("deployments"))
            feats.append(rec)
        outputs.append(("observations", _as_feature_collection(feats)))

    if deployments and (args.parts in ("auto", "all") or kind in ("full", "deployments")):
        feats = []
        for d in deployments:
            if not isinstance(d, dict):
                continue
            rec = _apply_mapping_to_record(d, map_dep, part="deployments", header=header, record_type=args.record_type_deployment, default_geometry=facility_geometry)
            if not rec["properties"].get("title"):
                man = d.get("manufacturer")
                mod = d.get("model")
                ser = d.get("serialNumber")
                bits = [b for b in [man, mod, f"SN {ser}" if ser else None] if isinstance(b, str) and b]
                rec["properties"]["title"] = " ".join(bits) if bits else "Deployment"
            for k in ("dataGeneration", "instrumentOperatingStatus", "geospatialLocation"):
                if k in d:
                    rec["properties"].setdefault(f"wmdr:{k}", d.get(k))
            feats.append(rec)
        outputs.append(("deployments", _as_feature_collection(feats)))

    stem = inp.stem
    for tag, fc in outputs:
        out_path = out_dir / f"{stem}_{tag}.geojson" if kind == "full" else out_dir / f"{stem}.geojson"
        out_path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")

    if not outputs:
        raise SystemExit("No outputs produced (input did not contain requested part(s)).")


if __name__ == "__main__":
    main()
