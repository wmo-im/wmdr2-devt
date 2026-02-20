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
import re
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
    src: str                    # wmdr10-json
    dst: str                    # records-part1
    cardinality: str = ""
    link_relation: str = ""     # records-part1-link-relation
    link_group: str = ""        # records-part1-link-group
    default: str = ""
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
                    src=src,
                    dst=dst,
                    cardinality=(raw.get("records-part1-cardinality") or "").strip(),
                    link_relation=(raw.get("records-part1-link-relation") or "").strip(),
                    link_group=(raw.get("records-part1-link-group") or "").strip(),
                    default=(raw.get("records-part1-default") or "").strip(),
                    description=(raw.get("description") or "").strip(),
                )
            )
    return by_part


def _load_mapping_simple(path: Path) -> List[MapRow]:
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
                    src=src,
                    dst=dst,
                    cardinality=(raw.get("records-part1-cardinality") or "").strip(),
                    link_relation=(raw.get("records-part1-link-relation") or "").strip(),
                    link_group=(raw.get("records-part1-link-group") or "").strip(),
                    default=(raw.get("record-part1-default") or "").strip(),
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
    """Load YAML config if available.

    If PyYAML is not installed or the config file is missing/unreadable, returns {}.
    """
    if yaml is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}



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
    """Parse WMDR geoLocation (often a space-separated 'lat lon [z]') into GeoJSON Point.

    Returns a 2D or 3D Point depending on whether a third coordinate (altitude/height) is present.
    """
    coords = _parse_pos_lon_lat_z(raw)
    if coords is None:
        return None
    return {"type": "Point", "coordinates": coords}

def _iso_or_open_end(v: Any) -> str:
    """Normalize end datetimes: None/'' -> '..' (open end)."""
    if v in (None, ""):
        return ".."
    return str(v)


def _extract_interval(entry: Dict[str, Any]) -> Tuple[Any, Any, bool]:
    """Return (start, end, used_time_wrapper).

    Supports both:
      - legacy keys: time.interval[0] / time.interval[1]
      - current keys: time.interval == [start, end]
      - direct keys: beginPosition / endPosition
    """
    t = entry.get("time")
    if isinstance(t, dict):
        interval = t.get("interval")
        if isinstance(interval, list) or isinstance(interval, tuple):
            start = interval[0] if len(interval) > 0 else None
            end = interval[1] if len(interval) > 1 else None
        else:
            start = t.get("interval[0]") or t.get("start") or t.get("begin")
            end = t.get("interval[1]") or t.get("end")
        return start, end, True

    start = entry.get("beginPosition") or entry.get("begin")
    end = entry.get("endPosition") or entry.get("end")
    return start, end, False

def _parse_pos_lon_lat_z(raw: Any) -> Optional[List[Any]]:
    """Parse WMDR 'lat lon [z]' string into [lon, lat, z?] list."""
    if raw is None:
        return None

    # Accept already-parsed GeoJSON geometry dicts
    if isinstance(raw, dict) and isinstance(raw.get("coordinates"), list):
        coords = raw.get("coordinates")
        # keep as-is (2D or 3D)
        return coords  # type: ignore[return-value]
    if isinstance(raw, dict):
        for k in ("pos", "value", "text", "geoLocation", "geometry"):
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
    out: List[Any] = [lon, lat]
    if len(nums) >= 3:
        z = nums[2]
        # store clean ints when possible
        if abs(z - round(z)) < 1e-9:
            out.append(int(round(z)))
        else:
            out.append(z)
    return out


def _to_history_lists(history_key: str, items: List[Dict[str, Any]], ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a list of per-interval history objects into list-based serialization."""
    # base key for possible value back-fill (e.g., descriptionHistory -> description)
    local = history_key.split(":", 1)[-1]
    base = local[:-7] if local.endswith("History") else local

    # candidates to back-fill values when history objects only carry time/id
    cand_vals: List[Any] = []
    if base in ctx:
        cand_vals.append(ctx.get(base))
    # also try namespace-preserving base
    if ":" in history_key:
        ns = history_key.split(":", 1)[0]
        ns_base = f"{ns}:{base}"
        if ns_base in ctx:
            cand_vals.append(ctx.get(ns_base))
    base_val = next((v for v in cand_vals if v not in (None, "", {}, [])), None)

    grouped: Dict[str, Dict[str, Any]] = {}
    used_time_wrapper_any = False

    for entry in items:
        if not isinstance(entry, dict):
            continue

        ident = entry.get("identifier") or entry.get("@gml:id") or entry.get("@id") or ""
        gid = str(ident) if ident else "__noid__"
        start, end, used_tw = _extract_interval(entry)
        used_time_wrapper_any = used_time_wrapper_any or used_tw

        grp = grouped.setdefault(gid, {"identifier": ident} if ident else {})
        dt_list = grp.setdefault("__dt_list__", [])
        dt_list.append([start, _iso_or_open_end(end)])

        # collect value fields
        value_keys = []
        for k, v in entry.items():
            if k in ("identifier", "@gml:id", "@id", "beginPosition", "endPosition", "time", "begin", "end"):
                continue
            value_keys.append(k)
            out_k = "hrefs" if k == "href" else k  # href -> hrefs
            grp.setdefault(out_k, []).append(v)

        # if there are no value fields, but we have a base value, attach it
        if not value_keys and base_val is not None:
            grp.setdefault(base, []).append(base_val)

    out: List[Dict[str, Any]] = []
    for gid, grp in grouped.items():
        dts = grp.pop("__dt_list__", [])
        if not dts:
            continue
        # match the examples:
        # - if history used begin/end and we only have one interval, flatten to [start, end]
        if not used_time_wrapper_any and len(dts) == 1:
            grp["datetimes"] = dts[0]
        else:
            grp["datetimes"] = dts
        out.append(grp)

    return out


def _to_geometry_history_moving_point(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert geospatialLocation history entries into a MovingPoint-like structure.

    Output matches the requested OGC Records-style representation:

      {
        "type": "MovingPoint",
        "coordinates": [[lon, lat, z], ...],
        "datetimes": [[begin, end], ...]
      }

    Entries are ordered oldest -> newest. If an end is missing, it is inferred from the next
    begin; the newest entry ends with '..'.
    """
    parsed: List[Dict[str, Any]] = []

    for entry in items:
        if not isinstance(entry, dict):
            continue

        start, end, _ = _extract_interval(entry)

        # normalize / validate start
        if start in (None, ""):
            continue
        start_s = str(start)

        coords = _parse_pos_lon_lat_z(entry.get("geometry") or entry.get("geoLocation") or entry.get("pos"))
        if coords is None:
            # sometimes stored as a Point dict under "geometry"
            g = entry.get("geometry")
            if isinstance(g, dict) and isinstance(g.get("coordinates"), list):
                coords = g.get("coordinates")  # type: ignore[assignment]
        if coords is None:
            continue

        parsed.append({"start": start_s, "end": None if end in (None, "") else str(end), "coords": coords})

    if not parsed:
        return None

    # oldest -> newest
    parsed.sort(key=lambda p: p["start"])

    # infer missing ends from next begin; newest ends with '..'
    for i, p in enumerate(parsed):
        if p["end"] in (None, "", ".."):
            if i < len(parsed) - 1:
                p["end"] = parsed[i + 1]["start"]
            else:
                p["end"] = ".."

    return {
        "type": "MovingPoint",
        "coordinates": [p["coords"] for p in parsed],
        "datetimes": [[p["start"], _iso_or_open_end(p["end"])] for p in parsed],
    }

def _rewrite_history_in_place(node: Any) -> None:
    """Recursively rewrite *History keys to list-based serialization in-place."""
    if isinstance(node, list):
        for it in node:
            _rewrite_history_in_place(it)
        return
    if not isinstance(node, dict):
        return

    # First handle history keys at this level
    for k, v in list(node.items()):
        if not isinstance(k, str):
            continue

        # geospatial location history -> geometryHistory (MovingPoint)
        if k.endswith("geospatialLocationHistory") and isinstance(v, list) and all(isinstance(x, dict) for x in v):
            mp = _to_geometry_history_moving_point(v)  # type: ignore[arg-type]
            if mp is not None:
                node["geometryHistory"] = mp
            node.pop(k, None)
            continue

        if k.endswith("History") and isinstance(v, list) and all(isinstance(x, dict) for x in v):
            node[k] = _to_history_lists(k, v, node)  # type: ignore[arg-type]

    # Recurse into remaining children
    for v in node.values():
        _rewrite_history_in_place(v)


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


def _infer_interval_side(src_path: str) -> int:
    """Infer whether a mapping row contributes the begin (0) or end (1) of a time interval."""
    last = (src_path.split(".")[-1] if src_path else "").lower()
    if last in ("endposition", "end", "dateclosed", "enddate"):
        return 1
    if last in ("beginposition", "begin", "dateestablished", "start", "startdate"):
        return 0
    # fallback heuristics
    if "end" in last and "begin" not in last and "start" not in last:
        return 1
    return 0


_DATE_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}Z$")
_MIDNIGHT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T00:00:00(?:Z|\+00:00)$")


def _normalize_interval_value(val: Any, *, prefer_date: bool) -> str:
    """Normalize interval values.

    - None/'' -> '..'
    - 'YYYY-MM-DDZ' -> 'YYYY-MM-DD'
    - If prefer_date and datetime is midnight -> date-only
    """
    if val in (None, ""):
        return ".."
    s = str(val).strip()
    if s in ("", "None"):
        return ".."
    if _DATE_Z_RE.match(s):
        s = s[:-1]
    if prefer_date and "T" in s and _MIDNIGHT_RE.match(s):
        s = s.split("T", 1)[0]
    return s


def _set_time_interval(root: Dict[str, Any], dst_path: str, val: Any, idx_tuple: Optional[Tuple[int, ...]], src_path: str) -> None:
    """Set/merge values into a `time.interval` array of length 2.

    The mapping file now targets `...time.interval` (without [0]/[1]).
    We infer whether a mapping row contributes the begin vs end side from the WMDR10 source path.
    """
    tokens = [t for t in dst_path.split(".") if t]
    if not tokens or tokens[-1] != "interval":
        _set_nested(root, dst_path, val, idx_tuple)
        return

    prefer_date = tokens == ["time", "interval"]
    side = _infer_interval_side(src_path)

    cur: Any = root
    idx_pos = 0

    # walk everything up to the 'interval' leaf
    for tok in tokens[:-1]:
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
            if lst[use_idx] is None or not isinstance(lst[use_idx], dict):
                lst[use_idx] = {}
            cur = lst[use_idx]
        else:
            if not isinstance(cur, dict):
                return
            if tok not in cur or not isinstance(cur[tok], dict):
                cur[tok] = {}
            cur = cur[tok]

    if not isinstance(cur, dict):
        return

    interval = cur.get("interval")
    if not isinstance(interval, list):
        interval = ["..", ".."]
        cur["interval"] = interval
    else:
        while len(interval) < 2:
            interval.append("..")
        if len(interval) > 2:
            interval = interval[:2]
            cur["interval"] = interval

    interval[side] = _normalize_interval_value(val, prefer_date=prefer_date)


def _dumps_compact_lists(obj: Any, *, indent: int = 2) -> str:
    """JSON pretty-printer that keeps simple lists (and lists-of-lists) on one line."""

    def is_scalar(x: Any) -> bool:
        return x is None or isinstance(x, (str, int, float, bool))

    def is_simple_list(lst: list) -> bool:
        if not lst:
            return True
        if all(is_scalar(x) for x in lst):
            return True
        if all(isinstance(x, list) and all(is_scalar(y) for y in x) for x in lst):
            return True
        return False

    def fmt(o: Any, level: int) -> str:
        pad = " " * (indent * level)

        if isinstance(o, dict):
            if not o:
                return "{}"
            items = []
            for k, v in o.items():
                kk = json.dumps(k, ensure_ascii=False)
                vv = fmt(v, level + 1)
                items.append(f"{pad}{' ' * indent}{kk}: {vv}")
            inner = ",\n".join(items)
            return "{\n" + inner + f"\n{pad}" + "}"

        if isinstance(o, list):
            if is_simple_list(o):
                return json.dumps(o, ensure_ascii=False, separators=(',', ': '))
            if not o:
                return "[]"
            items = [f"{pad}{' ' * indent}{fmt(v, level + 1)}" for v in o]
            inner = ",\n".join(items)
            return "[\n" + inner + f"\n{pad}" + "]"

        return json.dumps(o, ensure_ascii=False)

    return fmt(obj, 0) + "\n"

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

            # time interval special case (new mapping uses time.interval without [0]/[1])
            if row.dst == "time.interval" or row.dst.endswith(".time.interval"):
                _set_time_interval(rec, row.dst, val, idx_tuple, row.src)
                continue

            # geometry objects inside histories (e.g., properties.*History[*].geometry)
            if row.dst != "geometry" and row.dst.endswith(".geometry"):
                geom = _parse_geo_pos(val)
                if geom is not None:
                    _set_nested(rec, row.dst, geom, idx_tuple)
                else:
                    _set_nested(rec, row.dst, val, idx_tuple)
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

    # normalize WMDR history fields to list-based serialization
    _rewrite_history_in_place(rec.get("properties"))

    # Promote geometryHistory to the Feature root (as requested) and use newest coordinate for geometry.
    props = rec.get("properties")
    if isinstance(props, dict) and "geometryHistory" in props:
        gh = props.pop("geometryHistory")
        rec["geometryHistory"] = gh

        # Ensure geometry uses the newest coordinate from the history (last entry)
        if isinstance(gh, dict):
            coords = gh.get("coordinates")
            if isinstance(coords, list) and coords:
                rec["geometry"] = {"type": "Point", "coordinates": coords[-1]}

        # If top-level time.interval is missing, infer from history
        if "time" not in rec and isinstance(gh, dict):
            dts = gh.get("datetimes")
            if isinstance(dts, list) and dts:
                start0 = dts[0][0] if isinstance(dts[0], list) and len(dts[0]) > 0 else None
                endN = dts[-1][1] if isinstance(dts[-1], list) and len(dts[-1]) > 1 else None
                if start0 is not None or endN is not None:
                    rec["time"] = {"interval": [_normalize_interval_value(start0, prefer_date=True), _normalize_interval_value(endN, prefer_date=True)]}

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
    rt_fac = "stationPlatform"
    rt_obs = "observationCollection"
    rt_dep = "deployment"

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

            out_path.write_text(_dumps_compact_lists(fc, indent=2), encoding="utf-8")
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
