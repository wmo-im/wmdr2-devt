#!/usr/bin/env python3
"""
Convert WMDR10 XML files into a lean JSON representation and apply a small
post-processing normalization pass that helps the downstream WMDR2/GeoJSON
conversion.

Why this script now does more than a plain export
-------------------------------------------------
The WMDR10 class already performs the heavy XML -> simplified-JSON conversion.
This script adds a second, stage-1 normalization pass that is easier to keep
local here than to thread through the larger WMDR10 implementation right away.

The added normalization intentionally stays conservative:
- collapse a few remaining wrapper objects such as:
  * sampling: {"Sampling": null} -> sampling: null
  * result: {"ResultSet": {...}} -> result: {...}
- normalize geospatialLocation histories by sorting them by beginPosition and,
  when possible, inferring intermediate endPosition values from the next
  beginPosition. This makes location histories easier to convert into WMDR2
  temporalGeometry / MovingPoint objects downstream.

The conversion remains loss-minimizing for WMDR-relevant information.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _keynorm(value: str | None) -> str:
    """Return a relaxed key name for wrapper comparisons."""
    if value is None:
        return ""
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _parse_iso_like(value: Any) -> datetime | None:
    """Parse a WMDR date or datetime string into a datetime when possible."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text == "..":
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _is_open_interval_end(value: Any) -> bool:
    """Return True when the value represents an open or missing interval end."""
    return value in (None, "", "..")


def _collapse_local_wrappers(node: Any, parent_key: str | None = None) -> Any:
    """
    Collapse a few remaining wrapper patterns left after WMDR10 simplification.

    Conservative rules only:
    - {'Sampling': None} under 'sampling' -> None
    - {'ResultSet': {...}} under 'result' -> {...}
    - same-name wrappers such as {'schedule': {'Schedule': {...}}} -> {...}
    """
    if isinstance(node, list):
        return [_collapse_local_wrappers(item, parent_key=parent_key) for item in node]

    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        out[key] = _collapse_local_wrappers(value, parent_key=key)

    if len(out) == 1:
        (only_key, only_val), = out.items()
        pnorm = _keynorm(parent_key)
        onorm = _keynorm(only_key)

        if pnorm == "sampling" and onorm == "sampling" and only_val is None:
            return None

        if pnorm == "result" and onorm == "resultset":
            return only_val

        if pnorm and onorm and (pnorm == onorm or onorm == pnorm.rstrip("s")):
            return only_val

    return out


def _normalize_geospatial_location_history_inplace(node: Any) -> None:
    """
    Recursively normalize geospatialLocation histories in place.

    For list-valued geospatialLocation histories:
    - sort by beginPosition ascending when available
    - if an entry has no endPosition and the following entry has a beginPosition,
      infer the current endPosition from that following beginPosition

    This preserves the original geoLocation strings while making history
    intervals explicit enough for later MovingPoint generation.
    """
    if isinstance(node, list):
        for item in node:
            _normalize_geospatial_location_history_inplace(item)
        return

    if not isinstance(node, dict):
        return

    for value in node.values():
        _normalize_geospatial_location_history_inplace(value)

    geoloc = node.get("geospatialLocation")
    if not isinstance(geoloc, list) or not geoloc:
        return
    if not all(isinstance(item, dict) for item in geoloc):
        return

    def sort_key(item: dict[str, Any]) -> tuple[int, datetime]:
        dt = _parse_iso_like(item.get("beginPosition"))
        if dt is None:
            return (1, datetime.max)
        return (0, dt)

    geoloc.sort(key=sort_key)

    for idx in range(len(geoloc) - 1):
        current = geoloc[idx]
        following = geoloc[idx + 1]
        next_begin = following.get("beginPosition")
        if _is_open_interval_end(current.get("endPosition")) and next_begin not in (None, "", ".."):
            current["endPosition"] = next_begin


def normalize_stage1_payload(data: dict[str, Any]) -> dict[str, Any]:
    """
    Apply the stage-1 post-processing normalization to a WMDR10 JSON payload.
    """
    normalized = copy.deepcopy(data)
    normalized = _collapse_local_wrappers(normalized)
    _normalize_geospatial_location_history_inplace(normalized)
    return normalized


def main() -> None:
    # Imported lazily so the normalization helpers remain importable for tests
    # without requiring the full repository package layout.
    from utils.config import load_config
    from wmdr10.wmdr10 import WMDR10

    config = load_config(Path("config.yaml"))
    source_path = Path(config["convert_wmdr10_xml_to_wmdr10_json"]["source"])
    target_path = Path(config["convert_wmdr10_xml_to_wmdr10_json"]["target"])
    target_path.mkdir(parents=True, exist_ok=True)

    xml_files = list(source_path.glob("*.xml"))
    for xml_file in xml_files:
        wmdr10 = WMDR10(xml_file)
        wmdr10.data = normalize_stage1_payload(wmdr10.data)

        wmdr10.export(path=target_path / xml_file.name.replace(".xml", ""))

        print(f"{wmdr10.export(parts='header', path=target_path / xml_file.name)} created.")
        print(f"{wmdr10.export(parts='facility', path=target_path / xml_file.name)} created.")
        print(f"{wmdr10.export(parts='observations', path=target_path / xml_file.name)} created.")
        print(f"{wmdr10.export(parts='deployments', path=target_path / xml_file.name)} created.")

        print(f"{wmdr10.export(parts='observations', index=1, path=target_path / xml_file.name)} created.")
        print(f"{wmdr10.export(parts='observations', index=5, path=target_path / xml_file.name)} created.")
        print(f"{wmdr10.export(parts='deployments', index=1, path=target_path / xml_file.name)} created.")
        print(f"{wmdr10.export(parts='deployments', index=3, path=target_path / xml_file.name)} created.")
        print(f"Finished processing '{xml_file.name}'.")


if __name__ == "__main__":
    main()
