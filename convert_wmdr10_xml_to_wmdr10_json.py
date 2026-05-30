#!/usr/bin/env python3
"""
Convert WMDR 1.0 XML files into a lean WMDR10 JSON representation.

The heavy XML -> simplified JSON conversion is still done by ``WMDR10``.  This
script applies a conservative stage-1 normalization pass before exporting one
full-record JSON file per XML file for the downstream WMDR10 -> WMDR2 converter.

Stage-1 normalization currently does the following:

- Collapse a few remaining wrapper objects, for example::

      sampling: {"Sampling": null} -> sampling: null
      result: {"ResultSet": {...}} -> result: {...}

- Normalize common ``unknown`` placeholders in scalar strings.
- Normalize geospatial-location histories by sorting by ``beginPosition`` and,
  when possible, inferring intermediate ``endPosition`` values from the next
  ``beginPosition``.
- Normalize XML/GML source identifiers:

  * preserve source identifiers for referenceable WMDR entities, especially
    deployments, contacts, equipment, and instruments;
  * expose preserved source identifiers as plain ``id`` fields;
  * remove XML/GML bookkeeping identifiers from descriptive/container objects
    such as descriptions, territories, schedules, reference datums, and history
    wrappers.

This lets XML-producing systems keep stable deployment/contact/equipment IDs
into WMDR2 without carrying arbitrary ``@gml:id`` fields through the whole JSON
pipeline. Partial/debug exports are intentionally not produced.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


# Context keys under which an XML/GML id identifies a real, referenceable WMDR
# entity and should therefore be preserved as a plain JSON ``id``.
ENTITY_ID_CONTEXT_KEYS = {
    "deployment",
    "deployments",
    "equipment",
    "equipments",
    "instrument",
    "instruments",
    "contact",
    "contacts",
    "responsibleparty",
    "recordowner",
    "operator",
    "operators",
    "manufacturer",
    "manufacturers",
}

XML_ID_KEYS = {
    "@gml:id",
    "gml:id",
    "@id",
    "@xml:id",
    "xml:id",
}

XML_BOOKKEEPING_KEYS = {
    "@xmlns",
    "xmlns",
    "schemaLocation",
    "@schemaLocation",
    "@xsi:schemaLocation",
    "xsi:schemaLocation",
}


def _keynorm(value: str | None) -> str:
    """Return a relaxed key name for wrapper and context comparisons."""
    if value is None:
        return ""
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _parse_iso_like(value: Any) -> datetime | None:
    """Parse a WMDR date or datetime string into a ``datetime`` when possible."""
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


def _normalize_unknown_string(value: Any) -> Any:
    """Normalize common unknown placeholders in plain text strings."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    # Preserve codelist URIs.  The downstream converter decides whether a URI
    # ending in /unknown is meaningful for the target element.
    if text.startswith(("http://", "https://")):
        return value

    normalized = re.sub(
        r"\((unknown|null|none|nil)\)",
        "unknown",
        text,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\bunknown\s*/\s*unknown\b",
        "unknown",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\bunknown\s+unknown\b",
        "unknown",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if normalized.lower() in {"unknown", "none", "null", "nil"}:
        return "unknown"

    return normalized


def _normalize_unknown_strings_inplace(node: Any) -> Any:
    """Recursively normalize unknown placeholders in scalar string values."""
    if isinstance(node, list):
        return [_normalize_unknown_strings_inplace(item) for item in node]

    if isinstance(node, dict):
        return {
            key: _normalize_unknown_strings_inplace(value)
            for key, value in node.items()
        }

    return _normalize_unknown_string(node)


def _collapse_local_wrappers(node: Any, parent_key: str | None = None) -> Any:
    """Collapse a few remaining wrapper patterns left after WMDR10 simplification.

    Conservative rules only:

    - ``{"Sampling": None}`` under ``sampling`` -> ``None``
    - ``{"ResultSet": {...}}`` under ``result`` -> ``{...}``
    - same-name wrappers such as ``{"schedule": {"Schedule": {...}}}`` -> ``{...}``
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


def _is_entity_id_context(path: Sequence[str]) -> bool:
    """Return True when the current object path denotes a referenceable entity."""
    return any(_keynorm(item) in ENTITY_ID_CONTEXT_KEYS for item in path)


def _looks_like_xml_generated_id(value: Any) -> bool:
    """Return True for IDs that look like XML/GML generated identifiers."""
    if not isinstance(value, str):
        return False

    text = value.strip()
    if not text:
        return False

    # Typical WMDR/GML examples seen in generated source JSON.
    if text.startswith(("id_", "uuid.", "uuid_")):
        return True

    # UUID-like strings, optionally without a leading ``id_``.
    return bool(
        re.fullmatch(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            text,
        )
    )


def _normalize_source_ids(node: Any, path: tuple[str, ...] = ()) -> Any:
    """Normalize XML/GML source IDs according to WMDR entity context.

    XML ``@gml:id`` values are useful only when they identify referenceable WMDR
    entities.  In those contexts they are retained as plain ``id`` fields.  In
    non-entity contexts they are removed as XML bookkeeping.
    """
    if isinstance(node, list):
        return [_normalize_source_ids(item, path=path) for item in node]

    if not isinstance(node, dict):
        return node

    entity_context = _is_entity_id_context(path)
    preserved_id: str | None = None
    out: dict[str, Any] = {}

    for key, value in node.items():
        if key in XML_BOOKKEEPING_KEYS:
            continue

        if key in XML_ID_KEYS:
            if entity_context and isinstance(value, str) and value.strip():
                preserved_id = value.strip()
            continue

        # Some simplification pipelines may already have converted @gml:id to
        # ``id``.  Keep such ids for entity-like objects, but remove generated
        # XML ids from descriptive/container objects.
        if key == "id" and not entity_context and _looks_like_xml_generated_id(value):
            continue

        out[key] = _normalize_source_ids(value, path=path + (key,))

    if preserved_id and not out.get("id"):
        return {"id": preserved_id, **out}

    return out


def _normalize_geospatial_location_history_inplace(node: Any) -> None:
    """Recursively normalize geospatial-location histories in place.

    For list-valued ``geospatialLocation`` histories:

    - sort by ``beginPosition`` ascending when available;
    - if an entry has no ``endPosition`` and the following entry has a
      ``beginPosition``, infer the current ``endPosition`` from that following
      ``beginPosition``.

    This preserves original ``geoLocation`` strings while making history
    intervals explicit enough for later WMDR2 temporal-geometry generation.
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
    """Apply stage-1 post-processing normalization to a WMDR10 JSON payload."""
    normalized = copy.deepcopy(data)
    normalized = _collapse_local_wrappers(normalized)
    normalized = _normalize_unknown_strings_inplace(normalized)
    normalized = _normalize_source_ids(normalized)
    _normalize_geospatial_location_history_inplace(normalized)
    return normalized


def _discover_config_path(explicit: Path | None = None) -> Path:
    """Return the config path for CLI execution."""
    if explicit is not None:
        path = explicit.expanduser()
        return path if path.is_absolute() else Path.cwd() / path

    candidates: list[Path] = []
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        candidates.extend([base / "config.yaml", base / "config.yml"])
        candidates.extend(parent / "config.yaml" for parent in base.parents)
        candidates.extend(parent / "config.yml" for parent in base.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists() and candidate.is_file():
            return candidate

    return Path("config.yaml")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert WMDR10 XML files to normalized WMDR10 JSON files."
    )
    parser.add_argument("--config", type=Path, help="Path to config.yaml/config.yml.")
    parser.add_argument("--source", type=Path, help="Input directory containing WMDR10 XML files.")
    parser.add_argument("--target", type=Path, help="Output directory for WMDR10 JSON files.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run XML -> normalized WMDR10 JSON conversion."""
    # Imported lazily so normalization helpers remain importable in tests
    # without requiring the full repository package layout.
    from utils.config import load_config
    from wmdr10.wmdr10 import WMDR10

    args = _build_arg_parser().parse_args(argv)
    config_path = _discover_config_path(args.config)
    config = load_config(config_path)
    section = config["convert_wmdr10_xml_to_wmdr10_json"]

    source_path = args.source or Path(section["source"])
    target_path = args.target or Path(section["target"])

    if not source_path.is_absolute():
        source_path = config_path.parent / source_path
    if not target_path.is_absolute():
        target_path = config_path.parent / target_path

    target_path.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(source_path.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No XML files found in {source_path}")

    print(f"Using config: {config_path}")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")

    for xml_file in xml_files:
        wmdr10 = WMDR10(xml_file)
        wmdr10.data = normalize_stage1_payload(wmdr10.data)

        output_base = target_path / xml_file.with_suffix("").name
        output_path = wmdr10.export(path=output_base)
        print(f"{output_path} created.")
        print(f"Finished processing '{xml_file.name}'.")


if __name__ == "__main__":
    main()
