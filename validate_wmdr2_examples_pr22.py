#!/usr/bin/env python3
"""Validate WMDR2 JSON examples against a PR-22 WMDR2 schema.

The PR-22 schema uses the public field name
``properties.observingCapabilities`` while the current wmdr2-devt examples use
``properties.observationSeries``. This validator applies a small set of
in-memory compatibility adaptations before validation. It does not modify the
source example files.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

VERSION = "validate_wmdr2_examples_pr22.py pr22-application-areas-operating-status-v8"

# These records are generated from source material that is currently known not
# to contain enough information for strict PR-22 schema validation, or contains
# local phone numbers that intentionally fail the strict OGC Contact/E.164
# pattern.  The script only suppresses the expected error category for each
# listed record, and only when --allow-known-nonvalidating is used.
KNOWN_NONVALIDATING: dict[str, set[str]] = {
    "20200304_0-20000-0-06494.json": {"missing_time"},
    "20251218_0-20000-0-45004.json": {"missing_time"},
    "20260318_0-20008-0-DAV.json": {"missing_time"},
    "20220511_0-404-0-63707.json": {"phone_e164"},
    "20250314_0-404-800-4AA01.json": {"phone_e164"},
    "20250504_0-20008-0-NRB.json": {"phone_e164"},
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _is_plain_int(value: Any) -> bool:
    # bool is a subclass of int; it must not be converted as a code value.
    return isinstance(value, int) and not isinstance(value, bool)


def _path_text(parts: list[Any]) -> str:
    return "/".join(str(part) for part in parts) or "<root>"


def _stringify_plain_ints(value: Any, path: list[Any], notes: list[str]) -> Any:
    """Convert integer code values to strings, preserving shape."""
    if _is_plain_int(value):
        notes.append(f"converted integer {_path_text(path)} to string")
        return str(value)
    if isinstance(value, list):
        converted: list[Any] = []
        changed = False
        for index, item in enumerate(value):
            if _is_plain_int(item):
                converted.append(str(item))
                changed = True
            else:
                converted.append(item)
        if changed:
            notes.append(f"converted integer item(s) in {_path_text(path)} to string")
        return converted
    return value


def _code_from_status_entry(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("nilReason", "@nilReason"):
            if isinstance(value.get(key), str) and value.get(key):
                return value[key]
        for key in ("instrumentOperatingStatus", "operatingStatus", "href", "url", "value", "#text", "text"):
            item = value.get(key)
            if item not in (None, "", [], {}):
                return _code_from_status_entry(item)
        return None
    if _is_plain_int(value):
        return str(value)
    return value


def _status_time(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    time_obj = value.get("time")
    if isinstance(time_obj, dict):
        interval = time_obj.get("interval")
        if isinstance(interval, list) and interval:
            return {"interval": interval}
    start = value.get("validFrom") or value.get("beginPosition") or value.get("begin") or value.get("from") or value.get("start")
    end = value.get("validTo") or value.get("endPosition") or value.get("end") or value.get("stop")
    if start not in (None, "", [], {}) or end not in (None, "", [], {}):
        return {"interval": [start if start not in (None, "", [], {}) else "..", end if end not in (None, "", [], {}) else ".."]}
    return None


def _split_legacy_operating_status_history(config: dict[str, Any], path: list[Any], notes: list[str]) -> list[dict[str, Any]]:
    """Split old list-valued operatingStatus histories for validation only."""
    raw_status = config.get("operatingStatus")
    if not isinstance(raw_status, list):
        return [config]

    variants: list[dict[str, Any]] = []
    for item in raw_status:
        status_code = _code_from_status_entry(item)
        if status_code in (None, "", [], {}):
            continue
        variant = deepcopy(config)
        variant["operatingStatus"] = status_code
        status_time = _status_time(item)
        if status_time:
            variant["time"] = status_time
        variants.append(variant)

    if variants:
        notes.append(f"split legacy list-valued {_path_text(path + ['operatingStatus'])} into scalar configurations")
        return variants
    return [config]


def _adapt_nil_reasons(value: Any, path: list[Any], notes: list[str]) -> Any:
    """Replace simple {"nilReason": "..."} objects by the string reason.

    This is a temporary compatibility adaptation for the PR-22 schema, whose
    controlled-vocabulary fields are currently string-only in several places.
    """
    if isinstance(value, dict):
        if set(value.keys()) == {"nilReason"} and isinstance(value.get("nilReason"), str):
            notes.append(f"converted nilReason object at {_path_text(path)} to string")
            return value["nilReason"]
        return {key: _adapt_nil_reasons(item, path + [key], notes) for key, item in value.items()}
    if isinstance(value, list):
        return [_adapt_nil_reasons(item, path + [index], notes) for index, item in enumerate(value)]
    return value


def _iter_observing_capabilities(props: dict[str, Any]) -> list[dict[str, Any]]:
    capabilities = props.get("observingCapabilities")
    if isinstance(capabilities, list):
        return [item for item in capabilities if isinstance(item, dict)]
    if isinstance(capabilities, dict):
        return [capabilities]
    return []


def adapt_for_pr22_schema(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return a schema-compatible copy of a WMDR2 example and adaptation notes."""
    doc = deepcopy(payload)
    notes: list[str] = []

    doc = _adapt_nil_reasons(doc, [], notes)
    if not isinstance(doc, dict):
        return doc, notes

    props = doc.get("properties")
    if not isinstance(props, dict):
        return doc, notes

    # wmdr2-devt uses the model term ObservationSeries as JSON property
    # `observationSeries`.  PR #22 uses `observingCapabilities` as the public
    # schema property.  Rename only in the validation copy.
    if "observationSeries" in props:
        observation_series = props.pop("observationSeries")
        if "observingCapabilities" in props:
            props["observingCapabilities"] = _as_list(props["observingCapabilities"]) + _as_list(observation_series)
            notes.append("merged properties.observationSeries into existing properties.observingCapabilities")
        else:
            props["observingCapabilities"] = observation_series
            notes.append("renamed properties.observationSeries to properties.observingCapabilities")

    for capability_index, capability in enumerate(_iter_observing_capabilities(props)):
        capability_path = ["properties", "observingCapabilities", capability_index]

        # PR-22 schema currently treats code-valued strings more strictly than
        # the devt examples.  Convert only the integer code values known to
        # occur along the XML -> WMDR1 -> WMDR2 path.
        if "observedProperty" in capability:
            capability["observedProperty"] = _stringify_plain_ints(
                capability["observedProperty"],
                capability_path + ["observedProperty"],
                notes,
            )

        # PR-22 and WMDR2 use the plural multi-valued field
        # `applicationAreas`.  Older generated devt examples may still contain
        # the singular `applicationArea`; adapt it only in memory and always
        # present a list to the schema validator.
        application_area_values: list[Any] = []
        if "applicationAreas" in capability:
            application_area_values.extend(_as_list(capability.get("applicationAreas")))
        if "applicationArea" in capability:
            application_area_values.extend(_as_list(capability.pop("applicationArea")))
            notes.append("renamed properties.observingCapabilities[].applicationArea to applicationAreas")
        if application_area_values:
            capability["applicationAreas"] = _stringify_plain_ints(
                application_area_values,
                capability_path + ["applicationAreas"],
                notes,
            )

        configurations = capability.get("observingConfigurations")
        if isinstance(configurations, list):
            expanded_configurations: list[Any] = []
            for config_index, config in enumerate(configurations):
                if not isinstance(config, dict):
                    expanded_configurations.append(config)
                    continue
                config_variants = _split_legacy_operating_status_history(
                    config,
                    capability_path + ["observingConfigurations", config_index],
                    notes,
                )
                for variant in config_variants:
                    if "observingMethod" in variant:
                        variant["observingMethod"] = _stringify_plain_ints(
                            variant["observingMethod"],
                            capability_path + ["observingConfigurations", config_index, "observingMethod"],
                            notes,
                        )
                    if "operatingStatus" in variant:
                        variant["operatingStatus"] = _stringify_plain_ints(
                            variant["operatingStatus"],
                            capability_path + ["observingConfigurations", config_index, "operatingStatus"],
                            notes,
                        )
                    expanded_configurations.append(variant)
            capability["observingConfigurations"] = expanded_configurations

    return doc, notes


def iter_json_examples(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("*.json"))
    return sorted(set(files))


def is_probable_schema(path: Path) -> bool:
    lower_parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    return "schemas" in lower_parts or "schema" in name or "bundled" in name


def _error_category(error: ValidationError) -> str | None:
    path = [str(part) for part in error.absolute_path]

    if error.validator == "required" and error.message == "'time' is a required property":
        if "observingConfigurations" in path or "observingProcedures" in path:
            return "missing_time"

    if error.validator == "pattern" and "contacts" in path and "phones" in path and path[-1:] == ["value"]:
        return "phone_e164"

    return None


def _format_errors(path: Path, errors: list[ValidationError]) -> str:
    lines = [f"FAIL {path}"]
    for err in errors[:20]:
        location = _path_text(list(err.absolute_path))
        lines.append(f"  - {location}: {err.message}")
    if len(errors) > 20:
        lines.append(f"  ... {len(errors) - 20} more errors")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "positional_paths",
        nargs="*",
        type=Path,
        help="JSON example file(s) or directory/directories to validate. Positional form is retained for compatibility.",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        type=Path,
        default=["results/wmdr2_json_examples"],
        help="JSON example file(s) or directory/directories to validate.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("schemas/wmdr2-bundled.json"),
        help="Schema to validate against. Default: schemas/wmdr2-bundled.json",
    )
    parser.add_argument(
        "--no-alias",
        action="store_true",
        help="Do not apply PR-22 compatibility adaptations.",
    )
    parser.add_argument(
        "--show-adaptations",
        action="store_true",
        help="Print files where PR-22 compatibility adaptations were applied.",
    )
    parser.add_argument(
        "--allow-known-nonvalidating",
        action="store_true",
        help="Do not fail for known source-derived validation errors.",
    )
    parser.add_argument(
        "--show-known-nonvalidating",
        action="store_true",
        help="Print the known source-derived validation errors that were tolerated.",
    )
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args(argv)

    input_paths = [*args.positional_paths, *args.paths]
    if not input_paths:
        parser.error("provide JSON example paths either positionally or via --paths")

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    files = [path for path in iter_json_examples(input_paths) if not is_probable_schema(path)]
    if not files:
        print("No JSON example files found.", file=sys.stderr)
        return 2

    failures: list[str] = []
    ok_count = 0
    adapted_count = 0
    known_nonvalidating_count = 0

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - command-line diagnostics
            failures.append(f"{path}: cannot parse JSON: {exc}")
            continue

        notes: list[str] = []
        validation_payload = payload
        if not args.no_alias and isinstance(payload, dict):
            validation_payload, notes = adapt_for_pr22_schema(payload)
            if notes:
                adapted_count += 1
                if args.show_adaptations:
                    print(f"ADAPT {path}: " + "; ".join(notes))

        errors = sorted(validator.iter_errors(validation_payload), key=lambda err: list(err.absolute_path))
        if not errors:
            ok_count += 1
            print(f"OK   {path}")
            continue

        error_categories = {_error_category(error) for error in errors}
        if args.allow_known_nonvalidating and None not in error_categories:
            allowed = KNOWN_NONVALIDATING.get(path.name, set())
            categories = {category for category in error_categories if category is not None}
            if categories and categories.issubset(allowed):
                known_nonvalidating_count += 1
                if args.show_known_nonvalidating:
                    print(_format_errors(path, errors))
                continue

        failures.append(_format_errors(path, errors))

    print(
        f"\nChecked {len(files)} JSON file(s); "
        f"{ok_count} valid; "
        f"{adapted_count} adapted for PR-22 schema; "
        f"{known_nonvalidating_count} known non-validating."
    )

    if failures:
        print("\nFailures:")
        print("\n".join(failures[:50]))
        if len(failures) > 50:
            print(f"... {len(failures) - 50} more failing file(s)")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
