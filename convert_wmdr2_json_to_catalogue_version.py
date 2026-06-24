#!/usr/bin/env python3
"""
convert_wmdr2_json_to_catalogue_version.py

Convert embedded WMDR2 facility records to the catalogue-based representation.

The converter reads facility-centric WMDR2 JSON Features, collects full contact
and instrument objects into shared catalogues, and writes facility records with
minimal inline contact stubs and deployment-level instrument references.  The
output intentionally avoids a custom ``wmdr2`` wrapper property.

For the v0.2.x reusable-deployment model, ``properties.deployments[*].instrument``
is a scalar reference to one reusable instrument.  This module preserves that
scalar shape when externalising instruments; older versions accidentally wrapped
it as a one-item list.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - optional dependency for CLI convenience
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_PATTERN = "*.json"
CATALOGUE_MEDIA_TYPE = "application/json"


@dataclass(frozen=True)
class CataloguePaths:
    source: Path
    records_path: Path
    contacts_path: Path
    instruments_path: Path
    pattern: str = DEFAULT_PATTERN
    recursive: bool = True


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _non_empty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _clean_none(obj: Any, *, _path: tuple[str, ...] = ()) -> Any:
    """Remove empty members, preserving aligned ``temporalGeometry.methods`` slots."""

    def preserve_empty_list(path: tuple[str, ...]) -> bool:
        return len(path) >= 2 and path[-2:] == ("temporalGeometry", "methods")

    if isinstance(obj, dict):
        cleaned = {key: _clean_none(value, _path=_path + (key,)) for key, value in obj.items()}
        return {key: value for key, value in cleaned.items() if value not in (None, "", [], {})}

    if isinstance(obj, list):
        cleaned = [_clean_none(item, _path=_path) for item in obj]
        return [
            item
            for item in cleaned
            if item not in ("", {}) and (item != [] or preserve_empty_list(_path))
        ]

    return obj


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unknown"


def _stable_hash(payload: Any, length: int = 10) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def _dedupe_scalars(items: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for item in items:
        marker = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        if marker not in seen:
            seen.add(marker)
            out.append(item)
    return out


def _dedupe_dicts(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        marker = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        if marker not in seen:
            seen.add(marker)
            out.append(item)
    return out


def _normalize_email_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    email = value.strip().lower()
    if not email or "@" not in email:
        return None
    return email


def _contact_emails(contact: dict[str, Any]) -> list[str]:
    emails: list[str] = []
    for item in _as_list(contact.get("emails")):
        if isinstance(item, str):
            email = _normalize_email_value(item)
        elif isinstance(item, dict):
            email = _normalize_email_value(item.get("value") or item.get("email") or item.get("href"))
        else:
            email = None
        if email and email not in emails:
            emails.append(email)
    return emails


def contact_identifier(contact: dict[str, Any]) -> str:
    """Return a deterministic catalogue id for a contact."""
    emails = _contact_emails(contact)
    if emails:
        return f"contact:{emails[0]}"

    name = str(contact.get("name") or "").strip()
    organization = str(contact.get("organization") or "").strip()
    if name and organization:
        return f"contact:{_slug(name)}--{_slug(organization)}"

    seed = {
        "identifier": contact.get("identifier"),
        "name": name,
        "organization": organization,
        "position": contact.get("position"),
        "phones": contact.get("phones"),
        "addresses": contact.get("addresses"),
        "links": contact.get("links"),
    }
    base = _slug("--".join(part for part in (name, organization, str(contact.get("identifier") or "")) if part))
    return f"contact:{base}--{_stable_hash(seed, 8)}"


def _catalogue_href(catalogue_path: Path, records_path: Path, identifier: str) -> str:
    rel = Path(os.path.relpath(catalogue_path, start=records_path))
    return f"{rel.as_posix()}#{identifier}"


def _minimal_contact_stub(contact: dict[str, Any], identifier: str, *, href: str) -> dict[str, Any]:
    """Return a minimal OGC Records contact object that points to the catalogue."""
    stub: dict[str, Any] = {"identifier": identifier}

    if _non_empty(contact.get("name")):
        stub["name"] = contact["name"]
    if _non_empty(contact.get("organization")):
        stub["organization"] = contact["organization"]
    if "name" not in stub and "organization" not in stub:
        stub["name"] = identifier.removeprefix("contact:")
    if _non_empty(contact.get("roles")):
        stub["roles"] = contact["roles"]

    stub["links"] = [
        {
            "rel": "about",
            "href": href,
            "type": CATALOGUE_MEDIA_TYPE,
        }
    ]
    return _clean_none(stub)


def _merge_objects(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Conservative object merge: preserve existing values, add missing details."""
    merged = copy.deepcopy(existing)
    for key, value in incoming.items():
        if not _non_empty(value):
            continue
        if key not in merged or not _non_empty(merged.get(key)):
            merged[key] = copy.deepcopy(value)
            continue
        if isinstance(merged[key], list):
            current = merged[key]
            for item in _as_list(value):
                marker = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
                if not any(json.dumps(old, sort_keys=True, ensure_ascii=False, default=str) == marker for old in current):
                    current.append(copy.deepcopy(item))
        elif isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_objects(merged[key], value)
        # Scalar conflicts intentionally keep the first value for id stability.
    return _clean_none(merged)


def instrument_identifier(instrument: dict[str, Any]) -> str:
    raw = instrument.get("id") or instrument.get("identifier")
    if isinstance(raw, str) and raw.strip():
        raw = raw.strip()
        return raw if raw.startswith("instrument:") else f"instrument:{_slug(raw)}"

    seed = {
        "manufacturer": instrument.get("manufacturer"),
        "model": instrument.get("model"),
        "type": instrument.get("type"),
        "title": instrument.get("title"),
        "description": instrument.get("description"),
        "observedProperty": instrument.get("observedProperty"),
        "observedGeometry": instrument.get("observedGeometry"),
    }
    human = "--".join(
        _slug(str(part))
        for part in (instrument.get("manufacturer"), instrument.get("model"), instrument.get("title"))
        if isinstance(part, str) and part.strip()
    )
    return f"instrument:{human or 'unknown'}--{_stable_hash(seed, 8)}"


def _normalize_instrument_ref(value: Any) -> str | None:
    """Normalize one instrument reference or inline instrument object to an id."""
    if isinstance(value, dict):
        return instrument_identifier(value)
    if isinstance(value, str) and value.strip():
        ref = value.strip()
        return ref if ref.startswith("instrument:") else f"instrument:{_slug(ref)}"
    return None


def _deployment_instrument_ref(value: Any) -> str | None:
    """Return the scalar v0.2.x deployment instrument reference.

    WMDR2 v0.2.x reusable deployment definitions point to one instrument.  If a
    legacy input contains a list, the first non-empty reference is retained.
    """
    for item in _as_list(value):
        ref = _normalize_instrument_ref(item)
        if ref:
            return ref
    return None


def _iter_json_files(root: Path, *, pattern: str, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".json" else []
    if not root.is_dir():
        return []
    walker = root.rglob if recursive else root.glob
    return sorted(path for path in walker(pattern) if path.is_file() and path.suffix.lower() == ".json")


def convert_record_to_catalogue_version(
    record: dict[str, Any],
    *,
    contacts: dict[str, dict[str, Any]],
    instruments: dict[str, dict[str, Any]],
    records_path: Path,
    contacts_path: Path,
) -> dict[str, Any]:
    """Externalize one facility record and update shared catalogue maps."""
    out = copy.deepcopy(record)
    properties = out.setdefault("properties", {})
    if not isinstance(properties, dict):
        return out

    # Contacts: keep minimal OGC Records-compliant contact objects inline.
    contact_stubs: list[dict[str, Any]] = []
    for raw_contact in _as_list(properties.get("contacts")):
        if not isinstance(raw_contact, dict):
            continue
        full_contact = _clean_none(copy.deepcopy(raw_contact))
        identifier = contact_identifier(full_contact)
        full_contact["identifier"] = identifier
        contacts[identifier] = _merge_objects(contacts.get(identifier, {}), full_contact)
        href = _catalogue_href(contacts_path, records_path, identifier)
        contact_stubs.append(_minimal_contact_stub(full_contact, identifier, href=href))
    if contact_stubs:
        properties["contacts"] = _dedupe_dicts(contact_stubs)

    # Instruments: move full details to instruments.json.
    for raw_instrument in _as_list(properties.get("instruments")):
        if not isinstance(raw_instrument, dict):
            continue
        full_instrument = _clean_none(copy.deepcopy(raw_instrument))
        identifier = instrument_identifier(full_instrument)
        full_instrument["id"] = identifier
        instruments[identifier] = _merge_objects(instruments.get(identifier, {}), full_instrument)
    properties.pop("instruments", None)

    # Ensure deployment instrument refs are stable scalar ids.  The v0.2.x
    # reusable deployment model uses one instrument reference, not a list.
    for deployment in _as_list(properties.get("deployments")):
        if not isinstance(deployment, dict):
            continue
        ref = _deployment_instrument_ref(deployment.get("instrument"))
        if ref:
            deployment["instrument"] = ref

    return _clean_none(out)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _should_skip_generated_catalogue_input(path: Path, paths: CataloguePaths) -> bool:
    """Return True for derived catalogue outputs that should not be reprocessed."""
    if _is_relative_to(path, paths.records_path):
        return True
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path.absolute()
    for catalogue_file in (paths.contacts_path, paths.instruments_path):
        try:
            if resolved == catalogue_file.resolve():
                return True
        except Exception:
            if resolved == catalogue_file.absolute():
                return True
    return False


def _relative_output_path(source_file: Path, source_root: Path) -> Path:
    if source_root.is_file():
        return Path(source_file.name)
    try:
        return source_file.relative_to(source_root)
    except ValueError:
        return Path(source_file.name)


def _write_catalogues(
    *,
    contacts: dict[str, dict[str, Any]],
    instruments: dict[str, dict[str, Any]],
    contacts_path: Path,
    instruments_path: Path,
) -> None:
    contacts_path.parent.mkdir(parents=True, exist_ok=True)
    contacts_path.write_text(
        json.dumps({"contacts": sorted(contacts.values(), key=lambda item: item.get("identifier", ""))}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    instruments_path.write_text(
        json.dumps({"instruments": sorted(instruments.values(), key=lambda item: item.get("id", ""))}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def convert_catalogue_files(source_files: Iterable[Path], paths: CataloguePaths) -> list[Path]:
    """Externalize a known set of generated WMDR2 facility record files."""
    contacts: dict[str, dict[str, Any]] = {}
    instruments: dict[str, dict[str, Any]] = {}
    written_records: list[Path] = []

    for source_file in sorted(source_files):
        if _should_skip_generated_catalogue_input(source_file, paths):
            continue
        payload = json.loads(source_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        catalogue_record = convert_record_to_catalogue_version(
            payload,
            contacts=contacts,
            instruments=instruments,
            records_path=paths.records_path,
            contacts_path=paths.contacts_path,
        )
        relative = _relative_output_path(source_file, paths.source)
        target_file = paths.records_path / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(json.dumps(catalogue_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written_records.append(target_file)

    _write_catalogues(
        contacts=contacts,
        instruments=instruments,
        contacts_path=paths.contacts_path,
        instruments_path=paths.instruments_path,
    )
    return written_records


def convert_to_catalogue_version(paths: CataloguePaths) -> list[Path]:
    source_files = [
        source_file
        for source_file in _iter_json_files(paths.source, pattern=paths.pattern, recursive=paths.recursive)
        if not _should_skip_generated_catalogue_input(source_file, paths)
    ]
    return convert_catalogue_files(source_files, paths)


def _paths_from_config(config_path: Path) -> CataloguePaths:
    if yaml is None:
        raise SystemExit("PyYAML is required when using --config.")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = cfg.get("convert_wmdr10_json_to_wmdr2_json") or {}
    if not isinstance(section, dict):
        raise SystemExit("Missing convert_wmdr10_json_to_wmdr2_json config section.")
    catalogues = section.get("catalogues") or {}
    if not isinstance(catalogues, dict):
        raise SystemExit("catalogues must be a mapping.")
    if "source" in catalogues:
        raise SystemExit("catalogues.source is obsolete; catalogue input is always the converter target.")

    base = config_path.parent
    target = section.get("target")
    if not target:
        raise SystemExit("Missing convert_wmdr10_json_to_wmdr2_json.target; it is used as catalogue input.")

    source = Path(str(target)).expanduser()
    records_path = Path(str(catalogues.get("records_path") or (source / "catalogue_representation"))).expanduser()
    contacts_path = Path(str(catalogues.get("contacts_path") or (source / "catalogues" / "contacts.json"))).expanduser()
    instruments_path = Path(str(catalogues.get("instruments_path") or (source / "catalogues" / "instruments.json"))).expanduser()

    def abs_path(path: Path) -> Path:
        return path if path.is_absolute() else base / path

    return CataloguePaths(
        source=abs_path(source),
        records_path=abs_path(records_path),
        contacts_path=abs_path(contacts_path),
        instruments_path=abs_path(instruments_path),
        pattern=str(section.get("pattern") or DEFAULT_PATTERN),
        recursive=bool(section.get("recursive", True)),
    )


# Backwards-compatible aliases for older imports/tests.
externalize_record = convert_record_to_catalogue_version
externalize_catalogue_files = convert_catalogue_files
externalize_catalogues = convert_to_catalogue_version


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert embedded WMDR2 facility records to the catalogue-based representation.")
    parser.add_argument("--config", default=Path("config.yaml"), type=Path, help="Read catalogue paths from config.yaml.")
    parser.add_argument("--source", type=Path, help="Source WMDR2 facility records, usually converter target output.")
    parser.add_argument("--records-path", type=Path, help="Target folder for rewritten facility records.")
    parser.add_argument("--contacts-path", type=Path, help="Target contacts.json path.")
    parser.add_argument("--instruments-path", type=Path, help="Target instruments.json path.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args(argv)

    explicit_paths_provided = all(
        value is not None
        for value in (args.source, args.records_path, args.contacts_path, args.instruments_path)
    )
    if explicit_paths_provided:
        paths = CataloguePaths(
            source=args.source,
            records_path=args.records_path,
            contacts_path=args.contacts_path,
            instruments_path=args.instruments_path,
            pattern=args.pattern,
            recursive=not args.no_recursive,
        )
    elif args.config.exists():
        paths = _paths_from_config(args.config)
    else:
        missing = [
            name
            for name, value in (
                ("--source", args.source),
                ("--records-path", args.records_path),
                ("--contacts-path", args.contacts_path),
                ("--instruments-path", args.instruments_path),
            )
            if value is None
        ]
        raise SystemExit(
            f"Config file not found: {args.config}. "
            f"Alternatively pass all explicit path arguments: {', '.join(missing)}"
        )

    written = convert_to_catalogue_version(paths)
    print(f"Wrote {len(written)} catalogue-based WMDR2 record(s) to {paths.records_path}")
    print(f"Wrote contacts catalogue to {paths.contacts_path}")
    print(f"Wrote instruments catalogue to {paths.instruments_path}")


if __name__ == "__main__":
    main()
