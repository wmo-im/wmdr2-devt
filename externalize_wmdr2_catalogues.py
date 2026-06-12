#!/usr/bin/env python3
"""
externalize_wmdr2_catalogues.py

First-pass catalogue externalizer for WMDR2 facility records.

It reads facility-centric WMDR2 JSON Features, collects full contact and
instrument objects into shared catalogues, and writes facility records with
minimal inline contact stubs and deployment-level instrument references.

The output intentionally avoids a custom `wmdr2` wrapper property.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency for CLI convenience
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
    """Remove empty object members, but preserve aligned temporalGeometry methods.

    ``temporalGeometry.methods`` is an aligned array whose items are lists of
    geopositioning-method terms. Empty inner lists are meaningful there: they
    mean that no method is declared for the corresponding coordinate/date.
    """

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
    """Return deterministic catalogue id for a contact.

    Priority:
    1. first e-mail address: contact:<lowercase-email>
    2. name + organization: contact:<slug-name>--<slug-organization>
    3. available name/organization/existing identifier + stable hash
    """
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
    try:
        rel = catalogue_path.relative_to(records_path)
    except ValueError:
        rel = Path("..") / catalogue_path.name
    return f"{rel.as_posix()}#{identifier}"


def _minimal_contact_stub(contact: dict[str, Any], identifier: str, *, href: str) -> dict[str, Any]:
    """Return a minimal OGC Records contact object that points to the catalogue."""
    stub: dict[str, Any] = {"identifier": identifier}

    # OGC Records contact objects should contain at least name or organization.
    if _non_empty(contact.get("name")):
        stub["name"] = contact["name"]
    if _non_empty(contact.get("organization")):
        stub["organization"] = contact["organization"]
    if "name" not in stub and "organization" not in stub:
        # Last-resort schema-safety fallback. Full details remain in contacts.json.
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
        # For scalar conflicts, keep the first value for id stability and avoid silent churn.
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
        "observableVariables": instrument.get("observableVariables"),
        "observableGeometry": instrument.get("observableGeometry"),
    }
    human = "--".join(
        _slug(str(part))
        for part in (instrument.get("manufacturer"), instrument.get("model"), instrument.get("title"))
        if isinstance(part, str) and part.strip()
    )
    return f"instrument:{human or 'unknown'}--{_stable_hash(seed, 8)}"


def _iter_json_files(root: Path, *, pattern: str, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() == ".json" else []
    if not root.is_dir():
        return []
    walker = root.rglob if recursive else root.glob
    return sorted(path for path in walker(pattern) if path.is_file() and path.suffix.lower() == ".json")


def externalize_record(
    record: dict[str, Any],
    *,
    contacts: dict[str, dict[str, Any]],
    instruments: dict[str, dict[str, Any]],
    records_path: Path,
    contacts_path: Path,
) -> dict[str, Any]:
    """Externalize one facility record and update the shared catalogue maps."""
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

    # Instruments: move full details to instruments.json. Deployments already carry refs.
    for raw_instrument in _as_list(properties.get("instruments")):
        if not isinstance(raw_instrument, dict):
            continue
        full_instrument = _clean_none(copy.deepcopy(raw_instrument))
        identifier = instrument_identifier(full_instrument)
        full_instrument["id"] = identifier
        instruments[identifier] = _merge_objects(instruments.get(identifier, {}), full_instrument)
    if "instruments" in properties:
        properties.pop("instruments", None)

    # Ensure deployment instrument refs use stable instrument ids where possible.
    known_ids = set(instruments)
    for deployment in _as_list(properties.get("deployments")):
        if not isinstance(deployment, dict):
            continue
        refs: list[str] = []
        for item in _as_list(deployment.get("instrument")):
            if isinstance(item, dict):
                ident = instrument_identifier(item)
                item = ident
            elif isinstance(item, str) and item.strip():
                item = item.strip()
                if not item.startswith("instrument:"):
                    item = f"instrument:{_slug(item)}"
            else:
                continue
            refs.append(item)
        if refs:
            deployment["instrument"] = _dedupe_scalars(refs)

    return _clean_none(out)


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


def externalize_catalogues(paths: CataloguePaths) -> list[Path]:
    contacts: dict[str, dict[str, Any]] = {}
    instruments: dict[str, dict[str, Any]] = {}
    written_records: list[Path] = []

    for source_file in _iter_json_files(paths.source, pattern=paths.pattern, recursive=paths.recursive):
        payload = json.loads(source_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        externalized = externalize_record(
            payload,
            contacts=contacts,
            instruments=instruments,
            records_path=paths.records_path,
            contacts_path=paths.contacts_path,
        )
        relative = source_file.name if paths.source.is_file() else source_file.relative_to(paths.source)
        target_file = paths.records_path / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(json.dumps(externalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written_records.append(target_file)

    paths.contacts_path.parent.mkdir(parents=True, exist_ok=True)
    paths.contacts_path.write_text(
        json.dumps({"contacts": sorted(contacts.values(), key=lambda item: item.get("identifier", ""))}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    paths.instruments_path.parent.mkdir(parents=True, exist_ok=True)
    paths.instruments_path.write_text(
        json.dumps({"instruments": sorted(instruments.values(), key=lambda item: item.get("id", ""))}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return written_records


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

    base = config_path.parent
    source = Path(str(catalogues.get("source") or section.get("target") or section.get("source"))).expanduser()
    records_path = Path(str(catalogues.get("records_path") or section.get("target"))).expanduser()
    contacts_path = Path(str(catalogues.get("contacts_path") or "contacts.json")).expanduser()
    instruments_path = Path(str(catalogues.get("instruments_path") or "instruments.json")).expanduser()

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Externalize WMDR2 contacts and instruments into shared catalogues.")
    parser.add_argument("--config", default="config.yaml", type=Path, help="Read catalogue paths from config.yaml.")
    parser.add_argument("--source", type=Path, help="Source WMDR2 facility records, usually converter target output.")
    parser.add_argument("--records-path", type=Path, help="Target folder for rewritten facility records.")
    parser.add_argument("--contacts-path", type=Path, help="Target contacts.json path.")
    parser.add_argument("--instruments-path", type=Path, help="Target instruments.json path.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    if args.config:
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
        if missing:
            raise SystemExit(f"Missing required argument(s): {', '.join(missing)}")
        paths = CataloguePaths(
            source=args.source,
            records_path=args.records_path,
            contacts_path=args.contacts_path,
            instruments_path=args.instruments_path,
            pattern=args.pattern,
            recursive=not args.no_recursive,
        )

    written = externalize_catalogues(paths)
    print(f"Wrote {len(written)} externalized WMDR2 record(s) to {paths.records_path}")
    print(f"Wrote contacts catalogue to {paths.contacts_path}")
    print(f"Wrote instruments catalogue to {paths.instruments_path}")


if __name__ == "__main__":
    main()
