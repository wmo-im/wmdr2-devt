#!/usr/bin/env python3
"""Externalize WMDR2 contacts and instrument type metadata into catalogues.

The converter accepts both WMDR2 v0.3.1 full records with instrument references on
``observationSeries[*].observingConfigurations[*].instrument``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import re
import copy


@dataclass(frozen=True)
class CataloguePaths:
    source: Path
    records_path: Path
    contacts_path: Path
    instruments_path: Path


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


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "value"




def _last_segment(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().strip("<>").rstrip("/#")
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    return text or None


def _is_role_codelist_reference(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().strip("<>")
    segment = _last_segment(text) or text
    return segment in {"CI_RoleCode", "RoleCode"} or text.endswith("#CI_RoleCode") or text.endswith("#RoleCode")


def _normalize_role(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("codeListValue", "@codeListValue", "role", "value", "#text", "text", "name"):
            candidate = value.get(key)
            if _is_role_codelist_reference(candidate):
                continue
            role = _normalize_role(candidate)
            if role:
                return role
        for key in ("href", "url", "codeList", "@codeList"):
            candidate = value.get(key)
            if _is_role_codelist_reference(candidate):
                continue
            role = _normalize_role(candidate)
            if role:
                return role
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().strip("<>")
    if _is_role_codelist_reference(text):
        return None
    segment = _last_segment(text) or text
    if _is_role_codelist_reference(segment) or segment.lower() in {"unknown", "none", "null", "nil"}:
        return None
    return segment


def _normalize_roles(value: Any) -> list[str]:
    roles: list[str] = []
    for item in _as_list(value):
        role = _normalize_role(item)
        if role and role not in roles:
            roles.append(role)
    return roles


def _sanitize_contact_roles(contact: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = dict(contact)
    roles = _normalize_roles(sanitized.get("roles") or sanitized.get("role"))
    sanitized.pop("role", None)
    if roles:
        sanitized["roles"] = roles
    else:
        sanitized.pop("roles", None)
    return sanitized


def contact_uid(contact: Mapping[str, Any]) -> str:
    for email in _as_list(contact.get("emails") or contact.get("email")):
        if isinstance(email, str) and "@" in email:
            return f"contact:{email.strip().lower()}"
    name = contact.get("name") or contact.get("title") or ""
    org = contact.get("organization") or contact.get("organisation") or contact.get("organisationName") or ""
    return f"contact:{_slug(name)}--{_slug(org)}"


def _inline_contact(contact: Mapping[str, Any], uid: str) -> dict[str, Any]:
    contact = _sanitize_contact_roles(contact)
    out: dict[str, Any] = {"uid": uid}
    for key in ("name", "organization", "roles"):
        if contact.get(key) not in (None, "", [], {}):
            out[key] = contact[key]
    out["links"] = [{"rel": "about", "href": f"../catalogues/contacts.json#{uid}", "type": "application/json"}]
    return _clean(out)


def _instrument_catalogue_uid(instrument: Mapping[str, Any]) -> str | None:
    manufacturer = instrument.get("manufacturer")
    model = instrument.get("model")
    if manufacturer not in (None, "", [], {}) and model not in (None, "", [], {}):
        return f"instrument:{_slug(manufacturer)}--{_slug(model)}"
    raw_id = instrument.get("uid") or instrument.get("id")
    if isinstance(raw_id, str) and raw_id.startswith("instrument:"):
        # Keep existing type-like identifiers, but not serial-number-only instances.
        if "serialNumber" in instrument and len(instrument) <= 2:
            return None
        return raw_id
    return None


def _instrument_catalogue_entry(instrument: Mapping[str, Any]) -> dict[str, Any] | None:
    iid = _instrument_catalogue_uid(instrument)
    if not iid:
        return None
    out: dict[str, Any] = {"uid": iid}
    for key in ("manufacturer", "model", "verticalRange", "observingMethods"):
        if instrument.get(key) not in (None, "", [], {}):
            out[key] = instrument[key]
    if set(out) == {"uid"}:
        return None
    return _clean(out)


def _iter_record_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*.json") if p.is_file())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def convert_to_catalogue_version(paths: CataloguePaths) -> list[Path]:
    contacts: dict[str, dict[str, Any]] = {}
    instruments: dict[str, dict[str, Any]] = {}
    written: list[Path] = []

    for source_path in _iter_record_paths(paths.source):
        record = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            continue
        rewritten = copy.deepcopy(record)
        props = rewritten.get("properties")
        source_props = record.get("properties")
        if not isinstance(props, dict) or not isinstance(source_props, dict):
            continue

        inline_contacts: list[dict[str, Any]] = []
        for contact in _as_list(source_props.get("contacts")):
            if not isinstance(contact, Mapping):
                continue
            entry = _sanitize_contact_roles(contact)
            cid = contact_uid(entry)
            entry["uid"] = cid
            entry.pop("id", None)
            entry.pop("identifier", None)
            # Merge by uid, preserving details from later duplicates.
            merged = {**contacts.get(cid, {}), **entry}
            contacts[cid] = _clean(merged)
            inline_contacts.append(_inline_contact(entry, cid))
        if inline_contacts:
            props["contacts"] = inline_contacts

        id_map: dict[str, str] = {}
        for inst in _as_list(source_props.get("instruments")):
            if not isinstance(inst, Mapping):
                continue
            entry = _instrument_catalogue_entry(inst)
            if not entry:
                continue
            old_id = inst.get("uid") or inst.get("id")
            new_id = entry["uid"]
            if isinstance(old_id, str):
                id_map[old_id] = new_id
            instruments[new_id] = {**instruments.get(new_id, {}), **entry}

        # v0.3.1: instrument refs live on observing configurations.
        for obs in _as_list(props.get("observationSeries")):
            if not isinstance(obs, dict):
                continue
            for cfg in _as_list(obs.get("observingConfigurations")):
                if isinstance(cfg, dict) and isinstance(cfg.get("instrument"), str):
                    if cfg["instrument"] in id_map:
                        cfg["instrument"] = id_map[cfg["instrument"]]
                    else:
                        cfg.pop("instrument", None)

        props.pop("instruments", None)
        rel = source_path.relative_to(paths.source) if paths.source.is_dir() else Path(source_path.name)
        out_path = paths.records_path / rel
        _write_json(out_path, rewritten)
        written.append(out_path)

    if contacts:
        _write_json(paths.contacts_path, {"contacts": sorted(contacts.values(), key=lambda c: c.get("uid", ""))})
    else:
        _write_json(paths.contacts_path, {"contacts": []})
    if instruments:
        _write_json(paths.instruments_path, {"instruments": sorted(instruments.values(), key=lambda i: i.get("uid", ""))})
    else:
        _write_json(paths.instruments_path, {"instruments": []})
    return written


def contact_identifier(contact: Mapping[str, Any]) -> str:
    """Backward-compatible alias for callers; returns the contact uid."""
    return contact_uid(contact)


__all__ = ["CataloguePaths", "contact_uid", "contact_identifier", "convert_to_catalogue_version"]
