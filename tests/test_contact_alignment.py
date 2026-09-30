from __future__ import annotations

import convert_wmdr10_json_to_wmdr2_json as converter
from schema_registry import validator_for_schema


def test_wmdr1_responsible_party_becomes_direct_ogc_contact_with_role() -> None:
    registry: dict[str, dict] = {}
    contacts = converter._contact_occurrences(
        {
            "organisationName": "Example Service",
            "contactInfo": {"address": {"electronicMailAddress": "ops@example.org"}},
            "role": "principalInvestigator",
        },
        registry,
    )
    assert contacts == [{
        "identifier": "contact:ops@example.org",
        "organization": "Example Service",
        "emails": [{"value": "ops@example.org"}],
        "roles": ["principalInvestigator"],
    }]


def test_same_contact_identity_can_have_context_specific_roles() -> None:
    registry: dict[str, dict] = {}
    raw = {
        "organisationName": "Example Service",
        "contactInfo": {"address": {"electronicMailAddress": "ops@example.org"}},
    }
    owner = converter._contact_occurrences(
        raw,
        registry,
        "owner",
        context="facility",
    )[0]
    leader = converter._contact_occurrences(
        raw,
        registry,
        "principalInvestigator",
        context="observation",
    )[0]
    assert owner["identifier"] == leader["identifier"]
    assert owner["roles"] == ["supervisor"]
    assert leader["roles"] == ["principalInvestigator"]


def test_facility_schema_accepts_official_contact_roles() -> None:
    validator = validator_for_schema("wmdr2-facility-properties.schema.json")
    value = {
        "type": "facility",
        "title": "Test",
        "facilityType": None,
        "contacts": [{
            "identifier": "contact:ops@example.org",
            "organization": "Example Service",
            "emails": [{"value": "ops@example.org"}],
            "roles": ["supervisor"],
        }],
    }
    assert list(validator.iter_errors(value)) == []
