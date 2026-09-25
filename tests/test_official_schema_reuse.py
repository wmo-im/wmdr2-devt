from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"
OFFICIAL_COMMIT = "d30f13c3be6395466a778360c4a4ef59625be6af"
OFFICIAL_URL = (
    "https://raw.githubusercontent.com/wmo-im/wmdr2/"
    f"{OFFICIAL_COMMIT}/schemas/wmdr2-bundled.json"
)

WCMP_COMMIT = "f05037aa8d8bf5911a44a511b7b99a0be009c9ab"
WCMP_URL = (
    "https://raw.githubusercontent.com/wmo-im/wcmp2/"
    f"{WCMP_COMMIT}/schemas/wcmpRecordGeoJSON.yaml"
)


def _load(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _refs(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        ref = value.get("$ref")
        if isinstance(ref, str):
            found.append(ref)
        for child in value.values():
            found.extend(_refs(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_refs(child))
    return found


def test_record_reuses_official_root_properties() -> None:
    schema = _load("wmdr2-record-feature.schema.json")
    for key in ("id", "conformsTo", "type"):
        ref = schema["properties"][key]["$ref"]
        assert ref == OFFICIAL_URL + f"#/properties/{key}"

    geometry_refs = _refs(schema["properties"]["geometry"])
    assert OFFICIAL_URL + "#/properties/geometry" in geometry_refs

    time_refs = _refs(schema["properties"]["time"])
    assert (
        "wmdr2-common.schema.json#/$defs/timeObject"
        in time_refs
    )

    link_refs = _refs(schema["properties"]["links"])
    assert OFFICIAL_URL + "#/properties/links/items" in link_refs


def test_facility_reuses_official_compatible_properties() -> None:
    schema = _load("wmdr2-facility-properties.schema.json")
    refs = _refs(schema)
    expected = (
        "type",
        "title",
        "description",
        "externalIds",
        "created",
        "updated",
        "additionalIds",
        "additionalTitles",
        "facilityType",
        "wmoRegion",
        "territories",
    )
    for key in expected:
        fragment = f"#/properties/properties/properties/{key}"
        assert any(ref == OFFICIAL_URL + fragment for ref in refs), key

    assert (
        OFFICIAL_URL
        + "#/properties/properties/properties/contacts/items"
    ) in refs


def test_observation_reuses_official_observation_semantics() -> None:
    schema = _load("wmdr2-observation.schema.json")
    refs = _refs(schema)
    for key in ("observedGeometry", "observedProperty", "programAffiliations"):
        fragment = f"#/definitions/Observations/properties/{key}"
        assert OFFICIAL_URL + fragment in refs


def test_configuration_reuses_official_properties_except_instrument() -> None:
    schema = _load("wmdr2-configuration.schema.json")
    refs = _refs(schema)
    for key in (
        "id",
        "geometry",
        "observingMethod",
        "operatingStatus",
        "sourceOfObservation",
        "instrumentSerialNumber",
        "verticalDistance",
    ):
        fragment = f"#/definitions/Configurations/properties/{key}"
        assert OFFICIAL_URL + fragment in refs

    time_refs = _refs(schema["properties"]["time"])
    assert "wmdr2-common.schema.json#/$defs/timePeriod" in time_refs

    instrument = schema["properties"]["instrument"]
    assert "$ref" not in instrument
    assert instrument["type"] == ["string", "integer", "null"]
    assert "Intentional devt divergence" in instrument["$comment"]


def test_instrument_extends_official_instrument_definition() -> None:
    schema = _load("wmdr2-instrument.schema.json")
    assert schema["allOf"][0]["$ref"] == OFFICIAL_URL + "#/definitions/Instrument"


def test_all_official_references_are_pinned_not_main() -> None:
    names = (
        "wmdr2-common.schema.json",
        "wmdr2-instrument.schema.json",
        "wmdr2-configuration.schema.json",
        "wmdr2-observation.schema.json",
        "wmdr2-facility-properties.schema.json",
        "wmdr2-record-feature.schema.json",
    )
    for name in names:
        refs = _refs(_load(name))
        official_refs = [
            ref for ref in refs
            if "raw.githubusercontent.com/wmo-im/wmdr2/" in ref
        ]
        assert official_refs, f"{name} contains no official WMDR2 reference"
        assert all(f"/{OFFICIAL_COMMIT}/" in ref for ref in official_refs)


def test_vendored_official_snapshots_are_present() -> None:
    for path in (
        SCHEMA_DIR / "official" / "wmdr2-bundled.json",
        SCHEMA_DIR / "official" / "wcmpRecordGeoJSON.yaml",
    ):
        assert path.exists(), (
            f"{path} is missing; run "
            "`python schemas/sync_official_wmdr2_schema.py`"
        )


def test_time_reuses_pinned_wcmp_temporal_fragments() -> None:
    common = _load("wmdr2-common.schema.json")
    time_object = common["$defs"]["timeObject"]
    refs = _refs(time_object)

    for key in ("date", "timestamp", "interval"):
        assert (
            WCMP_URL
            + f"#/properties/time/oneOf/1/properties/{key}"
        ) in refs

    assert time_object["properties"]["resolution"] == {
        "$ref": "#/$defs/iso8601Duration"
    }


def test_time_period_reuses_wcmp_interval_but_corrects_resolution() -> None:
    common = _load("wmdr2-common.schema.json")
    time_period = common["$defs"]["timePeriod"]
    refs = _refs(time_period)

    assert (
        WCMP_URL
        + "#/properties/time/oneOf/1/properties/interval"
    ) in refs
    assert time_period["properties"]["resolution"] == {
        "$ref": "#/$defs/iso8601Duration"
    }
