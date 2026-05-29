from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = PROJECT_ROOT / "schemas"

COMMON_SCHEMA_FILE = "wmdr2-common.schema.json"
RECORD_SCHEMA_FILE = "wmdr2-record-feature.schema.json"
FEATURE_COLLECTION_SCHEMA_FILE = "wmdr2-feature-collection.schema.json"


def _load_schema(filename: str) -> dict[str, Any]:
    path = SCHEMA_DIR / filename
    if not path.exists():
        pytest.fail(
            f"Schema file not found: {path}\n"
            f"Expected schema files to be committed under: {SCHEMA_DIR}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


COMMON_SCHEMA = _load_schema(COMMON_SCHEMA_FILE)
RECORD_SCHEMA = _load_schema(RECORD_SCHEMA_FILE)
FEATURE_COLLECTION_SCHEMA = _load_schema(FEATURE_COLLECTION_SCHEMA_FILE)

REGISTRY = (
    Registry()
    .with_resource(
        COMMON_SCHEMA["$id"],
        Resource.from_contents(COMMON_SCHEMA, default_specification=DRAFT202012),
    )
    .with_resource(
        RECORD_SCHEMA["$id"],
        Resource.from_contents(RECORD_SCHEMA, default_specification=DRAFT202012),
    )
    .with_resource(
        FEATURE_COLLECTION_SCHEMA["$id"],
        Resource.from_contents(
            FEATURE_COLLECTION_SCHEMA,
            default_specification=DRAFT202012,
        ),
    )
)


def _validator(schema: dict[str, Any]) -> Draft202012Validator:
    return Draft202012Validator(schema, registry=REGISTRY)


def _validate(schema: dict[str, Any], instance: dict[str, Any]) -> list[str]:
    return sorted(err.message for err in _validator(schema).iter_errors(instance))


def _is_valid(schema: dict[str, Any], instance: dict[str, Any]) -> bool:
    return _validator(schema).is_valid(instance)


def _valid_facility_record_feature() -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": "facility:0-20000-0-06494",
        "geometry": {
            "type": "Point",
            "coordinates": [6.0733333333, 50.5108333333, 671],
        },
        "time": None,
        "temporalGeometry": {
            "coordinates": [
                [6.0733333333, 50.5108333333, 671],
                [6.0734, 50.5109, 671],
            ],
            "dates": [
                "2016-04-28",
                "2024-01-17",
            ],
        },
        "conformsTo": [
            "http://www.opengis.net/spec/ogcapi-records-1/1.0/conf/record-core",
            "https://schemas.wmo.int/wmdr/2.0/core/full-record",
        ],
        "properties": {
            "type": "facility",
            "title": "MONT-RIGI",
            "created": "2020-03-04T00:00:00Z",
            "updated": "2020-03-04T00:00:00Z",
            "description": "Example facility record.",
            "externalIds": [
                {
                    "scheme": "WMO:WIGOS",
                    "value": "0-20000-0-06494",
                }
            ],
            "contacts": [
                {
                    "id": "contact:example",
                    "organization": "Royal Meteorological Institute of Belgium",
                    "roles": ["owner"],
                }
            ],
            "keywords": [
                "0-20000-0-06494",
                "MONT-RIGI",
            ],
            "facilityType": "landFixed",
            "wmoRegion": "europe",
            "temporalTerritory": {
                "territory": ["BEL"],
                "dates": ["2016-04-28"],
            },
            "temporalClimateZone": {
                "climateZone": ["Cfb"],
                "dates": [".."],
            },
            "temporalSurfaceCover": {
                "surfaceCover": ["grassland"],
                "dates": [".."],
            },
            "temporalProgramAffiliation": {
                "programAffiliation": [
                    "GOSGeneral",
                    "GOSGeneral",
                    "GBON",
                ],
                "reportingStatus": [
                    "operational",
                    "closed",
                    "operational",
                ],
                "dates": [
                    "2000-08-17",
                    "2025-05-28",
                    "2022-09-08",
                ],
            },
            "schedules": [
                {
                    "@type": "Event",
                    "uid": "schedule_daily_12",
                    "start": "0001-01-01T12:00:00",
                    "timeZone": "UTC",
                    "duration": "PT0S",
                    "recurrenceRules": [
                        {
                            "@type": "RecurrenceRule",
                            "frequency": "daily",
                        }
                    ],
                }
            ],
            "observations": [
                {
                    "id": "observation:179",
                    "title": "domain: atmosphere; geometry: point; variable: 179 Cloud amount",
                    "time": {
                        "interval": ["2016-04-29", ".."],
                    },
                    "observedVariable": 179,
                    "observedGeometryType": "point",
                    "observedDomain": "atmosphere",
                    "reporting": {
                        "internationalExchange": [True, False],
                        "temporalReportingInterval": ["PT1H", "PT10M"],
                        "uom": [None, "mm"],
                    },
                    "deployments": [
                        "deployment:id_af2ac7ee-a215-4e90-974c-f4499458cc06"
                    ],
                }
            ],
            "deployments": [
                {
                    "id": "deployment:id_af2ac7ee-a215-4e90-974c-f4499458cc06",
                    "time": {
                        "interval": ["2016-04-29", ".."],
                    },
                    "observingMethod": "automatic",
                    "localReferenceSurface": "localGround",
                    "instrument": ["instrument:example"],
                    "serialNumbers": {
                        "serialNumber": ["ABC123"],
                        "dates": [".."],
                    },
                    "temporalObservingSchedule": {
                        "observingSchedule": ["schedule_daily_12"],
                        "dates": ["2025-01-01"],
                    },
                }
            ],
            "instruments": [
                {
                    "id": "instrument:example",
                    "manufacturer": "Vaisala",
                    "model": "ExampleModel",
                }
            ],
        },
    }


def test_facility_centric_record_feature_is_valid() -> None:
    instance = _valid_facility_record_feature()

    assert _validate(RECORD_SCHEMA, instance) == []


def test_record_time_allows_null_when_unknown() -> None:
    instance = _valid_facility_record_feature()
    instance["time"] = None

    assert _validate(RECORD_SCHEMA, instance) == []






def test_time_intervals_use_date_resolution_only() -> None:
    instance = _valid_facility_record_feature()
    instance["time"] = {"interval": ["2000-08-17", "2025-05-28"]}
    assert _validate(RECORD_SCHEMA, instance) == []

    instance["time"] = {"interval": ["2000-08-17T00:00:00Z", "2025-05-28T00:00:00Z"]}
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_nested_time_intervals_use_date_resolution_only() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["observations"][0]["time"] = {"interval": ["2016-04-29", ".."]}
    assert _validate(RECORD_SCHEMA, instance) == []

    instance["properties"]["observations"][0]["time"] = {"interval": ["2016-04-29T00:00:00Z", ".."]}
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_temporal_geometry_belongs_at_root_not_properties() -> None:
    instance = _valid_facility_record_feature()

    assert _validate(RECORD_SCHEMA, instance) == []

    instance["properties"]["temporalGeometry"] = deepcopy(instance["temporalGeometry"])

    assert not _is_valid(RECORD_SCHEMA, instance)




def test_temporal_geometry_does_not_use_moving_point_type() -> None:
    instance = _valid_facility_record_feature()

    assert _validate(RECORD_SCHEMA, instance) == []

    instance["temporalGeometry"]["type"] = "MovingPoint"

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_record_schema_rejects_feature_collection() -> None:
    instance = {
        "type": "FeatureCollection",
        "features": [],
    }

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_non_empty_feature_collection_is_valid_when_wrapping_records() -> None:
    instance = {
        "type": "FeatureCollection",
        "features": [
            deepcopy(_valid_facility_record_feature()),
        ],
    }

    assert _validate(FEATURE_COLLECTION_SCHEMA, instance) == []


def test_empty_feature_collection_is_invalid() -> None:
    instance = {
        "type": "FeatureCollection",
        "features": [],
    }

    errors = _validate(FEATURE_COLLECTION_SCHEMA, instance)

    assert errors
    assert any("should be non-empty" in msg or "is too short" in msg for msg in errors)


def test_properties_wmdr2_wrapper_is_invalid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["wmdr2"] = {}

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_themes_are_invalid_in_core_record() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["themes"] = []

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_lifecycle_dates_are_not_repeated_in_properties() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["dateEstablished"] = "2000-08-17Z"

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_observation_deployments_are_id_references_not_objects() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["observations"][0]["deployments"] = [
        {"id": "deployment:id_af2ac7ee-a215-4e90-974c-f4499458cc06"}
    ]

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_old_international_reporting_schedule_is_invalid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["observations"][0]["internationalReportingSchedule"] = [
        {
            "internationalExchange": True,
            "temporalReportingInterval": "PT1H",
        }
    ]

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_reporting_has_no_id_or_interval() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["observations"][0]["reporting"]["interval"] = ["unknown"]

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_deployment_temporal_observing_schedule_references_schedules() -> None:
    instance = _valid_facility_record_feature()
    schedule_ref = instance["properties"]["deployments"][0]["temporalObservingSchedule"]

    assert schedule_ref == {
        "observingSchedule": ["schedule_daily_12"],
        "dates": ["2025-01-01"],
    }
    assert _validate(RECORD_SCHEMA, instance) == []


def test_observation_temporal_observing_schedule_is_invalid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["observations"][0]["temporalObservingSchedule"] = {
        "observingSchedule": ["schedule_daily_12"],
        "dates": ["2025-01-01"],
    }

    assert not _is_valid(RECORD_SCHEMA, instance)




def test_temporal_observing_schedule_has_no_id() -> None:
    """Legacy test name kept so cached VS Code node IDs still resolve.

    The current model keeps temporalObservingSchedule on deployments, but the
    object is only a dated reference to first-class schedule UIDs. It must not
    carry a source XML id.
    """
    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["temporalObservingSchedule"]["id"] = "schedule-source-id"

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_schedule_uid_uses_jscalendar_safe_id() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["schedules"][0]["uid"] = "schedule:daily-12"

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_schedule_recurrence_rule_requires_jscalendar_type() -> None:
    instance = _valid_facility_record_feature()
    del instance["properties"]["schedules"][0]["recurrenceRules"][0]["@type"]

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_schedule_recurrence_override_null_is_invalid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["schedules"][0]["recurrenceOverrides"] = {
        "2025-07-14T12:00:00": None
    }

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_deployment_title_is_invalid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["title"] = "Deployment title"

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_deployment_manufacturer_model_are_invalid_but_serial_numbers_are_valid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["manufacturer"] = "Vaisala"

    assert not _is_valid(RECORD_SCHEMA, instance)

    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["model"] = "ExampleModel"

    assert not _is_valid(RECORD_SCHEMA, instance)

    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["serialNumber"] = "ABC123"

    assert not _is_valid(RECORD_SCHEMA, instance)

    instance = _valid_facility_record_feature()

    assert _validate(RECORD_SCHEMA, instance) == []


def test_instrument_serial_numbers_are_invalid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["instruments"][0]["serialNumbers"] = {
        "serialNumber": ["ABC123"],
        "dates": [".."],
    }

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_observation_observed_variable_can_be_numeric_code() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["observations"][0]["observedVariable"] = 12006
    instance["properties"]["observations"][0]["observedDomain"] = "atmosphere"

    assert _validate(RECORD_SCHEMA, instance) == []

def test_contact_instructions_are_invalid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["contacts"][0]["contactInstructions"] = "RA VI"

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_wmdr2_full_record_conformance_is_required() -> None:
    instance = _valid_facility_record_feature()

    assert _validate(RECORD_SCHEMA, instance) == []

    instance["conformsTo"].remove("https://schemas.wmo.int/wmdr/2.0/core/full-record")

    assert not _is_valid(RECORD_SCHEMA, instance)



def test_observation_description_is_not_part_of_current_core_model() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["observations"][0]["description"] = None

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_inline_observing_schedule_object_is_invalid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["temporalObservingSchedule"] = [
        {"interval": "unknown"}
    ]

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_schedule_event_requires_jscalendar_event_type() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["schedules"][0]["@type"] = "Task"

    assert not _is_valid(RECORD_SCHEMA, instance)
