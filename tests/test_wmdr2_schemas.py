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
        "time": {
            "interval": ["..", ".."],
        },
        "conformsTo": [
            "http://www.opengis.net/spec/ogcapi-records-1/1.0/conf/record-core",
            "https://schemas.wmo.int/wmdr/2.0/core/facility-record",
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
            "facilityType": "http://codes.wmo.int/wmdr/FacilityType/landFixed",
            "wmoRegion": "http://codes.wmo.int/wmdr/WMORegion/europe",
            "temporalTerritory": {
                "territory": ["http://codes.wmo.int/wmdr/TerritoryName/BEL"],
                "datetimes": ["2016-04-28T00:00:00Z"],
            },
            "temporalClimateZone": {
                "climateZone": ["http://codes.wmo.int/wmdr/ClimateZone/Cfb"],
                "datetimes": [".."],
            },
            "temporalSurfaceCover": {
                "surfaceCover": ["http://codes.wmo.int/wmdr/SurfaceCover/grassland"],
                "datetimes": [".."],
            },
            "temporalGeometry": {
                "type": "MovingPoint",
                "coordinates": [
                    [6.0733333333, 50.5108333333, 671],
                    [6.0734, 50.5109, 671],
                ],
                "datetimes": [
                    "2016-04-28T00:00:00Z",
                    "2024-01-17T00:00:00Z",
                ],
            },
            "temporalProgramAffiliation": [
                {
                    "programAffiliation": [
                        "http://codes.wmo.int/wmdr/ProgramAffiliation/GOSGeneral"
                    ],
                    "reportingStatus": [
                        "http://codes.wmo.int/wmdr/ReportingStatus/operational",
                        "http://codes.wmo.int/wmdr/ReportingStatus/closed",
                    ],
                    "datetimes": [
                        "2000-08-17T00:00:00Z",
                        "2025-05-28T00:00:00Z",
                    ],
                }
            ],
            "observations": [
                {
                    "id": "observation:http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/179",
                    "title": "variable 179: Cloud amount; domain: Atmosphere",
                    "description": "Observed variable 179; geometry type point",
                    "time": {
                        "interval": ["2016-04-29T00:00:00Z", ".."],
                    },
                    "observedVariable": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/179",
                    "observedGeometryType": "http://codes.wmo.int/wmdr/Geometry/point",
                    "observedDomain": "https://codes.wmo.int/wmdr/Domain/atmosphere",
                    "internationalReportingSchedule": [
                        {
                            "internationalExchange": True,
                            "temporalReportingInterval": "PT1H",
                        }
                    ],
                    "deployments": [
                        "deployment:id_af2ac7ee-a215-4e90-974c-f4499458cc06"
                    ],
                }
            ],
            "deployments": [
                {
                    "id": "deployment:id_af2ac7ee-a215-4e90-974c-f4499458cc06",
                    "time": {
                        "interval": ["2016-04-29T00:00:00Z", ".."],
                    },
                    "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/automatic",
                    "localReferenceSurface": "http://codes.wmo.int/wmdr/ReferenceSurfaceType/localGround",
                    "temporalObservingSchedule": [
                        {
                            "interval": "unknown",
                            "diurnalBaseTime": "06:00:00Z",
                        }
                    ],
                }
            ],
        },
    }


def test_facility_centric_record_feature_is_valid() -> None:
    instance = _valid_facility_record_feature()

    assert _validate(RECORD_SCHEMA, instance) == []


def test_record_time_allows_unknown_interval() -> None:
    instance = _valid_facility_record_feature()
    instance["time"] = {"interval": ["..", ".."]}

    assert _validate(RECORD_SCHEMA, instance) == []


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


def test_international_reporting_schedule_has_no_id_or_interval() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["observations"][0]["internationalReportingSchedule"][0]["interval"] = "unknown"

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_temporal_observing_schedule_has_no_id() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["temporalObservingSchedule"][0]["id"] = "schedule-1"

    assert not _is_valid(RECORD_SCHEMA, instance)


def test_deployment_title_is_invalid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["title"] = "Deployment title"

    assert not _is_valid(RECORD_SCHEMA, instance)
