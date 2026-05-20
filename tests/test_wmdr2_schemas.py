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
            f"Expected schema files to be committed under: {SCHEMA_DIR}\n"
            "If the schemas live elsewhere, update SCHEMA_DIR in this test."
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
            "coordinates": [6.0, 50.0, 671],
        },
        "time": {
            "interval": ["2016-04-28T00:00:00Z", ".."],
        },
        "conformsTo": [
            "http://www.opengis.net/spec/ogcapi-records-1/1.0/conf/record-core"
        ],
        "properties": {
            "type": "facility",
            "title": "MONT-RIGI",
            "description": "desc",
            "externalIds": [
                {
                    "scheme": "WMO:WIGOS",
                    "value": "0-20000-0-06494",
                }
            ],
            "wmdr2": {
                "temporalProgramAffiliation": [
                    {
                        "programAffiliation": [
                            "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"
                        ],
                        "reportingStatus": [
                            {
                                "value": (
                                    "http://codes.wmo.int/wmdr/"
                                    "ReportingStatus/operational"
                                ),
                                "id": "rs1",
                                "time": {
                                    "interval": ["2022-09-08T00:00:00Z", ".."],
                                },
                            }
                        ],
                    }
                ],
            },
        },
    }


def test_facility_centric_record_feature_is_valid() -> None:
    instance = _valid_facility_record_feature()

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