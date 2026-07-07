from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

import convert_wmdr10_json_to_wmdr2_json as module


def _validator() -> Draft202012Validator:
    base = Path(__file__).resolve().parents[1] / "schemas"
    common = json.loads((base / "wmdr2-common.schema.json").read_text(encoding="utf-8"))
    schema = json.loads((base / "wmdr2-record-feature.schema.json").read_text(encoding="utf-8"))
    registry = (
        Registry()
        .with_resource(
            common["$id"],
            Resource.from_contents(common, default_specification=DRAFT202012),
        )
        .with_resource(
            schema["$id"],
            Resource.from_contents(schema, default_specification=DRAFT202012),
        )
    )
    return Draft202012Validator(schema, registry=registry)


def test_v030_record_validates_against_schema() -> None:
    record = module.convert_payload(
        {
            "facility": {
                "identifier": "0-TEST",
                "name": "Test",
                "geospatialLocation": {"geoLocation": "46 7 100", "beginPosition": "2020-01-01"},
            },
            "observationSeries": [
                {
                    "observedProperty": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/263",
                    "deployments": [
                        {
                            "beginPosition": "2020-01-01",
                            "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/106",
                            "instrumentOperatingStatus": "operational",
                            "sourceOfObservation": "manualReading",
                            "heightAboveLocalReferenceSurface": {"@uom": "m", "#text": "2"},
                        }
                    ],
                    "reporting": {"beginPosition": "2020-01-01", "internationalExchange": "true", "temporalReportingInterval": "PT1H", "spatialReportingInterval": "point"},
                }
            ],
        },
        source_name="0-TEST",
    )
    errors = sorted(_validator().iter_errors(record), key=lambda e: list(e.path))
    assert errors == []


def test_schema_rejects_legacy_deployment_property() -> None:
    record = module.convert_payload(
        {
            "facility": {"identifier": "0-TEST", "name": "Test", "geospatialLocation": {"geoLocation": "46 7 100"}},
            "observationSeries": [{"observedProperty": 263}],
        }
    )
    record["properties"]["observationSeries"][0]["observingConfigurations"][0]["deployment"] = "deployment:bad"
    errors = list(_validator().iter_errors(record))
    assert any("False schema" in error.message or "False" in error.message for error in errors)


def test_temporal_geometry_methods_require_non_empty_strings() -> None:
    common = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "wmdr2-common.schema.json").read_text(encoding="utf-8"))
    schema = common["$defs"]["temporalGeometry"]
    validator = Draft202012Validator(schema)
    payload = {
        "type": "MovingPoint",
        "coordinates": [[7.0, 46.0], [7.1, 46.1]],
        "dates": ["2020-01-01", "2021-01-01"],
        "methods": [[""]],
    }
    assert list(validator.iter_errors(payload))


def test_schema_rejects_unknown_valid_from_for_observing_configuration() -> None:
    record = module.convert_payload(
        {
            "facility": {"identifier": "0-TEST", "name": "Test", "geospatialLocation": {"geoLocation": "46 7 100"}},
            "observationSeries": [{"observedProperty": 263}],
        },
        source_name="0-TEST",
    )
    cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
    assert cfg["validFrom"] == ".."
    errors = list(_validator().iter_errors(record))
    assert any(list(error.path)[-2:] == ["validFrom"] or list(error.path)[-1:] == ["validFrom"] for error in errors)


def test_schema_rejects_unknown_valid_from_for_observing_procedure() -> None:
    record = module.convert_payload(
        {
            "facility": {"identifier": "0-TEST", "name": "Test", "geospatialLocation": {"geoLocation": "46 7 100"}},
            "observationSeries": [
                {
                    "observedProperty": 263,
                    "deployments": [
                        {
                            "beginPosition": "2020-01-01",
                            "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/106",
                            "instrumentOperatingStatus": "operational",
                            "sourceOfObservation": "manualReading",
                        }
                    ],
                    "coverage": {"temporalSamplingInterval": "PT1H"},
                }
            ],
        },
        source_name="0-TEST",
    )
    procedure = record["properties"]["observationSeries"][0]["observingProcedures"][0]
    assert procedure["validFrom"] == ".."
    errors = list(_validator().iter_errors(record))
    assert any(list(error.path)[-2:] == ["validFrom"] or list(error.path)[-1:] == ["validFrom"] for error in errors)
