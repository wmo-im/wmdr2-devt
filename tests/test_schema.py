from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

import convert_wmdr10_json_to_wmdr2_json as converter

ROOT = Path(__file__).resolve().parents[1]


def _validator() -> Draft202012Validator:
    schema = json.loads((ROOT / "schemas" / "wmdr2-record-feature.schema.json").read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _valid_record() -> dict[str, Any]:
    return converter.convert_payload(
        {
            "facility": {
                "identifier": "wsi:0-20000-0-TEST",
                "name": "Test",
                "geospatialLocation": "46 7 100",
                "beginPosition": "2020-01-01",
            },
            "observationSeries": [
                {
                    "observedProperty": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006",
                    "deployments": [
                        {
                            "beginPosition": "2020-01-01",
                            "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/266",
                            "sourceOfObservation": "automaticReading",
                        }
                    ],
                }
            ],
        }
    )


def test_generated_record_validates_against_wmdr2_record_schema() -> None:
    errors = sorted(_validator().iter_errors(_valid_record()), key=lambda error: list(error.path))
    assert errors == []


def test_schema_rejects_namespaced_facility_id() -> None:
    record = _valid_record()
    record["id"] = "wsi:0-20000-0-TEST"
    errors = list(_validator().iter_errors(record))
    assert errors



def test_schema_rejects_non_iso_time_resolution_word() -> None:
    record = _valid_record()
    record["time"]["resolution"] = "day"
    errors = list(_validator().iter_errors(record))
    assert errors


def test_schema_rejects_observing_configuration_keywords_and_temporal_geometry() -> None:
    record = _valid_record()
    cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
    cfg["keywords"] = ["not-a-discovery-object"]
    assert list(_validator().iter_errors(record))

    record = _valid_record()
    cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
    cfg["temporalGeometry"] = {
        "type": "MovingPoint",
        "coordinates": [[7.0, 46.0, 100]],
        "dates": ["2020-01-01"],
    }
    assert list(_validator().iter_errors(record))

def test_schema_rejects_time_on_reporting_procedure() -> None:
    record = _valid_record()
    record["properties"]["observationSeries"][0]["reportingProcedures"] = [
        {"time": {"interval": ["2020-01-01", ".."]}, "internationalExchange": True}
    ]
    errors = list(_validator().iter_errors(record))
    assert errors



def test_schema_rejects_temporal_reporting_interval_on_reporting_procedure() -> None:
    record = _valid_record()
    record["properties"]["observationSeries"][0]["reportingProcedures"] = [
        {"temporalReportingInterval": "PT1H", "internationalExchange": True}
    ]
    errors = list(_validator().iter_errors(record))
    assert errors


def test_schema_accepts_aggregation_interval_on_reusable_schedule() -> None:
    record = _valid_record()
    record["properties"]["schedules"] = [
        {"uid": "schedule_abc", "@type": "Event", "wmo.int:aggregationInterval": "PT1H"}
    ]
    errors = list(_validator().iter_errors(record))
    assert errors == []


def test_schema_accepts_serial_number_on_observing_configuration() -> None:
    record = _valid_record()
    cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
    cfg["serialNumber"] = "SN-001"
    errors = list(_validator().iter_errors(record))
    assert errors == []


def test_schema_rejects_serial_number_on_instrument_catalogue_entry() -> None:
    record = _valid_record()
    record["properties"]["instruments"] = [
        {"id": "instrument:maker-model", "manufacturer": "Maker", "model": "Model", "serialNumber": "SN-001"}
    ]
    errors = list(_validator().iter_errors(record))
    assert errors


def test_schema_rejects_multiple_operating_status_values_on_observing_configuration() -> None:
    record = _valid_record()
    cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
    cfg["operatingStatus"] = ["operational", "standby"]
    errors = list(_validator().iter_errors(record))
    assert errors
