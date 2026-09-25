from __future__ import annotations

from copy import deepcopy
from typing import Any

from schema_registry import validator_for_schema


FACILITY_TYPE = "http://codes.wmo.int/wmdr/FacilityType/landFixed"
OBSERVED_PROPERTY = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
OBSERVED_GEOMETRY = "http://codes.wmo.int/wmdr/Geometry/point"
DOMAIN = "http://codes.wmo.int/wmdr/Domain/atmosphere"
PROGRAM = "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"
OBSERVING_METHOD = "http://codes.wmo.int/wmdr/ObservingMethod/266"
OPERATING_STATUS = "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/operational"
SOURCE_OF_OBSERVATION = "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading"
DATA_POLICY = "http://codes.wmo.int/wmdr/DataPolicy/noLimitation"


def concept(uri: str) -> dict[str, str]:
    return {"id": uri}


def _valid_record() -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": "0-20008-0-THE",
        "conformsTo": ["http://wigos.wmo.int/spec/wmdr/2/conf/core"],
        "geometry": {"type": "Point", "coordinates": [22.95, 40.63, 5.0]},
        "time": {"interval": ["2020-01-01", ".."], "resolution": "P1D"},
        "links": [
            {
                "rel": "self",
                "href": "https://example.org/stations/0-20008-0-THE",
            }
        ],
        "properties": {
            "type": "facility",
            "title": "Test facility",
            "created": "2020-01-02T00:00:00Z",
            "facilityType": concept(FACILITY_TYPE),
            "contacts": [
                {
                    "identifier": "contact:ops@example.org",
                    "organization": "Example Met Service",
                    "emails": [{"value": "ops@example.org"}],
                }
            ],
            "observations": [
                {
                    "id": "12006-point",
                    "observedProperty": concept(OBSERVED_PROPERTY),
                    "observedGeometry": concept(OBSERVED_GEOMETRY),
                    "observedFeature": {"domain": concept(DOMAIN)},
                    "programAffiliations": [
                        {"programAffiliation": concept(PROGRAM)}
                    ],
                    "configurations": [
                        {
                            "id": "cfg-1",
                            "time": {"interval": ["2020-01-01", ".."]},
                            "observingMethod": concept(OBSERVING_METHOD),
                            "operatingStatus": concept(OPERATING_STATUS),
                            "sourceOfObservation": concept(SOURCE_OF_OBSERVATION),
                        }
                    ],
                    "reportingProcedures": [
                        {
                            "internationalExchange": False,
                            "dataPolicy": concept(DATA_POLICY),
                        }
                    ],
                }
            ],
        },
    }


def _errors(record: dict[str, Any]):
    validator = validator_for_schema("wmdr2-record-feature.schema.json")
    return sorted(validator.iter_errors(record), key=lambda error: list(error.path))


def test_generated_record_validates_against_wmdr2_record_schema() -> None:
    assert _errors(_valid_record()) == []


def test_schema_preserves_devt_root_cardinalities() -> None:
    for key in ("id", "conformsTo", "type", "geometry", "properties"):
        record = _valid_record()
        record.pop(key)
        assert _errors(record), key

    record = _valid_record()
    record.pop("links")
    assert _errors(record) == []


def test_schema_preserves_devt_facility_cardinalities() -> None:
    for key in ("type", "title", "facilityType"):
        record = _valid_record()
        record["properties"].pop(key)
        assert _errors(record), key

    for key in ("created", "contacts", "observations"):
        record = _valid_record()
        record["properties"].pop(key)
        assert _errors(record) == [], key


def test_schema_rejects_namespaced_facility_id() -> None:
    record = _valid_record()
    record["id"] = "wsi:0-20008-0-THE"
    assert _errors(record)


def test_schema_rejects_non_iso_time_resolution_word() -> None:
    record = _valid_record()
    record["time"]["resolution"] = "day"
    assert _errors(record)


def test_schema_rejects_old_observation_and_configuration_names() -> None:
    record = _valid_record()
    props = record["properties"]
    props["observationSeries"] = props.pop("observations")
    assert _errors(record)

    record = _valid_record()
    observation = record["properties"]["observations"][0]
    observation["observingConfigurations"] = observation.pop("configurations")
    assert _errors(record)


def test_schema_rejects_time_on_reporting_procedure() -> None:
    record = _valid_record()
    procedure = record["properties"]["observations"][0]["reportingProcedures"][0]
    procedure["time"] = {"interval": ["2020-01-01", ".."]}
    assert _errors(record)


def test_schema_requires_temporal_interval_for_international_exchange() -> None:
    record = _valid_record()
    procedure = record["properties"]["observations"][0]["reportingProcedures"][0]
    procedure["internationalExchange"] = True
    assert _errors(record)

    procedure["temporalReportingInterval"] = "PT1H"
    assert _errors(record) == []


def test_schema_accepts_aggregation_interval_on_reusable_schedule() -> None:
    record = _valid_record()
    record["properties"]["schedules"] = [
        {
            "uid": "schedule_abc",
            "@type": "Event",
            "start": "0001-01-01",
            "wmo.int:aggregationInterval": "PT1H",
        }
    ]
    assert _errors(record) == []


def test_schema_accepts_instrument_serial_number_on_configuration() -> None:
    record = _valid_record()
    cfg = record["properties"]["observations"][0]["configurations"][0]
    cfg["instrumentSerialNumber"] = "SN-001"
    assert _errors(record) == []


def test_schema_rejects_old_serial_number_on_configuration() -> None:
    record = _valid_record()
    cfg = record["properties"]["observations"][0]["configurations"][0]
    cfg["serialNumber"] = "SN-001"
    assert _errors(record)


def test_schema_rejects_serial_number_on_instrument_catalogue_entry() -> None:
    record = _valid_record()
    record["properties"]["instruments"] = [
        {
            "id": "maker-model",
            "manufacturer": "Maker",
            "model": "Model",
            "serialNumber": "SN-001",
        }
    ]
    assert _errors(record)


def test_schema_accepts_context_local_instrument_id() -> None:
    record = _valid_record()
    record["properties"]["instruments"] = [
        {"id": "maker-model", "manufacturer": "Maker", "model": "Model"}
    ]
    record["properties"]["observations"][0]["configurations"][0]["instrument"] = "maker-model"
    assert _errors(record) == []


def test_schema_rejects_embedded_instrument_on_configuration() -> None:
    record = _valid_record()
    cfg = record["properties"]["observations"][0]["configurations"][0]
    cfg["instrument"] = {"id": "maker-model"}
    assert _errors(record)


def test_schema_rejects_multiple_operating_status_values_on_configuration() -> None:
    record = _valid_record()
    cfg = record["properties"]["observations"][0]["configurations"][0]
    cfg["operatingStatus"] = [concept(OPERATING_STATUS), concept(OPERATING_STATUS)]
    assert _errors(record)


def test_negative_tests_start_from_a_valid_fixture() -> None:
    assert _errors(deepcopy(_valid_record())) == []


def test_schema_allows_configuration_without_operating_status() -> None:
    record = _valid_record()
    cfg = record["properties"]["observations"][0]["configurations"][0]
    cfg.pop("operatingStatus")
    assert _errors(record) == []
