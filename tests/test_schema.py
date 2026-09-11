from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"
COMMON_SCHEMA = SCHEMA_DIR / "wmdr2-common.schema.json"
RECORD_SCHEMA = SCHEMA_DIR / "wmdr2-record-feature.schema.json"

FACILITY_TYPE = "http://codes.wmo.int/wmdr/FacilityType/landFixed"
OBSERVED_PROPERTY = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
OBSERVED_GEOMETRY = "http://codes.wmo.int/wmdr/Geometry/point"
DOMAIN = "http://codes.wmo.int/wmdr/Domain/atmosphere"
PROGRAM = "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"
OBSERVING_METHOD = "http://codes.wmo.int/wmdr/ObservingMethod/266"
OPERATING_STATUS = "http://codes.wmo.int/wmdr/OperatingStatus/operational"
SOURCE_OF_OBSERVATION = "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading"
DATA_POLICY = "http://codes.wmo.int/wmdr/DataPolicy/noLimitation"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validator() -> Draft202012Validator:
    schema = _load(RECORD_SCHEMA)
    registry = Registry()

    if COMMON_SCHEMA.exists():
        common = _load(COMMON_SCHEMA)
        if "$id" in common:
            registry = registry.with_resource(
                common["$id"],
                Resource.from_contents(common, default_specification=DRAFT202012),
            )

    if "$id" in schema:
        registry = registry.with_resource(
            schema["$id"],
            Resource.from_contents(schema, default_specification=DRAFT202012),
        )

    return Draft202012Validator(
        schema,
        registry=registry,
        format_checker=FormatChecker(),
    )


def _valid_record() -> dict[str, Any]:
    """Return a genuinely schema-valid Core record.

    Negative tests below mutate exactly one aspect of this record. Keeping this
    fixture valid is essential: otherwise a test can appear to reject the
    intended construct while actually failing for an unrelated missing Core
    element.
    """
    return {
        "type": "Feature",
        "id": "0-20008-0-THE",
        "conformsTo": ["http://wigos.wmo.int/spec/wmdr/2/conf/core"],
        "geometry": {
            "type": "Point",
            "coordinates": [22.95, 40.63, 5.0],
        },
        "time": {
            "interval": ["2020-01-01", ".."],
            "resolution": "P1D",
        },
        "properties": {
            "type": "facility",
            "title": "Test facility",
            "facilityType": FACILITY_TYPE,
            "observationSeries": [
                {
                    "id": "observationSeries:test",
                    "observedProperty": OBSERVED_PROPERTY,
                    "observedGeometry": OBSERVED_GEOMETRY,
                    "observedFeature": {
                        "domain": DOMAIN,
                    },
                    "programAffiliations": [PROGRAM],
                    "observingConfigurations": [
                        {
                            "time": {
                                "interval": ["2020-01-01", ".."],
                            },
                            "observingMethod": OBSERVING_METHOD,
                            "operatingStatus": OPERATING_STATUS,
                            "sourceOfObservation": SOURCE_OF_OBSERVATION,
                        }
                    ],
                    "reportingProcedures": [
                        {
                            "internationalExchange": False,
                            "dataPolicy": DATA_POLICY,
                        }
                    ],
                }
            ],
        },
    }


def _errors(record: dict[str, Any]):
    return sorted(_validator().iter_errors(record), key=lambda error: list(error.path))


def test_generated_record_validates_against_wmdr2_record_schema() -> None:
    assert _errors(_valid_record()) == []


def test_schema_rejects_namespaced_facility_id() -> None:
    record = _valid_record()
    record["id"] = "wsi:0-20008-0-THE"
    assert _errors(record)


def test_schema_rejects_non_iso_time_resolution_word() -> None:
    record = _valid_record()
    record["time"]["resolution"] = "day"
    assert _errors(record)


def test_schema_rejects_observing_configuration_keywords_and_temporal_geometry() -> None:
    for key, value in (
        ("keywords", ["legacy"]),
        ("temporalGeometry", {"type": "MovingPoint"}),
    ):
        record = _valid_record()
        cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
        cfg[key] = value
        assert _errors(record), key


def test_schema_rejects_time_on_reporting_procedure() -> None:
    record = _valid_record()
    procedure = record["properties"]["observationSeries"][0]["reportingProcedures"][0]
    procedure["time"] = {"interval": ["2020-01-01", ".."]}
    assert _errors(record)


def test_schema_accepts_temporal_reporting_interval_when_exchange_is_false() -> None:
    record = _valid_record()
    procedure = record["properties"]["observationSeries"][0]["reportingProcedures"][0]
    procedure["temporalReportingInterval"] = "PT1H"
    assert _errors(record) == []


def test_schema_requires_complete_international_exchange_reporting_metadata() -> None:
    record = _valid_record()
    procedure = record["properties"]["observationSeries"][0]["reportingProcedures"][0]
    procedure["internationalExchange"] = True
    assert _errors(record)

    procedure["temporalReportingInterval"] = "PT1H"
    procedure["temporalAggregate"] = "PT10M"
    procedure["reportingSchedules"] = ["schedule_reporting"]
    record["properties"]["schedules"] = [
        {
            "uid": "schedule_reporting",
            "@type": "Event",
            "start": "0001-01-01 00:00:00",
            "recurrenceRules": [{"frequency": "hourly"}],
        }
    ]
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


def test_schema_accepts_serial_number_on_observing_configuration() -> None:
    record = _valid_record()
    cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
    cfg["serialNumber"] = "SN-001"
    assert _errors(record) == []


def test_schema_rejects_serial_number_on_instrument_catalogue_entry() -> None:
    record = _valid_record()
    record["properties"]["instruments"] = [
        {
            "id": "instrument:maker-model",
            "manufacturer": "Maker",
            "model": "Model",
            "serialNumber": "SN-001",
        }
    ]
    assert _errors(record)


def test_schema_rejects_multiple_operating_status_values_on_observing_configuration() -> None:
    record = _valid_record()
    cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
    cfg["operatingStatus"] = [OPERATING_STATUS, OPERATING_STATUS]
    assert _errors(record)


def test_negative_tests_start_from_a_valid_fixture() -> None:
    """Guard against the false-positive failure mode that prompted this rewrite."""
    assert _errors(deepcopy(_valid_record())) == []


def test_schema_allows_observing_configuration_without_operating_status() -> None:
    record = _valid_record()
    cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
    cfg.pop("operatingStatus")
    assert _errors(record) == []


def test_schema_does_not_require_temporal_aggregate_for_international_exchange() -> None:
    record = _valid_record()
    procedure = record["properties"]["observationSeries"][0]["reportingProcedures"][0]
    procedure["internationalExchange"] = True
    procedure["temporalReportingInterval"] = "PT1H"
    procedure["reportingSchedules"] = ["schedule_reporting"]
    record["properties"]["schedules"] = [
        {
            "uid": "schedule_reporting",
            "@type": "Event",
            "start": "0001-01-01 00:00:00",
            "recurrenceRules": [{"frequency": "hourly"}],
        }
    ]
    procedure.pop("temporalAggregate", None)
    assert _errors(record) == []
