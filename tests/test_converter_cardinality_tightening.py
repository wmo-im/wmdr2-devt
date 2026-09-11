from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = ROOT / "convert_wmdr10_json_to_wmdr2_json.py"


def load_converter():
    spec = importlib.util.spec_from_file_location("wmdr2_converter", CONVERTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_preserves_untimed_scalar_territory() -> None:
    converter = load_converter()
    uri = "http://codes.wmo.int/wmdr/TerritoryName/CHE"
    assert converter._normalize_territories(uri) == [{"territory": uri}]


def test_converter_preserves_untimed_mapping_territory() -> None:
    converter = load_converter()
    uri = "http://codes.wmo.int/wmdr/TerritoryName/CHE"
    source = {"territoryName": {"href": uri}}
    assert converter._normalize_territories(source) == [{"territory": uri}]


def test_converter_preserves_timed_territory_when_time_is_recorded() -> None:
    converter = load_converter()
    uri = "http://codes.wmo.int/wmdr/TerritoryName/CHE"
    source = {
        "territoryName": {"href": uri},
        "beginPosition": "2020-01-01",
        "endPosition": "..",
    }
    assert converter._normalize_territories(source) == [
        {"territory": uri, "time": {"interval": ["2020-01-01", ".."]}}
    ]


def test_converter_preserves_untimed_scalar_program_affiliation() -> None:
    converter = load_converter()
    uri = "http://codes.wmo.int/wmdr/ProgramAffiliation/GAW"
    assert converter._normalize_program_affiliations(uri) == [{"program": uri}]


def test_converter_preserves_untimed_mapping_program_affiliation() -> None:
    converter = load_converter()
    uri = "http://codes.wmo.int/wmdr/ProgramAffiliation/GAW"
    source = {"programAffiliation": {"href": uri}}
    assert converter._normalize_program_affiliations(source) == [{"program": uri}]


def test_converter_preserves_untimed_reporting_status() -> None:
    converter = load_converter()
    program = "http://codes.wmo.int/wmdr/ProgramAffiliation/GAW"
    status = "http://codes.wmo.int/wmdr/ReportingStatus/operational"
    source = {
        "programAffiliation": {"href": program},
        "reportingStatus": {"reportingStatus": {"href": status}},
    }
    assert converter._normalize_program_affiliations(source) == [
        {"program": program, "reportingStatus": status}
    ]


def test_schedule_normalizer_rejects_identifier_only_or_start_only() -> None:
    converter = load_converter()
    assert converter._normalize_schedule_object({"uid": "schedule_empty"}) is None
    assert (
        converter._normalize_schedule_object(
            {"uid": "schedule_empty", "start": "0001-01-01"}
        )
        is None
    )


def test_schedule_normalizer_requires_more_than_diurnal_base_time() -> None:
    converter = load_converter()
    assert (
        converter._normalize_schedule_object(
            {
                "uid": "schedule_empty",
                "start": "0001-01-01",
                "wmo.int:diurnalBaseTime": "00:00:00",
            }
        )
        is None
    )


def test_schedule_normalizer_accepts_sampling_cadence_and_supplies_start() -> None:
    converter = load_converter()
    schedule = converter._normalize_schedule_object(
        {"uid": "schedule_sampling", "wmo.int:samplingFrequency": "10 min"},
        kind="observing",
    )
    assert schedule is not None
    assert schedule["uid"] == "schedule_sampling"
    assert schedule["start"] == converter.CANONICAL_SCHEDULE_START_DATE
    assert schedule["wmo.int:samplingFrequency"] == "PT10M"


def test_schedule_normalizer_accepts_reporting_cadence() -> None:
    converter = load_converter()
    schedule = converter._normalize_schedule_object(
        {"uid": "schedule_reporting", "temporalReportingInterval": "1 h"},
        kind="reporting",
    )
    assert schedule is not None
    assert schedule["wmo.int:aggregationInterval"] == "PT1H"


def test_environment_converter_keeps_surface_cover_value_and_scheme() -> None:
    converter = load_converter()
    value = "http://codes.wmo.int/wmdr/SurfaceCover/grass"
    scheme = "http://codes.wmo.int/wmdr/SurfaceCoverClassification/LCZ"
    source = {
        "surfaceCover": {
            "surfaceCover": {"href": value},
            "surfaceCoverClassification": {"href": scheme},
        }
    }
    environment = converter._environment_from_facility(source)
    assert environment == [{"surfaceCover": {"value": value, "scheme": scheme}}]

def test_converter_unwraps_single_item_program_affiliation_list() -> None:
    converter = load_converter()
    uri = "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"
    source = {"programAffiliation": [uri]}
    assert converter._normalize_program_affiliations(source) == [{"program": uri}]


def test_environment_converter_omits_unknown_optional_topography() -> None:
    converter = load_converter()
    source = {"topographyBathymetry": {"localTopography": "unknown"}}
    assert converter._environment_from_facility(source) == []


def test_explicit_schedule_keeps_contextual_diurnal_base_time() -> None:
    converter = load_converter()
    source = {
        "reporting": {
            "diurnalBaseTime": "6",
            "reportingSchedule": {
                "recurrenceRules": [{"frequency": "hourly"}]
            },
        }
    }
    schedule = converter._schedule_from_source(source, kind="reporting")
    assert schedule is not None
    assert schedule["wmo.int:diurnalBaseTime"] == "06:00:00"
    assert schedule["recurrenceRules"] == [{"frequency": "hourly"}]

