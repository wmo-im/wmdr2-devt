from __future__ import annotations

from typing import Any

from schema_registry import validator_for_def, validator_for_schema


OBSERVED = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
GEOMETRY = "http://codes.wmo.int/wmdr/Geometry/point"
PROGRAM = "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"
DOMAIN = "http://codes.wmo.int/wmdr/Domain/atmosphere"
METHOD = "http://codes.wmo.int/wmdr/ObservingMethod/266"
STATUS = "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/operational"
SOURCE = "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading"
REFERENCE_SURFACE = "http://codes.wmo.int/wmdr/ReferenceSurfaceType/localGround"
UNIT_M = "http://codes.wmo.int/wmdr/unit/m"
EXPOSURE = "http://codes.wmo.int/wmdr/Exposure/good"
DATA_POLICY = "http://codes.wmo.int/wmdr/DataPolicy/noLimitation"
FACILITY_TYPE = "http://codes.wmo.int/wmdr/FacilityType/landFixed"
WMO_REGION = "https://codes.wmo.int/wmdr/WMORegion/6"


def concept(uri: str) -> dict[str, str]:
    return {"id": uri}


def _errors_schema(name: str, value: Any):
    return list(validator_for_schema(name).iter_errors(value))


def _errors_def(name: str, value: Any):
    return list(
        validator_for_def("wmdr2-common.schema.json", name).iter_errors(value)
    )


def test_controlled_concept_requires_absolute_http_uri() -> None:
    assert _errors_def("controlledConcept", concept(DOMAIN)) == []
    assert _errors_def(
        "controlledConcept",
        concept(DOMAIN.replace("http://", "https://")),
    ) == []
    assert _errors_def("controlledConcept", DOMAIN)
    assert _errors_def("controlledConcept", {"id": "atmosphere"})
    assert _errors_def("controlledConcept", {"id": 12006})
    assert _errors_def("controlledConcept", {"nilReason": "unknown"})


def test_controlled_concept_or_null_accepts_concept_or_json_null() -> None:
    assert _errors_def("controlledConceptOrNull", concept(OBSERVED)) == []
    assert _errors_def("controlledConceptOrNull", None) == []
    assert _errors_def(
        "controlledConceptOrNull",
        {"nilReason": "unknown"},
    )


def _valid_configuration() -> dict[str, Any]:
    return {
        "id": "cfg-1",
        "time": {"interval": ["2020-01-01", ".."]},
        "observingMethod": concept(METHOD),
        "operatingStatus": concept(STATUS),
        "sourceOfObservation": concept(SOURCE),
    }


def test_configuration_requires_id_and_time() -> None:
    valid = _valid_configuration()
    assert _errors_schema("wmdr2-configuration.schema.json", valid) == []
    for key in ("id", "time"):
        invalid = dict(valid)
        invalid.pop(key)
        assert _errors_schema("wmdr2-configuration.schema.json", invalid), key


def test_configuration_controlled_optional_values_are_concept_or_null() -> None:
    valid = _valid_configuration()
    for key in ("observingMethod", "sourceOfObservation", "operatingStatus"):
        candidate = dict(valid)
        candidate[key] = None
        assert _errors_schema("wmdr2-configuration.schema.json", candidate) == []
        candidate[key] = "unknown"
        assert _errors_schema("wmdr2-configuration.schema.json", candidate)


def test_configuration_instrument_is_reference_not_embedded_object() -> None:
    valid = _valid_configuration()
    assert _errors_schema(
        "wmdr2-configuration.schema.json",
        valid | {"instrument": "vaisala-hmp155"},
    ) == []
    assert _errors_schema(
        "wmdr2-configuration.schema.json",
        valid | {"instrument": {"id": "vaisala-hmp155"}},
    )


def test_optional_exposure_is_controlled_and_not_nillable() -> None:
    valid = _valid_configuration()
    assert _errors_schema(
        "wmdr2-configuration.schema.json",
        valid | {"exposure": concept(EXPOSURE)},
    ) == []
    assert _errors_schema(
        "wmdr2-configuration.schema.json",
        valid | {"exposure": None},
    )
    assert _errors_schema(
        "wmdr2-configuration.schema.json",
        valid | {"exposure": "good"},
    )


def test_vertical_distance_uses_official_structure() -> None:
    valid = _valid_configuration()
    vertical = {
        "distances": [2.0],
        "unit": concept(UNIT_M),
        "referenceSurface": concept(REFERENCE_SURFACE),
    }
    assert _errors_schema(
        "wmdr2-configuration.schema.json",
        valid | {"verticalDistance": vertical},
    ) == []

    incomplete = dict(vertical)
    incomplete.pop("referenceSurface")
    assert _errors_schema(
        "wmdr2-configuration.schema.json",
        valid | {"verticalDistance": incomplete},
    )


def test_vertical_distance_required_concepts_may_be_explicit_null() -> None:
    valid = _valid_configuration()
    vertical = {
        "distances": [2.0],
        "unit": None,
        "referenceSurface": None,
    }
    assert _errors_schema(
        "wmdr2-configuration.schema.json",
        valid | {"verticalDistance": vertical},
    ) == []


def test_old_vertical_distance_and_serial_names_are_rejected() -> None:
    valid = _valid_configuration()
    for old_key, value in (
        ("verticalDistanceFromReferenceSurface", {"value": 2.0}),
        ("referenceSurface", concept(REFERENCE_SURFACE)),
        ("serialNumber", "SN-1"),
    ):
        assert _errors_schema(
            "wmdr2-configuration.schema.json",
            valid | {old_key: value},
        )
    assert _errors_schema(
        "wmdr2-configuration.schema.json",
        valid | {"instrumentSerialNumber": "SN-1"},
    ) == []


def _valid_observation() -> dict[str, Any]:
    return {
        "id": "12006-point",
        "observedProperty": concept(OBSERVED),
        "observedGeometry": concept(GEOMETRY),
        "observedFeature": {"domain": concept(DOMAIN)},
        "programAffiliations": [
            {"programAffiliation": concept(PROGRAM)}
        ],
        "configurations": [
            {"id": "cfg-1", "time": {"interval": ["2020-01-01", ".."]}}
        ],
    }


def test_observation_requires_devt_core_values() -> None:
    valid = _valid_observation()
    assert _errors_schema("wmdr2-observation.schema.json", valid) == []
    for key in (
        "id",
        "observedProperty",
        "observedGeometry",
        "observedFeature",
        "programAffiliations",
        "configurations",
    ):
        invalid = dict(valid)
        invalid.pop(key)
        assert _errors_schema("wmdr2-observation.schema.json", invalid), key


def test_observation_rejects_old_names_and_scalar_controlled_values() -> None:
    valid = _valid_observation()
    assert _errors_schema(
        "wmdr2-observation.schema.json",
        valid | {"observedProperty": OBSERVED},
    )
    assert _errors_schema(
        "wmdr2-observation.schema.json",
        valid | {"observedGeometry": GEOMETRY},
    )
    assert _errors_schema(
        "wmdr2-observation.schema.json",
        valid | {"observationSeries": []},
    )
    assert _errors_schema(
        "wmdr2-observation.schema.json",
        valid | {"observingConfigurations": []},
    )


def test_program_affiliation_uses_official_names_and_optional_dates() -> None:
    valid = _valid_observation()
    valid["programAffiliations"] = [
        {
            "programAffiliation": concept(PROGRAM),
            "reportingStatus": concept(
                "http://codes.wmo.int/wmdr/ReportingStatus/operational"
            ),
            "dates": ["2020-01-01", ".."],
        }
    ]
    assert _errors_schema("wmdr2-observation.schema.json", valid) == []

    invalid = _valid_observation()
    invalid["programAffiliations"] = [{"program": concept(PROGRAM)}]
    assert _errors_schema("wmdr2-observation.schema.json", invalid)


def test_observation_rejects_independent_time() -> None:
    valid = _valid_observation()
    assert _errors_schema(
        "wmdr2-observation.schema.json",
        valid | {"time": {"interval": ["2020-01-01", ".."]}},
    )


def test_reporting_false_exchange_does_not_require_temporal_values() -> None:
    value = {
        "internationalExchange": False,
        "dataPolicy": concept(DATA_POLICY),
    }
    assert _errors_def("reportingProcedure", value) == []


def test_reporting_true_exchange_requires_interval() -> None:
    base = {
        "internationalExchange": True,
        "dataPolicy": concept(DATA_POLICY),
    }
    assert _errors_def("reportingProcedure", base)
    assert _errors_def(
        "reportingProcedure",
        base | {"temporalReportingInterval": "PT1H"},
    ) == []


def test_reporting_temporal_values_use_iso8601_duration() -> None:
    value = {
        "internationalExchange": True,
        "dataPolicy": concept(DATA_POLICY),
        "temporalReportingInterval": "1 hour",
    }
    assert _errors_def("reportingProcedure", value)


def test_spatial_reporting_interval_is_not_a_codelist_value() -> None:
    value = {
        "internationalExchange": False,
        "dataPolicy": concept(DATA_POLICY),
        "spatialReportingInterval": "point",
    }
    assert _errors_def("reportingProcedure", value) == []


def test_instrument_extends_official_model_without_serial_number() -> None:
    valid = {
        "id": "vaisala-hmp155",
        "manufacturer": "Vaisala",
        "model": "HMP155",
        "observingMethods": [concept(METHOD)],
    }
    assert _errors_schema("wmdr2-instrument.schema.json", valid) == []
    assert _errors_schema(
        "wmdr2-instrument.schema.json",
        valid | {"serialNumber": "SN-1"},
    )
