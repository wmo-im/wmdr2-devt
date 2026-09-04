from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "wmdr2-record-feature.schema.json"


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validator_for(def_name: str) -> Draft202012Validator:
    schema = _schema()
    wrapper = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": f"#/$defs/{def_name}",
        "$defs": schema["$defs"],
    }
    return Draft202012Validator(wrapper, format_checker=FormatChecker())


def _errors(def_name: str, value: Any):
    return list(_validator_for(def_name).iter_errors(value))


OBSERVED = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
GEOMETRY = "http://codes.wmo.int/wmdr/Geometry/point"
PROGRAM = "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"
DOMAIN = "http://codes.wmo.int/wmdr/Domain/atmosphere"
DOMAIN_FEATURE = "http://codes.wmo.int/wmdr/DomainFeature/nearSurface"
METHOD = "http://codes.wmo.int/wmdr/ObservingMethod/266"
STATUS = "http://codes.wmo.int/wmdr/OperatingStatus/operational"
SOURCE = "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading"
REFERENCE_SURFACE = "http://codes.wmo.int/wmdr/LocalReferenceSurface/localGround"
EXPOSURE = "http://codes.wmo.int/wmdr/Exposure/good"
DATA_POLICY = "http://codes.wmo.int/wmdr/DataPolicy/noLimitation"
FACILITY_TYPE = "http://codes.wmo.int/wmdr/FacilityType/landFixed"
WMO_REGION = "http://codes.wmo.int/wmdr/WMORegion/6"


def test_controlled_value_accepts_http_and_https_absolute_uris() -> None:
    assert _errors("controlledValue", DOMAIN) == []
    assert _errors("controlledValue", DOMAIN.replace("http://", "https://")) == []


def test_controlled_value_rejects_compact_and_non_uri_values() -> None:
    assert _errors("controlledValue", "atmosphere")
    assert _errors("controlledValue", 12006)
    assert _errors("controlledValue", True)
    assert _errors("controlledValue", ["atmosphere"])
    assert _errors("controlledValue", {"nilReason": "unknown"})


def test_nillable_controlled_value_accepts_uri_or_nilreason() -> None:
    assert _errors("nillableControlledValue", OBSERVED) == []
    assert _errors("nillableControlledValue", {"nilReason": "unknown"}) == []
    assert _errors("nillableControlledValue", "12006")


def test_facility_type_is_required_and_wmo_region_is_optional_non_nillable() -> None:
    valid = {
        "type": "facility",
        "title": "Test",
        "facilityType": FACILITY_TYPE,
    }
    assert _errors("facilityProperties", valid) == []
    assert _errors("facilityProperties", {"type": "facility", "title": "Test"})
    assert _errors(
        "facilityProperties",
        valid | {"wmoRegion": {"nilReason": "unknown"}},
    )
    assert _errors("facilityProperties", valid | {"wmoRegion": WMO_REGION}) == []


def test_facility_rejects_singular_program_affiliation_alias() -> None:
    valid = {
        "type": "facility",
        "title": "Test",
        "facilityType": FACILITY_TYPE,
    }
    invalid = valid | {
        "programAffiliation": [
            {
                "time": {"interval": ["2020-01-01", ".."]},
                "program": PROGRAM,
            }
        ]
    }
    assert _errors("facilityProperties", invalid)


def test_program_affiliation_uses_program_and_string_specific_id() -> None:
    valid = {
        "time": {"interval": ["2020-01-01", ".."]},
        "program": PROGRAM,
        "reportingStatus": "http://codes.wmo.int/wmdr/ReportingStatus/operational",
        "programSpecificFacilityId": "GBON-123",
    }
    assert _errors("programAffiliation", valid) == []

    nested_alias = valid | {"programAffiliation": PROGRAM}
    assert _errors("programAffiliation", nested_alias)

    bad_status = valid | {"reportingStatus": "operational"}
    assert _errors("programAffiliation", bad_status)


def test_observed_feature_requires_controlled_domain() -> None:
    assert _errors("observedFeature", {"domain": DOMAIN}) == []
    assert _errors("observedFeature", {})
    assert _errors("observedFeature", {"domain": "atmosphere"})
    assert _errors(
        "observedFeature",
        {"domain": DOMAIN, "domainFeature": {"nilReason": "unknown"}},
    )
    assert _errors(
        "observedFeature",
        {"domain": DOMAIN, "domainFeature": DOMAIN_FEATURE, "featureName": "air"},
    ) == []


def _valid_observation_series() -> dict[str, Any]:
    return {
        "id": "observationSeries:test",
        "observedProperty": OBSERVED,
        "observedGeometry": GEOMETRY,
        "observedFeature": {"domain": DOMAIN},
        "programAffiliations": [PROGRAM],
    }


def test_observation_series_core_values_are_required() -> None:
    valid = _valid_observation_series()
    assert _errors("observationSeries", valid) == []

    for key in (
        "id",
        "observedProperty",
        "observedGeometry",
        "observedFeature",
        "programAffiliations",
    ):
        invalid = dict(valid)
        invalid.pop(key)
        assert _errors("observationSeries", invalid), key


def test_observation_series_allows_nilreason_for_mandatory_controlled_values() -> None:
    valid = _valid_observation_series()

    for key in ("observedProperty", "observedGeometry"):
        candidate = dict(valid)
        candidate[key] = {"nilReason": "unknown"}
        assert _errors("observationSeries", candidate) == [], key

    candidate = dict(valid)
    candidate["programAffiliations"] = {"nilReason": "unknown"}
    assert _errors("observationSeries", candidate) == []


def test_observation_series_rejects_compact_values_and_transitional_aliases() -> None:
    valid = _valid_observation_series()

    assert _errors("observationSeries", valid | {"observedProperty": 12006})
    assert _errors("observationSeries", valid | {"observedGeometry": "point"})
    assert _errors("observationSeries", valid | {"programAffiliations": ["GBON"]})
    assert _errors("observationSeries", valid | {"programAffiliation": [PROGRAM]})
    assert _errors("observationSeries", valid | {"reporting": []})
    assert _errors("observationSeries", valid | {"uid": "observationSeries:old"})


def _valid_configuration() -> dict[str, Any]:
    return {
        "time": {"interval": ["2020-01-01", ".."]},
        "observingMethod": METHOD,
        "operatingStatus": STATUS,
        "sourceOfObservation": SOURCE,
    }


def test_observing_configuration_core_values_are_required() -> None:
    valid = _valid_configuration()
    assert _errors("observingConfiguration", valid) == []

    for key in ("time", "observingMethod", "sourceOfObservation"):
        invalid = dict(valid)
        invalid.pop(key)
        assert _errors("observingConfiguration", invalid), key

    without_status = dict(valid)
    without_status.pop("operatingStatus")
    assert _errors("observingConfiguration", without_status) == []


def test_observing_configuration_accepts_nilreason_for_mandatory_values() -> None:
    valid = _valid_configuration()

    for key in ("observingMethod", "sourceOfObservation"):
        candidate = dict(valid)
        candidate[key] = {"nilReason": "unknown"}
        assert _errors("observingConfiguration", candidate) == [], key


def test_operating_status_is_optional_controlled_and_not_nillable() -> None:
    valid = _valid_configuration()

    without_status = dict(valid)
    without_status.pop("operatingStatus")
    assert _errors("observingConfiguration", without_status) == []

    assert _errors("observingConfiguration", valid) == []

    with_nil = dict(valid)
    with_nil["operatingStatus"] = {"nilReason": "unknown"}
    assert _errors("observingConfiguration", with_nil)

    with_compact = dict(valid)
    with_compact["operatingStatus"] = "operational"
    assert _errors("observingConfiguration", with_compact)


def test_optional_exposure_is_controlled_and_not_nillable() -> None:
    valid = _valid_configuration()
    assert _errors("observingConfiguration", valid | {"exposure": EXPOSURE}) == []
    assert _errors(
        "observingConfiguration",
        valid | {"exposure": {"nilReason": "unknown"}},
    )
    assert _errors("observingConfiguration", valid | {"exposure": "good"})


def test_vertical_distance_requires_reference_surface() -> None:
    valid = _valid_configuration()
    distance = {"verticalDistanceFromReferenceSurface": {"value": 2.0}}

    assert _errors("observingConfiguration", valid | distance)
    assert _errors(
        "observingConfiguration",
        valid | distance | {"referenceSurface": REFERENCE_SURFACE},
    ) == []


def test_reference_surface_is_controlled_and_not_nillable() -> None:
    valid = _valid_configuration()

    assert _errors(
        "observingConfiguration",
        valid | {"referenceSurface": REFERENCE_SURFACE},
    ) == []
    assert _errors(
        "observingConfiguration",
        valid | {"referenceSurface": "localGround"},
    )
    assert _errors(
        "observingConfiguration",
        valid | {"referenceSurface": {"nilReason": "unknown"}},
    )


def test_reporting_false_exchange_does_not_require_temporal_values() -> None:
    value = {
        "internationalExchange": False,
        "dataPolicy": DATA_POLICY,
    }
    assert _errors("reportingProcedure", value) == []


def test_reporting_true_exchange_requires_interval_and_schedule_not_aggregate() -> None:
    base = {
        "internationalExchange": True,
        "dataPolicy": DATA_POLICY,
    }
    assert _errors("reportingProcedure", base)

    valid = base | {
        "temporalReportingInterval": "PT1H",
        "reportingSchedules": ["schedule_example"],
    }
    assert _errors("reportingProcedure", valid) == []

    with_aggregate = valid | {
        "temporalAggregate": "PT10M",
    }
    assert _errors("reportingProcedure", with_aggregate) == []


def test_reporting_true_exchange_still_requires_each_core_reporting_field() -> None:
    valid = {
        "internationalExchange": True,
        "dataPolicy": DATA_POLICY,
        "temporalReportingInterval": "PT1H",
        "reportingSchedules": ["schedule_example"],
    }

    missing_interval = dict(valid)
    missing_interval.pop("temporalReportingInterval")
    assert _errors("reportingProcedure", missing_interval)

    missing_schedule = dict(valid)
    missing_schedule.pop("reportingSchedules")
    assert _errors("reportingProcedure", missing_schedule)


def test_reporting_temporal_values_use_iso8601_duration() -> None:
    value = {
        "internationalExchange": True,
        "dataPolicy": DATA_POLICY,
        "temporalReportingInterval": "1 hour",
        "temporalAggregate": "PT10M",
        "reportingSchedules": ["schedule_example"],
    }
    assert _errors("reportingProcedure", value)


def test_data_policy_is_mandatory_controlled_and_nillable() -> None:
    assert _errors(
        "reportingProcedure",
        {"internationalExchange": False},
    )

    assert _errors(
        "reportingProcedure",
        {
            "internationalExchange": False,
            "dataPolicy": {"nilReason": "unknown"},
        },
    ) == []

    assert _errors(
        "reportingProcedure",
        {
            "internationalExchange": False,
            "dataPolicy": "noLimitation",
        },
    )


def test_spatial_reporting_interval_is_not_a_codelist_value() -> None:
    value = {
        "internationalExchange": False,
        "dataPolicy": DATA_POLICY,
        "spatialReportingInterval": "point",
    }
    assert _errors("reportingProcedure", value) == []


def test_reporting_rejects_transitional_reporting_alias() -> None:
    value = {
        "internationalExchange": False,
        "dataPolicy": DATA_POLICY,
        "reporting": "legacy",
    }
    assert _errors("reportingProcedure", value)


def test_legacy_codevalue_remains_transitional_for_unreviewed_properties() -> None:
    assert _errors("codeValue", "weatherForecasting") == []
    assert _errors("codeValue", 12006) == []
    assert _errors("codeValue", {"legacy": "value"}) == []


def test_observation_series_rejects_observed_domain_alias_and_independent_time() -> None:
    valid = {
        "id": "observationSeries:test",
        "observedProperty": OBSERVED,
        "observedGeometry": GEOMETRY,
        "observedFeature": {"domain": DOMAIN},
        "programAffiliations": [PROGRAM],
    }
    assert _errors("observationSeries", valid) == []
    assert _errors("observationSeries", valid | {"observedDomain": {"domain": DOMAIN}})
    assert _errors("observationSeries", valid | {"time": {"interval": ["2020-01-01", ".."]}})


def test_facility_rejects_independent_operating_status() -> None:
    valid = {"type": "facility", "title": "Test", "facilityType": FACILITY_TYPE}
    assert _errors("facilityProperties", valid) == []
    assert _errors("facilityProperties", valid | {"operatingStatus": STATUS})


def test_program_specific_facility_title_is_optional_plain_string() -> None:
    valid = {"program": PROGRAM, "time": {"interval": ["2020-01-01", ".."]}}
    assert _errors("programAffiliation", valid) == []
    assert _errors(
        "programAffiliation",
        valid | {"programSpecificFacilityTitle": "Programme-specific station title"},
    ) == []
    assert _errors(
        "programAffiliation",
        valid | {"programSpecificFacilityTitle": {"href": PROGRAM}},
    )


def test_source_temporal_aliases_are_rejected_on_public_history_objects() -> None:
    cfg = {
        "time": {"interval": ["2020-01-01", ".."]},
        "observingMethod": METHOD,
        "operatingStatus": STATUS,
        "sourceOfObservation": SOURCE,
    }
    for key in ("beginPosition", "endPosition", "validFrom", "validTo"):
        assert _errors("observingConfiguration", cfg | {key: "2020-01-01"}), key

    reporting = {
        "internationalExchange": False,
        "dataPolicy": DATA_POLICY,
    }
    for key in ("beginPosition", "endPosition", "validFrom", "validTo"):
        assert _errors("reportingProcedure", reporting | {key: "2020-01-01"}), key
