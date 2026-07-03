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
FACILITY_SETS_SCHEMA_FILE = "wmdr2-facility-sets.schema.json"


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
FACILITY_SETS_SCHEMA = _load_schema(FACILITY_SETS_SCHEMA_FILE)

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
        FACILITY_SETS_SCHEMA["$id"],
        Resource.from_contents(FACILITY_SETS_SCHEMA, default_specification=DRAFT202012),
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
        "time": {"interval": ["2016-04-28", ".."]},
        "temporalGeometry": {
            "type": "MovingPoint",
            "coordinates": [
                [6.0733333333, 50.5108333333, 671],
                [6.0734, 50.5109, 671],
            ],
            "dates": ["2016-04-28", "2024-01-17"],
            "methods": [["gps"], []],
        },
        "conformsTo": ["http://wigos.wmo.int/spec/wmdr/2/conf/core"],
        "properties": {
            "type": "facility",
            "title": "MONT-RIGI",
            "created": "2020-03-04T00:00:00Z",
            "updated": "2020-03-04T00:00:00Z",
            "description": "Example facility record.",
            "contacts": [
                {
                    "id": "contact:example",
                    "organization": "Royal Meteorological Institute of Belgium",
                    "roles": ["owner"],
                }
            ],
            "keywords": ["0-20000-0-06494", "MONT-RIGI"],
            "facilitySets": ["facilitySet:gaw"],
            "facilityType": "landFixed",
            "wmoRegion": "europe",
            "territory": [
                {"territory": "BEL", "date": "2016-04-28"},
            ],
            "environment": [
                {"date": "..", "climateZone": "Cfb", "surfaceCover": "grassland"},
                {
                    "date": "2020-01-01",
                    "population": [1000, None],
                    "perimeter_km": [10, 50],
                    "surfaceRoughness": "rough",
                    "topographyBathymetry": {
                        "localTopography": "flat",
                        "relativeElevation": "middle",
                        "topographicContext": "rises",
                        "altitudeOrDepth": "veryHighAltitude",
                    },
                },
            ],
            "programAffiliation": [
                {
                    "programAffiliation": "GOSGeneral",
                    "reportingStatus": "operational",
                    "date": "2000-08-17",
                    "programSpecificFacilityId": "GOS-06494",
                },
                {
                    "programAffiliation": "GBON",
                    "reportingStatus": "operational",
                    "date": "2022-09-08",
                },
            ],
            "schedules": [
                {
                    "@type": "Event",
                    "uid": "schedule_daily_12",
                    "start": "0001-01-01T12:00:00",
                    "timeZone": "UTC",
                    "duration": "PT0S",
                    "recurrenceRules": [
                        {"@type": "RecurrenceRule", "frequency": "daily"},
                    ],
                }
            ],
            "reporting": [
                {
                    "id": "reporting:hourly-open",
                    "internationalExchange": True,
                    "temporalAggregate": "PT1H",
                },
                {
                    "id": "reporting:ten-minute-restricted",
                    "internationalExchange": False,
                    "temporalAggregate": "PT10M",
                },
            ],
            "observationSeries": [
                {
                    "id": "observationSeries:179",
                    "title": "domain: atmosphere; geometry: point; variable: 179 Cloud amount",
                    "observedProperty": 179,
                    "observedGeometry": "point",
                    "observedFeature": {"domain": "atmosphere"},
                    "programAffiliation": ["GAWregional"],
                    "officialStatus": [
                        {"officialStatus": "primary", "date": ".."}
                    ],
                    "observingConfigurations": [
                        {"date": "2016-04-29", "deployment": "deployment:id_af2ac7ee-a215-4e90-974c-f4499458cc06", "observingMethod": {"nilReason": "unknown"}, "operatingStatus": "operational", "exposure": "good"},
                    ],
                    "observingProcedures": [
                        {"date": "2025-01-01", "strategy": "unknown", "observingSchedules": ["schedule_daily_12"]},
                    ],
                    "reporting": [
                        {"date": "2016-04-29", "strategy": "unknown", "reporting": "reporting:hourly-open", "uom": None},
                        {"date": "2020-01-01", "strategy": "unknown", "reporting": "reporting:ten-minute-restricted", "uom": "mm"},
                    ],
                }
            ],
            "deployments": [
                {
                    "id": "deployment:id_af2ac7ee-a215-4e90-974c-f4499458cc06",
                    "instrument": "instrument:example",
                    "serialNumber": "ABC123",
                    "sourceOfObservation": "automatic",
                    "referenceSurface": "localGround",
                    "verticalDistanceFromReferenceSurface": {"value": 2.0, "uom": "m"},
                    "geometry": {"type": "Point", "coordinates": [6.0733333333, 50.5108333333, 671]},
                }
            ],
            "instruments": [
                {
                    "id": "instrument:example",
                    "observingMethods": [266],
                    "manufacturer": "Vaisala",
                    "model": "ExampleModel",
                    "verticalRange": {"min": 0, "max": 30},
                }
            ],
        },
    }

def test_facility_centric_record_feature_is_valid() -> None:
    assert _validate(RECORD_SCHEMA, _valid_facility_record_feature()) == []



def test_reporting_definitions_are_reusable_and_history_uses_references() -> None:
    instance = _valid_facility_record_feature()
    observation = instance["properties"]["observationSeries"][0]
    reporting_ids = {item["id"] for item in instance["properties"]["reporting"]}
    assert {item["reporting"] for item in observation["reporting"]} <= reporting_ids
    observation["reporting"][0]["internationalExchange"] = True
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_observation_program_affiliations_are_plain_code_lists() -> None:
    instance = _valid_facility_record_feature()
    observation = instance["properties"]["observationSeries"][0]
    observation["programAffiliation"] = ["GAWregional", "GBON"]
    assert _validate(RECORD_SCHEMA, instance) == []


def test_observation_program_affiliation_allows_structured_code_values() -> None:
    instance = _valid_facility_record_feature()
    observation = instance["properties"]["observationSeries"][0]
    observation.pop("programAffiliation", None)
    observation["programAffiliation"] = [
        {"programAffiliation": "GAWregional", "date": ".."},
    ]
    assert _is_valid(RECORD_SCHEMA, instance)


def test_facility_historical_program_affiliation_is_valid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["programAffiliation"] = [
        {
            "programAffiliation": "GAWregional",
            "reportingStatus": "operational",
            "programSpecificFacilityId": "GAW-123",
            "date": "..",
        }
    ]
    assert _validate(RECORD_SCHEMA, instance) == []

def test_facility_temporal_reporting_status_is_not_part_of_current_core_model() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["temporalReportingStatus"] = [
        {"reportingStatus": "operational", "date": "2020-01-01"}
    ]
    assert not _is_valid(RECORD_SCHEMA, instance)


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


def test_temporal_geometry_is_the_root_aligned_temporal_object() -> None:
    instance = _valid_facility_record_feature()
    assert _validate(RECORD_SCHEMA, instance) == []

    instance["properties"]["temporalGeometry"] = deepcopy(instance["temporalGeometry"])
    assert not _is_valid(RECORD_SCHEMA, instance)

    instance = _valid_facility_record_feature()
    instance["temporalGeometry"].pop("type")
    assert not _is_valid(RECORD_SCHEMA, instance)

    instance = _valid_facility_record_feature()
    instance["temporalGeometry"]["type"] = "Point"
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_temporal_geometry_methods_are_optional_aligned_term_lists() -> None:
    instance = _valid_facility_record_feature()
    instance["temporalGeometry"].pop("methods")
    assert _validate(RECORD_SCHEMA, instance) == []

    instance = _valid_facility_record_feature()
    instance["temporalGeometry"]["methods"] = ["gps", None]
    assert not _is_valid(RECORD_SCHEMA, instance)

    instance = _valid_facility_record_feature()
    instance["temporalGeometry"]["methods"] = [["gps"], [None]]
    assert not _is_valid(RECORD_SCHEMA, instance)

    instance = _valid_facility_record_feature()
    instance["temporalGeometry"]["methods"] = [["gps"], []]
    assert _validate(RECORD_SCHEMA, instance) == []

    instance = _valid_facility_record_feature()
    instance["temporalGeometry"] = {
        "type": "MovingPoint",
        "coordinates": [[6.0733333333, 50.5108333333, 671]],
        "dates": ["2016-04-28"],
        "methods": [["gps"]],
    }
    assert _validate(RECORD_SCHEMA, instance) == []


def test_historical_environment_topography_bathymetry_object_is_valid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["environment"][0]["topographyBathymetry"] = {
        "localTopography": "slope",
        "relativeElevation": "middle",
        "topographicContext": "rises",
        "altitudeOrDepth": "veryHighAltitude",
    }
    assert _validate(RECORD_SCHEMA, instance) == []

def test_environment_obsolete_topography_bathymetry_timeline_is_invalid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["environment"][0]["temporalTopographyBathymetry"] = [
        {"topographyBathymetry": {"localTopography": "flat"}, "date": "2020-01-01"},
    ]
    assert not _is_valid(RECORD_SCHEMA, instance)

def test_environment_obsolete_population_density_timeline_is_invalid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["environment"][0]["temporalPopulationDensities"] = [
        {"populationDensity": [100.0, 200.0], "date": "2020-01-01"},
    ]
    assert not _is_valid(RECORD_SCHEMA, instance)

def test_historical_environment_population_requires_population_perimeter_and_date() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["environment"] = [
        {"date": "2020-01-01", "population": [1000, None], "perimeter_km": [10, 50]},
    ]
    assert _validate(RECORD_SCHEMA, instance) == []

    instance = _valid_facility_record_feature()
    instance["properties"]["environment"] = [
        {"date": "2020-01-01", "population": 1000},
    ]
    assert not _is_valid(RECORD_SCHEMA, instance)

def test_environmental_histories_are_invalid_as_direct_facility_properties() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"].pop("environment", None)
    instance["properties"]["temporalClimateZone"] = [
        {"climateZone": "Cfb", "date": "1982-03-13"},
    ]
    assert not _is_valid(RECORD_SCHEMA, instance)

def test_external_ids_and_singular_facility_set_are_invalid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["externalIds"] = [{"value": "0-20000-0-06494"}]
    assert not _is_valid(RECORD_SCHEMA, instance)

    instance = _valid_facility_record_feature()
    instance["properties"]["facilitySet"] = "GAW"
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_facility_sets_catalog_schema() -> None:
    catalog = {
        "facilitySets": [
            {
                "id": "facilitySet:gaw",
                "title": "GAW",
                "description": "Global Atmosphere Watch facilities.",
            }
        ]
    }
    assert _validate(FACILITY_SETS_SCHEMA, catalog) == []


def test_observation_description_is_not_part_of_current_core_model() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["observationSeries"][0]["description"] = None
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_observation_series_time_is_not_part_of_v0252_model() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["observationSeries"][0]["time"] = {"interval": ["2016-04-29", ".."]}
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_observation_series_direct_deployments_are_not_part_of_v0252_model() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["observationSeries"][0]["deployments"] = [
        "deployment:id_af2ac7ee-a215-4e90-974c-f4499458cc06",
    ]
    assert not _is_valid(RECORD_SCHEMA, instance)

def test_root_deployments_are_stable_contexts_in_v0252_model() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"] = [
        {"id": "deployment:legacy", "instrument": "instrument:example"},
    ]
    assert _is_valid(RECORD_SCHEMA, instance)

def test_deployment_vertical_distance_accepts_structured_quantity() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["verticalDistanceFromReferenceSurface"] = {"value": 2.0, "uom": "m"}
    assert _is_valid(RECORD_SCHEMA, instance)

def test_deployment_date_operating_status_and_exposure_are_not_part_of_stable_context() -> None:
    instance = _valid_facility_record_feature()
    deployment = instance["properties"]["deployments"][0]
    deployment["date"] = "2020-01-01"
    assert not _is_valid(RECORD_SCHEMA, instance)

    instance = _valid_facility_record_feature()
    deployment = instance["properties"]["deployments"][0]
    deployment["operatingStatus"] = "operational"
    assert not _is_valid(RECORD_SCHEMA, instance)

    instance = _valid_facility_record_feature()
    deployment = instance["properties"]["deployments"][0]
    deployment["exposure"] = "good"
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_reusable_deployment_serial_number_and_observation_official_status_are_valid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["serialNumber"] = "XYZ"
    instance["properties"]["observationSeries"][0]["officialStatus"] = [
        {"officialStatus": "primary", "date": "2024-01-01"}
    ]
    assert _is_valid(RECORD_SCHEMA, instance)

def test_deployment_serial_numbers_parallel_array_is_invalid() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["serialNumbers"] = {
        "serialNumber": ["ABC123"],
        "dates": [".."],
    }
    assert not _is_valid(RECORD_SCHEMA, instance)



def test_observation_series_observing_configuration_value_is_required_and_accepts_nil_reason() -> None:
    instance = _valid_facility_record_feature()
    method = instance["properties"]["observationSeries"][0]["observingConfigurations"][0]
    assert method["observingMethod"] == {"nilReason": "unknown"}
    assert _validate(RECORD_SCHEMA, instance) == []

    method.pop("observingMethod")
    assert any("'observingMethod' is a required property" in message for message in _validate(RECORD_SCHEMA, instance))


def test_observing_method_value_accepts_known_codes_on_instrument_and_observation_series() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["instruments"][0]["observingMethods"] = ["266", 267]
    instance["properties"]["observationSeries"][0]["observingConfigurations"] = [
        {"date": "2016-04-29", "observingMethod": "266"},
        {"date": "2017-01-01", "observingMethod": 267},
    ]
    assert _is_valid(RECORD_SCHEMA, instance)



def test_observing_method_rejects_literal_unknown_string_use_nil_reason_instead() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["observationSeries"][0]["observingConfigurations"] = [
        {"date": "2016-04-29", "observingMethod": "unknown"}
    ]
    assert not _is_valid(RECORD_SCHEMA, instance)

    instance["properties"]["observationSeries"][0]["observingConfigurations"] = [
        {"date": "2016-04-29", "observingMethod": {"nilReason": "unknown"}}
    ]
    assert _is_valid(RECORD_SCHEMA, instance)

def test_observing_configuration_requires_date_property() -> None:
    instance = _valid_facility_record_feature()
    method = instance["properties"]["observationSeries"][0]["observingConfigurations"][0]
    method.pop("date")
    assert any("'date' is a required property" in message for message in _validate(RECORD_SCHEMA, instance))

def test_instrument_serial_number_is_not_a_catalogue_property() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["instruments"][0]["serialNumber"] = "ABC123"
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_instrument_vertical_range_requires_min_and_max() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["instruments"][0]["verticalRange"] = {"min": 0, "max": 30}
    assert _validate(RECORD_SCHEMA, instance) == []

    incomplete = _valid_facility_record_feature()
    incomplete["properties"]["instruments"][0]["verticalRange"] = {"min": 0}
    assert any("'max' is a required property" in message for message in _validate(RECORD_SCHEMA, incomplete))


def test_instrument_vertical_range_min_and_max_are_numeric() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["instruments"][0]["verticalRange"] = {"min": "0", "max": 30}
    assert any("'0' is not of type 'number'" in message for message in _validate(RECORD_SCHEMA, instance))


def test_instrument_observed_property_is_not_a_catalogue_property() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["instruments"][0]["observedProperty"] = [179, "free text variable"]
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_instrument_observed_geometry_is_not_a_catalogue_property() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["instruments"][0]["observedGeometry"] = "point"
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_wmdr2_core_conformance_is_required() -> None:
    instance = _valid_facility_record_feature()
    instance["conformsTo"] = []
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_wmdr2_core_conformance_is_the_only_allowed_value() -> None:
    instance = _valid_facility_record_feature()
    instance["conformsTo"] = [
        "http://wigos.wmo.int/spec/wmdr/2/conf/core",
        "http://www.opengis.net/spec/ogcapi-records-1/1.0/conf/record-core",
    ]
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_wmdr2_core_conformance_rejects_https_variant() -> None:
    instance = _valid_facility_record_feature()
    instance["conformsTo"] = ["https://wigos.wmo.int/spec/wmdr/2/conf/core"]
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_schema_descriptions_carry_wmdr1_documentation() -> None:
    deployment = COMMON_SCHEMA["$defs"]["deployment"]
    instrument = COMMON_SCHEMA["$defs"]["instrument"]
    historical_environment = COMMON_SCHEMA["$defs"]["environment"]
    assert "stable deployed observer/instrument context" in deployment["description"]
    assert "Manufacturer of the equipment" in instrument["properties"]["manufacturer"]["description"]
    assert "Environmental context" in historical_environment["description"]

def test_observation_uses_observed_property_not_observed_variable() -> None:
    instance = _valid_facility_record_feature()
    observation = instance["properties"]["observationSeries"][0]
    assert "observedProperty" in observation
    observation["observedVariable"] = observation.pop("observedProperty")
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_observed_feature_accepts_domain_feature_and_feature_name() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["observationSeries"][0]["observedFeature"] = {
        "domain": "atmosphere",
        "domainFeature": "near-surface-air",
        "featureName": "2 m air",
    }
    assert _validate(RECORD_SCHEMA, instance) == []

def test_observed_domain_is_not_part_of_current_core_model() -> None:
    instance = _valid_facility_record_feature()
    observation = instance["properties"]["observationSeries"][0]
    observation["observedDomain"] = observation.pop("observedFeature")
    assert not _is_valid(RECORD_SCHEMA, instance)

def test_observation_series_does_not_carry_reference_surface() -> None:
    instance = _valid_facility_record_feature()
    observation = instance["properties"]["observationSeries"][0]
    observation["referenceSurface"] = "localGround"
    assert not _is_valid(RECORD_SCHEMA, instance)


def test_deployment_uses_reference_surface_not_local_reference_surface() -> None:
    instance = _valid_facility_record_feature()
    deployment = instance["properties"]["deployments"][0]
    assert "referenceSurface" in deployment
    deployment["localReferenceSurface"] = deployment.pop("referenceSurface")
    assert not _is_valid(RECORD_SCHEMA, instance)

def test_deployment_vertical_distance_requires_quantity_object_with_value() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["verticalDistanceFromReferenceSurface"] = 2.0
    assert not _is_valid(RECORD_SCHEMA, instance)

    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["verticalDistanceFromReferenceSurface"] = {"uom": "m"}
    assert not _is_valid(RECORD_SCHEMA, instance)

def test_deployment_geometry_is_valid_point_geometry() -> None:
    instance = _valid_facility_record_feature()
    instance["properties"]["deployments"][0]["geometry"] = {
        "type": "Point",
        "coordinates": [7.0, 46.0, 100],
    }
    assert _validate(RECORD_SCHEMA, instance) == []

    instance["properties"]["deployments"][0]["temporalGeometry"] = {
        "type": "MovingPoint",
        "coordinates": [[7.0, 46.0, 100]],
        "dates": ["2021-01-01"],
    }
    assert not _is_valid(RECORD_SCHEMA, instance)

def test_observation_uses_observed_geometry_not_observed_geometry_type() -> None:
    instance = _valid_facility_record_feature()
    observation = instance["properties"]["observationSeries"][0]
    assert "observedGeometry" in observation
    observation["observedGeometryType"] = observation.pop("observedGeometry")
    assert not _is_valid(RECORD_SCHEMA, instance)
