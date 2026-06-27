from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convert_wmdr10_json_to_wmdr2_json import convert_payload


def _sample_payload() -> dict:
    return {
        "header": {"fileDateTime": "20260511"},
        "facility": {
            "identifier": "0-99999-0-TST",
            "name": "Test station",
            "dateEstablished": "2020-01-01",
            "programAffiliation": [
                {
                    "programAffiliation": "GOSGeneral",
                    "beginPosition": "2020-01-01",
                    "reportingStatus": "operational",
                }
            ],
            "environment": {
                "temporalClimateZone": [
                    {"date": "2020-01-01", "climateZone": "temperate"}
                ],
                "temporalSurfaceRoughness": [
                    {"date": "2020-01-01", "surfaceRoughness": "low"}
                ],
            },
            "geospatialLocation": {
                "pos": "46.0 7.0 500",
                "beginPosition": "2020-01-01",
            },
        },
        "observationSeries": [
            {
                "identifier": "obs-1",
                "observedVariable": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006",
                "observedGeometry": "point",
                "applicationArea": ["weatherForecasting"],
                "programAffiliation": "GOSGeneral",
                "deployments": [
                    {
                        "identifier": "dep-1",
                        "beginPosition": "2020-01-01",
                        "manufacturer": "Acme",
                        "model": "WX-1",
                        "serialNumber": "SN-001",
                        "sourceOfObservation": "automaticReading",
                        "referenceSurface": "ground",
                        "representativeness": "local",
                        "exposure": "good",
                        "officialStatus": True,
                        "instrumentOperatingStatus": [
                            {
                                "instrumentOperatingStatus": "operational",
                                "beginPosition": "2020-01-01",
                            }
                        ],
                        "heightAboveLocalReferenceSurface": {
                            "@uom": "m",
                            "#text": "2.0",
                        },
                        "geospatialLocation": {
                            "pos": "46.0 7.0 502",
                            "beginPosition": "2020-01-01",
                        },
                        "dataGeneration": [
                            {
                                "beginPosition": "2020-01-01",
                                "temporalSamplingInterval": "PT10M",
                                "reporting": {
                                    "internationalExchange": "true",
                                    "temporalAggregate": "PT10M",
                                    "uom": "K",
                                    "levelOfData": "level1",
                                    "timeliness": "PT30M",
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_v024_observation_contains_model_aligned_historical_objects() -> None:
    record = convert_payload(_sample_payload(), source_name="sample")
    props = record["properties"]

    assert props["deployments"]
    assert props["instruments"]

    observation = props["observationSeries"][0]
    assert observation["id"].startswith("observationSeries:")
    assert "deployments" in observation
    assert "reporting" in observation
    assert "domain" not in observation
    assert "domainName" not in observation
    assert "observedDomain" not in observation

    assert observation["observedFeature"]["domain"] == "atmosphere"
    assert observation["sourceOfObservation"] == "automaticReading"
    assert observation["referenceSurface"] == "ground"
    assert observation["representativeness"] == "local"
    assert observation["verticalDistanceFromReferenceSurface"]["value"] == 2.0

    deployment_refs = observation["deployments"]
    assert isinstance(deployment_refs, list)
    assert deployment_refs

    deployment = props["deployments"][0]
    assert deployment_refs == [deployment["id"]]
    assert deployment["id"].startswith("deployment:")
    assert deployment["date"] == "2020-01-01"
    assert deployment["serialNumber"] == "SN-001"
    assert deployment["instrument"].startswith("instrument:")
    instrument = props["instruments"][0]
    assert "observingMethods" not in instrument
    assert observation["observingMethods"] == [{"date": "..", "observingMethod": {"nilReason": "unknown"}}]
    assert deployment["operatingStatus"] == "operational"
    assert deployment["geometry"]["type"] == "Point"
    assert deployment["exposure"] == "good"

    for key in (
        "officialStatus",
        "sourceOfObservation",
        "referenceSurface",
        "representativeness",
        "verticalDistanceFromReferenceSurface",
        "observingSchedule",
        "temporalObservingSchedule",
    ):
        assert key not in deployment


def test_v024_reporting_official_status_and_observing_procedures_are_dated_objects() -> None:
    record = convert_payload(_sample_payload(), source_name="sample")
    props = record["properties"]
    observation = props["observationSeries"][0]

    schedules = props["schedules"]
    assert schedules
    assert schedules[0]["uid"].startswith("schedule_")

    observing_procedures = observation["observingProcedures"]
    assert observing_procedures
    assert observing_procedures[0]["date"] == "2020-01-01"
    assert observing_procedures[0]["strategy"] == "unknown"
    assert observing_procedures[0]["observingSchedules"] == [schedules[0]["uid"]]

    reporting_defs = props["reporting"]
    assert reporting_defs
    reporting_def = reporting_defs[0]
    assert reporting_def["id"].startswith("reporting:")
    assert reporting_def["internationalExchange"] is True
    assert reporting_def["temporalAggregate"] == "PT10M"
    assert reporting_def["levelOfData"] == "level1"
    assert reporting_def["timeliness"] == "PT30M"

    historical_reporting = observation["reporting"]
    assert isinstance(historical_reporting, list)
    reporting = historical_reporting[0]
    assert reporting["date"] == "2020-01-01"
    assert reporting["strategy"] == "unknown"
    assert reporting["reporting"] == reporting_def["id"]
    assert reporting["uom"] == "K"
    for key in ("internationalExchange", "temporalAggregate", "levelOfData", "timeliness"):
        assert key not in reporting

    official = observation["officialStatus"]
    assert isinstance(official, list)
    assert official[0]["date"] == "2020-01-01"
    assert official[0]["officialStatus"] == "primary"


def test_v024_facility_histories_are_unwrapped() -> None:
    record = convert_payload(_sample_payload(), source_name="sample")
    props = record["properties"]

    assert "environment" in props
    assert "temporalProgramAffiliation" not in props
    assert "programAffiliation" in props

    environment = props["environment"]
    assert isinstance(environment, list)
    assert environment[0]["date"] == "2020-01-01"
    assert environment[0]["climateZone"] == "temperate"
    assert environment[0]["surfaceRoughness"] == "low"


def test_v024_schema_definitions_are_present() -> None:
    schema = json.loads((ROOT / "schemas" / "wmdr2-common.schema.json").read_text(encoding="utf-8"))
    defs = schema["$defs"]

    for name in (
        "observedFeature",
        "deployment",
        "reporting",
        "reportingProcedure",
        "observingProcedure",
        "officialStatus",
        "environment",
        "programAffiliation",
        "territory",
        "schedule",
    ):
        assert name in defs

    observation_props = defs["observation"]["properties"]
    assert "observedFeature" in observation_props
    assert "deployments" in observation_props
    assert "reporting" in observation_props
    assert "officialStatus" in observation_props
    assert "observingProcedures" in observation_props
    assert "deployments" in observation_props
    assert "reporting" in observation_props
