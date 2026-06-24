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
        "observations": [
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


def test_v022_observation_contains_model_aligned_historical_objects() -> None:
    record = convert_payload(_sample_payload(), source_name="sample")
    props = record["properties"]

    assert props["deployments"]
    assert props["instruments"]

    observation = props["observations"][0]
    assert observation["id"].startswith("observations:")
    assert "deployments" not in observation
    assert "reporting" not in observation
    assert "domain" not in observation
    assert "domainName" not in observation
    assert "observedDomain" not in observation

    assert observation["observedFeature"]["domain"] == "atmosphere"
    assert observation["sourceOfObservation"] == "automaticReading"
    assert observation["referenceSurface"] == "ground"
    assert observation["representativeness"] == "local"
    assert observation["verticalDistanceFromReferenceSurface"]["value"] == 2.0

    deployments = observation["historicalDeployments"]
    assert isinstance(deployments, list)
    assert deployments

    deployment = deployments[0]
    assert deployment["id"].startswith("historicalDeployment:")
    assert deployment["date"] == "2020-01-01"
    assert props["deployments"][0]["serialNumber"] == "SN-001"
    assert props["deployments"][0]["instrument"].startswith("instrument:")
    assert deployment["deployment"] == props["deployments"][0]["id"]
    assert deployment["operatingStatus"] == "operational"
    assert "serialNumber" not in deployment
    assert "instrument" not in deployment
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


def test_v022_historical_reporting_official_status_and_schedule_refs_are_dated_objects() -> None:
    record = convert_payload(_sample_payload(), source_name="sample")
    props = record["properties"]
    observation = props["observations"][0]

    schedules = props["schedules"]
    assert schedules
    assert schedules[0]["uid"].startswith("schedule_")

    schedule_refs = observation["observingSchedules"]
    assert schedule_refs
    assert schedule_refs[0]["date"] == "2020-01-01"
    assert schedule_refs[0]["schedule"] == schedules[0]["uid"]

    reporting_defs = props["reporting"]
    assert reporting_defs
    reporting_def = reporting_defs[0]
    assert reporting_def["id"].startswith("reporting:")
    assert reporting_def["internationalExchange"] is True
    assert reporting_def["temporalAggregate"] == "PT10M"
    assert reporting_def["levelOfData"] == "level1"
    assert reporting_def["timeliness"] == "PT30M"

    historical_reporting = observation["historicalReporting"]
    assert isinstance(historical_reporting, list)
    reporting = historical_reporting[0]
    assert reporting["date"] == "2020-01-01"
    assert reporting["reporting"] == reporting_def["id"]
    assert reporting["uom"] == "K"
    for key in ("internationalExchange", "temporalAggregate", "levelOfData", "timeliness"):
        assert key not in reporting

    official = observation["historicalOfficialStatus"]
    assert isinstance(official, list)
    assert official[0]["date"] == "2020-01-01"
    assert official[0]["officialStatus"] == "primary"


def test_v022_facility_histories_are_unwrapped() -> None:
    record = convert_payload(_sample_payload(), source_name="sample")
    props = record["properties"]

    assert "environment" not in props
    assert "historicalEnvironment" in props
    assert "temporalProgramAffiliation" not in props
    assert "historicalProgramAffiliation" in props

    historical_environment = props["historicalEnvironment"]
    assert isinstance(historical_environment, list)
    assert historical_environment[0]["date"] == "2020-01-01"
    assert historical_environment[0]["climateZone"] == "temperate"
    assert historical_environment[0]["surfaceRoughness"] == "low"


def test_v022_schema_definitions_are_present() -> None:
    schema = json.loads((ROOT / "schemas" / "wmdr2-common.schema.json").read_text(encoding="utf-8"))
    defs = schema["$defs"]

    for name in (
        "observedFeature",
        "historicalDeployment",
        "reporting",
        "historicalReporting",
        "historicalOfficialStatus",
        "historicalEnvironment",
        "historicalProgramAffiliation",
        "historicalTerritory",
        "scheduleReference",
    ):
        assert name in defs

    observation_props = defs["observation"]["properties"]
    assert "observedFeature" in observation_props
    assert "historicalDeployments" in observation_props
    assert "historicalReporting" in observation_props
    assert "historicalOfficialStatus" in observation_props
    assert "observingSchedules" in observation_props
    assert "deployments" not in observation_props
    assert "reporting" not in observation_props
