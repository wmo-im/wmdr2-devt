from pathlib import Path
import importlib.util
import json

MODULE_PATH = Path(__file__).with_name("convert_wmdr10_json_to_wmdr2_geojson.py")
spec = importlib.util.spec_from_file_location("wmdr2_converter", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_convert_payload_minimal_full_record():
    payload = {
        "header": {
            "fileDateTime": "2025-05-04T00:00:00Z",
            "recordOwner": {
                "organisationName": "Test Org",
                "contactInfo": {
                    "address": {"electronicMailAddress": ["test@example.org"]},
                    "onlineResource": {"url": "https://example.org"},
                },
                "role": "https://standards.iso.org/role/publisher",
            },
        },
        "facility": {
            "identifier": "0-20000-0-TEST",
            "name": "Test Station",
            "dateEstablished": "2020-01-01Z",
            "geospatialLocation": {
                "geoLocation": "-1.0 36.0 1795",
                "beginPosition": "2020-01-01T00:00:00Z",
                "endPosition": None,
            },
            "facilityType": "http://codes.wmo.int/wmdr/FacilityType/landFixed",
        },
        "observations": [
            {
                "@gml:id": "obs-1",
                "facility": "http://codes.wmo.int/wmdr/0-20000-0-TEST",
                "observedProperty": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/263",
                "beginPosition": "2025-01-01T00:00:00Z",
                "endPosition": None,
                "deployments": [
                    {
                        "@gml:id": "dep-1",
                        "facility": "http://codes.wmo.int/wmdr/0-20000-0-TEST",
                        "observedProperty": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/263",
                        "sourceOfObservation": "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading",
                        "manufacturer": "ACME",
                        "model": "X1",
                        "serialNumber": "42",
                        "beginPosition": "2025-01-01T00:00:00Z",
                        "endPosition": None,
                        "dataGeneration": [
                            {
                                "@gml:id": "dg-1",
                                "sampling": {"temporalSamplingInterval": "PT1M"},
                                "reporting": {
                                    "internationalExchange": "true",
                                    "temporalReportingInterval": "PT1H",
                                    "beginPosition": "2025-01-01T00:00:00Z",
                                    "endPosition": None,
                                    "coverage": {
                                        "startMonth": "1",
                                        "endMonth": "12",
                                        "startWeekday": "1",
                                        "endWeekday": "7",
                                        "startHour": "0",
                                        "endHour": "23",
                                        "startMinute": "0",
                                        "endMinute": "59",
                                        "diurnalBaseTime": "00:00:00Z",
                                    },
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }

    collection = module.convert_payload(payload, source_name="test.json")
    assert collection["type"] == "FeatureCollection"
    assert len(collection["features"]) == 3

    facility = collection["features"][0]
    assert facility["properties"]["type"] == "facility"
    assert facility["geometry"]["type"] == "Point"
    assert facility["geometry"]["coordinates"] == [36.0, -1.0, 1795]

    observation = collection["features"][1]
    assert observation["properties"]["wmdr2"]["observation"]["internationalExchange"] is True

    deployment = collection["features"][2]
    assert deployment["properties"]["type"] == "deployment"
    assert deployment["properties"]["wmdr2"]["deployment"]["procedure"]["instrumentModel"] == "X1"
    schedules = deployment["properties"]["wmdr2"]["deployment"]["temporalReportingSchedule"]
    assert schedules[0]["interval"] == "PT1H"
