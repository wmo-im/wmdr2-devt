from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convert_wmdr2_json_to_wmdr2_gc_dar import GcDarPaths, convert_files, convert_record_to_gc_dar


def _minimal_full_record() -> dict:
    return {
        "type": "Feature",
        "id": "facility:0-20000-0-06725",
        "geometry": {"type": "Point", "coordinates": [7.8232, 46.4204, 1540]},
        "time": {"interval": ["2000-08-17", ".."]},
        "conformsTo": ["http://wigos.wmo.int/spec/wmdr/2/conf/core"],
        "properties": {
            "type": "facility",
            "title": "Blatten",
            "keywords": ["0-20000-0-06725", "Blatten"],
            "programAffiliation": [
                {"date": "2000-08-17", "programAffiliation": "GOSGeneral", "reportingStatus": "operational"},
                {"date": "2022-09-08", "programAffiliation": "GBON", "reportingStatus": "operational"},
            ],
            "territory": [{"date": "2000-08-17", "territory": "CHE"}],
            "observationSeries": [
                {
                    "id": "observationSeries:12006:old",
                    "title": "historic air temperature",
                    "time": {"interval": ["2000-01-01", "2019-12-31"]},
                    "observedProperty": 12006,
                    "observedGeometry": "point",
                    "observedFeature": {"domain": "atmosphere", "domainFeature": "near-surface-air"},
                    "programAffiliation": ["GOSGeneral"],
                    "sourceOfObservation": "manualReading",
                    "observingConfigurations": [
                        {"date": "2000-01-01", "deployment": "deployment:old", "observingMethod": 1}
                    ],
                },
                {
                    "id": "observationSeries:12006",
                    "title": "domain: atmosphere; geometry: point; variable: 12006",
                    "time": {"interval": ["2020-01-01", ".."]},
                    "observedProperty": 12006,
                    "observedGeometry": "point",
                    "observedFeature": {"domain": "atmosphere", "domainFeature": "near-surface-air"},
                    "programAffiliation": ["GBON"],
                    "applicationArea": ["weatherForecasting"],
                    "sourceOfObservation": "automaticReading",
                    "observingConfigurations": [
                        {"date": "2020-01-01", "deployment": "deployment:dep-1", "observingMethod": 266}
                    ],
                    "officialStatus": [{"date": "2020-01-01", "officialStatus": "primary"}],
                    "reporting": [
                        {"date": "2020-01-01", "strategy": "unknown", "reporting": "reporting:hourly", "uom": "K"}
                    ],
                },
            ],
            "deployments": [
                {"id": "deployment:old", "date": "2000-01-01", "instrument": "instrument:old"},
                {"id": "deployment:dep-1", "date": "2020-01-01", "instrument": "instrument:aws-1"},
            ],
            "reporting": [
                {
                    "id": "reporting:hourly",
                    "internationalExchange": True,
                    "temporalAggregate": "PT1H",
                    "levelOfData": "level1",
                }
            ],
            "instruments": [
                {"id": "instrument:old", "manufacturer": "Old", "model": "Old"},
                {"id": "instrument:aws-1", "manufacturer": "Vaisala", "model": "HMP155"},
            ],
        },
    }


def test_convert_record_to_gc_dar_extracts_current_discovery_summary():
    dar = convert_record_to_gc_dar(
        _minimal_full_record(),
        source_filename="full.json",
        derived_at="2026-06-29T00:00:00Z",
        full_record_href_template="https://example.org/full/{plain_id}.json",
        dar_record_href_template="https://example.org/dar/{plain_id}.json",
    )

    assert dar["type"] == "Feature"
    assert dar["id"] == "facility:0-20000-0-06725"
    assert dar["conformsTo"] == ["http://wigos.wmo.int/spec/wmdr/2/conf/gc-dar"]
    props = dar["properties"]
    assert "summary" not in props
    assert props["territory"] == "CHE"
    assert props["programAffiliation"] == [
        {"program": "GOSGeneral", "reportingStatus": "operational", "date": "2000-08-17"},
        {"program": "GBON", "reportingStatus": "operational", "date": "2022-09-08"},
    ]
    assert props["observedProperty"] == [12006]
    assert props["observedFeatureDomain"] == ["atmosphere"]
    assert "instrument" not in props
    assert "sourceOfObservation" not in props
    assert "internationalExchange" not in props
    assert len(dar["properties"]["observationSeries"]) == 1
    series = dar["properties"]["observationSeries"][0]
    assert series["current"] is True
    assert series["sourceOfObservation"] == "automaticReading"
    assert series["observingMethod"] == 266
    assert series["deployment"] == "deployment:dep-1"
    assert series["instrument"] == "instrument:aws-1"
    assert series["uom"] == "K"
    assert series["internationalExchange"] is True
    assert series["temporalAggregate"] == "PT1H"
    assert series["levelOfData"] == "level1"
    assert "reportingDefinitions" not in dar["properties"]
    assert dar["properties"]["provenance"]["sourceContentHash"].startswith("sha256:")
    assert dar["properties"]["retrieval"]["fullRecord"]["href"] == "https://example.org/full/0-20000-0-06725.json"


def test_convert_files_writes_matching_relative_path(tmp_path: Path):
    source = tmp_path / "full"
    target = tmp_path / "dar"
    source.mkdir()
    (source / "record.json").write_text(json.dumps(_minimal_full_record()), encoding="utf-8")

    written = convert_files(GcDarPaths(source=source, target=target))

    assert written == [target / "record.json"]
    payload = json.loads((target / "record.json").read_text(encoding="utf-8"))
    assert payload["properties"]["profile"] == "wmdr2-gc-dar"
    assert "provenance" in payload["properties"]
