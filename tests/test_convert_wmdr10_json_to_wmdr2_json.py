from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "convert_wmdr10_json_to_wmdr2_json.py"

OBSERVED_179 = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/179"
OBSERVED_12006 = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
GEOMETRY_POINT = "http://codes.wmo.int/wmdr/Geometry/point"


def _load_converter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("convert_wmdr10_json_to_wmdr2_json", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_converter()


def _minimal_facility() -> dict[str, Any]:
    return {
        "identifier": "0-TEST",
        "name": "TEST",
        "geospatialLocation": "46 7 100",
    }


def _data_generation_with_coverage_and_reporting() -> dict[str, Any]:
    return {
        "beginPosition": "2020-01-01T00:00:00Z",
        "sampling": {
            "samplingStrategy": "continuous",
            "temporalSamplingInterval": "PT2S",
            "samplingTimePeriod": "PT2S",
        },
        "reporting": {
            "internationalExchange": "true",
            "temporalReportingInterval": "PT1H",
                "timeliness": "PT30M",
            "uom": "http://codes.wmo.int/wmdr/unit/mm",
            "dataPolicy": {
                "dataPolicy": "http://codes.wmo.int/wmdr/DataPolicy/noLimitation",
                "attribution": {"originator": {"role": None}},
            },
            "levelOfData": "http://codes.wmo.int/wmdr/LevelOfData/level1",
        },
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
    }


def _observation_with_deployment(*, duplicate_data_generation: bool = False) -> dict[str, Any]:
    data_generation = [_data_generation_with_coverage_and_reporting()]
    if duplicate_data_generation:
        data_generation.append(_data_generation_with_coverage_and_reporting())

    return {
        "observedProperty": OBSERVED_12006,
        "type": GEOMETRY_POINT,
        "deployments": [
            {
                "id": "dep1",
                "beginPosition": "2020-01-01T00:00:00Z",
                "sourceOfObservation": "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading",
                "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/266",
                "manufacturer": "Maker",
                "model": "Model",
                "serialNumber": "SN1",
                "dataGeneration": data_generation,
            }
        ],
    }


def test_observation_title_uses_label_and_domain_when_labels_available() -> None:
    module.CODE_LIST_LABELS.clear()
    module.CODE_LIST_LABELS.update({"ObservedVariableAtmosphere": {"179": "Cloud amount"}})

    title = module._format_observation_title(OBSERVED_179)

    assert title == "domain: atmosphere; variable: 179 Cloud amount"


def test_observation_title_includes_geometry_when_available() -> None:
    module.CODE_LIST_LABELS.clear()
    module.CODE_LIST_LABELS.update({"ObservedVariableAtmosphere": {"179": "Cloud amount"}})

    title = module._format_observation_title(OBSERVED_179, GEOMETRY_POINT)

    assert title == "domain: atmosphere; geometry: point; variable: 179 Cloud amount"


def test_observation_title_falls_back_to_code_and_domain_without_label() -> None:
    module.CODE_LIST_LABELS.clear()

    title = module._format_observation_title(OBSERVED_179)

    assert title == "domain: atmosphere; variable: 179"


def test_observation_description_collapses_unknown_unknown() -> None:
    obs = {
        "observedProperty": OBSERVED_179,
        "type": "point",
    }
    deployments = [
        {"manufacturer": "(unknown)", "model": "unknown", "observingMethod": None},
    ]

    desc = module._observation_description(obs, deployments)

    assert desc == "Observed variable 179; geometry type point; deployment procedure unknown"


def test_observation_description_humanizes_observing_method() -> None:
    obs = {
        "observedProperty": OBSERVED_179,
        "type": "point",
    }
    deployments = [
        {"observingMethod": "instrumentAutomaticReading"},
    ]

    desc = module._observation_description(obs, deployments)

    assert desc == (
        "Observed variable 179; geometry type point; "
        "deployment procedure instrument automatic reading"
    )


def test_temporal_observing_schedule_defaults_interval_unknown_when_only_id_present() -> None:
    data_generation = [
        {"@gml:id": "dg-1"},
    ]

    schedule = module._normalize_temporal_observing_schedule(data_generation)

    assert schedule == [{"interval": "unknown"}]


def test_temporal_reporting_schedule_retains_substantive_reporting_payload() -> None:
    data_generation = [
        {
            "@gml:id": "dg-1",
            "reporting": {"internationalExchange": "true"},
        },
    ]

    schedule = module._normalize_temporal_reporting_schedule(data_generation)

    assert schedule == [
        {
            "id": "dg-1",
            "interval": "unknown",
            "reporting": {"internationalExchange": True},
        }
    ]


def test_default_discovery_policy_has_no_themes() -> None:
    assert "themes" not in module.DEFAULT_DISCOVERY_POLICY["facility"]
    assert "themes" not in module.DEFAULT_DISCOVERY_POLICY["observation"]
    assert "themes" not in module.DEFAULT_DISCOVERY_POLICY["deployment"]
    assert module.DEFAULT_DISCOVERY_POLICY["facility"]["keywords"] == ["identifier", "name"]


def test_build_facility_feature_uses_current_core_model() -> None:
    record = module.build_facility_feature(
        _minimal_facility(),
        [_observation_with_deployment()],
        [],
        {},
        source_name="20200101_0-TEST",
    )

    props = record["properties"]

    assert record["type"] == "Feature"
    assert record["geometry"] == {"type": "Point", "coordinates": [7.0, 46.0, 100]}
    assert "wmdr2" not in props
    assert "themes" not in props
    assert props["observations"][0]["observedVariable"] == 12006
    assert props["observations"][0]["observedGeometryType"] == "point"
    assert props["observations"][0]["observedDomain"] == "atmosphere"
    assert "description" not in props["observations"][0]


def test_reporting_arrays_preserve_policy_attribution_and_level_of_data() -> None:
    record = module.build_facility_feature(
        _minimal_facility(),
        [_observation_with_deployment()],
        [],
        {},
        source_name="20200101_0-TEST",
    )

    reporting = record["properties"]["observations"][0]["reporting"]

    assert reporting["internationalExchange"] == [True]
    assert reporting["temporalAggregate"] == ["PT1H"]
    assert reporting["temporalTimeliness"] == {
        "timeliness": ["PT30M"],
        "dates": ["2020-01-01"],
    }
    assert reporting["uom"] == ["mm"]
    assert reporting["levelOfData"] == ["level1"]
    assert reporting["dataPolicy"] == [
        {
            "dataPolicy": "noLimitation",
            "attribution": {"originator": {"role": None}},
        }
    ]


def test_schedule_extensions_use_wmo_int_namespace_and_aggregation_object() -> None:
    record = module.build_facility_feature(
        _minimal_facility(),
        [_observation_with_deployment()],
        [],
        {},
        source_name="20200101_0-TEST",
    )

    schedule = record["properties"]["schedules"][0]

    assert schedule["@type"] == "Event"
    assert schedule["uid"].startswith("schedule_")
    assert schedule["start"] == "0001-01-01T00:00:00"
    assert schedule["timeZone"] == "UTC"
    assert schedule["duration"] == "P1D"
    assert schedule["recurrenceRules"] == [{"@type": "RecurrenceRule", "frequency": "daily"}]
    assert "wmo.int:diurnalBaseTime" not in schedule
    assert schedule["wmo.int:sampling"] == {
        "samplingStrategy": "continuous",
        "temporalSamplingInterval": "PT2S",
        "samplingTimePeriod": "PT2S",
    }
    assert schedule["wmo.int:aggregation"] == {
        "temporalAggregate": "PT1H",
        "diurnalBaseTime": "00:00:00",
    }
    assert "wmo.int:archiving" not in schedule
    assert "wmo.int:aggregating" not in schedule
    assert not any(key.startswith("wmdr2.wmo.int:") for key in schedule)


def test_duplicate_temporal_observing_schedule_references_are_removed() -> None:
    record = module.build_facility_feature(
        _minimal_facility(),
        [_observation_with_deployment(duplicate_data_generation=True)],
        [],
        {},
        source_name="20200101_0-TEST",
    )

    schedules = record["properties"]["schedules"]
    temporal_schedule = record["properties"]["deployments"][0]["temporalObservingSchedule"]

    assert len(schedules) == 1
    assert temporal_schedule == {
        "observingSchedule": [schedules[0]["uid"]],
        "dates": ["2020-01-01"],
    }


def test_facility_environment_wraps_environmental_histories() -> None:
    facility = _minimal_facility() | {
        "climateZone": {
            "climateZone": "http://codes.wmo.int/wmdr/ClimateZone/Cfb",
            "beginPosition": "1980-01-01T00:00:00Z",
        },
        "surfaceCover": {
            "surfaceCover": "http://codes.wmo.int/wmdr/SurfaceCover/urbanBuiltup",
            "beginPosition": "1981-01-01T00:00:00Z",
        },
        "populationDensity": {
            "populationDensity": [100, 200],
            "beginPosition": "1990-01-01T00:00:00Z",
        },
        "localTopography": {
            "value": "flat",
            "beginPosition": "1970-01-01T00:00:00Z",
        },
    }

    record = module.build_facility_feature(
        facility,
        [],
        [],
        {},
        source_name="20200101_0-TEST",
    )

    props = record["properties"]
    environment = props["environment"]

    assert "temporalClimateZone" not in props
    assert "temporalSurfaceCover" not in props
    assert "localTopography" not in props

    assert environment["temporalClimateZone"] == {
        "climateZone": ["Cfb"],
        "dates": ["1980-01-01"],
    }
    assert environment["temporalSurfaceCover"] == {
        "surfaceCover": ["urbanBuiltup"],
        "dates": ["1981-01-01"],
    }
    assert environment["temporalPopulation"] == {
        "populationDensity": [[100, 200]],
        "dates": ["1990-01-01"],
    }
    assert environment["temporalTopographyBathymetry"] == {
        "topographyBathymetry": [{"localTopography": {"value": "flat"}}],
        "dates": ["1970-01-01"],
    }
