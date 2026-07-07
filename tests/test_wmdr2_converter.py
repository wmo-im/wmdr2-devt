from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Any

import pytest

import convert_wmdr10_json_to_wmdr2_json as module

OBSERVED_12006 = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
OBSERVED_263 = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/263"
GEOMETRY_POINT = "http://codes.wmo.int/wmdr/Geometry/point"


def _minimal_facility() -> dict[str, Any]:
    return {
        "identifier": "0-TEST",
        "name": "Test facility",
        "geospatialLocation": {
            "geoLocation": "46 7 100",
            "beginPosition": "2021-01-01",
            "geopositioningMethod": "http://codes.wmo.int/wmdr/GeopositioningMethod/gps",
        },
    }


def _observation_with_deployment() -> dict[str, Any]:
    return {
        "observedProperty": OBSERVED_263,
        "observedDomain": {"domain": "atmosphere"},
        "programAffiliation": ["http://codes.wmo.int/wmdr/ProgramAffiliation/GAWregional"],
        "deployments": [
            {
                "id": "dep1",
                "beginPosition": "2020-01-01T00:00:00Z",
                "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/106",
                "instrumentOperatingStatus": "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/operational",
                "sourceOfObservation": "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading",
                "localReferenceSurface": "http://codes.wmo.int/wmdr/ReferenceSurface/ground",
                "location": "on the roof mast",
                "heightAboveLocalReferenceSurface": {"@uom": "m", "#text": "2.0"},
                "exposure": "http://codes.wmo.int/wmdr/Exposure/class1",
                "manufacturer": "Maker",
                "model": "Model",
                "serialNumber": "SN-001",
            }
        ],
    }


def test_build_facility_feature_uses_v030_model_without_deployment_class() -> None:
    record = module.build_facility_feature(
        _minimal_facility(), [_observation_with_deployment()], [], {"dateStamp": "2026-07-06"}, source_name="20260706_0-TEST"
    )
    props = record["properties"]
    assert "deployments" not in props
    obs = props["observationSeries"][0]
    cfg = obs["observingConfigurations"][0]
    assert cfg["validFrom"] == "2020-01-01"
    assert "deployment" not in cfg
    assert cfg["observingLocation"]["relativeLocation"] == "on the roof mast"
    assert cfg["observingLocation"]["verticalDistanceFromReferenceSurface"] == {"value": 2.0, "uom": "m"}
    assert cfg["observingMethod"] == 106
    assert cfg["operatingStatus"] == "operational"
    assert cfg["sourceOfObservation"] == "automaticReading"


def test_observing_configuration_unknown_method_is_nil_reason() -> None:
    obs = _observation_with_deployment()
    obs["deployments"][0]["observingMethod"] = "unknown"
    record = module.build_facility_feature(_minimal_facility(), [obs], [], {}, source_name="0-TEST")
    cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
    assert cfg["observingMethod"] == {"nilReason": "unknown"}


def test_method_only_observation_gets_valid_from_and_nil_source_status() -> None:
    record = module.build_facility_feature(
        _minimal_facility(), [{"observedProperty": OBSERVED_12006, "beginPosition": "2020-01-02"}], [], {}, source_name="0-TEST"
    )
    cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
    assert cfg["validFrom"] == "2020-01-02"
    assert cfg["observingMethod"] == {"nilReason": "unknown"}
    assert cfg["operatingStatus"] == {"nilReason": "unknown"}
    assert cfg["sourceOfObservation"] == {"nilReason": "unknown"}


def test_observing_procedure_keeps_valid_from_and_schedule_refs() -> None:
    obs = _observation_with_deployment()
    obs["coverage"] = {"beginPosition": "2020-01-01", "temporalSamplingInterval": "PT10M", "diurnalBaseTime": "06:00"}
    record = module.build_facility_feature(_minimal_facility(), [obs], [], {}, source_name="0-TEST")
    series = record["properties"]["observationSeries"][0]
    procedure = series["observingProcedures"][0]
    assert procedure["validFrom"] == "2020-01-01"
    assert procedure["strategy"] == "continuous"
    assert procedure["observingSchedules"]
    schedule = record["properties"]["schedules"][0]
    assert schedule["wmo.int:diurnalBaseTime"] == "06:00:00"


def test_reporting_is_inline_combined_reporting_procedure() -> None:
    reporting = [
        {
            "beginPosition": "2020-01-01",
            "reporting": {
                "internationalExchange": "true",
                "temporalReportingInterval": "PT1H",
                "spatialReportingInterval": "point",
                "uom": "http://codes.wmo.int/wmdr/unit/mm",
                "timeliness": "PT30M",
                "diurnalBaseTime": "00:00",
            },
        }
    ]
    registry: dict[str, dict[str, Any]] = {}
    normalized = module._normalize_observation_reporting(reporting, reporting_registry=registry)
    assert registry == {}
    assert normalized is not None
    assert normalized[0]["internationalExchange"] is True
    assert normalized[0]["temporalReportingInterval"] == "PT1H"
    assert normalized[0]["spatialReportingInterval"] == "point"
    assert normalized[0]["uom"] == "mm"
    assert "validFrom" not in normalized[0]
    assert normalized[0]["reportingSchedules"][0]["wmo.int:diurnalBaseTime"] == "00:00:00"


def test_reporting_schedules_can_be_reused_from_schedule_registry() -> None:
    reporting = [
        {
            "beginPosition": "2020-01-01",
            "reporting": {
                "internationalExchange": "true",
                "temporalReportingInterval": "PT1H",
                "spatialReportingInterval": "point",
                "diurnalBaseTime": "00:00",
            },
        },
        {
            "beginPosition": "2020-01-01",
            "reporting": {
                "internationalExchange": "true",
                "temporalReportingInterval": "PT1H",
                "spatialReportingInterval": "point",
                "diurnalBaseTime": "00:00",
            },
        },
    ]
    registry: dict[str, dict[str, Any]] = {}
    normalized = module._normalize_observation_reporting(reporting, schedule_registry=registry)
    assert normalized is not None
    assert len(registry) == 1
    uid = next(iter(registry))
    assert normalized[0]["reportingSchedules"] == [uid]
    assert registry[uid]["wmo.int:diurnalBaseTime"] == "00:00:00"


def test_facility_level_reusable_reporting_is_not_emitted() -> None:
    obs = _observation_with_deployment()
    obs["reportingProcedures"] = {"internationalExchange": "true", "temporalReportingInterval": "PT1H"}
    record = module.build_facility_feature(_minimal_facility(), [obs], [], {}, source_name="0-TEST")
    assert "reporting" not in record["properties"]
    assert record["properties"]["observationSeries"][0]["reportingProcedures"][0]["internationalExchange"] is True


def test_convert_payload_rejects_non_wmdr10_feature_input() -> None:
    converted = module.convert_payload({"type": "Feature", "id": "wsi:0-TEST", "properties": {"title": "Legacy"}})
    assert converted["id"].startswith("wsi:")
    assert converted["properties"]["title"] != "Legacy"


def test_environment_program_affiliation_and_territory_use_valid_from() -> None:
    facility = _minimal_facility() | {
        "temporalClimateZone": {"climateZone": "http://codes.wmo.int/wmdr/ClimateZone/Cfb", "beginPosition": "2020-01-01"},
        "programAffiliation": {"programAffiliation": "GAW", "beginPosition": "2020-01-01", "reportingStatus": "operational"},
        "territory": {"territory": "CHE", "beginPosition": "2020-01-01"},
    }
    record = module.build_facility_feature(facility, [], [], {}, source_name="0-TEST")
    assert record["properties"]["environment"][0]["validFrom"] == "2020-01-01"
    assert record["properties"]["programAffiliations"][0]["validFrom"] == "2020-01-01"
    assert record["properties"]["territory"][0]["validFrom"] == "2020-01-01"


def test_temporal_normalizers_keep_public_api() -> None:
    assert module._normalize_temporal_climate_zone([
        {"climateZone": "http://codes.wmo.int/wmdr/ClimateZone/Cfb", "beginPosition": "2020-01-01T00:00:00Z"},
        "http://codes.wmo.int/wmdr/ClimateZone/Af",
    ]) == [{"validFrom": "2020-01-01", "climateZone": "Cfb"}, {"validFrom": "..", "climateZone": "Af"}]
    assert module._normalize_temporal_surface_cover({
        "surfaceCover": "http://codes.wmo.int/wmdr/SurfaceCover/grassland",
        "surfaceClassification": {"href": "http://codes.wmo.int/wmdr/SurfaceClassification/local"},
        "beginPosition": "2020-01-01",
    }) == [{"validFrom": "2020-01-01", "surfaceCover": "grassland", "surfaceClassification": "local"}]
    assert module._normalize_temporal_surface_roughness({
        "surfaceRoughness": "http://codes.wmo.int/wmdr/SurfaceRoughness/rough", "beginPosition": "1991-01-01"
    }) == [{"validFrom": "1991-01-01", "surfaceRoughness": "rough"}]


def test_population_normalizer_uses_v030_valid_from() -> None:
    assert module._normalize_temporal_population({"population": "100", "perimeter_km": [10, 50], "beginPosition": "1990-01-01T00:00:00Z"}) == [
        {"validFrom": "1990-01-01", "population": [100.0, None], "perimeter_km": [10.0, 50.0]}
    ]


def test_official_status_still_uses_date_attribute_from_model() -> None:
    assert module._normalize_temporal_official_status({"officialStatus": True, "beginPosition": "2020-01-01"}) == [
        {"date": "2020-01-01", "officialStatus": "primary"}
    ]
    assert module._normalize_temporal_official_status({"officialStatus": False, "beginPosition": "2020-01-01"}) == [
        {"date": "2020-01-01", "officialStatus": "additional"}
    ]


def test_instrument_catalogue_is_type_metadata_only() -> None:
    instrument = module._normalize_instrument(
        {"manufacturer": "Maker", "model": "Model", "instrumentTitle": "Title", "serialNumber": "SN-001"},
        facility_id="0-TEST",
    )
    assert instrument is not None
    assert instrument["uid"] == "instrument:Maker--Model"
    assert "serialNumber" not in instrument


def test_vertical_range_alone_can_create_instrument_type() -> None:
    instrument = module._normalize_instrument({"verticalRangeMin": 10, "verticalRangeMax": 500}, facility_id="0-TEST")
    assert instrument is not None
    assert instrument["verticalRange"] == {"min": 10.0, "max": 500.0}


def test_instrument_observed_property_and_geometry_helpers() -> None:
    assert module._normalize_instrument_observed_property(
        {"observedProperty": [OBSERVED_12006, {"href": OBSERVED_12006}, {"description": "locally defined aerosol metric"}]}
    ) == [12006, "locally defined aerosol metric"]
    assert module._normalize_instrument_observed_geometry({"observedGeometry": GEOMETRY_POINT}) == "point"


def test_facility_set_helpers_are_still_available() -> None:
    assert module._facility_set_refs(["GAW", {"facilitySet": "GBON"}, "GAW"]) == ["facilitySet:GAW", "facilitySet:GBON"]
    assert module.facility_set_catalog_entry("GAW", description="Global Atmosphere Watch") == {
        "uid": "facilitySet:GAW",
        "title": "GAW",
        "description": "Global Atmosphere Watch",
    }


def test_convert_payload_accepts_split_payload_shape() -> None:
    converted = module.convert_payload(
        {"header": {"dateStamp": "2020-01-02"}, "facility": _minimal_facility(), "observationSeries": [_observation_with_deployment()], "deployments": []},
        source_name="20200102_0-TEST",
    )
    assert converted["id"] == "wsi:0-TEST"
    assert converted["properties"]["created"] == "2020-01-02T00:00:00Z"


def test_convert_file_writes_json_output(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    target = tmp_path / "out"
    source.write_text(json.dumps({"facility": _minimal_facility(), "observationSeries": []}), encoding="utf-8")
    output = module.convert_file(source, target)
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8"))["id"] == "wsi:0-TEST"


def test_load_code_list_labels_reads_csv_mapping(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text("domain,code,label\nObservedVariableAtmosphere,12006,Air temperature\n", encoding="utf-8")
    assert module._load_code_list_labels({"codeListLabels": {"files": [str(labels)]}}, base_dir=tmp_path) == {
        "ObservedVariableAtmosphere": {"12006": "Air temperature"}
    }


def test_catalogues_source_config_key_is_obsolete(tmp_path: Path) -> None:
    source = tmp_path / "wmdr10"
    target = tmp_path / "wmdr2"
    source.mkdir()
    (source / "record.json").write_text(json.dumps({"facility": _minimal_facility()}), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join([
            "convert_wmdr10_json_to_wmdr2_json:",
            f"  source: {source.as_posix()}",
            f"  target: {target.as_posix()}",
            "  catalogues:",
            "    enabled: true",
            "    source: old-derived-source",
        ]),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="catalogues.source is obsolete"):
        module.main(["--config", str(config)])
