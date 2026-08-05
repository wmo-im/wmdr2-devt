from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import convert_wmdr10_json_to_wmdr2_json as converter

OBSERVED_12006 = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
OBSERVING_METHOD_266 = "http://codes.wmo.int/wmdr/ObservingMethod/266"
SOURCE_AUTOMATIC = "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading"
REFERENCE_LOCAL_GROUND = "http://codes.wmo.int/wmdr/ReferenceSurfaceType/localGround"
APPLICATION_NOWCASTING = "http://codes.wmo.int/wmdr/ApplicationArea/nowcasting"
APPLICATION_ATMOS_COMP = "http://codes.wmo.int/wmdr/ApplicationArea/atmosphericCompositionMonitoring"
PROGRAM_GBON = "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"


OBSOLETE_OUTPUT_KEYS = {
    "observing" + "Location",
    "deployment",
    "deployments",
    "applicationArea",
    "validFrom",
    "validTo",
    "beginPosition",
    "endPosition",
}


def _walk_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _base_facility(**extra: Any) -> dict[str, Any]:
    facility: dict[str, Any] = {
        "identifier": "0-20000-0-TEST",
        "name": "Contract Test Facility",
        "geospatialLocation": "46 7 500",
        "beginPosition": "2000-01-01",
    }
    facility.update(extra)
    return facility


def test_mapping_contract_preserves_facility_environment_from_xml_derived_shape() -> None:
    record = converter.convert_record({
        "facility": _base_facility(
            climateZone={
                "climateZone": "http://codes.wmo.int/wmdr/ClimateZone/equatorialSavannahDrySummer",
                "beginPosition": "2009-01-06",
            },
            surfaceCover={
                "surfaceCover": "http://codes.wmo.int/wmdr/SurfaceCoverGlob2009/mosaicForest",
                "surfaceCoverClassification": "http://codes.wmo.int/wmdr/SurfaceCoverClassification/globCover2009",
                "beginPosition": "2009-01-06",
            },
            surfaceRoughness={
                "surfaceRoughness": "http://codes.wmo.int/wmdr/SurfaceRoughness/rough",
                "beginPosition": "2009-01-06",
            },
            topographyBathymetry={
                "localTopography": "http://codes.wmo.int/wmdr/LocalTopography/slope",
                "relativeElevation": "http://codes.wmo.int/wmdr/RelativeElevation/middle",
                "topographicContext": "http://codes.wmo.int/wmdr/TopographicContext/rises",
                "altitudeOrDepth": "http://codes.wmo.int/wmdr/AltitudeOrDepth/veryHighAltitude",
                "beginPosition": "2009-01-06",
            },
        )
    })

    assert record["properties"]["environment"] == [
        {
            "time": {"interval": ["2009-01-06", ".."]},
            "climateZone": "equatorialSavannahDrySummer",
            "surfaceCover": {"value": "mosaicForest", "scheme": "globCover2009"},
            "surfaceRoughness": "rough",
            "topographyBathymetry": {
                "localTopography": "slope",
                "relativeElevation": "middle",
                "topographicContext": "rises",
                "altitudeOrDepth": "veryHighAltitude",
            },
        }
    ]


def test_mapping_contract_preserves_observation_series_metadata_from_xml_derived_shape() -> None:
    record = converter.convert_record({
        "facility": _base_facility(),
        "observationSeries": [
            {
                "observedProperty": OBSERVED_12006,
                "type": "http://codes.wmo.int/wmdr/Geometry/point",
                "programAffiliation": {"href": PROGRAM_GBON},
                "deployments": [
                    {
                        "beginPosition": "2020-01-01",
                        "observingMethod": OBSERVING_METHOD_266,
                        "sourceOfObservation": SOURCE_AUTOMATIC,
                        "applicationArea": {"href": APPLICATION_NOWCASTING},
                    },
                    {
                        "beginPosition": "2021-01-01",
                        "observingMethod": OBSERVING_METHOD_266,
                        "sourceOfObservation": SOURCE_AUTOMATIC,
                        "applicationArea": {"applicationArea": APPLICATION_ATMOS_COMP},
                    },
                ],
            }
        ],
    })

    series = record["properties"]["observationSeries"][0]
    assert series["observedProperty"] == "12006"
    assert series["observedGeometry"] == "point"
    assert series["programAffiliations"] == ["GBON"]
    assert series["applicationAreas"] == ["nowcasting", "atmosphericCompositionMonitoring"]
    assert "applicationArea" not in series


def test_mapping_contract_preserves_observing_configuration_from_deployment_equipment() -> None:
    record = converter.convert_record({
        "facility": _base_facility(),
        "observationSeries": [
            {
                "observedProperty": OBSERVED_12006,
                "type": "http://codes.wmo.int/wmdr/Geometry/point",
                "deployments": [
                    {
                        "beginPosition": "2020-01-01",
                        "endPosition": "2022-12-31",
                        "observingMethod": OBSERVING_METHOD_266,
                        "sourceOfObservation": SOURCE_AUTOMATIC,
                        "localReferenceSurface": REFERENCE_LOCAL_GROUND,
                        "heightAboveLocalReferenceSurface": {"@uom": "m", "#text": "2.0"},
                        "manufacturer": "Maker",
                        "model": "Model",
                        "serialNumber": "SN-001",
                    }
                ],
            }
        ],
    })

    props = record["properties"]
    config = props["observationSeries"][0]["observingConfigurations"][0]
    assert config["time"] == {"interval": ["2020-01-01", "2022-12-31"]}
    assert config["observingMethod"] == "266"
    assert config["sourceOfObservation"] == "automaticReading"
    assert config["referenceSurface"] == "localGround"
    assert config["verticalDistanceFromReferenceSurface"] == {"value": 2.0, "uom": "m"}
    assert config["serialNumber"] == "SN-001"
    assert config["instrument"] == "instrument:maker-model"

    assert props["instruments"] == [
        {
            "id": "instrument:maker-model",
            "manufacturer": "Maker",
            "model": "Model",
        }
    ]
    assert "serialNumber" not in props["instruments"][0]


def test_mapping_contract_splits_temporal_operating_status_history() -> None:
    record = converter.convert_record({
        "facility": _base_facility(),
        "observationSeries": [
            {
                "observedProperty": OBSERVED_12006,
                "type": "http://codes.wmo.int/wmdr/Geometry/point",
                "deployments": [
                    {
                        "beginPosition": "2003-12-01",
                        "observingMethod": OBSERVING_METHOD_266,
                        "sourceOfObservation": SOURCE_AUTOMATIC,
                        "instrumentOperatingStatus": [
                            {
                                "instrumentOperatingStatus": "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/operational",
                                "beginPosition": "2003-12-01",
                                "endPosition": "2011-05-31",
                            },
                            {
                                "instrumentOperatingStatus": "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/inactive",
                                "beginPosition": "2011-06-01",
                            },
                        ],
                    }
                ],
            }
        ],
    })

    configs = record["properties"]["observationSeries"][0]["observingConfigurations"]
    assert [config["operatingStatus"] for config in configs] == ["operational", "inactive"]
    assert [config["time"] for config in configs] == [
        {"interval": ["2003-12-01", "2011-05-31"]},
        {"interval": ["2011-06-01", ".."]},
    ]


def test_mapping_contract_does_not_emit_obsolete_output_keys() -> None:
    record = converter.convert_record({
        "facility": _base_facility(),
        "observationSeries": [
            {
                "observedProperty": OBSERVED_12006,
                "type": "http://codes.wmo.int/wmdr/Geometry/point",
                "deployments": [
                    {
                        "beginPosition": "2020-01-01",
                        "observingMethod": OBSERVING_METHOD_266,
                        "sourceOfObservation": SOURCE_AUTOMATIC,
                        "applicationArea": APPLICATION_NOWCASTING,
                    }
                ],
            }
        ],
    })

    emitted_keys = {key for mapping in _walk_mappings(record) for key in mapping.keys()}
    assert emitted_keys.isdisjoint(OBSOLETE_OUTPUT_KEYS)
