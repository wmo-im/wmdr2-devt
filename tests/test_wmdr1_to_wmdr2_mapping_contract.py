from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import convert_wmdr10_json_to_wmdr2_json as converter

OBSERVED_12006 = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
OBSERVING_METHOD_266 = "http://codes.wmo.int/wmdr/ObservingMethod/266"
SOURCE_AUTOMATIC = "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading"
REFERENCE_LOCAL_GROUND = "http://codes.wmo.int/wmdr/ReferenceSurfaceType/localGround"
UNIT_M = "http://codes.wmo.int/wmdr/unit/m"
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
    "observationSeries",
    "observingConfigurations",
    "serialNumber",
    "referenceSurface",
    "verticalDistanceFromReferenceSurface",
}


def concept(uri: str) -> dict[str, str]:
    return {"id": uri}


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
            "climateZone": concept(
                "http://codes.wmo.int/wmdr/ClimateZone/equatorialSavannahDrySummer"
            ),
            "surfaceCover": {
                "value": concept(
                    "http://codes.wmo.int/wmdr/SurfaceCoverGlob2009/mosaicForest"
                ),
                "scheme": concept(
                    "http://codes.wmo.int/wmdr/SurfaceCoverClassification/globCover2009"
                ),
            },
            "surfaceRoughness": concept(
                "http://codes.wmo.int/wmdr/SurfaceRoughness/rough"
            ),
            "topographyBathymetry": {
                "localTopography": concept(
                    "http://codes.wmo.int/wmdr/LocalTopography/slope"
                ),
                "relativeElevation": concept(
                    "http://codes.wmo.int/wmdr/RelativeElevation/middle"
                ),
                "topographicContext": concept(
                    "http://codes.wmo.int/wmdr/TopographicContext/rises"
                ),
                "altitudeOrDepth": concept(
                    "http://codes.wmo.int/wmdr/AltitudeOrDepth/veryHighAltitude"
                ),
            },
        }
    ]


def test_mapping_contract_preserves_observation_metadata_from_xml_derived_shape() -> None:
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

    observation = record["properties"]["observations"][0]
    assert observation["observedProperty"] == concept(OBSERVED_12006)
    assert observation["observedGeometry"] == concept(
        "http://codes.wmo.int/wmdr/Geometry/point"
    )
    assert observation["programAffiliations"] == [
        {"programAffiliation": concept(PROGRAM_GBON)}
    ]
    assert observation["applicationAreas"] == [
        concept(APPLICATION_NOWCASTING),
        concept(APPLICATION_ATMOS_COMP),
    ]
    assert "applicationArea" not in observation


def test_mapping_contract_preserves_configuration_from_deployment_equipment() -> None:
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
    config = props["observations"][0]["configurations"][0]
    assert config["id"]
    assert config["time"] == {"interval": ["2020-01-01", "2022-12-31"]}
    assert config["observingMethod"] == concept(OBSERVING_METHOD_266)
    assert config["sourceOfObservation"] == concept(SOURCE_AUTOMATIC)
    assert config["verticalDistance"] == {
        "distances": [2.0],
        "unit": concept(UNIT_M),
        "referenceSurface": concept(REFERENCE_LOCAL_GROUND),
    }
    assert config["instrumentSerialNumber"] == "SN-001"
    assert config["instrument"] == "maker-model"

    assert props["instruments"] == [
        {
            "id": "maker-model",
            "manufacturer": "Maker",
            "model": "Model",
            "observingMethods": [concept(OBSERVING_METHOD_266)],
        }
    ]
    assert "serialNumber" not in props["instruments"][0]


def test_mapping_contract_splits_temporal_operating_status_history() -> None:
    operational = (
        "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/operational"
    )
    inactive = "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/inactive"
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
                                "instrumentOperatingStatus": operational,
                                "beginPosition": "2003-12-01",
                                "endPosition": "2011-05-31",
                            },
                            {
                                "instrumentOperatingStatus": inactive,
                                "beginPosition": "2011-06-01",
                            },
                        ],
                    }
                ],
            }
        ],
    })

    configs = record["properties"]["observations"][0]["configurations"]
    assert [config["operatingStatus"] for config in configs] == [
        concept(operational),
        concept(inactive),
    ]
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
