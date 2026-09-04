from __future__ import annotations

import copy

import convert_wmdr10_json_to_wmdr2_json as converter


OBSERVED = "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
GEOMETRY = "http://codes.wmo.int/wmdr/Geometry/point"
DOMAIN = "http://codes.wmo.int/wmdr/Domain/atmosphere"
PROGRAM = "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"
FACILITY_TYPE = "http://codes.wmo.int/wmdr/FacilityType/landFixed"
WMO_REGION = "http://codes.wmo.int/wmdr/WMORegion/6"
DATA_POLICY = "http://codes.wmo.int/wmdr/DataPolicy/noLimitation"


def test_normalize_code_value_preserves_absolute_wmdr_uri() -> None:
    assert converter._normalize_code_value(OBSERVED) == OBSERVED
    assert converter._normalize_code_value({"href": OBSERVED}) == OBSERVED


def test_legacy_compact_helper_no_longer_contracts_wmdr_uri() -> None:
    assert converter._compact_wmdr_code_value(FACILITY_TYPE) == FACILITY_TYPE
    assert converter._compact_wmdr_code_value({"href": PROGRAM}) == PROGRAM


def test_code_list_array_helper_preserves_program_uri() -> None:
    assert converter._compact_wmdr_code_values(
        {"programAffiliation": PROGRAM},
        "programAffiliation",
    ) == [PROGRAM]


def test_observed_domain_is_emitted_as_domain_concept_uri() -> None:
    assert converter._observed_domain_from_observed_variable(OBSERVED) == DOMAIN
    assert converter._observed_domain_object({"observedProperty": OBSERVED}) == {
        "domain": DOMAIN
    }


def test_observation_series_preserves_controlled_uris_and_has_no_independent_time() -> None:
    series = converter._observation_series_from_source(
        {
            "id": "observationSeries:test",
            "observedProperty": OBSERVED,
            "type": GEOMETRY,
            "programAffiliation": [
                {"programAffiliation": PROGRAM}
            ],
            "beginPosition": "2020-01-01",
        },
        0,
        [],
        {},
        {},
        {},
    )

    assert series["observedProperty"] == OBSERVED
    assert series["observedGeometry"] == GEOMETRY
    assert series["observedFeature"] == {"domain": DOMAIN}
    assert series["programAffiliations"] == [PROGRAM]
    assert "observedDomain" not in series
    assert "time" not in series


def test_readable_observation_title_stays_compact_despite_uri_metadata() -> None:
    title = converter._format_observation_title(OBSERVED, GEOMETRY)
    assert title is not None
    assert "domain: atmosphere" in title
    assert "geometry: point" in title
    assert "variable: 12006" in title
    assert "http://codes.wmo.int/" not in title


def test_reporting_procedure_retains_iso_duration_attributes() -> None:
    proc = converter._reporting_procedure_from_source(
        {
            "reporting": {
                "internationalExchange": False,
                "dataPolicy": DATA_POLICY,
                "temporalReportingInterval": "PT1H",
                "temporalAggregate": "PT10M",
            }
        },
        {},
        {},
    )
    assert proc is not None
    assert proc["internationalExchange"] is False
    assert proc["dataPolicy"] == DATA_POLICY
    assert proc["temporalReportingInterval"] == "PT1H"
    assert proc["temporalAggregate"] == "PT10M"


def test_schedule_structure_does_not_move_reporting_periods_to_schedule() -> None:
    record = {
        "properties": {
            "observationSeries": [
                {
                    "id": "observationSeries:test",
                    "reportingProcedures": [
                        {
                            "internationalExchange": False,
                            "dataPolicy": DATA_POLICY,
                            "temporalReportingInterval": "PT1H",
                            "temporalAggregate": "PT10M",
                        }
                    ],
                }
            ]
        }
    }
    converter._normalize_procedure_schedule_structure(record)
    proc = record["properties"]["observationSeries"][0]["reportingProcedures"][0]
    assert proc["temporalReportingInterval"] == "PT1H"
    assert proc["temporalAggregate"] == "PT10M"
    assert "wmo.int:aggregationInterval" not in proc


def test_finalizer_preserves_controlled_uri() -> None:
    value = {
        "properties": {
            "facilityType": FACILITY_TYPE,
            "observationSeries": [
                {
                    "observedProperty": OBSERVED,
                    "observedFeature": {"domain": DOMAIN},
                }
            ],
        }
    }
    assert converter._finalize_wmdr2_value(value) == value


def test_facility_controlled_values_remain_absolute_uris() -> None:
    record = converter.build_facility_feature(
        {
            "facility": {
                "identifier": "0-20008-0-TST",
                "name": "Test facility",
                "facilityType": FACILITY_TYPE,
                "wmoRegion": WMO_REGION,
                "geospatialLocation": "46 7 100",
            },
            "observationSeries": [],
        },
        source_name="test",
    )
    props = record["properties"]
    assert props["facilityType"] == FACILITY_TYPE
    assert props["wmoRegion"] == WMO_REGION


def test_source_wmdr1_object_is_not_mutated() -> None:
    source = {
        "facility": {
            "identifier": "0-20008-0-TST",
            "name": "Test facility",
            "facilityType": FACILITY_TYPE,
            "wmoRegion": WMO_REGION,
            "geospatialLocation": "46 7 100",
        },
        "observationSeries": [
            {
                "observedProperty": OBSERVED,
                "type": GEOMETRY,
                "programAffiliation": [{"programAffiliation": PROGRAM}],
            }
        ],
    }
    before = copy.deepcopy(source)
    converter.build_facility_feature(source, source_name="test")
    assert source == before


def test_historical_facility_program_affiliation_preserves_uri_and_specific_strings() -> None:
    affiliations = converter._normalize_program_affiliations(
        [
            {
                "programAffiliation": PROGRAM,
                "beginPosition": "2020-01-01",
                "programSpecificFacilityId": "GBON-123",
                "programSpecificFacilityTitle": "Programme-specific station title",
                "reportingStatus": "http://codes.wmo.int/wmdr/ReportingStatus/operational",
            }
        ]
    )
    assert len(affiliations) == 1
    affiliation = affiliations[0]
    assert affiliation["program"] == PROGRAM
    assert "programAffiliation" not in affiliation
    assert affiliation["programSpecificFacilityId"] == "GBON-123"
    assert affiliation["programSpecificFacilityTitle"] == "Programme-specific station title"


def test_converter_does_not_fabricate_missing_operating_status() -> None:
    record = converter.convert_record(
        {
            "facility": {
                "identifier": "0-20000-0-TEST",
                "name": "Test",
                "geospatialLocation": "46 7 500",
            },
            "observationSeries": [
                {
                    "observedProperty": OBSERVED,
                    "type": GEOMETRY,
                    "deployments": [
                        {
                            "beginPosition": "2020-01-01",
                            "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/266",
                            "sourceOfObservation": "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading",
                        }
                    ],
                }
            ],
        }
    )
    cfg = record["properties"]["observationSeries"][0]["observingConfigurations"][0]
    assert "operatingStatus" not in cfg


def test_converter_does_not_fabricate_temporal_aggregate() -> None:
    proc = converter._reporting_procedure_from_source(
        {
            "reporting": {
                "internationalExchange": True,
                "dataPolicy": DATA_POLICY,
                "temporalReportingInterval": "PT1H",
            }
        },
        {},
        {},
    )
    assert proc is not None
    assert proc["temporalReportingInterval"] == "PT1H"
    assert "temporalAggregate" not in proc


def test_optional_controlled_value_omits_nested_nilreason() -> None:
    assert (
        converter._optional_controlled_value(
            {
                "instrumentOperatingStatus": {
                    "nilReason": "unknown"
                }
            },
            "operatingStatus",
            "instrumentOperatingStatus",
        )
        is None
    )


def test_converter_omits_explicit_unknown_optional_operating_status() -> None:
    cfg = converter._observing_configuration_from_source(
        {
            "beginPosition": "2020-01-01",
            "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/266",
            "sourceOfObservation": "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading",
            "instrumentOperatingStatus": {
                "instrumentOperatingStatus": {
                    "nilReason": "unknown"
                }
            },
        },
        {},
        {},
    )

    assert "operatingStatus" not in cfg


def test_converter_preserves_explicit_optional_operating_status_uri() -> None:
    status_uri = (
        "http://codes.wmo.int/wmdr/"
        "InstrumentOperatingStatus/operational"
    )
    cfg = converter._observing_configuration_from_source(
        {
            "beginPosition": "2020-01-01",
            "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/266",
            "sourceOfObservation": "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading",
            "instrumentOperatingStatus": {
                "instrumentOperatingStatus": {
                    "href": status_uri
                }
            },
        },
        {},
        {},
    )

    assert cfg["operatingStatus"] == status_uri

