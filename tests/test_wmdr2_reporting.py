from __future__ import annotations

from convert_wmdr10_json_to_wmdr2_json import (
    _jscalendar_observing_schedule,
    _normalize_observation_reporting,
    convert_wmdr10_json_to_wmdr2_json,
)


def test_reporting_procedure_is_inline_v030() -> None:
    reporting_registry = {}
    result = _normalize_observation_reporting(
        {
            "reporting": {
                "date": "2024-01-01/..",
                "strategy": "routine",
                "internationalExchange": True,
                "dataPolicy": "wmo",
                "temporalReportingInterval": "PT1H",
                "spatialReportingInterval": "point",
                "timeliness": "near-real-time",
                "uom": "K",
            }
        },
        reporting_registry,
    )

    assert result is not None
    assert reporting_registry == {}
    procedure = result[0]
    assert procedure["date"] == "2024-01-01/.."
    assert procedure["internationalExchange"] is True
    assert procedure["temporalReportingInterval"] == "PT1H"
    assert procedure["spatialReportingInterval"] == "point"
    assert procedure["uom"] == "K"
    assert "reporting" not in procedure
    assert "id" not in procedure
    assert "validFrom" not in procedure


def test_schedule_emits_diurnal_base_time_v030() -> None:
    schedule = _jscalendar_observing_schedule(
        {
            "temporalSamplingInterval": "PT1H",
            "temporalReportingInterval": "PT24H",
            "diurnalBaseTime": "06:00",
        }
    )

    assert schedule is not None
    assert schedule["wmo.int:aggregationInterval"] == "PT24H"
    assert schedule["wmo.int:diurnalBaseTime"] == "06:00:00"
    assert "diurnalBaseTime" not in schedule


def test_converter_does_not_emit_facility_level_reporting() -> None:
    payload = {
        "header": {"fileDateTime": "2026-07-06"},
        "facility": {
            "identifier": "0-20000-0-TEST",
            "name": "Test Facility",
            "geospatialLocation": {"pos": "46.0 7.0 500", "date": "2024-01-01"},
        },
        "deployments": [
            {
                "id": "dep-1",
                "manufacturer": "Acme",
                "model": "Thermometer",
                "serialNumber": "S123",
                "observingMethod": "automaticReading",
            }
        ],
        "observationSeries": [
            {
                "observedProperty": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006",
                "observedGeometry": "point",
                "dataGeneration": {
                    "date": "2024-01-01/..",
                    "reporting": {
                        "strategy": "routine",
                        "internationalExchange": True,
                        "dataPolicy": "noLimitation",
                        "temporalReportingInterval": "PT1H",
                        "spatialReportingInterval": "point",
                        "uom": "m/s",
                        "diurnalBaseTime": "00:00",
                    },
                },
            }
        ],
    }

    out = convert_wmdr10_json_to_wmdr2_json(payload)
    props = out["properties"]
    assert "reporting" not in props
    assert props["observationSeries"][0]["reportingProcedures"][0]["spatialReportingInterval"] == "point"
    assert props["instruments"][0]["uid"].startswith("instrument:")
    assert "serialNumber" not in props["instruments"][0]
    assert props["deployments"][0]["serialNumber"] == "S123"

# v0.3.0 overrides for reporting tests.

def test_reporting_procedure_is_inline_v030() -> None:
    reporting_registry = {}
    result = _normalize_observation_reporting(
        {
            "reporting": {
                "date": "2024-01-01/..",
                "strategy": "routine",
                "internationalExchange": True,
                "dataPolicy": "wmo",
                "temporalReportingInterval": "PT1H",
                "spatialReportingInterval": "point",
                "timeliness": "near-real-time",
                "uom": "K",
            }
        },
        reporting_registry,
    )
    assert result is not None
    assert reporting_registry == {}
    procedure = result[0]
    assert procedure["internationalExchange"] is True
    assert procedure["temporalReportingInterval"] == "PT1H"
    assert procedure["spatialReportingInterval"] == "point"
    assert procedure["uom"] == "K"
    assert "reporting" not in procedure
    assert "id" not in procedure
    assert "validFrom" not in procedure
    assert "date" not in procedure


def test_converter_does_not_emit_facility_level_reporting() -> None:
    payload = {
        "header": {"fileDateTime": "2026-07-06"},
        "facility": {"identifier": "0-20000-0-TEST", "name": "Test Facility", "geospatialLocation": {"pos": "46.0 7.0 500", "date": "2024-01-01"}},
        "deployments": [{"id": "dep-1", "manufacturer": "Acme", "model": "Thermometer", "serialNumber": "S123", "observingMethod": "automaticReading"}],
        "observationSeries": [{"observedProperty": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006", "observedGeometry": "point", "dataGeneration": {"date": "2024-01-01/..", "reporting": {"strategy": "routine", "internationalExchange": True, "dataPolicy": "noLimitation", "temporalReportingInterval": "PT1H", "spatialReportingInterval": "point", "uom": "m/s", "diurnalBaseTime": "00:00"}}}],
    }
    out = convert_wmdr10_json_to_wmdr2_json(payload)
    props = out["properties"]
    assert "reporting" not in props
    assert "deployments" not in props
    assert props["observationSeries"][0]["reportingProcedures"][0]["spatialReportingInterval"] == "point"
    assert props["instruments"][0]["uid"].startswith("instrument:")
    assert "serialNumber" not in props["instruments"][0]
    assert props["observationSeries"][0]["observingConfigurations"][0]["serialNumber"] == "S123"


def test_converter_reuses_reporting_schedules() -> None:
    payload = {
        "header": {"fileDateTime": "2026-07-06"},
        "facility": {
            "identifier": "0-20000-0-TEST",
            "name": "Test Facility",
            "geospatialLocation": {"pos": "46.0 7.0 500", "date": "2024-01-01"},
        },
        "observationSeries": [
            {
                "observedProperty": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006",
                "dataGeneration": {
                    "date": "2024-01-01/..",
                    "reporting": {
                        "internationalExchange": True,
                        "temporalReportingInterval": "PT1H",
                        "spatialReportingInterval": "point",
                        "uom": "m/s",
                        "diurnalBaseTime": "00:00",
                    },
                },
            },
            {
                "observedProperty": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12007",
                "dataGeneration": {
                    "date": "2024-01-01/..",
                    "reporting": {
                        "internationalExchange": True,
                        "temporalReportingInterval": "PT1H",
                        "spatialReportingInterval": "point",
                        "uom": "m/s",
                        "diurnalBaseTime": "00:00",
                    },
                },
            },
        ],
    }
    out = convert_wmdr10_json_to_wmdr2_json(payload)
    schedules = out["properties"]["schedules"]
    assert len([s for s in schedules if s.get("wmo.int:aggregationInterval") == "PT1H"]) == 1
    schedule_ids = {schedule["uid"] for schedule in schedules}
    for series in out["properties"]["observationSeries"]:
        assert series["reportingProcedures"][0]["reportingSchedules"][0] in schedule_ids


def test_schedule_does_not_emit_misspelled_wmi_extension_namespace() -> None:
    schedule = _jscalendar_observing_schedule(
        {
            "temporalSamplingInterval": "PT1H",
            "temporalReportingInterval": "PT24H",
        }
    )
    assert schedule is not None
    assert "wmo.int:samplingFrequency" in schedule
    assert "wmo.int:aggregationInterval" in schedule
    assert all(not key.startswith("wmi.int:") for key in schedule)
