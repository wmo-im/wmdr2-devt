from __future__ import annotations

from typing import Any

import pytest

import convert_wmdr10_json_to_wmdr2_json as converter
from schema_registry import validator_for_def


def _validator():
    return validator_for_def("wmdr2-common.schema.json", "temporalGeometry")


def test_converter_emits_aligned_temporal_geometry_methods() -> None:
    method = "http://codes.wmo.int/wmdr/GeopositioningMethod/gps"
    entries = converter._facility_temporal_geometry_entries(
        {
            "geospatialLocation": {
                "geoLocation": "46.0 7.0 100",
                "beginPosition": "2020-01-01",
                "geopositioningMethod": method,
            },
            "geospatialLocationHistory": [
                {"geoLocation": "46.1 7.1 101", "beginPosition": "2021-01-01"},
            ],
        }
    )
    temporal_geometry = converter._temporal_geometry_extension(entries)

    assert temporal_geometry == {
        "type": "MovingPoint",
        "coordinates": [[7.0, 46.0, 100], [7.1, 46.1, 101]],
        "dates": ["2020-01-01", "2021-01-01"],
        "methods": [[{"id": method}], []],
    }
    _validator().validate(temporal_geometry)


@pytest.mark.parametrize(
    "methods",
    [
        ["gps"],
        [[], "gps"],
        [[123]],
        [[{"id": "gps"}]],
    ],
)
def test_temporal_geometry_schema_rejects_invalid_methods(methods: Any) -> None:
    payload = {
        "type": "MovingPoint",
        "coordinates": [[7.0, 46.0], [7.1, 46.1]],
        "dates": ["2020-01-01", "2021-01-01"],
        "methods": methods,
    }
    assert list(_validator().iter_errors(payload))


def test_clean_none_preserves_empty_temporal_geometry_method_slots() -> None:
    payload = {
        "temporalGeometry": {
            "methods": [
                [{"id": "http://codes.wmo.int/wmdr/GeopositioningMethod/gps"}],
                [],
            ]
        }
    }
    assert converter._clean_none(payload) == payload


def test_temporal_geometry_uses_open_marker_for_missing_date() -> None:
    temporal_geometry = converter._temporal_geometry_extension(
        [
            {"coordinates": [6, 45], "date": "2020-01-01"},
            {"coordinates": [7, 46], "date": None},
        ]
    )
    assert temporal_geometry is not None
    assert temporal_geometry["dates"] == ["2020-01-01", ".."]
