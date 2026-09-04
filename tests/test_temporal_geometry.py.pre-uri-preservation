from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import convert_wmdr10_json_to_wmdr2_json as converter

ROOT = Path(__file__).resolve().parents[1]


def _temporal_geometry_validator() -> Draft202012Validator:
    schema = json.loads((ROOT / "schemas" / "wmdr2-record-feature.schema.json").read_text(encoding="utf-8"))
    return Draft202012Validator({"$defs": schema["$defs"], **schema["$defs"]["temporalGeometry"]})


def test_converter_emits_aligned_temporal_geometry_methods() -> None:
    entries = converter._facility_temporal_geometry_entries(
        {
            "geospatialLocation": {
                "geoLocation": "46.0 7.0 100",
                "beginPosition": "2020-01-01",
                "geopositioningMethod": "http://codes.wmo.int/wmdr/GeopositioningMethod/gps",
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
        "methods": [["gps"], []],
    }
    _temporal_geometry_validator().validate(temporal_geometry)


@pytest.mark.parametrize("methods", [["gps"], [[], "gps"], [[123]]])
def test_temporal_geometry_schema_rejects_invalid_methods(methods: Any) -> None:
    payload = {
        "type": "MovingPoint",
        "coordinates": [[7.0, 46.0], [7.1, 46.1]],
        "dates": ["2020-01-01", "2021-01-01"],
        "methods": methods,
    }
    assert list(_temporal_geometry_validator().iter_errors(payload))


def test_clean_none_preserves_empty_temporal_geometry_method_slots() -> None:
    payload = {"temporalGeometry": {"methods": [["gps"], []]}}
    assert converter._clean_none(payload) == payload
