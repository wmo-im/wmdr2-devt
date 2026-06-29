import json
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convert_wmdr10_json_to_wmdr2_json import (  # noqa: E402
    _clean_none,
    _facility_temporal_geometry_entries,
    _temporal_geometry_extension,
)

SCHEMAS = ROOT / "schemas"


def _temporal_geometry_validator() -> Draft202012Validator:
    common_schema = json.loads((SCHEMAS / "wmdr2-common.schema.json").read_text(encoding="utf-8"))
    schema = {
        "$schema": common_schema.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
        "$defs": common_schema["$defs"],
        **common_schema["$defs"]["temporalGeometry"],
    }
    return Draft202012Validator(schema)


def _assert_valid_temporal_geometry(payload: dict[str, Any]) -> None:
    errors = sorted(_temporal_geometry_validator().iter_errors(payload), key=lambda error: list(error.path))
    assert not errors, "\n".join(error.message for error in errors)


def test_temporal_geometry_schema_accepts_aligned_methods_lists() -> None:
    _assert_valid_temporal_geometry(
        {
            "type": "MovingPoint",
            "coordinates": [[7.0, 46.0, 100], [7.1, 46.1, 101], [7.2, 46.2, 102]],
            "dates": ["2020-01-01", "2021-01-01", "2022-01-01"],
            "methods": [[], ["gps"], ["gps", "surveyed"]],
        }
    )


@pytest.mark.parametrize(
    "methods",
    [
        ["gps", "surveyed"],
        [[], "gps"],
        [["gps"], [123]],
        [[""]],
    ],
)
def test_temporal_geometry_schema_rejects_non_list_of_list_methods(methods: Any) -> None:
    payload = {
        "type": "MovingPoint",
        "coordinates": [[7.0, 46.0], [7.1, 46.1]],
        "dates": ["2020-01-01", "2021-01-01"],
        "methods": methods,
    }
    errors = list(_temporal_geometry_validator().iter_errors(payload))
    assert errors


def test_converter_emits_aligned_methods_from_geopositioning_method() -> None:
    facility = {
        "geospatialLocation": {
            "geoLocation": "46.0 7.0 100",
            "date": "2020-01-01",
            "geopositioningMethod": "http://codes.wmo.int/wmdr/GeopositioningMethod/gps",
        },
        "geospatialLocationHistory": [
            {
                "geoLocation": "46.1 7.1 101",
                "date": "2021-01-01",
                "geopositioningMethod": [
                    "http://codes.wmo.int/wmdr/GeopositioningMethod/gps",
                    {"href": "http://codes.wmo.int/wmdr/GeopositioningMethod/surveyed"},
                ],
            },
            {
                "geoLocation": "46.2 7.2 102",
                "date": "2022-01-01",
            },
        ],
    }

    entries = _facility_temporal_geometry_entries(facility)
    temporal_geometry = _temporal_geometry_extension(entries)

    assert temporal_geometry is not None
    assert temporal_geometry["dates"] == ["2020-01-01", "2021-01-01", "2022-01-01"]
    assert temporal_geometry["methods"] == [["gps"], ["gps", "surveyed"], []]


def test_clean_none_preserves_empty_temporal_geometry_methods_slots() -> None:
    payload = {"temporalGeometry": {"methods": [["gps"], []]}}
    assert _clean_none(payload) == {"temporalGeometry": {"methods": [["gps"], []]}}


def test_clean_none_does_not_preserve_empty_lists_for_unrelated_methods_keys() -> None:
    payload = {"properties": {"methods": [["abc"], []]}}
    assert _clean_none(payload) == {"properties": {"methods": [["abc"]]}}


def test_converter_omits_methods_when_absent() -> None:
    facility = {
        "geospatialLocation": {"geoLocation": "46.0 7.0 100", "date": "2020-01-01"},
        "geospatialLocationHistory": [
            {"geoLocation": "46.1 7.1 101", "date": "2021-01-01"},
        ],
    }

    temporal_geometry = _temporal_geometry_extension(_facility_temporal_geometry_entries(facility))

    assert temporal_geometry is not None
    assert "methods" not in temporal_geometry
