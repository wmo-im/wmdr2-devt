from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import convert_wmdr10_json_to_wmdr2_json as converter  # noqa: E402


def _environment_schema() -> dict:
    common = json.loads((ROOT / "schemas" / "wmdr2-common.schema.json").read_text(encoding="utf-8"))
    return {
        "$schema": common.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
        "$defs": common["$defs"],
        **common["$defs"]["environment"],
    }


def test_environment_converter_uses_temporal_population_and_topography_bathymetry() -> None:
    facility = {
        "population": [
            {
                "population": [12000, None],
                "perimeter_km": [10, 50],
                "beginPosition": "2020-01-01",
                "endPosition": None,
            }
        ],
        "topographyBathymetry": {
            "localTopography": "slope",
            "relativeElevation": "middle",
            "topographicContext": "rises",
            "altitudeOrDepth": "veryHighAltitude",
        },
    }

    environment = converter._normalize_environment(facility)

    assert environment == {
        "temporalPopulation": [
            {
                "population": [12000.0, None],
                "perimeter_km": [10.0, 50.0],
                "dates": ["2020-01-01", ".."],
            }
        ],
        "topographyBathymetry": {
            "localTopography": "slope",
            "relativeElevation": "middle",
            "topographicContext": "rises",
            "altitudeOrDepth": "veryHighAltitude",
        },
    }

    assert "temporalPopulationDensities" not in environment
    assert "temporalLocalTopography" not in environment
    assert "temporalRelativeElevation" not in environment
    assert "temporalTopographicContext" not in environment
    assert "temporalAltitudeOrDepth" not in environment


def test_environment_converter_defaults_population_perimeters_and_allows_unknown_second_population() -> None:
    environment = converter._normalize_environment(
        {
            "population": [
                {
                    "population": "12000",
                    "beginPosition": "2020-01-01",
                }
            ]
        }
    )

    assert environment == {
        "temporalPopulation": [
            {
                "population": [12000.0, None],
                "perimeter_km": [10.0, 50.0],
                "dates": ["2020-01-01", ".."],
            }
        ]
    }


def test_environment_schema_accepts_new_population_and_topography_shape() -> None:
    payload = {
        "temporalPopulation": [
            {
                "population": [12000, None],
                "perimeter_km": [10, 50],
                "dates": ["2020-01-01", ".."],
            }
        ],
        "topographyBathymetry": {
            "localTopography": "slope",
            "relativeElevation": "middle",
            "topographicContext": "rises",
            "altitudeOrDepth": "veryHighAltitude",
        },
    }

    Draft202012Validator(_environment_schema()).validate(payload)


@pytest.mark.parametrize(
    "obsolete_name",
    [
        "temporalPopulationDensities",
        "temporalLocalTopography",
        "temporalRelativeElevation",
        "temporalTopographicContext",
        "temporalAltitudeOrDepth",
        "temporalTopographyBathymetry",
    ],
)
def test_environment_schema_rejects_obsolete_environment_names(obsolete_name: str) -> None:
    with pytest.raises(ValidationError):
        Draft202012Validator(_environment_schema()).validate({obsolete_name: []})
