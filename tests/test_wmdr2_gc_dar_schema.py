from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convert_wmdr2_json_to_wmdr2_gc_dar import convert_record_to_gc_dar


def _load_schema(root: Path) -> dict[str, Any]:
    return json.loads(
        (root / "schemas" / "wmdr2-gc-dar-record.schema.json").read_text(encoding="utf-8")
    )


def _load_or_build_example(root: Path) -> dict[str, Any]:
    """Return a GC-DAR example even when the examples fixture was not copied.

    Some development setups copy only tests and schemas from a patch.  Falling
    back to a generated minimal record keeps the schema tests focused on schema
    behaviour instead of failing with FileNotFoundError.
    """
    example_path = root / "examples" / "wmdr2_gc_dar_example.json"
    if example_path.exists():
        return json.loads(example_path.read_text(encoding="utf-8"))

    full_example_path = root / "examples" / "wmdr2_full_minimal_example.json"
    if full_example_path.exists():
        full_example = json.loads(full_example_path.read_text(encoding="utf-8"))
    else:
        full_example = {
            "type": "Feature",
            "id": "facility:test",
            "geometry": {"type": "Point", "coordinates": [7.0, 46.0, 500.0]},
            "time": {"interval": ["2020-01-01", ".."]},
            "conformsTo": ["http://wigos.wmo.int/spec/wmdr/2/conf/core"],
            "properties": {
                "type": "facility",
                "profile": "wmdr2-full",
                "modelVersion": "0.2.5",
                "title": "Test facility",
                "summary": {"territory": "CHE"},
                "observationSeries": [
                    {
                        "id": "observationSeries:12006",
                        "time": {"interval": ["2020-01-01", ".."]},
                        "observedProperty": 12006,
                        "observedGeometry": "point",
                        "observedFeature": {"domain": "atmosphere"},
                        "programAffiliation": ["GOSGeneral"],
                    }
                ],
            },
        }

    return convert_record_to_gc_dar(full_example, source_filename="generated-test-fixture.json")


def test_example_validates_against_gc_dar_schema():
    root = Path(__file__).resolve().parents[1]
    schema = _load_schema(root)
    example = _load_or_build_example(root)

    jsonschema.Draft202012Validator(schema).validate(example)


def test_schema_rejects_non_dar_profile():
    root = Path(__file__).resolve().parents[1]
    schema = _load_schema(root)
    example = _load_or_build_example(root)
    example["properties"]["profile"] = "wmdr2-full"

    validator = jsonschema.Draft202012Validator(schema)
    errors = list(validator.iter_errors(example))
    assert errors
