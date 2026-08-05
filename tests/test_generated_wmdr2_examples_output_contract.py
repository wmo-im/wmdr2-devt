from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "results" / "wmdr2_json_examples"
OBSOLETE_OUTPUT_KEYS = {
    "observing" + "Location",
    "deployment",
    "deployments",
    "application" + "Area",
    "valid" + "From",
    "valid" + "To",
    "begin" + "Position",
    "end" + "Position",
    "surface" + "CoverClassification",
}


def _walk_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


@pytest.mark.skipif(not EXAMPLES.exists(), reason="generated WMDR2 example directory not present")
def test_generated_wmdr2_examples_do_not_emit_obsolete_model_keys() -> None:
    failures: list[str] = []

    for path in sorted(EXAMPLES.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        emitted = {key for mapping in _walk_mappings(record) for key in mapping.keys()}
        obsolete = sorted(emitted & OBSOLETE_OUTPUT_KEYS)
        if obsolete:
            failures.append(f"{path.relative_to(ROOT)}: {', '.join(obsolete)}")

    assert not failures, "Obsolete WMDR2 output keys found:\n" + "\n".join(failures)
