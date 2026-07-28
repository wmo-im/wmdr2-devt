
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validate_wmdr2_examples_pr22.py"
EXAMPLES = ROOT / "results" / "wmdr2_json_examples"
WMDR2_PR22_BUNDLED_SCHEMA=Path("~/Public/git/wmdr2/schemas/wmdr2-bundled.json").expanduser()


# @pytest.mark.pr22_schema
def test_wmdr2_examples_are_compatible_with_pr22_schema() -> None:
    schema = WMDR2_PR22_BUNDLED_SCHEMA
    if not schema:
        pytest.skip("Set WMDR2_PR22_BUNDLED_SCHEMA to run PR-22 schema compatibility validation")

    schema_path = Path(schema)
    if not schema_path.exists():
        pytest.skip(f"PR-22 bundled schema not found: {schema_path}")

    if not EXAMPLES.exists():
        pytest.skip(f"WMDR2 JSON example directory not found: {EXAMPLES}")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--paths",
            str(EXAMPLES),
            "--schema",
            str(schema_path),
            "--allow-known-nonvalidating",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr