from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from schema_registry import validator_for_schema


ROOT = Path(__file__).resolve().parents[1]
XML_CONVERTER = ROOT / "convert_wmdr10_xml_to_wmdr10_json.py"
JSON_CONVERTER = ROOT / "convert_wmdr10_json_to_wmdr2_json.py"
XML_SOURCE_DIR = ROOT / "resources" / "wmdr10_xml_examples"


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def e2e_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    if not XML_CONVERTER.exists() or not XML_SOURCE_DIR.exists():
        pytest.skip("XML converter or XML example resources are not present in this checkout")

    work = tmp_path_factory.mktemp("xml_to_wmdr2")
    wmdr10 = work / "wmdr10_json"
    wmdr2 = work / "wmdr2_json"

    xml_module = _load_module(XML_CONVERTER, "_wmdr2_e2e_xml_converter")
    json_module = _load_module(JSON_CONVERTER, "_wmdr2_e2e_json_converter")

    xml_module.main(
        [
            "--config",
            str(ROOT / "config.yaml"),
            "--source",
            str(XML_SOURCE_DIR),
            "--target",
            str(wmdr10),
        ]
    )
    json_module.main(
        [
            "--config",
            str(ROOT / "config.yaml"),
            "--source",
            str(wmdr10),
            "--target",
            str(wmdr2),
        ]
    )

    return {"wmdr10": wmdr10, "wmdr2": wmdr2}


def test_end_to_end_conversion_writes_one_wmdr2_record_per_xml(
    e2e_outputs: dict[str, Path],
) -> None:
    xml_files = sorted(XML_SOURCE_DIR.glob("*.xml"))
    assert xml_files
    for xml_path in xml_files:
        assert (e2e_outputs["wmdr2"] / f"{xml_path.stem}.json").exists()


def _error_path(error: Any) -> str:
    return "/".join(str(part) for part in error.path) or "<root>"


def test_end_to_end_wmdr2_records_validate_against_schema(
    e2e_outputs: dict[str, Path],
) -> None:
    validator = validator_for_schema("wmdr2-record-feature.schema.json")
    outputs = sorted(e2e_outputs["wmdr2"].glob("*.json"))
    assert outputs

    failures: list[str] = []
    for output in outputs:
        payload = json.loads(output.read_text(encoding="utf-8"))
        errors = sorted(
            validator.iter_errors(payload),
            key=lambda err: (list(err.path), err.message),
        )
        for error in errors:
            failures.append(
                f"{output.name}: {_error_path(error)}: {error.message}"
            )

    assert not failures, "\n".join(failures[:100])
