from __future__ import annotations

import json
from pathlib import Path

import pytest

import convert_wmdr10_json_to_wmdr2_json as converter


def test_main_without_arguments_uses_config_yaml_and_reports_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "record.json").write_text(
        json.dumps({"facility": {"identifier": "wsi:0-TEST", "name": "Test"}}), encoding="utf-8"
    )
    (tmp_path / "config.yaml").write_text(
        "\n".join(
            [
                "convert_wmdr10_json_to_wmdr2_json:",
                "  source: source",
                "  target: target",
                "  recursive: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert converter.main([]) == 0

    out = capsys.readouterr().out
    assert "wrote" in out
    assert "record.json" in out
    payload = json.loads((target / "record.json").read_text(encoding="utf-8"))
    assert payload["id"] == "0-TEST"


def test_main_writes_catalogues_when_enabled(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    catalogue_records = tmp_path / "catalogue_records"
    catalogues = tmp_path / "catalogues"
    source.mkdir()
    (source / "record.json").write_text(
        json.dumps(
            {
                "facility": {
                    "identifier": "0-TEST",
                    "name": "Test",
                    "contact": {
                        "organisationName": "Example Org",
                        "contactInfo": {"address": {"electronicMailAddress": "ops@example.org"}},
                        "role": "owner",
                    },
                },
                "observationSeries": [
                    {
                        "observedProperty": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006",
                        "deployments": [{"manufacturer": "Maker", "model": "Model", "beginPosition": "2020-01-01"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "convert_wmdr10_json_to_wmdr2_json:",
                f"  source: {source.as_posix()}",
                f"  target: {target.as_posix()}",
                "  catalogues:",
                "    enabled: true",
                f"    records_path: {catalogue_records.as_posix()}",
                f"    contacts_path: {(catalogues / 'contacts.json').as_posix()}",
                f"    instruments_path: {(catalogues / 'instruments.json').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    converter.main(["--config", str(config)])

    embedded = json.loads((target / "record.json").read_text(encoding="utf-8"))
    catalogue_record = json.loads((catalogue_records / "record.json").read_text(encoding="utf-8"))
    contacts = json.loads((catalogues / "contacts.json").read_text(encoding="utf-8"))["contacts"]
    instruments = json.loads((catalogues / "instruments.json").read_text(encoding="utf-8"))["instruments"]

    assert embedded["properties"]["contacts"]
    assert embedded["properties"]["instruments"]
    assert "contacts" not in catalogue_record["properties"]
    assert "instruments" not in catalogue_record["properties"]
    assert contacts[0]["identifier"] == "contact:ops@example.org"
    assert instruments[0]["id"].startswith("instrument:")


def test_main_accepts_source_target_aliases(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "in"
    target = tmp_path / "out"
    source.mkdir()
    (source / "record.json").write_text(
        json.dumps({"facility": {"identifier": "0-TEST", "name": "Test"}}),
        encoding="utf-8",
    )

    assert converter.main(["--source", str(source), "--target", str(target)]) == 0

    assert (target / "record.json").exists()
    assert "wrote" in capsys.readouterr().out
