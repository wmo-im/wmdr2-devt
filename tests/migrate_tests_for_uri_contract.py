#!/usr/bin/env python3
"""
Migrate the existing WMDR2 test suite to the URI-preserving tightened contract.

Run from the repository root AFTER installing the tightened schema/converter:

    python tools/migrate_tests_for_uri_contract.py
    pytest -q

The script preserves all existing tests and changes only expectations that
encode the superseded contract:
  * WMO controlled-value URIs contracted to final path segments;
  * observedFeature.domain represented as a bare notation;
  * ObservationSeries.time stored independently;
  * temporalReportingInterval moved into schedule aggregation;
  * temporal-geometry methods contracted to notations.

It also installs the corrected tests/test_schema.py supplied alongside this
script, whose _valid_record() is valid under the tightened schema.

Each changed file gets a one-time .pre-uri-contract backup.
The script is idempotent and reports replacements it cannot find.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Iterable


ROOT = Path.cwd()
TESTS = ROOT / "tests"
SELF_DIR = Path(__file__).resolve().parent

DOMAIN_ATMOSPHERE = "http://codes.wmo.int/wmdr/Domain/atmosphere"
DOMAIN_TERRESTRIAL = "http://codes.wmo.int/wmdr/Domain/terrestrial"
DOMAIN_OCEAN = "http://codes.wmo.int/wmdr/Domain/ocean"


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".pre-uri-contract")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backup: {backup}")


def _function_span(source: str, name: str) -> tuple[int, int]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            start_line = node.lineno
            decorators = getattr(node, "decorator_list", [])
            if decorators:
                start_line = min([start_line] + [item.lineno for item in decorators])
            if node.end_lineno is None:
                raise RuntimeError(f"Cannot determine end of {name}")
            return offsets[start_line - 1], offsets[node.end_lineno]
    raise KeyError(name)


def _patch_function(
    path: Path,
    function_name: str,
    replacements: Iterable[tuple[str, str]],
) -> tuple[int, list[str]]:
    source = path.read_text(encoding="utf-8")
    try:
        start, end = _function_span(source, function_name)
    except KeyError:
        return 0, [f"{path.name}::{function_name}: function not found"]

    block = source[start:end]
    changed = 0
    missing: list[str] = []

    for old, new in replacements:
        if new in block:
            continue
        if old in block:
            block = block.replace(old, new)
            changed += 1
        else:
            missing.append(f"{path.name}::{function_name}: pattern not found: {old[:90]!r}")

    if changed:
        updated = source[:start] + block + source[end:]
        ast.parse(updated)
        _backup(path)
        path.write_text(updated, encoding="utf-8")
    return changed, missing


def _patch_file_literal(path: Path, old: str, new: str) -> tuple[int, str | None]:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return 0, None
    if old not in text:
        return 0, f"{path.name}: pattern not found: {old[:100]!r}"
    updated = text.replace(old, new, 1)
    ast.parse(updated)
    _backup(path)
    path.write_text(updated, encoding="utf-8")
    return 1, None


def patch_converter_helpers() -> tuple[int, list[str]]:
    path = TESTS / "test_converter_helpers.py"
    total = 0
    missing: list[str] = []

    patches = {
        "test_normalize_code_value": [
            (
                '("http://codes.wmo.int/wmdr/unit/mm", "mm")',
                '("http://codes.wmo.int/wmdr/unit/mm", "http://codes.wmo.int/wmdr/unit/mm")',
            ),
            (
                '({"href": "http://codes.wmo.int/wmdr/LevelOfData/level1"}, "level1")',
                '({"href": "http://codes.wmo.int/wmdr/LevelOfData/level1"}, "http://codes.wmo.int/wmdr/LevelOfData/level1")',
            ),
        ],
        "test_compact_wmdr_code_value": [
            (
                '("http://codes.wmo.int/wmdr/unit/mm", "mm")',
                '("http://codes.wmo.int/wmdr/unit/mm", "http://codes.wmo.int/wmdr/unit/mm")',
            ),
            (
                '("https://codes.wmo.int/wmdr/FacilityType/landFixed", "landFixed")',
                '("https://codes.wmo.int/wmdr/FacilityType/landFixed", "https://codes.wmo.int/wmdr/FacilityType/landFixed")',
            ),
            (
                '({"href": "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"}, "GBON")',
                '({"href": "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"}, "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON")',
            ),
        ],
        "test_observed_domain_from_observed_variable": [
            (
                '(OBSERVED_12006, "atmosphere")',
                f'(OBSERVED_12006, "{DOMAIN_ATMOSPHERE}")',
            ),
            (
                '("http://codes.wmo.int/wmdr/ObservedVariableTerrestrial/12", "terrestrial")',
                f'("http://codes.wmo.int/wmdr/ObservedVariableTerrestrial/12", "{DOMAIN_TERRESTRIAL}")',
            ),
            (
                '("http://codes.wmo.int/wmdr/ObservedVariableOcean/1", "ocean")',
                f'("http://codes.wmo.int/wmdr/ObservedVariableOcean/1", "{DOMAIN_OCEAN}")',
            ),
        ],
    }

    for fn, replacements in patches.items():
        changed, miss = _patch_function(path, fn, replacements)
        total += changed
        missing.extend(miss)
    return total, missing


def patch_converter_tests() -> tuple[int, list[str]]:
    path = TESTS / "test_converter.py"
    total = 0
    missing: list[str] = []

    patches: dict[str, list[tuple[str, str]]] = {
        "test_observation_series_preserves_xml_converter_observation_series_metadata": [
            (
                'assert observation["observedGeometry"] == "point"',
                'assert observation["observedGeometry"] == GEOMETRY_POINT',
            ),
            (
                'assert observation["observedProperty"] == "12006"',
                'assert observation["observedProperty"] == OBSERVED_12006',
            ),
            (
                'assert observation["programAffiliations"] == ["GBON", "GOS"]',
                'assert observation["programAffiliations"] == [PROGRAM_GBON, PROGRAM_GOS]',
            ),
            (
                'assert observation["programAffiliations"] == ["GBON", "GOS", "GBON"]',
                'assert observation["programAffiliations"] == [PROGRAM_GBON, PROGRAM_GOS]',
            ),
            (
                'assert observation["applicationAreas"] == ["nowcasting", "aviation"]',
                'assert observation["applicationAreas"] == [APPLICATION_NOWCASTING, APPLICATION_AVIATION]',
            ),
        ],
        "test_numeric_codelist_values_are_emitted_as_strings": [
            (
                'assert observation["observedProperty"] == "12006"',
                'assert observation["observedProperty"] == "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"',
            ),
            (
                'assert observation["observedGeometry"] == "point"',
                'assert observation["observedGeometry"] == "http://codes.wmo.int/wmdr/Geometry/point"',
            ),
            (
                'assert cfg["referenceSurface"] == "localGround"',
                'assert cfg["referenceSurface"] == "http://codes.wmo.int/wmdr/ReferenceSurfaceType/localGround"',
            ),
        ],
        "test_convert_payload_emits_current_v031_shape": [
            (
                'assert observation["observedProperty"] == "12006"',
                'assert observation["observedProperty"] == OBSERVED_12006',
            ),
            (
                'assert observation["observedGeometry"] == "point"',
                'assert observation["observedGeometry"] == GEOMETRY_POINT',
            ),
            (
                'assert observation["observedFeature"] == {"domain": "atmosphere"}',
                f'assert observation["observedFeature"] == {{"domain": "{DOMAIN_ATMOSPHERE}"}}',
            ),
            (
                'assert observation["programAffiliations"] == ["GAWregional", "GBON"]',
                'assert observation["programAffiliations"] == [PROGRAM_GAW_REGIONAL, PROGRAM_GBON]',
            ),
            (
                'assert cfg["observingMethod"] == "266"',
                'assert cfg["observingMethod"] == "http://codes.wmo.int/wmdr/ObservingMethod/266"',
            ),
            (
                'assert cfg["operatingStatus"] == "operational"',
                'assert cfg["operatingStatus"] == "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/operational"',
            ),
            (
                'assert cfg["sourceOfObservation"] == "automaticReading"',
                'assert cfg["sourceOfObservation"] == "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading"',
            ),
            (
                'assert cfg["referenceSurface"] == "localGround"',
                'assert cfg["referenceSurface"] == "http://codes.wmo.int/wmdr/ReferenceSurfaceType/localGround"',
            ),
        ],
        "test_xml_derived_deployment_equipment": [
            (
                'assert observation["observedProperty"] == "572"',
                'assert observation["observedProperty"] == "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/572"',
            ),
            (
                'assert observation["observedGeometry"] == "point"',
                'assert observation["observedGeometry"] == "http://codes.wmo.int/wmdr/Geometry/point"',
            ),
            (
                'assert observation["programAffiliations"] == ["GAW"]',
                'assert observation["programAffiliations"] == ["http://codes.wmo.int/wmdr/ProgramAffiliation/GAW"]',
            ),
            (
                'assert cfg["observingMethod"] == "264"',
                'assert cfg["observingMethod"] == "http://codes.wmo.int/wmdr/ObservingMethodAtmosphere/264"',
            ),
            (
                'assert cfg["sourceOfObservation"] == "automaticReading"',
                'assert cfg["sourceOfObservation"] == "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading"',
            ),
            (
                'assert cfg["operatingStatus"] == "operational"',
                'assert cfg["operatingStatus"] == "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/operational"',
            ),
            (
                'assert cfg["referenceSurface"] == "localGround"',
                'assert cfg["referenceSurface"] == "http://codes.wmo.int/wmdr/ReferenceSurfaceType/localGround"',
            ),
        ],
        "test_normalize_existing_record_converts_all_source_temporal_keys": [
            (
                'assert series["time"] == {"interval": ["2020-02-01", "2023-02-01"]}',
                'assert "time" not in series',
            ),
        ],
        "test_observed_domain_from_observed_variable": [
            (
                '(OBSERVED_12006, "atmosphere")',
                f'(OBSERVED_12006, "{DOMAIN_ATMOSPHERE}")',
            ),
            (
                '("http://codes.wmo.int/wmdr/ObservedVariableTerrestrial/12", "terrestrial")',
                f'("http://codes.wmo.int/wmdr/ObservedVariableTerrestrial/12", "{DOMAIN_TERRESTRIAL}")',
            ),
        ],
        "test_compact_wmdr_code_value_only_compacts_wmdr_urls": [
            (
                '("http://codes.wmo.int/wmdr/unit/mm", "mm")',
                '("http://codes.wmo.int/wmdr/unit/mm", "http://codes.wmo.int/wmdr/unit/mm")',
            ),
            (
                '("https://codes.wmo.int/wmdr/FacilityType/landFixed", "landFixed")',
                '("https://codes.wmo.int/wmdr/FacilityType/landFixed", "https://codes.wmo.int/wmdr/FacilityType/landFixed")',
            ),
        ],
        "test_reporting_procedure_matches_v031_uml_attributes_and_uses_reusable_schedule": [
            (
                'assert "temporalReportingInterval" not in procedure',
                'assert procedure["temporalReportingInterval"] == "PT1H"',
            ),
            (
                'assert schedule["wmo.int:aggregationInterval"] == "PT1H"',
                'assert "wmo.int:aggregationInterval" not in schedule',
            ),
        ],
        "test_diurnal_coverage_sets_dummy_start_duration_and_shared_schedule": [
            (
                'assert schedule["wmo.int:aggregationInterval"] == "PT1H"',
                'assert reporting["temporalReportingInterval"] == "PT1H"\n    assert "wmo.int:aggregationInterval" not in schedule',
            ),
        ],
        "test_program_affiliation_with_explicit_time_is_emitted_as_temporal_object": [
            (
                '{"program": "GBON", "time": {"interval": ["2020-01-01", ".."]}}',
                '{"program": "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON", "time": {"interval": ["2020-01-01", ".."]}}',
            ),
        ],
        "test_facility_environment_is_emitted_as_time_bound_entries": [
            ('"climateZone": "equatorialSavannahDrySummer"', '"climateZone": "http://codes.wmo.int/wmdr/ClimateZone/equatorialSavannahDrySummer"'),
            ('"value": "mosaicForest"', '"value": "http://codes.wmo.int/wmdr/SurfaceCoverGlob2009/mosaicForest"'),
            ('"scheme": "globCover2009"', '"scheme": "http://codes.wmo.int/wmdr/SurfaceCoverClassification/globCover2009"'),
            ('"localTopography": "slope"', '"localTopography": "http://codes.wmo.int/wmdr/LocalTopography/slope"'),
            ('"relativeElevation": "middle"', '"relativeElevation": "http://codes.wmo.int/wmdr/RelativeElevation/middle"'),
            ('"topographicContext": "rises"', '"topographicContext": "http://codes.wmo.int/wmdr/TopographicContext/rises"'),
            ('"altitudeOrDepth": "veryHighAltitude"', '"altitudeOrDepth": "http://codes.wmo.int/wmdr/AltitudeOrDepth/veryHighAltitude"'),
        ],
    }

    for fn, replacements in patches.items():
        changed, miss = _patch_function(path, fn, replacements)
        total += changed
        # Some listed alternatives are intentionally optional. Suppress missing
        # reports when at least one alternative in the same semantic group has
        # already been updated by the current source version.
        missing.extend(miss)
    return total, missing


def patch_temporal_geometry() -> tuple[int, list[str]]:
    path = TESTS / "test_temporal_geometry.py"
    return _patch_function(
        path,
        "test_converter_emits_aligned_temporal_geometry_methods",
        [
            (
                '"methods": [["gps"], []]',
                '"methods": [["http://codes.wmo.int/wmdr/GeopositioningMethod/gps"], []]',
            )
        ],
    )


def patch_mapping_contract() -> tuple[int, list[str]]:
    path = TESTS / "test_wmdr1_to_wmdr2_mapping_contract.py"
    total = 0
    missing: list[str] = []

    patches = {
        "test_mapping_contract_preserves_facility_environment_from_xml_derived_shape": [
            ('"climateZone": "equatorialSavannahDrySummer"', '"climateZone": "http://codes.wmo.int/wmdr/ClimateZone/equatorialSavannahDrySummer"'),
            ('"value": "mosaicForest"', '"value": "http://codes.wmo.int/wmdr/SurfaceCoverGlob2009/mosaicForest"'),
            ('"scheme": "globCover2009"', '"scheme": "http://codes.wmo.int/wmdr/SurfaceCoverClassification/globCover2009"'),
            ('"surfaceRoughness": "rough"', '"surfaceRoughness": "http://codes.wmo.int/wmdr/SurfaceRoughness/rough"'),
            ('"localTopography": "slope"', '"localTopography": "http://codes.wmo.int/wmdr/LocalTopography/slope"'),
            ('"relativeElevation": "middle"', '"relativeElevation": "http://codes.wmo.int/wmdr/RelativeElevation/middle"'),
            ('"topographicContext": "rises"', '"topographicContext": "http://codes.wmo.int/wmdr/TopographicContext/rises"'),
            ('"altitudeOrDepth": "veryHighAltitude"', '"altitudeOrDepth": "http://codes.wmo.int/wmdr/AltitudeOrDepth/veryHighAltitude"'),
        ],
        "test_mapping_contract_preserves_observation_series_metadata_from_xml_derived_shape": [
            ('assert series["observedProperty"] == "12006"', 'assert series["observedProperty"] == OBSERVED_12006'),
            ('assert series["observedGeometry"] == "point"', 'assert series["observedGeometry"] == "http://codes.wmo.int/wmdr/Geometry/point"'),
            ('assert series["observedFeature"] == {"domain": "atmosphere"}', f'assert series["observedFeature"] == {{"domain": "{DOMAIN_ATMOSPHERE}"}}'),
            ('assert series["programAffiliations"] == ["GBON"]', 'assert series["programAffiliations"] == [PROGRAM_GBON]'),
            ('assert series["applicationAreas"] == ["nowcasting", "atmosphericCompositionMonitoring"]', 'assert series["applicationAreas"] == [APPLICATION_NOWCASTING, APPLICATION_ATMOS_COMP]'),
        ],
        "test_mapping_contract_preserves_observing_configuration_from_deployment_equipment": [
            ('assert config["observingMethod"] == "266"', 'assert config["observingMethod"] == OBSERVING_METHOD_266'),
            ('assert config["sourceOfObservation"] == "automaticReading"', 'assert config["sourceOfObservation"] == SOURCE_AUTOMATIC'),
            ('assert config["referenceSurface"] == "localGround"', 'assert config["referenceSurface"] == REFERENCE_LOCAL_GROUND'),
        ],
        "test_mapping_contract_splits_temporal_operating_status_history": [
            (
                'assert [config["operatingStatus"] for config in configs] == ["operational", "inactive"]',
                'assert [config["operatingStatus"] for config in configs] == [\n'
                '        "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/operational",\n'
                '        "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/inactive",\n'
                '    ]',
            ),
        ],
    }

    for fn, replacements in patches.items():
        changed, miss = _patch_function(path, fn, replacements)
        total += changed
        missing.extend(miss)
    return total, missing


def install_schema_test() -> tuple[int, list[str]]:
    source = SELF_DIR / "test_schema.py"
    target = TESTS / "test_schema.py"
    if not source.exists():
        return 0, [f"corrected test_schema.py not found next to migration script: {source}"]
    desired = source.read_text(encoding="utf-8")
    if target.exists() and target.read_text(encoding="utf-8") == desired:
        return 0, []
    _backup(target)
    target.write_text(desired, encoding="utf-8")
    ast.parse(desired)
    return 1, []


def main() -> int:
    required = [
        TESTS / "test_converter.py",
        TESTS / "test_converter_helpers.py",
        TESTS / "test_schema.py",
        TESTS / "test_temporal_geometry.py",
        TESTS / "test_wmdr1_to_wmdr2_mapping_contract.py",
    ]
    missing_files = [path for path in required if not path.exists()]
    if missing_files:
        for path in missing_files:
            print(f"missing: {path}")
        return 2

    total = 0
    warnings: list[str] = []

    for label, action in (
        ("test_converter.py", patch_converter_tests),
        ("test_converter_helpers.py", patch_converter_helpers),
        ("test_schema.py", install_schema_test),
        ("test_temporal_geometry.py", patch_temporal_geometry),
        ("test_wmdr1_to_wmdr2_mapping_contract.py", patch_mapping_contract),
    ):
        changed, miss = action()
        total += changed
        warnings.extend(miss)
        print(f"{label}: {changed} migration edit(s)")

    print(f"\nTotal migration edits: {total}")

    if warnings:
        print("\nPatterns not found (often means your local test already differs):")
        for warning in warnings:
            print(f"  - {warning}")
        print(
            "\nThese are not automatically fatal. Run pytest; remaining failures "
            "will show whether any missing pattern represents a still-stale expectation."
        )

    print("\nNext: pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
