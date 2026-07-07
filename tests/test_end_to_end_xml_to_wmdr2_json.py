from __future__ import annotations

import csv
import json
import importlib.util
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


ROOT = Path(__file__).resolve().parents[1]
XML_CONVERTER = ROOT / "convert_wmdr10_xml_to_wmdr10_json.py"
JSON_CONVERTER = ROOT / "convert_wmdr10_json_to_wmdr2_json.py"
XML_SOURCE_DIR = ROOT / "resources" / "wmdr10_xml_examples"
SCHEMA_DIR = ROOT / "schemas"
REPORT_DIR = Path(os.environ.get("WMDR2_E2E_REPORT_DIR", ROOT / "reports"))

# Source XML examples that are not marked with the legacy ``<!--Invalid xml:``
# marker but still lack mandatory history anchors for WMDR2 v0.3.1.  These
# examples must remain schema-invalid; the converter must not invent
# ``validFrom`` dates from neighbouring facility/program metadata.
KNOWN_SCHEMA_INVALID_XML_STEMS: dict[str, str] = {
    "20200304_0-20000-0-06494": (
        "source observations are missing required begin dates for "
        "observingConfiguration/observingProcedure histories"
    ),
    "20250529_0-410-0-22184": (
        "source observations are missing required begin dates for "
        "observingConfiguration histories"
    ),
}


def _load_repo_module(path: Path, module_name: str) -> Any:
    """Load a repository script as a module so pytest-cov can trace it.

    The E2E test intentionally calls converter ``main()`` functions directly
    instead of using subprocesses.  This keeps converter execution inside the
    pytest process, so normal ``pytest --cov`` runs measure the converter code.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Cannot import module {module_name!r} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_converter_main(module: Any, argv: list[str]) -> None:
    """Run a converter main function and report SystemExit cleanly."""
    try:
        module.main(argv)
    except SystemExit as exc:
        code = exc.code
        if code not in (None, 0):
            pytest.fail(f"Converter {module.__name__}.main({argv!r}) exited with {code!r}")




def _xml_declares_invalid(xml_path: Path) -> bool:
    try:
        head = xml_path.read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        return False
    return "<!--Invalid xml:" in head or "<!--Invalid XML:" in head


def _xml_expected_schema_invalid(xml_path: Path) -> bool:
    """Return True only when the WMDR2 output is expected to fail schema validation.

    A legacy ``<!--Invalid xml:`` marker means the source XML has known XML-level
    problems.  It does not necessarily mean that the subset converted into
    WMDR2 must be schema-invalid.  Keep those source-quality markers separate
    from fixtures that are expected to fail the WMDR2 JSON schema.
    """
    return xml_path.stem in KNOWN_SCHEMA_INVALID_XML_STEMS


def _xml_declares_source_invalid_but_schema_may_validate(xml_path: Path) -> bool:
    return _xml_declares_invalid(xml_path) and not _xml_expected_schema_invalid(xml_path)


def _xml_invalid_reason(xml_path: Path) -> str:
    return KNOWN_SCHEMA_INVALID_XML_STEMS.get(xml_path.stem, "source XML is expected to be schema-invalid")


def _display_path(path: Path) -> str:
    """Return a stable, readable path for repo and temporary files."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


COMMON_SCHEMA_FILE = "wmdr2-common.schema.json"
RECORD_SCHEMA_FILE = "wmdr2-record-feature.schema.json"

# XML/GML/parser bookkeeping that should not be treated as WMDR content loss.
IGNORED_XML_ATTRS = {
    "schemaLocation",
    "noNamespaceSchemaLocation",
    "nil",
    "nilReason",
    "type",  # xsi:type / gml type wrappers
}

# Only keep XML/GML ids when the element itself is a referenceable WMDR entity.
REFERENCEABLE_ID_TAGS = {
    "deployment",
    "Deployment",
    "equipment",
    "Equipment",
    "instrument",
    "Instrument",
    "contact",
    "Contact",
    "responsibleParty",
    "ResponsibleParty",
    "recordOwner",
    "RecordOwner",
}

# Whole values that are known to be structural or explicitly not carried into WMDR2.
IGNORED_VALUE_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^gmxCodelists\.xml#CI_RoleCode$"),
    re.compile(r"^https?://www\.w3\.org/"),
    re.compile(r"^https?://www\.opengis\.net/"),
    re.compile(r"^https?://standards\.iso\.org/"),
]


@dataclass(frozen=True)
class XmlEntry:
    xml_file: str
    section: str
    path: str
    kind: str
    value: str
    candidates: tuple[str, ...] = field(default_factory=tuple)


def _load_schema(filename: str) -> dict[str, Any]:
    path = SCHEMA_DIR / filename
    if not path.exists():
        pytest.fail(f"Schema file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def record_validator() -> Draft202012Validator:
    common = _load_schema(COMMON_SCHEMA_FILE)
    record = _load_schema(RECORD_SCHEMA_FILE)
    registry = (
        Registry()
        .with_resource(
            common["$id"],
            Resource.from_contents(common, default_specification=DRAFT202012),
        )
        .with_resource(
            record["$id"],
            Resource.from_contents(record, default_specification=DRAFT202012),
        )
    )
    return Draft202012Validator(record, registry=registry)


@pytest.fixture(scope="session")
def e2e_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Run the two-stage XML → WMDR10 JSON → WMDR2 JSON conversion."""
    if not XML_CONVERTER.exists():
        pytest.fail(f"Missing XML converter: {XML_CONVERTER}")
    if not JSON_CONVERTER.exists():
        pytest.fail(f"Missing WMDR2 converter: {JSON_CONVERTER}")
    if not XML_SOURCE_DIR.exists():
        pytest.fail(f"Missing XML source directory: {XML_SOURCE_DIR}")

    work = tmp_path_factory.mktemp("xml_to_wmdr2_e2e")
    wmdr10_dir = work / "wmdr10_json"
    wmdr2_dir = work / "wmdr2_json"

    xml_converter = _load_repo_module(XML_CONVERTER, "_wmdr2_e2e_xml_converter")
    json_converter = _load_repo_module(JSON_CONVERTER, "_wmdr2_e2e_json_converter")

    _run_converter_main(
        xml_converter,
        [
            "--config",
            str(ROOT / "config.yaml"),
            "--source",
            str(XML_SOURCE_DIR),
            "--target",
            str(wmdr10_dir),
        ],
    )

    # The XML converter now writes only full-record WMDR10 JSON files.  The
    # second-stage converter can therefore consume this directory directly.
    xml_stems = {xml_path.stem for xml_path in XML_SOURCE_DIR.glob("*.xml")}
    json_outputs = sorted(wmdr10_dir.glob("*.json"))
    json_stems = {json_path.stem for json_path in json_outputs}
    missing = sorted(xml_stems - json_stems)
    extra = sorted(json_stems - xml_stems)
    if missing:
        pytest.fail(f"Missing full WMDR10 JSON outputs for XML stem(s): {missing}")
    if extra:
        pytest.fail(
            "XML converter wrote unexpected JSON files. Partial exports should "
            f"not be produced anymore: {extra}"
        )

    _run_converter_main(
        json_converter,
        [
            "--config",
            str(ROOT / "config.yaml"),
            "--source",
            str(wmdr10_dir),
            "--target",
            str(wmdr2_dir),
        ],
    )

    return {
        "work": work,
        "wmdr10": wmdr10_dir,
        "wmdr2": wmdr2_dir,
    }


def test_end_to_end_conversion_writes_one_wmdr2_record_per_xml(
    e2e_outputs: dict[str, Path],
) -> None:
    xml_files = sorted(XML_SOURCE_DIR.glob("*.xml"))
    assert xml_files, f"No XML examples found under {XML_SOURCE_DIR}"

    for xml_path in xml_files:
        expected = e2e_outputs["wmdr2"] / f"{xml_path.stem}.json"
        assert expected.exists(), f"No WMDR2 JSON output written for {xml_path.name}: {expected}"


def test_end_to_end_wmdr2_records_validate_against_schema(
    e2e_outputs: dict[str, Path],
    record_validator: Draft202012Validator,
) -> None:
    outputs = [
        e2e_outputs["wmdr2"] / f"{xml_path.stem}.json"
        for xml_path in sorted(XML_SOURCE_DIR.glob("*.xml"))
    ]
    assert outputs, f"No XML examples found under {XML_SOURCE_DIR}"

    for output in outputs:
        xml_path = XML_SOURCE_DIR / f"{output.stem}.xml"
        payload = json.loads(output.read_text(encoding="utf-8"))
        errors = sorted(record_validator.iter_errors(payload), key=lambda err: list(err.path))
        if _xml_expected_schema_invalid(xml_path):
            reason = _xml_invalid_reason(xml_path)
            assert errors, (
                "Known schema-invalid source unexpectedly produced schema-valid WMDR2 JSON: "
                f"{_display_path(xml_path)} ({reason})"
            )
        elif _xml_declares_source_invalid_but_schema_may_validate(xml_path):
            # The XML source itself is flagged as invalid, but the subset that
            # survives conversion may still satisfy the WMDR2 JSON schema.  Both
            # outcomes are acceptable here; source-level validity is not the same
            # as target-schema validity.
            continue
        else:
            assert not errors, _format_schema_errors(output, errors)


def test_xml_to_wmdr2_semantic_content_loss_report_is_written(
    e2e_outputs: dict[str, Path],
) -> None:
    """Write a report of XML leaf values not found in the WMDR2 JSON output.

    This test is intentionally documentary by default. It fails only if report
    generation itself fails. To make unreviewed loss fail CI, run with:

        WMDR2_FAIL_ON_UNREVIEWED_LOSS=1 pytest -q tests/test_end_to_end_xml_to_wmdr2_json.py
    """
    report = build_loss_report(XML_SOURCE_DIR, e2e_outputs["wmdr2"])
    report_paths = write_loss_report(report, REPORT_DIR)

    assert report_paths["json"].exists()
    assert report_paths["csv"].exists()
    assert report_paths["summary_csv"].exists()
    assert report_paths["summary_md"].exists()
    assert report["summary"]["xml_files"] > 0

    if os.environ.get("WMDR2_FAIL_ON_UNREVIEWED_LOSS") == "1":
        assert report["summary"]["lost_entries"] == 0, (
            f"XML content loss detected. See {report_paths['json']} and {report_paths['csv']}"
        )


def build_loss_report(xml_dir: Path, wmdr2_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_entries = 0
    total_matched = 0
    total_lost = 0
    total_ignored = 0

    for xml_path in sorted(xml_dir.glob("*.xml")):
        json_path = wmdr2_dir / f"{xml_path.stem}.json"
        json_values: set[str] = set()
        if json_path.exists():
            json_values = collect_json_scalar_variants(json.loads(json_path.read_text(encoding="utf-8")))

        entries = collect_xml_entries(xml_path)
        lost: list[dict[str, Any]] = []
        matched = 0
        ignored = 0

        for entry in entries:
            total_entries += 1
            if is_intentionally_ignored(entry):
                ignored += 1
                total_ignored += 1
                continue
            if any(candidate in json_values for candidate in entry.candidates):
                matched += 1
                total_matched += 1
                continue
            lost.append(
                {
                    "section": entry.section,
                    "path": entry.path,
                    "kind": entry.kind,
                    "value": entry.value,
                    "candidateValues": list(entry.candidates),
                }
            )
            total_lost += 1

        files.append(
            {
                "xml": _display_path(xml_path),
                "wmdr2Json": _display_path(json_path) if json_path.exists() else None,
                "summary": {
                    "xmlEntries": len(entries),
                    "matched": matched,
                    "lost": len(lost),
                    "ignored": ignored,
                },
                "lost": lost,
            }
        )

    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Semantic XML leaf values not found in the generated WMDR2 JSON, "
            "after applying intentional-loss and code-list-compaction rules."
        ),
        "notes": [
            "This is a coverage heuristic: it checks whether XML leaf values survive somewhere in the WMDR2 JSON.",
            "It intentionally ignores XML namespaces, schema locations, wrapper ids, nil markers, and generic ISO role codelist references.",
            "Code-list URIs are matched against their compact WMDR2 values, e.g. ObservedVariableAtmosphere/12006 -> 12006.",
            "Date-times are matched against date-resolution WMDR2 values, e.g. 2025-05-28T00:00:00Z -> 2025-05-28.",
        ],
        "summary": {
            "xml_files": len(files),
            "xml_entries": total_entries,
            "matched_entries": total_matched,
            "lost_entries": total_lost,
            "ignored_entries": total_ignored,
        },
        "files": files,
    }


def write_loss_report(report: dict[str, Any], report_dir: Path) -> dict[str, Path]:
    """Write detailed and review-friendly XML-loss reports.

    The JSON file contains the full nested report. The detailed CSV contains
    one row per unmatched XML leaf. The summary CSV/Markdown files group these
    rows by source, section, path template, and value so reviewers can quickly
    see what is probably missing from the WMDR2 model or converter.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "xml_to_wmdr2_loss_report.json"
    csv_path = report_dir / "xml_to_wmdr2_loss_report.csv"
    summary_csv_path = report_dir / "xml_to_wmdr2_loss_summary.csv"
    summary_md_path = report_dir / "xml_to_wmdr2_loss_summary.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    detailed_rows: list[dict[str, Any]] = []
    for file_report in report["files"]:
        for lost in file_report["lost"]:
            detailed_rows.append(
                {
                    "xml": file_report["xml"],
                    "section": lost["section"],
                    "path": lost["path"],
                    "pathTemplate": _path_template(lost["path"]),
                    "kind": lost["kind"],
                    "value": lost["value"],
                    "candidateValues": " | ".join(lost["candidateValues"]),
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "xml",
                "section",
                "path",
                "pathTemplate",
                "kind",
                "value",
                "candidateValues",
            ],
        )
        writer.writeheader()
        writer.writerows(detailed_rows)

    grouped_rows = _group_loss_rows(detailed_rows)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "xml",
                "section",
                "pathTemplate",
                "kind",
                "value",
                "count",
                "examplePath",
                "candidateValues",
            ],
        )
        writer.writeheader()
        writer.writerows(grouped_rows)

    summary_md_path.write_text(_format_loss_summary_markdown(report, grouped_rows), encoding="utf-8")

    return {
        "json": json_path,
        "csv": csv_path,
        "summary_csv": summary_csv_path,
        "summary_md": summary_md_path,
    }


def _path_template(path: str) -> str:
    """Collapse repeated XML element indexes to make review groups readable."""
    return re.sub(r"\[\d+\]", "[]", path)


def _group_loss_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group detailed loss rows into reviewable unique loss signatures."""
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["xml"]),
            str(row["section"]),
            str(row["pathTemplate"]),
            str(row["kind"]),
            str(row["value"]),
        )
        item = grouped.setdefault(
            key,
            {
                "xml": row["xml"],
                "section": row["section"],
                "pathTemplate": row["pathTemplate"],
                "kind": row["kind"],
                "value": row["value"],
                "count": 0,
                "examplePath": row["path"],
                "candidateValues": row["candidateValues"],
            },
        )
        item["count"] += 1
    return sorted(
        grouped.values(),
        key=lambda row: (-int(row["count"]), str(row["xml"]), str(row["section"]), str(row["pathTemplate"])),
    )


def _format_loss_summary_markdown(report: dict[str, Any], grouped_rows: list[dict[str, Any]]) -> str:
    summary = report["summary"]
    lines = [
        "# XML to WMDR2 semantic content-loss summary",
        "",
        "This report is generated by `tests/test_end_to_end_xml_to_wmdr2_json.py`.",
        "It is a review aid, not a formal proof of lossless conversion.",
        "",
        "## Overall counts",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| XML files | {summary['xml_files']} |",
        f"| XML semantic leaf entries scanned | {summary['xml_entries']} |",
        f"| Matched entries | {summary['matched_entries']} |",
        f"| Ignored intentional/non-semantic entries | {summary['ignored_entries']} |",
        f"| Potentially lost entries | {summary['lost_entries']} |",
        "",
        "## How to interpret the files",
        "",
        "- `xml_to_wmdr2_loss_report.json` is the complete nested machine-readable report.",
        "- `xml_to_wmdr2_loss_report.csv` is the detailed row-by-row list of every unmatched XML leaf value.",
        "- `xml_to_wmdr2_loss_summary.csv` groups repeated rows by XML file, section, path template, and value.",
        "- This Markdown file gives the quick human review view.",
        "",
        "Rows in the loss reports mean: an XML text or attribute value was not found anywhere in the generated WMDR2 JSON after applying known transformations such as code-list compaction and date-time to date conversion.",
        "They are *potential losses*, not automatically bugs. Some may be intentional because the WMDR2 core model deliberately drops XML/GML bookkeeping and selected legacy metadata.",
        "",
        "## Per-file summary",
        "",
        "| XML | Scanned | Matched | Ignored | Potentially lost |",
        "|---|---:|---:|---:|---:|",
    ]

    for file_report in report["files"]:
        file_summary = file_report["summary"]
        lines.append(
            "| "
            f"`{file_report['xml']}` | "
            f"{file_summary['xmlEntries']} | "
            f"{file_summary['matched']} | "
            f"{file_summary['ignored']} | "
            f"{file_summary['lost']} |"
        )

    path_counts: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()
    for row in grouped_rows:
        path_counts[str(row["pathTemplate"])] += int(row["count"])
        section_counts[str(row["section"])] += int(row["count"])

    lines.extend(
        [
            "",
            "## Potential losses by section",
            "",
            "| Section | Count |",
            "|---|---:|",
        ]
    )
    for section, count in section_counts.most_common():
        lines.append(f"| {section} | {count} |")

    lines.extend(
        [
            "",
            "## Most frequent unmatched XML paths",
            "",
            "| Count | XML path template |",
            "|---:|---|",
        ]
    )
    for path_template, count in path_counts.most_common(50):
        lines.append(f"| {count} | `{path_template}` |")

    lines.extend(
        [
            "",
            "## Top grouped potential losses",
            "",
            "| Count | XML | Section | Path template | Value |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in grouped_rows[:100]:
        value = str(row["value"]).replace("|", "\\|")
        if len(value) > 160:
            value = value[:157] + "..."
        lines.append(
            "| "
            f"{row['count']} | "
            f"`{row['xml']}` | "
            f"{row['section']} | "
            f"`{row['pathTemplate']}` | "
            f"{value} |"
        )

    lines.extend(
        [
            "",
            "## Review workflow",
            "",
            "1. Start with the per-file counts above to find records with unusually high potential loss.",
            "2. Inspect `xml_to_wmdr2_loss_summary.csv` for grouped repeated losses.",
            "3. Use `examplePath` from the summary CSV or `path` from the detailed CSV to find the source XML element.",
            "4. Decide whether each group is an intentional model reduction, a converter gap, or a schema/model issue.",
            "5. After review, either update the converter/model or add a new intentional-ignore rule to this test with a comment.",
            "",
        ]
    )
    return "\n".join(lines)

def collect_xml_entries(xml_path: Path) -> list[XmlEntry]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    entries: list[XmlEntry] = []

    def walk(element: ET.Element, path: str) -> None:
        tag = local_name(element.tag)

        text = (element.text or "").strip()
        if text:
            entries.append(make_entry(xml_path, classify_section(path), f"{path}/text()", "text", text))

        for raw_attr, raw_value in element.attrib.items():
            attr = local_name(raw_attr)
            value = str(raw_value).strip()
            if not value:
                continue
            if attr in IGNORED_XML_ATTRS:
                continue
            if attr == "id" and tag not in REFERENCEABLE_ID_TAGS:
                continue
            # Keep href, uom, role, and ids on referenceable entities.
            entries.append(make_entry(xml_path, classify_section(path), f"{path}/@{attr}", "attribute", value))

        children = list(element)
        totals = Counter(local_name(child.tag) for child in children)
        seen: Counter[str] = Counter()
        for child in children:
            child_tag = local_name(child.tag)
            seen[child_tag] += 1
            suffix = f"{child_tag}[{seen[child_tag]}]" if totals[child_tag] > 1 else child_tag
            walk(child, f"{path}/{suffix}")

    walk(root, f"/{local_name(root.tag)}")
    return entries


def make_entry(xml_path: Path, section: str, path: str, kind: str, value: str) -> XmlEntry:
    candidates = tuple(sorted(normalize_value_variants(value)))
    return XmlEntry(
        xml_file=_display_path(xml_path),
        section=section,
        path=path,
        kind=kind,
        value=value,
        candidates=candidates,
    )


def collect_json_scalar_variants(obj: Any) -> set[str]:
    values: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                walk(nested)
            return
        if isinstance(value, list):
            for nested in value:
                walk(nested)
            return
        if value is None:
            values.add("null")
            return
        if isinstance(value, bool):
            values.add("true" if value else "false")
            return
        if isinstance(value, (int, float)):
            values.add(str(value))
            return
        if isinstance(value, str):
            values.update(normalize_value_variants(value))
            return
        values.add(str(value))

    walk(obj)
    return values


def normalize_value_variants(value: str) -> set[str]:
    raw = value.strip()
    variants: set[str] = {raw}
    if not raw:
        return variants

    # XML boolean text/attributes become JSON booleans.
    low = raw.lower()
    if low in {"true", "false"}:
        variants.add(low)

    # Date-time and date-with-Z values become date-resolution values in WMDR2.
    date_match = re.match(r"^(\d{4}-\d{2}-\d{2})(?:T.*)?Z?$", raw)
    if date_match:
        variants.add(date_match.group(1))

    # Code-list URI compaction: keep the actual codelist value only.
    if raw.startswith(("http://", "https://")):
        clean = raw.rstrip("/#")
        last = clean.rsplit("/", 1)[-1]
        parent = clean.rsplit("/", 1)[0].rsplit("/", 1)[-1]
        if last:
            variants.add(last)
            variants.add(last.lower())
        if last.isdigit():
            variants.add(str(int(last)))
        if parent.startswith("ObservedVariable"):
            domain = parent.removeprefix("ObservedVariable")
            if domain:
                variants.add(domain[:1].lower() + domain[1:])
                variants.add(domain.lower())

    # Local codelist refs such as gmxCodelists.xml#CI_RoleCode.
    if "#" in raw:
        variants.add(raw.rsplit("#", 1)[-1])

    # Parenthesized unknown values sometimes occur in simplified inputs.
    paren = re.fullmatch(r"\(([^()]+)\)", raw)
    if paren:
        variants.add(paren.group(1).strip())

    return {v for v in variants if v != ""}


def is_intentionally_ignored(entry: XmlEntry) -> bool:
    value = entry.value.strip()
    if any(pattern.match(value) for pattern in IGNORED_VALUE_PATTERNS):
        return True
    if entry.kind == "attribute" and entry.path.endswith("/@id"):
        # Non-referenceable XML ids are skipped during extraction. This branch
        # mainly protects hand-authored intermediate XML variants.
        return True
    return False


def classify_section(path: str) -> str:
    low = path.lower()
    if any(token in low for token in ("deployment", "datageneration", "sampling", "instrument", "equipment")):
        return "deployment"
    if any(token in low for token in ("observation", "observingcapability", "observedproperty", "observedvariable")):
        return "observation"
    if any(token in low for token in ("facility", "geospatial", "territory", "climate", "surfacecover")):
        return "facility"
    if any(token in low for token in ("contact", "responsibleparty", "recordowner")):
        return "contact"
    return "other"


def local_name(name: str) -> str:
    if "}" in name:
        return name.rsplit("}", 1)[-1]
    if ":" in name:
        return name.rsplit(":", 1)[-1]
    return name


def _format_schema_errors(path: Path, errors: Iterable[Any]) -> str:
    lines = [f"Schema validation failed for {path}:"]
    for error in errors:
        instance_path = "/".join(str(part) for part in error.path) or "<root>"
        schema_path = "/".join(str(part) for part in error.schema_path) or "<schema-root>"
        lines.append(f"- instance {instance_path}; schema {schema_path}: {error.message}")
    return "\n".join(lines)
