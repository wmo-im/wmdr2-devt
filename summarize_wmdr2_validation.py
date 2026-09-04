#!/usr/bin/env python3
"""Summarize current E2E schema-validation failures by file and error signature.

Run after pytest has generated or from a small wrapper that points this at a
directory of converted WMDR2 JSON files:

    python tools/summarize_wmdr2_validation.py <directory>

This is diagnostic only; it does not modify source data or tests.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def normalized_path(error) -> str:
    parts = list(error.path)
    # Collapse array indices so repeated occurrences group together.
    out = []
    for part in parts:
        out.append("*" if isinstance(part, int) else str(part))
    return "/".join(out) or "<root>"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_wmdr2_validation.py <wmdr2-json-directory>")

    directory = Path(sys.argv[1])
    schema_path = Path("schemas/wmdr2-record-feature.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    by_file = defaultdict(Counter)
    overall = Counter()

    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for error in validator.iter_errors(payload):
            signature = (normalized_path(error), error.message)
            by_file[path.name][signature] += 1
            overall[signature] += 1

    print("=== Overall validation signatures ===")
    for (path, message), count in overall.most_common():
        print(f"{count:4d}  {path}: {message}")

    print("\n=== By file ===")
    for filename, counts in sorted(by_file.items()):
        print(f"\n{filename}")
        for (path, message), count in counts.most_common():
            print(f"  {count:3d}  {path}: {message}")

    return 1 if overall else 0


if __name__ == "__main__":
    raise SystemExit(main())
