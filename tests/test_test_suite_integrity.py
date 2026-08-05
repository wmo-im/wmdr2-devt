from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def test_no_duplicate_test_function_names_in_same_file() -> None:
    """Avoid silent test shadowing by later definitions in the same module."""
    duplicates: list[str] = []

    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        seen: dict[str, list[int]] = defaultdict(list)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                seen[node.name].append(node.lineno)
        for name, lines in seen.items():
            if len(lines) > 1:
                rel = path.relative_to(ROOT)
                duplicates.append(f"{rel}:{name} defined at lines {', '.join(map(str, lines))}")

    assert not duplicates, "Duplicate test function names shadow earlier tests:\n" + "\n".join(duplicates)
