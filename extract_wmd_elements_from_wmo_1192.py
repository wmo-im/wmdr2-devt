#!/usr/bin/env python3
"""
Extract WMDS (WIGOS Metadata Standard) element definitions from CHAPTER 7 tables
of WMO-No. 1192 (2019 edition) into a CSV.

author: joerg.klausen@meteoswiss.ch

Config
------
Reads paths + code-table URL mappings from config.yaml:

wmo_1192:
  resource: <path to wmds_1192_en.pdf>
  result: <path to output CSV>

wmdr_code_tables:
  <code-table-id>: <url> | <nested-dict>

Output
------
CSV file with columns:
    category, id, name, definition, code_table, code_table_url, requirement


Notes on code_table_url
-----------------------
- If the extracted code_table is e.g. "1-01" and config has sub-tables under it
  (e.g. 1-01-01, 1-01-02, ...), we output the leaf URLs for those sub-tables.
- If the extracted code_table cell contains multiple IDs (e.g. "11-01;11-02"),
  code_table_url will be a semicolon-joined list of resolved URLs (including any
  sub-table URLs).
- If nothing is resolvable, code_table_url is empty.

Dependencies
------------
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pdfplumber
from utils.config import load_config

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pyyaml\n"
        "Install with: pip install pyyaml\n"
        f"Original error: {e}"
    )


CH7_TITLE = "CHAPTER 7. DETAILED SPECIFICATION OF WIGOS METADATA ELEMENTS"
REF_TITLE = "REFERENCES AND FURTHER READING"

STOPWORDS = {"of", "and", "or", "the", "a", "an", "to", "in", "for", "on", "at", "with"}


def collapse_ws(s: Optional[str]) -> str:
    """Collapse all whitespace (incl. newlines) to single spaces and strip."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def normalize_category(raw: str) -> str:
    """
    Convert headings like "OBSERVED VARIABLE" -> "Observed Variable"
    while keeping stopwords in lowercase (except if first word).
    Also preserves slash-separated parts (e.g., "STATION/PLATFORM").
    """
    raw = collapse_ws(raw).replace("\u00ad", "")  # soft hyphen

    def cap_word(w: str, first: bool) -> str:
        lw = w.lower()
        if (not first) and lw in STOPWORDS:
            return lw
        return lw[:1].upper() + lw[1:]

    parts = []
    for chunk in raw.split("/"):
        words = [w for w in chunk.strip().split(" ") if w]
        out = [cap_word(w, first=(i == 0)) for i, w in enumerate(words)]
        parts.append(" ".join(out))
    return "/".join(parts)


def extract_code_table(cell: Optional[str]) -> str:
    """
    Extract one or more code-table IDs from a cell.
    Returns semicolon-separated unique IDs in appearance order.
    Accepts IDs like 1-01 and sub IDs like 1-01-01.
    """
    if not cell:
        return ""
    toks = re.findall(r"\b\d{1,2}-\d{2}(?:-\d{2})?\b", cell)
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return ";".join(out)


def resolve_path(base_dir: Path, p: str) -> Path:
    """Resolve a path relative to base_dir unless it's already absolute."""
    pp = Path(p)
    return pp if pp.is_absolute() else (base_dir / pp)


def find_ch7_page_range(pdf: Any) -> Tuple[int, int]:
    """
    Return (start_page_index, end_page_index_exclusive) for Chapter 7.
    Uses the last occurrence of Chapter 7 title (skips table of contents).
    """
    ch7_hits = []
    ref_hits = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if CH7_TITLE in text:
            ch7_hits.append(i)
        if REF_TITLE in text:
            ref_hits.append(i)

    if not ch7_hits:
        raise RuntimeError("Could not locate Chapter 7 title in the PDF.")
    if not ref_hits:
        raise RuntimeError("Could not locate References section in the PDF.")

    start = ch7_hits[-1]
    end = ref_hits[-1]
    if end <= start:
        raise RuntimeError(f"Invalid page range detected: start={start}, end={end}")
    return start, end


def category_headings_in_order(text: str) -> List[str]:
    """Return normalized category headings found on the page, in textual order."""
    headings = []
    for m in re.finditer(r"CATEGORY\s+\d+\s*:\s*([A-Z0-9/\- ]+)", text):
        headings.append(normalize_category(m.group(1)))
    return headings


def _leaf_urls(value: Any) -> Iterable[str]:
    """
    Yield all URL strings found in nested structures like:
      - "http://..."
      - {"1-01-01": "http://...", ...}
      - {"1-04": {"1-04": "http://...", "1-04-02": "http://..."}, ...}
    """
    if value is None:
        return
    if isinstance(value, str):
        u = value.strip()
        if u:
            yield u
        return
    if isinstance(value, dict):
        for v in value.values():
            yield from _leaf_urls(v)
        return
    if isinstance(value, list):
        for v in value:
            yield from _leaf_urls(v)
        return
    # ignore other types


def _collect_urls_for_ct_id(ct: str, mapping: Dict[str, Any]) -> List[str]:
    """
    Resolve URLs for a single code-table ID.
    - If mapping[ct] is a string: include it.
    - If mapping[ct] is a dict: include *all leaf URLs* under it.
    - Also include any top-level keys starting with f"{ct}-" (flat sub-table style).
    Deduplicates while preserving order.
    """
    out: List[str] = []
    seen: set[str] = set()

    def add(u: str) -> None:
        if u not in seen:
            seen.add(u)
            out.append(u)

    # exact match
    if ct in mapping:
        for u in _leaf_urls(mapping.get(ct)):
            add(u)

    # flat sub-keys (if present in config)
    prefix = ct + "-"
    for k, v in mapping.items():
        if isinstance(k, str) and k.startswith(prefix):
            for u in _leaf_urls(v):
                add(u)

    return out


def code_table_url_for(code_table: str, mapping: Dict[str, Any]) -> str:
    """
    Map code_table like "3-01" or "11-01;11-02" to a semicolon-joined URL string.
    Uses nested sub-table URLs when present (e.g., 1-01 -> 1-01-01, 1-01-02, ...).
    """
    if not code_table:
        return ""

    urls: List[str] = []
    seen: set[str] = set()

    for ct in [c.strip() for c in code_table.split(";") if c.strip()]:
        for u in _collect_urls_for_ct_id(ct, mapping):
            if u not in seen:
                seen.add(u)
                urls.append(u)

    return ";".join(urls)


def extract_records(pdf_path: Path, code_table_map: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract all Chapter 7 element rows into a list of dict records."""
    records: List[Dict[str, str]] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        start, end = find_ch7_page_range(pdf)
        current_category: Optional[str] = None

        for page_index in range(start, end):
            page = pdf.pages[page_index]
            text = page.extract_text() or ""
            headings = category_headings_in_order(text)

            if headings:
                current_category = headings[0]

            tables = page.extract_tables() or []
            for ti, table in enumerate(tables):
                # Some pages contain 2 categories (e.g., Category 9 and 10)
                table_category = headings[ti] if ti < len(headings) else (current_category or "")

                for row in table:
                    if not row or not row[0]:
                        continue
                    if collapse_ws(row[0]).upper() == "ID":
                        continue

                    rid = collapse_ws(row[0])
                    if not re.match(r"^\d{1,2}-\d{2}$", rid):
                        continue  # not an element row

                    # Expected columns: ID, Name, Definition, Note, Code table, M/C/O
                    name = collapse_ws(row[1] if len(row) > 1 else "")
                    definition = collapse_ws(row[2] if len(row) > 2 else "")
                    code_table = extract_code_table(row[4] if len(row) > 4 else "")
                    requirement = collapse_ws(row[5] if len(row) > 5 else "")
                    code_table_url = code_table_url_for(code_table, code_table_map)

                    records.append(
                        {
                            "category": table_category,
                            "id": rid,
                            "name": name,
                            "definition": definition,
                            "code_table": code_table,
                            "code_table_url": code_table_url,
                            "requirement": requirement,
                        }
                    )

    return records


def write_csv(records: List[Dict[str, str]], out_path: Path) -> None:
    """Write records to CSV with the required header and UTF-8 encoding."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "id",
        "name",
        "definition",
        "code_table",
        "code_table_url",
        "requirement",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to config.yaml (default: script folder/config.yaml)",
    )
    ap.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Override input PDF path (otherwise from config: wmo_1192.resource)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override output CSV path (otherwise from config: wmo_1192.result)",
    )
    args = ap.parse_args()

    cfg_path = args.config
    cfg = load_config(cfg_path)
    base_dir = cfg_path.parent

    wmo_1192 = cfg.get("wmo_1192") or {}
    if not isinstance(wmo_1192, dict):
        raise ValueError("config.yaml: 'wmo_1192' must be a mapping/dict")

    resource = wmo_1192.get("resource")
    result = wmo_1192.get("result")
    if not resource or not result:
        raise ValueError("config.yaml: wmo_1192.resource and wmo_1192.result are required")

    pdf_path = args.pdf.resolve() if args.pdf else resolve_path(base_dir, str(resource)).resolve()
    out_path = args.out.resolve() if args.out else resolve_path(base_dir, str(result)).resolve()

    code_table_map = cfg.get("wmdr_code_tables") or {}
    if not isinstance(code_table_map, dict):
        raise ValueError("config.yaml: 'wmdr_code_tables' must be a mapping/dict")

    records = extract_records(pdf_path, code_table_map)
    write_csv(records, out_path)

    print(f"Wrote {len(records)} rows to: {out_path}")


if __name__ == "__main__":
    main()
