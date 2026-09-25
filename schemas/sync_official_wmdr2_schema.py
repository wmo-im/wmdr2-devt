#!/usr/bin/env python3
"""Fetch the pinned official WMDR2 bundled schema used by wmdr2-devt."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


OFFICIAL_COMMIT = "d30f13c3be6395466a778360c4a4ef59625be6af"
OFFICIAL_BLOB_SHA1 = "649bdea87f7a641bd009a4ee118f2b5a77ee0234"
OFFICIAL_URL = (
    "https://raw.githubusercontent.com/wmo-im/wmdr2/"
    f"{OFFICIAL_COMMIT}/schemas/wmdr2-bundled.json"
)
OFFICIAL_TARGET = (
    Path(__file__).resolve().parent / "official" / "wmdr2-bundled.json"
)

WCMP_COMMIT = "f05037aa8d8bf5911a44a511b7b99a0be009c9ab"
WCMP_BLOB_SHA1 = "73ac9326613473e2534b48c7b33ab6eb1f1e50b3"
WCMP_URL = (
    "https://raw.githubusercontent.com/wmo-im/wcmp2/"
    f"{WCMP_COMMIT}/schemas/wcmpRecordGeoJSON.yaml"
)
WCMP_TARGET = (
    Path(__file__).resolve().parent / "official" / "wcmpRecordGeoJSON.yaml"
)



def git_blob_sha1(data: bytes) -> str:
    """Return the Git object SHA-1 for file bytes."""
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def verify(data: bytes, expected_blob_sha1: str, label: str) -> None:
    """Verify that bytes are exactly the pinned upstream Git blob."""
    actual = git_blob_sha1(data)
    if actual != expected_blob_sha1:
        raise RuntimeError(
            f"{label} snapshot does not match the pinned Git blob: "
            f"expected {expected_blob_sha1}, got {actual}"
        )


def _read_source(url: str, source: Path | None) -> tuple[bytes, str]:
    if source is None:
        with urlopen(url, timeout=30) as response:
            return response.read(), url
    expanded = source.expanduser()
    return expanded.read_bytes(), str(expanded)


def _sync_one(
    *,
    url: str,
    target: Path,
    expected_blob_sha1: str,
    label: str,
    source: Path | None = None,
) -> Path:
    data, origin = _read_source(url, source)
    verify(data, expected_blob_sha1, label)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(
        f"wrote {target} from {origin}; "
        f"{label} blob {expected_blob_sha1}"
    )
    return target


def sync(
    wmdr2_source: Path | None = None,
    wcmp_source: Path | None = None,
) -> None:
    """Fetch/copy and verify the pinned official schema snapshots."""
    _sync_one(
        url=OFFICIAL_URL,
        target=OFFICIAL_TARGET,
        expected_blob_sha1=OFFICIAL_BLOB_SHA1,
        label=f"wmo-im/wmdr2@{OFFICIAL_COMMIT}",
        source=wmdr2_source,
    )
    _sync_one(
        url=WCMP_URL,
        target=WCMP_TARGET,
        expected_blob_sha1=WCMP_BLOB_SHA1,
        label=f"wmo-im/wcmp2@{WCMP_COMMIT}",
        source=wcmp_source,
    )


def _check_one(
    *,
    target: Path,
    expected_blob_sha1: str,
    label: str,
) -> None:
    if not target.exists():
        raise FileNotFoundError(
            f"{target} is missing; run {Path(__file__).name} without --check"
        )
    verify(target.read_bytes(), expected_blob_sha1, label)
    print(f"verified {target}: {label} blob {expected_blob_sha1}")


def check() -> None:
    """Verify the existing local snapshots."""
    _check_one(
        target=OFFICIAL_TARGET,
        expected_blob_sha1=OFFICIAL_BLOB_SHA1,
        label=f"wmo-im/wmdr2@{OFFICIAL_COMMIT}",
    )
    _check_one(
        target=WCMP_TARGET,
        expected_blob_sha1=WCMP_BLOB_SHA1,
        label=f"wmo-im/wcmp2@{WCMP_COMMIT}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing local snapshot instead of downloading it",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help=(
            "copy the pinned wmdr2-bundled.json from a local file instead of "
            "downloading it; the Git blob is still verified"
        ),
    )
    parser.add_argument(
        "--wcmp-source",
        type=Path,
        help=(
            "copy the pinned WCMP wcmpRecordGeoJSON.yaml from a local file "
            "instead of downloading it; the Git blob is still verified"
        ),
    )
    args = parser.parse_args()
    if args.check and (args.source is not None or args.wcmp_source is not None):
        parser.error("--check cannot be combined with source options")
    check() if args.check else sync(args.source, args.wcmp_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
