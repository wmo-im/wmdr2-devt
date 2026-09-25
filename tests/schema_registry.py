from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"
OFFICIAL_SCHEMA_PATH = SCHEMA_DIR / "official" / "wmdr2-bundled.json"
WCMP_SCHEMA_PATH = SCHEMA_DIR / "official" / "wcmpRecordGeoJSON.yaml"
OFFICIAL_COMMIT = "d30f13c3be6395466a778360c4a4ef59625be6af"
OFFICIAL_SCHEMA_URI = (
    "https://raw.githubusercontent.com/wmo-im/wmdr2/"
    f"{OFFICIAL_COMMIT}/schemas/wmdr2-bundled.json"
)
WCMP_COMMIT = "f05037aa8d8bf5911a44a511b7b99a0be009c9ab"
WCMP_SCHEMA_URI = (
    "https://raw.githubusercontent.com/wmo-im/wcmp2/"
    f"{WCMP_COMMIT}/schemas/wcmpRecordGeoJSON.yaml"
)


def load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def schema_registry() -> Registry:
    if not OFFICIAL_SCHEMA_PATH.exists():
        raise RuntimeError(
            f"Missing {OFFICIAL_SCHEMA_PATH}. Run "
            "`python schemas/sync_official_wmdr2_schema.py` first."
        )

    registry = Registry()

    official = json.loads(OFFICIAL_SCHEMA_PATH.read_text(encoding="utf-8"))
    official_resource = Resource.from_contents(
        official,
        default_specification=DRAFT202012,
    )
    registry = registry.with_resource(OFFICIAL_SCHEMA_URI, official_resource)

    official_id = official.get("$id")
    if isinstance(official_id, str):
        registry = registry.with_resource(official_id, official_resource)

    if not WCMP_SCHEMA_PATH.exists():
        raise RuntimeError(
            f"Missing {WCMP_SCHEMA_PATH}. Run "
            "`python schemas/sync_official_wmdr2_schema.py` first."
        )
    wcmp = yaml.safe_load(WCMP_SCHEMA_PATH.read_text(encoding="utf-8"))
    wcmp_resource = Resource.from_contents(
        wcmp,
        default_specification=DRAFT202012,
    )
    registry = registry.with_resource(WCMP_SCHEMA_URI, wcmp_resource)
    wcmp_id = wcmp.get("$id")
    if isinstance(wcmp_id, str):
        registry = registry.with_resource(wcmp_id, wcmp_resource)

    for path in sorted(SCHEMA_DIR.glob("wmdr2-*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resource = Resource.from_contents(
            schema,
            default_specification=DRAFT202012,
        )
        schema_id = schema.get("$id")
        if not isinstance(schema_id, str):
            raise RuntimeError(f"{path} has no $id")
        registry = registry.with_resource(schema_id, resource)

    return registry


def validator_for_schema(name: str) -> Draft202012Validator:
    schema = load_schema(name)
    return Draft202012Validator(
        schema,
        registry=schema_registry(),
        format_checker=FormatChecker(),
    )


def validator_for_def(name: str, def_name: str) -> Draft202012Validator:
    schema = load_schema(name)
    wrapper = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://wmo-im.github.io/wmdr2-devt/tests/{name}-{def_name}.json",
        "$ref": f"{schema['$id']}#/$defs/{def_name}",
    }
    return Draft202012Validator(
        wrapper,
        registry=schema_registry(),
        format_checker=FormatChecker(),
    )
