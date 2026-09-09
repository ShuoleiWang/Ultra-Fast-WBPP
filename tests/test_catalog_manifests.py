from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
CATALOGS = ROOT / "resources" / "catalogs"


def test_catalog_examples_validate() -> None:
    schema = json.loads(
        (CATALOGS / "catalog-manifest-v1.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    examples = sorted(
        path
        for path in CATALOGS.glob("*.json")
        if not path.name.endswith(".schema.json")
    )
    assert examples
    for path in examples:
        validator.validate(json.loads(path.read_text(encoding="utf-8")))


def test_download_url_requires_hash_and_size() -> None:
    schema = json.loads(
        (CATALOGS / "catalog-manifest-v1.schema.json").read_text(encoding="utf-8")
    )
    value = json.loads(
        (CATALOGS / "astap-external-v1.json").read_text(encoding="utf-8")
    )
    value["artifacts"][0]["url"] = "https://example.invalid/catalog.bin"
    errors = list(Draft202012Validator(schema).iter_errors(value))
    assert any("sha256" in str(error) and "sizeBytes" in str(error) for error in errors)
