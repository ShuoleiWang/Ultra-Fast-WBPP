"""The byte forms every identity and receipt digest is computed over.

Receipts, installed-catalog sets, selections and correspondence tables are
identified by the SHA-256 of their canonical JSON.  These bytes are part of
the persisted contracts: an installed catalog or a saved selection made by an
earlier build must keep its identity, so the encodings here never change.
"""

from __future__ import annotations

import json
import os
from typing import Any

from lightframeqc.content_hash import file_sha256


def canonical_json(value: Any) -> bytes:
    """Compact, key-sorted UTF-8 JSON that rejects NaN and infinities."""

    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_json_document(value: Any) -> bytes:
    """:func:`canonical_json` with a final newline: how receipts are written
    (their ``receiptId`` hashes exactly these bytes of the receipt core)."""

    return canonical_json(value) + b"\n"


def sha256_digest(path: str | os.PathLike[str]) -> str:
    """``sha256:<hex>`` of a file's bytes."""

    return "sha256:" + file_sha256(path)


__all__ = ["canonical_json", "canonical_json_document", "sha256_digest"]
