from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest import mock

from lightframeqc import content_hash
from lightframeqc.identity import compute_file_identity


def test_file_is_read_once_per_stat_identity(tmp_path: Path) -> None:
    content_hash.clear_cache()
    path = tmp_path / "frame.bin"
    payload = os.urandom(3 * 1024 * 1024 + 17)
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    real_open = Path.open
    opens: list[str] = []

    def counting_open(self: Path, *args, **kwargs):
        if self == path:
            opens.append(str(self))
        return real_open(self, *args, **kwargs)

    with mock.patch.object(Path, "open", counting_open):
        assert content_hash.file_sha256(path) == expected
        assert content_hash.file_sha256(path) == expected
        # The race-aware identity hasher shares the cache and skips the read too.
        identity = compute_file_identity(path)
    assert identity.sha256 == expected
    assert opens == [str(path)]
    assert content_hash.cache_size() == 1


def test_rewritten_file_is_hashed_again(tmp_path: Path) -> None:
    content_hash.clear_cache()
    path = tmp_path / "frame.bin"
    path.write_bytes(b"first" * 1000)
    first = content_hash.file_sha256(path)
    # Same size, new bytes: ctime (and usually mtime) change, so the cached
    # digest no longer applies.
    path.write_bytes(b"other" * 1000)
    second = content_hash.file_sha256(path)
    assert first != second
    assert second == hashlib.sha256(b"other" * 1000).hexdigest()


def test_identity_hasher_records_its_digest_for_plain_lookups(tmp_path: Path) -> None:
    content_hash.clear_cache()
    path = tmp_path / "frame.bin"
    path.write_bytes(b"identity-first" * 500)
    identity = compute_file_identity(path)
    with mock.patch.object(Path, "open", side_effect=AssertionError("must not read")) as opened:
        assert content_hash.file_sha256(path) == identity.sha256
    assert opened.call_count == 0
