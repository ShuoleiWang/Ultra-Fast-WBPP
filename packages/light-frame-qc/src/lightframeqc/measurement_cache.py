"""Advisory per-frame cache of one complete :class:`FrameMeasurement`.

A blink session and the production run that follows it measure exactly the
same Lights: the same read, the same SEP extraction and the same
native-resolution PSF pass, for the same result.  This cache stores the
whole measurement of one frame under the frame's content identity, the
measurement configuration and a fingerprint of the implementation, so the
second pass reuses it instead of measuring again.

Entries are disposable in the same way as :mod:`lightframeqc.analysis_cache`:
an unavailable fingerprint, an unsupported value, a changed source file and
any cache I/O or validation failure all become a miss and a fresh
measurement.  A value is stored only after the module has proved that
restoring it reproduces the measured object exactly, so a field that does
not survive the canonical encoding disables the entry rather than changing
a measurement.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, fields, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

import numpy as np

from .analysis_cache import (
    _encode,
    _unique_object,
    implementation_fingerprint,
    private_cache_directory,
    prune_cache_directory,
)
from .config import QcConfig
from .models import FileIdentity, FrameMeasurement, FrameMetadata, FrameRole, Star

_SCHEMA = 1
# One entry is dominated by its two star catalogs: about 0.6 MB for a 26 MP
# Light at the default maximum_stars, so the size bound admits roughly 850
# frames, several complete sessions, and the count bound never binds first.
_MAX_ENTRY_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_ENTRIES = 1024
_MODULES = (
    "measurement_cache", "analysis_cache", "measure", "readers", "identity",
    "models", "config", "metadata", "native_psf", "cfa", "fits_bands", "xisf",
    "statistics", "source_extraction",
)
_PACKAGES = ("numpy", "astropy", "sep", "PIL")

_STAR_FLOAT_FIELDS = ("x", "y", "flux", "peak", "a", "b", "theta", "fwhm", "ellipticity")
_STAR_INT_FIELDS = ("flags", "support_pixels", "detection_pixels")
_MEASUREMENT_FIELDS = tuple(
    item.name
    for item in fields(FrameMeasurement)
    if item.name not in {"stars", "raw_stars", "metadata", "identity"}
)
_METADATA_FIELDS = tuple(item.name for item in fields(FrameMetadata))


def _packed_stars(stars: list[Star] | None) -> str | None:
    """Base64 of the exact little-endian float64/int64 image of a star list.

    The same layout as :mod:`lightframeqc.analysis_cache` hashes, kept as
    text so one entry stays a single canonical JSON document.  ``None``
    integers become -1, which no real count or flag takes.
    """

    if stars is None:
        return None
    if not stars:
        return ""
    floats = np.array(
        [[getattr(star, name) for name in _STAR_FLOAT_FIELDS] for star in stars],
        dtype="<f8",
    )
    integers = np.array(
        [[-1 if getattr(star, name) is None else getattr(star, name) for name in _STAR_INT_FIELDS] for star in stars],
        dtype="<i8",
    )
    return base64.b64encode(floats.tobytes(order="C") + integers.tobytes(order="C")).decode("ascii")


def _unpacked_stars(value: Any) -> list[Star] | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Invalid cached star catalog")
    if not value:
        return []
    raw = base64.b64decode(value.encode("ascii"), validate=True)
    width = 8 * (len(_STAR_FLOAT_FIELDS) + len(_STAR_INT_FIELDS))
    if not raw or len(raw) % width:
        raise ValueError("Invalid cached star catalog length")
    count = len(raw) // width
    split = count * 8 * len(_STAR_FLOAT_FIELDS)
    floats = np.frombuffer(raw[:split], dtype="<f8").reshape(count, len(_STAR_FLOAT_FIELDS))
    integers = np.frombuffer(raw[split:], dtype="<i8").reshape(count, len(_STAR_INT_FIELDS))
    stars: list[Star] = []
    for row, counts in zip(floats, integers, strict=True):
        values = {name: float(row[index]) for index, name in enumerate(_STAR_FLOAT_FIELDS)}
        values["flags"] = int(counts[0])
        values["support_pixels"] = None if counts[1] < 0 else int(counts[1])
        values["detection_pixels"] = None if counts[2] < 0 else int(counts[2])
        stars.append(Star(**values))
    return stars


def _measurement_payload(measurement: FrameMeasurement) -> dict[str, Any]:
    metadata = asdict(measurement.metadata)
    metadata["role"] = measurement.metadata.role.value
    metadata["observed_at"] = (
        measurement.metadata.observed_at.isoformat()
        if measurement.metadata.observed_at is not None
        else None
    )
    return {
        "measurement": {name: getattr(measurement, name) for name in _MEASUREMENT_FIELDS},
        "metadata": metadata,
        "identity": None if measurement.identity is None else asdict(measurement.identity),
        "stars": _packed_stars(measurement.stars),
        "rawStars": _packed_stars(measurement.raw_stars),
    }


def _restore(payload: Any) -> FrameMeasurement:
    """Rebuild one measurement; any deviation from the stored shape raises."""

    if not isinstance(payload, dict) or set(payload) != {"measurement", "metadata", "identity", "stars", "rawStars"}:
        raise ValueError("Invalid cached measurement payload")
    values, metadata, identity = payload["measurement"], payload["metadata"], payload["identity"]
    if not isinstance(values, dict) or set(values) != set(_MEASUREMENT_FIELDS):
        raise ValueError("Invalid cached measurement fields")
    if not isinstance(metadata, dict) or set(metadata) != set(_METADATA_FIELDS):
        raise ValueError("Invalid cached metadata fields")
    observed_at = metadata["observed_at"]
    if observed_at is not None and not isinstance(observed_at, str):
        raise ValueError("Invalid cached observation time")
    restored_metadata = FrameMetadata(
        **{name: value for name, value in metadata.items() if name not in {"role", "observed_at"}},
        role=FrameRole(metadata["role"]),
        observed_at=None if observed_at is None else datetime.fromisoformat(observed_at),
    )
    if identity is not None and (
        not isinstance(identity, dict) or set(identity) != {item.name for item in fields(FileIdentity)}
    ):
        raise ValueError("Invalid cached file identity")
    stars = _unpacked_stars(payload["stars"])
    if stars is None:
        raise ValueError("Invalid cached measurement without a star catalog")
    return FrameMeasurement(
        **values,
        metadata=restored_metadata,
        identity=None if identity is None else FileIdentity(**identity),
        stars=stars,
        raw_stars=_unpacked_stars(payload["rawStars"]),
    )


class FrameMeasurementCache:
    """Store and restore complete measurements of individual Light frames.

    ``stats`` is only for a caller that runs in this process; the frame
    workers report their outcome with the measurement instead.
    """

    def __init__(
        self,
        directory: Path | str,
        config: QcConfig,
        *,
        image_index: int = 0,
        max_full_decode_bytes: int | None = None,
        stats: dict[str, int] | None = None,
    ):
        self.directory = Path(directory) / "frame-measurement-v1"
        self.config = config
        self.image_index = image_index
        self.max_full_decode_bytes = max_full_decode_bytes
        self.stats = stats

    def key(self, path: str | os.PathLike[str], identity: FileIdentity) -> str | None:
        """The entry name for one frame's exact bytes under this measurement.

        ``identity`` is the content-addressed identity the measurement has
        already computed, so the digest of the file is part of the key and no
        code, configuration or pixel change can be answered from the cache.
        The resolved pathname is part of it too, because a measurement
        records the file it was made from.  ``make_thumbnails`` is excluded:
        it only chooses whether a preview image is written next to the
        report, never a measured value.
        """

        try:
            config = asdict(self.config)
            config.pop("make_thumbnails")
            return hashlib.sha256(
                _encode([
                    _SCHEMA,
                    identity.sha256,
                    identity.size_bytes,
                    identity.mtime_ns,
                    str(Path(path).expanduser().resolve(strict=True)),
                    self.image_index,
                    self.max_full_decode_bytes,
                    config,
                    implementation_fingerprint(_SCHEMA, _MODULES, _PACKAGES),
                ])
            ).hexdigest()
        except Exception:
            return None

    def load(self, key: str | None) -> FrameMeasurement | None:
        try:
            if key is None:
                raise ValueError("Cache key is unavailable")
            private_cache_directory(self.directory)
            descriptor = os.open(
                self.directory / f"{key}.json",
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            )
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_ENTRY_BYTES or (
                    os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & 0o077)
                ):
                    raise ValueError("Invalid cache file")
                raw = stream.read(_MAX_ENTRY_BYTES + 1)
            if len(raw) > _MAX_ENTRY_BYTES:
                raise ValueError("Oversized cache file")
            envelope = json.loads(raw, object_pairs_hook=_unique_object)
            if (
                not isinstance(envelope, dict)
                or set(envelope) != {"schema", "key", "payload", "sha256"}
                or envelope["schema"] != _SCHEMA
                or envelope["key"] != key
            ):
                raise ValueError("Invalid cache envelope")
            if hashlib.sha256(_encode(envelope["payload"])).hexdigest() != envelope["sha256"]:
                raise ValueError("Cache checksum mismatch")
            measurement = _restore(envelope["payload"])
            if measurement.status != "MEASURED":
                raise ValueError("Only measured frames are cached")
            self._count("hits")
            return measurement
        except Exception:
            self._count("misses")
            return None

    def store(self, key: str | None, measurement: FrameMeasurement) -> bool:
        """Write one measurement; a value that cannot be restored exactly is
        silently not cached, so a later run measures it again."""

        if key is None or measurement.status != "MEASURED" or measurement.identity is None:
            return False
        temporary: str | None = None
        try:
            # The preview image path is a per-run destination, never a
            # measured value; a restored measurement carries whatever the
            # run that reused it wrote (or nothing).
            payload = _measurement_payload(replace(measurement, thumbnail_path=None))
            encoded = _encode(payload)
            if len(encoded) > _MAX_ENTRY_BYTES:
                return False
            # Proof, not assumption: an entry exists only when decoding it
            # reproduces the measured object field for field.
            if _restore(json.loads(encoded, object_pairs_hook=_unique_object)) != replace(
                measurement, thumbnail_path=None
            ):
                return False
            data = _encode({
                "schema": _SCHEMA,
                "key": key,
                "payload": payload,
                "sha256": hashlib.sha256(encoded).hexdigest(),
            })
            if len(data) > _MAX_ENTRY_BYTES:
                return False
            private_cache_directory(self.directory)
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".measurement-", suffix=".tmp", delete=False) as stream:
                temporary = stream.name
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / f"{key}.json")
            temporary = None
            prune_cache_directory(self.directory, _MAX_ENTRIES, _MAX_TOTAL_BYTES)
            self._count("writes")
            return True
        except Exception:
            return False
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    def _count(self, name: str) -> None:
        if self.stats is not None:
            self.stats[name] = self.stats.get(name, 0) + 1


__all__ = ["FrameMeasurementCache"]
