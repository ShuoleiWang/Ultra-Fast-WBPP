"""Advisory JSON cache for complete, measured light-frame cohorts.

Only group analysis is cached. Source measurement and the final quality gate
remain the caller's responsibility on every run. Entries are intentionally
disposable: unsupported inputs, unavailable fingerprints and any cache I/O or
validation failure all become misses.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime
from functools import lru_cache
import hashlib
import importlib
import json
import marshal
import math
import os
from pathlib import Path
import platform
import stat
import sys
import tempfile
from typing import Any

import numpy as np

from .config import QcConfig
from .models import Confidence, Decision, FrameFeatures, FrameMeasurement, FrameResult, RegistrationMetrics, Star


_SCHEMA = 1
_MAX_ENTRY_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_BYTES = 128 * 1024 * 1024
_MAX_ENTRIES = 64
_MAX_FRAMES = 512
_MAX_KEY_BYTES = 64 * 1024 * 1024
_MODULES = (
    "analysis", "analysis_cache", "registration", "triangle_bootstrap",
    "grouping", "statistics", "metadata", "config", "models", "morphology",
)
_PACKAGES = ("numpy", "scipy", "skimage", "astroalign", "astropy")
_RESULT_FIELDS = {
    "path", "group_id", "reference_path", "decision", "confidence", "reasons",
    "warnings", "registration", "features", "star_count", "grid",
}
_SUMMARY_FIELDS = {
    "groupId", "frameCount", "referencePath", "filter", "camera", "target",
    "exposureSeconds", "decisions",
}
_GRID_FIELDS = {
    "expectedStars", "referenceExpectedStars", "matchedStars", "completeness",
    "transparencyResidualMag", "coarseTransparencyResidualMag", "missingMask",
    "supportedOverlapMask", "backgroundDeltaRobustSigma", "textureRatio",
}


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported cache input: {type(value).__name__}")


def _encode(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False, default=_json_default,
    ).encode("ascii")


@lru_cache(maxsize=8)
def implementation_fingerprint(
    schema: int, modules: tuple[str, ...], packages: tuple[str, ...]
) -> str:
    """Hash source or frozen loader code; never deserialize executable cache data.

    No manual-version fallback: if a packaged loader cannot expose its code,
    caching is disabled. This is preferable to reusing stale scientific results.
    The small fingerprint is shared for this process's loaded implementation;
    every cache in the package names the modules and runtime packages its own
    stored values depend on, so a change to either is a miss.
    """
    digest = hashlib.sha256()
    digest.update(_encode([schema, sys.version, platform.machine(), sys.byteorder]))
    for name in modules:
        module = importlib.import_module(f"lightframeqc.{name}")
        loader = module.__spec__.loader
        source_path = Path(module.__file__) if getattr(module, "__file__", None) else None
        if source_path is not None and source_path.suffix == ".py" and source_path.is_file():
            contents = source_path.read_bytes()
        else:
            code = loader.get_code(module.__name__)
            if code is None:
                raise ValueError("Cache implementation fingerprint is unavailable")
            contents = marshal.dumps(code)
        digest.update(_encode(name))
        digest.update(hashlib.sha256(contents).digest())
    for name in packages:
        version = getattr(importlib.import_module(name), "__version__", None)
        if not isinstance(version, str) or not version:
            raise ValueError("Cache runtime version is unavailable")
        digest.update(_encode([name, version]))
    return digest.hexdigest()


def _implementation_fingerprint() -> str:
    return implementation_fingerprint(_SCHEMA, _MODULES, _PACKAGES)


_STAR_FLOAT_FIELDS = ("x", "y", "flux", "peak", "a", "b", "theta", "fwhm", "ellipticity")
_STAR_INT_FIELDS = ("flags", "support_pixels", "detection_pixels")


def _packed_stars(stars: list[Star] | None) -> bytes:
    """Exact little-endian float64/int64 image of a star list for hashing.

    A frame carries thousands of stars twice over (supported and raw); their
    JSON text was most of the key's cost.  ``None`` integers become -1, which
    no real count takes, and a missing list is distinguished from an empty one.
    """

    if stars is None:
        return b"none"
    if not stars:
        return b"empty"
    floats = np.array(
        [[getattr(star, name) for name in _STAR_FLOAT_FIELDS] for star in stars],
        dtype="<f8",
    )
    integers = np.array(
        [[-1 if getattr(star, name) is None else getattr(star, name) for name in _STAR_INT_FIELDS] for star in stars],
        dtype="<i8",
    )
    return len(stars).to_bytes(8, "big") + floats.tobytes(order="C") + integers.tobytes(order="C")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate cache key")
        result[key] = value
    return result


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_grid(grid: Any, config: QcConfig) -> None:
    if not isinstance(grid, dict):
        raise ValueError("Invalid cached grid")
    if not grid:
        return
    optional = {"consensusDimmingResidualMag", "coarseConsensusDimmingResidualMag"}
    if not _GRID_FIELDS <= grid.keys() or grid.keys() - (_GRID_FIELDS | optional | {"rows", "columns"}):
        raise ValueError("Invalid cached grid fields")
    if grid.get("rows") != config.grid_rows or grid.get("columns") != config.grid_columns:
        raise ValueError("Invalid cached grid dimensions")
    for name, values in grid.items():
        if name in {"rows", "columns"}:
            continue
        rows, columns = (4, 4) if name.startswith("coarse") else (config.grid_rows, config.grid_columns)
        if not isinstance(values, list) or len(values) != rows:
            raise ValueError("Invalid cached grid rows")
        for row in values:
            if not isinstance(row, list) or len(row) != columns:
                raise ValueError("Invalid cached grid columns")
            if any(value is not None and type(value) is not bool and not _number(value) for value in row):
                raise ValueError("Invalid cached grid cell")


def _restore(payload: Any, frames: list[FrameMeasurement], config: QcConfig) -> tuple[list[dict[str, Any]], list[FrameResult]]:
    if not isinstance(payload, dict) or set(payload) != {"groups", "results"}:
        raise ValueError("Invalid cached payload")
    summaries, records = payload["groups"], payload["results"]
    by_path = {frame.metadata.path: frame for frame in frames}
    if not isinstance(records, list) or len(records) != len(frames):
        raise ValueError("Incomplete cached cohort")
    if not isinstance(summaries, list) or not 1 <= len(summaries) <= len(frames):
        raise ValueError("Invalid cached groups")
    results: list[FrameResult] = []
    remaining = set(by_path)
    for record in records:
        if not isinstance(record, dict) or set(record) != _RESULT_FIELDS:
            raise ValueError("Invalid cached result fields")
        path, reference_path = record["path"], record["reference_path"]
        if path not in remaining or reference_path not in by_path or not isinstance(record["group_id"], str):
            raise ValueError("Invalid cached frame membership")
        remaining.remove(path)
        frame = by_path[path]
        expected_count = frame.detected_source_count if frame.detected_source_count is not None else len(frame.stars)
        if type(record["star_count"]) is not int or record["star_count"] != expected_count:
            raise ValueError("Invalid cached star count")
        for name in ("reasons", "warnings"):
            if not isinstance(record[name], list) or not all(isinstance(item, str) for item in record[name]):
                raise ValueError("Invalid cached explanation")
        features = record["features"]
        if not isinstance(features, dict) or set(features) != {field.name for field in fields(FrameFeatures)}:
            raise ValueError("Invalid cached features")
        for field in fields(FrameFeatures):
            value = features[field.name]
            if value is None and "None" in str(field.type):
                continue
            if str(field.type).startswith("bool"):
                if type(value) is not bool:
                    raise ValueError("Invalid cached boolean feature")
                continue
            if not _number(value) or (str(field.type).startswith("int") and type(value) is not int):
                raise ValueError("Invalid cached feature value")
        registration = record["registration"]
        if not isinstance(registration, dict) or set(registration) != {field.name for field in fields(RegistrationMetrics)}:
            raise ValueError("Invalid cached registration")
        if type(registration["ok"]) is not bool or type(registration["matched_stars"]) is not int:
            raise ValueError("Invalid cached match count")
        if not _number(registration["match_fraction"]) or not 0 <= registration["match_fraction"] <= 1:
            raise ValueError("Invalid cached match fraction")
        for name, count in (("source_indices", len(frame.stars)), ("reference_indices", len(by_path[reference_path].stars))):
            indices = registration[name]
            if not isinstance(indices, list) or len(indices) != registration["matched_stars"] or any(type(index) is not int or not 0 <= index < count for index in indices):
                raise ValueError("Invalid cached match indices")
            if len(set(indices)) != len(indices):
                raise ValueError("Duplicate cached match indices")
        rms = registration["rms_pixels"]
        if rms is not None and (not _number(rms) or rms < 0):
            raise ValueError("Invalid cached registration residual")
        if registration["error"] is not None and not isinstance(registration["error"], str):
            raise ValueError("Invalid cached registration error")
        matrix = registration["matrix"]
        if matrix is not None and (
            not isinstance(matrix, list) or len(matrix) != 3 or
            any(not isinstance(row, list) or len(row) != 3 or not all(_number(value) for value in row) for row in matrix)
        ):
            raise ValueError("Invalid cached transform")
        _validate_grid(record["grid"], config)
        results.append(FrameResult(
            **{name: value for name, value in record.items() if name not in {"decision", "confidence", "features", "registration"}},
            decision=Decision(record["decision"]), confidence=Confidence(record["confidence"]),
            features=FrameFeatures(**features), registration=RegistrationMetrics(**registration),
            metadata=frame.metadata, identity=frame.identity, thumbnail_path=frame.thumbnail_path,
        ))
    group_ids: set[str] = set()
    for summary in summaries:
        if not isinstance(summary, dict) or set(summary) != _SUMMARY_FIELDS:
            raise ValueError("Invalid cached summary fields")
        group_id = summary["groupId"]
        if not isinstance(group_id, str) or group_id in group_ids:
            raise ValueError("Duplicate cached group")
        group_ids.add(group_id)
        members = [result for result in results if result.group_id == group_id]
        reference_path = summary["referencePath"]
        if not members or reference_path not in {member.path for member in members} or any(member.reference_path != reference_path for member in members):
            raise ValueError("Invalid cached reference")
        reference = by_path[reference_path].metadata
        expected = {
            "groupId": group_id, "frameCount": len(members), "referencePath": reference_path,
            "filter": reference.filter_name, "camera": reference.camera, "target": reference.target,
            "exposureSeconds": reference.exposure_seconds,
            "decisions": {decision.value: sum(item.decision == decision for item in members) for decision in Decision},
        }
        if summary != expected:
            raise ValueError("Cached summary does not match its frames")
    if group_ids != {result.group_id for result in results}:
        raise ValueError("Missing cached group")
    return summaries, results


# Every FrameMeasurement field except the star lists, the nested dataclasses
# (encoded separately) and the presentation-only thumbnail path.
_FRAME_KEY_FIELDS = tuple(
    item.name
    for item in fields(FrameMeasurement)
    if item.name not in {"stars", "raw_stars", "metadata", "identity", "thumbnail_path"}
)


class GroupAnalysisCache:
    def __init__(self, directory: Path | str, config: QcConfig, stats: dict[str, int] | None = None):
        self.directory = Path(directory) / "group-analysis-v1"
        self.config = config
        self.stats = stats

    def _count(self, name: str) -> None:
        if self.stats is not None:
            self.stats[name] = self.stats.get(name, 0) + 1

    def key(self, base_id: str, frames: list[FrameMeasurement]) -> str | None:
        try:
            if not 1 <= len(frames) <= _MAX_FRAMES or len({frame.metadata.path for frame in frames}) != len(frames):
                return None
            config = asdict(self.config)
            config.pop("make_thumbnails")
            digest = hashlib.sha256(_encode([base_id, config, _implementation_fingerprint()]))
            size = 0
            for frame in frames:
                # Include all scientific inputs and freshly measured identity,
                # even optional metadata and source counts. Only the report's
                # presentation-specific thumbnail destination is excluded.
                # Star lists are hashed from their exact binary image; the
                # rest of the measurement is small enough for canonical JSON.
                value = {name: getattr(frame, name) for name in _FRAME_KEY_FIELDS}
                value["metadata"] = asdict(frame.metadata)
                value["identity"] = None if frame.identity is None else asdict(frame.identity)
                parts = [
                    _encode(value),
                    _packed_stars(frame.stars),
                    _packed_stars(frame.raw_stars),
                ]
                for data in parts:
                    size += len(data)
                    if size > _MAX_KEY_BYTES:
                        return None
                    digest.update(len(data).to_bytes(8, "big"))
                    digest.update(data)
            return digest.hexdigest()
        except Exception:
            return None

    def _private_directory(self) -> None:
        private_cache_directory(self.directory)

    def load(self, key: str | None, frames: list[FrameMeasurement]) -> tuple[list[dict[str, Any]], list[FrameResult]] | None:
        try:
            if key is None:
                raise ValueError("Cache key is unavailable")
            self._private_directory()
            path = self.directory / f"{key}.json"
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            )
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_ENTRY_BYTES or (os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & 0o077)):
                    raise ValueError("Invalid cache file")
                raw = stream.read(_MAX_ENTRY_BYTES + 1)
            if len(raw) > _MAX_ENTRY_BYTES:
                raise ValueError("Oversized cache file")
            envelope = json.loads(raw, object_pairs_hook=_unique_object)
            if not isinstance(envelope, dict) or set(envelope) != {"schema", "key", "payload", "sha256"} or envelope["schema"] != _SCHEMA or envelope["key"] != key:
                raise ValueError("Invalid cache envelope")
            encoded = _encode(envelope["payload"])
            if hashlib.sha256(encoded).hexdigest() != envelope["sha256"]:
                raise ValueError("Cache checksum mismatch")
            result = _restore(envelope["payload"], frames, self.config)
            self._count("hits")
            return result
        except Exception:
            self._count("misses")
            return None

    def store(self, key: str | None, summaries: list[dict[str, Any]], results: list[FrameResult]) -> None:
        if key is None:
            return
        temporary: str | None = None
        try:
            records = []
            for result in results:
                record = {name: getattr(result, name) for name in _RESULT_FIELDS}
                record["features"] = asdict(result.features)
                record["registration"] = asdict(result.registration)
                records.append(record)
            # Metadata, identity, thumbnail paths and quality gates are never
            # restored from a cache. Current measured objects supply them.
            payload = {"groups": summaries, "results": records}
            encoded = _encode(payload)
            if len(encoded) > _MAX_ENTRY_BYTES:
                return
            data = _encode({"schema": _SCHEMA, "key": key, "payload": payload, "sha256": hashlib.sha256(encoded).hexdigest()})
            if len(data) > _MAX_ENTRY_BYTES:
                return
            self._private_directory()
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".analysis-", suffix=".tmp", delete=False) as stream:
                temporary = stream.name
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / f"{key}.json")
            temporary = None
            self._prune()
        except Exception:
            pass
        finally:
            if temporary is not None:
                try:
                    Path(temporary).unlink(missing_ok=True)
                except OSError:
                    pass

    def _prune(self) -> None:
        prune_cache_directory(self.directory, _MAX_ENTRIES, _MAX_TOTAL_BYTES)


def private_cache_directory(directory: Path) -> None:
    """Create (or accept) a cache directory only this user can read."""

    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or (os.name == "posix" and (info.st_uid != os.getuid() or info.st_mode & 0o077)):
        raise ValueError("Cache directory is not private")


def prune_cache_directory(directory: Path, max_entries: int, max_total_bytes: int) -> None:
    """Keep the newest entries within both bounds; every entry is disposable."""

    entries = []
    for path in directory.glob("*.json"):
        if len(path.stem) != 64 or any(character not in "0123456789abcdef" for character in path.stem):
            continue
        try:
            info = path.lstat()
            if stat.S_ISREG(info.st_mode):
                entries.append((info.st_mtime_ns, path, info.st_size))
        except OSError:
            continue
    entries.sort(reverse=True)
    total = 0
    for index, (_, path, size) in enumerate(entries):
        total += size
        if index >= max_entries or total > max_total_bytes:
            path.unlink(missing_ok=True)
