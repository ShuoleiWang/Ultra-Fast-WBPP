"""Read-only Light quality preflight for the native desktop review step."""

from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Iterable

from lightframeqc.analysis import analyze_measurements
from lightframeqc.config import DEFAULT_CONFIG
from lightframeqc.measure import measure_paths
from lightframeqc.models import GateDisposition, FrameRole
from lightframeqc.quality_gate import GatePolicy, evaluate_quality_gate
from lightframeqc.readers import probe_frame_metadata

from ..hardware import detect_hardware
from ..performance_profile import select_execution_tuning
from .cache import quality_cache_directory
from .review_preview import (
    MAX_REVIEW_PREVIEWS as _MAX_REVIEW_PREVIEWS,
    MAX_TOTAL_PREVIEW_BYTES as _MAX_TOTAL_PREVIEW_BYTES,
    bounded_review_preview as _bounded_preview,
)


class QualityPreflightError(RuntimeError):
    """A desktop quality preflight could not establish auditable evidence."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def inspect_light_quality(
    paths: Iterable[str | Path], *, workers: int | None = None
) -> dict[str, Any]:
    """Measure authoritative Light files and return compact, content-bound gates.

    The temporary thumbnail workspace is private and removed before return.  The
    actual E2E run repeats the gate and verifies any approval against the current
    source bytes, policy, and complete science request.
    """

    canonical: list[Path] = []
    seen: set[str] = set()
    for value in paths:
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file():
            raise QualityPreflightError(
                "QUALITY_INPUT_NOT_FILE", f"quality input is not a regular file: {path}"
            )
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        metadata = probe_frame_metadata(path)
        if metadata.role is not FrameRole.LIGHT:
            raise QualityPreflightError(
                "QUALITY_INPUT_NOT_LIGHT",
                f"quality preflight accepts authoritative Light frames only: {path}",
            )
        canonical.append(path)
    if not canonical:
        raise QualityPreflightError(
            "QUALITY_NO_LIGHTS", "quality preflight requires at least one Light frame"
        )

    tuning = select_execution_tuning(detect_hardware())
    selected_workers = tuning.qc_workers if workers is None else workers
    if (
        isinstance(selected_workers, bool)
        or not isinstance(selected_workers, int)
        or selected_workers < 1
    ):
        raise QualityPreflightError(
            "QUALITY_WORKER_COUNT_INVALID", "quality workers must be a positive integer"
        )
    config = replace(DEFAULT_CONFIG, make_thumbnails=True)
    policy = GatePolicy()
    timings: dict[str, float] = {}
    cache_stats: dict[str, int] = {}
    measurement_stats: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="ultra-fast-wbpp-qc-") as temporary:
        started = perf_counter()
        measurements = measure_paths(
            canonical, temporary, config, workers=selected_workers, stats=measurement_stats
        )
        timings["measurementSeconds"] = perf_counter() - started
        started = perf_counter()
        analysis_stats: dict[str, Any] = {}
        _groups, results = analyze_measurements(
            measurements, config, cache_directory=quality_cache_directory(), cache_stats=cache_stats,
            workers=selected_workers, stats=analysis_stats,
        )
        timings["analysisSeconds"] = perf_counter() - started
        started = perf_counter()
        evaluate_quality_gate(results, measurements, policy)
        timings["gateSeconds"] = perf_counter() - started

        frames: list[dict[str, Any]] = []
        counts = {disposition.value: 0 for disposition in GateDisposition}
        preview_count = 0
        preview_bytes = 0
        for result in results:
            gate = result.quality_gate
            if gate is None:
                raise QualityPreflightError(
                    "QUALITY_GATE_MISSING", f"quality gate produced no result for {result.path}"
                )
            identity = result.identity
            source_sha256 = (
                f"sha256:{identity.sha256}" if identity is not None else None
            )
            preview: bytes | None = None
            if (
                gate.disposition in {GateDisposition.REVIEW, GateDisposition.HARD_FAIL}
                and preview_count < _MAX_REVIEW_PREVIEWS
            ):
                candidate = _bounded_preview(result.thumbnail_path)
                if (
                    candidate is not None
                    and preview_bytes + len(candidate) <= _MAX_TOTAL_PREVIEW_BYTES
                ):
                    preview = candidate
                    preview_count += 1
                    preview_bytes += len(candidate)
            counts[gate.disposition.value] += 1
            frames.append(
                {
                    "path": str(Path(result.path).resolve(strict=True)),
                    "sourceSha256": source_sha256,
                    "disposition": gate.disposition.value,
                    "decision": result.decision.value,
                    "confidence": result.confidence.value,
                    "starCount": result.star_count,
                    "summary": gate.summary,
                    # A frame without a preflight transform (cloud, no stars)
                    # fails the run's registration the same way; the GUI must
                    # not offer to approve it.
                    "registrable": bool(result.registration.ok),
                    "previewDataUrl": (
                        "data:image/png;base64," + base64.b64encode(preview).decode("ascii")
                        if preview is not None
                        else None
                    ),
                    "previewSha256": (
                        "sha256:" + hashlib.sha256(preview).hexdigest()
                        if preview is not None
                        else None
                    ),
                    "evidence": [item.serializable() for item in gate.evidence],
                }
            )
    frames.sort(key=lambda item: str(item["path"]).casefold())
    return {
        "schemaVersion": 1,
        "gatePolicyDigest": policy.canonical_digest(),
        "workers": selected_workers,
        "counts": counts,
        "frames": frames,
        "timings": timings,
        "measurement": measurement_stats,
        "analysis": analysis_stats,
        "analysisCache": cache_stats,
    }


__all__ = ["QualityPreflightError", "inspect_light_quality"]
