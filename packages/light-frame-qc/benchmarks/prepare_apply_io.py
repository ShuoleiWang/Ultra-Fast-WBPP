#!/usr/bin/env python3
"""Reproducible prepare/apply I/O-count microbenchmark using temporary FITS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from astropy.io import fits
import numpy as np

from lightframeqc.identity import compute_file_identity
from lightframeqc.models import (
    Confidence,
    Decision,
    EvidenceFamily,
    EvidenceSeverity,
    FrameFeatures,
    FrameResult,
    GateDisposition,
    QualityEvidence,
    QualityGateResult,
    RegistrationMetrics,
)
from lightframeqc import prepare
from lightframeqc.quality_gate import GatePolicy
from lightframeqc.readers import probe_frame_metadata


def _write_frame(
    path: Path,
    *,
    role: str,
    target: str,
    seed: int,
    width: int,
    height: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(seed).integers(
        0, 65535, size=(height, width), dtype=np.uint16
    )
    hdu = fits.PrimaryHDU(pixels)
    hdu.header["IMAGETYP"] = role
    hdu.header["OBJECT"] = target
    hdu.header["FILTER"] = "R"
    hdu.header["EXPTIME"] = 300.0 if role == "Light Frame" else 0.75
    hdu.header["INSTRUME"] = "QHY268M"
    hdu.header["XBINNING"] = 1
    hdu.header["YBINNING"] = 1
    hdu.header["GAIN"] = 0
    hdu.header["OFFSET"] = 30
    hdu.writeto(path)
    timestamp = 1_780_000_000_123_456_789 + seed
    os.utime(path, ns=(timestamp, timestamp))


def _result(path: Path, decision: Decision) -> FrameResult:
    policy = GatePolicy()
    disposition = (
        GateDisposition.PASS
        if decision is Decision.KEEP
        else GateDisposition.REVIEW
    )
    evidence = []
    if disposition is GateDisposition.REVIEW:
        evidence = [
            QualityEvidence(
                code="BENCHMARK.REVIEW",
                family=EvidenceFamily.PROVENANCE,
                severity=EvidenceSeverity.REVIEW,
                message="benchmark manifest-only frame",
            )
        ]
    result = FrameResult(
        path=str(path.resolve()),
        group_id="benchmark-group",
        reference_path=str(path.resolve()),
        decision=decision,
        confidence=Confidence.HIGH,
        reasons=[],
        warnings=[],
        registration=RegistrationMetrics(
            ok=True, matched_stars=40, match_fraction=0.9
        ),
        features=FrameFeatures(),
        metadata=probe_frame_metadata(path),
        star_count=100,
    )
    result.identity = compute_file_identity(path)
    result.quality_gate = QualityGateResult(
        disposition=disposition,
        evidence=evidence,
        summary="prepare I/O benchmark gate",
        version=f"{policy.version}@{policy.canonical_digest()}",
        policy_digest=policy.canonical_digest(),
        policy=policy.serializable(),
    )
    return result


def run(width: int, height: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="openastroflow-prepare-benchmark-") as raw:
        root = Path(raw)
        results = []
        source_paths: set[str] = set()
        for name, target, seed, decision in (
            ("target-a.fits", "Target A", 1, Decision.KEEP),
            ("target-b.fits", "Target B", 2, Decision.KEEP),
            ("excluded.fits", "Target A", 3, Decision.REVIEW),
        ):
            path = root / name
            _write_frame(
                path,
                role="Light Frame",
                target=target,
                seed=seed,
                width=width,
                height=height,
            )
            results.append(_result(path, decision))
            source_paths.add(str(path.resolve()))

        flat = root / "flats" / "masterFlat_R.fits"
        _write_frame(
            flat,
            role="Master Flat",
            target="FlatWizard",
            seed=4,
            width=width,
            height=height,
        )
        source_paths.add(str(flat.resolve()))
        destination = root / "prepared"
        plan = prepare.build_prepare_plan(results, [flat], destination)

        real_identity = prepare._compute_file_identity
        real_stream_copy = prepare._stream_copy
        source_hash_bytes = 0
        target_hash_bytes = 0
        source_hash_calls = 0
        target_hash_calls = 0
        copy_bytes = 0
        copy_calls = 0
        source_digest_passes = 0

        def counted_identity(path: str | os.PathLike[str]):
            nonlocal source_hash_bytes, target_hash_bytes
            nonlocal source_hash_calls, target_hash_calls
            resolved = str(Path(path).resolve(strict=True))
            size = Path(path).stat().st_size
            if resolved in source_paths:
                source_hash_calls += 1
                source_hash_bytes += size
            else:
                target_hash_calls += 1
                target_hash_bytes += size
            return real_identity(path)

        def counted_stream(source, target, digest):
            nonlocal copy_bytes, copy_calls, source_digest_passes
            copied = real_stream_copy(source, target, digest)
            copy_calls += 1
            copy_bytes += copied
            source_digest_passes += int(digest is not None)
            return copied

        prepare._compute_file_identity = counted_identity
        prepare._stream_copy = counted_stream
        try:
            started = time.perf_counter()
            prepare.apply_prepare_plan(plan, destination)
            elapsed = time.perf_counter() - started
        finally:
            prepare._compute_file_identity = real_identity
            prepare._stream_copy = real_stream_copy

        return {
            "fixture": {
                "width": width,
                "height": height,
                "passedLights": 2,
                "manifestOnlyLights": 1,
                "sharedMasterFlats": 1,
                "publishedFiles": 4,
            },
            "elapsedSeconds": round(elapsed, 6),
            "sourceFullHashCalls": source_hash_calls,
            "sourceFullHashBytes": source_hash_bytes,
            "sourceCopyCalls": copy_calls,
            "sourceCopyBytes": copy_bytes,
            "sourceDigestPasses": source_digest_passes,
            "targetFullHashCalls": target_hash_calls,
            "targetFullHashBytes": target_hash_bytes,
            "contentReadBytesTotal": source_hash_bytes
            + copy_bytes
            + target_hash_bytes,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    arguments = parser.parse_args()
    if arguments.width < 32 or arguments.height < 32:
        parser.error("width and height must both be at least 32")
    print(json.dumps(run(arguments.width, arguments.height), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
