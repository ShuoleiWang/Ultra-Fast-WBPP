from __future__ import annotations

from pathlib import Path
import os
import json
from typing import Any

from astropy.io import fits
import numpy as np
import pytest

from openastroflow_engine.backends import DeviceKind
from openastroflow_engine.calibration import (
    FrameExpression,
    IntegrationMapPaths,
    IntegrationParameters,
    integrate_expressions,
)
from openastroflow_engine.hardware import CpuFamily, HardwareProfile
import openastroflow_engine.metal_integration as metal
from openastroflow_engine.performance_profile import select_execution_tuning


def _write(path: Path, value: float, shape: tuple[int, int] = (96, 8)) -> Path:
    fits.writeto(path, np.full(shape, value, dtype=np.float32), overwrite=False)
    return path


def _apple_profile(chip: str, optimization: str) -> HardwareProfile:
    return HardwareProfile(
        operating_system="macOS",
        architecture="arm64",
        cpu_brand=chip,
        cpu_family=CpuFamily.APPLE_M,
        devices=(DeviceKind.CPU, DeviceKind.METAL),
        cpu_backend="apple-silicon-cpu-v1",
        accelerator_backend="metal-generic-v1",
        optimization_profile=optimization,
    )


class _FakeMetalExecutor:
    calls: list[dict[str, Any]] = []

    def __init__(self, **_: Any) -> None:
        self.library_path = Path(__file__).resolve()

    def __enter__(self) -> _FakeMetalExecutor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def run_masked_tile(
        self, samples: np.ndarray, rejection_mask: np.ndarray, **kwargs: Any
    ) -> metal.NativeTileResult:
        type(self).calls.append(
            {"shape": samples.shape, "maskShape": rejection_mask.shape, **kwargs}
        )
        accepted, rejected, _ = metal._portable_mask_counts(samples, rejection_mask)
        valid = np.isfinite(samples) & (rejection_mask == 0)
        integrated = np.sum(
            np.where(valid, samples, 0.0), axis=0, dtype=np.float64
        ) / np.maximum(np.count_nonzero(valid, axis=0), 1)
        integrated = integrated.astype(np.float32)
        return metal.NativeTileResult(
            integrated=integrated,
            accepted=accepted,
            rejected=rejected,
            stats={
                "deviceName": "Fake Apple GPU",
                "executedOnGpu": True,
                "wallSeconds": 0.01,
                "gpuSeconds": 0.005,
                "submittedBufferBytes": samples.nbytes,
                "inputNormalizationScale": 1.0,
                "outputRescale": 1.0,
                "parityGate": {
                    "performed": bool(kwargs["compare_cpu"]),
                    "pixelsCompared": integrated.size,
                    "finiteMaskEqual": True,
                    "rejectionCountsEqual": True,
                    "maxAbs": 0.0,
                    "rmse": 0.0,
                    "maxAbsLimit": 2e-6,
                    "rmseLimit": 2e-7,
                    "passed": True,
                },
            },
        )


@pytest.mark.parametrize(
    (
        "chip",
        "optimization",
        "memory_gib",
        "expected_backend",
        "expected_rows",
        "height",
        "expected_workers",
    ),
    (
        (
            "Apple M2 Max",
            "apple-silicon-generic-v1",
            32,
            "generic-apple-metal",
            48,
            192,
            4,
        ),
        (
            "Apple M3 Pro",
            "apple-m3-pro-tuned-v1",
            36,
            "m3-pro-tuned",
            64,
            512,
            8,
        ),
    ),
)
def test_apple_profiles_drive_real_tile_and_buffer_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chip: str,
    optimization: str,
    memory_gib: int,
    expected_backend: str,
    expected_rows: int,
    height: int,
    expected_workers: int,
) -> None:
    _FakeMetalExecutor.calls = []
    monkeypatch.setattr(metal, "NativeMetalExecutor", _FakeMetalExecutor)
    paths = [
        _write(
            tmp_path / f"frame-{index}.fits",
            10.0 + index,
            (height, 8),
        )
        for index in range(5)
    ]
    profile = _apple_profile(chip, optimization)
    tuning = select_execution_tuning(
        profile, logical_cores=12, physical_memory_bytes=memory_gib * 1024**3
    )

    result = metal.integrate_registered_group(
        [FrameExpression(str(path)) for path in paths],
        tmp_path / "master.fits",
        parameters=IntegrationParameters(max_memory_bytes=1024 * 1024),
        requested_backend=expected_backend,
        hardware=profile,
        tuning=tuning,
    )

    assert result.execution["selectedBackend"] == expected_backend
    assert result.execution["acceleratorUsed"] is True
    assert result.execution["tileRows"] == expected_rows
    assert result.execution["inflightBuffers"] == 2
    assert result.execution["peakInflightBuffers"] == 2
    assert result.execution["configuredCpuWorkers"] == tuning.cpu_workers
    assert result.execution["cpuWorkersUsed"] == expected_workers
    assert (
        result.execution["memoryModel"]["estimatedPeakBytes"]
        <= result.execution["integrationMemoryBudgetBytes"]
    )
    assert result.execution["fastMath"] is False
    assert result.execution["parityGate"]["passed"] is True
    assert result.execution["parityGate"]["scope"] == "first-middle-last-tiles"
    assert result.execution["parityGate"]["sampledTileCount"] == 3
    assert len(_FakeMetalExecutor.calls) == expected_workers
    assert all(call["shape"][1] == expected_rows for call in _FakeMetalExecutor.calls)
    assert sum(bool(call["compare_cpu"]) for call in _FakeMetalExecutor.calls) == 3


def test_ninety_six_frames_use_one_full_stack_metal_reduction_without_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeMetalExecutor.calls = []
    monkeypatch.setattr(metal, "NativeMetalExecutor", _FakeMetalExecutor)
    paths = [
        _write(tmp_path / f"frame-{index:03d}.fits", 1.0 if index < 4 else 2.0, (2, 2))
        for index in range(96)
    ]
    profile = _apple_profile("Apple M3 Pro", "apple-m3-pro-tuned-v1")
    tuning = select_execution_tuning(
        profile, logical_cores=12, physical_memory_bytes=36 * 1024**3
    )

    result = metal.integrate_registered_group(
        [FrameExpression(str(path)) for path in paths],
        tmp_path / "master.fits",
        requested_backend="m3-pro-tuned",
        hardware=profile,
        tuning=tuning,
    )

    assert result.frame_count == 96
    assert result.accepted_samples + result.rejected_samples == 96 * 4
    assert result.execution["selectedBackend"] == "m3-pro-tuned"
    assert result.execution["acceleratorUsed"] is True
    assert result.execution["inputFramesTruncated"] is False
    assert result.execution["rejectionMask"]["partialMeanBatching"] is False
    assert result.execution["parityGate"]["sampledTileCount"] == 1
    assert _FakeMetalExecutor.calls[0]["shape"][0] == 96
    assert _FakeMetalExecutor.calls[0]["maskShape"][0] == 96
    with fits.open(result.output_path, memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, 2.0, atol=1e-6)


def test_m3_profile_is_not_applied_to_other_apple_chips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeMetalExecutor.calls = []
    monkeypatch.setattr(metal, "NativeMetalExecutor", _FakeMetalExecutor)
    paths = [_write(tmp_path / f"frame-{index}.fits", 1.0 + index) for index in range(5)]
    profile = _apple_profile("Apple M4", "apple-silicon-generic-v1")
    tuning = select_execution_tuning(
        profile, logical_cores=10, physical_memory_bytes=24 * 1024**3
    )

    result = metal.integrate_registered_group(
        [FrameExpression(str(path)) for path in paths],
        tmp_path / "master.fits",
        requested_backend="m3-pro-tuned",
        hardware=profile,
        tuning=tuning,
    )

    assert result.execution["selectedBackend"] == "generic-apple-metal"
    assert "not applied" in result.execution["fallbackReason"]


def test_low_memory_m3_pro_uses_generic_metal_tuning() -> None:
    profile = _apple_profile("Apple M3 Pro", "apple-m3-pro-tuned-v1")
    tuning = select_execution_tuning(
        profile, logical_cores=8, physical_memory_bytes=8 * 1024**3
    )

    selected, note = metal._requested_selection("auto", profile, tuning)

    assert tuning.profile_id == "apple-silicon-generic-v1"
    assert selected == "generic-apple-metal"
    assert note is None


def test_realistic_deep_stack_memory_plan_preserves_cap_and_tile_height() -> None:
    budget = 4 * 1024**3

    workers, rows, bytes_per_row = metal._metal_memory_plan(
        width=6252,
        height=4176,
        frame_count=96,
        configured_workers=8,
        inflight_buffers=2,
        target_tile_rows=64,
        memory_budget_bytes=budget,
    )

    assert workers == 3
    assert rows == 64
    assert bytes_per_row * rows <= budget
    assert (
        metal._metal_parallel_bytes_per_row(
            width=6252,
            frame_count=96,
            preparation_workers=4,
            inflight_buffers=2,
        )
        * rows
        > budget
    )


def test_maximum_stack_memory_plan_reduces_rows_without_exceeding_cap() -> None:
    budget = 4 * 1024**3

    workers, rows, bytes_per_row = metal._metal_memory_plan(
        width=6252,
        height=4176,
        frame_count=metal.MAXIMUM_METAL_REJECTION_FRAMES,
        configured_workers=8,
        inflight_buffers=2,
        target_tile_rows=64,
        memory_budget_bytes=budget,
    )

    assert workers == 1
    assert 1 <= rows < 64
    assert bytes_per_row * rows <= budget


def test_rejection_maps_are_apple_profile_and_tile_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(metal, "NativeMetalExecutor", _FakeMetalExecutor)
    rng = np.random.default_rng(19)
    frame_count, height, width = 9, 128, 16
    row_scale = np.linspace(0.001, 1.0, height, dtype=np.float32)
    stack = (
        rng.normal(0.0, 1.0, (frame_count, height, width)).astype(np.float32)
        * row_scale[None, :, None]
    )
    stack[0] += np.float32(0.01)
    stack[:, 3, 2] = np.nan
    stack[:, 3, 3] = [1, 1, 1, 1, 1, 100, np.inf, np.inf, np.inf]
    stack[:, 3, 4] = [1, 1, 100, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    stack[8, 91, 7] += np.float32(100.0)
    paths = []
    for index, values in enumerate(stack):
        path = tmp_path / f"profile-{index:02d}.fits"
        fits.writeto(path, values, overwrite=False)
        paths.append(path)

    runs = []
    profiles = (
        (
            _apple_profile("Apple M2 Max", "apple-silicon-generic-v1"),
            32,
        ),
        (
            _apple_profile("Apple M3 Pro", "apple-m3-pro-tuned-v1"),
            36,
        ),
    )
    for index, (profile, memory_gib) in enumerate(profiles):
        tuning = select_execution_tuning(
            profile,
            logical_cores=12,
            physical_memory_bytes=memory_gib * 1024**3,
        )
        accepted = tmp_path / f"profile-{index}-accepted.fits"
        coverage = tmp_path / f"profile-{index}-coverage.fits"
        rejected = tmp_path / f"profile-{index}-rejected.fits"
        result = metal.integrate_registered_group(
            [FrameExpression(str(path)) for path in paths],
            tmp_path / f"profile-{index}-master.fits",
            parameters=IntegrationParameters(
                max_memory_bytes=1024 * 1024,
                max_statistics_samples=100,
                minimum_rejection_frames=5,
            ),
            requested_backend=(
                "m3-pro-tuned" if profile.m3_pro_tuned else "generic-apple-metal"
            ),
            hardware=profile,
            tuning=tuning,
            map_paths=IntegrationMapPaths(accepted, coverage, rejected),
        )
        runs.append((result, accepted, coverage, rejected))

    assert [run[0].tile_rows for run in runs] == [48, 64]
    assert (
        runs[0][0].execution["rejectionMask"]["sigmaFloor"]
        == runs[1][0].execution["rejectionMask"]["sigmaFloor"]
    )
    assert runs[0][0].accepted_samples == runs[1][0].accepted_samples
    assert runs[0][0].rejected_samples == runs[1][0].rejected_samples
    for left, right in zip(runs[0][1:], runs[1][1:], strict=True):
        assert np.array_equal(fits.getdata(left), fits.getdata(right), equal_nan=True)

    cpu_accepted = tmp_path / "profile-cpu-accepted.fits"
    cpu_coverage = tmp_path / "profile-cpu-coverage.fits"
    cpu_rejected = tmp_path / "profile-cpu-rejected.fits"
    integrate_expressions(
        [FrameExpression(str(path)) for path in paths],
        tmp_path / "profile-cpu-master.fits",
        parameters=IntegrationParameters(
            max_memory_bytes=1024 * 1024,
            max_statistics_samples=100,
            minimum_rejection_frames=5,
        ),
        map_paths=IntegrationMapPaths(
            cpu_accepted, cpu_coverage, cpu_rejected
        ),
    )
    for accelerated, portable in zip(
        runs[0][1:],
        (cpu_accepted, cpu_coverage, cpu_rejected),
        strict=True,
    ):
        assert np.array_equal(
            fits.getdata(accelerated), fits.getdata(portable), equal_nan=True
        )
    np.testing.assert_array_equal(fits.getdata(cpu_accepted)[3, 3:5], [5, 3])
    np.testing.assert_array_equal(fits.getdata(cpu_rejected)[3, 3:5], [1, 0])


def test_auto_prefers_native_cpu_kernels_and_explicit_metal_still_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openastroflow_engine import native_kernels

    monkeypatch.setattr(metal, "NativeMetalExecutor", _FakeMetalExecutor)
    paths = [_write(tmp_path / f"auto-{index}.fits", 10.0 + index, (96, 8)) for index in range(5)]
    profile = _apple_profile("Apple M3 Pro", "apple-m3-pro-tuned-v1")
    tuning = select_execution_tuning(profile, logical_cores=12, physical_memory_bytes=36 * 1024**3)
    automatic = metal.integrate_registered_group(
        [FrameExpression(str(path)) for path in paths],
        tmp_path / "auto-master.fits",
        parameters=IntegrationParameters(max_memory_bytes=1024 * 1024),
        requested_backend="auto",
        hardware=profile,
        tuning=tuning,
    )
    if native_kernels.load_native_kernels() is not None:
        assert automatic.execution["selectedBackend"] == "portable-cpu"
        assert automatic.execution["selectionPolicy"] == "native-cpu-kernels-preferred"
        assert automatic.execution["fallbackReason"] == metal.NATIVE_CPU_AUTO_REASON
        assert automatic.execution["acceleratorUsed"] is False
        assert automatic.execution["rejectionMask"]["kernel"] == native_kernels.MAD_KERNEL_ID
    else:
        assert automatic.execution["selectedBackend"] == "m3-pro-tuned"
    explicit = metal.integrate_registered_group(
        [FrameExpression(str(path)) for path in paths],
        tmp_path / "explicit-master.fits",
        parameters=IntegrationParameters(max_memory_bytes=1024 * 1024),
        requested_backend="m3-pro-tuned",
        hardware=profile,
        tuning=tuning,
    )
    assert explicit.execution["selectedBackend"] == "m3-pro-tuned"
    assert explicit.execution["acceleratorUsed"] is True
    np.testing.assert_array_equal(
        fits.getdata(tmp_path / "auto-master.fits"), fits.getdata(tmp_path / "explicit-master.fits")
    )


def test_portable_mask_counts_separate_nan_coverage_from_rejection() -> None:
    samples = np.asarray(
        [
            [[1.0, np.nan, 3.0]],
            [[2.0, 4.0, np.nan]],
            [[3.0, 5.0, 6.0]],
        ],
        dtype=np.float32,
    )
    mask = np.zeros(samples.shape, dtype=np.uint8)
    mask[2, 0, 0] = 1
    mask[0, 0, 2] = 1

    accepted, rejected, unavailable = metal._portable_mask_counts(samples, mask)

    np.testing.assert_array_equal(accepted, [[2, 2, 1]])
    np.testing.assert_array_equal(rejected, [[1, 0, 1]])
    np.testing.assert_array_equal(unavailable, [[0, 1, 1]])


def test_ctypes_request_keeps_all_array_owners_alive() -> None:
    samples = np.ones((5, 2, 2), dtype=np.float32)
    scales = np.ones((5, 2, 2), dtype=np.float32)
    offsets = np.zeros((5, 2, 2), dtype=np.float32)
    weights = np.ones(5, dtype=np.float32)

    request, _, _, _, _ = metal.NativeMetalExecutor._request_and_output(
        samples,
        image_height=2,
        first_row=0,
        scales=scales,
        offsets=offsets,
        weights=weights,
        sigma_clip=4.0,
    )

    assert len(request._array_owners) == 4  # type: ignore[attr-defined]
    assert request.sample_count == samples.size


@pytest.mark.skipif(os.name == "nt", reason="POSIX native-file mode gate")
def test_world_writable_native_library_candidate_is_rejected(tmp_path: Path) -> None:
    candidate = tmp_path / "libopenastroflow_native.dylib"
    candidate.write_bytes(b"not-a-library")
    candidate.chmod(0o666)

    discovered = metal._candidate_library_paths(candidate)

    assert candidate.resolve() not in discovered


@pytest.mark.skipif(
    os.environ.get("OAF_RUN_REAL_METAL") != "1",
    reason="opt-in test requires a real Apple Metal device",
)
def test_real_m3_pro_ninety_six_frame_full_stack_metal(tmp_path: Path) -> None:
    shape = (192, 64)
    paths = []
    for index in range(96):
        values = np.full(shape, 2.0 + index * 0.001, dtype=np.float32)
        if index == 0:
            values[0, 0] = np.nan
        elif index == 1:
            values[96, 1] = np.nan
        elif index == 2:
            values[191, 2] = np.nan
        path = tmp_path / f"frame-{index:03d}.fits"
        fits.writeto(path, values, overwrite=False)
        paths.append(path)

    metal_maps = IntegrationMapPaths(
        tmp_path / "metal-accepted.fits",
        tmp_path / "metal-coverage.fits",
        tmp_path / "metal-rejected.fits",
    )
    cpu_maps = IntegrationMapPaths(
        tmp_path / "cpu-accepted.fits",
        tmp_path / "cpu-coverage.fits",
        tmp_path / "cpu-rejected.fits",
    )

    result = metal.integrate_registered_group(
        [FrameExpression(str(path)) for path in paths],
        tmp_path / "master.fits",
        requested_backend="m3-pro-tuned",
        map_paths=metal_maps,
    )
    cpu = integrate_expressions(
        [FrameExpression(str(path)) for path in paths],
        tmp_path / "cpu-master.fits",
        map_paths=cpu_maps,
    )

    execution = result.execution
    (tmp_path / "metal-execution.json").write_text(
        json.dumps(execution, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert result.frame_count == 96
    assert result.accepted_samples + result.rejected_samples == 96 * 192 * 64 - 3
    assert execution["selectedBackend"] == "m3-pro-tuned"
    assert execution["acceleratorUsed"] is True
    assert "M3 Pro" in execution["deviceName"]
    assert execution["inputFramesTruncated"] is False
    assert execution["rejectionMask"]["partialMeanBatching"] is False
    assert execution["parityGate"]["passed"] is True
    assert execution["parityGate"]["sampledTileCount"] == 3
    assert execution["parityGate"]["sampledFirstRows"] == [0, 64, 128]
    assert result.accepted_samples == cpu.accepted_samples
    assert result.rejected_samples == cpu.rejected_samples
    for metal_path, cpu_path in zip(
        (
            metal_maps.accepted_count,
            metal_maps.coverage,
            metal_maps.rejection_count,
        ),
        (cpu_maps.accepted_count, cpu_maps.coverage, cpu_maps.rejection_count),
        strict=True,
    ):
        assert np.array_equal(
            fits.getdata(metal_path), fits.getdata(cpu_path), equal_nan=True
        )
    with fits.open(result.output_path, memmap=False) as metal_hdul, fits.open(
        cpu.output_path, memmap=False
    ) as cpu_hdul:
        metal_pixels = np.asarray(metal_hdul[0].data, dtype=np.float64)
        cpu_pixels = np.asarray(cpu_hdul[0].data, dtype=np.float64)
    assert np.array_equal(np.isfinite(metal_pixels), np.isfinite(cpu_pixels))
    difference = metal_pixels - cpu_pixels
    assert float(np.max(np.abs(difference))) <= 2e-5
    assert float(np.sqrt(np.mean(np.square(difference)))) <= 2e-6
