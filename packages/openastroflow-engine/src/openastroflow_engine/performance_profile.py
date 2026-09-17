"""Capability- and memory-derived execution tuning; never scientific semantics.

The tuning is a table keyed by platform, CPU family, memory band and core
count.  Every value here only decides *where and how much* work runs at once
(worker counts, tile rows, in-flight buffers, sampling stride); tile size,
memory budget and thread count never change a pixel, and the differential
tests hold that invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from .hardware import CpuFamily, HardwareProfile
from .platform import current


GIB = 1024**3


@dataclass(frozen=True, slots=True)
class ExecutionTuning:
    profile_id: str
    cpu_workers: int
    qc_workers: int
    registration_memory_bytes: int
    integration_memory_bytes: int
    integration_tile_rows: int
    gpu_inflight_buffers: int
    local_normalization_sample_stride: int
    fast_math: bool = False
    evidence_class: str = "compatible-generic"
    memory_bytes: int = 0
    memory_source: str = "unavailable"
    logical_cores: int = 1
    # Native-kernel thread budget per stage (the fused warp shares it among
    # its in-flight Lights; integration tiles use all of it).  Distinct from
    # ``cpu_workers``, which is the memory-bound number of Lights in flight.
    kernel_threads: int = 1

    def serializable(self) -> dict[str, object]:
        return {
            "profileId": self.profile_id,
            "cpuWorkers": self.cpu_workers,
            "qcWorkers": self.qc_workers,
            "registrationMemoryBytes": self.registration_memory_bytes,
            "integrationMemoryBytes": self.integration_memory_bytes,
            "integrationTileRows": self.integration_tile_rows,
            "gpuInflightBuffers": self.gpu_inflight_buffers,
            "localNormalizationSampleStride": self.local_normalization_sample_stride,
            "fastMath": self.fast_math,
            "evidenceClass": self.evidence_class,
            "memoryBytes": self.memory_bytes,
            "memorySource": self.memory_source,
            "logicalCores": self.logical_cores,
            "kernelThreads": self.kernel_threads,
        }


@dataclass(frozen=True, slots=True)
class _Band:
    """Worker/tile/buffer values of one memory band of a profile table."""

    below_gib: float
    workers: int
    tile_rows: int
    gpu_inflight_buffers: int
    sample_stride: int


# Generic Apple silicon: every M-series chip has at least eight cores; previews
# and warps are bounded per worker, so worker counts scale with memory.
_APPLE_SILICON_BANDS = (
    _Band(12, workers=2, tile_rows=24, gpu_inflight_buffers=1, sample_stride=3),
    _Band(16, workers=4, tile_rows=32, gpu_inflight_buffers=1, sample_stride=3),
    _Band(24, workers=4, tile_rows=32, gpu_inflight_buffers=1, sample_stride=2),
    _Band(float("inf"), workers=8, tile_rows=48, gpu_inflight_buffers=2, sample_stride=2),
)
# x86-64 (Windows, Linux): CPU only.  Each in-flight Light holds about
# 12 bytes per pixel decoded, so workers follow memory; the native kernels
# spread every warp and tile over the remaining cores regardless.
_X86_64_BANDS = (
    _Band(12, workers=2, tile_rows=32, gpu_inflight_buffers=0, sample_stride=3),
    _Band(16, workers=4, tile_rows=32, gpu_inflight_buffers=0, sample_stride=3),
    _Band(24, workers=4, tile_rows=48, gpu_inflight_buffers=0, sample_stride=2),
    _Band(float("inf"), workers=8, tile_rows=48, gpu_inflight_buffers=0, sample_stride=2),
)
_PORTABLE_BANDS = (
    _Band(16, workers=2, tile_rows=32, gpu_inflight_buffers=0, sample_stride=3),
    _Band(float("inf"), workers=4, tile_rows=48, gpu_inflight_buffers=0, sample_stride=3),
)


def _band(table: tuple[_Band, ...], memory: int) -> _Band:
    for band in table:
        if memory < band.below_gib * GIB:
            return band
    return table[-1]


def _physical_memory_bytes() -> int:
    """Physical memory from the platform service layer (never a fixed guess)."""

    return int(current().memory_status().total_bytes)


def select_execution_tuning(
    hardware: HardwareProfile,
    *,
    logical_cores: int | None = None,
    physical_memory_bytes: int | None = None,
) -> ExecutionTuning:
    """Choose bounded resource values without changing numerical parameters.

    M3 Pro receives the measured overlap profile only when enough memory is
    present.  Every other machine is on a capability-derived generic profile
    of its platform table; unknown future chips never inherit machine-specific
    values by name.  Core and memory facts come from the hardware profile
    (the platform service layer) unless overridden.
    """

    if logical_cores is None:
        logical_cores = hardware.logical_cores if hardware.logical_cores > 1 else None
    cores = max(1, int(logical_cores or os.cpu_count() or 1))
    if physical_memory_bytes is None:
        physical_memory_bytes = hardware.memory_bytes or None
    memory_source = hardware.memory_source if physical_memory_bytes else "platform"
    memory = max(2 * GIB, int(physical_memory_bytes or _physical_memory_bytes()))
    if physical_memory_bytes is None:
        memory_source = current().memory_status().source
    usable = max(1 * GIB, int(memory * 0.62))
    # Fused calibrate+register keeps whole decoded Lights in memory (about
    # 12 bytes per pixel each); the registration budget therefore sets how
    # many Lights are in flight while the native kernels use every core.
    registration = min(max(128 * 1024**2, usable // 6), 4 * GIB)
    integration = min(max(256 * 1024**2, usable // 4), 4 * GIB)
    facts = {
        "registration_memory_bytes": registration,
        "integration_memory_bytes": integration,
        "memory_bytes": memory,
        "memory_source": memory_source,
        "logical_cores": cores,
    }

    if hardware.m3_pro_tuned and memory >= 24 * GIB:
        # Measured on the M3 Pro: eight kernel threads (its six performance
        # cores plus two efficiency cores) were the validated optimum.
        return ExecutionTuning(
            profile_id="apple-m3-pro-tuned-v1",
            cpu_workers=min(8, cores),
            qc_workers=min(8, cores),
            integration_tile_rows=64,
            gpu_inflight_buffers=2,
            local_normalization_sample_stride=2,
            evidence_class="performance-validated-m3-pro",
            kernel_threads=min(8, cores),
            **facts,
        )

    if hardware.apple_silicon:
        band = _band(_APPLE_SILICON_BANDS, memory)
        workers = max(1, min(band.workers, cores))
        return ExecutionTuning(
            profile_id="apple-silicon-generic-v1",
            cpu_workers=workers,
            qc_workers=workers,
            integration_tile_rows=band.tile_rows,
            gpu_inflight_buffers=band.gpu_inflight_buffers,
            local_normalization_sample_stride=band.sample_stride,
            kernel_threads=workers,
            **facts,
        )

    if hardware.cpu_family == CpuFamily.X86_64 and hardware.platform_id in {"windows", "linux"}:
        # Workers are memory-bound (decoded Lights in flight); the native
        # kernels spread each warp and tile over every logical core, including
        # SMT siblings and efficiency cores (dynamic chunking balances them).
        band = _band(_X86_64_BANDS, memory)
        workers = max(1, min(band.workers, cores))
        return ExecutionTuning(
            profile_id=f"{hardware.platform_id}-x86-64-cpu-v1",
            cpu_workers=workers,
            qc_workers=max(1, min(8, cores)) if memory >= 12 * GIB else workers,
            integration_tile_rows=band.tile_rows,
            gpu_inflight_buffers=0,
            local_normalization_sample_stride=band.sample_stride,
            kernel_threads=min(64, cores),
            **facts,
        )

    band = _band(_PORTABLE_BANDS, memory)
    workers = max(1, min(band.workers, cores))
    return ExecutionTuning(
        profile_id="portable-cpu-generic-v1",
        cpu_workers=workers,
        qc_workers=min(2, workers),
        integration_tile_rows=band.tile_rows,
        gpu_inflight_buffers=0,
        local_normalization_sample_stride=band.sample_stride,
        kernel_threads=min(64, cores),
        **facts,
    )


__all__ = ["ExecutionTuning", "GIB", "select_execution_tuning"]
