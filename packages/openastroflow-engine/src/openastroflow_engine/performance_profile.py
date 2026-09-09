"""Capability- and memory-derived execution tuning; never scientific semantics."""

from __future__ import annotations

from dataclasses import dataclass
import os

from .hardware import HardwareProfile


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
        }


def _physical_memory_bytes() -> int:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return 8 * GIB


def select_execution_tuning(
    hardware: HardwareProfile,
    *,
    logical_cores: int | None = None,
    physical_memory_bytes: int | None = None,
) -> ExecutionTuning:
    """Choose bounded resource values without changing numerical parameters.

    M3 Pro receives the measured overlap profile only when enough memory is
    present. Every other Apple chip remains on a capability-derived generic
    profile; unknown future M chips never inherit M3-specific values by name.
    """

    cores = max(1, int(logical_cores or os.cpu_count() or 1))
    memory = max(2 * GIB, int(physical_memory_bytes or _physical_memory_bytes()))
    usable = max(1 * GIB, int(memory * 0.62))
    registration = min(max(128 * 1024**2, usable // 8), 2 * GIB)
    integration = min(max(256 * 1024**2, usable // 4), 4 * GIB)

    if hardware.m3_pro_tuned and memory >= 24 * GIB:
        return ExecutionTuning(
            profile_id="apple-m3-pro-tuned-v1",
            cpu_workers=min(8, cores),
            qc_workers=min(2, cores),
            registration_memory_bytes=registration,
            integration_memory_bytes=integration,
            integration_tile_rows=64,
            gpu_inflight_buffers=2,
            local_normalization_sample_stride=2,
            evidence_class="performance-validated-m3-pro",
        )

    if hardware.apple_silicon:
        if memory < 12 * GIB:
            workers, rows, buffers = 1, 24, 1
        elif memory < 24 * GIB:
            workers, rows, buffers = min(2, cores), 32, 1
        else:
            workers, rows, buffers = min(4, cores), 48, 2
        return ExecutionTuning(
            profile_id="apple-silicon-generic-v1",
            cpu_workers=workers,
            qc_workers=min(2, workers),
            registration_memory_bytes=registration,
            integration_memory_bytes=integration,
            integration_tile_rows=rows,
            gpu_inflight_buffers=buffers,
            local_normalization_sample_stride=2 if memory >= 16 * GIB else 3,
        )

    windows = hardware.operating_system.casefold() == "windows"
    workers = min(4 if memory >= 16 * GIB else 2, cores)
    return ExecutionTuning(
        profile_id="windows-cpu-generic-v1" if windows else "portable-cpu-generic-v1",
        cpu_workers=max(1, workers),
        qc_workers=min(2, max(1, workers)),
        registration_memory_bytes=registration,
        integration_memory_bytes=integration,
        integration_tile_rows=32 if memory < 16 * GIB else 48,
        gpu_inflight_buffers=0,
        local_normalization_sample_stride=3,
    )


__all__ = ["ExecutionTuning", "GIB", "select_execution_tuning"]
