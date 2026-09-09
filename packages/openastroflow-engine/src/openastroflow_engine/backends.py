from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from .models import json_value


class StageKind(StrEnum):
    CALIBRATION = "CALIBRATION"
    QUALITY_GATE = "QUALITY_GATE"
    REGISTRATION = "REGISTRATION"
    INTEGRATION = "INTEGRATION"
    DRIZZLE = "DRIZZLE"
    SOLVER = "SOLVER"


class DeviceKind(StrEnum):
    CPU = "CPU"
    METAL = "METAL"
    CUDA = "CUDA"
    DIRECTML = "DIRECTML"


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    backend_id: str
    stage: StageKind
    display_name: str
    version: str
    available: bool
    execution_ready: bool
    devices: tuple[DeviceKind, ...] = ()
    capabilities: tuple[str, ...] = ()
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "backendId": self.backend_id,
            "stage": self.stage.value,
            "displayName": self.display_name,
            "version": self.version,
            "available": self.available,
            "executionReady": self.execution_ready,
            "devices": [device.value for device in self.devices],
            "capabilities": list(self.capabilities),
            "reason": self.reason,
            "metadata": json_value(self.metadata, "backend.metadata"),
        }


@runtime_checkable
class Backend(Protocol):
    """Common seam implemented by future native and external executors."""

    @property
    def descriptor(self) -> BackendDescriptor: ...

    def validate_options(self, options: dict[str, Any]) -> tuple[str, ...]: ...


class BackendRegistry:
    def __init__(self, backends: tuple[Backend, ...] | list[Backend] = ()) -> None:
        self._backends: dict[str, Backend] = {}
        for backend in backends:
            self.register(backend)

    def register(self, backend: Backend) -> None:
        backend_id = backend.descriptor.backend_id
        if not backend_id.strip():
            raise ValueError("backend id cannot be empty")
        if backend_id in self._backends:
            raise ValueError(f"duplicate backend id: {backend_id}")
        self._backends[backend_id] = backend

    def get(self, backend_id: str) -> Backend | None:
        return self._backends.get(backend_id)

    def for_stage(self, stage: StageKind) -> tuple[Backend, ...]:
        return tuple(
            backend
            for backend in self._backends.values()
            if backend.descriptor.stage == stage
        )

    def choose(self, stage: StageKind, requested: str) -> Backend | None:
        if requested != "auto":
            backend = self.get(requested)
            return backend if backend and backend.descriptor.stage == stage else None
        candidates = self.for_stage(stage)
        ready = next(
            (
                backend
                for backend in candidates
                if backend.descriptor.available and backend.descriptor.execution_ready
            ),
            None,
        )
        if ready is not None:
            return ready
        available = next(
            (backend for backend in candidates if backend.descriptor.available), None
        )
        return available if available is not None else candidates[0] if candidates else None

    def serializable(self) -> list[dict[str, Any]]:
        return [
            self._backends[key].descriptor.serializable()
            for key in sorted(self._backends)
        ]


__all__ = [
    "Backend",
    "BackendDescriptor",
    "BackendRegistry",
    "DeviceKind",
    "StageKind",
]
