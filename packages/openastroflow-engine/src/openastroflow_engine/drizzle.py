from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .backends import Backend, BackendDescriptor, DeviceKind, StageKind


@dataclass(frozen=True, slots=True)
class DrizzleCapabilities:
    scales: tuple[int, ...]
    drop_shrink_min: float
    drop_shrink_max: float
    supports_cfa_drizzle: bool
    supports_rejection_maps: bool

    def serializable(self) -> dict[str, Any]:
        return {
            "scales": list(self.scales),
            "dropShrink": {"minimum": self.drop_shrink_min, "maximum": self.drop_shrink_max},
            "supportsCfaDrizzle": self.supports_cfa_drizzle,
            "supportsRejectionMaps": self.supports_rejection_maps,
        }


@dataclass(frozen=True, slots=True)
class DrizzleRequest:
    registered_inputs: tuple[str, ...]
    output_path: str
    scale: int
    drop_shrink: float
    cfa_drizzle: bool = False


@dataclass(frozen=True, slots=True)
class DrizzleResult:
    completed: bool
    output_path: str | None = None
    error: str | None = None


@runtime_checkable
class DrizzleBackend(Backend, Protocol):
    @property
    def drizzle_capabilities(self) -> DrizzleCapabilities: ...

    def drizzle(self, request: DrizzleRequest) -> DrizzleResult: ...


@dataclass(frozen=True, slots=True)
class DeclarativeDrizzleBackend:
    descriptor: BackendDescriptor
    drizzle_capabilities: DrizzleCapabilities

    def validate_options(self, options: dict[str, Any]) -> tuple[str, ...]:
        errors: list[str] = []
        unknown = sorted(set(options) - {"scale", "dropShrink", "cfaDrizzle"})
        errors.extend(f"unknown drizzle option: {key}" for key in unknown)
        scale = options.get("scale", 2)
        drop_shrink = options.get("dropShrink", 0.9)
        cfa_drizzle = options.get("cfaDrizzle", False)
        if isinstance(scale, bool) or scale not in self.drizzle_capabilities.scales:
            errors.append(
                f"scale {scale!r} is unsupported; expected one of {self.drizzle_capabilities.scales}"
            )
        if isinstance(drop_shrink, bool) or not isinstance(drop_shrink, (int, float)):
            errors.append("dropShrink must be a number")
        elif not (
            self.drizzle_capabilities.drop_shrink_min
            <= float(drop_shrink)
            <= self.drizzle_capabilities.drop_shrink_max
        ):
            errors.append(
                "dropShrink is outside backend range "
                f"[{self.drizzle_capabilities.drop_shrink_min}, "
                f"{self.drizzle_capabilities.drop_shrink_max}]"
            )
        if cfa_drizzle and not self.drizzle_capabilities.supports_cfa_drizzle:
            errors.append("CFA drizzle is unsupported by this backend")
        return tuple(errors)

    def drizzle(self, request: DrizzleRequest) -> DrizzleResult:
        return DrizzleResult(
            completed=False,
            error=self.descriptor.reason or "drizzle execution backend is unavailable",
        )


def drizzle_backends() -> tuple[DeclarativeDrizzleBackend, ...]:
    capabilities = DrizzleCapabilities(
        scales=(1, 2, 3),
        drop_shrink_min=0.1,
        drop_shrink_max=1.0,
        supports_cfa_drizzle=False,
        supports_rejection_maps=True,
    )
    return (
        DeclarativeDrizzleBackend(
            descriptor=BackendDescriptor(
                backend_id="native-drizzle",
                stage=StageKind.DRIZZLE,
                display_name="Ultra-Fast WBPP Native Drizzle",
                version="protocol-v1",
                available=False,
                execution_ready=False,
                devices=(DeviceKind.CPU, DeviceKind.METAL),
                capabilities=(
                    "geometric-drizzle-contract",
                    "drop-shrink-contract",
                    "rejection-map-input-contract",
                ),
                reason="The drizzle contract is implemented, but the pixel executor is not bundled in this release.",
                metadata={"drizzle": capabilities.serializable()},
            ),
            drizzle_capabilities=capabilities,
        ),
    )


__all__ = [
    "DeclarativeDrizzleBackend",
    "DrizzleBackend",
    "DrizzleCapabilities",
    "DrizzleRequest",
    "DrizzleResult",
    "drizzle_backends",
]
