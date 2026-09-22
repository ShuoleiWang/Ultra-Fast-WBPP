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


@runtime_checkable
class DrizzleCapabilityProvider(Backend, Protocol):
    @property
    def drizzle_capabilities(self) -> DrizzleCapabilities: ...


@dataclass(frozen=True, slots=True)
class NativeDrizzleCapabilities:
    descriptor: BackendDescriptor
    drizzle_capabilities: DrizzleCapabilities

    def validate_options(self, options: dict[str, Any]) -> tuple[str, ...]:
        errors: list[str] = []
        unknown = sorted(set(options) - {"scale", "dropShrink", "cfaDrizzle", "kernel"})
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
        kernel = options.get("kernel", "square")
        if not isinstance(kernel, str) or kernel not in {"square", "circular", "gaussian", "point"}:
            errors.append("kernel must be square, circular, gaussian or point")
        return tuple(errors)


def drizzle_backends() -> tuple[NativeDrizzleCapabilities, ...]:
    """The native drizzle backend's declared contract (the execution seam is
    ``drizzle_native``; availability follows the native kernel library)."""

    from .drizzle_native import SUPPORTED_SCALES
    from .native_kernels import DRIZZLE_KERNEL_ID, load_native_kernels

    kernels = load_native_kernels()
    ready = kernels is not None and hasattr(kernels, "drizzle_band")
    capabilities = DrizzleCapabilities(
        scales=SUPPORTED_SCALES,
        drop_shrink_min=0.1,
        drop_shrink_max=1.0,
        supports_cfa_drizzle=True,
        supports_rejection_maps=True,
    )
    return (
        NativeDrizzleCapabilities(
            descriptor=BackendDescriptor(
                backend_id="native-drizzle",
                stage=StageKind.DRIZZLE,
                display_name="Ultra-Fast WBPP Native Drizzle",
                version=DRIZZLE_KERNEL_ID,
                available=ready,
                execution_ready=ready,
                devices=(DeviceKind.CPU,),
                capabilities=(
                    "geometric-drizzle-contract",
                    "drop-shrink-contract",
                    "rejection-map-input-contract",
                    "cfa-drizzle-contract",
                ),
                reason=None if ready else "the native kernel library is not loaded",
                metadata={"drizzle": capabilities.serializable()},
            ),
            drizzle_capabilities=capabilities,
        ),
    )


__all__ = [
    "NativeDrizzleCapabilities",
    "DrizzleBackend",
    "DrizzleCapabilities",
    "drizzle_backends",
]

# Historical capability imports; execution is integrate_drizzle_group.
DrizzleBackend = DrizzleCapabilityProvider
DeclarativeDrizzleBackend = NativeDrizzleCapabilities
