"""The solver backends a run can use, in the planner's ``auto`` order.

The request/result types live in :mod:`.base`; the execution adapters build on
them, and this registry composes the adapters, so the base module never
imports an implementation.
"""

from __future__ import annotations

from ..backends import BackendDescriptor, DeviceKind, StageKind
from .astap import AstapSolverBackend
from .astrometry_net import AstrometryNetSolverBackend
from .base import DeclarativeSolverBackend, SolverBackend


def solver_backends() -> tuple[SolverBackend, ...]:
    # Lazy imports avoid a cycle: execution adapters consume the request/result
    # models defined above, while this registry is the public composition root.
    from .astap import AstapSolverBackend
    from .astrometry_net import AstrometryNetSolverBackend

    common = ("seed-hints", "celestial-wcs", "fail-closed-wcs-validation")
    # Registry order is the planner's ``auto`` preference; keep it aligned with
    # the runtime chain (astrometry-net, astap, native) so the plan names the
    # solver that will actually run first when both are science-ready.
    return (
        AstrometryNetSolverBackend(),
        AstapSolverBackend(),
        DeclarativeSolverBackend(
            BackendDescriptor(
                backend_id="native",
                stage=StageKind.SOLVER,
                display_name="Ultra-Fast WBPP Native Solver",
                version="protocol-v1",
                available=False,
                execution_ready=False,
                devices=(DeviceKind.CPU, DeviceKind.METAL),
                capabilities=common + ("offline-index-provider-seam",),
                reason="The native solver protocol exists, but no native solver implementation is bundled yet.",
            )
        ),
    )


__all__ = ["solver_backends"]
