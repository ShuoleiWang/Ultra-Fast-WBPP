"""Hard exclusion guards: defects the pipeline cannot model or recover."""

from __future__ import annotations

from .features import FrameSelectionFeatures
from .parameters import SelectionParameters

# Quality-gate codes that always exclude a frame: not a usable Light, broken
# measurement or identity, or a defect that destroys the PSF/rejection model.
HARD_GATE_CODES: frozenset[str] = frozenset(
    {
        "GATE_MEASUREMENT_MISSING",
        "GATE_MEASUREMENT_FAILED",
        "GATE_NOT_A_LIGHT_FRAME",
        "GATE_IDENTITY_MISSING",
        "GATE_IDENTITY_MISMATCH",
        "GATE_FINITE_FRACTION_INVALID",
        "GATE_FINITE_FRACTION_HARD",
        "GATE_NEAR_CONSTANT_IMAGE",
        "GATE_FRAGMENTED_TRAILING_HARD",
        "GATE_COHERENT_TRAILING_HARD",
        # Until region weight maps exist, a hard occlusion cannot be used partially.
        "GATE_OCCLUSION_HARD",
        # GATE_MULTI_FAMILY_CLOUD_HARD is deliberately absent: a uniform
        # transparency loss is modelled by the normalization scale, so it is a
        # gray-zone defect; the transparency and extinction guards below bound it.
    }
)

# REVIEW codes that mean "not enough evidence", not "this frame is defective".
INSUFFICIENT_EVIDENCE_CODES: frozenset[str] = frozenset(
    {
        "GATE_INSUFFICIENT_COHORT",
        "GATE_NIGHT_UNRESOLVED",
        "GATE_INSUFFICIENT_NIGHT_BASELINE",
        "GATE_MORPHOLOGY_SAMPLE_REVIEW",
        "GATE_SOURCE_COUNT_MISSING",
        "GATE_FINITE_FRACTION_MISSING",
        "GATE_FINITE_FRACTION_REVIEW",
        "GATE_DYNAMIC_RANGE_MISSING",
        "GATE_COMMON_FOOTPRINT_REVIEW",
        "GATE_REFERENCE_NOT_CONNECTED",
        "GATE_ABSOLUTE_LOW_SOURCE_COUNT",
    }
)


def hard_exclusion_reasons(
    features: FrameSelectionFeatures,
    parameters: SelectionParameters,
    *,
    transparency_baseline: float | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return (code, message) pairs for every guard the frame trips.

    The QC transparency ratio is relative to the group's star-richest frame,
    which is not necessarily a clear-sky frame, so the floor compares the
    frame with ``transparency_baseline`` (the median ratio of the group's
    PASS frames) when one is known.
    """

    reasons: list[tuple[str, str]] = []
    for code in features.hard_fail_codes:
        if code in HARD_GATE_CODES:
            reasons.append((f"SEL_GUARD_{code}", f"quality gate hard failure {code}"))
    if not features.registration_ok and not features.registration_estimated:
        reasons.append(
            (
                "SEL_GUARD_REGISTRATION_FAILED",
                "the frame could not be registered to the group reference",
            )
        )
    if features.transparency is not None:
        baseline = (
            transparency_baseline
            if transparency_baseline is not None and transparency_baseline > 0
            else 1.0
        )
        relative = features.transparency / baseline
        if relative < parameters.transparency_floor:
            reasons.append(
                (
                    "SEL_GUARD_TRANSPARENCY_BELOW_NORMALIZATION_FLOOR",
                    f"stellar transparency {features.transparency:.2f} is "
                    f"{relative:.2f}x the group's clear-frame median, below the "
                    f"normalization floor {parameters.transparency_floor:.2f}",
                )
            )
    if (
        features.extra_extinction_mag is not None
        and features.extra_extinction_mag > parameters.extra_extinction_exclusion_mag
    ):
        reasons.append(
            (
                "SEL_GUARD_EXTRA_EXTINCTION",
                f"extra extinction {features.extra_extinction_mag:.2f} mag exceeds "
                f"{parameters.extra_extinction_exclusion_mag:.2f} mag",
            )
        )
    return tuple(reasons)
