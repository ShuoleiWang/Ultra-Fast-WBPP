"""Unattended selection decisions from features, guards and counterfactuals."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .counterfactual import CounterfactualReport, FrameCounterfactual
from .features import FrameSelectionFeatures
from .guards import HARD_GATE_CODES, INSUFFICIENT_EVIDENCE_CODES, hard_exclusion_reasons
from .parameters import SelectionParameters
from .region import REGION_WEIGHT_ALGORITHM, RegionWeightMap
from .weights import psf_factors

ACTION_EXCLUDE = "EXCLUDE"
ACTION_KEEP = "KEEP"
# A region weight map that blanks more than this fraction of the field leaves
# too little of the frame to be worth its rejection and normalization cost.
MOSTLY_BLANK_FRACTION = 0.5
# Defects a region weight map handles inside the frame: the blocked or dimmed
# cells get weight zero or a reduced weight, the rest of the frame is clean.
REGION_HANDLED_CODES: frozenset[str] = frozenset(
    {"GATE_OCCLUSION_REVIEW", "GATE_SPATIAL_DIMMING_REVIEW", "GATE_SPATIAL_DIMMING_STRONG"}
)
# Uniform transparency evidence the normalization scale models exactly: a
# night dimmer than the cohort's clear envelope, or a frame a little below
# the nightly airmass envelope.  The transparency and extinction guards bound
# the loss and the noise weight already scales with it, so these codes cost
# no confidence.
NORMALIZATION_MODELLED_CODES: frozenset[str] = frozenset(
    {"GATE_NIGHT_ZEROPOINT_SHIFT_REVIEW", "GATE_TEMPORAL_EXTINCTION_REVIEW"}
)


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    path: str
    action: str
    confidence: float
    reasons: tuple[tuple[str, str], ...]
    soft: bool = False
    psf_factor: float = 1.0
    counterfactual: FrameCounterfactual | None = None
    suggestion: str | None = None

    @property
    def admitted(self) -> bool:
        return self.action != ACTION_EXCLUDE

    @property
    def weight_multiplier(self) -> float:
        return 0.0 if self.action == ACTION_EXCLUDE else self.confidence * self.psf_factor

    def serializable(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "action": self.action,
            "confidence": self.confidence,
            "psfFactor": self.psf_factor,
            "weightMultiplier": self.weight_multiplier,
            "soft": self.soft,
            "reasons": [{"code": code, "message": message} for code, message in self.reasons],
            "counterfactual": (
                self.counterfactual.serializable() if self.counterfactual is not None else None
            ),
            "suggestion": self.suggestion,
        }


def _psf_exclusion(
    features: FrameSelectionFeatures, parameters: SelectionParameters
) -> tuple[tuple[str, str], ...]:
    """Reasons to exclude a frame whose PSF is beyond the priority cut-offs."""

    profile = parameters.profile
    scale = parameters.threshold_scale
    night_cut = profile.fwhm_night_exclusion_ratio * scale
    group_cut = profile.fwhm_group_exclusion_ratio * scale
    if features.fwhm_night_ratio is not None and features.fwhm_night_ratio > night_cut:
        return (
            (
                "SEL_PSF_BEYOND_NIGHT_CUTOFF",
                f"FWHM is {features.fwhm_night_ratio:.2f}x the best of its night "
                f"(cut-off {night_cut:.2f} for priority {profile.priority})",
            ),
        )
    if features.fwhm_group_ratio is not None and features.fwhm_group_ratio > group_cut:
        return (
            (
                "SEL_PSF_BEYOND_GROUP_CUTOFF",
                f"FWHM is {features.fwhm_group_ratio:.2f}x the best of its group "
                f"(cut-off {group_cut:.2f} for priority {profile.priority})",
            ),
        )
    return ()


def _occlusion_exclusion(
    features: FrameSelectionFeatures, parameters: SelectionParameters, *, mapped: bool
) -> tuple[tuple[str, str], ...]:
    """A large occluded region is excluded unless a region weight map blanks it.

    Pixel rejection does not remove a blocked area from the mean once it is a
    sizeable minority of the stack, and the block-noise weight even rewards
    the low-noise blocked pixels, so the frame would dominate the master.
    """

    if mapped:
        return ()
    area = features.occlusion_area
    if area is None or area < 0.15:
        return ()
    density = features.occlusion_density
    if density is not None and density > 0.50:
        return ()
    return (
        (
            "SEL_OCCLUSION_AREA_EXCLUDED",
            f"a connected region covering {area:.0%} of the field lost its stars "
            "(no region weight map blanks it)",
        ),
    )


def _defect_codes(features: FrameSelectionFeatures) -> list[str]:
    """Gate codes that name a defect (not a guard, not missing evidence)."""

    return [
        code
        for code in (*features.review_codes, *features.hard_fail_codes)
        if code not in INSUFFICIENT_EVIDENCE_CODES and code not in HARD_GATE_CODES
    ]


def _gray_zone(
    features: FrameSelectionFeatures, parameters: SelectionParameters, *, mapped: bool
) -> tuple[str, float, tuple[tuple[str, str], ...]]:
    """Decide a frame whose gate evidence names a defect the pipeline can model.

    A mapped frame's region-handled defects cost no confidence: the map blanks
    or reduces the affected cells, so only the remaining defects decide.
    """

    reasons: list[tuple[str, str]] = []
    remaining: list[str] = []
    for code in _defect_codes(features):
        if code in NORMALIZATION_MODELLED_CODES:
            reasons.append(
                (
                    "SEL_KEEP_NORMALIZED_" + code,
                    f"kept: normalization models {code}; the transparency guards bound it",
                )
            )
            continue
        if mapped and code in REGION_HANDLED_CODES:
            reasons.append(
                (
                    "SEL_KEEP_REGION_WEIGHTED_" + code,
                    f"kept: the region weight map handles {code}",
                )
            )
            continue
        remaining.append(code)
    if not remaining:
        return ACTION_KEEP, 1.0, tuple(reasons)
    confidence = parameters.defect_review_confidence
    for code in remaining:
        if code in {
            "GATE_TRAILING_REVIEW",
            "GATE_OCCLUSION_REVIEW",
            "GATE_REGISTRATION_REVIEW",
            "GATE_MULTI_FAMILY_CLOUD_HARD",
        }:
            confidence = min(confidence, 0.5)
        reasons.append(("SEL_KEEP_DOWNWEIGHTED_" + code, f"kept with reduced weight: {code}"))
    return ACTION_KEEP, confidence, tuple(reasons)


def decide(
    features: Sequence[FrameSelectionFeatures],
    parameters: SelectionParameters,
    *,
    approved_paths: Iterable[str] = (),
    region_maps: Mapping[str, RegionWeightMap] | None = None,
) -> list[SelectionDecision]:
    """Decide every frame under ``parameters.policy`` (never ``legacy-gate``).

    ``region_maps`` are the frames' region weight maps by path.  A mapped
    frame keeps its clean area instead of being excluded for a blocked or
    dimmed region, unless the map blanks more than half of the field.
    """

    parameters.validate()
    if not parameters.unattended:
        raise ValueError("decide() is only defined for unattended policies")
    approved = {str(path) for path in approved_paths}
    maps = dict(region_maps or {})
    factors = psf_factors(features, parameters)
    baselines: dict[str, float | None] = {}
    for group_id in {item.group_id for item in features}:
        clear = [
            item.transparency
            for item in features
            if item.group_id == group_id
            and item.gate_disposition == "PASS"
            and item.transparency is not None
        ]
        if not clear:
            clear = [
                item.transparency
                for item in features
                if item.group_id == group_id and item.transparency is not None
            ]
        baselines[group_id] = float(np.median(clear)) if clear else None
    decisions: list[SelectionDecision] = []
    for item in features:
        guards = hard_exclusion_reasons(
            item, parameters, transparency_baseline=baselines.get(item.group_id)
        )
        if guards:
            decisions.append(SelectionDecision(item.path, ACTION_EXCLUDE, 0.0, guards))
            continue
        if item.path in approved:
            decisions.append(
                SelectionDecision(
                    item.path, ACTION_KEEP, 1.0, (("SEL_APPROVED", "explicitly approved"),)
                )
            )
            continue
        if parameters.policy == "include-all":
            decisions.append(
                SelectionDecision(
                    item.path, ACTION_KEEP, 1.0, (("SEL_INCLUDE_ALL", "diagnostic policy"),)
                )
            )
            continue
        psf_reasons = _psf_exclusion(item, parameters)
        if psf_reasons:
            decisions.append(
                SelectionDecision(item.path, ACTION_EXCLUDE, 0.0, psf_reasons, soft=True)
            )
            continue
        region_map = maps.get(item.path)
        if region_map is not None and region_map.zero_fraction > MOSTLY_BLANK_FRACTION:
            decisions.append(
                SelectionDecision(
                    item.path,
                    ACTION_EXCLUDE,
                    0.0,
                    (
                        (
                            "SEL_REGION_MOSTLY_BLANK",
                            "the region weight map blanks "
                            f"{region_map.zero_fraction:.0%} of the field",
                        ),
                    ),
                    soft=True,
                )
            )
            continue
        occlusion_reasons = _occlusion_exclusion(
            item, parameters, mapped=region_map is not None
        )
        if occlusion_reasons:
            decisions.append(
                SelectionDecision(item.path, ACTION_EXCLUDE, 0.0, occlusion_reasons, soft=True)
            )
            continue
        factor = factors.get(item.path, 1.0)
        defect_codes = _defect_codes(item)
        if item.gate_disposition == "PASS" and not defect_codes:
            decisions.append(SelectionDecision(item.path, ACTION_KEEP, 1.0, (), psf_factor=factor))
            continue
        if not defect_codes:
            decisions.append(
                SelectionDecision(
                    item.path,
                    ACTION_KEEP,
                    parameters.insufficient_evidence_confidence,
                    tuple(
                        ("SEL_KEEP_INSUFFICIENT_EVIDENCE_" + code, "kept: evidence insufficient, not a defect")
                        for code in item.review_codes
                    )
                    or (("SEL_KEEP_INSUFFICIENT_EVIDENCE", "kept: evidence insufficient"),),
                    psf_factor=factor,
                )
            )
            continue
        action, confidence, reasons = _gray_zone(
            item, parameters, mapped=region_map is not None
        )
        decisions.append(
            SelectionDecision(item.path, action, confidence, reasons, psf_factor=factor)
        )

    # Soft-exclusion guard: rule-based exclusions above the guard fraction are
    # downgraded to reduced-weight inclusion until a counterfactual confirms them.
    total = len(decisions)
    soft = [index for index, item in enumerate(decisions) if item.soft]
    if total and len(soft) / total > parameters.soft_exclusion_fraction_guard:
        for index in soft:
            item = decisions[index]
            decisions[index] = replace(
                item,
                action=ACTION_KEEP,
                confidence=0.5,
                reasons=item.reasons
                + (
                    (
                        "SEL_SOFT_EXCLUSION_GUARD",
                        f"{len(soft)} of {total} frames would be excluded by rules; kept with reduced weight",
                    ),
                ),
            )
    return decisions


def _outlier_threshold(values: Sequence[float | None], multiplier: float) -> float | None:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)])
    if finite.size < 3:
        return None
    median = float(np.median(finite))
    madn = float(1.4826 * np.median(np.abs(finite - median)))
    return median + multiplier * max(madn, 1e-6)


def annotate_with_counterfactual(
    decisions: Sequence[SelectionDecision],
    report: CounterfactualReport | None,
    parameters: SelectionParameters,
) -> list[SelectionDecision]:
    """Attach counterfactual evidence and a suggestion; the action is unchanged.

    A frame is suggested for exclusion only when the tile bootstrap had enough
    tiles, its interval excludes the harm threshold, and it stands out from its
    group (the leave-one-out deltas of a homogeneous group are all alike, so an
    estimator bias cannot flag every frame).
    """

    if report is None:
        return list(decisions)
    by_path = {frame.path: frame for frame in report.frames}
    enough_tiles = report.tiles_used >= parameters.minimum_counterfactual_tiles
    depth_outlier = _outlier_threshold(
        [frame.delta_depth_mag for frame in report.frames], parameters.outlier_madn
    )
    background_outlier = _outlier_threshold(
        [frame.delta_background_sigma for frame in report.frames], parameters.outlier_madn
    )
    fwhm_outlier = _outlier_threshold(
        [frame.delta_fwhm_px for frame in report.frames], parameters.outlier_madn
    )
    annotated: list[SelectionDecision] = []
    for item in decisions:
        frame = by_path.get(item.path)
        if frame is None:
            annotated.append(item)
            continue
        suggestion = None
        depth_low = frame.delta_depth_ci[0] if frame.delta_depth_ci else None
        depth_high = frame.delta_depth_ci[1] if frame.delta_depth_ci else None
        background_low = frame.delta_background_ci[0] if frame.delta_background_ci else None
        background_high = frame.delta_background_ci[1] if frame.delta_background_ci else None
        depth_harmful = (
            depth_low is not None
            and depth_low > parameters.harmful_depth_mag
            and frame.delta_depth_mag is not None
            and depth_outlier is not None
            and frame.delta_depth_mag > depth_outlier
        )
        background_harmful = (
            background_low is not None
            and background_low > parameters.harmful_background_sigma
            and frame.delta_background_sigma is not None
            and background_outlier is not None
            and frame.delta_background_sigma > background_outlier
        )
        fwhm_harmful = (
            parameters.priority == "resolution"
            and frame.delta_fwhm_px is not None
            and frame.delta_fwhm_px > parameters.harmful_fwhm_px
            and fwhm_outlier is not None
            and frame.delta_fwhm_px > fwhm_outlier
        )
        depth_beneficial = (
            depth_high is not None and depth_high < -parameters.beneficial_depth_mag
        )
        # Background structure is a tie-breaker: a frame that measurably
        # deepens the master is not removed for a background residual the
        # depth gain outweighs.
        harmful = enough_tiles and (
            depth_harmful or fwhm_harmful or (background_harmful and not depth_beneficial)
        )
        beneficial = depth_beneficial and (background_high is None or background_high < 0)
        if harmful:
            suggestion = "EXCLUDE_CONFIRMED_HARMFUL"
        elif beneficial and item.confidence < 1.0:
            suggestion = "RESTORE_FULL_WEIGHT"
        annotated.append(replace(item, counterfactual=frame, suggestion=suggestion))
    return annotated


def confirmed_harmful(decisions: Sequence[SelectionDecision]) -> list[SelectionDecision]:
    """Admitted frames whose counterfactual confirmed they harm the master."""

    return [
        item
        for item in decisions
        if item.admitted and item.suggestion == "EXCLUDE_CONFIRMED_HARMFUL"
    ]


def exclude_confirmed(
    decisions: Sequence[SelectionDecision], harmful: Iterable[str]
) -> list[SelectionDecision]:
    """Return decisions with the listed frames excluded by the counterfactual."""

    targets = {str(path) for path in harmful}
    result: list[SelectionDecision] = []
    for item in decisions:
        if item.path in targets:
            result.append(
                replace(
                    item,
                    action=ACTION_EXCLUDE,
                    reasons=item.reasons
                    + (
                        (
                            "SEL_COUNTERFACTUAL_CONFIRMED_HARMFUL",
                            "excluded: the master measured better without this frame",
                        ),
                    ),
                    suggestion=None,
                )
            )
        else:
            result.append(item)
    return result


def selection_receipt(
    parameters: SelectionParameters,
    features: Sequence[FrameSelectionFeatures],
    decisions: Sequence[SelectionDecision],
    reports: dict[str, CounterfactualReport] | None = None,
    region_maps: Mapping[str, RegionWeightMap] | None = None,
) -> dict[str, Any]:
    """The ``qualityControl.selection`` receipt block."""

    counts = {ACTION_KEEP: 0, ACTION_EXCLUDE: 0, "KEEP_REDUCED_WEIGHT": 0}
    for item in decisions:
        if item.action == ACTION_EXCLUDE:
            counts[ACTION_EXCLUDE] += 1
        elif item.confidence < 1.0:
            counts["KEEP_REDUCED_WEIGHT"] += 1
        else:
            counts[ACTION_KEEP] += 1
    return {
        "parameters": parameters.describe(),
        "counts": counts,
        "frames": [item.serializable() for item in decisions],
        "features": [item.serializable() for item in features],
        "counterfactual": (
            {name: report.serializable() for name, report in sorted(reports.items())}
            if reports
            else None
        ),
        "regionWeights": (
            {
                "algorithm": REGION_WEIGHT_ALGORITHM,
                "frames": [region_maps[path].serializable() for path in sorted(region_maps)],
            }
            if region_maps
            else None
        ),
    }
