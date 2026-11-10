"""The ``unattended-v2`` policy: cross-night evidence, a value model and nights as blocks.

v2 decides every frame from the quality gate, the blink flags' cross-night
channel statistics and a value ``V`` (the blink reference proxy relative to
the channel's clean frames).  Firm rules exclude what the pipeline cannot
model (geometry, trails, non-uniform cloud, a uniformly moonlit night); soft
rules (PSF, negligible value) are value-ordered and never take a group below
``max(2, ceil(n / 2))``; uniform transparency loss is kept to a stellar scale
of 4 and weighted by the measured noise.  Nights whose frames look suspect
become blocks the integration measures with a leave-block-out counterfactual.
Every threshold is a constant of this module and is written to the receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from lightframeqc.blink_flags import BlinkFrameFlags, BlinkFrameInput, ChannelStatistics

from .counterfactual import CounterfactualReport
from .features import FrameSelectionFeatures
from .guards import HARD_GATE_CODES, INSUFFICIENT_EVIDENCE_CODES
from .parameters import SelectionParameters
from .region import RegionWeightMap

V2_ALGORITHM = "unattended-selection-v2"
POLICY_V2 = "unattended-v2"


@dataclass(frozen=True, slots=True)
class V2Thresholds:
    """Every threshold of the policy; not recipe keys, written to the receipt."""

    # Firm guards.
    minimum_overlap: float = 0.90
    minimum_stars: int = 20
    extra_hard_codes: tuple[str, ...] = (
        "GATE_MULTI_FAMILY_CLOUD_HARD",
        "GATE_SPATIAL_DIMMING_STRONG",
        "GATE_SOURCE_RETENTION_STRONG",
    )
    # Cross-night: the blink combined rule (sky >= 1.6 x skyClean and source
    # ratio <= 0.60) excludes; a bright sky alone is suspect from 1.3 x.
    sky_suspect_ratio: float = 1.3
    # Non-uniform sky: a background-shape change together with a brighter sky.
    background_shape: float = 0.50
    region_map_maximum_blank: float = 0.50
    occlusion_area_exclusion: float = 0.15
    occlusion_density_floor: float = 0.50
    # Uniform-loss floors (stellar scale 4 = 1.5 mag; the blink exclude level).
    transparency_floor: float = 0.25
    source_ratio_floor: float = 0.45
    # Value V = S / S_median over the clean set (S the blink reference proxy).
    value_floor: float = 0.06
    value_suspect: float = 0.36
    # PSF cut-off, FWHM over the clean median, per priority.
    psf_cutoff: Mapping[str, float] = field(
        default_factory=lambda: {"depth": 1.6, "balanced": 1.3, "resolution": 1.2}
    )
    # Exponent p of the weight term (F_median / F) ** (2 p), per priority.  The
    # cut-off already bounds the PSF spread, and within it a FWHM^-2 weight
    # costs more noise than it gains in FWHM (RCS2 R/G/B, the same frames:
    # 1.6-2.7 % sharper, 2.5-4.1 % noisier), so only ``resolution`` trades.
    psf_exponent: Mapping[str, float] = field(
        default_factory=lambda: {"depth": 0.0, "balanced": 0.0, "resolution": 1.0}
    )
    trailing_ellipticity: float = 0.35
    trailing_coherence: float = 0.25
    # Source retention is explained by transparency up to this relative value.
    retention_explained_transparency: float = 0.80
    confidence_suspect: float = 0.5
    confidence_shape: float = 0.7
    confidence_night_outlier: float = 0.7
    # Soft exclusions never take a group below max(minimum, ceil(fraction * n)).
    soft_guard_minimum: int = 2
    soft_guard_fraction: float = 0.5
    # Blocks (nights, split at a gap) measured by the leave-block-out counterfactual.
    block_gap_hours: float = 2.0
    block_maximum_per_group: int = 4
    block_suspect_fraction: float = 0.5
    block_harmful_background_sigma: float = 0.25
    block_depth_cost_mag: float = 0.010
    block_beneficial_depth_mag: float = 0.005
    minimum_counterfactual_tiles: int = 16

    def describe(self) -> dict[str, Any]:
        return {
            "algorithm": V2_ALGORITHM,
            "minimumOverlap": self.minimum_overlap,
            "minimumStars": self.minimum_stars,
            "extraHardCodes": list(self.extra_hard_codes),
            "skySuspectRatio": self.sky_suspect_ratio,
            "backgroundShape": self.background_shape,
            "regionMapMaximumBlank": self.region_map_maximum_blank,
            "occlusionAreaExclusion": self.occlusion_area_exclusion,
            "occlusionDensityFloor": self.occlusion_density_floor,
            "transparencyFloor": self.transparency_floor,
            "sourceRatioFloor": self.source_ratio_floor,
            "valueFloor": self.value_floor,
            "valueSuspect": self.value_suspect,
            "psfCutoff": dict(self.psf_cutoff),
            "psfExponent": dict(self.psf_exponent),
            "trailingEllipticity": self.trailing_ellipticity,
            "trailingCoherence": self.trailing_coherence,
            "retentionExplainedTransparency": self.retention_explained_transparency,
            "confidenceSuspect": self.confidence_suspect,
            "confidenceShape": self.confidence_shape,
            "confidenceNightOutlier": self.confidence_night_outlier,
            "softGuardMinimum": self.soft_guard_minimum,
            "softGuardFraction": self.soft_guard_fraction,
            "blockGapHours": self.block_gap_hours,
            "blockMaximumPerGroup": self.block_maximum_per_group,
            "blockSuspectFraction": self.block_suspect_fraction,
            "blockHarmfulBackgroundSigma": self.block_harmful_background_sigma,
            "blockDepthCostMag": self.block_depth_cost_mag,
            "blockBeneficialDepthMag": self.block_beneficial_depth_mag,
            "minimumCounterfactualTiles": self.minimum_counterfactual_tiles,
            "gradientAmplitude": "not measured in-run",
        }


V2_THRESHOLDS = V2Thresholds()


@dataclass(frozen=True, slots=True)
class BlinkContext:
    """Blink inputs, flags and channel statistics keyed by resolved path / channel."""

    inputs: Mapping[str, BlinkFrameInput]
    flags: Mapping[str, BlinkFrameFlags]
    statistics: Mapping[str, ChannelStatistics]


@dataclass(frozen=True, slots=True)
class SuspectBlock:
    """A night (or part of one) of a group that the integration measures as a set."""

    group_id: str
    name: str
    paths: tuple[str, ...]
    reasons: tuple[str, ...]
    median_value: float | None
    median_sky_ratio: float | None

    def serializable(self) -> dict[str, Any]:
        return {
            "groupId": self.group_id,
            "name": self.name,
            "paths": list(self.paths),
            "reasons": list(self.reasons),
            "medianValue": self.median_value,
            "medianSkyRatio": self.median_sky_ratio,
        }


@dataclass(frozen=True, slots=True)
class ChannelSummary:
    group_id: str
    target: str
    filter_name: str
    frame_count: int
    clean_count: int
    sky_clean: float | None
    fwhm_median: float | None
    s_median: float | None
    transparency_median: float | None
    sources_best: float | None
    blocks: int

    def serializable(self) -> dict[str, Any]:
        return {
            "groupId": self.group_id,
            "target": self.target,
            "filter": self.filter_name,
            "frameCount": self.frame_count,
            "cleanSetSize": self.clean_count,
            "baselines": {
                "skyClean": self.sky_clean if self.sky_clean is not None else "unavailable (fewer than 3 clean frames)",
                "fwhmMedian": self.fwhm_median,
                "sMedian": self.s_median,
                "transparencyMedian": self.transparency_median,
                "sourcesBest": self.sources_best,
            },
            "blocks": self.blocks,
        }


def _resolved(path: str) -> str:
    try:
        return str(Path(path).resolve(strict=True))
    except OSError:
        return str(Path(path))


def blink_context(
    inputs: Iterable[BlinkFrameInput],
    flags: Iterable[BlinkFrameFlags],
    statistics: Mapping[str, ChannelStatistics],
) -> BlinkContext:
    """Key blink evidence by resolved path so it joins the selection features."""

    return BlinkContext(
        inputs={_resolved(item.path): item for item in inputs},
        flags={_resolved(item.path): item for item in flags},
        statistics=dict(statistics),
    )


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _median(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.median(finite)) if finite else None


def _metric(flags: BlinkFrameFlags | None, key: str) -> float | None:
    return _finite(flags.metrics.get(key)) if flags is not None else None


def _raw_value(
    item: FrameSelectionFeatures,
    blink: BlinkFrameInput | None,
    fwhm_median: float | None,
    sources_best: float | None,
) -> float | None:
    """The blink reference proxy with the clean median FWHM as the PSF anchor."""

    if blink is None or fwhm_median is None or sources_best is None or sources_best <= 0:
        return None
    transparency = item.transparency
    noise = blink.sky_mad
    fwhm = item.fwhm_native
    if (
        transparency is None
        or noise is None
        or fwhm is None
        or transparency <= 0
        or noise <= 0
        or fwhm <= 0
    ):
        return None
    roundness = max(0.0, min(1.0, 1.0 - (item.ellipticity or 0.0)))
    completeness = min(1.0, item.star_count / sources_best)
    value = (transparency / (1.4826 * noise)) ** 2 * (fwhm_median / fwhm) ** 2 * roundness * completeness
    return value if math.isfinite(value) and value > 0 else None


def _observed(blink: BlinkFrameInput | None) -> datetime | None:
    if blink is None or not blink.observed_at:
        return None
    try:
        moment = datetime.fromisoformat(str(blink.observed_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    # Naive UTC throughout, so aware and naive stamps compare.
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


def _blocks_of_group(
    members: Sequence[FrameSelectionFeatures],
    context: BlinkContext,
    thresholds: V2Thresholds,
) -> dict[str, str]:
    """Block name per path: the observing night, split where a gap exceeds the bound."""

    by_night: dict[str, list[FrameSelectionFeatures]] = {}
    for item in members:
        by_night.setdefault(item.night_id or "UNKNOWN", []).append(item)
    names: dict[str, str] = {}
    for night, items in by_night.items():
        timed = sorted(
            items,
            key=lambda item: (
                _observed(context.inputs.get(item.path)) or datetime.max.replace(tzinfo=None),
                item.path,
            ),
        )
        part = 1
        previous: datetime | None = None
        for item in timed:
            moment = _observed(context.inputs.get(item.path))
            if (
                previous is not None
                and moment is not None
                and (moment - previous).total_seconds() > thresholds.block_gap_hours * 3600.0
            ):
                part += 1
            if moment is not None:
                previous = moment
            names[item.path] = night if part == 1 else f"{night}#{part}"
    return names


def _clean_set(
    members: Sequence[FrameSelectionFeatures],
    context: BlinkContext,
    stats: ChannelStatistics | None,
) -> list[FrameSelectionFeatures]:
    """Blink's clean set; falls back to the gate's PASS frames, then to the registered ones."""

    sources_best = stats.sources_best if stats is not None else None
    clean: list[FrameSelectionFeatures] = []
    for item in members:
        blink = context.inputs.get(item.path)
        if blink is None or item.gate_disposition == "HARD_FAIL" or not item.registration_ok:
            continue
        extinction = blink.extinction_mag
        ratio = item.star_count / sources_best if sources_best else None
        if (
            extinction is not None
            and extinction < 0.35
            and ratio is not None
            and ratio >= 0.60
            and blink.sky is not None
        ):
            clean.append(item)
    if not clean:
        clean = [item for item in members if item.gate_disposition == "PASS" and item.registration_ok]
    if not clean:
        clean = [item for item in members if item.registration_ok or item.registration_estimated]
    return clean


# --- per-frame decision ----------------------------------------------------


def _firm_reasons(
    item: FrameSelectionFeatures,
    blink: BlinkFrameInput | None,
    flags: BlinkFrameFlags | None,
    transparency_median: float | None,
    source_ratio: float | None,
    thresholds: V2Thresholds,
) -> list[tuple[str, str]]:
    reasons: list[tuple[str, str]] = []
    hard = set(item.hard_fail_codes) | (set(item.review_codes) & set(thresholds.extra_hard_codes))
    for code in sorted(hard):
        if code in HARD_GATE_CODES or code in thresholds.extra_hard_codes:
            reasons.append((f"SEL_GUARD_{code}", f"quality gate {code}: not modelled, excluded"))
    if not item.registration_ok and not item.registration_estimated:
        reasons.append(("SEL_GUARD_REGISTRATION_FAILED", "the frame could not be registered to the group reference"))
    if item.star_count < thresholds.minimum_stars:
        reasons.append(
            ("SEL_GUARD_FEW_STARS", f"{item.star_count} stars detected, fewer than {thresholds.minimum_stars}")
        )
    if item.overlap_fraction is not None and item.overlap_fraction < thresholds.minimum_overlap:
        reasons.append(
            (
                "SEL_GUARD_OVERLAP",
                f"overlap {item.overlap_fraction:.2f} with the group reference is below {thresholds.minimum_overlap:.2f}",
            )
        )
    if flags is not None:
        sky_flag = next((flag for flag in flags.flags if flag.code == "BLINK_SKY_BRIGHT" and flag.combined), None)
        if sky_flag is not None:
            reasons.append(
                (
                    "SEL_NIGHT_SKY_BRIGHT",
                    f"sky {sky_flag.value:.2f}x the channel's clean sky with a source ratio of "
                    f"{source_ratio:.2f}: a moonlit or hazy frame that lost its stars"
                    if sky_flag.value is not None and source_ratio is not None
                    else "sky bright against the channel's clean sky and stars lost",
                )
            )
    if (
        blink is not None
        and blink.background_shape is not None
        and blink.background_shape >= thresholds.background_shape
    ):
        # A shape change with a brighter sky is scattered light the
        # normalization grid does not remove; with thin cloud alone it is the
        # light-cloud band the observer keeps, so it only costs confidence.
        sky_ratio = _metric(flags, "skyRatio")
        if sky_ratio is not None and sky_ratio >= thresholds.sky_suspect_ratio:
            reasons.append(
                (
                    "SEL_BACKGROUND_SHAPE",
                    f"background shape differs by {blink.background_shape:.2f} from its flip family "
                    f"with a sky {sky_ratio:.2f}x the clean sky: scattered light the normalization grid does not remove",
                )
            )
    if transparency_median is not None and item.transparency is not None and transparency_median > 0:
        relative = item.transparency / transparency_median
        if relative < thresholds.transparency_floor:
            reasons.append(
                (
                    "SEL_TRANSPARENCY_FLOOR",
                    f"transparency {item.transparency:.2f} is {relative:.2f}x the clean median, "
                    f"below the floor {thresholds.transparency_floor:.2f} (stellar scale 4)",
                )
            )
    if source_ratio is not None and source_ratio < thresholds.source_ratio_floor:
        reasons.append(
            (
                "SEL_SOURCES_LOW",
                f"source ratio {source_ratio:.2f} is below {thresholds.source_ratio_floor:.2f} of the channel's richest frames",
            )
        )
    if (
        "GATE_TRAILING_REVIEW" in item.review_codes
        and item.ellipticity is not None
        and item.ellipticity >= thresholds.trailing_ellipticity
        and (item.orientation_coherence or 0.0) >= thresholds.trailing_coherence
    ):
        reasons.append(
            (
                "SEL_TRAILING",
                f"median ellipticity {item.ellipticity:.2f} with orientation coherence "
                f"{item.orientation_coherence:.2f}: trailing",
            )
        )
    return reasons


def _region_reasons(
    item: FrameSelectionFeatures,
    region_map: RegionWeightMap | None,
    value: float | None,
    maps_unavailable: str | None,
    thresholds: V2Thresholds,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(firm exclusions, keep notes) for the defects a region map may handle."""

    exclusions: list[tuple[str, str]] = []
    notes: list[tuple[str, str]] = []
    dimming = "GATE_SPATIAL_DIMMING_REVIEW" in item.review_codes
    occlusion = "GATE_OCCLUSION_REVIEW" in item.review_codes
    if region_map is not None and region_map.zero_fraction > thresholds.region_map_maximum_blank:
        exclusions.append(
            (
                "SEL_REGION_MOSTLY_BLANK",
                f"the region weight map blanks {region_map.zero_fraction:.0%} of the field",
            )
        )
        return exclusions, notes
    if dimming:
        if region_map is None:
            why = maps_unavailable or "no region weight map covers the dimmed cells"
            exclusions.append(
                (
                    "SEL_PATCHY_CLOUD",
                    f"localized transparency loss (p90 {item.spatial_dimming_p90_mag if item.spatial_dimming_p90_mag is not None else float('nan'):.2f} mag) and {why}",
                )
            )
        else:
            remaining = None if value is None else value * (1.0 - region_map.zero_fraction)
            if remaining is not None and remaining < thresholds.value_floor:
                exclusions.append(
                    (
                        "SEL_PATCHY_CLOUD",
                        f"the map keeps {1.0 - region_map.zero_fraction:.0%} of the field and "
                        f"the remaining value {remaining:.3f} is below {thresholds.value_floor:.2f}",
                    )
                )
            else:
                notes.append(
                    (
                        "SEL_KEEP_REGION_WEIGHTED_GATE_SPATIAL_DIMMING_REVIEW",
                        f"kept: the region weight map blanks {region_map.zero_fraction:.0%} of the field",
                    )
                )
    if occlusion:
        area = item.occlusion_area
        density = item.occlusion_density
        if region_map is not None:
            notes.append(
                ("SEL_KEEP_REGION_WEIGHTED_GATE_OCCLUSION_REVIEW", "kept: the region weight map blanks the blocked cells")
            )
        elif (
            area is not None
            and area >= thresholds.occlusion_area_exclusion
            and not (density is not None and density > thresholds.occlusion_density_floor)
        ):
            exclusions.append(
                (
                    "SEL_OCCLUSION_AREA_EXCLUDED",
                    f"a connected region covering {area:.0%} of the field lost its stars and "
                    f"{maps_unavailable or 'no region weight map blanks it'}",
                )
            )
    return exclusions, notes


@dataclass
class _Candidate:
    item: FrameSelectionFeatures
    value: float | None
    components: dict[str, Any]
    block: str
    psf_factor: float
    completeness: float
    firm: list[tuple[str, str]]
    soft: list[tuple[str, str]]
    notes: list[tuple[str, str]]
    confidence: float
    suspect: list[str]


def decide_v2(
    features: Sequence[FrameSelectionFeatures],
    parameters: SelectionParameters,
    context: BlinkContext,
    *,
    approved_paths: Iterable[str] = (),
    region_maps: Mapping[str, RegionWeightMap] | None = None,
    region_maps_unavailable: str | None = None,
    thresholds: V2Thresholds = V2_THRESHOLDS,
) -> tuple[list[Any], dict[str, ChannelSummary], list[SuspectBlock]]:
    """Decide every frame under ``unattended-v2``.

    Returns the decisions (``SelectionDecision``), the per-group summaries and
    the suspect blocks the integration should measure.  ``region_maps_unavailable``
    names why no map was built (canvas drizzle) so the exclusions say so.
    """

    from .policy import ACTION_EXCLUDE, ACTION_KEEP, SelectionDecision

    approved = {str(path) for path in approved_paths}
    maps = dict(region_maps or {})
    exponent = thresholds.psf_exponent[parameters.priority]
    psf_cut = thresholds.psf_cutoff[parameters.priority]
    decisions: dict[str, SelectionDecision] = {}
    summaries: dict[str, ChannelSummary] = {}
    blocks: list[SuspectBlock] = []
    groups: dict[str, list[FrameSelectionFeatures]] = {}
    for item in features:
        groups.setdefault(item.group_id, []).append(item)

    for group_id, members in groups.items():
        stats = context.statistics.get(group_id)
        clean = _clean_set(members, context, stats)
        fwhm_median = _median(item.fwhm_native for item in clean)
        transparency_median = _median(item.transparency for item in clean)
        sources_best = stats.sources_best if stats is not None else None
        if sources_best is None:
            counts = [float(item.star_count) for item in members if item.star_count > 0]
            sources_best = max(counts) if counts else None
        s_median = _median(
            _raw_value(item, context.inputs.get(item.path), fwhm_median, sources_best) for item in clean
        )
        block_names = _blocks_of_group(members, context, thresholds)
        candidates: list[_Candidate] = []
        for item in members:
            blink = context.inputs.get(item.path)
            flags = context.flags.get(item.path)
            raw = _raw_value(item, blink, fwhm_median, sources_best)
            value = raw / s_median if raw is not None and s_median else None
            source_ratio = (
                item.star_count / sources_best if sources_best else None
            )
            sky_ratio = _metric(flags, "skyRatio")
            fwhm_ratio = (
                item.fwhm_native / fwhm_median
                if item.fwhm_native is not None and fwhm_median
                else None
            )
            transparency_relative = (
                item.transparency / transparency_median
                if item.transparency is not None and transparency_median
                else None
            )
            completeness = min(1.0, item.star_count / sources_best) if sources_best else 1.0
            psf_factor = 1.0
            if exponent and fwhm_ratio is not None and fwhm_ratio > 0:
                psf_factor = float(min(16.0, max(1.0 / 16.0, fwhm_ratio ** (-2.0 * exponent))))
            components = {
                "transparencyRelative": transparency_relative,
                "sigma": blink.sky_mad if blink is not None else None,
                "skyRatio": sky_ratio,
                "fwhmRatio": fwhm_ratio,
                "ellipticity": item.ellipticity,
                "sourceRatio": source_ratio,
                "extinctionMag": blink.extinction_mag if blink is not None else None,
                "dimmingP90Mag": item.spatial_dimming_p90_mag,
                "backgroundShape": blink.background_shape if blink is not None else None,
                "overlap": item.overlap_fraction,
                "blinkFlags": list(flags.codes) if flags is not None else [],
            }
            candidate = _Candidate(
                item=item,
                value=value,
                components=components,
                block=block_names.get(item.path, item.night_id or "UNKNOWN"),
                psf_factor=psf_factor,
                completeness=completeness,
                firm=_firm_reasons(item, blink, flags, transparency_median, source_ratio, thresholds),
                soft=[],
                notes=[],
                confidence=1.0,
                suspect=[],
            )
            region_exclusions, region_notes = _region_reasons(
                item, maps.get(item.path), value, region_maps_unavailable, thresholds
            )
            candidate.firm.extend(region_exclusions)
            candidate.notes.extend(region_notes)
            if candidate.firm:
                candidates.append(candidate)
                continue
            if item.path in approved:
                candidate.notes.append(("SEL_APPROVED", "explicitly approved"))
                candidates.append(candidate)
                continue
            if fwhm_ratio is not None and fwhm_ratio > psf_cut:
                candidate.soft.append(
                    (
                        "SEL_PSF_BEYOND_MEDIAN",
                        f"FWHM is {fwhm_ratio:.2f}x the clean median (cut-off {psf_cut:.2f} for priority {parameters.priority})",
                    )
                )
            if value is not None and value < thresholds.value_floor:
                candidate.soft.append(
                    (
                        "SEL_VALUE_NEGLIGIBLE",
                        f"value {value:.3f} of a clean frame is below {thresholds.value_floor:.2f}",
                    )
                )
            # Confidence and suspect marks of a kept frame.
            if flags is not None and any(flag.code == "BLINK_SKY_BRIGHT" for flag in flags.flags) and sky_ratio is not None:
                # Blink's attention level (sky >= 1.6x) with the stars retained.
                candidate.confidence = min(candidate.confidence, thresholds.confidence_suspect)
                candidate.suspect.append("SKY_BRIGHT")
                candidate.notes.append(("SEL_KEEP_SKY_BRIGHT", f"kept with reduced weight: sky {sky_ratio:.2f}x the clean sky, stars retained"))
            if blink is not None and blink.background_shape is not None and blink.background_shape >= thresholds.background_shape:
                candidate.confidence = min(candidate.confidence, thresholds.confidence_shape)
                candidate.suspect.append("BACKGROUND_SHAPE")
                candidate.notes.append(("SEL_KEEP_BACKGROUND_SHAPE", f"kept with reduced weight: background shape {blink.background_shape:.2f}"))
            for code in item.review_codes:
                if code in INSUFFICIENT_EVIDENCE_CODES:
                    candidate.notes.append(("SEL_KEEP_INSUFFICIENT_EVIDENCE_" + code, "kept: evidence insufficient, not a defect"))
                elif code in {"GATE_NIGHT_ZEROPOINT_SHIFT_REVIEW", "GATE_TEMPORAL_EXTINCTION_REVIEW"}:
                    candidate.notes.append(("SEL_KEEP_NORMALIZED_" + code, f"kept: normalization models {code}"))
                elif code == "GATE_SOURCE_RETENTION_REVIEW":
                    if transparency_relative is not None and transparency_relative <= thresholds.retention_explained_transparency:
                        candidate.notes.append(("SEL_KEEP_RETENTION_EXPLAINED", f"kept: stars lost with transparency {transparency_relative:.2f}x the clean median"))
                    else:
                        candidate.confidence = min(candidate.confidence, thresholds.confidence_suspect)
                        candidate.suspect.append("SOURCES_LOW_UNEXPLAINED")
                        candidate.notes.append(("SEL_KEEP_DOWNWEIGHTED_" + code, "kept with reduced weight: stars lost at normal transparency"))
                elif code in {"GATE_BACKGROUND_REVIEW", "GATE_BACKGROUND_STRONG", "GATE_NOISE_REVIEW"}:
                    candidate.confidence = min(candidate.confidence, thresholds.confidence_night_outlier)
                    candidate.suspect.append("NIGHT_OUTLIER")
                    candidate.notes.append(("SEL_KEEP_DOWNWEIGHTED_" + code, f"kept with reduced weight: {code}"))
                elif code in {"GATE_TRAILING_REVIEW", "GATE_REGISTRATION_REVIEW"}:
                    candidate.confidence = min(candidate.confidence, thresholds.confidence_suspect)
                    candidate.notes.append(("SEL_KEEP_DOWNWEIGHTED_" + code, f"kept with reduced weight: {code}"))
                elif code not in {"GATE_SPATIAL_DIMMING_REVIEW", "GATE_OCCLUSION_REVIEW"}:
                    candidate.notes.append(("SEL_NOTED_" + code, f"noted: {code} is decided by the PSF and value rules"))
            if value is not None and value < thresholds.value_suspect:
                candidate.suspect.append("LOW_VALUE")
            candidates.append(candidate)

        # Soft guard: value-ordered, never below max(minimum, ceil(fraction * n)).
        in_play = [candidate for candidate in candidates if not candidate.firm]
        floor = max(thresholds.soft_guard_minimum, math.ceil(thresholds.soft_guard_fraction * len(in_play)))
        soft = sorted(
            (candidate for candidate in in_play if candidate.soft),
            key=lambda candidate: (candidate.value if candidate.value is not None else -1.0, candidate.item.path),
        )
        remaining = len(in_play)
        for candidate in soft:
            if remaining - 1 < floor:
                candidate.soft.append(
                    (
                        "SEL_SOFT_GUARD_KEPT",
                        f"kept with reduced weight: excluding it would leave fewer than {floor} of {len(in_play)} frames",
                    )
                )
                candidate.confidence = min(candidate.confidence, 0.5)
                candidate.notes.extend(candidate.soft)
                candidate.soft = []
                continue
            remaining -= 1

        for candidate in candidates:
            item = candidate.item
            if candidate.firm:
                action, confidence, reasons, is_firm = ACTION_EXCLUDE, 0.0, candidate.firm, True
            elif candidate.soft:
                action, confidence, reasons, is_firm = ACTION_EXCLUDE, 0.0, candidate.soft, False
            else:
                action, confidence, reasons, is_firm = ACTION_KEEP, candidate.confidence, candidate.notes, False
            decisions[item.path] = SelectionDecision(
                item.path,
                action,
                confidence,
                tuple(reasons),
                soft=(action == ACTION_EXCLUDE and not is_firm),
                psf_factor=candidate.psf_factor,
                firm=is_firm,
                value=candidate.value,
                components=candidate.components,
                completeness_factor=candidate.completeness,
                block=candidate.block,
                suspect=tuple(candidate.suspect) if action == ACTION_KEEP else (),
            )

        # Suspect blocks of the group.
        by_block: dict[str, list[_Candidate]] = {}
        for candidate in candidates:
            if decisions[candidate.item.path].admitted:
                by_block.setdefault(candidate.block, []).append(candidate)
        block_records: list[SuspectBlock] = []
        for name, items in by_block.items():
            median_value = _median(candidate.value for candidate in items)
            median_sky = _median(candidate.components.get("skyRatio") for candidate in items)
            marked = sum(1 for candidate in items if candidate.suspect)
            reasons: list[str] = []
            if median_sky is not None and median_sky >= thresholds.sky_suspect_ratio:
                reasons.append("SKY_BRIGHT")
            if median_value is not None and median_value < thresholds.value_suspect:
                reasons.append("LOW_VALUE")
            if items and marked / len(items) >= thresholds.block_suspect_fraction:
                reasons.append("SUSPECT_FRAMES")
            if reasons:
                block_records.append(
                    SuspectBlock(
                        group_id=group_id,
                        name=name,
                        paths=tuple(sorted(candidate.item.path for candidate in items)),
                        reasons=tuple(reasons),
                        median_value=median_value,
                        median_sky_ratio=median_sky,
                    )
                )
        block_records.sort(key=lambda record: (record.median_value if record.median_value is not None else -1.0, record.name))
        blocks.extend(block_records[: thresholds.block_maximum_per_group])
        summaries[group_id] = ChannelSummary(
            group_id=group_id,
            target=members[0].target,
            filter_name=members[0].filter_name,
            frame_count=len(members),
            clean_count=len(clean) if stats is None or stats.clean_count == 0 else stats.clean_count,
            sky_clean=stats.sky_clean if stats is not None else None,
            fwhm_median=fwhm_median,
            s_median=s_median,
            transparency_median=transparency_median,
            sources_best=sources_best,
            blocks=len(by_block),
        )
    return [decisions[item.path] for item in features], summaries, blocks


# --- blocks and the counterfactual -----------------------------------------


def block_indices(block: SuspectBlock, ordered_paths: Sequence[str]) -> tuple[int, ...]:
    """Indices of the block's frames in an integration group's frame order."""

    positions = {path: index for index, path in enumerate(ordered_paths)}
    return tuple(sorted(positions[path] for path in block.paths if path in positions))


def confirmed_harmful_blocks(
    report: CounterfactualReport, thresholds: V2Thresholds = V2_THRESHOLDS
) -> dict[str, dict[str, Any]]:
    """Blocks whose removal measurably improves the master, by block name.

    A block is removed when the background residual improves by more than
    the harm threshold (interval lower bound) at a depth cost below the
    bound, or when depth itself improves beyond the beneficial bound.
    """

    harmful: dict[str, dict[str, Any]] = {}
    if report.tiles_used < thresholds.minimum_counterfactual_tiles:
        return harmful
    for block in report.blocks:
        depth = block.delta_depth_mag
        depth_low = block.delta_depth_ci[0] if block.delta_depth_ci else None
        background_low = block.delta_background_ci[0] if block.delta_background_ci else None
        background_harm = (
            background_low is not None
            and background_low > thresholds.block_harmful_background_sigma
            and depth is not None
            and depth > -thresholds.block_depth_cost_mag
        )
        depth_harm = depth_low is not None and depth_low > thresholds.block_beneficial_depth_mag
        if background_harm or depth_harm:
            harmful[block.name] = {
                **block.serializable(),
                "because": "background" if background_harm else "depth",
            }
    return harmful


def suspect_channels(
    decisions: Sequence[Any],
    features: Sequence[FrameSelectionFeatures],
    summaries: Mapping[str, ChannelSummary],
    blocks: Sequence[SuspectBlock],
    block_outcomes: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Groups a user should look at in Blink, with the codes and one sentence.

    ``block_outcomes`` maps ``group_id/block`` to ``REMOVED``, ``KEPT`` or
    ``NOT_MEASURED``; a suspect block that was not removed, a channel without
    a clean-sky baseline, or a soft guard that kept frames make the channel suspect.
    """

    records: list[dict[str, Any]] = []
    group_by_path = {item.path: item.group_id for item in features}
    by_group: dict[str, list[Any]] = {}
    for item in decisions:
        by_group.setdefault(group_by_path.get(item.path, ""), []).append(item)
    for group_id, summary in summaries.items():
        codes: list[str] = []
        parts: list[str] = []
        kept_blocks = [
            block for block in blocks
            if block.group_id == group_id and block_outcomes.get(f"{group_id}/{block.name}", "NOT_MEASURED") != "REMOVED"
        ]
        if kept_blocks:
            measured = [block for block in kept_blocks if block_outcomes.get(f"{group_id}/{block.name}") == "KEPT"]
            unmeasured = [block for block in kept_blocks if block not in measured]
            if measured:
                codes.append("SUSPECT_BLOCK_KEPT")
                parts.append(
                    f"{len(measured)} suspect night(s) ({', '.join(block.name for block in measured)}) measured and kept"
                )
            if unmeasured:
                codes.append("SUSPECT_BLOCK_UNMEASURED")
                parts.append(
                    f"{len(unmeasured)} suspect night(s) ({', '.join(block.name for block in unmeasured)}) could not be measured"
                )
        if summary.sky_clean is None and summary.frame_count > 3:
            codes.append("NO_SKY_BASELINE")
            parts.append("no clean-sky baseline (fewer than 3 clean frames), so a uniformly bright night is invisible")
        guarded = [
            item for item in by_group.get(group_id, ())
            if any(code == "SEL_SOFT_GUARD_KEPT" for code, _ in item.reasons)
        ]
        if guarded:
            codes.append("SOFT_GUARD_KEPT")
            parts.append(f"{len(guarded)} frame(s) kept at reduced weight by the minimum-count guard")
        if codes:
            records.append(
                {
                    "target": summary.target,
                    "filter": summary.filter_name,
                    "codes": codes,
                    "summary": "; ".join(parts) + ".",
                }
            )
    return records


def receipt_block(
    thresholds: V2Thresholds,
    summaries: Mapping[str, ChannelSummary],
    decisions: Sequence[Any],
    blocks: Sequence[SuspectBlock],
    block_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The ``unattendedV2`` block of ``qc/selection.json``."""

    counts = {"excludedFirm": 0, "excludedSoft": 0, "excludedByCounterfactual": 0, "keptReducedWeight": 0, "kept": 0}
    for item in decisions:
        if item.admitted:
            counts["keptReducedWeight" if item.confidence < 1.0 else "kept"] += 1
        elif any(code in {"SEL_COUNTERFACTUAL_CONFIRMED_HARMFUL", "SEL_BLOCK_COUNTERFACTUAL_REMOVED"} for code, _ in item.reasons):
            counts["excludedByCounterfactual"] += 1
        elif item.firm:
            counts["excludedFirm"] += 1
        else:
            counts["excludedSoft"] += 1
    return {
        "thresholds": thresholds.describe(),
        "counts": counts,
        "channels": {group_id: summary.serializable() for group_id, summary in sorted(summaries.items())},
        "suspectBlocks": [block.serializable() for block in blocks],
        "blocks": list(block_results),
    }


__all__ = [
    "BlinkContext",
    "ChannelSummary",
    "POLICY_V2",
    "SuspectBlock",
    "V2Thresholds",
    "V2_ALGORITHM",
    "V2_THRESHOLDS",
    "blink_context",
    "block_indices",
    "confirmed_harmful_blocks",
    "decide_v2",
    "receipt_block",
    "suspect_channels",
]
