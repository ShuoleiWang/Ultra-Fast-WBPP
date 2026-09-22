"""Blink-style screening flags: absolute, cross-night criteria per channel.

The quality gate judges a frame against the other frames of its own night,
so a uniformly bad night (moonlit, hazy) is self-consistent and passes.  The
blink flags compare every frame of a channel (one QC group: target, filter,
camera geometry, exposure bucket) with the channel's *clean set* instead:
sky level, detected-source count, extinction, PSF width and background shape
are absolute within the channel, whichever night the frame belongs to.

Nothing is excluded silently.  A flag sets a *default decision* and carries
its value, threshold and a one-line reason; the user decides frame by frame
in the blink view, and the run records every override.  Two severities:
``EXCLUDE`` pre-marks the frame as ``DROP``; ``ATTENTION`` keeps it but
highlights it.  Gate codes that say "the gate could not judge" (a
single-frame night, a missing source count) become notes, not flags.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .models import FrameMeasurement, FrameResult, GateDisposition
from .nightly_statistics import observing_night
from .statistics import safe_log_extinction

BLINK_FLAGS_VERSION = "blink-flags-v1"

SEVERITY_EXCLUDE = "EXCLUDE"
SEVERITY_ATTENTION = "ATTENTION"
DECISION_KEEP = "KEEP"
DECISION_DROP = "DROP"

# Gate codes that say the gate could not establish evidence, not that the
# frame is bad (the single-frame B night the user kept).  They are reported
# as notes so the reviewer knows the frame was not judged automatically.
EVIDENCE_INSUFFICIENCY_CODES = frozenset(
    {
        "GATE_INSUFFICIENT_COHORT",
        "GATE_INSUFFICIENT_NIGHT_BASELINE",
        "GATE_NIGHT_UNRESOLVED",
        "GATE_MORPHOLOGY_SAMPLE_REVIEW",
        "GATE_SOURCE_COUNT_MISSING",
        "GATE_FINITE_FRACTION_REVIEW",
        "GATE_FINITE_FRACTION_MISSING",
        "GATE_DYNAMIC_RANGE_MISSING",
        "GATE_REFERENCE_NOT_CONNECTED",
    }
)

# Gate evidence -> blink flag (code, severity).  Codes covered by an absolute
# flag are listed in ``_GATE_CODES_COVERED`` and never duplicated.
GATE_FLAG_MAPPING: dict[str, tuple[str, str]] = {
    "GATE_COHERENT_TRAILING_HARD": ("BLINK_TRAILING", SEVERITY_EXCLUDE),
    "GATE_FRAGMENTED_TRAILING_HARD": ("BLINK_TRAILING", SEVERITY_EXCLUDE),
    "GATE_TRAILING_REVIEW": ("BLINK_TRAILING", SEVERITY_ATTENTION),
    "GATE_FOCUS_SEEING_REVIEW": ("BLINK_FOCUS", SEVERITY_ATTENTION),
    "GATE_NIGHT_FOCUS_SHIFT_REVIEW": ("BLINK_FOCUS", SEVERITY_ATTENTION),
    "GATE_OCCLUSION_HARD": ("BLINK_OBSTRUCTION", SEVERITY_EXCLUDE),
    "GATE_OCCLUSION_REVIEW": ("BLINK_OBSTRUCTION", SEVERITY_ATTENTION),
    "GATE_SPATIAL_DIMMING_STRONG": ("BLINK_CLOUD_PATCHY", SEVERITY_EXCLUDE),
    "GATE_SPATIAL_DIMMING_REVIEW": ("BLINK_CLOUD_PATCHY", SEVERITY_ATTENTION),
    "GATE_MULTI_FAMILY_CLOUD_HARD": ("BLINK_CLOUD_THICK", SEVERITY_EXCLUDE),
    "GATE_SOURCE_RETENTION_STRONG": ("BLINK_SOURCES_LOW", SEVERITY_ATTENTION),
    "GATE_SOURCE_RETENTION_REVIEW": ("BLINK_SOURCES_LOW", SEVERITY_ATTENTION),
    "GATE_COMMON_FOOTPRINT_REVIEW": ("BLINK_FIELD_MISMATCH", SEVERITY_ATTENTION),
    "GATE_REGISTRATION_REVIEW": ("BLINK_REGISTRATION_WEAK", SEVERITY_ATTENTION),
    "GATE_BACKGROUND_STRONG": ("BLINK_NIGHT_OUTLIER", SEVERITY_ATTENTION),
    "GATE_BACKGROUND_REVIEW": ("BLINK_NIGHT_OUTLIER", SEVERITY_ATTENTION),
    "GATE_NOISE_REVIEW": ("BLINK_NIGHT_OUTLIER", SEVERITY_ATTENTION),
    "GATE_MEASUREMENT_FAILED": ("BLINK_UNMEASURABLE", SEVERITY_EXCLUDE),
    "GATE_MEASUREMENT_MISSING": ("BLINK_UNMEASURABLE", SEVERITY_EXCLUDE),
    "GATE_IDENTITY_MISSING": ("BLINK_UNMEASURABLE", SEVERITY_EXCLUDE),
    "GATE_IDENTITY_MISMATCH": ("BLINK_UNMEASURABLE", SEVERITY_EXCLUDE),
    "GATE_FINITE_FRACTION_HARD": ("BLINK_UNMEASURABLE", SEVERITY_EXCLUDE),
    "GATE_FINITE_FRACTION_INVALID": ("BLINK_UNMEASURABLE", SEVERITY_EXCLUDE),
    "GATE_NEAR_CONSTANT_IMAGE": ("BLINK_UNMEASURABLE", SEVERITY_EXCLUDE),
    "GATE_NOT_A_LIGHT_FRAME": ("BLINK_UNMEASURABLE", SEVERITY_EXCLUDE),
}

# Extinction and the absolute source count are judged by the absolute flags
# with their own (wider) thresholds; the gate's nightly verdicts on them are
# deliberately not repeated as flags.
_GATE_CODES_COVERED = frozenset(
    {
        "GATE_TEMPORAL_EXTINCTION_STRONG",
        "GATE_TEMPORAL_EXTINCTION_REVIEW",
        "GATE_NIGHT_ZEROPOINT_SHIFT_REVIEW",
        "GATE_ABSOLUTE_LOW_SOURCE_COUNT",
    }
)

_GATE_FLAG_MESSAGES: dict[str, str] = {
    "BLINK_TRAILING": "Elongated or trailed stars",
    "BLINK_FOCUS": "Focus or seeing degraded against the night",
    "BLINK_OBSTRUCTION": "A connected region of the field has no stars (obstruction)",
    "BLINK_CLOUD_PATCHY": "Localized transparency loss (patchy cloud)",
    "BLINK_CLOUD_THICK": "Several independent cloud evidence families agree",
    "BLINK_SOURCES_LOW": "Detected-source retention below the nightly clear envelope",
    "BLINK_FIELD_MISMATCH": "Common-field coverage is insufficient: wrong field or a large offset",
    "BLINK_REGISTRATION_WEAK": "Registration evidence is below the automatic-pass boundary",
    "BLINK_NIGHT_OUTLIER": "Background or noise is an outlier within its night",
    "BLINK_UNMEASURABLE": "The frame could not be measured or is not a light frame",
}


@dataclass(frozen=True, slots=True)
class BlinkFlagPolicy:
    """Every threshold of the blink flags; its digest names the rule set."""

    version: str = BLINK_FLAGS_VERSION
    # Clean set: frames with low extinction, a normal star count, no
    # HARD_FAIL and a registration transform.  The channel's clean sky is
    # the median sky of this set and needs a minimum of frames.
    clean_extinction_mag: float = 0.35
    clean_source_ratio: float = 0.60
    minimum_clean_frames: int = 3
    # Channel references: the robust richest star count and the robust best
    # PSF width; the percentile falls back to the extreme on small channels.
    sources_best_percentile: float = 90.0
    sources_best_percentile_minimum_frames: int = 10
    fwhm_best_percentile: float = 10.0
    # BLINK_SKY_BRIGHT: image median over the clean sky.  ATTENTION alone;
    # EXCLUDE together with a source ratio at or below the combined level.
    sky_bright_ratio: float = 1.6
    sky_bright_exclude_source_ratio: float = 0.60
    # BLINK_SOURCES_LOW: star count over the channel's richest count.
    sources_low_attention_ratio: float = 0.60
    sources_low_exclude_ratio: float = 0.45
    # BLINK_EXTINCTION: airmass-corrected extra extinction in magnitudes.
    extinction_attention_mag: float = 0.50
    extinction_exclude_mag: float = 1.00
    # BLINK_BACKGROUND_SHAPE: P95-P5 of the normalized background grid
    # difference to the frame's flip family over the outer cells (see
    # ``background_shapes``); the central box (this fraction of each
    # dimension) holds the target and is excluded.
    background_shape_attention: float = 0.50
    background_shape_core_fraction: float = 0.40
    # BLINK_GRADIENT_AMPLITUDE (calibrated data only): flux-scaled gradient
    # amplitude over the clean set's.
    gradient_amplitude_attention_ratio: float = 2.0
    gradient_amplitude_exclude_ratio: float = 3.0
    # BLINK_FWHM_WIDE: PSF FWHM over the channel's best.
    fwhm_wide_attention_ratio: float = 1.30
    fwhm_wide_exclude_ratio: float = 1.60
    # BLINK_STARS_ELONGATED: median ellipticity (the gate's review level).
    stars_elongated_attention: float = 0.30
    # BLINK_FEW_STARS: fewer detected sources than this is always EXCLUDE.
    few_stars_minimum: int = 20

    def validate(self) -> None:
        integer_fields = {
            "minimum_clean_frames",
            "sources_best_percentile_minimum_frames",
            "few_stars_minimum",
        }
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "version":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("blink flag policy version cannot be empty")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{item.name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{item.name} must be finite and non-negative")
        if not 0 < self.sources_best_percentile <= 100 or not 0 < self.fwhm_best_percentile <= 100:
            raise ValueError("percentiles must be in (0, 100]")
        if not 0 <= self.background_shape_core_fraction < 1:
            raise ValueError("background_shape_core_fraction must be in [0, 1)")
        if not 0 <= self.clean_source_ratio <= 1 or not 0 <= self.sky_bright_exclude_source_ratio <= 1:
            raise ValueError("source ratio thresholds must be in [0, 1]")
        if not 0 <= self.sources_low_exclude_ratio <= self.sources_low_attention_ratio <= 1:
            raise ValueError("source-low thresholds must satisfy 0 <= exclude <= attention <= 1")
        if self.sky_bright_ratio < 1:
            raise ValueError("sky_bright_ratio must be at least 1")
        for attention, exclude in (
            (self.extinction_attention_mag, self.extinction_exclude_mag),
            (self.gradient_amplitude_attention_ratio, self.gradient_amplitude_exclude_ratio),
            (self.fwhm_wide_attention_ratio, self.fwhm_wide_exclude_ratio),
        ):
            if exclude < attention:
                raise ValueError("an EXCLUDE threshold cannot be below its ATTENTION threshold")
        if not 0 <= self.stars_elongated_attention <= 1:
            raise ValueError("stars_elongated_attention must be in [0, 1]")

    def serializable(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def canonical_digest(self) -> str:
        payload = json.dumps(
            self.serializable(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BlinkFrameInput:
    """The per-frame numbers the flags and the reference score use.

    Built from ``FrameResult`` + ``FrameMeasurement`` by :func:`blink_inputs`
    in production and from JSON in the regression fixture, so the rules can
    be replayed without the frames.
    """

    path: str
    channel_id: str
    target: str
    filter_name: str
    night: str | None
    observed_at: str | None
    airmass: float | None
    sky: float | None
    sky_mad: float | None
    star_count: int
    transparency: float | None
    extinction_mag: float | None
    fwhm_native: float | None
    ellipticity: float | None
    eccentricity: float | None
    registration_ok: bool
    matched_stars: int
    registration_rms: float | None
    overlap: float | None
    background_shape: float | None
    gate_disposition: str | None
    gate_codes: tuple[str, ...] = ()
    transform: tuple[tuple[float, float, float], ...] | None = None
    source_sha256: str | None = None
    is_qc_reference: bool = False
    # Flux-scaled amplitude (ADU) of the calibrated background gradient; only
    # a blink session with calibration masters fills it.
    gradient_adu: float | None = None

    @property
    def name(self) -> str:
        return self.path.replace("\\", "/").rsplit("/", 1)[-1]

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value["gate_codes"] = list(self.gate_codes)
        value["transform"] = (
            [list(row) for row in self.transform] if self.transform is not None else None
        )
        return value

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BlinkFrameInput":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"blink frame input has unknown keys: {', '.join(unknown)}")
        values = dict(raw)
        values["gate_codes"] = tuple(str(code) for code in values.get("gate_codes", ()))
        transform = values.get("transform")
        values["transform"] = (
            tuple(tuple(float(item) for item in row) for row in transform)
            if transform is not None
            else None
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class BlinkFlag:
    code: str
    severity: str
    message: str
    value: float | None = None
    threshold: float | None = None
    combined: bool = False

    def serializable(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "value": self.value,
            "threshold": self.threshold,
            "combined": self.combined,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ChannelStatistics:
    """Per-channel references the absolute flags are measured against."""

    channel_id: str
    frame_count: int
    clean_count: int
    sky_clean: float | None
    sources_best: float | None
    fwhm_best: float | None
    gradient_clean: float | None

    def serializable(self) -> dict[str, Any]:
        return {
            "channelId": self.channel_id,
            "frameCount": self.frame_count,
            "cleanCount": self.clean_count,
            "skyClean": self.sky_clean,
            "sourcesBest": self.sources_best,
            "fwhmBest": self.fwhm_best,
            "gradientClean": self.gradient_clean,
        }


@dataclass(frozen=True, slots=True)
class BlinkFrameFlags:
    path: str
    channel_id: str
    flags: tuple[BlinkFlag, ...]
    default_decision: str
    notes: tuple[str, ...]
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def exclude(self) -> bool:
        return any(item.severity == SEVERITY_EXCLUDE for item in self.flags)

    @property
    def attention(self) -> bool:
        return not self.exclude and any(item.severity == SEVERITY_ATTENTION for item in self.flags)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(item.code for item in self.flags)

    def serializable(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "channelId": self.channel_id,
            "defaultDecision": self.default_decision,
            "flags": [item.serializable() for item in self.flags],
            "notes": list(self.notes),
            "metrics": dict(self.metrics),
        }


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _ratio(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference <= 0:
        return None
    return value / reference


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _grid_array(grid: Sequence[Sequence[float | None]] | None) -> np.ndarray | None:
    if not grid:
        return None
    values = np.array(
        [[np.nan if item is None else float(item) for item in row] for row in grid],
        dtype=np.float64,
    )
    return values if values.ndim == 2 and values.size > 0 else None


def background_shape_statistic(
    grid: Sequence[Sequence[float | None]] | np.ndarray | None, core_fraction: float = 0.40
) -> float | None:
    """P95-P5 of a normalized background-difference grid outside the core.

    The analysis grid holds each frame's 16x16 SEP background and the QC
    reference's, both normalized to unit cell spread, then differenced: a
    sky-level-invariant *shape* difference.  The central box (the target) is
    excluded so nebulosity never counts as a gradient.
    """

    values = grid.copy() if isinstance(grid, np.ndarray) else _grid_array(grid)
    if values is None:
        return None
    rows, columns = values.shape
    half = core_fraction / 2.0
    values[
        int(rows * (0.5 - half)) : int(rows * (0.5 + half)),
        int(columns * (0.5 - half)) : int(columns * (0.5 + half)),
    ] = np.nan
    outer = values[np.isfinite(values)]
    if outer.size < 4:
        return None
    return float(np.percentile(outer, 95) - np.percentile(outer, 5))


# Frames of a flip family enter its consensus grid when they are this
# transparent (light cloud stays in, thick cloud does not) and the family
# needs this many members before a consensus is trusted.
SHAPE_FAMILY_EXTINCTION_MAG = 0.60
SHAPE_FAMILY_MINIMUM_FRAMES = 3
# The analysis grid is normalized to unit cell spread; when a frame's raw
# background varies by less than this fraction of its sky across the cells
# (a flat field at the noise level), that normalization amplifies noise and
# the shape statistic is left undefined.
SHAPE_MINIMUM_RELATIVE_SPREAD = 0.005


def _flip_parity(matrix: Sequence[Sequence[float]] | None) -> str | None:
    """``"same"`` or ``"flipped"`` orientation relative to the QC reference.

    A meridian flip rotates the camera by 180 degrees; the similarity
    transform's rotation is read from its first column.
    """

    if matrix is None:
        return None
    try:
        cosine = float(matrix[0][0])
    except (IndexError, TypeError, ValueError):
        return None
    if not math.isfinite(cosine):
        return None
    return "flipped" if cosine < 0 else "same"


def _relative_background_spread(measurement: FrameMeasurement | None) -> float | None:
    if measurement is None or not measurement.background_grid:
        return None
    grid = _grid_array(measurement.background_grid)
    if grid is None:
        return None
    finite = grid[np.isfinite(grid)]
    if finite.size < 4:
        return None
    center = float(np.median(finite))
    if abs(center) < 1e-9:
        return None
    return 1.4826 * float(np.median(np.abs(finite - center))) / abs(center)


def background_shapes(
    results: Sequence[FrameResult],
    core_fraction: float = 0.40,
    measurements: Mapping[str, FrameMeasurement] | None = None,
) -> dict[str, float | None]:
    """Per-frame background shape statistic against the frame's flip family.

    The QC previews are raw, so their background is dominated by vignetting,
    which rotates with the camera at a meridian flip while the registered
    grids are in the reference's orientation: a flipped frame differs from
    an unflipped reference by about one cell spread whatever its sky does.
    Each frame is therefore compared with the median difference grid of the
    frames sharing its orientation (registered, not HARD_FAIL, extinction
    below ``SHAPE_FAMILY_EXTINCTION_MAG``); a family with fewer than
    ``SHAPE_FAMILY_MINIMUM_FRAMES`` such members has no statistic, and
    neither has a frame whose raw background is flat at the noise level
    (``SHAPE_MINIMUM_RELATIVE_SPREAD``, judged on ``measurements``).
    """

    grids: dict[str, np.ndarray] = {}
    families: dict[str, str] = {}
    for result in results:
        grid = _grid_array(result.grid.get("backgroundDeltaRobustSigma") if result.grid else None)
        parity = _flip_parity(result.registration.matrix)
        if grid is None or parity is None:
            continue
        if measurements is not None:
            spread = _relative_background_spread(measurements.get(result.path))
            if spread is not None and spread < SHAPE_MINIMUM_RELATIVE_SPREAD:
                continue
        grids[result.path] = grid
        families[result.path] = parity
    members: dict[str, list[np.ndarray]] = defaultdict(list)
    for result in results:
        if result.path not in grids or not result.registration.ok:
            continue
        gate = result.quality_gate
        if gate is not None and gate.disposition is GateDisposition.HARD_FAIL:
            continue
        extinction = _finite(result.features.extra_extinction_mag)
        if extinction is None:
            extinction = _finite(result.features.nightly_extinction_residual)
        if extinction is None:
            extinction = _finite(safe_log_extinction(_finite(result.features.transparency_ratio)))
        if extinction is None or extinction >= SHAPE_FAMILY_EXTINCTION_MAG:
            continue
        members[families[result.path]].append(grids[result.path])
    consensus = {
        parity: np.nanmedian(np.stack(items), axis=0)
        for parity, items in members.items()
        if len(items) >= SHAPE_FAMILY_MINIMUM_FRAMES
    }
    shapes: dict[str, float | None] = {}
    for result in results:
        grid = grids.get(result.path)
        family = consensus.get(families.get(result.path, ""))
        shapes[result.path] = (
            background_shape_statistic(grid - family, core_fraction)
            if grid is not None and family is not None
            else None
        )
    return shapes


def _observed_at_text(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def blink_inputs(
    results: Iterable[FrameResult],
    measurements: Iterable[FrameMeasurement] | Mapping[str, FrameMeasurement] | None = None,
    policy: BlinkFlagPolicy | None = None,
    *,
    night_boundary_hours: float = 12.0,
    observing_timezone: str | None = None,
) -> list[BlinkFrameInput]:
    """Project gate-evaluated results onto the blink inputs, in result order."""

    selected = policy or BlinkFlagPolicy()
    selected.validate()
    measurement_by_path: dict[str, FrameMeasurement] = {}
    if measurements is not None:
        items = measurements.values() if isinstance(measurements, Mapping) else measurements
        for item in items:
            measurement_by_path[item.metadata.path] = item
    ordered = list(results)
    by_channel: dict[str, list[FrameResult]] = defaultdict(list)
    for result in ordered:
        by_channel[result.group_id].append(result)
    shapes: dict[str, float | None] = {}
    for channel_results in by_channel.values():
        shapes.update(
            background_shapes(
                channel_results, selected.background_shape_core_fraction, measurement_by_path
            )
        )
    inputs: list[BlinkFrameInput] = []
    for result in ordered:
        features = result.features
        measurement = measurement_by_path.get(result.path)
        sky = _finite(measurement.image_median) if measurement is not None else None
        if sky is None:
            sky = _finite(features.image_median)
        sky_mad = _finite(measurement.image_mad) if measurement is not None else None
        if sky_mad is None:
            sky_mad = _finite(features.image_mad)
        # Airmass-corrected extra extinction when an airmass model exists,
        # else the nightly residual, else the raw extinction against the QC
        # reference (a channel without airmass metadata still gets flags).
        extinction = _finite(features.extra_extinction_mag)
        if extinction is None:
            extinction = _finite(features.nightly_extinction_residual)
        if extinction is None:
            extinction = _finite(safe_log_extinction(_finite(features.transparency_ratio)))
        fwhm = _finite(features.psf_fwhm_native_pixels)
        if fwhm is None:
            fwhm = _finite(features.median_fwhm_native_pixels)
        if fwhm is None and measurement is not None:
            preview_fwhm = _finite(features.median_fwhm_preview_pixels)
            scale_x = _finite(measurement.preview_scale_x)
            scale_y = _finite(measurement.preview_scale_y)
            if preview_fwhm is not None and scale_x is not None and scale_y is not None:
                fwhm = preview_fwhm * math.sqrt(scale_x * scale_y)
        gate = result.quality_gate
        try:
            night = observing_night(
                result.metadata.observed_at, night_boundary_hours, observing_timezone
            )
        except (TypeError, ValueError):
            night = None
        registration = result.registration
        matrix = registration.matrix
        inputs.append(
            BlinkFrameInput(
                path=result.path,
                channel_id=result.group_id,
                target=str(result.metadata.target or ""),
                filter_name=str(result.metadata.filter_name or ""),
                night=night,
                observed_at=_observed_at_text(result.metadata.observed_at),
                airmass=_finite(result.metadata.airmass),
                sky=sky,
                sky_mad=sky_mad,
                star_count=int(result.star_count),
                transparency=_finite(features.transparency_ratio),
                extinction_mag=extinction,
                fwhm_native=fwhm,
                ellipticity=_finite(features.median_ellipticity),
                eccentricity=_finite(features.median_eccentricity),
                registration_ok=bool(registration.ok),
                matched_stars=int(registration.matched_stars),
                registration_rms=_finite(registration.rms_pixels),
                overlap=_finite(features.overlap_fraction),
                background_shape=shapes.get(result.path),
                gate_disposition=gate.disposition.value if gate is not None else None,
                gate_codes=tuple(item.code for item in gate.evidence) if gate is not None else (),
                transform=(
                    tuple(tuple(float(item) for item in row) for row in matrix)
                    if matrix is not None
                    else None
                ),
                source_sha256=(
                    f"sha256:{result.identity.sha256}" if result.identity is not None else None
                ),
                is_qc_reference=(
                    result.reference_path is not None and result.reference_path == result.path
                ),
            )
        )
    return inputs


def _percentile_or_extreme(
    values: Sequence[float], percentile: float, *, minimum_frames: int, high: bool
) -> float | None:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if finite.size == 0:
        return None
    if finite.size < minimum_frames:
        return float(np.max(finite) if high else np.min(finite))
    return float(np.percentile(finite, percentile))


def _channel_statistics(
    frames: Sequence[BlinkFrameInput], policy: BlinkFlagPolicy
) -> ChannelStatistics:
    not_hard_fail = [
        frame for frame in frames if frame.gate_disposition != GateDisposition.HARD_FAIL.value
    ]
    sources_best = _percentile_or_extreme(
        [float(frame.star_count) for frame in not_hard_fail if frame.star_count > 0],
        policy.sources_best_percentile,
        minimum_frames=policy.sources_best_percentile_minimum_frames,
        high=True,
    )
    fwhm_values = [frame.fwhm_native for frame in not_hard_fail if frame.fwhm_native is not None and frame.fwhm_native > 0]
    fwhm_best = (
        float(max(np.min(fwhm_values), np.percentile(fwhm_values, policy.fwhm_best_percentile)))
        if fwhm_values
        else None
    )
    clean = [
        frame
        for frame in not_hard_fail
        if frame.registration_ok
        and frame.extinction_mag is not None
        and frame.extinction_mag < policy.clean_extinction_mag
        and _ratio(float(frame.star_count), sources_best) is not None
        and _ratio(float(frame.star_count), sources_best) >= policy.clean_source_ratio
        and frame.sky is not None
    ]
    sky_clean = (
        float(np.median([frame.sky for frame in clean]))
        if len(clean) >= policy.minimum_clean_frames
        else None
    )
    gradients = [
        frame.gradient_adu / frame.transparency
        for frame in clean
        if frame.gradient_adu is not None and frame.transparency is not None and frame.transparency > 0
    ]
    gradient_clean = (
        float(np.median(gradients)) if len(gradients) >= policy.minimum_clean_frames else None
    )
    return ChannelStatistics(
        channel_id=frames[0].channel_id if frames else "",
        frame_count=len(frames),
        clean_count=len(clean),
        sky_clean=sky_clean,
        sources_best=sources_best,
        fwhm_best=fwhm_best,
        gradient_clean=gradient_clean if gradient_clean and gradient_clean > 0 else None,
    )


def channel_statistics(
    inputs: Sequence[BlinkFrameInput], policy: BlinkFlagPolicy | None = None
) -> dict[str, ChannelStatistics]:
    """The clean-sky, richest-count and best-FWHM references of every channel."""

    selected = policy or BlinkFlagPolicy()
    selected.validate()
    by_channel: dict[str, list[BlinkFrameInput]] = defaultdict(list)
    for frame in inputs:
        by_channel[frame.channel_id].append(frame)
    return {
        channel_id: _channel_statistics(frames, selected)
        for channel_id, frames in by_channel.items()
    }


def _absolute_flags(
    frame: BlinkFrameInput, stats: ChannelStatistics, policy: BlinkFlagPolicy
) -> tuple[list[BlinkFlag], dict[str, Any]]:
    flags: list[BlinkFlag] = []
    sky_ratio = _ratio(frame.sky, stats.sky_clean)
    source_ratio = _ratio(float(frame.star_count), stats.sources_best)
    fwhm_ratio = _ratio(frame.fwhm_native, stats.fwhm_best)
    gradient_ratio = (
        _ratio(frame.gradient_adu / frame.transparency, stats.gradient_clean)
        if frame.gradient_adu is not None and frame.transparency is not None and frame.transparency > 0
        else None
    )
    extinction = frame.extinction_mag

    # Sky level against the channel's clean sky; a brighter but transparent
    # night is only highlighted, a bright night that also lost its stars is
    # the moonlit/hazy case and is pre-dropped (both flags say "combined").
    sky_bright = sky_ratio is not None and sky_ratio >= policy.sky_bright_ratio
    sources_attention = (
        source_ratio is not None and source_ratio <= policy.sources_low_attention_ratio
    )
    combined = bool(
        sky_bright
        and source_ratio is not None
        and source_ratio <= policy.sky_bright_exclude_source_ratio
    )
    if sky_bright:
        assert sky_ratio is not None
        message = f"Sky {sky_ratio:.2f}x the channel's clean-sky level"
        if combined:
            assert source_ratio is not None
            message += f" and {source_ratio * 100:.0f} % of its stars: moonlit or hazy night"
        flags.append(
            BlinkFlag(
                "BLINK_SKY_BRIGHT",
                SEVERITY_EXCLUDE if combined else SEVERITY_ATTENTION,
                message,
                value=_round(sky_ratio),
                threshold=policy.sky_bright_ratio,
                combined=combined,
            )
        )
    if source_ratio is not None:
        if source_ratio <= policy.sources_low_exclude_ratio:
            flags.append(
                BlinkFlag(
                    "BLINK_SOURCES_LOW",
                    SEVERITY_EXCLUDE,
                    f"Only {source_ratio * 100:.0f} % of the channel's best star count",
                    value=_round(source_ratio),
                    threshold=policy.sources_low_exclude_ratio,
                )
            )
        elif sources_attention:
            flags.append(
                BlinkFlag(
                    "BLINK_SOURCES_LOW",
                    SEVERITY_ATTENTION,
                    f"{source_ratio * 100:.0f} % of the channel's best star count",
                    value=_round(source_ratio),
                    threshold=policy.sources_low_attention_ratio,
                    combined=combined,
                )
            )
    if extinction is not None and extinction >= policy.extinction_attention_mag:
        exclude = extinction >= policy.extinction_exclude_mag
        flags.append(
            BlinkFlag(
                "BLINK_EXTINCTION",
                SEVERITY_EXCLUDE if exclude else SEVERITY_ATTENTION,
                f"Extra extinction {extinction:.2f} mag against the clear-sky envelope",
                value=_round(extinction),
                threshold=(
                    policy.extinction_exclude_mag if exclude else policy.extinction_attention_mag
                ),
            )
        )
    if (
        frame.background_shape is not None
        and frame.background_shape >= policy.background_shape_attention
    ):
        flags.append(
            BlinkFlag(
                "BLINK_BACKGROUND_SHAPE",
                SEVERITY_ATTENTION,
                f"Background shape differs from the channel's by {frame.background_shape:.2f} "
                "(spread of the normalized grid difference)",
                value=_round(frame.background_shape),
                threshold=policy.background_shape_attention,
            )
        )
    if gradient_ratio is not None and gradient_ratio >= policy.gradient_amplitude_attention_ratio:
        exclude = gradient_ratio >= policy.gradient_amplitude_exclude_ratio
        flags.append(
            BlinkFlag(
                "BLINK_GRADIENT_AMPLITUDE",
                SEVERITY_EXCLUDE if exclude else SEVERITY_ATTENTION,
                f"Calibrated background gradient {gradient_ratio:.1f}x the clean set's after flux scaling",
                value=_round(gradient_ratio),
                threshold=(
                    policy.gradient_amplitude_exclude_ratio
                    if exclude
                    else policy.gradient_amplitude_attention_ratio
                ),
            )
        )
    if fwhm_ratio is not None and fwhm_ratio >= policy.fwhm_wide_attention_ratio:
        exclude = fwhm_ratio >= policy.fwhm_wide_exclude_ratio
        flags.append(
            BlinkFlag(
                "BLINK_FWHM_WIDE",
                SEVERITY_EXCLUDE if exclude else SEVERITY_ATTENTION,
                f"PSF FWHM {fwhm_ratio:.2f}x the channel's best",
                value=_round(fwhm_ratio),
                threshold=(
                    policy.fwhm_wide_exclude_ratio if exclude else policy.fwhm_wide_attention_ratio
                ),
            )
        )
    if frame.ellipticity is not None and frame.ellipticity >= policy.stars_elongated_attention:
        flags.append(
            BlinkFlag(
                "BLINK_STARS_ELONGATED",
                SEVERITY_ATTENTION,
                f"Median star ellipticity {frame.ellipticity:.2f}",
                value=_round(frame.ellipticity),
                threshold=policy.stars_elongated_attention,
            )
        )
    if not frame.registration_ok:
        flags.append(
            BlinkFlag(
                "BLINK_UNREGISTRABLE",
                SEVERITY_EXCLUDE,
                "No registration transform could be estimated; the run would fail on this frame",
            )
        )
    if frame.star_count < policy.few_stars_minimum:
        flags.append(
            BlinkFlag(
                "BLINK_FEW_STARS",
                SEVERITY_EXCLUDE,
                f"Only {frame.star_count} sources detected",
                value=float(frame.star_count),
                threshold=float(policy.few_stars_minimum),
            )
        )
    metrics = {
        "sky": _round(frame.sky, 2),
        "skyRatio": _round(sky_ratio),
        "starCount": int(frame.star_count),
        "sourceRatio": _round(source_ratio),
        "extinctionMag": _round(extinction),
        "transparency": _round(frame.transparency),
        "fwhmNative": _round(frame.fwhm_native),
        "fwhmRatio": _round(fwhm_ratio),
        "ellipticity": _round(frame.ellipticity),
        "eccentricity": _round(frame.eccentricity),
        "registrationRms": _round(frame.registration_rms),
        "matchedStars": int(frame.matched_stars),
        "overlap": _round(frame.overlap),
        "backgroundShape": _round(frame.background_shape),
        "gradientRatio": _round(gradient_ratio),
    }
    return flags, metrics


def _gate_flags(
    frame: BlinkFrameInput, existing: Sequence[BlinkFlag]
) -> tuple[list[BlinkFlag], list[str]]:
    flags: list[BlinkFlag] = []
    notes: list[str] = []
    present = {item.code for item in existing}
    for code in frame.gate_codes:
        if code in EVIDENCE_INSUFFICIENCY_CODES:
            notes.append(code)
            continue
        if code in _GATE_CODES_COVERED:
            continue
        mapped = GATE_FLAG_MAPPING.get(code)
        if mapped is None:
            continue
        flag_code, severity = mapped
        if flag_code == "BLINK_REGISTRATION_WEAK" and not frame.registration_ok:
            continue  # BLINK_UNREGISTRABLE already says it
        if flag_code in present:
            continue
        present.add(flag_code)
        flags.append(
            BlinkFlag(flag_code, severity, f"{_GATE_FLAG_MESSAGES[flag_code]} ({code})")
        )
    return flags, notes


def compute_flags(
    inputs: Sequence[BlinkFrameInput], policy: BlinkFlagPolicy | None = None
) -> list[BlinkFrameFlags]:
    """Flag every frame against its channel; the result is in input order."""

    selected = policy or BlinkFlagPolicy()
    selected.validate()
    stats = channel_statistics(inputs, selected)
    results: list[BlinkFrameFlags] = []
    for frame in inputs:
        absolute, metrics = _absolute_flags(frame, stats[frame.channel_id], selected)
        mapped, notes = _gate_flags(frame, absolute)
        flags = tuple(absolute + mapped)
        exclude = any(item.severity == SEVERITY_EXCLUDE for item in flags)
        results.append(
            BlinkFrameFlags(
                path=frame.path,
                channel_id=frame.channel_id,
                flags=flags,
                default_decision=DECISION_DROP if exclude else DECISION_KEEP,
                notes=tuple(dict.fromkeys(notes)),
                metrics=metrics,
            )
        )
    return results


def _median(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return float(np.median(finite)) if finite else None


def night_summaries(
    inputs: Sequence[BlinkFrameInput], flags: Sequence[BlinkFrameFlags]
) -> list[dict[str, Any]]:
    """Per channel and night: counts, median sky/source/extinction, default drop.

    A night whose every frame carries an EXCLUDE flag is marked
    ``defaultDropNight`` so the blink view can offer "drop night" first.
    """

    if len(inputs) != len(flags):
        raise ValueError("night_summaries needs one flag record per input")
    grouped: dict[tuple[str, str], list[tuple[BlinkFrameInput, BlinkFrameFlags]]] = {}
    for frame, record in zip(inputs, flags, strict=True):
        if frame.path != record.path:
            raise ValueError("night_summaries inputs and flags are not aligned")
        grouped.setdefault((frame.channel_id, frame.night or "UNKNOWN"), []).append((frame, record))
    summaries: list[dict[str, Any]] = []
    for (channel_id, night), items in grouped.items():
        exclude = sum(record.exclude for _, record in items)
        attention = sum(record.attention for _, record in items)
        summaries.append(
            {
                "channelId": channel_id,
                "night": night,
                "frameCount": len(items),
                "medianSky": _round(_median(frame.sky for frame, _ in items), 2),
                "skyRatio": _round(_median(record.metrics.get("skyRatio") for _, record in items)),
                "medianSourceRatio": _round(
                    _median(record.metrics.get("sourceRatio") for _, record in items)
                ),
                "medianExtinction": _round(_median(frame.extinction_mag for frame, _ in items)),
                "exclude": exclude,
                "attention": attention,
                "defaultDropNight": bool(items) and exclude == len(items),
            }
        )
    return summaries


__all__ = [
    "BLINK_FLAGS_VERSION",
    "BlinkFlag",
    "BlinkFlagPolicy",
    "BlinkFrameFlags",
    "BlinkFrameInput",
    "ChannelStatistics",
    "DECISION_DROP",
    "DECISION_KEEP",
    "EVIDENCE_INSUFFICIENCY_CODES",
    "GATE_FLAG_MAPPING",
    "SEVERITY_ATTENTION",
    "SEVERITY_EXCLUDE",
    "SHAPE_FAMILY_EXTINCTION_MAG",
    "SHAPE_FAMILY_MINIMUM_FRAMES",
    "SHAPE_MINIMUM_RELATIVE_SPREAD",
    "background_shape_statistic",
    "background_shapes",
    "blink_inputs",
    "channel_statistics",
    "compute_flags",
    "night_summaries",
]
