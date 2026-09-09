"""Independent, conservative admission gate for WBPP preparation.

The existing classifier explains likely defects.  This module answers a
different question: has a light frame provided enough independent evidence to
be copied automatically?  A frame passes only when every mandatory check is
available and no evidence family requests review.  Metrics from the same
physical family are collapsed to their highest severity before disposition.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

import numpy as np

from .config import QcConfig
from .morphology import measure_fragmented_trails
from .models import (
    EvidenceFamily,
    EvidenceSeverity,
    FrameMeasurement,
    FrameResult,
    FrameRole,
    GateDisposition,
    QualityEvidence,
    QualityGateResult,
    RegistrationMetrics,
)
from .nightly_statistics import (
    _observing_timezone,
    fit_nightly_extinction_envelope,
    night_robust_baseline,
    observing_night,
)
from .statistics import safe_log_extinction


@dataclass(frozen=True, slots=True)
class GatePolicy:
    version: str = "quality-gate-v1"
    # Implementation evidence changes must invalidate previous manual
    # approvals even when the user-facing thresholds remain identical.
    evidence_revision: int = field(default=3, init=False)
    minimum_pass_group_frames: int = 8
    minimum_night_frames_for_pass: int = 3
    minimum_registration_matches: int = 30
    minimum_registration_fraction: float = 0.25
    maximum_registration_rms: float = 1.5
    minimum_overlap_fraction: float = 0.50
    minimum_detected_sources: int = 20
    minimum_morphology_stars: int = 15
    finite_fraction_review: float = 0.999
    finite_fraction_hard: float = 0.95
    weak_spatial_dimming_mag: float = 0.18
    strong_spatial_dimming_mag: float = 0.45
    weak_extinction_mag: float = 0.35
    strong_extinction_mag: float = 0.45
    # A uniform night-wide shift is ambiguous: throughput, seasonal pointing,
    # filter state, or a different sky pedestal can change it without a bad
    # frame.  Keep sub-0.5 mag shifts admissible when every spatial, morphology,
    # background, registration, and within-night transparency family is clean;
    # the ordinary global normalization and quality weights handle that scalar
    # difference downstream. Extreme shifts still require review.
    night_zeropoint_review_mag: float = 0.50
    weak_source_retention: float = 0.70
    strong_source_retention: float = 0.45
    weak_background_z: float = 4.0
    strong_background_z: float = 6.0
    weak_background_relative: float = 0.25
    strong_background_relative: float = 0.50
    weak_noise_z: float = 4.0
    weak_noise_low_ratio: float = 0.55
    weak_noise_high_ratio: float = 1.80
    focus_ratio: float = 1.12
    focus_z: float = 3.5
    focus_delta_pixels: float = 0.25
    focus_standalone_ratio: float = 1.30
    night_focus_review_ratio: float = 1.15
    night_fwhm_review_ratio: float = 1.12
    trailing_review_median: float = 0.30
    trailing_review_p90: float = 0.55
    trailing_hard_median: float = 0.45
    trailing_hard_fraction: float = 0.60
    trailing_hard_coherence: float = 0.75
    trailing_hard_relative: float = 1.50
    trailing_ellipticity_cutoff: float = 0.35
    minimum_trailing_stars: int = 30
    weak_occlusion_area: float = 0.05
    strong_occlusion_area: float = 0.15
    very_strong_occlusion_area: float = 0.30
    weak_occlusion_density: float = 0.35
    strong_occlusion_density: float = 0.15
    weak_boundary_support: float = 0.50
    strong_boundary_support: float = 0.65
    minimum_background_support: float = 0.50
    night_boundary_hours: float = 12.0
    observing_timezone: str | None = None

    def validate(self) -> None:
        integer_fields = {
            "minimum_pass_group_frames",
            "minimum_night_frames_for_pass",
            "minimum_registration_matches",
            "minimum_detected_sources",
            "minimum_morphology_stars",
            "minimum_trailing_stars",
        }
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.minimum_pass_group_frames < 8:
            raise ValueError("minimum_pass_group_frames must be at least eight")
        if self.minimum_night_frames_for_pass < 2:
            raise ValueError("minimum_night_frames_for_pass must be at least two")
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "version":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("gate policy version cannot be empty")
                continue
            if item.name == "observing_timezone":
                _observing_timezone(value)
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{item.name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{item.name} must be finite")
        if not 0 < self.finite_fraction_hard <= self.finite_fraction_review <= 1:
            raise ValueError("finite fraction thresholds are invalid")
        if not 0 <= self.strong_source_retention <= self.weak_source_retention <= 1:
            raise ValueError("source retention thresholds are invalid")
        if not 0 <= self.night_boundary_hours < 24:
            raise ValueError("night_boundary_hours must be in [0, 24)")
        if not (
            0
            <= self.weak_occlusion_area
            <= self.strong_occlusion_area
            <= self.very_strong_occlusion_area
            <= 1
        ):
            raise ValueError("occlusion area thresholds are invalid")

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

    @classmethod
    def from_qc_config(cls, config: QcConfig) -> "GatePolicy":
        return cls(
            weak_spatial_dimming_mag=config.weak_spatial_transparency_p90_mag,
            strong_spatial_dimming_mag=config.strong_spatial_transparency_p90_mag,
            weak_extinction_mag=config.weak_extra_extinction_mag,
            strong_extinction_mag=config.strong_extra_extinction_mag,
            weak_source_retention=config.weak_completeness_ratio,
            strong_source_retention=config.strong_completeness_ratio,
            minimum_overlap_fraction=config.minimum_overlap_fraction,
            weak_occlusion_area=config.weak_occlusion_area,
            strong_occlusion_area=config.strong_occlusion_area,
            very_strong_occlusion_area=config.very_strong_occlusion_area,
            weak_occlusion_density=config.weak_occlusion_density_ratio,
            strong_occlusion_density=config.strong_occlusion_density_ratio,
            weak_boundary_support=config.weak_boundary_support,
            strong_boundary_support=config.strong_boundary_support,
            observing_timezone=config.observing_timezone,
        )


_SEVERITY_RANK = {
    EvidenceSeverity.INFO: 0,
    EvidenceSeverity.WARNING: 1,
    EvidenceSeverity.REVIEW: 2,
    EvidenceSeverity.ERROR: 3,
    EvidenceSeverity.HARD_FAIL: 4,
}


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _evidence(
    code: str,
    family: EvidenceFamily,
    severity: EvidenceSeverity,
    message: str,
    *,
    value: Any = None,
    threshold: Any = None,
    units: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> QualityEvidence:
    return QualityEvidence(
        code=code,
        family=family,
        severity=severity,
        message=message,
        value=value,
        threshold=threshold,
        units=units,
        details=dict(details or {}),
    )


def _add_family(
    families: dict[EvidenceFamily, QualityEvidence], item: QualityEvidence
) -> None:
    previous = families.get(item.family)
    if previous is None or (
        _SEVERITY_RANK[item.severity], item.code
    ) > (
        _SEVERITY_RANK[previous.severity], previous.code
    ):
        families[item.family] = item


def _night_peer_stats(
    values: list[float | None], nights: list[str | None], index: int
) -> tuple[float | None, float | None]:
    current_night = nights[index]
    if current_night is None:
        return None, None
    peers = np.asarray(
        [
            value
            for peer, (value, night) in enumerate(zip(values, nights, strict=True))
            if peer != index and night == current_night and value is not None
        ],
        dtype=np.float64,
    )
    if not peers.size:
        return None, None
    center = float(np.median(peers))
    scale = 1.4826 * float(np.median(np.abs(peers - center)))
    return center, scale if scale > 1e-9 else None


def _robust_anomaly(
    value: float | None,
    baseline: float | None,
    scale: float | None,
) -> tuple[float | None, float | None, float | None]:
    if value is None or baseline is None:
        return None, None, None
    ratio = value / baseline if abs(baseline) > 1e-12 else None
    delta = value - baseline
    # A perfectly stable peer set has MAD=0.  It must not make a large outlier
    # invisible, so use a small relative floor rather than dropping z evidence.
    effective_scale = (
        scale if scale is not None and scale > 0 else max(abs(baseline) * 0.02, 1e-9)
    )
    z = delta / effective_scale
    return ratio, delta, z


def _grid_median(grid: list[list[float | None]]) -> float | None:
    values = [
        float(value)
        for row in grid
        for value in row
        if value is not None and math.isfinite(value)
    ]
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def _night_low_envelopes(
    values: list[float | None], nights: list[str | None]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, night in zip(values, nights, strict=True):
        if value is not None and night is not None:
            grouped[night].append(value)
    result: dict[str, float] = {}
    for night, samples in grouped.items():
        ordered = sorted(samples)
        count = max(1, int(math.ceil(0.10 * len(ordered))))
        result[night] = float(np.median(np.asarray(ordered[:count], dtype=np.float64)))
    return result


def _populate_native_morphology(
    result: FrameResult, measurement: FrameMeasurement, policy: GatePolicy
) -> None:
    features = result.features
    fragmented = measure_fragmented_trails(
        measurement.raw_stars if measurement.raw_stars is not None else measurement.stars,
        measurement.preview_width, measurement.preview_height,
    )
    if fragmented.available:
        features.fragmented_trailing_detected = fragmented.detected
        features.fragmented_trail_chain_count = fragmented.chain_count
        features.fragmented_trail_fraction = fragmented.fragment_fraction
        features.fragmented_trail_coherence = fragmented.orientation_coherence
        features.fragmented_trail_consensus_fraction = fragmented.consensus_fraction
        features.fragmented_trail_occupied_cells = fragmented.occupied_cells
        features.fragmented_trail_spatial_minor_fraction = fragmented.spatial_minor_fraction
    usable = [star for star in measurement.stars if star.flags == 0]
    if len(usable) < policy.minimum_morphology_stars:
        usable = list(measurement.stars)
    scale_x = _finite(measurement.preview_scale_x) or 1.0
    scale_y = _finite(measurement.preview_scale_y) or scale_x
    scale = math.sqrt(scale_x * scale_y)
    fwhm = [star.fwhm * scale for star in usable if math.isfinite(star.fwhm)]
    ellipticity = [
        star.ellipticity for star in usable if math.isfinite(star.ellipticity)
    ]
    axis_ratio = [
        min(star.a, star.b) / max(star.a, star.b)
        for star in usable
        if star.a > 0 and star.b > 0 and math.isfinite(star.a) and math.isfinite(star.b)
    ]
    eccentricity = [
        math.sqrt(max(0.0, 1.0 - ratio * ratio)) for ratio in axis_ratio
    ]
    if features.median_fwhm_native_pixels is None and fwhm:
        features.median_fwhm_native_pixels = float(np.median(fwhm))
    if features.p90_fwhm_native_pixels is None and fwhm:
        features.p90_fwhm_native_pixels = float(np.percentile(fwhm, 90))
    if features.median_eccentricity is None and eccentricity:
        features.median_eccentricity = float(np.median(eccentricity))
    if features.median_ellipticity is None and ellipticity:
        features.median_ellipticity = float(np.median(ellipticity))
    if features.p90_ellipticity is None and ellipticity:
        features.p90_ellipticity = float(np.percentile(ellipticity, 90))
    if features.median_axis_ratio is None and axis_ratio:
        features.median_axis_ratio = float(np.median(axis_ratio))
    if features.valid_morphology_star_count is None:
        features.valid_morphology_star_count = len(usable)
    if ellipticity:
        if features.elongated_fraction is None:
            features.elongated_fraction = sum(
                value >= policy.trailing_ellipticity_cutoff for value in ellipticity
            ) / len(ellipticity)
        oriented = [
            (float(star.ellipticity), float(star.theta))
            for star in usable
            if math.isfinite(star.ellipticity)
            and math.isfinite(star.theta)
            and star.ellipticity > 0
        ]
        if features.orientation_coherence is None and oriented:
            weights = np.asarray([item[0] for item in oriented], dtype=np.float64)
            angles = np.asarray([item[1] for item in oriented], dtype=np.float64)
            vector = np.sum(weights * np.exp(2j * angles))
            features.orientation_coherence = float(
                abs(vector) / max(float(np.sum(weights)), 1e-12)
            )


def _cohort_key(result: FrameResult) -> str:
    # analysis group ids already include target/field, filter, camera, geometry,
    # binning, gain/offset and exposure bucket.  Keep the gate independent of
    # the grouping implementation while preserving that scientific boundary.
    return result.group_id


def _policy(value: GatePolicy | QcConfig | None) -> GatePolicy:
    if value is None:
        result = GatePolicy()
    elif isinstance(value, GatePolicy):
        result = value
    elif isinstance(value, QcConfig):
        result = GatePolicy.from_qc_config(value)
    else:
        raise TypeError("quality gate config must be GatePolicy, QcConfig, or None")
    result.validate()
    return result


def evaluate_quality_gate(
    results: Iterable[FrameResult],
    measurements: Iterable[FrameMeasurement],
    config: GatePolicy | QcConfig | None = None,
) -> list[QualityGateResult]:
    """Evaluate and attach a conservative gate result to every frame result."""

    policy = _policy(config)
    policy_digest = policy.canonical_digest()
    ordered_results = list(results)
    measurement_by_path: dict[str, FrameMeasurement] = {}
    for measurement in measurements:
        path = measurement.metadata.path
        if path in measurement_by_path:
            raise ValueError(f"duplicate quality-gate measurement path: {path}")
        measurement_by_path[path] = measurement
    result_paths = [result.path for result in ordered_results]
    if len(set(result_paths)) != len(result_paths):
        raise ValueError("duplicate quality-gate result path")

    cohorts: dict[str, list[int]] = defaultdict(list)
    for index, result in enumerate(ordered_results):
        cohorts[_cohort_key(result)].append(index)

    gates: list[QualityGateResult | None] = [None] * len(ordered_results)
    for cohort_id in sorted(cohorts):
        indices = cohorts[cohort_id]
        cohort_results = [ordered_results[index] for index in indices]
        cohort_measurements = [measurement_by_path.get(result.path) for result in cohort_results]
        for result, measurement in zip(
            cohort_results, cohort_measurements, strict=True
        ):
            if measurement is not None:
                _populate_native_morphology(result, measurement, policy)
        nights = [
            observing_night(
                result.metadata.observed_at,
                policy.night_boundary_hours,
                policy.observing_timezone,
            )
            for result in cohort_results
        ]
        night_counts: dict[str, int] = defaultdict(int)
        for night in nights:
            if night is not None:
                night_counts[night] += 1
        extinction = [safe_log_extinction(result.features.transparency_ratio) for result in cohort_results]
        extinction_fit = fit_nightly_extinction_envelope(
            [result.metadata.airmass for result in cohort_results], extinction, nights
        )
        hfr_values = [_finite(result.features.nina_hfr_pixels) for result in cohort_results]
        fwhm_values = [_finite(result.features.median_fwhm_native_pixels) for result in cohort_results]
        background_values = [
            _finite(measurement.image_median) if measurement is not None else None
            for measurement in cohort_measurements
        ]
        noise_values = [
            (
                _grid_median(measurement.texture_grid)
                or _finite(measurement.image_mad)
            )
            if measurement is not None
            else None
            for measurement in cohort_measurements
        ]
        source_values = [
            float(measurement.detected_source_count)
            if measurement is not None and measurement.detected_source_count is not None
            else None
            for measurement in cohort_measurements
        ]
        source_baselines = night_robust_baseline(source_values, nights, "high")
        hfr_baselines = night_robust_baseline(hfr_values, nights, "low")
        fwhm_baselines = night_robust_baseline(fwhm_values, nights, "low")
        night_hfr = _night_low_envelopes(hfr_values, nights)
        night_fwhm = _night_low_envelopes(fwhm_values, nights)
        best_hfr = min(night_hfr.values()) if len(night_hfr) >= 2 else None
        best_fwhm = min(night_fwhm.values()) if len(night_fwhm) >= 2 else None
        best_night_offset = (
            min(extinction_fit.diagnostics.night_offsets.values())
            if len(extinction_fit.diagnostics.night_offsets) >= 2
            else None
        )

        # The geometric reference must be supported by at least one real edge;
        # its identity transform is never accepted as standalone registration.
        peers_by_reference: dict[str, list[RegistrationMetrics]] = defaultdict(list)
        for peer_result in cohort_results:
            if (
                peer_result.reference_path
                and peer_result.path != peer_result.reference_path
                and peer_result.registration.ok
            ):
                peers_by_reference[peer_result.reference_path].append(
                    peer_result.registration
                )

        for local_index, (result, measurement) in enumerate(
            zip(cohort_results, cohort_measurements, strict=True)
        ):
            families: dict[EvidenceFamily, QualityEvidence] = {}
            night_id = nights[local_index]
            if measurement is None:
                _add_family(
                    families,
                    _evidence(
                        "GATE_MEASUREMENT_MISSING",
                        EvidenceFamily.IDENTITY,
                        EvidenceSeverity.HARD_FAIL,
                        "No measurement is bound to this frame result.",
                    ),
                )
            else:
                result.features.image_median = measurement.image_median
                result.features.image_mad = measurement.image_mad
                if result.metadata.role is not FrameRole.LIGHT:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_NOT_A_LIGHT_FRAME",
                            EvidenceFamily.ROLE,
                            EvidenceSeverity.HARD_FAIL,
                            "Only authoritative light frames can pass the gate.",
                            value=result.metadata.role.value,
                        ),
                    )
                if measurement.status != "MEASURED" or measurement.error_code:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_MEASUREMENT_FAILED",
                            EvidenceFamily.IDENTITY,
                            EvidenceSeverity.HARD_FAIL,
                            "The frame could not be measured safely.",
                            details={"errorCode": measurement.error_code or "MEASUREMENT_FAILED"},
                        ),
                    )
                if measurement.identity is None or result.identity is None:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_IDENTITY_MISSING",
                            EvidenceFamily.IDENTITY,
                            EvidenceSeverity.HARD_FAIL,
                            "Content identity is missing.",
                        ),
                    )
                elif measurement.identity != result.identity:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_IDENTITY_MISMATCH",
                            EvidenceFamily.IDENTITY,
                            EvidenceSeverity.HARD_FAIL,
                            "Measurement and result identities disagree.",
                        ),
                    )
                finite_fraction = _finite(measurement.finite_fraction)
                if finite_fraction is None:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_FINITE_FRACTION_MISSING",
                            EvidenceFamily.PIXEL_STATISTICS,
                            EvidenceSeverity.REVIEW,
                            "Finite-pixel coverage was not measured.",
                        ),
                    )
                elif not 0 <= finite_fraction <= 1:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_FINITE_FRACTION_INVALID",
                            EvidenceFamily.PIXEL_STATISTICS,
                            EvidenceSeverity.HARD_FAIL,
                            "Finite-pixel coverage is outside [0, 1].",
                            value=finite_fraction,
                        ),
                    )
                elif finite_fraction < policy.finite_fraction_hard:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_FINITE_FRACTION_HARD",
                            EvidenceFamily.PIXEL_STATISTICS,
                            EvidenceSeverity.HARD_FAIL,
                            "Too much of the image is non-finite.",
                            value=finite_fraction,
                            threshold=policy.finite_fraction_hard,
                        ),
                    )
                elif finite_fraction < policy.finite_fraction_review:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_FINITE_FRACTION_REVIEW",
                            EvidenceFamily.PIXEL_STATISTICS,
                            EvidenceSeverity.REVIEW,
                            "Non-finite image area requires review.",
                            value=finite_fraction,
                            threshold=policy.finite_fraction_review,
                        ),
                    )
                dynamic_range = _finite(measurement.dynamic_range)
                median_value = _finite(measurement.image_median) or 0.0
                if dynamic_range is None:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_DYNAMIC_RANGE_MISSING",
                            EvidenceFamily.PIXEL_STATISTICS,
                            EvidenceSeverity.REVIEW,
                            "Robust image dynamic range was not measured.",
                        ),
                    )
                elif dynamic_range <= max(1e-8, abs(median_value) * 1e-8):
                    _add_family(
                        families,
                        _evidence(
                            "GATE_NEAR_CONSTANT_IMAGE",
                            EvidenceFamily.PIXEL_STATISTICS,
                            EvidenceSeverity.HARD_FAIL,
                            "The image has no usable dynamic range.",
                            value=dynamic_range,
                        ),
                    )

            if len(indices) < policy.minimum_pass_group_frames:
                _add_family(
                    families,
                    _evidence(
                        "GATE_INSUFFICIENT_COHORT",
                        EvidenceFamily.PROVENANCE,
                        EvidenceSeverity.REVIEW,
                        "Fewer than eight comparable frames cannot establish an automatic pass.",
                        value=len(indices),
                        threshold=policy.minimum_pass_group_frames,
                        details={"cohortId": cohort_id},
                    ),
                )
            if night_id is None:
                _add_family(
                    families,
                    _evidence(
                        "GATE_NIGHT_UNRESOLVED",
                        EvidenceFamily.METADATA,
                        EvidenceSeverity.REVIEW,
                        "Observation night cannot be resolved.",
                        ),
                    )
            elif night_counts[night_id] < policy.minimum_night_frames_for_pass:
                _add_family(
                    families,
                    _evidence(
                        "GATE_INSUFFICIENT_NIGHT_BASELINE",
                        EvidenceFamily.PROVENANCE,
                        EvidenceSeverity.REVIEW,
                        "Too few same-night peers are available for condition baselines.",
                        value=night_counts[night_id],
                        threshold=policy.minimum_night_frames_for_pass,
                        details={"nightId": night_id},
                    ),
                )

            registration = result.registration
            if result.path == result.reference_path:
                best_peer = max(
                    peers_by_reference.get(result.path, []),
                    key=lambda item: (
                        item.matched_stars,
                        item.match_fraction,
                        -(
                            _finite(item.rms_pixels)
                            if _finite(item.rms_pixels) is not None
                            else math.inf
                        ),
                    ),
                    default=None,
                )
                registration = best_peer or registration
                if best_peer is None:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_REFERENCE_NOT_CONNECTED",
                            EvidenceFamily.REGISTRATION,
                            EvidenceSeverity.REVIEW,
                            "The geometric reference has no independent registration edge.",
                        ),
                    )
            rms = _finite(registration.rms_pixels)
            registration_pass = bool(
                registration.ok
                and registration.matched_stars >= policy.minimum_registration_matches
                and registration.match_fraction >= policy.minimum_registration_fraction
                and rms is not None
                and rms <= policy.maximum_registration_rms
            )
            if not registration_pass:
                _add_family(
                    families,
                    _evidence(
                        "GATE_REGISTRATION_REVIEW",
                        EvidenceFamily.REGISTRATION,
                        EvidenceSeverity.REVIEW,
                        "Registration evidence is below the automatic-pass boundary.",
                        value={
                            "ok": registration.ok,
                            "matches": registration.matched_stars,
                            "fraction": registration.match_fraction,
                            "rms": rms,
                        },
                        threshold={
                            "matches": policy.minimum_registration_matches,
                            "fraction": policy.minimum_registration_fraction,
                            "maximumRms": policy.maximum_registration_rms,
                        },
                    ),
                )
            overlap = _finite(result.features.overlap_fraction)
            sparse_overlap_inferred = bool(
                overlap is None
                and registration.ok
                and registration.matched_stars >= policy.minimum_registration_matches
                and registration.match_fraction >= 0.75
            )
            if not sparse_overlap_inferred and (
                overlap is None or overlap < policy.minimum_overlap_fraction
            ):
                _add_family(
                    families,
                    _evidence(
                        "GATE_COMMON_FOOTPRINT_REVIEW",
                        EvidenceFamily.REGISTRATION,
                        EvidenceSeverity.REVIEW,
                        "Common-field coverage is insufficient or unavailable.",
                        value=overlap,
                        threshold=policy.minimum_overlap_fraction,
                    ),
                )

            extinction_residual = extinction_fit.residuals[local_index]
            result.features.nightly_extinction_residual = extinction_residual
            airmass_explained = bool(
                extinction_fit.diagnostics.reliable
                and extinction_residual is not None
                and extinction_residual < policy.weak_extinction_mag
            )
            if extinction_fit.diagnostics.reliable and extinction_residual is not None:
                if extinction_residual >= policy.strong_extinction_mag:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_TEMPORAL_EXTINCTION_STRONG",
                            EvidenceFamily.TRANSPARENCY,
                            EvidenceSeverity.ERROR,
                            "Extinction exceeds the nightly airmass clear envelope.",
                            value=extinction_residual,
                            threshold=policy.strong_extinction_mag,
                            units="mag",
                        ),
                    )
                elif extinction_residual >= policy.weak_extinction_mag:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_TEMPORAL_EXTINCTION_REVIEW",
                            EvidenceFamily.TRANSPARENCY,
                            EvidenceSeverity.REVIEW,
                            "Nightly airmass-corrected extinction requires review.",
                            value=extinction_residual,
                            threshold=policy.weak_extinction_mag,
                            units="mag",
                        ),
                    )

            if (
                night_id is not None
                and best_night_offset is not None
                and night_id in extinction_fit.diagnostics.night_offsets
            ):
                night_shift = (
                    extinction_fit.diagnostics.night_offsets[night_id]
                    - best_night_offset
                )
                if night_shift >= policy.night_zeropoint_review_mag:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_NIGHT_ZEROPOINT_SHIFT_REVIEW",
                            EvidenceFamily.TRANSPARENCY,
                            EvidenceSeverity.REVIEW,
                            "The entire observing night is dimmer than the cohort clear envelope.",
                            value=night_shift,
                            threshold=policy.night_zeropoint_review_mag,
                            units="mag",
                            details={"nightId": night_id},
                        ),
                    )
            spatial = _finite(result.features.spatial_dimming_p90_mag)
            if spatial is not None and spatial >= policy.strong_spatial_dimming_mag:
                _add_family(
                    families,
                    _evidence(
                        "GATE_SPATIAL_DIMMING_STRONG",
                        EvidenceFamily.CONSENSUS,
                        EvidenceSeverity.ERROR,
                        "Strong localized transparency loss is present.",
                        value=spatial,
                        threshold=policy.strong_spatial_dimming_mag,
                        units="mag",
                    ),
                )
            elif spatial is not None and spatial >= policy.weak_spatial_dimming_mag:
                _add_family(
                    families,
                    _evidence(
                        "GATE_SPATIAL_DIMMING_REVIEW",
                        EvidenceFamily.CONSENSUS,
                        EvidenceSeverity.REVIEW,
                        "Localized transparency loss requires review.",
                        value=spatial,
                        threshold=policy.weak_spatial_dimming_mag,
                        units="mag",
                    ),
                )

            # Matched-star completeness is geometric evidence within the common
            # footprint and is not erased by airmass.  Raw thresholded source
            # counts, however, can fall sharply at low altitude even on a clear
            # frame, so those are suppressed when the extinction model explains
            # the global change.
            retention_values = [_finite(result.features.star_completeness)]
            if not airmass_explained:
                retention_values.append(
                    _finite(result.features.detected_source_ratio)
                )
            if (
                not airmass_explained
                and source_baselines[local_index]
                and source_values[local_index] is not None
            ):
                retention_values.append(source_values[local_index] / source_baselines[local_index])
            retention = min((value for value in retention_values if value is not None), default=None)
            if retention is not None:
                severity = None
                code = ""
                threshold = None
                if retention <= policy.strong_source_retention:
                    severity = EvidenceSeverity.ERROR
                    code = "GATE_SOURCE_RETENTION_STRONG"
                    threshold = policy.strong_source_retention
                elif retention <= policy.weak_source_retention:
                    severity = EvidenceSeverity.REVIEW
                    code = "GATE_SOURCE_RETENTION_REVIEW"
                    threshold = policy.weak_source_retention
                if severity is not None:
                    _add_family(
                        families,
                        _evidence(
                            code,
                            EvidenceFamily.CONSENSUS,
                            severity,
                            "Detected-source retention is below the nightly clear envelope.",
                            value=retention,
                            threshold=threshold,
                        ),
                    )
            if (
                source_values[local_index] is not None
                and source_values[local_index] < policy.minimum_detected_sources
            ):
                _add_family(
                    families,
                    _evidence(
                        "GATE_ABSOLUTE_LOW_SOURCE_COUNT",
                        EvidenceFamily.CONSENSUS,
                        EvidenceSeverity.REVIEW,
                        "Too few sources were detected for an automatic pass.",
                        value=source_values[local_index],
                        threshold=policy.minimum_detected_sources,
                    ),
                )
            elif source_values[local_index] is None:
                _add_family(
                    families,
                    _evidence(
                        "GATE_SOURCE_COUNT_MISSING",
                        EvidenceFamily.CONSENSUS,
                        EvidenceSeverity.REVIEW,
                        "Detected-source count is unavailable.",
                    ),
                )
            morphology_count_now = result.features.valid_morphology_star_count or 0
            if morphology_count_now < policy.minimum_morphology_stars:
                _add_family(
                    families,
                    _evidence(
                        "GATE_MORPHOLOGY_SAMPLE_REVIEW",
                        EvidenceFamily.MORPHOLOGY,
                        EvidenceSeverity.REVIEW,
                        "Too few valid stars are available for shape assessment.",
                        value=morphology_count_now,
                        threshold=policy.minimum_morphology_stars,
                    ),
                )

            background, background_scale = _night_peer_stats(background_values, nights, local_index)
            background_ratio, _, background_z = _robust_anomaly(
                background_values[local_index], background, background_scale
            )
            result.features.background_z = background_z
            if background_ratio is not None and background_z is not None:
                relative = abs(background_ratio - 1.0)
                if (
                    abs(background_z) >= policy.strong_background_z
                    and relative >= policy.strong_background_relative
                ):
                    _add_family(
                        families,
                        _evidence(
                            "GATE_BACKGROUND_STRONG",
                            EvidenceFamily.BACKGROUND,
                            EvidenceSeverity.ERROR,
                            "Background signal is a strong nightly outlier.",
                            value={"z": background_z, "ratio": background_ratio},
                        ),
                    )
                elif (
                    abs(background_z) >= policy.weak_background_z
                    and relative >= policy.weak_background_relative
                ):
                    _add_family(
                        families,
                        _evidence(
                            "GATE_BACKGROUND_REVIEW",
                            EvidenceFamily.BACKGROUND,
                            EvidenceSeverity.REVIEW,
                            "Background signal is a nightly outlier.",
                            value={"z": background_z, "ratio": background_ratio},
                        ),
                    )
            noise, noise_scale = _night_peer_stats(noise_values, nights, local_index)
            noise_ratio, _, noise_z = _robust_anomaly(noise_values[local_index], noise, noise_scale)
            result.features.noise_z = noise_z
            if noise_ratio is not None and noise_z is not None and (
                noise_ratio <= policy.weak_noise_low_ratio
                or noise_ratio >= policy.weak_noise_high_ratio
            ) and abs(noise_z) >= policy.weak_noise_z:
                _add_family(
                    families,
                    _evidence(
                        "GATE_NOISE_REVIEW",
                        EvidenceFamily.BACKGROUND,
                        EvidenceSeverity.REVIEW,
                        "Background noise is a nightly outlier.",
                        value={"z": noise_z, "ratio": noise_ratio},
                    ),
                )

            hfr_baseline = hfr_baselines[local_index]
            fwhm_baseline = fwhm_baselines[local_index]
            focus_anomalies: list[dict[str, float | None]] = []
            for focus_series, focus_value, focus_baseline in (
                (hfr_values, hfr_values[local_index], hfr_baseline),
                (fwhm_values, fwhm_values[local_index], fwhm_baseline),
            ):
                _, focus_scale = _night_peer_stats(
                    focus_series,
                    nights,
                    local_index,
                )
                ratio, delta, z = _robust_anomaly(focus_value, focus_baseline, focus_scale)
                abnormal = bool(
                    ratio is not None
                    and delta is not None
                    and ratio >= policy.focus_ratio
                    and delta >= policy.focus_delta_pixels
                    and z is not None
                    and z >= policy.focus_z
                )
                focus_anomalies.append(
                    {"ratio": ratio, "delta": delta, "z": z, "abnormal": float(abnormal)}
                )
            jointly_abnormal = all(item["abnormal"] == 1.0 for item in focus_anomalies)
            standalone_severe = any(
                item["ratio"] is not None
                and float(item["ratio"]) >= policy.focus_standalone_ratio
                for item in focus_anomalies
            )
            if jointly_abnormal or standalone_severe:
                _add_family(
                    families,
                    _evidence(
                        "GATE_FOCUS_SEEING_REVIEW",
                        EvidenceFamily.MORPHOLOGY,
                        EvidenceSeverity.REVIEW,
                        "HFR and PSF width jointly indicate degraded focus or seeing.",
                        value={"hfr": focus_anomalies[0], "fwhm": focus_anomalies[1]},
                    ),
                )
            if (
                night_id is not None
                and best_fwhm is not None
                and night_id in night_fwhm
            ):
                night_hfr_ratio = (
                    night_hfr[night_id] / max(best_hfr, 1e-9)
                    if best_hfr is not None and night_id in night_hfr
                    else None
                )
                night_fwhm_ratio = night_fwhm[night_id] / max(best_fwhm, 1e-9)
                jointly_degraded = (
                    night_hfr_ratio is not None
                    and night_hfr_ratio >= policy.night_focus_review_ratio
                    and night_fwhm_ratio >= policy.night_fwhm_review_ratio
                )
                # N.I.N.A. HFR is optional. A whole blurry night establishes
                # its own poor within-night baseline, so retain the same
                # standalone severe FWHM boundary across comparable nights.
                standalone_degraded = (
                    night_fwhm_ratio >= policy.focus_standalone_ratio
                    and night_fwhm[night_id] - best_fwhm >= policy.focus_delta_pixels
                )
                if jointly_degraded or standalone_degraded:
                    _add_family(
                        families,
                        _evidence(
                            "GATE_NIGHT_FOCUS_SHIFT_REVIEW",
                            EvidenceFamily.MORPHOLOGY,
                            EvidenceSeverity.REVIEW,
                            "Night-level PSF width, with HFR when available, is degraded versus the cohort best night.",
                            value={
                                "hfrRatio": night_hfr_ratio,
                                "fwhmRatio": night_fwhm_ratio,
                            },
                            threshold={
                                "hfrRatio": policy.night_focus_review_ratio,
                                "fwhmRatio": policy.night_fwhm_review_ratio,
                                "standaloneFwhmRatio": policy.focus_standalone_ratio,
                            },
                            details={"nightId": night_id},
                        ),
                    )

            median_e = _finite(result.features.median_ellipticity)
            p90_e = _finite(result.features.p90_ellipticity)
            elongated = _finite(result.features.elongated_fraction)
            coherence = _finite(result.features.orientation_coherence)
            morphology_count = result.features.valid_morphology_star_count or 0
            e_values = [_finite(item.features.median_ellipticity) for item in cohort_results]
            e_baseline, _ = _night_peer_stats(e_values, nights, local_index)
            e_relative = (
                median_e / max(e_baseline, 1e-6)
                if median_e is not None and e_baseline is not None
                else None
            )
            if result.features.fragmented_trailing_detected:
                _add_family(
                    families,
                    _evidence(
                        "GATE_FRAGMENTED_TRAILING_HARD",
                        EvidenceFamily.MORPHOLOGY,
                        EvidenceSeverity.HARD_FAIL,
                        "Many stars form coherent, deblended tracking trails across the field.",
                        value={
                            "chains": result.features.fragmented_trail_chain_count,
                            "fragmentFraction": result.features.fragmented_trail_fraction,
                            "orientationCoherence": result.features.fragmented_trail_coherence,
                            "directionConsensus": result.features.fragmented_trail_consensus_fraction,
                            "occupiedCells": result.features.fragmented_trail_occupied_cells,
                            "spatialMinorFraction": result.features.fragmented_trail_spatial_minor_fraction,
                        },
                    ),
                )
            if (
                morphology_count >= policy.minimum_trailing_stars
                and median_e is not None
                and median_e >= policy.trailing_hard_median
                and elongated is not None
                and elongated >= policy.trailing_hard_fraction
                and coherence is not None
                and coherence >= policy.trailing_hard_coherence
                and e_relative is not None
                and e_relative >= policy.trailing_hard_relative
            ):
                _add_family(
                    families,
                    _evidence(
                        "GATE_COHERENT_TRAILING_HARD",
                        EvidenceFamily.MORPHOLOGY,
                        EvidenceSeverity.HARD_FAIL,
                        "A strong, coherent full-field trail is present.",
                        value={
                            "medianEllipticity": median_e,
                            "elongatedFraction": elongated,
                            "orientationCoherence": coherence,
                            "relative": e_relative,
                        },
                    ),
                )
            elif (
                (median_e is not None and median_e >= policy.trailing_review_median)
                or (p90_e is not None and p90_e >= policy.trailing_review_p90)
            ):
                _add_family(
                    families,
                    _evidence(
                        "GATE_TRAILING_REVIEW",
                        EvidenceFamily.MORPHOLOGY,
                        EvidenceSeverity.REVIEW,
                        "Elongated star profiles require review.",
                        value={"medianEllipticity": median_e, "p90Ellipticity": p90_e},
                    ),
                )

            area = _finite(result.features.largest_missing_region) or 0.0
            density = _finite(result.features.missing_inside_outside_ratio)
            boundary = _finite(result.features.boundary_support) or 0.0
            background_support = _finite(result.features.background_support) or 0.0
            hard_occlusion = bool(
                (
                    area >= policy.strong_occlusion_area
                    and density is not None
                    and density <= policy.strong_occlusion_density
                    and boundary >= policy.strong_boundary_support
                    and background_support >= policy.minimum_background_support
                )
                or (
                    area >= policy.very_strong_occlusion_area
                    and density is not None
                    and density <= policy.weak_occlusion_density
                    and background_support >= policy.minimum_background_support
                )
            )
            if hard_occlusion:
                _add_family(
                    families,
                    _evidence(
                        "GATE_OCCLUSION_HARD",
                        EvidenceFamily.OCCLUSION,
                        EvidenceSeverity.HARD_FAIL,
                        "A large connected hard obstruction is present.",
                        value={"area": area, "density": density, "boundary": boundary},
                    ),
                )
            elif area >= policy.weak_occlusion_area and sum(
                (
                    density is not None and density <= policy.weak_occlusion_density,
                    boundary >= policy.weak_boundary_support,
                    background_support >= policy.minimum_background_support,
                )
            ) >= 2:
                _add_family(
                    families,
                    _evidence(
                        "GATE_OCCLUSION_REVIEW",
                        EvidenceFamily.OCCLUSION,
                        EvidenceSeverity.REVIEW,
                        "Connected missing-star geometry requires review.",
                        value={"area": area, "density": density, "boundary": boundary},
                    ),
                )

            cloud_prefixes = {
                EvidenceFamily.CONSENSUS: (
                    "GATE_SPATIAL_DIMMING_",
                    "GATE_SOURCE_RETENTION_",
                ),
                EvidenceFamily.TRANSPARENCY: ("GATE_TEMPORAL_EXTINCTION_",),
                EvidenceFamily.BACKGROUND: (
                    "GATE_BACKGROUND_",
                    "GATE_NOISE_",
                ),
            }
            cloud_families = [
                family
                for family, prefixes in cloud_prefixes.items()
                if family in families
                and families[family].code.startswith(prefixes)
                and _SEVERITY_RANK[families[family].severity]
                >= _SEVERITY_RANK[EvidenceSeverity.REVIEW]
            ]
            cloud_strong = any(
                _SEVERITY_RANK[families[family].severity]
                >= _SEVERITY_RANK[EvidenceSeverity.ERROR]
                for family in cloud_families
            )
            if (
                len(indices) >= policy.minimum_pass_group_frames
                and registration_pass
                and len(cloud_families) >= 2
                and cloud_strong
            ):
                _add_family(
                    families,
                    _evidence(
                        "GATE_MULTI_FAMILY_CLOUD_HARD",
                        EvidenceFamily.TRANSPARENCY,
                        EvidenceSeverity.HARD_FAIL,
                        "Multiple independent cloud evidence families agree.",
                        details={"families": sorted(family.value for family in cloud_families)},
                    ),
                )

            evidence = sorted(families.values(), key=lambda item: item.family.value)
            hard = any(item.severity is EvidenceSeverity.HARD_FAIL for item in evidence)
            review = any(
                _SEVERITY_RANK[item.severity] >= _SEVERITY_RANK[EvidenceSeverity.REVIEW]
                for item in evidence
            )
            disposition = (
                GateDisposition.HARD_FAIL
                if hard
                else GateDisposition.REVIEW
                if review
                else GateDisposition.PASS
            )
            gate = QualityGateResult(
                disposition=disposition,
                evidence=evidence,
                summary=(
                    f"cohort={cohort_id}; night={night_id or 'UNKNOWN'}; "
                    f"policy={policy_digest}; disposition={disposition.value}"
                ),
                version=f"{policy.version}@{policy_digest}",
                policy_digest=policy_digest,
                policy=policy.serializable(),
            )
            result.quality_gate = gate
            gates[indices[local_index]] = gate

    assert all(gate is not None for gate in gates)
    return [gate for gate in gates if gate is not None]


__all__ = ["GatePolicy", "evaluate_quality_gate"]
