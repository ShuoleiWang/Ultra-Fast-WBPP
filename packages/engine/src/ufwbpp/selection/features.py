"""Per-frame selection features extracted from the Light Frame QC results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lightframeqc.models import FrameMeasurement, FrameResult
from lightframeqc.nightly_statistics import observing_night


@dataclass(frozen=True, slots=True)
class FrameSelectionFeatures:
    path: str
    group_id: str
    filter_name: str
    target: str
    night_id: str | None
    is_reference: bool
    star_count: int
    registration_ok: bool
    registration_estimated: bool
    registration_rms: float | None
    matched_stars: int
    transparency: float | None
    extra_extinction_mag: float | None
    fwhm_native: float | None
    ellipticity: float | None
    orientation_coherence: float | None
    spatial_dimming_p90_mag: float | None
    star_completeness: float | None
    detected_source_ratio: float | None
    overlap_fraction: float | None
    occlusion_area: float | None
    occlusion_density: float | None
    boundary_support: float | None
    background_support: float | None
    background_z: float | None
    noise_z: float | None
    gate_disposition: str
    hard_fail_codes: tuple[str, ...]
    review_codes: tuple[str, ...]
    wing_fraction: float | None = None
    fwhm_source: str = "none"
    fwhm_night_ratio: float | None = None
    fwhm_group_ratio: float | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "groupId": self.group_id,
            "filter": self.filter_name,
            "target": self.target,
            "nightId": self.night_id,
            "isReference": self.is_reference,
            "starCount": self.star_count,
            "registrationOk": self.registration_ok,
            "registrationEstimated": self.registration_estimated,
            "registrationRms": self.registration_rms,
            "matchedStars": self.matched_stars,
            "transparency": self.transparency,
            "extraExtinctionMag": self.extra_extinction_mag,
            "fwhmNative": self.fwhm_native,
            "ellipticity": self.ellipticity,
            "orientationCoherence": self.orientation_coherence,
            "spatialDimmingP90Mag": self.spatial_dimming_p90_mag,
            "starCompleteness": self.star_completeness,
            "detectedSourceRatio": self.detected_source_ratio,
            "overlapFraction": self.overlap_fraction,
            "occlusionArea": self.occlusion_area,
            "occlusionDensity": self.occlusion_density,
            "boundarySupport": self.boundary_support,
            "backgroundSupport": self.background_support,
            "backgroundZ": self.background_z,
            "noiseZ": self.noise_z,
            "gateDisposition": self.gate_disposition,
            "hardFailCodes": list(self.hard_fail_codes),
            "reviewCodes": list(self.review_codes),
            "wingFraction": self.wing_fraction,
            "fwhmSource": self.fwhm_source,
            "fwhmNightRatio": self.fwhm_night_ratio,
            "fwhmGroupRatio": self.fwhm_group_ratio,
        }


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _resolved(path: str) -> str:
    try:
        return str(Path(path).resolve(strict=True))
    except OSError:
        return str(Path(path))


def _robust_best(values: Sequence[float]) -> float | None:
    """A robust 'best' (smallest) FWHM: the 10th percentile, never below the minimum."""

    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)])
    if finite.size == 0:
        return None
    if finite.size < 4:
        return float(np.min(finite))
    return float(max(np.min(finite), np.percentile(finite, 10)))


def extract_features(
    results: Sequence[FrameResult],
    measurements: Sequence[FrameMeasurement] | Mapping[str, FrameMeasurement] | None = None,
    *,
    night_boundary_hours: float = 12.0,
    observing_timezone: str | None = None,
) -> list[FrameSelectionFeatures]:
    """Extract the selection features of every QC result, in result order.

    FWHM ratios are relative to the robust best FWHM of the frame's night
    (within its group) and of its whole group; both use native-pixel FWHM when
    the gate populated it, otherwise the preview FWHM rescaled through the
    measurement's preview scale.
    """

    measurement_by_path: dict[str, FrameMeasurement] = {}
    if measurements is not None:
        items = measurements.values() if isinstance(measurements, Mapping) else measurements
        for item in items:
            measurement_by_path[_resolved(item.metadata.path)] = item

    partial: list[FrameSelectionFeatures] = []
    for result in results:
        features = result.features
        gate = result.quality_gate
        hard_codes: list[str] = []
        review_codes: list[str] = []
        if gate is not None:
            for item in gate.evidence:
                severity = getattr(item.severity, "value", str(item.severity))
                if severity == "HARD_FAIL":
                    hard_codes.append(item.code)
                elif severity in {"REVIEW", "ERROR"}:
                    review_codes.append(item.code)
        resolved = _resolved(result.path)
        fwhm_source = "native-stamps"
        fwhm = _finite(getattr(features, "psf_fwhm_native_pixels", None))
        if fwhm is None:
            fwhm_source = "preview-scaled"
            fwhm = _finite(features.median_fwhm_native_pixels)
        if fwhm is None:
            preview_fwhm = _finite(features.median_fwhm_preview_pixels)
            measurement = measurement_by_path.get(resolved)
            scale = None
            if measurement is not None:
                sx = _finite(getattr(measurement, "preview_scale_x", None))
                sy = _finite(getattr(measurement, "preview_scale_y", None))
                if sx is not None and sy is not None:
                    scale = float(np.sqrt(sx * sy))
            if preview_fwhm is not None and scale is not None:
                fwhm = preview_fwhm * scale
        if fwhm is None:
            fwhm_source = "none"
        night = None
        try:
            night = observing_night(
                result.metadata.observed_at, night_boundary_hours, observing_timezone
            )
        except (TypeError, ValueError):
            night = None
        registration = result.registration
        extra_extinction = _finite(features.nightly_extinction_residual)
        if extra_extinction is None:
            extra_extinction = _finite(features.extra_extinction_mag)
        partial.append(
            FrameSelectionFeatures(
                path=resolved,
                group_id=result.group_id,
                filter_name=str(result.metadata.filter_name or ""),
                target=str(result.metadata.target or ""),
                night_id=night,
                is_reference=(
                    result.reference_path is not None
                    and _resolved(result.reference_path) == resolved
                ),
                star_count=int(result.star_count),
                registration_ok=bool(registration.ok),
                registration_estimated=registration.matrix is not None,
                registration_rms=_finite(registration.rms_pixels),
                matched_stars=int(registration.matched_stars),
                transparency=_finite(features.transparency_ratio),
                extra_extinction_mag=extra_extinction,
                fwhm_native=fwhm,
                ellipticity=_finite(features.median_ellipticity),
                orientation_coherence=_finite(features.orientation_coherence),
                spatial_dimming_p90_mag=_finite(features.spatial_dimming_p90_mag),
                star_completeness=_finite(features.star_completeness),
                detected_source_ratio=_finite(features.detected_source_ratio),
                overlap_fraction=_finite(features.overlap_fraction),
                occlusion_area=_finite(features.largest_missing_region),
                occlusion_density=_finite(features.missing_inside_outside_ratio),
                boundary_support=_finite(features.boundary_support),
                background_support=_finite(features.background_support),
                background_z=_finite(features.background_z),
                noise_z=_finite(features.noise_z),
                gate_disposition=(
                    getattr(gate.disposition, "value", str(gate.disposition))
                    if gate is not None
                    else "HARD_FAIL"
                ),
                hard_fail_codes=tuple(hard_codes),
                review_codes=tuple(review_codes),
                wing_fraction=_finite(getattr(features, "psf_wing_fraction", None)),
                fwhm_source=fwhm_source,
            )
        )

    # FWHM ratios against the robust best of the night and of the group.
    group_best: dict[str, float | None] = {}
    night_best: dict[tuple[str, str | None], float | None] = {}
    for group_id in {item.group_id for item in partial}:
        members = [item for item in partial if item.group_id == group_id]
        group_best[group_id] = _robust_best(
            [item.fwhm_native for item in members if item.fwhm_native is not None]
        )
        for night_id in {item.night_id for item in members}:
            night_best[(group_id, night_id)] = _robust_best(
                [
                    item.fwhm_native
                    for item in members
                    if item.night_id == night_id and item.fwhm_native is not None
                ]
            )
    complete: list[FrameSelectionFeatures] = []
    for item in partial:
        night_ratio = None
        group_ratio = None
        if item.fwhm_native is not None:
            best_night = night_best.get((item.group_id, item.night_id))
            best_group = group_best.get(item.group_id)
            if best_night is not None and best_night > 0:
                night_ratio = item.fwhm_native / best_night
            if best_group is not None and best_group > 0:
                group_ratio = item.fwhm_native / best_group
        complete.append(
            FrameSelectionFeatures(
                **{**asdict(item), "fwhm_night_ratio": night_ratio, "fwhm_group_ratio": group_ratio}
            )
        )
    return complete
