"""Native quality weights derived from the registration SEP catalog."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Literal, Sequence

import numpy as np
import sep
from scipy.spatial import cKDTree

from .pipeline import FrameAnalysis, FrameTransform


Normalization = Literal["group-median", "first"]


@dataclass(frozen=True, slots=True)
class StellarScaleEstimate:
    source_index: int
    reference_index: int
    filter_name: str | None
    scale: float | None
    status: str
    evidence: dict[str, Any]


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        (np.asarray(points, dtype=np.float64), np.ones(len(points), dtype=np.float64))
    )
    mapped = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    valid = np.isfinite(mapped[:, 2]) & (np.abs(mapped[:, 2]) > 1e-12)
    output = np.full((len(points), 2), np.nan, dtype=np.float64)
    output[valid] = mapped[valid, :2] / mapped[valid, 2:3]
    return output


def _one_to_one_indices(
    source: np.ndarray,
    reference: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_finite = np.all(np.isfinite(source), axis=1)
    reference_finite = np.all(np.isfinite(reference), axis=1)
    source_indices = np.flatnonzero(source_finite)
    reference_indices = np.flatnonzero(reference_finite)
    if not source_indices.size or not reference_indices.size:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    tree = cKDTree(reference[reference_indices])
    neighbours = tree.query_ball_point(source[source_indices], radius)
    edges: list[tuple[float, int, int]] = []
    for local_source, candidates in enumerate(neighbours):
        source_index = int(source_indices[local_source])
        for local_reference in candidates:
            reference_index = int(reference_indices[int(local_reference)])
            distance = float(
                np.linalg.norm(source[source_index] - reference[reference_index])
            )
            edges.append((distance, source_index, reference_index))
    edges.sort()
    used_source: set[int] = set()
    used_reference: set[int] = set()
    selected: list[tuple[int, int]] = []
    for _distance, source_index, reference_index in edges:
        if source_index in used_source or reference_index in used_reference:
            continue
        used_source.add(source_index)
        used_reference.add(reference_index)
        selected.append((source_index, reference_index))
    return (
        np.asarray([item[0] for item in selected], dtype=np.int64),
        np.asarray([item[1] for item in selected], dtype=np.int64),
    )


def _aperture_fluxes(
    analysis: FrameAnalysis,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    native_per_preview = math.sqrt(analysis.scale_x * analysis.scale_y)
    radius = max(2.0, 10.0 / native_per_preview)
    annulus_inner = max(radius + 2.0, 45.0 / native_per_preview)
    annulus_outer = max(annulus_inner + 2.0, 65.0 / native_per_preview)
    points = analysis.catalog.points[indices]
    image = np.asarray(analysis.preview, dtype=np.float32)
    invalid = ~np.isfinite(image)
    finite = image[~invalid]
    if finite.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            {
                "apertureRadiusPreviewPixels": radius,
                "annulusInnerPreviewPixels": annulus_inner,
                "annulusOuterPreviewPixels": annulus_outer,
            },
        )
    work = image.copy()
    work[invalid] = np.float32(np.median(finite))
    flux, _error, flags = sep.sum_circle(
        work,
        points[:, 0],
        points[:, 1],
        radius,
        mask=invalid,
        bkgann=(annulus_inner, annulus_outer),
        subpix=1,
    )
    valid = (
        np.isfinite(flux)
        & (flux > 0)
        & (np.asarray(flags, dtype=np.int64) == 0)
    )
    return (
        np.asarray(flux[valid], dtype=np.float64),
        np.asarray(indices[valid], dtype=np.int64),
        {
            "apertureRadiusPreviewPixels": radius,
            "annulusInnerPreviewPixels": annulus_inner,
            "annulusOuterPreviewPixels": annulus_outer,
        },
    )


def _robust_ratio(
    values: np.ndarray,
    *,
    minimum_samples: int,
) -> tuple[float, float, int]:
    selected = np.asarray(values, dtype=np.float64)
    selected = selected[
        np.isfinite(selected) & (selected >= 0.5) & (selected <= 2.0)
    ]
    if selected.size < minimum_samples:
        raise ValueError("too few bounded stellar flux ratios")
    for _ in range(3):
        center = float(np.median(selected))
        sigma = float(1.4826 * np.median(np.abs(selected - center)))
        if not math.isfinite(sigma) or sigma <= np.finfo(np.float64).eps:
            break
        keep = np.abs(selected - center) <= 4.0 * sigma
        if int(np.count_nonzero(keep)) < minimum_samples or np.all(keep):
            break
        selected = selected[keep]
    scale = float(np.median(selected))
    sigma = float(1.4826 * np.median(np.abs(selected - scale)))
    return scale, sigma, int(selected.size)


NORMALIZATION_REFERENCE_RULE = "flattest-background-among-top-quality-v1"
NORMALIZATION_REFERENCE_QUALITY_FRACTION = 0.75
NORMALIZATION_REFERENCE_SKY_WEIGHT = 0.02
NORMALIZATION_REFERENCE_TILE_PREVIEW_PIXELS = 32


def background_flatness(analysis: FrameAnalysis) -> dict[str, float]:
    """Large-scale background amplitude of one calibrated preview.

    Tile medians (over the darker 70% of each tile, which excludes stars) on
    the calibrated preview; the p05-p95 span of those medians is the
    gradient the master would inherit from this frame, in frame units.
    """

    preview = np.asarray(analysis.preview, dtype=np.float64)
    tile = NORMALIZATION_REFERENCE_TILE_PREVIEW_PIXELS
    height, width = preview.shape
    rows, columns = max(1, height // tile), max(1, width // tile)
    levels = np.full((rows, columns), np.nan)
    for i in range(rows):
        for j in range(columns):
            block = preview[i * tile : min(height, (i + 1) * tile), j * tile : min(width, (j + 1) * tile)]
            block = block[np.isfinite(block)]
            if block.size < 16:
                continue
            cut = np.quantile(block, 0.70)
            levels[i, j] = float(np.median(block[block <= cut]))
    finite = levels[np.isfinite(levels)]
    if finite.size < 4:
        return {"gradientSpan": float("nan"), "sky": float(analysis.catalog.background)}
    low, high = np.quantile(finite, (0.05, 0.95))
    return {"gradientSpan": float(high - low), "sky": float(np.median(finite))}


def select_normalization_reference(
    indices: Sequence[int],
    analyses: Sequence[FrameAnalysis],
    quality_weights: Sequence[float],
) -> tuple[int, dict[str, Any]]:
    """Pick the normalization reference of one filter group.

    The integrated master inherits the large-scale background of the frame
    every other frame is matched to.  The frame with the smallest
    large-scale background span wins, with a small preference for a low sky
    (residual flat-field structure is proportional to the sky level, and a
    lower sky also means less noise around the reference's own gradient);
    the worst quarter by quality is excluded first so a poor frame never
    anchors the group.
    """

    ordered = sorted(indices, key=lambda index: (-float(quality_weights[index]), index))
    keep = max(1, math.ceil(NORMALIZATION_REFERENCE_QUALITY_FRACTION * len(ordered)))
    candidates = ordered[:keep]
    flatness = {index: background_flatness(analyses[index]) for index in candidates}

    def score(index: int) -> float:
        value = flatness[index]
        span = value["gradientSpan"]
        if not math.isfinite(span):
            span = float("inf")
        return span + NORMALIZATION_REFERENCE_SKY_WEIGHT * value["sky"]

    reference_index = min(candidates, key=lambda index: (score(index), index))
    return reference_index, {
        "rule": NORMALIZATION_REFERENCE_RULE,
        "qualityFractionConsidered": NORMALIZATION_REFERENCE_QUALITY_FRACTION,
        "skyWeight": NORMALIZATION_REFERENCE_SKY_WEIGHT,
        "candidateCount": len(candidates),
        "referenceSky": flatness[reference_index]["sky"],
        "referenceGradientSpan": flatness[reference_index]["gradientSpan"],
        "referenceScore": score(reference_index),
        "candidateScores": {
            str(index): {"score": score(index), **flatness[index]} for index in candidates
        },
        "groupSkyRange": [
            float(min(analyses[index].catalog.background for index in indices)),
            float(max(analyses[index].catalog.background for index in indices)),
        ],
        "referenceQualityWeight": float(quality_weights[reference_index]),
    }


def estimate_stellar_scale_hints(
    analyses: Sequence[FrameAnalysis],
    transforms: Sequence[FrameTransform],
    quality_weights: Sequence[float],
    *,
    match_radius_preview_pixels: float = 2.5,
    minimum_scale_stars: int = 16,
) -> tuple[StellarScaleEstimate, ...]:
    """Estimate same-filter reference/source throughput from matched stars."""

    if not (len(analyses) == len(transforms) == len(quality_weights)):
        raise ValueError("analyses, transforms, and quality_weights must align")
    groups: dict[str | None, list[int]] = defaultdict(list)
    for index, analysis in enumerate(analyses):
        groups[analysis.filter_name].append(index)
    estimates: list[StellarScaleEstimate | None] = [None] * len(analyses)
    for filter_name, indices in groups.items():
        reference_index, reference_selection = select_normalization_reference(
            indices, analyses, quality_weights
        )
        reference = analyses[reference_index]
        reference_transform = transforms[reference_index].preview_matrix
        if reference_transform is None:
            raise ValueError("reference registration transform is unavailable")
        estimates[reference_index] = StellarScaleEstimate(
            reference_index,
            reference_index,
            filter_name,
            1.0,
            "REFERENCE_IDENTITY",
            {
                "method": "matched-preview-aperture-median-mad-v1",
                "matchedSources": reference.catalog.count,
                "acceptedScaleStars": reference.catalog.count,
                "scaleSigma": 0.0,
                "sourceExposureSeconds": reference.exposure_seconds,
                "referenceExposureSeconds": reference.exposure_seconds,
                "exposureCorrection": 1.0,
                "scaleDomain": "post-linear-exposure-normalization",
                "referenceSelection": reference_selection,
            },
        )
        reference_global = _transform_points(
            reference.catalog.points, np.asarray(reference_transform, dtype=np.float64)
        )
        for source_index in indices:
            if source_index == reference_index:
                continue
            source = analyses[source_index]
            source_transform = transforms[source_index].preview_matrix
            evidence: dict[str, Any] = {
                "method": "matched-preview-aperture-median-mad-v1",
                "minimumScaleStars": minimum_scale_stars,
                "matchRadiusPreviewPixels": match_radius_preview_pixels,
            }
            try:
                if source_transform is None:
                    raise ValueError("source registration transform is unavailable")
                source_global = _transform_points(
                    source.catalog.points,
                    np.asarray(source_transform, dtype=np.float64),
                )
                source_indices, reference_indices = _one_to_one_indices(
                    source_global,
                    reference_global,
                    match_radius_preview_pixels,
                )
                evidence["matchedSources"] = int(source_indices.size)
                source_flux, retained_source, aperture = _aperture_fluxes(
                    source, source_indices
                )
                reference_flux, retained_reference, _ = _aperture_fluxes(
                    reference, reference_indices
                )
                source_by_index = {
                    int(index): float(value)
                    for index, value in zip(retained_source, source_flux, strict=True)
                }
                reference_by_index = {
                    int(index): float(value)
                    for index, value in zip(
                        retained_reference, reference_flux, strict=True
                    )
                }
                ratios = [
                    reference_by_index[int(reference_index_value)]
                    / source_by_index[int(source_index_value)]
                    for source_index_value, reference_index_value in zip(
                        source_indices, reference_indices, strict=True
                    )
                    if int(source_index_value) in source_by_index
                    and int(reference_index_value) in reference_by_index
                ]
                source_exposure = source.exposure_seconds
                reference_exposure = reference.exposure_seconds
                if (
                    source_exposure is None
                    or reference_exposure is None
                    or not math.isfinite(source_exposure)
                    or not math.isfinite(reference_exposure)
                    or source_exposure <= 0
                    or reference_exposure <= 0
                ):
                    raise ValueError(
                        "positive source/reference exposure is required for stellar scale"
                    )
                exposure_correction = source_exposure / reference_exposure
                exposure_normalized_ratios = np.asarray(
                    ratios, dtype=np.float64
                ) * exposure_correction
                scale, sigma, accepted = _robust_ratio(
                    exposure_normalized_ratios,
                    minimum_samples=minimum_scale_stars,
                )
                evidence.update(
                    {
                        **aperture,
                        "apertureValidPairs": len(ratios),
                        "acceptedScaleStars": accepted,
                        "scaleSigma": sigma,
                        "sourceExposureSeconds": source_exposure,
                        "referenceExposureSeconds": reference_exposure,
                        "exposureCorrection": exposure_correction,
                        "scaleDomain": "post-linear-exposure-normalization",
                    }
                )
                estimates[source_index] = StellarScaleEstimate(
                    source_index,
                    reference_index,
                    filter_name,
                    scale,
                    "STELLAR_SCALE_ACCEPTED",
                    evidence,
                )
            except (ValueError, RuntimeError) as error:
                evidence["reason"] = f"{type(error).__name__}: {error}"
                estimates[source_index] = StellarScaleEstimate(
                    source_index,
                    reference_index,
                    filter_name,
                    None,
                    "STELLAR_SCALE_UNAVAILABLE",
                    evidence,
                )
    assert all(item is not None for item in estimates)
    return tuple(item for item in estimates if item is not None)


def normalize_quality_weights(
    analyses: Sequence[FrameAnalysis],
    *,
    mode: Normalization = "group-median",
) -> tuple[float, ...]:
    """Normalize deterministic native scores within each optical filter."""

    if mode not in {"group-median", "first"}:
        raise ValueError("mode must be 'group-median' or 'first'")
    groups: dict[str | None, list[int]] = defaultdict(list)
    for index, analysis in enumerate(analyses):
        groups[analysis.filter_name].append(index)
    output = np.zeros(len(analyses), dtype=np.float64)
    for indices in groups.values():
        scores = np.asarray(
            [analyses[index].native_quality_score for index in indices],
            dtype=np.float64,
        )
        finite_positive = scores[np.isfinite(scores) & (scores > 0)]
        if finite_positive.size == 0:
            continue
        denominator = (
            float(np.median(finite_positive))
            if mode == "group-median"
            else float(scores[0])
        )
        if not math.isfinite(denominator) or denominator <= 0:
            continue
        output[indices] = scores / denominator
    return tuple(float(value) for value in output)


def diagnostic_scores(analysis: FrameAnalysis) -> dict[str, float]:
    """Alternative simple score families for one-off oracle comparison."""

    flux = np.asarray(analysis.catalog.flux, dtype=np.float64)
    peak = np.asarray(analysis.catalog.peak, dtype=np.float64)
    signal = analysis.total_star_signal
    integrated = analysis.total_integrated_flux
    noise = max(float(analysis.catalog.noise), np.finfo(np.float64).tiny)
    background = max(float(analysis.catalog.background), np.finfo(np.float64).tiny)
    psf = max(float(analysis.psf_scale), np.finfo(np.float64).tiny)
    mean_flux = float(np.sum(flux / np.maximum(np.asarray(analysis.catalog.fwhm) ** 2, 0.25)))
    return {
        "psfCoherencePerDetection": signal * psf * psf
        / (integrated * max(analysis.catalog.detected_count, 1)),
        "peakSignalOverNoisePsf2": signal / (noise * psf * psf),
        "peakSignalOverNoisePsf": signal / (noise * psf),
        "peakSignalOverNoise2Psf2": signal / (noise * noise * psf * psf),
        "quadraturePeakOverNoisePsf2": float(np.sqrt(np.sum(peak * peak)))
        / (noise * psf * psf),
        "integratedFluxOverNoisePsf2": integrated / (noise * psf * psf),
        "quadratureFluxOverNoisePsf2": float(np.sqrt(np.sum(flux * flux)))
        / (noise * psf * psf),
        "pclPsfSignalProxyPeak": integrated * signal / (noise * background),
        "pclPsfSignalProxyMean": integrated * mean_flux / (noise * background),
        "pclPsfSignalProxyPeakSquared": signal * signal / (noise * background),
    }
