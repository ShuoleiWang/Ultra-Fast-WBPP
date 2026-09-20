"""Independent registration vertical slice for the Ultra-Fast WBPP native backend.

This module intentionally has no PixInsight/PCL dependency.  Detection runs on
bounded FITS/XISF previews.  The estimated similarity matrices are lifted to
full-resolution coordinates, where the same contract can feed a CPU or Metal
resampler.  The included SciPy warp is the correctness/reference backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import astroalign
from astropy.io import fits
import numpy as np
from numpy.typing import NDArray
import sep
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import (
    AffineTransform as SkimageAffineTransform,
    ProjectiveTransform,
    SimilarityTransform,
)
from lightframeqc.cfa import is_cfa_pattern, luminance as cfa_luminance, normalize_pattern as normalize_cfa_pattern
from lightframeqc.xisf import XISF

from lightframeqc.parallel import FrameRunner
from lightframeqc.readers import ImagePreview, read_frame_preview


FloatImage = NDArray[np.float32]
Float64Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CalibrationPlan:
    """Optional master calibration used before detection or full-image warp.

    ``flat_paths`` is keyed by normalized filter name and ``dark_paths`` by
    exact Light exposure seconds.  Bias and Dark application scales are
    resolved once by the caller from explicit numeric-domain evidence.  If
    both are supplied and the selected Dark includes bias, the calibrated
    signal is ``light - biasScale*bias - (darkScale*dark - biasScale*bias)``.
    A supplied flat is normalized by its finite median before division.
    """

    bias_path: str | None = None
    dark_path: str | None = None
    dark_paths: Mapping[float, str] = field(default_factory=dict)
    dark_bias_included_by_exposure: Mapping[float, bool] = field(default_factory=dict)
    dark_application_scale_by_exposure: Mapping[float, float] = field(
        default_factory=dict
    )
    flat_paths: Mapping[str, str] = field(default_factory=dict)
    dark_scale: float = 1.0
    dark_includes_bias: bool = True
    bias_application_scale: float = 1.0
    flat_floor_fraction: float = 0.05
    light_scale: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.dark_scale) or self.dark_scale < 0:
            raise ValueError("dark_scale must be finite and nonnegative")
        if (
            not math.isfinite(self.bias_application_scale)
            or self.bias_application_scale <= 0
        ):
            raise ValueError("bias_application_scale must be finite and positive")
        if not 0 < self.flat_floor_fraction < 1:
            raise ValueError("flat_floor_fraction must be in (0, 1)")
        if self.light_scale is not None and (
            not math.isfinite(self.light_scale) or self.light_scale <= 0
        ):
            raise ValueError("light_scale must be finite and positive")
        if self.dark_path is not None and self.dark_paths:
            raise ValueError("dark_path and dark_paths are mutually exclusive")
        seen: list[float] = []
        for raw_exposure, path in self.dark_paths.items():
            if isinstance(raw_exposure, bool):
                raise ValueError("dark exposure keys must be finite and positive")
            try:
                exposure = float(raw_exposure)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "dark exposure keys must be finite and positive"
                ) from error
            if not math.isfinite(exposure) or exposure <= 0:
                raise ValueError("dark exposure keys must be finite and positive")
            if not isinstance(path, str) or not path.strip():
                raise ValueError("dark master paths must be non-empty strings")
            if any(
                math.isclose(exposure, other, rel_tol=0.0, abs_tol=1e-6)
                for other in seen
            ):
                raise ValueError("dark exposure keys must be unique")
            seen.append(exposure)
        if set(float(value) for value in self.dark_bias_included_by_exposure) - set(
            float(value) for value in self.dark_paths
        ):
            raise ValueError("dark bias semantics contains an unknown exposure")
        if set(float(value) for value in self.dark_application_scale_by_exposure) - set(
            float(value) for value in self.dark_paths
        ):
            raise ValueError("dark application scales contain an unknown exposure")
        if any(
            not isinstance(value, bool)
            for value in self.dark_bias_included_by_exposure.values()
        ):
            raise ValueError("dark bias semantics values must be boolean")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in self.dark_application_scale_by_exposure.values()
        ):
            raise ValueError("dark application scales must be finite and positive")

    def flat_for(self, filter_name: str | None) -> str | None:
        if filter_name is None:
            return None
        folded = filter_name.strip().upper()
        for key, value in self.flat_paths.items():
            if str(key).strip().upper() == folded:
                return str(value)
        return None

    def dark_for(self, exposure_seconds: float | None) -> str | None:
        if self.dark_path is not None:
            return self.dark_path
        if exposure_seconds is None or not math.isfinite(exposure_seconds):
            if self.dark_paths:
                raise ValueError("Light exposure is required to select a MasterDark")
            return None
        matches = [
            str(path)
            for exposure, path in self.dark_paths.items()
            if math.isclose(
                float(exposure), exposure_seconds, rel_tol=0.0, abs_tol=1e-6
            )
        ]
        if len(matches) != 1:
            if self.dark_paths:
                raise ValueError(
                    f"no exact MasterDark matches Light exposure {exposure_seconds:.9g}s"
                )
            return None
        return matches[0]

    def dark_includes_bias_for(self, exposure_seconds: float | None) -> bool:
        if not self.dark_bias_included_by_exposure:
            return self.dark_includes_bias
        if exposure_seconds is None:
            raise ValueError("Light exposure is required to select MasterDark bias semantics")
        matches = [
            value
            for exposure, value in self.dark_bias_included_by_exposure.items()
            if math.isclose(float(exposure), exposure_seconds, rel_tol=0.0, abs_tol=1e-6)
        ]
        if len(matches) != 1:
            raise ValueError("no exact MasterDark bias semantics matches Light exposure")
        return matches[0]

    def dark_application_scale_for(self, exposure_seconds: float | None) -> float:
        if not self.dark_application_scale_by_exposure:
            return 1.0
        if exposure_seconds is None:
            raise ValueError("Light exposure is required to select MasterDark application scale")
        matches = [
            float(value)
            for exposure, value in self.dark_application_scale_by_exposure.items()
            if math.isclose(float(exposure), exposure_seconds, rel_tol=0.0, abs_tol=1e-6)
        ]
        if len(matches) != 1:
            raise ValueError("no exact MasterDark application scale matches Light exposure")
        return matches[0]

    @property
    def enabled(self) -> bool:
        return bool(
            self.bias_path or self.dark_path or self.dark_paths or self.flat_paths
        )


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    preview_long_edge: int = 1600
    detection_sigma: float = 4.5
    minimum_source_pixels: int = 5
    maximum_stars: int = 1200
    background_box: int = 64

    def __post_init__(self) -> None:
        if self.preview_long_edge < 256:
            raise ValueError("preview_long_edge must be at least 256")
        if not 1 <= self.detection_sigma <= 20:
            raise ValueError("detection_sigma must be in [1, 20]")
        if self.minimum_source_pixels < 1 or self.maximum_stars < 20:
            raise ValueError("invalid source-count limits")


@dataclass(frozen=True, slots=True)
class RegistrationConfig:
    max_control_points: int = 100
    match_radius_px: float = 5.0
    residual_threshold_px: float = 2.5
    max_rms_px: float = 2.0
    min_inliers: int = 12
    min_inlier_ratio: float = 0.12
    max_trials: int = 1000
    refine_full_centroids: bool = True
    full_centroid_radius_px: int = 7
    full_residual_threshold_px: float = 1.5
    full_transform_model: str = "affine"
    full_min_inlier_ratio: float = 0.25
    full_min_span_fraction: float = 0.25
    full_max_scale_deviation: float = 0.05
    full_max_anisotropy: float = 1.05
    full_max_corner_delta_px: float = 32.0
    # Gaussian window (sigma, full-resolution pixels) of the core-weighted
    # centroid used by the full-resolution refinement.  ``None`` selects the
    # default window; ``0`` selects the plain (wing-weighted) centre of mass.
    full_centroid_window_sigma_px: float | None = None

    def __post_init__(self) -> None:
        if self.full_transform_model not in {"affine", "projective"}:
            raise ValueError("full_transform_model must be affine or projective")
        if self.full_centroid_radius_px < 2:
            raise ValueError("full_centroid_radius_px must be at least 2")
        if self.full_centroid_window_sigma_px is not None and (
            not math.isfinite(self.full_centroid_window_sigma_px)
            or self.full_centroid_window_sigma_px < 0
        ):
            raise ValueError("full_centroid_window_sigma_px must be finite and nonnegative")
        if (
            not math.isfinite(self.full_residual_threshold_px)
            or self.full_residual_threshold_px <= 0
        ):
            raise ValueError("full_residual_threshold_px must be finite and positive")
        for name in ("full_min_inlier_ratio", "full_min_span_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if (
            not math.isfinite(self.full_max_scale_deviation)
            or not 0.0 < self.full_max_scale_deviation < 0.5
        ):
            raise ValueError("full_max_scale_deviation must be finite and in (0, 0.5)")
        if (
            not math.isfinite(self.full_max_anisotropy)
            or self.full_max_anisotropy < 1.0
        ):
            raise ValueError("full_max_anisotropy must be finite and >= 1")
        if (
            not math.isfinite(self.full_max_corner_delta_px)
            or self.full_max_corner_delta_px <= 0
        ):
            raise ValueError("full_max_corner_delta_px must be finite and positive")


@dataclass(frozen=True, slots=True)
class StarCatalog:
    points: Float64Array
    flux: Float64Array
    peak: Float64Array
    fwhm: Float64Array
    background: float
    noise: float
    detected_count: int

    @property
    def count(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    path: str
    filter_name: str | None
    preview: FloatImage
    source_width: int
    source_height: int
    scale_x: float
    scale_y: float
    catalog: StarCatalog
    read_seconds: float
    calibration_seconds: float
    detection_seconds: float
    exposure_seconds: float | None = None

    @property
    def quality_score(self) -> float:
        if self.catalog.count == 0:
            return 0.0
        median_fwhm = float(np.median(self.catalog.fwhm))
        # Cap count so one noisy frame cannot win solely through over-detection.
        return min(self.catalog.count, 500) / max(median_fwhm * median_fwhm, 0.25)

    @property
    def total_star_signal(self) -> float:
        # A sum of PSF peak signals is deliberately used instead of segmented
        # aperture flux.  On saturated astronomical frames, poor seeing spreads
        # clipped cores and can make aperture flux increase as quality falls;
        # peak concentration retains the desired direction.
        return float(np.sum(self.catalog.peak, dtype=np.float64))

    @property
    def total_integrated_flux(self) -> float:
        return float(np.sum(self.catalog.flux, dtype=np.float64))

    @property
    def psf_scale(self) -> float:
        if self.catalog.count == 0:
            return float("inf")
        return float(np.median(self.catalog.fwhm))

    @property
    def native_quality_score(self) -> float:
        """Deterministic PSF-coherence quality proxy from the existing SEP pass.

        ``peak/integrated`` measures signal concentration.  Multiplying by the
        median PSF area removes the first-order Gaussian seeing dependence, and
        the uncapped detection count penalizes clutter/noise detections.  The
        global RMS remains an explicit measured guard/evidence field.
        """

        denominator = self.total_integrated_flux * self.catalog.detected_count
        if not math.isfinite(denominator) or denominator <= 0:
            return 0.0
        return self.total_star_signal * self.psf_scale * self.psf_scale / denominator


@dataclass(frozen=True, slots=True)
class FrameTransform:
    path: str
    filter_name: str | None
    preview_matrix: Float64Array | None
    full_matrix: Float64Array | None
    source_count: int
    reference_count: int
    match_count: int
    inlier_count: int
    inlier_ratio: float
    rms_preview_px: float
    rms_full_px: float
    accepted: bool
    reason: str | None
    registration_seconds: float
    bridge_path: str | None = None
    transform_model: str = "similarity"
    full_refine_inliers: int = 0
    full_refine_seconds: float = 0.0
    full_refine_evidence: Mapping[str, object] = field(default_factory=dict)
    warp_seconds: float = 0.0
    warp_pearson: float | None = None
    warp_valid_fraction: float | None = None


@dataclass(frozen=True, slots=True)
class RegistrationRun:
    reference_index: int
    analyses: tuple[FrameAnalysis, ...]
    transforms: tuple[FrameTransform, ...]
    analysis_wall_seconds: float
    registration_wall_seconds: float
    total_seconds: float


def _normalized_filter(preview: ImagePreview) -> str | None:
    value = preview.metadata.filter_name
    if value is None:
        value = preview.metadata.header.get("FILTER")
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _same_preview_geometry(left: ImagePreview, right: ImagePreview) -> None:
    if (
        left.source_width != right.source_width
        or left.source_height != right.source_height
        or left.data.shape != right.data.shape
    ):
        raise ValueError(
            "calibration master geometry differs from light preview: "
            f"{right.source_width}x{right.source_height}/{right.data.shape} vs "
            f"{left.source_width}x{left.source_height}/{left.data.shape}"
        )


def _calibrate_arrays(
    light: FloatImage,
    *,
    bias: FloatImage | None,
    dark: FloatImage | None,
    flat: FloatImage | None,
    plan: CalibrationPlan,
) -> FloatImage:
    result = np.array(light, dtype=np.float32, order="C", copy=True)
    if plan.light_scale is not None:
        result *= np.float32(plan.light_scale)
    scaled_bias = (
        bias * np.float32(plan.bias_application_scale)
        if bias is not None
        else None
    )
    if scaled_bias is not None:
        result -= scaled_bias
    if dark is not None:
        scaled_dark = np.float32(plan.dark_scale) * dark
        if scaled_bias is not None and plan.dark_includes_bias:
            result -= scaled_dark - scaled_bias
        else:
            result -= scaled_dark
    if flat is not None:
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            raise ValueError("flat contains no finite pixels")
        location = float(np.median(finite))
        if not math.isfinite(location) or abs(location) < np.finfo(np.float32).tiny:
            raise ValueError("flat has invalid normalization median")
        normalized = flat / np.float32(location)
        floor = np.float32(plan.flat_floor_fraction)
        valid = np.isfinite(normalized) & (normalized > floor)
        result = np.divide(
            result,
            normalized,
            out=np.full_like(result, np.nan),
            where=valid,
        )
    return np.ascontiguousarray(result, dtype=np.float32)


def calibrate_image(
    light: FloatImage,
    plan: CalibrationPlan,
    *,
    filter_name: str | None,
    bias: FloatImage | None = None,
    dark: FloatImage | None = None,
    flat: FloatImage | None = None,
) -> FloatImage:
    """Calibrate an already decoded image with caller-supplied master arrays."""

    del filter_name  # The caller used it to select ``flat``; retained in API.
    return _calibrate_arrays(light, bias=bias, dark=dark, flat=flat, plan=plan)


def _read_calibrated_preview(
    path: str,
    config: DetectionConfig,
    plan: CalibrationPlan | None,
    master_cache: dict[str, ImagePreview],
    master_lock: threading.Lock | None,
) -> tuple[ImagePreview, FloatImage, float]:
    light = read_frame_preview(path, max_long_edge=config.preview_long_edge)
    image = np.asarray(light.data, dtype=np.float32)
    if plan is None or not plan.enabled:
        return light, np.ascontiguousarray(image), 0.0

    started = time.perf_counter()
    filter_name = _normalized_filter(light)

    def master(master_path: str | None) -> FloatImage | None:
        if master_path is None:
            return None
        key = str(Path(master_path).expanduser().resolve(strict=True))
        if master_lock is None:
            item = master_cache.get(key)
            if item is None:
                item = read_frame_preview(key, max_long_edge=config.preview_long_edge)
                master_cache[key] = item
        else:
            with master_lock:
                item = master_cache.get(key)
                if item is None:
                    item = read_frame_preview(key, max_long_edge=config.preview_long_edge)
                    master_cache[key] = item
        _same_preview_geometry(light, item)
        return np.asarray(item.data, dtype=np.float32)

    flat_path = plan.flat_for(filter_name)
    if plan.flat_paths and flat_path is None:
        raise ValueError(f"no flat configured for filter {filter_name!r}")
    bias = master(plan.bias_path)
    dark = master(plan.dark_for(light.metadata.exposure_seconds))
    flat = master(flat_path)
    preview_plan = replace(
        plan,
        dark_includes_bias=plan.dark_includes_bias_for(light.metadata.exposure_seconds),
        dark_scale=(
            plan.dark_scale
            * plan.dark_application_scale_for(light.metadata.exposure_seconds)
        ),
    )
    calibrated = _calibrate_arrays(
        image,
        bias=bias,
        dark=dark,
        flat=flat,
        plan=preview_plan,
    )
    return light, calibrated, time.perf_counter() - started


def detect_stars(image: FloatImage, config: DetectionConfig) -> StarCatalog:
    """Detect and rank stars on a mono Float32 image with SEP."""

    work = np.array(image, dtype=np.float32, order="C", copy=True)
    invalid = ~np.isfinite(work)
    finite = work[~invalid]
    if finite.size < max(1024, work.size // 20):
        raise ValueError("image has too few finite pixels")
    median = float(np.median(finite))
    work[invalid] = np.float32(median)
    box = max(16, min(config.background_box, work.shape[0] // 4, work.shape[1] // 4))
    background = sep.Background(work, mask=invalid, bw=box, bh=box, fw=3, fh=3)
    residual = np.ascontiguousarray(work - background.back(), dtype=np.float32)
    rms = np.ascontiguousarray(background.rms(), dtype=np.float32)
    floor = max(float(background.globalrms), np.finfo(np.float32).eps)
    rms[~np.isfinite(rms) | (rms <= 0)] = np.float32(floor)
    objects = sep.extract(
        residual,
        config.detection_sigma,
        err=rms,
        mask=invalid,
        minarea=config.minimum_source_pixels,
    )
    if objects.size == 0:
        return StarCatalog(
            points=np.empty((0, 2), dtype=np.float64),
            flux=np.empty(0, dtype=np.float64),
            peak=np.empty(0, dtype=np.float64),
            fwhm=np.empty(0, dtype=np.float64),
            background=float(background.globalback),
            noise=float(background.globalrms),
            detected_count=0,
        )

    x = np.asarray(objects["x"], dtype=np.float64)
    y = np.asarray(objects["y"], dtype=np.float64)
    flux = np.asarray(objects["flux"], dtype=np.float64)
    peak = np.asarray(objects["peak"], dtype=np.float64)
    a = np.asarray(objects["a"], dtype=np.float64)
    b = np.asarray(objects["b"], dtype=np.float64)
    flags = np.asarray(objects["flag"], dtype=np.int64)
    finite_rows = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(flux)
        & np.isfinite(peak)
        & np.isfinite(a)
        & np.isfinite(b)
        & (flux > 0)
        & (a > 0)
        & (b > 0)
        & (flags == 0)
    )
    x, y, flux, peak, a, b = (
        values[finite_rows] for values in (x, y, flux, peak, a, b)
    )
    detected_count = int(flux.size)
    fwhm = 2.354820045 * np.sqrt(a * b)
    order = np.argsort(-flux, kind="stable")[: config.maximum_stars]
    return StarCatalog(
        points=np.ascontiguousarray(np.column_stack((x[order], y[order])), dtype=np.float64),
        flux=np.ascontiguousarray(flux[order], dtype=np.float64),
        peak=np.ascontiguousarray(peak[order], dtype=np.float64),
        fwhm=np.ascontiguousarray(fwhm[order], dtype=np.float64),
        background=float(background.globalback),
        noise=float(background.globalrms),
        detected_count=detected_count,
    )


def analyze_frame(
    path: str,
    *,
    detection: DetectionConfig | None = None,
    calibration: CalibrationPlan | None = None,
    master_cache: dict[str, ImagePreview] | None = None,
    master_lock: threading.Lock | None = None,
) -> FrameAnalysis:
    config = detection or DetectionConfig()
    cache = master_cache if master_cache is not None else {}
    started = time.perf_counter()
    preview, pixels, calibration_seconds = _read_calibrated_preview(
        path, config, calibration, cache, master_lock
    )
    after_read = time.perf_counter()
    catalog = detect_stars(pixels, config)
    finished = time.perf_counter()
    return FrameAnalysis(
        path=str(Path(path).expanduser().resolve(strict=True)),
        filter_name=_normalized_filter(preview),
        preview=pixels,
        source_width=preview.source_width,
        source_height=preview.source_height,
        scale_x=preview.scale_x,
        scale_y=preview.scale_y,
        catalog=catalog,
        read_seconds=max(0.0, after_read - started - calibration_seconds),
        calibration_seconds=calibration_seconds,
        detection_seconds=finished - after_read,
        exposure_seconds=preview.metadata.exposure_seconds,
    )


class _StatKeyedMasterCache(dict):
    """Master preview cache whose keys carry the file's size and mtime, so a
    master rewritten at the same path is decoded again."""

    @staticmethod
    def _key(path: str) -> str:
        info = Path(path).stat()
        return f"{path}|{info.st_size}|{info.st_mtime_ns}"

    def get(self, key: str, default: ImagePreview | None = None) -> ImagePreview | None:  # type: ignore[override]
        return super().get(self._key(key), default)

    def __setitem__(self, key: str, value: ImagePreview) -> None:
        super().__setitem__(self._key(key), value)


# Master previews decoded by this process's frame analyses; each worker
# process of a pool fills its own copy once.
_MASTER_PREVIEWS = _StatKeyedMasterCache()
_MASTER_PREVIEWS_LOCK = threading.Lock()


def _analyze_frame_task(
    task: tuple[str, DetectionConfig | None, CalibrationPlan | None],
) -> FrameAnalysis:
    path, detection, calibration = task
    return analyze_frame(
        path,
        detection=detection,
        calibration=calibration,
        master_cache=_MASTER_PREVIEWS,
        master_lock=_MASTER_PREVIEWS_LOCK,
    )


def analyze_frames(
    paths: Iterable[str],
    *,
    detection: DetectionConfig | None = None,
    calibration: CalibrationPlan | None = None,
    workers: int = 4,
) -> tuple[FrameAnalysis, ...]:
    """Analyze multiple frames in stable order with a shared master cache.

    SEP's detection holds the GIL, so a batch of frames runs in spawned
    worker processes (``lightframeqc.parallel`` chooses processes from twelve
    frames, threads below that, and falls back to threads when a pool cannot
    start); every frame runs the same function, so the values never depend
    on the executor.
    """

    ordered = [str(Path(path).expanduser().resolve(strict=True)) for path in paths]
    if not ordered:
        return ()
    if workers < 1:
        raise ValueError("workers must be positive")
    tasks = [(path, detection, calibration) for path in ordered]
    if workers == 1:
        return tuple(_analyze_frame_task(task) for task in tasks)
    with FrameRunner(workers, len(ordered)) as runner:
        return tuple(runner.map(_analyze_frame_task, tasks))


def choose_reference(analyses: Sequence[FrameAnalysis]) -> int:
    if not analyses:
        raise ValueError("cannot select a reference from no frames")
    return max(
        range(len(analyses)),
        key=lambda index: (analyses[index].quality_score, -index),
    )


def _one_to_one_matches(
    source: Float64Array,
    reference: Float64Array,
    model: SimilarityTransform,
    radius: float,
) -> tuple[Float64Array, Float64Array, Float64Array]:
    transformed = np.asarray(model(source), dtype=np.float64)
    tree = cKDTree(reference)
    neighbours = tree.query_ball_point(transformed, radius)
    edges: list[tuple[float, int, int]] = []
    for source_index, candidates in enumerate(neighbours):
        for reference_index in candidates:
            distance = float(
                np.linalg.norm(transformed[source_index] - reference[reference_index])
            )
            edges.append((distance, source_index, int(reference_index)))
    edges.sort()
    used_source: set[int] = set()
    used_reference: set[int] = set()
    selected: list[tuple[int, int, float]] = []
    for distance, source_index, reference_index in edges:
        if source_index in used_source or reference_index in used_reference:
            continue
        used_source.add(source_index)
        used_reference.add(reference_index)
        selected.append((source_index, reference_index, distance))
    if not selected:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty.copy(), np.empty(0, dtype=np.float64)
    source_indices = np.asarray([item[0] for item in selected], dtype=np.int64)
    reference_indices = np.asarray([item[1] for item in selected], dtype=np.int64)
    distances = np.asarray([item[2] for item in selected], dtype=np.float64)
    return source[source_indices], reference[reference_indices], distances


def _refine_transform(
    source: StarCatalog,
    reference: StarCatalog,
    config: RegistrationConfig,
) -> tuple[SimilarityTransform, Float64Array, int]:
    initial = _translation_rotation_bootstrap(source, reference, config)
    if initial is None:
        initial, _ = astroalign.find_transform(
            source.points,
            reference.points,
            max_control_points=config.max_control_points,
        )
    return _refine_from_initial(source, reference, initial, config)


def _translation_rotation_bootstrap(
    source: StarCatalog,
    reference: StarCatalog,
    config: RegistrationConfig,
) -> SimilarityTransform | None:
    """Fast catalog-offset bootstrap for same-scale 0/180-degree frames."""

    count = min(240, source.count, reference.count)
    if count < 3:
        return None
    source_points = source.points[:count]
    reference_points = reference.points[:count]
    best_model: SimilarityTransform | None = None
    best_matches = 0
    bin_width = max(3.0, config.match_radius_px)
    for angle in (0.0, math.pi):
        rotation = SimilarityTransform(rotation=angle)
        rotated = np.asarray(rotation(source_points), dtype=np.float64)
        differences = reference_points[:, None, :] - rotated[None, :, :]
        flat = differences.reshape((-1, 2))
        bins = np.rint(flat / bin_width).astype(np.int64)
        unique, counts = np.unique(bins, axis=0, return_counts=True)
        if counts.size == 0:
            continue
        winning_bin = unique[int(np.argmax(counts))]
        selected = np.all(bins == winning_bin, axis=1)
        translation = np.median(flat[selected], axis=0)
        model = SimilarityTransform(rotation=angle, translation=translation)
        matched_source, _, _ = _one_to_one_matches(
            source.points,
            reference.points,
            model,
            max(config.match_radius_px * 1.5, bin_width * 1.5),
        )
        if matched_source.shape[0] > best_matches:
            best_matches = int(matched_source.shape[0])
            best_model = model
    return best_model if best_matches >= max(12, config.min_inliers) else None


def _refine_from_initial(
    source: StarCatalog,
    reference: StarCatalog,
    initial: SimilarityTransform,
    config: RegistrationConfig,
) -> tuple[SimilarityTransform, Float64Array, int]:
    """Refine a close source->reference bootstrap against all detected stars."""

    model: SimilarityTransform = initial
    for iteration in range(2):
        matched_source, matched_reference, _ = _one_to_one_matches(
            source.points, reference.points, model, config.match_radius_px
        )
        if matched_source.shape[0] < 3:
            break
        refined, mask = ransac(
            (matched_source, matched_reference),
            SimilarityTransform,
            min_samples=2,
            residual_threshold=config.residual_threshold_px,
            max_trials=config.max_trials,
            rng=np.random.default_rng(iteration),
        )
        if refined is None or mask is None or np.count_nonzero(mask) < 2:
            break
        model = refined

    matched_source, matched_reference, _ = _one_to_one_matches(
        source.points, reference.points, model, config.match_radius_px
    )
    if matched_source.shape[0] < 2:
        raise ValueError("too few catalog matches")
    residuals = np.linalg.norm(model(matched_source) - matched_reference, axis=1)
    inliers = residuals <= config.residual_threshold_px
    if np.count_nonzero(inliers) >= 2:
        factory = getattr(SimilarityTransform, "from_estimate", None)
        if factory is not None:
            least_squares = factory(matched_source[inliers], matched_reference[inliers])
        else:
            least_squares = SimilarityTransform()
            if not least_squares.estimate(matched_source[inliers], matched_reference[inliers]):
                least_squares = None
        if least_squares is not None:
            model = least_squares
            matched_source, matched_reference, _ = _one_to_one_matches(
                source.points, reference.points, model, config.match_radius_px
            )
            residuals = np.linalg.norm(model(matched_source) - matched_reference, axis=1)
            inliers = residuals <= config.residual_threshold_px
    return model, residuals, int(np.count_nonzero(inliers))


def _result_from_model(
    source: FrameAnalysis,
    reference: FrameAnalysis,
    model: SimilarityTransform,
    residuals: Float64Array,
    inlier_count: int,
    config: RegistrationConfig,
    *,
    started: float,
    bridge_path: str | None = None,
) -> FrameTransform:
    mask = residuals <= config.residual_threshold_px
    rms = float(np.sqrt(np.mean(np.square(residuals[mask])))) if np.any(mask) else float("inf")
    denominator = min(source.catalog.count, reference.catalog.count)
    ratio = inlier_count / denominator if denominator else 0.0
    accepted = (
        inlier_count >= config.min_inliers
        and ratio >= config.min_inlier_ratio
        and rms <= config.max_rms_px
    )
    preview_matrix = np.asarray(model.params, dtype=np.float64)
    full_matrix = full_resolution_matrix(preview_matrix, source, reference)
    full_scale = 0.5 * (reference.scale_x + reference.scale_y)
    return FrameTransform(
        path=source.path,
        filter_name=source.filter_name,
        preview_matrix=preview_matrix,
        full_matrix=full_matrix,
        source_count=source.catalog.count,
        reference_count=reference.catalog.count,
        match_count=int(residuals.size),
        inlier_count=inlier_count,
        inlier_ratio=float(ratio),
        rms_preview_px=rms,
        rms_full_px=rms * full_scale,
        accepted=accepted,
        reason=None if accepted else "registration thresholds not met",
        registration_seconds=time.perf_counter() - started,
        bridge_path=bridge_path,
    )


def _apply_homography(matrix: Float64Array, points: Float64Array) -> Float64Array:
    homogeneous = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
    transformed = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


CENTROID_WINDOW_ITERATIONS = 12
CENTROID_WINDOW_CONVERGENCE_PX = 1.0e-4
# Window of the core-weighted centroid: the sigma that SExtractor XWIN/YWIN
# and sep.winpos use for typical seeing (FWHM 3-5 px).  A fixed window keeps
# one position definition for every frame, filter and night of a run; the
# preview moment FWHM is not used because block averaging inflates it.
DEFAULT_CENTROID_WINDOW_SIGMA_PX = 2.0


def _local_centroids(
    image: FloatImage,
    approximate: Float64Array,
    radius: int,
    *,
    window_sigma: float | None = None,
) -> tuple[Float64Array, NDArray[np.int64]]:
    """Refine approximate star positions with small raw-image patches.

    The plain centre of mass of a patch is wing-weighted: on an asymmetric
    PSF (coma, tilt, tracking) it sits a night- and filter-dependent fraction
    of a pixel away from the star core, so filters imaged under different
    PSF shapes register their cores against each other with that offset.
    With ``window_sigma`` the centre of mass only seeds a Gaussian-windowed
    centroid (SExtractor XWIN/YWIN) whose window is re-centred on every
    iteration, which converges on the core.
    """

    height, width = image.shape
    size = 2 * radius + 1
    positions = np.asarray(approximate, dtype=np.float64).reshape((-1, 2))
    empty = (np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.int64))
    if positions.shape[0] == 0:
        return empty
    if not np.all(np.isfinite(positions)):
        raise ValueError("cannot convert float NaN to integer")
    # Every patch is gathered at once and reduced per star with the same
    # Float64 arithmetic as a one-star-at-a-time loop: patches are cast
    # elementwise, each border median and each patch sum reduce one star's
    # contiguous samples, so the centroids are value-identical.
    center = np.rint(positions).astype(np.int64)  # round half to even, as round()
    x0 = center[:, 0] - radius
    y0 = center[:, 1] - radius
    inside = (
        (x0 >= 0) & (y0 >= 0) & (x0 + size <= width) & (y0 + size <= height)
    )
    candidates = np.flatnonzero(inside)
    if candidates.size == 0:
        return empty
    offsets = np.arange(size, dtype=np.int64)
    rows = y0[candidates, None, None] + offsets[None, :, None]
    columns = x0[candidates, None, None] + offsets[None, None, :]
    patches = np.asarray(image[rows, columns], dtype=np.float64)
    finite = np.all(np.isfinite(patches.reshape(candidates.size, -1)), axis=1)
    candidates = candidates[finite]
    if candidates.size == 0:
        return empty
    patches = patches[finite]
    border = np.concatenate(
        (patches[:, 0, :], patches[:, -1, :], patches[:, 1:-1, 0], patches[:, 1:-1, -1]),
        axis=1,
    )
    background = np.median(border, axis=1)
    signal = np.maximum(patches - background[:, None, None], 0.0)
    flat_signal = signal.reshape(candidates.size, -1)
    total = np.sum(flat_signal, axis=1)
    positive = np.isfinite(total) & (total > 0)
    candidates = candidates[positive]
    if candidates.size == 0:
        return empty
    signal = signal[positive]
    total = total[positive]
    yy, xx = np.indices((size, size), dtype=np.float64)
    local_x = np.sum((signal * xx).reshape(candidates.size, -1), axis=1) / total
    local_y = np.sum((signal * yy).reshape(candidates.size, -1), axis=1) / total
    if window_sigma is not None and window_sigma > 0.0:
        local_x, local_y = _windowed_centroids(signal, xx, yy, local_x, local_y, window_sigma)
    centroid_x = x0[candidates] + local_x
    centroid_y = y0[candidates] + local_y
    limit = radius * 0.5
    close = np.fromiter(
        (
            math.hypot(float(cx) - float(x), float(cy) - float(y)) <= limit
            for cx, cy, (x, y) in zip(centroid_x, centroid_y, positions[candidates], strict=True)
        ),
        dtype=np.bool_,
        count=int(candidates.size),
    )
    refined = np.column_stack((centroid_x[close], centroid_y[close]))
    return (
        np.asarray(refined, dtype=np.float64).reshape((-1, 2)),
        np.asarray(candidates[close], dtype=np.int64),
    )


def _windowed_centroids(
    signal: Float64Array,
    xx: Float64Array,
    yy: Float64Array,
    start_x: Float64Array,
    start_y: Float64Array,
    window_sigma: float,
) -> tuple[Float64Array, Float64Array]:
    """Iterate Gaussian-windowed centroids of background-subtracted patches."""

    count = signal.shape[0]
    flat_signal = signal.reshape(count, -1)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)
    current_x = np.array(start_x, dtype=np.float64)
    current_y = np.array(start_y, dtype=np.float64)
    denominator = 2.0 * float(window_sigma) ** 2
    limit_x = float(xx.shape[1] - 1)
    limit_y = float(yy.shape[0] - 1)
    for _ in range(CENTROID_WINDOW_ITERATIONS):
        weight = np.exp(
            -(
                (flat_x[None, :] - current_x[:, None]) ** 2
                + (flat_y[None, :] - current_y[:, None]) ** 2
            )
            / denominator
        )
        weighted = flat_signal * weight
        weighted_total = np.sum(weighted, axis=1)
        movable = np.isfinite(weighted_total) & (weighted_total > 0.0)
        safe_total = np.where(movable, weighted_total, 1.0)
        next_x = np.where(movable, np.sum(weighted * flat_x[None, :], axis=1) / safe_total, current_x)
        next_y = np.where(movable, np.sum(weighted * flat_y[None, :], axis=1) / safe_total, current_y)
        # The window never leaves the gathered patch.
        next_x = np.clip(next_x, 0.0, limit_x)
        next_y = np.clip(next_y, 0.0, limit_y)
        shift = np.maximum(np.abs(next_x - current_x), np.abs(next_y - current_y))
        current_x, current_y = next_x, next_y
        if not np.any(shift > CENTROID_WINDOW_CONVERGENCE_PX):
            break
    return current_x, current_y


def _core_window_sigma(config: RegistrationConfig) -> float:
    """Window sigma of the core-weighted centroid; one value for a whole run."""

    if config.full_centroid_window_sigma_px is not None:
        return float(config.full_centroid_window_sigma_px)
    return DEFAULT_CENTROID_WINDOW_SIGMA_PX


def _refine_full_resolution(
    result: FrameTransform,
    source: FrameAnalysis,
    reference: FrameAnalysis,
    source_image: FloatImage,
    reference_image: FloatImage,
    config: RegistrationConfig,
) -> FrameTransform:
    if result.preview_matrix is None or result.full_matrix is None:
        return result
    started = time.perf_counter()

    def rejected(reason: str, **evidence: object) -> FrameTransform:
        return replace(
            result,
            full_refine_seconds=time.perf_counter() - started,
            full_refine_evidence={
                "status": "REJECTED",
                "reason": reason,
                **evidence,
            },
        )

    coarse_model = SimilarityTransform(matrix=result.preview_matrix)
    source_preview, reference_preview, _ = _one_to_one_matches(
        source.catalog.points,
        reference.catalog.points,
        coarse_model,
        config.match_radius_px,
    )
    if source_preview.shape[0] < 12:
        return rejected(
            "PREVIEW_MATCHES_INSUFFICIENT",
            previewMatchCount=int(source_preview.shape[0]),
        )
    source_scale = _preview_to_full_matrix(source.scale_x, source.scale_y)
    reference_scale = _preview_to_full_matrix(reference.scale_x, reference.scale_y)
    source_approximate = _apply_homography(source_scale, source_preview)
    reference_approximate = _apply_homography(reference_scale, reference_preview)
    window_sigma = _core_window_sigma(config)
    source_centroids, source_indices = _local_centroids(
        source_image,
        source_approximate,
        config.full_centroid_radius_px,
        window_sigma=window_sigma,
    )
    reference_centroids, reference_indices = _local_centroids(
        reference_image,
        reference_approximate,
        config.full_centroid_radius_px,
        window_sigma=window_sigma,
    )
    common = np.intersect1d(source_indices, reference_indices, assume_unique=True)
    if common.size < 12:
        return rejected(
            "FULL_CENTROIDS_INSUFFICIENT",
            commonCentroidCount=int(common.size),
        )
    source_positions = {int(value): index for index, value in enumerate(source_indices)}
    reference_positions = {int(value): index for index, value in enumerate(reference_indices)}
    full_source = np.asarray(
        [source_centroids[source_positions[int(value)]] for value in common],
        dtype=np.float64,
    )
    full_reference = np.asarray(
        [reference_centroids[reference_positions[int(value)]] for value in common],
        dtype=np.float64,
    )
    model_class: type[SkimageAffineTransform] | type[ProjectiveTransform]
    minimum_samples: int
    if config.full_transform_model == "projective":
        model_class = ProjectiveTransform
        minimum_samples = 4
    else:
        model_class = SkimageAffineTransform
        minimum_samples = 3
    try:
        model, inliers = ransac(
            (full_source, full_reference),
            model_class,
            min_samples=minimum_samples,
            residual_threshold=config.full_residual_threshold_px,
            max_trials=config.max_trials,
            rng=np.random.default_rng(0),
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return rejected(
            "RANSAC_FAILED",
            commonCentroidCount=int(common.size),
        )
    if model is None or inliers is None or np.count_nonzero(inliers) < 12:
        return rejected(
            "RANSAC_INLIERS_INSUFFICIENT",
            commonCentroidCount=int(common.size),
            ransacInlierCount=(
                int(np.count_nonzero(inliers)) if inliers is not None else 0
            ),
        )
    factory = getattr(model_class, "from_estimate", None)
    if factory is not None:
        least_squares = factory(full_source[inliers], full_reference[inliers])
        if least_squares:
            model = least_squares
    residuals = np.linalg.norm(model(full_source) - full_reference, axis=1)
    final_inliers = residuals <= config.full_residual_threshold_px
    inlier_count = int(np.count_nonzero(final_inliers))
    if inlier_count < 12:
        return rejected(
            "FINAL_INLIERS_INSUFFICIENT",
            commonCentroidCount=int(common.size),
            finalInlierCount=inlier_count,
        )
    rms = float(np.sqrt(np.mean(np.square(residuals[final_inliers]))))
    full_matrix = np.asarray(model.params, dtype=np.float64)
    inlier_ratio = inlier_count / int(common.size)
    selected_source = full_source[final_inliers]
    selected_reference = full_reference[final_inliers]
    source_span = (
        float(np.ptp(selected_source[:, 0]) / max(source.source_width - 1, 1)),
        float(np.ptp(selected_source[:, 1]) / max(source.source_height - 1, 1)),
    )
    reference_span = (
        float(
            np.ptp(selected_reference[:, 0]) / max(reference.source_width - 1, 1)
        ),
        float(
            np.ptp(selected_reference[:, 1]) / max(reference.source_height - 1, 1)
        ),
    )
    singular_values = np.linalg.svd(full_matrix[:2, :2], compute_uv=False)
    minimum_singular = float(np.min(singular_values))
    maximum_singular = float(np.max(singular_values))
    anisotropy = maximum_singular / max(minimum_singular, np.finfo(np.float64).tiny)
    scale_minimum = 1.0 - config.full_max_scale_deviation
    scale_maximum = 1.0 + config.full_max_scale_deviation
    source_corners = np.asarray(
        [
            [0.0, 0.0],
            [source.source_width - 1.0, 0.0],
            [0.0, source.source_height - 1.0],
            [source.source_width - 1.0, source.source_height - 1.0],
            [(source.source_width - 1.0) / 2.0, (source.source_height - 1.0) / 2.0],
        ],
        dtype=np.float64,
    )
    refined_corners = _apply_homography(full_matrix, source_corners)
    coarse_corners = _apply_homography(
        np.asarray(result.full_matrix, dtype=np.float64), source_corners
    )
    corner_delta = float(
        np.max(np.linalg.norm(refined_corners - coarse_corners, axis=1))
    )
    gate_evidence = {
        "status": "ACCEPTED",
        "centroid": "gaussian-window-core-v1" if window_sigma > 0.0 else "plain-centre-of-mass-v1",
        "centroidWindowSigmaPixels": window_sigma,
        "commonCentroidCount": int(common.size),
        "finalInlierCount": inlier_count,
        "finalInlierRatio": inlier_ratio,
        "rmsPixels": rms,
        "sourceSpanFraction": list(source_span),
        "referenceSpanFraction": list(reference_span),
        "linearSingularValues": [float(value) for value in singular_values],
        "linearAnisotropy": anisotropy,
        "maximumCornerDeltaFromPreviewLiftPixels": corner_delta,
    }
    gate_reasons: list[str] = []
    if inlier_ratio < config.full_min_inlier_ratio:
        gate_reasons.append("INLIER_RATIO_LOW")
    if min(*source_span, *reference_span) < config.full_min_span_fraction:
        gate_reasons.append("SPATIAL_SPAN_LOW")
    if not (
        scale_minimum <= minimum_singular <= scale_maximum
        and scale_minimum <= maximum_singular <= scale_maximum
    ):
        gate_reasons.append("LINEAR_SCALE_IMPLAUSIBLE")
    if anisotropy > config.full_max_anisotropy:
        gate_reasons.append("LINEAR_ANISOTROPY_HIGH")
    if corner_delta > config.full_max_corner_delta_px:
        gate_reasons.append("CORNER_DELTA_HIGH")
    if gate_reasons:
        return rejected(
            "FULL_MODEL_GATE_FAILED",
            **{**gate_evidence, "status": "REJECTED", "gateReasons": gate_reasons},
        )
    preview_matrix = np.linalg.inv(reference_scale) @ full_matrix @ source_scale
    return replace(
        result,
        preview_matrix=preview_matrix,
        full_matrix=full_matrix,
        rms_full_px=rms,
        transform_model=f"{config.full_transform_model}-full-centroid",
        full_refine_inliers=inlier_count,
        full_refine_seconds=time.perf_counter() - started,
        full_refine_evidence=gate_evidence,
    )


def _preview_to_full_matrix(scale_x: float, scale_y: float) -> Float64Array:
    # Pixel-center mapping for a block-mean preview.
    return np.asarray(
        [
            [scale_x, 0.0, (scale_x - 1.0) * 0.5],
            [0.0, scale_y, (scale_y - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def full_resolution_matrix(
    preview_matrix: Float64Array,
    source: FrameAnalysis,
    reference: FrameAnalysis,
) -> Float64Array:
    source_scale = _preview_to_full_matrix(source.scale_x, source.scale_y)
    reference_scale = _preview_to_full_matrix(reference.scale_x, reference.scale_y)
    return reference_scale @ preview_matrix @ np.linalg.inv(source_scale)


def _estimate_one(
    source: FrameAnalysis,
    reference: FrameAnalysis,
    config: RegistrationConfig,
) -> FrameTransform:
    started = time.perf_counter()
    if source.path == reference.path:
        identity = np.eye(3, dtype=np.float64)
        return FrameTransform(
            path=source.path,
            filter_name=source.filter_name,
            preview_matrix=identity,
            full_matrix=identity,
            source_count=source.catalog.count,
            reference_count=reference.catalog.count,
            match_count=source.catalog.count,
            inlier_count=source.catalog.count,
            inlier_ratio=1.0,
            rms_preview_px=0.0,
            rms_full_px=0.0,
            accepted=True,
            reason=None,
            registration_seconds=time.perf_counter() - started,
        )
    try:
        model, residuals, inlier_count = _refine_transform(
            source.catalog, reference.catalog, config
        )
    except Exception as error:
        return FrameTransform(
            path=source.path,
            filter_name=source.filter_name,
            preview_matrix=None,
            full_matrix=None,
            source_count=source.catalog.count,
            reference_count=reference.catalog.count,
            match_count=0,
            inlier_count=0,
            inlier_ratio=0.0,
            rms_preview_px=float("inf"),
            rms_full_px=float("inf"),
            accepted=False,
            reason=f"{type(error).__name__}: {error}",
            registration_seconds=time.perf_counter() - started,
        )
    return _result_from_model(
        source,
        reference,
        model,
        residuals,
        inlier_count,
        config,
        started=started,
    )


def _estimate_via_bridge(
    source: FrameAnalysis,
    bridge: FrameAnalysis,
    bridge_to_reference: FrameTransform,
    reference: FrameAnalysis,
    config: RegistrationConfig,
) -> FrameTransform | None:
    """Recover a hard direct match through a same-filter registered frame."""

    if bridge_to_reference.preview_matrix is None or not bridge_to_reference.accepted:
        return None
    started = time.perf_counter()
    source_to_bridge = _estimate_one(source, bridge, config)
    if source_to_bridge.preview_matrix is None or not source_to_bridge.accepted:
        return None
    composed = bridge_to_reference.preview_matrix @ source_to_bridge.preview_matrix
    try:
        initial = SimilarityTransform(matrix=composed)
        model, residuals, inlier_count = _refine_from_initial(
            source.catalog, reference.catalog, initial, config
        )
    except Exception:
        return None
    return _result_from_model(
        source,
        reference,
        model,
        residuals,
        inlier_count,
        config,
        started=started,
        bridge_path=bridge.path,
    )


def warp_image(
    image: FloatImage,
    source_to_reference: Float64Array,
    output_shape: tuple[int, int],
    *,
    order: int = 3,
    cval: float = float("nan"),
) -> FloatImage:
    """Warp a source image into reference coordinates using SciPy.

    ``source_to_reference`` uses homogeneous ``(x, y)`` coordinates.  SciPy's
    output-to-input convention and ``(y, x)`` axis order are handled here.
    """

    inverse = np.linalg.inv(np.asarray(source_to_reference, dtype=np.float64))
    if not np.allclose(inverse[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1.0e-12):
        height, width = output_shape
        output = np.empty(output_shape, dtype=np.float32)
        source = np.asarray(image, dtype=np.float32)
        if order > 1:
            source = ndimage.spline_filter(source, order=order, output=np.float32)
        x = np.arange(width, dtype=np.float64)[None, :]
        for y0 in range(0, height, 128):
            y1 = min(height, y0 + 128)
            y = np.arange(y0, y1, dtype=np.float64)[:, None]
            denominator = inverse[2, 0] * x + inverse[2, 1] * y + inverse[2, 2]
            source_x = (inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]) / denominator
            source_y = (inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]) / denominator
            output[y0:y1] = ndimage.map_coordinates(
                source,
                (source_y, source_x),
                order=order,
                mode="constant",
                cval=cval,
                prefilter=False,
            )
        return np.ascontiguousarray(output)
    matrix_yx = np.asarray(
        [[inverse[1, 1], inverse[1, 0]], [inverse[0, 1], inverse[0, 0]]],
        dtype=np.float64,
    )
    offset_yx = np.asarray([inverse[1, 2], inverse[0, 2]], dtype=np.float64)
    warped = ndimage.affine_transform(
        np.asarray(image, dtype=np.float32),
        matrix_yx,
        offset=offset_yx,
        output_shape=output_shape,
        order=order,
        mode="constant",
        cval=cval,
        prefilter=order > 1,
    )
    return np.ascontiguousarray(warped, dtype=np.float32)


def _robust_warp_similarity(reference: FloatImage, candidate: FloatImage) -> tuple[float, float]:
    finite = np.isfinite(reference) & np.isfinite(candidate)
    valid_fraction = float(np.count_nonzero(finite) / finite.size)
    if np.count_nonzero(finite) < 100:
        return float("nan"), valid_fraction
    left = np.asarray(reference[finite], dtype=np.float64)
    right = np.asarray(candidate[finite], dtype=np.float64)
    # Normalize away filter/exposure differences; spatial agreement remains.
    for values in (left, right):
        values -= np.median(values)
        scale = 1.4826 * np.median(np.abs(values))
        if scale > 0:
            values /= scale
    return float(np.corrcoef(left, right)[0, 1]), valid_fraction


def register_analyses(
    analyses: Sequence[FrameAnalysis],
    *,
    reference_index: int | None = None,
    config: RegistrationConfig | None = None,
    validate_warp: bool = True,
    workers: int = 1,
) -> tuple[int, tuple[FrameTransform, ...]]:
    """Estimate every transform; ``workers`` bounds concurrent full-resolution
    refinements, which are independent per frame and give identical results
    for any worker count."""

    selected = config or RegistrationConfig()
    if workers < 1:
        raise ValueError("workers must be positive")
    index = choose_reference(analyses) if reference_index is None else reference_index
    if index < 0 or index >= len(analyses):
        raise IndexError("reference_index is out of range")
    reference = analyses[index]
    results: list[FrameTransform | None] = [None] * len(analyses)
    results[index] = _estimate_one(reference, reference, selected)

    # Build one small transform tree per filter.  Only one filter anchor pays
    # the harder cross-filter bootstrap; remaining frames use a reliable
    # same-filter bootstrap and compose matrices before a final all-catalog
    # refinement.  Besides being more robust, this avoids spending seconds
    # exhausting irrelevant cross-filter triangle lists on every frame.
    groups: dict[str | None, list[int]] = {}
    for analysis_index, analysis in enumerate(analyses):
        groups.setdefault(analysis.filter_name, []).append(analysis_index)
    # Groups are independent (each writes its own indices of ``results``), so
    # they run concurrently and share the worker budget between them.
    group_workers = max(1, min(workers, len(groups)))
    member_budget = max(1, workers // group_workers)

    def process_group(group_indices: list[int]) -> None:
        if index in group_indices:
            anchor_index = index
        else:
            anchor_index = -1
            # Stable input order is preferable here: a "best" SEP score can be
            # filter-biased and does not predict cross-filter triangle overlap.
            for candidate_index in group_indices:
                candidate = _estimate_one(analyses[candidate_index], reference, selected)
                results[candidate_index] = candidate
                if candidate.accepted:
                    anchor_index = candidate_index
                    break
            if anchor_index < 0:
                # Preserve diagnostics for every member if no group anchor can
                # reach the global reference.
                for candidate_index in group_indices:
                    if results[candidate_index] is None:
                        results[candidate_index] = _estimate_one(
                            analyses[candidate_index], reference, selected
                        )
                return

        anchor_result = results[anchor_index]
        assert anchor_result is not None
        pending = [
            source_index
            for source_index in group_indices
            if results[source_index] is None
        ]

        def estimate_member(source_index: int) -> FrameTransform:
            if anchor_index == index:
                return _estimate_one(analyses[source_index], reference, selected)
            recovered = _estimate_via_bridge(
                analyses[source_index],
                analyses[anchor_index],
                anchor_result,
                reference,
                selected,
            )
            # Direct fallback retains generality for unusual same-filter
            # failures without penalizing the normal fast path.
            return recovered or _estimate_one(
                analyses[source_index], reference, selected
            )

        # Member estimates depend only on the anchor and the shared catalogs,
        # so they run concurrently; results keep input order.
        member_workers = max(1, min(member_budget, len(pending)))
        if member_workers == 1:
            estimates = [estimate_member(source_index) for source_index in pending]
        else:
            with ThreadPoolExecutor(max_workers=member_workers) as executor:
                estimates = list(executor.map(estimate_member, pending))
        for source_index, estimate in zip(pending, estimates, strict=True):
            results[source_index] = estimate

    if group_workers == 1:
        for group_indices in groups.values():
            process_group(group_indices)
    else:
        with ThreadPoolExecutor(max_workers=group_workers) as executor:
            list(executor.map(process_group, groups.values()))

    if selected.refine_full_centroids:
        reference_image = read_full_image(reference.path)
        refine_indices = [
            source_index
            for source_index, optional_result in enumerate(results)
            if optional_result is not None
            and source_index != index
            and optional_result.accepted
        ]

        def refine(source_index: int) -> FrameTransform:
            coarse = results[source_index]
            assert coarse is not None
            # Each worker decodes its own full-resolution frame; the shared
            # reference image is read-only.
            source_image = read_full_image(analyses[source_index].path)
            return _refine_full_resolution(
                coarse,
                analyses[source_index],
                reference,
                source_image,
                reference_image,
                selected,
            )

        refine_workers = max(1, min(workers, len(refine_indices)))
        if refine_workers == 1:
            refined = [refine(source_index) for source_index in refine_indices]
        else:
            with ThreadPoolExecutor(max_workers=refine_workers) as executor:
                refined = list(executor.map(refine, refine_indices))
        for source_index, result in zip(refine_indices, refined, strict=True):
            results[source_index] = result

    def validate(position: int) -> FrameTransform:
        analysis = analyses[position]
        optional_result = results[position]
        assert optional_result is not None
        result = optional_result
        if validate_warp and result.accepted and result.preview_matrix is not None:
            started = time.perf_counter()
            warped = warp_image(
                analysis.preview,
                result.preview_matrix,
                reference.preview.shape,
                order=1,
            )
            elapsed = time.perf_counter() - started
            pearson, valid_fraction = _robust_warp_similarity(reference.preview, warped)
            result = FrameTransform(
                **{
                    **{name: getattr(result, name) for name in result.__dataclass_fields__},
                    "warp_seconds": elapsed,
                    "warp_pearson": pearson,
                    "warp_valid_fraction": valid_fraction,
                }
            )
        return result

    # Each frame's preview warp check reads the shared reference preview and
    # its own result only, so the checks run concurrently in input order.
    validate_workers = max(1, min(workers, len(analyses)))
    if validate_workers == 1:
        validated = [validate(position) for position in range(len(analyses))]
    else:
        with ThreadPoolExecutor(max_workers=validate_workers) as executor:
            validated = list(executor.map(validate, range(len(analyses))))
    return index, tuple(validated)


def run_registration(
    paths: Iterable[str],
    *,
    detection: DetectionConfig | None = None,
    registration: RegistrationConfig | None = None,
    calibration: CalibrationPlan | None = None,
    reference_path: str | None = None,
    validate_warp: bool = True,
    workers: int = 4,
) -> RegistrationRun:
    started = time.perf_counter()
    ordered = [str(Path(path).expanduser().resolve(strict=True)) for path in paths]
    if len(ordered) < 2:
        raise ValueError("at least two frames are required")
    if workers < 1:
        raise ValueError("workers must be positive")
    analysis_started = time.perf_counter()
    analyses = analyze_frames(
        ordered,
        detection=detection,
        calibration=calibration,
        workers=workers,
    )
    analysis_wall_seconds = time.perf_counter() - analysis_started
    reference_index = None
    if reference_path is not None:
        canonical = str(Path(reference_path).expanduser().resolve(strict=True))
        try:
            reference_index = ordered.index(canonical)
        except ValueError as error:
            raise ValueError("reference_path is not one of the input frames") from error
    registration_started = time.perf_counter()
    reference_index, transforms = register_analyses(
        analyses,
        reference_index=reference_index,
        config=registration,
        validate_warp=validate_warp,
        workers=workers,
    )
    registration_wall_seconds = time.perf_counter() - registration_started
    accepted = [item for item in transforms if item.accepted and item.full_matrix is not None]
    if len(accepted) != len(transforms):
        failures = [Path(item.path).name for item in transforms if not item.accepted]
        raise RuntimeError(f"registration failed for {len(failures)} frame(s): {failures}")
    return RegistrationRun(
        reference_index=reference_index,
        analyses=analyses,
        transforms=transforms,
        analysis_wall_seconds=analysis_wall_seconds,
        registration_wall_seconds=registration_wall_seconds,
        total_seconds=time.perf_counter() - started,
    )


_CFA_HEADER_KEYS = ("BAYERPAT", "BAYERPATN", "CFAPAT", "CFAPATTERN")


def _cfa_luminance_if_mosaic(data: FloatImage, pattern: Any) -> FloatImage:
    """A Bayer mosaic is measured on its debayered luminance, whose star
    profiles are smooth at the mosaic's own pixel coordinates."""

    if is_cfa_pattern(pattern):
        return np.ascontiguousarray(cfa_luminance(data, normalize_cfa_pattern(pattern)), dtype=np.float32)
    return data


def read_full_image(path: str) -> FloatImage:
    """Decode the first mono FITS/XISF image as native contiguous Float32.

    A Bayer mosaic (``BAYERPAT`` or the PixInsight CFA property) comes back
    as its full-resolution luminance so centroids are unbiased.
    """

    source = Path(path).expanduser().resolve(strict=True)
    folded = source.name.casefold()
    if folded.endswith((".fit", ".fits", ".fts", ".fit.fz", ".fits.fz", ".fts.fz")):
        with fits.open(
            source,
            mode="readonly",
            memmap=True,
            do_not_scale_image_data=True,
            uint=False,
            checksum=False,
        ) as hdul:
            hdu = next((item for item in hdul if int(item.header.get("NAXIS", 0) or 0) >= 2), None)
            if hdu is None or hdu.data is None:
                raise ValueError(f"no image in {source}")
            data = np.asarray(hdu.data, dtype=np.float32)
            data = np.squeeze(data)
            if data.ndim != 2:
                raise ValueError(f"expected mono image, got {data.shape}")
            bscale = float(hdu.header.get("BSCALE", 1.0) or 1.0)
            bzero = float(hdu.header.get("BZERO", 0.0) or 0.0)
            if bscale != 1.0 or bzero != 0.0:
                data = data * np.float32(bscale) + np.float32(bzero)
            pattern = next((hdu.header.get(key) for key in _CFA_HEADER_KEYS if hdu.header.get(key)), None)
            return _cfa_luminance_if_mosaic(np.ascontiguousarray(data, dtype=np.float32), pattern)
    if folded.endswith(".xisf"):
        document = XISF(str(source))
        data = np.asarray(document.read_image(0, data_format="channels_last"))
        data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(f"expected mono image, got {data.shape}")
        pattern = None
        images = document.get_images_metadata()
        if images:
            keywords = images[0].get("FITSKeywords", {}) if isinstance(images[0], dict) else {}
            for key in _CFA_HEADER_KEYS:
                entries = keywords.get(key) if isinstance(keywords, dict) else None
                if entries:
                    pattern = entries[0].get("value") if isinstance(entries[0], dict) else entries[0]
                    break
            if pattern is None:
                properties = images[0].get("XISFProperties", {}) if isinstance(images[0], dict) else {}
                entry = properties.get("PCL:CFASourcePattern") if isinstance(properties, dict) else None
                pattern = entry.get("value") if isinstance(entry, dict) else entry
        return _cfa_luminance_if_mosaic(np.ascontiguousarray(data, dtype=np.float32), pattern)
    raise ValueError(f"unsupported image format: {source}")
