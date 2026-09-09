"""Star-catalog registration for light-frame quality control.

The public transform direction is always ``source -> reference``.  Coordinates
are preview-image pixels in ``(x, y)`` order.

Accepted catalog inputs
-----------------------
``numpy.ndarray`` or another two-dimensional array-like
    Shape ``(N, 2)``.  Columns are ``x`` and ``y``.  When no flux is supplied,
    callers should put the brightest/reliable stars first because astroalign
    limits the number of control points it considers.
mapping (for example, ``dict``)
    Either ``{"points": array_like, "flux": optional_array_like}`` or
    ``{"x": array_like, "y": array_like, "flux": optional_array_like}``.
object
    An object exposing either ``.points`` or both ``.x`` and ``.y``.  A
    one-dimensional ``.flux`` attribute is optional.
sequence of row objects
    For direct use with detection models such as ``list[Star]``, every row may
    expose scalar ``.x`` and ``.y`` attributes plus an optional ``.flux``.

Flux is used only to order astroalign control points.  Registration and RMS are
computed from coordinates.  Invalid/non-finite coordinates and exact duplicate
coordinates are ignored while the returned indices continue to refer to the
caller's original catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import SimilarityTransform

from .triangle_bootstrap import find_catalog_transform


FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]


@runtime_checkable
class StarCatalogObject(Protocol):
    """Structural type accepted by :func:`register_star_catalogs`.

    Implementations may expose ``points`` instead of ``x``/``y``; runtime
    normalization supports both forms even though a Protocol cannot express
    that either/or relationship precisely.
    """

    x: ArrayLike
    y: ArrayLike


StarCatalogInput: TypeAlias = ArrayLike | Mapping[str, Any] | StarCatalogObject


@dataclass(frozen=True, slots=True)
class RegistrationThresholds:
    """Acceptance and robust-fit thresholds in preview-image pixels."""

    min_inliers: int = 12
    min_inlier_ratio: float = 0.25
    max_rms_px: float = 2.5
    match_radius_px: float = 5.0
    residual_threshold_px: float = 3.0
    max_control_points: int = 100
    max_trials: int = 1_000
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.min_inliers < 2:
            raise ValueError("min_inliers must be at least 2")
        if not 0.0 <= self.min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be between 0 and 1")
        if self.max_rms_px <= 0.0:
            raise ValueError("max_rms_px must be positive")
        if self.match_radius_px <= 0.0:
            raise ValueError("match_radius_px must be positive")
        if self.residual_threshold_px <= 0.0:
            raise ValueError("residual_threshold_px must be positive")
        if self.max_control_points < 3:
            raise ValueError("max_control_points must be at least 3")
        if self.max_trials < 1:
            raise ValueError("max_trials must be positive")


@dataclass(frozen=True, slots=True)
class CatalogMatches:
    """One-to-one nearest-neighbour matches under a supplied transform."""

    source_points: FloatArray
    reference_points: FloatArray
    source_indices: IntArray
    reference_indices: IntArray
    distances_px: FloatArray

    @property
    def count(self) -> int:
        return int(self.source_indices.size)


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    """Similarity-transform estimate and conservative QC acceptance decision.

    ``estimated`` says whether a transform was found.  ``accepted`` additionally
    requires at least 12 inliers, at least a 25% inlier ratio, and RMS no greater
    than 2.5 preview pixels under the default thresholds.

    The inlier-ratio denominator is ``min(source_count, reference_count)``.  That
    keeps a clean subset catalog from being penalized merely because the other
    frame detected many more faint stars.
    """

    transform: SimilarityTransform | None
    source_count: int
    reference_count: int
    matched_source_points: FloatArray
    matched_reference_points: FloatArray
    matched_source_indices: IntArray
    matched_reference_indices: IntArray
    residuals_px: FloatArray
    inlier_mask: BoolArray
    rms_px: float
    inlier_ratio: float
    accepted: bool
    reason_codes: tuple[str, ...]
    detail: str | None = None

    @property
    def estimated(self) -> bool:
        return self.transform is not None

    @property
    def match_count(self) -> int:
        return int(self.matched_source_indices.size)

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def matrix(self) -> FloatArray | None:
        if self.transform is None:
            return None
        return np.asarray(self.transform.params, dtype=np.float64).copy()

    def apply(self, points: ArrayLike) -> FloatArray:
        """Apply the source-to-reference transform to an ``(N, 2)`` array."""

        if self.transform is None:
            raise RuntimeError("registration did not produce a transform")
        coordinates = _coerce_points(points, label="points")
        return np.asarray(self.transform(coordinates), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _Catalog:
    points: FloatArray
    original_indices: IntArray
    flux: FloatArray | None


def _coerce_points(value: ArrayLike, *, label: str) -> FloatArray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{label} must have shape (N, 2) in (x, y) order")
    return points


def _extract_catalog(catalog: StarCatalogInput, *, label: str) -> _Catalog:
    flux_value: Any = None

    if isinstance(catalog, Mapping):
        if "points" in catalog:
            points = _coerce_points(catalog["points"], label=f"{label}['points']")
        elif "x" in catalog and "y" in catalog:
            x = np.asarray(catalog["x"], dtype=np.float64)
            y = np.asarray(catalog["y"], dtype=np.float64)
            if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
                raise ValueError(f"{label} x and y must be equal-length 1-D arrays")
            points = np.column_stack((x, y))
        else:
            raise ValueError(f"{label} mapping must contain 'points' or both 'x' and 'y'")
        flux_value = catalog.get("flux")
    elif hasattr(catalog, "points"):
        points = _coerce_points(getattr(catalog, "points"), label=f"{label}.points")
        flux_value = getattr(catalog, "flux", None)
    elif hasattr(catalog, "x") and hasattr(catalog, "y"):
        x = np.asarray(getattr(catalog, "x"), dtype=np.float64)
        y = np.asarray(getattr(catalog, "y"), dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError(f"{label}.x and {label}.y must be equal-length 1-D arrays")
        points = np.column_stack((x, y))
        flux_value = getattr(catalog, "flux", None)
    else:
        value = np.asarray(catalog)
        if value.dtype.names and {"x", "y"}.issubset(value.dtype.names):
            points = np.column_stack(
                (
                    np.asarray(value["x"], dtype=np.float64),
                    np.asarray(value["y"], dtype=np.float64),
                )
            )
            if "flux" in value.dtype.names:
                flux_value = value["flux"]
        elif value.ndim == 1 and value.size == 0:
            points = np.empty((0, 2), dtype=np.float64)
        elif value.ndim == 1 and all(
            hasattr(row, "x") and hasattr(row, "y") for row in value
        ):
            points = np.asarray(
                [(float(getattr(row, "x")), float(getattr(row, "y"))) for row in value],
                dtype=np.float64,
            )
            if all(hasattr(row, "flux") for row in value):
                flux_value = np.asarray(
                    [float(getattr(row, "flux")) for row in value],
                    dtype=np.float64,
                )
        else:
            points = _coerce_points(value, label=label)

    original_indices = np.arange(points.shape[0], dtype=np.int64)
    flux: FloatArray | None = None
    if flux_value is not None:
        flux = np.asarray(flux_value, dtype=np.float64)
        if flux.ndim != 1 or flux.shape[0] != points.shape[0]:
            raise ValueError(f"{label} flux must be a 1-D array with one value per point")

    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    original_indices = original_indices[finite]
    if flux is not None:
        flux = flux[finite]

    if points.size:
        # Exact duplicates make two-point similarity samples degenerate.  Keep
        # the first caller-visible index (normally the brightest after source
        # detection) and preserve the catalog's original ordering.
        _, unique_positions = np.unique(points, axis=0, return_index=True)
        unique_positions.sort()
        points = points[unique_positions]
        original_indices = original_indices[unique_positions]
        if flux is not None:
            flux = flux[unique_positions]

    return _Catalog(points=points, original_indices=original_indices, flux=flux)


def _control_points(catalog: _Catalog) -> FloatArray:
    if catalog.flux is None:
        return catalog.points
    finite_flux = np.where(np.isfinite(catalog.flux), catalog.flux, -np.inf)
    order = np.argsort(-finite_flux, kind="stable")
    return catalog.points[order]


def match_star_catalogs(
    source: StarCatalogInput,
    reference: StarCatalogInput,
    transform: SimilarityTransform,
    *,
    max_distance_px: float = 5.0,
) -> CatalogMatches:
    """Return deterministic one-to-one matches after applying ``transform``.

    All source/reference pairs within ``max_distance_px`` are considered.  Edges
    are consumed shortest-first so a crowded reference star cannot be assigned
    to more than one source star.
    """

    if max_distance_px <= 0.0:
        raise ValueError("max_distance_px must be positive")

    source_catalog = _extract_catalog(source, label="source")
    reference_catalog = _extract_catalog(reference, label="reference")
    if source_catalog.points.size == 0 or reference_catalog.points.size == 0:
        return _empty_matches()

    transformed = np.asarray(transform(source_catalog.points), dtype=np.float64)
    tree = cKDTree(reference_catalog.points)
    neighbours = tree.query_ball_point(transformed, r=max_distance_px)

    edges: list[tuple[float, int, int]] = []
    for source_index, candidates in enumerate(neighbours):
        for reference_index in candidates:
            distance = float(
                np.linalg.norm(
                    transformed[source_index] - reference_catalog.points[reference_index]
                )
            )
            edges.append((distance, source_index, int(reference_index)))
    edges.sort(key=lambda edge: (edge[0], edge[1], edge[2]))

    used_source: set[int] = set()
    used_reference: set[int] = set()
    selected: list[tuple[int, int, float]] = []
    for distance, source_index, reference_index in edges:
        if source_index in used_source or reference_index in used_reference:
            continue
        used_source.add(source_index)
        used_reference.add(reference_index)
        selected.append((source_index, reference_index, distance))

    selected.sort(key=lambda match: match[0])
    if not selected:
        return _empty_matches()

    source_positions = np.fromiter((item[0] for item in selected), dtype=np.int64)
    reference_positions = np.fromiter((item[1] for item in selected), dtype=np.int64)
    distances = np.fromiter((item[2] for item in selected), dtype=np.float64)
    return CatalogMatches(
        source_points=source_catalog.points[source_positions],
        reference_points=reference_catalog.points[reference_positions],
        source_indices=source_catalog.original_indices[source_positions],
        reference_indices=reference_catalog.original_indices[reference_positions],
        distances_px=distances,
    )


def _empty_matches() -> CatalogMatches:
    return CatalogMatches(
        source_points=np.empty((0, 2), dtype=np.float64),
        reference_points=np.empty((0, 2), dtype=np.float64),
        source_indices=np.empty(0, dtype=np.int64),
        reference_indices=np.empty(0, dtype=np.int64),
        distances_px=np.empty(0, dtype=np.float64),
    )


def _robust_similarity(
    source_points: FloatArray,
    reference_points: FloatArray,
    thresholds: RegistrationThresholds,
) -> SimilarityTransform | None:
    if source_points.shape[0] < 2:
        return None
    if source_points.shape[0] == 2:
        return _least_squares_similarity(source_points, reference_points)

    kwargs: dict[str, Any] = {
        "min_samples": 2,
        "residual_threshold": thresholds.residual_threshold_px,
        "max_trials": thresholds.max_trials,
    }
    parameters = inspect.signature(ransac).parameters
    if "rng" in parameters:
        kwargs["rng"] = np.random.default_rng(thresholds.random_seed)
    else:  # scikit-image < 0.23
        kwargs["random_state"] = thresholds.random_seed

    try:
        model, inliers = ransac(
            (source_points, reference_points),
            SimilarityTransform,
            **kwargs,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    if model is None or inliers is None or np.count_nonzero(inliers) < 2:
        return None
    return model


def _least_squares_similarity(
    source_points: FloatArray,
    reference_points: FloatArray,
) -> SimilarityTransform | None:
    """Estimate without emitting the scikit-image 0.26 deprecation warning."""

    factory = getattr(SimilarityTransform, "from_estimate", None)
    if factory is not None:  # scikit-image >= 0.26
        model = factory(source_points, reference_points)
        return model if model else None

    # Compatibility with the project's supported scikit-image 0.25 floor.
    model = SimilarityTransform()
    return model if model.estimate(source_points, reference_points) else None


def _failure_result(
    source_count: int,
    reference_count: int,
    *,
    code: str,
    detail: str | None = None,
) -> RegistrationResult:
    return RegistrationResult(
        transform=None,
        source_count=source_count,
        reference_count=reference_count,
        matched_source_points=np.empty((0, 2), dtype=np.float64),
        matched_reference_points=np.empty((0, 2), dtype=np.float64),
        matched_source_indices=np.empty(0, dtype=np.int64),
        matched_reference_indices=np.empty(0, dtype=np.int64),
        residuals_px=np.empty(0, dtype=np.float64),
        inlier_mask=np.empty(0, dtype=np.bool_),
        rms_px=float("inf"),
        inlier_ratio=0.0,
        accepted=False,
        reason_codes=(code,),
        detail=detail,
    )


def register_star_catalogs(
    source: StarCatalogInput,
    reference: StarCatalogInput,
    *,
    thresholds: RegistrationThresholds | None = None,
) -> RegistrationResult:
    """Estimate and validate a source-to-reference similarity transform.

    Astroalign supplies the rotation/scale/translation bootstrap, including
    arbitrary dither and 180-degree meridian-flip rotations.  SciPy performs
    all-catalog one-to-one matching and scikit-image RANSAC robustly refines the
    similarity transform before RMS and threshold evaluation.

    Invalid catalog *shape* raises ``ValueError``.  A valid but unregistrable
    pair returns a non-estimated result so batch QC can degrade gracefully.
    """

    limits = thresholds or RegistrationThresholds()
    source_catalog = _extract_catalog(source, label="source")
    reference_catalog = _extract_catalog(reference, label="reference")
    source_count = int(source_catalog.points.shape[0])
    reference_count = int(reference_catalog.points.shape[0])

    # Astroalign's geometric hash requires triangles, even though a similarity
    # transform itself has a two-point minimal sample.
    if min(source_count, reference_count) < 3:
        return _failure_result(
            source_count,
            reference_count,
            code="INSUFFICIENT_STARS_FOR_ESTIMATE",
        )

    try:
        initial_transform = find_catalog_transform(
            _control_points(source_catalog),
            _control_points(reference_catalog),
            max_control_points=limits.max_control_points,
        )
    except Exception as exc:  # External matcher failures are per-frame QC data.
        return _failure_result(
            source_count,
            reference_count,
            code="TRANSFORM_NOT_FOUND",
            detail=f"{type(exc).__name__}: {exc}",
        )

    model: SimilarityTransform = initial_transform
    # Two match/refine passes bring all stars (not just astroalign control
    # points) into the fit while retaining deterministic one-to-one matching.
    for _ in range(2):
        matches = match_star_catalogs(
            source_catalog.points,
            reference_catalog.points,
            model,
            max_distance_px=limits.match_radius_px,
        )
        refined = _robust_similarity(matches.source_points, matches.reference_points, limits)
        if refined is None:
            break
        model = refined

    final_matches = match_star_catalogs(
        source_catalog.points,
        reference_catalog.points,
        model,
        max_distance_px=limits.match_radius_px,
    )
    if final_matches.count < 2:
        return _failure_result(
            source_count,
            reference_count,
            code="TOO_FEW_MATCHES",
        )

    transformed = np.asarray(model(final_matches.source_points), dtype=np.float64)
    residuals = np.linalg.norm(transformed - final_matches.reference_points, axis=1)
    inlier_mask = residuals <= limits.residual_threshold_px

    # A final least-squares estimate on robust inliers reduces bias in the RMS.
    if np.count_nonzero(inlier_mask) >= 2:
        least_squares_model = _least_squares_similarity(
            final_matches.source_points[inlier_mask],
            final_matches.reference_points[inlier_mask],
        )
        if least_squares_model is not None:
            model = least_squares_model
            final_matches = match_star_catalogs(
                source_catalog.points,
                reference_catalog.points,
                model,
                max_distance_px=limits.match_radius_px,
            )
            transformed = np.asarray(model(final_matches.source_points), dtype=np.float64)
            residuals = np.linalg.norm(transformed - final_matches.reference_points, axis=1)
            inlier_mask = residuals <= limits.residual_threshold_px

    inlier_count = int(np.count_nonzero(inlier_mask))
    denominator = min(source_count, reference_count)
    inlier_ratio = float(inlier_count / denominator) if denominator else 0.0
    rms_px = (
        float(np.sqrt(np.mean(np.square(residuals[inlier_mask]))))
        if inlier_count
        else float("inf")
    )

    reason_codes: list[str] = []
    if inlier_count < limits.min_inliers:
        reason_codes.append("INSUFFICIENT_INLIERS")
    if inlier_ratio < limits.min_inlier_ratio:
        reason_codes.append("LOW_INLIER_RATIO")
    if not np.isfinite(rms_px) or rms_px > limits.max_rms_px:
        reason_codes.append("HIGH_RMS")

    return RegistrationResult(
        transform=model,
        source_count=source_count,
        reference_count=reference_count,
        matched_source_points=final_matches.source_points,
        matched_reference_points=final_matches.reference_points,
        matched_source_indices=source_catalog.original_indices[final_matches.source_indices],
        matched_reference_indices=reference_catalog.original_indices[
            final_matches.reference_indices
        ],
        residuals_px=residuals,
        inlier_mask=inlier_mask,
        rms_px=rms_px,
        inlier_ratio=inlier_ratio,
        accepted=not reason_codes,
        reason_codes=tuple(reason_codes),
    )


__all__ = [
    "CatalogMatches",
    "RegistrationResult",
    "RegistrationThresholds",
    "StarCatalogInput",
    "StarCatalogObject",
    "match_star_catalogs",
    "register_star_catalogs",
]
