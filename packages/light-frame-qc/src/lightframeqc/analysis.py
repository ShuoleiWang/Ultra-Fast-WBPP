from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
from skimage.transform import SimilarityTransform

from .config import QcConfig
from .analysis_cache import GroupAnalysisCache
from .grouping import build_groups
from .metadata import field_token
from .morphology import measure_fragmented_trails
from .parallel import FrameRunner
from .models import (
    Confidence,
    Decision,
    FrameFeatures,
    FrameMeasurement,
    FrameResult,
    FrameRole,
    RegistrationMetrics,
)


_HFR_PATTERN = re.compile(r"(?:^|_)HFR(?P<value>[0-9]+(?:\.[0-9]+)?)(?:_|$)", re.I)
from .registration import (
    RegistrationResult,
    RegistrationThresholds,
    register_star_catalogs,
)
from .statistics import (
    fit_clear_airmass_envelope,
    mad,
    median,
    percentile,
    robust_z,
    safe_log_extinction,
)


def _catalog(frame: FrameMeasurement) -> dict[str, np.ndarray]:
    return {
        "points": np.asarray([(star.x, star.y) for star in frame.stars], dtype=np.float64),
        "flux": np.asarray([star.flux for star in frame.stars], dtype=np.float64),
    }


def _limits(config: QcConfig, *, grouping: bool = False) -> RegistrationThresholds:
    return RegistrationThresholds(
        min_inliers=8 if grouping else config.minimum_registration_matches,
        min_inlier_ratio=0.15 if grouping else config.minimum_registration_fraction,
        max_rms_px=config.maximum_registration_rms_pixels,
        match_radius_px=max(4.0, config.maximum_registration_rms_pixels * 2.0),
        residual_threshold_px=config.maximum_registration_rms_pixels,
        max_control_points=100,
    )


def _registration(
    source: FrameMeasurement,
    reference: FrameMeasurement,
    config: QcConfig,
    *,
    grouping: bool = False,
) -> RegistrationResult:
    source_catalog = _catalog(source)
    reference_catalog = _catalog(reference)
    thresholds = _limits(config, grouping=grouping)
    best: RegistrationResult | None = None
    # astroalign 2.6 randomizes its internal triangle RANSAC from system
    # entropy.  Dense fields occasionally take an unlucky sample even for an
    # exact catalog subset.  Retry only failed estimates and retain the best
    # evidence; successful real-frame registrations still pay for one pass.
    for _attempt in range(3):
        result = register_star_catalogs(
            source_catalog,
            reference_catalog,
            thresholds=thresholds,
        )
        if best is None or (
            result.inlier_count,
            result.inlier_ratio,
            -result.rms_px,
        ) > (
            best.inlier_count,
            best.inlier_ratio,
            -best.rms_px,
        ):
            best = result
        if result.accepted:
            return result
    assert best is not None
    return best


def _split_auto_fields(
    group_id: str, frames: list[FrameMeasurement], config: QcConfig
) -> list[tuple[str, list[FrameMeasurement]]]:
    if not frames or field_token(frames[0].metadata) != "AUTO" or len(frames) == 1:
        return [(group_id, frames)]

    clusters: list[list[FrameMeasurement]] = []
    for frame in sorted(frames, key=lambda item: (-len(item.stars), item.metadata.path)):
        destination: int | None = None
        for index, cluster in enumerate(clusters):
            reference = cluster[0]
            if min(len(frame.stars), len(reference.stars)) < 8:
                continue
            registration = _registration(frame, reference, config, grouping=True)
            if registration.accepted:
                destination = index
                break
        if destination is None:
            clusters.append([frame])
        else:
            clusters[destination].append(frame)

    return [
        (f"{group_id}-field-{index + 1}", sorted(cluster, key=lambda item: item.metadata.path))
        for index, cluster in enumerate(clusters)
    ]


def _reference_score(frame: FrameMeasurement) -> tuple[int, float, str]:
    pi_score = frame.pixinsight.get(
        "psf_signal_weight", frame.pixinsight.get("psfSignalWeight")
    )
    try:
        pi_value = float(pi_score) if pi_score is not None else 0.0
    except (TypeError, ValueError):
        pi_value = 0.0
    detected = _detected_count(frame)
    # Geometry and completeness need the star-richest frame.  An optional
    # PixInsight metric is only a tie-breaker, never a stronger authority than
    # the uncapped detector count.
    return (detected, math.log1p(max(pi_value, 0.0)), frame.metadata.path)


def _supported_reference(
    frames: list[FrameMeasurement], config: QcConfig
) -> tuple[FrameMeasurement, dict[str, RegistrationResult]]:
    """Check a star-rich reference against at most two independent peers.

    A denser but unrelated field must not become the sole authority for a
    normal majority. Probe no more than three catalog pairs, using a small
    bootstrap and the complete catalogs for validation. Failed bootstraps are
    inconclusive; if no alternate pair establishes a supported reference, the
    ordinary registration and fail-closed gate retain authority.
    """

    # A tracking trail can be deblended into many compact detections and win
    # the source-count ranking. Require candidate references to be free of
    # independently established, distributed fragment-chain evidence.
    candidates = [
        frame for frame in frames
        if not measure_fragmented_trails(
            frame.raw_stars if frame.raw_stars is not None else frame.stars,
            frame.preview_width, frame.preview_height,
        ).detected
    ] or frames
    preferred = max(candidates, key=_reference_score)
    if len(frames) < 8:
        # This cohort cannot earn an automatic PASS in the first place.
        return preferred, {}
    peers: list[FrameMeasurement] = []
    for index in (0, len(candidates) // 2, len(candidates) - 1):
        candidate = candidates[index]
        if candidate is not preferred and all(candidate is not item for item in peers):
            peers.append(candidate)
        if len(peers) == 2:
            break
    # This bounds the bootstrap work, without lowering the full-catalog
    # minimum match count, inlier fraction, or RMS acceptance requirement.
    limits = replace(
        _limits(config),
        max_control_points=32,
        min_inliers=max(30, config.minimum_registration_matches),
        max_rms_px=min(1.5, config.maximum_registration_rms_pixels),
    )
    for peer in peers:
        registration = register_star_catalogs(_catalog(peer), _catalog(preferred), thresholds=limits)
        if registration.accepted:
            # A small bootstrap with only partial support is enough to avoid
            # replacing the reference, but should not replace the ordinary
            # full-control-point registration of that peer.
            cached = {peer.metadata.path: registration} if registration.inlier_ratio >= 0.90 else {}
            return preferred, cached
    if len(peers) == 2:
        registration = register_star_catalogs(_catalog(peers[1]), _catalog(peers[0]), thresholds=limits)
        if registration.accepted:
            # A pair meeting the ordinary complete-catalog acceptance bounds
            # supplies independent geometry; an extra 75% requirement here
            # would unnecessarily exclude supported catalogs with missing stars.
            # Two failed, bounded bootstraps cannot prove a different target.
            # They do establish missing independent geometry for this frame:
            # retain it as REVIEW instead of repeating the expensive rejected
            # bootstrap three more times. The verified majority can continue.
            unsupported = RegistrationResult(
                transform=None,
                source_count=len(preferred.stars),
                reference_count=len(peers[0].stars),
                matched_source_points=np.empty((0, 2), dtype=np.float64),
                matched_reference_points=np.empty((0, 2), dtype=np.float64),
                matched_source_indices=np.empty(0, dtype=np.int64),
                matched_reference_indices=np.empty(0, dtype=np.int64),
                residuals_px=np.empty(0, dtype=np.float64),
                inlier_mask=np.empty(0, dtype=np.bool_),
                rms_px=math.inf,
                inlier_ratio=0.0,
                accepted=False,
                reason_codes=("REFERENCE_CONNECTIVITY_UNRESOLVED",),
                detail="The candidate has no geometric support from two independently connected peers.",
            )
            cached = {preferred.metadata.path: unsupported}
            if registration.inlier_ratio >= 0.90:
                cached[peers[1].metadata.path] = registration
            return peers[0], cached
    return preferred, {}


def _detected_count(frame: FrameMeasurement) -> int:
    """Return uncapped SEP source count, with a synthetic/import fallback."""

    if frame.detected_source_count is not None:
        return frame.detected_source_count
    return len(frame.stars)


def _detected_ratio(
    frame: FrameMeasurement, reference: FrameMeasurement
) -> float | None:
    # Do not mix an uncapped SEP count with a possibly capped imported catalog.
    if (frame.detected_source_count is None) != (
        reference.detected_source_count is None
    ):
        return None
    reference_count = _detected_count(reference)
    return _detected_count(frame) / reference_count if reference_count > 0 else None


def _identity_registration(frame: FrameMeasurement) -> RegistrationResult:
    count = len(frame.stars)
    points = np.asarray([(star.x, star.y) for star in frame.stars], dtype=np.float64)
    indices = np.arange(count, dtype=np.int64)
    return RegistrationResult(
        transform=SimilarityTransform(),
        source_count=count,
        reference_count=count,
        matched_source_points=points,
        matched_reference_points=points.copy(),
        matched_source_indices=indices,
        matched_reference_indices=indices.copy(),
        residuals_px=np.zeros(count, dtype=np.float64),
        inlier_mask=np.ones(count, dtype=np.bool_),
        rms_px=0.0,
        inlier_ratio=1.0,
        accepted=count >= 1,
        reason_codes=(),
    )


def _matrix(registration: RegistrationResult) -> list[list[float]] | None:
    value = registration.matrix
    return value.tolist() if value is not None else None


def _registration_metrics(registration: RegistrationResult) -> RegistrationMetrics:
    error = None
    if registration.reason_codes:
        error = ",".join(registration.reason_codes)
        if registration.detail:
            error += ": " + registration.detail
    return RegistrationMetrics(
        ok=registration.accepted,
        matched_stars=registration.inlier_count,
        match_fraction=registration.inlier_ratio,
        rms_pixels=(registration.rms_px if math.isfinite(registration.rms_px) else None),
        matrix=_matrix(registration),
        source_indices=registration.matched_source_indices[
            registration.inlier_mask
        ].astype(int).tolist(),
        reference_indices=registration.matched_reference_indices[
            registration.inlier_mask
        ].astype(int).tolist(),
        error=error,
    )


def _cell(x: float, y: float, width: int, height: int, rows: int, columns: int) -> tuple[int, int] | None:
    if width <= 0 or height <= 0 or not (0 <= x < width and 0 <= y < height):
        return None
    column = min(columns - 1, int(x / width * columns))
    row = min(rows - 1, int(y / height * rows))
    return row, column


def _reference_point_is_in_source(
    x: float,
    y: float,
    frame: FrameMeasurement,
    inverse: Any,
) -> bool:
    """Whether a reference coordinate lies in the candidate frame footprint.

    ``inverse`` is the registration's inverse transform, built once per frame
    (``transform.inverse`` inverts the matrix on every access; applying the
    prebuilt object to one point is the same arithmetic).
    """

    if inverse is None:
        return False
    try:
        source = np.asarray(
            inverse(np.asarray([[x, y]], dtype=np.float64)),
            dtype=np.float64,
        )[0]
    except (ValueError, np.linalg.LinAlgError):
        return False
    return bool(
        np.all(np.isfinite(source))
        and 0.0 <= source[0] < frame.preview_width
        and 0.0 <= source[1] < frame.preview_height
    )


def _finite_grid(grid: list[list[float | None]]) -> np.ndarray:
    if not grid:
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray(
        [[np.nan if value is None else float(value) for value in row] for row in grid],
        dtype=np.float64,
    )


def _remap_candidate_grid(
    frame: FrameMeasurement,
    reference: FrameMeasurement,
    registration: RegistrationResult,
    grid: list[list[float | None]],
    rows: int,
    columns: int,
) -> np.ndarray:
    source_grid = _finite_grid(grid)
    result = np.full((rows, columns), np.nan, dtype=np.float64)
    if source_grid.shape != (rows, columns) or registration.transform is None:
        return result
    # Inverse sampling gives every reference-grid cell one deterministic source
    # sample.  Forward voting left collision holes under rotation and made a
    # rotated detector pattern look like missing background evidence.
    for row in range(rows):
        for column in range(columns):
            reference_x = (column + 0.5) / columns * reference.preview_width
            reference_y = (row + 0.5) / rows * reference.preview_height
            source = np.asarray(
                registration.transform.inverse(
                    np.asarray([[reference_x, reference_y]], dtype=np.float64)
                ),
                dtype=np.float64,
            )[0]
            source_cell = _cell(
                float(source[0]),
                float(source[1]),
                frame.preview_width,
                frame.preview_height,
                rows,
                columns,
            )
            if source_cell is None:
                continue
            value = source_grid[source_cell]
            if math.isfinite(value):
                result[row, column] = float(value)
    return result


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    rows, columns = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for start_row in range(rows):
        for start_column in range(columns):
            if not mask[start_row, start_column] or seen[start_row, start_column]:
                continue
            component: list[tuple[int, int]] = []
            queue = deque([(start_row, start_column)])
            seen[start_row, start_column] = True
            while queue:
                row, column = queue.popleft()
                component.append((row, column))
                for next_row, next_column in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                ):
                    if (
                        0 <= next_row < rows
                        and 0 <= next_column < columns
                        and mask[next_row, next_column]
                        and not seen[next_row, next_column]
                    ):
                        seen[next_row, next_column] = True
                        queue.append((next_row, next_column))
            components.append(component)
    return components


def _spatial_features(
    frame: FrameMeasurement,
    reference: FrameMeasurement,
    registration: RegistrationResult,
    config: QcConfig,
) -> tuple[FrameFeatures, dict[str, Any]]:
    features = FrameFeatures()
    rows, columns = config.grid_rows, config.grid_columns
    reference_expected = np.zeros((rows, columns), dtype=np.int32)
    expected = np.zeros((rows, columns), dtype=np.int32)
    matched = np.zeros((rows, columns), dtype=np.int32)
    cell_residuals: list[list[list[float]]] = [[[] for _ in range(columns)] for _ in range(rows)]
    # Sparse star fields cannot populate three stars in each 16x16 cell.
    # Preserve independently measured coarse-cell flux evidence for the same
    # multi-frame consensus, rather than silently losing local-cloud checks.
    coarse_residuals: list[list[list[float]]] = [[[] for _ in range(4)] for _ in range(4)]

    valid = registration.inlier_mask
    source_indices = registration.matched_source_indices[valid]
    reference_indices = registration.matched_reference_indices[valid]
    ref_stars = reference.stars
    source_stars = frame.stars
    try:
        inverse = registration.transform.inverse if registration.transform is not None else None
    except (ValueError, np.linalg.LinAlgError):
        inverse = None

    for star in ref_stars:
        location = _cell(
            star.x,
            star.y,
            reference.preview_width,
            reference.preview_height,
            rows,
            columns,
        )
        if location is not None:
            reference_expected[location] += 1
        # Stars outside the candidate's transformed footprint are not missing:
        # they are simply outside the common field after dither/rotation.
        if not _reference_point_is_in_source(star.x, star.y, frame, inverse):
            continue
        location = _cell(
            star.x,
            star.y,
            reference.preview_width,
            reference.preview_height,
            rows,
            columns,
        )
        if location is not None:
            expected[location] += 1

    exposure_scale = 1.0
    if (
        frame.metadata.exposure_seconds
        and reference.metadata.exposure_seconds
        and frame.metadata.exposure_seconds > 0
    ):
        exposure_scale = reference.metadata.exposure_seconds / frame.metadata.exposure_seconds

    ratio_records: list[tuple[int, float]] = []
    for source_index, reference_index in zip(source_indices, reference_indices, strict=True):
        if source_index >= len(source_stars) or reference_index >= len(ref_stars):
            continue
        source_star = source_stars[int(source_index)]
        reference_star = ref_stars[int(reference_index)]
        location = _cell(
            reference_star.x,
            reference_star.y,
            reference.preview_width,
            reference.preview_height,
            rows,
            columns,
        )
        if location is None:
            continue
        matched[location] += 1
        if source_star.flux > 0 and reference_star.flux > 0:
            ratio_records.append(
                (int(reference_index), source_star.flux / reference_star.flux * exposure_scale)
            )

    ratios = [ratio for _, ratio in ratio_records if math.isfinite(ratio) and ratio > 0]
    global_ratio = median(ratios)
    features.transparency_ratio = global_ratio
    if global_ratio and global_ratio > 0:
        residuals: list[float] = []
        for reference_index, ratio in ratio_records:
            if ratio <= 0:
                continue
            residual = -2.5 * math.log10(ratio / global_ratio)
            residuals.append(residual)
            star = ref_stars[reference_index]
            location = _cell(
                star.x,
                star.y,
                reference.preview_width,
                reference.preview_height,
                rows,
                columns,
            )
            if location is not None:
                cell_residuals[location[0]][location[1]].append(residual)
            coarse_location = _cell(
                star.x, star.y, reference.preview_width, reference.preview_height, 4, 4
            )
            if coarse_location is not None:
                coarse_residuals[coarse_location[0]][coarse_location[1]].append(residual)
        features.spatial_transparency_mad_mag = mad(residuals, 0.0)
        # Individual-star MAD can miss a cloud bank that affects fewer than
        # half of the surviving stars.  Cell medians suppress faint-star noise;
        # their high percentile retains sensitivity to a localized patch while
        # the per-frame global ratio above removes genuine whole-frame changes.
        cell_medians: list[float] = []
        for row in range(rows):
            for column in range(columns):
                if len(cell_residuals[row][column]) < 3:
                    continue
                cell_value = median(cell_residuals[row][column])
                if cell_value is not None and math.isfinite(cell_value):
                    cell_medians.append(cell_value)
        if len(cell_medians) >= 16:
            features.spatial_transparency_p90_mag = percentile(
                [abs(value) for value in cell_medians], 90
            )

    expected_total = int(expected.sum())
    matched_total = int(matched.sum())
    features.star_completeness = (
        matched_total / expected_total if expected_total > 0 else None
    )
    completeness = np.divide(
        matched,
        expected,
        out=np.full((rows, columns), np.nan, dtype=np.float64),
        where=expected > 0,
    )
    global_completeness = features.star_completeness or 0.0

    candidate_background = _remap_candidate_grid(
        frame,
        reference,
        registration,
        frame.background_grid,
        rows,
        columns,
    )
    reference_background = _finite_grid(reference.background_grid)
    candidate_texture = _remap_candidate_grid(
        frame,
        reference,
        registration,
        frame.texture_grid,
        rows,
        columns,
    )
    reference_texture = _finite_grid(reference.texture_grid)

    def normalize(values: np.ndarray) -> np.ndarray:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return np.full_like(values, np.nan)
        location = float(np.median(finite))
        scale = max(1.4826 * float(np.median(np.abs(finite - location))), 1e-8)
        return (values - location) / scale

    normalized_candidate_background = normalize(candidate_background)
    background_delta = normalized_candidate_background - normalize(reference_background)
    texture_ratio = np.divide(
        candidate_texture,
        reference_texture,
        out=np.full_like(candidate_texture, np.nan),
        where=np.isfinite(reference_texture) & (reference_texture > 1e-12),
    )

    missing = (
        (expected >= 3)
        & np.isfinite(completeness)
        & (completeness <= np.maximum(0.20, global_completeness * 0.35))
    )
    reference_supported = reference_expected >= 3
    supported = (expected >= 3) & reference_supported
    supported_cells = int(np.count_nonzero(supported))
    reference_supported_cells = int(np.count_nonzero(reference_supported))
    features.overlap_fraction = (
        supported_cells / reference_supported_cells
        if reference_supported_cells > 0
        else None
    )
    missing &= supported
    components = _connected_components(missing)
    largest = max(components, key=len) if components else []
    features.largest_missing_region = (
        len(largest) / supported_cells if supported_cells > 0 else None
    )

    component_mask = np.zeros((rows, columns), dtype=bool)
    for row, column in largest:
        component_mask[row, column] = True
    outside_mask = supported & ~component_mask
    inside_expected = int(expected[component_mask].sum())
    inside_matched = int(matched[component_mask].sum())
    outside_expected = int(expected[outside_mask].sum())
    outside_matched = int(matched[outside_mask].sum())
    inside_density = inside_matched / inside_expected if inside_expected else 1.0
    outside_density = outside_matched / outside_expected if outside_expected else 1.0
    features.missing_inside_outside_ratio = (
        inside_density / max(outside_density, 1e-6) if largest else None
    )

    background_evidence = component_mask & (
        (np.abs(background_delta) >= 3.0)
        | (np.isfinite(texture_ratio) & (texture_ratio <= 0.50))
    )
    features.background_support = (
        float(background_evidence[component_mask].mean()) if largest else None
    )

    boundary_total = 0
    boundary_supported = 0
    for row, column in largest:
        for other_row, other_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if not (0 <= other_row < rows and 0 <= other_column < columns):
                continue
            if component_mask[other_row, other_column]:
                continue
            boundary_total += 1
            local_jump = (
                math.isfinite(completeness[row, column])
                and math.isfinite(completeness[other_row, other_column])
                and abs(completeness[row, column] - completeness[other_row, other_column]) >= 0.5
            )
            background_jump = (
                math.isfinite(candidate_background[row, column])
                and math.isfinite(candidate_background[other_row, other_column])
                and abs(
                    normalized_candidate_background[row, column]
                    - normalized_candidate_background[other_row, other_column]
                )
                >= 3.0
            )
            if local_jump or background_jump:
                boundary_supported += 1
    features.boundary_support = (
        boundary_supported / boundary_total if boundary_total else None
    )

    residual_grid = [
        [
            median(cell_residuals[row][column])
            if len(cell_residuals[row][column]) >= 3
            else None
            for column in range(columns)
        ]
        for row in range(rows)
    ]
    grid = {
        "rows": rows,
        "columns": columns,
        "expectedStars": expected.astype(int).tolist(),
        "referenceExpectedStars": reference_expected.astype(int).tolist(),
        "matchedStars": matched.astype(int).tolist(),
        "completeness": [
            [None if not math.isfinite(value) else float(value) for value in row]
            for row in completeness
        ],
        "transparencyResidualMag": residual_grid,
        "coarseTransparencyResidualMag": [
            [median(values) if len(values) >= 3 else None for values in row]
            for row in coarse_residuals
        ],
        "missingMask": component_mask.tolist(),
        "supportedOverlapMask": supported.tolist(),
        "backgroundDeltaRobustSigma": [
            [None if not math.isfinite(value) else float(value) for value in row]
            for row in background_delta
        ],
        "textureRatio": [
            [None if not math.isfinite(value) else float(value) for value in row]
            for row in texture_ratio
        ],
    }
    return features, grid


def _apply_spatial_consensus(results: list[FrameResult]) -> None:
    """Score directional spatial extinction against a multi-frame clear envelope.

    The input grids have already had each frame's global flux scale removed.
    Restore that scale before comparing frames: an uneven, faint frame must
    not make its relatively brighter cells brighter than genuinely clear sky.
    Build the envelope in this common extinction system, then remove each
    candidate's global offset again to isolate spatial rather than uniform
    dimming. Global transparency and airmass retain their separate evidence.
    """

    if len(results) < 3:
        return
    shapes = {
        (
            int(result.grid.get("rows", 0)),
            int(result.grid.get("columns", 0)),
        )
        for result in results
        if result.grid.get("transparencyResidualMag")
    }
    if len(shapes) != 1:
        return
    rows, columns = next(iter(shapes))
    if rows <= 0 or columns <= 0:
        return

    global_extinction = []
    for result in results:
        ratio = result.features.transparency_ratio
        global_extinction.append(
            -2.5 * math.log10(ratio)
            if ratio is not None and math.isfinite(ratio) and ratio > 0
            else math.nan
        )

    grids: list[np.ndarray] = []
    for result, extinction in zip(results, global_extinction, strict=True):
        raw = result.grid.get("transparencyResidualMag")
        if not raw:
            grids.append(np.full((rows, columns), np.nan, dtype=np.float64))
            continue
        values = np.asarray(
            [
                [np.nan if value is None else float(value) for value in row]
                for row in raw
            ],
            dtype=np.float64,
        )
        grids.append(
            values + extinction
            if values.shape == (rows, columns)
            else np.full((rows, columns), np.nan, dtype=np.float64)
        )

    envelope = np.full((rows, columns), np.nan, dtype=np.float64)
    for row in range(rows):
        for column in range(columns):
            samples = np.asarray(
                [
                    grid[row, column]
                    for grid in grids
                    if math.isfinite(grid[row, column])
                ],
                dtype=np.float64,
            )
            if samples.size >= 3:
                # A low percentile follows the least-extinguished observations
                # while avoiding a literal single-frame minimum.
                envelope[row, column] = float(np.percentile(samples, 10))

    for result, values in zip(results, grids, strict=True):
        delta = values - envelope
        finite = delta[np.isfinite(delta)]
        if finite.size < 16:
            result.features.spatial_dimming_p90_mag = None
            result.features.spatial_brightening_p90_mag = None
            continue
        delta = delta - float(np.median(finite))
        finite = delta[np.isfinite(delta)]
        result.features.spatial_transparency_mad_mag = mad(finite, 0.0)
        result.features.spatial_transparency_p90_mag = percentile(
            np.abs(finite), 90
        )
        dimming = percentile(finite, 90)
        brightening = percentile(-finite, 90)
        result.features.spatial_dimming_p90_mag = (
            max(0.0, dimming) if dimming is not None else None
        )
        result.features.spatial_brightening_p90_mag = (
            max(0.0, brightening) if brightening is not None else None
        )
        result.grid["consensusDimmingResidualMag"] = [
            [None if not math.isfinite(value) else float(value) for value in row]
            for row in delta
        ]

    # Coarsening changes spatial resolution, not the extinction thresholds.
    # Require at least eight cells with three matched stars apiece, each bound
    # to a >=3-frame clear envelope. This only supplies the existing spatial
    # evidence family: by itself it requests review, never a hard cloud reject.
    coarse_grids: list[np.ndarray] = []
    for result, extinction in zip(results, global_extinction, strict=True):
        raw = result.grid.get("coarseTransparencyResidualMag")
        values = np.asarray(raw, dtype=np.float64) if raw else np.empty((0, 0))
        coarse_grids.append(
            values + extinction if values.shape == (4, 4) else np.full((4, 4), np.nan)
        )
    coarse_envelope = np.full((4, 4), np.nan)
    for row in range(4):
        for column in range(4):
            samples = [grid[row, column] for grid in coarse_grids if math.isfinite(grid[row, column])]
            if len(samples) >= 3:
                coarse_envelope[row, column] = float(np.percentile(samples, 10))
    for result, values in zip(results, coarse_grids, strict=True):
        if result.features.spatial_dimming_p90_mag is not None:
            continue
        delta = values - coarse_envelope
        finite = delta[np.isfinite(delta)]
        if finite.size < 8:
            continue
        delta = delta - float(np.median(finite))
        finite = delta[np.isfinite(delta)]
        result.features.spatial_transparency_mad_mag = mad(finite, 0.0)
        result.features.spatial_transparency_p90_mag = percentile(np.abs(finite), 90)
        result.features.spatial_dimming_p90_mag = max(0.0, float(np.percentile(finite, 90)))
        result.features.spatial_brightening_p90_mag = max(0.0, float(np.percentile(-finite, 90)))
        result.grid["coarseConsensusDimmingResidualMag"] = [
            [None if not math.isfinite(value) else float(value) for value in row]
            for row in delta
        ]


def _confidence(
    group_size: int, registration: RegistrationResult, config: QcConfig
) -> Confidence:
    if not registration.accepted:
        return Confidence.LOW if registration.estimated else Confidence.NONE
    if (
        group_size >= config.high_confidence_group_frames
        and registration.inlier_count >= config.high_confidence_registration_matches
        and registration.rms_px <= config.high_confidence_rms_pixels
    ):
        return Confidence.HIGH
    if (
        group_size >= config.minimum_group_frames
        and registration.inlier_count >= config.minimum_registration_matches
        and registration.rms_px <= config.maximum_registration_rms_pixels
    ):
        return Confidence.MEDIUM
    return Confidence.LOW


def _nina_hfr(frame: FrameMeasurement) -> float | None:
    header_value = frame.metadata.header.get("HFR")
    if header_value is not None:
        try:
            value = float(header_value)
        except (TypeError, ValueError):
            value = float("nan")
        if math.isfinite(value) and value > 0:
            return value
    match = _HFR_PATTERN.search(frame.metadata.path.rsplit("/", 1)[-1])
    if not match:
        return None
    value = float(match.group("value"))
    return value if math.isfinite(value) and value > 0 else None


def _populate_morphology(
    results: list[FrameResult], measurements: list[FrameMeasurement]
) -> None:
    for result, measurement in zip(results, measurements, strict=True):
        usable = [star for star in measurement.stars if star.flags == 0]
        if len(usable) < 30:
            usable = list(measurement.stars)
        fwhm = [star.fwhm for star in usable if math.isfinite(star.fwhm)]
        ellipticity = [
            star.ellipticity
            for star in usable
            if math.isfinite(star.ellipticity)
        ]
        features = result.features
        features.nina_hfr_pixels = _nina_hfr(measurement)
        features.median_fwhm_preview_pixels = median(fwhm)
        features.p90_fwhm_preview_pixels = percentile(fwhm, 90)
        features.median_ellipticity = median(ellipticity)
        features.p90_ellipticity = percentile(ellipticity, 90)
        native = measurement.native_psf or {}
        if isinstance(native, dict) and native.get("r50Pixels") is not None:
            features.psf_r50_native_pixels = float(native["r50Pixels"])
            features.psf_fwhm_native_pixels = float(native["fwhmPixels"])
            features.psf_wing_fraction = (
                float(native["wingFraction"]) if native.get("wingFraction") is not None else None
            )
            features.psf_native_star_count = int(native.get("starCount") or 0)


def _robust_center_scale(values: list[float | None]) -> tuple[float | None, float | None]:
    center = median(values)
    if center is None:
        return None, None
    dispersion = mad(values, center)
    if dispersion is None:
        return center, None
    scale = 1.4826 * dispersion
    return center, scale if scale > 1e-9 else None


def _high_outlier(
    value: float | None,
    center: float | None,
    scale: float | None,
    *,
    minimum_ratio: float,
    minimum_z: float,
    standalone_ratio: float,
) -> bool:
    if value is None or center is None or center <= 0:
        return False
    ratio = value / center
    if ratio >= standalone_ratio:
        return True
    return bool(
        scale is not None
        and scale > 0
        and ratio >= minimum_ratio
        and (value - center) / scale >= minimum_z
    )


def _apply_scores(
    results: list[FrameResult], measurements: list[FrameMeasurement], config: QcConfig
) -> None:
    background_values = [measurement.image_median for measurement in measurements]
    hfr_center, hfr_scale = _robust_center_scale(
        [item.features.nina_hfr_pixels for item in results]
    )
    fwhm_center, fwhm_scale = _robust_center_scale(
        [item.features.median_fwhm_preview_pixels for item in results]
    )
    ellipticity_center, ellipticity_scale = _robust_center_scale(
        [item.features.median_ellipticity for item in results]
    )
    p90_ellipticity_center, p90_ellipticity_scale = _robust_center_scale(
        [item.features.p90_ellipticity for item in results]
    )

    extinction = [safe_log_extinction(item.features.transparency_ratio) for item in results]
    extra = fit_clear_airmass_envelope(
        [measurement.metadata.airmass for measurement in measurements], extinction
    )
    for result, measurement, extra_extinction in zip(results, measurements, extra, strict=True):
        features = result.features
        features.extra_extinction_mag = extra_extinction
        reasons: list[str] = []
        cloud_score = 0
        cloud_families = 0
        has_strong_cloud = False

        # Only directional dimming against the multi-frame bright envelope is
        # cloud evidence.  Local brightening remains diagnostic and cannot
        # make a cloudy reference contaminate otherwise clear frames.
        spatial_dimming = features.spatial_dimming_p90_mag
        if (
            spatial_dimming is not None
            and spatial_dimming >= config.strong_spatial_transparency_p90_mag
        ):
            cloud_score += 2
            cloud_families += 1
            has_strong_cloud = True
            reasons.append("CLOUD_SPATIAL_TRANSPARENCY_STRONG")
        elif (
            spatial_dimming is not None
            and spatial_dimming >= config.weak_spatial_transparency_p90_mag
        ):
            cloud_score += 1
            cloud_families += 1
            reasons.append("CLOUD_SPATIAL_TRANSPARENCY")

        completeness = features.star_completeness
        if (
            completeness is not None
            and completeness <= config.strong_completeness_ratio
        ):
            cloud_score += 2
            cloud_families += 1
            has_strong_cloud = True
            reasons.append("CLOUD_STAR_COMPLETENESS_STRONG")
        elif (
            completeness is not None
            and completeness <= config.weak_completeness_ratio
        ):
            cloud_score += 1
            cloud_families += 1
            reasons.append("CLOUD_STAR_COMPLETENESS")

        # The raw SEP count is intentionally never an automatic cloud family
        # unless an independent airmass/time extinction model also says the
        # loss is unexplained.  Without that model it can still force REVIEW,
        # but cannot combine with another signal to auto-reject a normally
        # dimmer low-altitude frame.
        detected_ratio = features.detected_source_ratio
        if (
            detected_ratio is not None
            and detected_ratio <= config.strong_completeness_ratio
        ):
            if extra_extinction is None:
                cloud_score += 2
                reasons.append("SOURCE_COUNT_DROP_UNMODELED_STRONG")
            elif extra_extinction >= config.weak_extra_extinction_mag:
                cloud_score += 2
                cloud_families += 1
                has_strong_cloud = True
                reasons.append("CLOUD_SOURCE_COUNT_STRONG")
        elif (
            detected_ratio is not None
            and detected_ratio <= config.weak_completeness_ratio
        ):
            if extra_extinction is None:
                cloud_score += 1
                reasons.append("SOURCE_COUNT_DROP_UNMODELED")
            elif extra_extinction >= config.weak_extra_extinction_mag:
                cloud_score += 1
                cloud_families += 1
                reasons.append("CLOUD_SOURCE_COUNT")

        if extra_extinction is not None and extra_extinction >= config.strong_extra_extinction_mag:
            cloud_score += 2
            cloud_families += 1
            has_strong_cloud = True
            reasons.append("CLOUD_TEMPORAL_EXTINCTION_STRONG")
        elif extra_extinction is not None and extra_extinction >= config.weak_extra_extinction_mag:
            cloud_score += 1
            cloud_families += 1
            reasons.append("CLOUD_TEMPORAL_EXTINCTION")

        background_z = robust_z(measurement.image_median, background_values, scale_floor=1e-8)
        if background_z is not None and abs(background_z) >= 3.0:
            cloud_score += 1
            cloud_families += 1
            reasons.append("CLOUD_BACKGROUND_SCATTER")

        features.cloud_score = cloud_score

        area = features.largest_missing_region or 0.0
        density = features.missing_inside_outside_ratio
        boundary = features.boundary_support or 0.0
        background_support = features.background_support or 0.0
        occlusion_score = 0
        strong_occlusion = False
        if area >= config.weak_occlusion_area:
            occlusion_score += 1
            reasons.append("OCC_LARGE_CONNECTED_REGION")
            # A collapsed single 16x16 cell is common in dense natural star
            # fields and is not a wall.  Density, boundary and background only
            # become obstruction evidence after a materially connected area
            # exists.
            if density is not None and density <= config.weak_occlusion_density_ratio:
                occlusion_score += 1
                reasons.append("OCC_STAR_DENSITY_COLLAPSE")
            if boundary >= config.weak_boundary_support:
                occlusion_score += 1
                reasons.append("OCC_BOUNDARY")
            if background_support >= 0.50:
                occlusion_score += 1
                reasons.append("OCC_BACKGROUND_OR_TEXTURE")
        if (
            area >= config.strong_occlusion_area
            and density is not None
            and density <= config.strong_occlusion_density_ratio
            and background_support >= 0.50
            and boundary >= config.strong_boundary_support
        ):
            strong_occlusion = True
            occlusion_score = max(occlusion_score, 5)
            reasons.append("OCC_HARD_GEOMETRY_STRONG")
        elif (
            area >= config.very_strong_occlusion_area
            and density is not None
            and density <= config.weak_occlusion_density_ratio
            and background_support >= 0.50
        ):
            strong_occlusion = True
            occlusion_score = max(occlusion_score, 5)
            reasons.append("OCC_VERY_LARGE_REGION_STRONG")
        features.occlusion_score = occlusion_score

        hfr_outlier = _high_outlier(
            features.nina_hfr_pixels,
            hfr_center,
            hfr_scale,
            minimum_ratio=1.08,
            minimum_z=4.0,
            standalone_ratio=1.15,
        )
        fwhm_outlier = _high_outlier(
            features.median_fwhm_preview_pixels,
            fwhm_center,
            fwhm_scale,
            minimum_ratio=1.08,
            minimum_z=4.0,
            standalone_ratio=1.18,
        )
        shape_score = 0
        if hfr_outlier or fwhm_outlier:
            shape_score = 2 if hfr_outlier else 1
            reasons.append("SHAPE_FOCUS_OR_SEEING_OUTLIER")

        median_trailing = _high_outlier(
            features.median_ellipticity,
            ellipticity_center,
            ellipticity_scale,
            minimum_ratio=1.20,
            minimum_z=4.0,
            standalone_ratio=1.50,
        ) and (features.median_ellipticity or 0.0) >= 0.35
        p90_trailing = _high_outlier(
            features.p90_ellipticity,
            p90_ellipticity_center,
            p90_ellipticity_scale,
            minimum_ratio=1.15,
            minimum_z=4.0,
            standalone_ratio=1.40,
        ) and (features.p90_ellipticity or 0.0) >= 0.65
        if median_trailing or p90_trailing:
            shape_score = max(shape_score, 2)
            reasons.append("SHAPE_TRAILING_OUTLIER")
        features.shape_score = shape_score

        assessable = result.confidence in (Confidence.MEDIUM, Confidence.HIGH)
        automatic_reject_allowed = result.confidence is Confidence.HIGH
        if strong_occlusion and automatic_reject_allowed:
            result.decision = Decision.REJECT_OCCLUSION
        elif (
            automatic_reject_allowed
            and cloud_score >= config.automatic_cloud_score
            and cloud_families >= 2
            and has_strong_cloud
        ):
            result.decision = Decision.REJECT_CLOUD
        elif cloud_score >= 2 or occlusion_score >= 2 or shape_score >= 2:
            result.decision = Decision.REVIEW
        elif assessable:
            result.decision = Decision.KEEP
        else:
            result.decision = Decision.UNASSESSABLE
        result.reasons = sorted(set(reasons))


@dataclass(frozen=True)
class _FrameTask:
    """One frame's registration and spatial features against the reference."""

    frame: FrameMeasurement
    reference: FrameMeasurement
    is_reference: bool
    registration: RegistrationResult | None
    config: QcConfig


def _analyze_frame(task: _FrameTask) -> tuple[RegistrationResult, FrameFeatures, dict[str, Any]]:
    """Per-frame part of a group analysis, importable so a child process can run it."""

    registration = (
        _identity_registration(task.frame)
        if task.is_reference
        else task.registration or _registration(task.frame, task.reference, task.config)
    )
    if registration.estimated:
        features, grid = _spatial_features(task.frame, task.reference, registration, task.config)
    else:
        features, grid = FrameFeatures(), {}
    return registration, features, grid


def _analyze_group(
    group_id: str,
    frames: list[FrameMeasurement],
    config: QcConfig,
    runner: FrameRunner | None = None,
) -> tuple[dict[str, Any], list[FrameResult]]:
    reference, cached_registrations = _supported_reference(frames, config)
    tasks = [
        _FrameTask(
            frame=frame,
            reference=reference,
            is_reference=frame is reference,
            registration=None if frame is reference else cached_registrations.get(frame.metadata.path),
            config=config,
        )
        for frame in frames
    ]
    # Frames are independent once the reference is fixed; the group steps
    # below (morphology, consensus, scores) see them in their original order.
    analyses = runner.map(_analyze_frame, tasks) if runner is not None else [_analyze_frame(task) for task in tasks]
    results: list[FrameResult] = []
    ordered_measurements: list[FrameMeasurement] = []
    for frame, (registration, features, grid) in zip(frames, analyses, strict=True):
        confidence = _confidence(len(frames), registration, config)
        warnings: list[str] = []
        if len(frames) < config.minimum_group_frames:
            warnings.append("INSUFFICIENT_GROUP_FRAMES")
        if not registration.accepted:
            warnings.extend(registration.reason_codes)
        features.detected_source_ratio = _detected_ratio(frame, reference)
        if (
            features.overlap_fraction is not None
            and features.overlap_fraction < config.minimum_overlap_fraction
        ):
            confidence = Confidence.LOW
            warnings.append("LOW_COMMON_FOOTPRINT")
        results.append(
            FrameResult(
                path=frame.metadata.path,
                group_id=group_id,
                reference_path=reference.metadata.path,
                decision=Decision.UNASSESSABLE,
                confidence=confidence,
                reasons=[],
                warnings=sorted(set(warnings)),
                registration=_registration_metrics(registration),
                features=features,
                metadata=frame.metadata,
                star_count=_detected_count(frame),
                thumbnail_path=frame.thumbnail_path,
                grid=grid,
                identity=frame.identity,
            )
        )
        ordered_measurements.append(frame)
    _populate_morphology(results, ordered_measurements)
    _apply_spatial_consensus(results)
    _apply_scores(results, ordered_measurements, config)
    summary = {
        "groupId": group_id,
        "frameCount": len(frames),
        "referencePath": reference.metadata.path,
        "filter": reference.metadata.filter_name,
        "camera": reference.metadata.camera,
        "target": reference.metadata.target,
        "exposureSeconds": reference.metadata.exposure_seconds,
        "decisions": {
            decision.value: sum(item.decision == decision for item in results)
            for decision in Decision
        },
    }
    return summary, results


def analyze_measurements(
    measurements: Iterable[FrameMeasurement], config: QcConfig, *,
    cache_directory: Path | str | None = None,
    cache_stats: dict[str, int] | None = None,
    workers: int = 1,
    stats: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[FrameResult]]:
    """Group the measurements and analyze every group against its reference.

    ``workers`` bounds the per-frame registration/feature work (see
    :mod:`lightframeqc.parallel`); ``stats`` receives what ran.
    """

    config.validate()
    frames = list(measurements)
    summaries: list[dict[str, Any]] = []
    results: list[FrameResult] = []
    cache = GroupAnalysisCache(cache_directory, config, cache_stats) if cache_directory is not None else None

    measured_paths: set[str] = set()
    with FrameRunner(workers, len(frames)) as runner:
        for base_id, base_frames in build_groups(frames, config):
            cache_key = cache.key(base_id, base_frames) if cache is not None else None
            cached = cache.load(cache_key, base_frames) if cache is not None else None
            if cached is None:
                base_summaries: list[dict[str, Any]] = []
                base_results: list[FrameResult] = []
                for group_id, group_frames in _split_auto_fields(base_id, base_frames, config):
                    summary, group_results = _analyze_group(group_id, group_frames, config, runner)
                    base_summaries.append(summary)
                    base_results.extend(group_results)
                if cache is not None:
                    cache.store(cache_key, base_summaries, base_results)
            else:
                base_summaries, base_results = cached
            summaries.extend(base_summaries)
            results.extend(base_results)
            measured_paths.update(item.metadata.path for item in base_frames)
        if stats is not None:
            stats.update(runner.stats)

    for frame in frames:
        if frame.metadata.path in measured_paths:
            continue
        if frame.error_code:
            code = frame.error_code
        elif frame.metadata.role is FrameRole.UNKNOWN:
            code = "UNKNOWN_FRAME_ROLE"
        elif frame.metadata.role is not FrameRole.LIGHT:
            code = "NOT_A_LIGHT_FRAME_" + frame.metadata.role.value
        else:
            code = "MEASUREMENT_FAILED"
        results.append(
            FrameResult(
                path=frame.metadata.path,
                group_id="unassessable",
                reference_path=None,
                decision=Decision.UNASSESSABLE,
                confidence=Confidence.NONE,
                reasons=[code],
                warnings=[frame.error_message] if frame.error_message else [],
                registration=RegistrationMetrics(error=code),
                features=FrameFeatures(),
                metadata=frame.metadata,
                star_count=_detected_count(frame),
                thumbnail_path=frame.thumbnail_path,
                identity=frame.identity,
            )
        )
    results.sort(key=lambda item: item.path)
    summaries.sort(key=lambda item: item["groupId"])
    return summaries, results
