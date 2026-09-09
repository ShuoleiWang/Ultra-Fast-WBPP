"""Conservative catalog evidence for stars split into tracking-trail fragments.

Ordinary SEP shape statistics can miss a trail deblended into several nearly
round sources.  This check keeps the blend flag and looks for *many* elongated
groups with a shared direction across the image.  It is evidence of a defect,
not proof of acceptable morphology when the result is negative.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .models import Star


@dataclass(frozen=True, slots=True)
class FragmentedTrailMetrics:
    available: bool = False
    considered_stars: int = 0
    candidate_chain_count: int = 0
    chain_count: int = 0
    consensus_fraction: float = 0.0
    fragment_count: int = 0
    fragment_fraction: float = 0.0
    orientation_coherence: float | None = None
    occupied_cells: int = 0
    spatial_minor_fraction: float = 0.0
    link_radius_pixels: float | None = None
    bright_blend_fraction: float | None = None
    detected: bool = False


def measure_fragmented_trails(
    stars: Sequence[Star], preview_width: int, preview_height: int
) -> FragmentedTrailMetrics:
    """Find distributed coherent chains without rereading image pixels.

    Work is bounded to the brightest 500 finite, positive, in-frame detections.
    SEP's blended bit is retained; detections with other extraction flags are
    omitted. A candidate chain needs at least three detections, at least half
    blended, an axial ratio of four, and a length of ten PSF minor-axis widths.
    Six chains affecting at least 10% of the bright catalog must form a majority
    direction consensus and occupy a two-dimensional footprint before
    ``detected`` is set. A single satellite, close doubles, and isolated galaxy
    knots therefore do not supply sufficient evidence.

    The fixed bounds are intentionally conservative evidence requirements;
    ``detected=False`` must never be used as a standalone admission criterion.
    """
    if (
        not math.isfinite(preview_width)
        or not math.isfinite(preview_height)
        or preview_width <= 0
        or preview_height <= 0
    ):
        return FragmentedTrailMetrics()

    usable = [
        star
        for star in stars
        if all(math.isfinite(value) for value in (star.x, star.y, star.flux, star.a, star.b))
        and 0 <= star.x < preview_width
        and 0 <= star.y < preview_height
        and star.flux > 0
        and star.a > 0
        and star.b > 0
        and star.flags & ~1 == 0
    ]
    usable.sort(key=lambda star: (-star.flux, star.y, star.x))
    usable = usable[:500]
    count = len(usable)
    if count < 30:
        return FragmentedTrailMetrics(considered_stars=count)

    points = np.asarray([(star.x, star.y) for star in usable], dtype=np.float64)
    blended = np.asarray([bool(star.flags & 1) for star in usable])
    # The faint catalog can be dominated by one-pixel residuals. Use bright
    # detections, including blended fragments, to estimate the transverse PSF.
    minor_width = float(np.median([min(star.a, star.b) for star in usable[:100]]))
    radius = min(10.0 * minor_width, 0.025 * min(preview_width, preview_height))
    bright_blend_fraction = float(np.mean(blended[:100]))

    parents = np.arange(count)

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = int(parents[index])
        return index

    for first, second in cKDTree(points).query_pairs(radius, output_type="ndarray"):
        parents[root(int(second))] = root(int(first))
    components: dict[int, list[int]] = {}
    for index in range(count):
        components.setdefault(root(index), []).append(index)

    centers: list[np.ndarray] = []
    angles: list[float] = []
    sizes: list[int] = []
    for indices in components.values():
        if len(indices) < 3 or float(np.mean(blended[indices])) < 0.5:
            continue
        group = points[indices]
        center = np.mean(group, axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(group.T))
        axis = eigenvectors[:, -1]
        span = float(np.ptp((group - center) @ axis))
        # A PSF-width floor prevents duplicate or almost coincident centroids
        # from producing an arbitrarily large aspect ratio.
        axial_ratio = math.sqrt(
            max(0.0, float(eigenvalues[-1]))
            / max(float(eigenvalues[0]), minor_width * minor_width)
        )
        if axial_ratio < 4.0 or span < 10.0 * minor_width:
            continue
        centers.append(center)
        angles.append(math.atan2(float(axis[1]), float(axis[0])))
        sizes.append(len(indices))

    candidate_count = len(centers)
    consensus: list[int] = []
    if angles:
        # Curved tracks can produce a second, shorter family of fragments.
        # Select the largest 30-degree axial-angle window, then require it to
        # contain a majority of all candidate chains. This avoids cancellation
        # by other shapes without accepting a small chance orientation cluster.
        axial_angles = np.mod(angles, math.pi)
        for start in axial_angles:
            indices = np.flatnonzero(
                np.mod(axial_angles - start, math.pi) <= math.pi / 6 + 1e-12
            ).tolist()
            if (len(indices), sum(sizes[index] for index in indices)) > (
                len(consensus), sum(sizes[index] for index in consensus)
            ):
                consensus = indices

    fragment_count = sum(sizes[index] for index in consensus)
    consensus_fraction = len(consensus) / candidate_count if candidate_count else 0.0

    coherence: float | None = None
    occupied_cells = 0
    spatial_minor_fraction = 0.0
    if consensus:
        # Equal weight per independent chain prevents one long satellite from
        # dominating the orientation statistic.
        coherence = min(
            1.0,
            float(abs(np.mean(np.exp(2j * np.asarray(angles)[consensus])))),
        )
        normalized_centers = np.asarray(centers)[consensus] / (preview_width, preview_height)
        cells = np.floor(normalized_centers * 4).astype(np.int64)
        occupied_cells = len({tuple(cell) for cell in cells})
        if len(consensus) > 1:
            spatial_minor_fraction = math.sqrt(
                max(0.0, float(np.linalg.eigvalsh(np.cov(normalized_centers.T))[0]))
            )

    fraction = fragment_count / count
    detected = (
        len(consensus) >= 6
        and consensus_fraction > 0.5
        and fraction >= 0.10
        and coherence is not None
        and coherence >= 0.85
        and occupied_cells >= 6
        and spatial_minor_fraction >= 0.10
    )
    return FragmentedTrailMetrics(
        available=True,
        considered_stars=count,
        candidate_chain_count=candidate_count,
        chain_count=len(consensus),
        consensus_fraction=consensus_fraction,
        fragment_count=fragment_count,
        fragment_fraction=fraction,
        orientation_coherence=coherence,
        occupied_cells=occupied_cells,
        spatial_minor_fraction=spatial_minor_fraction,
        link_radius_pixels=radius,
        bright_blend_fraction=bright_blend_fraction,
        detected=detected,
    )
