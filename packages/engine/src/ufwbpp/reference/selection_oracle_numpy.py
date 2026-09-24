"""Naive leave-one-out oracle for testing ``LeaveOneOutAccumulator``.

Loops over tiles, subsets and blocks with plain NumPy; no vectorisation over
frames.  Only suitable for small synthetic stacks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def block_mean_image(
    samples: NDArray[np.float32],
    accepted: NDArray[np.bool_],
    weights: NDArray[np.float64],
    block: int,
) -> NDArray[np.float64]:
    """Block-level weighted mean over the accepted samples of all frames."""

    frames, rows, width = samples.shape
    rows_b, cols_b = rows // block, width // block
    result = np.full((rows_b, cols_b), np.nan)
    for r in range(rows_b):
        for c in range(cols_b):
            numerator = 0.0
            denominator = 0.0
            for f in range(frames):
                patch = samples[f, r * block : (r + 1) * block, c * block : (c + 1) * block]
                mask = accepted[f, r * block : (r + 1) * block, c * block : (c + 1) * block]
                numerator += weights[f] * float(patch[mask].sum(dtype=np.float64))
                denominator += weights[f] * float(mask.sum())
            if denominator > 0:
                result[r, c] = numerator / denominator
    return result


def tile_depth_madn(
    means: NDArray[np.float64], usable: NDArray[np.bool_]
) -> float:
    rows_b, cols_b = means.shape
    r, c = np.nonzero(usable)
    design = np.column_stack(
        (np.ones(r.size), c / max(1, cols_b - 1), r / max(1, rows_b - 1))
    )
    values = means[usable]
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))))


def usable_blocks(
    integrated: NDArray[np.float32], block: int, star_sigma: float
) -> NDArray[np.bool_]:
    rows_b, cols_b = integrated.shape[0] // block, integrated.shape[1] // block
    finite = np.isfinite(integrated)
    values = integrated[finite]
    median = float(np.median(values))
    spread = float(1.4826 * np.median(np.abs(values - median)))
    threshold = median + star_sigma * max(spread, 1e-12)
    usable = np.zeros((rows_b, cols_b), dtype=bool)
    for r in range(rows_b):
        for c in range(cols_b):
            patch = integrated[r * block : (r + 1) * block, c * block : (c + 1) * block]
            usable[r, c] = bool(np.all(np.isfinite(patch)) and patch.max() <= threshold)
    return usable


def leave_one_out_depths(
    samples: NDArray[np.float32],
    accepted: NDArray[np.bool_],
    weights: NDArray[np.float64],
    integrated: NDArray[np.float32],
    *,
    block: int = 8,
    star_sigma: float = 5.0,
) -> NDArray[np.float64]:
    """Depth MADN of the full stack (index 0) and of every leave-one-out stack."""

    frames = samples.shape[0]
    usable = usable_blocks(integrated, block, star_sigma)
    depths = np.empty(frames + 1)
    full = block_mean_image(samples, accepted, weights, block)
    usable &= np.isfinite(full)
    depths[0] = tile_depth_madn(full, usable)
    for f in range(frames):
        keep = np.ones(frames, dtype=bool)
        keep[f] = False
        means = block_mean_image(samples[keep], accepted[keep], weights[keep], block)
        depths[f + 1] = tile_depth_madn(means, usable & np.isfinite(means))
    return depths
