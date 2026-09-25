from __future__ import annotations

import itertools

import numpy as np
import pytest

from ufwbpp.stacking.crop import histogram_rectangle


def _exhaustive_rectangle(heights: np.ndarray, row: int):
    """Independent interval oracle, including the existing tuple tie-break."""
    candidates = []
    for left in range(len(heights)):
        for right in range(left + 1, len(heights) + 1):
            height = int(np.min(heights[left:right]))
            if height:
                candidates.append(
                    (height * (right - left), row - height + 1, left, row + 1, right)
                )
    return max(candidates, default=None)


@pytest.mark.parametrize("width", range(7))
def test_histogram_matches_every_small_mask_height(width: int) -> None:
    # Covers zero runs, repeated plateaus, holes, and tied maximum rectangles.
    for values in itertools.product(range(3), repeat=width):
        heights = np.asarray(values, dtype=np.int64)
        assert histogram_rectangle(heights, 5) == _exhaustive_rectangle(heights, 5)


def test_histogram_random_run_lengths_and_strided_input() -> None:
    rng = np.random.default_rng(44281)
    for _ in range(150):
        values = np.repeat(rng.integers(0, 101, 8), rng.integers(1, 8, 8))
        backing = np.empty(len(values) * 2, dtype=np.int64)
        backing[::2] = values
        heights = backing[::2]
        assert histogram_rectangle(heights, 200) == _exhaustive_rectangle(heights, 200)


def test_crop_mask_histogram_matches_exhaustive_pixel_rectangle() -> None:
    rng = np.random.default_rng(739)
    for _ in range(100):
        mask = rng.random((5, 7)) > 0.2
        heights = np.zeros(mask.shape[1], dtype=np.int64)
        best = None
        for row_index, row in enumerate(mask):
            heights = np.where(row, heights + 1, 0)
            candidate = histogram_rectangle(heights, row_index)
            if candidate is not None and (best is None or candidate > best):
                best = candidate
        candidates = []
        for top in range(mask.shape[0]):
            for bottom in range(top + 1, mask.shape[0] + 1):
                for left in range(mask.shape[1]):
                    for right in range(left + 1, mask.shape[1] + 1):
                        if mask[top:bottom, left:right].all():
                            candidates.append(
                                ((bottom - top) * (right - left), top, left, bottom, right)
                            )
        assert best == max(candidates, default=None)
