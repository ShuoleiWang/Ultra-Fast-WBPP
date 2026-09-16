"""Exact vectorized replacements for per-slice NumPy robust statistics.

These helpers reproduce the arithmetic of the NumPy reference calls they
replace (``np.nanmedian`` over one axis, ``np.median`` of a compacted finite
subset) so callers can evaluate many slices at once without changing a single
value. They contain no scientific parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def nanmedian_frames(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """``np.nanmedian(values, axis=0)`` for a Float32 (frames, columns) array.

    NumPy's masked-array path sorts each column, takes the middle finite
    sample (odd counts) or the Float32 mean of the two middle samples (even
    counts); this reproduces that arithmetic with one sort and two gathers, so
    the result is value-identical while avoiding the per-column overhead of
    the masked implementation. Columns without finite samples are NaN. The
    same arithmetic yields ``np.median`` of each column's finite samples, so a
    NaN-padded column equals the median of its compacted finite values.
    """

    if values.ndim != 2 or values.dtype != np.float32:
        raise ValueError("expected a Float32 (frames, columns) array")
    frames, columns = values.shape
    if frames == 0 or columns == 0:
        return np.full(columns, np.nan, dtype=np.float32)
    ordered = np.sort(values, axis=0)  # NaN sorts last
    finite_count = frames - np.count_nonzero(np.isnan(values), axis=0)
    high = finite_count // 2
    low = np.where(finite_count % 2 == 1, high, high - 1)
    empty = finite_count == 0
    high = np.where(empty, 0, high)
    low = np.where(empty, 0, low)
    column_index = np.arange(columns)
    pair = np.stack((ordered[low, column_index], ordered[high, column_index]))
    result = np.mean(pair, axis=0)
    result[empty] = np.nan
    return np.asarray(result, dtype=np.float32)


def nanmedian_rows(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Median of each row's finite samples of a Float32 (rows, samples) array.

    Equals ``np.median(row[np.isfinite(row)])`` per row (NaN for rows without
    finite samples); ``inf`` samples must already be masked to NaN.
    """

    return nanmedian_frames(np.ascontiguousarray(values.T))
