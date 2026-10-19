"""Small NumPy oracle for square drops through a forward canvas lattice.

It visits every source pixel and output pixel without native row culling.
The source's finite values are unit-weight samples; the result is the
Float64 signal sum and mapped-area weight on the output grid.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _clip(vertices: NDArray[np.float64], axis: int, limit: float, keep_above: bool) -> NDArray[np.float64]:
    result = []
    for current, previous in zip(vertices, np.roll(vertices, 1, axis=0), strict=True):
        inside = (current[axis] >= limit) if keep_above else (current[axis] <= limit)
        was_inside = (previous[axis] >= limit) if keep_above else (previous[axis] <= limit)
        if inside != was_inside:
            fraction = (limit - previous[axis]) / (current[axis] - previous[axis])
            result.append(previous + fraction * (current - previous))
        if inside:
            result.append(current)
    return np.asarray(result, dtype=np.float64).reshape(-1, 2)


def square_lattice_drizzle(
    source: NDArray[np.float32],
    registration: NDArray[np.float64],
    lattice_x: NDArray[np.float64],
    lattice_y: NDArray[np.float64],
    origin: tuple[float, float],
    spacing: float,
    output_shape: tuple[int, int],
    *,
    pixfrac: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map each drop's four corners, then integrate its exact polygon area
    over output-pixel squares. Lattice nodes hold output coordinates.
    """

    total, weight = (np.zeros(output_shape, dtype=np.float64) for _ in range(2))
    corners = 0.5 * pixfrac * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    for row, column in np.ndindex(source.shape):
        value = float(source[row, column])
        if not np.isfinite(value):
            continue
        points = np.c_[corners + (column, row), np.ones(4)] @ registration.T
        points = points[:, :2] / points[:, 2:3]
        position = (points - origin) / spacing
        cell = np.floor(position).astype(np.int64)
        cell[:, 0] = np.clip(cell[:, 0], 0, lattice_x.shape[1] - 2)
        cell[:, 1] = np.clip(cell[:, 1], 0, lattice_x.shape[0] - 2)
        fraction = position - cell
        x, y = cell.T
        fx, fy = fraction.T
        mapped = []
        for nodes in (lattice_x, lattice_y):
            top = nodes[y, x] + fx * (nodes[y, x + 1] - nodes[y, x])
            bottom = nodes[y + 1, x] + fx * (nodes[y + 1, x + 1] - nodes[y + 1, x])
            mapped.append(top + fy * (bottom - top))
        polygon = np.column_stack(mapped)
        for oy, ox in np.ndindex(output_shape):
            clipped = polygon
            for axis, limit, above in ((0, ox - 0.5, True), (0, ox + 0.5, False), (1, oy - 0.5, True), (1, oy + 0.5, False)):
                clipped = _clip(clipped, axis, limit, above)
                if len(clipped) < 3:
                    break
            if len(clipped) >= 3:
                q = np.roll(clipped, -1, axis=0)
                area = 0.5 * abs(float(np.sum(clipped[:, 0] * q[:, 1] - q[:, 0] * clipped[:, 1])))
                total[oy, ox] += value * area
                weight[oy, ox] += area
    return total, weight
