"""Deterministic Lanczos-3 tap weights: the reference of the native table.

The six normalized tap weights of a sub-pixel fraction ``f`` in ``[0, 1)``
are read from a table of ``TABLE_INTERVALS + 3`` nodes at ``f = i/N``
(``i = -1 .. N+1``) with cubic Lagrange interpolation.  The node values
come from their own Taylor series in a fixed Float64 operation order, never
from a math library, so every platform and library version builds the same
table; ``native/src/Lanczos3Table.cpp`` performs the same operations
in the same order and the two agree value for value (see the identity test).

Why a table: the previous evaluation spent six exact transcendental calls
and 48 divisions per output pixel and depended on the host's libm (a macOS
upgrade changed the master hashes once); the table's interpolation error is
below 1e-12, four orders of magnitude under the Float32 resolution of the
weights, so a registered pixel differs from the libm-exact value by at most
a unit in the last place of Float32 for a small fraction of pixels.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from numpy.typing import NDArray

TABLE_INTERVALS = 2048
TABLE_NODES = TABLE_INTERVALS + 3
SERIES_TERMS = 12
TAP_OFFSETS = (-2, -1, 0, 1, 2, 3)
PI = 3.141592653589793


def deterministic_sinpi(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """``sin(pi * v)`` by reduction to ``|r| <= 0.5`` and a nested Taylor
    series, elementwise, in the operation order of the native kernel."""

    v = np.asarray(values, dtype=np.float64)
    n = np.floor(v + 0.5)
    r = v - n
    x = PI * r
    x2 = x * x
    term = np.ones_like(x)
    for k in range(SERIES_TERMS, 0, -1):
        denominator = float((2 * k) * (2 * k + 1))
        term = 1.0 - x2 * term / denominator
    s = x * term
    odd = (n.astype(np.int64) & 1) != 0
    return np.where(odd, -s, s)


def _raw_weights(distance: NDArray[np.float64]) -> NDArray[np.float64]:
    """``sinc(d) * sinc(d/3)`` for signed tap distances.

    The table's two nodes outside ``[0, 1]`` use the smooth analytic
    continuation instead of the ``|d| >= 3`` cut-off (``sinc(3)`` is exactly
    zero anyway), so the interpolation sees a smooth function right up to
    the edges of the fraction range.
    """

    d = np.asarray(distance, dtype=np.float64)
    result = np.zeros(d.shape, dtype=np.float64)
    inside = d != 0.0
    result[d == 0.0] = 1.0
    if np.any(inside):
        di = d[inside]
        primary = deterministic_sinpi(di) / (PI * di)
        reduced = di / 3.0
        secondary = deterministic_sinpi(reduced) / (PI * reduced)
        result[inside] = primary * secondary
    return result


@lru_cache(maxsize=1)
def node_table() -> NDArray[np.float64]:
    """``(TABLE_NODES, 6)`` normalized weights; row ``i + 1`` is ``f = i/N``."""

    nodes = (np.arange(TABLE_NODES, dtype=np.float64) - 1.0) / float(TABLE_INTERVALS)
    raw = np.empty((TABLE_NODES, 6), dtype=np.float64)
    for tap, offset in enumerate(TAP_OFFSETS):
        raw[:, tap] = _raw_weights(nodes - float(offset))
    total = np.zeros(TABLE_NODES, dtype=np.float64)
    for tap in range(6):
        total = total + raw[:, tap]
    table = raw / total[:, None]
    table.setflags(write=False)
    return table


def tap_weights_float64(fractions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalized Float64 weights, ``(n, 6)``, before the Float32 rounding."""

    fraction = np.asarray(fractions, dtype=np.float64)
    table = node_table()
    u = fraction * float(TABLE_INTERVALS)
    floor_u = np.floor(u)
    i = floor_u.astype(np.int64)
    t = u - floor_u
    # Cubic Lagrange basis through the four nodes at t = -1, 0, 1, 2.
    a = t + 1.0
    b = t - 1.0
    c = t - 2.0
    b0 = -((t * b) * c) / 6.0
    b1 = ((a * b) * c) / 2.0
    b2 = -((a * t) * c) / 2.0
    b3 = ((a * t) * b) / 6.0
    p0 = table[i]
    p1 = table[i + 1]
    p2 = table[i + 2]
    p3 = table[i + 3]
    values = ((b0[:, None] * p0 + b1[:, None] * p1) + b2[:, None] * p2) + b3[:, None] * p3
    total = np.zeros(fraction.shape, dtype=np.float64)
    for tap in range(6):
        total = total + values[:, tap]
    return values / total[:, None]


def tap_weights(fractions: NDArray[np.float64]) -> tuple[NDArray[np.float32], ...]:
    """Normalized Float32 weights of the six taps for fractions in ``[0, 1)``."""

    values = tap_weights_float64(fractions)
    return tuple(np.asarray(values[:, tap], dtype=np.float32) for tap in range(6))


def exact_tap_weights(fraction: float) -> tuple[float, ...]:
    """The libm-based weights (for tests of the table's accuracy only)."""

    total = 0.0
    raw = []
    for offset in TAP_OFFSETS:
        d = fraction - offset
        if abs(d) >= 3.0:
            weight = 0.0
        elif d == 0.0:
            weight = 1.0
        else:
            weight = (math.sin(PI * d) / (PI * d)) * (math.sin(PI * d / 3.0) / (PI * d / 3.0))
        raw.append(weight)
        total += weight
    return tuple(weight / total for weight in raw)


__all__ = [
    "PI",
    "SERIES_TERMS",
    "TABLE_INTERVALS",
    "TABLE_NODES",
    "TAP_OFFSETS",
    "deterministic_sinpi",
    "exact_tap_weights",
    "node_table",
    "tap_weights",
    "tap_weights_float64",
]
