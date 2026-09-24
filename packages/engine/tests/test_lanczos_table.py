"""The deterministic Lanczos-3 weight table: the native and Python tables are
identical, the interpolation reproduces the exact weights far below Float32
resolution, and the special fractions are exact."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ufwbpp import lanczos_table as lt
from ufwbpp.native_kernels import load_native_kernels

KERNELS = load_native_kernels()


def test_deterministic_sinpi_matches_libm_and_is_exact_at_integers() -> None:
    values = np.linspace(-4.0, 4.0, 200_001)
    ours = lt.deterministic_sinpi(values)
    reference = np.array([math.sin(math.pi * float(v)) for v in values])
    assert np.max(np.abs(ours - reference)) < 4e-15
    integers = np.arange(-6, 7, dtype=np.float64)
    assert np.all(lt.deterministic_sinpi(integers) == 0.0)
    assert abs(lt.deterministic_sinpi(np.array([0.5]))[0] - 1.0) < 4e-16
    assert abs(lt.deterministic_sinpi(np.array([-0.5]))[0] + 1.0) < 4e-16


def test_table_interpolation_is_far_below_float32_resolution() -> None:
    rng = np.random.default_rng(7)
    fractions = np.concatenate(
        [rng.uniform(0.0, 1.0, 20_000), [0.0, 0.5, 1.0 / 2048, 3.0 / 4096, 1e-9, 1.0 - 1e-9, 1.0 - 2.0**-30]]
    )
    interpolated = lt.tap_weights_float64(fractions)
    exact = np.array([lt.exact_tap_weights(float(f)) for f in fractions])
    assert np.max(np.abs(interpolated - exact)) < 1e-12
    weights = np.stack(lt.tap_weights(fractions))
    assert weights.dtype == np.float32 and weights.shape == (6, fractions.size)
    # Float32 weights differ from the Float32 of the exact weights only
    # where the exact value sits next to a rounding boundary: rarely, and
    # never by more than one unit in the last place.
    exact32 = exact.T.astype(np.float32)
    differing = weights != exact32
    assert differing.mean() < 2e-3
    difference = np.abs(weights.astype(np.float64) - exact32.astype(np.float64))
    one_ulp = np.spacing(np.abs(exact32)).astype(np.float64)
    assert np.all(difference <= one_ulp + 1e-12)
    sums = weights.astype(np.float64).sum(axis=0)
    assert np.max(np.abs(sums - 1.0)) < 4e-7


def test_special_fractions_are_exact() -> None:
    zero = np.stack(lt.tap_weights(np.array([0.0])))[:, 0]
    assert zero.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    half = np.stack(lt.tap_weights(np.array([0.5])))[:, 0]
    assert half[2] == half[3] and half[1] == half[4] and half[0] == half[5]
    assert half[2] > 0.6 and half[1] < 0.0 and half[0] > 0.0
    table = lt.node_table()
    assert table.shape == (lt.TABLE_NODES, 6) and not table.flags.writeable
    assert table[1].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # f = 0
    assert table[lt.TABLE_INTERVALS + 1].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # f = 1


@pytest.mark.skipif(KERNELS is None, reason="native kernel library not available")
def test_native_table_is_identical_to_the_python_table() -> None:
    assert KERNELS is not None
    native = KERNELS.lanczos3_table()
    assert native.shape == lt.node_table().shape
    assert np.array_equal(native, lt.node_table())
