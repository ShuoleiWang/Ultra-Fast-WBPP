from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import theilslopes


def finite_array(values: Iterable[float | None]) -> np.ndarray:
    return np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(value)],
        dtype=np.float64,
    )


def median(values: Iterable[float | None]) -> float | None:
    array = finite_array(values)
    return float(np.median(array)) if array.size else None


def mad(values: Iterable[float | None], center: float | None = None) -> float | None:
    array = finite_array(values)
    if not array.size:
        return None
    location = float(np.median(array)) if center is None else center
    return float(np.median(np.abs(array - location)))


def percentile(values: Iterable[float | None], q: float) -> float | None:
    array = finite_array(values)
    return float(np.percentile(array, q)) if array.size else None


def robust_z(
    value: float | None,
    values: Iterable[float | None],
    *,
    scale_floor: float = 1e-6,
) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    location = median(values)
    dispersion = mad(values, location)
    if location is None or dispersion is None:
        return None
    scale = max(1.4826 * dispersion, scale_floor)
    return (value - location) / scale


def fit_clear_airmass_envelope(
    airmass: list[float | None], extinction_mag: list[float | None]
) -> list[float | None]:
    valid = [
        (index, float(x), float(y))
        for index, (x, y) in enumerate(zip(airmass, extinction_mag, strict=True))
        if x is not None
        and y is not None
        and math.isfinite(x)
        and math.isfinite(y)
        and 0.9 <= x <= 10.0
    ]
    result: list[float | None] = [None] * len(airmass)
    if len(valid) < 4:
        return result

    x_values = np.asarray([x for _, x, _ in valid], dtype=np.float64)
    y_values = np.asarray([y for _, _, y in valid], dtype=np.float64)
    if float(np.ptp(x_values)) < 0.15:
        clear_zero = float(np.percentile(y_values, 5))
        for index, _, y in valid:
            result[index] = y - clear_zero
        return result

    slope, intercept, _, _ = theilslopes(y_values, x_values)
    slope = float(np.clip(slope, -0.05, 3.0))
    residuals = y_values - (slope * x_values + intercept)
    # A low clear envelope remains anchored when clouds are the majority.  The
    # high automatic thresholds and independent-family rule absorb ordinary
    # photometric scatter without learning a cloudy median as "normal".
    clear_zero = float(np.percentile(residuals, 5))
    for index, x, y in valid:
        result[index] = y - (slope * x + intercept + clear_zero)
    return result


def safe_log_extinction(ratio: float | None) -> float | None:
    if ratio is None or not math.isfinite(ratio) or ratio <= 0:
        return None
    return -2.5 * math.log10(ratio)
