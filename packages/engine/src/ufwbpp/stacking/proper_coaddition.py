"""Proper (ZOGY 2017) coaddition of one registered, normalized Light group.

This is an **additional** product.  The ordinary rejection/integration master
stays the run's primary master and is untouched; ``proper_coadd_group`` reads
exactly the samples the ordinary integration consumed, reuses its per-pixel
rejection decisions and writes one more linear image beside it.

Model (Zackay & Ofek 2017, "Proper image subtraction and coaddition", eq. 8-9)
for frames ``M_j = F_j * T (x) P_j + eps_j`` with background noise ``sigma_j``::

    R_hat = sum_j (F_j / sigma_j^2) conj(P_hat_j) M_hat_j
            / sqrt( sum_j (F_j^2 / sigma_j^2) |P_hat_j|^2 )
    P_hat_R = sqrt( sum_j (F_j^2 / sigma_j^2) |P_hat_j|^2 ) / F_R
    F_R     = sqrt( sum_j F_j^2 / sigma_j^2 )

``R`` as written by the reference has unit noise variance; the published
product is ``R / F_R + sky``, which is ``T (x) P_R`` plus white noise of
sigma ``1 / F_R`` -- that is, the same photometric units as the normalized
frames and the ordinary master, with the ideal stacked background noise.  The
matched-filter score image ``S`` is deliberately **not** published: it is a
detection statistic, not an image.

Everything here runs only when the recipe opts in; nothing in this module is
reachable from a default run.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..image_io.fits import (
    CalibrationError,
    FitsFrame,
    FitsFloatWriter,
    atomic_publish_file,
    temporary_output,
)
from ..platform import remove_file
from .integration import (
    FrameExpression,
    _canonical_expression,
    _expression_rows,
    _open_expression_sources,
)

# Algorithm identity.  A receipt written by a different version of the maths
# is distinguishable from this one by this string alone.
PROPER_COADD_ALGORITHM_ID = "zogy-proper-coadd-v1"

# Stamp geometry of the empirical PSF: 31x31 around each star.
PSF_STAMP_RADIUS = 15
# Peak detection threshold above the background sigma, and the star budget.
PSF_DETECTION_SIGMA = 30.0
PSF_MINIMUM_STARS = 8
PSF_MAXIMUM_STARS = 300
# The brightest few percent of detections are dropped: they are the ones most
# likely to be saturated or non-linear.
PSF_BRIGHT_DROP_FRACTION = 0.05
# Moffat beta of the analytic fallback PSF.
PSF_FALLBACK_MOFFAT_BETA = 2.5
# Frames read, sky-subtracted and PSF-measured ahead of the transform, each on
# its own thread.  The measurement dominates a frame's cost and depends on
# that frame alone; the transforms and the accumulation stay in frame order.
PREPARE_AHEAD = 3

OUTLIER_HANDLINGS = ("reuse-rejection", "none")


@dataclass(frozen=True, slots=True)
class ProperCoadditionParameters:
    """The opt-in proper-coaddition block, shaped like ``DrizzleOptions``."""

    enabled: bool = False
    outlier_handling: str = "reuse-rejection"
    apodization_pixels: int = 64

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("proper_coaddition.enabled must be a boolean")
        if self.outlier_handling not in OUTLIER_HANDLINGS:
            raise ValueError(
                "proper_coaddition.outlier_handling must be one of: "
                + ", ".join(OUTLIER_HANDLINGS)
            )
        if (
            isinstance(self.apodization_pixels, bool)
            or not isinstance(self.apodization_pixels, int)
            or not 0 <= self.apodization_pixels <= 512
        ):
            raise ValueError(
                "proper_coaddition.apodization_pixels must be an integer in [0, 512]"
            )

    def serializable(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "outlierHandling": self.outlier_handling,
            "apodizationPixels": self.apodization_pixels,
        }


@dataclass(frozen=True, slots=True)
class FramePsf:
    """One frame's PSF: how it was obtained and what it measures."""

    source: str
    star_count: int
    fwhm_pixels: float
    kernel: NDArray[np.float32] = field(repr=False)

    def serializable(self) -> dict[str, Any]:
        return {
            "psfSource": self.source,
            "psfStars": self.star_count,
            "psfFwhmPixels": self.fwhm_pixels,
        }


@dataclass(frozen=True, slots=True)
class ProperCoadditionResult:
    output_path: str
    shape: tuple[int, int]
    frame_count: int
    flux_scale_norm: float
    coadd_fwhm_pixels: float
    sky_added: float
    replaced_samples: int
    floored_spectrum_fraction: float
    frames: tuple[Mapping[str, Any], ...]
    timing_seconds: Mapping[str, float]
    working_set_bytes: int
    padded_shape: tuple[int, int]
    frames_prepared_ahead: int = 1
    output_sha256: str | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "algorithm": PROPER_COADD_ALGORITHM_ID,
            "outputPath": self.output_path,
            "shape": list(self.shape),
            "paddedShape": list(self.padded_shape),
            "frameCount": self.frame_count,
            "fluxScaleSumSqrt": self.flux_scale_norm,
            "coaddPsfFwhmPixels": self.coadd_fwhm_pixels,
            "skyAddedBack": self.sky_added,
            "replacedSamples": self.replaced_samples,
            "flooredSpectrumFraction": self.floored_spectrum_fraction,
            "outputSha256": self.output_sha256,
            "frames": [dict(item) for item in self.frames],
            "timingSeconds": {
                key: round(value, 3) for key, value in self.timing_seconds.items()
            },
            "workingSetBytes": self.working_set_bytes,
            "framesPreparedAhead": self.frames_prepared_ahead,
        }


def _scipy_fft() -> Any:
    try:
        import scipy.fft as scipy_fft
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise CalibrationError(
            "PROPER_COADD_UNAVAILABLE",
            "proper coaddition needs scipy.fft; install the engine's dependencies",
        ) from error
    return scipy_fft


def _ndimage() -> Any:
    try:
        import scipy.ndimage as scipy_ndimage
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise CalibrationError(
            "PROPER_COADD_UNAVAILABLE",
            "proper coaddition needs scipy.ndimage; install the engine's dependencies",
        ) from error
    return scipy_ndimage


def _padded_shape(shape: tuple[int, int], taper: int) -> tuple[int, int]:
    """A 5-smooth-ish FFT shape that holds the frame plus its guard band.

    The frame sits inside a border of ``taper`` pixels on every side, so the
    apodization lives entirely in that border and the frame's own pixels keep
    unit weight.  Rounding up to a fast transform length also keeps a large
    prime factor out of the transform (6252 = 2^2 * 3 * 521).
    """

    fft = _scipy_fft()
    return (
        int(fft.next_fast_len(shape[0] + 2 * taper)),
        int(fft.next_fast_len(shape[1] + 2 * taper)),
    )


def working_set_bytes(shape: tuple[int, int], taper: int = 64, prepared: int = 1) -> int:
    """Peak resident bytes of one group's coaddition, before the budget check.

    Accumulators (complex64 spectrum + float32 squared modulus), the master
    that replaces rejected samples, one frame, one frame spectrum, one PSF
    spectrum, and one transform workspace of a spectrum's size; ``prepared``
    frames in flight, each with its invalid mask and the star detection's
    temporaries (one of them is the frame counted above).
    """

    height, width = _padded_shape(shape, taper)
    spectrum = height * (width // 2 + 1) * 8
    real = height * width * 4
    master = shape[0] * shape[1] * 4
    # Accumulators (numerator, its accumulation temporary, |P|^2), the two
    # transforms in flight, the master, the frame, the transform grid and the
    # two full-frame temporaries the star detection allocates; plus, for every
    # further frame prepared ahead, the frame, its invalid mask and the
    # detection's two temporaries.
    ahead = max(0, int(prepared) - 1) * (3 * master + master // 4)
    return 4 * spectrum + spectrum // 2 + master + 5 * real + ahead


def _background_sigma(values: NDArray[np.float32], *, rows: int = 512) -> float:
    """Robust background sigma from adjacent-column differences.

    Differencing removes the sky gradient and the extended signal, so the MAD
    of the differences measures the pixel noise rather than the scene; the
    same estimator the ordinary integration's frame-noise weights use.  A
    uniform row subsample keeps the temporaries small on a 26 MP frame.
    """

    step = max(1, values.shape[0] // rows)
    sample = values[::step]
    finite = np.isfinite(sample)
    differences = np.diff(sample, axis=1)
    usable = finite[:, :-1] & finite[:, 1:]
    samples = differences[usable]
    if samples.size < 1024:
        return float("nan")
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    return float(1.4826 * mad / math.sqrt(2.0)) if mad > 0 else float("nan")


def _sky_level(values: NDArray[np.float32], *, rows: int = 512) -> float:
    """Median of a uniform row subsample: the frame's background level."""

    height = values.shape[0]
    step = max(1, height // rows)
    sample = values[::step]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


# Subpixel factor of the half-maximum area measurement below.  A 31x31 stamp
# counted at its own sampling quantizes a 2.6 px FWHM to steps of ~0.9 px.
FWHM_UPSAMPLE = 4


def _half_max_fwhm(kernel: NDArray[np.float32]) -> float:
    """FWHM of a PSF kernel from the area enclosed by its half-maximum level.

    Area based rather than profile based: it needs no fit and is insensitive
    to the exact centre.  The kernel is cubic-upsampled first so the pixel
    quantization of the enclosed area does not dominate a narrow PSF.
    """

    peak = float(np.max(kernel))
    if not math.isfinite(peak) or peak <= 0:
        return float("nan")
    fine = _ndimage().zoom(
        np.asarray(kernel, dtype=np.float64), FWHM_UPSAMPLE, order=3, grid_mode=False
    )
    peak = float(np.max(fine))
    if not math.isfinite(peak) or peak <= 0:
        return float("nan")
    area = int(np.count_nonzero(fine >= 0.5 * peak)) / float(FWHM_UPSAMPLE**2)
    if area <= 0:
        return float("nan")
    return float(2.0 * math.sqrt(area / math.pi))


def _moffat_kernel(fwhm: float, radius: int) -> NDArray[np.float32]:
    """Unit-sum circular Moffat of the given FWHM (beta fixed)."""

    beta = PSF_FALLBACK_MOFFAT_BETA
    alpha = fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    r2 = axis[:, None] ** 2 + axis[None, :] ** 2
    kernel = (1.0 + r2 / (alpha * alpha)) ** (-beta)
    total = float(np.sum(kernel))
    return np.asarray(kernel / total, dtype=np.float32)


def _candidate_peaks(
    values: NDArray[np.float32], sigma: float, radius: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]:
    """Isolated local maxima of a sky-subtracted frame, brightest first."""

    ndimage = _ndimage()
    height, width = values.shape
    margin = 2 * radius + 2
    if height <= 2 * margin or width <= 2 * margin:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=np.float32)
    interior = values[margin : height - margin, margin : width - margin]
    finite = np.nan_to_num(interior, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    threshold = np.float32(PSF_DETECTION_SIGMA * sigma)
    peaks = ndimage.maximum_filter(finite, size=5, mode="nearest")
    hits = (finite >= threshold) & (finite == peaks)
    rows, columns = np.nonzero(hits)
    if rows.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=np.float32)
    brightness = finite[rows, columns]
    # Deterministic order: brightest first, ties broken by position.
    order = np.lexsort((columns, rows, -brightness))
    return rows[order] + margin, columns[order] + margin, brightness[order]


def _psf_star_indices(
    rows: NDArray[np.int64], columns: NDArray[np.int64], radius: int
) -> NDArray[np.int64]:
    """The detections (brightest first) whose stamps are stacked.

    Isolation is judged against every detection, the brightest included, so a
    star beside a bright and possibly saturated one never enters the stack;
    only then are the brightest few percent dropped as likely non-linear.
    """

    dropped = (
        max(1, int(round(PSF_BRIGHT_DROP_FRACTION * rows.size))) if rows.size > 20 else 0
    )
    isolated = _isolated(rows, columns, radius, dropped + PSF_MAXIMUM_STARS)
    return isolated[isolated >= dropped][:PSF_MAXIMUM_STARS]


def _isolated(
    rows: NDArray[np.int64], columns: NDArray[np.int64], radius: int, limit: int
) -> NDArray[np.int64]:
    """Indices of peaks with no other candidate inside their stamp box."""

    cell = 2 * radius + 1
    buckets: dict[tuple[int, int], list[int]] = {}
    for index in range(rows.size):
        buckets.setdefault((int(rows[index]) // cell, int(columns[index]) // cell), []).append(index)
    selected: list[int] = []
    for index in range(rows.size):
        row, column = int(rows[index]), int(columns[index])
        crowded = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for other in buckets.get((row // cell + dy, column // cell + dx), ()):
                    if other == index:
                        continue
                    if (
                        abs(int(rows[other]) - row) <= cell
                        and abs(int(columns[other]) - column) <= cell
                    ):
                        crowded = True
                        break
                if crowded:
                    break
            if crowded:
                break
        if not crowded:
            selected.append(index)
            if len(selected) >= limit:
                break
    return np.asarray(selected, dtype=np.int64)


def _stack_star_stamps(
    values: NDArray[np.float32],
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    radius: int,
    invalid: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float32] | None, int]:
    """Sigma-clipped mean of recentred, unit-sum star stamps.

    ``invalid`` marks the pixels that held no data before they were filled
    for the transform; a stamp touching one is skipped.
    """

    ndimage = _ndimage()
    size = 2 * radius + 1
    axis = np.arange(size, dtype=np.float64) - radius
    stamps: list[NDArray[np.float32]] = []
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        stamp = np.asarray(
            values[row - radius : row + radius + 1, column - radius : column + radius + 1],
            dtype=np.float64,
        )
        if stamp.shape != (size, size) or not np.all(np.isfinite(stamp)):
            continue
        # Stamps that straddle the registered frame's invalid border (filled
        # for the transform) are skipped.  The mask says so exactly; testing
        # for exact zeros would also reject every stamp of quantized data,
        # where many sky-subtracted samples are exactly the median.
        if invalid is not None and bool(
            np.any(invalid[row - radius : row + radius + 1, column - radius : column + radius + 1])
        ):
            continue
        # Local background from the stamp's outer ring, so a residual gradient
        # does not leak into the wings.
        ring = np.concatenate(
            (stamp[0], stamp[-1], stamp[1:-1, 0], stamp[1:-1, -1])
        )
        stamp = stamp - float(np.median(ring))
        positive = np.clip(stamp, 0.0, None)
        total = float(np.sum(positive))
        if total <= 0:
            continue
        centre_y = float(np.sum(positive * axis[:, None]) / total)
        centre_x = float(np.sum(positive * axis[None, :]) / total)
        if abs(centre_y) > 2.0 or abs(centre_x) > 2.0:
            continue
        shifted = ndimage.shift(
            stamp, (-centre_y, -centre_x), order=3, mode="constant", cval=0.0
        )
        flux = float(np.sum(shifted))
        if not math.isfinite(flux) or flux <= 0:
            continue
        stamps.append(np.asarray(shifted / flux, dtype=np.float32))
    if len(stamps) < PSF_MINIMUM_STARS:
        return None, len(stamps)
    block = np.stack(stamps, axis=0).astype(np.float64)
    keep = np.ones(block.shape, dtype=bool)
    for _ in range(3):
        counts = np.count_nonzero(keep, axis=0)
        centre = np.where(
            counts > 0,
            np.median(np.where(keep, block, np.nan), axis=0),
            0.0,
        )
        deviation = np.abs(block - centre[None, :, :])
        scale = 1.4826 * np.median(np.where(keep, deviation, np.nan), axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.inf)
        keep = deviation <= 3.0 * scale[None, :, :]
    counts = np.count_nonzero(keep, axis=0)
    stacked = np.where(
        counts > 0,
        np.sum(np.where(keep, block, 0.0), axis=0) / np.maximum(counts, 1),
        0.0,
    )
    stacked = np.clip(stacked, 0.0, None)
    total = float(np.sum(stacked))
    if not math.isfinite(total) or total <= 0:
        return None, len(stamps)
    return np.asarray(stacked / total, dtype=np.float32), len(stamps)


def measure_frame_psf(
    values: NDArray[np.float32],
    sigma: float,
    *,
    radius: int = PSF_STAMP_RADIUS,
    fallback_fwhm: float | None = None,
    invalid: NDArray[np.bool_] | None = None,
) -> FramePsf:
    """Measure one frame's PSF by stacking its own stars; never assume one.

    ``values`` is the sky-subtracted registered frame and ``invalid`` the
    pixels that held no data before they were filled.  The analytic Moffat is
    used only when too few usable stars survive detection, isolation and
    recentring, and the receipt says so per frame.
    """

    if math.isfinite(sigma) and sigma > 0:
        rows, columns, _ = _candidate_peaks(values, sigma, radius)
        if rows.size:
            selected = _psf_star_indices(rows, columns, radius)
            kernel, used = _stack_star_stamps(
                values, rows[selected], columns[selected], radius, invalid
            )
            if kernel is not None:
                return FramePsf(
                    source="measured-star-stack",
                    star_count=used,
                    fwhm_pixels=_half_max_fwhm(kernel),
                    kernel=kernel,
                )
    return _fallback_psf(fallback_fwhm, radius)


def _fallback_psf(fallback_fwhm: float | None, radius: int) -> FramePsf:
    """The recorded analytic fallback: a Moffat of the given (or a 3.5 px) FWHM."""

    fwhm = fallback_fwhm if fallback_fwhm and math.isfinite(fallback_fwhm) else 3.5
    return FramePsf(
        source=f"moffat-beta-{PSF_FALLBACK_MOFFAT_BETA:g}-fallback",
        star_count=0,
        fwhm_pixels=float(fwhm),
        kernel=_moffat_kernel(float(fwhm), radius),
    )


def _guard_window(padded: int, extent: int, taper: int) -> NDArray[np.float32]:
    """Unit weight over the frame, a half-cosine ramp across the guard band.

    The frame occupies ``[taper, taper + extent)`` of a ``padded``-long axis
    and keeps weight 1 throughout, so the published pixels are never
    attenuated; the mirrored guard band on either side falls to zero over
    ``taper`` pixels, which is what removes the transform's wrap-around step.
    """

    window = np.zeros(padded, dtype=np.float32)
    window[taper : taper + extent] = 1.0
    if taper <= 0:
        return window
    ramp = np.asarray(
        0.5 * (1.0 - np.cos(np.pi * (np.arange(taper, dtype=np.float64) + 0.5) / taper)),
        dtype=np.float32,
    )
    window[:taper] = ramp
    tail = min(taper, padded - taper - extent)
    if tail > 0:
        window[taper + extent : taper + extent + tail] = ramp[::-1][:tail]
    return window


def _unpack_accepted(bits: NDArray[np.uint8], width: int) -> NDArray[np.bool_]:
    return np.unpackbits(bits, axis=1, count=width).astype(bool)


def proper_coadd_group(
    expressions: Sequence[FrameExpression],
    output_path: str | os.PathLike[str],
    *,
    master_path: str | os.PathLike[str],
    shape: tuple[int, int],
    flux_scales: Sequence[float],
    accepted_bits: Sequence[NDArray[np.uint8]] | None,
    metadata: Mapping[str, Any] | None = None,
    parameters: ProperCoadditionParameters,
    division_floor: float = 1e-12,
    max_memory_bytes: int,
    workers: int = 1,
    durable: bool = True,
) -> ProperCoadditionResult:
    """Coadd one registered, normalized group properly; create-only output.

    ``expressions`` are exactly the integration's expressions (registered
    frames with the group's normalization applied), ``master_path`` the
    uncropped ordinary master whose surviving robust mean replaces rejected
    samples, and ``accepted_bits`` the per-frame packed accepted masks the
    ordinary integration recorded.
    """

    parameters.validate()
    fft = _scipy_fft()
    canonical = tuple(_canonical_expression(item) for item in expressions)
    if not canonical:
        raise CalibrationError("NO_INPUTS", "proper coaddition needs at least one frame")
    if len(flux_scales) != len(canonical):
        raise ValueError("flux scale count differs from the frame count")
    reuse_rejection = parameters.outlier_handling == "reuse-rejection"
    if reuse_rejection and (accepted_bits is None or len(accepted_bits) != len(canonical)):
        raise CalibrationError(
            "PROPER_COADD_REJECTION_UNAVAILABLE",
            "reuse-rejection needs one accepted-sample mask per frame",
        )
    budget = int(max_memory_bytes)
    # As many frames in flight as the budget holds, down to one at a time.
    ahead = max(1, min(PREPARE_AHEAD, len(canonical)))
    while ahead > 1 and working_set_bytes(shape, int(parameters.apodization_pixels), ahead) > budget:
        ahead -= 1
    required = working_set_bytes(shape, int(parameters.apodization_pixels), ahead)
    if required > budget:
        raise CalibrationError(
            "PROPER_COADD_MEMORY",
            f"proper coaddition needs about {required} bytes for a "
            f"{shape[0]}x{shape[1]} group but the budget is {budget} bytes",
        )
    destination = Path(output_path)
    if destination.exists() or os.path.lexists(destination):
        raise CalibrationError(
            "OUTPUT_EXISTS", "refusing to overwrite output", path=str(destination)
        )

    height, width = shape
    radius = PSF_STAMP_RADIUS
    taper = int(parameters.apodization_pixels)
    padded = _padded_shape(shape, taper)
    window_y = _guard_window(padded[0], height, taper)
    window_x = _guard_window(padded[1], width, taper)
    spectrum_shape = (padded[0], padded[1] // 2 + 1)
    numerator = np.zeros(spectrum_shape, dtype=np.complex64)
    denominator = np.zeros(spectrum_shape, dtype=np.float32)
    timing: dict[str, float] = {"read": 0.0, "psf": 0.0, "fft": 0.0, "write": 0.0}
    started = time.perf_counter()

    frame_records: list[dict[str, Any]] = []
    replaced_total = 0
    sky_levels: list[float] = []
    flux_over_variance = 0.0
    measured_fwhm: list[float] = []

    temporary = temporary_output(destination)
    try:
        with ExitStack() as stack:
            sources = _open_expression_sources(stack, canonical)
            master = stack.enter_context(FitsFrame(Path(master_path)))
            if tuple(master.shape) != shape:
                raise CalibrationError(
                    "PROPER_COADD_GEOMETRY",
                    "the ordinary master and the registered frames differ in shape",
                )
            master_values = master.read_rows(0, height)
            work = np.empty(padded, dtype=np.float32)
            def prepare(index: int) -> tuple[Any, ...]:
                """Read, clean, sky-subtract and measure one frame (its own thread)."""

                phase = time.perf_counter()
                values = _expression_rows(
                    canonical[index], sources, 0, height, division_floor=division_floor
                )
                replaced = 0
                if reuse_rejection:
                    assert accepted_bits is not None
                    accepted = _unpack_accepted(accepted_bits[index], width)
                    # A sample the ordinary path rejected must not enter the
                    # transform: ZOGY assumes Gaussian noise, and a satellite
                    # trail or a hot pixel is neither.  The surviving robust
                    # mean of that pixel takes its place; the frames are on
                    # the reference's photometric scale already, so the master
                    # needs no further scaling.
                    substitute = ~accepted & np.isfinite(master_values)
                    replaced = int(np.count_nonzero(substitute & np.isfinite(values)))
                    values[substitute] = master_values[substitute]
                    del accepted, substitute
                sky = _sky_level(values)
                sigma = _background_sigma(values)
                if not math.isfinite(sky) or not math.isfinite(sigma) or sigma <= 0:
                    raise CalibrationError(
                        "PROPER_COADD_NOISE_UNMEASURED",
                        f"frame {index} has no measurable background sigma",
                    )
                np.subtract(values, np.float32(sky), out=values)
                # NaN borders would poison the whole periodic transform; the PSF
                # stamps must know which pixels were filled.
                invalid = ~np.isfinite(values)
                np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                read_seconds = time.perf_counter() - phase
                phase = time.perf_counter()
                # No fallback here: a frame without enough stars takes the
                # median FWHM of the frames before it, resolved in frame order
                # below, so the result never depends on the thread schedule.
                psf = measure_frame_psf(values, sigma, radius=radius, invalid=invalid)
                del invalid
                return values, replaced, sky, sigma, psf, read_seconds, time.perf_counter() - phase

            preparing: deque[Future[tuple[Any, ...]]] = deque()
            preparer = stack.enter_context(
                ThreadPoolExecutor(max_workers=ahead, thread_name_prefix="ufwbpp-proper")
            )
            stack.callback(preparer.shutdown, wait=True, cancel_futures=True)
            for index in range(ahead):
                preparing.append(preparer.submit(prepare, index))
            for index in range(len(canonical)):
                values, replaced, sky, sigma, psf, read_seconds, psf_seconds = (
                    preparing.popleft().result()
                )
                if index + ahead < len(canonical):
                    preparing.append(preparer.submit(prepare, index + ahead))
                timing["read"] += read_seconds
                timing["psf"] += psf_seconds
                replaced_total += replaced
                sky_levels.append(sky)
                if psf.source == "measured-star-stack" and math.isfinite(psf.fwhm_pixels):
                    measured_fwhm.append(psf.fwhm_pixels)
                elif psf.source != "measured-star-stack":
                    psf = _fallback_psf(
                        float(np.median(measured_fwhm)) if measured_fwhm else None, radius
                    )

                phase = time.perf_counter()
                # The frame is mirrored into the guard band and the band is
                # faded out, so the transform sees no wrap-around step while
                # every published pixel keeps unit weight.
                work[...] = np.pad(
                    values,
                    (
                        (taper, padded[0] - height - taper),
                        (taper, padded[1] - width - taper),
                    ),
                    mode="reflect",
                )
                del values
                work *= window_y[:, None]
                work *= window_x[None, :]
                frame_spectrum = fft.rfft2(work, workers=workers)
                work[...] = 0.0
                # The kernel's centre sits at the origin with wraparound, so
                # the coadd stays aligned with the registered grid.
                work[: radius + 1, : radius + 1] = psf.kernel[radius:, radius:]
                work[: radius + 1, padded[1] - radius :] = psf.kernel[radius:, :radius]
                work[padded[0] - radius :, : radius + 1] = psf.kernel[:radius, radius:]
                work[padded[0] - radius :, padded[1] - radius :] = psf.kernel[:radius, :radius]
                psf_spectrum = fft.rfft2(work, workers=workers)

                flux = float(flux_scales[index])
                inverse_variance = 1.0 / (sigma * sigma)
                numerator += np.asarray(
                    np.float32(flux * inverse_variance)
                    * np.conjugate(psf_spectrum)
                    * frame_spectrum,
                    dtype=np.complex64,
                )
                denominator += np.asarray(
                    np.float32(flux * flux * inverse_variance)
                    * (psf_spectrum.real**2 + psf_spectrum.imag**2),
                    dtype=np.float32,
                )
                flux_over_variance += flux * flux * inverse_variance
                del frame_spectrum, psf_spectrum
                timing["fft"] += time.perf_counter() - phase

                frame_records.append(
                    {
                        "index": index,
                        "fluxScale": flux,
                        "backgroundSigma": sigma,
                        "backgroundLevel": sky,
                        "replacedSamples": replaced,
                        **psf.serializable(),
                    }
                )

            flux_norm = math.sqrt(flux_over_variance)
            if not math.isfinite(flux_norm) or flux_norm <= 0:
                raise CalibrationError(
                    "PROPER_COADD_DEGENERATE", "the coadd's flux normalization vanished"
                )
            phase = time.perf_counter()
            root = np.sqrt(denominator, dtype=np.float32)
            # Numerator and denominator vanish together, so the ratio stays
            # well conditioned wherever the transforms carry any information
            # at all.  The only frequencies that must be dropped are the ones
            # where the denominator is at, or under, the float32 transform's
            # own round-off; a floor any higher would silently low-pass the
            # product and make its background look quieter than it is.
            floor = np.float32(1e-14) * np.float32(root.max())
            usable = root > floor
            floored_fraction = float(
                1.0 - np.count_nonzero(usable) / float(usable.size)
            )
            numerator[~usable] = 0
            np.divide(numerator, root, out=numerator, where=usable)
            # The coadd's own PSF, from the same accumulator.
            coadd_psf_spectrum = np.asarray(
                root / np.float32(flux_norm), dtype=np.float32
            ).astype(np.complex64)
            del root, usable, denominator
            coadd = fft.irfft2(numerator, s=padded, workers=workers)
            del numerator
            coadd_psf = fft.irfft2(coadd_psf_spectrum, s=padded, workers=workers)
            del coadd_psf_spectrum
            # R is written with unit noise variance; dividing by F_R puts it in
            # the normalized frames' photometric units, where its noise is the
            # ideal stacked 1 / F_R.
            sky_added = float(np.mean(np.asarray(sky_levels, dtype=np.float64)))
            centre = np.roll(
                np.roll(coadd_psf, radius, axis=0), radius, axis=1
            )[: 2 * radius + 1, : 2 * radius + 1]
            coadd_fwhm = _half_max_fwhm(np.asarray(centre, dtype=np.float32))
            del coadd_psf, centre
            timing["fft"] += time.perf_counter() - phase

            phase = time.perf_counter()
            output_metadata = dict(metadata or {})
            output_metadata.setdefault("OAFSTATE", "UNSOLVED_WORKING")
            output_metadata["OAFPCOAD"] = PROPER_COADD_ALGORITHM_ID
            output_metadata["OAFPCFR"] = flux_norm
            output_metadata["OAFPCFWH"] = coadd_fwhm
            output_metadata["OAFPCSKY"] = sky_added
            output_metadata["OAFPCAPO"] = taper
            output_metadata["OAFPCREP"] = replaced_total
            output_metadata["OAFPCOUT"] = parameters.outlier_handling
            output_metadata["OAFNFRM"] = len(canonical)
            with FitsFloatWriter(
                temporary, shape, output_metadata, durable=durable
            ) as writer:
                band = max(1, min(height, 1024))
                scale = np.float32(1.0 / flux_norm)
                for y0 in range(0, height, band):
                    y1 = min(height, y0 + band)
                    rows = np.asarray(
                        coadd[taper + y0 : taper + y1, taper : taper + width],
                        dtype=np.float32,
                    ) * scale
                    rows += np.float32(sky_added)
                    writer.write_rows(y0, rows)
            digest = writer.sha256
            del coadd
            timing["write"] += time.perf_counter() - phase
        atomic_publish_file(temporary, destination)
    finally:
        remove_file(temporary)

    timing["total"] = time.perf_counter() - started
    return ProperCoadditionResult(
        output_path=str(destination),
        shape=shape,
        frame_count=len(canonical),
        flux_scale_norm=flux_norm,
        coadd_fwhm_pixels=coadd_fwhm,
        sky_added=sky_added,
        replaced_samples=replaced_total,
        floored_spectrum_fraction=floored_fraction,
        frames=tuple(frame_records),
        timing_seconds=timing,
        working_set_bytes=required,
        padded_shape=padded,
        frames_prepared_ahead=ahead,
        output_sha256=digest,
    )


__all__ = [
    "PROPER_COADD_ALGORITHM_ID",
    "FramePsf",
    "ProperCoadditionParameters",
    "ProperCoadditionResult",
    "measure_frame_psf",
    "proper_coadd_group",
    "working_set_bytes",
]
