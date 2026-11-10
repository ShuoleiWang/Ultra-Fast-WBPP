"""A super luminance: the broadband channels of one grid combined into one
deeper luminance image.

Each channel is put on the reference channel's flux scale (the median ratio of
aperture fluxes at the reference's isolated, unsaturated stars) and weighted by
the inverse variance of its sky noise in the reference's units, so the result
is the minimum-variance estimate of the reference's signal.  The noise is the
robust scatter of neighbouring-pixel differences on the sky, not the sky's own
scatter: on a deep master that scatter is mostly faint galaxies and background
structure that every channel shares, which do not average down and would pull
the weights towards equal.

A channel whose stars, stacked at the reference's star positions, are clearly
wider or more elongated than the reference's is left out: its photons would
blur the luminance more than they deepen it.  Measured at the same stars, the
comparison does not depend on how deep each channel is, and an offset from the
reference's star centres reads as elongation.  A channel covering less of the
reference's pixels is left out too, so the depth does not jump at its edge.
Every channel's sky level is replaced by the reference's: the result keeps the
reference's background and its footprint.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from astropy.io import fits
import numpy as np
from numpy.typing import NDArray

from ..image_io.fits import FitsFloatWriter


SUPER_LUMINANCE_ALGORITHM = "super-luminance-inverse-variance-v2"
# The FILTER of the product; the colour builder reads it as the L role.
SUPER_LUMINANCE_FILTER = "SuperL"
# Channels that take part: the broadband filters (narrowband light is not luminance).
BROADBAND_KEYS = ("l", "r", "g", "b")
# A channel is left out when its stars are this much wider, or this much
# more elongated, than the reference's ...
MAXIMUM_FWHM_RATIO = 1.15
MAXIMUM_ELLIPTICITY_EXCESS = 0.08
# ... or when it covers less than this share of the reference's pixels.
MINIMUM_COVERAGE_FRACTION = 0.98
MINIMUM_SCALE_STARS = 20
# Stars are measured in a 3 x 3 grid of boxes spread over the field, so a
# channel whose stars degrade towards the edges is judged by its whole field.
_MEASUREMENT_BOX = 384
_MINIMUM_STACK_STARS = 5
_SKY_TILE = 128
_SKY_TILES_PER_AXIS = 16
_BAND_ROWS = 512
_STRUCTURAL_KEYS = frozenset(
    {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "BZERO", "BSCALE", "CHECKSUM", "DATASUM"}
)


def _boxes(shape: tuple[int, int]) -> list[tuple[slice, slice]]:
    height, width = shape
    size_y, size_x = min(height // 3, _MEASUREMENT_BOX), min(width // 3, _MEASUREMENT_BOX)
    boxes = []
    for fy in (1.0 / 6.0, 0.5, 5.0 / 6.0):
        for fx in (1.0 / 6.0, 0.5, 5.0 / 6.0):
            y0, x0 = int(fy * height) - size_y // 2, int(fx * width) - size_x // 2
            boxes.append((slice(y0, y0 + size_y), slice(x0, x0 + size_x)))
    return boxes


def _reference_stars(data: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Positions of the isolated, unsaturated stars of one reference box and
    their isophotal FWHMs; empty when the box is mostly outside the data or
    too crowded to extract."""

    import sep
    from scipy.spatial import cKDTree

    empty = np.empty(0)
    finite = np.isfinite(data)
    if np.count_nonzero(finite) < finite.size // 2:
        return empty, empty, empty
    work = np.ascontiguousarray(np.where(finite, data, np.nanmedian(data)))
    background = sep.Background(work, mask=~finite, bw=64, bh=64)
    try:
        objects = sep.extract(work - background.back(), 10.0, err=background.globalrms, mask=~finite, minarea=9)
    except Exception:
        # SEP raises a bare Exception when a galaxy fills the box beyond its pixel stack.
        return empty, empty, empty
    if len(objects) < 2:
        return empty, empty, empty
    fwhm = 2.3548 * np.sqrt(np.clip(objects["a"] * objects["b"], 0.0, None))
    keep = (objects["flag"] == 0) & (objects["b"] > 0)
    # The brightest tenth may be saturated: their cores are flat.
    keep &= objects["peak"] < np.percentile(objects["peak"], 90.0)
    if keep.any():
        # Isolated: no other detection within four FWHM.
        points = np.column_stack((objects["x"], objects["y"]))
        nearest = cKDTree(points).query(points, k=2)[0][:, 1]
        keep &= nearest >= 4.0 * float(np.median(fwhm[keep]))
    return objects["x"][keep], objects["y"][keep], fwhm[keep]


def _stacked_kernel(
    data: NDArray[np.float64], rows: NDArray[np.int64], columns: NDArray[np.int64], radius: int, norms: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Median of the stars' stamps, each less its border's median and divided
    by its star's expected flux."""

    stamps = []
    for row, column, norm in zip(rows.tolist(), columns.tolist(), norms.tolist()):
        stamp = data[row - radius : row + radius + 1, column - radius : column + radius + 1]
        border = np.concatenate((stamp[0], stamp[-1], stamp[1:-1, 0], stamp[1:-1, -1]))
        stamps.append((stamp - np.median(border)) / norm)
    return np.median(np.stack(stamps), axis=0)


def _half_maximum_fwhm(kernel: NDArray[np.float64]) -> float:
    """FWHM from the area above half maximum, counted on a 4x cubic upsampling
    so the pixel grid does not quantize a narrow PSF."""

    from scipy import ndimage

    fine = ndimage.zoom(kernel, 4, order=3, grid_mode=False)
    peak = float(np.max(fine))
    return 2.0 * math.sqrt(np.count_nonzero(fine >= 0.5 * peak) / 16.0 / math.pi) if peak > 0 else math.nan


def _moments(kernel: NDArray[np.float64], centre: tuple[float, float], radius: float) -> tuple[float, float, float]:
    """Centroid offset from ``centre`` and ellipticity (1 - b/a of the second
    moments about ``centre``) of the kernel's positive part within ``radius``
    of ``centre``; NaN when nothing positive is there."""

    axis = np.arange(kernel.shape[0], dtype=np.float64) - kernel.shape[0] // 2
    dy, dx = axis[:, None] - centre[0], axis[None, :] - centre[1]
    weight = np.where(dy * dy + dx * dx <= radius * radius, np.clip(kernel, 0.0, None), 0.0)
    total = float(np.sum(weight))
    if not total > 0:
        return math.nan, math.nan, math.nan
    mean_y, mean_x, yy, xx, xy = (float(np.sum(weight * term)) / total for term in (dy, dx, dy * dy, dx * dx, dx * dy))
    half_sum, half_difference = 0.5 * (yy + xx), math.hypot(0.5 * (yy - xx), xy)
    ellipticity = 1.0 - math.sqrt(max(half_sum - half_difference, 0.0) / (half_sum + half_difference)) if half_sum > 0 else math.nan
    return mean_y, mean_x, ellipticity


def _sky_tiles(data: NDArray[np.float32]) -> list[tuple[slice, slice]]:
    """The faintest quarter of a regular grid of tiles that lie in the data: the sky."""

    height, width = data.shape
    size_y, size_x = min(_SKY_TILE, height), min(_SKY_TILE, width)
    levels = []
    for y in sorted({int(value) for value in np.linspace(0, height - size_y, _SKY_TILES_PER_AXIS)}):
        for x in sorted({int(value) for value in np.linspace(0, width - size_x, _SKY_TILES_PER_AXIS)}):
            sample = np.asarray(data[y : y + size_y : 4, x : x + size_x : 4], dtype=np.float32)
            if np.isfinite(sample).mean() >= 0.95:
                levels.append((float(np.nanmedian(sample)), y, x))
    levels.sort()
    return [(slice(y, y + size_y), slice(x, x + size_x)) for _, y, x in levels[: max(1, len(levels) // 4)]]


def _sky_level_and_noise(data: NDArray[np.float32], tiles: list[tuple[slice, slice]]) -> tuple[float, float] | None:
    """The median sky level of the tiles and their pixel noise: the robust
    scatter of differences between neighbouring pixels along both axes."""

    levels, variances = [], []
    for tile_slices in tiles:
        tile = np.asarray(data[tile_slices], dtype=np.float32)
        finite = tile[np.isfinite(tile)]
        scatter = []
        for axis in (0, 1):
            differences = np.abs(np.diff(tile, axis=axis)).ravel()
            differences = differences[np.isfinite(differences)]
            if differences.size >= tile.size // 2:
                # Sky differences centre on zero, so their median absolute
                # value is their median absolute deviation.
                scatter.append(1.4826 * float(np.median(differences)))
        if len(scatter) == 2:
            levels.append(float(np.median(finite)))
            variances.append(0.25 * (scatter[0] ** 2 + scatter[1] ** 2))
    if not variances:
        return None
    return float(np.median(levels)), math.sqrt(float(np.median(variances)))


def _recorded(measurements: Mapping[str, Any]) -> dict[str, Any]:
    """Measurements as a receipt holds them: one that could not be made (a
    value that is not finite) is ``None``, which canonical JSON can state."""

    return {
        name: None if isinstance(value, float) and not math.isfinite(value) else value
        for name, value in measurements.items()
    }


def build_super_luminance(
    channels: Mapping[str, Path],
    reference_key: str,
    destination: Path,
    *,
    filter_names: Mapping[str, str],
) -> dict[str, Any]:
    """Write the super luminance of ``channels`` (filter key -> mono FITS on
    one grid, with a verified WCS) to ``destination`` with the reference
    channel's header; returns its record.  Nothing is written when fewer
    than two broadband channels qualify."""

    import sep

    keys = [key for key in BROADBAND_KEYS if key in channels]
    record: dict[str, Any] = {"algorithm": SUPER_LUMINANCE_ALGORITHM, "status": "NOT_CREATED"}
    if len(keys) < 2:
        return {**record, "reason": "fewer than two broadband channels"}
    reference = reference_key if reference_key in keys else keys[0]
    record["referenceFilter"] = filter_names.get(reference, reference)
    opened = {key: fits.open(Path(channels[key]), mode="readonly", memmap=True) for key in keys}
    try:
        images = {key: next(hdu for hdu in hdul if hdu.data is not None and hdu.data.ndim == 2) for key, hdul in opened.items()}
        data = {key: hdu.data for key, hdu in images.items()}
        shape = data[reference].shape
        if any(item.shape != shape for item in data.values()):
            return {**record, "reason": "the channels do not share one grid"}

        # Stars: found on the reference, measured at the same positions in
        # every channel, box by box.
        boxes = [{key: np.asarray(data[key][box], dtype=np.float64) for key in keys} for box in _boxes(shape)]
        found = [_reference_stars(box[reference]) for box in boxes]
        isophotal = np.concatenate([item[2] for item in found])
        if isophotal.size < MINIMUM_SCALE_STARS:
            return {**record, "reason": f"fewer than {MINIMUM_SCALE_STARS} isolated stars on the reference"}
        size = float(np.median(isophotal))
        radius = min(24, max(6, math.ceil(3.0 * size)))
        reach = math.ceil(5.0 * size) + 1
        fluxes: list[dict[str, tuple[NDArray[np.float64], NDArray[np.bool_]]]] = []
        for box, (x, y, _) in zip(boxes, found):
            measured = {}
            for key in keys:
                # Only stars whose aperture and annulus lie in this channel's
                # data: SEP refuses an annulus without valid pixels.
                finite = np.isfinite(box[key])
                inside = np.array([
                    bool(finite[row - reach : row + reach + 1, column - reach : column + reach + 1].all())
                    and reach <= row < finite.shape[0] - reach and reach <= column < finite.shape[1] - reach
                    for row, column in zip(np.rint(y).astype(int).tolist(), np.rint(x).astype(int).tolist())
                ], dtype=bool)
                flux, usable = np.zeros(x.size), np.zeros(x.size, dtype=bool)
                if inside.any():
                    values, _, flags = sep.sum_circle(
                        np.ascontiguousarray(np.where(finite, box[key], 0.0)), x[inside], y[inside], 2.0 * size,
                        bkgann=(3.0 * size, 5.0 * size),
                    )
                    flux[inside], usable[inside] = values, flags == 0
                measured[key] = (flux, usable)
            fluxes.append(measured)
        scales: dict[str, float] = {}
        scale_stars: dict[str, int] = {}
        for key in keys:
            ratios = np.concatenate([
                (item[key][0] / item[reference][0])[item[key][1] & item[reference][1] & (item[key][0] > 0) & (item[reference][0] > 0)]
                for item in fluxes
            ])
            scale_stars[key] = int(ratios.size)
            if ratios.size >= MINIMUM_SCALE_STARS:
                scales[key] = float(np.median(ratios))

        # Star shapes: per box, every scaled channel's stack at the
        # reference's stars, its moments taken about the reference's centre.
        shapes: dict[str, list[tuple[float, ...]]] = {key: [] for key in keys}
        stacks: list[tuple[dict[str, NDArray[np.float64]], tuple[float, float], float, float, float]] = []
        for box, (x, y, _), measured in zip(boxes, found, fluxes):
            if reference not in scales:
                break
            rows, columns = np.rint(y).astype(np.int64), np.rint(x).astype(np.int64)
            height, width = box[reference].shape
            usable = (rows >= radius) & (columns >= radius) & (rows < height - radius) & (columns < width - radius)
            usable &= measured[reference][1] & (measured[reference][0] > 0)
            for index in np.flatnonzero(usable):
                stamp = (slice(rows[index] - radius, rows[index] + radius + 1), slice(columns[index] - radius, columns[index] + radius + 1))
                usable[index] = all(bool(np.isfinite(box[key][stamp]).all()) for key in scales)
            if np.count_nonzero(usable) < _MINIMUM_STACK_STARS:
                continue
            kernels = {
                key: _stacked_kernel(box[key], rows[usable], columns[usable], radius, measured[reference][0][usable] * scales[key])
                for key in scales
            }
            reference_fwhm = _half_maximum_fwhm(kernels[reference])
            if not reference_fwhm > 0:
                continue
            window = 1.5 * reference_fwhm
            centre_y, centre_x, _ = _moments(kernels[reference], (0.0, 0.0), window)
            _, _, reference_ellipticity = _moments(kernels[reference], (centre_y, centre_x), window)
            stacks.append((kernels, (centre_y, centre_x), window, reference_fwhm, reference_ellipticity))
            for key, kernel in kernels.items():
                fwhm = _half_maximum_fwhm(kernel)
                offset_y, offset_x, ellipticity = _moments(kernel, (centre_y, centre_x), window)
                shapes[key].append(
                    (fwhm, fwhm / reference_fwhm, ellipticity, ellipticity - reference_ellipticity, math.hypot(offset_y, offset_x))
                )

        tiles = _sky_tiles(data[reference])
        step = 8
        reference_finite = np.isfinite(data[reference][::step, ::step])
        channel_records: dict[str, Any] = {}
        weights: dict[str, float] = {}
        levels: dict[str, float] = {}
        for key in keys:
            coverage = float(np.count_nonzero(np.isfinite(data[key][::step, ::step]) & reference_finite)) / max(
                1, int(np.count_nonzero(reference_finite))
            )
            noise = _sky_level_and_noise(data[key], tiles)
            entry: dict[str, Any] = {
                "filter": filter_names.get(key, key),
                "scale": scales.get(key),
                "scaleStars": scale_stars[key],
                "shapeBoxes": len(shapes[key]),
                "coverageFraction": coverage,
                "skyLevel": None if noise is None else noise[0],
                "skyNoise": None if noise is None else noise[1],
            }
            if shapes[key]:
                medians = np.median(np.asarray(shapes[key]), axis=0)
                entry.update(
                    fwhmPixels=float(medians[0]),
                    fwhmRatio=float(medians[1]),
                    ellipticity=float(medians[2]),
                    ellipticityExcess=float(medians[3]),
                    starOffsetPixels=float(medians[4]),
                )
            # Comparisons are written to fail on a NaN measurement.
            reason = None
            if key not in scales or not scales[key] > 0:
                reason = f"fewer than {MINIMUM_SCALE_STARS} stars to scale it"
            elif not shapes[key]:
                reason = "too few isolated stars to measure its star shapes"
            elif not (math.isfinite(entry["fwhmRatio"]) and math.isfinite(entry["ellipticityExcess"])):
                reason = "its star shapes could not be measured"
            elif not entry["fwhmRatio"] <= MAXIMUM_FWHM_RATIO:
                reason = f"stars {entry['fwhmRatio']:.2f}x wider than the reference's"
            elif not entry["ellipticityExcess"] <= MAXIMUM_ELLIPTICITY_EXCESS:
                reason = f"stars {entry['ellipticityExcess']:+.2f} more elongated than the reference's"
            elif not coverage >= MINIMUM_COVERAGE_FRACTION:
                reason = f"covers {coverage:.1%} of the reference's pixels"
            elif noise is None or not noise[1] > 0:
                reason = "no sky to measure its noise"
            if reason is None:
                # Inverse variance of its noise in the reference's units.
                weights[key] = scales[key] ** 2 / noise[1] ** 2
                levels[key] = noise[0]
            entry["included"] = reason is None
            if reason is not None:
                entry["reason"] = reason
            channel_records[key] = _recorded(entry)
        record["channels"] = channel_records
        if reference not in weights:
            return {**record, "reason": f"the reference is unusable: {channel_records[reference]['reason']}"}
        if len(weights) < 2:
            return {**record, "reason": "no channel besides the reference qualifies"}
        total = sum(weights.values())
        for key, weight in weights.items():
            channel_records[key]["weight"] = weight / total
        # The signal-to-noise gain over the reference alone.
        record["effectiveGain"] = math.sqrt(total / weights[reference])
        included = [key for key in keys if key in weights]
        # The result's stars, against the reference's: the stacks combine
        # with the channels' weights exactly as the images do.
        combined = []
        for kernels, centre, window, reference_fwhm, reference_ellipticity in stacks:
            kernel = sum(kernels[key] * (weights[key] / total) for key in included)
            combined.append((_half_maximum_fwhm(kernel) / reference_fwhm, _moments(kernel, centre, window)[2] - reference_ellipticity))
        medians = np.median(np.asarray(combined), axis=0)
        record["stars"] = _recorded(
            {"fwhmRatio": float(medians[0]), "ellipticityExcess": float(medians[1]), "boxes": len(combined)}
        )

        metadata: dict[str, Any] = {
            card.keyword: card.value
            for card in images[reference].header.cards
            if card.keyword and card.keyword not in _STRUCTURAL_KEYS and card.keyword not in ("COMMENT", "HISTORY")
        }
        metadata.update(
            FILTER=SUPER_LUMINANCE_FILTER,
            OAFSUPL=SUPER_LUMINANCE_ALGORITHM,
            OAFSLREF=filter_names.get(reference, reference),
            OAFSLCH=",".join(filter_names.get(key, key) for key in included),
            OAFSLWT=",".join(f"{weights[key] / total:.4f}" for key in included),
            OAFSLGN=round(record["effectiveGain"], 4),
        )
        reference_level = np.float32(levels[reference])
        height, width = shape
        with FitsFloatWriter(destination, shape, metadata) as writer:
            for y0 in range(0, height, _BAND_ROWS):
                y1 = min(height, y0 + _BAND_ROWS)
                footprint = np.isfinite(np.asarray(data[reference][y0:y1], dtype=np.float32))
                numerator = np.zeros((y1 - y0, width), dtype=np.float32)
                denominator = np.zeros((y1 - y0, width), dtype=np.float32)
                for key in included:
                    rows_data = np.asarray(data[key][y0:y1], dtype=np.float32)
                    finite = np.isfinite(rows_data)
                    factor = np.float32(weights[key] / scales[key])
                    numerator += np.where(finite, (rows_data - np.float32(levels[key])) * factor, np.float32(0.0))
                    denominator += np.where(finite, np.float32(weights[key]), np.float32(0.0))
                with np.errstate(invalid="ignore", divide="ignore"):
                    writer.write_rows(y0, np.where(footprint, reference_level + numerator / denominator, np.float32(np.nan)))
        record.update(status="CREATED", sha256=writer.sha256)
        return record
    finally:
        # An error keeps this frame alive in its traceback: the mapped pixels
        # are released here, or the channels stay mapped (and on Windows
        # cannot be renamed) while the caller handles the failure.
        images = data = None
        for hdul in opened.values():
            hdul.close()


__all__ = ["SUPER_LUMINANCE_ALGORITHM", "SUPER_LUMINANCE_FILTER", "build_super_luminance"]
