"""Temporal, spatially coherent trail masks for ordinary integration.

A trail must be absent from the robust registered reference and have distributed
positive evidence along a long narrow corridor. Ordinary MAD rejection decisions
are preserved; the corridor mask can reject additional samples.

At pixels affected by that additional rejection, an optional group-relative sky
model adjusts the temporary integration values to preserve the original accepted
background. This avoids stripes when the removed frame has a different sky level.
Stored inputs and pixels outside those corridors remain unchanged.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from .residual_background import ResidualBackgroundAlignment

ALGORITHM = "temporal-residual-supported-line-corridor-v1"
MINIMUM_LENGTH_PIXELS = 256


@dataclass(frozen=True, slots=True)
class TransientTrail:
    frame: int
    normal_x: float
    normal_y: float
    distance: float
    start: float
    stop: float
    half_width: float
    seed_count: int
    supported_segments: int
    profile_sigma: float

    def serializable(self) -> dict[str, Any]:
        return {
            "frameIndex": self.frame,
            "normal": [self.normal_x, self.normal_y],
            "distancePixels": self.distance,
            "alongStartPixels": self.start,
            "alongStopPixels": self.stop,
            "halfWidthPixels": self.half_width,
            "seedCount": self.seed_count,
            "supportedSegments": self.supported_segments,
            "profileSigma": self.profile_sigma,
        }


@dataclass(frozen=True, slots=True)
class TransientRejectionModel:
    bin_factor: int
    trails: tuple[TransientTrail, ...] = ()
    status: str = "APPLIED"
    background_alignment: ResidualBackgroundAlignment | None = None

    def normalize_rejected_rows(self, values: NDArray[np.float32], first_row: int,
                                original: NDArray[np.bool_], accepted: NDArray[np.bool_]) -> None:
        """Adjust a temporary tile only where the spatial mask removed samples.

        ``original`` is the ordinary MAD acceptance mask; ``accepted`` includes
        the additional trail rejection. The background correction is anchored to
        the original weighted acceptance set, never to a single reference frame.
        """
        if self.background_alignment is None:
            return
        changed = np.any(original & ~accepted, axis=0)
        if not np.any(changed):
            return
        # Only rows that contain an additionally rejected pixel need the
        # per-pixel correction; the model is evaluated on those contiguous
        # row runs alone, which gives the same values as the full tile.
        changed_rows = np.flatnonzero(np.any(changed, axis=1))
        weights = np.asarray(self.background_alignment.weights)[:, None, None]
        run_start = int(changed_rows[0])
        previous = run_start
        runs: list[tuple[int, int]] = []
        for row in changed_rows[1:]:
            row = int(row)
            if row != previous + 1:
                runs.append((run_start, previous + 1))
                run_start = row
            previous = row
        runs.append((run_start, previous + 1))
        for row0, row1 in runs:
            correction = np.zeros(
                (values.shape[0], row1 - row0, values.shape[2]), dtype=values.dtype
            )
            self.background_alignment.apply_rows(correction, first_row + row0)
            original_rows = original[:, row0:row1]
            changed_rows_mask = changed[row0:row1]
            denominator = np.sum(original_rows * weights, axis=0)
            anchor = np.divide(
                np.sum(np.where(original_rows, correction, 0) * weights, axis=0),
                denominator, out=np.zeros_like(denominator), where=denominator > 0,
            )
            # Equivalent to mean_new(X-C) + mean_original(C). No correction is
            # applied elsewhere, so ordinary integration controls stay bitwise equal.
            for index in range(values.shape[0]):
                block = values[index, row0:row1]
                block[changed_rows_mask] += (
                    correction[index, changed_rows_mask] - anchor[changed_rows_mask]
                ).astype(np.float32)

    def serializable(self) -> dict[str, Any]:
        return {
            "algorithm": ALGORITHM,
            "status": self.status,
            "binFactor": self.bin_factor,
            "reference": "registered-group-temporal-median",
            "backgroundRemoval": "detection-only-clipped-smooth-residual",
            "minimumLengthPixels": MINIMUM_LENGTH_PIXELS,
            "seedSigma": 2.5,
            "minimumSupportedSegments": 6,
            "minimumProfileSigma": 6.0,
            "tileInvariant": True,
            "intensitiesModified": self.background_alignment is not None,
            "storedInputsModified": False,
            "skyAlignmentApplication": "additional-spatial-rejection-pixels-only",
            "skyReference": "original-mad-accepted-weighted-background",
            "ordinaryMadDecisions": "unchanged-original-input-values",
            "backgroundAlignment": (self.background_alignment.serializable()
                                    if self.background_alignment is not None else None),
            "trails": [trail.serializable() for trail in self.trails],
        }

    def reject_rows(self, accepted: NDArray[np.bool_], first_row: int,
                    enough_samples: NDArray[np.bool_]) -> None:
        height, width = accepted.shape[1:]
        x = np.arange(width, dtype=np.float64)[None, :]
        y = np.arange(first_row, first_row + height, dtype=np.float64)[:, None]
        for trail in self.trails:
            distance = x * trail.normal_x + y * trail.normal_y - trail.distance
            along = -x * trail.normal_y + y * trail.normal_x
            corridor = ((np.abs(distance) <= trail.half_width)
                        & (along >= trail.start) & (along <= trail.stop)
                        & enough_samples)
            accepted[trail.frame, corridor] = False


def detect_transient_trails(values: NDArray[np.float32], bin_factor: int,
                            *, workers: int = 1) -> TransientRejectionModel:
    """Fit deterministic corridors from block-mean registered frame samples.

    Frames are analysed independently against the shared temporal median, so
    ``workers`` frames run concurrently; the fitted corridors are identical
    for any worker count and keep frame order.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter
    from skimage.transform import hough_line, hough_line_peaks

    frame_count, height, width = values.shape
    if frame_count < 5 or min(height, width) * bin_factor < MINIMUM_LENGTH_PIXELS:
        return TransientRejectionModel(bin_factor, status="NOT_APPLICABLE")
    finite = np.isfinite(values)
    common = np.sum(finite, axis=0) >= max(5, (frame_count + 1) // 2)
    reference = np.zeros((height, width), dtype=np.float32)
    if np.any(common):
        reference[common] = np.nanmedian(values[:, common], axis=0)
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 1440, endpoint=False)
    # Shared stars, diffraction spikes, and resolved structure cannot seed
    # a trail; the structure map depends only on the reference.
    structure = reference - gaussian_filter(reference, 3)

    def frame_trails(frame_index: int) -> list[TransientTrail]:
        trails: list[TransientTrail] = []
        valid = common & finite[frame_index]
        if np.count_nonzero(valid) < 256:
            return trails
        residual = np.where(valid, values[frame_index] - reference, 0.0)
        location = float(np.median(residual[valid]))
        sigma = float(1.4826 * np.median(np.abs(residual[valid] - location)))
        if not np.isfinite(sigma) or sigma <= 0:
            return trails
        # Clipping keeps a satellite itself from raising the background model.
        clipped = np.clip(residual - location, -3 * sigma, 3 * sigma)
        support = gaussian_filter(valid.astype(np.float32), 8)
        background = gaussian_filter(np.where(valid, clipped, 0), 8)
        background /= np.maximum(support, 0.01)
        residual = residual - location - background
        sigma = float(1.4826 * np.median(np.abs(
            residual[valid] - np.median(residual[valid]))))
        if not np.isfinite(sigma) or sigma <= 0:
            return trails
        # The final corridor can cross protected structure, using other
        # frames there.
        protected = binary_dilation(structure > 4 * sigma, iterations=2)
        # Normalized smoothing remains valid at a footprint boundary: requiring
        # almost full support would truncate every fitted trail by several
        # smoothing radii before the image edge.
        valid &= ~protected & (support > 0.25)
        seeds = valid & (residual > 2.5 * sigma)
        hough, angles, distances = hough_line(seeds, theta=theta)
        peaks, peak_angles, peak_distances = hough_line_peaks(
            hough, angles, distances, threshold=12,
            min_distance=4, min_angle=8, num_peaks=16,
        )
        for peak, angle, distance in zip(peaks, peak_angles, peak_distances, strict=True):
            nx, ny = float(np.cos(angle)), float(np.sin(angle))
            cross = xx * nx + yy * ny - distance
            near = seeds & (np.abs(cross) <= 1.5)
            py, px = np.nonzero(near)
            if px.size < 12:
                continue
            # Refine the Hough angle using only distributed seed coordinates.
            points = np.column_stack((px, py)).astype(np.float64)
            centroid = np.mean(points, axis=0)
            _, _, axes = np.linalg.svd(points - centroid, full_matrices=False)
            normal = axes[1]
            if normal @ np.array([nx, ny]) < 0:
                normal *= -1
            nx, ny = map(float, normal)
            distance = float(centroid @ normal)
            cross = xx * nx + yy * ny - distance
            along = -xx * ny + yy * nx
            seed_along = -px * ny + py * nx
            start, stop = np.quantile(seed_along, [0.01, 0.99])
            length = float(stop - start)
            if length * bin_factor < MINIMUM_LENGTH_PIXELS:
                continue
            core = valid & (np.abs(cross) < 1.5) & (along >= start) & (along <= stop)
            if np.count_nonzero(core) < 64:
                continue
            # Independent positions along a line, not a bright point source,
            # must account for the excess in at least 6 of 8 spatial segments.
            clipped_signal = np.clip(residual, -4 * sigma, 4 * sigma)
            supported = 0
            for lo, hi in zip(np.linspace(start, stop, 9)[:-1],
                              np.linspace(start, stop, 9)[1:], strict=True):
                segment = core & (along >= lo) & (along < hi)
                if (np.count_nonzero(segment) >= 8
                        and np.mean(clipped_signal[segment]) > 0.3 * sigma):
                    supported += 1
            # A sub-bin vertical/horizontal trail can occupy only one of the
            # three core columns. A median would erase that coherent evidence.
            signal = float(np.mean(clipped_signal[core]))
            significance = signal / sigma * np.sqrt(length / 3)
            if supported < 6 or significance < 6:
                continue
            # Measure the cross-track profile, including sub-threshold wings.
            # The uncertainty uses independent along-track positions; one bin
            # of support padding covers bin phase and interpolation wings.
            profile = []
            for offset in range(-8, 9):
                band = (valid & (np.abs(cross - offset) < 0.5)
                        & (along >= start) & (along <= stop))
                profile.append(float(np.median(residual[band]))
                               if np.count_nonzero(band) >= 16 else 0.0)
            threshold = max(0.15, 3 / np.sqrt(length / 3)) * sigma
            left = right = 8
            while left > 0 and profile[left - 1] > threshold:
                left -= 1
            while right < 16 and profile[right + 1] > threshold:
                right += 1
            half_width = float(max(8 - left, right - 8) + 1.5)
            if half_width >= 8 or 2 * half_width * length > height * width * 0.05:
                continue
            # Deduplicate neighboring Hough peaks for the same corridor.
            if any(t.frame == frame_index and abs(t.normal_x * nx + t.normal_y * ny) > .999
                   and abs(t.distance / bin_factor - distance) < half_width + 2 for t in trails):
                continue
            shift = (bin_factor - 1) / 2
            trails.append(TransientTrail(
                frame_index, nx, ny,
                float(distance * bin_factor + shift * (nx + ny)),
                float(start * bin_factor + shift * (nx - ny) - 2 * bin_factor),
                float(stop * bin_factor + shift * (nx - ny) + 2 * bin_factor),
                half_width * bin_factor, int(px.size), supported, significance,
            ))
        return trails

    frame_workers = max(1, min(workers, frame_count))
    if frame_workers == 1:
        per_frame = [frame_trails(index) for index in range(frame_count)]
    else:
        with ThreadPoolExecutor(
            max_workers=frame_workers, thread_name_prefix="oaf-trails"
        ) as executor:
            per_frame = list(executor.map(frame_trails, range(frame_count)))
    trails = [trail for frame in per_frame for trail in frame]
    return TransientRejectionModel(bin_factor, tuple(trails))
