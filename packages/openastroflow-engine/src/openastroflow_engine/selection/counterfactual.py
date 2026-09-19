"""Leave-one-out counterfactual evidence computed inside the integration loop.

For every integration tile the accumulator receives the frame samples, the
rejection mask, the frame weights and the integrated tile.  It reduces the
tile to 8x8 block sums per frame, from which the block-level weighted mean of
the full stack and of every leave-one-out stack follows analytically:

    m_{-i} = (S - w_i * sum_i) / (W - w_i * count_i)

Three master-quality proxies are then evaluated per frame:

* depth: the robust noise of the plane-removed block means over star-free
  blocks (an 8-px-scale effective noise), summarised as the median over tiles;
* background: the RMS of the second-order polynomial residual of a coarse
  background map (tile rows x 256-px column segments), in units of the block
  noise of the full master;
* PSF: the weighted quadratic-mean FWHM proxy, without pixels.

Each frame's counterfactual is the change of the proxy when the frame is
removed, signed so that a positive value means "the master is better without
this frame".  Tile and cell bootstraps give confidence intervals.  The
rejection masks come from the full stack (a second-order approximation that
the report records).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

COUNTERFACTUAL_MODE = "analytic-leave-one-out-block8-v1"


@dataclass(frozen=True, slots=True)
class TileObservation:
    first_row: int
    samples: NDArray[np.float32]
    accepted: NDArray[np.bool_]
    weights: NDArray[np.float64]
    integrated: NDArray[np.float32]
    # Per-sample region weights (frame-major, the layout of ``samples``) when
    # the integration applied region weight maps; None otherwise.
    sample_weights: NDArray[np.float32] | None = None


@dataclass(frozen=True, slots=True)
class FrameCounterfactual:
    index: int
    path: str
    weight: float
    delta_depth_mag: float | None
    delta_depth_ci: tuple[float, float] | None
    delta_background_sigma: float | None
    delta_background_ci: tuple[float, float] | None
    delta_fwhm_px: float | None

    def serializable(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "path": self.path,
            "weight": self.weight,
            "deltaDepthMag": self.delta_depth_mag,
            "deltaDepthCi": list(self.delta_depth_ci) if self.delta_depth_ci else None,
            "deltaBackgroundSigma": self.delta_background_sigma,
            "deltaBackgroundCi": (
                list(self.delta_background_ci) if self.delta_background_ci else None
            ),
            "deltaFwhmPx": self.delta_fwhm_px,
        }


@dataclass(frozen=True, slots=True)
class CounterfactualReport:
    mode: str
    frames: tuple[FrameCounterfactual, ...]
    tiles_observed: int
    tiles_used: int
    block: int
    sigma_block_all: float | None
    background_rms_all_sigma: float | None
    fwhm_all_px: float | None
    bootstrap: int
    rejection_mask_source: str = "full-stack"

    def serializable(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "tilesObserved": self.tiles_observed,
            "tilesUsed": self.tiles_used,
            "block": self.block,
            "sigmaBlockAll": self.sigma_block_all,
            "backgroundRmsAllSigma": self.background_rms_all_sigma,
            "fwhmAllPx": self.fwhm_all_px,
            "bootstrap": self.bootstrap,
            "rejectionMaskSource": self.rejection_mask_source,
            "signConvention": "positive = master improves when the frame is removed",
            "frames": [frame.serializable() for frame in self.frames],
        }


def _madn(values: NDArray[np.floating]) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    median = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - median)))


def _plane_residual_madn(
    means: NDArray[np.float64], design: NDArray[np.float64]
) -> NDArray[np.float64]:
    """MADN of the plane residual for every column of ``means`` (blocks x sets)."""

    coefficients, *_ = np.linalg.lstsq(design, means, rcond=None)
    residual = means - design @ coefficients
    median = np.median(residual, axis=0)
    return 1.4826 * np.median(np.abs(residual - median[None, :]), axis=0)


class LeaveOneOutAccumulator:
    """Tile observer that accumulates leave-one-out master statistics."""

    def __init__(
        self,
        paths: Sequence[str],
        *,
        block: int = 8,
        background_segment_blocks: int = 32,
        star_sigma: float = 5.0,
        minimum_blocks: int = 24,
        statistics_rows: int = 64,
    ) -> None:
        if block < 2:
            raise ValueError("block must be at least 2")
        if statistics_rows < block:
            raise ValueError("statistics_rows must be at least one block")
        self.paths = tuple(str(path) for path in paths)
        self.block = int(block)
        # Integration bands can be thousands of rows; statistics are gathered
        # on fixed-height sub-tiles so tile counts (and the tile bootstrap) do
        # not depend on the memory budget that sized the bands.
        self.statistics_rows = int(statistics_rows)
        self.background_segment_blocks = int(background_segment_blocks)
        self.star_sigma = float(star_sigma)
        self.minimum_blocks = int(minimum_blocks)
        self._depth: list[NDArray[np.float64]] = []
        self._background: list[NDArray[np.float64]] = []
        self._tile_rows: list[float] = []
        self._segment_columns: NDArray[np.float64] | None = None
        self._weights: NDArray[np.float64] | None = None
        self.tiles_observed = 0
        self.tiles_used = 0

    def __call__(self, observation: TileObservation) -> None:
        samples = observation.samples
        accepted = observation.accepted
        weights = np.asarray(observation.weights, dtype=np.float64)
        frames, rows, width = samples.shape
        if frames != len(self.paths):
            raise ValueError("observation frame count does not match the accumulator")
        self.tiles_observed += 1
        if self._weights is None:
            self._weights = weights.copy()
        for start in range(0, rows, self.statistics_rows):
            stop = min(rows, start + self.statistics_rows)
            if stop - start < self.block:
                continue
            self._observe_rows(
                observation.first_row + start,
                samples[:, start:stop],
                accepted[:, start:stop],
                weights,
                observation.integrated[start:stop],
                (
                    observation.sample_weights[:, start:stop]
                    if observation.sample_weights is not None
                    else None
                ),
            )

    def _observe_rows(
        self,
        first_row: int,
        samples: NDArray[np.float32],
        accepted: NDArray[np.bool_],
        weights: NDArray[np.float64],
        integrated: NDArray[np.float32],
        sample_weights: NDArray[np.float32] | None = None,
    ) -> None:
        frames, rows, width = samples.shape
        block = self.block
        rows_b, cols_b = rows // block, width // block
        if rows_b == 0 or cols_b == 0:
            return
        used_rows, used_cols = rows_b * block, cols_b * block
        a = accepted[:, :used_rows, :used_cols]
        x = samples[:, :used_rows, :used_cols]
        if sample_weights is None:
            # Unit sample weights: the block sums count accepted samples.
            values = np.where(a, x, np.float32(0))
            membership: NDArray[Any] = a
        else:
            # Region-weighted samples: a sample contributes its region weight
            # to the block weight, exactly as the reduction kernel weights it.
            membership = np.where(a, sample_weights[:, :used_rows, :used_cols], np.float32(0))
            values = membership * x
        sums = values.reshape(frames, rows_b, block, cols_b, block).sum(
            axis=(2, 4), dtype=np.float64
        )
        counts = membership.reshape(frames, rows_b, block, cols_b, block).sum(
            axis=(2, 4), dtype=np.float64
        )
        total_sum = np.einsum("f,fij->ij", weights, sums)
        total_weight = np.einsum("f,fij->ij", weights, counts)
        numerator = np.concatenate(
            (total_sum[None], total_sum[None] - weights[:, None, None] * sums), axis=0
        )
        denominator = np.concatenate(
            (total_weight[None], total_weight[None] - weights[:, None, None] * counts),
            axis=0,
        )
        means = np.full(numerator.shape, np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=means, where=denominator > 0)

        image = integrated[:used_rows, :used_cols]
        finite = np.isfinite(image)
        finite_values = image[finite]
        if finite_values.size < self.minimum_blocks * block * block:
            return
        median = float(np.median(finite_values))
        spread = float(1.4826 * np.median(np.abs(finite_values - median)))
        threshold = median + self.star_sigma * max(spread, 1e-12)
        blocks_image = np.where(finite, image, -np.inf).reshape(rows_b, block, cols_b, block)
        block_max = blocks_image.max(axis=(1, 3))
        block_finite = finite.reshape(rows_b, block, cols_b, block).all(axis=(1, 3))
        usable = block_finite & (block_max <= threshold) & (total_weight > 0)
        usable &= np.all(np.isfinite(means), axis=0)
        if int(np.count_nonzero(usable)) < self.minimum_blocks:
            return

        row_index, col_index = np.nonzero(usable)
        design = np.column_stack(
            (
                np.ones(row_index.size, dtype=np.float64),
                col_index.astype(np.float64) / max(1, cols_b - 1),
                row_index.astype(np.float64) / max(1, rows_b - 1),
            )
        )
        selected = means[:, usable].T  # (blocks, frames + 1)
        self._depth.append(_plane_residual_madn(selected, design))

        segments = max(1, cols_b // self.background_segment_blocks)
        background = np.full((frames + 1, segments), np.nan, dtype=np.float64)
        for segment in range(segments):
            start = segment * self.background_segment_blocks
            stop = cols_b if segment == segments - 1 else start + self.background_segment_blocks
            cell = usable[:, start:stop]
            if int(np.count_nonzero(cell)) < 4:
                continue
            background[:, segment] = np.median(means[:, :, start:stop][:, cell], axis=1)
        self._background.append(background)
        self._tile_rows.append(first_row + used_rows / 2.0)
        if self._segment_columns is None or self._segment_columns.size != segments:
            centres = []
            for segment in range(segments):
                start = segment * self.background_segment_blocks
                stop = cols_b if segment == segments - 1 else start + self.background_segment_blocks
                centres.append((start + stop) / 2.0 * block)
            self._segment_columns = np.asarray(centres, dtype=np.float64)
        self.tiles_used += 1

    # -- finalisation -------------------------------------------------------

    def finalize(
        self,
        *,
        fwhm_by_frame: Sequence[float | None] | None = None,
        bootstrap: int = 200,
        seed: int = 0,
    ) -> CounterfactualReport:
        frames = len(self.paths)
        weights = (
            self._weights if self._weights is not None else np.ones(frames, dtype=np.float64)
        )
        rng = np.random.default_rng(seed)
        depth = np.asarray(self._depth, dtype=np.float64) if self._depth else np.empty((0, frames + 1))
        delta_depth: list[float | None] = [None] * frames
        depth_ci: list[tuple[float, float] | None] = [None] * frames
        sigma_all: float | None = None
        if depth.shape[0] >= 1:
            sigma_all = float(np.nanmedian(depth[:, 0]))
            with np.errstate(divide="ignore", invalid="ignore"):
                point = -2.5 * np.log10(np.nanmedian(depth[:, 1:], axis=0) / sigma_all)
            low = np.full(frames, np.nan)
            high = np.full(frames, np.nan)
            if depth.shape[0] >= 2:
                # A tile bootstrap needs at least two tiles; a single tile gives
                # a point estimate without an interval.
                draws = np.empty((bootstrap, frames), dtype=np.float64)
                for draw in range(bootstrap):
                    pick = rng.integers(0, depth.shape[0], depth.shape[0])
                    sample = depth[pick]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        draws[draw] = -2.5 * np.log10(
                            np.nanmedian(sample[:, 1:], axis=0) / np.nanmedian(sample[:, 0])
                        )
                low = np.nanpercentile(draws, 2.5, axis=0)
                high = np.nanpercentile(draws, 97.5, axis=0)
            for index in range(frames):
                if np.isfinite(point[index]):
                    delta_depth[index] = float(point[index])
                    if np.isfinite(low[index]) and np.isfinite(high[index]):
                        depth_ci[index] = (float(low[index]), float(high[index]))

        delta_background: list[float | None] = [None] * frames
        background_ci: list[tuple[float, float] | None] = [None] * frames
        background_all: float | None = None
        if self._background and self._segment_columns is not None and sigma_all:
            grid = np.stack(self._background, axis=1)  # (frames+1, tiles, segments)
            tiles_y = np.asarray(self._tile_rows, dtype=np.float64)
            cols_x = self._segment_columns
            yy, xx = np.meshgrid(tiles_y, cols_x, indexing="ij")
            valid = np.all(np.isfinite(grid), axis=0)
            if int(np.count_nonzero(valid)) >= 12:
                x = xx[valid]
                y = yy[valid]
                x = (x - x.mean()) / max(np.ptp(x), 1.0)
                y = (y - y.mean()) / max(np.ptp(y), 1.0)
                design = np.column_stack((np.ones_like(x), x, y, x * x, x * y, y * y))
                values = grid[:, valid].T  # (cells, frames+1)

                def rms_residual(design_: NDArray[np.float64], values_: NDArray[np.float64]) -> NDArray[np.float64]:
                    coefficients, *_ = np.linalg.lstsq(design_, values_, rcond=None)
                    residual = values_ - design_ @ coefficients
                    return np.sqrt(np.mean(residual * residual, axis=0))

                rms = rms_residual(design, values) / sigma_all
                background_all = float(rms[0])
                point = rms[0] - rms[1:]
                cells = values.shape[0]
                draws = np.empty((max(1, bootstrap // 2), frames), dtype=np.float64)
                for draw in range(draws.shape[0]):
                    pick = rng.integers(0, cells, cells)
                    sample = rms_residual(design[pick], values[pick]) / sigma_all
                    draws[draw] = sample[0] - sample[1:]
                low = np.percentile(draws, 2.5, axis=0)
                high = np.percentile(draws, 97.5, axis=0)
                for index in range(frames):
                    delta_background[index] = float(point[index])
                    background_ci[index] = (float(low[index]), float(high[index]))

        delta_fwhm: list[float | None] = [None] * frames
        fwhm_all: float | None = None
        if fwhm_by_frame is not None:
            fwhm = np.asarray(
                [np.nan if value is None else float(value) for value in fwhm_by_frame],
                dtype=np.float64,
            )
            known = np.isfinite(fwhm) & (weights > 0)
            if int(np.count_nonzero(known)) >= 2:
                squares = np.where(known, fwhm * fwhm, 0.0)
                w = np.where(known, weights, 0.0)
                total_w = float(w.sum())
                total_s = float((w * squares).sum())
                fwhm_all = float(np.sqrt(total_s / total_w))
                for index in range(frames):
                    if not known[index]:
                        continue
                    rest_w = total_w - w[index]
                    if rest_w <= 0:
                        continue
                    rest = float(np.sqrt((total_s - w[index] * squares[index]) / rest_w))
                    delta_fwhm[index] = fwhm_all - rest

        report_frames = tuple(
            FrameCounterfactual(
                index=index,
                path=self.paths[index],
                weight=float(weights[index]),
                delta_depth_mag=delta_depth[index],
                delta_depth_ci=depth_ci[index],
                delta_background_sigma=delta_background[index],
                delta_background_ci=background_ci[index],
                delta_fwhm_px=delta_fwhm[index],
            )
            for index in range(frames)
        )
        return CounterfactualReport(
            mode=COUNTERFACTUAL_MODE,
            frames=report_frames,
            tiles_observed=self.tiles_observed,
            tiles_used=self.tiles_used,
            block=self.block,
            sigma_block_all=sigma_all,
            background_rms_all_sigma=background_all,
            fwhm_all_px=fwhm_all,
            bootstrap=bootstrap,
        )
