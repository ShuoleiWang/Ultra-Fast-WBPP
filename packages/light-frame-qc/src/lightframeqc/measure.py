"""SEP-based measurements on bounded light-frame previews."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import hashlib
import math
import os
from pathlib import Path
import re
import secrets
from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import sep

from .config import DEFAULT_CONFIG, QcConfig
from .identity import (
    FileIdentityError,
    compute_file_identity,
    verify_file_identity_stat,
)
from .models import FrameMeasurement, FrameMetadata, Star
from .readers import (
    DEFAULT_MAX_FULL_DECODE_BYTES,
    FrameReadError,
    ImagePreview,
    probe_frame_metadata,
    read_frame_preview,
)


class FrameMeasurementError(RuntimeError):
    """A preview could not be measured with the requested safe settings."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class MeasurementSettings:
    """Direct, testable settings for one SEP measurement."""

    detection_sigma: float = 4.5
    minimum_source_pixels: int = 5
    minimum_source_support_pixels: int = 3
    maximum_stars: int = 2500
    grid_rows: int = 16
    grid_columns: int = 16

    def validate(self) -> None:
        if not math.isfinite(self.detection_sigma) or not 1.0 <= self.detection_sigma <= 20.0:
            raise ValueError("detection_sigma must be finite and in [1, 20]")
        if isinstance(self.minimum_source_pixels, bool) or self.minimum_source_pixels < 1:
            raise ValueError("minimum_source_pixels must be positive")
        if (
            isinstance(self.minimum_source_support_pixels, bool)
            or not isinstance(self.minimum_source_support_pixels, int)
            or self.minimum_source_support_pixels < 1
        ):
            raise ValueError("minimum_source_support_pixels must be a positive integer")
        if isinstance(self.maximum_stars, bool) or self.maximum_stars < 1:
            raise ValueError("maximum_stars must be positive")
        if self.grid_rows < 1 or self.grid_columns < 1:
            raise ValueError("grid dimensions must be positive")

    @classmethod
    def from_config(cls, config: QcConfig) -> "MeasurementSettings":
        return cls(
            detection_sigma=config.detection_sigma,
            minimum_source_pixels=config.minimum_source_pixels,
            minimum_source_support_pixels=config.minimum_source_support_pixels,
            maximum_stars=config.maximum_stars,
            grid_rows=config.grid_rows,
            grid_columns=config.grid_columns,
        )


def _native_float32(image: NDArray[Any]) -> NDArray[np.float32]:
    # A private writable copy keeps SEP isolated from ImagePreview's immutable
    # public array and guarantees native byte order/C contiguity.
    return np.array(image, dtype=np.float32, order="C", copy=True)


def _median_mad(values: NDArray[Any]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise FrameMeasurementError("NO_FINITE_PIXELS", "preview contains no finite samples")
    location = float(np.median(finite))
    dispersion = float(np.median(np.abs(finite - location)))
    return location, dispersion


def _background_mesh_size(length: int) -> int:
    # SEP accepts partial edge boxes.  Keeping at least four boxes along a
    # normal preview dimension avoids fitting the background to individual
    # stars while remaining valid for small test/ROI images.
    return max(8, min(64, max(8, length // 4)))


def _background_filter_size(length: int, box_size: int) -> int:
    return 3 if math.ceil(length / box_size) >= 3 else 1


def _grid(
    values: NDArray[Any],
    valid: NDArray[np.bool_],
    *,
    rows: int,
    columns: int,
    metric: Callable[[NDArray[np.float64]], float],
) -> list[list[float | None]]:
    height, width = values.shape
    y_edges = np.linspace(0, height, rows + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, columns + 1, dtype=np.int64)
    result: list[list[float | None]] = []
    for row in range(rows):
        cells: list[float | None] = []
        for column in range(columns):
            y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
            x0, x1 = int(x_edges[column]), int(x_edges[column + 1])
            if y1 <= y0 or x1 <= x0:
                cells.append(None)
                continue
            cell_values = np.asarray(values[y0:y1, x0:x1], dtype=np.float64)
            cell_valid = valid[y0:y1, x0:x1] & np.isfinite(cell_values)
            selected = cell_values[cell_valid]
            if selected.size == 0:
                cells.append(None)
                continue
            measured = float(metric(selected))
            cells.append(measured if math.isfinite(measured) else None)
        result.append(cells)
    return result


def _robust_sigma(values: NDArray[np.float64]) -> float:
    center = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - center)))


def _stars_from_sep(objects: NDArray[Any], maximum_stars: int) -> tuple[list[Star], int]:
    candidates: list[Star] = []
    column_names = objects.dtype.names or ()
    fwhm_factor = 2.0 * math.sqrt(2.0 * math.log(2.0))
    for item in objects:
        x = float(item["x"])
        y = float(item["y"])
        flux = float(item["flux"])
        peak = float(item["peak"])
        a = float(item["a"])
        b = float(item["b"])
        theta = float(item["theta"])
        if not all(math.isfinite(value) for value in (x, y, flux, peak, a, b, theta)):
            continue
        if flux <= 0.0 or a <= 0.0 or b <= 0.0:
            continue
        major = max(a, b)
        minor = min(a, b)
        candidates.append(
            Star(
                x=x,
                y=y,
                flux=flux,
                peak=peak,
                a=major,
                b=minor,
                theta=theta,
                fwhm=fwhm_factor * math.sqrt(major * minor),
                ellipticity=max(0.0, min(1.0, 1.0 - minor / major)),
                flags=int(item["flag"]),
                support_pixels=int(item["tnpix"]) if "tnpix" in column_names else None,
                detection_pixels=int(item["npix"]) if "npix" in column_names else None,
            )
        )
    candidates.sort(key=lambda star: (-star.flux, star.y, star.x))
    return candidates[:maximum_stars], len(candidates)


def measure_preview(
    preview: ImagePreview,
    *,
    settings: MeasurementSettings | None = None,
    thumbnail_path: str | os.PathLike[str] | None = None,
) -> FrameMeasurement:
    """Measure background, stars and a 16x16 texture grid on one preview.

    Star coordinates are always preview pixels in ``(x, y)`` order.  The
    source/preview relationship remains available through
    :class:`~lightframeqc.readers.ImagePreview`.
    """

    selected = settings or MeasurementSettings()
    selected.validate()
    height, width = preview.data.shape
    if height < 16 or width < 16:
        raise FrameMeasurementError(
            "PREVIEW_TOO_SMALL", f"preview is {width}x{height}; at least 16x16 is required"
        )

    image = _native_float32(preview.data)
    invalid = ~np.isfinite(image)
    image_median, image_mad = _median_mad(image)
    finite_fraction = 1.0 - float(np.count_nonzero(invalid)) / image.size
    if finite_fraction < 0.10:
        raise FrameMeasurementError(
            "TOO_FEW_FINITE_PIXELS",
            f"only {finite_fraction:.3%} of preview pixels are finite",
        )
    finite_values = np.asarray(image[~invalid], dtype=np.float64)
    image_p001, image_p999 = (
        float(value) for value in np.percentile(finite_values, (0.1, 99.9))
    )
    dynamic_range = max(0.0, image_p999 - image_p001)
    image[invalid] = np.float32(image_median)

    box_width = _background_mesh_size(width)
    box_height = _background_mesh_size(height)
    try:
        background = sep.Background(
            image,
            mask=invalid,
            bw=box_width,
            bh=box_height,
            fw=_background_filter_size(width, box_width),
            fh=_background_filter_size(height, box_height),
        )
        background_map = np.ascontiguousarray(background.back(), dtype=np.float32)
        rms_map = np.ascontiguousarray(background.rms(), dtype=np.float32)
    except Exception as error:
        raise FrameMeasurementError("SEP_BACKGROUND_FAILED", str(error)) from error

    residual = np.ascontiguousarray(image - background_map, dtype=np.float32)
    residual_finite = residual[~invalid]
    residual_sigma = _robust_sigma(np.asarray(residual_finite, dtype=np.float64))
    numerical_floor = max(
        np.finfo(np.float32).eps * max(1.0, abs(image_median)),
        1.0e-12,
    )
    noise_floor = max(float(background.globalrms), residual_sigma, numerical_floor)
    bad_rms = ~np.isfinite(rms_map) | (rms_map <= numerical_floor)
    if np.any(bad_rms):
        rms_map[bad_rms] = np.float32(noise_floor)

    try:
        objects, segmentation = sep.extract(
            residual,
            selected.detection_sigma,
            err=rms_map,
            mask=invalid,
            minarea=selected.minimum_source_pixels,
            segmentation_map=True,
        )
    except Exception as error:
        raise FrameMeasurementError("SEP_EXTRACTION_FAILED", str(error)) from error

    # SEP minarea applies after the detection filter: a single bright pixel
    # can produce a nine-pixel object. Require independently measured support
    # in the unconvolved data for the astrometric/photometric catalog. Preserve
    # raw detections for fragmented-trail morphology, using this same extract.
    all_stars, raw_detected_source_count = _stars_from_sep(objects, len(objects))
    supported_stars = [
        star
        for star in all_stars
        if star.support_pixels is None
        or star.support_pixels >= selected.minimum_source_support_pixels
    ]
    detected_source_count = len(supported_stars)
    stars = supported_stars[: selected.maximum_stars]
    raw_stars = all_stars[: selected.maximum_stars]
    background_valid = ~invalid
    star_free = background_valid & (segmentation == 0)
    background_grid = _grid(
        background_map,
        background_valid,
        rows=selected.grid_rows,
        columns=selected.grid_columns,
        metric=lambda values: float(np.median(values)),
    )
    texture_grid = _grid(
        residual,
        star_free,
        rows=selected.grid_rows,
        columns=selected.grid_columns,
        metric=_robust_sigma,
    )

    written_thumbnail: str | None = None
    if thumbnail_path is not None:
        written_thumbnail = save_thumbnail_png(preview, thumbnail_path)

    return FrameMeasurement(
        metadata=preview.metadata,
        stars=stars,
        detected_source_count=detected_source_count,
        raw_stars=raw_stars,
        raw_detected_source_count=raw_detected_source_count,
        preview_width=width,
        preview_height=height,
        preview_scale_x=preview.scale_x,
        preview_scale_y=preview.scale_y,
        finite_fraction=finite_fraction,
        image_p001=image_p001,
        image_p999=image_p999,
        dynamic_range=dynamic_range,
        image_median=image_median,
        image_mad=image_mad,
        background_grid=background_grid,
        texture_grid=texture_grid,
        thumbnail_path=written_thumbnail,
        reader_backend=preview.reader_backend,
        status="MEASURED",
    )


def _thumbnail_pixels(
    preview: ImagePreview,
    *,
    lower_percentile: float,
    upper_percentile: float,
    asinh_strength: float,
) -> NDArray[np.uint8]:
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("thumbnail percentiles must satisfy 0 <= low < high <= 100")
    if not math.isfinite(asinh_strength) or asinh_strength <= 0.0:
        raise ValueError("asinh_strength must be positive")
    data = np.asarray(preview.data, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise FrameMeasurementError("NO_FINITE_PIXELS", "cannot render an empty thumbnail")
    low, high = np.percentile(finite, (lower_percentile, upper_percentile))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)):
        raise FrameMeasurementError("THUMBNAIL_RANGE", "thumbnail range is non-finite")
    if high <= low:
        high = low + max(abs(float(low)) * 1.0e-6, 1.0e-6)
    normalized = np.nan_to_num(
        (data.astype(np.float64) - low) / (high - low),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    np.clip(normalized, 0.0, 1.0, out=normalized)
    stretched = np.arcsinh(asinh_strength * normalized) / math.asinh(asinh_strength)
    return np.asarray(np.rint(stretched * 255.0), dtype=np.uint8)


def save_thumbnail_png(
    preview: ImagePreview,
    output_path: str | os.PathLike[str],
    *,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    asinh_strength: float = 10.0,
) -> str:
    """Write a new grayscale PNG without ever overwriting an existing path."""

    destination = Path(output_path).expanduser()
    if destination.suffix.casefold() != ".png":
        raise FrameMeasurementError("THUMBNAIL_FORMAT", "thumbnail path must end in .png")
    source = Path(preview.path).resolve(strict=True)
    resolved_destination = destination.resolve(strict=False)
    if resolved_destination == source:
        raise FrameMeasurementError(
            "THUMBNAIL_IS_INPUT", "thumbnail destination resolves to the input frame"
        )
    if destination.exists():
        raise FrameMeasurementError(
            "THUMBNAIL_EXISTS", f"refusing to overwrite {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    pixels = _thumbnail_pixels(
        preview,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        asinh_strength=asinh_strength,
    )
    encoded = BytesIO()
    Image.fromarray(pixels, mode="L").save(encoded, format="PNG", optimize=False)
    try:
        with destination.open("xb") as stream:
            stream.write(encoded.getbuffer())
            stream.flush()
    except FileExistsError as error:
        raise FrameMeasurementError(
            "THUMBNAIL_EXISTS", f"refusing to overwrite {destination}"
        ) from error
    return str(destination.resolve(strict=True))


def measure_frame(
    path: str | os.PathLike[str],
    *,
    config: QcConfig | None = None,
    thumbnail_path: str | os.PathLike[str] | None = None,
    image_index: int = 0,
    max_full_decode_bytes: int | None = DEFAULT_MAX_FULL_DECODE_BYTES,
) -> FrameMeasurement:
    """Read and measure one frame, raising a coded error on failure."""

    selected_config = config or DEFAULT_CONFIG
    selected_config.validate()
    identity = compute_file_identity(path)
    try:
        preview = read_frame_preview(
            path,
            max_long_edge=selected_config.preview_long_edge,
            image_index=image_index,
            max_full_decode_bytes=max_full_decode_bytes,
        )
        measurement = measure_preview(
            preview,
            settings=MeasurementSettings.from_config(selected_config),
            thumbnail_path=thumbnail_path,
        )
    except Exception as error:
        try:
            verify_file_identity_stat(path, identity)
        except FileIdentityError as changed:
            raise FrameMeasurementError(
                "FILE_CHANGED_DURING_MEASUREMENT", changed.detail
            ) from changed
        raise

    try:
        verify_file_identity_stat(path, identity)
    except FileIdentityError as error:
        raise FrameMeasurementError(
            "FILE_CHANGED_DURING_MEASUREMENT", error.detail
        ) from error
    measurement.identity = identity
    return measurement


def _error_measurement(path: str | os.PathLike[str], error: Exception) -> FrameMeasurement:
    absolute = str(Path(path).expanduser().absolute())
    code = getattr(error, "code", error.__class__.__name__.upper())
    detail = getattr(error, "detail", str(error))
    try:
        metadata = probe_frame_metadata(path)
    except (FrameReadError, OSError, ValueError):
        metadata = FrameMetadata(path=absolute)
    try:
        identity = compute_file_identity(path)
    except (FileIdentityError, OSError, ValueError):
        identity = None
    return FrameMeasurement(
        metadata=metadata,
        status="ERROR",
        error_code=str(code),
        error_message=str(detail),
        identity=identity,
    )


def measure_frame_safe(
    path: str | os.PathLike[str],
    *,
    config: QcConfig | None = None,
    thumbnail_path: str | os.PathLike[str] | None = None,
    image_index: int = 0,
    max_full_decode_bytes: int | None = DEFAULT_MAX_FULL_DECODE_BYTES,
) -> FrameMeasurement:
    """Measure one frame while representing expected decode/SEP errors as data."""

    try:
        return measure_frame(
            path,
            config=config,
            thumbnail_path=thumbnail_path,
            image_index=image_index,
            max_full_decode_bytes=max_full_decode_bytes,
        )
    except (
        FileIdentityError,
        FrameReadError,
        FrameMeasurementError,
        OSError,
        ValueError,
    ) as error:
        return _error_measurement(path, error)


def measurement_star_catalog(measurement: FrameMeasurement) -> dict[str, NDArray[np.float64]]:
    """Project model stars to the catalog shape accepted by registration.py."""

    return {
        "x": np.asarray([star.x for star in measurement.stars], dtype=np.float64),
        "y": np.asarray([star.y for star in measurement.stars], dtype=np.float64),
        "flux": np.asarray([star.flux for star in measurement.stars], dtype=np.float64),
    }


def _thumbnail_name(run_token: str, index: int, path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", path.stem).strip("-._") or "frame"
    stem = stem[:64]
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{run_token}-{index:06d}-{stem}-{digest}.png"


def measure_paths(
    paths: Iterable[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    config: QcConfig,
    workers: int = 1,
) -> list[FrameMeasurement]:
    """Measure paths in stable order, optionally with bounded parallelism.

    The default is deliberately one frame at a time.  Increasing ``workers``
    permits that many simultaneous frame decoders and therefore multiplies the
    memory ceiling by approximately the same factor.  Per-frame failures are
    returned with ``status='ERROR'`` so a long acquisition is fully audited.
    """

    config.validate()
    if isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    ordered_paths = [Path(path).expanduser() for path in paths]
    if not ordered_paths:
        return []

    report_root = Path(output_dir).expanduser()
    thumbnail_root: Path | None = None
    if config.make_thumbnails:
        thumbnail_root = report_root / "thumbnails"
        thumbnail_root.mkdir(parents=True, exist_ok=True)
    # A fresh token makes rerunning into an existing report directory safe.
    # Old review thumbnails remain recoverable and no file is overwritten.
    run_token = secrets.token_hex(5)

    def run(item: tuple[int, Path]) -> FrameMeasurement:
        index, frame_path = item
        thumbnail = (
            thumbnail_root / _thumbnail_name(run_token, index, frame_path)
            if thumbnail_root is not None
            else None
        )
        return measure_frame_safe(
            frame_path,
            config=config,
            thumbnail_path=thumbnail,
        )

    indexed = list(enumerate(ordered_paths))
    if workers == 1:
        return [run(item) for item in indexed]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lightframeqc") as pool:
        return list(pool.map(run, indexed))


__all__ = [
    "FrameMeasurementError",
    "MeasurementSettings",
    "measure_frame",
    "measure_frame_safe",
    "measure_paths",
    "measure_preview",
    "measurement_star_catalog",
    "save_thumbnail_png",
]
