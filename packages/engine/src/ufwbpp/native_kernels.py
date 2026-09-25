"""ctypes bridge for the portable multithreaded CPU kernels.

The native library (the repository's ``native/`` sources, installed as a
Release build into ``ufwbpp/native``) exports kernels that reproduce the
NumPy reference arithmetic of the ordinary mono pipeline value for value:

* ``warp_lanczos3``: normalized, domain-bounded 6x6 Lanczos-3 affine warp,
* ``mad_rejection``: full-stack per-pixel median/MAD sigma clipping with the
  v2 scale model (row-pooled MAD, per-frame noise factors),
* ``masked_weighted_mean``: exact Float64 frame-order weighted mean,
* ``tile_offsets``: per-tile additive normalization offsets,
* ``add_offset_grid``: the bilinear normalization offset grid added to rows,
* ``radon_line_peaks``: multi-scale fast-Radon line peaks of the transient
  trail detector.

Every caller keeps its NumPy implementation as the portable fallback.  The
kernels are optional: a missing library, a library without the symbols, an ABI
mismatch, or ``UFWBPP_DISABLE_NATIVE_KERNELS=1`` leaves the Python path
in charge.  Because ctypes releases the GIL for the duration of a call, several
Python worker threads can run kernels concurrently, and each kernel additionally
splits its own work across ``threads`` native threads.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import stat
import threading
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


NATIVE_ABI_VERSION = 1
DISABLE_ENVIRONMENT_VARIABLE = "UFWBPP_DISABLE_NATIVE_KERNELS"
WARP_KERNEL_ID = "native-cpu-lanczos3-warp-v3-table2048"
MAD_KERNEL_ID = "native-cpu-mad-rejection-v2"
MEAN_KERNEL_ID = "native-cpu-masked-mean-v1"
TILE_OFFSET_KERNEL_ID = "native-cpu-tile-offsets-v1"
RADON_KERNEL_ID = "native-cpu-radon-peaks-v1"
DRIZZLE_KERNEL_ID = "native-cpu-drizzle-v1"
DEBAYER_KERNEL_ID = "native-cpu-debayer-bilinear-v1"
OFFSET_GRID_KERNEL_ID = "native-cpu-offset-grid-v1"
DRIZZLE_KERNELS = {"square": 0, "circular": 1, "gaussian": 2, "point": 3}
_MAXIMUM_KERNEL_THREADS = 64


class NativeKernelError(RuntimeError):
    """A native kernel could not be loaded or refused a request."""


def _library_filename() -> str:
    from .platform import current

    return current().native_library_filename()


def candidate_library_paths(explicit: str | os.PathLike[str] | None) -> tuple[Path, ...]:
    """Return trusted, existing native-library candidates in priority order."""

    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    environment = os.environ.get("UFWBPP_NATIVE_LIBRARY", "").strip()
    if environment:
        candidates.append(Path(environment).expanduser())
    # Only the installed Release library: the unoptimized test build in
    # build/native is 6-7x slower and must never serve a real run.
    candidates.append(Path(__file__).resolve().parent / "native" / _library_filename())
    found = ctypes.util.find_library("ufwbpp_native")
    if found and os.path.isabs(found):
        candidates.append(Path(found))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if not resolved.is_file():
            continue
        if os.name != "nt":
            metadata = resolved.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o022:
                continue
            if metadata.st_uid not in {0, os.getuid()}:
                continue
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return tuple(unique)


class _WarpRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("source_width", ctypes.c_uint32),
        ("source_height", ctypes.c_uint32),
        ("output_width", ctypes.c_uint32),
        ("first_row", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("source_samples", ctypes.POINTER(ctypes.c_float)),
        ("source_sample_count", ctypes.c_size_t),
        ("inverse", ctypes.c_double * 6),
        ("domain_scale", ctypes.c_float),
        ("reserved_scale", ctypes.c_float),
    ]


class _WarpRequestV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("source_width", ctypes.c_uint32),
        ("source_height", ctypes.c_uint32),
        ("output_width", ctypes.c_uint32),
        ("first_row", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("source_samples", ctypes.POINTER(ctypes.c_float)),
        ("source_sample_count", ctypes.c_size_t),
        ("inverse", ctypes.c_double * 9),
        ("domain_scale", ctypes.c_float),
        ("reserved_scale", ctypes.c_float),
    ]


class _MadRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("frame_count", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("minimum_rejection_frames", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("frame_major_samples", ctypes.POINTER(ctypes.c_float)),
        ("sample_count", ctypes.c_size_t),
        ("sigma_clip", ctypes.c_float),
        ("group_sigma_floor", ctypes.c_float),
        ("absolute_floor", ctypes.c_float),
        ("epsilon_floor", ctypes.c_float),
    ]


class _MadRequestV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("frame_count", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("minimum_rejection_frames", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("frame_major_samples", ctypes.POINTER(ctypes.c_float)),
        ("sample_count", ctypes.c_size_t),
        ("sigma_clip", ctypes.c_float),
        ("group_sigma_floor", ctypes.c_float),
        ("absolute_floor", ctypes.c_float),
        ("epsilon_floor", ctypes.c_float),
        ("frame_scales", ctypes.POINTER(ctypes.c_float)),
        ("frame_scale_count", ctypes.c_size_t),
        ("pool_half_width", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _MeanRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("frame_count", ctypes.c_uint32),
        ("row_count", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("frame_major_samples", ctypes.POINTER(ctypes.c_float)),
        ("sample_count", ctypes.c_size_t),
        ("frame_major_accepted", ctypes.POINTER(ctypes.c_uint8)),
        ("accepted_count", ctypes.c_size_t),
        ("frame_weights", ctypes.POINTER(ctypes.c_double)),
        ("weight_count", ctypes.c_size_t),
    ]


class _MeanRequestV2(ctypes.Structure):
    _fields_ = [
        *_MeanRequestV1._fields_,
        ("frame_major_sample_weights", ctypes.POINTER(ctypes.c_float)),
        ("sample_weight_count", ctypes.c_size_t),
    ]


class _MeanOutputV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("integrated", ctypes.POINTER(ctypes.c_float)),
        ("accepted_samples", ctypes.POINTER(ctypes.c_uint16)),
        ("rejected_samples", ctypes.POINTER(ctypes.c_uint16)),
        ("pixel_capacity", ctypes.c_size_t),
    ]


class _TileOffsetRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("tile_count", ctypes.c_uint32),
        ("minimum_samples", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("target", ctypes.POINTER(ctypes.c_double)),
        ("reference", ctypes.POINTER(ctypes.c_double)),
        ("sample_count", ctypes.c_size_t),
        ("boundaries", ctypes.POINTER(ctypes.c_uint64)),
        ("boundary_count", ctypes.c_size_t),
        ("scale", ctypes.c_double),
        ("lower_quantile", ctypes.c_double),
        ("upper_quantile", ctypes.c_double),
        ("residual_clip_sigma", ctypes.c_double),
    ]


class _TileOffsetOutputV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("offset", ctypes.POINTER(ctypes.c_double)),
        ("count", ctypes.POINTER(ctypes.c_uint32)),
        ("residual_mad", ctypes.POINTER(ctypes.c_double)),
        ("valid", ctypes.POINTER(ctypes.c_uint8)),
        ("tile_capacity", ctypes.c_size_t),
    ]


class _RadonPeakRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("minimum_rows", ctypes.c_uint32),
        ("minimum_scale_samples", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("detection_z", ctypes.c_float),
        ("image", ctypes.POINTER(ctypes.c_float)),
        ("image_count", ctypes.c_size_t),
        ("weight", ctypes.POINTER(ctypes.c_uint8)),
        ("weight_count", ctypes.c_size_t),
        ("minimum_coverage", ctypes.c_double),
        ("minimum_count", ctypes.c_double),
    ]


_RADON_PEAK_DTYPE = np.dtype(
    [
        ("level", np.uint32),
        ("block", np.uint32),
        ("shift_index", np.uint32),
        ("column", np.uint32),
        ("z", np.float32),
        ("reserved", np.uint32),
    ]
)
_RADON_PEAK_INITIAL_CAPACITY = 4096


class _RadonPeakOutputV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("peaks", ctypes.c_void_p),
        ("peak_capacity", ctypes.c_size_t),
        ("peak_count", ctypes.c_size_t),
    ]


class _DebayerRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("pattern", ctypes.c_uint8 * 4),
        ("mosaic", ctypes.POINTER(ctypes.c_float)),
        ("mosaic_count", ctypes.c_size_t),
        ("planes", ctypes.POINTER(ctypes.c_float)),
        ("plane_count", ctypes.c_size_t),
    ]


class _DrizzleRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("source_width", ctypes.c_uint32),
        ("source_rows", ctypes.c_uint32),
        ("source_row0", ctypes.c_uint32),
        ("scale", ctypes.c_uint32),
        ("kernel", ctypes.c_uint32),
        ("output_width", ctypes.c_uint32),
        ("output_rows", ctypes.c_uint32),
        ("output_row0", ctypes.c_uint32),
        ("mask_width", ctypes.c_uint32),
        ("mask_height", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("cfa_pattern", ctypes.c_uint8 * 4),
        ("channel", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
        ("normalization_scale", ctypes.c_float),
        ("normalization_offset", ctypes.c_float),
        ("frame_weight", ctypes.c_float),
        ("reserved_float", ctypes.c_float),
        ("pixfrac", ctypes.c_double),
        ("forward", ctypes.c_double * 9),
        ("source", ctypes.POINTER(ctypes.c_float)),
        ("source_count", ctypes.c_size_t),
        ("grid", ctypes.POINTER(ctypes.c_double)),
        ("grid_count", ctypes.c_size_t),
        ("grid_x_nodes", ctypes.POINTER(ctypes.c_double)),
        ("grid_x_count", ctypes.c_size_t),
        ("grid_y_nodes", ctypes.POINTER(ctypes.c_double)),
        ("grid_y_count", ctypes.c_size_t),
        ("weight_grid", ctypes.POINTER(ctypes.c_double)),
        ("weight_grid_count", ctypes.c_size_t),
        ("weight_grid_x_nodes", ctypes.POINTER(ctypes.c_double)),
        ("weight_grid_x_count", ctypes.c_size_t),
        ("weight_grid_y_nodes", ctypes.POINTER(ctypes.c_double)),
        ("weight_grid_y_count", ctypes.c_size_t),
        ("mask", ctypes.POINTER(ctypes.c_uint8)),
        ("mask_count", ctypes.c_size_t),
        ("output_sum", ctypes.POINTER(ctypes.c_double)),
        ("output_weight", ctypes.POINTER(ctypes.c_double)),
        ("output_count", ctypes.c_size_t),
        ("output_touched", ctypes.POINTER(ctypes.c_uint8)),
    ]


class _OffsetGridRequestV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("threads", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("values", ctypes.POINTER(ctypes.c_float)),
        ("value_count", ctypes.c_size_t),
        ("rows", ctypes.POINTER(ctypes.c_int64)),
        ("row_count", ctypes.c_size_t),
        ("grid", ctypes.POINTER(ctypes.c_double)),
        ("grid_count", ctypes.c_size_t),
        ("x_nodes", ctypes.POINTER(ctypes.c_double)),
        ("x_node_count", ctypes.c_size_t),
        ("y_nodes", ctypes.POINTER(ctypes.c_double)),
        ("y_node_count", ctypes.c_size_t),
    ]


class _CpuFeaturesV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("architecture", ctypes.c_uint32),
        ("features", ctypes.c_char * 256),
        ("brand", ctypes.c_char * 64),
    ]


_CPU_ARCHITECTURES = {0: "unknown", 1: "x86-64", 2: "arm64"}


_ERROR_BYTES = 1024
_REQUIRED_SYMBOLS = (
    "ufwbpp_native_abi_version",
    "ufwbpp_native_cpu_warp_lanczos3_v1",
    "ufwbpp_native_cpu_warp_lanczos3_v2",
    "ufwbpp_native_cpu_mad_rejection_v1",
    "ufwbpp_native_cpu_mad_rejection_v2",
    "ufwbpp_native_cpu_masked_mean_v1",
    "ufwbpp_native_cpu_tile_offsets_v1",
    "ufwbpp_native_cpu_radon_peaks_v1",
    "ufwbpp_native_cpu_drizzle_v1",
    "ufwbpp_native_cpu_debayer_bilinear_v1",
    "ufwbpp_native_cpu_add_offset_grid_v1",
    "ufwbpp_native_lanczos3_table_v1",
    "ufwbpp_native_default_kernel_threads_v1",
)


def _thread_count(value: int | None) -> int:
    if value is None:
        return default_kernel_threads()
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("native kernel thread count must be a positive integer")
    return min(int(value), _MAXIMUM_KERNEL_THREADS)


class NativeKernels:
    """Validated owner of one loaded native kernel library."""

    def __init__(self, library_path: str | os.PathLike[str]) -> None:
        self.library_path = Path(library_path).expanduser().resolve(strict=True)
        try:
            self._library = ctypes.CDLL(str(self.library_path))
        except OSError as error:
            raise NativeKernelError(
                f"unable to load native kernels from {self.library_path}: {error}"
            ) from error
        missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(self._library, name)]
        if missing:
            raise NativeKernelError(
                "native library misses portable kernel symbols: " + ", ".join(missing)
            )
        library = self._library
        library.ufwbpp_native_abi_version.argtypes = []
        library.ufwbpp_native_abi_version.restype = ctypes.c_uint32
        version = int(library.ufwbpp_native_abi_version())
        if version != NATIVE_ABI_VERSION:
            raise NativeKernelError(
                f"native ABI version {version} differs from {NATIVE_ABI_VERSION}"
            )
        error_arguments = [ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
        library.ufwbpp_native_cpu_warp_lanczos3_v1.argtypes = [
            ctypes.POINTER(_WarpRequestV1),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            *error_arguments,
        ]
        library.ufwbpp_native_cpu_warp_lanczos3_v1.restype = ctypes.c_int
        library.ufwbpp_native_cpu_warp_lanczos3_v2.argtypes = [
            ctypes.POINTER(_WarpRequestV2),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            *error_arguments,
        ]
        library.ufwbpp_native_cpu_warp_lanczos3_v2.restype = ctypes.c_int
        library.ufwbpp_native_cpu_mad_rejection_v1.argtypes = [
            ctypes.POINTER(_MadRequestV1),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            *error_arguments,
        ]
        library.ufwbpp_native_cpu_mad_rejection_v1.restype = ctypes.c_int
        library.ufwbpp_native_cpu_mad_rejection_v2.argtypes = [
            ctypes.POINTER(_MadRequestV2),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            *error_arguments,
        ]
        library.ufwbpp_native_cpu_mad_rejection_v2.restype = ctypes.c_int
        library.ufwbpp_native_cpu_masked_mean_v1.argtypes = [
            ctypes.POINTER(_MeanRequestV1),
            ctypes.POINTER(_MeanOutputV1),
            *error_arguments,
        ]
        library.ufwbpp_native_cpu_masked_mean_v1.restype = ctypes.c_int
        # Optional: per-sample weights (region weight maps); older libraries
        # lack the symbol and callers fall back to the NumPy reduction.
        self.has_sample_weight_support = hasattr(library, "ufwbpp_native_cpu_masked_mean_v2")
        if self.has_sample_weight_support:
            library.ufwbpp_native_cpu_masked_mean_v2.argtypes = [
                ctypes.POINTER(_MeanRequestV2),
                ctypes.POINTER(_MeanOutputV1),
                *error_arguments,
            ]
            library.ufwbpp_native_cpu_masked_mean_v2.restype = ctypes.c_int
        library.ufwbpp_native_cpu_tile_offsets_v1.argtypes = [
            ctypes.POINTER(_TileOffsetRequestV1),
            ctypes.POINTER(_TileOffsetOutputV1),
            *error_arguments,
        ]
        library.ufwbpp_native_cpu_tile_offsets_v1.restype = ctypes.c_int
        library.ufwbpp_native_cpu_radon_peaks_v1.argtypes = [
            ctypes.POINTER(_RadonPeakRequestV1),
            ctypes.POINTER(_RadonPeakOutputV1),
            *error_arguments,
        ]
        library.ufwbpp_native_cpu_radon_peaks_v1.restype = ctypes.c_int
        library.ufwbpp_native_cpu_drizzle_v1.argtypes = [
            ctypes.POINTER(_DrizzleRequestV1),
            *error_arguments,
        ]
        library.ufwbpp_native_cpu_drizzle_v1.restype = ctypes.c_int
        library.ufwbpp_native_cpu_add_offset_grid_v1.argtypes = [
            ctypes.POINTER(_OffsetGridRequestV1),
            *error_arguments,
        ]
        library.ufwbpp_native_cpu_add_offset_grid_v1.restype = ctypes.c_int
        library.ufwbpp_native_default_kernel_threads_v1.argtypes = []
        library.ufwbpp_native_default_kernel_threads_v1.restype = ctypes.c_uint32
        self.hardware_threads = max(1, int(library.ufwbpp_native_default_kernel_threads_v1()))
        # Optional since ABI 1 libraries built before the probe existed load too.
        self._features_probe = getattr(library, "ufwbpp_native_cpu_features_v1", None)
        if self._features_probe is not None:
            self._features_probe.argtypes = [ctypes.POINTER(_CpuFeaturesV1), *error_arguments]
            self._features_probe.restype = ctypes.c_int
        self._cpu_features: tuple[str, str, tuple[str, ...]] | None = None
        self._sha256: str | None = None

    @property
    def library_sha256(self) -> str:
        """SHA-256 of the loaded library file (receipt evidence), computed once."""

        if self._sha256 is None:
            digest = hashlib.sha256()
            with open(self.library_path, "rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            self._sha256 = digest.hexdigest()
        return self._sha256

    def _probe_cpu(self) -> tuple[str, str, tuple[str, ...]]:
        if self._cpu_features is None:
            if self._features_probe is None:
                self._cpu_features = ("unknown", "", ())
            else:
                request = _CpuFeaturesV1()
                request.struct_size = ctypes.sizeof(_CpuFeaturesV1)
                error = ctypes.create_string_buffer(_ERROR_BYTES)
                status = int(self._features_probe(ctypes.byref(request), error, ctypes.sizeof(error)))
                if status != 0:
                    self._raise(error, status, "native CPU feature probe")
                names = request.features.decode("ascii", errors="replace")
                self._cpu_features = (
                    _CPU_ARCHITECTURES.get(int(request.architecture), "unknown"),
                    request.brand.decode("ascii", errors="replace").strip(),
                    tuple(name for name in names.split(",") if name),
                )
        return self._cpu_features

    def cpu_architecture(self) -> str:
        """Architecture the library was compiled for: ``x86-64``, ``arm64``."""

        return self._probe_cpu()[0]

    def cpu_brand(self) -> str:
        """cpuid brand string (x86-64 only; empty elsewhere)."""

        return self._probe_cpu()[1]

    def cpu_features(self) -> tuple[str, ...]:
        """ISA extensions usable by the running OS, e.g. ``("sse4.2", "avx2")``."""

        return self._probe_cpu()[2]

    def describe(self) -> dict[str, Any]:
        """Receipt-ready facts about the loaded library."""

        return {
            "loaded": True,
            "libraryPath": str(self.library_path),
            "sha256": f"sha256:{self.library_sha256}",
            "abiVersion": NATIVE_ABI_VERSION,
            "hardwareThreads": self.hardware_threads,
            "cpuArchitecture": self.cpu_architecture(),
            "cpuBrand": self.cpu_brand(),
            "cpuFeatures": list(self.cpu_features()),
            "kernels": [
                WARP_KERNEL_ID, MAD_KERNEL_ID, MEAN_KERNEL_ID, TILE_OFFSET_KERNEL_ID, OFFSET_GRID_KERNEL_ID,
            ],
        }

    @staticmethod
    def _raise(buffer: ctypes.Array[Any], status: int, kernel: str) -> None:
        detail = bytes(buffer.value).decode("utf-8", errors="replace").strip()
        raise NativeKernelError(
            f"{kernel} failed with status {status}: {detail or 'no detail'}"
        )

    def warp_lanczos3(
        self,
        source: NDArray[np.float32],
        inverse: NDArray[np.float64],
        *,
        first_row: int,
        row_count: int,
        output_width: int,
        domain_scale: float,
        threads: int | None = None,
    ) -> NDArray[np.float32]:
        """Warp one band of output rows from a native Float32 source image.

        ``inverse`` is the output-to-input map: a 2x3 or 3x3 affine matrix, or
        a 3x3 projective matrix whose last row divides both coordinates.
        """

        source_values = np.ascontiguousarray(source, dtype=np.float32)
        if source_values.ndim != 2:
            raise ValueError("warp source must be a two-dimensional image")
        matrix = np.ascontiguousarray(inverse, dtype=np.float64)
        if matrix.shape == (2, 3):
            matrix = np.vstack((matrix, np.asarray([[0.0, 0.0, 1.0]])))
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("warp inverse must be a finite 2x3 or 3x3 matrix")
        if row_count < 1 or output_width < 1 or first_row < 0:
            raise ValueError("warp band geometry must be positive")
        height, width = source_values.shape
        request = _WarpRequestV2()
        request.struct_size = ctypes.sizeof(_WarpRequestV2)
        request.source_width = int(width)
        request.source_height = int(height)
        request.output_width = int(output_width)
        request.first_row = int(first_row)
        request.row_count = int(row_count)
        request.threads = _thread_count(threads)
        request.source_samples = source_values.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        request.source_sample_count = source_values.size
        request.inverse = (ctypes.c_double * 9)(*(float(value) for value in matrix.ravel()))
        request.domain_scale = float(domain_scale)
        destination = np.empty((int(row_count), int(output_width)), dtype=np.float32)
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        status = int(
            self._library.ufwbpp_native_cpu_warp_lanczos3_v2(
                ctypes.byref(request),
                destination.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                destination.size,
                error,
                ctypes.sizeof(error),
            )
        )
        if status != 0:
            self._raise(error, status, "native Lanczos-3 warp")
        return destination

    def mad_rejection(
        self,
        samples: NDArray[np.float32],
        *,
        sigma_clip: float,
        minimum_rejection_frames: int,
        group_sigma_floor: float,
        absolute_floor: float,
        epsilon_floor: float,
        frame_scales: Sequence[float] | NDArray[Any] | None = None,
        pool_half_width: int = 0,
        threads: int | None = None,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
        """Return (accepted, center) for a frame-major (F, R, W) Float32 stack.

        ``frame_scales`` (one Float32 factor per frame) and ``pool_half_width``
        select the v2 scale model; ``None``/0 reproduce the v1 per-pixel MAD
        decisions exactly.
        """

        values = np.ascontiguousarray(samples, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError("MAD rejection samples must be frame-major 3-D")
        frames, rows, width = values.shape
        if isinstance(pool_half_width, bool) or int(pool_half_width) < 0:
            raise ValueError("pool_half_width must be a nonnegative integer")
        scales: NDArray[np.float32] | None = None
        if frame_scales is not None:
            scales = np.ascontiguousarray(frame_scales, dtype=np.float32).ravel()
            if scales.size != frames:
                raise ValueError("MAD rejection frame scale count differs from the frame count")
            if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError("MAD rejection frame scales must be finite and positive")
        request = _MadRequestV2()
        request.struct_size = ctypes.sizeof(_MadRequestV2)
        request.frame_count = int(frames)
        request.row_count = int(rows)
        request.width = int(width)
        request.minimum_rejection_frames = int(minimum_rejection_frames)
        request.threads = _thread_count(threads)
        request.frame_major_samples = values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        request.sample_count = values.size
        request.sigma_clip = float(sigma_clip)
        request.group_sigma_floor = float(group_sigma_floor)
        request.absolute_floor = float(absolute_floor)
        request.epsilon_floor = float(epsilon_floor)
        if scales is not None:
            request.frame_scales = scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            request.frame_scale_count = scales.size
        else:
            request.frame_scales = ctypes.POINTER(ctypes.c_float)()
            request.frame_scale_count = 0
        request.pool_half_width = int(pool_half_width)
        request.reserved = 0
        accepted = np.empty(values.shape, dtype=np.uint8)
        center = np.empty((rows, width), dtype=np.float32)
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        status = int(
            self._library.ufwbpp_native_cpu_mad_rejection_v2(
                ctypes.byref(request),
                accepted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                accepted.size,
                center.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                center.size,
                error,
                ctypes.sizeof(error),
            )
        )
        if status != 0:
            self._raise(error, status, "native MAD rejection")
        return accepted.view(np.bool_), center

    def masked_weighted_mean(
        self,
        samples: NDArray[np.float32],
        accepted: NDArray[Any],
        weights: NDArray[np.float64],
        *,
        threads: int | None = None,
        sample_weights: NDArray[Any] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.uint16], NDArray[np.uint16]]:
        """Return (integrated, accepted_count, rejected_count) for one tile.

        ``sample_weights`` (frame-major Float32, the layout of ``samples``)
        multiplies each frame weight per sample; it requires a library with
        the V2 entry point (``has_sample_weight_support``).
        """

        values = np.ascontiguousarray(samples, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError("masked mean samples must be frame-major 3-D")
        mask = np.ascontiguousarray(accepted)
        if mask.dtype == np.bool_:
            mask = mask.view(np.uint8)
        elif mask.dtype != np.uint8:
            mask = np.ascontiguousarray(mask != 0).view(np.uint8)
        if mask.shape != values.shape:
            raise ValueError("masked mean acceptance mask must match the samples")
        frame_weights = np.ascontiguousarray(weights, dtype=np.float64).ravel()
        frames, rows, width = values.shape
        if frame_weights.size != frames:
            raise ValueError("masked mean weight count differs from the frame count")
        sample_weight_values: NDArray[np.float32] | None = None
        if sample_weights is not None:
            if not getattr(self, "has_sample_weight_support", False):
                raise NativeKernelError(
                    "the native library has no per-sample weight entry point (masked mean V2)"
                )
            sample_weight_values = np.ascontiguousarray(sample_weights, dtype=np.float32)
            if sample_weight_values.shape != values.shape:
                raise ValueError("masked mean sample weights must match the samples")
        request = _MeanRequestV2() if sample_weight_values is not None else _MeanRequestV1()
        request.struct_size = ctypes.sizeof(type(request))
        request.frame_count = int(frames)
        request.row_count = int(rows)
        request.width = int(width)
        request.threads = _thread_count(threads)
        request.frame_major_samples = values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        request.sample_count = values.size
        request.frame_major_accepted = mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        request.accepted_count = mask.size
        request.frame_weights = frame_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        request.weight_count = frame_weights.size
        if sample_weight_values is not None:
            request.frame_major_sample_weights = sample_weight_values.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            )
            request.sample_weight_count = sample_weight_values.size
        integrated = np.empty((rows, width), dtype=np.float32)
        accepted_count = np.empty((rows, width), dtype=np.uint16)
        rejected_count = np.empty((rows, width), dtype=np.uint16)
        output = _MeanOutputV1()
        output.struct_size = ctypes.sizeof(_MeanOutputV1)
        output.integrated = integrated.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output.accepted_samples = accepted_count.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint16)
        )
        output.rejected_samples = rejected_count.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint16)
        )
        output.pixel_capacity = integrated.size
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        entry = (
            self._library.ufwbpp_native_cpu_masked_mean_v2
            if sample_weight_values is not None
            else self._library.ufwbpp_native_cpu_masked_mean_v1
        )
        status = int(
            entry(
                ctypes.byref(request),
                ctypes.byref(output),
                error,
                ctypes.sizeof(error),
            )
        )
        if status != 0:
            self._raise(error, status, "native masked weighted mean")
        return integrated, accepted_count, rejected_count

    def tile_offsets(
        self,
        target: NDArray[np.float64],
        reference: NDArray[np.float64],
        boundaries: NDArray[np.uint64],
        *,
        scale: float,
        lower_quantile: float,
        upper_quantile: float,
        minimum_samples: int,
        residual_clip_sigma: float,
        threads: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.uint32], NDArray[np.float64], NDArray[np.bool_]]:
        """Return (offset, count, residual_mad, valid) per concatenated tile."""

        target_values = np.ascontiguousarray(target, dtype=np.float64).ravel()
        reference_values = np.ascontiguousarray(reference, dtype=np.float64).ravel()
        bounds = np.ascontiguousarray(boundaries, dtype=np.uint64).ravel()
        if target_values.size != reference_values.size:
            raise ValueError("tile offset target and reference sample counts differ")
        if bounds.size < 2:
            raise ValueError("tile offsets require at least one tile")
        tiles = int(bounds.size - 1)
        request = _TileOffsetRequestV1()
        request.struct_size = ctypes.sizeof(_TileOffsetRequestV1)
        request.tile_count = tiles
        request.minimum_samples = int(minimum_samples)
        request.threads = _thread_count(threads)
        request.target = target_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        request.reference = reference_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        request.sample_count = target_values.size
        request.boundaries = bounds.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        request.boundary_count = bounds.size
        request.scale = float(scale)
        request.lower_quantile = float(lower_quantile)
        request.upper_quantile = float(upper_quantile)
        request.residual_clip_sigma = float(residual_clip_sigma)
        offset = np.empty(tiles, dtype=np.float64)
        count = np.empty(tiles, dtype=np.uint32)
        residual_mad = np.empty(tiles, dtype=np.float64)
        valid = np.empty(tiles, dtype=np.uint8)
        output = _TileOffsetOutputV1()
        output.struct_size = ctypes.sizeof(_TileOffsetOutputV1)
        output.offset = offset.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        output.count = count.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        output.residual_mad = residual_mad.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        output.valid = valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        output.tile_capacity = tiles
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        status = int(
            self._library.ufwbpp_native_cpu_tile_offsets_v1(
                ctypes.byref(request), ctypes.byref(output), error, ctypes.sizeof(error)
            )
        )
        if status != 0:
            self._raise(error, status, "native tile offsets")
        return offset, count, residual_mad, valid.view(np.bool_)

    def add_offset_grid(
        self,
        values: NDArray[np.float32],
        rows: NDArray[np.int64],
        grid: NDArray[np.float64],
        x_nodes: NDArray[np.float64],
        y_nodes: NDArray[np.float64],
        *,
        threads: int | None = 1,
    ) -> None:
        """Add the bilinear offset grid to ``values`` in place.

        ``values`` is a C-contiguous Float32 ``(len(rows), width)`` array;
        ``rows`` are the frame rows of its rows.  The native kernel and
        ``calibration._add_offset_grid_rows`` produce identical values.
        """

        if values.dtype != np.float32 or values.ndim != 2 or not values.flags["C_CONTIGUOUS"]:
            raise ValueError("offset grid values must be a C-contiguous 2-D Float32 array")
        row_values = np.ascontiguousarray(rows, dtype=np.int64)
        grid_values = np.ascontiguousarray(grid, dtype=np.float64)
        x_values = np.ascontiguousarray(x_nodes, dtype=np.float64)
        y_values = np.ascontiguousarray(y_nodes, dtype=np.float64)
        if row_values.shape != (values.shape[0],) or grid_values.shape != (y_values.size, x_values.size):
            raise ValueError("offset grid rows or node grid do not match")
        request = _OffsetGridRequestV1()
        request.struct_size = ctypes.sizeof(_OffsetGridRequestV1)
        request.width = int(values.shape[1])
        request.threads = _thread_count(threads)
        request.values = values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        request.value_count = values.size
        request.rows = row_values.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        request.row_count = row_values.size
        request.grid = grid_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        request.grid_count = grid_values.size
        request.x_nodes = x_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        request.x_node_count = x_values.size
        request.y_nodes = y_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        request.y_node_count = y_values.size
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        status = int(
            self._library.ufwbpp_native_cpu_add_offset_grid_v1(
                ctypes.byref(request), error, ctypes.sizeof(error)
            )
        )
        if status != 0:
            self._raise(error, status, "native offset grid")

    def radon_line_peaks(
        self,
        image: NDArray[np.float32],
        weight: NDArray[np.uint8],
        *,
        size: int,
        minimum_rows: int,
        detection_z: float,
        minimum_coverage: float,
        minimum_count: float,
        minimum_scale_samples: int,
        threads: int | None = None,
    ) -> list[tuple[int, NDArray[np.intp], NDArray[np.intp], NDArray[np.intp], NDArray[np.float32]]]:
        """Line peaks per dyadic level: (n, blocks, shift_index, columns, z).

        ``image`` and ``weight`` are the ``(height, width)`` samples and 0/1
        weights of one orientation; ``size`` is the dyadic canvas height.
        Levels ascend and each level lists its peaks in row-major order,
        exactly the ``np.nonzero`` order of the NumPy reference.
        """

        samples = np.ascontiguousarray(image, dtype=np.float32)
        weights = np.ascontiguousarray(weight, dtype=np.uint8)
        if samples.ndim != 2 or samples.shape != weights.shape:
            raise ValueError("radon peaks need matching 2-D image and weight arrays")
        height, width = samples.shape
        request = _RadonPeakRequestV1()
        request.struct_size = ctypes.sizeof(_RadonPeakRequestV1)
        request.width = int(width)
        request.height = int(height)
        request.size = int(size)
        request.minimum_rows = int(minimum_rows)
        request.minimum_scale_samples = int(minimum_scale_samples)
        request.threads = _thread_count(threads)
        request.detection_z = float(detection_z)
        request.image = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        request.image_count = samples.size
        request.weight = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        request.weight_count = weights.size
        request.minimum_coverage = float(minimum_coverage)
        request.minimum_count = float(minimum_count)
        capacity = _RADON_PEAK_INITIAL_CAPACITY
        while True:
            peaks = np.zeros(capacity, dtype=_RADON_PEAK_DTYPE)
            output = _RadonPeakOutputV1()
            output.struct_size = ctypes.sizeof(_RadonPeakOutputV1)
            output.peaks = peaks.ctypes.data
            output.peak_capacity = capacity
            output.peak_count = 0
            error = ctypes.create_string_buffer(_ERROR_BYTES)
            status = int(
                self._library.ufwbpp_native_cpu_radon_peaks_v1(
                    ctypes.byref(request), ctypes.byref(output), error, ctypes.sizeof(error)
                )
            )
            if status == 2 and int(output.peak_count) > capacity:
                capacity = int(output.peak_count)
                continue
            if status != 0:
                self._raise(error, status, "native radon line peaks")
            break
        found = peaks[: int(output.peak_count)]
        levels: list[tuple[int, NDArray[np.intp], NDArray[np.intp], NDArray[np.intp], NDArray[np.float32]]] = []
        for level in np.unique(found["level"]):
            rows = found[found["level"] == level]
            levels.append(
                (
                    int(level),
                    rows["block"].astype(np.intp),
                    rows["shift_index"].astype(np.intp),
                    rows["column"].astype(np.intp),
                    np.ascontiguousarray(rows["z"], dtype=np.float32),
                )
            )
        return levels

    def lanczos3_table(self) -> NDArray[np.float64]:
        """The native library's deterministic Lanczos-3 weight table,
        ``(nodes, 6)`` Float64, for the identity test against the Python
        reference ``lanczos_table.node_table``."""

        from .lanczos_table import TABLE_NODES

        values = np.empty((TABLE_NODES, 6), dtype=np.float64)
        count = ctypes.c_uint32(0)
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        status = int(
            self._library.ufwbpp_native_lanczos3_table_v1(
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                values.size,
                ctypes.byref(count),
                error,
                ctypes.sizeof(error),
            )
        )
        if status != 0:
            self._raise(error, status, "native Lanczos-3 table")
        if int(count.value) != TABLE_NODES:
            raise NativeKernelError(f"native Lanczos-3 table has {count.value} nodes, expected {TABLE_NODES}")
        return values

    def debayer_bilinear(
        self,
        mosaic: NDArray[np.float32],
        pattern: Sequence[int],
        *,
        threads: int | None = None,
    ) -> NDArray[np.float32]:
        """Bilinear demosaic into ``(3, H, W)`` planes, value-identical to
        ``lightframeqc.cfa.bilinear_debayer``; ``pattern`` gives the channel
        (0 R, 1 G, 2 B) of the tile positions (0,0), (0,1), (1,0), (1,1)."""

        samples = np.ascontiguousarray(mosaic, dtype=np.float32)
        if samples.ndim != 2 or samples.size == 0:
            raise ValueError("debayer needs a nonempty 2-D mosaic")
        layout = tuple(int(value) for value in pattern)
        if len(layout) != 4 or any(value not in (0, 1, 2) for value in layout):
            raise ValueError("debayer pattern needs four channel indices in 0..2")
        planes = np.empty((3,) + samples.shape, dtype=np.float32)
        request = _DebayerRequestV1()
        request.struct_size = ctypes.sizeof(_DebayerRequestV1)
        request.width = int(samples.shape[1])
        request.height = int(samples.shape[0])
        request.threads = _thread_count(threads)
        request.pattern = (ctypes.c_uint8 * 4)(*layout)
        request.mosaic = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        request.mosaic_count = samples.size
        request.planes = planes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        request.plane_count = planes.size
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        status = int(
            self._library.ufwbpp_native_cpu_debayer_bilinear_v1(ctypes.byref(request), error, ctypes.sizeof(error))
        )
        if status != 0:
            self._raise(error, status, "native debayer")
        return planes

    def drizzle_band(
        self,
        source: NDArray[np.float32],
        *,
        source_row0: int,
        forward: NDArray[np.float64],
        scale: int,
        pixfrac: float,
        kernel: str,
        output_sum: NDArray[np.float64],
        output_weight: NDArray[np.float64],
        output_row0: int = 0,
        normalization_scale: float = 1.0,
        normalization_offset: float = 0.0,
        grid: NDArray[np.float64] | None = None,
        grid_x_nodes: NDArray[np.float64] | None = None,
        grid_y_nodes: NDArray[np.float64] | None = None,
        weight_grid: NDArray[np.float64] | None = None,
        weight_grid_x_nodes: NDArray[np.float64] | None = None,
        weight_grid_y_nodes: NDArray[np.float64] | None = None,
        mask: NDArray[np.uint8] | None = None,
        cfa_pattern: Sequence[int] = (0, 1, 1, 2),
        channel: int = 255,
        frame_weight: float = 1.0,
        threads: int | None = None,
        output_touched: NDArray[np.uint8] | None = None,
    ) -> None:
        """Drizzle the source rows onto the output band accumulators in place.

        ``forward`` maps input pixel centres to output pixel centres (the
        registration matrix times the scale); ``output_sum`` and
        ``output_weight`` are ``(rows, width)`` Float64 arrays that receive
        ``sum(w*a*v)`` and ``sum(w*a)``.  ``mask`` is the reference-grid
        acceptance mask (1 accepted), ``grid``/nodes the additive offset grid
        of the frame's normalization and ``weight_grid``/nodes an optional
        per-pixel weight multiplier (region weights).  ``output_touched``, a
        UInt8 array of the band's shape, is set to 1 wherever the frame
        contributed positive weight (it is never cleared).
        """

        samples = np.ascontiguousarray(source, dtype=np.float32)
        if samples.ndim != 2:
            raise ValueError("drizzle source must be a 2-D array")
        if output_sum.ndim != 2 or output_sum.shape != output_weight.shape:
            raise ValueError("drizzle accumulators must be matching 2-D arrays")
        if not (output_sum.flags["C_CONTIGUOUS"] and output_weight.flags["C_CONTIGUOUS"]):
            raise ValueError("drizzle accumulators must be C-contiguous")
        if output_sum.dtype != np.float64 or output_weight.dtype != np.float64:
            raise ValueError("drizzle accumulators must be Float64")
        matrix = np.ascontiguousarray(forward, dtype=np.float64).reshape(9)
        if kernel not in DRIZZLE_KERNELS:
            raise ValueError(f"unknown drizzle kernel {kernel!r}")
        request = _DrizzleRequestV1()
        request.struct_size = ctypes.sizeof(_DrizzleRequestV1)
        request.source_width = int(samples.shape[1])
        request.source_rows = int(samples.shape[0])
        request.source_row0 = int(source_row0)
        request.scale = int(scale)
        request.kernel = DRIZZLE_KERNELS[kernel]
        request.output_width = int(output_sum.shape[1])
        request.output_rows = int(output_sum.shape[0])
        request.output_row0 = int(output_row0)
        request.threads = _thread_count(threads)
        pattern = tuple(int(value) for value in cfa_pattern)
        if len(pattern) != 4:
            raise ValueError("cfa_pattern needs four channel indices")
        request.cfa_pattern = (ctypes.c_uint8 * 4)(*pattern)
        request.channel = int(channel)
        request.normalization_scale = float(normalization_scale)
        request.normalization_offset = float(normalization_offset)
        request.frame_weight = float(frame_weight)
        request.pixfrac = float(pixfrac)
        request.forward = (ctypes.c_double * 9)(*matrix.tolist())
        request.source = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        request.source_count = samples.size
        keep_alive: list[Any] = [samples]
        if grid is not None:
            grid_values = np.ascontiguousarray(grid, dtype=np.float64)
            x_nodes = np.ascontiguousarray(grid_x_nodes, dtype=np.float64)
            y_nodes = np.ascontiguousarray(grid_y_nodes, dtype=np.float64)
            if grid_values.shape != (y_nodes.size, x_nodes.size):
                raise ValueError("drizzle offset grid shape must be (y nodes, x nodes)")
            keep_alive += [grid_values, x_nodes, y_nodes]
            request.grid = grid_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            request.grid_count = grid_values.size
            request.grid_x_nodes = x_nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            request.grid_x_count = x_nodes.size
            request.grid_y_nodes = y_nodes.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            request.grid_y_count = y_nodes.size
        if weight_grid is not None:
            weight_values = np.ascontiguousarray(weight_grid, dtype=np.float64)
            weight_x = np.ascontiguousarray(weight_grid_x_nodes, dtype=np.float64)
            weight_y = np.ascontiguousarray(weight_grid_y_nodes, dtype=np.float64)
            if weight_values.shape != (weight_y.size, weight_x.size):
                raise ValueError("drizzle weight grid shape must be (y nodes, x nodes)")
            keep_alive += [weight_values, weight_x, weight_y]
            request.weight_grid = weight_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            request.weight_grid_count = weight_values.size
            request.weight_grid_x_nodes = weight_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            request.weight_grid_x_count = weight_x.size
            request.weight_grid_y_nodes = weight_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            request.weight_grid_y_count = weight_y.size
        if mask is not None:
            mask_values = np.ascontiguousarray(mask, dtype=np.uint8)
            if mask_values.ndim != 2:
                raise ValueError("drizzle mask must be a 2-D array")
            keep_alive.append(mask_values)
            request.mask = mask_values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            request.mask_count = mask_values.size
            request.mask_width = int(mask_values.shape[1])
            request.mask_height = int(mask_values.shape[0])
        request.output_sum = output_sum.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        request.output_weight = output_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        request.output_count = output_sum.size
        if output_touched is not None:
            if (
                output_touched.shape != output_sum.shape
                or output_touched.dtype != np.uint8
                or not output_touched.flags["C_CONTIGUOUS"]
            ):
                raise ValueError("drizzle touch flags must be a C-contiguous UInt8 array of the band's shape")
            request.output_touched = output_touched.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        error = ctypes.create_string_buffer(_ERROR_BYTES)
        status = int(
            self._library.ufwbpp_native_cpu_drizzle_v1(ctypes.byref(request), error, ctypes.sizeof(error))
        )
        if status != 0:
            self._raise(error, status, "native drizzle")
        del keep_alive


_LOCK = threading.Lock()
_CACHE: dict[str, NativeKernels | None] = {}


def native_kernels_disabled() -> bool:
    return os.environ.get(DISABLE_ENVIRONMENT_VARIABLE, "").strip() not in {"", "0", "false", "no"}


def load_native_kernels(
    library_path: str | os.PathLike[str] | None = None,
) -> NativeKernels | None:
    """Return the cached native kernels, or ``None`` when the NumPy path must run.

    Candidates are tried in priority order and the first library exporting the
    complete kernel symbol set wins, so an older library without the kernels
    never blocks a newer sibling.
    """

    if native_kernels_disabled():
        return None
    key = os.path.normcase(str(Path(library_path).expanduser())) if library_path else ""
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        loaded: NativeKernels | None = None
        for candidate in candidate_library_paths(library_path):
            try:
                loaded = NativeKernels(candidate)
                break
            except NativeKernelError:
                continue
        _CACHE[key] = loaded
        if loaded is not None:
            _install_debayer_accelerator(loaded)
        return loaded


def _install_debayer_accelerator(kernels: NativeKernels) -> None:
    """Let ``lightframeqc.cfa.bilinear_debayer`` run on the native kernel; the
    kernel reproduces the NumPy reference value for value."""

    try:
        from lightframeqc import cfa
    except Exception:  # pragma: no cover - lightframeqc is a hard dependency
        return

    def accelerated(mosaic: NDArray[np.float32], layout: tuple[int, int, int, int]) -> NDArray[np.float32]:
        return kernels.debayer_bilinear(mosaic, layout)

    cfa.set_debayer_accelerator(accelerated)


def reset_native_kernel_cache() -> None:
    """Forget loaded libraries; used by tests that switch libraries or the env."""

    with _LOCK:
        _CACHE.clear()
    try:
        from lightframeqc import cfa

        cfa.set_debayer_accelerator(None)
    except Exception:  # pragma: no cover
        pass


def describe_native_kernels() -> dict[str, Any]:
    """Receipt/doctor evidence: the loaded library or why the NumPy path runs."""

    if native_kernels_disabled():
        return {
            "loaded": False,
            "reason": f"disabled by {DISABLE_ENVIRONMENT_VARIABLE}",
            "libraryPath": None,
        }
    kernels = load_native_kernels()
    if kernels is None:
        candidates = [str(path) for path in candidate_library_paths(None)]
        return {
            "loaded": False,
            "reason": (
                "no candidate library exports the kernel symbols"
                if candidates
                else "native library not found"
            ),
            "libraryPath": None,
            "candidates": candidates,
            "expectedFilename": _library_filename(),
        }
    return kernels.describe()


@lru_cache(maxsize=1)
def default_kernel_threads() -> int:
    """Threads per kernel call: the tuning row's native-kernel thread budget."""

    try:
        from .hardware import detect_hardware
        from .performance_profile import select_execution_tuning

        threads = int(select_execution_tuning(detect_hardware()).kernel_threads)
    except Exception:  # pragma: no cover - defensive: tuning never blocks pixels
        threads = int(os.cpu_count() or 1)
    return max(1, min(threads, _MAXIMUM_KERNEL_THREADS))


__all__ = [
    "DISABLE_ENVIRONMENT_VARIABLE",
    "MAD_KERNEL_ID",
    "MEAN_KERNEL_ID",
    "NATIVE_ABI_VERSION",
    "NativeKernelError",
    "DRIZZLE_KERNELS",
    "DRIZZLE_KERNEL_ID",
    "NativeKernels",
    "RADON_KERNEL_ID",
    "TILE_OFFSET_KERNEL_ID",
    "WARP_KERNEL_ID",
    "default_kernel_threads",
    "describe_native_kernels",
    "load_native_kernels",
    "native_kernels_disabled",
    "reset_native_kernel_cache",
]
