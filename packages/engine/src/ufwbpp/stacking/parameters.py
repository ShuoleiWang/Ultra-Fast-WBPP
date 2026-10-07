"""Parameters and result of one pixel run, and the registration transform types."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..calibration.inputs import MasterMetadataOverride, RawFrameMetadataOverride
from ..calibration.policy import STRICT, WORKFLOWS, workflow_receipt
from ..image_io.xisf import XisfDecodePolicy
from ..integrity import canonical_json_document
from .drizzle_native import DrizzleGroupInputs
from .integration import CalibrationError, IntegrationParameters
from .metal_integration import SUPPORTED_BACKENDS
from .normalization import GlobalNormalizationParameters
from .proper_coaddition import ProperCoadditionParameters


PIPELINE_VERSION = "portable-pixel-pipeline-v1"


OUTPUT_STATE = "UNSOLVED_WORKING"


REGISTRATION_RESAMPLERS = frozenset({"bilinear", "lanczos-3-clamped"})


@dataclass(frozen=True, slots=True)
class PixelTransform:
    """Input-pixel to output-pixel homogeneous transform.

    The last row is ``(0, 0, 1)`` for an affine map.  A projective map (any
    other finite last row) is accepted as well: registration against frames
    of another night or hour angle needs the two perspective terms that a
    tilt or differential refraction adds, which an affine fit leaves as a
    field-dependent misregistration of several tenths of a pixel.
    """

    matrix: tuple[tuple[float, float, float], ...]

    @classmethod
    def identity(cls) -> PixelTransform:
        return cls(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

    @classmethod
    def from_value(
        cls, value: PixelTransform | Sequence[Sequence[float]]
    ) -> PixelTransform:
        if isinstance(value, cls):
            result = value
        else:
            try:
                result = cls(
                    tuple(tuple(float(item) for item in row) for row in value)
                )
            except (TypeError, ValueError) as error:
                raise CalibrationError(
                    "TRANSFORM_INVALID", "transform must be a numeric 3x3 matrix"
                ) from error
        result.validated_matrix()
        return result

    def validated_matrix(self) -> NDArray[np.float64]:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise CalibrationError(
                "TRANSFORM_INVALID", "transform must be a finite 3x3 matrix"
            )
        if matrix[2, 2] == 0.0:
            raise CalibrationError(
                "TRANSFORM_INVALID", "homogeneous transform must be normalized (m22 != 0)"
            )
        determinant = float(np.linalg.det(matrix))
        linear_determinant = float(np.linalg.det(matrix[:2, :2]))
        if (
            not math.isfinite(determinant)
            or abs(determinant) < 1e-12
            or not math.isfinite(linear_determinant)
            or abs(linear_determinant) < 1e-12
        ):
            raise CalibrationError("TRANSFORM_SINGULAR", "transform is singular")
        return matrix

    @property
    def is_projective(self) -> bool:
        matrix = self.validated_matrix()
        return not bool(np.array_equal(matrix[2], (0.0, 0.0, 1.0)))

    @property
    def is_identity(self) -> bool:
        return bool(
            np.allclose(
                self.validated_matrix(), np.eye(3), rtol=0.0, atol=1e-12
            )
        )

    def serializable(self) -> list[list[float]]:
        return [list(row) for row in self.matrix]


# Compatibility import; serialized matrices and numerical evaluation are unchanged.
AffineTransform = PixelTransform


OUTPUT_GRID_ALGORITHM = "canvas-window-lattice-v1"


@dataclass(frozen=True, eq=False)
class OutputGrid:
    """The output pixel grid of one mosaic panel: a window of the canvas.

    Every Light of the run is resampled straight onto this grid instead of
    the registration reference's own grid, so a panel master needs no second
    interpolation to join the mosaic.  ``reference_x``/``reference_y``
    (``(rows, columns)``, row-major) hold the registration-reference pixel
    coordinates (zero-based) of window pixel ``(column*spacing,
    row*spacing)``; a node outside the reference's mapping is NaN.  A Light
    registered to the reference by ``transform`` is sampled at
    ``transform^-1(reference(x, y))``: exactly at the nodes, bilinearly
    between them (the lattice warp).  ``canvas_origin`` is the window's
    zero-based ``(x, y)`` offset on the canvas and ``wcs`` the window's
    celestial WCS cards (the canvas projection with CRPIX shifted to the
    window).
    """

    width: int
    height: int
    spacing: int
    reference_x: NDArray[np.float64]
    reference_y: NDArray[np.float64]
    canvas_origin: tuple[int, int]
    wcs: Mapping[str, Any]

    def validate(self) -> None:
        if not (isinstance(self.width, int) and isinstance(self.height, int)) or self.width < 6 or self.height < 6:
            raise ValueError("output_grid must be at least 6x6 pixels")
        if (
            not isinstance(self.spacing, int)
            or isinstance(self.spacing, bool)
            or self.spacing < 1
            or self.spacing & (self.spacing - 1)
        ):
            raise ValueError("output_grid spacing must be a power of two")
        expected = self.node_shape
        for name, nodes in (("reference_x", self.reference_x), ("reference_y", self.reference_y)):
            if not isinstance(nodes, np.ndarray) or nodes.dtype != np.float64 or nodes.shape != expected:
                raise ValueError(f"output_grid {name} must be a float64 array of shape {expected}")
            if np.isinf(nodes).any():
                raise ValueError(f"output_grid {name} may hold NaN but no infinite node")
        if not np.isfinite(self.reference_x).any():
            raise ValueError("output_grid maps no window pixel into the reference frame")
        if not np.array_equal(np.isnan(self.reference_x), np.isnan(self.reference_y)):
            raise ValueError("output_grid nodes must be NaN in both coordinates or neither")
        if len(self.canvas_origin) != 2 or not all(isinstance(value, int) for value in self.canvas_origin):
            raise ValueError("output_grid canvas_origin must be two integers")
        for key in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"):
            if key not in self.wcs:
                raise ValueError(f"output_grid wcs lacks {key}")

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def node_shape(self) -> tuple[int, int]:
        return (
            (self.height - 1 + self.spacing - 1) // self.spacing + 1,
            (self.width - 1 + self.spacing - 1) // self.spacing + 1,
        )

    @property
    def digest(self) -> str:
        """SHA-256 of the geometry, the node coordinates and the WCS."""

        hasher = hashlib.sha256()
        hasher.update(
            canonical_json_document(
                {
                    "algorithm": OUTPUT_GRID_ALGORITHM,
                    "width": self.width,
                    "height": self.height,
                    "spacing": self.spacing,
                    "canvasOrigin": list(self.canvas_origin),
                    "wcs": {key: self.wcs[key] for key in sorted(self.wcs)},
                }
            )
        )
        for nodes in (self.reference_x, self.reference_y):
            hasher.update(np.ascontiguousarray(nodes, dtype="<f8").tobytes())
        return hasher.hexdigest()

    def frame_lattice(self, transform: PixelTransform) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The input coordinates of a Light at the nodes: the inverse of its
        input-to-reference ``transform`` applied to the reference nodes, in
        the warp's reference order (``((m00*x) + (m01*y)) + m02``, then the
        projective division)."""

        inverse = np.linalg.inv(transform.validated_matrix())
        x, y = self.reference_x, self.reference_y
        input_x = inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]
        input_y = inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]
        if not (inverse[2, 0] == 0.0 and inverse[2, 1] == 0.0 and inverse[2, 2] == 1.0):
            denominator = inverse[2, 0] * x + inverse[2, 1] * y + inverse[2, 2]
            input_x = input_x / denominator
            input_y = input_y / denominator
        return np.ascontiguousarray(input_x), np.ascontiguousarray(input_y)

    def serializable(self) -> dict[str, Any]:
        return {
            "algorithm": OUTPUT_GRID_ALGORITHM,
            "width": self.width,
            "height": self.height,
            "latticeSpacing": self.spacing,
            "canvasOrigin": list(self.canvas_origin),
            "wcs": {key: self.wcs[key] for key in sorted(self.wcs)},
            "sha256": self.digest,
        }


@dataclass(frozen=True, slots=True)
class PipelineParameters:
    calibration_workflow: str = STRICT
    integration: IntegrationParameters = field(default_factory=IntegrationParameters)
    registration_memory_bytes: int = 256 * 1024 * 1024
    registration_resampler: str = "lanczos-3-clamped"
    auto_crop: bool = True
    minimum_crop_fraction: float = 0.25
    preview_max_long_edge: int = 1600
    dark_temperature_tolerance_celsius: float = 3.0
    ordinary_integration_backend: str = "auto"
    native_library_path: str | None = None
    metal_source_path: str | None = None
    xisf_decode: XisfDecodePolicy = field(default_factory=XisfDecodePolicy)
    global_normalization: GlobalNormalizationParameters = field(
        default_factory=GlobalNormalizationParameters
    )
    # Opt-in ZOGY proper coaddition: one ADDITIONAL linear product per group,
    # from the same registered, normalized frames and the same per-pixel
    # rejection decisions.  The ordinary master is unaffected either way.
    proper_coaddition: ProperCoadditionParameters = field(
        default_factory=ProperCoadditionParameters
    )
    master_metadata_overrides: tuple[MasterMetadataOverride, ...] = ()
    raw_frame_metadata_overrides: tuple[RawFrameMetadataOverride, ...] = ()
    # Calibrated Lights are computed in memory and registered directly.  They
    # are written to disk only when a consumer (Drizzle, the public portable
    # pipeline) needs them; the ordinary E2E path keeps them transient.
    materialize_calibrated_lights: bool = True
    # Drizzle consumes the ordinary integration's per-frame products: with
    # this flag every group also keeps its calibrated paths, transforms,
    # normalization coefficients, weights and packed rejection masks in the
    # result (``PipelineResult.drizzle_groups``).  Requires materialized
    # calibrated Lights.
    capture_drizzle_inputs: bool = False
    # Cosmetic correction of hot pixels: pixels of the subtracted master dark
    # that lie more than this many robust sigmas above its median are
    # replaced in every calibrated Light by the median of their eight
    # neighbours before registration.  Unstable hot pixels leave residuals
    # after dark subtraction that pixel rejection cannot always remove when
    # several frames share one pointing.  ``None`` disables the correction.
    cosmetic_hot_pixel_sigma: float | None = 3.0
    # Published pipeline outputs are fsynced.  An enclosing run whose whole
    # pipeline directory is transient (the E2E work tree) turns this off and
    # relies on its own fsynced promotion of the final products.
    durable_intermediates: bool = True
    # The folder WBPP grouping keywords are read below (the common ancestor
    # of every input of the enclosing run, so all its stages read the same
    # keywords); ``None`` derives it from the inputs.  A local path derived
    # from the inputs: not serialized.
    grouping_keyword_root: str | None = None
    # A mosaic panel's canvas window: every Light is resampled onto it
    # instead of onto the registration reference's own grid (no shared crop;
    # drizzle and proper coaddition are not available on it yet).
    output_grid: OutputGrid | None = None

    def validate(self) -> None:
        if self.calibration_workflow not in WORKFLOWS:
            raise ValueError("unsupported calibration_workflow")
        if self.output_grid is not None:
            if not isinstance(self.output_grid, OutputGrid):
                raise ValueError("output_grid must be an OutputGrid")
            self.output_grid.validate()
            if self.capture_drizzle_inputs:
                raise ValueError("drizzle onto a mosaic canvas window is not supported yet")
            if self.proper_coaddition.enabled:
                raise ValueError("proper coaddition on a mosaic canvas window is not supported yet")
        if self.cosmetic_hot_pixel_sigma is not None and (
            isinstance(self.cosmetic_hot_pixel_sigma, bool)
            or not math.isfinite(self.cosmetic_hot_pixel_sigma)
            or self.cosmetic_hot_pixel_sigma < 1.0
        ):
            raise ValueError("cosmetic_hot_pixel_sigma must be None or a finite value >= 1")
        if not isinstance(self.materialize_calibrated_lights, bool):
            raise ValueError("materialize_calibrated_lights must be a boolean")
        if not isinstance(self.capture_drizzle_inputs, bool):
            raise ValueError("capture_drizzle_inputs must be a boolean")
        if self.capture_drizzle_inputs and not self.materialize_calibrated_lights:
            raise ValueError("capture_drizzle_inputs requires materialize_calibrated_lights")
        if not isinstance(self.durable_intermediates, bool):
            raise ValueError("durable_intermediates must be a boolean")
        self.integration.validate()
        if self.registration_memory_bytes < 1024:
            raise ValueError("registration_memory_bytes is too small")
        if self.registration_resampler not in REGISTRATION_RESAMPLERS:
            raise ValueError(
                "registration_resampler must be bilinear or lanczos-3-clamped"
            )
        if not 0 < self.minimum_crop_fraction <= 1:
            raise ValueError("minimum_crop_fraction must be in (0, 1]")
        if self.preview_max_long_edge < 16:
            raise ValueError("preview_max_long_edge must be at least 16")
        if (
            not math.isfinite(self.dark_temperature_tolerance_celsius)
            or self.dark_temperature_tolerance_celsius < 0
        ):
            raise ValueError("dark_temperature_tolerance_celsius must be finite and nonnegative")
        if self.ordinary_integration_backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                "ordinary_integration_backend must be auto, portable-cpu, "
                "generic-apple-metal, or m3-pro-tuned"
            )
        self.xisf_decode.validate()
        self.global_normalization.validate()
        self.proper_coaddition.validate()
        seen_overrides: set[str] = set()
        for override in self.master_metadata_overrides:
            override.validate()
            if override.source_sha256 in seen_overrides:
                raise ValueError("master_metadata_overrides contains a duplicate source digest")
            seen_overrides.add(override.source_sha256)
        seen_raw_overrides: set[str] = set()
        for override in self.raw_frame_metadata_overrides:
            override.validate()
            if override.source_sha256 in seen_raw_overrides:
                raise ValueError("raw_frame_metadata_overrides contains a duplicate source digest")
            seen_raw_overrides.add(override.source_sha256)

    def serializable(self) -> dict[str, Any]:
        return {
            "calibrationWorkflow": self.calibration_workflow,
            "calibrationPolicy": workflow_receipt(self.calibration_workflow),
            "integration": self.integration.serializable(),
            "registrationMemoryBytes": self.registration_memory_bytes,
            "registrationResampler": self.registration_resampler,
            "autoCrop": self.auto_crop,
            "minimumCropFraction": self.minimum_crop_fraction,
            "previewMaxLongEdge": self.preview_max_long_edge,
            "darkTemperatureToleranceCelsius": self.dark_temperature_tolerance_celsius,
            "ordinaryIntegrationBackend": self.ordinary_integration_backend,
            "nativeLibraryPath": self.native_library_path,
            "metalSourcePath": self.metal_source_path,
            "xisfDecode": self.xisf_decode.serializable(),
            "globalNormalization": self.global_normalization.serializable(),
            "masterMetadataOverrides": [
                item.serializable() for item in self.master_metadata_overrides
            ],
            "rawFrameMetadataOverrides": [
                item.serializable() for item in self.raw_frame_metadata_overrides
            ],
            "materializeCalibratedLights": self.materialize_calibrated_lights,
            "captureDrizzleInputs": self.capture_drizzle_inputs,
            # Serialized only when asked for, so every digest built over these
            # parameters is unchanged for a run that does not use it.
            **(
                {"properCoaddition": self.proper_coaddition.serializable()}
                if self.proper_coaddition.enabled
                else {}
            ),
            "cosmeticHotPixelSigma": self.cosmetic_hot_pixel_sigma,
            "durableIntermediates": self.durable_intermediates,
            **({"outputGrid": self.output_grid.serializable()} if self.output_grid is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    output_directory: str
    receipt_path: str
    state: str
    master_light_paths: tuple[str, ...]
    preview_paths: tuple[str, ...]
    # Per filter, the drizzle inputs captured by ``capture_drizzle_inputs``
    # (in-memory rejection masks; not part of the serializable receipt).
    drizzle_groups: Mapping[str, DrizzleGroupInputs] = field(default_factory=dict)
    # Per filter, the additional proper-coaddition product, when the recipe
    # asked for one.  The ordinary masters above stay the primary products.
    proper_coadd_paths: Mapping[str, str] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "outputDirectory": self.output_directory,
            "receiptPath": self.receipt_path,
            "state": self.state,
            "masterLightPaths": list(self.master_light_paths),
            "previewPaths": list(self.preview_paths),
            **(
                {"properCoaddPaths": dict(self.proper_coadd_paths)}
                if self.proper_coadd_paths
                else {}
            ),
        }
