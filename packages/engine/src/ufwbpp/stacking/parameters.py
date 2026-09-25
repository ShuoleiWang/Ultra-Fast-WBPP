"""Parameters and result of one pixel run, and the registration transform types."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..calibration.inputs import MasterMetadataOverride, RawFrameMetadataOverride
from ..calibration.policy import STRICT, WORKFLOWS, workflow_receipt
from ..image_io.xisf import XisfDecodePolicy
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

    def validate(self) -> None:
        if self.calibration_workflow not in WORKFLOWS:
            raise ValueError("unsupported calibration_workflow")
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
