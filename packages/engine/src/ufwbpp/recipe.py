from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
import re
from typing import Any, Mapping

from .calibration import COMBINATIONS, DEFAULT_COMBINATION
from .calibration_policy import STRICT, WORKFLOWS
from .selection.parameters import SelectionParameters


class RecipeError(ValueError):
    code = "RECIPE_INVALID"


class Requirement(StrEnum):
    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"
    DISABLED = "DISABLED"


class SolverPolicy(StrEnum):
    REQUIRED = "REQUIRED"
    DISABLED = "DISABLED"


_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class MasterMetadataOverrideRecipe:
    source_sha256: str
    camera: str | None = None
    gain: float | None = None
    offset: float | None = None
    binning_x: int | None = None
    binning_y: int | None = None
    filter_name: str | None = None
    cfa_pattern: str | None = None
    readout_mode: str | None = None
    temperature_celsius: float | None = None
    exposure_seconds: float | None = None
    bias_included: bool | None = None
    numeric_domain: str | None = None
    normalized_unit_scale: float | None = None

    @classmethod
    def from_dict(cls, raw: Any, index: int) -> MasterMetadataOverrideRecipe:
        name = f"calibration.masterMetadataOverrides[{index}]"
        value = _mapping(raw, name)
        _unknown(
            value,
            {
                "sourceSha256", "camera", "gain", "offset", "binning", "filter",
                "cfaPattern", "readoutMode", "temperatureCelsius", "exposureSeconds",
                "biasIncluded", "numericDomain", "normalizedUnitScale",
            },
            name,
        )
        digest = value.get("sourceSha256")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise RecipeError(f"{name}.sourceSha256 must be a lowercase sha256: digest")
        strings: dict[str, str | None] = {}
        for key in ("camera", "filter", "cfaPattern", "readoutMode"):
            item = value.get(key)
            if item is not None and (not isinstance(item, str) or not item.strip()):
                raise RecipeError(f"{name}.{key} must be a non-empty string or null")
            strings[key] = item.strip() if item is not None else None
        binning = value.get("binning")
        if binning is not None and (
            not isinstance(binning, list)
            or len(binning) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in binning)
        ):
            raise RecipeError(f"{name}.binning must be [positive x, positive y]")
        numbers: dict[str, float | None] = {}
        for key in ("gain", "offset", "temperatureCelsius", "exposureSeconds"):
            item = value.get(key)
            if item is None:
                numbers[key] = None
                continue
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise RecipeError(f"{name}.{key} must be numeric")
            item = float(item)
            if not math.isfinite(item):
                raise RecipeError(f"{name}.{key} must be finite")
            numbers[key] = item
        if numbers["exposureSeconds"] is not None and numbers["exposureSeconds"] < 0:
            raise RecipeError(f"{name}.exposureSeconds must be nonnegative")
        bias_included = value.get("biasIncluded")
        if bias_included is not None and not isinstance(bias_included, bool):
            raise RecipeError(f"{name}.biasIncluded must be boolean or null")
        numeric_domain = value.get("numericDomain")
        normalized_unit_scale = value.get("normalizedUnitScale")
        if (numeric_domain is None) != (normalized_unit_scale is None):
            raise RecipeError(
                f"{name}.numericDomain and normalizedUnitScale must be declared together"
            )
        if numeric_domain is not None:
            if (
                not isinstance(numeric_domain, str)
                or numeric_domain.strip().upper()
                not in {"NORMALIZED_UNIT", "SENSOR_CODE"}
            ):
                raise RecipeError(
                    f"{name}.numericDomain must be NORMALIZED_UNIT or SENSOR_CODE"
                )
            if (
                isinstance(normalized_unit_scale, bool)
                or not isinstance(normalized_unit_scale, (int, float))
                or not math.isfinite(float(normalized_unit_scale))
                or float(normalized_unit_scale) <= 0
            ):
                raise RecipeError(
                    f"{name}.normalizedUnitScale must be finite and positive"
                )
        return cls(
            source_sha256=digest,
            camera=strings["camera"],
            gain=numbers["gain"],
            offset=numbers["offset"],
            binning_x=binning[0] if binning is not None else None,
            binning_y=binning[1] if binning is not None else None,
            filter_name=strings["filter"],
            cfa_pattern=strings["cfaPattern"],
            readout_mode=strings["readoutMode"],
            temperature_celsius=numbers["temperatureCelsius"],
            exposure_seconds=numbers["exposureSeconds"],
            bias_included=bias_included,
            numeric_domain=(
                numeric_domain.strip().upper()
                if isinstance(numeric_domain, str)
                else None
            ),
            normalized_unit_scale=(
                float(normalized_unit_scale)
                if normalized_unit_scale is not None
                else None
            ),
        )

    def serializable(self) -> dict[str, Any]:
        return {
            "sourceSha256": self.source_sha256,
            "camera": self.camera,
            "gain": self.gain,
            "offset": self.offset,
            "binning": [self.binning_x, self.binning_y] if self.binning_x is not None else None,
            "filter": self.filter_name,
            "cfaPattern": self.cfa_pattern,
            "readoutMode": self.readout_mode,
            "temperatureCelsius": self.temperature_celsius,
            "exposureSeconds": self.exposure_seconds,
            "biasIncluded": self.bias_included,
            "numericDomain": self.numeric_domain,
            "normalizedUnitScale": self.normalized_unit_scale,
        }


@dataclass(frozen=True, slots=True)
class ReviewApprovalRecipe:
    source_sha256: str
    gate_policy_digest: str
    request_digest: str

    @classmethod
    def from_dict(cls, raw: Any, index: int) -> ReviewApprovalRecipe:
        value = _mapping(raw, f"reviewApprovals[{index}]")
        _unknown(
            value,
            {"sourceSha256", "gatePolicyDigest", "requestDigest"},
            f"reviewApprovals[{index}]",
        )
        fields = {
            "source_sha256": value.get("sourceSha256"),
            "gate_policy_digest": value.get("gatePolicyDigest"),
            "request_digest": value.get("requestDigest"),
        }
        for name, digest in fields.items():
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                raise RecipeError(
                    f"reviewApprovals[{index}].{name} must be a lowercase sha256: digest"
                )
        return cls(**fields)

    def serializable(self) -> dict[str, str]:
        return {
            "sourceSha256": self.source_sha256,
            "gatePolicyDigest": self.gate_policy_digest,
            "requestDigest": self.request_digest,
        }


@dataclass(frozen=True, slots=True)
class RawFrameMetadataOverrideRecipe:
    """Explicit content-bound confirmation for otherwise unknown raw metadata."""

    source_sha256: str
    cfa_pattern: str

    @classmethod
    def from_dict(cls, raw: Any, index: int) -> RawFrameMetadataOverrideRecipe:
        name = f"rawFrameMetadataOverrides[{index}]"
        value = _mapping(raw, name)
        _unknown(value, {"sourceSha256", "cfaPattern"}, name)
        digest = value.get("sourceSha256")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise RecipeError(f"{name}.sourceSha256 must be a lowercase sha256: digest")
        cfa = value.get("cfaPattern")
        if not isinstance(cfa, str) or not cfa.strip():
            raise RecipeError(f"{name}.cfaPattern must be a non-empty string")
        normalized = cfa.strip().upper()
        if normalized in {"UNKNOWN", "UNSPECIFIED"}:
            raise RecipeError(f"{name}.cfaPattern must explicitly confirm NONE or a CFA pattern")
        return cls(digest, normalized)

    def serializable(self) -> dict[str, str]:
        return {
            "sourceSha256": self.source_sha256,
            "cfaPattern": self.cfa_pattern,
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RecipeError(f"{name} must be an object")
    return value


def _unknown(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise RecipeError(f"{name} contains unknown keys: {', '.join(unexpected)}")


def _enum(enum_type: type[StrEnum], value: Any, name: str) -> Any:
    try:
        return enum_type(str(value).upper())
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise RecipeError(f"{name} must be one of: {choices}") from error


@dataclass(frozen=True, slots=True)
class CalibrationRecipe:
    flat: Requirement = Requirement.REQUIRED
    dark: Requirement = Requirement.OPTIONAL
    bias: Requirement = Requirement.REQUIRED
    allow_masters: bool = True
    workflow: str = STRICT
    master_metadata_overrides: tuple[MasterMetadataOverrideRecipe, ...] = ()

    @classmethod
    def from_dict(cls, raw: Any) -> CalibrationRecipe:
        value = _mapping(raw, "calibration")
        _unknown(
            value,
            {"flat", "dark", "bias", "allowMasters", "masterMetadataOverrides", "workflow"},
            "calibration",
        )
        workflow = value.get("workflow", STRICT)
        if not isinstance(workflow, str) or workflow not in WORKFLOWS:
            raise RecipeError("calibration.workflow must be strict-v1 or mono-standard-v1")
        allow_masters = value.get("allowMasters", True)
        if not isinstance(allow_masters, bool):
            raise RecipeError("calibration.allowMasters must be boolean")
        raw_overrides = value.get("masterMetadataOverrides", [])
        if not isinstance(raw_overrides, list):
            raise RecipeError("calibration.masterMetadataOverrides must be an array")
        overrides = tuple(
            MasterMetadataOverrideRecipe.from_dict(item, index)
            for index, item in enumerate(raw_overrides)
        )
        if len({item.source_sha256 for item in overrides}) != len(overrides):
            raise RecipeError("calibration.masterMetadataOverrides contains duplicate sourceSha256 values")
        return cls(
            flat=_enum(Requirement, value.get("flat", "REQUIRED"), "calibration.flat"),
            dark=_enum(Requirement, value.get("dark", "OPTIONAL"), "calibration.dark"),
            bias=_enum(Requirement, value.get("bias", "REQUIRED"), "calibration.bias"),
            allow_masters=allow_masters,
            workflow=workflow,
            master_metadata_overrides=overrides,
        )

    def serializable(self) -> dict[str, Any]:
        return {
            "flat": self.flat.value,
            "dark": self.dark.value,
            "bias": self.bias.value,
            "allowMasters": self.allow_masters,
            "workflow": self.workflow,
            "masterMetadataOverrides": [
                item.serializable() for item in self.master_metadata_overrides
            ],
        }


@dataclass(frozen=True, slots=True)
class SolverRecipe:
    policy: SolverPolicy = SolverPolicy.REQUIRED
    backend: str = "auto"
    search_radius_degrees: float = 15.0

    @classmethod
    def from_dict(cls, raw: Any) -> SolverRecipe:
        value = _mapping(raw, "solver")
        _unknown(value, {"policy", "backend", "searchRadiusDegrees"}, "solver")
        backend = value.get("backend", "auto")
        if not isinstance(backend, str) or not backend.strip():
            raise RecipeError("solver.backend must be a non-empty string")
        radius = value.get("searchRadiusDegrees", 15.0)
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            raise RecipeError("solver.searchRadiusDegrees must be numeric")
        radius = float(radius)
        if not 0 < radius <= 180:
            raise RecipeError("solver.searchRadiusDegrees must be in (0, 180]")
        return cls(
            policy=_enum(SolverPolicy, value.get("policy", "REQUIRED"), "solver.policy"),
            backend=backend.strip(),
            search_radius_degrees=radius,
        )

    def serializable(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "backend": self.backend,
            "searchRadiusDegrees": self.search_radius_degrees,
        }


@dataclass(frozen=True, slots=True)
class DrizzleRecipe:
    enabled: bool = False
    backend: str = "auto"
    scale: int = 2
    drop_shrink: float = 0.9
    cfa_drizzle: bool = False
    kernel: str = "square"

    @classmethod
    def from_dict(cls, raw: Any) -> DrizzleRecipe:
        value = _mapping(raw, "drizzle")
        _unknown(
            value,
            {"enabled", "backend", "scale", "dropShrink", "cfaDrizzle", "kernel"},
            "drizzle",
        )
        enabled = value.get("enabled", False)
        cfa_drizzle = value.get("cfaDrizzle", False)
        if not isinstance(enabled, bool):
            raise RecipeError("drizzle.enabled must be boolean")
        if not isinstance(cfa_drizzle, bool):
            raise RecipeError("drizzle.cfaDrizzle must be boolean")
        backend = value.get("backend", "auto")
        if not isinstance(backend, str) or not backend.strip():
            raise RecipeError("drizzle.backend must be a non-empty string")
        scale = value.get("scale", 2)
        if isinstance(scale, bool) or not isinstance(scale, int) or not 1 <= scale <= 4:
            raise RecipeError("drizzle.scale must be an integer from 1 to 4")
        kernel = value.get("kernel", "square")
        if not isinstance(kernel, str) or kernel not in {"square", "circular", "gaussian", "point"}:
            raise RecipeError("drizzle.kernel must be square, circular, gaussian or point")
        drop_shrink = value.get("dropShrink", 0.9)
        if isinstance(drop_shrink, bool) or not isinstance(drop_shrink, (int, float)):
            raise RecipeError("drizzle.dropShrink must be numeric")
        drop_shrink = float(drop_shrink)
        if not 0 < drop_shrink <= 1:
            raise RecipeError("drizzle.dropShrink must be in (0, 1]")
        return cls(
            enabled=enabled,
            backend=backend.strip(),
            scale=scale,
            drop_shrink=drop_shrink,
            cfa_drizzle=cfa_drizzle,
            kernel=kernel,
        )

    def serializable(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "scale": self.scale,
            "dropShrink": self.drop_shrink,
            "cfaDrizzle": self.cfa_drizzle,
            "kernel": self.kernel,
        }


@dataclass(frozen=True, slots=True)
class ProperCoadditionRecipe:
    """The optional ``properCoaddition`` block: ZOGY coaddition, default off.

    Produces one additional linear product per filter; the ordinary
    rejection/integration master stays the primary product and is untouched.
    """

    enabled: bool = False
    outlier_handling: str = "reuse-rejection"
    apodization_pixels: int = 64

    @classmethod
    def from_dict(cls, raw: Any) -> ProperCoadditionRecipe:
        value = _mapping(raw, "properCoaddition")
        _unknown(
            value, {"enabled", "outlierHandling", "apodizationPixels"}, "properCoaddition"
        )
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise RecipeError("properCoaddition.enabled must be boolean")
        outlier_handling = value.get("outlierHandling", "reuse-rejection")
        if outlier_handling not in {"reuse-rejection", "none"}:
            raise RecipeError(
                "properCoaddition.outlierHandling must be reuse-rejection or none"
            )
        apodization = value.get("apodizationPixels", 64)
        if (
            isinstance(apodization, bool)
            or not isinstance(apodization, int)
            or not 0 <= apodization <= 512
        ):
            raise RecipeError(
                "properCoaddition.apodizationPixels must be an integer from 0 to 512"
            )
        return cls(
            enabled=enabled,
            outlier_handling=outlier_handling,
            apodization_pixels=apodization,
        )

    def serializable(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "outlierHandling": self.outlier_handling,
            "apodizationPixels": self.apodization_pixels,
        }


@dataclass(frozen=True, slots=True)
class IntegrationRecipe:
    """The optional ``integration`` block: today only the combination rule."""

    combination: str = DEFAULT_COMBINATION

    @classmethod
    def from_dict(cls, raw: Any) -> IntegrationRecipe:
        value = _mapping(raw, "integration")
        _unknown(value, {"combination"}, "integration")
        combination = value.get("combination", DEFAULT_COMBINATION)
        if combination not in COMBINATIONS:
            raise RecipeError(
                "integration.combination must be one of: " + ", ".join(COMBINATIONS)
            )
        return cls(combination=combination)

    def serializable(self) -> dict[str, Any]:
        return {"combination": self.combination}


@dataclass(frozen=True, slots=True)
class LocalNormalizationRecipe:
    """Disabled legacy recipe field; the experimental pixel path was retired."""

    enabled: bool = False
    tile_size_pixels: int = 256

    def __post_init__(self) -> None:
        if self.enabled is not False:
            raise RecipeError("LOCAL_NORMALIZATION_REMOVED: use the default stellar-scale and background normalization")

    @classmethod
    def from_dict(cls, raw: Any) -> LocalNormalizationRecipe:
        value = _mapping(raw, "localNormalization")
        _unknown(value, {"enabled", "tileSizePixels"}, "localNormalization")
        enabled = value.get("enabled", False)
        tile_size = value.get("tileSizePixels", 256)
        if not isinstance(enabled, bool):
            raise RecipeError("localNormalization.enabled must be boolean")
        if enabled:
            raise RecipeError("LOCAL_NORMALIZATION_REMOVED: use the default stellar-scale and background normalization")
        if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 64:
            raise RecipeError("localNormalization.tileSizePixels must be an integer >= 64")
        return cls(enabled=enabled, tile_size_pixels=tile_size)

    def serializable(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "tileSizePixels": self.tile_size_pixels}


@dataclass(frozen=True, slots=True)
class SelectionRecipe:
    """The optional ``selection`` block: unattended Light selection policy."""

    parameters: SelectionParameters = field(default_factory=SelectionParameters)
    present: bool = False

    @classmethod
    def from_dict(cls, raw: Any) -> SelectionRecipe:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise RecipeError("selection must be an object")
        try:
            return cls(parameters=SelectionParameters.from_mapping(raw), present=True)
        except ValueError as error:
            raise RecipeError(str(error)) from error

    def serializable(self) -> dict[str, Any]:
        return self.parameters.serializable()


@dataclass(frozen=True, slots=True)
class Recipe:
    calibration: CalibrationRecipe = field(default_factory=CalibrationRecipe)
    selection: SelectionRecipe = field(default_factory=SelectionRecipe)
    solver: SolverRecipe = field(default_factory=SolverRecipe)
    drizzle: DrizzleRecipe = field(default_factory=DrizzleRecipe)
    proper_coaddition: ProperCoadditionRecipe = field(
        default_factory=ProperCoadditionRecipe
    )
    integration: IntegrationRecipe = field(default_factory=IntegrationRecipe)
    local_normalization: LocalNormalizationRecipe = field(
        default_factory=LocalNormalizationRecipe
    )
    output_format: str = "FITS"
    overwrite: bool = False
    review_approvals: tuple[ReviewApprovalRecipe, ...] = ()
    raw_frame_metadata_overrides: tuple[RawFrameMetadataOverrideRecipe, ...] = ()
    schema_version: int = 1

    @classmethod
    def from_dict(cls, raw: Any) -> Recipe:
        value = _mapping(raw, "recipe")
        _unknown(
            value,
            {
                "schemaVersion",
                "calibration",
                "solver",
                "drizzle",
                "properCoaddition",
                "integration",
                "localNormalization",
                "outputFormat",
                "overwrite",
                "reviewApprovals",
                "rawFrameMetadataOverrides",
                "selection",
            },
            "recipe",
        )
        schema_version = value.get("schemaVersion", 1)
        if schema_version != 1:
            raise RecipeError("only recipe schemaVersion 1 is supported")
        output_format = value.get("outputFormat", "FITS")
        if not isinstance(output_format, str) or output_format.upper() not in {"FITS", "XISF"}:
            raise RecipeError("outputFormat must be FITS or XISF")
        overwrite = value.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise RecipeError("overwrite must be boolean")
        raw_approvals = value.get("reviewApprovals", [])
        if not isinstance(raw_approvals, list):
            raise RecipeError("reviewApprovals must be an array")
        approvals = tuple(
            ReviewApprovalRecipe.from_dict(item, index)
            for index, item in enumerate(raw_approvals)
        )
        if len({item.source_sha256 for item in approvals}) != len(approvals):
            raise RecipeError("reviewApprovals contains duplicate sourceSha256 values")
        raw_frame_overrides_value = value.get("rawFrameMetadataOverrides", [])
        if not isinstance(raw_frame_overrides_value, list):
            raise RecipeError("rawFrameMetadataOverrides must be an array")
        raw_frame_overrides = tuple(
            RawFrameMetadataOverrideRecipe.from_dict(item, index)
            for index, item in enumerate(raw_frame_overrides_value)
        )
        if len({item.source_sha256 for item in raw_frame_overrides}) != len(raw_frame_overrides):
            raise RecipeError("rawFrameMetadataOverrides contains duplicate sourceSha256 values")
        proper_coaddition = ProperCoadditionRecipe.from_dict(
            value.get("properCoaddition")
        )
        if proper_coaddition.enabled and DrizzleRecipe.from_dict(value.get("drizzle")).enabled:
            raise RecipeError(
                "properCoaddition.enabled cannot be combined with drizzle.enabled: "
                "the drizzled master lives on a finer grid, so the proper coadd "
                "would have no same-grid solved master to inherit a verified WCS from"
            )
        return cls(
            calibration=CalibrationRecipe.from_dict(value.get("calibration")),
            solver=SolverRecipe.from_dict(value.get("solver")),
            drizzle=DrizzleRecipe.from_dict(value.get("drizzle")),
            proper_coaddition=proper_coaddition,
            integration=IntegrationRecipe.from_dict(value.get("integration")),
            local_normalization=LocalNormalizationRecipe.from_dict(
                value.get("localNormalization")
            ),
            output_format=output_format.upper(),
            overwrite=overwrite,
            review_approvals=approvals,
            raw_frame_metadata_overrides=raw_frame_overrides,
            selection=SelectionRecipe.from_dict(value.get("selection")),
            schema_version=1,
        )

    def serializable(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "calibration": self.calibration.serializable(),
            "solver": self.solver.serializable(),
            "drizzle": self.drizzle.serializable(),
            "localNormalization": self.local_normalization.serializable(),
            "outputFormat": self.output_format,
            "overwrite": self.overwrite,
            "reviewApprovals": [
                approval.serializable() for approval in self.review_approvals
            ],
            "rawFrameMetadataOverrides": [
                item.serializable() for item in self.raw_frame_metadata_overrides
            ],
            # Only an explicit selection block takes part in the recipe digest,
            # so recipes written before the block existed keep their digest.
            **({"selection": self.selection.serializable()} if self.selection.present else {}),
            # Likewise the opt-in advanced algorithms: a recipe that does not
            # ask for them serializes, and digests, exactly as before.
            **(
                {"properCoaddition": self.proper_coaddition.serializable()}
                if self.proper_coaddition.enabled
                else {}
            ),
            **(
                {"integration": self.integration.serializable()}
                if self.integration.combination != DEFAULT_COMBINATION
                else {}
            ),
        }


__all__ = [
    "COMBINATIONS",
    "DEFAULT_COMBINATION",
    "CalibrationRecipe",
    "DrizzleRecipe",
    "IntegrationRecipe",
    "ProperCoadditionRecipe",
    "LocalNormalizationRecipe",
    "MasterMetadataOverrideRecipe",
    "Recipe",
    "RecipeError",
    "Requirement",
    "ReviewApprovalRecipe",
    "RawFrameMetadataOverrideRecipe",
    "SolverPolicy",
    "SolverRecipe",
]
