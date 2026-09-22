from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
import math
from typing import Any


class Decision(StrEnum):
    KEEP = "KEEP"
    REVIEW = "REVIEW"
    REJECT_CLOUD = "REJECT_CLOUD"
    REJECT_OCCLUSION = "REJECT_OCCLUSION"
    UNASSESSABLE = "UNASSESSABLE"


class Confidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class FrameRole(StrEnum):
    """Scientific role resolved from authoritative frame metadata.

    Raw and master calibration frames are deliberately distinct.  In
    particular, :class:`FrameRole.MASTER_LIGHT` is an integration product and
    must never be mistaken for an unprocessed light frame.
    """

    LIGHT = "LIGHT"
    RAW_FLAT = "RAW_FLAT"
    MASTER_FLAT = "MASTER_FLAT"
    DARK = "DARK"
    MASTER_DARK = "MASTER_DARK"
    BIAS = "BIAS"
    MASTER_BIAS = "MASTER_BIAS"
    MASTER_LIGHT = "MASTER_LIGHT"
    UNKNOWN = "UNKNOWN"


class GateDisposition(StrEnum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    HARD_FAIL = "HARD_FAIL"


class EvidenceSeverity(StrEnum):
    """Severity of one independently auditable gate observation."""

    INFO = "INFO"
    WARNING = "WARNING"
    REVIEW = "REVIEW"
    ERROR = "ERROR"
    HARD_FAIL = "HARD_FAIL"


class EvidenceFamily(StrEnum):
    IDENTITY = "IDENTITY"
    ROLE = "ROLE"
    METADATA = "METADATA"
    PIXEL_STATISTICS = "PIXEL_STATISTICS"
    MORPHOLOGY = "MORPHOLOGY"
    TRANSPARENCY = "TRANSPARENCY"
    BACKGROUND = "BACKGROUND"
    NOISE = "NOISE"
    REGISTRATION = "REGISTRATION"
    OCCLUSION = "OCCLUSION"
    PROVENANCE = "PROVENANCE"
    CONSENSUS = "CONSENSUS"


def _strict_json_value(value: Any, field_name: str) -> Any:
    """Return JSON-native data or fail before a report can be published."""

    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_value(item, f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} contains a non-string key")
            result[key] = _strict_json_value(item, f"{field_name}.{key}")
        return result
    raise TypeError(f"{field_name} contains unsupported type {type(value).__name__}")


@dataclass(frozen=True)
class QualityEvidence:
    code: str
    family: EvidenceFamily
    severity: EvidenceSeverity
    message: str
    value: Any = None
    threshold: Any = None
    units: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        if not self.code.strip():
            raise ValueError("quality evidence code cannot be empty")
        if not self.message.strip():
            raise ValueError("quality evidence message cannot be empty")
        return {
            "code": self.code,
            "family": self.family.value,
            "severity": self.severity.value,
            "message": self.message,
            "value": _strict_json_value(self.value, "qualityEvidence.value"),
            "threshold": _strict_json_value(
                self.threshold, "qualityEvidence.threshold"
            ),
            "units": self.units,
            "details": _strict_json_value(
                self.details, "qualityEvidence.details"
            ),
        }


@dataclass
class QualityGateResult:
    disposition: GateDisposition
    evidence: list[QualityEvidence] = field(default_factory=list)
    summary: str = ""
    version: str = "quality-gate-v1"
    policy_digest: str = ""
    policy: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        if not self.version.strip():
            raise ValueError("quality gate version cannot be empty")
        return {
            "disposition": self.disposition.value,
            "summary": self.summary,
            "version": self.version,
            "policyDigest": self.policy_digest,
            "policy": _strict_json_value(self.policy, "qualityGate.policy"),
            "evidence": [item.serializable() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Stable, content-addressed identity for one regular input file."""

    sha256: str
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int

    def serializable(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "sizeBytes": self.size_bytes,
            "mtimeNs": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
        }


@dataclass(frozen=True)
class Star:
    x: float
    y: float
    flux: float
    peak: float
    a: float
    b: float
    theta: float
    fwhm: float
    ellipticity: float
    flags: int = 0
    support_pixels: int | None = None
    detection_pixels: int | None = None


@dataclass
class FrameMetadata:
    path: str
    width: int = 0
    height: int = 0
    channels: int = 0
    filter_name: str = "UNKNOWN"
    exposure_seconds: float | None = None
    gain: float | None = None
    offset: float | None = None
    binning_x: int = 1
    binning_y: int = 1
    camera: str = "UNKNOWN"
    target: str = "UNKNOWN"
    ra_degrees: float | None = None
    dec_degrees: float | None = None
    airmass: float | None = None
    altitude_degrees: float | None = None
    azimuth_degrees: float | None = None
    observed_at: datetime | None = None
    header: dict[str, Any] = field(default_factory=dict)
    role: FrameRole = FrameRole.UNKNOWN
    role_evidence: list[str] = field(default_factory=list)
    role_conflicts: list[str] = field(default_factory=list)
    cfa_pattern: str = "UNKNOWN"
    readout_mode: str = "UNKNOWN"
    binning_known: bool = False
    image_count: int = 1

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        value["observed_at"] = (
            self.observed_at.isoformat() if self.observed_at is not None else None
        )
        return value


@dataclass
class FrameMeasurement:
    metadata: FrameMetadata
    # Supported sources used by registration/photometry; the count remains
    # uncapped even when the returned catalog is limited by maximum_stars.
    stars: list[Star] = field(default_factory=list)
    detected_source_count: int | None = None
    preview_width: int = 0
    preview_height: int = 0
    image_median: float | None = None
    image_mad: float | None = None
    background_grid: list[list[float | None]] = field(default_factory=list)
    texture_grid: list[list[float | None]] = field(default_factory=list)
    thumbnail_path: str | None = None
    reader_backend: str = ""
    status: str = "MEASURED"
    error_code: str | None = None
    error_message: str | None = None
    pixinsight: dict[str, Any] = field(default_factory=dict)
    identity: FileIdentity | None = None
    preview_scale_x: float | None = None
    preview_scale_y: float | None = None
    finite_fraction: float | None = None
    image_p001: float | None = None
    image_p999: float | None = None
    dynamic_range: float | None = None
    # Original finite, positive SEP detections preserve fragmented trails and
    # extraction provenance. None identifies imported/legacy measurements.
    raw_stars: list[Star] | None = None
    raw_detected_source_count: int | None = None
    # Native-resolution PSF of the brightest stars (half-flux radius, FWHM as
    # 2 r50, wing fraction beyond one FWHM); None for imported measurements.
    native_psf: dict[str, Any] | None = None


@dataclass
class RegistrationMetrics:
    ok: bool = False
    matched_stars: int = 0
    match_fraction: float = 0.0
    rms_pixels: float | None = None
    matrix: list[list[float]] | None = None
    source_indices: list[int] = field(default_factory=list)
    reference_indices: list[int] = field(default_factory=list)
    error: str | None = None


@dataclass
class FrameFeatures:
    transparency_ratio: float | None = None
    extra_extinction_mag: float | None = None
    star_completeness: float | None = None
    detected_source_ratio: float | None = None
    spatial_transparency_mad_mag: float | None = None
    spatial_transparency_p90_mag: float | None = None
    spatial_dimming_p90_mag: float | None = None
    spatial_brightening_p90_mag: float | None = None
    overlap_fraction: float | None = None
    largest_missing_region: float | None = None
    missing_inside_outside_ratio: float | None = None
    boundary_support: float | None = None
    background_support: float | None = None
    nina_hfr_pixels: float | None = None
    median_fwhm_preview_pixels: float | None = None
    p90_fwhm_preview_pixels: float | None = None
    median_ellipticity: float | None = None
    p90_ellipticity: float | None = None
    image_median: float | None = None
    image_mad: float | None = None
    median_fwhm_native_pixels: float | None = None
    p90_fwhm_native_pixels: float | None = None
    median_axis_ratio: float | None = None
    median_eccentricity: float | None = None
    elongated_fraction: float | None = None
    orientation_coherence: float | None = None
    valid_morphology_star_count: int | None = None
    fragmented_trailing_detected: bool | None = None
    fragmented_trail_chain_count: int | None = None
    fragmented_trail_fraction: float | None = None
    fragmented_trail_coherence: float | None = None
    fragmented_trail_consensus_fraction: float | None = None
    fragmented_trail_occupied_cells: int | None = None
    fragmented_trail_spatial_minor_fraction: float | None = None
    nightly_extinction_residual: float | None = None
    psf_r50_native_pixels: float | None = None
    psf_fwhm_native_pixels: float | None = None
    psf_wing_fraction: float | None = None
    psf_native_star_count: int | None = None
    background_z: float | None = None
    noise_z: float | None = None
    cloud_score: int = 0
    occlusion_score: int = 0
    shape_score: int = 0


@dataclass
class FrameResult:
    path: str
    group_id: str
    reference_path: str | None
    decision: Decision
    confidence: Confidence
    reasons: list[str]
    warnings: list[str]
    registration: RegistrationMetrics
    features: FrameFeatures
    metadata: FrameMetadata
    star_count: int
    thumbnail_path: str | None = None
    grid: dict[str, Any] = field(default_factory=dict)
    identity: FileIdentity | None = None
    quality_gate: QualityGateResult | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "groupId": self.group_id,
            "referencePath": self.reference_path,
            "decision": self.decision.value,
            "confidence": self.confidence.value,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "registration": asdict(self.registration),
            "features": asdict(self.features),
            "metadata": self.metadata.serializable(),
            "starCount": self.star_count,
            "thumbnailPath": self.thumbnail_path,
            "grid": self.grid,
            "sourceIdentity": (
                self.identity.serializable() if self.identity is not None else None
            ),
            "qualityGate": (
                self.quality_gate.serializable()
                if self.quality_gate is not None
                else None
            ),
        }


@dataclass
class RunResult:
    schema_version: int
    algorithm_version: str
    generated_at: datetime
    inputs: list[str]
    output_directory: str
    config: dict[str, Any]
    groups: list[dict[str, Any]]
    frames: list[FrameResult]
    warnings: list[str] = field(default_factory=list)

    def serializable(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "algorithmVersion": self.algorithm_version,
            "generatedAt": self.generated_at.isoformat(),
            "inputs": self.inputs,
            "outputDirectory": self.output_directory,
            "config": self.config,
            "groups": self.groups,
            "warnings": self.warnings,
            "frames": [frame.serializable() for frame in self.frames],
        }
