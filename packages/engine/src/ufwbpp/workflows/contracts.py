"""Requests, results and progress for the single-target scientific workflow."""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import StrEnum
import hashlib
import json
import math
import re
from typing import Any, Callable, Mapping
from lightframeqc import DEFAULT_CONFIG, QcConfig
from lightframeqc.quality_gate import GatePolicy
from lightframeqc.blink_flags import BlinkFlagPolicy
from ..integrity import canonical_json_document
from ..pixel_pipeline import PipelineParameters
from ..selection import SelectionParameters
from ..drizzle_native import SUPPORTED_SCALES as DRIZZLE_SCALES_SUPPORTED, SUPPORTED_KERNELS as DRIZZLE_KERNELS_SUPPORTED


class IntegrationMode(StrEnum):
    ORDINARY = "ordinary"
    DRIZZLE = "drizzle"


class E2EState(StrEnum):
    SOLVED = "SOLVED"
    UNSOLVED_WORKING = "UNSOLVED_WORKING"


class ProgressStage(StrEnum):
    INVENTORY = "inventory"
    QUALITY_CONTROL = "quality-control"
    CALIBRATION = "calibration"
    REGISTRATION = "registration"
    INTEGRATION = "integration"
    DRIZZLE = "drizzle"
    ASTROMETRY = "astrometry"
    PREVIEW = "preview"
    VERIFY = "verify"
    PUBLISH = "publish"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: ProgressStage
    status: str
    current: int = 0
    total: int = 0
    message: str = ""

    def serializable(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status,
            "current": self.current,
            "total": self.total,
            "message": self.message,
        }


ProgressCallback = Callable[[ProgressEvent], None]


class E2EError(RuntimeError):
    """Stable fail-closed error for invalid or incomplete E2E execution."""

    def __init__(self, code: str, message: str, *, path: str | None = None) -> None:
        self.code = code
        self.path = path
        detail = f"{path}: {message}" if path else message
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ReviewApproval:
    """One explicit, content- and request-bound REVIEW admission."""

    source_sha256: str
    gate_policy_digest: str
    request_digest: str

    def serializable(self) -> dict[str, str]:
        return {
            "sourceSha256": self.source_sha256,
            "gatePolicyDigest": self.gate_policy_digest,
            "requestDigest": self.request_digest,
        }


SELECTION_KIND = "ultra-fast-wbpp-selection"


SELECTION_POLICY = "explicit-v1"


MAX_SELECTION_DECISIONS = 10_000


_SELECTION_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


_SELECTION_ORIGIN_KEYS = ("sessionId", "blinkManifestSha256", "flagsPolicyDigest", "createdAt")


@dataclass(frozen=True, slots=True)
class ExplicitDecision:
    """One Light's KEEP/DROP as the reviewer decided it in the blink view."""

    source_sha256: str
    decision: str
    default_decision: str | None = None
    flags: tuple[str, ...] = ()
    note: str | None = None

    def serializable(self) -> dict[str, Any]:
        value: dict[str, Any] = {"sourceSha256": self.source_sha256, "decision": self.decision}
        if self.default_decision is not None:
            value["defaultDecision"] = self.default_decision
        if self.flags:
            value["flags"] = list(self.flags)
        if self.note is not None:
            value["note"] = self.note
        return value


@dataclass(frozen=True, slots=True)
class ExplicitSelection:
    """The ``selection-v1`` file: every Light's decision and where it came from.

    ``digest`` is the SHA-256 of the canonical JSON of the validated file and
    is recorded as ``selectionDigest`` in the receipts.  ``origin`` is kept
    for the audit trail and never enforced: a hand-written file is valid.
    """

    decisions: tuple[ExplicitDecision, ...]
    undecided: str = "ERROR"
    origin: Mapping[str, str] | None = None
    digest: str = ""

    @property
    def by_source(self) -> dict[str, ExplicitDecision]:
        return {item.source_sha256: item for item in self.decisions}

    def serializable(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schemaVersion": 1,
            "kind": SELECTION_KIND,
            "policy": SELECTION_POLICY,
            "undecided": self.undecided,
            "decisions": [item.serializable() for item in self.decisions],
        }
        if self.origin is not None:
            value["origin"] = dict(self.origin)
        return value


def parse_explicit_selection(raw: Any) -> ExplicitSelection:
    """Validate a ``selection-v1`` object; every problem is ``SELECTION_INVALID``."""

    def invalid(message: str) -> E2EError:
        return E2EError("SELECTION_INVALID", message)

    if not isinstance(raw, Mapping):
        raise invalid("selection must be a JSON object")
    allowed = {"schemaVersion", "kind", "policy", "origin", "undecided", "decisions"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise invalid("selection has unsupported fields: " + ", ".join(str(item) for item in unknown))
    if raw.get("schemaVersion") != 1:
        raise invalid("selection schemaVersion must be 1")
    if raw.get("kind") != SELECTION_KIND:
        raise invalid(f"selection kind must be {SELECTION_KIND}")
    if raw.get("policy") != SELECTION_POLICY:
        raise invalid(f"selection policy must be {SELECTION_POLICY}")
    undecided = raw.get("undecided", "ERROR")
    if undecided not in {"ERROR", "DROP", "KEEP"}:
        raise invalid("selection undecided must be ERROR, DROP or KEEP")
    origin_raw = raw.get("origin")
    origin: dict[str, str] | None = None
    if origin_raw is not None:
        if not isinstance(origin_raw, Mapping) or set(origin_raw) - set(_SELECTION_ORIGIN_KEYS):
            raise invalid("selection origin may hold only " + ", ".join(_SELECTION_ORIGIN_KEYS))
        origin = {}
        for key in _SELECTION_ORIGIN_KEYS:
            if key in origin_raw:
                value = origin_raw[key]
                if not isinstance(value, str) or len(value) > 512:
                    raise invalid(f"selection origin.{key} must be a short string")
                origin[key] = value
    decisions_raw = raw.get("decisions")
    if not isinstance(decisions_raw, list):
        raise invalid("selection decisions must be an array")
    if len(decisions_raw) > MAX_SELECTION_DECISIONS:
        raise invalid(f"selection holds more than {MAX_SELECTION_DECISIONS} decisions")
    decisions: list[ExplicitDecision] = []
    seen: set[str] = set()
    for index, item in enumerate(decisions_raw):
        if not isinstance(item, Mapping) or set(item) - {"sourceSha256", "decision", "defaultDecision", "flags", "note"}:
            raise invalid(f"selection decision {index} has unsupported fields")
        digest = item.get("sourceSha256")
        if not isinstance(digest, str) or _SELECTION_DIGEST.fullmatch(digest) is None:
            raise invalid(f"selection decision {index} needs a lowercase sha256: digest")
        if digest in seen:
            raise invalid(f"selection decision {index} repeats {digest}")
        seen.add(digest)
        decision = item.get("decision")
        if decision not in {"KEEP", "DROP"}:
            raise invalid(f"selection decision {index} must be KEEP or DROP")
        default = item.get("defaultDecision")
        if default is not None and default not in {"KEEP", "DROP"}:
            raise invalid(f"selection decision {index} defaultDecision must be KEEP or DROP")
        flags_raw = item.get("flags", [])
        if not isinstance(flags_raw, list) or any(
            not isinstance(flag, str) or not flag.strip() or len(flag) > 64 for flag in flags_raw
        ):
            raise invalid(f"selection decision {index} flags must be short strings")
        note = item.get("note")
        if note is not None and (not isinstance(note, str) or len(note) > 2_000):
            raise invalid(f"selection decision {index} note must be a string of at most 2000 characters")
        decisions.append(
            ExplicitDecision(
                source_sha256=digest,
                decision=decision,
                default_decision=default,
                flags=tuple(flags_raw),
                note=note,
            )
        )
    selection = ExplicitSelection(decisions=tuple(decisions), undecided=undecided, origin=origin)
    digest = "sha256:" + hashlib.sha256(canonical_json_document(selection.serializable())).hexdigest()
    return replace(selection, digest=digest)


@dataclass(frozen=True, slots=True)
class DrizzleOptions:
    scale: int = 2
    pixfrac: float = 0.9
    kernel: str = "square"
    cfa_drizzle: bool = False
    tile_rows: int = 256
    max_tile_bytes: int = 256 * 1024**2
    max_output_pixels: int = 128 * 1024**2
    max_working_set_bytes: int = 4 * 1024**3
    minimum_coverage_fraction: float = 0.90
    maximum_null_fraction: float = 0.10
    minimum_distinct_dither_phases: int = 3
    minimum_dither_phase_separation_pixels: float = 0.15
    minimum_dither_span_pixels: float = 0.35
    maximum_fwhm_for_upsampling_pixels: float = 3.0
    rejection_sigma: float = 6.0
    rejection_minimum_frames: int = 3

    def validate(self) -> None:
        if isinstance(self.scale, bool) or self.scale not in DRIZZLE_SCALES_SUPPORTED:
            raise E2EError(
                "DRIZZLE_SCALE_INVALID",
                f"drizzle scale must be one of {DRIZZLE_SCALES_SUPPORTED}",
            )
        if not math.isfinite(self.pixfrac) or not 0.1 <= self.pixfrac <= 1.0:
            raise E2EError("DRIZZLE_PIXFRAC_INVALID", "pixfrac must be in [0.1, 1]")
        if self.kernel not in DRIZZLE_KERNELS_SUPPORTED:
            raise E2EError(
                "DRIZZLE_KERNEL_INVALID",
                f"drizzle kernel must be one of {DRIZZLE_KERNELS_SUPPORTED}",
            )
        if not isinstance(self.cfa_drizzle, bool):
            raise E2EError("DRIZZLE_CFA_INVALID", "cfa_drizzle must be boolean")
        for name in (
            "tile_rows",
            "max_tile_bytes",
            "max_output_pixels",
            "max_working_set_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise E2EError("DRIZZLE_LIMIT_INVALID", f"{name} must be positive")
        for name in ("minimum_coverage_fraction", "maximum_null_fraction"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise E2EError(
                    "DRIZZLE_GATE_INVALID", f"{name} must be in [0, 1]"
                )
        if self.minimum_coverage_fraction < 0.90:
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "minimum_coverage_fraction cannot be below the production floor 0.90",
            )
        if self.maximum_null_fraction > 0.10:
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "maximum_null_fraction cannot exceed the production ceiling 0.10",
            )
        if (
            isinstance(self.minimum_distinct_dither_phases, bool)
            or not isinstance(self.minimum_distinct_dither_phases, int)
            or self.minimum_distinct_dither_phases < 3
        ):
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "minimum_distinct_dither_phases must be at least 3",
            )
        if not (
            math.isfinite(self.minimum_dither_phase_separation_pixels)
            and 0.0 < self.minimum_dither_phase_separation_pixels <= math.sqrt(0.5)
        ):
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "minimum dither phase separation must be in (0, sqrt(0.5)]",
            )
        if not (
            math.isfinite(self.minimum_dither_span_pixels)
            and 0.0 < self.minimum_dither_span_pixels <= 0.5
        ):
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "minimum dither span must be in (0, 0.5]",
            )
        if not (
            math.isfinite(self.maximum_fwhm_for_upsampling_pixels)
            and self.maximum_fwhm_for_upsampling_pixels > 0
        ):
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "maximum_fwhm_for_upsampling_pixels must be positive",
            )
        if self.maximum_fwhm_for_upsampling_pixels > 3.0:
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "maximum_fwhm_for_upsampling_pixels cannot exceed the production ceiling 3.0",
            )
        if not math.isfinite(self.rejection_sigma) or self.rejection_sigma < 3.0:
            raise E2EError(
                "DRIZZLE_GATE_INVALID", "rejection_sigma must be at least 3"
            )
        if (
            isinstance(self.rejection_minimum_frames, bool)
            or not isinstance(self.rejection_minimum_frames, int)
            or self.rejection_minimum_frames < 3
        ):
            raise E2EError(
                "DRIZZLE_GATE_INVALID",
                "rejection_minimum_frames must be at least 3",
            )

    def serializable(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "pixfrac": self.pixfrac,
            "kernel": self.kernel,
            "cfaDrizzle": self.cfa_drizzle,
            "tileRows": self.tile_rows,
            "maxTileBytes": self.max_tile_bytes,
            "maxOutputPixels": self.max_output_pixels,
            "maxWorkingSetBytes": self.max_working_set_bytes,
            "minimumCoverageFraction": self.minimum_coverage_fraction,
            "maximumNullFraction": self.maximum_null_fraction,
            "minimumDistinctDitherPhases": self.minimum_distinct_dither_phases,
            "minimumDitherPhaseSeparationPixels": self.minimum_dither_phase_separation_pixels,
            "minimumDitherSpanPixels": self.minimum_dither_span_pixels,
            "maximumFwhmForUpsamplingPixels": self.maximum_fwhm_for_upsampling_pixels,
            "rejectionSigma": self.rejection_sigma,
            "rejectionMinimumFrames": self.rejection_minimum_frames,
        }


@dataclass(frozen=True, slots=True)
class E2ERequest:
    light_files: tuple[str, ...]
    flat_files: tuple[str, ...]
    bias_files: tuple[str, ...]
    output_directory: str
    dark_files: tuple[str, ...] = ()
    master_bias_files: tuple[str, ...] = ()
    master_dark_files: tuple[str, ...] = ()
    master_flat_files: tuple[str, ...] = ()
    review_approvals: tuple[ReviewApproval, ...] = ()
    recipe_digest: str | None = None
    integration_mode: IntegrationMode = IntegrationMode.ORDINARY
    workers: int = 1
    qc_config: QcConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    gate_policy: GatePolicy = field(default_factory=GatePolicy)
    pipeline_parameters: PipelineParameters = field(default_factory=PipelineParameters)
    selection: SelectionParameters = field(default_factory=SelectionParameters)
    # The blink review's decisions; present exactly when ``selection.policy``
    # is ``explicit-v1``.  The flags policy is recorded with them.
    explicit_selection: ExplicitSelection | None = None
    blink_flags_policy: BlinkFlagPolicy = field(default_factory=BlinkFlagPolicy)
    drizzle: DrizzleOptions = field(default_factory=DrizzleOptions)
    ra_hint_degrees: float | None = None
    dec_hint_degrees: float | None = None
    field_of_view_degrees: float | None = None
    search_radius_degrees: float | None = None
    min_matches: int = 12
    max_rms_arcsec: float = 2.0
    # Filters of one run are registered onto one reference grid and cropped
    # identically, so their masters share every pixel.  Each master is still
    # solved on its own; the fresh solves must agree within this many pixels
    # (solver precision, not the 0.05 px grid-identity gate) before the best
    # one is written to every same-grid master.
    same_grid_wcs_tolerance_pixels: float = 1.0
    registration_detection: Any = field(default=None, repr=False, compare=False)
    registration_config: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class E2EResult:
    success: bool
    code: str
    state: E2EState
    output_directory: str | None
    evidence_directory: str | None
    receipt_path: str
    product_paths: tuple[str, ...]
    preview_paths: tuple[str, ...]
    passed_light_paths: tuple[str, ...]
    excluded_light_paths: tuple[str, ...]
    message: str | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "code": self.code,
            "state": self.state.value,
            "outputDirectory": self.output_directory,
            "evidenceDirectory": self.evidence_directory,
            "receiptPath": self.receipt_path,
            "productPaths": list(self.product_paths),
            "previewPaths": list(self.preview_paths),
            "passedLightPaths": list(self.passed_light_paths),
            "excludedLightPaths": list(self.excluded_light_paths),
            "message": self.message,
        }

