"""Blink screening: flags, reference and the per-channel evidence of a set of Lights.

The same evidence serves two callers.  ``blink-measure`` (the desktop's
"Blink & select" step) measures the Lights, computes it, renders the
normalized previews and writes a session manifest; ``run_e2e`` recomputes it
from its own quality pass and writes ``qc/blink.json`` next to the QC
manifest, so a run with an explicit selection is self-describing without the
session.  Nothing here touches the pipeline's registration, normalization or
integration references.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from lightframeqc.analysis import analyze_measurements
from lightframeqc.blink_flags import (
    BlinkFlagPolicy,
    BlinkFrameFlags,
    BlinkFrameInput,
    ChannelStatistics,
    blink_inputs,
    channel_statistics,
    compute_flags,
    night_summaries,
)
from lightframeqc.blink_reference import (
    BLINK_REFERENCE_RULE,
    BlinkReference,
    BlinkScore,
    choose_references,
    score_frames,
)
from lightframeqc.config import DEFAULT_CONFIG, QcConfig
from lightframeqc.measure import linear_preview_name, measure_paths
from lightframeqc.models import FrameMeasurement, FrameResult, FrameRole
from lightframeqc.parallel import FrameRunner
from lightframeqc.quality_gate import GatePolicy, evaluate_quality_gate
from lightframeqc.readers import probe_frame_metadata, read_frame_preview

from . import __version__
from .blink_previews import (
    FILMSTRIP_FORMATS,
    ChannelStretch,
    FramePreviewResult,
    FramePreviewSpec,
    PreviewCalibration,
    calibrate_linear,
    STRETCH_HARD_TARGET,
    channel_stretch,
    compose_to_reference,
    render_previews,
    safe_stem,
)
from .hardware import detect_hardware
from .performance_profile import select_execution_tuning
from .quality_cache import quality_cache_directory

BLINK_EVIDENCE_KIND = "blink-evidence-v1"
BLINK_MANIFEST_KIND = "blink-manifest-v1"
MAX_BLINK_LIGHTS = 10_000
MAX_BLINK_MASTERS = 64
LINEAR_DIRECTORY = "linear"
FILMSTRIP_DIRECTORY = "filmstrip"
# The same filmstrip images under the harder stretch, for the contrast toggle.
FILMSTRIP_HARD_DIRECTORY = "filmstrip-hard"
ZOOM_DIRECTORY = "zoom"
# A master dark serves a Light whose exposure is within this fraction.
DARK_EXPOSURE_TOLERANCE = 0.05

ProgressCallback = Callable[[str, str], None]


class BlinkSessionError(RuntimeError):
    """A blink session could not be established; ``code`` is stable."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class BlinkEvidence:
    """Flags, scores and references of every Light, in manifest order.

    ``inputs``/``flags``/``scores`` are aligned lists sorted by channel and
    then by observation time; ``index`` of a frame is its position here.
    """

    inputs: tuple[BlinkFrameInput, ...]
    flags: tuple[BlinkFrameFlags, ...]
    scores: tuple[BlinkScore, ...]
    references: Mapping[str, BlinkReference]
    statistics: Mapping[str, ChannelStatistics]
    nights: tuple[dict[str, Any], ...]
    flags_policy: BlinkFlagPolicy
    gate_policy_digest: str

    @property
    def flags_by_path(self) -> dict[str, BlinkFrameFlags]:
        return {record.path: record for record in self.flags}

    @property
    def reference_paths(self) -> dict[str, str]:
        return {channel_id: reference.path for channel_id, reference in self.references.items()}

    @property
    def counts(self) -> dict[str, int]:
        exclude = sum(record.exclude for record in self.flags)
        attention = sum(record.attention for record in self.flags)
        return {
            "frames": len(self.flags),
            "exclude": exclude,
            "attention": attention,
            "clean": len(self.flags) - exclude - attention,
        }

    def channel_ids(self) -> list[str]:
        return list(dict.fromkeys(frame.channel_id for frame in self.inputs))

    def frame_records(self) -> list[dict[str, Any]]:
        """Per-frame manifest records without previews or transforms."""

        index_by_path = {frame.path: index for index, frame in enumerate(self.inputs)}
        records: list[dict[str, Any]] = []
        for frame, record, score in zip(self.inputs, self.flags, self.scores, strict=True):
            reference = self.references.get(frame.channel_id)
            records.append(
                {
                    "index": index_by_path[frame.path],
                    "channelId": frame.channel_id,
                    "filter": frame.filter_name,
                    "target": frame.target,
                    "night": frame.night,
                    "path": frame.path,
                    "name": frame.name,
                    "sourceSha256": frame.source_sha256,
                    "observedAt": frame.observed_at,
                    "airmass": frame.airmass,
                    "reference": reference is not None and reference.path == frame.path,
                    "defaultDecision": record.default_decision,
                    "flags": [item.serializable() for item in record.flags],
                    "notes": list(record.notes),
                    "gate": {
                        "disposition": frame.gate_disposition,
                        "codes": list(frame.gate_codes),
                    },
                    "metrics": dict(record.metrics),
                    "score": {
                        "log10": None if score.score_log10 is None else round(score.score_log10, 4),
                        "z": None if score.score_z is None else round(score.score_z, 3),
                        "rank": score.rank,
                        "candidate": score.candidate,
                    },
                }
            )
        return records

    def channel_records(self) -> list[dict[str, Any]]:
        index_by_path = {frame.path: index for index, frame in enumerate(self.inputs)}
        records: list[dict[str, Any]] = []
        for channel_id in self.channel_ids():
            frames = [frame for frame in self.inputs if frame.channel_id == channel_id]
            reference = self.references.get(channel_id)
            statistics = self.statistics.get(channel_id)
            records.append(
                {
                    "channelId": channel_id,
                    "target": frames[0].target,
                    "filter": frames[0].filter_name,
                    "frameCount": len(frames),
                    "reference": (
                        {
                            "index": index_by_path[reference.path],
                            "sourceSha256": reference.source_sha256,
                            "rule": reference.rule,
                            "candidacy": reference.candidacy,
                            "candidateCount": reference.candidate_count,
                        }
                        if reference is not None
                        else None
                    ),
                    "statistics": (
                        {
                            "skyClean": statistics.sky_clean,
                            "cleanCount": statistics.clean_count,
                            "sourcesBest": statistics.sources_best,
                            "fwhmBest": statistics.fwhm_best,
                            "gradientClean": statistics.gradient_clean,
                        }
                        if statistics is not None
                        else None
                    ),
                    "nights": [
                        {key: value for key, value in night.items() if key != "channelId"}
                        for night in self.nights
                        if night["channelId"] == channel_id
                    ],
                }
            )
        return records

    def serializable(self) -> dict[str, Any]:
        """The run's ``qc/blink.json``: flags, reference, scores and nights."""

        return {
            "schemaVersion": 1,
            "kind": BLINK_EVIDENCE_KIND,
            "gatePolicyDigest": self.gate_policy_digest,
            "flagsPolicyDigest": self.flags_policy.canonical_digest(),
            "flagsPolicy": self.flags_policy.serializable(),
            "referenceRule": BLINK_REFERENCE_RULE,
            "counts": self.counts,
            "channels": self.channel_records(),
            "frames": self.frame_records(),
        }


def _manifest_order(inputs: Iterable[BlinkFrameInput]) -> list[BlinkFrameInput]:
    """Channel order (first appearance) then observation time, then path."""

    ordered = list(inputs)
    channel_rank = {channel_id: rank for rank, channel_id in enumerate(dict.fromkeys(frame.channel_id for frame in ordered))}
    return sorted(
        ordered,
        key=lambda frame: (channel_rank[frame.channel_id], frame.observed_at or "￿", frame.path),
    )


def _fallback_reference(frames: Sequence[BlinkFrameInput]) -> BlinkReference:
    """A channel without any scorable frame still gets a reference: the
    richest frame (then the earliest) anchors the previews unregistered."""

    best = min(
        frames,
        key=lambda frame: (-frame.star_count, frame.observed_at or "￿", frame.path),
    )
    return BlinkReference(
        channel_id=best.channel_id,
        path=best.path,
        rule=BLINK_REFERENCE_RULE,
        candidacy="unscored",
        candidate_count=0,
        score=None,
        source_sha256=best.source_sha256,
    )


def blink_evidence(
    results: Sequence[FrameResult],
    measurements: Sequence[FrameMeasurement] | Mapping[str, FrameMeasurement] | None,
    *,
    flags_policy: BlinkFlagPolicy | None = None,
    gate_policy: GatePolicy | None = None,
    qc_config: QcConfig | None = None,
    gradients: Mapping[str, float] | None = None,
) -> BlinkEvidence:
    """Compute flags, scores, references and night summaries for gate-evaluated results.

    ``gradients`` are optional calibrated gradient amplitudes (ADU) per path,
    supplied by a blink session that had calibration masters.
    """

    policy = flags_policy or BlinkFlagPolicy()
    policy.validate()
    gate = gate_policy or GatePolicy()
    inputs = blink_inputs(
        results,
        measurements,
        policy,
        night_boundary_hours=gate.night_boundary_hours,
        observing_timezone=(
            qc_config.observing_timezone if qc_config is not None else gate.observing_timezone
        ),
    )
    if gradients:
        inputs = [
            replace(frame, gradient_adu=gradients.get(frame.path))
            if frame.path in gradients
            else frame
            for frame in inputs
        ]
    ordered = _manifest_order(inputs)
    flags = compute_flags(ordered, policy)
    scores = score_frames(ordered, flags)
    references = dict(choose_references(scores, flags, inputs=ordered))
    for channel_id in dict.fromkeys(frame.channel_id for frame in ordered):
        if channel_id not in references:
            references[channel_id] = _fallback_reference(
                [frame for frame in ordered if frame.channel_id == channel_id]
            )
    return BlinkEvidence(
        inputs=tuple(ordered),
        flags=tuple(flags),
        scores=tuple(scores),
        references=references,
        statistics=channel_statistics(ordered, policy),
        nights=tuple(night_summaries(ordered, flags)),
        flags_policy=policy,
        gate_policy_digest=gate.canonical_digest(),
    )


@dataclass(frozen=True, slots=True)
class BlinkPreviewOptions:
    filmstrip_scale: int = 8
    zoom_scale: int = 4
    filmstrip_format: str = "jpeg"
    jpeg_quality: int = 85
    display_algorithm: str = "shared-stretch-v1"

    def validate(self) -> None:
        if self.display_algorithm not in {"shared-stretch-v1", "blink-complementary-display-v2"}:
            raise BlinkSessionError("BLINK_REQUEST_INVALID", "unsupported preview display algorithm")
        for name in ("filmstrip_scale", "zoom_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 64:
                raise BlinkSessionError("BLINK_REQUEST_INVALID", f"previews.{name} must be an integer in [1, 64]")
        if self.filmstrip_scale < self.zoom_scale:
            raise BlinkSessionError("BLINK_REQUEST_INVALID", "previews.filmstripScale cannot be finer than zoomScale")
        if self.filmstrip_format not in FILMSTRIP_FORMATS:
            raise BlinkSessionError("BLINK_REQUEST_INVALID", "previews.filmstripFormat must be jpeg or png")
        if isinstance(self.jpeg_quality, bool) or not isinstance(self.jpeg_quality, int) or not 50 <= self.jpeg_quality <= 100:
            raise BlinkSessionError("BLINK_REQUEST_INVALID", "previews.jpegQuality must be an integer in [50, 100]")

    def serializable(self) -> dict[str, Any]:
        return {
            "filmstripScale": self.filmstrip_scale,
            "zoomScale": self.zoom_scale,
            "filmstripFormat": self.filmstrip_format,
            "jpegQuality": self.jpeg_quality,
            "displayAlgorithm": self.display_algorithm,
        }


@dataclass(frozen=True, slots=True)
class BlinkMaster:
    role: str
    path: str
    filter_name: str | None = None
    exposure_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class BlinkRequest:
    """The private ``blink-measure`` request (schema 1)."""

    light_paths: tuple[str, ...]
    session_directory: str
    workers: int | None = None
    previews: BlinkPreviewOptions = BlinkPreviewOptions()
    master_flats: tuple[BlinkMaster, ...] = ()
    master_darks: tuple[BlinkMaster, ...] = ()
    master_bias: str | None = None

    @classmethod
    def from_mapping(cls, raw: Any) -> "BlinkRequest":
        def invalid(message: str) -> BlinkSessionError:
            return BlinkSessionError("BLINK_REQUEST_INVALID", message)

        if not isinstance(raw, Mapping):
            raise invalid("blink request must be a JSON object")
        allowed = {"schemaVersion", "lightPaths", "sessionDirectory", "workers", "previews", "masterFlats", "masterDarks", "masterBias"}
        unknown = sorted(set(raw) - allowed)
        if unknown or raw.get("schemaVersion") != 1:
            raise invalid("blink request has an unsupported schema or fields: " + ", ".join(str(item) for item in unknown))
        paths = raw.get("lightPaths")
        if (
            not isinstance(paths, list)
            or not paths
            or len(paths) > MAX_BLINK_LIGHTS
            or not all(isinstance(item, str) and item.strip() for item in paths)
        ):
            raise invalid(f"lightPaths must contain between 1 and {MAX_BLINK_LIGHTS} non-empty paths")
        session = raw.get("sessionDirectory")
        if not isinstance(session, str) or not session.strip():
            raise invalid("sessionDirectory is required")
        workers = raw.get("workers")
        if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 64):
            raise invalid("workers must be an integer in [1, 64]")
        previews_raw = raw.get("previews", {})
        if not isinstance(previews_raw, Mapping):
            raise invalid("previews must be an object")
        preview_keys = {"filmstripScale": "filmstrip_scale", "zoomScale": "zoom_scale", "filmstripFormat": "filmstrip_format", "jpegQuality": "jpeg_quality", "displayAlgorithm": "display_algorithm"}
        if set(previews_raw) - set(preview_keys):
            raise invalid("previews has unsupported fields")
        previews = BlinkPreviewOptions(**{preview_keys[key]: value for key, value in previews_raw.items()})
        previews.validate()

        def masters(key: str, *, with_filter: bool) -> tuple[BlinkMaster, ...]:
            items = raw.get(key, [])
            if not isinstance(items, list) or len(items) > MAX_BLINK_MASTERS:
                raise invalid(f"{key} must be an array of at most {MAX_BLINK_MASTERS} entries")
            result: list[BlinkMaster] = []
            for index, item in enumerate(items):
                fields = {"filter", "path"} if with_filter else {"path", "exposureSeconds"}
                if not isinstance(item, Mapping) or set(item) - fields or "path" not in item:
                    raise invalid(f"{key}[{index}] is invalid")
                path = item.get("path")
                if not isinstance(path, str) or not path.strip():
                    raise invalid(f"{key}[{index}].path must be a non-empty string")
                filter_name = item.get("filter")
                if with_filter and (not isinstance(filter_name, str) or not filter_name.strip()):
                    raise invalid(f"{key}[{index}].filter must be a non-empty string")
                exposure = item.get("exposureSeconds")
                if exposure is not None and (isinstance(exposure, bool) or not isinstance(exposure, (int, float)) or not math.isfinite(float(exposure)) or exposure <= 0):
                    raise invalid(f"{key}[{index}].exposureSeconds must be positive")
                result.append(
                    BlinkMaster(
                        role="MASTER_FLAT" if with_filter else "MASTER_DARK",
                        path=path,
                        filter_name=filter_name.strip().upper() if isinstance(filter_name, str) else None,
                        exposure_seconds=float(exposure) if exposure is not None else None,
                    )
                )
            return tuple(result)

        flats = masters("masterFlats", with_filter=True)
        if len({item.filter_name for item in flats}) != len(flats):
            raise invalid("masterFlats must name each filter once")
        bias = raw.get("masterBias")
        if bias is not None and (not isinstance(bias, str) or not bias.strip()):
            raise invalid("masterBias must be a non-empty string")
        return cls(
            light_paths=tuple(paths),
            session_directory=session,
            workers=workers,
            previews=previews,
            master_flats=flats,
            master_darks=masters("masterDarks", with_filter=False),
            master_bias=bias,
        )


def _canonical_lights(paths: Sequence[str]) -> list[Path]:
    canonical: list[Path] = []
    seen: set[str] = set()
    for value in paths:
        try:
            path = Path(value).expanduser().resolve(strict=True)
        except OSError as error:
            raise BlinkSessionError("BLINK_REQUEST_INVALID", f"light path cannot be resolved: {value} ({error})") from error
        if not path.is_file():
            raise BlinkSessionError("BLINK_REQUEST_INVALID", f"light path is not a regular file: {path}")
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        metadata = probe_frame_metadata(path)
        if metadata.role is not FrameRole.LIGHT:
            raise BlinkSessionError(
                "BLINK_INPUT_NOT_LIGHT",
                f"blink-measure accepts authoritative Light frames only: {path}",
            )
        canonical.append(path)
    if not canonical:
        raise BlinkSessionError("BLINK_NO_LIGHTS", "blink-measure requires at least one Light frame")
    return canonical


def _create_session_directory(requested: str, lights: Sequence[Path]) -> Path:
    session = Path(requested).expanduser()
    if os.path.lexists(session):
        raise BlinkSessionError("BLINK_SESSION_EXISTS", f"blink session directory must be new: {session}")
    parent = session.parent
    parent.mkdir(parents=True, exist_ok=True)
    resolved = parent.resolve(strict=True) / session.name
    # Never inside a source folder (and no source inside the session).
    for light in lights:
        if light.parent == resolved or light.parent in resolved.parents or resolved in light.parents:
            raise BlinkSessionError("BLINK_REQUEST_INVALID", "the session directory cannot lie in a source folder")
    try:
        resolved.mkdir(parents=False, exist_ok=False)
    except FileExistsError as error:
        raise BlinkSessionError("BLINK_SESSION_EXISTS", f"blink session directory must be new: {resolved}") from error
    return resolved


def _inventory_sha256(results: Sequence[FrameResult]) -> str:
    """Content-bound identity of the measured set (names and digests)."""

    entries = sorted(
        (
            {
                "name": Path(result.path).name,
                "sha256": result.identity.sha256 if result.identity is not None else "",
            }
            for result in results
        ),
        key=lambda item: (item["sha256"], item["name"]),
    )
    payload = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _fit_quadratic_span(values: np.ndarray, core_fraction: float = 0.40) -> float | None:
    """P99-P1 of a quadratic surface fitted to the outer cells of a grid."""

    rows, columns = values.shape
    grid = values.astype(np.float64).copy()
    half = core_fraction / 2.0
    grid[int(rows * (0.5 - half)) : int(rows * (0.5 + half)), int(columns * (0.5 - half)) : int(columns * (0.5 + half))] = np.nan
    mask = np.isfinite(grid)
    if np.count_nonzero(mask) < 12:
        return None
    yy, xx = np.indices(grid.shape, dtype=np.float64)
    x = (xx[mask] - columns / 2.0) / columns
    y = (yy[mask] - rows / 2.0) / rows
    design = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    coefficients, *_ = np.linalg.lstsq(design, grid[mask], rcond=None)
    fitted = design @ coefficients
    return float(np.percentile(fitted, 99) - np.percentile(fitted, 1))


def _master_preview(path: str, long_edge: int) -> tuple[np.ndarray, float | None] | None:
    try:
        preview = read_frame_preview(Path(path).expanduser(), max_long_edge=long_edge)
    except Exception:
        return None
    return np.asarray(preview.data, dtype=np.float32), preview.metadata.exposure_seconds


def calibrated_gradients(
    request: BlinkRequest,
    lights: Sequence[Path],
    measurements: Sequence[FrameMeasurement],
    linear_root: Path,
    *,
    long_edge: int,
    core_fraction: float = 0.40,
) -> tuple[dict[str, float], list[str]]:
    """Calibrated background gradient amplitude (ADU) per Light, when possible.

    Needs the filter's master flat and a pedestal source (an exposure-matched
    master dark, else the master bias): with the flat alone the pedestal
    leaves an inverse-vignetting term as large as a moonlit gradient (see
    ``build/plans/a11-raw-grid-gradient.md``).  Masters in the normalized
    [0, 1] domain are scaled to the 16-bit range of the Lights.
    """

    notes: list[str] = []
    if not request.master_flats:
        return {}, notes
    flats = {master.filter_name: master for master in request.master_flats}
    darks: list[tuple[BlinkMaster, np.ndarray, float | None]] = []
    for master in request.master_darks:
        loaded = _master_preview(master.path, long_edge)
        if loaded is None:
            notes.append(f"master dark unreadable: {Path(master.path).name}")
            continue
        darks.append((master, loaded[0], master.exposure_seconds or loaded[1]))
    bias = _master_preview(request.master_bias, long_edge) if request.master_bias else None
    if request.master_bias and bias is None:
        notes.append(f"master bias unreadable: {Path(request.master_bias).name}")
    flat_previews: dict[str, np.ndarray] = {}
    gradients: dict[str, float] = {}
    for index, (light, measurement) in enumerate(zip(lights, measurements, strict=True)):
        if measurement.status != "MEASURED":
            continue
        filter_name = str(measurement.metadata.filter_name or "").strip().upper()
        flat = flats.get(filter_name)
        if flat is None:
            continue
        if filter_name not in flat_previews:
            loaded = _master_preview(flat.path, long_edge)
            if loaded is None:
                notes.append(f"master flat unreadable: {Path(flat.path).name}")
                flats.pop(filter_name)
                continue
            flat_previews[filter_name] = loaded[0]
        flat_data = flat_previews[filter_name]
        exposure = measurement.metadata.exposure_seconds
        pedestal: np.ndarray | None = None
        for master, data, dark_exposure in darks:
            if exposure is None or dark_exposure is None:
                continue
            if abs(dark_exposure - exposure) <= DARK_EXPOSURE_TOLERANCE * exposure:
                pedestal = data
                break
        if pedestal is None and bias is not None:
            pedestal = bias[0]
        if pedestal is None:
            continue
        try:
            linear = np.load(linear_root / linear_preview_name(index), allow_pickle=False).astype(np.float32)
        except (OSError, ValueError):
            continue
        if linear.shape != flat_data.shape or linear.shape != pedestal.shape:
            notes.append(f"master geometry differs from the Lights for filter {filter_name}")
            continue
        light_median = float(np.nanmedian(linear))
        scale = 65535.0 if float(np.nanmax(pedestal)) <= 1.001 and light_median > 2.0 else 1.0
        flat_norm = flat_data / max(float(np.nanmedian(flat_data)), 1e-9)
        with np.errstate(divide="ignore", invalid="ignore"):
            calibrated = (linear - pedestal * scale) / np.where(flat_norm > 0.05, flat_norm, np.nan)
        rows, columns = calibrated.shape
        cells = np.full((16, 16), np.nan, dtype=np.float64)
        y_edges = np.linspace(0, rows, 17, dtype=np.int64)
        x_edges = np.linspace(0, columns, 17, dtype=np.int64)
        for row in range(16):
            for column in range(16):
                block = calibrated[y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]]
                finite = block[np.isfinite(block)]
                if finite.size:
                    cells[row, column] = float(np.median(finite))
        span = _fit_quadratic_span(cells, core_fraction)
        if span is not None:
            gradients[str(light)] = span
    return gradients, notes


def _preview_calibration(
    request: BlinkRequest, filter_name: str, exposure: float | None, long_edge: int
) -> PreviewCalibration | None:
    """The flat of the filter plus an exposure-matched dark or the bias."""

    flat = next((item for item in request.master_flats if item.filter_name == filter_name), None)
    if flat is None:
        return None
    pedestal: str | None = None
    for dark in request.master_darks:
        dark_exposure = dark.exposure_seconds
        if dark_exposure is None:
            loaded = _master_preview(dark.path, long_edge)
            dark_exposure = loaded[1] if loaded is not None else None
        if exposure is not None and dark_exposure is not None and abs(dark_exposure - exposure) <= DARK_EXPOSURE_TOLERANCE * exposure:
            pedestal = dark.path
            break
    if pedestal is None and request.master_bias:
        pedestal = request.master_bias
    return PreviewCalibration(flat_path=flat.path, pedestal_path=pedestal, long_edge=long_edge)


def _preview_specs(
    evidence: BlinkEvidence,
    results_by_path: Mapping[str, FrameResult],
    measurements_by_path: Mapping[str, FrameMeasurement],
    linear_by_path: Mapping[str, Path],
    session: Path,
    options: BlinkPreviewOptions,
    request: BlinkRequest,
    long_edge: int,
) -> tuple[list[FramePreviewSpec], dict[str, ChannelStretch], dict[str, dict[str, Any]]]:
    specs: list[FramePreviewSpec] = []
    stretches: dict[str, ChannelStretch] = {}
    geometry: dict[str, dict[str, Any]] = {}
    divisor = max(1, round(options.filmstrip_scale / options.zoom_scale))
    for channel_id in evidence.channel_ids():
        reference = evidence.references[channel_id]
        reference_result = results_by_path.get(reference.path)
        reference_measurement = measurements_by_path.get(reference.path)
        reference_linear = linear_by_path.get(reference.path)
        if reference_measurement is None or reference_linear is None or not reference_linear.exists():
            continue
        try:
            reference_data = np.load(reference_linear, allow_pickle=False).astype(np.float32)
        except (OSError, ValueError):
            continue
        calibration = _preview_calibration(
            request,
            str(reference_measurement.metadata.filter_name or "").strip().upper(),
            reference_measurement.metadata.exposure_seconds,
            long_edge,
        )
        reference_data, reference_calibrated = calibrate_linear(reference_data, calibration)
        if not reference_calibrated:
            calibration = None
        stretch = channel_stretch(reference_data, divisor)
        stretch_hard = channel_stretch(reference_data, divisor, target=STRETCH_HARD_TARGET)
        stretches[channel_id] = stretch
        output_shape = (int(reference_data.shape[0]), int(reference_data.shape[1]))
        reference_sky = (
            float(np.nanmedian(reference_data))
            if reference_calibrated
            else float(reference_measurement.image_median or stretch.sky_reference)
        )
        reference_transparency = (
            reference_result.features.transparency_ratio if reference_result is not None else None
        )
        reference_matrix = reference_result.registration.matrix if reference_result is not None else None
        geometry[channel_id] = {
            "zoom": [output_shape[1], output_shape[0]],
            "filmstrip": [math.ceil(output_shape[1] / divisor), math.ceil(output_shape[0] / divisor)],
            "sourceShape": [int(reference_measurement.metadata.height), int(reference_measurement.metadata.width)],
            "zoomScale": round(float(reference_measurement.preview_scale_x or 1.0), 4),
            "filmstripScale": round(float(reference_measurement.preview_scale_x or 1.0) * divisor, 4),
            "calibration": calibration.serializable() if calibration is not None else None,
        }
        for index, frame in enumerate(evidence.inputs):
            if frame.channel_id != channel_id:
                continue
            measurement = measurements_by_path.get(frame.path)
            result = results_by_path.get(frame.path)
            linear = linear_by_path.get(frame.path)
            if measurement is None or linear is None or not linear.exists():
                continue
            transform = (
                None
                if frame.path == reference.path
                else compose_to_reference(
                    result.registration.matrix if result is not None and result.registration.ok else None,
                    reference_matrix,
                )
            )
            if frame.path == reference.path:
                transform = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
            sky = float(measurement.image_median) if measurement.image_median is not None else reference_sky
            transparency = frame.transparency
            flux_scale = (
                float(reference_transparency) / float(transparency)
                if reference_transparency is not None and transparency is not None and transparency > 0
                else 1.0
            )
            stem = safe_stem(frame.path)
            suffix = "jpg" if options.filmstrip_format == "jpeg" else "png"
            specs.append(
                FramePreviewSpec(
                    index=index,
                    linear_path=str(linear),
                    filmstrip_path=str(session / FILMSTRIP_DIRECTORY / f"{index:04d}-{safe_stem(frame.filter_name, 16)}-{stem}.{suffix}"),
                    filmstrip_hard_path=str(session / FILMSTRIP_HARD_DIRECTORY / f"{index:04d}-{safe_stem(frame.filter_name, 16)}-{stem}.{suffix}"),
                    zoom_path=str(session / ZOOM_DIRECTORY / f"{index:04d}-{safe_stem(frame.filter_name, 16)}-{stem}.png"),
                    transform=transform,
                    output_shape=output_shape,
                    sky=sky,
                    flux_scale=flux_scale,
                    reference_sky=reference_sky,
                    stretch=stretch,
                    stretch_hard=stretch_hard,
                    filmstrip_format=options.filmstrip_format,
                    jpeg_quality=options.jpeg_quality,
                    filmstrip_divisor=divisor,
                    calibration=calibration,
                )
            )
    return specs, stretches, geometry


def run_blink_session(
    request: BlinkRequest,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Measure, analyse, gate, flag, score and render the Lights of a blink session.

    Returns the ``blink-manifest-v1`` manifest, which is also written to
    ``manifest.json`` in the (newly created) session directory before the
    function returns.
    """

    def emit(stage: str, message: str) -> None:
        if progress is not None:
            progress(stage, message)

    request.previews.validate()
    lights = _canonical_lights(request.light_paths)
    session = _create_session_directory(request.session_directory, lights)
    tuning = select_execution_tuning(detect_hardware())
    workers = tuning.qc_workers if request.workers is None else request.workers
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise BlinkSessionError("BLINK_REQUEST_INVALID", "workers must be a positive integer")
    # The stretched review thumbnails are the legacy review path; the blink
    # previews come from the linear previews kept next to the session.
    config = replace(DEFAULT_CONFIG, make_thumbnails=False)
    gate_policy = GatePolicy()
    flags_policy = BlinkFlagPolicy()
    timings: dict[str, float] = {}
    linear_root = session / LINEAR_DIRECTORY
    frame_runner = FrameRunner(workers, len(lights))
    try:
        emit("measure", f"measuring {len(lights)} Light frames")
        started = perf_counter()
        measurement_stats: dict[str, Any] = {}
        measurement_cache_stats: dict[str, int] = {}
        measurements = measure_paths(
            lights, session, config, workers=workers, stats=measurement_stats,
            runner=frame_runner, linear_directory=linear_root,
            cache_directory=quality_cache_directory(), cache_stats=measurement_cache_stats,
        )
        timings["measurementSeconds"] = perf_counter() - started
        emit("analyze", "analyzing star fields")
        started = perf_counter()
        analysis_stats: dict[str, Any] = {}
        cache_stats: dict[str, int] = {}
        _groups, results = analyze_measurements(
            measurements, config, cache_directory=quality_cache_directory(), cache_stats=cache_stats,
            workers=workers, stats=analysis_stats, runner=frame_runner,
        )
        timings["analysisSeconds"] = perf_counter() - started
        started = perf_counter()
        evaluate_quality_gate(results, measurements, gate_policy)
        timings["gateSeconds"] = perf_counter() - started
        started = perf_counter()
        gradients, notes = calibrated_gradients(
            request, lights, measurements, linear_root,
            long_edge=config.preview_long_edge, core_fraction=flags_policy.background_shape_core_fraction,
        )
        timings["gradientSeconds"] = perf_counter() - started
        emit("flags", "computing blink flags and references")
        started = perf_counter()
        evidence = blink_evidence(
            results, measurements, flags_policy=flags_policy, gate_policy=gate_policy,
            qc_config=config, gradients=gradients,
        )
        timings["flagsSeconds"] = perf_counter() - started
        emit("previews", f"rendering {len(evidence.inputs)} normalized previews")
        started = perf_counter()
        results_by_path = {result.path: result for result in results}
        measurements_by_path = {measurement.metadata.path: measurement for measurement in measurements}
        linear_by_path = {str(light): linear_root / linear_preview_name(index) for index, light in enumerate(lights)}
        if request.previews.display_algorithm == "blink-complementary-display-v2":
            from .blink_diagnostic_render import prepare_diagnostic_specs
            evidence, specs, stretches, geometry = prepare_diagnostic_specs(
                evidence, results_by_path, measurements_by_path, linear_by_path,
                session, request, config.preview_long_edge,
            )
        else:
            specs, stretches, geometry = _preview_specs(
                evidence, results_by_path, measurements_by_path, linear_by_path, session,
                request.previews, request, config.preview_long_edge,
            )
        rendered = {item.index: item for item in render_previews(specs, workers=workers, runner=frame_runner)}
        timings["previewSeconds"] = perf_counter() - started
    finally:
        frame_runner.close()
        shutil.rmtree(linear_root, ignore_errors=True)
    failed = [item for item in rendered.values() if item.error is not None]
    if specs and len(failed) == len(specs):
        raise BlinkSessionError("BLINK_PREVIEW_FAILED", f"no preview could be rendered: {failed[0].error}")

    manifest = _manifest(
        request, session, lights, evidence, results, rendered, stretches, geometry, timings,
        measurement_stats, analysis_stats, cache_stats, measurement_cache_stats, notes, workers, gate_policy,
    )
    manifest_path = session / "manifest.json"
    encoded = (json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with manifest_path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def _manifest(
    request: BlinkRequest,
    session: Path,
    lights: Sequence[Path],
    evidence: BlinkEvidence,
    results: Sequence[FrameResult],
    rendered: Mapping[int, FramePreviewResult],
    stretches: Mapping[str, ChannelStretch],
    geometry: Mapping[str, dict[str, Any]],
    timings: Mapping[str, float],
    measurement_stats: Mapping[str, Any],
    analysis_stats: Mapping[str, Any],
    cache_stats: Mapping[str, int],
    measurement_cache_stats: Mapping[str, int],
    notes: Sequence[str],
    workers: int,
    gate_policy: GatePolicy,
) -> dict[str, Any]:
    results_by_path = {result.path: result for result in results}
    reference_matrix = {
        channel_id: results_by_path[reference.path].registration.matrix
        if reference.path in results_by_path
        else None
        for channel_id, reference in evidence.references.items()
    }
    frames: list[dict[str, Any]] = []
    for record, frame in zip(evidence.frame_records(), evidence.inputs, strict=True):
        preview = rendered.get(record["index"])
        result = results_by_path.get(frame.path)
        reference = evidence.references.get(frame.channel_id)
        is_reference = reference is not None and reference.path == frame.path
        transform = (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
            if is_reference
            else compose_to_reference(
                result.registration.matrix if result is not None and result.registration.ok else None,
                reference_matrix.get(frame.channel_id),
            )
        )
        registered = preview is not None and preview.registered
        frames.append(
            {
                **record,
                "previews": {
                    "filmstrip": (
                        Path(preview.filmstrip_path).relative_to(session).as_posix()
                        if preview is not None and preview.filmstrip_path
                        else None
                    ),
                    "zoom": (
                        Path(preview.zoom_path).relative_to(session).as_posix()
                        if preview is not None and preview.zoom_path
                        else None
                    ),
                    "filmstripHard": (
                        Path(preview.filmstrip_hard_path).relative_to(session).as_posix()
                        if preview is not None and preview.filmstrip_hard_path
                        else None
                    ),
                    "coverage": None if preview is None else (None if preview.coverage is None else round(preview.coverage, 4)),
                    "filmstripBytes": preview.filmstrip_bytes if preview is not None else 0,
                    "filmstripHardBytes": preview.filmstrip_hard_bytes if preview is not None else 0,
                    "zoomBytes": preview.zoom_bytes if preview is not None else 0,
                    "error": preview.error if preview is not None else "not rendered",
                    **({"diagnostic": preview.diagnostic_previews} if preview is not None and preview.diagnostic_previews is not None else {}),
                },
                **({"diagnostics": preview.diagnostics} if preview is not None and preview.diagnostics is not None else {}),
                "transformToReference": (
                    [[round(value, 6) for value in row] for row in transform] if transform is not None else None
                ),
                "normalization": {
                    "skyOffset": (
                        round(float(preview.sky), 3)
                        if preview is not None and preview.sky is not None
                        else round(float(frame.sky), 3) if frame.sky is not None else None
                    ),
                    "fluxScale": round(float(evidence_flux_scale(evidence, frame)), 6),
                    "registered": registered,
                    "calibrated": preview is not None and preview.calibrated,
                },
            }
        )
    channels: list[dict[str, Any]] = []
    for channel in evidence.channel_records():
        channel_id = channel["channelId"]
        stretch = stretches.get(channel_id)
        channels.append(
            {
                **channel,
                "stretch": stretch.serializable() if stretch is not None else None,
                "previewGeometry": geometry.get(channel_id),
            }
        )
    counts = evidence.counts
    return {
        "schemaVersion": 1,
        "kind": BLINK_MANIFEST_KIND,
        "sessionId": session.name,
        "sessionDirectory": str(session),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "engineVersion": __version__,
        "gatePolicyDigest": gate_policy.canonical_digest(),
        "flagsPolicyDigest": evidence.flags_policy.canonical_digest(),
        "flagsPolicy": evidence.flags_policy.serializable(),
        "referenceRule": "calibrated-local-noise-psf-v2" if request.previews.display_algorithm == "blink-complementary-display-v2" else BLINK_REFERENCE_RULE,
        "inventorySha256": _inventory_sha256(results),
        "workers": workers,
        "previewOptions": request.previews.serializable(),
        "calibration": {
            "masterFlats": [Path(item.path).name for item in request.master_flats],
            "masterDarks": [Path(item.path).name for item in request.master_darks],
            "masterBias": Path(request.master_bias).name if request.master_bias else None,
            "gradientFrames": sum(1 for frame in evidence.inputs if frame.gradient_adu is not None),
            "notes": list(notes),
        },
        "timings": {key: round(value, 3) for key, value in timings.items()},
        "measurement": dict(measurement_stats),
        "analysis": dict(analysis_stats),
        "analysisCache": dict(cache_stats),
        "measurementCache": dict(measurement_cache_stats),
        "counts": counts,
        "channels": channels,
        "frames": frames,
    }


def evidence_flux_scale(evidence: BlinkEvidence, frame: BlinkFrameInput) -> float:
    reference = evidence.references.get(frame.channel_id)
    if reference is None:
        return 1.0
    reference_frame = next((item for item in evidence.inputs if item.path == reference.path), None)
    if (
        reference_frame is None
        or reference_frame.transparency is None
        or frame.transparency is None
        or frame.transparency <= 0
    ):
        return 1.0
    return float(reference_frame.transparency) / float(frame.transparency)


__all__ = [
    "BLINK_EVIDENCE_KIND",
    "BLINK_MANIFEST_KIND",
    "BlinkEvidence",
    "BlinkMaster",
    "BlinkPreviewOptions",
    "BlinkRequest",
    "BlinkSessionError",
    "blink_evidence",
    "calibrated_gradients",
    "run_blink_session",
]
