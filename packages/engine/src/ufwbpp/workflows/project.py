"""Transactional multi-target, mosaic, RGB/LRGB product orchestration.

The single-field :mod:`ufwbpp.workflows.single_target` executor remains the only
raw-frame science path.  This module composes it without weakening any gate:

* READY Light frames are partitioned by acquisition target and filter;
* raw calibration frames are integrated once into a shared master library;
* every target runs once with all of its filters, so the filters of one
  target are registered onto one reference frame, cropped to one common
  rectangle, and each master still completes QC through a fresh solve;
* multi-panel filters are reprojected only from SOLVED panels, pass exact
  coverage/overlap/seam gates, and then receive a new final plate solution;
* filters that already share their pixel grid are copied unchanged onto the
  reference grid (no resampling), while independently solved mosaics are
  reprojected onto it, before optional RGB/LRGB construction; and
* one outer create-only directory rename is the only success commit.

The reprojection WCS of a working mosaic is explicitly never treated as a
solution.  A failed final solve publishes only ``<output>.unsolved`` evidence.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import traceback
from typing import Any, Callable, Mapping, NamedTuple, Sequence

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from lightframeqc.cfa import CHANNEL_NAMES, is_cfa_pattern, normalize_pattern as normalize_cfa_pattern
from lightframeqc.metadata import grouping_keyword_root
from lightframeqc.source_extraction import cached_extraction_self_test

from ..calibration.inputs import MasterMetadataOverride
from ..calibration.matching import BIAS, DARK, FLAT
from ..calibration.policy import MONO_STANDARD, workflow_receipt
from ..integrity import canonical_json_document, sha256_digest
from ..models import AssetRole, AssetStatus, FrameAsset, ProjectInventory
from ..path_budget import DETAILS_DIRECTORY, PROJECT_STAGING_SUFFIX, RUNS_DIRECTORY, target_key
from ..platform import remove_file, rename_with_retry
from ..products.color import (
    ColorProductError,
    ColorProductRequest,
    ColorProductResult,
    build_color_product,
)
from ..products.mosaic import (
    MosaicError,
    MosaicRequest,
    MosaicResult,
    ReprojectProvider,
    build_solved_panel_mosaic,
)
from ..products.preview import render_auto_stretch_preview
from ..solvers.base import SolverBackend, canonical_wcs_sha256, validate_wcs_header
from ..stacking.crop import histogram_rectangle
from ..stacking.integration import CalibrationError
from .common import _fsync_directory, _relativize_solver_attempts, _rename_directory_no_replace, _safe_token
from .contracts import ProgressEvent
from .registration import _build_registration_masters
from .review import bind_review_approval_selections
from .sharing import _sanitize_shareable_tree, _share_safe_value
from .single_target import (
    E2EError,
    E2ERequest,
    E2EResult,
    E2EState,
    ProgressCallback,
    ProgressStage,
    run_e2e,
)
from .solve import (
    _SolverHints,
    _read_image_header,
    _solve_one,
    _solution_geometry,
)
from .sources import (
    _canonical_inputs,
    _capture_sources,
    _input_frame_info,
    _stage_e2e_xisf_inputs,
    _verify_sources,
)


PROJECT_E2E_VERSION = "ultra-fast-wbpp-project-e2e-v1"
# Everything that is not a final channel, a preview or the receipt lives under
# ``DETAILS_DIRECTORY`` (named in ``path_budget`` with the other layout parts).


class ProjectE2EError(RuntimeError):
    """Stable failure at the multi-product transaction boundary."""

    def __init__(self, code: str, message: str, *, path: str | None = None) -> None:
        self.code = code
        self.path = path
        detail = f"{path}: {message}" if path else message
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class SciencePanel:
    target: str
    target_key: str
    filter_name: str
    filter_key: str
    light_files: tuple[str, ...]
    # A colour channel panel of a Bayer (OSC) Light set: the Lights' own
    # filter and the channel the panel's master carries.  None for mono.
    source_filter: str | None = None
    cfa_channel: str | None = None
    cfa_pattern: str | None = None

    @property
    def panel_id(self) -> str:
        return f"{_safe_token(self.target_key)}__{_safe_token(self.filter_key)}"

    def serializable(self) -> dict[str, Any]:
        return {
            "panelId": self.panel_id,
            "target": self.target,
            "targetKey": self.target_key,
            "filter": self.filter_name,
            "filterKey": self.filter_key,
            "lightFiles": list(self.light_files),
            **(
                {
                    "sourceFilter": self.source_filter,
                    "cfaChannel": self.cfa_channel,
                    "cfaPattern": self.cfa_pattern,
                }
                if self.cfa_channel is not None
                else {}
            ),
        }


@dataclass(frozen=True, slots=True)
class ProjectLayout:
    panels: tuple[SciencePanel, ...]

    @property
    def target_keys(self) -> tuple[str, ...]:
        return tuple(sorted({panel.target_key for panel in self.panels}))

    @property
    def filter_keys(self) -> tuple[str, ...]:
        return tuple(sorted({panel.filter_key for panel in self.panels}))

    @property
    def requires_project_orchestration(self) -> bool:
        # Preserve the established single-target ``run`` contract.  Explicit
        # ``run-project`` still creates RGB for a one-target multi-filter set.
        return len(self.target_keys) > 1

    @property
    def target_runs(self) -> tuple[tuple[SciencePanel, tuple[SciencePanel, ...]], ...]:
        """Panels grouped per target: one multi-filter E2E run each.

        Every filter of a target is registered onto the same reference frame
        inside one run, so the filter masters share their pixel grid exactly
        and LRGB composition never has to resample them.  The group descriptor
        is a synthetic panel that carries every Light of the target.
        """

        groups: list[tuple[SciencePanel, tuple[SciencePanel, ...]]] = []
        for target_key in self.target_keys:
            panels = tuple(panel for panel in self.panels if panel.target_key == target_key)
            descriptor = SciencePanel(
                target=panels[0].target,
                target_key=target_key,
                filter_name="+".join(panel.filter_name for panel in panels),
                filter_key="+".join(panel.filter_key for panel in panels),
                # The colour channel panels of a Bayer Light set share their
                # Lights; the run receives each file once.
                light_files=tuple(
                    dict.fromkeys(path for panel in panels for path in panel.light_files)
                ),
            )
            groups.append((descriptor, panels))
        return tuple(groups)

    def serializable(self) -> dict[str, Any]:
        return {
            "panelCount": len(self.panels),
            "targetCount": len(self.target_keys),
            "filterCount": len(self.filter_keys),
            "targets": list(self.target_keys),
            "filters": list(self.filter_keys),
            "panels": [panel.serializable() for panel in self.panels],
        }


@dataclass(frozen=True, slots=True)
class ProjectProgressEvent(ProgressEvent):
    """Add project context without changing the single-panel event contract."""

    overall_fraction: float = 0.0
    scope: str = "project"
    project_stage: str | None = None
    panel: SciencePanel | None = None
    panel_index: int | None = None
    panel_count: int = 0

    def serializable(self) -> dict[str, Any]:
        value = ProgressEvent.serializable(self)
        value.update(overallFraction=self.overall_fraction, scope=self.scope, panelCount=self.panel_count)
        if self.project_stage is not None:
            value["stage"] = self.project_stage
        if self.panel is not None:
            value.update(panelId=self.panel.panel_id, panelTarget=self.panel.target,
                         panelFilter=self.panel.filter_name, panelIndex=self.panel_index)
        return value


class _ProjectProgress:
    """Count planned work, not elapsed time: Light-stage units plus final operations.

    Each panel stage costs its input Light count. Each final image operation
    costs one average panel's Light count; raw shared calibration counts its
    input frames. Opaque operations advance only at real stage boundaries.
    """

    def __init__(self, layout: ProjectLayout, request: E2ERequest, callback: ProgressCallback | None):
        self.layout, self.callback = layout, callback
        self.stages = tuple(stage for stage in (
            ProgressStage.INVENTORY, ProgressStage.QUALITY_CONTROL, ProgressStage.CALIBRATION,
            ProgressStage.REGISTRATION, ProgressStage.INTEGRATION, ProgressStage.DRIZZLE,
            ProgressStage.ASTROMETRY, ProgressStage.PREVIEW, ProgressStage.VERIFY, ProgressStage.PUBLISH,
        ) if stage is not ProgressStage.DRIZZLE or request.integration_mode.value == "drizzle")
        lights = sum(len(panel.light_files) for panel in layout.panels)
        image_unit = lights / len(layout.panels)
        self.run_count = len(layout.target_runs)
        mosaics = sum(sum(panel.filter_key == key for panel in layout.panels) > 1 for key in layout.filter_keys)
        self.phase_units = {
            "prepare": max(1, len(request.bias_files) + len(request.dark_files) + len(request.flat_files)),
            "mosaic": image_unit * 2 * mosaics,
            "alignment": image_unit * len(layout.filter_keys),
            "color": image_unit * (len(layout.filter_keys) + int({"r", "g", "b"}.issubset(layout.filter_keys))),
            "verify": image_unit,
            "publish": image_unit,
        }
        self.total_units = lights * len(self.stages) + sum(self.phase_units.values())
        self.completed: dict[str, float] = {}
        self.last_fraction = 0.0

    def _emit(self, event: ProgressEvent, key: str, units: float, **context: Any) -> None:
        self.completed[key] = max(self.completed.get(key, 0.0), units)
        self.last_fraction = max(self.last_fraction, min(0.99, sum(self.completed.values()) / self.total_units))
        if self.callback is not None:
            self.callback(ProjectProgressEvent(
                event.stage, event.status, event.current, event.total, event.message,
                overall_fraction=self.last_fraction, panel_count=self.run_count, **context,
            ))

    def phase(self, name: str, current: int, total: int, message: str) -> None:
        fraction = min(1.0, max(0.0, current / total)) if total else 1.0
        self._emit(ProgressEvent(ProgressStage.VERIFY, "completed" if current == total else "running",
                                 current, total, message), name, self.phase_units[name] * fraction,
                   project_stage=name)

    def panel_callback(self, index: int, panel: SciencePanel) -> ProgressCallback:
        def forward(event: ProgressEvent) -> None:
            stage = ProgressStage.PUBLISH if event.stage is ProgressStage.COMPLETE else event.stage
            key = f"panel:{panel.panel_id}:{stage.value}"
            units = self.completed.get(key, 0.0)
            if stage in self.stages:
                fraction = (min(1.0, max(0.0, event.current / event.total)) if event.total > 0
                            else 1.0 if event.status == "completed" else 0.0)
                units = len(panel.light_files) * fraction
            self._emit(replace(event, stage=stage), key, units, scope="panel", panel=panel, panel_index=index)
        return forward

    def panel_done(self, index: int, panel: SciencePanel) -> None:
        # A successfully returned, receipt-checked panel proves all its stages,
        # including runners that only emit coarse or no intermediate events.
        for stage in self.stages:
            self.completed[f"panel:{panel.panel_id}:{stage.value}"] = len(panel.light_files)
        self.panel_callback(index, panel)(ProgressEvent(ProgressStage.PUBLISH, "completed", 1, 1,
                                                       f"panel {panel.target}/{panel.filter_name} verified"))


@dataclass(frozen=True, slots=True)
class ProjectE2ERequest:
    inventory: ProjectInventory
    e2e_request: E2ERequest
    output_directory: str
    minimum_mosaic_covered_fraction: float = 0.70
    minimum_pair_overlap_pixels: int = 16
    minimum_pair_overlap_fraction: float = 0.005
    maximum_seam_normalized_mad: float = 0.25
    minimum_channel_alignment_coverage: float = 0.98
    channel_wcs_tolerance_pixels: float = 0.05
    # GUI/CLI REVIEW admissions as the reviewer made them: the preflight
    # Light digest and the gate-policy digest.  They are bound into full
    # approvals per target run, against that run's own request, so a project
    # with several targets or filters can admit reviewed frames too.
    review_selections: tuple[Mapping[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectE2EResult:
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
    mono_filters: tuple[str, ...] = ()
    color_product_path: str | None = None
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
            "monoFilters": list(self.mono_filters),
            "colorProductPath": self.color_product_path,
            "message": self.message,
        }


PanelRunner = Callable[..., E2EResult]
MosaicBuilder = Callable[..., MosaicResult]
ColorBuilder = Callable[..., ColorProductResult]


_CALIBRATION_TARGETS = {
    "bias",
    "dark",
    "flat",
    "calibration",
    "calibrations",
    "masterbias",
    "masterdark",
    "masterflat",
}
_FILTER_ALIASES = {
    "r": "R",
    "red": "R",
    "g": "G",
    "green": "G",
    "b": "B",
    "blue": "B",
    "l": "L",
    "lum": "L",
    "luminance": "L",
    "ha": "HA",
    "halpha": "HA",
    "hydrogenalpha": "HA",
    "oiii": "OIII",
    "oxygeniii": "OIII",
    "sii": "SII",
    "sulfurii": "SII",
    "sulphurii": "SII",
}


def _normalized_token(value: str) -> str:
    return target_key(value)


def _science_target(asset: FrameAsset) -> tuple[str, str]:
    display = str(asset.target).strip()
    key = _normalized_token(display)
    if not key or key in {"unknown", "none", "unspecified"}:
        raise ProjectE2EError(
            "SCIENCE_TARGET_MISSING",
            "every Light requires an explicit NINA OBJECT/target so panels cannot be mixed",
            path=asset.path,
        )
    if key in _CALIBRATION_TARGETS:
        raise ProjectE2EError(
            "CALIBRATION_TARGET_AS_LIGHT",
            "a Light is labelled as a calibration target",
            path=asset.path,
        )
    return display, key


def _science_filter(asset: FrameAsset) -> tuple[str, str]:
    display = str(asset.filter_name).strip()
    key = _normalized_token(display)
    if not key or key in {"unknown", "none", "unspecified"}:
        raise ProjectE2EError(
            "SCIENCE_FILTER_MISSING",
            "every Light requires explicit FILTER metadata",
            path=asset.path,
        )
    canonical = _FILTER_ALIASES.get(key, display.upper())
    return canonical, _normalized_token(canonical)


def classify_project_layout(inventory: ProjectInventory) -> ProjectLayout:
    """Classify READY Lights into deterministic scientific target/filter panels."""

    if not isinstance(inventory, ProjectInventory):
        raise ProjectE2EError("PROJECT_INVENTORY_INVALID", "inventory has the wrong type")
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for asset in inventory.assets:
        if asset.role is not AssetRole.LIGHT:
            continue
        if asset.status is not AssetStatus.READY:
            raise ProjectE2EError(
                "LIGHT_NOT_READY", "conflicted or unreadable Light cannot enter a panel", path=asset.path
            )
        target, target_key = _science_target(asset)
        panel = dict(asset.grouping_keywords).get("PANEL")
        if panel is not None:
            # WBPP's PANEL keyword (PANEL_2/): one mosaic panel per value, as
            # its post-processing grouping integrates them separately.
            target = f"{target} PANEL {panel}"
            target_key = _normalized_token(target)
        filter_name, filter_key = _science_filter(asset)
        pattern = normalize_cfa_pattern(asset.cfa_pattern)
        # A Bayer (OSC) Light yields the three colour channel panels R, G and
        # B of its target; the pixel pipeline debayers it into those groups.
        if is_cfa_pattern(pattern):
            expansions = [
                (channel, _normalized_token(channel), filter_name, channel, pattern)
                for channel in CHANNEL_NAMES
            ]
        else:
            expansions = [(filter_name, filter_key, None, None, None)]
        for panel_filter, panel_key, source_filter, channel, panel_pattern in expansions:
            group = grouped.setdefault(
                (target_key, panel_key),
                {
                    "target": target,
                    "targetKey": target_key,
                    "filter": panel_filter,
                    "filterKey": panel_key,
                    "sourceFilter": source_filter,
                    "cfaChannel": channel,
                    "cfaPattern": panel_pattern,
                    "lights": [],
                },
            )
            if (group["cfaChannel"], group["sourceFilter"], group["cfaPattern"]) != (channel, source_filter, panel_pattern):
                raise ProjectE2EError(
                    "CFA_CHANNEL_FILTER_COLLISION",
                    f"target {target} mixes a Bayer Light set with mono {panel_filter} Lights or a second "
                    "Bayer filter; one target takes one Bayer filter and no mono R/G/B filters beside it",
                    path=asset.path,
                )
            group["lights"].append(asset.path)
    if not grouped:
        raise ProjectE2EError("NO_LIGHTS", "project inventory contains no READY Lights")
    panels = tuple(
        SciencePanel(
            target=value["target"],
            target_key=value["targetKey"],
            filter_name=value["filter"],
            filter_key=value["filterKey"],
            light_files=tuple(sorted(value["lights"], key=str.casefold)),
            source_filter=value["sourceFilter"],
            cfa_channel=value["cfaChannel"],
            cfa_pattern=value["cfaPattern"],
        )
        for _, value in sorted(grouped.items())
    )
    return ProjectLayout(panels)


def project_requires_orchestration(inventory: ProjectInventory) -> bool:
    return classify_project_layout(inventory).requires_project_orchestration



def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    data = canonical_json_document(value)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _artifact(path: Path, root: Path, kind: str, *, digests: Mapping[Path, str] | None = None) -> dict[str, Any]:
    info = path.stat()
    digest = digests.get(path) if digests is not None else None
    return {
        "path": path.relative_to(root).as_posix(),
        "relativePath": path.relative_to(root).as_posix(),
        "kind": kind,
        "sha256": digest if digest is not None else sha256_digest(path),
        "sizeBytes": info.st_size,
    }


def _sha256_many(paths: Sequence[Path], *, workers: int = 4) -> dict[Path, str]:
    """Digest independent files concurrently; hashing releases the GIL, so the
    final products (a gigabyte-scale RGB cube beside the masters) overlap."""

    unique = list(dict.fromkeys(paths))
    if len(unique) <= 1 or workers <= 1:
        return {path: sha256_digest(path) for path in unique}
    with ThreadPoolExecutor(max_workers=min(workers, len(unique)), thread_name_prefix="ufwbpp-digest") as pool:
        return dict(zip(unique, pool.map(sha256_digest, unique), strict=True))


def _screening_record(receipt_path: Path, staging: Path, target: str) -> dict[str, Any] | None:
    """The run's screening summary with preview paths relative to the project."""

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    screening = receipt.get("qualityControl", {}).get("screening")
    if not isinstance(screening, Mapping):
        return None
    run_root = receipt_path.parent.relative_to(staging).as_posix()
    frames = []
    for frame in screening.get("frames", []):
        if not isinstance(frame, Mapping):
            continue
        preview = frame.get("reviewPreview")
        frames.append(
            {
                **frame,
                "target": target,
                "reviewPreview": f"{run_root}/{preview}" if isinstance(preview, str) else None,
            }
        )
    return {
        "target": target,
        "counts": dict(screening.get("counts", {})),
        "admitted": screening.get("admitted"),
        "excluded": screening.get("excluded"),
        "frames": frames,
        "selectionPolicy": receipt.get("qualityControl", {}).get("selectionPolicy"),
    }


def _project_screening(subruns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Screening across every target run: totals plus the frames needing a decision."""

    counts: dict[str, int] = {}
    frames: list[dict[str, Any]] = []
    admitted = 0
    excluded = 0
    policies: list[str] = []
    for record in subruns:
        screening = record.get("screening")
        if not isinstance(screening, Mapping):
            continue
        for key, value in screening.get("counts", {}).items():
            counts[key] = counts.get(key, 0) + int(value)
        admitted += int(screening.get("admitted") or 0)
        excluded += int(screening.get("excluded") or 0)
        frames.extend(screening.get("frames", []))
        policy = screening.get("selectionPolicy")
        if isinstance(policy, str) and policy not in policies:
            policies.append(policy)
    return {
        "counts": counts,
        "admitted": admitted,
        "excluded": excluded,
        "frames": frames,
        # Every run of a project shares the request's policy; None until a
        # run has reported one.
        "selectionPolicy": policies[0] if len(policies) == 1 else None,
    }


def _accepted_quality(receipt_path: Path, filter_name: str) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    filters = receipt.get("astrometry", {}).get("filters", {})
    record = filters.get(filter_name)
    if not isinstance(record, Mapping):
        raise ProjectE2EError(
            "PANEL_ASTROMETRY_EVIDENCE_MISSING",
            f"panel receipt has no astrometry record for {filter_name}",
        )
    attempts = record.get("attempts")
    accepted = (
        next(
            (
                item
                for item in reversed(attempts)
                if isinstance(item, Mapping) and item.get("accepted") is True
            ),
            None,
        )
        if isinstance(attempts, list)
        else None
    )
    quality = (
        accepted.get("result", {}).get("astrometricQuality")
        if isinstance(accepted, Mapping)
        else None
    )
    if not isinstance(quality, Mapping) or quality.get("catalogManaged") is not True:
        raise ProjectE2EError(
            "PANEL_ASTROMETRY_EVIDENCE_MISSING",
            "panel final product lacks managed catalog correspondence evidence",
        )
    return dict(quality)


def _astrometry_gui_evidence(path: Path, quality: Mapping[str, Any]) -> dict[str, Any]:
    try:
        header, shape = _read_image_header(path)
    except E2EError:
        with fits.open(path, mode="readonly", memmap=True, checksum=True) as hdul:
            hdu = next(
                (item for item in hdul
                 if item.data is not None and item.data.ndim == 3 and item.data.shape[0] == 3),
                None,
            )
            if hdu is None:
                raise ProjectE2EError(
                    "FINAL_PRODUCT_GEOMETRY_INVALID", "expected a mono image or three-plane RGB cube", path=str(path)
                )
            header = hdul[0].header.copy()
            if hdu is not hdul[0]:
                header.extend(hdu.header, update=True, strip=True)
            shape = tuple(int(value) for value in hdu.data.shape[-2:])
    # A linear RGB product has NAXIS=3, but its sky coordinates and SIP
    # distortion describe only the two spatial axes. Select those axes before
    # constructing WCS; .celestial alone runs too late for SIP validation.
    try:
        spatial_wcs = WCS(header, naxis=2, relax=False).celestial
        if spatial_wcs.pixel_n_dim != 2 or not spatial_wcs.has_celestial:
            raise ValueError("the selected spatial axes are not a two-dimensional celestial WCS")
        spatial_header = spatial_wcs.to_header(relax=True)
    except Exception as error:
        raise ProjectE2EError("FINAL_PRODUCT_WCS_INVALID", str(error), path=str(path)) from error
    validation = validate_wcs_header(spatial_header, image_shape=shape)
    if not validation.valid:
        raise ProjectE2EError("FINAL_PRODUCT_WCS_INVALID", validation.message, path=str(path))
    geometry = _solution_geometry(spatial_header, shape)
    return {
        **dict(quality),
        "referenceFrame": str(header.get("RADESYS", "ICRS") or "ICRS"),
        "projection": str(header.get("CTYPE1", "RA---TAN")).upper().split("---")[-1].split("-")[0],
        "wcsSha256": canonical_wcs_sha256(spatial_header),
        "centerRaDegrees": geometry["centerRaDegrees"],
        "centerDecDegrees": geometry["centerDecDegrees"],
        "pixelScaleArcsec": geometry["pixelScaleArcsec"],
        "rotationDegrees": geometry["rotationDegrees"],
        "imageShape": geometry["imageShape"],
        "state": "SOLVED",
    }


def _validate_request(request: ProjectE2ERequest) -> tuple[Path, Path, ProjectLayout]:
    if not isinstance(request, ProjectE2ERequest):
        raise ProjectE2EError("PROJECT_REQUEST_INVALID", "request has the wrong type")
    layout = classify_project_layout(request.inventory)
    output = Path(request.output_directory).expanduser().resolve(strict=False)
    evidence = output.with_name(output.name + ".unsolved")
    for path, code in ((output, "OUTPUT_EXISTS"), (evidence, "FAILURE_OUTPUT_EXISTS")):
        if os.path.lexists(path):
            raise ProjectE2EError(code, "destination must be new", path=str(path))
    if Path(request.e2e_request.output_directory).expanduser().resolve(strict=False) != output:
        raise ProjectE2EError(
            "PROJECT_OUTPUT_MISMATCH",
            "outer output and base E2E output must name the same directory",
        )
    base_lights = {
        str(path)
        for path in _canonical_inputs(request.e2e_request.light_files, "LIGHT", required=True)
    }
    inventory_lights = {path for panel in layout.panels for path in panel.light_files}
    if base_lights != inventory_lights:
        raise ProjectE2EError(
            "PROJECT_LIGHT_SET_MISMATCH",
            "base E2E request Lights do not match the inventory panel partition",
        )
    if request.e2e_request.review_approvals and len(layout.panels) != 1:
        raise ProjectE2EError(
            "PROJECT_REVIEW_APPROVAL_SCOPE_UNSUPPORTED",
            "pre-bound REVIEW approvals are supported only when the project contains one target/filter panel; pass review selections instead",
        )
    if request.e2e_request.review_approvals and request.review_selections:
        raise ProjectE2EError(
            "PROJECT_REVIEW_APPROVAL_SCOPE_UNSUPPORTED",
            "a project request carries either pre-bound approvals or review selections, not both",
        )
    if request.e2e_request.explicit_selection is not None and request.review_selections:
        raise ProjectE2EError(
            "SELECTION_POLICY_CONFLICT",
            "a project request carries either an explicit selection or review selections, not both",
        )
    digest_pattern = re.compile(r"^sha256:[0-9a-f]{64}$")
    seen_selection_digests: set[str] = set()
    for position, selection in enumerate(request.review_selections):
        if (
            not isinstance(selection, Mapping)
            or set(selection) != {"sourceSha256", "gatePolicyDigest"}
            or any(not isinstance(value, str) or digest_pattern.fullmatch(value) is None for value in selection.values())
            or selection["sourceSha256"] in seen_selection_digests
        ):
            raise ProjectE2EError(
                "REVIEW_SELECTION_INVALID",
                f"review selection {position} must carry unique lowercase sourceSha256/gatePolicyDigest values",
            )
        seen_selection_digests.add(selection["sourceSha256"])
    numeric = {
        "minimum_mosaic_covered_fraction": request.minimum_mosaic_covered_fraction,
        "minimum_pair_overlap_fraction": request.minimum_pair_overlap_fraction,
        "maximum_seam_normalized_mad": request.maximum_seam_normalized_mad,
        "minimum_channel_alignment_coverage": request.minimum_channel_alignment_coverage,
        "channel_wcs_tolerance_pixels": request.channel_wcs_tolerance_pixels,
    }
    for name, value in numeric.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ProjectE2EError("PROJECT_GATE_INVALID", f"{name} must be finite and positive")
    if request.minimum_mosaic_covered_fraction > 1 or request.minimum_pair_overlap_fraction > 1:
        raise ProjectE2EError("PROJECT_GATE_INVALID", "coverage fractions cannot exceed 1")
    if not 0.9 <= request.minimum_channel_alignment_coverage <= 1.0:
        raise ProjectE2EError(
            "PROJECT_GATE_INVALID",
            "minimum_channel_alignment_coverage must be in [0.9, 1]",
        )
    return output, evidence, layout


def _all_sources(request: E2ERequest) -> tuple[Any, ...]:
    grouped = (
        ("LIGHT", request.light_files),
        ("FLAT", request.flat_files),
        ("DARK", request.dark_files),
        ("BIAS", request.bias_files),
        ("MASTER_BIAS", request.master_bias_files),
        ("MASTER_DARK", request.master_dark_files),
        ("MASTER_FLAT", request.master_flat_files),
    )
    flattened: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for role, values in grouped:
        for path in _canonical_inputs(values, role, required=False):
            key = os.path.normcase(str(path))
            if key in seen:
                raise ProjectE2EError("INPUT_ROLE_OVERLAP", "one source appears more than once", path=str(path))
            seen.add(key)
            flattened.append((role, path))
    return tuple(_capture_sources(flattened, workers=max(1, int(request.workers))))


def _build_shared_calibration(
    base: E2ERequest,
    root: Path,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[MasterMetadataOverride, ...],
    dict[str, Any],
]:
    """Build raw calibration only once and return supplied-master inputs."""

    raw_present = bool(base.bias_files or base.dark_files or base.flat_files)
    if not raw_present:
        receipt = {
            "schemaVersion": 1,
            "stage": "shared-calibration-library",
            "calibrationPolicy": workflow_receipt(base.pipeline_parameters.calibration_workflow),
            "mode": "REUSED_SUPPLIED_MASTERS",
            "masterBiasFiles": list(base.master_bias_files),
            "masterDarkFiles": list(base.master_dark_files),
            "masterFlatFiles": list(base.master_flat_files),
            "rawIntegrationCount": 0,
        }
        _write_json(root / "receipt.json", receipt)
        return (
            base.master_bias_files,
            base.master_dark_files,
            base.master_flat_files,
            (),
            receipt,
        )

    raw_groups = (
        ("BIAS", _canonical_inputs(base.bias_files, "BIAS", required=False)),
        ("DARK", _canonical_inputs(base.dark_files, "DARK", required=False)),
        ("FLAT", _canonical_inputs(base.flat_files, "FLAT", required=False)),
        (
            "MASTER_BIAS",
            _canonical_inputs(base.master_bias_files, "MASTER_BIAS", required=False),
        ),
        (
            "MASTER_DARK",
            _canonical_inputs(base.master_dark_files, "MASTER_DARK", required=False),
        ),
        (
            "MASTER_FLAT",
            _canonical_inputs(base.master_flat_files, "MASTER_FLAT", required=False),
        ),
    )
    staged, aliases, conversions, _ = _stage_e2e_xisf_inputs(
        raw_groups, root / "pixel-inputs", base.pipeline_parameters
    )
    calibration = _build_registration_masters(
        biases=staged["BIAS"],
        darks=staged["DARK"],
        flats=staged["FLAT"],
        supplied_biases=staged["MASTER_BIAS"],
        supplied_darks=staged["MASTER_DARK"],
        supplied_flats=staged["MASTER_FLAT"],
        lights=_canonical_inputs(base.light_files, "LIGHT", required=True),
        directory=root / "masters",
        pipeline_parameters=base.pipeline_parameters,
        source_aliases=aliases,
        xisf_conversions=conversions,
    )
    def master_source(value: str) -> str:
        # Reused XISF masters were decoded only for this shared build. Their
        # metadata overrides bind the original content, and each panel must
        # establish its own conversion evidence from that same source. Newly
        # generated masters have no alias and retain their derived identity.
        return str(aliases.get(str(Path(value)), Path(value)))

    raw_receipt = calibration.receipt
    generated_root = (root / "masters").resolve(strict=True)

    def raw_info(path: Path) -> Any:
        original = aliases.get(str(path), path)
        return _input_frame_info(
            path,
            base.pipeline_parameters,
            override_identity_path=original,
        )

    def generated_override(
        generated: Path,
        reference: Any,
        *,
        bias_included: bool | None,
        declare_additive_numeric_domain: bool,
        grouping_keywords: tuple[tuple[str, str], ...],
    ) -> MasterMetadataOverride:
        if reference.temperature_celsius is None and base.pipeline_parameters.calibration_workflow != MONO_STANDARD:
            raise ProjectE2EError(
                "RAW_MASTER_METADATA_MISSING",
                "shared generated-master metadata requires a known raw reference temperature; no value can be inferred safely",
                path=reference.path,
            )
        return MasterMetadataOverride(
            source_sha256=sha256_digest(generated),
            camera=reference.camera,
            gain=reference.gain,
            offset=reference.offset,
            binning_x=reference.binning_x,
            binning_y=reference.binning_y,
            filter_name=reference.filter_name,
            cfa_pattern=reference.cfa_pattern,
            readout_mode=reference.readout_mode,
            temperature_celsius=reference.temperature_celsius,
            exposure_seconds=reference.exposure_seconds,
            bias_included=bias_included,
            numeric_domain=(
                (
                    "NORMALIZED_UNIT"
                    if reference.numeric_domain == "NORMALIZED_UNIT"
                    else "SENSOR_CODE"
                )
                if declare_additive_numeric_domain
                else None
            ),
            normalized_unit_scale=(
                reference.normalized_unit_scale
                if declare_additive_numeric_domain
                else None
            ),
            grouping_keywords=grouping_keywords,
        )

    # Every master a Light of the project uses, one per calibration group:
    # a generated one carries its group's metadata and grouping keywords in a
    # content-bound override, a supplied one keeps its own path.
    trusted_generated: list[MasterMetadataOverride] = []
    outputs: dict[str, list[str]] = {BIAS: [], DARK: [], FLAT: []}
    for kind in (BIAS, DARK, FLAT):
        for key in sorted(calibration.match.used(kind)):
            group = calibration.match.groups[kind][key]
            master = calibration.masters[(kind, key)]
            resolved = master.resolve(strict=True)
            if group.supplied_master or not resolved.is_relative_to(generated_root):
                outputs[kind].append(master_source(str(master)))
                continue
            outputs[kind].append(str(resolved))
            trusted_generated.append(
                generated_override(
                    resolved,
                    raw_info(Path(group.members[0])),
                    bias_included=True if kind == DARK else None,
                    declare_additive_numeric_domain=kind != FLAT,
                    grouping_keywords=group.traits.keywords,
                )
            )
    master_biases = tuple(outputs[BIAS])
    master_darks = tuple(outputs[DARK])
    master_flats = tuple(outputs[FLAT])
    receipt = {
        "schemaVersion": 1,
        "stage": "shared-calibration-library",
        "mode": "BUILT_ONCE_OR_REUSED",
        "rawIntegrationCount": 1,
        "registrationCalibration": raw_receipt,
        "masterBiasFiles": list(master_biases),
        "masterDarkFiles": list(master_darks),
        "masterFlatFiles": list(master_flats),
        "trustedGeneratedMasterOverrides": [
            {
                **item.serializable(),
                "trustBoundary": "UFWBPP_SHARED_CALIBRATION_OUTPUT",
            }
            for item in trusted_generated
        ],
    }
    _write_json(root / "receipt.json", receipt)
    return master_biases, master_darks, master_flats, tuple(trusted_generated), receipt


def _find_filter_products(result: E2EResult, expected_filters: Sequence[str]) -> dict[str, Path]:
    """Bind one solved master to each expected filter of a target run."""

    if not result.success or result.state is not E2EState.SOLVED:
        raise ProjectE2EError(
            result.code,
            "target subrun did not publish solved products",
            path=result.evidence_directory,
        )
    paths = [Path(path).resolve(strict=True) for path in result.product_paths]
    if len(paths) != len(expected_filters):
        raise ProjectE2EError(
            "PANEL_PRODUCT_MULTIPLICITY_INVALID",
            f"one target must produce exactly one master per filter "
            f"({len(expected_filters)}), found {len(paths)}",
        )
    products: dict[str, Path] = {}
    for path in paths:
        header, shape = _read_image_header(path)
        validation = validate_wcs_header(header, image_shape=shape)
        if not validation.valid or header.get("OAFSTATE") != "SOLVED" or header.get("OAFWCS") != "SOLVED":
            raise ProjectE2EError(
                "PANEL_PRODUCT_NOT_SOLVED",
                f"{validation.code}: {validation.message}",
                path=str(path),
            )
        observed = _FILTER_ALIASES.get(
            _normalized_token(str(header.get("FILTER", ""))),
            str(header.get("FILTER", "")).upper(),
        )
        if observed not in expected_filters or observed in products:
            raise ProjectE2EError(
                "PANEL_PRODUCT_FILTER_MISMATCH",
                f"expected one master for each of {list(expected_filters)}, found {header.get('FILTER')!r}",
                path=str(path),
            )
        products[observed] = path
    return products


def _final_mosaic_hints(path: Path, base: E2ERequest) -> _SolverHints:
    header, shape = _read_image_header(path)
    geometry = _solution_geometry(header, shape)
    return _SolverHints(
        ra_degrees=float(geometry["centerRaDegrees"]),
        dec_degrees=float(geometry["centerDecDegrees"]),
        field_of_view_degrees=float(geometry["fieldWidthDegrees"]),
        search_radius_degrees=(
            base.search_radius_degrees
            if base.search_radius_degrees is not None
            else max(2.0, float(geometry["fieldWidthDegrees"]) * 0.75)
        ),
        provenance="VERIFIED_REPROJECTION_GRID_HINT_ONLY",
        evidence={
            "workingMosaicSha256": sha256_digest(path),
            "workingMosaicState": str(header.get("OAFSTATE")),
            "propagatedWcsIsFinalSolution": False,
            "geometry": geometry,
        },
    )


def _supports_keyword(function: Callable[..., Any], name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in parameters
    )


def _source_identity(path: Path) -> dict[str, Any]:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ProjectE2EError("PRODUCT_SOURCE_INVALID", "expected a regular file", path=str(path))
    return {
        "path": str(path),
        "sha256": sha256_digest(path),
        "sizeBytes": info.st_size,
        "mtimeNs": info.st_mtime_ns,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def _five_point_pixel_error(reference: Path, source: Path) -> tuple[float, tuple[int, int], tuple[int, int]]:
    reference_header, reference_shape = _read_image_header(reference)
    source_header, source_shape = _read_image_header(source)
    reference_wcs = WCS(reference_header, relax=False).celestial
    source_wcs = WCS(source_header, relax=False).celestial
    height, width = source_shape
    points = np.asarray(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [0.0, height - 1.0],
            [width - 1.0, height - 1.0],
            [(width - 1.0) / 2.0, 0.0],
            [(width - 1.0) / 2.0, height - 1.0],
            [0.0, (height - 1.0) / 2.0],
            [width - 1.0, (height - 1.0) / 2.0],
            [(width - 1.0) / 2.0, (height - 1.0) / 2.0],
        ],
        dtype=np.float64,
    )
    sky = source_wcs.all_pix2world(points, 0)
    projected = reference_wcs.all_world2pix(sky, 0)
    if not np.all(np.isfinite(projected)):
        return math.inf, reference_shape, source_shape
    return float(np.max(np.abs(projected - points))), reference_shape, source_shape


def _publish_file(source: Path, destination: Path) -> str:
    """Expose ``source`` at ``destination`` without a second copy on disk.

    Final products are the run-level masters and color files byte for byte,
    so they are hard links wherever the volume allows one; a volume without
    hard links gets a verified copy.  Returns the mode used.  ``destination``
    is create-only either way.
    """

    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=4 * 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if sha256_digest(source) != sha256_digest(destination):
            raise ProjectE2EError(
                "ARTIFACT_HASH_MISMATCH", "published product differs from its source", path=str(destination)
            )
        return "COPY"
    # Solver outputs are born private (0600, from a temporary file); a final
    # product gets the permissions a file created here would have.
    os.chmod(destination, 0o666 & ~_current_umask())
    return "HARDLINK"


def _current_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _channel_sky_overlap(reference: Path, source: Path) -> dict[str, Any]:
    """Screen sky displacement independently of pixel-axis rotation or parity.

    This is a coarse preflight only. The actual reprojection footprint must
    still pass the stricter full-resolution channel coverage requirement.
    """
    reference_header, reference_shape = _read_image_header(reference)
    source_header, source_shape = _read_image_header(source)
    reference_wcs = WCS(reference_header, relax=False).celestial
    source_wcs = WCS(source_header, relax=False).celestial
    rh, rw = reference_shape
    sh, sw = source_shape
    reference_center = reference_wcs.pixel_to_world((rw - 1) / 2, (rh - 1) / 2)
    source_center = source_wcs.pixel_to_world((sw - 1) / 2, (sh - 1) / 2)
    center_separation = float(reference_center.separation(source_center).degree)
    reference_edges = reference_wcs.pixel_to_world(
        np.asarray([0.0, rw - 1.0, (rw - 1) / 2, (rw - 1) / 2]),
        np.asarray([(rh - 1) / 2, (rh - 1) / 2, 0.0, rh - 1.0]),
    )
    reference_span = max(
        float(reference_edges[0].separation(reference_edges[1]).degree),
        float(reference_edges[2].separation(reference_edges[3]).degree),
    )
    # Retain the quarter-field gross-displacement bound, now in sky angles.
    maximum_center_separation = 0.25 * reference_span
    xx, yy = np.meshgrid(np.linspace(0, rw - 1, 9), np.linspace(0, rh - 1, 9))
    sky = reference_wcs.all_pix2world(np.column_stack((xx.ravel(), yy.ravel())), 0)
    projected = source_wcs.all_world2pix(sky, 0, quiet=True)
    inside = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= -0.5)
        & (projected[:, 0] <= sw - 0.5)
        & (projected[:, 1] >= -0.5)
        & (projected[:, 1] <= sh - 0.5)
    )
    return {
        "centerSeparationDegrees": center_separation,
        "referenceAngularSpanDegrees": reference_span,
        "maximumCenterSeparationDegrees": maximum_center_separation,
        "footprintSampleCount": int(inside.size),
        "overlappingFootprintSamples": int(np.count_nonzero(inside)),
        "sampledReferenceCoverageFraction": float(np.mean(inside)),
    }


def _align_channel(
    source: Path,
    reference: Path,
    destination: Path,
    *,
    filter_name: str,
    reference_filter: str,
    provider: ReprojectProvider,
    tolerance_pixels: float,
    minimum_coverage: float,
) -> dict[str, Any]:
    source_identity = _source_identity(source)
    reference_identity = _source_identity(reference)
    error, reference_shape, source_shape = _five_point_pixel_error(reference, source)
    if source_shape == reference_shape and error <= tolerance_pixels:
        publication = _publish_file(source, destination)
        return {
            "filter": filter_name,
            "mode": "EXACT_GEOMETRY_COPY",
            "referenceFilter": reference_filter,
            "wcsProvenance": "INDEPENDENT_SOLVE",
            "publication": publication,
            "maximumFivePointErrorPixels": error,
            "maximumNinePointResidualPixels": error,
            "sampleCount": 9,
            "coverageFraction": 1.0,
            "source": source_identity,
            "reference": reference_identity,
            "output": _source_identity(destination),
        }

    sky_overlap = _channel_sky_overlap(reference, source)
    separation = sky_overlap["centerSeparationDegrees"]
    maximum_separation = sky_overlap["maximumCenterSeparationDegrees"]
    if (
        not math.isfinite(error)
        or not math.isfinite(separation)
        or not math.isfinite(maximum_separation)
        or maximum_separation <= 0
        or separation > maximum_separation
        or sky_overlap["overlappingFootprintSamples"] == 0
    ):
        raise ProjectE2EError(
            "CHANNEL_WCS_GROSS_MISMATCH",
            f"{filter_name} sky center differs by {separation:.6g} degrees from "
            f"{reference_filter} (maximum {maximum_separation:.6g}); "
            f"{sky_overlap['overlappingFootprintSamples']}/"
            f"{sky_overlap['footprintSampleCount']} reference footprint samples overlap",
            path=str(source),
        )

    function = provider.reproject_function
    if not callable(function):
        raise ProjectE2EError(
            "CHANNEL_ALIGNMENT_BACKEND_UNAVAILABLE",
            "channel grids differ and no per-image reprojection backend is available",
        )
    source_header, _ = _read_image_header(source)
    reference_header, _ = _read_image_header(reference)
    with fits.open(source, mode="readonly", memmap=False, checksum=True) as hdul:
        source_hdu = next(item for item in hdul if item.data is not None and item.data.ndim == 2)
        source_data = np.asarray(source_hdu.data, dtype=np.float32).copy()
    source_wcs = WCS(source_header, relax=False).celestial
    reference_wcs = WCS(reference_header, relax=False).celestial
    kwargs: dict[str, Any] = {}
    if _supports_keyword(function, "shape_out"):
        kwargs["shape_out"] = reference_shape
    if _supports_keyword(function, "return_footprint"):
        kwargs["return_footprint"] = True
    try:
        raw = function((source_data, source_wcs), reference_wcs, **kwargs)
    except Exception as error_value:
        raise ProjectE2EError("CHANNEL_ALIGNMENT_FAILED", str(error_value), path=str(source)) from error_value
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise ProjectE2EError("CHANNEL_ALIGNMENT_INVALID", "backend must return science and footprint")
    science = np.asarray(raw[0], dtype=np.float32)
    footprint = np.asarray(raw[1], dtype=np.float32)
    if science.shape != reference_shape or footprint.shape != reference_shape:
        raise ProjectE2EError("CHANNEL_ALIGNMENT_INVALID", "backend returned the wrong geometry")
    covered = footprint > 0
    fraction = float(np.count_nonzero(covered) / footprint.size)
    if fraction < minimum_coverage or not np.all(np.isfinite(science[covered])):
        raise ProjectE2EError(
            "CHANNEL_ALIGNMENT_COVERAGE_FAILED",
            f"{filter_name} alignment coverage {fraction:.6g} is below {minimum_coverage:.6g}",
        )
    science = science.copy()
    science[~covered] = np.nan
    header = reference_wcs.to_header(relax=True)
    for key in ("OBJECT", "INSTRUME"):
        if key in source_header:
            header[key] = source_header[key]
    header["FILTER"] = filter_name
    header["OAFSTATE"] = "SOLVED"
    header["OAFWCS"] = "SOLVED"
    header["OAFPROD"] = "ALIGNED_MONO"
    header["OAFALGN"] = (reference_filter, "Color reference filter")
    header["OAFWCSPR"] = ("REPROJECTED_INDEPENDENT_SOLVE", "WCS provenance of this grid")
    header.add_history("Ultra-Fast WBPP: independently solved mono reprojected to solved color reference grid")
    fits.writeto(destination, science, header, overwrite=False, checksum=True)
    validation = validate_wcs_header(header, image_shape=reference_shape)
    if not validation.valid:
        raise ProjectE2EError("CHANNEL_ALIGNMENT_WCS_INVALID", validation.message)
    output_error, output_reference_shape, output_shape = _five_point_pixel_error(
        reference, destination
    )
    if (
        output_reference_shape != output_shape
        or not math.isfinite(output_error)
        or output_error > tolerance_pixels
    ):
        raise ProjectE2EError(
            "CHANNEL_ALIGNMENT_WCS_RESIDUAL_FAILED",
            f"aligned {filter_name} grid differs from reference by {output_error:.6g} pixels",
        )
    return {
        "filter": filter_name,
        "mode": "REPROJECTED_INDEPENDENT_SOLVE",
        "referenceFilter": reference_filter,
        "wcsProvenance": "REPROJECTED_INDEPENDENT_SOLVE",
        "maximumFivePointErrorPixelsBeforeAlignment": error,
        "maximumNinePointResidualPixelsBeforeAlignment": error,
        "maximumNinePointResidualPixelsAfterAlignment": output_error,
        "sampleCount": 9,
        "coverageFraction": fraction,
        "skyOverlapPreflight": sky_overlap,
        "source": source_identity,
        "reference": reference_identity,
        "output": _source_identity(destination),
        "propagatedWorkingMosaicUsedAsSolution": False,
    }


def _crop_final_channels(
    channels: Mapping[str, Path],
    *,
    minimum_retained_fraction: float,
    max_memory_bytes: int,
) -> dict[str, Any]:
    """Crop staged channels to the largest rectangle with finite common support.

    Reprojection encodes missing footprint as NaN. Scan that actual support,
    including luminance, without applying any brightness threshold. The same
    integer translation is applied to every channel's WCS (including the SIP
    reference pixel); no resampling or pixel replacement takes place here.
    """
    if not channels:
        raise ProjectE2EError("FINAL_AUTOCROP_EMPTY", "no final channels")
    if not 0 < minimum_retained_fraction <= 1:
        raise ProjectE2EError("FINAL_AUTOCROP_INVALID", "minimum retained fraction must be in (0, 1]")
    headers_shapes = {key: _read_image_header(path) for key, path in channels.items()}
    shape = next(iter(headers_shapes.values()))[1]
    if any(item[1] != shape for item in headers_shapes.values()):
        raise ProjectE2EError("FINAL_AUTOCROP_GEOMETRY_INVALID", "final channel shapes differ")
    height, width = shape
    # FITS-backed planes stay on disk; scratch is one tile plus a histogram.
    bytes_per_row = width * 16
    if max_memory_bytes < bytes_per_row:
        raise ProjectE2EError("MEMORY_BUDGET_TOO_SMALL", "one final crop-mask row exceeds memory budget")
    tile_rows = min(height, max(1, max_memory_bytes // bytes_per_row))
    heights = np.zeros(width, dtype=np.int64)
    best: tuple[int, int, int, int, int] | None = None
    common_count = 0
    invalid_counts = dict.fromkeys(channels, 0)
    with ExitStack() as stack:
        planes = {}
        for key, path in channels.items():
            hdul = stack.enter_context(fits.open(path, mode="readonly", memmap=True, checksum=True))
            planes[key] = next(item for item in hdul if item.data is not None and item.data.ndim == 2)
        for y0 in range(0, height, tile_rows):
            y1 = min(height, y0 + tile_rows)
            common = np.ones((y1 - y0, width), dtype=bool)
            for key, plane in planes.items():
                finite = np.isfinite(plane.data[y0:y1])
                invalid_counts[key] += int(finite.size - np.count_nonzero(finite))
                common &= finite
            common_count += int(np.count_nonzero(common))
            for local_y, row in enumerate(common):
                heights = np.where(row, heights + 1, 0)
                candidate = histogram_rectangle(heights, y0 + local_y)
                if candidate is not None and (best is None or candidate > best):
                    best = candidate
    if best is None or best[0] == 0:
        raise ProjectE2EError("FINAL_AUTOCROP_EMPTY", "final channels have no common finite rectangle")
    area, top, left, bottom, right = best
    retained_fraction = area / (height * width)
    if retained_fraction < minimum_retained_fraction:
        raise ProjectE2EError(
            "FINAL_AUTOCROP_TOO_SMALL",
            f"common final crop retains only {retained_fraction:.3%} of the reference frame; "
            f"minimum is {minimum_retained_fraction:.3%}",
        )
    applied = (top, left, bottom, right) != (0, 0, height, width)
    evidence: dict[str, Any] = {
        "mode": "COMMON_FINITE_MAXIMUM_RECTANGLE",
        "applied": applied,
        "inputImageShape": list(shape),
        "outputImageShape": [bottom - top, right - left],
        "bounds": {"top": top, "left": left, "bottom": bottom, "right": right},
        "boundsConvention": "ZERO_BASED_HALF_OPEN_REFERENCE_GRID",
        "retainedPixels": area,
        "retainedFraction": retained_fraction,
        "minimumRetainedFraction": minimum_retained_fraction,
        "commonFinitePixelsBeforeCrop": common_count,
        "commonFiniteFractionBeforeCrop": common_count / (height * width),
        "invalidPixelsBeforeCrop": invalid_counts,
        "invalidPixelsAfterCrop": dict.fromkeys(channels, 0),
        "wcsTransform": {"crpix1Offset": -left, "crpix2Offset": -top},
        "worldCoordinatesPreserved": True,
        "pixelValuesUnchanged": True,
    }
    if not applied:
        return evidence
    points = np.asarray([
        [0, 0], [right - left - 1, 0], [0, bottom - top - 1],
        [right - left - 1, bottom - top - 1],
        [(right - left - 1) / 2, (bottom - top - 1) / 2],
    ], dtype=np.float64)
    for key, path in channels.items():
        header = headers_shapes[key][0].copy()
        original_wcs = WCS(header, naxis=2, relax=False).celestial
        # FITS SIP coefficients are relative to CRPIX. Keeping coefficients
        # unchanged while translating CRPIX also translates the SIP origin.
        header["CRPIX1"] = float(header["CRPIX1"]) - left
        header["CRPIX2"] = float(header["CRPIX2"]) - top
        cropped_wcs = WCS(header, naxis=2, relax=False).celestial
        if not np.allclose(
            original_wcs.all_pix2world(points + [left, top], 0),
            cropped_wcs.all_pix2world(points, 0), rtol=0, atol=1e-10,
        ):
            raise ProjectE2EError("FINAL_AUTOCROP_WCS_INVALID", "cropping changed sky coordinates", path=str(path))
        validation = validate_wcs_header(header, image_shape=(bottom - top, right - left))
        if not validation.valid:
            raise ProjectE2EError("FINAL_AUTOCROP_WCS_INVALID", validation.message, path=str(path))
        header["OAFFCROP"] = ("COMMON", "Final channel common finite-support crop")
        header["OAFFCTOP"] = (top, "Zero-based top before final crop")
        header["OAFFCLEF"] = (left, "Zero-based left before final crop")
        header["OAFFCBOT"] = (bottom, "Exclusive bottom before final crop")
        header["OAFFCRGT"] = (right, "Exclusive right before final crop")
        header["OAFFCRET"] = (retained_fraction, "Retained area fraction of final reference grid")
        header.add_history("Ultra-Fast WBPP: common finite-support crop after final channel alignment; CRPIX translated")
        temporary = path.with_name(f".{path.name}.final-crop")
        try:
            with fits.open(path, mode="readonly", memmap=True, checksum=True) as hdul:
                plane = next(item for item in hdul if item.data is not None and item.data.ndim == 2)
                # Stream the FITS-backed slice; this does not allocate a full
                # additional science frame or alter any retained pixel value.
                fits.writeto(temporary, plane.data[top:bottom, left:right], header, overwrite=False, checksum=True)
            # Leaving the ``with`` block releases the memory map of ``path``
            # (no array of it is retained), which the replace below needs on
            # Windows.
            rename_with_retry(temporary, path, replace=True)
        finally:
            remove_file(temporary)
    return evidence


def _failure_result(
    *,
    staging: Path,
    evidence: Path,
    code: str,
    message: str,
    sources: Sequence[Any],
    layout: ProjectLayout,
    records: Mapping[str, Any],
    passed: Sequence[str],
    excluded: Sequence[str],
) -> ProjectE2EResult:
    _sanitize_shareable_tree(staging, sources)
    public_records = _share_safe_value(
        dict(records),
        staging=staging,
        source_tokens={
            item.path: f"source/{item.source_id}/{Path(item.path).name}"
            for item in sources
        },
    )
    public_layout = _share_safe_value(
        layout.serializable(),
        staging=staging,
        source_tokens={
            item.path: f"source/{item.source_id}/{Path(item.path).name}"
            for item in sources
        },
    )
    receipt_core = {
        "schemaVersion": 1,
        "pipelineVersion": PROJECT_E2E_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "success": False,
        "state": E2EState.UNSOLVED_WORKING.value,
        "code": code,
        "message": message,
        "layout": public_layout,
        "sources": [item.serializable() for item in sources],
        "execution": public_records,
        "publication": {
            "successDirectoryPublished": False,
            "evidenceDirectoryPublished": True,
            "atomic": True,
            "noReplace": True,
        },
    }
    receipt_id = "sha256:" + hashlib.sha256(canonical_json_document(receipt_core)).hexdigest()
    _write_json(staging / "receipt.json", {"receiptId": receipt_id, **receipt_core})
    _rename_directory_no_replace(staging, evidence)
    _fsync_directory(evidence.parent)
    return ProjectE2EResult(
        success=False,
        code=code,
        state=E2EState.UNSOLVED_WORKING,
        output_directory=None,
        evidence_directory=str(evidence),
        receipt_path=str(evidence / "receipt.json"),
        product_paths=(),
        preview_paths=(),
        passed_light_paths=tuple(passed),
        excluded_light_paths=tuple(excluded),
        message=message,
    )


@dataclass(frozen=True, slots=True)
class _ProjectRun:
    """What every stage of one project run shares: the request, the staging
    tree and its records, the admitted and excluded Lights, and progress."""

    request: ProjectE2ERequest
    output: Path
    evidence: Path
    layout: ProjectLayout
    sources: tuple[Any, ...]
    staging: Path
    details: Path
    records: dict[str, Any]
    passed: list[str]
    excluded: list[str]
    project_progress: "_ProjectProgress"


class _SharedMasters(NamedTuple):
    direct_approved_panel: bool
    masters_bias: tuple[str, ...]
    masters_dark: tuple[str, ...]
    masters_flat: tuple[str, ...]
    trusted_generated_master_overrides: tuple[MasterMetadataOverride, ...]


class _AlignedChannels(NamedTuple):
    reference_key: str
    reference_filter: str
    mono_paths: dict[str, Path]
    output_quality: dict[str, dict[str, Any]]
    astrometry_provenance: dict[str, dict[str, Any]]


class _ColorOutputs(NamedTuple):
    preview_paths: list[Path]
    color_result: ColorProductResult | None
    color_paths: dict[str, Path]


def _check_project_request(
    request: ProjectE2ERequest,
    solver_backends: Sequence[SolverBackend],
    mosaic_provider: ReprojectProvider | None,
) -> tuple[Path, Path, ProjectLayout, ReprojectProvider | None, tuple[Any, ...], Any, dict[str, Any]]:
    """Refuse a request that cannot run before anything is created."""

    output, evidence, layout = _validate_request(request)
    if not solver_backends:
        raise ProjectE2EError("SOLVER_CHAIN_EMPTY", "at least one final-solve backend is required")
    if mosaic_provider is None and (
        any(sum(panel.filter_key == key for panel in layout.panels) > 1 for key in layout.filter_keys)
        or {"r", "g", "b"}.issubset(set(layout.filter_keys))
    ):
        from ..products.mosaic import load_reproject_provider

        capability, mosaic_provider = load_reproject_provider()
        if mosaic_provider is None:
            raise ProjectE2EError(
                "MOSAIC_BACKEND_UNAVAILABLE", capability.reason or "reproject is unavailable"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    sources = _all_sources(request.e2e_request)
    if any(Path(item.path) == output or output in Path(item.path).parents for item in sources):
        raise ProjectE2EError("OUTPUT_ALIASES_SOURCE", "output cannot contain or alias a source")
    # Every REVIEW selection must name a Light of exactly one target run; a
    # stale digest from another data set is refused before any work starts.
    run_digests = {
        group.target_key: {sha256_digest(Path(path).resolve(strict=True)) for path in group.light_files}
        for group, _panels in layout.target_runs
    }
    unmatched = sorted(
        digest for digest in {selection["sourceSha256"] for selection in request.review_selections}
        if sum(digest in digests for digests in run_digests.values()) != 1
    )
    if unmatched:
        raise ProjectE2EError(
            "REVIEW_APPROVAL_SOURCE_AMBIGUOUS",
            "review selections must identify exactly one current Light: " + ", ".join(unmatched),
        )
    # Every explicit decision must name a Light of exactly one target run;
    # each run then receives the decisions of its own Lights.
    explicit_selection = request.e2e_request.explicit_selection
    if explicit_selection is not None:
        occurrences = {
            digest: sum(digest in digests for digests in run_digests.values())
            for digest in explicit_selection.by_source
        }
        unknown = sorted(digest for digest, count in occurrences.items() if count == 0)
        if unknown:
            raise ProjectE2EError(
                "SELECTION_SOURCE_UNKNOWN",
                "selection digests name no current Light: " + ", ".join(unknown),
            )
        ambiguous = sorted(digest for digest, count in occurrences.items() if count > 1)
        if ambiguous:
            raise ProjectE2EError(
                "SELECTION_SOURCE_AMBIGUOUS",
                "selection digests name more than one current Light: " + ", ".join(ambiguous),
            )
    return output, evidence, layout, mosaic_provider, sources, explicit_selection, run_digests


def _prepare_shared_calibration(run: _ProjectRun) -> _SharedMasters:
    """Build the shared calibration library once, or keep a directly approved request as is."""

    request, staging, details, records, project_progress = run.request, run.staging, run.details, run.records, run.project_progress
    project_progress.phase("prepare", 0, 1, "building shared calibration library")
    shared_root = details / "shared-calibration"
    shared_root.mkdir()
    direct_approved_panel = bool(request.e2e_request.review_approvals)
    if direct_approved_panel:
        # The approval digest is bound to the original calibration/source
        # request. Rewriting raw calibration into shared masters would
        # change that context. A one-panel project can safely preserve the
        # exact request and still use run_e2e's within-run master reuse.
        masters_bias = ()
        masters_dark = ()
        masters_flat = ()
        trusted_generated_master_overrides = ()
        shared_receipt = {
            "schemaVersion": 1,
            "stage": "shared-calibration-library",
            "mode": "SINGLE_PANEL_DIRECT_APPROVED_REQUEST",
            "rawIntegrationCount": 0,
        }
        _write_json(shared_root / "receipt.json", shared_receipt)
    else:
        (
            masters_bias,
            masters_dark,
            masters_flat,
            trusted_generated_master_overrides,
            shared_receipt,
        ) = _build_shared_calibration(request.e2e_request, shared_root)
    records["sharedCalibration"] = {
        "receipt": (shared_root / "receipt.json").relative_to(staging).as_posix(),
        "receiptSha256": sha256_digest(shared_root / "receipt.json"),
        "mode": shared_receipt["mode"],
        "rawIntegrationCount": shared_receipt["rawIntegrationCount"],
    }
    project_progress.phase("prepare", 1, 1, "shared calibration library verified")
    return _SharedMasters(direct_approved_panel, masters_bias, masters_dark, masters_flat, trusted_generated_master_overrides)


def _run_target_panels(run: _ProjectRun, shared: _SharedMasters, explicit_selection: Any, run_digests: dict[str, Any], solver_backends: Sequence[SolverBackend], panel_runner: PanelRunner) -> tuple[dict[str, list[tuple[SciencePanel, Path]]], dict[str, dict[str, Any]]] | ProjectE2EResult:
    """Run every target once with all of its filters; a failed target run fails the project."""

    direct_approved_panel, masters_bias, masters_dark, masters_flat, trusted_generated_master_overrides = shared
    request, evidence, layout, sources, staging, details, records, passed, excluded, project_progress = run.request, run.evidence, run.layout, run.sources, run.staging, run.details, run.records, run.passed, run.excluded, run.project_progress
    panels_by_filter: dict[str, list[tuple[SciencePanel, Path]]] = {}
    panel_quality_by_path: dict[str, dict[str, Any]] = {}
    runs_root = details / RUNS_DIRECTORY
    runs_root.mkdir()
    # One run per target carries every filter of that target: the run
    # registers all Lights onto one reference frame and crops the filter
    # masters to one common rectangle, so they share their pixel grid.
    for index, (group, panels) in enumerate(layout.target_runs, start=1):
        panel_progress = project_progress.panel_callback(index, group)
        panel_progress(ProgressEvent(ProgressStage.INVENTORY, "started",
                                     message=f"target {group.target} ({group.filter_name})"))
        sub_output = runs_root / _safe_token(group.target_key)
        panel_digests = run_digests[group.target_key]
        run_explicit_selection = (
            replace(
                explicit_selection,
                decisions=tuple(
                    decision
                    for decision in explicit_selection.decisions
                    if decision.source_sha256 in panel_digests
                ),
            )
            if explicit_selection is not None
            else None
        )
        if direct_approved_panel:
            sub_request = replace(
                request.e2e_request,
                light_files=group.light_files,
                output_directory=str(sub_output),
            )
        else:
            panel_pipeline_parameters = replace(
                request.e2e_request.pipeline_parameters,
                raw_frame_metadata_overrides=tuple(
                    item
                    for item in request.e2e_request.pipeline_parameters.raw_frame_metadata_overrides
                    if item.source_sha256 in panel_digests
                ),
                master_metadata_overrides=(
                    *request.e2e_request.pipeline_parameters.master_metadata_overrides,
                    *trusted_generated_master_overrides,
                ),
            )
            sub_request = replace(
                request.e2e_request,
                light_files=group.light_files,
                flat_files=(),
                dark_files=(),
                bias_files=(),
                master_bias_files=masters_bias,
                master_dark_files=masters_dark,
                master_flat_files=masters_flat,
                output_directory=str(sub_output),
                review_approvals=(),
                pipeline_parameters=panel_pipeline_parameters,
                explicit_selection=run_explicit_selection,
            )
            run_selections = [
                selection for selection in request.review_selections
                if selection["sourceSha256"] in panel_digests
            ]
            if run_selections:
                # Bound against this run's own request: its Lights, the
                # shared masters it will use, the gate policy and the
                # pipeline parameters, exactly what run_e2e re-verifies.
                sub_request = bind_review_approval_selections(sub_request, run_selections)
        sub_result = panel_runner(
            sub_request,
            solver_backends=solver_backends,
            progress=panel_progress,
        )
        passed.extend(sub_result.passed_light_paths)
        excluded.extend(sub_result.excluded_light_paths)
        record = {
            "target": group.target,
            "targetKey": group.target_key,
            "panels": [panel.serializable() for panel in panels],
            "success": sub_result.success,
            "code": sub_result.code,
            "receipt": Path(sub_result.receipt_path).relative_to(staging).as_posix(),
            "receiptSha256": sha256_digest(Path(sub_result.receipt_path)),
        }
        records["subruns"].append(record)
        try:
            record["screening"] = _screening_record(Path(sub_result.receipt_path), staging, group.target)
        except (OSError, ValueError, json.JSONDecodeError):
            record["screening"] = None
        records["screening"] = _project_screening(records["subruns"])
        if not sub_result.success:
            sub_message = sub_result.message or f"{group.target}/{group.filter_name}: {sub_result.code}"
            _verify_sources(sources)
            return _failure_result(
                staging=staging,
                evidence=evidence,
                code=sub_result.code,
                message=sub_message,
                sources=sources,
                layout=layout,
                records=records,
                passed=passed,
                excluded=excluded,
            )
        products_by_filter = _find_filter_products(
            sub_result, [panel.filter_name for panel in panels]
        )
        record["solvedProducts"] = {}
        for panel in panels:
            product = products_by_filter[panel.filter_name]
            panel_quality_by_path[str(product)] = _accepted_quality(
                Path(sub_result.receipt_path), panel.filter_name
            )
            panels_by_filter.setdefault(panel.filter_key, []).append((panel, product))
            record["solvedProducts"][panel.filter_name] = {
                "path": product.relative_to(staging).as_posix(),
                "sha256": sha256_digest(product),
            }
        project_progress.panel_done(index, group)
    return panels_by_filter, panel_quality_by_path


def _build_filter_mosaics(run: _ProjectRun, panels_by_filter: dict[str, list[tuple[SciencePanel, Path]]], panel_quality_by_path: dict[str, dict[str, Any]], solver_backends: Sequence[SolverBackend], mosaic_builder: MosaicBuilder, mosaic_provider: ReprojectProvider | None) -> tuple[dict[str, tuple[str, Path]], dict[str, dict[str, Any]]] | ProjectE2EResult:
    """Mosaic each filter's panels and solve the mosaic afresh; single panels pass through."""

    request, evidence, layout, sources, staging, details, records, passed, excluded, project_progress = run.request, run.evidence, run.layout, run.sources, run.staging, run.details, run.records, run.passed, run.excluded, run.project_progress
    final_sources: dict[str, tuple[str, Path]] = {}
    final_quality: dict[str, dict[str, Any]] = {}
    mosaic_root = details / "mosaics"
    mosaic_root.mkdir()
    mosaic_total = 2 * sum(len(panels) > 1 for panels in panels_by_filter.values())
    mosaic_completed = 0
    project_progress.phase("mosaic", 0, mosaic_total, "building and solving filter mosaics" if mosaic_total else "single panels require no mosaic")
    for filter_key, panels in sorted(panels_by_filter.items()):
        display_filter = panels[0][0].filter_name
        if len(panels) == 1:
            final_sources[filter_key] = (display_filter, panels[0][1])
            final_quality[filter_key] = panel_quality_by_path[str(panels[0][1])]
            records["mosaics"][filter_key] = {
                "mode": "SINGLE_SOLVED_PANEL",
                "panelCount": 1,
                "freshFinalSolveRequired": False,
            }
            continue
        assert mosaic_provider is not None
        working_directory = mosaic_root / f"{_safe_token(filter_key)}-working"
        mosaic = mosaic_builder(
            MosaicRequest(
                panel_paths=tuple(str(path) for _, path in panels),
                output_directory=str(working_directory),
                minimum_covered_fraction=request.minimum_mosaic_covered_fraction,
                minimum_pair_overlap_pixels=request.minimum_pair_overlap_pixels,
                minimum_pair_overlap_fraction=request.minimum_pair_overlap_fraction,
                maximum_seam_normalized_mad=request.maximum_seam_normalized_mad,
            ),
            provider=mosaic_provider,
        )
        working = Path(mosaic.mosaic_path).resolve(strict=True)
        mosaic_completed += 1
        project_progress.phase("mosaic", mosaic_completed, mosaic_total, f"{display_filter}: mosaic built; fresh solve pending")
        working_header, _ = _read_image_header(working)
        if working_header.get("OAFSTATE") != "NEEDS_FINAL_SOLVE" or working_header.get("OAFWCS") != "PROPAGATED":
            raise ProjectE2EError(
                "MOSAIC_STATE_INVALID",
                "working mosaic must remain NEEDS_FINAL_SOLVE/PROPAGATED",
                path=str(working),
            )
        solved_directory = mosaic_root / f"{_safe_token(filter_key)}-solved"
        solved_directory.mkdir()
        solved_path = solved_directory / f"master_light_{_safe_token(display_filter)}_mosaic_wcs.fits"
        hints = _final_mosaic_hints(working, request.e2e_request)
        solved, attempts = _solve_one(
            input_path=working,
            output_path=solved_path,
            backends=solver_backends,
            hints=hints,
            min_matches=request.e2e_request.min_matches,
            max_rms_arcsec=request.e2e_request.max_rms_arcsec,
        )
        records["mosaics"][filter_key] = {
            "mode": "SOLVED_PANEL_MOSAIC",
            "panelCount": len(panels),
            "workingReceipt": Path(mosaic.receipt_path).relative_to(staging).as_posix(),
            "workingReceiptSha256": sha256_digest(Path(mosaic.receipt_path)),
            "workingState": mosaic.state,
            "propagatedWcsIsFinalSolution": False,
        }
        records["finalSolves"][filter_key] = {
            "status": "SOLVED" if solved else "UNSOLVED",
            "input": working.relative_to(staging).as_posix(),
            "output": solved_path.relative_to(staging).as_posix() if solved else None,
            "hints": hints.serializable(),
            "attempts": _relativize_solver_attempts(attempts, staging),
        }
        if not solved:
            _verify_sources(sources)
            return _failure_result(
                staging=staging,
                evidence=evidence,
                code="MOSAIC_FINAL_SOLVE_REQUIRED",
                message=f"filter {display_filter} mosaic did not pass a fresh final solve",
                sources=sources,
                layout=layout,
                records=records,
                passed=passed,
                excluded=excluded,
            )
        final_sources[filter_key] = (display_filter, solved_path)
        accepted_attempt = next(
            item for item in reversed(attempts) if item.get("accepted") is True
        )
        quality = accepted_attempt.get("result", {}).get("astrometricQuality")
        if not isinstance(quality, Mapping) or quality.get("catalogManaged") is not True:
            raise ProjectE2EError(
                "MOSAIC_ASTROMETRY_EVIDENCE_MISSING",
                "fresh mosaic solve lacks managed catalog evidence",
            )
        final_quality[filter_key] = dict(quality)
        mosaic_completed += 1
        project_progress.phase("mosaic", mosaic_completed, mosaic_total, f"{display_filter}: fresh mosaic solve verified")
    return final_sources, final_quality


def _align_channels(run: _ProjectRun, final_sources: dict[str, tuple[str, Path]], final_quality: dict[str, dict[str, Any]], mosaic_provider: ReprojectProvider | None) -> _AlignedChannels:
    """Put every solved channel on the reference grid and crop them to one rectangle."""

    request, staging, records, project_progress = run.request, run.staging, run.records, run.project_progress
    # Luminance carries the detail of an LRGB product, so when channels do
    # not already share their grid it is the one that stays unresampled.
    reference_key = next(
        (key for key in ("l", "r", "g", "b") if key in final_sources),
        sorted(final_sources)[0],
    )
    reference_filter, reference_path = final_sources[reference_key]
    mono_paths: dict[str, Path] = {}
    output_quality: dict[str, dict[str, Any]] = {}
    astrometry_provenance: dict[str, dict[str, Any]] = {}
    project_progress.phase("alignment", 0, len(final_sources), "aligning solved channels onto the reference grid")
    for key, (filter_name, source) in sorted(final_sources.items()):
        # ``<FILTER>.fits`` so PixInsight labels the opened image with the
        # channel name (its view identifier comes from the file name).
        destination = staging / f"{_safe_token(filter_name)}.fits"
        if key == reference_key:
            publication = _publish_file(source, destination)
            alignment = {
                "filter": filter_name,
                "mode": "REFERENCE_SOLVED_GRID",
                "referenceFilter": reference_filter,
                "wcsProvenance": "INDEPENDENT_SOLVE",
                "publication": publication,
                "coverageFraction": 1.0,
                "source": _source_identity(source),
                "output": _source_identity(destination),
            }
        else:
            assert mosaic_provider is not None
            alignment = _align_channel(
                source,
                reference_path,
                destination,
                filter_name=filter_name,
                reference_filter=reference_filter,
                provider=mosaic_provider,
                tolerance_pixels=request.channel_wcs_tolerance_pixels,
                minimum_coverage=request.minimum_channel_alignment_coverage,
            )
        source_solution = _astrometry_gui_evidence(source, final_quality[key])
        reference_solution = _astrometry_gui_evidence(
            reference_path, final_quality[reference_key]
        )
        if alignment["mode"] == "REPROJECTED_INDEPENDENT_SOLVE":
            output_quality[key] = final_quality[reference_key]
            provenance = {
                "type": "PROPAGATED_VERIFIED",
                "freshSolveOnThisPixelGrid": False,
                "referenceFilter": reference_filter,
                "referenceSolution": reference_solution,
                "sourceSolution": source_solution,
                "reprojection": {
                    "backendId": mosaic_provider.backend_id if mosaic_provider else None,
                    "backendVersion": mosaic_provider.version if mosaic_provider else None,
                    "coverageFraction": alignment["coverageFraction"],
                    "sampleCount": alignment["sampleCount"],
                    "maximumNinePointResidualPixelsBeforeAlignment": alignment[
                        "maximumNinePointResidualPixelsBeforeAlignment"
                    ],
                    "maximumNinePointResidualPixelsAfterAlignment": alignment[
                        "maximumNinePointResidualPixelsAfterAlignment"
                    ],
                },
                "qualityAppliesTo": "REFERENCE_SOLVED_WCS_PROPAGATED_TO_VERIFIED_GRID",
            }
        else:
            output_quality[key] = final_quality[key]
            provenance = {
                "type": "FRESH_SOLVE_UNCHANGED_GRID",
                "freshSolveOnThisPixelGrid": True,
                "sourceSolution": source_solution,
                "qualityAppliesTo": "SOURCE_SOLVED_WCS_UNCHANGED",
            }
        alignment["astrometryProvenance"] = provenance
        astrometry_provenance[key] = provenance
        mono_paths[key] = destination
        records["alignment"][key] = alignment
        project_progress.phase("alignment", len(mono_paths), len(final_sources), f"{filter_name}: channel alignment verified")

    if request.e2e_request.pipeline_parameters.auto_crop:
        final_crop = _crop_final_channels(
            mono_paths,
            minimum_retained_fraction=request.e2e_request.pipeline_parameters.minimum_crop_fraction,
            max_memory_bytes=request.e2e_request.pipeline_parameters.registration_memory_bytes,
        )
        records["finalCrop"] = final_crop
        for key, path in mono_paths.items():
            # Final identities must describe the cropped files consumed by
            # previews, RGB and GUI verification, not the intermediate grid.
            records["alignment"][key]["output"] = _source_identity(path)
            if final_crop["applied"]:
                # The crop rewrote the file, so it no longer shares the
                # run master's storage.
                records["alignment"][key]["publication"] = "CROPPED_COPY"
                provenance = astrometry_provenance[key]
                provenance["type"] = "PROPAGATED_VERIFIED"
                provenance["freshSolveOnThisPixelGrid"] = False
                provenance["finalCrop"] = final_crop
                provenance["qualityAppliesTo"] = "SOLVED_WCS_TRANSLATED_TO_COMMON_CROPPED_GRID"
    else:
        records["finalCrop"] = {"mode": "DISABLED", "applied": False}
    return _AlignedChannels(reference_key, reference_filter, mono_paths, output_quality, astrometry_provenance)


def _render_previews_and_color(run: _ProjectRun, final_sources: dict[str, tuple[str, Path]], aligned: _AlignedChannels, color_builder: ColorBuilder) -> _ColorOutputs:
    """Render the mono previews and, with R, G and B present, the colour products."""

    reference_key, reference_filter, mono_paths, output_quality, astrometry_provenance = aligned
    request, staging, details, records, project_progress = run.request, run.staging, run.details, run.records, run.project_progress
    preview_root = staging / "previews"
    preview_root.mkdir()
    preview_paths: list[Path] = []
    color_total = len(mono_paths) + int({"r", "g", "b"}.issubset(mono_paths))
    project_progress.phase("color", 0, color_total, "rendering mono previews and color products")
    for key, path in sorted(mono_paths.items()):
        preview = preview_root / f"{path.stem}.png"
        render_auto_stretch_preview(
            path,
            preview,
            max_long_edge=request.e2e_request.pipeline_parameters.preview_max_long_edge,
            max_memory_bytes=request.e2e_request.pipeline_parameters.registration_memory_bytes,
        )
        preview_paths.append(preview)
        project_progress.phase("color", len(preview_paths), color_total, f"{final_sources[key][0]}: preview rendered")

    color_result: ColorProductResult | None = None
    color_paths: dict[str, Path] = {}
    if {"r", "g", "b"}.issubset(mono_paths):
        color_result = color_builder(
            ColorProductRequest(
                red_path=str(mono_paths["r"]),
                green_path=str(mono_paths["g"]),
                blue_path=str(mono_paths["b"]),
                luminance_path=str(mono_paths["l"]) if "l" in mono_paths else None,
                output_directory=str(details / "color"),
                wcs_tolerance_pixels=request.channel_wcs_tolerance_pixels,
                preview_max_long_edge=request.e2e_request.pipeline_parameters.preview_max_long_edge,
            )
        )
        # The color stage publishes into ``details/color``; its FITS and
        # previews surface at the top level like the mono channels.
        color_publication = {}
        for name, source in (
            ("linearRgb", Path(color_result.linear_rgb_path)),
            ("previewTiff", Path(color_result.preview_tiff_path)),
            ("previewPng", Path(color_result.preview_png_path)),
        ):
            final = (staging if name == "linearRgb" else preview_root) / source.name
            color_publication[name] = _publish_file(source, final)
            color_paths[name] = final
        records["color"] = {
            "status": "SOLVED_LRGB" if "l" in mono_paths else "SOLVED_RGB",
            "receipt": Path(color_result.receipt_path).relative_to(staging).as_posix(),
            "receiptSha256": sha256_digest(Path(color_result.receipt_path)),
            "publication": color_publication,
            "luminanceParticipated": "l" in mono_paths,
            "astrometryProvenance": {
                "type": "PROPAGATED_VERIFIED",
                "freshSolveOnRgbCube": False,
                "referenceFilter": reference_filter,
                "referenceSolution": _astrometry_gui_evidence(
                    mono_paths[reference_key], output_quality[reference_key]
                ),
                "channelGeometryVerified": True,
                "finalCrop": records["finalCrop"],
                "channelProvenanceTypes": {
                    key: astrometry_provenance[key]["type"]
                    for key in sorted(mono_paths)
                },
                "qualityAppliesTo": "REFERENCE_SOLVED_WCS_PROPAGATED_TO_RGB_GRID",
            },
        }
    else:
        records["color"] = {
            "status": "NOT_CREATED_MISSING_RGB_CHANNELS",
            "availableFilters": sorted(final_sources),
            "missingRequiredChannels": sorted({"r", "g", "b"} - set(mono_paths)),
            "monoProductsPublished": True,
        }

    project_progress.phase("color", color_total, color_total, "color and preview products created")
    return _ColorOutputs(preview_paths, color_result, color_paths)


def _verify_and_publish(run: _ProjectRun, panels_by_filter: dict[str, list[tuple[SciencePanel, Path]]], final_sources: dict[str, tuple[str, Path]], aligned: _AlignedChannels, color: _ColorOutputs) -> ProjectE2EResult:
    """Verify sources and products, write the receipt and publish the project atomically."""

    preview_paths, color_result, color_paths = color
    reference_key, reference_filter, mono_paths, output_quality, astrometry_provenance = aligned
    request, output, layout, sources, staging, records, passed, excluded, project_progress = run.request, run.output, run.layout, run.sources, run.staging, run.records, run.passed, run.excluded, run.project_progress
    project_progress.phase("verify", 0, 1, "verifying source identities, final products, and receipts")
    _verify_sources(sources)
    _sanitize_shareable_tree(staging, sources)
    public_records = _share_safe_value(
        records,
        staging=staging,
        source_tokens={
            item.path: f"source/{item.source_id}/{Path(item.path).name}"
            for item in sources
        },
    )
    public_layout = _share_safe_value(
        layout.serializable(),
        staging=staging,
        source_tokens={
            item.path: f"source/{item.source_id}/{Path(item.path).name}"
            for item in sources
        },
    )
    final_artifacts = []
    final_digests = _sha256_many(
        [
            *(path for _key, path in sorted(mono_paths.items())),
            *(
                (color_paths["linearRgb"], color_paths["previewTiff"], color_paths["previewPng"])
                if color_result is not None
                else ()
            ),
            *preview_paths,
        ]
    )
    for key, path in sorted(mono_paths.items()):
        artifact = _artifact(path, staging, "SOLVED_MONO_FITS", digests=final_digests)
        artifact["filter"] = final_sources[key][0]
        artifact["astrometry"] = _astrometry_gui_evidence(
            path, output_quality[key]
        )
        artifact["astrometryProvenance"] = astrometry_provenance[key]
        artifact["finalGate"] = {
            "status": "PASS",
            "managedCatalogRequired": True,
            "freshMosaicSolve": len(panels_by_filter[key]) > 1,
            "channelAlignmentPassed": True,
            "freshSolveOnThisPixelGrid": astrometry_provenance[key][
                "freshSolveOnThisPixelGrid"
            ],
            "propagatedReferenceWcsVerified": astrometry_provenance[key]["type"]
            == "PROPAGATED_VERIFIED",
        }
        final_artifacts.append(artifact)
    product_paths = list(mono_paths.values())
    if color_result is not None:
        for path, kind in (
            (color_paths["linearRgb"], "LINEAR_RGB_FITS"),
            (color_paths["previewTiff"], "RGB_PREVIEW_TIFF_16"),
            (color_paths["previewPng"], "RGB_PREVIEW_PNG_16"),
        ):
            final_artifacts.append(_artifact(path, staging, kind, digests=final_digests))
            product_paths.append(path)
        rgb_artifact = next(
            item for item in final_artifacts if item["kind"] == "LINEAR_RGB_FITS"
        )
        rgb_artifact["astrometry"] = _astrometry_gui_evidence(
            color_paths["linearRgb"], output_quality[reference_key]
        )
        rgb_artifact["astrometryProvenance"] = records["color"][
            "astrometryProvenance"
        ]
        rgb_artifact["wcsProvenance"] = "PROPAGATED_VERIFIED"
        rgb_artifact["finalGate"] = {
            "status": "PASS",
            "rgbChannelsPresent": True,
            "luminanceParticipated": "l" in mono_paths,
            "strictGeometryAndWcsAlignment": True,
            "freshSolveOnRgbCube": False,
            "propagatedReferenceWcsVerified": True,
        }
    for path in preview_paths:
        final_artifacts.append(_artifact(path, staging, "MONO_PREVIEW_PNG", digests=final_digests))

    code = "PROJECT_COLOR_SUCCEEDED" if color_result is not None else "PROJECT_MONO_SUCCEEDED"
    receipt_core = {
        "schemaVersion": 1,
        "pipelineVersion": PROJECT_E2E_VERSION,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "success": True,
        "state": E2EState.SOLVED.value,
        "code": code,
        "recipeDigest": request.e2e_request.recipe_digest,
        "calibrationPolicy": workflow_receipt(request.e2e_request.pipeline_parameters.calibration_workflow),
        "layout": public_layout,
        "sources": [item.serializable() for item in sources],
        "execution": public_records,
        "finalProducts": {
            "monoFilters": [final_sources[key][0] for key in sorted(final_sources)],
            "rgbCreated": color_result is not None,
            "luminanceParticipated": color_result is not None and "l" in mono_paths,
            "artifacts": final_artifacts,
            "guiArtifacts": final_artifacts,
            "resultGate": {
                "status": "PASS",
                "allMonoProductsSolved": True,
                "managedCatalogEvidenceRequired": True,
                "sourceIdentityVerifiedAtCommit": True,
                "mosaicCoverageOverlapSeamPassed": True,
                "propagatedAlignedChannelsVerified": all(
                    value["type"]
                    in {"FRESH_SOLVE_UNCHANGED_GRID", "PROPAGATED_VERIFIED"}
                    for value in astrometry_provenance.values()
                ),
                "rgbState": (
                    "SOLVED_LRGB"
                    if color_result is not None and "l" in mono_paths
                    else "SOLVED_RGB"
                    if color_result is not None
                    else "MONO_ONLY_CHANNELS_MISSING"
                ),
            },
        },
        "publication": {
            "atomic": True,
            "noReplace": True,
            "outerTransaction": True,
            "sourceMutation": False,
            "failurePublishesUnsolvedEvidenceOnly": True,
        },
    }
    receipt_id = "sha256:" + hashlib.sha256(canonical_json_document(receipt_core)).hexdigest()
    _write_json(staging / "receipt.json", {"receiptId": receipt_id, **receipt_core})
    project_progress.phase("verify", 1, 1, "final products and receipt verified")
    project_progress.phase("publish", 0, 1, "atomically publishing the complete project")
    _fsync_directory(staging)
    _rename_directory_no_replace(staging, output)
    _fsync_directory(output.parent)
    # Progress cannot turn an already committed success into a failure.
    # The desktop's independent artifact verification releases the final 1%.
    try:
        project_progress.phase("publish", 1, 1, "project published; desktop verification pending")
    except Exception:
        pass
    return ProjectE2EResult(
        success=True,
        code=code,
        state=E2EState.SOLVED,
        output_directory=str(output),
        evidence_directory=None,
        receipt_path=str(output / "receipt.json"),
        product_paths=tuple(str(output / path.relative_to(staging)) for path in product_paths),
        preview_paths=tuple(str(output / path.relative_to(staging)) for path in preview_paths),
        passed_light_paths=tuple(passed),
        excluded_light_paths=tuple(excluded),
        mono_filters=tuple(final_sources[key][0] for key in sorted(final_sources)),
        color_product_path=(
            str(output / color_paths["linearRgb"].relative_to(staging))
            if color_result is not None
            else None
        ),
    )


def run_project_e2e(
    request: ProjectE2ERequest,
    *,
    solver_backends: Sequence[SolverBackend],
    progress: ProgressCallback | None = None,
    mosaic_provider: ReprojectProvider | None = None,
    panel_runner: PanelRunner = run_e2e,
    mosaic_builder: MosaicBuilder = build_solved_panel_mosaic,
    color_builder: ColorBuilder = build_color_product,
) -> ProjectE2EResult:
    """Run a complete multi-target project and commit one final directory."""

    base = request.e2e_request
    if base.pipeline_parameters.grouping_keyword_root is None:
        # The shared calibration and every target run read WBPP grouping
        # keywords below the same folder: the ancestor of all project inputs.
        request = replace(
            request,
            e2e_request=replace(
                base,
                pipeline_parameters=replace(
                    base.pipeline_parameters,
                    grouping_keyword_root=grouping_keyword_root(
                        (
                            *base.light_files,
                            *base.flat_files,
                            *base.dark_files,
                            *base.bias_files,
                            *base.master_bias_files,
                            *base.master_dark_files,
                            *base.master_flat_files,
                        )
                    ),
                ),
            ),
        )
    output, evidence, layout, mosaic_provider, sources, explicit_selection, run_digests = _check_project_request(
        request, solver_backends, mosaic_provider
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=PROJECT_STAGING_SUFFIX, dir=output.parent))
    # The published directory holds only the final channels, ``previews`` and
    # ``receipt.json`` at its top level; every intermediate stage lives under
    # ``details`` so the result reads at a glance.
    details = staging / DETAILS_DIRECTORY
    details.mkdir()
    records: dict[str, Any] = {
        "subruns": [],
        "mosaics": {},
        "finalSolves": {},
        "alignment": {},
        # The panel runs record the same verdict; here it is visible at the
        # project level without opening a run receipt.
        "sourceExtraction": dict(cached_extraction_self_test()),
    }
    passed: list[str] = []
    excluded: list[str] = []
    project_progress = _ProjectProgress(layout, request.e2e_request, progress)
    run = _ProjectRun(request, output, evidence, layout, sources, staging, details, records, passed, excluded, project_progress)
    try:
        shared = _prepare_shared_calibration(run)
        targets = _run_target_panels(run, shared, explicit_selection, run_digests, solver_backends, panel_runner)
        if isinstance(targets, ProjectE2EResult):
            return targets
        panels_by_filter, panel_quality_by_path = targets
        mosaics = _build_filter_mosaics(
            run, panels_by_filter, panel_quality_by_path, solver_backends, mosaic_builder, mosaic_provider
        )
        if isinstance(mosaics, ProjectE2EResult):
            return mosaics
        final_sources, final_quality = mosaics
        aligned = _align_channels(run, final_sources, final_quality, mosaic_provider)
        color = _render_previews_and_color(run, final_sources, aligned, color_builder)
        return _verify_and_publish(run, panels_by_filter, final_sources, aligned, color)
    except (ProjectE2EError, E2EError, MosaicError, ColorProductError, CalibrationError, OSError) as error:
        if not staging.exists():
            raise
        code = getattr(error, "code", "PROJECT_EXECUTION_FAILED")
        try:
            _verify_sources(sources)
        except Exception as drift:
            code = getattr(drift, "code", "SOURCE_CHANGED")
            error = drift
        return _failure_result(
            staging=staging,
            evidence=evidence,
            code=str(code),
            message=str(error),
            sources=sources,
            layout=layout,
            records=records,
            passed=passed,
            excluded=excluded,
        )
    except Exception as error:
        if not staging.exists():
            raise
        # Preserve completed science products and a bounded, path-safe stack
        # for unexpected finalization errors. Nothing is published as success.
        diagnostic = {
            "schemaVersion": 1,
            "exceptionType": type(error).__name__,
            "message": str(error)[-16_384:],
            "stack": [
                {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
                for frame in traceback.extract_tb(error.__traceback__)[-32:]
            ],
        }
        diagnostic_path = staging / "diagnostics" / "unexpected-failure.json"
        diagnostic_path.parent.mkdir(exist_ok=True)
        _write_json(diagnostic_path, diagnostic)
        records["unexpectedFailure"] = {
            "diagnostic": diagnostic_path.relative_to(staging).as_posix(),
            "exceptionType": diagnostic["exceptionType"],
        }
        code = "PROJECT_UNEXPECTED_FAILURE"
        message = f"{type(error).__name__}: {error}"
        try:
            _verify_sources(sources)
        except Exception as drift:
            code = getattr(drift, "code", "SOURCE_CHANGED")
            message = str(drift)
        try:
            return _failure_result(
                staging=staging, evidence=evidence, code=str(code), message=message,
                sources=sources, layout=layout, records=records, passed=passed, excluded=excluded,
            )
        except Exception as evidence_error:
            # Sanitization or receipt serialization can itself be the defect.
            # Do not recursively invoke that writer or delete the private tree.
            message += f"; evidence publication failed: {type(evidence_error).__name__}: {evidence_error}"
            return ProjectE2EResult(
                success=False, code=str(code), state=E2EState.UNSOLVED_WORKING,
                output_directory=None, evidence_directory=str(staging),
                receipt_path=str(diagnostic_path), product_paths=(), preview_paths=(),
                passed_light_paths=tuple(passed), excluded_light_paths=tuple(excluded),
                message=message,
            )


__all__ = [
    "PROJECT_E2E_VERSION",
    "ProjectE2EError",
    "ProjectE2ERequest",
    "ProjectE2EResult",
    "ProjectLayout",
    "SciencePanel",
    "classify_project_layout",
    "project_requires_orchestration",
    "run_project_e2e",
]
