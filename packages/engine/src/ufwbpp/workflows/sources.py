"""The inputs of a run: content-bound source identities, XISF staging into bounded FITS, and verification that nothing changed underneath the run."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import math
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping, Sequence

from lightframeqc.cfa import is_cfa_pattern
from lightframeqc.readers import probe_frame_metadata

from ..calibration.inputs import InternalSourceIdentity, with_path_metadata
from ..calibration.policy import apply_mono_workflow, bias_from_header, MONO_STANDARD
from ..image_io.xisf import convert_xisf_to_fits, preflight_xisf_header
from ..integrity import sha256_digest
from ..platform import remove_file
from ..stacking.integration import CalibrationError, FrameInfo, normalize_role, read_frame_info
from ..stacking.parameters import PipelineParameters
from .common import _emit
from .contracts import ProgressStage, E2EError, E2ERequest, ProgressCallback


@dataclass(frozen=True, slots=True)
class _SourceIdentity:
    path: str
    role: str
    sha256: str
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int

    @property
    def source_id(self) -> str:
        payload = f"{self.role}\0{self.sha256}\0{Path(self.path).name}".encode("utf-8")
        return "src-" + hashlib.sha256(payload).hexdigest()[:20]

    def serializable(self) -> dict[str, Any]:
        """Return the share-safe identity written to published receipts."""

        return {
            "sourceId": self.source_id,
            "role": self.role,
            "displayName": Path(self.path).name,
            "sha256": self.sha256,
            "sizeBytes": self.size_bytes,
        }

    def local_serializable(self) -> dict[str, Any]:
        """Return execution-local binding data; never write it to a product."""

        return {
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "sizeBytes": self.size_bytes,
            "mtimeNs": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
        }


def _input_frame_info(
    path: Path,
    parameters: PipelineParameters,
    *,
    override_identity_path: Path | None = None,
    override_source_identity: _SourceIdentity | InternalSourceIdentity | None = None,
) -> FrameInfo:
    if path.suffix.casefold() != ".xisf":
        info = read_frame_info(path)
    else:
        preflight_xisf_header(path, parameters.xisf_decode)
        try:
            metadata = probe_frame_metadata(path)
        except Exception as error:
            code = getattr(error, "code", "XISF_HEADER_ERROR")
            raise CalibrationError(code, str(error), path=str(path)) from error
        header = metadata.header

        def number(*keys: str) -> float | None:
            for key in keys:
                value = header.get(key)
                try:
                    result = float(value) if value is not None else math.nan
                except (TypeError, ValueError):
                    continue
                if math.isfinite(result):
                    return result
            return None

        info = FrameInfo(
            path=str(path),
            role=normalize_role(metadata.role.value),
            shape=(metadata.height, metadata.width),
            filter_name=metadata.filter_name,
            exposure_seconds=metadata.exposure_seconds,
            temperature_celsius=number(
                "CCD-TEMP", "CCD_TEMP", "SENSORT", "SENSOR-T", "CAMTEMP"
            ),
            camera=metadata.camera,
            gain=metadata.gain,
            offset=metadata.offset,
            binning_x=metadata.binning_x,
            binning_y=metadata.binning_y,
            cfa_pattern=metadata.cfa_pattern,
            readout_mode=metadata.readout_mode,
            target=metadata.target,
            bias_included=bias_from_header(header),
        )
    identity_path = override_identity_path or path
    info = with_path_metadata(info, identity_path, keyword_root=parameters.grouping_keyword_root)
    if (
        identity_path.suffix.casefold() == ".xisf"
        and path.suffix.casefold() != ".xisf"
    ):
        if (
            info.numeric_domain_authority != "SELF_DECLARED_HEADER"
            or info.numeric_domain == "UNDECLARED"
            or info.normalized_unit_scale is None
        ):
            raise CalibrationError(
                "XISF_NUMERIC_DOMAIN_UNDECLARED",
                "private XISF conversion lacks explicit finite bounds for its numeric domain",
                path=str(identity_path),
            )
        info = replace(
            info, numeric_domain_authority="TRUSTED_XISF_CONVERSION"
        )
    matches = []
    if parameters.raw_frame_metadata_overrides:
        if override_source_identity is None:
            digest = sha256_digest(identity_path)
        else:
            canonical_identity_path = identity_path.expanduser().resolve(strict=True)
            if override_source_identity.path != str(canonical_identity_path):
                raise CalibrationError(
                    "SOURCE_IDENTITY_CACHE_SET_MISMATCH",
                    "metadata override digest is not bound to this exact source path",
                    path=str(canonical_identity_path),
                )
            digest = override_source_identity.sha256
        matches = [
            item
            for item in parameters.raw_frame_metadata_overrides
            if item.source_sha256 == digest
        ]
    if len(matches) > 1:
        raise CalibrationError(
            "RAW_FRAME_METADATA_OVERRIDE_SOURCE_AMBIGUOUS",
            "multiple raw-frame overrides bind the same source",
            path=str(identity_path),
        )
    if matches:
        current = info.cfa_pattern.strip().upper()
        confirmed = matches[0].cfa_pattern.strip().upper()
        if current not in {"", "UNKNOWN", "UNSPECIFIED"} and current != confirmed:
            raise CalibrationError(
                "RAW_CFA_OVERRIDE_CONFLICT",
                f"explicit source CFA {current} cannot be replaced by {confirmed}",
                path=str(identity_path),
            )
        info = replace(info, cfa_pattern=confirmed)
    return apply_mono_workflow(info, parameters.calibration_workflow)


def _stage_e2e_xisf_inputs(
    grouped: Sequence[tuple[str, tuple[Path, ...]]],
    directory: Path,
    parameters: PipelineParameters,
    source_identities: Mapping[str, _SourceIdentity] | None = None,
) -> tuple[
    dict[str, tuple[Path, ...]],
    dict[str, Path],
    list[dict[str, Any]],
    dict[str, str],
]:
    directory.mkdir(parents=True, exist_ok=False)
    aliases: dict[str, Path] = {}
    conversions: list[dict[str, Any]] = []
    staged_digests: dict[str, str] = {}
    staged_groups: dict[str, tuple[Path, ...]] = {}
    sequence = 0
    for role, paths in grouped:
        staged_paths: list[Path] = []
        for path in paths:
            sequence += 1
            expected = (
                source_identities.get(str(path.resolve(strict=True)))
                if source_identities is not None
                else None
            )
            if source_identities is not None and expected is None:
                raise E2EError(
                    "SOURCE_IDENTITY_CACHE_SET_MISMATCH",
                    "pixel staging is missing the captured source identity",
                    path=str(path),
                )
            if path.suffix.casefold() == ".xisf":
                staged = directory / f"{sequence:06d}_{role.casefold()}.fits"
                receipt = convert_xisf_to_fits(path, staged, policy=parameters.xisf_decode)
                if expected is not None and (
                    receipt.source_sha256 != expected.sha256
                    or receipt.source_size_bytes != expected.size_bytes
                ):
                    remove_file(staged)
                    raise E2EError(
                        "SOURCE_CHANGED",
                        "XISF conversion does not match the captured source identity",
                        path=str(path),
                    )
                conversions.append({"role": role, **receipt.serializable()})
                staged_digests[str(staged)] = receipt.converted_sha256
            else:
                # FITS sources are consumed in place through read-only handles.
                # Their captured stat identity is rechecked at every trust
                # boundary and their content is rehashed once before
                # publication, so a private byte copy adds no detection that
                # the final gate does not already provide.
                staged = path
            staged_paths.append(staged)
            aliases[str(staged)] = path
        staged_groups[role] = tuple(staged_paths)
    return staged_groups, aliases, conversions, staged_digests


def _verify_staged_pixel_inputs(
    staged_digests: Mapping[str, str],
    identities: Sequence[_SourceIdentity] = (),
) -> None:
    """Recheck private conversions by content and original sources by identity."""

    for value, expected in staged_digests.items():
        path = Path(value)
        try:
            actual = sha256_digest(path)
        except OSError as error:
            raise E2EError(
                "PRIVATE_PIXEL_STAGING_CHANGED",
                "private pixel snapshot disappeared",
                path=value,
            ) from error
        if actual != expected:
            raise E2EError(
                "PRIVATE_PIXEL_STAGING_CHANGED",
                "private pixel snapshot changed during execution",
                path=value,
            )
    _verify_source_stat_identities(identities)


def _verify_source_stat_identities(identities: Sequence[_SourceIdentity]) -> None:
    for identity in identities:
        path = Path(identity.path)
        try:
            current = path.stat(follow_symlinks=False)
        except OSError as error:
            raise E2EError("SOURCE_CHANGED", "source disappeared", path=identity.path) from error
        actual = (current.st_size, current.st_mtime_ns, current.st_dev, current.st_ino)
        expected = (
            identity.size_bytes,
            identity.mtime_ns,
            identity.device,
            identity.inode,
        )
        if actual != expected:
            raise E2EError(
                "SOURCE_CHANGED",
                "source identity changed during execution",
                path=identity.path,
            )


def _canonical_inputs(values: Iterable[str], role: str, *, required: bool) -> tuple[Path, ...]:
    paths: list[Path] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise E2EError("INPUT_PATH_INVALID", f"{role} paths must be non-empty strings")
        try:
            path = Path(value).expanduser().resolve(strict=True)
        except OSError as error:
            raise E2EError("INPUT_MISSING", f"{role} input cannot be resolved", path=value) from error
        try:
            mode = path.stat(follow_symlinks=False).st_mode
        except OSError as error:
            raise E2EError("INPUT_STAT_FAILED", str(error), path=str(path)) from error
        if not stat.S_ISREG(mode):
            raise E2EError("INPUT_NOT_REGULAR", f"{role} input is not a regular file", path=str(path))
        key = os.path.normcase(str(path))
        if key in seen:
            raise E2EError("DUPLICATE_INPUT", f"duplicate {role} input", path=str(path))
        seen.add(key)
        paths.append(path)
    if required and not paths:
        raise E2EError("INPUT_GROUP_EMPTY", f"at least one {role} frame is required")
    return tuple(sorted(paths, key=lambda item: os.path.normcase(str(item))))


def _capture_sources(
    flattened: Sequence[tuple[str, Path]], *, workers: int = 1
) -> tuple[_SourceIdentity, ...]:
    """Hash every source once, several files at a time; order is preserved."""

    if not flattened:
        return ()
    count = max(1, min(int(workers), len(flattened), 8))
    if count == 1:
        return tuple(_capture_source(path, role) for role, path in flattened)
    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="ufwbpp-inventory") as pool:
        return tuple(pool.map(lambda item: _capture_source(item[1], item[0]), flattened))


def _capture_source(path: Path, role: str) -> _SourceIdentity:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise E2EError("INPUT_NOT_REGULAR", "input is not a regular file", path=str(path))
    digest = sha256_digest(path)
    after = path.stat(follow_symlinks=False)
    before_tuple = (before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino)
    after_tuple = (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino)
    if before_tuple != after_tuple:
        raise E2EError("SOURCE_CHANGED", "source changed while inventorying", path=str(path))
    return _SourceIdentity(
        str(path),
        role,
        digest,
        after.st_size,
        after.st_mtime_ns,
        after.st_dev,
        after.st_ino,
    )


def _trusted_source_identity_bindings(
    identities: Mapping[str, _SourceIdentity],
    paths: Sequence[Path],
) -> dict[str, InternalSourceIdentity]:
    result: dict[str, InternalSourceIdentity] = {}
    for path in paths:
        canonical = path.expanduser().resolve(strict=True)
        identity = identities.get(str(canonical))
        if identity is None:
            raise E2EError(
                "TRUSTED_GENERATED_CALIBRATION_SOURCE_DRIFT",
                "upstream source identity is missing from the generated-master handoff",
                path=str(canonical),
            )
        current = canonical.stat(follow_symlinks=False)
        actual_stat = (
            current.st_size,
            current.st_mtime_ns,
            current.st_dev,
            current.st_ino,
        )
        expected_stat = (
            identity.size_bytes,
            identity.mtime_ns,
            identity.device,
            identity.inode,
        )
        # The private pixel snapshot was copied through one open handle and its
        # digest was checked against this captured identity before this handoff.
        # Rechecking stat here detects path replacement; the final publication
        # gate performs the deliberate second full hash of every original.
        if actual_stat != expected_stat:
            raise E2EError(
                "TRUSTED_GENERATED_CALIBRATION_SOURCE_DRIFT",
                "source stat identity changed before the generated-master trust handoff",
                path=str(canonical),
            )
        result[identity.path] = InternalSourceIdentity(
            path=identity.path,
            sha256=identity.sha256,
            size_bytes=identity.size_bytes,
            mtime_ns=identity.mtime_ns,
            device=identity.device,
            inode=identity.inode,
        )
    if len(result) != len(paths):
        raise E2EError(
            "TRUSTED_GENERATED_CALIBRATION_SOURCE_DRIFT",
            "generated-master source bindings do not match the selected input set",
        )
    return result


def _verify_source(identity: _SourceIdentity) -> None:
    path = Path(identity.path)
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as error:
        raise E2EError("SOURCE_CHANGED", "source disappeared", path=identity.path) from error
    actual = (current.st_size, current.st_mtime_ns, current.st_dev, current.st_ino)
    expected = (
        identity.size_bytes,
        identity.mtime_ns,
        identity.device,
        identity.inode,
    )
    if actual != expected or sha256_digest(path) != identity.sha256:
        raise E2EError("SOURCE_CHANGED", "source identity changed during execution", path=identity.path)


def _verify_sources(identities: Sequence[_SourceIdentity], *, workers: int = 4) -> None:
    """Rehash every original; files are checked concurrently, failures surface as one."""

    count = max(1, min(int(workers), len(identities), 8))
    if count <= 1:
        for identity in identities:
            _verify_source(identity)
        return
    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="ufwbpp-verify") as pool:
        list(pool.map(_verify_source, identities))


@dataclass(frozen=True)
class _E2ESources:
    """The run's validated sources: canonical paths by role and the one
    path/stat-bound identity captured for each."""

    output: Path
    failure_directory: Path
    lights: tuple[Path, ...]
    flats: tuple[Path, ...]
    biases: tuple[Path, ...]
    darks: tuple[Path, ...]
    master_biases: tuple[Path, ...]
    master_darks: tuple[Path, ...]
    master_flats: tuple[Path, ...]
    identities: tuple[_SourceIdentity, ...]
    identity_by_path: dict[str, _SourceIdentity]

    def pixel_groups(self, lights: Sequence[Path]) -> tuple[tuple[str, tuple[Path, ...]], ...]:
        """Calibration sources with ``lights`` in the pixel pipeline's role order."""

        return (
            ("BIAS", self.biases),
            ("DARK", self.darks),
            ("FLAT", self.flats),
            ("MASTER_BIAS", self.master_biases),
            ("MASTER_DARK", self.master_darks),
            ("MASTER_FLAT", self.master_flats),
            ("LIGHT", tuple(lights)),
        )


def _validated_sources(request: E2ERequest, progress: ProgressCallback | None) -> _E2ESources:
    """Canonical, role-checked sources, each hashed exactly once."""

    output = Path(request.output_directory).expanduser().resolve(strict=False)
    failure_directory = output.with_name(output.name + ".unsolved")
    if os.path.lexists(output):
        raise E2EError("OUTPUT_EXISTS", "output directory must be new", path=str(output))
    if os.path.lexists(failure_directory):
        raise E2EError(
            "FAILURE_OUTPUT_EXISTS",
            "move the previous UNSOLVED evidence before retrying",
            path=str(failure_directory),
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    _emit(progress, ProgressStage.INVENTORY, "started", "validating explicit source files")
    lights = _canonical_inputs(request.light_files, "LIGHT", required=True)
    flats = _canonical_inputs(request.flat_files, "FLAT", required=False)
    biases = _canonical_inputs(request.bias_files, "BIAS", required=False)
    darks = _canonical_inputs(request.dark_files, "DARK", required=False)
    master_biases = _canonical_inputs(request.master_bias_files, "MASTER_BIAS", required=False)
    master_darks = _canonical_inputs(request.master_dark_files, "MASTER_DARK", required=False)
    master_flats = _canonical_inputs(request.master_flat_files, "MASTER_FLAT", required=False)
    if len(master_biases) > 1 or (biases and master_biases) or (not biases and not master_biases and request.pipeline_parameters.calibration_workflow != MONO_STANDARD):
        raise E2EError(
            "BIAS_SOURCE_AMBIGUOUS",
            "supply exactly one Bias source mode: raw Bias frames or one MasterBias",
        )
    if not flats and not master_flats:
        raise E2EError("MASTER_FLAT_MISSING", "raw Flats or supplied MasterFlats are required")
    all_grouped = (
        ("LIGHT", lights),
        ("FLAT", flats),
        ("DARK", darks),
        ("BIAS", biases),
        ("MASTER_BIAS", master_biases),
        ("MASTER_DARK", master_darks),
        ("MASTER_FLAT", master_flats),
    )
    flattened = [(role, path) for role, paths in all_grouped for path in paths]
    canonical_names = [os.path.normcase(str(path)) for _, path in flattened]
    if len(set(canonical_names)) != len(canonical_names):
        raise E2EError("INPUT_ROLE_OVERLAP", "one source appears in multiple roles")
    if any(path == output or output in path.parents for _, path in flattened):
        raise E2EError("OUTPUT_ALIASES_SOURCE", "output cannot contain or alias a source")
    # Capture every source exactly once.  All preflight/override lookups below
    # reuse only this path/stat-bound identity; private snapshot creation still
    # re-reads through one open handle and final publication still rehashes the
    # originals, so eliminating duplicate inventory passes weakens no drift gate.
    identities = _capture_sources(flattened, workers=request.workers)
    identity_by_path = {identity.path: identity for identity in identities}
    raw_digests: dict[str, list[Path]] = {}
    for identity in identities:
        if identity.role in {"LIGHT", "FLAT", "DARK", "BIAS"}:
            raw_digests.setdefault(identity.sha256, []).append(Path(identity.path))
    for override in request.pipeline_parameters.raw_frame_metadata_overrides:
        if len(raw_digests.get(override.source_sha256, [])) != 1:
            raise E2EError(
                "RAW_FRAME_METADATA_OVERRIDE_SOURCE_AMBIGUOUS",
                "raw-frame override digest must bind exactly one current E2E source",
            )
    for expected_role, path in flattened:
        try:
            frame_info = _input_frame_info(
                path,
                request.pipeline_parameters,
                override_source_identity=identity_by_path[str(path)],
            )
            actual_role = normalize_role(frame_info.role)
        except CalibrationError as error:
            raise E2EError(error.code, str(error), path=str(path)) from error
        if actual_role != expected_role:
            raise E2EError(
                "FRAME_ROLE_MISMATCH",
                f"expected {expected_role}, found {actual_role}",
                path=str(path),
            )
        if expected_role in {"LIGHT", "FLAT", "DARK", "BIAS"}:
            cfa = frame_info.cfa_pattern.strip().upper()
            if cfa in {"", "UNKNOWN", "UNSPECIFIED"}:
                raise E2EError(
                    "CFA_CONFIRMATION_REQUIRED",
                    f"raw {expected_role} CFA metadata is unknown; add a hash-bound rawFrameMetadataOverrides confirmation",
                    path=str(path),
                )
            if cfa != "NONE" and not is_cfa_pattern(cfa):
                raise E2EError(
                    "CFA_PATTERN_UNSUPPORTED",
                    f"CFA pattern {cfa!r} of raw {expected_role} is not supported (RGGB, BGGR, GRBG, GBRG are)",
                    path=str(path),
                )
    _emit(progress, ProgressStage.INVENTORY, "completed", f"validated {len(identities)} source files")
    return _E2ESources(
        output=output,
        failure_directory=failure_directory,
        lights=lights,
        flats=flats,
        biases=biases,
        darks=darks,
        master_biases=master_biases,
        master_darks=master_darks,
        master_flats=master_flats,
        identities=tuple(identities),
        identity_by_path=identity_by_path,
    )
