from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Iterable

from lightframeqc.models import FrameRole
from lightframeqc.readers import (
    FrameReadError,
    discover_paths,
    probe_frame_metadata,
)

from .models import (
    AssetRole,
    AssetStatus,
    FrameAsset,
    InventoryIssue,
    IssueSeverity,
    ProjectInventory,
    SourceStat,
)


_ROLE_MAP = {
    FrameRole.LIGHT: AssetRole.LIGHT,
    FrameRole.RAW_FLAT: AssetRole.FLAT,
    FrameRole.DARK: AssetRole.DARK,
    FrameRole.BIAS: AssetRole.BIAS,
    FrameRole.MASTER_FLAT: AssetRole.MASTER_FLAT,
    FrameRole.MASTER_DARK: AssetRole.MASTER_DARK,
    FrameRole.MASTER_BIAS: AssetRole.MASTER_BIAS,
    FrameRole.MASTER_LIGHT: AssetRole.MASTER_LIGHT,
    FrameRole.UNKNOWN: AssetRole.UNKNOWN,
}


class InventoryBuildError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _source_stat(path: Path) -> SourceStat:
    stat = path.stat(follow_symlinks=False)
    if not path.is_file():
        raise OSError("not a regular file")
    return SourceStat(
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        device=stat.st_dev,
        inode=stat.st_ino,
    )


def _asset_id(path: Path, source_stat: SourceStat) -> str:
    return _digest(
        {
            "path": os.path.normcase(str(path)),
            "sizeBytes": source_stat.size_bytes,
            "mtimeNs": source_stat.mtime_ns,
            "device": source_stat.device,
            "inode": source_stat.inode,
        }
    )


def _format(path: Path) -> str:
    return "XISF" if path.name.casefold().endswith(".xisf") else "FITS"


def _group_id(
    *,
    role: AssetRole,
    width: int,
    height: int,
    channels: int,
    filter_name: str,
    target: str,
    camera: str,
    exposure_seconds: float | None,
    temperature_celsius: float | None,
    gain: float | None,
    offset: float | None,
    binning_x: int,
    binning_y: int,
    cfa_pattern: str,
    readout_mode: str,
) -> str:
    group_role = {
        AssetRole.MASTER_FLAT: AssetRole.FLAT,
        AssetRole.MASTER_DARK: AssetRole.DARK,
        AssetRole.MASTER_BIAS: AssetRole.BIAS,
    }.get(role, role)
    payload = {
        "role": group_role.value,
        "geometry": [width, height, channels],
        "filter": filter_name if group_role in {AssetRole.LIGHT, AssetRole.FLAT} else None,
        "target": target if group_role == AssetRole.LIGHT else None,
        "camera": camera,
        "exposureSeconds": exposure_seconds
        if group_role in {AssetRole.LIGHT, AssetRole.DARK}
        else None,
        "temperatureCelsius": temperature_celsius
        if group_role in {AssetRole.LIGHT, AssetRole.DARK}
        else None,
        "gain": gain,
        "offset": offset,
        "binning": [binning_x, binning_y],
        "cfaPattern": cfa_pattern,
        "readoutMode": readout_mode,
    }
    return _digest(payload)


def _temperature_celsius(header: dict[str, object]) -> float | None:
    for key in ("CCD-TEMP", "CCD_TEMP", "SENSORT", "SENSOR-T", "CAMTEMP"):
        value = header.get(key)
        try:
            number = float(value) if value is not None else math.nan
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _cfa_is_explicit(header: dict[str, object]) -> bool:
    keys = {str(key).strip().upper() for key in header}
    return bool(keys & {"BAYERPAT", "BAYERPATTERN", "CFA", "CFAPAT", "COLORTYP"})


def _default_project_name(roots: tuple[Path, ...]) -> str:
    if len(roots) == 1:
        return roots[0].stem or roots[0].name or "Ultra-Fast WBPP Project"
    try:
        common = Path(os.path.commonpath([str(path) for path in roots]))
    except ValueError:
        # Multiple Windows drives have no common path.
        return "Ultra-Fast WBPP Project"
    return common.name or "Ultra-Fast WBPP Project"


def _normalized_roots(
    inputs: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    if isinstance(inputs, (str, os.PathLike)):
        inputs = (inputs,)
    roots: list[Path] = []
    seen: set[str] = set()
    for value in inputs:
        path = Path(value).expanduser()
        if not path.exists():
            raise InventoryBuildError("INPUT_NOT_FOUND", f"input does not exist: {path}")
        resolved = path.resolve(strict=True)
        key = os.path.normcase(str(resolved))
        if key not in seen:
            roots.append(resolved)
            seen.add(key)
    if not roots:
        raise InventoryBuildError("NO_INPUTS", "at least one input path is required")
    return tuple(sorted(roots, key=lambda path: os.path.normcase(str(path))))


def inventory_manifest_sha256(inventory: ProjectInventory) -> str:
    """Return the raw SHA-256 of the canonical inventory snapshot.

    The digest includes resolved paths, role/metadata probes, source stat
    identities, and inventory issues.  Pixel execution performs stronger full
    content hashes before publication; this manifest digest binds planning.
    """

    encoded = json.dumps(
        inventory.serializable(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_digest(path: str, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = limit
    with open(path, "rb") as stream:
        while remaining is None or remaining > 0:
            block = stream.read(1 << 20 if remaining is None else min(1 << 20, remaining))
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def _duplicate_lights(assets: list[FrameAsset]) -> dict[int, str]:
    """Byte-identical copies of a ready Light: index -> the kept original.

    Candidates share size and the first 64 KiB (header and early pixels); only
    they are hashed in full, so a normal import reads 64 KiB per Light at most
    once and never a whole frame.
    """

    by_size: dict[int, list[int]] = {}
    for index, asset in enumerate(assets):
        if asset.role == AssetRole.LIGHT and asset.status == AssetStatus.READY and asset.source_stat is not None:
            by_size.setdefault(asset.source_stat.size_bytes, []).append(index)
    duplicates: dict[int, str] = {}
    for indices in by_size.values():
        if len(indices) < 2:
            continue
        by_prefix: dict[str, list[int]] = {}
        for index in indices:
            by_prefix.setdefault(_content_digest(assets[index].path, 64 * 1024), []).append(index)
        for candidates in by_prefix.values():
            if len(candidates) < 2:
                continue
            first_by_content: dict[str, int] = {}
            for index in candidates:
                original = first_by_content.setdefault(_content_digest(assets[index].path), index)
                if original != index:
                    duplicates[index] = assets[original].path
    return duplicates


def _examples(paths: list[str], count: int = 2) -> str:
    return ", ".join(Path(path).name for path in paths[:count]) + (", …" if len(paths) > count else "")


def inventory_project(
    inputs: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    *,
    name: str | None = None,
) -> ProjectInventory:
    """Recursively inventory NINA FITS/XISF assets using header-only probes.

    The returned source identities are stat snapshots, not content hashes. They
    make a plan auditable without forcing a full read of every large frame.
    Execution backends must content-hash and revalidate their inputs before
    publishing outputs.
    """

    roots = _normalized_roots(inputs)
    try:
        paths = discover_paths(roots)
    except FrameReadError as error:
        raise InventoryBuildError(error.code, str(error)) from error

    assets: list[FrameAsset] = []
    issues: list[InventoryIssue] = []
    processed_lights: list[tuple[str, tuple[str, ...]]] = []
    master_lights: list[str] = []
    for path in paths:
        try:
            source_stat = _source_stat(path)
        except OSError as error:
            issues.append(
                InventoryIssue(
                    code="SOURCE_STAT_FAILED",
                    severity=IssueSeverity.ERROR,
                    message=str(error),
                    path=str(path),
                )
            )
            continue
        asset_id = _asset_id(path, source_stat)
        try:
            metadata = probe_frame_metadata(path)
        except FrameReadError as error:
            assets.append(
                FrameAsset(
                    asset_id=asset_id,
                    path=str(path),
                    format=_format(path),
                    role=AssetRole.UNKNOWN,
                    status=AssetStatus.UNREADABLE,
                    source_stat=source_stat,
                    error_code=error.code,
                    error_message=error.detail,
                )
            )
            issues.append(
                InventoryIssue(
                    code=error.code,
                    severity=IssueSeverity.ERROR,
                    message=error.detail,
                    path=str(path),
                )
            )
            continue

        role = _ROLE_MAP[metadata.role]
        # A Light PixInsight already calibrated or registered (WBPP output)
        # keeps IMAGETYP=LIGHT; processing it again would repeat the dark
        # subtraction and flat division, so it is refused, never guessed.
        processed = tuple(metadata.processing) if role == AssetRole.LIGHT else ()
        conflicts = (
            *metadata.role_conflicts,
            *(
                (f"already {' and '.join(step.lower() for step in processed)} by PixInsight; import the raw Light",)
                if processed
                else ()
            ),
        )
        status = AssetStatus.CONFLICT if conflicts else AssetStatus.READY
        observed_at = (
            metadata.observed_at.isoformat() if metadata.observed_at is not None else None
        )
        group_id = _group_id(
            role=role,
            width=metadata.width,
            height=metadata.height,
            channels=metadata.channels,
            filter_name=metadata.filter_name,
            target=metadata.target,
            camera=metadata.camera,
            exposure_seconds=metadata.exposure_seconds,
            temperature_celsius=_temperature_celsius(metadata.header),
            gain=metadata.gain,
            offset=metadata.offset,
            binning_x=metadata.binning_x,
            binning_y=metadata.binning_y,
            cfa_pattern=metadata.cfa_pattern,
            readout_mode=metadata.readout_mode,
        )
        asset = FrameAsset(
            asset_id=asset_id,
            path=str(path),
            format=_format(path),
            role=role,
            status=status,
            width=metadata.width,
            height=metadata.height,
            channels=metadata.channels,
            filter_name=metadata.filter_name,
            target=metadata.target,
            camera=metadata.camera,
            exposure_seconds=metadata.exposure_seconds,
            temperature_celsius=_temperature_celsius(metadata.header),
            gain=metadata.gain,
            offset=metadata.offset,
            binning_x=metadata.binning_x,
            binning_y=metadata.binning_y,
            cfa_pattern=metadata.cfa_pattern,
            cfa_explicit=_cfa_is_explicit(metadata.header),
            readout_mode=metadata.readout_mode,
            observed_at=observed_at,
            role_evidence=tuple(metadata.role_evidence),
            role_conflicts=conflicts,
            group_id=group_id,
            source_stat=source_stat,
        )
        assets.append(asset)
        if metadata.role_conflicts:
            issues.append(
                InventoryIssue(
                    code="ROLE_CONFLICT",
                    severity=IssueSeverity.ERROR,
                    message="; ".join(metadata.role_conflicts),
                    path=str(path),
                    details={"evidence": metadata.role_evidence},
                )
            )
        elif processed:
            processed_lights.append((str(path), processed))
        elif role == AssetRole.MASTER_LIGHT:
            master_lights.append(str(path))
        elif role == AssetRole.UNKNOWN:
            issues.append(
                InventoryIssue(
                    code="ROLE_UNKNOWN",
                    severity=IssueSeverity.WARNING,
                    message="frame role could not be established from metadata or path",
                    path=str(path),
                    details={"evidence": metadata.role_evidence},
                )
            )

    if processed_lights:
        paths = [path for path, _ in processed_lights]
        issues.append(
            InventoryIssue(
                code="PROCESSED_LIGHT",
                severity=IssueSeverity.ERROR,
                message=(
                    f"{len(paths)} Light(s) were already calibrated or registered by PixInsight "
                    f"(WBPP output such as {_examples(paths)}); import the raw Lights instead"
                ),
                path=paths[0],
                details={
                    "count": len(paths),
                    "calibrated": sum("CALIBRATED" in steps for _, steps in processed_lights),
                    "registered": sum("REGISTERED" in steps for _, steps in processed_lights),
                    "paths": paths,
                },
            )
        )
    duplicates = _duplicate_lights(assets)
    for index, original in duplicates.items():
        assets[index] = replace(
            assets[index],
            status=AssetStatus.CONFLICT,
            role_conflicts=(*assets[index].role_conflicts, f"byte-identical copy of {original}"),
        )
    if duplicates:
        copies = [assets[index].path for index in sorted(duplicates)]
        issues.append(
            InventoryIssue(
                code="DUPLICATE_LIGHT",
                severity=IssueSeverity.ERROR,
                message=(
                    f"{len(copies)} Light(s) are byte-identical copies of other imported Lights "
                    f"({_examples(copies)}); remove the copies so no exposure counts twice"
                ),
                path=copies[0],
                details={"copies": {assets[index].path: original for index, original in sorted(duplicates.items())}},
            )
        )
    if master_lights:
        issues.append(
            InventoryIssue(
                code="MASTER_LIGHT_IGNORED",
                severity=IssueSeverity.WARNING,
                message=(
                    f"{len(master_lights)} integrated master Light(s) ({_examples(master_lights)}) are "
                    "products, not inputs, and are ignored"
                ),
                path=master_lights[0],
                details={"paths": master_lights},
            )
        )

    if not any(asset.role == AssetRole.LIGHT and asset.status == AssetStatus.READY for asset in assets):
        issues.append(
            InventoryIssue(
                code="NO_LIGHTS",
                severity=IssueSeverity.ERROR,
                message="the selected inputs contain no unprocessed Light frames",
            )
        )

    if name is not None and not isinstance(name, str):
        raise InventoryBuildError("INVALID_PROJECT_NAME", "project name must be a string")
    project_name = (name or _default_project_name(roots)).strip()
    if not project_name:
        raise InventoryBuildError("INVALID_PROJECT_NAME", "project name cannot be empty")
    project_id = _digest(
        {
            "roots": [os.path.normcase(str(path)) for path in roots],
            "assets": [asset.asset_id for asset in assets],
        }
    )
    return ProjectInventory(
        project_id=project_id,
        name=re.sub(r"\s+", " ", project_name),
        source_roots=tuple(str(path) for path in roots),
        assets=tuple(assets),
        issues=tuple(issues),
    )


__all__ = ["InventoryBuildError", "inventory_project"]
