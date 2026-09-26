from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import math
from typing import Any


def json_value(value: Any, name: str = "value") -> Any:
    """Convert typed model data to strict, finite JSON-native values."""

    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite float")
        return value
    if isinstance(value, (tuple, list)):
        return [json_value(item, f"{name}[]") for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{name} contains a non-string key")
        return {key: json_value(item, f"{name}.{key}") for key, item in value.items()}
    raise TypeError(f"{name} contains unsupported type {type(value).__name__}")


class AssetRole(StrEnum):
    LIGHT = "LIGHT"
    FLAT = "FLAT"
    DARK = "DARK"
    BIAS = "BIAS"
    MASTER_FLAT = "MASTER_FLAT"
    MASTER_DARK = "MASTER_DARK"
    MASTER_BIAS = "MASTER_BIAS"
    MASTER_LIGHT = "MASTER_LIGHT"
    UNKNOWN = "UNKNOWN"


class AssetStatus(StrEnum):
    READY = "READY"
    CONFLICT = "CONFLICT"
    UNREADABLE = "UNREADABLE"


class IssueSeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class InventoryIssue:
    code: str
    severity: IssueSeverity
    message: str
    path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def serializable(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "path": self.path,
            "details": json_value(self.details, "issue.details"),
        }


@dataclass(frozen=True, slots=True)
class SourceStat:
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int

    def serializable(self) -> dict[str, int]:
        return {
            "sizeBytes": self.size_bytes,
            "mtimeNs": self.mtime_ns,
            "device": self.device,
            "inode": self.inode,
        }


@dataclass(frozen=True, slots=True)
class FrameAsset:
    asset_id: str
    path: str
    format: str
    role: AssetRole
    status: AssetStatus
    width: int = 0
    height: int = 0
    channels: int = 0
    filter_name: str = "UNKNOWN"
    target: str = "UNKNOWN"
    camera: str = "UNKNOWN"
    exposure_seconds: float | None = None
    temperature_celsius: float | None = None
    gain: float | None = None
    offset: float | None = None
    binning_x: int = 1
    binning_y: int = 1
    cfa_pattern: str = "UNKNOWN"
    cfa_explicit: bool = False
    readout_mode: str = "UNKNOWN"
    observed_at: str | None = None
    role_evidence: tuple[str, ...] = ()
    role_conflicts: tuple[str, ...] = ()
    group_id: str = ""
    source_stat: SourceStat | None = None
    error_code: str | None = None
    error_message: str | None = None
    # WBPP grouping keywords from the path (NIGHT, SESSION, PANEL, ...).
    grouping_keywords: tuple[tuple[str, str], ...] = ()

    @property
    def is_master(self) -> bool:
        return self.role in {
            AssetRole.MASTER_FLAT,
            AssetRole.MASTER_DARK,
            AssetRole.MASTER_BIAS,
            AssetRole.MASTER_LIGHT,
        }

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        value["status"] = self.status.value
        value["assetId"] = value.pop("asset_id")
        value["filter"] = value.pop("filter_name")
        value["exposureSeconds"] = value.pop("exposure_seconds")
        value["temperatureCelsius"] = value.pop("temperature_celsius")
        value["binningX"] = value.pop("binning_x")
        value["binningY"] = value.pop("binning_y")
        value["cfaPattern"] = value.pop("cfa_pattern")
        value["cfaExplicit"] = value.pop("cfa_explicit")
        value["readoutMode"] = value.pop("readout_mode")
        value["observedAt"] = value.pop("observed_at")
        value["roleEvidence"] = list(value.pop("role_evidence"))
        value["roleConflicts"] = list(value.pop("role_conflicts"))
        value["groupId"] = value.pop("group_id")
        source_stat = self.source_stat
        value["sourceStat"] = source_stat.serializable() if source_stat else None
        value.pop("source_stat", None)
        value["errorCode"] = value.pop("error_code")
        value["errorMessage"] = value.pop("error_message")
        keywords = value.pop("grouping_keywords")
        if keywords:
            value["groupingKeywords"] = {name: text for name, text in keywords}
        return json_value(value, "asset")


@dataclass(frozen=True, slots=True)
class ProjectInventory:
    project_id: str
    name: str
    source_roots: tuple[str, ...]
    assets: tuple[FrameAsset, ...]
    issues: tuple[InventoryIssue, ...] = ()
    schema_version: int = 1

    @property
    def counts(self) -> dict[str, int]:
        counts = {role.value: 0 for role in AssetRole}
        for asset in self.assets:
            counts[asset.role.value] += 1
        return counts

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == IssueSeverity.ERROR for issue in self.issues)

    def serializable(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "projectId": self.project_id,
            "name": self.name,
            "sourceRoots": list(self.source_roots),
            "counts": self.counts,
            "assets": [asset.serializable() for asset in self.assets],
            "issues": [issue.serializable() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class Project:
    """Typed input boundary shared by CLI, worker, and future GUI clients."""

    inventory: ProjectInventory
    output_directory: str

    def serializable(self) -> dict[str, Any]:
        return {
            "inventory": self.inventory.serializable(),
            "outputDirectory": self.output_directory,
        }
