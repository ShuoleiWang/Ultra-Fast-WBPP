"""Shared metadata policy for planning and pixel execution.

Standard mono records assumptions without inventing acquisition metadata.
"""
from dataclasses import replace
from typing import Any, Mapping

STRICT = "strict-v1"
MONO_STANDARD = "mono-standard-v1"
WORKFLOWS = frozenset({STRICT, MONO_STANDARD})


def unknown(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().upper() in {"", "UNKNOWN", "UNSPECIFIED"})


def canonical(value: Any) -> Any:
    return value.strip().upper() if isinstance(value, str) else value


def cfa_for_workflow(value: Any, workflow: str) -> Any:
    return "NONE" if workflow == MONO_STANDARD and unknown(value) else canonical(value)


def same_metadata(left: Any, right: Any, workflow: str = STRICT, *, required: bool = False) -> bool:
    if unknown(left) or unknown(right):
        return workflow == MONO_STANDARD and not required
    return canonical(left) == canonical(right)


def apply_mono_workflow(info: Any, workflow: str) -> Any:
    if workflow == MONO_STANDARD and unknown(info.cfa_pattern):
        return replace(info, cfa_pattern="NONE")
    return info


def metadata_changes(override: Any) -> dict[str, Any]:
    fields = ("camera", "gain", "offset", "binning_x", "binning_y", "filter_name", "cfa_pattern", "readout_mode", "temperature_celsius", "exposure_seconds")
    return {name: canonical(value) for name in fields if not unknown(value := getattr(override, name))}


def bias_from_header(header: Mapping[str, Any]) -> bool | None:
    # These explicit declarations have a defined polarity. Never infer from
    # image statistics, temperature or a filename.
    value = canonical(header.get("OAFBIAS"))
    if value in {"INCLUDED", "SUBTRACTED"}:
        return value == "INCLUDED"
    value = header.get("BIASINC")
    if isinstance(value, bool):
        return value
    if canonical(value) in {"T", "TRUE", "F", "FALSE"}:
        return canonical(value) in {"T", "TRUE"}
    return None


def resolve_dark_bias(explicit: bool | None, header: bool | None, workflow: str) -> bool | None:
    if explicit is not None:
        return explicit
    if workflow == MONO_STANDARD:
        return True if header is None else header
    return None


def workflow_receipt(workflow: str) -> dict[str, Any]:
    return {"workflow": workflow, "missingCfa": "MONO_WORKFLOW_DEFAULT" if workflow == MONO_STANDARD else "REQUIRE_CONFIRMATION", "unknownAcquisitionMetadata": "PRESERVED_UNKNOWN", "knownConflicts": "REJECT", "undeclaredMasterDarkBiasIncluded": True if workflow == MONO_STANDARD else None}


def can_omit_bias(frames: Any, darks: Any, workflow: str) -> bool:
    """Bias is unnecessary only if every raw target subtracts a Bias-inclusive Dark.

    Detailed camera/temperature/ambiguity checks remain separate and mandatory.
    """
    import math
    if workflow != MONO_STANDARD:
        return False
    targets = list(frames)
    available = list(darks)
    return bool(targets) and all(
        frame.exposure_seconds is not None and any(
            included is True and dark.exposure_seconds is not None and math.isclose(frame.exposure_seconds, dark.exposure_seconds, rel_tol=0.0, abs_tol=1e-6)
            for dark, included in available
        ) for frame in targets
    )


def conflicting_profile_fields(infos: Any, workflow: str) -> list[str]:
    # An unknown reference must not mask contradictions between two other files.
    if workflow != MONO_STANDARD:
        return []
    values = list(infos)
    return [name for name in ("camera", "gain", "offset", "binning_x", "binning_y", "readout_mode") if len({canonical(getattr(info, name)) for info in values if not unknown(getattr(info, name))}) > 1]


def acquisition_receipt(info: Any) -> dict[str, Any]:
    names = {"camera": "camera", "gain": "gain", "offset": "offset", "readoutMode": "readout_mode", "temperatureCelsius": "temperature_celsius", "exposureSeconds": "exposure_seconds", "binningX": "binning_x", "binningY": "binning_y"}
    values = {key: getattr(info, attr) for key, attr in names.items()}
    return {"values": values, "unrecordedFields": [key for key, value in values.items() if unknown(value)]}
