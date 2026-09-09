from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Iterable

from .config import QcConfig
from .metadata import field_token
from .models import FrameMeasurement, FrameRole


def _optional_bucket(value: float | None, digits: int = 3) -> str:
    return "?" if value is None else f"{value:.{digits}f}"


def _base_key(frame: FrameMeasurement) -> tuple[str, ...]:
    metadata = frame.metadata
    return (
        metadata.camera,
        f"{metadata.width}x{metadata.height}x{metadata.channels}",
        f"{metadata.binning_x}x{metadata.binning_y}",
        metadata.filter_name,
        _optional_bucket(metadata.gain),
        _optional_bucket(metadata.offset),
        field_token(metadata),
    )


def _split_exposures(
    frames: list[FrameMeasurement], tolerance_fraction: float
) -> list[list[FrameMeasurement]]:
    ordered = sorted(
        frames,
        key=lambda frame: (
            frame.metadata.exposure_seconds is None,
            frame.metadata.exposure_seconds or 0.0,
            frame.metadata.path,
        ),
    )
    groups: list[list[FrameMeasurement]] = []
    centers: list[float | None] = []
    for frame in ordered:
        exposure = frame.metadata.exposure_seconds
        destination = None
        for index, center in enumerate(centers):
            if exposure is None or center is None:
                if exposure is center:
                    destination = index
                    break
                continue
            if abs(exposure - center) <= max(exposure, center) * tolerance_fraction:
                destination = index
                break
        if destination is None:
            groups.append([frame])
            centers.append(exposure)
        else:
            groups[destination].append(frame)
            values = [
                item.metadata.exposure_seconds
                for item in groups[destination]
                if item.metadata.exposure_seconds is not None
            ]
            centers[destination] = sum(values) / len(values) if values else None
    return groups


def build_groups(
    frames: Iterable[FrameMeasurement], config: QcConfig
) -> list[tuple[str, list[FrameMeasurement]]]:
    buckets: dict[tuple[str, ...], list[FrameMeasurement]] = defaultdict(list)
    for frame in frames:
        if frame.status == "MEASURED" and frame.metadata.role is FrameRole.LIGHT:
            buckets[_base_key(frame)].append(frame)

    result: list[tuple[str, list[FrameMeasurement]]] = []
    for key in sorted(buckets):
        for exposure_group in _split_exposures(
            buckets[key], config.group_exposure_tolerance_fraction
        ):
            descriptor = {
                "base": key,
                "exposure": sorted(
                    {
                        frame.metadata.exposure_seconds for frame in exposure_group
                    },
                    key=lambda value: (value is None, value or 0),
                ),
            }
            digest = hashlib.sha256(
                json.dumps(descriptor, ensure_ascii=True).encode("utf-8")
            ).hexdigest()[:12]
            result.append((f"group-{digest}", sorted(exposure_group, key=lambda x: x.metadata.path)))
    return result
