from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QcConfig:
    preview_long_edge: int = 2048
    grid_rows: int = 16
    grid_columns: int = 16
    detection_sigma: float = 4.5
    minimum_source_pixels: int = 5
    # Actual above-threshold pixels before SEP's detection convolution.
    minimum_source_support_pixels: int = 3
    maximum_stars: int = 2500
    group_exposure_tolerance_fraction: float = 0.05
    minimum_group_frames: int = 4
    high_confidence_group_frames: int = 8
    minimum_registration_matches: int = 12
    high_confidence_registration_matches: int = 30
    minimum_registration_fraction: float = 0.25
    maximum_registration_rms_pixels: float = 2.5
    high_confidence_rms_pixels: float = 1.5
    weak_spatial_transparency_p90_mag: float = 0.18
    strong_spatial_transparency_p90_mag: float = 0.45
    weak_completeness_ratio: float = 0.70
    strong_completeness_ratio: float = 0.45
    weak_extra_extinction_mag: float = 0.35
    strong_extra_extinction_mag: float = 0.45
    minimum_overlap_fraction: float = 0.50
    weak_occlusion_area: float = 0.05
    strong_occlusion_area: float = 0.15
    very_strong_occlusion_area: float = 0.30
    weak_occlusion_density_ratio: float = 0.35
    strong_occlusion_density_ratio: float = 0.15
    weak_boundary_support: float = 0.50
    strong_boundary_support: float = 0.65
    automatic_cloud_score: int = 4
    make_thumbnails: bool = True
    observing_timezone: str | None = None

    def validate(self) -> None:
        integer_fields = (
            "preview_long_edge",
            "grid_rows",
            "grid_columns",
            "minimum_source_pixels",
            "minimum_source_support_pixels",
            "maximum_stars",
            "minimum_group_frames",
            "high_confidence_group_frames",
            "minimum_registration_matches",
            "high_confidence_registration_matches",
            "automatic_cloud_score",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in integer_fields or item.name in {
                "make_thumbnails",
                "observing_timezone",
            }:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{item.name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{item.name} must be finite")
        if not isinstance(self.make_thumbnails, bool):
            raise ValueError("make_thumbnails must be boolean")
        if self.observing_timezone is not None:
            if not isinstance(self.observing_timezone, str) or not self.observing_timezone.strip():
                raise ValueError("observing_timezone must be None or a non-empty string")
            from .nightly_statistics import _observing_timezone

            _observing_timezone(self.observing_timezone)
        if self.preview_long_edge < 256:
            raise ValueError("preview_long_edge must be at least 256")
        if self.grid_rows < 4 or self.grid_columns < 4:
            raise ValueError("quality grids must be at least 4x4")
        if not 1.0 <= self.detection_sigma <= 20.0:
            raise ValueError("detection_sigma must be in [1, 20]")
        if self.minimum_source_pixels < 1:
            raise ValueError("minimum_source_pixels must be positive")
        if self.minimum_source_support_pixels < 1:
            raise ValueError("minimum_source_support_pixels must be positive")
        if self.maximum_stars < self.minimum_registration_matches:
            raise ValueError("maximum_stars cannot be below minimum_registration_matches")
        if self.minimum_group_frames < 2:
            raise ValueError("minimum_group_frames must be at least two")
        if self.high_confidence_group_frames < self.minimum_group_frames:
            raise ValueError("high confidence group size cannot be smaller")
        if self.high_confidence_registration_matches < self.minimum_registration_matches:
            raise ValueError("high confidence registration matches cannot be smaller")
        if self.maximum_registration_rms_pixels <= 0:
            raise ValueError("maximum registration RMS must be positive")
        if not 0 < self.high_confidence_rms_pixels <= self.maximum_registration_rms_pixels:
            raise ValueError("high confidence RMS must be positive and no larger than maximum")
        if self.automatic_cloud_score < 1:
            raise ValueError("automatic_cloud_score must be positive")
        for name in (
            "group_exposure_tolerance_fraction",
            "minimum_registration_fraction",
            "weak_completeness_ratio",
            "strong_completeness_ratio",
            "minimum_overlap_fraction",
            "weak_occlusion_area",
            "strong_occlusion_area",
            "very_strong_occlusion_area",
            "weak_occlusion_density_ratio",
            "strong_occlusion_density_ratio",
            "weak_boundary_support",
            "strong_boundary_support",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        positive_pairs = (
            (
                "weak_spatial_transparency_p90_mag",
                "strong_spatial_transparency_p90_mag",
            ),
            ("weak_extra_extinction_mag", "strong_extra_extinction_mag"),
        )
        for weak_name, strong_name in positive_pairs:
            weak = getattr(self, weak_name)
            strong = getattr(self, strong_name)
            if weak < 0 or strong < weak:
                raise ValueError(f"require 0 <= {weak_name} <= {strong_name}")
        if self.strong_completeness_ratio > self.weak_completeness_ratio:
            raise ValueError("strong completeness ratio cannot exceed weak ratio")
        if not (
            self.weak_occlusion_area
            <= self.strong_occlusion_area
            <= self.very_strong_occlusion_area
        ):
            raise ValueError("occlusion area thresholds must be nondecreasing")
        if self.strong_occlusion_density_ratio > self.weak_occlusion_density_ratio:
            raise ValueError("strong occlusion density ratio cannot exceed weak ratio")
        if self.strong_boundary_support < self.weak_boundary_support:
            raise ValueError("strong boundary support cannot be below weak support")

    def serializable(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = QcConfig()


def load_config(path: str | Path | None) -> QcConfig:
    if path is None:
        DEFAULT_CONFIG.validate()
        return DEFAULT_CONFIG
    config_path = Path(path).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a JSON object")
    allowed = {item.name for item in fields(QcConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError("unknown configuration field(s): " + ", ".join(unknown))
    result = QcConfig(**raw)
    result.validate()
    return result
