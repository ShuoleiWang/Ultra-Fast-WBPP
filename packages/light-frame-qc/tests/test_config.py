from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from lightframeqc.config import DEFAULT_CONFIG, QcConfig, load_config


def test_default_config_file_matches_valid_built_in_defaults() -> None:
    path = Path(__file__).resolve().parents[1] / "default-config.json"

    loaded = load_config(path)

    assert loaded == DEFAULT_CONFIG
    loaded.validate()


@pytest.mark.parametrize(
    "config",
    [
        replace(DEFAULT_CONFIG, preview_long_edge=True),
        replace(DEFAULT_CONFIG, minimum_source_support_pixels=True),
        replace(DEFAULT_CONFIG, minimum_source_support_pixels=0),
        replace(DEFAULT_CONFIG, minimum_source_support_pixels=-1),
        replace(DEFAULT_CONFIG, minimum_source_support_pixels=2.5),
        replace(DEFAULT_CONFIG, detection_sigma=float("nan")),
        replace(
            DEFAULT_CONFIG,
            weak_spatial_transparency_p90_mag=0.8,
            strong_spatial_transparency_p90_mag=0.4,
        ),
        replace(
            DEFAULT_CONFIG,
            strong_completeness_ratio=0.9,
            weak_completeness_ratio=0.7,
        ),
        replace(DEFAULT_CONFIG, minimum_overlap_fraction=1.1),
        replace(DEFAULT_CONFIG, automatic_cloud_score=0),
        replace(DEFAULT_CONFIG, observing_timezone="Mars/Olympus"),
    ],
)
def test_invalid_or_reversed_thresholds_fail_closed(config: QcConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_load_config_rejects_unknown_and_nonfinite_values(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"not_a_setting": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown configuration"):
        load_config(unknown)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"detection_sigma": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_config(nonfinite)


def test_actual_pixel_support_threshold_is_explicit_and_serialized(tmp_path: Path) -> None:
    path = tmp_path / "support.json"
    path.write_text(json.dumps({"minimum_source_support_pixels": 4}), encoding="utf-8")
    configured = load_config(path)

    assert DEFAULT_CONFIG.minimum_source_pixels == 5
    assert DEFAULT_CONFIG.minimum_source_support_pixels == 3
    assert configured.minimum_source_support_pixels == 4
    assert configured.serializable()["minimum_source_support_pixels"] == 4
