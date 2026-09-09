from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest

import lightframeqc.measure as measure_module
from lightframeqc.measure import (
    FrameMeasurementError,
    MeasurementSettings,
    measure_preview,
)
from lightframeqc.models import (
    Confidence,
    Decision,
    EvidenceFamily,
    EvidenceSeverity,
    FrameFeatures,
    FrameMetadata,
    FrameResult,
    FrameRole,
    GateDisposition,
    QualityEvidence,
    QualityGateResult,
    RegistrationMetrics,
    RunResult,
)
from lightframeqc.readers import ImagePreview
from lightframeqc.report import write_reports


def _preview(tmp_path: Path) -> ImagePreview:
    rng = np.random.default_rng(20260901)
    data = rng.normal(1_000.0, 7.0, size=(64, 96)).astype(np.float32)
    data[0, 0] = np.nan
    data[4, 9] = np.inf
    data = np.ascontiguousarray(data)
    data.setflags(write=False)
    return ImagePreview(
        path=str(tmp_path / "synthetic.fits"),
        data=data,
        metadata=FrameMetadata(
            path=str(tmp_path / "synthetic.fits"),
            width=384,
            height=192,
            channels=1,
            role=FrameRole.LIGHT,
        ),
        source_width=384,
        source_height=192,
        source_channels=1,
        block_size=4,
        reader_backend="test-preview",
    )


def test_measure_preview_persists_scale_finite_fraction_and_robust_range(
    tmp_path: Path,
) -> None:
    preview = _preview(tmp_path)
    finite = np.asarray(preview.data[np.isfinite(preview.data)], dtype=np.float64)
    expected_p001, expected_p999 = np.percentile(finite, (0.1, 99.9))

    measurement = measure_preview(
        preview,
        settings=MeasurementSettings(maximum_stars=100),
    )

    assert measurement.preview_scale_x == pytest.approx(4.0)
    assert measurement.preview_scale_y == pytest.approx(3.0)
    assert measurement.finite_fraction == pytest.approx(finite.size / preview.data.size)
    assert measurement.image_p001 == pytest.approx(expected_p001)
    assert measurement.image_p999 == pytest.approx(expected_p999)
    assert measurement.dynamic_range == pytest.approx(expected_p999 - expected_p001)
    assert measurement.image_median == pytest.approx(np.median(finite))
    assert measurement.image_mad == pytest.approx(
        np.median(np.abs(finite - np.median(finite)))
    )
    # A block-mean preview cannot prove an original-pixel saturation fraction.
    assert not hasattr(measurement, "saturation_fraction")


def test_sep_background_failure_preserves_the_actual_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview = _preview(tmp_path)

    def fail_background(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic SEP background failure")

    monkeypatch.setattr(measure_module.sep, "Background", fail_background)
    with pytest.raises(FrameMeasurementError) as raised:
        measure_preview(preview)

    assert raised.value.code == "SEP_BACKGROUND_FAILED"
    assert raised.value.detail == "synthetic SEP background failure"
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_quality_gate_models_emit_strict_enum_free_json() -> None:
    evidence = QualityEvidence(
        code="MORPHOLOGY.ELONGATED",
        family=EvidenceFamily.MORPHOLOGY,
        severity=EvidenceSeverity.REVIEW,
        message="Elongated-star fraction exceeds the review boundary.",
        value=0.27,
        threshold=0.20,
        units="fraction",
        details={"validStars": 84, "sources": ["SEP", "native-scale"]},
    )
    gate = QualityGateResult(
        disposition=GateDisposition.REVIEW,
        summary="Shape evidence requires review.",
        evidence=[evidence],
    )

    payload = gate.serializable()
    encoded = json.dumps(payload, allow_nan=False)

    assert payload["disposition"] == "REVIEW"
    assert payload["evidence"][0]["family"] == "MORPHOLOGY"
    assert payload["evidence"][0]["severity"] == "REVIEW"
    assert "GateDisposition" not in encoded

    invalid = QualityEvidence(
        code="NOISE.NONFINITE",
        family=EvidenceFamily.NOISE,
        severity=EvidenceSeverity.ERROR,
        message="Non-finite values are forbidden.",
        value=float("nan"),
    )
    with pytest.raises(ValueError, match="non-finite"):
        invalid.serializable()


def test_reports_expose_gate_and_key_quality_metrics(tmp_path: Path) -> None:
    evidence = QualityEvidence(
        code="MORPHOLOGY.HARD_TRAIL",
        family=EvidenceFamily.MORPHOLOGY,
        severity=EvidenceSeverity.HARD_FAIL,
        message="Coherent elongation indicates tracking failure.",
        value=0.61,
        threshold=0.45,
        units="fraction",
    )
    result = FrameResult(
        path=str(tmp_path / "bad-frame.fits"),
        group_id="group-R",
        reference_path=None,
        decision=Decision.REVIEW,
        confidence=Confidence.HIGH,
        reasons=["shape evidence"],
        warnings=[],
        registration=RegistrationMetrics(ok=True, matched_stars=80),
        features=FrameFeatures(
            image_median=1_004.5,
            image_mad=6.2,
            median_fwhm_native_pixels=5.8,
            p90_fwhm_native_pixels=7.1,
            median_axis_ratio=0.58,
            median_eccentricity=0.81,
            elongated_fraction=0.61,
            orientation_coherence=0.92,
            valid_morphology_star_count=84,
            nightly_extinction_residual=0.17,
            background_z=2.4,
            noise_z=1.8,
        ),
        metadata=FrameMetadata(
            path=str(tmp_path / "bad-frame.fits"),
            filter_name="R",
            role=FrameRole.LIGHT,
        ),
        star_count=84,
        quality_gate=QualityGateResult(
            disposition=GateDisposition.HARD_FAIL,
            summary="Tracking failure",
            evidence=[evidence],
        ),
    )
    run = RunResult(
        schema_version=2,
        algorithm_version="test",
        generated_at=datetime.now(timezone.utc),
        inputs=[str(tmp_path)],
        output_directory=str(tmp_path / "report"),
        config={},
        groups=[],
        frames=[result],
    )

    outputs = write_reports(run)
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    with outputs["csv"].open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    html = outputs["html"].read_text(encoding="utf-8")

    assert payload["frames"][0]["qualityGate"]["disposition"] == "HARD_FAIL"
    assert payload["frames"][0]["qualityGate"]["evidence"][0]["code"] == (
        "MORPHOLOGY.HARD_TRAIL"
    )
    assert payload["frames"][0]["features"]["background_z"] == 2.4
    assert row["gate_disposition"] == "HARD_FAIL"
    assert row["gate_evidence_count"] == "1"
    assert row["gate_evidence_codes"] == "MORPHOLOGY.HARD_TRAIL"
    assert row["median_fwhm_native_pixels"] == "5.8"
    assert row["orientation_coherence"] == "0.92"
    assert "GATE HARD_FAIL" in html
    assert "MORPHOLOGY.HARD_TRAIL" in html
