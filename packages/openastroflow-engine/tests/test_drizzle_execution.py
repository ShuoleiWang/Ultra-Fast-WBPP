from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from astropy.io import fits
import numpy as np
import pytest

import openastroflow_engine.drizzle_execution as drizzle_execution
from openastroflow_engine.drizzle_execution import (
    DrizzleExecutionError,
    DrizzleExecutionRequest,
    DrizzleFrameInput,
    DrizzleProvider,
    execute_drizzle,
    projective_input_to_output_pixmap,
    stsci_drizzle_capability,
    validate_drizzle_request,
    verify_drizzle_result,
)


IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def write_image(path: Path, data: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.asarray(data), overwrite=False)
    return path


class NumpyPointAccumulator:
    """Small independent point-kernel oracle with the STScI array protocol."""

    def __init__(
        self,
        *,
        out_shape: tuple[int, int],
        kernel: str,
        fillval: str,
        disable_ctx: bool = True,
    ) -> None:
        assert kernel == "point"
        assert fillval == "NaN"
        assert disable_ctx is True
        self._weighted_sum = np.zeros(out_shape, dtype=np.float64)
        self._weight = np.zeros(out_shape, dtype=np.float32)
        self.call_shapes: list[tuple[int, int]] = []

    @property
    def out_wht(self) -> np.ndarray:
        return self._weight

    @property
    def out_img(self) -> np.ndarray:
        output = np.full(self._weight.shape, np.nan, dtype=np.float32)
        covered = self._weight > 0
        output[covered] = (self._weighted_sum[covered] / self._weight[covered]).astype(
            np.float32
        )
        return output

    def add_image(
        self,
        data: np.ndarray,
        exptime: float,
        pixmap: np.ndarray,
        *,
        weight_map: np.ndarray,
        wht_scale: float,
        pixfrac: float,
        pixel_scale_ratio: float,
        in_units: str,
    ) -> None:
        assert exptime > 0
        assert 0.0 < pixfrac <= 1.0
        assert pixel_scale_ratio > 0
        assert in_units == "cps"
        assert pixmap.shape == (*data.shape, 2)
        self.call_shapes.append(data.shape)
        x = np.floor(pixmap[..., 0] + 0.5).astype(np.int64)
        y = np.floor(pixmap[..., 1] + 0.5).astype(np.int64)
        weights = np.asarray(weight_map, dtype=np.float64) * float(wht_scale)
        valid = (
            np.isfinite(data)
            & np.isfinite(pixmap[..., 0])
            & np.isfinite(pixmap[..., 1])
            & (weights > 0)
            & (x >= 0)
            & (x < self._weight.shape[1])
            & (y >= 0)
            & (y < self._weight.shape[0])
        )
        np.add.at(
            self._weighted_sum, (y[valid], x[valid]), data[valid] * weights[valid]
        )
        np.add.at(self._weight, (y[valid], x[valid]), weights[valid].astype(np.float32))


class FailingAccumulator(NumpyPointAccumulator):
    def add_image(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected numerical failure")


def provider(
    accumulator: type[NumpyPointAccumulator] = NumpyPointAccumulator,
    created: list[NumpyPointAccumulator] | None = None,
) -> DrizzleProvider:
    def factory(**kwargs: Any) -> NumpyPointAccumulator:
        instance = accumulator(**kwargs)
        if created is not None:
            created.append(instance)
        return instance

    return DrizzleProvider("numpy-test-oracle", "test-v1", factory)


def request(
    tmp_path: Path,
    frames: tuple[DrizzleFrameInput, ...],
    *,
    output_shape: tuple[int, int],
    scale: int = 1,
    tile_rows: int = 3,
) -> DrizzleExecutionRequest:
    return DrizzleExecutionRequest(
        frames=frames,
        output_path=str(tmp_path / "result" / "master-drizzle.fits"),
        receipt_path=str(tmp_path / "result" / "master-drizzle.receipt.json"),
        output_shape=output_shape,
        scale=scale,
        pixfrac=1.0,
        kernel="point",
        tile_rows=tile_rows,
        minimum_distinct_dither_phases=1,
        minimum_dither_span_pixels=0.0,
        median_fwhm_native_pixels=1.5 if scale > 1 else None,
    )


def test_missing_optional_stsci_dependency_is_an_explicit_false_capability() -> None:
    def missing(_: str) -> object:
        raise ModuleNotFoundError("drizzle")

    capability = stsci_drizzle_capability(missing)

    assert capability.available is False
    assert capability.execution_ready is False
    assert capability.backend_id == "stsci-drizzle-cpu"
    assert "unavailable" in (capability.reason or "")
    assert capability.serializable()["mappingConvention"] == "OUTPUT_TO_INPUT"


def test_real_stsci_array_adapter_when_optional_dependency_is_present(
    tmp_path: Path,
) -> None:
    pytest.importorskip("drizzle.resample")
    values = np.array([[1, 2], [3, 4]], np.float32)
    science = write_image(tmp_path / "input" / "science.fits", values)
    drizzle_request = request(
        tmp_path,
        (DrizzleFrameInput(str(science), output_to_input_projective=IDENTITY),),
        output_shape=(2, 2),
        tile_rows=2,
    )

    result = execute_drizzle(drizzle_request)

    assert result.completed is True
    assert result.backend_id == "stsci-drizzle-cpu"
    with fits.open(result.output_path) as hdul:
        np.testing.assert_allclose(hdul["SCI"].data, values, atol=1e-6)
        np.testing.assert_allclose(hdul["WHT"].data, 1.0, atol=1e-6)


def test_projective_inversion_is_exact_for_two_times_geometry() -> None:
    output_to_input = (
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 1.0),
    )

    pixmap = projective_input_to_output_pixmap(output_to_input, (2, 3))

    np.testing.assert_allclose(pixmap[..., 0], [[0.0, 2.0, 4.0], [0.0, 2.0, 4.0]])
    np.testing.assert_allclose(pixmap[..., 1], [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])


def test_fake_backend_executes_weighted_masked_tiled_drizzle_and_receipts(
    tmp_path: Path,
) -> None:
    first = write_image(
        tmp_path / "input" / "first.fits", np.array([[1, 2], [3, 4]], np.float32)
    )
    second = write_image(
        tmp_path / "input" / "second.fits", np.array([[3, 4], [5, 6]], np.float32)
    )
    second_weight = write_image(
        tmp_path / "input" / "second-weight.fits", np.full((2, 2), 3.0, np.float32)
    )
    second_rejection = write_image(
        tmp_path / "input" / "second-reject.fits", np.array([[0, 1], [0, 0]], np.uint8)
    )
    frames = (
        DrizzleFrameInput(str(first), output_to_input_projective=IDENTITY),
        DrizzleFrameInput(
            str(second),
            output_to_input_projective=IDENTITY,
            weight_path=str(second_weight),
            rejection_mask_path=str(second_rejection),
        ),
    )
    before = {
        path: path.read_bytes()
        for path in (first, second, second_weight, second_rejection)
    }
    before_modes = {path: path.stat().st_mode for path in before}
    os.chmod(first, 0o400)
    before_modes[first] = first.stat().st_mode
    created: list[NumpyPointAccumulator] = []

    result = execute_drizzle(
        request(tmp_path, frames, output_shape=(2, 2), tile_rows=2),
        provider=provider(created=created),
    )

    assert result.completed is True
    assert result.code == "DRIZZLE_SUCCEEDED"
    assert result.output_path is not None
    assert result.receipt_path is not None
    assert len(created) == 1
    assert created[0].call_shapes == [(2, 2), (2, 2)]
    with fits.open(result.output_path) as hdul:
        assert [hdu.name for hdu in hdul] == ["SCI", "WHT", "COVERAGE"]
        np.testing.assert_allclose(hdul["SCI"].data, [[2.5, 2.0], [4.5, 5.5]])
        np.testing.assert_allclose(hdul["WHT"].data, [[4.0, 1.0], [4.0, 4.0]])
        np.testing.assert_array_equal(hdul["COVERAGE"].data, [[2, 1], [2, 2]])
        assert hdul[0].header["DRZSCALE"] == 1
        assert hdul[0].header["DRIZKERN"] == "point"
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["status"] == "succeeded"
    assert receipt["statistics"]["nullPixels"] == 0
    assert receipt["statistics"]["coveragePercentiles"]["p50"] == 2.0
    assert receipt["statistics"]["rejectionMasksProvided"] == 1
    assert receipt["statistics"]["rejectionMaskPixels"] == 1
    assert receipt["inputs"][0]["rejection"]["maskApplied"] is False
    assert receipt["inputs"][0]["rejection"]["maskRejectedPixels"] == 0
    assert receipt["inputs"][1]["rejection"]["maskApplied"] is True
    assert receipt["inputs"][1]["rejection"]["maskRejectedPixels"] == 1
    assert receipt["execution"]["sourceMutation"] is False
    assert (
        receipt["artifact"]["sha256"]
        == hashlib.sha256(Path(result.output_path).read_bytes()).hexdigest()
    )
    assert (
        result.receipt_sha256
        == hashlib.sha256(Path(result.receipt_path).read_bytes()).hexdigest()
    )
    for path, content in before.items():
        assert path.read_bytes() == content
        assert path.stat().st_mode == before_modes[path]


def test_dense_output_to_input_pixmap_is_inverted_with_residual_gate(
    tmp_path: Path,
) -> None:
    science_values = np.arange(9, dtype=np.float32).reshape(3, 3)
    science = write_image(tmp_path / "input" / "science.fits", science_values)
    y, x = np.indices((3, 3), dtype=np.float64)
    dense = write_image(tmp_path / "input" / "o2i.fits", np.dstack([x, y]))
    frame = DrizzleFrameInput(str(science), output_to_input_pixmap_path=str(dense))

    result = execute_drizzle(
        request(tmp_path, (frame,), output_shape=(3, 3), tile_rows=3),
        provider=provider(),
    )

    assert result.completed is True
    with fits.open(result.output_path) as hdul:
        np.testing.assert_allclose(hdul["SCI"].data, science_values)
        np.testing.assert_allclose(hdul["WHT"].data, 1.0)
        np.testing.assert_array_equal(hdul["COVERAGE"].data, 1)
    assert result.receipt is not None
    assert result.receipt["inputs"][0]["mappingKind"] == "DENSE_OUTPUT_TO_INPUT"


def test_row_chunks_are_bounded_and_never_emit_a_degenerate_one_row_tile(
    tmp_path: Path,
) -> None:
    values = np.arange(10, dtype=np.float32).reshape(5, 2)
    science = write_image(tmp_path / "input" / "science.fits", values)
    created: list[NumpyPointAccumulator] = []
    drizzle_request = request(
        tmp_path,
        (DrizzleFrameInput(str(science), output_to_input_projective=IDENTITY),),
        output_shape=(5, 2),
        tile_rows=3,
    )

    result = execute_drizzle(drizzle_request, provider=provider(created=created))

    assert result.completed is True
    assert created[0].call_shapes == [(3, 2), (2, 2)]
    with fits.open(result.output_path) as hdul:
        np.testing.assert_allclose(hdul["SCI"].data, values)


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"scale": True}, "SCALE_UNSUPPORTED"),
        ({"scale": 4}, "SCALE_UNSUPPORTED"),
        ({"pixfrac": 0.0}, "PIXFRAC_INVALID"),
        ({"kernel": "magic"}, "KERNEL_UNSUPPORTED"),
        ({"output_shape": (0, 2)}, "INVALID_REQUEST"),
        ({"tile_rows": 5000}, "TILE_LIMIT_INVALID"),
        ({"minimum_coverage_fraction": 0.89}, "COVERAGE_POLICY_UNSAFE"),
        ({"maximum_null_fraction": 0.11}, "COVERAGE_POLICY_UNSAFE"),
    ],
)
def test_invalid_scientific_or_resource_options_fail_before_output(
    tmp_path: Path,
    changes: dict[str, Any],
    code: str,
) -> None:
    science = write_image(
        tmp_path / "input" / "science.fits", np.ones((2, 2), np.float32)
    )
    base: dict[str, Any] = {
        "frames": (
            DrizzleFrameInput(str(science), output_to_input_projective=IDENTITY),
        ),
        "output_path": str(tmp_path / "result" / "master.fits"),
        "receipt_path": str(tmp_path / "result" / "master.json"),
        "output_shape": (2, 2),
        "scale": 1,
        "pixfrac": 1.0,
        "kernel": "point",
        "tile_rows": 2,
        "minimum_distinct_dither_phases": 1,
        "minimum_dither_span_pixels": 0.0,
    }
    base.update(changes)
    drizzle_request = DrizzleExecutionRequest(**base)

    result = execute_drizzle(drizzle_request, provider=provider())

    assert result.completed is False
    assert result.code == code
    assert result.output_path is None
    assert not Path(base["receipt_path"]).exists()


def test_backend_failure_leaves_no_output_or_receipt(tmp_path: Path) -> None:
    science = write_image(
        tmp_path / "input" / "science.fits", np.ones((2, 2), np.float32)
    )
    drizzle_request = request(
        tmp_path,
        (DrizzleFrameInput(str(science), output_to_input_projective=IDENTITY),),
        output_shape=(2, 2),
    )

    result = execute_drizzle(drizzle_request, provider=provider(FailingAccumulator))

    assert result.completed is False
    assert result.code == "BACKEND_EXECUTION_FAILED"
    assert result.output_path is None
    assert not Path(drizzle_request.output_path).exists()
    assert not Path(drizzle_request.receipt_path).exists()


def test_existing_output_is_never_replaced_or_reported_as_success(
    tmp_path: Path,
) -> None:
    science = write_image(
        tmp_path / "input" / "science.fits", np.ones((2, 2), np.float32)
    )
    drizzle_request = request(
        tmp_path,
        (DrizzleFrameInput(str(science), output_to_input_projective=IDENTITY),),
        output_shape=(2, 2),
    )
    output = Path(drizzle_request.output_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(b"user-owned sentinel")

    result = execute_drizzle(drizzle_request, provider=provider())

    assert result.completed is False
    assert result.code == "OUTPUT_EXISTS"
    assert output.read_bytes() == b"user-owned sentinel"
    assert not Path(drizzle_request.receipt_path).exists()


def test_atomic_pair_rollback_preserves_a_racing_user_receipt(tmp_path: Path) -> None:
    staged_output = tmp_path / "staged.fits"
    staged_receipt = tmp_path / "staged.json"
    output = tmp_path / "published.fits"
    receipt = tmp_path / "published.json"
    staged_output.write_bytes(b"new output")
    staged_receipt.write_bytes(b"new receipt")
    receipt.write_bytes(b"user-owned receipt")

    with pytest.raises(DrizzleExecutionError) as raised:
        drizzle_execution._publish_new_pair(
            staged_output, output, staged_receipt, receipt
        )

    assert raised.value.code == "OUTPUT_EXISTS"
    assert not output.exists()
    assert receipt.read_bytes() == b"user-owned receipt"


def test_directory_sync_failure_rolls_back_both_new_publications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged_output = tmp_path / "staged.fits"
    staged_receipt = tmp_path / "staged.json"
    output = tmp_path / "published.fits"
    receipt = tmp_path / "published.json"
    staged_output.write_bytes(b"new output")
    staged_receipt.write_bytes(b"new receipt")

    def fail_sync(_: Path) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(drizzle_execution, "_fsync_directory", fail_sync)

    with pytest.raises(DrizzleExecutionError) as raised:
        drizzle_execution._publish_new_pair(
            staged_output, output, staged_receipt, receipt
        )

    assert raised.value.code == "ATOMIC_PUBLICATION_FAILED"
    assert not output.exists()
    assert not receipt.exists()


def test_execution_uses_identity_bound_target_if_source_symlink_changes(
    tmp_path: Path,
) -> None:
    good_values = np.array([[1, 2], [3, 4]], np.float32)
    good = write_image(tmp_path / "input" / "good.fits", good_values)
    bad = write_image(tmp_path / "input" / "bad.fits", np.full((2, 2), 99, np.float32))
    source_link = tmp_path / "input" / "current.fits"
    source_link.symlink_to(good.name)

    def swapping_factory(**kwargs: Any) -> NumpyPointAccumulator:
        source_link.unlink()
        source_link.symlink_to(bad.name)
        return NumpyPointAccumulator(**kwargs)

    drizzle_request = request(
        tmp_path,
        (DrizzleFrameInput(str(source_link), output_to_input_projective=IDENTITY),),
        output_shape=(2, 2),
    )

    result = execute_drizzle(
        drizzle_request,
        provider=DrizzleProvider("symlink-test", "test-v1", swapping_factory),
    )

    assert result.completed is True
    with fits.open(result.output_path) as hdul:
        np.testing.assert_allclose(hdul["SCI"].data, good_values)
    assert result.receipt is not None
    assert result.receipt["inputs"][0]["calibrated"]["path"] == str(good.resolve())


def test_scale_mismatch_and_singular_transform_fail_closed(tmp_path: Path) -> None:
    science = write_image(
        tmp_path / "input" / "science.fits", np.ones((2, 2), np.float32)
    )
    identity_at_2x = request(
        tmp_path,
        (DrizzleFrameInput(str(science), output_to_input_projective=IDENTITY),),
        output_shape=(4, 4),
        scale=2,
    )
    singular_frame = DrizzleFrameInput(
        str(science),
        output_to_input_projective=(
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    )
    singular = request(
        tmp_path / "singular",
        (singular_frame,),
        output_shape=(2, 2),
    )

    mismatch_result = execute_drizzle(identity_at_2x, provider=provider())
    singular_result = execute_drizzle(singular, provider=provider())

    assert mismatch_result.completed is False
    assert mismatch_result.code == "MAPPING_SCALE_MISMATCH"
    assert singular_result.completed is False
    assert singular_result.code == "TRANSFORM_SINGULAR"


def test_validation_rejects_different_publication_directories(tmp_path: Path) -> None:
    science = write_image(
        tmp_path / "input" / "science.fits", np.ones((2, 2), np.float32)
    )
    drizzle_request = DrizzleExecutionRequest(
        frames=(DrizzleFrameInput(str(science), output_to_input_projective=IDENTITY),),
        output_path=str(tmp_path / "one" / "master.fits"),
        receipt_path=str(tmp_path / "two" / "receipt.json"),
        output_shape=(2, 2),
        scale=1,
        pixfrac=1.0,
        kernel="point",
        minimum_distinct_dither_phases=1,
        minimum_dither_span_pixels=0.0,
    )

    with pytest.raises(Exception, match="share one parent"):
        validate_drizzle_request(drizzle_request)


def test_default_science_gate_rejects_three_frames_without_subpixel_dither(
    tmp_path: Path,
) -> None:
    frames = tuple(
        DrizzleFrameInput(
            str(
                write_image(
                    tmp_path / "input" / f"science-{index}.fits",
                    np.ones((4, 4), np.float32),
                )
            ),
            output_to_input_projective=IDENTITY,
        )
        for index in range(3)
    )
    drizzle_request = DrizzleExecutionRequest(
        frames=frames,
        output_path=str(tmp_path / "result" / "master.fits"),
        receipt_path=str(tmp_path / "result" / "receipt.json"),
        output_shape=(4, 4),
        scale=1,
        pixfrac=1.0,
        kernel="point",
    )

    result = execute_drizzle(drizzle_request, provider=provider())

    assert result.completed is False
    assert result.code == "DITHER_PHASES_INSUFFICIENT"
    assert not Path(drizzle_request.output_path).exists()
    assert not Path(drizzle_request.receipt_path).exists()


def test_twox_output_with_only_quarter_coverage_is_not_published(tmp_path: Path) -> None:
    science = write_image(
        tmp_path / "input" / "science.fits", np.ones((4, 4), np.float32)
    )
    output_to_input_at_2x = (
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 1.0),
    )
    drizzle_request = request(
        tmp_path,
        (
            DrizzleFrameInput(
                str(science), output_to_input_projective=output_to_input_at_2x
            ),
        ),
        output_shape=(8, 8),
        scale=2,
        tile_rows=4,
    )

    result = execute_drizzle(drizzle_request, provider=provider())

    assert result.completed is False
    assert result.code == "COVERAGE_BELOW_MINIMUM"
    assert not Path(drizzle_request.output_path).exists()
    assert not Path(drizzle_request.receipt_path).exists()


def test_verifier_rejects_changed_coverage_artifact_and_changed_receipt(
    tmp_path: Path,
) -> None:
    science = write_image(
        tmp_path / "input" / "science.fits", np.arange(16, dtype=np.float32).reshape(4, 4)
    )
    first_request = request(
        tmp_path / "coverage-tamper",
        (DrizzleFrameInput(str(science), output_to_input_projective=IDENTITY),),
        output_shape=(4, 4),
        tile_rows=4,
    )
    first_result = execute_drizzle(first_request, provider=provider())
    assert first_result.completed is True
    verify_drizzle_result(first_request, first_result)
    with fits.open(first_result.output_path, mode="update", memmap=False) as hdul:
        hdul["COVERAGE"].data[0, 0] = 0
        hdul.flush()
    with pytest.raises(DrizzleExecutionError) as coverage_error:
        verify_drizzle_result(first_request, first_result)
    assert coverage_error.value.code == "ARTIFACT_IDENTITY_MISMATCH"

    second_request = request(
        tmp_path / "receipt-tamper",
        (DrizzleFrameInput(str(science), output_to_input_projective=IDENTITY),),
        output_shape=(4, 4),
        tile_rows=4,
    )
    second_result = execute_drizzle(second_request, provider=provider())
    assert second_result.completed is True
    receipt_path = Path(second_result.receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["statistics"]["coveredPixels"] -= 1
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(DrizzleExecutionError) as receipt_error:
        verify_drizzle_result(second_request, second_result)
    assert receipt_error.value.code == "RECEIPT_IDENTITY_MISMATCH"

    third_request = request(
        tmp_path / "resigned-receipt-tamper",
        (DrizzleFrameInput(str(science), output_to_input_projective=IDENTITY),),
        output_shape=(4, 4),
        tile_rows=4,
    )
    third_result = execute_drizzle(third_request, provider=provider())
    assert third_result.completed is True
    forged = dict(third_result.receipt or {})
    forged["scienceGate"] = dict(forged["scienceGate"])
    forged["scienceGate"]["coverage"] = dict(
        forged["scienceGate"]["coverage"]
    )
    forged["scienceGate"]["coverage"]["observedCoverageFraction"] = 0.91
    core = {key: value for key, value in forged.items() if key != "receiptId"}
    canonical_core = (
        json.dumps(core, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    forged["receiptId"] = "sha256:" + hashlib.sha256(canonical_core).hexdigest()
    forged_bytes = (
        json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    Path(third_result.receipt_path).write_bytes(forged_bytes)
    forged_result = replace(
        third_result,
        receipt=forged,
        receipt_sha256=hashlib.sha256(forged_bytes).hexdigest(),
    )
    with pytest.raises(DrizzleExecutionError) as semantic_error:
        verify_drizzle_result(third_request, forged_result)
    assert semantic_error.value.code == "COVERAGE_RECEIPT_MISMATCH"
