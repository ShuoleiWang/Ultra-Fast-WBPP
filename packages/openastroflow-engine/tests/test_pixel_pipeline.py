from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

from astropy.io import fits
import numpy as np
import pytest

from openastroflow_engine.calibration import CalibrationError, IntegrationParameters
from openastroflow_engine.global_normalization import (
    GlobalNormalizationParameters,
    StellarScaleHint,
)
from openastroflow_engine.path_budget import STAGING_SUFFIX
from openastroflow_engine.pixel_pipeline import (
    AffineTransform,
    MasterMetadataOverride,
    PipelineParameters,
    run_portable_pipeline,
)


def _write_frame(
    path: Path,
    role: str,
    data: np.ndarray,
    *,
    filter_name: str = "R",
    exposure: float = 30.0,
    float_storage: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["IMAGETYP"] = role
    header["FILTER"] = filter_name
    header["OBJECT"] = "SYNTHETIC"
    header["INSTRUME"] = "SYNTH-CAM"
    header["EXPTIME"] = exposure
    header["GAIN"] = 100
    header["OFFSET"] = 50
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    header["READOUTM"] = "MODE-1"
    header["BAYERPAT"] = "NONE"
    header["CCD-TEMP"] = -10.0
    # Raw camera fixtures use an auditable unsigned integer storage domain.
    # Float supplied masters are enabled only in the explicit hash-bound domain
    # tests below.
    pixels = (
        np.asarray(data, dtype=np.float32)
        if float_storage
        else np.asarray(np.rint(data), dtype=np.uint16)
    )
    fits.writeto(
        path,
        pixels,
        header,
        overwrite=False,
    )
    return path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _dataset(
    root: Path,
) -> tuple[
    list[Path], list[Path], list[Path], list[Path], np.ndarray, np.ndarray
]:
    height, width = 12, 16
    response = np.broadcast_to(
        np.linspace(0.8, 1.2, width, dtype=np.float32), (height, width)
    ).copy()
    y, x = np.mgrid[:height, :width]
    signal = (500.0 + 3.0 * x + 2.0 * y).astype(np.float32)
    signal[5, 8] += 700
    biases = [
        _write_frame(
            root / "bias" / f"bias_{index}.fits",
            "Bias",
            np.full((height, width), value),
            exposure=0.001,
        )
        for index, value in enumerate((99.0, 100.0, 101.0))
    ]
    darks = [
        _write_frame(root / "dark" / f"dark_{index}.fits", "Dark", np.full((height, width), value))
        for index, value in enumerate((119.0, 120.0, 121.0))
    ]
    flats = [
        _write_frame(
            root / "flat" / f"flat_{index}.fits",
            "Flat",
            100.0 + 1000.0 * response,
            exposure=2.0,
        )
        for index in range(3)
    ]
    lights: list[Path] = []
    for index in range(5):
        raw = 120.0 + signal * response
        raw = raw.copy()
        if index == 4:
            raw[2, 3] += 8000.0 * response[2, 3]
        lights.append(
            _write_frame(root / "light" / f"light_{index}.fits", "Light", raw)
        )
    # Return the exact noiseless target represented by the integer camera
    # samples, rather than the pre-quantization analytic arrays.
    stored_flat = np.rint(100.0 + 1000.0 * response).astype(np.float32)
    effective_response = (stored_flat - 100.0) / np.float32(1000.0)
    stored_light = np.rint(120.0 + signal * response).astype(np.float32)
    effective_signal = (stored_light - 120.0) / effective_response
    return biases, darks, flats, lights, effective_signal, effective_response


def _parameters() -> PipelineParameters:
    return PipelineParameters(
        integration=IntegrationParameters(
            sigma_clip=4.0,
            minimum_rejection_frames=3,
            max_memory_bytes=4096,
            max_statistics_samples=100,
        ),
        registration_memory_bytes=4096,
        preview_max_long_edge=64,
        global_normalization=GlobalNormalizationParameters(enabled=False),
    )


def test_parallel_warps_preserve_fits_and_ordered_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openastroflow_engine.pixel_pipeline as pipeline

    biases, darks, flats, lights, _signal, _response = _dataset(tmp_path / "raw")
    tuning = pipeline.select_execution_tuning(pipeline.detect_hardware())
    parameters = replace(
        _parameters(),
        registration_memory_bytes=64 * 1024,
        ordinary_integration_backend="portable-cpu",
        auto_crop=False,
    )
    transforms = {
        str(path): AffineTransform.from_value(
            ((1.0, 0.0, 0.15 * index), (0.0, 1.0, 0.09 * index), (0.0, 0.0, 1.0))
        )
        for index, path in enumerate(lights)
    }
    receipts = []
    for workers in (1, 4):
        monkeypatch.setattr(
            pipeline, "select_execution_tuning",
            lambda _hardware, workers=workers: replace(tuning, cpu_workers=workers, kernel_threads=workers),
        )
        result = run_portable_pipeline(
            bias_files=biases, dark_files=darks, flat_files=flats, light_files=lights,
            output_directory=tmp_path / f"workers-{workers}",
            parameters=parameters, transforms=transforms,
        )
        receipt = json.loads(Path(result.receipt_path).read_text())
        execution = receipt["statistics"]["registration"]
        assert execution["cpuWorkersUsed"] == workers
        assert execution["perWorkerMemoryBudgetBytes"] * workers <= parameters.registration_memory_bytes
        # The Lights of the final submission round take the idle workers' CPU
        # share; the identical outputs below show the thread count is immaterial.
        light_count = len(lights)
        rounds = -(-light_count // workers)
        assert execution["tailLights"] == light_count - (rounds - 1) * workers
        assert execution["tailNativeThreads"] == max(
            execution["nativeThreadsPerWorker"], workers // execution["tailLights"]
        )
        receipts.append(receipt)

    # Full artifact hashes cover registered/master pixels, FITS headers, maps
    # and previews. Provenance order stays CAL/REGISTERED for each input frame.
    assert receipts[0]["outputs"] == receipts[1]["outputs"]

    def without_execution_tuning(registration: dict) -> dict:
        # Threads and tile rows reflect the per-worker memory/CPU budget.
        # Artifact hashes above still require identical scientific outputs.
        stripped = {}
        for path, entry in registration.items():
            entry = dict(entry)
            if isinstance(entry.get("execution"), dict):
                if "tileRows" in entry["execution"]:
                    assert entry["execution"]["tileRows"] > 0
                entry["execution"] = {
                    key: value for key, value in entry["execution"].items()
                    if key not in {"nativeThreads", "tileRows"}
                }
            stripped[path] = entry
        return stripped

    assert without_execution_tuning(receipts[0]["registration"]) == without_execution_tuning(receipts[1]["registration"])
    assert receipts[0]["inputs"] == receipts[1]["inputs"]
    light_kinds = [item["kind"] for item in receipts[1]["outputs"] if item["kind"] in {"CALIBRATED_LIGHT", "REGISTERED_LIGHT"}]
    assert light_kinds == ["CALIBRATED_LIGHT", "REGISTERED_LIGHT"] * len(lights)


@pytest.mark.parametrize(('budget_rows', 'cpu_workers', 'expected_workers'),
                         [(3, 8, 3), (10, 12, 8), (10, 2, 2)])
def test_parallel_warps_share_budget_and_keep_result_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    budget_rows: int, cpu_workers: int, expected_workers: int,
) -> None:
    import openastroflow_engine.pixel_pipeline as pipeline
    from openastroflow_engine.calibration import PixelStatistics, read_frame_info

    source = _write_frame(tmp_path / "source.fits", "Light", np.ones((12, 16)))
    info = read_frame_info(source)
    transform = AffineTransform.from_value(((1, 0, 0.2), (0, 1, 0.3), (0, 0, 1)))
    jobs = [pipeline._RegistrationJob(source, tmp_path / f"{i}.fits", transform, info) for i in range(expected_workers * 2)]
    one_row = info.shape[1] * 192
    budget = one_row * budget_rows + 2
    barrier = threading.Barrier(expected_workers)
    lock = threading.Lock()
    observed = []

    def register(_source, destination, *_args, max_memory_bytes, **_kwargs):
        with lock:
            observed.append((threading.get_ident(), max_memory_bytes))
        barrier.wait(timeout=5)
        return PixelStatistics(1, 0, 0.0, 1.0, float(destination.stem))

    monkeypatch.setattr(pipeline, "_register_frame", register)
    results = pipeline._register_frames(jobs, max_memory_bytes=budget, resampler="lanczos-3-clamped", cpu_workers=cpu_workers)
    assert [item.mean for item in results] == list(range(len(jobs)))
    assert len({thread for thread, _memory in observed}) == expected_workers
    assert {memory for _thread, memory in observed} == {budget // expected_workers}


def test_parallel_warp_failure_waits_for_active_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openastroflow_engine.pixel_pipeline as pipeline
    from openastroflow_engine.calibration import PixelStatistics, read_frame_info

    source = _write_frame(tmp_path / "source.fits", "Light", np.ones((12, 16)))
    info = read_frame_info(source)
    jobs = [pipeline._RegistrationJob(source, tmp_path / f"{i}.fits", AffineTransform.identity(), info) for i in range(2)]
    barrier = threading.Barrier(2)
    failed = threading.Event()
    release_writer = threading.Event()
    writer_finished = threading.Event()

    def register(_source, destination, *_args, **_kwargs):
        barrier.wait(timeout=5)
        if destination.stem == "0":
            failed.set()
            raise CalibrationError("TEST_WARP_FAILURE", "injected warp failure")
        assert release_writer.wait(timeout=5)
        writer_finished.set()
        return PixelStatistics(1, 0, 0.0, 1.0, 0.5)

    monkeypatch.setattr(pipeline, "_register_frame", register)
    with ThreadPoolExecutor(max_workers=1) as caller:
        pending = caller.submit(pipeline._register_frames, jobs, max_memory_bytes=4096, resampler="bilinear", cpu_workers=2)
        try:
            assert failed.wait(timeout=5)
            assert not pending.done()
        finally:
            release_writer.set()
        with pytest.raises(CalibrationError, match="TEST_WARP_FAILURE"):
            pending.result(timeout=5)
    assert writer_finished.is_set()


def _dark_override(path: Path, *, bias_included: bool) -> MasterMetadataOverride:
    return MasterMetadataOverride(
        source_sha256=_sha(path),
        camera="SYNTH-CAM",
        gain=100,
        offset=50,
        binning_x=1,
        binning_y=1,
        filter_name="R",
        cfa_pattern="NONE",
        readout_mode="MODE-1",
        temperature_celsius=-10.0,
        exposure_seconds=30.0,
        bias_included=bias_included,
    )


def test_raw_to_unsolved_master_is_numerically_correct_and_audited(
    tmp_path: Path,
) -> None:
    biases, darks, flats, lights, signal, response = _dataset(tmp_path / "raw")
    source_hashes = {path: _sha(path) for path in (*biases, *darks, *flats, *lights)}
    output = tmp_path / "result"

    result = run_portable_pipeline(
        bias_files=biases,
        dark_files=darks,
        flat_files=flats,
        light_files=lights,
        output_directory=output,
        parameters=_parameters(),
    )

    assert result.state == "UNSOLVED_WORKING"
    assert Path(result.receipt_path).is_file()
    assert all(Path(path).is_file() for path in result.master_light_paths)
    assert all(Path(path).is_file() for path in result.preview_paths)
    master_bias = output / "masters" / "master_bias.fits"
    master_dark = output / "masters" / "master_dark_30s.fits"
    master_flat = output / "masters" / "master_flat_R.fits"
    master_light = output / "masters" / "master_light_R.fits"
    with fits.open(master_bias, memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, 100.0, atol=2e-5)
    with fits.open(master_dark, memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, 120.0, atol=2e-5)
        assert hdul[0].header["OAFBIAS"] == "INCLUDED"
    with fits.open(master_flat, memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, response, atol=2e-5)
        assert hdul[0].header["OAFBIAS"] == "SUBTRACTED"
    with fits.open(master_light, memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, signal, atol=3e-3)
        header = hdul[0].header
        assert header["OAFSTATE"] == "UNSOLVED_WORKING"
        assert header["OAFWCS"] == "UNSOLVED"
        assert "CTYPE1" not in header
        assert "CRVAL1" not in header
    assert (output / "previews" / "master_light_R.png").is_file()
    assert {path: _sha(path) for path in source_hashes} == source_hashes

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["state"] == "UNSOLVED_WORKING"
    assert receipt["astrometry"]["status"] == "UNSOLVED"
    assert receipt["astrometry"]["wcsValidated"] is False
    assert receipt["drizzle"]["status"] == "NOT_RUN"
    # No staging path (``.<name>.<random>.stage``) may leak into the receipt.
    assert all(STAGING_SUFFIX not in str(value) for value in receipt["statistics"].values())
    for artifact in receipt["outputs"]:
        artifact_path = output / artifact["path"]
        assert artifact_path.is_file()
        assert artifact["sha256"] == _sha(artifact_path)
    master_integration = receipt["statistics"]["integrationGroups"]["R"]["integration"]
    assert master_integration["tileRows"] < signal.shape[0]
    assert master_integration["rejectedSamples"] >= 1
    assert {
        item["resampler"] for item in receipt["registration"].values()
    } == {"identity-exact"}
    assert {
        item["resamplerAlgorithm"] for item in receipt["registration"].values()
    } == {"identity-exact-copy-v1"}


def test_unknown_calibration_profile_metadata_fails_closed(tmp_path: Path) -> None:
    biases, darks, flats, lights, _, _ = _dataset(tmp_path / "raw")
    with fits.open(darks[0], mode="update", memmap=False) as hdul:
        del hdul[0].header["GAIN"]
        hdul.flush()
    with pytest.raises(CalibrationError) as error:
        run_portable_pipeline(
            bias_files=biases,
            dark_files=darks,
            flat_files=flats,
            light_files=lights,
            output_directory=tmp_path / "result",
            parameters=_parameters(),
        )
    assert error.value.code == "CALIBRATION_PROFILE_MISMATCH"


def test_supplied_masters_are_reused_once_and_never_recalibrated(
    tmp_path: Path,
) -> None:
    biases, darks, flats, lights, signal, response = _dataset(tmp_path / "raw")
    masters = tmp_path / "supplied-masters"
    master_bias = _write_frame(
        masters / "master_bias.fits",
        "Master Bias",
        np.full(signal.shape, 100.0),
        exposure=0.001,
    )
    master_dark = _write_frame(
        masters / "master_dark_30s.fits",
        "Master Dark",
        np.full(signal.shape, 120.0),
        exposure=30.0,
    )
    # Deliberately use a non-unit supplied MasterFlat.  The pipeline must
    # normalize it at application time without modifying or calibrating it.
    master_flat = _write_frame(
        masters / "master_flat_R.fits",
        "Master Flat",
        5000.0 * response,
        exposure=2.0,
    )
    source_hashes = {
        path: _sha(path) for path in (master_bias, master_dark, master_flat)
    }
    for path in source_hashes:
        path.chmod(0o444)

    output = tmp_path / "master-reuse"
    run_portable_pipeline(
        master_bias_file=master_bias,
        master_dark_files=(master_dark,),
        master_flat_files=(master_flat,),
        light_files=lights,
        output_directory=output,
        parameters=replace(
            _parameters(),
            master_metadata_overrides=(_dark_override(master_dark, bias_included=True),),
        ),
    )

    with fits.open(output / "masters" / "master_light_R.fits", memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, signal, atol=3e-3)
    assert {path: _sha(path) for path in source_hashes} == source_hashes
    assert not (output / "masters" / "master_bias.fits").exists()
    assert not (output / "masters" / "master_dark_30s.fits").exists()
    assert not (output / "masters" / "master_flat_R.fits").exists()
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    calibration = receipt["statistics"]["calibration"]
    assert calibration["masterBias"]["mode"] == "REUSED_SUPPLIED_MASTER"
    assert calibration["masterDark:30"]["calibrationApplied"] is False
    assert calibration["masterFlat:R"]["calibrationApplied"] is False
    assert {item["role"] for item in receipt["inputs"]} >= {
        "MASTER_BIAS",
        "MASTER_DARK",
        "MASTER_FLAT",
    }


def test_float_code_masters_require_hash_bound_domain_and_are_not_rescaled_twice(
    tmp_path: Path,
) -> None:
    _biases, _darks, _flats, lights, signal, response = _dataset(tmp_path / "raw")
    masters = tmp_path / "float-code-masters"
    master_bias = _write_frame(
        masters / "master_bias.fits",
        "Master Bias",
        np.full(signal.shape, 100.0),
        exposure=0.001,
        float_storage=True,
    )
    master_dark = _write_frame(
        masters / "master_dark.fits",
        "Master Dark",
        np.full(signal.shape, 120.0),
        exposure=30.0,
        float_storage=True,
    )
    master_flat = _write_frame(
        masters / "master_flat.fits",
        "Master Flat",
        5000.0 * response,
        exposure=2.0,
        float_storage=True,
    )

    def override(path: Path, *, bias_included: bool | None) -> MasterMetadataOverride:
        return MasterMetadataOverride(
            source_sha256=_sha(path),
            camera="SYNTH-CAM",
            gain=100,
            offset=50,
            binning_x=1,
            binning_y=1,
            filter_name="R",
            cfa_pattern="NONE",
            readout_mode="MODE-1",
            temperature_celsius=-10,
            exposure_seconds=30.0 if bias_included is not None else 0.001,
            bias_included=bias_included,
            numeric_domain="SENSOR_CODE",
            normalized_unit_scale=65535.0,
        )

    parameters = replace(
        _parameters(),
        master_metadata_overrides=(
            override(master_bias, bias_included=None),
            override(master_dark, bias_included=True),
        ),
    )
    output = tmp_path / "float-code-result"
    run_portable_pipeline(
        master_bias_file=master_bias,
        master_dark_files=(master_dark,),
        master_flat_files=(master_flat,),
        light_files=lights,
        output_directory=output,
        parameters=parameters,
    )

    with fits.open(output / "masters" / "master_light_R.fits", memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, signal, atol=3e-3)
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    calibrated = next(
        item for item in receipt["outputs"] if item["kind"] == "CALIBRATED_LIGHT"
    )
    assert calibrated["details"]["additiveApplicationScale"] == 1.0
    assert calibrated["details"]["biasApplicationScale"] is None
    domains = {item["role"]: item for item in receipt["pixelNumericDomains"]}
    assert domains["MASTER_BIAS"]["authority"] == "CONTENT_BOUND_OVERRIDE"
    assert domains["MASTER_DARK"]["authority"] == "CONTENT_BOUND_OVERRIDE"

    ambiguous_dark = replace(
        override(master_dark, bias_included=True),
        numeric_domain=None,
        normalized_unit_scale=None,
    )
    ambiguous_bias = replace(
        override(master_bias, bias_included=None),
        numeric_domain=None,
        normalized_unit_scale=None,
    )
    with pytest.raises(CalibrationError) as captured:
        run_portable_pipeline(
            master_bias_file=master_bias,
            master_dark_files=(master_dark,),
            master_flat_files=(master_flat,),
            light_files=lights,
            output_directory=tmp_path / "ambiguous-float-code-result",
            parameters=replace(
                _parameters(),
                master_metadata_overrides=(ambiguous_bias, ambiguous_dark),
            ),
        )
    assert captured.value.code == "CALIBRATION_NUMERIC_DOMAIN_AMBIGUOUS"


def test_supplied_bias_subtracted_master_dark_subtracts_bias_and_dark_once(
    tmp_path: Path,
) -> None:
    _biases, _darks, _flats, lights, signal, response = _dataset(tmp_path / "raw")
    masters = tmp_path / "bias-subtracted-masters"
    master_bias = _write_frame(
        masters / "master_bias.fits",
        "Master Bias",
        np.full(signal.shape, 100.0),
        exposure=0.001,
    )
    master_dark = _write_frame(
        masters / "master_dark_30s.fits",
        "Master Dark",
        np.full(signal.shape, 20.0),
        exposure=30.0,
    )
    master_flat = _write_frame(
        masters / "master_flat_R.fits",
        "Master Flat",
        5000.0 * response,
        exposure=2.0,
    )
    output = tmp_path / "bias-subtracted-result"

    run_portable_pipeline(
        master_bias_file=master_bias,
        master_dark_files=(master_dark,),
        master_flat_files=(master_flat,),
        light_files=lights,
        output_directory=output,
        parameters=replace(
            _parameters(),
            master_metadata_overrides=(_dark_override(master_dark, bias_included=False),),
        ),
    )

    with fits.open(output / "masters" / "master_light_R.fits", memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, signal, atol=3e-3)
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["statistics"]["calibration"]["masterDark:30"]["biasIncluded"] is False


def test_supplied_master_dark_without_hash_bound_bias_semantics_fails_closed(
    tmp_path: Path,
) -> None:
    _biases, _darks, _flats, lights, signal, response = _dataset(tmp_path / "raw")
    masters = tmp_path / "missing-semantics-masters"
    master_bias = _write_frame(
        masters / "master_bias.fits", "Master Bias", np.full(signal.shape, 100.0), exposure=0.001
    )
    master_dark = _write_frame(
        masters / "master_dark.fits", "Master Dark", np.full(signal.shape, 120.0), exposure=30.0
    )
    master_flat = _write_frame(
        masters / "master_flat.fits", "Master Flat", 5000.0 * response, exposure=2.0
    )

    with pytest.raises(CalibrationError) as captured:
        run_portable_pipeline(
            master_bias_file=master_bias,
            master_dark_files=(master_dark,),
            master_flat_files=(master_flat,),
            light_files=lights,
            output_directory=tmp_path / "missing-semantics-output",
            parameters=_parameters(),
        )
    assert captured.value.code == "MASTER_DARK_BIAS_SEMANTICS_REQUIRED"


def test_registration_quality_weights_drive_pixels_and_emit_actual_maps(
    tmp_path: Path,
) -> None:
    biases, darks, flats, lights, signal, response = _dataset(tmp_path / "raw")
    selected = lights[:3]
    offsets = (0.0, 100.0, 200.0)
    for path, offset in zip(selected, offsets, strict=True):
        with fits.open(path, mode="update", memmap=False) as hdul:
            hdul[0].data[:] = 120.0 + (signal + offset) * response
            hdul.flush()
    parameters = _parameters()
    parameters = PipelineParameters(
        integration=IntegrationParameters(
            sigma_clip=4.0,
            minimum_rejection_frames=4,
            max_memory_bytes=4096,
            max_statistics_samples=100,
        ),
        registration_memory_bytes=parameters.registration_memory_bytes,
        preview_max_long_edge=parameters.preview_max_long_edge,
        ordinary_integration_backend="portable-cpu",
        global_normalization=GlobalNormalizationParameters(enabled=False),
    )
    quality = {
        str(selected[0]): 0.8,
        str(selected[1]): 0.1,
        str(selected[2]): 0.1,
    }
    output = tmp_path / "quality-weighted"

    run_portable_pipeline(
        bias_files=biases,
        dark_files=darks,
        flat_files=flats,
        light_files=selected,
        output_directory=output,
        quality_weights=quality,
        parameters=parameters,
    )

    with fits.open(output / "masters" / "master_light_R.fits", memmap=False) as hdul:
        decoded = []
        for path in selected:
            with fits.open(path, memmap=False) as source_hdul:
                decoded.append(np.asarray(source_hdul[0].data, dtype=np.float32))
        calibrated = [(item - np.float32(120.0)) / response for item in decoded]
    map_root = output / "coverage"
    with fits.open(map_root / "R_acceptedSampleCount.fits", memmap=False) as hdul:
        np.testing.assert_array_equal(hdul[0].data, 3.0)
    with fits.open(map_root / "R_coverageFraction.fits", memmap=False) as hdul:
        np.testing.assert_array_equal(hdul[0].data, 1.0)
    with fits.open(map_root / "R_rejectionCount.fits", memmap=False) as hdul:
        np.testing.assert_array_equal(hdul[0].data, 0.0)
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    integration = receipt["statistics"]["integrationGroups"]["R"]["integration"]
    np.testing.assert_allclose(
        integration["weightComponents"]["registrationQuality"],
        (0.8, 0.1, 0.1),
        atol=1e-12,
    )
    raw_weights = np.asarray(
        integration["weightComponents"]["noise"], dtype=np.float64
    ) * np.asarray((0.8, 0.1, 0.1), dtype=np.float64)
    expected_weights = raw_weights / np.sum(raw_weights)
    np.testing.assert_allclose(integration["weights"], expected_weights, atol=1e-12)
    expected = sum(
        weight * item
        for weight, item in zip(expected_weights, calibrated, strict=True)
    )
    with fits.open(output / "masters" / "master_light_R.fits", memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, expected, atol=5e-3)
    assert set(integration["maps"]) == {
        "acceptedSampleCount",
        "coverageFraction",
        "rejectionCount",
    }


def test_default_global_normalization_aligns_background_and_records_coefficients(
    tmp_path: Path,
) -> None:
    biases, darks, flats, lights, signal, _flat_response = _dataset(tmp_path / "raw")
    selected = lights[:3]
    exact_signal = np.rint(signal).astype(np.uint16)
    for path in flats:
        with fits.open(path, mode="update", memmap=False) as hdul:
            hdul[0].data[:] = 1100
            hdul.flush()
    for path, offset in zip(selected, (0.0, 100.0, 200.0), strict=True):
        with fits.open(path, mode="update", memmap=False) as hdul:
            hdul[0].data[:] = 120 + exact_signal + int(offset)
            hdul.flush()
    quality = {
        str(selected[0]): 0.8,
        str(selected[1]): 0.1,
        str(selected[2]): 0.1,
    }
    parameters = replace(
        _parameters(),
        ordinary_integration_backend="portable-cpu",
        global_normalization=GlobalNormalizationParameters(
            enabled=True,
            maximum_samples=1_024,
            minimum_samples=128,
            upper_quantile=0.90,
        ),
    )
    output = tmp_path / "globally-normalized"
    stellar_hints = {
        str(path): StellarScaleHint(
            source_path=str(path.resolve(strict=True)),
            reference_path=str(selected[0].resolve(strict=True)),
            filter_name="R",
            source_sha256=_sha(path),
            reference_sha256=_sha(selected[0]),
            scale=1.0,
            status=(
                "REFERENCE_IDENTITY"
                if path == selected[0]
                else "STELLAR_SCALE_ACCEPTED"
            ),
            evidence={"acceptedScaleStars": 40},
        )
        for path in selected
    }

    run_portable_pipeline(
        bias_files=biases,
        dark_files=darks,
        flat_files=flats,
        light_files=selected,
        output_directory=output,
        quality_weights=quality,
        stellar_scale_hints=stellar_hints,
        parameters=parameters,
    )

    with fits.open(output / "masters" / "master_light_R.fits", memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, exact_signal, atol=5e-3)
        assert hdul[0].header["OAFNORM"] == "GLOBAL_STELLAR"
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    normalization = receipt["statistics"]["integrationGroups"]["R"][
        "globalNormalization"
    ]
    assert normalization["status"] == "APPLIED"
    assert normalization["evidence"]["referenceIndex"] == 0
    assert [item["mode"] for item in normalization["evidence"]["frames"]] == [
        "REFERENCE_IDENTITY",
        "STELLAR_SCALE_SCALAR_OFFSET",
        "STELLAR_SCALE_SCALAR_OFFSET",
    ]
    np.testing.assert_allclose(
        [item["offset"] for item in normalization["evidence"]["frames"]],
        (0.0, -100.0, -200.0),
        atol=5e-3,
    )


@pytest.mark.parametrize(
    ("changed_field", "expected_code"),
    (
        ("source_sha256", "STELLAR_SCALE_HINT_IDENTITY_MISMATCH"),
        ("filter_name", "STELLAR_SCALE_HINT_FILTER_MISMATCH"),
    ),
)
def test_stellar_scale_hint_content_identity_and_filter_are_fail_closed(
    tmp_path: Path,
    changed_field: str,
    expected_code: str,
) -> None:
    biases, darks, flats, lights, _signal, _response = _dataset(tmp_path / "raw")
    light = lights[0]
    values = {
        "source_path": str(light.resolve(strict=True)),
        "reference_path": str(light.resolve(strict=True)),
        "filter_name": "R",
        "source_sha256": _sha(light),
        "reference_sha256": _sha(light),
        "scale": 1.0,
        "status": "REFERENCE_IDENTITY",
        "evidence": {},
    }
    values[changed_field] = "sha256:" + "0" * 64 if changed_field.endswith("sha256") else "B"
    with pytest.raises(CalibrationError) as captured:
        run_portable_pipeline(
            bias_files=biases,
            dark_files=darks,
            flat_files=flats,
            light_files=(light,),
            output_directory=tmp_path / "rejected-hint",
            stellar_scale_hints={str(light): StellarScaleHint(**values)},
            parameters=replace(
                _parameters(),
                global_normalization=GlobalNormalizationParameters(
                    enabled=True,
                    maximum_samples=1_024,
                    minimum_samples=128,
                    upper_quantile=0.90,
                ),
            ),
        )
    assert captured.value.code == expected_code


def test_raw_flats_use_exact_matched_bias_included_dark_when_available(
    tmp_path: Path,
) -> None:
    biases, darks, flats, lights, signal, response = _dataset(tmp_path / "raw")
    flat_darks = [
        _write_frame(
            tmp_path / "raw" / "dark" / f"flat_dark_{index}.fits",
            "Dark",
            np.full(signal.shape, value),
            exposure=2.0,
        )
        for index, value in enumerate((129.0, 130.0, 131.0))
    ]
    for path in flats:
        with fits.open(path, mode="update", memmap=False) as hdul:
            hdul[0].data[:] = 130.0 + 1000.0 * response
            hdul.flush()
    output = tmp_path / "flat-dark-calibrated"

    run_portable_pipeline(
        bias_files=biases,
        dark_files=(*darks, *flat_darks),
        flat_files=flats,
        light_files=lights,
        output_directory=output,
        parameters=_parameters(),
    )

    with fits.open(output / "masters" / "master_flat_R.fits", memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, response, atol=2e-5)
    with fits.open(output / "masters" / "master_light_R.fits", memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, signal, atol=3e-3)
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    flat_artifact = next(
        item for item in receipt["outputs"] if item["kind"] == "MASTER_FLAT"
    )
    assert {
        item["mode"] for item in flat_artifact["details"]["calibrationSources"]
    } == {"MATCHED_BIAS_INCLUDED_DARK"}


def test_same_filter_mixed_exposures_use_each_dark_and_linear_normalization(
    tmp_path: Path,
) -> None:
    biases, darks_30, flats, lights_30, signal, response = _dataset(
        tmp_path / "raw"
    )
    darks_60 = [
        _write_frame(
            tmp_path / "raw" / "dark" / f"dark_60_{index}.fits",
            "Dark",
            np.full(signal.shape, value),
            exposure=60.0,
        )
        for index, value in enumerate((139.0, 140.0, 141.0))
    ]
    lights_60 = [
        _write_frame(
            tmp_path / "raw" / "light" / f"light_60_{index}.fits",
            "Light",
            140.0 + 2.0 * signal * response,
            exposure=60.0,
        )
        for index in range(3)
    ]
    selected_30 = lights_30[:3]
    output = tmp_path / "mixed-exposure"

    run_portable_pipeline(
        bias_files=biases,
        dark_files=(*darks_30, *darks_60),
        flat_files=flats,
        light_files=(*selected_30, *lights_60),
        output_directory=output,
        parameters=_parameters(),
    )

    with fits.open(output / "masters" / "master_light_R.fits", memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, signal, atol=4e-3)
        assert hdul[0].header["EXPTIME"] == 30.0
        assert hdul[0].header["OAFINTTM"] == 270.0
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    group = receipt["statistics"]["integrationGroups"]["R"]
    assert group["exposureNormalization"] == {
        "method": "LINEAR_REFERENCE_EXPOSURE",
        "referenceSeconds": 30.0,
        "sourceExposureSeconds": [30.0, 60.0],
        "totalIntegrationSeconds": 270.0,
    }
    calibrated = [
        item
        for item in receipt["outputs"]
        if item["kind"] == "CALIBRATED_LIGHT"
    ]
    assert {
        item["details"]["exposureNormalization"]["scale"]
        for item in calibrated
    } == {0.5, 1.0}


def test_dark_temperature_mismatch_fails_closed(tmp_path: Path) -> None:
    biases, darks, flats, lights, _, _ = _dataset(tmp_path / "raw")
    for path in darks:
        fits.setval(path, "CCD-TEMP", value=0.0)
    with pytest.raises(CalibrationError) as error:
        run_portable_pipeline(
            bias_files=biases,
            dark_files=darks,
            flat_files=flats,
            light_files=lights,
            output_directory=tmp_path / "result",
            parameters=_parameters(),
        )
    assert error.value.code == "CALIBRATION_PROFILE_MISMATCH"


def test_affine_translation_uses_common_autocrop(tmp_path: Path) -> None:
    biases, darks, flats, lights, signal, _ = _dataset(tmp_path / "raw")
    translation = AffineTransform(
        ((1.0, 0.0, 2.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    output = tmp_path / "translated"

    run_portable_pipeline(
        bias_files=biases,
        dark_files=darks,
        flat_files=flats,
        light_files=lights,
        output_directory=output,
        transforms={path.name: translation for path in lights},
        parameters=_parameters(),
    )

    with fits.open(output / "masters" / "master_light_R.fits", memmap=False) as hdul:
        assert hdul[0].data.shape == (signal.shape[0] - 4, signal.shape[1] - 4)
        np.testing.assert_allclose(hdul[0].data, signal[2:-2, 2:-2], atol=3e-3)
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    crop = receipt["statistics"]["integrationGroups"]["R"]["crop"]
    assert crop == [2, 4, signal.shape[0] - 2, signal.shape[1]]
    assert receipt["parameters"]["registrationResampler"] == "lanczos-3-clamped"
    assert {
        item["resampler"] for item in receipt["registration"].values()
    } == {"lanczos-3-clamped"}
    assert all(not item["identity"] for item in receipt["registration"].values())


def test_filters_of_one_run_share_one_autocrop_rectangle(tmp_path: Path) -> None:
    """Different dithers per filter must not leave the masters on different grids."""

    biases, darks, flats, lights, signal_r, _ = _dataset(tmp_path / "raw")
    height, width = signal_r.shape
    for index in range(3):
        flats.append(
            _write_frame(
                tmp_path / "raw" / "flat_g" / f"flat_g_{index}.fits",
                "Flat",
                100.0 + 1000.0 * np.ones((height, width), dtype=np.float32),
                filter_name="G",
                exposure=2.0,
            )
        )
    green_lights = [
        _write_frame(
            tmp_path / "raw" / "light_g" / f"light_g_{index}.fits",
            "Light",
            120.0 + signal_r * 0.6,
            filter_name="G",
        )
        for index in range(5)
    ]
    # R Lights are shifted by +2 columns, G Lights by +2 rows: the per-filter
    # valid rectangles differ, but every master must land on their intersection.
    transforms = {
        path.name: AffineTransform(((1.0, 0.0, 2.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        for path in lights
    }
    transforms.update(
        {
            path.name: AffineTransform(((1.0, 0.0, 0.0), (0.0, 1.0, 2.0), (0.0, 0.0, 1.0)))
            for path in green_lights
        }
    )
    output = tmp_path / "shared-crop"

    run_portable_pipeline(
        bias_files=biases,
        dark_files=darks,
        flat_files=flats,
        light_files=lights + green_lights,
        output_directory=output,
        transforms=transforms,
        parameters=_parameters(),
    )

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    groups = receipt["statistics"]["integrationGroups"]
    assert groups["R"]["groupCrop"] == [2, 4, height - 2, width]
    assert groups["G"]["groupCrop"] == [4, 2, height, width - 2]
    shared = [4, 4, height - 2, width - 2]
    assert groups["R"]["crop"] == shared and groups["G"]["crop"] == shared
    assert groups["R"]["cropSharedAcrossFilters"] is True
    with fits.open(output / "masters" / "master_light_R.fits", memmap=False) as red:
        with fits.open(output / "masters" / "master_light_G.fits", memmap=False) as green:
            assert red[0].data.shape == green[0].data.shape == (height - 6, width - 6)
            # Output pixel (r, c) holds signal (r, c-2) for R and (r-2, c) for G.
            np.testing.assert_allclose(red[0].data, signal_r[4:-2, 2:-4], atol=3e-3)
            # The G Lights are stored as integers with a flat of exactly one,
            # so their master equals the rounded synthetic signal.
            np.testing.assert_allclose(green[0].data, 0.6 * signal_r[2:-4, 4:-2], atol=0.5)


def test_flats_and_linear_masters_are_built_per_filter(tmp_path: Path) -> None:
    biases, darks, flats, lights, signal_r, _ = _dataset(tmp_path / "raw")
    height, width = signal_r.shape
    response_g = np.broadcast_to(
        np.linspace(1.15, 0.85, width, dtype=np.float32), (height, width)
    ).copy()
    signal_g = signal_r * 0.6
    for index in range(3):
        flats.append(
            _write_frame(
                tmp_path / "raw" / "flat_g" / f"flat_g_{index}.fits",
                "Flat",
                100.0 + 1000.0 * response_g,
                filter_name="G",
                exposure=2.0,
            )
        )
    for index in range(5):
        lights.append(
            _write_frame(
                tmp_path / "raw" / "light_g" / f"light_g_{index}.fits",
                "Light",
                120.0 + signal_g * response_g,
                filter_name="G",
            )
        )
    output = tmp_path / "multifilter"

    result = run_portable_pipeline(
        bias_files=biases,
        dark_files=darks,
        flat_files=flats,
        light_files=lights,
        output_directory=output,
        parameters=_parameters(),
    )

    assert len(result.master_light_paths) == 2
    assert (output / "masters" / "master_flat_R.fits").is_file()
    assert (output / "masters" / "master_flat_G.fits").is_file()
    with fits.open(output / "masters" / "master_light_G.fits", memmap=False) as hdul:
        with fits.open(output / "masters" / "master_flat_G.fits", memmap=False) as flat_hdul:
            decoded_response_g = np.asarray(flat_hdul[0].data, dtype=np.float32)
        with fits.open(lights[-1], memmap=False) as source_hdul:
            decoded_light_g = np.asarray(source_hdul[0].data, dtype=np.float32)
        expected_g = (decoded_light_g - np.float32(120.0)) / decoded_response_g
        np.testing.assert_allclose(hdul[0].data, expected_g, atol=3e-3)
    with fits.open(output / "masters" / "master_light_R.fits", memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, signal_r, atol=3e-3)


def test_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    biases, darks, flats, lights, _, _ = _dataset(tmp_path / "raw")
    output = tmp_path / "result"
    run_portable_pipeline(
        bias_files=biases,
        dark_files=darks,
        flat_files=flats,
        light_files=lights,
        output_directory=output,
        parameters=_parameters(),
    )
    receipt_hash = _sha(output / "receipt.json")

    with pytest.raises(CalibrationError) as captured:
        run_portable_pipeline(
            bias_files=biases,
            dark_files=darks,
            flat_files=flats,
            light_files=lights,
            output_directory=output,
            parameters=_parameters(),
        )
    assert captured.value.code == "OUTPUT_EXISTS"
    assert _sha(output / "receipt.json") == receipt_hash


@pytest.mark.parametrize("mismatch", ["filter", "dark-exposure"])
def test_calibration_mismatch_fails_before_publication(
    tmp_path: Path, mismatch: str
) -> None:
    biases, darks, flats, lights, _, _ = _dataset(tmp_path / "raw")
    if mismatch == "filter":
        with fits.open(flats[0], mode="update") as hdul:
            hdul[0].header["FILTER"] = "G"
        with fits.open(flats[1], mode="update") as hdul:
            hdul[0].header["FILTER"] = "G"
        with fits.open(flats[2], mode="update") as hdul:
            hdul[0].header["FILTER"] = "G"
    else:
        for dark in darks:
            with fits.open(dark, mode="update") as hdul:
                hdul[0].header["EXPTIME"] = 60.0
    output = tmp_path / "must-not-exist"

    with pytest.raises(CalibrationError) as captured:
        run_portable_pipeline(
            bias_files=biases,
            dark_files=darks,
            flat_files=flats,
            light_files=lights,
            output_directory=output,
            parameters=_parameters(),
        )
    assert captured.value.code in {"MASTER_FLAT_MISSING", "DARK_EXPOSURE_MISMATCH"}
    assert not output.exists()


def test_master_dark_hot_pixels_are_replaced_before_registration(tmp_path: Path) -> None:
    from openastroflow_engine.pixel_pipeline import _hot_pixel_map, _replace_hot_pixels

    rng = np.random.default_rng(3)
    dark = rng.normal(120.0, 1.0, (64, 80)).astype(np.float32)
    dark[10, 20] = 900.0   # hot
    dark[40, 60] = 135.0   # warm, still > 3 sigma
    dark[0, 0] = 500.0     # hot on the border
    rows, columns, evidence = _hot_pixel_map(dark, 3.0)
    assert evidence["count"] >= 3
    assert {(10, 20), (40, 60), (0, 0)} <= set(zip(rows.tolist(), columns.tolist()))
    light = np.full((64, 80), 500.0, dtype=np.float32)
    light[10, 20] = 5000.0
    light[40, 60] = 640.0
    light[0, 0] = 2000.0
    star = light.copy()
    _replace_hot_pixels(light, rows, columns)
    assert light[10, 20] == 500.0 and light[40, 60] == 500.0 and light[0, 0] == 500.0
    # Untouched pixels are exactly preserved.
    untouched = np.ones(light.shape, dtype=bool)
    untouched[rows, columns] = False
    np.testing.assert_array_equal(light[untouched], star[untouched])
    # A dark without dispersion carries no hot-pixel evidence.
    rows, columns, evidence = _hot_pixel_map(np.full((8, 8), 120.0, dtype=np.float32), 3.0)
    assert rows.size == 0 and evidence["count"] == 0


def test_integration_noise_weights_ignore_sky_gradients(tmp_path: Path) -> None:
    from openastroflow_engine.calibration import (
        FrameExpression,
        IntegrationParameters,
        _normalized_noise_weights,
        _open_expression_sources,
        _validate_expression_shapes,
    )
    from contextlib import ExitStack

    rng = np.random.default_rng(5)
    shape = (256, 320)
    y, x = np.indices(shape, dtype=np.float64)
    flat_frame = (500.0 + rng.normal(0.0, 4.0, shape)).astype(np.float32)
    # Same pixel noise, but a strong moonlit gradient across the frame.
    gradient_frame = (500.0 + 120.0 * x / shape[1] + 80.0 * y / shape[0] + rng.normal(0.0, 4.0, shape)).astype(np.float32)
    paths = []
    for name, data in (("flat", flat_frame), ("gradient", gradient_frame)):
        path = tmp_path / f"{name}.fits"
        fits.writeto(path, data)
        paths.append(path)
    expressions = tuple(FrameExpression(str(path)) for path in paths)
    with ExitStack() as stack:
        sources = _open_expression_sources(stack, expressions)
        shape_ = _validate_expression_shapes(expressions, sources)
        weights, _ = _normalized_noise_weights(
            expressions, sources, shape_, IntegrationParameters(max_statistics_samples=20_000)
        )
    # Equal noise must give (nearly) equal weights despite the gradient.
    assert abs(weights[0] / weights[1] - 1.0) < 0.15
