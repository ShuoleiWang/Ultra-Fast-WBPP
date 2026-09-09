from __future__ import annotations

import hashlib
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from openastroflow_engine.calibration import (
    CalibrationError,
    FitsFrame,
    FrameExpression,
    IntegrationMapPaths,
    IntegrationParameters,
    integrate_expressions,
    read_frame_info,
    write_expression,
)


def _write(path: Path, data: np.ndarray) -> Path:
    header = fits.Header()
    header["IMAGETYP"] = "Light"
    fits.writeto(path, data, header, overwrite=False)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_unsigned_fits_scaling_is_read_correctly_and_source_stays_read_only(
    tmp_path: Path,
) -> None:
    source = _write(
        tmp_path / "unsigned.fits",
        np.asarray([[0, 32768, 40000, 65535]], dtype=np.uint16),
    )
    before = _sha(source)
    with FitsFrame(source) as frame:
        values = frame.read_rows(0, 1)
    assert values.tolist() == [[0.0, 32768.0, 40000.0, 65535.0]]
    assert _sha(source) == before


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
def test_wide_unsigned_fits_preserves_low_adu_before_float_conversion(
    tmp_path: Path, dtype: type,
) -> None:
    expected = np.tile(
        np.asarray([[0, 0, 0, 1, 2, 10, 63, 64, 127, 128, 129, 1000, 1000, 1000]], dtype=dtype),
        (6, 1),
    )
    source = _write(tmp_path / "wide-unsigned.fits", expected)
    before = _sha(source)
    x, y = np.meshgrid(np.arange(expected.shape[1], dtype=float), [2.0])
    with FitsFrame(source) as frame:
        np.testing.assert_array_equal(frame.read_rows(0, 6), expected.astype(np.float32))
        np.testing.assert_array_equal(frame.sample_bilinear(x, y), expected[2:3].astype(np.float32))
        np.testing.assert_allclose(
            frame.sample_lanczos3_clamped(x[:, 2:-2], y[:, 2:-2]),
            expected[2:3, 2:-2].astype(np.float32), atol=1e-6,
        )
    assert _sha(source) == before


def test_scaled_integer_fits_applies_offset_before_float32_rounding(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "scaled.fits",
        np.asarray([[-2147483647, -2147483646, -2147483644]], dtype=np.int32),
    )
    fits.setval(source, "BSCALE", value=0.25)
    fits.setval(source, "BZERO", value=float(1 << 29))
    with FitsFrame(source) as frame:
        np.testing.assert_array_equal(frame.read_rows(0, 1), [[0.25, 0.5, 1.0]])


def test_flat_response_must_be_positive_but_negative_light_signal_is_retained(
    tmp_path: Path,
) -> None:
    light = _write(tmp_path / "light.fits", np.asarray([[-10, 10, 10, 10, 10]], dtype=np.float32))
    flat = _write(tmp_path / "flat.fits", np.asarray([[1, -1, 0, np.inf, 2]], dtype=np.float32))
    destination = tmp_path / "calibrated.fits"
    write_expression(FrameExpression(str(light), divide_path=str(flat)), destination)
    np.testing.assert_array_equal(fits.getdata(destination), [[-10, np.nan, np.nan, np.nan, 5]])


def test_rejection_uses_only_finite_samples_and_per_pixel_minimum(tmp_path: Path) -> None:
    paths = []
    for index, value in enumerate([10] * 5 + [100] + [np.inf] * 5):
        pixels = np.full((4, 8), 10, dtype=np.float32)
        pixels[0, 0] = value
        pixels[0, 1] = ([10, 10, 100] + [np.nan] * 8)[index]
        pixels[0, 2] = ([10, 10] + [np.inf] * 9)[index]
        pixels[0, 3] = np.nan
        paths.append(_write(tmp_path / f"finite-{index}.fits", pixels))
    for budget in (3000, 100_000):
        destination = tmp_path / f"finite-master-{budget}.fits"
        maps = [tmp_path / f"{kind}-{budget}.fits" for kind in ("accepted", "coverage", "rejected")]
        result = integrate_expressions(
            [FrameExpression(str(path)) for path in paths], destination,
            parameters=IntegrationParameters(minimum_rejection_frames=5, max_memory_bytes=budget),
            map_paths=IntegrationMapPaths(*maps),
        )
        np.testing.assert_array_equal(fits.getdata(destination)[0, :4], [10, 40, 10, np.nan])
        np.testing.assert_array_equal(fits.getdata(maps[0])[0, :4], [5, 3, 2, 0])
        np.testing.assert_array_equal(fits.getdata(maps[2])[0, :4], [1, 0, 0, 0])
        assert result.rejected_samples == 1


def test_integer_storage_domain_uses_physical_endpoints_and_rejects_other_units(
    tmp_path: Path,
) -> None:
    source = _write(
        tmp_path / "unsigned-domain.fits",
        np.asarray([[0, 65535]], dtype=np.uint16),
    )
    info = read_frame_info(source)
    assert info.numeric_domain == "INTEGER_16_PHYSICAL_0_BASED"
    assert info.normalized_unit_scale == 65535.0
    assert info.numeric_domain_authority == "FITS_STORAGE_ENDPOINTS"
    evidence = dict(info.numeric_domain_evidence)
    assert evidence["BITPIX"] == 16
    assert evidence["BSCALE"] == 1.0
    assert evidence["BZERO"] == 32768.0
    assert evidence["derivedPhysicalLow"] == 0.0
    assert evidence["derivedPhysicalHigh"] == 65535.0

    fits.setval(source, "BUNIT", value="electron")
    incompatible = read_frame_info(source)
    assert incompatible.numeric_domain == "UNDECLARED"
    assert incompatible.normalized_unit_scale is None
    assert incompatible.numeric_domain_authority == "FITS_BUNIT_UNSUPPORTED"


def test_tiled_robust_integration_rejects_a_cosmic_ray(tmp_path: Path) -> None:
    height, width = 9, 16
    base = (
        np.arange(height, dtype=np.float32)[:, None] * 2
        + np.arange(width, dtype=np.float32)[None, :]
        + 100
    )
    sources: list[Path] = []
    for index in range(5):
        values = base.copy()
        if index == 4:
            values[4, 7] += 10_000
        sources.append(_write(tmp_path / f"light_{index}.fits", values))
    hashes = {path: _sha(path) for path in sources}
    output = tmp_path / "master.fits"
    parameters = IntegrationParameters(
        sigma_clip=4.0,
        minimum_rejection_frames=3,
        max_memory_bytes=4096,
        max_statistics_samples=100,
    )

    result = integrate_expressions(
        (FrameExpression(str(path)) for path in sources),
        output,
        metadata={"IMAGETYP": "Master Light", "OAFSTATE": "UNSOLVED_WORKING"},
        parameters=parameters,
    )

    assert result.tile_rows < height
    assert result.rejected_samples >= 1
    assert sum(result.weights) == pytest.approx(1.0)
    with fits.open(output, memmap=False) as hdul:
        np.testing.assert_allclose(hdul[0].data, base, rtol=0, atol=2e-4)
        assert hdul[0].header["OAFSTATE"] == "UNSOLVED_WORKING"
    assert {path: _sha(path) for path in sources} == hashes

    with pytest.raises(CalibrationError) as captured:
        integrate_expressions(
            (FrameExpression(str(path)) for path in sources), output
        )
    assert captured.value.code == "OUTPUT_EXISTS"


def test_zero_accepted_pixel_remains_nan_instead_of_finite_median_fallback(
    tmp_path: Path,
) -> None:
    sources: list[Path] = []
    for index, value in enumerate((0.0, 1.0, 2.0, 3.0)):
        data = np.full((4, 4), 10.0, dtype=np.float32)
        data[0, 0] = value
        sources.append(_write(tmp_path / f"strict_{index}.fits", data))
    output = tmp_path / "strict-master.fits"
    accepted = tmp_path / "strict-accepted.fits"
    coverage = tmp_path / "strict-coverage.fits"
    rejected = tmp_path / "strict-rejected.fits"

    integrate_expressions(
        (FrameExpression(str(path)) for path in sources),
        output,
        parameters=IntegrationParameters(
            sigma_clip=1e-9,
            minimum_rejection_frames=3,
            max_memory_bytes=4096,
            max_statistics_samples=100,
        ),
        map_paths=IntegrationMapPaths(accepted, coverage, rejected),
    )

    assert np.isnan(fits.getdata(output)[0, 0])
    assert fits.getdata(accepted)[0, 0] == 0
    assert fits.getdata(coverage)[0, 0] == 0
    assert fits.getdata(rejected)[0, 0] == 4


def _mixed_noise_sources(tmp_path: Path) -> list[Path]:
    rng = np.random.default_rng(7)
    frame_count, height, width = 9, 8, 16
    row_scales = np.asarray(
        [0.001, 0.001, 0.001, 0.001, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float32,
    )
    stack = (
        rng.normal(0.0, 1.0, (frame_count, height, width)).astype(np.float32)
        * row_scales[None, :, None]
    )
    stack[0] += np.float32(0.01)
    stack[:, 0, 0] = np.nan
    stack[8, 6, 9] += np.float32(100.0)
    return [
        _write(tmp_path / f"mixed-{index:02d}.fits", values)
        for index, values in enumerate(stack)
    ]


def _integrate_with_maps(
    tmp_path: Path,
    sources: list[Path],
    *,
    memory_bytes: int,
    stem: str,
):
    output = tmp_path / f"{stem}.fits"
    accepted = tmp_path / f"{stem}-accepted.fits"
    coverage = tmp_path / f"{stem}-coverage.fits"
    rejected = tmp_path / f"{stem}-rejected.fits"
    result = integrate_expressions(
        [FrameExpression(str(path)) for path in sources],
        output,
        parameters=IntegrationParameters(
            sigma_clip=4.0,
            minimum_rejection_frames=3,
            max_memory_bytes=memory_bytes,
            max_statistics_samples=100,
        ),
        map_paths=IntegrationMapPaths(accepted, coverage, rejected),
    )
    return result, output, accepted, coverage, rejected


def test_rejection_and_cpu_pixels_are_tile_and_memory_invariant(
    tmp_path: Path,
) -> None:
    sources = _mixed_noise_sources(tmp_path)
    runs = [
        _integrate_with_maps(
            tmp_path,
            sources,
            memory_bytes=memory,
            stem=f"memory-{memory}",
        )
        for memory in (3000, 7000, 100_000)
    ]

    assert [item[0].tile_rows for item in runs] == [1, 2, 8]
    reference = runs[0]
    reference_floor = reference[0].execution["rejectionMask"]["sigmaFloor"]
    assert reference_floor["algorithm"] == "fixed-grid-group-mad-plus-float32-ulp-v1"
    assert reference_floor["tileInvariant"] is True
    assert reference_floor["coordinateCount"] <= 100
    assert reference_floor["usableSigmaCount"] > 0
    for run in runs[1:]:
        assert run[0].accepted_samples == reference[0].accepted_samples
        assert run[0].rejected_samples == reference[0].rejected_samples
        assert run[0].execution["rejectionMask"]["sigmaFloor"] == reference_floor
        for reference_path, candidate_path in zip(reference[1:], run[1:], strict=True):
            assert np.array_equal(
                fits.getdata(reference_path),
                fits.getdata(candidate_path),
                equal_nan=True,
            )


def test_rejection_floor_and_maps_are_frame_order_invariant(tmp_path: Path) -> None:
    sources = _mixed_noise_sources(tmp_path)
    forward = _integrate_with_maps(
        tmp_path, sources, memory_bytes=7000, stem="forward"
    )
    reverse = _integrate_with_maps(
        tmp_path, list(reversed(sources)), memory_bytes=7000, stem="reverse"
    )

    assert (
        forward[0].execution["rejectionMask"]["sigmaFloor"]
        == reverse[0].execution["rejectionMask"]["sigmaFloor"]
    )
    for forward_path, reverse_path in zip(forward[2:], reverse[2:], strict=True):
        assert np.array_equal(
            fits.getdata(forward_path), fits.getdata(reverse_path), equal_nan=True
        )


def test_zero_mad_and_nan_samples_use_recorded_numerical_floor(
    tmp_path: Path,
) -> None:
    sources: list[Path] = []
    for index in range(5):
        values = np.full((4, 8), 100.0, dtype=np.float32)
        values[:, 0] = np.nan
        if index == 4:
            values[2, 4] += np.float32(0.01)
        sources.append(_write(tmp_path / f"zero-mad-{index}.fits", values))

    result, output, accepted, _, rejected = _integrate_with_maps(
        tmp_path, sources, memory_bytes=4096, stem="zero-mad"
    )
    floor = result.execution["rejectionMask"]["sigmaFloor"]
    assert floor["sampledSigmaMedian"] is None
    assert floor["groupSigmaFloor"] == pytest.approx(1.0e-7)
    assert floor["float32EpsilonFactor"] == 16.0
    assert np.all(np.isnan(fits.getdata(output)[:, 0]))
    assert np.all(fits.getdata(accepted)[:, 0] == 0)
    assert np.all(fits.getdata(rejected)[:, 0] == 0)
    assert fits.getdata(rejected)[2, 4] == 1
