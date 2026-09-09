from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pytest

from openastroflow_engine.mosaic import (
    MOSAIC_STATE,
    MosaicError,
    MosaicRequest,
    ReprojectProvider,
    build_solved_panel_mosaic,
    reproject_capability,
)


def _wcs(*, crpix1: float = 6.5) -> WCS:
    value = WCS(naxis=2)
    value.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    value.wcs.crpix = [crpix1, 4.5]
    value.wcs.crval = [150.0, 20.0]
    value.wcs.cd = np.asarray([[-0.001, 0.0], [0.0, 0.001]])
    return value


def _panel(path: Path, value: float, *, crpix1: float, filter_name: str = "R") -> Path:
    header = _wcs(crpix1=crpix1).to_header(relax=True)
    header["FILTER"] = filter_name
    header["OBJECT"] = "SYNTHETIC-MOSAIC"
    header["OAFSTATE"] = "SOLVED"
    header["OAFWCS"] = "SOLVED"
    fits.writeto(
        path,
        np.full((8, 12), value, dtype=np.float32),
        header,
        overwrite=False,
        checksum=True,
    )
    return path


class FakeReproject:
    def __init__(self, *, fail_coadd: bool = False) -> None:
        self.fail_coadd = fail_coadd
        self.optimal_calls = 0
        self.coadd_calls = 0

    def find_optimal(self, inputs: Any, *, auto_rotate: bool = False) -> tuple[WCS, tuple[int, int]]:
        assert len(inputs) == 2
        assert auto_rotate is False
        self.optimal_calls += 1
        return _wcs(crpix1=10.5), (8, 20)

    def coadd(
        self,
        inputs: Any,
        output_wcs: WCS,
        *,
        shape_out: tuple[int, int],
        reproject_function: Any,
        combine_function: str,
        match_background: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert len(inputs) == 2
        assert output_wcs.has_celestial
        assert shape_out == (8, 20)
        assert reproject_function is not None
        assert combine_function in {"mean", "sum"}
        assert match_background is False
        self.coadd_calls += 1
        if self.fail_coadd:
            raise RuntimeError("injected coadd failure")
        coverage = np.zeros(shape_out, dtype=np.float32)
        total = np.zeros(shape_out, dtype=np.float32)
        for input_data in inputs:
            science, footprint = self.interp(
                input_data,
                output_wcs,
                shape_out=shape_out,
                return_footprint=True,
            )
            selected = footprint > 0
            coverage[selected] += footprint[selected]
            total[selected] += science[selected] * footprint[selected]
        science = np.full(shape_out, np.nan, dtype=np.float32)
        selected = coverage > 0
        science[selected] = (
            total[selected] / coverage[selected]
            if combine_function == "mean"
            else total[selected]
        )
        return science, coverage

    @staticmethod
    def interp(
        input_data: Any,
        output_wcs: WCS,
        *,
        shape_out: tuple[int, int],
        return_footprint: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        data, input_wcs = input_data
        science = np.full(shape_out, np.nan, dtype=np.float32)
        footprint = np.zeros(shape_out, dtype=np.float32)
        start = int(round(output_wcs.wcs.crpix[0] - input_wcs.wcs.crpix[0]))
        science[:, start : start + 12] = data
        footprint[:, start : start + 12] = 1.0
        assert return_footprint
        return science, footprint

    def provider(self) -> ReprojectProvider:
        return ReprojectProvider(
            backend_id="fake-reproject",
            version="test-1",
            find_optimal_celestial_wcs=self.find_optimal,
            reproject_and_coadd=self.coadd,
            reproject_function=self.interp,
        )


def _panels(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _panel(tmp_path / "panel-1.fits", 10.0, crpix1=10.5),
        _panel(tmp_path / "panel-2.fits", 30.0, crpix1=2.5),
    )


def test_reproject_dependency_absence_reports_capability_false() -> None:
    def missing_import(name: str) -> Any:
        raise ModuleNotFoundError(name)

    capability = reproject_capability(missing_import)
    assert capability.capable is False
    assert capability.available is False
    assert capability.execution_ready is False
    assert capability.serializable()["capability"] is False
    assert "unavailable" in (capability.reason or "")


def test_capability_checks_required_reproject_surface() -> None:
    package = SimpleNamespace(__version__="1.0", reproject_interp=lambda *args: None)
    mosaicking = SimpleNamespace(find_optimal_celestial_wcs=lambda inputs: None)

    def importer(name: str) -> Any:
        return package if name == "reproject" else mosaicking

    capability = reproject_capability(importer)
    assert capability.available is True
    assert capability.capable is False
    assert "reproject_and_coadd" in (capability.reason or "")


def test_solved_panel_mosaic_records_coverage_and_needs_final_solve(tmp_path: Path) -> None:
    panels = _panels(tmp_path)
    fake = FakeReproject()
    output = tmp_path / "mosaic"
    result = build_solved_panel_mosaic(
        MosaicRequest(tuple(str(path) for path in panels), str(output)),
        provider=fake.provider(),
    )

    assert result.state == MOSAIC_STATE
    assert fake.optimal_calls == 1
    assert fake.coadd_calls == 1
    with fits.open(result.mosaic_path, checksum=True) as hdul:
        assert [hdu.name for hdu in hdul] == ["PRIMARY", "FOOTPRINT", "COVERAGE"]
        assert hdul[0].header["OAFSTATE"] == "NEEDS_FINAL_SOLVE"
        assert hdul[0].header["OAFWCS"] == "PROPAGATED"
        assert hdul[0].header["FILTER"] == "R"
        assert hdul[0].header["OAFNPAN"] == 2
        assert np.all(hdul["FOOTPRINT"].data[:, :])
        assert float(np.max(hdul["COVERAGE"].data)) == 2.0
        assert np.allclose(hdul[0].data[:, 8:12], 10.0)
        assert np.allclose(hdul[0].data[:, :], 10.0)

    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["state"] == "NEEDS_FINAL_SOLVE"
    assert receipt["requiresFinalSolve"] is True
    assert receipt["propagatedWcsIsFinalSolution"] is False
    assert receipt["outputGrid"]["wcsRole"] == "REPROJECTION_GRID_ONLY"
    assert receipt["overlap"]["coveredPixels"] == 160
    assert receipt["overlap"]["multipleContributorPixels"] == 32
    assert receipt["overlap"]["maximumCoverage"] == 2.0
    assert len(receipt["panels"]) == 2
    assert all("sha256" in item["identity"] for item in receipt["panels"])
    photometric = receipt["coadd"]["photometricNormalization"]
    assert photometric["referencePanelIndex"] == 0
    assert photometric["panelCorrections"][1]["appliedGain"] == pytest.approx(1 / 3)
    assert photometric["panelCorrections"][1]["appliedOffset"] == pytest.approx(0)
    assert photometric["correctedOverlapVerification"]["connected"] is True
    assert all(
        item["seam"]["normalizedMad"] == pytest.approx(0)
        for item in photometric["correctedOverlapVerification"]["pairwise"]
    )

    with pytest.raises(MosaicError) as captured:
        build_solved_panel_mosaic(
            MosaicRequest(tuple(str(path) for path in panels), str(output)),
            provider=fake.provider(),
        )
    assert captured.value.code == "OUTPUT_EXISTS"


def test_mosaic_coadd_failure_is_atomic(tmp_path: Path) -> None:
    panels = _panels(tmp_path)
    output = tmp_path / "coadd-failure"
    with pytest.raises(MosaicError) as captured:
        build_solved_panel_mosaic(
            MosaicRequest(tuple(str(path) for path in panels), str(output)),
            provider=FakeReproject(fail_coadd=True).provider(),
        )
    assert captured.value.code == "REPROJECT_COADD_FAILED"
    assert not output.exists()
    assert not list(tmp_path.glob(".coadd-failure.staging-*"))


def test_sum_keeps_cumulative_coverage_and_accepts_filter_aliases(tmp_path: Path) -> None:
    first = _panel(tmp_path / "sum-1.fits", 10.0, crpix1=10.5, filter_name="R")
    second = _panel(tmp_path / "sum-2.fits", 30.0, crpix1=2.5, filter_name="Red")
    output = tmp_path / "sum-mosaic"
    result = build_solved_panel_mosaic(
        MosaicRequest((str(first), str(second)), str(output), combine_function="sum"),
        provider=FakeReproject().provider(),
    )
    with fits.open(result.mosaic_path) as hdul:
        assert np.allclose(hdul[0].data[:, 8:12], 20.0)
        assert float(np.max(hdul["COVERAGE"].data)) == 2.0
    assert result.receipt["overlap"]["multipleContributorPixels"] == 32


def test_mosaic_rejects_unknown_filter_and_noncoverage_combine_mode(tmp_path: Path) -> None:
    first, second = _panels(tmp_path)
    with fits.open(second, mode="update") as hdul:
        del hdul[0].header["FILTER"]
        hdul.flush()
    output = tmp_path / "unknown-filter"
    with pytest.raises(MosaicError) as captured:
        build_solved_panel_mosaic(
            MosaicRequest((str(first), str(second)), str(output)),
            provider=FakeReproject().provider(),
        )
    assert captured.value.code == "PANEL_FILTER_MISSING"
    assert not output.exists()

    with pytest.raises(MosaicError) as captured:
        build_solved_panel_mosaic(
            MosaicRequest((str(first), str(second)), str(tmp_path / "bad-combine"), combine_function="first"),
            provider=FakeReproject().provider(),
        )
    assert captured.value.code == "COMBINE_FUNCTION_INVALID"


def test_mosaic_auto_rotate_fails_closed_when_shapely_is_unavailable(tmp_path: Path) -> None:
    panels = _panels(tmp_path)
    fake = FakeReproject().provider()
    provider = ReprojectProvider(
        backend_id=fake.backend_id,
        version=fake.version,
        find_optimal_celestial_wcs=fake.find_optimal_celestial_wcs,
        reproject_and_coadd=fake.reproject_and_coadd,
        reproject_function=fake.reproject_function,
        auto_rotate_available=False,
        auto_rotate_reason="test: shapely missing",
    )
    output = tmp_path / "auto-rotate"
    with pytest.raises(MosaicError) as captured:
        build_solved_panel_mosaic(
            MosaicRequest(tuple(str(path) for path in panels), str(output), auto_rotate=True),
            provider=provider,
        )
    assert captured.value.code == "MOSAIC_AUTO_ROTATE_UNAVAILABLE"
    assert not output.exists()


def test_mosaic_publish_failure_removes_staging_and_does_not_publish(tmp_path: Path) -> None:
    panels = _panels(tmp_path)
    output = tmp_path / "publish-failure"

    def fail_publish(_source: Path, _destination: Path) -> None:
        raise OSError("injected publication failure")

    with pytest.raises(MosaicError) as captured:
        build_solved_panel_mosaic(
            MosaicRequest(tuple(str(path) for path in panels), str(output)),
            provider=FakeReproject().provider(),
            publisher=fail_publish,
        )
    assert captured.value.code == "ATOMIC_PUBLICATION_FAILED"
    assert not output.exists()
    assert not list(tmp_path.glob(".publish-failure.staging-*"))
