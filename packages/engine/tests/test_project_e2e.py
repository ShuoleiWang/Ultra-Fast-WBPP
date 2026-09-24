from __future__ import annotations

from dataclasses import replace
import errno
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from astropy.io import fits
from astropy.wcs import Sip, WCS
import numpy as np
import pytest

from ufwbpp.workflows.single_target import E2EError, E2ERequest, E2EResult, E2EState, ProgressEvent, ProgressStage, ReviewApproval
from ufwbpp.cli import _load_project_request, _verify_expected_source_roles
from ufwbpp.inventory import inventory_project
from ufwbpp.mosaic import ReprojectProvider
from ufwbpp.workflows.project import (
    ProjectE2EError,
    ProjectE2ERequest,
    ProjectLayout,
    SciencePanel,
    _ProjectProgress,
    _align_channel,
    _astrometry_gui_evidence,
    _crop_final_channels,
    classify_project_layout,
    run_project_e2e,
)
from ufwbpp.solver import (
    AstrometricQuality,
    SolutionKind,
    SolverIndexArtifact,
    SolverResult,
    SolverStatus,
    WcsParity,
)


def _managed_quality() -> AstrometricQuality:
    return AstrometricQuality(
        matched_stars=40,
        rms_pixels=0.1,
        rms_arcsec=0.4,
        parity=WcsParity.NEGATIVE,
        catalog_identity="1" * 64,
        index_identities=("astrometry.net:index:4200:healpix:1:hpnside:1",),
        correspondence_sha256="2" * 64,
        catalog_managed=True,
        installed_set_identity="3" * 64,
        catalog_manifest_sha256="4" * 64,
        index_artifacts=(
            SolverIndexArtifact(
                index_id="4200",
                relative_name="index-4200.fits",
                size_bytes=4096,
                sha256="5" * 64,
                manifest_sha256="4" * 64,
                installed_set_identity="3" * 64,
            ),
        ),
    )


def _output_wcs(*, crpix: tuple[float, float] = (7.5, 7.5), crval1: float = 150.0) -> WCS:
    value = WCS(naxis=2)
    value.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    value.wcs.crpix = list(crpix)
    value.wcs.crval = [crval1, 20.0]
    value.wcs.cd = np.asarray([[-0.001, 0.0], [0.0, 0.001]])
    return value


def _science_header(target: str, filter_name: str, *, role: str = "Light") -> fits.Header:
    header = fits.Header()
    header["IMAGETYP"] = role
    header["OBJECT"] = target
    header["FILTER"] = filter_name
    header["INSTRUME"] = "SYNTH-CAM"
    header["EXPTIME"] = 60.0 if role == "Light" else 2.0
    header["GAIN"] = 100
    header["OFFSET"] = 20
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    header["READOUTM"] = "MODE-1"
    header["BAYERPAT"] = "NONE"
    header["CCD-TEMP"] = -10.0
    return header


def _write(path: Path, data: np.ndarray, header: fits.Header) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.asarray(data), header, overwrite=False, checksum=True)
    return path


def _opposite_axis_channels(
    tmp_path: Path, *, source_ra: float = 150.0, source_shift: float = 0.0,
) -> tuple[Path, Path, np.ndarray]:
    shape = (64, 80)
    y, x = np.indices(shape, dtype=np.float32)
    data = 100 + 0.4 * x + 0.3 * y + 20 * np.exp(-((x - 25) ** 2 + (y - 43) ** 2) / 80)
    reference_wcs = _output_wcs(crpix=((shape[1] + 1) / 2, (shape[0] + 1) / 2))
    source_wcs = _output_wcs(crpix=((shape[1] + 1) / 2 + source_shift, (shape[0] + 1) / 2), crval1=source_ra)
    source_wcs.wcs.cd = -reference_wcs.wcs.cd
    reference = _write(tmp_path / "R.fits", data, reference_wcs.to_header(relax=True))
    source = _write(tmp_path / "B.fits", data[::-1, ::-1], source_wcs.to_header(relax=True))
    return source, reference, data


def test_channel_alignment_accepts_same_sky_with_opposite_pixel_axes(tmp_path: Path) -> None:
    reproject = pytest.importorskip("reproject")
    source, reference, expected = _opposite_axis_channels(tmp_path)
    destination = tmp_path / "aligned-B.fits"
    result = _align_channel(
        source, reference, destination, filter_name="B", reference_filter="R",
        provider=replace(FakeReproject().provider(), reproject_function=reproject.reproject_interp),
        tolerance_pixels=0.05, minimum_coverage=0.98,
    )
    assert result["mode"] == "REPROJECTED_INDEPENDENT_SOLVE"
    assert result["maximumNinePointResidualPixelsBeforeAlignment"] > 70
    assert result["skyOverlapPreflight"]["centerSeparationDegrees"] < 1e-8
    assert result["skyOverlapPreflight"]["overlappingFootprintSamples"] == 81
    assert result["coverageFraction"] == 1.0
    assert result["maximumNinePointResidualPixelsAfterAlignment"] < 0.05
    np.testing.assert_allclose(fits.getdata(destination), expected, rtol=1e-6)


def test_rotated_channel_with_wrong_sky_center_is_rejected_before_reprojection(tmp_path: Path) -> None:
    source, reference, _ = _opposite_axis_channels(tmp_path, source_ra=152.0)
    def forbidden_reproject(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("wrong-field source must be rejected before reprojection")
    with pytest.raises(ProjectE2EError) as captured:
        _align_channel(
            source, reference, tmp_path / "wrong-B.fits", filter_name="B", reference_filter="R",
            provider=replace(FakeReproject().provider(), reproject_function=forbidden_reproject),
            tolerance_pixels=0.05, minimum_coverage=0.98,
        )
    assert captured.value.code == "CHANNEL_WCS_GROSS_MISMATCH"
    assert "sky center differs" in str(captured.value)
    assert not (tmp_path / "wrong-B.fits").exists()


def test_rotated_overlapping_channel_still_requires_full_resolution_coverage(tmp_path: Path) -> None:
    reproject = pytest.importorskip("reproject")
    source, reference, _ = _opposite_axis_channels(tmp_path, source_shift=4.0)
    with pytest.raises(ProjectE2EError) as captured:
        _align_channel(
            source, reference, tmp_path / "partial-B.fits", filter_name="B", reference_filter="R",
            provider=replace(FakeReproject().provider(), reproject_function=reproject.reproject_interp),
            tolerance_pixels=0.05, minimum_coverage=0.98,
        )
    assert captured.value.code == "CHANNEL_ALIGNMENT_COVERAGE_FAILED"
    assert not (tmp_path / "partial-B.fits").exists()


def _project(tmp_path: Path, *, filters: tuple[str, ...] = ("R", "G", "B")) -> tuple[Any, E2ERequest, list[Path]]:
    lights: list[Path] = []
    for target_index in range(4):
        for filter_name in filters:
            for sequence in range(3):
                lights.append(
                    _write(
                        tmp_path / "input" / f"dunpai{target_index + 1}" / filter_name / f"light-{sequence}.fits",
                        np.full((8, 8), 100 + 10 * target_index + sequence, dtype=np.uint16),
                        _science_header(f"dunpai{target_index + 1}", filter_name),
                    )
                )
    master_bias = _write(
        tmp_path / "input" / "masters" / "master-bias.fits",
        np.zeros((8, 8), dtype=np.float32),
        _science_header("calibration", "NONE", role="Master Bias"),
    )
    master_flats = [
        _write(
            tmp_path / "input" / "masters" / f"master-flat-{filter_name}.fits",
            np.ones((8, 8), dtype=np.float32),
            _science_header("calibration", filter_name, role="Master Flat"),
        )
        for filter_name in filters
    ]
    inventory = inventory_project([tmp_path / "input"])
    request = E2ERequest(
        light_files=tuple(str(path) for path in lights),
        flat_files=(),
        bias_files=(),
        master_bias_files=(str(master_bias),),
        master_flat_files=tuple(str(path) for path in master_flats),
        output_directory=str(tmp_path / "product"),
        ra_hint_degrees=150.0,
        dec_hint_degrees=20.0,
        field_of_view_degrees=0.014,
        search_radius_degrees=2.0,
    )
    return inventory, request, lights


class FakePanelRunner:
    def __init__(self, *, drift_source: Path | None = None) -> None:
        self.calls = 0
        self.drift_source = drift_source

    def __call__(self, request: E2ERequest, **_kwargs: Any) -> E2EResult:
        # One run carries every filter of one target; every filter master of
        # the run shares the target's pixel grid, as the real pipeline does.
        self.calls += 1
        filters: dict[str, str] = {}
        target = ""
        for path in request.light_files:
            header = fits.getheader(Path(path))
            target = str(header["OBJECT"])
            filters.setdefault(str(header["FILTER"]), target)
        target_index = int(target[-1]) - 1
        starts = ((0, 0), (6, 0), (0, 6), (6, 6))
        start_x, start_y = starts[target_index]
        output = Path(request.output_directory)
        quality = _managed_quality().serializable()
        products: list[Path] = []
        astrometry: dict[str, Any] = {}
        for filter_name in sorted(filters):
            solved_header = _output_wcs(crpix=(7.5 - start_x, 7.5 - start_y)).to_header(relax=True)
            solved_header["IMAGETYP"] = "Master Light"
            solved_header["OBJECT"] = target
            solved_header["FILTER"] = filter_name
            solved_header["OAFSTATE"] = "SOLVED"
            solved_header["OAFWCS"] = "SOLVED"
            product = output / "products" / filter_name / f"master_light_{filter_name}_wcs.fits"
            product.parent.mkdir(parents=True)
            value = {"R": 10.0, "G": 8.0, "B": 6.0, "L": 12.0}.get(filter_name, 5.0)
            local_y, local_x = np.indices((8, 8), dtype=np.float32)
            sky_signal = (
                value
                + 0.15 * (local_x + start_x)
                + 0.10 * (local_y + start_y)
            )
            panel_gain = 1.0 + 0.10 * target_index
            panel_offset = 2.0 * target_index
            fits.writeto(
                product,
                (sky_signal * panel_gain + panel_offset).astype(np.float32),
                solved_header,
                checksum=True,
            )
            products.append(product)
            astrometry[filter_name] = {
                "status": "SOLVED",
                "output": str(product.relative_to(output)),
                "attempts": [
                    {
                        "accepted": True,
                        "result": {"astrometricQuality": quality},
                    }
                ],
            }
        receipt = output / "receipt.json"
        # One excluded frame per target with a review preview, as the real
        # run records it, so the project receipt's aggregation is exercised.
        review = output / "qc" / "review" / "0000-excluded.png"
        review.parent.mkdir(parents=True)
        review.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        screening = {
            "counts": {"PASS": len(request.light_files) - 1, "REVIEW": 1, "HARD_FAIL": 0},
            "admitted": len(request.light_files) - 1,
            "excluded": 1,
            "frames": [{
                "path": request.light_files[0], "disposition": "REVIEW", "admitted": False,
                "summary": "fake review", "evidence": ["fake evidence"], "starCount": 3,
                "reviewPreview": "qc/review/0000-excluded.png",
            }],
        }
        receipt.write_text(
            json.dumps({
                "pipelineVersion": "fake-panel-e2e", "astrometry": {"filters": astrometry},
                "qualityControl": {"screening": screening},
            }),
            encoding="utf-8",
        )
        if self.drift_source is not None and self.calls == 1:
            with self.drift_source.open("ab") as stream:
                stream.write(b"drift")
        return E2EResult(
            success=True,
            code="E2E_SUCCEEDED",
            state=E2EState.SOLVED,
            output_directory=str(output),
            evidence_directory=None,
            receipt_path=str(receipt),
            product_paths=tuple(str(product) for product in products),
            preview_paths=(),
            passed_light_paths=request.light_files,
            excluded_light_paths=(),
        )


class FakeReproject:
    def __init__(self, *, sparse_coverage: bool = False) -> None:
        self.sparse_coverage = sparse_coverage

    def find_optimal(self, inputs: Any, **_kwargs: Any) -> tuple[WCS, tuple[int, int]]:
        assert len(inputs) == 4
        return _output_wcs(), (14, 14)

    def interp(
        self,
        input_data: Any,
        output_wcs: WCS,
        *,
        shape_out: tuple[int, int],
        return_footprint: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        data, input_wcs = input_data
        start_x = int(round(output_wcs.wcs.crpix[0] - input_wcs.wcs.crpix[0]))
        start_y = int(round(output_wcs.wcs.crpix[1] - input_wcs.wcs.crpix[1]))
        science = np.full(shape_out, np.nan, dtype=np.float32)
        footprint = np.zeros(shape_out, dtype=np.float32)
        height, width = data.shape
        science[start_y : start_y + height, start_x : start_x + width] = data
        footprint[start_y : start_y + height, start_x : start_x + width] = 1.0
        if self.sparse_coverage:
            footprint[:, 4:] = 0
            science[:, 4:] = np.nan
        assert return_footprint
        return science, footprint

    def coadd(
        self,
        inputs: Any,
        output_wcs: WCS,
        *,
        shape_out: tuple[int, int],
        **_kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        total = np.zeros(shape_out, dtype=np.float64)
        coverage = np.zeros(shape_out, dtype=np.float32)
        for input_data in inputs:
            science, footprint = self.interp(
                input_data, output_wcs, shape_out=shape_out
            )
            selected = footprint > 0
            total[selected] += science[selected]
            coverage[selected] += footprint[selected]
        output = np.full(shape_out, np.nan, dtype=np.float32)
        selected = coverage > 0
        output[selected] = (total[selected] / coverage[selected]).astype(np.float32)
        return output, coverage

    def provider(self) -> ReprojectProvider:
        return ReprojectProvider(
            backend_id="fake-reproject",
            version="1",
            find_optimal_celestial_wcs=self.find_optimal,
            reproject_and_coadd=self.coadd,
            reproject_function=self.interp,
        )


class ManagedCopySolver:
    backend_id = "fake-managed-solver"

    def __init__(
        self,
        *,
        fail: bool = False,
        wrong_filter: str | None = None,
        rotate_filter: str | None = None,
        sip: bool = False,
    ) -> None:
        self.fail = fail
        self.wrong_filter = wrong_filter
        self.rotate_filter = rotate_filter
        self.sip = sip

    def solve(self, request: Any) -> SolverResult:
        if self.fail:
            return SolverResult(
                backend_id=self.backend_id,
                status=SolverStatus.FAILED,
                solution_kind=SolutionKind.NONE,
                backend_confirmed=False,
                error="injected failure",
            )
        with fits.open(request.input_path, mode="readonly", memmap=False) as hdul:
            header = hdul[0].header.copy()
            if self.wrong_filter is not None and str(header.get("FILTER")) == self.wrong_filter:
                header["CRVAL1"] = float(header["CRVAL1"]) + 1.0
            if self.rotate_filter is not None and str(header.get("FILTER")) == self.rotate_filter:
                angle = np.deg2rad(1.0)
                matrix = np.asarray(WCS(header).celestial.pixel_scale_matrix)
                rotation = np.asarray(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                rotated = rotation @ matrix
                for key in (
                    "PC1_1",
                    "PC1_2",
                    "PC2_1",
                    "PC2_2",
                    "CDELT1",
                    "CDELT2",
                ):
                    if key in header:
                        del header[key]
                header["CD1_1"] = rotated[0, 0]
                header["CD1_2"] = rotated[0, 1]
                header["CD2_1"] = rotated[1, 0]
                header["CD2_2"] = rotated[1, 1]
            if self.sip:
                celestial = WCS(header, naxis=2)
                a = np.zeros((3, 3))
                b = np.zeros((3, 3))
                a[2, 0], b[0, 2] = 2e-4, -3e-4
                celestial.sip = Sip(a, b, None, None, celestial.wcs.crpix)
                celestial.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
                header.update(celestial.to_header(relax=True))
            hdul[0].header = header
            hdul.writeto(request.output_path, overwrite=False, checksum=True)
            shape = tuple(int(value) for value in hdul[0].data.shape)
        return SolverResult(
            backend_id=self.backend_id,
            status=SolverStatus.SOLVED,
            solution_kind=SolutionKind.SOLVED,
            backend_confirmed=True,
            header=header,
            image_shape=shape,
            output_path=request.output_path,
            astrometric_quality=_managed_quality(),
            evidence={"verified": True},
        )

    def verify_result(self, result: SolverResult) -> bool:
        return bool(result.evidence.get("verified")) and Path(result.output_path or "").is_file()


def test_four_panel_rgb_project_runs_mosaics_fresh_solves_and_color(tmp_path: Path) -> None:
    inventory, base, _lights = _project(tmp_path)
    layout = classify_project_layout(inventory)
    assert layout.target_keys == ("dunpai1", "dunpai2", "dunpai3", "dunpai4")
    assert len(layout.panels) == 12
    runner = FakePanelRunner()
    events = []

    def progressing_runner(request, **kwargs):
        callback = kwargs["progress"]
        callback(ProgressEvent(ProgressStage.QUALITY_CONTROL, "completed", 1, 1))
        callback(ProgressEvent(ProgressStage.INTEGRATION, "started"))
        result = runner(request, **kwargs)
        callback(ProgressEvent(ProgressStage.COMPLETE, "completed", 1, 1))
        return result

    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(),),
        panel_runner=progressing_runner,
        progress=events.append,
        mosaic_provider=FakeReproject().provider(),
    )

    assert result.success is True
    assert result.code == "PROJECT_COLOR_SUCCEEDED"
    assert result.mono_filters == ("B", "G", "R")
    # One multi-filter run per target keeps every filter master of a target
    # on one pixel grid.
    assert runner.calls == 4
    fractions = [event.overall_fraction for event in events]
    assert fractions == sorted(fractions)
    assert all(0 <= value < 1 for value in fractions)
    assert fractions[-1] == 0.99  # Desktop receipt verification owns completion.
    panels = [event for event in events if event.scope == "panel"]
    assert {event.panel_index for event in panels} == set(range(1, 5))
    assert all(event.panel_count == 4 and event.panel is not None for event in panels)
    assert {event.panel.filter_name for event in panels} == {"B+G+R"}
    assert all(event.stage is not ProgressStage.COMPLETE for event in panels)
    assert max(event.overall_fraction for event in panels) < fractions[-1]
    phases = {event.project_stage for event in events if event.scope == "project"}
    assert phases == {"prepare", "mosaic", "alignment", "color", "verify", "publish"}
    assert next(event for event in events if event.project_stage == "alignment").current == 0
    assert result.color_product_path is not None
    with fits.open(result.color_product_path) as hdul:
        assert hdul[0].data.shape == (3, 14, 14)
        assert hdul[0].header["OAFSTATE"] == "SOLVED"
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["execution"]["sourceExtraction"]["deterministic"] is True
    assert receipt["execution"]["sharedCalibration"]["rawIntegrationCount"] == 0
    assert all(
        item["workingState"] == "NEEDS_FINAL_SOLVE"
        and item["propagatedWcsIsFinalSolution"] is False
        for item in receipt["execution"]["mosaics"].values()
    )
    assert all(item["status"] == "SOLVED" for item in receipt["execution"]["finalSolves"].values())
    serialized = Path(result.receipt_path).read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "device" not in serialized.casefold()
    assert "inode" not in serialized.casefold()
    assert "mtime" not in serialized.casefold()


def test_project_progress_weights_lights_and_does_not_credit_failure_cleanup(tmp_path: Path) -> None:
    _inventory, base, _lights = _project(tmp_path)
    small = SciencePanel("small", "small", "R", "r", ("one",))
    large = SciencePanel("large", "large", "R", "r", ("two", "three", "four"))
    events = []
    tracker = _ProjectProgress(ProjectLayout((small, large)), base, events.append)
    small_progress = tracker.panel_callback(1, small)
    large_progress = tracker.panel_callback(2, large)
    small_progress(ProgressEvent(ProgressStage.QUALITY_CONTROL, "completed"))
    one_light = events[-1].overall_fraction
    large_progress(ProgressEvent(ProgressStage.QUALITY_CONTROL, "completed"))
    assert events[-1].overall_fraction == pytest.approx(4 * one_light)
    large_progress(ProgressEvent(ProgressStage.PUBLISH, "started", message="publishing UNSOLVED evidence"))
    large_progress(ProgressEvent(ProgressStage.FAILED, "completed"))
    assert events[-1].overall_fraction == pytest.approx(4 * one_light)
    assert events[-1].stage is ProgressStage.FAILED
    # A regressing or overlarge stage counter cannot move backwards or overflow.
    small_progress(ProgressEvent(ProgressStage.REGISTRATION, "progress", 9, 2))
    high = events[-1].overall_fraction
    small_progress(ProgressEvent(ProgressStage.REGISTRATION, "progress", 0, 2))
    assert events[-1].overall_fraction == high
    assert high == pytest.approx(5 * one_light)
    assert ProgressEvent(ProgressStage.QUALITY_CONTROL, "started").serializable() == {
        "stage": "quality-control", "status": "started", "current": 0, "total": 0, "message": "",
    }


def test_single_panel_review_approval_preserves_exact_raw_request(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    lights = [
        _write(
            input_root / "dunpai1" / "R" / f"light-{index}.fits",
            np.full((8, 8), 100 + index, dtype=np.float32),
            _science_header("dunpai1", "R"),
        )
        for index in range(3)
    ]
    master_bias = _write(
        input_root / "masters" / "master-bias.fits",
        np.zeros((8, 8), dtype=np.float32),
        _science_header("calibration", "NONE", role="Master Bias"),
    )
    master_flat = _write(
        input_root / "masters" / "master-flat-R.fits",
        np.ones((8, 8), dtype=np.float32),
        _science_header("calibration", "R", role="Master Flat"),
    )
    inventory = inventory_project([input_root])
    approval = ReviewApproval(
        source_sha256="sha256:" + "1" * 64,
        gate_policy_digest="sha256:" + "2" * 64,
        request_digest="sha256:" + "3" * 64,
    )
    base = E2ERequest(
        light_files=tuple(str(path) for path in lights),
        flat_files=(),
        bias_files=(),
        master_bias_files=(str(master_bias),),
        master_flat_files=(str(master_flat),),
        review_approvals=(approval,),
        output_directory=str(tmp_path / "product"),
    )
    observed: list[E2ERequest] = []
    runner = FakePanelRunner()

    def capture(request: E2ERequest, **kwargs: Any) -> E2EResult:
        observed.append(request)
        return runner(request, **kwargs)

    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(),),
        panel_runner=capture,
    )

    assert result.success is True
    assert observed[0].review_approvals == (approval,)
    assert observed[0].master_bias_files == base.master_bias_files
    assert observed[0].master_flat_files == base.master_flat_files
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["execution"]["sharedCalibration"]["mode"] == (
        "SINGLE_PANEL_DIRECT_APPROVED_REQUEST"
    )


def test_review_selections_are_bound_per_target_run_in_a_multi_panel_project(tmp_path: Path) -> None:
    """A GUI reviewer admits frames of several targets and filters; each
    target run receives the approvals of its own Lights, bound to that run's
    request (its Lights, the shared masters it uses, the gate policy)."""

    inventory, base, lights = _project(tmp_path, filters=("R", "G", "B"))
    digest = lambda path: "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731
    policy = base.gate_policy.canonical_digest()
    chosen = [path for path in lights if path.parent.parent.name == "dunpai2" and path.parent.name == "R"][:2]
    chosen.append(next(path for path in lights if path.parent.parent.name == "dunpai3" and path.parent.name == "B"))
    selections = tuple({"sourceSha256": digest(path), "gatePolicyDigest": policy} for path in chosen)
    observed: list[E2ERequest] = []
    runner = FakePanelRunner()

    def capture(request: E2ERequest, **kwargs: Any) -> E2EResult:
        observed.append(request)
        return runner(request, **kwargs)

    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory, review_selections=selections),
        solver_backends=(ManagedCopySolver(),),
        panel_runner=capture,
        mosaic_provider=FakeReproject().provider(),
    )

    assert result.success is True
    by_target = {Path(request.light_files[0]).parent.parent.name: request for request in observed}
    assert sorted(by_target) == ["dunpai1", "dunpai2", "dunpai3", "dunpai4"]
    assert by_target["dunpai1"].review_approvals == ()
    assert by_target["dunpai4"].review_approvals == ()
    assert sorted(item.source_sha256 for item in by_target["dunpai2"].review_approvals) == sorted(digest(path) for path in chosen[:2])
    assert [item.source_sha256 for item in by_target["dunpai3"].review_approvals] == [digest(chosen[2])]
    for request in (by_target["dunpai2"], by_target["dunpai3"]):
        assert {item.gate_policy_digest for item in request.review_approvals} == {policy}
        assert all(item.request_digest.startswith("sha256:") for item in request.review_approvals)
    # The two runs' approvals are bound to different requests.
    assert by_target["dunpai2"].review_approvals[0].request_digest != by_target["dunpai3"].review_approvals[0].request_digest

    stale_base = replace(base, output_directory=str(tmp_path / "product-stale"))
    stale = ProjectE2ERequest(
        inventory, stale_base, stale_base.output_directory,
        review_selections=(*selections, {"sourceSha256": "sha256:" + "f" * 64, "gatePolicyDigest": policy}),
    )
    with pytest.raises(ProjectE2EError) as excinfo:
        run_project_e2e(stale, solver_backends=(ManagedCopySolver(),), panel_runner=FakePanelRunner(), mosaic_provider=FakeReproject().provider())
    assert excinfo.value.code == "REVIEW_APPROVAL_SOURCE_AMBIGUOUS"

    malformed_base = replace(base, output_directory=str(tmp_path / "product-malformed"))
    malformed = ProjectE2ERequest(
        inventory, malformed_base, malformed_base.output_directory,
        review_selections=({"sourceSha256": digest(chosen[0]), "gatePolicyDigest": policy, "extra": "x"},),
    )
    with pytest.raises(ProjectE2EError) as excinfo:
        run_project_e2e(malformed, solver_backends=(ManagedCopySolver(),), panel_runner=FakePanelRunner(), mosaic_provider=FakeReproject().provider())
    assert excinfo.value.code == "REVIEW_SELECTION_INVALID"


def test_published_layout_is_channels_previews_receipt_and_details(tmp_path: Path) -> None:
    inventory, base, _ = _project(tmp_path, filters=("R", "G", "B", "L"))
    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(),),
        panel_runner=FakePanelRunner(),
        mosaic_provider=FakeReproject().provider(),
    )
    assert result.success is True
    output = Path(result.output_directory)
    # Top level: one FITS per channel named after it (PixInsight labels an
    # opened image with the file stem), the color cube, previews, receipt.
    assert sorted(path.name for path in output.iterdir()) == [
        "B.fits", "G.fits", "L.fits", "LRGB.fits", "R.fits", "details", "previews", "receipt.json",
    ]
    assert sorted(path.name for path in (output / "previews").iterdir()) == [
        "B.png", "G.png", "L.png", "LRGB.png", "LRGB.tiff", "R.png",
    ]
    assert sorted(path.name for path in (output / "details").iterdir()) == [
        "color", "mosaics", "runs", "shared-calibration",
    ]
    for name in ("B", "G", "L", "R"):
        assert fits.getheader(output / f"{name}.fits")["FILTER"] == name
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["execution"]["sharedCalibration"]["receipt"] == "details/shared-calibration/receipt.json"
    assert receipt["execution"]["color"]["receipt"] == "details/color/receipt.json"
    assert {item["relativePath"] for item in receipt["finalProducts"]["guiArtifacts"]} == {
        "B.fits", "G.fits", "L.fits", "R.fits", "LRGB.fits", "previews/B.png", "previews/G.png",
        "previews/L.png", "previews/R.png", "previews/LRGB.png", "previews/LRGB.tiff",
    }
    assert result.product_paths == tuple(
        str(output / name)
        for name in ("B.fits", "G.fits", "L.fits", "R.fits", "LRGB.fits", "previews/LRGB.tiff", "previews/LRGB.png")
    )
    assert result.color_product_path == str(output / "LRGB.fits")
    # A channel whose grid is final is the solved master itself, not a copy:
    # same storage, header untouched.
    alignment = receipt["execution"]["alignment"]
    assert alignment["l"]["mode"] == "REFERENCE_SOLVED_GRID"
    assert alignment["l"]["publication"] == "HARDLINK"
    assert alignment["l"]["referenceFilter"] == "L"
    solved_l = output / receipt["execution"]["finalSolves"]["l"]["output"]
    assert (output / "L.fits").stat().st_ino == solved_l.stat().st_ino
    assert "OAFALGN" not in fits.getheader(output / "L.fits")
    assert (output / "LRGB.fits").stat().st_ino == (output / "details/color/LRGB.fits").stat().st_ino
    assert receipt["execution"]["color"]["publication"] == {
        "linearRgb": "HARDLINK", "previewTiff": "HARDLINK", "previewPng": "HARDLINK",
    }
    # Screening is summed over the four target runs; every frame that needed
    # a decision keeps its target and a preview path inside the project.
    screening = receipt["execution"]["screening"]
    assert screening["counts"] == {"PASS": 44, "REVIEW": 4, "HARD_FAIL": 0}
    assert (screening["admitted"], screening["excluded"]) == (44, 4)
    assert len(screening["frames"]) == 4
    for frame in screening["frames"]:
        assert frame["disposition"] == "REVIEW" and frame["admitted"] is False
        assert frame["target"].startswith("DUNPAI")
        assert frame["path"].startswith("source/")
        assert (output / frame["reviewPreview"]).is_file()
        assert frame["reviewPreview"].startswith("details/runs/")


def test_publish_file_links_and_gives_a_private_solver_output_normal_permissions(tmp_path: Path) -> None:
    import ufwbpp.workflows.project as project_module

    source = tmp_path / "master_light_L_wcs.fits"
    source.write_bytes(b"solved master")
    source.chmod(0o600)
    destination = tmp_path / "L.fits"
    assert project_module._publish_file(source, destination) == "HARDLINK"
    assert destination.stat().st_ino == source.stat().st_ino
    assert destination.stat().st_mode & 0o777 == 0o666 & ~project_module._current_umask()


def test_publish_file_copies_when_the_volume_has_no_hard_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ufwbpp.workflows.project as project_module

    source = tmp_path / "source.bin"
    source.write_bytes(b"solved master")
    destination = tmp_path / "final.bin"

    def refuse_link(src: Any, dst: Any, **kwargs: Any) -> None:
        raise OSError(errno.EPERM, "hard links are not supported")

    monkeypatch.setattr(project_module.os, "link", refuse_link)
    assert project_module._publish_file(source, destination) == "COPY"
    assert destination.read_bytes() == b"solved master"
    assert destination.stat().st_ino != source.stat().st_ino
    with pytest.raises(FileExistsError):
        project_module._publish_file(source, destination)
    monkeypatch.undo()
    with pytest.raises(FileExistsError):
        project_module._publish_file(source, destination)


def test_missing_rgb_channel_publishes_solved_mono_only(tmp_path: Path) -> None:
    inventory, base, _ = _project(tmp_path, filters=("R", "G"))
    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(),),
        panel_runner=FakePanelRunner(),
        mosaic_provider=FakeReproject().provider(),
    )
    assert result.success is True
    assert result.code == "PROJECT_MONO_SUCCEEDED"
    assert result.color_product_path is None
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["execution"]["color"]["status"] == "NOT_CREATED_MISSING_RGB_CHANNELS"
    assert receipt["execution"]["color"]["missingRequiredChannels"] == ["b"]


def test_luminance_channel_participates_in_lrgb_instead_of_being_ignored(tmp_path: Path) -> None:
    inventory, base, _ = _project(tmp_path, filters=("R", "G", "B", "L"))
    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(),),
        panel_runner=FakePanelRunner(),
        mosaic_provider=FakeReproject().provider(),
    )
    assert result.success is True
    with fits.open(result.color_product_path or "") as hdul:
        assert hdul[0].header["OAFLUM"]
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["execution"]["color"]["status"] == "SOLVED_LRGB"
    assert receipt["execution"]["color"]["luminanceParticipated"] is True
    assert receipt["finalProducts"]["luminanceParticipated"] is True


def test_reprojected_channel_uses_reference_quality_and_declares_propagated_provenance(
    tmp_path: Path,
) -> None:
    inventory, base, _ = _project(tmp_path, filters=("R", "G", "B"))
    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(rotate_filter="G"),),
        panel_runner=FakePanelRunner(),
        mosaic_provider=FakeReproject().provider(),
    )
    assert result.success is True
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    green_alignment = receipt["execution"]["alignment"]["g"]
    assert green_alignment["mode"] == "REPROJECTED_INDEPENDENT_SOLVE"
    provenance = green_alignment["astrometryProvenance"]
    assert provenance["type"] == "PROPAGATED_VERIFIED"
    assert provenance["freshSolveOnThisPixelGrid"] is False
    assert provenance["qualityAppliesTo"] == "REFERENCE_SOLVED_WCS_PROPAGATED_TO_VERIFIED_GRID"
    assert provenance["reprojection"]["sampleCount"] == 9
    assert provenance["reprojection"]["maximumNinePointResidualPixelsBeforeAlignment"] > 0.05
    assert provenance["reprojection"]["maximumNinePointResidualPixelsAfterAlignment"] <= 0.05

    artifacts = receipt["finalProducts"]["guiArtifacts"]
    red = next(item for item in artifacts if item.get("filter") == "R")
    green = next(item for item in artifacts if item.get("filter") == "G")
    rgb = next(item for item in artifacts if item["kind"] == "LINEAR_RGB_FITS")
    assert green["astrometry"]["wcsSha256"] == red["astrometry"]["wcsSha256"]
    assert green["astrometry"]["wcsSha256"] == provenance["referenceSolution"]["wcsSha256"]
    assert green["astrometry"]["wcsSha256"] != provenance["sourceSolution"]["wcsSha256"]
    assert green["finalGate"]["freshSolveOnThisPixelGrid"] is False
    assert green["finalGate"]["propagatedReferenceWcsVerified"] is True
    assert rgb["astrometryProvenance"]["type"] == "PROPAGATED_VERIFIED"
    assert rgb["astrometryProvenance"]["freshSolveOnRgbCube"] is False
    assert rgb["finalGate"]["propagatedReferenceWcsVerified"] is True


def test_raw_calibration_library_is_integrated_once_for_all_panels(tmp_path: Path) -> None:
    inventory, supplied_base, _ = _project(tmp_path, filters=("R",))
    raw_biases = tuple(
        _write(
            tmp_path / "input" / "raw-bias" / f"bias-{index}.fits",
            np.full((8, 8), 100 + index - 1, dtype=np.uint16),
            _science_header("calibration", "NONE", role="Bias"),
        )
        for index in range(3)
    )
    raw_flats = tuple(
        _write(
            tmp_path / "input" / "raw-flat" / f"flat-{index}.fits",
            np.full((8, 8), 1100 + index, dtype=np.uint16),
            _science_header("calibration", "R", role="Flat"),
        )
        for index in range(3)
    )
    inventory = inventory_project([tmp_path / "input"])
    base = replace(
        supplied_base,
        bias_files=tuple(str(path) for path in raw_biases),
        flat_files=tuple(str(path) for path in raw_flats),
        master_bias_files=(),
        master_flat_files=(),
    )

    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(),),
        panel_runner=FakePanelRunner(),
        mosaic_provider=FakeReproject().provider(),
    )

    assert result.success is True
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["execution"]["sharedCalibration"]["rawIntegrationCount"] == 1
    assert receipt["execution"]["sharedCalibration"]["mode"] == "BUILT_ONCE_OR_REUSED"
    assert len(receipt["execution"]["subruns"]) == 4


def test_run_project_request_json_binds_all_source_roles_without_cli_path_echo(tmp_path: Path) -> None:
    _inventory, base, _ = _project(tmp_path)
    input_root = tmp_path / "input"
    sources = [
        {
            "sourceId": f"panel-{index}",
            "hostPath": str(input_root / f"dunpai{index}"),
            "expectedRole": "LIGHT",
            "recursive": True,
        }
        for index in range(1, 5)
    ]
    sources.append(
        {
            "sourceId": "bias",
            "hostPath": str(input_root / "masters" / "master-bias.fits"),
            "expectedRole": "MASTER_BIAS",
            "recursive": False,
        }
    )
    for filter_name in ("R", "G", "B"):
        sources.append(
            {
                "sourceId": f"flat-{filter_name}",
                "hostPath": str(input_root / "masters" / f"master-flat-{filter_name}.fits"),
                "expectedRole": "MASTER_FLAT",
                "recursive": False,
            }
        )
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "sources": sources,
                "outputDirectory": base.output_directory,
                "projectName": "Shield four panel",
                "recipe": {},
                "solverHints": {
                    "raDegrees": 150.0,
                    "decDegrees": 20.0,
                    "fieldOfViewDegrees": 0.014,
                },
                "execution": {"workers": 2},
            }
        ),
        encoding="utf-8",
    )

    paths, name, output, _recipe, options = _load_project_request(str(request_path))
    inventory = inventory_project(paths, name=name)
    _verify_expected_source_roles(inventory, options["expectedRoles"])

    assert output == base.output_directory
    assert name == "Shield four panel"
    assert len(classify_project_layout(inventory).panels) == 12
    wrong = dict(options["expectedRoles"])
    wrong[str((input_root / "masters" / "master-bias.fits").resolve())] = "MASTER_FLAT"
    with pytest.raises(Exception, match="expected MASTER_FLAT"):
        _verify_expected_source_roles(inventory, wrong)


def test_panel_exception_detail_reaches_project_result(tmp_path: Path) -> None:
    inventory, base, _ = _project(tmp_path, filters=("R",))

    def failed_panel(*_args: Any, **_kwargs: Any) -> E2EResult:
        raise E2EError("REGISTRATION_FAILED", "at least two frames are required")

    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(),),
        panel_runner=failed_panel,
        mosaic_provider=FakeReproject().provider(),
    )
    receipt = json.loads(Path(result.receipt_path).read_text())
    assert not result.success
    assert not Path(base.output_directory).exists()
    assert result.code == "REGISTRATION_FAILED"
    assert result.serializable()["message"] == receipt["message"]
    assert result.message == "REGISTRATION_FAILED: at least two frames are required"


def test_panel_qc_failure_preserves_reason_and_evidence(tmp_path: Path) -> None:
    inventory, base, _ = _project(tmp_path, filters=("R",))
    detail = "quality gate admitted 1 Light frame; at least 2 are required"

    def failed_panel(request: E2ERequest, **_kwargs: Any) -> E2EResult:
        evidence = Path(request.output_directory + ".unsolved")
        (evidence / "qc").mkdir(parents=True)
        (evidence / "qc" / "manifest.json").write_text('{"frames": []}')
        receipt = evidence / "receipt.json"
        receipt.write_text(json.dumps({"code": "QC_INSUFFICIENT_LIGHTS", "message": detail}))
        return E2EResult(
            success=False,
            code="QC_INSUFFICIENT_LIGHTS",
            state=E2EState.UNSOLVED_WORKING,
            output_directory=None,
            evidence_directory=str(evidence),
            receipt_path=str(receipt),
            product_paths=(),
            preview_paths=(),
            passed_light_paths=request.light_files[:1],
            excluded_light_paths=request.light_files[1:],
            message=detail,
        )

    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(),),
        panel_runner=failed_panel,
        mosaic_provider=FakeReproject().provider(),
    )
    assert result.code == "QC_INSUFFICIENT_LIGHTS"
    assert result.serializable()["message"] == detail
    assert len(result.passed_light_paths) == 1
    assert len(result.excluded_light_paths) == 2
    assert not Path(base.output_directory).exists()
    receipt = json.loads(Path(result.receipt_path).read_text())
    sub_receipt = Path(result.evidence_directory or "") / receipt["execution"]["subruns"][0]["receipt"]
    assert json.loads(sub_receipt.read_text())["message"] == detail
    assert (sub_receipt.parent / "qc" / "manifest.json").is_file()


def test_low_mosaic_coverage_and_final_solver_failure_publish_unsolved_only(tmp_path: Path) -> None:
    inventory, base, _ = _project(tmp_path / "coverage", filters=("R",))
    coverage = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(),),
        panel_runner=FakePanelRunner(),
        mosaic_provider=FakeReproject(sparse_coverage=True).provider(),
    )
    assert coverage.success is False
    assert coverage.code in {
        "MOSAIC_COVERAGE_GATE_FAILED",
        "MOSAIC_OVERLAP_SEAM_GATE_FAILED",
        "MOSAIC_SEAM_EVIDENCE_INVALID",
    }
    assert not Path(base.output_directory).exists()
    assert Path(coverage.evidence_directory or "").is_dir()

    inventory2, base2, _ = _project(tmp_path / "solver", filters=("R",))
    solver = run_project_e2e(
        ProjectE2ERequest(inventory2, base2, base2.output_directory),
        solver_backends=(ManagedCopySolver(fail=True),),
        panel_runner=FakePanelRunner(),
        mosaic_provider=FakeReproject().provider(),
    )
    assert solver.success is False
    assert solver.code == "MOSAIC_FINAL_SOLVE_REQUIRED"
    assert not Path(base2.output_directory).exists()


def test_wrong_channel_wcs_and_source_drift_fail_without_success_tree(tmp_path: Path) -> None:
    inventory, base, _ = _project(tmp_path / "wrong-wcs")
    wrong = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(wrong_filter="G"),),
        panel_runner=FakePanelRunner(),
        mosaic_provider=FakeReproject().provider(),
    )
    assert wrong.success is False
    assert wrong.code in {"CHANNEL_WCS_GROSS_MISMATCH", "SOLVER_CENTER_OUTSIDE_HINT_RADIUS"}
    assert not Path(base.output_directory).exists()

    inventory2, base2, lights = _project(tmp_path / "drift", filters=("R",))
    drift = run_project_e2e(
        ProjectE2ERequest(inventory2, base2, base2.output_directory),
        solver_backends=(ManagedCopySolver(),),
        panel_runner=FakePanelRunner(drift_source=lights[0]),
        mosaic_provider=FakeReproject().provider(),
    )
    assert drift.success is False
    assert drift.code == "SOURCE_CHANGED"
    assert not Path(base2.output_directory).exists()


def test_rgb_sip_cube_publishes_spatial_gui_astrometry(tmp_path: Path) -> None:
    inventory, base, _ = _project(tmp_path)
    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(sip=True),),
        panel_runner=FakePanelRunner(), mosaic_provider=FakeReproject().provider(),
    )
    assert result.success is True
    assert result.code == "PROJECT_COLOR_SUCCEEDED"
    cube_header = fits.getheader(result.color_product_path)
    assert cube_header["NAXIS"] == 3
    assert cube_header["NAXIS3"] == 3
    assert cube_header["A_ORDER"] == cube_header["B_ORDER"] == 2
    receipt = json.loads(Path(result.receipt_path).read_text())
    artifacts = receipt["finalProducts"]["guiArtifacts"]
    rgb = next(item for item in artifacts if item["kind"] == "LINEAR_RGB_FITS")
    red = next(item for item in artifacts if item["kind"] == "SOLVED_MONO_FITS" and item["filter"] == "R")
    assert rgb["astrometry"]["imageShape"] == [14, 14]
    for key in ("centerRaDegrees", "centerDecDegrees", "pixelScaleArcsec", "rotationDegrees", "wcsSha256"):
        assert rgb["astrometry"][key] == red["astrometry"][key]
    assert receipt["finalProducts"]["resultGate"]["status"] == "PASS"


def test_unexpected_final_metadata_error_preserves_completed_products_and_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ufwbpp.workflows.project as project_module
    inventory, base, _ = _project(tmp_path)
    original = project_module._astrometry_gui_evidence
    def fail_rgb_metadata(path: Path, quality: Any) -> Any:
        if path.name == "RGB.fits":
            raise ValueError("injected final RGB metadata failure")
        return original(path, quality)
    monkeypatch.setattr(project_module, "_astrometry_gui_evidence", fail_rgb_metadata)
    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(),), panel_runner=FakePanelRunner(),
        mosaic_provider=FakeReproject().provider(),
    )
    assert result.success is False
    assert result.code == "PROJECT_UNEXPECTED_FAILURE"
    assert "ValueError: injected final RGB metadata failure" in result.message
    assert not Path(base.output_directory).exists()
    evidence = Path(result.evidence_directory)
    assert (evidence / "RGB.fits").is_file()
    assert (evidence / "details/color/RGB.fits").is_file()
    assert len(list((evidence / "details/runs").glob("*/receipt.json"))) == 4
    receipt = json.loads(Path(result.receipt_path).read_text())
    diagnostic = json.loads((evidence / receipt["execution"]["unexpectedFailure"]["diagnostic"]).read_text())
    assert diagnostic["exceptionType"] == "ValueError"
    assert diagnostic["message"] == "injected final RGB metadata failure"
    assert diagnostic["stack"][-1]["function"] == "fail_rgb_metadata"
    assert all(not Path(frame["file"]).is_absolute() for frame in diagnostic["stack"])
    assert receipt["publication"]["successDirectoryPublished"] is False


def test_rgb_gui_astrometry_rejects_noncelestial_cube(tmp_path: Path) -> None:
    cube = _write(tmp_path / "invalid-rgb.fits", np.ones((3, 8, 8)), fits.Header())
    with pytest.raises(ProjectE2EError) as captured:
        _astrometry_gui_evidence(cube, _managed_quality().serializable())
    assert captured.value.code == "FINAL_PRODUCT_WCS_INVALID"


def test_unexpected_error_keeps_private_diagnostics_if_evidence_writer_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ufwbpp.workflows.project as project_module
    inventory, base, _ = _project(tmp_path, filters=("R",))
    def failed_panel(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("unexpected panel failure")
    calls = []
    def failed_evidence(**_kwargs: Any) -> Any:
        calls.append(True)
        raise TypeError("injected receipt serialization failure")
    monkeypatch.setattr(project_module, "_failure_result", failed_evidence)
    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(),), panel_runner=failed_panel,
        mosaic_provider=FakeReproject().provider(),
    )
    assert result.success is False
    assert result.code == "PROJECT_UNEXPECTED_FAILURE"
    assert calls == [True]
    assert "evidence publication failed: TypeError" in result.message
    assert not Path(base.output_directory).exists()
    assert Path(result.evidence_directory).is_dir()
    diagnostic = json.loads(Path(result.receipt_path).read_text())
    assert diagnostic["exceptionType"] == "ValueError"
    assert diagnostic["message"] == "unexpected panel failure"


def _brute_valid_rectangle(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    height, width = mask.shape
    return max(
        ((bottom - top) * (right - left), top, left, bottom, right)
        for top in range(height) for bottom in range(top + 1, height + 1)
        for left in range(width) for right in range(left + 1, width + 1)
        if mask[top:bottom, left:right].all()
    )


def test_final_crop_is_global_maximum_of_common_support_including_luminance(tmp_path: Path) -> None:
    rng = np.random.default_rng(71)
    shape = (9, 12)
    y, x = np.indices(shape)
    # Staircase edges model a rotated footprint; L adds an independent edge.
    masks = {
        "r": np.ones(shape, dtype=bool),
        "g": y >= x // 4,
        "b": x < 11 - y // 4,
        "l": x >= 2,
    }
    originals = {}
    paths = {}
    for key, mask in masks.items():
        data = rng.normal(size=shape).astype(np.float32)
        data[3:5, 4:7] = 0  # Zero and negative sky values remain valid pixels.
        data[~mask] = np.nan
        originals[key] = data.copy()
        paths[key] = _write(tmp_path / f"{key}.fits", data, _output_wcs().to_header())
    expected = _brute_valid_rectangle(np.logical_and.reduce(list(masks.values())))
    result = _crop_final_channels(paths, minimum_retained_fraction=0.25, max_memory_bytes=12 * 16 * 2)
    area, top, left, bottom, right = expected
    assert result["bounds"] == dict(top=top, left=left, bottom=bottom, right=right)
    assert result["retainedPixels"] == area
    assert result["invalidPixelsAfterCrop"] == dict.fromkeys(paths, 0)
    for key, path in paths.items():
        data = fits.getdata(path)
        assert np.isfinite(data).all()
        np.testing.assert_array_equal(data, originals[key][top:bottom, left:right])
        assert np.any(data == 0)


@pytest.mark.parametrize("sip", [False, True])
def test_final_crop_preserves_tan_and_sip_world_coordinates(tmp_path: Path, sip: bool) -> None:
    celestial = _output_wcs(crpix=(22.3, 18.7))
    if sip:
        a, b = np.zeros((4, 4)), np.zeros((4, 4))
        a[2, 0], a[1, 1], b[0, 2], b[2, 1] = 2e-4, 1e-5, -3e-4, 1e-7
        celestial.sip = Sip(a, b, None, None, celestial.wcs.crpix)
        celestial.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    data = np.arange(40 * 50, dtype=np.float32).reshape(40, 50)
    data[:3] = np.nan
    data[:, :4] = np.nan
    data[-2:] = np.nan
    data[:, -5:] = np.nan
    path = _write(tmp_path / "R.fits", data, celestial.to_header(relax=True))
    result = _crop_final_channels({"r": path}, minimum_retained_fraction=0.5, max_memory_bytes=8192)
    assert result["bounds"] == dict(top=3, left=4, bottom=38, right=45)
    with fits.open(path, checksum=True) as hdul:
        assert hdul[0].verify_checksum() == 1
        assert hdul[0].verify_datasum() == 1
        final_wcs = WCS(hdul[0].header)
        np.testing.assert_array_equal(hdul[0].data, data[3:38, 4:45])
        np.testing.assert_allclose(final_wcs.wcs.crpix, celestial.wcs.crpix - [4, 3])
        if sip:
            np.testing.assert_array_equal(final_wcs.sip.a, celestial.sip.a)
            np.testing.assert_array_equal(final_wcs.sip.b, celestial.sip.b)
            np.testing.assert_allclose(final_wcs.sip.crpix, celestial.sip.crpix - [4, 3])
        points = np.random.default_rng(17).uniform([0, 0], [40, 34], size=(100, 2))
        np.testing.assert_allclose(
            final_wcs.all_pix2world(points, 0),
            celestial.all_pix2world(points + [4, 3], 0), rtol=0, atol=1e-11,
        )


def test_final_crop_retained_fraction_guard_does_not_modify_channels(tmp_path: Path) -> None:
    paths = {}
    before = {}
    for key in ("r", "l"):
        data = np.ones((10, 12), dtype=np.float32)
        if key == "l":
            data[:, :8] = np.nan
        path = _write(tmp_path / f"{key}.fits", data, _output_wcs().to_header())
        paths[key], before[key] = path, path.read_bytes()
    with pytest.raises(ProjectE2EError) as captured:
        _crop_final_channels(paths, minimum_retained_fraction=0.5, max_memory_bytes=8192)
    assert captured.value.code == "FINAL_AUTOCROP_TOO_SMALL"
    assert all(path.read_bytes() == before[key] for key, path in paths.items())


def test_project_crops_final_alignment_edges_before_rgb_and_gui_hashes(tmp_path: Path) -> None:
    inventory, base, _ = _project(tmp_path, filters=("R", "G", "B", "L"))
    fake = FakeReproject()

    def corner_footprint(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        science, footprint = fake.interp(*args, **kwargs)
        science[0, -1], footprint[0, -1] = np.nan, 0
        return science, footprint

    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(ManagedCopySolver(rotate_filter="G", sip=True),),
        panel_runner=FakePanelRunner(),
        mosaic_provider=replace(fake.provider(), reproject_function=corner_footprint),
    )
    assert result.success is True
    receipt = json.loads(Path(result.receipt_path).read_text())
    crop = receipt["execution"]["finalCrop"]
    assert crop["applied"] is True
    assert crop["bounds"] == dict(top=1, left=0, bottom=14, right=14)
    assert crop["outputImageShape"] == [13, 14]
    assert crop["invalidPixelsBeforeCrop"] == dict(b=0, g=1, l=0, r=0)
    wcs_hashes = set()
    artifacts = [item for item in receipt["finalProducts"]["guiArtifacts"]
                 if item["kind"] in {"SOLVED_MONO_FITS", "LINEAR_RGB_FITS"}]
    assert len(artifacts) == 5
    for artifact in artifacts:
        path = Path(result.output_directory) / artifact["path"]
        assert artifact["sha256"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert artifact["astrometry"]["imageShape"] == [13, 14]
        assert artifact["astrometryProvenance"]["type"] == "PROPAGATED_VERIFIED"
        wcs_hashes.add(artifact["astrometry"]["wcsSha256"])
        with fits.open(path) as hdul:
            assert hdul[0].data.shape[-2:] == (13, 14)
            assert np.isfinite(hdul[0].data).all()
    assert len(wcs_hashes) == 1
    for alignment in receipt["execution"]["alignment"].values():
        path = Path(result.output_directory) / alignment["output"]["path"].removeprefix("artifact/")
        assert alignment["output"]["sha256"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
