from __future__ import annotations

from ufwbpp.stacking import masters as masters_module
from ufwbpp.workflows import common as common_module
from ufwbpp.workflows import products as products_module
from ufwbpp.workflows import registration as registration_module
from ufwbpp.workflows import screening as screening_module
from ufwbpp.workflows import sources as sources_module
from ufwbpp.stacking import records as records_module

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import stat
import sys
from types import SimpleNamespace
from typing import Any

from astropy.io import fits
import numpy as np
import pytest


# The registration package is a sibling distribution.  The product build
# installs it; repository tests exercise the exact checked-in implementation.
REGISTRATION_SOURCE = Path(__file__).resolve().parents[3] / "packages" / "registration" / "src"
if str(REGISTRATION_SOURCE) not in sys.path:
    sys.path.insert(0, str(REGISTRATION_SOURCE))

from ufwbpp.native_kernels import load_native_kernels
from ufwbpp.workflows.single_target import E2ERequest, E2EState, IntegrationMode, ReviewApproval, run_e2e
from ufwbpp.workflows.contracts import DrizzleOptions, ProgressEvent
from ufwbpp.workflows.single_target import (
    E2EError,
    _refine_solved_wcs,
    _unify_same_grid_solutions,
    _validate_cross_filter_wcs,
)
from ufwbpp.workflows.solve import (
    _SolverHints,
    _inherit_same_grid_solution,
    _solve_one,
    _verify_rewritten_wcs,
    _withdraw_inherited_solutions,
)
from ufwbpp.workflows.registration import _build_registration_masters, _register_lights
from ufwbpp.workflows.screening import _drizzle_sampling_evidence, _inferred_solver_hints
from ufwbpp.workflows.sharing import _share_safe_receipt_core, _share_safe_string
from ufwbpp.workflows.sources import _SourceIdentity
from ufwbpp.solvers.base import (
    AstrometricQuality,
    SolutionKind,
    SolverResult,
    SolverIndexArtifact,
    SolverStatus,
    WcsParity,
)
from ufwbpp.stacking.pipeline import run_portable_pipeline
from ufwbpp.calibration.inputs import MasterMetadataOverride, RawFrameMetadataOverride
from ufwbpp.stacking.parameters import PipelineParameters
from ufwbpp.workflows.project import _build_shared_calibration


def test_late_top_level_receipt_evidence_is_share_safe(tmp_path: Path) -> None:
    source = _SourceIdentity(
        path=str(tmp_path / "private" / "light.fits"),
        role="LIGHT",
        sha256="sha256:" + "1" * 64,
        size_bytes=123,
        mtime_ns=4,
        device=5,
        inode=6,
    )
    receipt = _share_safe_receipt_core(
        {
            "nativeLibrary": {
                "path": str(tmp_path / "build" / "libufwbpp_native.dylib"),
                "sha256": "sha256:" + "2" * 64,
                "device": 7,
                "inode": 8,
                "mtimeNs": 9,
            },
            "sourcePath": source.path,
        },
        staging=tmp_path / "staging",
        identities=(source,),
    )
    serialized = json.dumps(receipt, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert receipt["nativeLibrary"]["path"] == "local-redacted/libufwbpp_native.dylib"
    assert set(receipt["nativeLibrary"]) == {"path", "sha256"}
    assert receipt["sourcePath"].startswith("source/src-")


@pytest.mark.parametrize(
    ("staging", "master_path"),
    [
        (Path("/staging/run"), "/staging/run/work/calibration/master_bias.fits"),
        (PureWindowsPath("C:/staging/run"), "C:\\staging\\run\\work\\calibration\\master_bias.fits"),
        (PureWindowsPath("C:/staging/run"), "C:/staging/run/work/calibration/master_bias.fits"),
    ],
)
def test_generated_artifact_paths_use_portable_separators(staging, master_path) -> None:
    public_path = _share_safe_string(master_path, staging=staging, source_tokens={})
    assert public_path == "artifact/work/calibration/master_bias.fits"
    assert _share_safe_string(str(staging), staging=staging, source_tokens={}) == "artifact/."


def test_receipts_name_the_files_of_their_tree_with_forward_slashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative path prints with backslashes on Windows.  Receipts name a
    file of their tree with forward slashes on every host, so the receipt of
    a run made on Windows resolves anywhere: the writers run here with
    relative paths of the Windows flavour."""

    import pathlib

    import ufwbpp.products.color as color_module

    staging = tmp_path / "run"
    product = staging / "products" / "L" / "master_light_L_wcs.fits"
    product.parent.mkdir(parents=True)
    product.write_bytes(b"pixels")
    relative_to = pathlib.PurePath.relative_to
    monkeypatch.setattr(
        pathlib.PurePath,
        "relative_to",
        lambda self, *other, **options: PureWindowsPath(*relative_to(self, *other, **options).parts),
    )
    assert str(product.relative_to(staging)) == "products\\L\\master_light_L_wcs.fits"

    named = "products/L/master_light_L_wcs.fits"
    assert [item["path"] for item in common_module._artifact_records(staging, (staging / "products",))] == [named]
    (attempt,) = common_module._relativize_solver_attempts(
        [{"result": {"outputPath": str(product)}, "artifact": {"path": str(product)}}], staging
    )
    assert attempt["result"]["outputPath"] == attempt["artifact"]["path"] == named
    assert color_module._artifact(product, staging, "LINEAR_RGB_FITS")["path"] == named


def _header(
    role: str,
    *,
    exposure: float,
    observed_at: str,
    filter_name: str = "R",
) -> fits.Header:
    header = fits.Header()
    header["IMAGETYP"] = role
    header["FILTER"] = filter_name
    header["OBJECT"] = "SYNTHETIC-FIELD"
    header["INSTRUME"] = "SYNTHETIC-CAMERA"
    header["EXPTIME"] = exposure
    header["GAIN"] = 100
    header["OFFSET"] = 20
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    header["READOUTM"] = "MODE-1"
    header["CCD-TEMP"] = -10.0
    header["BAYERPAT"] = "NONE"
    header["DATE-OBS"] = observed_at
    header["RA"] = 150.0
    header["DEC"] = 20.0
    return header


def _write(path: Path, data: np.ndarray, header: fits.Header) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.asarray(data, dtype=np.uint16), header, overwrite=False)
    return path


def _stars(shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    rng = np.random.default_rng(20260901)
    y, x = np.indices(shape, dtype=np.float64)
    image = np.full(shape, 500.0, dtype=np.float64)
    # A stable, uncrowded field with enough profiles for the production gate's
    # >=30-match morphology and registration evidence.
    positions: list[tuple[float, float]] = []
    while len(positions) < 58:
        candidate = (float(rng.uniform(8, 120)), float(rng.uniform(8, 120)))
        if all(np.hypot(candidate[0] - px, candidate[1] - py) >= 8.0 for px, py in positions):
            positions.append(candidate)
    for index, (cx, cy) in enumerate(positions):
        amplitude = 1800.0 + 80.0 * (index % 9)
        image += amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * 0.8**2))
    image += rng.normal(0.0, 2.0, shape)
    return image


def _subpixel_shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Small dependency-free bilinear shift used by the drizzle E2E oracle."""

    height, width = image.shape
    y, x = np.indices(image.shape, dtype=np.float64)
    source_x = x - dx
    source_y = y - dy
    inside = (
        (source_x >= 0.0)
        & (source_x <= width - 1.0)
        & (source_y >= 0.0)
        & (source_y <= height - 1.0)
    )
    safe_x = np.clip(source_x, 0.0, width - 1.0)
    safe_y = np.clip(source_y, 0.0, height - 1.0)
    x0 = np.floor(safe_x).astype(np.int64)
    y0 = np.floor(safe_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = safe_x - x0
    ty = safe_y - y0
    shifted = (
        image[y0, x0] * (1.0 - tx) * (1.0 - ty)
        + image[y0, x1] * tx * (1.0 - ty)
        + image[y1, x0] * (1.0 - tx) * ty
        + image[y1, x1] * tx * ty
    )
    shifted[~inside] = float(np.median(image))
    return shifted


@pytest.fixture(scope="module")
def synthetic_project(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, ...]]:
    root = tmp_path_factory.mktemp("e2e-synthetic")
    shape = (128, 128)
    rng = np.random.default_rng(73)
    y, x = np.indices(shape, dtype=np.float64)
    response = 0.82 + 0.18 * (1.0 - ((x - 63.5) ** 2 + (y - 63.5) ** 2) / (2 * 92.0**2))
    response = np.clip(response, 0.72, 1.0)
    bias_level = 1000.0
    dark_signal = 14.0

    biases = tuple(
        _write(
            root / "BIAS" / f"bias_{index:02d}.fits",
            bias_level + rng.normal(0.0, 1.0, shape),
            _header("Bias", exposure=0.001, observed_at="2026-01-01T18:00:00Z"),
        )
        for index in range(3)
    )
    darks = tuple(
        _write(
            root / "DARK" / f"dark_{index:02d}.fits",
            bias_level + dark_signal + rng.normal(0.0, 1.2, shape),
            _header("Dark", exposure=60.0, observed_at="2026-01-01T18:10:00Z"),
        )
        for index in range(3)
    )
    flats = tuple(
        _write(
            root / "FLAT" / f"flat_R_{index:02d}.fits",
            bias_level + 24_000.0 * response + rng.normal(0.0, 3.0, shape),
            _header("Flat", exposure=2.0, observed_at="2026-01-01T18:20:00Z"),
        )
        for index in range(3)
    )

    base = _stars(shape)
    shifts = (
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
        (0.1, 0.4),
    )
    lights: list[Path] = []
    for index, (dx, dy) in enumerate(shifts):
        shifted = _subpixel_shift(base, dx, dy)
        raw = bias_level + dark_signal + shifted * response + rng.normal(0.0, 1.0, shape)
        # The ninth frame is deliberately a singleton observing night.  Its
        # pixels are healthy, but production GatePolicy must label provenance
        # insufficient as REVIEW and the orchestrator must not integrate it.
        observed = (
            f"2026-01-01T20:{index:02d}:00Z"
            if index < 8
            else "2026-01-03T20:00:00Z"
        )
        lights.append(
            _write(
                root / "LIGHT" / f"light_R_{index:02d}.fits",
                raw,
                _header("Light", exposure=60.0, observed_at=observed),
            )
        )

    # Prove the pipeline can consume genuinely read-only caller-owned inputs.
    for path in (*biases, *darks, *flats, *lights):
        path.chmod(0o444)
    return {
        "biases": biases,
        "darks": darks,
        "flats": flats,
        "lights": tuple(lights),
    }


class FakeSolver:
    backend_id = "fake-catalog-solver"

    def __init__(
        self,
        *,
        center_ra_degrees: float = 150.0,
        matched_stars: int = 30,
        rms_arcsec: float = 0.5,
        field_width_degrees: float = 3.0,
        receipt_drift: bool = False,
        provide_quality: bool = True,
    ) -> None:
        self.inputs: list[str] = []
        self.center_ra_degrees = center_ra_degrees
        self.matched_stars = matched_stars
        self.rms_arcsec = rms_arcsec
        self.field_width_degrees = field_width_degrees
        self.receipt_drift = receipt_drift
        self.provide_quality = provide_quality

    def solve(self, request: Any) -> SolverResult:
        self.inputs.append(request.input_path)
        with fits.open(request.input_path, mode="readonly", memmap=False) as hdul:
            image_hdu = next(item for item in hdul if item.data is not None and item.data.ndim == 2)
            height, width = image_hdu.data.shape
            primary = hdul[0].header
            primary["CTYPE1"] = "RA---TAN"
            primary["CTYPE2"] = "DEC--TAN"
            primary["CRPIX1"] = (width + 1.0) / 2.0
            primary["CRPIX2"] = (height + 1.0) / 2.0
            primary["CRVAL1"] = self.center_ra_degrees
            primary["CRVAL2"] = 20.0
            pixel_scale_degrees = self.field_width_degrees / (width - 1)
            primary["CD1_1"] = -pixel_scale_degrees
            primary["CD1_2"] = 0.0
            primary["CD2_1"] = 0.0
            primary["CD2_2"] = pixel_scale_degrees
            primary["OAFSTATE"] = "SOLVED"
            hdul.writeto(request.output_path, overwrite=False, checksum=True)
            solved_header = primary.copy()
        quality = (
            AstrometricQuality(
                matched_stars=self.matched_stars,
                rms_pixels=self.rms_arcsec / (pixel_scale_degrees * 3600.0),
                rms_arcsec=self.rms_arcsec,
                parity=WcsParity.NEGATIVE,
                catalog_identity="1" * 64,
                index_identities=("astrometry.net:index:4200:healpix:-1:hpnside:0",),
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
            if self.provide_quality
            else None
        )
        receipt_quality = quality.serializable() if quality is not None else {
            "status": "UNAVAILABLE"
        }
        if self.receipt_drift and quality is not None:
            receipt_quality = {**receipt_quality, "matchedStars": quality.matched_stars + 1}
        return SolverResult(
            backend_id=self.backend_id,
            status=SolverStatus.SOLVED,
            solution_kind=SolutionKind.SOLVED,
            backend_confirmed=True,
            header=solved_header,
            image_shape=(height, width),
            output_path=request.output_path,
            astrometric_quality=quality,
            evidence={
                "fakeReceiptVerified": True,
                "astrometricQuality": receipt_quality,
            },
        )

    def verify_result(self, result: SolverResult) -> bool:
        expected_quality = (
            result.astrometric_quality.serializable()
            if result.astrometric_quality is not None
            else {"status": "UNAVAILABLE"}
        )
        return (
            bool(result.evidence.get("fakeReceiptVerified"))
            and result.evidence.get("astrometricQuality") == expected_quality
            and Path(result.output_path or "").is_file()
        )


class FailingSolver:
    backend_id = "fake-failing-solver"

    def solve(self, request: Any) -> SolverResult:
        return SolverResult(
            backend_id=self.backend_id,
            status=SolverStatus.FAILED,
            solution_kind=SolutionKind.NONE,
            backend_confirmed=False,
            error="injected catalog miss",
        )

    def verify_result(self, result: SolverResult) -> bool:
        return False


def _request(
    project: dict[str, tuple[Path, ...]],
    output: Path,
    *,
    lights: tuple[Path, ...] | None = None,
    mode: IntegrationMode = IntegrationMode.ORDINARY,
) -> E2ERequest:
    return E2ERequest(
        light_files=tuple(str(path) for path in (lights or project["lights"][:8])),
        flat_files=tuple(str(path) for path in project["flats"]),
        dark_files=tuple(str(path) for path in project["darks"]),
        bias_files=tuple(str(path) for path in project["biases"]),
        output_directory=str(output),
        integration_mode=mode,
        workers=2,
        ra_hint_degrees=150.0,
        dec_hint_degrees=20.0,
        field_of_view_degrees=3.0,
        search_radius_degrees=5.0,
        drizzle=DrizzleOptions(scale=2, pixfrac=0.9, kernel="square", tile_rows=64),
    )


def _all_source_bytes(project: dict[str, tuple[Path, ...]]) -> dict[Path, bytes]:
    return {path: path.read_bytes() for values in project.values() for path in values}


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def test_solver_hints_derive_nina_fov_and_explicit_coordinates_win_over_the_header(
    tmp_path: Path,
) -> None:
    request = E2ERequest(
        light_files=("unused.fits",),
        flat_files=("unused-flat.fits",),
        bias_files=("unused-bias.fits",),
        output_directory=str(tmp_path / "unused"),
        ra_hint_degrees=12.0,
        dec_hint_degrees=55.0,
        field_of_view_degrees=1.95,
        search_radius_degrees=5.0,
    )
    frames = tuple(
        SimpleNamespace(
            metadata=SimpleNamespace(
                ra_degrees=282.14 + offset,
                dec_degrees=-7.40,
                width=6252,
                height=4176,
                header={
                    "FOCALLEN": 1200.0,
                    "XPIXSZ": 3.76,
                    "YPIXSZ": 3.76,
                    "XBINNING": 1,
                    "YBINNING": 1,
                },
            )
        )
        for offset in (-0.01, 0.0, 0.01)
    )

    hints = _inferred_solver_hints(request, [frame.metadata for frame in frames])

    # The header consensus (282.14, -7.40) is recorded; the user's coordinates seed the solver.
    assert (hints.ra_degrees, hints.dec_degrees) == (12.0, 55.0)
    assert hints.evidence["coordinates"]["metadata"]["raDegrees"] == pytest.approx(282.14, abs=1e-6)
    assert hints.field_of_view_degrees == pytest.approx(1.122, abs=0.002)
    assert hints.evidence["coordinates"]["consistency"] == "CONFLICT"
    assert hints.evidence["coordinates"]["selection"] == "explicit"
    assert "conflicting-NINA/FITS-header-ignored" in hints.provenance
    assert hints.evidence["fieldOfView"]["consistency"] == "CONFLICT"
    assert hints.evidence["fieldOfView"]["selection"] == "derived"
    assert "conflicting-explicit-FOV-ignored" in hints.provenance


def test_solver_hints_retain_consistent_explicit_nina_constraints(tmp_path: Path) -> None:
    request = E2ERequest(
        light_files=("unused.fits",),
        flat_files=("unused-flat.fits",),
        bias_files=("unused-bias.fits",),
        output_directory=str(tmp_path / "unused"),
        ra_hint_degrees=282.15,
        dec_hint_degrees=-7.41,
        field_of_view_degrees=1.12,
        search_radius_degrees=5.0,
    )
    frame = SimpleNamespace(
        metadata=SimpleNamespace(
            ra_degrees=282.14,
            dec_degrees=-7.40,
            width=6252,
            height=4176,
            header={"FOCALLEN": 1200.0, "XPIXSZ": 3.76, "YPIXSZ": 3.76},
        )
    )

    hints = _inferred_solver_hints(request, (frame.metadata,))

    assert hints.ra_degrees == 282.15
    assert hints.dec_degrees == -7.41
    assert hints.field_of_view_degrees == 1.12
    assert hints.evidence["coordinates"]["consistency"] == "CONSISTENT"
    assert hints.evidence["fieldOfView"]["consistency"] == "CONSISTENT"


@pytest.mark.skipif(
    os.environ.get("UFWBPP_RUN_REAL_METAL") != "1",
    reason="opt-in test requires a real Apple Metal device",
)
def test_real_m3_pro_e2e_receipt_proves_metal_production_path(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    output = tmp_path / "real-metal-e2e"
    request = _request(synthetic_project, output)
    # Auto now prefers native CPU kernels; this opt-in test must explicitly
    # request the Metal adapter whose execution evidence it verifies.
    request = replace(request, pipeline_parameters=replace(
        request.pipeline_parameters, ordinary_integration_backend="m3-pro-tuned"
    ))
    result = run_e2e(
        request,
        solver_backends=(FakeSolver(),),
    )

    assert result.success is True
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    execution = receipt["integration"]["ordinaryExecutions"]["R"]
    assert execution["selectedBackend"] == "m3-pro-tuned"
    assert execution["acceleratorUsed"] is True
    assert "M3 Pro" in execution["deviceName"]
    assert execution["parityGate"]["passed"] is True
    assert execution["fastMath"] is False


def test_complete_raw_to_verified_wcs_success_is_atomic_and_read_only(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    before = _all_source_bytes(synthetic_project)
    progress: list[ProgressEvent] = []
    solver = FakeSolver()
    output = tmp_path / "ordinary-solved"

    result = run_e2e(
        _request(synthetic_project, output),
        solver_backends=(solver,),
        progress=progress.append,
    )

    assert result.success is True
    assert result.state is E2EState.SOLVED
    assert result.output_directory == str(output)
    assert result.evidence_directory is None
    assert len(result.product_paths) == 1
    assert len(result.preview_paths) == 1
    assert not (tmp_path / "ordinary-solved.unsolved").exists()
    with fits.open(result.product_paths[0]) as hdul:
        assert hdul[0].header["OAFSTATE"] == "SOLVED"
        assert hdul[0].header["CTYPE1"] == "RA---TAN"
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["success"] is True
    assert receipt["state"] == "SOLVED"
    assert receipt["astrometry"]["requiredForSuccess"] is True
    assert receipt["astrometry"]["filters"]["R"]["status"] == "SOLVED"
    assert receipt["astrometry"]["qualityPolicy"] == {
        "minMatches": 12,
        "maxRmsArcsec": 2.0,
    }
    assert receipt["astrometry"]["crossFilterValidation"]["valid"] is True
    # The fake solver has no catalog stars to refit on.
    assert receipt["astrometry"]["catalogRefinement"] == {
        "status": "NOT_APPLICABLE",
        "reason": "no managed star catalog",
    }
    accepted_attempt = receipt["astrometry"]["filters"]["R"]["attempts"][-1]
    assert accepted_attempt["result"]["astrometricQuality"]["matchedStars"] == 30
    assert accepted_attempt["hintValidation"]["valid"] is True
    qc_manifest = json.loads(
        (output / "qc" / "manifest.json").read_text(encoding="utf-8")
    )
    assert isinstance(qc_manifest, dict)
    assert qc_manifest["counts"] == {"HARD_FAIL": 0, "PASS": 8, "REVIEW": 0}
    assert len(qc_manifest["frames"]) == 8
    assert qc_manifest["gatePolicy"]["minimum_pass_group_frames"] == 8
    assert qc_manifest["gatePolicyDigest"].startswith("sha256:")
    assert (output / "coverage" / "coverage.json").is_file()
    coverage = json.loads(
        (output / "coverage" / "coverage.json").read_text(encoding="utf-8")
    )["filters"]["R"]
    assert set(coverage["maps"]) == {
        "acceptedSampleCount",
        "coverageFraction",
        "rejectionCount",
    }
    assert all((output / relative).is_file() for relative in coverage["maps"].values())
    registration_receipt = json.loads(
        (output / "receipts" / "registration.json").read_text(encoding="utf-8")
    )
    pipeline_receipt = json.loads(
        (output / "receipts" / "pixel-pipeline.json").read_text(encoding="utf-8")
    )
    calibration = pipeline_receipt["statistics"]["calibration"]
    for key in ("masterBias", "masterDark:60", "masterFlat:R"):
        assert calibration[key]["mode"] == "REUSED_E2E_GENERATED_MASTER"
        assert calibration[key]["calibrationApplied"] is False
        assert calibration[key]["doubleBiasSubtraction"] is False
    assert calibration["masterDark:60"]["biasIncluded"] is True
    assert calibration["masterFlat:R"]["applicationScale"] == 1.0
    trusted_reuse = pipeline_receipt["trustedGeneratedCalibration"]
    assert trusted_reuse["status"] == "REUSED_E2E_GENERATED_MASTER"
    assert trusted_reuse["upstreamRegistrationCalibrationReceiptSha256"].startswith(
        "sha256:"
    )
    assert trusted_reuse["sourceContentManifestSha256"].startswith("sha256:")
    assert trusted_reuse["privateExecutionManifestBound"] is True
    assert trusted_reuse["upstreamRegistrationCalibrationReceiptSha256"] == (
        _content_sha256(output / "receipts" / "registration-calibration.json")
    )
    assert {item["role"] for item in trusted_reuse["originalRawSourceProvenance"]} == {
        "BIAS",
        "DARK",
        "FLAT",
    }
    integration = pipeline_receipt["statistics"]["integrationGroups"]["R"][
        "integration"
    ]
    source_order = [
        item["path"] for item in pipeline_receipt["inputs"] if item["role"] == "LIGHT"
    ]
    expected_quality = np.asarray(
        [registration_receipt["qualityWeightsBySource"][path] for path in source_order]
    )
    expected_quality /= np.sum(expected_quality)
    np.testing.assert_allclose(
        integration["weightComponents"]["registrationQuality"],
        expected_quality,
        atol=1e-12,
    )
    assert {event.stage.value for event in progress} >= {
        "quality-control",
        "calibration",
        "registration",
        "integration",
        "astrometry",
        "preview",
        "verify",
        "publish",
        "complete",
    }
    assert all(path.read_bytes() == content for path, content in before.items())
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in before)
    for public_json in output.rglob("*.json"):
        serialized = public_json.read_text(encoding="utf-8")
        assert str(tmp_path) not in serialized
        assert '"device"' not in serialized
        assert '"inode"' not in serialized
        assert '"mtimeNs"' not in serialized


def test_single_field_generated_master_reuse_matches_raw_rebuild_pixel_oracle(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ufwbpp.workflows.integration as integration_module

    e2e_master_integrations: list[str] = []
    e2e_master_sources: list[str] = []
    pixel_master_integrations: list[str] = []
    captured: dict[str, Any] = {}
    original_e2e_integrate = registration_module.integrate_expressions
    original_pixel_integrate = masters_module.integrate_expressions
    original_pixel_run = integration_module.run_portable_pipeline_fits

    def count_e2e_integrations(expressions: Any, output: Path, **kwargs: Any) -> Any:
        materialized = tuple(expressions)
        if Path(output).name.startswith("master_"):
            e2e_master_integrations.append(Path(output).name)
            e2e_master_sources.extend(item.source_path for item in materialized)
        return original_e2e_integrate(materialized, output, **kwargs)

    def count_pixel_integrations(expressions: Any, output: Path, **kwargs: Any) -> Any:
        if Path(output).name.startswith("master_"):
            pixel_master_integrations.append(Path(output).name)
        return original_pixel_integrate(expressions, output, **kwargs)

    def capture_pixel_run(**kwargs: Any) -> Any:
        aliases = kwargs["_source_aliases"]
        captured["transforms"] = {
            str(aliases[path]): value for path, value in kwargs["transforms"].items()
        }
        captured["quality_weights"] = {
            str(aliases[path]): value
            for path, value in kwargs["quality_weights"].items()
        }
        captured["stellar_scale_hints"] = {
            str(aliases[path]): replace(
                hint,
                source_path=str(aliases[path]),
                reference_path=str(aliases[hint.reference_path]),
            )
            for path, hint in kwargs["stellar_scale_hints"].items()
        }
        captured["trusted"] = kwargs["_trusted_generated_calibration"]
        return original_pixel_run(**kwargs)

    monkeypatch.setattr(registration_module, "integrate_expressions", count_e2e_integrations)
    monkeypatch.setattr(masters_module, "integrate_expressions", count_pixel_integrations)
    monkeypatch.setattr(integration_module, "run_portable_pipeline_fits", capture_pixel_run)

    source_bytes_before = _all_source_bytes(synthetic_project)
    output = tmp_path / "reuse-e2e"
    result = run_e2e(
        _request(synthetic_project, output),
        solver_backends=(FakeSolver(),),
    )
    assert result.success is True
    assert sorted(e2e_master_integrations) == [
        "master_bias.fits",
        "master_dark_60s.fits",
        "master_flat_R.fits",
    ]
    assert pixel_master_integrations == []
    assert captured["trusted"] is not None
    assert e2e_master_sources
    # FITS sources are consumed in place through read-only handles: every
    # master was integrated directly from an original, and no private copy of
    # any original was created or left behind.
    original_source_paths = {
        str(path) for paths in synthetic_project.values() for path in paths
    }
    assert set(e2e_master_sources) <= original_source_paths
    assert _all_source_bytes(synthetic_project) == source_bytes_before
    registration_receipt = json.loads(
        (output / "receipts" / "registration.json").read_text(encoding="utf-8")
    )
    pixel_receipt = json.loads(
        (output / "receipts" / "pixel-pipeline.json").read_text(encoding="utf-8")
    )
    registered_hints = {
        item["sourceSha256"]: item
        for item in registration_receipt["stellarScaleHints"]
    }
    normalized_frames = pixel_receipt["statistics"]["integrationGroups"]["R"][
        "globalNormalization"
    ]["evidence"]["frames"]
    assert sum(frame["mode"] == "REFERENCE_IDENTITY" for frame in normalized_frames) == 1
    for frame in normalized_frames:
        stellar = frame["evidence"]["stellarScale"]
        assert stellar["sourceSha256"] in registered_hints
        assert stellar["scale"] == registered_hints[stellar["sourceSha256"]]["scale"]
        assert stellar["evidence"]["scaleDomain"] == "post-linear-exposure-normalization"
        assert frame["evidence"]["backgroundCovarianceUsedForScale"] is False

    # The previous behavior is retained as an explicit raw-rebuild oracle.  It
    # receives the exact transforms and quality weights consumed by the reuse
    # path, so every output pixel and integration map must be identical.
    oracle = tmp_path / "raw-rebuild-oracle"
    run_portable_pipeline(
        bias_files=synthetic_project["biases"],
        dark_files=synthetic_project["darks"],
        flat_files=synthetic_project["flats"],
        light_files=synthetic_project["lights"][:8],
        output_directory=oracle,
        transforms=captured["transforms"],
        quality_weights=captured["quality_weights"],
        stellar_scale_hints=captured["stellar_scale_hints"],
        parameters=PipelineParameters(),
    )
    with fits.open(result.product_paths[0], memmap=False) as reused, fits.open(
        oracle / "masters" / "master_light_R.fits", memmap=False
    ) as rebuilt:
        np.testing.assert_array_equal(reused[0].data, rebuilt[0].data)
    coverage = json.loads(
        (output / "coverage" / "coverage.json").read_text(encoding="utf-8")
    )["filters"]["R"]["maps"]
    oracle_maps = {
        "acceptedSampleCount": oracle / "coverage" / "R_acceptedSampleCount.fits",
        "coverageFraction": oracle / "coverage" / "R_coverageFraction.fits",
        "rejectionCount": oracle / "coverage" / "R_rejectionCount.fits",
    }
    for name, oracle_path in oracle_maps.items():
        with fits.open(output / coverage[name], memmap=False) as reused, fits.open(
            oracle_path, memmap=False
        ) as rebuilt:
            np.testing.assert_array_equal(reused[0].data, rebuilt[0].data)


@pytest.mark.parametrize(
    ("target", "expected_code"),
    (
        ("master", "TRUSTED_GENERATED_MASTER_CHANGED"),
        ("receipt", "TRUSTED_GENERATED_CALIBRATION_RECEIPT_CHANGED"),
        # FITS sources are consumed in place (no private byte copy), so a
        # Light mutated between the trust handoff and pixel consumption is
        # caught by its captured stat identity as a source change.
        ("source", "SOURCE_CHANGED"),
    ),
)
def test_single_field_generated_master_handoff_detects_tampering(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    expected_code: str,
) -> None:
    import ufwbpp.workflows.integration as integration_module

    project = synthetic_project
    if target == "source":
        # Never mutate the shared module fixture; tamper a private copy.
        copied: dict[str, tuple[Path, ...]] = {}
        for role, paths in synthetic_project.items():
            values: list[Path] = []
            for source in paths:
                destination = tmp_path / "tamper-input" / role / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                destination.chmod(0o644)
                values.append(destination)
            copied[role] = tuple(values)
        project = copied
    original = integration_module.run_portable_pipeline_fits

    def tamper_before_consume(**kwargs: Any) -> Any:
        trusted = kwargs["_trusted_generated_calibration"]
        path = Path(
            trusted.master_flats[0].path
            if target == "master"
            else (
                trusted.upstream_receipt_path
                if target == "receipt"
                else kwargs["light_files"][0]
            )
        )
        with path.open("ab") as stream:
            stream.write(b"tamper")
        return original(**kwargs)

    monkeypatch.setattr(
        integration_module, "run_portable_pipeline_fits", tamper_before_consume
    )
    output = tmp_path / f"tampered-{target}"
    with pytest.raises(E2EError) as raised:
        run_e2e(
            _request(project, output),
            solver_backends=(FakeSolver(),),
        )
    assert raised.value.code == expected_code
    assert not output.exists()


def test_e2e_reuses_identity_bound_digests_and_excludes_review_light(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    all_sources = tuple(
        path for values in synthetic_project.values() for path in values
    )
    source_paths = {str(path.resolve(strict=True)) for path in all_sources}
    request = replace(
        _request(
            synthetic_project,
            tmp_path / "fresh-source-hash-e2e",
            lights=synthetic_project["lights"],
        ),
        pipeline_parameters=replace(
            PipelineParameters(),
            raw_frame_metadata_overrides=tuple(
                RawFrameMetadataOverride(_content_sha256(path), "NONE")
                for path in all_sources
            ),
        ),
    )
    original_e2e_hash = sources_module.sha256_digest
    original_pixel_hash = records_module.hash_calibration_file
    e2e_source_hashes: list[str] = []
    pixel_source_hashes: list[str] = []

    def count_e2e(path: Path) -> str:
        canonical = str(path.resolve(strict=True))
        if canonical in source_paths:
            e2e_source_hashes.append(canonical)
        return original_e2e_hash(path)

    def count_pixel(path: Path) -> str:
        canonical = str(path.resolve(strict=True))
        if canonical in source_paths:
            pixel_source_hashes.append(canonical)
        return original_pixel_hash(path)

    monkeypatch.setattr(common_module, "sha256_digest", count_e2e)
    monkeypatch.setattr(sources_module, "sha256_digest", count_e2e)
    monkeypatch.setattr(registration_module, "sha256_digest", count_e2e)
    monkeypatch.setattr(products_module, "sha256_digest", count_e2e)
    from ufwbpp.calibration import inputs as calibration_inputs
    monkeypatch.setattr(records_module, "hash_calibration_file", count_pixel)
    monkeypatch.setattr(calibration_inputs, "hash_calibration_file", count_pixel)
    result = run_e2e(request, solver_backends=(FakeSolver(),))

    assert result.success is True
    assert len(result.passed_light_paths) == 8
    assert len(result.excluded_light_paths) == 1
    passed_source_paths = {
        str(path.resolve(strict=True))
        for path in (
            *synthetic_project["biases"],
            *synthetic_project["darks"],
            *synthetic_project["flats"],
            *synthetic_project["lights"][:8],
        )
    }
    review_path = str(synthetic_project["lights"][8].resolve(strict=True))
    # The pixel pipeline receives the inventory digests bound to each source's
    # stat identity, so it never rereads an original merely to rehash it.
    assert pixel_source_hashes == []
    assert review_path not in pixel_source_hashes
    assert passed_source_paths <= source_paths
    assert set(e2e_source_hashes) == source_paths
    # One inventory hash plus one deliberate final publication hash.  Override
    # lookup, source selection, and generated-master handoff reuse the captured
    # path/stat-bound digest instead of rereading originals.
    assert all(e2e_source_hashes.count(path) == 2 for path in source_paths)


def test_final_source_rehash_detects_content_drift_with_restored_stat_identity(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    copied: dict[str, tuple[Path, ...]] = {}
    for role, paths in synthetic_project.items():
        values: list[Path] = []
        for path in paths:
            destination = tmp_path / "drift-input" / role / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            destination.chmod(0o644)
            values.append(destination)
        copied[role] = tuple(values)
    drifted_source = copied["lights"][0]

    class MutatingSolver(FakeSolver):
        def solve(self, request: Any) -> SolverResult:
            result = super().solve(request)
            before = drifted_source.stat(follow_symlinks=False)
            with fits.open(drifted_source, mode="update", memmap=False) as hdul:
                hdul[0].data[0, 0] = np.uint16(int(hdul[0].data[0, 0]) + 1)
                hdul.flush(output_verify="exception")
            os.utime(
                drifted_source,
                ns=(before.st_atime_ns, before.st_mtime_ns),
                follow_symlinks=False,
            )
            after = drifted_source.stat(follow_symlinks=False)
            assert (
                after.st_size,
                after.st_mtime_ns,
                after.st_dev,
                after.st_ino,
            ) == (
                before.st_size,
                before.st_mtime_ns,
                before.st_dev,
                before.st_ino,
            )
            return result

    output = tmp_path / "source-drift-output"
    with pytest.raises(E2EError) as raised:
        run_e2e(_request(copied, output), solver_backends=(MutatingSolver(),))

    assert raised.value.code == "SOURCE_CHANGED"
    assert not output.exists()
    assert not output.with_name(output.name + ".unsolved").exists()


def test_qc_review_is_manifested_and_never_enters_pixel_pipeline(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    output = tmp_path / "review-excluded"

    result = run_e2e(
        _request(synthetic_project, output, lights=synthetic_project["lights"]),
        solver_backends=(FakeSolver(),),
    )

    review_path = str(synthetic_project["lights"][8])
    assert result.success is True
    assert review_path in result.excluded_light_paths
    assert review_path not in result.passed_light_paths
    manifest = json.loads((output / "qc" / "manifest.json").read_text(encoding="utf-8"))
    frame = next(
        item
        for item in manifest["frames"]
        if item["path"].endswith("/" + Path(review_path).name)
    )
    assert frame["qualityGate"]["disposition"] == "REVIEW"
    assert any(
        item["code"] == "GATE_INSUFFICIENT_NIGHT_BASELINE"
        for item in frame["qualityGate"]["evidence"]
    )
    # The excluded frame gets a bounded review preview and a compact
    # screening record in the receipt, so the desktop can show why it is out
    # without reading the manifest or the raw file.
    assert frame["reviewPreviewPath"].startswith("qc/review/")
    preview = output / frame["reviewPreviewPath"]
    assert preview.is_file() and preview.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert 0 < preview.stat().st_size <= 512 * 1024
    assert all(item.get("reviewPreviewPath") is None for item in manifest["frames"] if item["qualityGate"]["disposition"] == "PASS")
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    # The SEP determinism self-test is part of every run's evidence.
    source_extraction = receipt["execution"]["sourceExtraction"]
    assert source_extraction["version"] == "sep-extraction-self-test-v1"
    assert source_extraction["deterministic"] is True and source_extraction["warnings"] == []
    screening = receipt["qualityControl"]["screening"]
    assert screening["counts"] == {"PASS": 8, "REVIEW": 1, "HARD_FAIL": 0}
    assert (screening["admitted"], screening["excluded"]) == (8, 1)
    [excluded] = screening["frames"]
    assert excluded["path"].endswith("/" + Path(review_path).name)
    assert excluded["disposition"] == "REVIEW" and excluded["admitted"] is False
    assert excluded["reviewPreview"] == frame["reviewPreviewPath"]
    assert excluded["summary"] and excluded["evidence"]
    pipeline_receipt = json.loads(
        (output / "receipts" / "pixel-pipeline.json").read_text(encoding="utf-8")
    )
    integrated_sources = {
        item["path"] for item in pipeline_receipt["inputs"] if item["role"] == "LIGHT"
    }
    assert review_path not in integrated_sources
    assert {Path(path).name for path in integrated_sources} == {
        Path(path).name for path in result.passed_light_paths
    }


def test_hash_bound_unknown_cfa_confirmation_reaches_registration_and_pixels(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    original = synthetic_project["lights"][0]
    light = tmp_path / "unknown-cfa-light.fits"
    shutil.copyfile(original, light)
    with fits.open(light, mode="update") as hdul:
        del hdul[0].header["BAYERPAT"]
        hdul.flush()
    light.chmod(0o444)
    project = {
        **synthetic_project,
        "lights": (light, *synthetic_project["lights"][1:]),
    }
    missing_output = tmp_path / "unknown-cfa-missing"
    with pytest.raises(E2EError) as missing:
        run_e2e(
            _request(project, missing_output),
            solver_backends=(FakeSolver(),),
        )
    assert missing.value.code == "CFA_CONFIRMATION_REQUIRED"
    assert not missing_output.exists()

    confirmed_output = tmp_path / "unknown-cfa-confirmed"
    base = _request(project, confirmed_output)
    confirmed = replace(
        base,
        pipeline_parameters=replace(
            base.pipeline_parameters,
            raw_frame_metadata_overrides=(
                RawFrameMetadataOverride(
                    source_sha256=_content_sha256(light),
                    cfa_pattern="NONE",
                ),
            ),
        ),
    )
    result = run_e2e(confirmed, solver_backends=(FakeSolver(),))
    assert result.success is True
    pixel_receipt = json.loads(
        (confirmed_output / "receipts" / "pixel-pipeline.json").read_text(
            encoding="utf-8"
        )
    )
    assert pixel_receipt["parameters"]["rawFrameMetadataOverrides"] == [
        {
            "sourceSha256": _content_sha256(light),
            "cfaPattern": "NONE",
        }
    ]


def test_all_raw_roles_require_cfa_confirmation_before_registration_calibration(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    project: dict[str, tuple[Path, ...]] = {}
    for role, paths in synthetic_project.items():
        copied: list[Path] = []
        for index, source in enumerate(paths):
            destination = tmp_path / "all-raw-unknown" / role / f"{index:03d}.fits"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            with fits.open(destination, mode="update") as hdul:
                if "BAYERPAT" in hdul[0].header:
                    del hdul[0].header["BAYERPAT"]
                hdul.flush()
            destination.chmod(0o444)
            copied.append(destination)
        project[role] = tuple(copied)

    base = _request(project, tmp_path / "all-raw-confirmed")
    light_digests = {
        _content_sha256(Path(path)) for path in base.light_files
    }
    partial = replace(
        base,
        output_directory=str(tmp_path / "all-raw-partial"),
        pipeline_parameters=replace(
            base.pipeline_parameters,
            raw_frame_metadata_overrides=tuple(
                RawFrameMetadataOverride(digest, "NONE")
                for digest in sorted(light_digests)
            ),
        ),
    )
    with pytest.raises(E2EError) as incomplete:
        run_e2e(partial, solver_backends=(FakeSolver(),))
    assert incomplete.value.code == "CFA_CONFIRMATION_REQUIRED"
    assert not Path(partial.output_directory).exists()

    all_raw_paths = tuple(
        Path(path)
        for values in (
            base.light_files,
            base.flat_files,
            base.dark_files,
            base.bias_files,
        )
        for path in values
    )
    confirmed = replace(
        base,
        pipeline_parameters=replace(
            base.pipeline_parameters,
            raw_frame_metadata_overrides=tuple(
                RawFrameMetadataOverride(_content_sha256(path), "NONE")
                for path in all_raw_paths
            ),
        ),
    )
    result = run_e2e(confirmed, solver_backends=(FakeSolver(),))
    assert result.success is True
    registration = json.loads(
        (Path(result.output_directory or "") / "receipts" / "registration-calibration.json").read_text(
            encoding="utf-8"
        )
    )
    assert registration["stage"] == "registration-calibration-masters"


def test_review_approval_is_hash_policy_and_request_bound(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    all_lights = synthetic_project["lights"]
    baseline_output = tmp_path / "review-approval-baseline"
    baseline_request = _request(
        synthetic_project, baseline_output, lights=all_lights
    )
    baseline = run_e2e(
        baseline_request,
        solver_backends=(FakeSolver(),),
    )
    assert baseline.success is True
    manifest = json.loads(
        (baseline_output / "qc" / "manifest.json").read_text(encoding="utf-8")
    )
    review_path = all_lights[8]
    approval = ReviewApproval(
        source_sha256=_content_sha256(review_path),
        gate_policy_digest=manifest["gatePolicyDigest"],
        request_digest=manifest["manualReviewApprovals"]["requestDigest"],
    )

    policy_drift_output = tmp_path / "review-approval-policy-drift"
    policy_drift_request = replace(
        baseline_request,
        output_directory=str(policy_drift_output),
        review_approvals=(
            replace(approval, gate_policy_digest="sha256:" + "b" * 64),
        ),
    )
    with pytest.raises(E2EError) as policy_error:
        run_e2e(policy_drift_request, solver_backends=(FakeSolver(),))
    assert policy_error.value.code == "REVIEW_APPROVAL_POLICY_DRIFT"
    assert not policy_drift_output.exists()

    approved_output = tmp_path / "review-approval-accepted"
    approved_request = replace(
        baseline_request,
        output_directory=str(approved_output),
        review_approvals=(approval,),
    )
    approved = run_e2e(
        approved_request,
        solver_backends=(FakeSolver(),),
    )
    assert str(review_path) in approved.passed_light_paths
    assert str(review_path) not in approved.excluded_light_paths
    approved_manifest = json.loads(
        (approved_output / "qc" / "manifest.json").read_text(encoding="utf-8")
    )
    accepted = approved_manifest["manualReviewApprovals"]["accepted"]
    assert len(accepted) == 1
    assert accepted[0] == {
        "admitted": True,
        "gateDisposition": "REVIEW",
        "gatePolicyDigest": approval.gate_policy_digest,
        "path": accepted[0]["path"],
        "requestDigest": approval.request_digest,
        "sourceSha256": approval.source_sha256,
    }
    assert accepted[0]["path"].startswith("source/src-")
    assert accepted[0]["path"].endswith("/" + review_path.name)

    drifted = replace(
        approved_request,
        output_directory=str(tmp_path / "review-approval-drifted"),
        pipeline_parameters=replace(
            approved_request.pipeline_parameters, auto_crop=False
        ),
    )
    with pytest.raises(E2EError) as error:
        run_e2e(drifted, solver_backends=(FakeSolver(),))
    assert error.value.code == "REVIEW_APPROVAL_REQUEST_DRIFT"
    assert not (tmp_path / "review-approval-drifted").exists()


def test_registration_calibration_selects_dark_per_light_exposure(
    tmp_path: Path,
) -> None:
    from ufwbpp_registration import DetectionConfig, analyze_frames

    shape = (128, 128)
    bias_level = 1000.0
    base = _stars(shape)
    biases = tuple(
        _write(
            tmp_path / "multi" / "BIAS" / f"bias_{index}.fits",
            np.full(shape, bias_level),
            _header("Bias", exposure=0.001, observed_at="2026-01-01T18:00:00Z"),
        )
        for index in range(3)
    )
    darks: list[Path] = []
    for exposure, signal in ((60.0, 10.0), (120.0, 20.0)):
        for index in range(3):
            darks.append(
                _write(
                    tmp_path
                    / "multi"
                    / "DARK"
                    / f"dark_{int(exposure)}_{index}.fits",
                    np.full(shape, bias_level + signal),
                    _header(
                        "Dark",
                        exposure=exposure,
                        observed_at="2026-01-01T18:10:00Z",
                    ),
                )
            )
    flats = tuple(
        _write(
            tmp_path / "multi" / "FLAT" / f"flat_{filter_name}_{index}.fits",
            np.full(shape, bias_level + 24_000.0),
            _header(
                "Flat",
                exposure=2.0,
                observed_at="2026-01-01T18:20:00Z",
                filter_name=filter_name,
            ),
        )
        for filter_name in ("R", "G")
        for index in range(3)
    )
    lights = tuple(
        _write(
            tmp_path / "multi" / "LIGHT" / f"light_{filter_name}.fits",
            bias_level + dark_signal + base,
            _header(
                "Light",
                exposure=exposure,
                observed_at="2026-01-01T20:00:00Z",
                filter_name=filter_name,
            ),
        )
        for filter_name, exposure, dark_signal in (
            ("R", 60.0, 10.0),
            ("G", 120.0, 20.0),
        )
    )
    calibration = _build_registration_masters(
        biases=biases,
        darks=tuple(darks),
        flats=flats,
        supplied_biases=(),
        supplied_darks=(),
        supplied_flats=(),
        lights=lights,
        directory=tmp_path / "multi" / "registration-masters",
        pipeline_parameters=PipelineParameters(),
    )
    plan, receipt = calibration.plan, calibration.receipt

    assert [Path(plan.masters_for(str(light)).dark_path).name for light in lights] == [
        "master_dark_60s.fits",
        "master_dark_120s.fits",
    ]
    assert set(receipt["registrationDarksByExposure"]) == {"60", "120"}
    analyses = analyze_frames(
        [str(path) for path in lights],
        detection=DetectionConfig(preview_long_edge=256),
        calibration=plan,
        workers=1,
    )
    np.testing.assert_allclose(analyses[0].preview, base, atol=2.0)
    np.testing.assert_allclose(analyses[1].preview, base, atol=2.0)

    with pytest.raises(E2EError) as error:
        _build_registration_masters(
            biases=biases,
            darks=tuple(path for path in darks if "dark_60_" in path.name),
            flats=flats,
            supplied_biases=(),
            supplied_darks=(),
            supplied_flats=(),
            lights=lights,
            directory=tmp_path / "multi" / "missing-dark-masters",
            pipeline_parameters=PipelineParameters(),
        )
    assert error.value.code == "DARK_EXPOSURE_MISMATCH"


def test_e2e_reuses_user_supplied_masters_without_double_calibration(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    masters = tmp_path / "e2e-supplied-masters"

    def median_pixels(paths: tuple[Path, ...]) -> np.ndarray:
        return np.median(
            np.stack([fits.getdata(path).astype(np.float64) for path in paths]),
            axis=0,
        )

    bias_pixels = median_pixels(synthetic_project["biases"])
    master_bias = _write(
        masters / "master_bias.fits",
        bias_pixels,
        _header("Master Bias", exposure=0.001, observed_at="2026-01-01T18:00:00Z"),
    )
    master_dark = _write(
        masters / "master_dark_60s.fits",
        median_pixels(synthetic_project["darks"]),
        _header("Master Dark", exposure=60.0, observed_at="2026-01-01T18:10:00Z"),
    )
    master_flat = _write(
        masters / "master_flat_R.fits",
        median_pixels(synthetic_project["flats"]) - bias_pixels,
        _header(
            "Master Flat",
            exposure=2.0,
            observed_at="2026-01-01T18:20:00Z",
        ),
    )
    master_hashes = {
        path: _content_sha256(path)
        for path in (master_bias, master_dark, master_flat)
    }
    output = tmp_path / "master-input-e2e"
    request = replace(
        _request(synthetic_project, output),
        bias_files=(),
        dark_files=(),
        flat_files=(),
        master_bias_files=(str(master_bias),),
        master_dark_files=(str(master_dark),),
        master_flat_files=(str(master_flat),),
        pipeline_parameters=replace(
            PipelineParameters(),
            master_metadata_overrides=(
                MasterMetadataOverride(
                    source_sha256=_content_sha256(master_dark),
                    camera="SYNTHETIC-CAMERA",
                    gain=100,
                    offset=20,
                    binning_x=1,
                    binning_y=1,
                    filter_name="R",
                    cfa_pattern="NONE",
                    readout_mode="MODE-1",
                    temperature_celsius=-10.0,
                    exposure_seconds=60.0,
                    bias_included=True,
                ),
            ),
        ),
    )

    result = run_e2e(request, solver_backends=(FakeSolver(),))

    assert result.success is True
    assert {
        path: _content_sha256(path) for path in master_hashes
    } == master_hashes
    registration_receipt = json.loads(
        (output / "receipts" / "registration-calibration.json").read_text(
            encoding="utf-8"
        )
    )
    assert registration_receipt["masterBias"]["mode"] == "REUSED_SUPPLIED_MASTER"
    assert registration_receipt["masterDarks"]["60"]["calibrationApplied"] is False
    assert registration_receipt["masterFlats"]["R"]["calibrationApplied"] is False
    pipeline_receipt = json.loads(
        (output / "receipts" / "pixel-pipeline.json").read_text(encoding="utf-8")
    )
    assert {item["role"] for item in pipeline_receipt["inputs"]} >= {
        "MASTER_BIAS",
        "MASTER_DARK",
        "MASTER_FLAT",
    }


def test_shared_raw_calibration_injects_trusted_generated_master_semantics_into_real_panel_e2e(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    base = _request(synthetic_project, tmp_path / "unused-project-output")
    shared_root = tmp_path / "shared-calibration"
    shared_root.mkdir()
    (
        master_biases,
        master_darks,
        master_flats,
        trusted_overrides,
        shared_receipt,
    ) = _build_shared_calibration(base, shared_root)

    assert len(master_biases) == 1
    assert len(master_darks) == 1
    assert len(master_flats) == 1
    dark_override = next(item for item in trusted_overrides if item.bias_included is not None)
    assert dark_override.bias_included is True
    assert dark_override.source_sha256 == _content_sha256(Path(master_darks[0]))
    assert shared_receipt["trustedGeneratedMasterOverrides"]

    child_parameters = replace(
        base.pipeline_parameters,
        master_metadata_overrides=trusted_overrides,
    )
    child = replace(
        base,
        bias_files=(),
        dark_files=(),
        flat_files=(),
        master_bias_files=master_biases,
        master_dark_files=master_darks,
        master_flat_files=master_flats,
        output_directory=str(tmp_path / "real-panel-from-shared-masters"),
        pipeline_parameters=child_parameters,
    )
    result = run_e2e(child, solver_backends=(FakeSolver(),))
    assert result.success is True
    calibration_receipt = json.loads(
        (
            Path(result.output_directory or "")
            / "receipts"
            / "registration-calibration.json"
        ).read_text(encoding="utf-8")
    )
    assert calibration_receipt["masterDarks"]["60"]["biasIncluded"] is True


def test_project_shared_calibration_preserves_xisf_master_identity_into_real_panel(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    from lightframeqc.xisf import XISF
    from ufwbpp.inventory import inventory_project
    from ufwbpp.workflows.project import ProjectE2ERequest, run_project_e2e

    masters = []
    overrides = []
    for role, key, exposure, bias_included in (
        ("Master Bias", "biases", 0.001, None),
        ("Master Dark", "darks", 60.0, True),
    ):
        path = tmp_path / (key + ".xisf")
        pixels = np.median(
            np.stack([fits.getdata(source).astype(np.float64) for source in synthetic_project[key]]),
            axis=0,
        ) / 65535.0
        XISF.write(
            str(path), pixels.astype(np.float32)[:, :, None],
            image_metadata={
                "id": "integration", "imageType": role.replace(" ", ""),
                "FITSKeywords": {"IMAGETYP": [{"value": repr(role), "comment": ""}]},
            },
        )
        masters.append(path)
        overrides.append(MasterMetadataOverride(
            source_sha256=_content_sha256(path), camera="SYNTHETIC-CAMERA",
            gain=100, offset=20, binning_x=1, binning_y=1, filter_name="R",
            cfa_pattern="NONE", readout_mode="MODE-1", temperature_celsius=-10.0,
            exposure_seconds=exposure, bias_included=bias_included,
        ))
    flats = []
    for index in range(20):
        source = synthetic_project["flats"][index % 3]
        header = fits.getheader(source)
        header["COMMENT"] = f"independent flat {index}"
        flats.append(_write(tmp_path / "flats" / f"flat-{index}.fits", fits.getdata(source), header))
    base = replace(
        _request(synthetic_project, tmp_path / "project-xisf-masters"),
        bias_files=(), dark_files=(), flat_files=tuple(str(path) for path in flats),
        master_bias_files=(str(masters[0]),), master_dark_files=(str(masters[1]),),
        pipeline_parameters=replace(PipelineParameters(), master_metadata_overrides=tuple(overrides)),
    )
    inventory = inventory_project([*base.light_files, *base.flat_files, *masters])
    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(FakeSolver(),),
    )
    assert result.success is True, result.code
    assert len(result.passed_light_paths) == 8
    assert [_content_sha256(path) for path in masters] == [item.source_sha256 for item in overrides]


@pytest.mark.parametrize("solved", [True, False])
def test_every_digest_a_published_project_cites_is_the_digest_of_the_published_file(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
    solved: bool,
) -> None:
    """The contract of the receipts' digests, on a project through the real
    pipeline (raw Bias, Dark and Flats build the shared calibration; R, G and
    B make the colour cube) and on the evidence tree of a failed solve: every
    SHA-256 a published JSON document cites for a file of the result is the
    digest of that file, and every receipt's ``receiptId`` is its own."""

    from receipt_audit import audit_published_digests
    from ufwbpp.inventory import inventory_project
    from ufwbpp.workflows.project import ProjectE2ERequest, run_project_e2e

    filters = ("R", "G", "B") if solved else ("R",)
    flats: list[Path] = []
    lights: list[Path] = []
    for filter_name in filters:
        for role, sources, target in (("flat", synthetic_project["flats"], flats), ("light", synthetic_project["lights"][:8], lights)):
            for index, source in enumerate(sources):
                header = fits.getheader(source)
                header["FILTER"] = filter_name
                target.append(_write(tmp_path / "in" / f"{role}_{filter_name}_{index}.fits", fits.getdata(source), header))
    base = replace(
        _request(synthetic_project, tmp_path / "product", lights=tuple(lights)),
        flat_files=tuple(str(path) for path in flats),
    )
    inventory = inventory_project([*lights, *flats, *synthetic_project["darks"], *synthetic_project["biases"]])
    result = run_project_e2e(
        ProjectE2ERequest(inventory, base, base.output_directory),
        solver_backends=(FakeSolver() if solved else FailingSolver(),),
    )
    assert result.success is solved, result.message
    published = Path(result.output_directory or result.evidence_directory or "")
    checked, problems = audit_published_digests(published)
    assert problems == []

    # The audit saw the citations it is about, not an empty tree.
    receipt = json.loads((published / "receipt.json").read_text(encoding="utf-8"))
    (subrun,) = receipt["execution"]["subruns"]
    cited = {line.split(" (")[0] for line in checked}
    expected = {
        "receipt.json: receiptId",
        f"{subrun['receipt']}: receiptId",
        "receipt.json: $.execution.sharedCalibration.receipt -> details/shared-calibration/receipt.json",
        f"receipt.json: $.execution.subruns[0].receipt -> {subrun['receipt']}",
    }
    if solved:
        expected |= {
            "details/color/receipt.json: receiptId",
            "receipt.json: $.execution.color.receipt -> details/color/receipt.json",
            "details/color/receipt.json: $.inputs[0].path -> R.fits",
            "details/color/receipt.json: $.artifacts[0].path -> details/color/RGB.fits",
            *(
                f"receipt.json: $.finalProducts.guiArtifacts[{index}].relativePath -> {artifact['relativePath']}"
                for index, artifact in enumerate(receipt["finalProducts"]["guiArtifacts"])
            ),
            *(
                f"receipt.json: $.execution.subruns[0].solvedProducts.{name}.path -> {product['path']}"
                for name, product in subrun["solvedProducts"].items()
            ),
        }
        assert len(receipt["finalProducts"]["guiArtifacts"]) == 9
    assert expected <= cited, sorted(expected - cited)
    # What the run wrote about its own files reads the same on every host.
    run_receipt = json.loads((published / subrun["receipt"]).read_text(encoding="utf-8"))
    record = run_receipt["astrometry"]["filters"]["R"]
    assert record["input"] == "work/pixel-pipeline/masters/master_light_R.fits"
    assert record["output"] == ("products/R/master_light_R_wcs.fits" if solved else None)
    coverage = json.loads((published / subrun["receipt"]).with_name("coverage").joinpath("coverage.json").read_text(encoding="utf-8"))
    assert coverage["filters"]["R"]["maps"]["coverageFraction"] == "coverage/R_coverageFraction.fits"
    # The generated masters, the run's artifacts and the promoted maps are among them.
    run_root = Path(subrun["receipt"]).parent.as_posix()
    assert sum(line.startswith(f"{run_root}/receipt.json: $.artifacts[") for line in cited) >= 10
    assert any(line.startswith("details/shared-calibration/receipt.json: $.registrationCalibration.artifacts[") for line in cited)


def test_the_digest_audit_refuses_a_wrong_citation_and_reads_a_rewritten_attempt_as_history(tmp_path: Path) -> None:
    """What the audit of a published tree accepts and refuses, on a tree made
    by hand: a citation must be the digest (and size) of the file it names; a
    solver attempt whose file a later attempt rewrote states the bytes before
    the rewrite, and the last attempt that names the file states the file."""

    from receipt_audit import audit_published_digests
    from ufwbpp.integrity import canonical_json_document

    def sha256(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()

    def publish(root: Path, *, last_attempt: bytes = b"refitted", artifact: bytes = b"refitted", size: int = 8) -> Path:
        master = root / "products" / "L" / "master_light_L_wcs.fits"
        master.parent.mkdir(parents=True)
        master.write_bytes(b"refitted")
        (root / "receipts").mkdir()
        (root / "receipts" / "pixel-pipeline.json").write_text(
            # Paths of the pipeline's own work tree, which is not published, are not citations of this tree.
            json.dumps({"outputs": [{"path": "products/L/master_light_L_wcs.fits", "sha256": sha256(b"work file")}]})
        )
        named = "products/L/master_light_L_wcs.fits"
        core = {
            "artifacts": [{"path": named, "sha256": sha256(artifact), "sizeBytes": size}],
            "astrometry": {"filters": {"L": {"attempts": [
                {"artifact": {"path": named, "sha256": sha256(b"solved")}},
                {"artifact": {"path": named, "sha256": sha256(last_attempt)}},
            ]}}},
            "staged": {"output": "artifact/" + named, "outputSha256": sha256(b"refitted")},
        }
        receipt = {"receiptId": "sha256:" + hashlib.sha256(canonical_json_document(core)).hexdigest(), **core}
        (root / "receipt.json").write_bytes(canonical_json_document(receipt))
        return root

    checked, problems = audit_published_digests(publish(tmp_path / "sound"))
    assert problems == [] and len(checked) == 4  # the receipt's id, its artifact, the last attempt, the staged name
    assert len(audit_published_digests(publish(tmp_path / "stale-attempt", last_attempt=b"solved"))[1]) == 1
    assert len(audit_published_digests(publish(tmp_path / "wrong-digest", artifact=b"other"))[1]) == 1
    assert len(audit_published_digests(publish(tmp_path / "wrong-size", size=9))[1]) == 1
    edited = publish(tmp_path / "edited")
    (edited / "receipt.json").write_bytes((edited / "receipt.json").read_bytes().replace(b'"sizeBytes":8', b'"sizeBytes":8,"note":1'))
    assert audit_published_digests(edited)[1] == ["receipt.json: receiptId is not the SHA-256 of the document's canonical JSON"]


@pytest.mark.parametrize("case", ["rename-refused", "no-rename-works", "receipt-held"])
def test_a_finished_run_whose_final_rename_fails_is_kept_not_deleted(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """A complete run is never deleted because its last rename failed: its
    products become the run's ``.unsolved`` evidence under a failure receipt
    (they stay in the staging directory when that rename fails too), the
    failure is ``PUBLICATION_FAILED`` and names the directory that holds them,
    and no receipt outside a published directory says SOLVED."""

    import ufwbpp.workflows.single_target as e2e_module
    from receipt_audit import audit_published_digests
    from ufwbpp import platform as platform_services

    output = tmp_path / "out"
    evidence = tmp_path / "out.unsolved"
    services = platform_services.current()
    rename_directory = services.rename_directory_no_replace
    refused = {output} if case == "rename-refused" else {output, evidence}

    def rename(source: Path, destination: Path) -> None:
        if Path(destination) in refused:
            raise PermissionError(13, "The process cannot access the file", str(destination))
        rename_directory(source, destination)

    monkeypatch.setattr(type(services), "rename_directory_no_replace", lambda self, source, destination: rename(source, destination))
    if case == "receipt-held":
        def held(path: Path, **kwargs: Any) -> bool:
            if Path(path).name == "receipt.json":
                raise PermissionError(13, "The process cannot access the file", str(path))
            return platform_services.remove_file(path, **kwargs)

        monkeypatch.setattr(e2e_module, "remove_file", held)
        with pytest.raises(E2EError) as captured:
            run_e2e(_request(synthetic_project, output), solver_backends=(FakeSolver(),))
        # Nothing could replace the receipt: the error names the kept directory and says what it is not.
        (kept,) = [path for path in tmp_path.iterdir()]
        assert captured.value.code == "PUBLICATION_FAILED" and captured.value.path == str(kept)
        assert kept.name.startswith(".out.") and kept.name.endswith(".stage")
        assert "which is not a published result" in str(captured.value)
        assert (kept / "products" / "R" / "master_light_R_wcs.fits").is_file()
        return

    result = run_e2e(_request(synthetic_project, output), solver_backends=(FakeSolver(),))

    assert result.success is False and result.code == "PUBLICATION_FAILED" and result.state is E2EState.UNSOLVED_WORKING
    assert result.output_directory is None and result.product_paths == ()
    kept = Path(result.evidence_directory)
    assert [path for path in tmp_path.iterdir()] == [kept]
    if case == "rename-refused":
        assert kept == evidence
    else:
        assert kept.name.startswith(".out.") and kept.name.endswith(".stage")
    assert "the finished run was not published: out: the directory could not be renamed" in result.message
    assert result.message.endswith(
        f"its products are kept in {kept}"
        if case == "rename-refused"
        else f"the evidence was not published (out.unsolved: the directory could not be renamed: Permission denied) and stays in {kept}"
    )
    # The finished master, its preview and the integration maps are there; the private work is not.
    assert fits.getheader(kept / "products" / "R" / "master_light_R_wcs.fits")["OAFSTATE"] == "SOLVED"
    assert (kept / "previews" / "master_light_R.png").is_file()
    assert sorted(path.name for path in (kept / "coverage").glob("*.fits")) == [
        "R_acceptedSampleCount.fits", "R_coverageFraction.fits", "R_rejectionCount.fits",
    ]
    assert not (kept / "work").exists()
    receipt_text = (kept / "receipt.json").read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert Path(result.receipt_path) == kept / "receipt.json"
    assert (receipt["success"], receipt["state"], receipt["code"]) == (False, "UNSOLVED_WORKING", "PUBLICATION_FAILED")
    assert receipt["publication"]["requestedOutputPublished"] is False
    # What the run found is recorded as found: every filter was solved.
    assert receipt["astrometry"]["status"] == "SOLVED" and receipt["astrometry"]["filters"]["R"]["status"] == "SOLVED"
    assert str(tmp_path) not in receipt_text
    assert audit_published_digests(kept)[1] == []


def test_integration_maps_are_published_as_the_files_the_pipeline_wrote(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three integration maps of a filter reach the run's ``coverage``
    folder through the create-only publication (a hard link where the volume
    has them), not through a copy that is hashed twice: the published file is
    the one the pipeline wrote, under the digest the pipeline recorded."""

    published: list[tuple[str, str]] = []
    hashed: list[str] = []
    publish = products_module.atomic_publish_file
    digest = products_module.sha256_digest

    def spy_publish(source: Path, destination: Path, **kwargs: Any) -> None:
        published.append((Path(source).name, _content_sha256(Path(source))))
        publish(source, destination, **kwargs)
        assert not Path(source).exists()

    def spy_digest(path: Path) -> str:
        hashed.append(Path(path).name)
        return digest(path)

    monkeypatch.setattr(products_module, "atomic_publish_file", spy_publish)
    monkeypatch.setattr(products_module, "sha256_digest", spy_digest)
    output = tmp_path / "maps"
    result = run_e2e(_request(synthetic_project, output), solver_backends=(FakeSolver(),))
    assert result.success is True

    names = ["R_acceptedSampleCount.fits", "R_coverageFraction.fits", "R_rejectionCount.fits"]
    assert [name for name, _ in published] == names and not set(hashed) & set(names)
    pipeline = json.loads((output / "receipts" / "pixel-pipeline.json").read_text(encoding="utf-8"))
    # The pipeline's own receipt writes its paths with the host's separator.
    recorded = {
        item["path"].replace("\\", "/").rsplit("/", 1)[-1]: item["sha256"]
        for item in pipeline["outputs"]
        if item["path"].replace("\\", "/").startswith("coverage/")
    }
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    artifacts = {item["path"]: item["sha256"] for item in receipt["artifacts"]}
    coverage = json.loads((output / "coverage" / "coverage.json").read_text(encoding="utf-8"))
    assert sorted(coverage["filters"]["R"]["maps"].values()) == [f"coverage/{name}" for name in names]
    for name, written in published:
        assert _content_sha256(output / "coverage" / name) == written == recorded[name] == artifacts[f"coverage/{name}"]
    # A name that is taken is never replaced.
    (tmp_path / "taken.fits").write_bytes(b"other")
    with pytest.raises(Exception) as refused:
        publish(output / "coverage" / names[0], tmp_path / "taken.fits")
    assert getattr(refused.value, "code", None) == "OUTPUT_EXISTS" and (tmp_path / "taken.fits").read_bytes() == b"other"


@pytest.mark.parametrize("held", [False, True], ids=["run-rename-refused", "no-rename-works"])
def test_a_target_run_that_cannot_be_published_inside_its_project_keeps_its_products_in_the_evidence(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
    monkeypatch: pytest.MonkeyPatch,
    held: bool,
) -> None:
    """A finished target run that cannot take its name fails the project,
    and its master stays in the project's evidence.  While a file of it is
    held, no directory above it can be renamed either: the project then
    stays in its staging directory, which the result names."""

    from receipt_audit import audit_published_digests
    from ufwbpp import platform as platform_services
    from ufwbpp.inventory import inventory_project
    from ufwbpp.workflows.project import ProjectE2ERequest, run_project_e2e

    services = platform_services.current()
    rename_directory = services.rename_directory_no_replace

    def rename(source: Path, destination: Path) -> None:
        destination = Path(destination)
        if (destination.parent.name == "runs" and (held or not destination.name.endswith(".unsolved"))) or (
            held and destination.name == "product.unsolved"
        ):
            raise PermissionError(13, "The process cannot access the file", str(destination))
        rename_directory(source, destination)

    monkeypatch.setattr(type(services), "rename_directory_no_replace", lambda self, source, destination: rename(source, destination))
    base = _request(synthetic_project, tmp_path / "product")
    inventory = inventory_project(
        [*synthetic_project["lights"][:8], *synthetic_project["flats"], *synthetic_project["darks"], *synthetic_project["biases"]]
    )
    result = run_project_e2e(ProjectE2ERequest(inventory, base, base.output_directory), solver_backends=(FakeSolver(),))

    assert result.success is False and result.code == "PUBLICATION_FAILED"
    evidence = Path(result.evidence_directory)
    assert [path for path in tmp_path.iterdir()] == [evidence]
    (run,) = (evidence / "details" / "runs").iterdir()
    if held:
        assert evidence.name.startswith(".product.") and evidence.name.endswith(".pstage")
        assert run.name.startswith(".SYNTHETICFIELD.") and run.name.endswith(".stage")
        assert result.message.endswith(
            f"the evidence was not published (product.unsolved: the directory could not be renamed: Permission denied) and stays in {evidence}"
        )
    else:
        assert evidence == tmp_path / "product.unsolved" and run.name == "SYNTHETICFIELD.unsolved"
    assert fits.getheader(run / "products" / "R" / "master_light_R_wcs.fits")["OAFSTATE"] == "SOLVED"
    receipt_text = (evidence / "receipt.json").read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    assert Path(result.receipt_path) == evidence / "receipt.json"
    assert receipt["success"] is False and receipt["code"] == "PUBLICATION_FAILED"
    assert receipt["publication"]["evidenceDirectoryPublished"] is (not held)
    assert receipt["message"].startswith("the finished run was not published: ")
    # A path inside a sentence keeps the host's separators after its staging prefix.
    assert receipt["message"].replace("\\", "/").endswith(f"artifact/details/runs/{run.name}")
    assert json.loads((run / "receipt.json").read_text(encoding="utf-8"))["success"] is False
    assert str(tmp_path) not in receipt_text
    assert audit_published_digests(evidence)[1] == []


def test_evidence_that_cannot_be_renamed_stays_staged_and_keeps_its_failure(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed run whose ``.unsolved`` name cannot be taken reports the
    failure it had, not the rename, and names the staging directory that
    still holds its evidence."""

    from ufwbpp import platform as platform_services

    services = platform_services.current()
    rename_directory = services.rename_directory_no_replace

    def rename(source: Path, destination: Path) -> None:
        if Path(destination).name == "out.unsolved":
            raise PermissionError(13, "The process cannot access the file", str(destination))
        rename_directory(source, destination)

    monkeypatch.setattr(type(services), "rename_directory_no_replace", lambda self, source, destination: rename(source, destination))
    result = run_e2e(_request(synthetic_project, tmp_path / "out"), solver_backends=(FailingSolver(),))

    assert result.success is False and result.code == "ASTROMETRY_REQUIRED"
    (kept,) = tmp_path.iterdir()
    assert Path(result.evidence_directory) == kept and kept.name.startswith(".out.") and kept.name.endswith(".stage")
    assert result.message == (
        "one or more filters did not produce a verified new WCS solution; the evidence was not published "
        f"(out.unsolved: the directory could not be renamed: Permission denied) and stays in {kept}"
    )
    receipt = json.loads((kept / "receipt.json").read_text(encoding="utf-8"))
    assert (receipt["success"], receipt["code"]) == (False, "ASTROMETRY_REQUIRED")
    assert (kept / "qc" / "manifest.json").is_file() and not (kept / "work").exists()


def test_directory_publication_reports_output_exists_only_when_the_output_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ufwbpp import platform as platform_services

    staged = tmp_path / ".out.stage"
    staged.mkdir()
    (staged / "product.fits").write_bytes(b"pixels")
    services = platform_services.current()

    def refused(self: Any, source: Path, destination: Path) -> None:
        raise PermissionError(13, "The process cannot access the file", str(destination))

    with monkeypatch.context() as patched:
        patched.setattr(type(services), "rename_directory_no_replace", refused)
        with pytest.raises(E2EError) as failed:
            common_module._rename_directory_no_replace(staged, tmp_path / "out")
        assert failed.value.code == "PUBLICATION_FAILED" and failed.value.path == str(tmp_path / "out")
        assert "the directory could not be renamed: Permission denied" in str(failed.value)
        # The failed rename left the name to another program's directory.
        (tmp_path / "taken").mkdir()
        with pytest.raises(E2EError) as raced:
            common_module._rename_directory_no_replace(staged, tmp_path / "taken")
        assert raced.value.code == "OUTPUT_EXISTS"
    with pytest.raises(E2EError) as existing:
        common_module._rename_directory_no_replace(staged, tmp_path / "taken")
    assert existing.value.code == "OUTPUT_EXISTS"
    # Whatever the failure, the staged tree is untouched and nothing was replaced.
    assert (staged / "product.fits").read_bytes() == b"pixels" and list((tmp_path / "taken").iterdir()) == []
    common_module._rename_directory_no_replace(staged, tmp_path / "out")
    assert (tmp_path / "out" / "product.fits").read_bytes() == b"pixels"


def test_single_admitted_light_stops_before_calibration_and_preserves_qc(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ufwbpp.workflows.single_target as e2e_module
    from lightframeqc.models import GateDisposition

    original_gate = screening_module.evaluate_quality_gate

    def one_admitted(results, measurements, policy):
        original_gate(results, measurements, policy)
        for index, result in enumerate(results):
            result.quality_gate.disposition = GateDisposition.PASS if index == 0 else GateDisposition.REVIEW

    def calibration_must_not_start(*args, **kwargs):
        pytest.fail("Calibration must not start when only one Light is admitted")

    monkeypatch.setattr(screening_module, "evaluate_quality_gate", one_admitted)
    monkeypatch.setattr(e2e_module, "_build_registration_masters", calibration_must_not_start)
    output = tmp_path / "one-admitted"
    result = run_e2e(_request(synthetic_project, output), solver_backends=(FailingSolver(),))

    assert not result.success and result.code == "QC_INSUFFICIENT_LIGHTS"
    assert "1 of 8 Light frames admitted" in result.message
    assert "at least 2" in result.serializable()["message"]
    assert not output.exists()
    evidence = Path(result.evidence_directory)
    assert (evidence / "qc" / "manifest.json").is_file()
    assert list((evidence / "qc" / "thumbnails").glob("*.png"))
    assert not (evidence / "work").exists()
    receipt = json.loads(Path(result.receipt_path).read_text())
    assert receipt["qualityControl"]["passedLights"] == 1
    assert receipt["message"] == result.message


@pytest.mark.parametrize(
    ("solver", "inheritance"),
    (
        (FailingSolver(), "NOT_APPLICABLE"),
        # It missed only the RMS gate, but no catalog refit can verify it.
        (FakeSolver(rms_arcsec=3.0), "WITHDRAWN"),
    ),
)
def test_solver_failure_blocks_final_publication_and_retains_unsolved_evidence(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
    solver: Any,
    inheritance: str,
) -> None:
    output = tmp_path / "solver-miss"

    result = run_e2e(
        _request(synthetic_project, output),
        solver_backends=(solver,),
    )

    assert result.success is False
    assert result.code == "ASTROMETRY_REQUIRED"
    assert result.state is E2EState.UNSOLVED_WORKING
    assert not output.exists()
    assert result.evidence_directory == str(tmp_path / "solver-miss.unsolved")
    evidence = Path(result.evidence_directory)
    assert not (evidence / "work").exists()
    assert (evidence / "receipts" / "pixel-pipeline.json").is_file()
    assert not any(
        part in {"calibrated", "registered", "pixel-inputs", "registration-calibration"}
        for path in evidence.rglob("*")
        for part in path.parts
    )
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["success"] is False
    assert receipt["state"] == "UNSOLVED_WORKING"
    assert receipt["astrometry"]["status"] == "UNSOLVED"
    assert receipt["astrometry"]["filters"]["R"]["status"] == "UNSOLVED"
    assert receipt["astrometry"]["sameGridInheritance"]["status"] == inheritance
    assert not list(evidence.rglob("master_light_*_wcs.fits"))
    assert receipt["artifacts"]
    assert all(not item["path"].startswith("work/") for item in receipt["artifacts"])


@pytest.mark.parametrize(
    ("solver", "validation_key", "code"),
    (
        (
            FakeSolver(center_ra_degrees=230.0),
            "hintValidation",
            "SOLVER_CENTER_OUTSIDE_HINT_RADIUS",
        ),
        (
            FakeSolver(matched_stars=8),
            "validation",
            "SOLVER_MATCH_COUNT_BELOW_MINIMUM",
        ),
        (
            FakeSolver(rms_arcsec=3.0),
            "validation",
            "SOLVER_RMS_ABOVE_MAXIMUM",
        ),
        (
            FakeSolver(rms_arcsec=3.1),
            "validation",
            "SOLVER_RMS_ABOVE_MAXIMUM",
        ),
        (
            FakeSolver(rms_arcsec=3.0, center_ra_degrees=230.0),
            "hintValidation",
            "SOLVER_CENTER_OUTSIDE_HINT_RADIUS",
        ),
        (
            FakeSolver(field_width_degrees=0.8),
            "hintValidation",
            "SOLVER_SCALE_OUTSIDE_HINT",
        ),
    ),
)
def test_scientifically_wrong_but_mathematically_valid_solution_is_rejected(
    tmp_path: Path,
    solver: FakeSolver,
    validation_key: str,
    code: str,
) -> None:
    source = _write(
        tmp_path / "master-unsolved.fits",
        np.ones((128, 128), dtype=np.uint16),
        _header("Light", exposure=60.0, observed_at="2026-01-01T20:00:00Z"),
    )
    output = tmp_path / "master-solved.fits"

    solved, attempts = _solve_one(
        input_path=source,
        output_path=output,
        backends=(solver,),
        hints=_SolverHints(150.0, 20.0, 3.0, 5.0, "NINA/FITS-header"),
        min_matches=12,
        max_rms_arcsec=2.0,
    )

    assert solved is False
    assert attempts[0][validation_key]["code"] == code
    # Only a solve that missed nothing but the RMS gate, and that narrowly, may seed the catalog refit.
    assert ("refitSeedWcs" in attempts[0]) is (code == "SOLVER_RMS_ABOVE_MAXIMUM" and solver.rms_arcsec <= 3.0)
    assert not output.exists()


def test_solver_quality_receipt_drift_is_rejected(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "master-unsolved.fits",
        np.ones((128, 128), dtype=np.uint16),
        _header("Light", exposure=60.0, observed_at="2026-01-01T20:00:00Z"),
    )
    output = tmp_path / "master-solved.fits"

    solved, attempts = _solve_one(
        input_path=source,
        output_path=output,
        backends=(FakeSolver(receipt_drift=True),),
        hints=_SolverHints(150.0, 20.0, 3.0, 5.0, "explicit"),
        min_matches=12,
        max_rms_arcsec=2.0,
    )

    assert solved is False
    assert attempts[0]["executionVerified"] is False
    assert not output.exists()


def test_backend_without_reliable_match_rms_evidence_falls_through(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "master-unsolved.fits",
        np.ones((128, 128), dtype=np.uint16),
        _header("Light", exposure=60.0, observed_at="2026-01-01T20:00:00Z"),
    )
    output = tmp_path / "master-solved.fits"
    unavailable_quality = FakeSolver(provide_quality=False)
    accepted = FakeSolver()

    solved, attempts = _solve_one(
        input_path=source,
        output_path=output,
        backends=(unavailable_quality, accepted),
        hints=_SolverHints(150.0, 20.0, 3.0, 5.0, "explicit"),
        min_matches=12,
        max_rms_arcsec=2.0,
    )

    assert solved is True
    assert attempts[0]["validation"]["code"] == "SOLVER_QUALITY_EVIDENCE_MISSING"
    assert attempts[0]["accepted"] is False
    assert attempts[1]["accepted"] is True
    assert output.is_file()


def test_cross_filter_wcs_mismatch_is_rejected(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "master-unsolved.fits",
        np.ones((128, 128), dtype=np.uint16),
        _header("Light", exposure=60.0, observed_at="2026-01-01T20:00:00Z"),
    )
    red = tmp_path / "red.fits"
    blue = tmp_path / "blue.fits"
    FakeSolver(center_ra_degrees=150.0).solve(
        SimpleNamespace(input_path=str(source), output_path=str(red))
    )
    FakeSolver(center_ra_degrees=154.0).solve(
        SimpleNamespace(input_path=str(source), output_path=str(blue))
    )

    validation = _validate_cross_filter_wcs({"R": red, "B": blue})

    assert validation.valid is False
    assert validation.code == "CROSS_FILTER_WCS_MISMATCH"


def _same_grid_products(
    tmp_path: Path, *, blue_offset_pixels: float, same_crop: bool = True
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    """Two solved masters of one registered grid whose fresh solves disagree slightly."""

    products: dict[str, Path] = {}
    records: dict[str, Any] = {}
    field_width = 0.5
    pixel_scale_degrees = field_width / 127
    for filter_name, offset in (("R", 0.0), ("B", blue_offset_pixels)):
        source = _write(
            tmp_path / f"{filter_name}-unsolved.fits",
            np.ones((128, 128), dtype=np.uint16) * (10 if filter_name == "R" else 20),
            _header("Light", exposure=60.0, observed_at="2026-01-01T20:00:00Z", filter_name=filter_name),
        )
        output = tmp_path / f"{filter_name}-solved.fits"
        solver = FakeSolver(center_ra_degrees=150.0 + offset * pixel_scale_degrees)
        solver.field_width_degrees = field_width
        solver.solve(SimpleNamespace(input_path=str(source), output_path=str(output)))
        with fits.open(output, mode="update", memmap=False) as hdul:
            hdul[0].header["OAFWCS"] = "SOLVED"
            hdul.flush(output_verify="exception")
        products[filter_name] = output
        records[filter_name] = {
            "attempts": [
                {"accepted": True, "result": {"astrometricQuality": {"rmsPixels": 0.2}}}
            ]
        }
    crop_r = [3, 5, 131, 133]
    receipt = {
        "statistics": {
            "integrationGroups": {
                "R": {"crop": crop_r},
                "B": {"crop": crop_r if same_crop else [4, 5, 132, 133]},
            }
        }
    }
    return products, records, receipt


def test_same_grid_solutions_are_verified_then_unified_without_touching_pixels(
    tmp_path: Path,
) -> None:
    products, records, receipt = _same_grid_products(tmp_path, blue_offset_pixels=0.3)
    before_red = np.asarray(fits.getdata(products["R"]))
    assert _validate_cross_filter_wcs(products).valid is False

    record = _unify_same_grid_solutions(
        products, pixel_pipeline_receipt=receipt, solver_records=records, tolerance_pixels=1.0
    )

    assert record["status"] == "APPLIED"
    assert record["adoptedFilter"] == "B"  # equal RMS and no luminance: alphabetical
    red = record["filters"]["R"]
    assert red["rewritten"] is True
    assert 0.25 < red["ownVersusAdoptedMaximumPixels"] < 0.35
    assert red["sha256Before"] != red["sha256After"]
    assert record["filters"]["B"]["rewritten"] is False
    np.testing.assert_array_equal(np.asarray(fits.getdata(products["R"])), before_red)
    header = fits.getheader(products["R"])
    assert header["OAFWCSSG"] == "B"
    assert header["OAFSTATE"] == "SOLVED" and header["OAFWCS"] == "SOLVED"
    assert _validate_cross_filter_wcs(products).valid is True


def test_same_grid_solutions_that_disagree_beyond_solver_precision_are_rejected(
    tmp_path: Path,
) -> None:
    products, records, receipt = _same_grid_products(tmp_path, blue_offset_pixels=6.0)
    before_sha = hashlib.sha256(products["B"].read_bytes()).hexdigest()

    record = _unify_same_grid_solutions(
        products, pixel_pipeline_receipt=receipt, solver_records=records, tolerance_pixels=1.0
    )

    assert record["status"] == "MISMATCH"
    assert record["code"] == "SAME_GRID_WCS_MISMATCH"
    assert record["filters"]["R"]["effectiveTolerancePixels"] == 1.0
    assert hashlib.sha256(products["B"].read_bytes()).hexdigest() == before_sha


def test_masters_with_different_crops_are_not_unified(tmp_path: Path) -> None:
    products, records, receipt = _same_grid_products(
        tmp_path, blue_offset_pixels=0.3, same_crop=False
    )
    record = _unify_same_grid_solutions(
        products, pixel_pipeline_receipt=receipt, solver_records=records, tolerance_pixels=1.0
    )
    assert record["status"] == "NOT_APPLICABLE"
    assert _validate_cross_filter_wcs(products).valid is False


@pytest.mark.parametrize("unsolved_filters", ((), ("R",), ("L", "R")))
@pytest.mark.parametrize("catalog", ("dense", "exact", "unverified", "sparse", "none", "misplaced", "wrong", "half-wrong"))
def test_solved_grid_is_refitted_on_catalog_stars(tmp_path: Path, catalog: str, unsolved_filters: tuple[str, ...]) -> None:
    from mosaic_synthetic import Sky, TruthSolver, make_sky, render, true_header
    from ufwbpp.mosaic.canvas import celestial_wcs
    from ufwbpp.workflows.canvas import _catalog_verifier

    # A fast corrected train: cubic barrel distortion the solver's pure TAN
    # misses; "exact" is a solver that already has the true solution.
    sky = make_sky((150.0, 2.0), 0.4, 40 if catalog == "sparse" else 700)
    shape = (600, 900)
    truth = true_header((150.0, 2.0), shape, scale_arcsec=2.0, distortion=0.0 if catalog == "exact" else 4.0)
    tan = {key: value for key, value in truth.items() if not key.startswith(("A_", "B_"))}
    tan.update(CTYPE1="RA---TAN", CTYPE2="DEC--TAN")
    products: dict[str, Path] = {}
    unsolved: dict[str, Path] = {}
    pixels: dict[str, np.ndarray] = {}
    for filter_name, seed in (("L", 1), ("R", 2)):
        image = render(sky, truth, shape, fwhm=3.0, background=100.0, noise=2.0, rng=np.random.default_rng(seed))
        header = fits.Header(list(tan.items()))
        header["OAFSTATE"] = "SOLVED"
        header["OAFWCS"] = "SOLVED"
        (tmp_path / filter_name).mkdir()
        products[filter_name] = tmp_path / filter_name / f"master_light_{filter_name}_wcs.fits"
        if filter_name in unsolved_filters:
            # Its own solve failed a gate: only its unsolved master exists.
            unsolved[filter_name] = tmp_path / f"{filter_name}-unsolved.fits"
            fits.PrimaryHDU(image.astype(np.float32), fits.Header([("OAFSTATE", "UNSOLVED_WORKING")])).writeto(unsolved[filter_name])
        else:
            fits.PrimaryHDU(image.astype(np.float32), header).writeto(products[filter_name])
        # A copy: on Windows a mapped file cannot take the rewritten, longer header.
        pixels[filter_name] = fits.getdata(unsolved.get(filter_name, products[filter_name]), memmap=False)
    records = {
        name: {"output": f"{name}/master_light_{name}_wcs.fits", "attempts": [{"accepted": True, "artifact": {}}]}
        for name in products
    }
    for name in unsolved:
        records[name] = {"status": "UNSOLVED", "output": None, "attempts": [{"accepted": False, "code": "SOLVER_RMS_ABOVE_MAXIMUM"}]}
    if unsolved:
        solved = {name: path for name, path in products.items() if name not in unsolved}
        grids = {"L": [0, 0, 600, 900], "R": [0, 0, 600, 900]}
        arguments = {"products_dir": tmp_path, "staging": tmp_path, "solver_records": records}
        if not solved:
            # No filter passed, and none has a solve that missed only the RMS gate.
            assert _inherit_same_grid_solution(unsolved, solved, **arguments, grid_signatures=grids)[1] == {}
            # Both solves missed only the RMS gate: the lower-RMS one seeds the refit.
            for name, rms_pixels in (("L", 1.5), ("R", 1.2)):
                records[name]["attempts"][0].update(result={"astrometricQuality": {"rmsPixels": rms_pixels}}, refitSeedWcs=dict(tan))
        # Masters cropped to another rectangle never take each other's solve.
        assert _inherit_same_grid_solution(unsolved, solved, **arguments, grid_signatures={**grids, "R": [1, 0, 601, 900]})[1] == {}
        assert not any(products[name].exists() for name in unsolved)
        inheritance, inherited = _inherit_same_grid_solution(unsolved, solved, **arguments, grid_signatures=grids)
        assert inheritance["adoptedFilter"] == ("L" if solved else "R")
        assert inheritance["adoptedSolve"] == ("ACCEPTED" if solved else "MISSED_ONLY_THE_RMS_GATE")
        assert inherited == {name: products[name] for name in unsolved}
    before = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in products.items()}
    backends = [TruthSolver(sky, scale_arcsec=2.0)] if catalog != "none" else [SimpleNamespace()]
    verifier = _catalog_verifier(backends)
    if catalog == "unverified":
        # The R master does not confirm the refit on catalog stars.
        accept = verifier
        verifier = lambda image, header, directory: (  # noqa: E731
            {**accept(image, header, directory), "matchedStars": 3} if directory.parent.name == "R" else accept(image, header, directory)
        )
    # The masters verify on another catalog than the refit's, as on an old
    # Tycho-2 index: a tenth of its stars misplaced ("misplaced"), or a refit
    # off from it at every star ("wrong") or on nearly half the stars ("half-wrong").
    moved_fraction, moved_arcsec = {"misplaced": (0.1, 8.0), "wrong": (1.0, 3.0), "half-wrong": (0.4, 8.0)}.get(catalog, (0.0, 0.0))
    if moved_fraction:
        moved = np.arange(sky.dec.size) < moved_fraction * sky.dec.size
        verifying = Sky(sky.ra, sky.dec + moved * moved_arcsec / 3600.0, sky.flux)
        verifier = _catalog_verifier([TruthSolver(verifying, scale_arcsec=2.0, match_radius_pixels=6.0)])

    record = _refine_solved_wcs(
        products,
        solver_records=records,
        backends=backends,
        verifier=verifier,
        min_matches=20,
        max_rms_arcsec=2.0,
        workers=2,
    )

    if catalog not in ("dense", "misplaced"):
        # No catalog, too few stars to pay for a detection, no gain on held-out
        # stars, or a master that does not verify: the solver's WCS stays.
        expected = {
            "none": ("NOT_APPLICABLE", None),
            "sparse": ("KEPT_SOLVER_WCS", "MOSAIC_CATALOG_TOO_SPARSE"),
            "exact": ("KEPT_SOLVER_WCS", "WCS_REFINEMENT_NOT_BETTER"),
            "unverified": ("KEPT_SOLVER_WCS", "WCS_REFINEMENT_UNVERIFIED"),
            "wrong": ("KEPT_SOLVER_WCS", "WCS_REFINEMENT_UNVERIFIED"),
            "half-wrong": ("KEPT_SOLVER_WCS", "WCS_REFINEMENT_UNVERIFIED"),
        }[catalog]
        assert (record["status"], record.get("code")) == expected
        if catalog in ("wrong", "half-wrong"):
            # Too many stars are off to call them outliers: all of them count.
            assert record["verificationResidual"]["outliers"] == 0
            assert record["verificationResidual"]["rmsArcsec"] == record["verificationResidual"]["rawRmsArcsec"] > 2.0
        assert {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in products.items()} == before
        assert all(len(item["attempts"]) == 1 for item in records.values())
        assert not any((path.parent / "catalog-refit-verification").exists() for path in products.values())
        if unsolved:
            # Not verified on their own pixels: these masters keep no product and stay unsolved.
            _withdraw_inherited_solutions(inherited, records)
            assert not any(products[name].exists() or records[name]["output"] for name in unsolved)
        return
    assert record["status"] == "APPLIED"
    assert record["verificationStatistic"] == "correspondence-clipped-rms-v1"
    assert record["referenceFilter"] == "L" and record["order"] >= 3
    assert record["holdoutRmsArcsec"] < 0.2 * record["solverRmsArcsec"]
    for name, path in products.items():
        np.testing.assert_array_equal(np.asarray(fits.getdata(path)), pixels[name])
        header = fits.getheader(path)
        assert header["OAFWCSRF"] == record["order"]
        assert header["OAFSTATE"] == "SOLVED" and header["OAFWCS"] == "SOLVED"
        assert header.get("OAFWCSPR") == ("INHERITED_SAME_GRID" if name in unsolved else None)
        # The refit's own verification is the master's accepted solve, bound to its bytes.
        solver_attempt, refit_attempt = records[name]["attempts"]
        assert solver_attempt["accepted"] is (name not in unsolved)
        assert refit_attempt["accepted"] is True and refit_attempt["backendId"] == "catalog-refit-verification"
        assert refit_attempt["result"]["astrometricQuality"]["catalogManaged"] is True
        verified = record["rewritten"][name]
        assert refit_attempt["result"]["astrometricQuality"]["rmsArcsec"] == verified["verifiedRmsArcsec"] <= 2.0
        if catalog == "misplaced":
            # The misplaced stars alone fail the plain RMS; they cannot decide the outcome.
            assert verified["verifiedRawRmsArcsec"] > 2.0
            assert 0.05 < verified["verifiedOutliers"] / verified["verifiedMatches"] <= 0.15
        assert Path(refit_attempt["artifact"]["path"]) == Path(name, f"master_light_{name}_wcs.fits")
        assert refit_attempt["artifact"]["sha256"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert _validate_cross_filter_wcs(products).valid is True
    # The refit maps every pixel where the true solution does, corners included.
    yy, xx = np.mgrid[0 : shape[0] : 50, 0 : shape[1] : 50]
    refit = np.column_stack(celestial_wcs(fits.getheader(products["L"])).all_pix2world(xx.ravel(), yy.ravel(), 0))
    exact = np.column_stack(celestial_wcs(truth).all_pix2world(xx.ravel(), yy.ravel(), 0))
    error = np.hypot((refit[:, 0] - exact[:, 0]) * np.cos(np.radians(exact[:, 1])), refit[:, 1] - exact[:, 1]) * 3600
    assert error.max() < 0.3 * 2.0
    # A master carrying exactly its WCS passes the rewrite check, however slowly the inverse converges.
    strong = fits.Header(list(true_header((150.0, 2.0), shape, scale_arcsec=2.0, distortion=40.0).items()))
    strong["OAFSTATE"] = strong["OAFWCS"] = "SOLVED"
    fits.PrimaryHDU(pixels["L"], strong).writeto(tmp_path / "strong.fits")
    _verify_rewritten_wcs(tmp_path / "strong.fits", strong, shape, error_code="UNEXPECTED_REWRITE_INVALID")


@pytest.mark.parametrize("mutation", ("rotate90", "mirror", "sip-edge"))
def test_cross_filter_wcs_direct_grid_gate_rejects_geometry_drift(
    tmp_path: Path, mutation: str
) -> None:
    source = _write(
        tmp_path / "geometry-source.fits",
        np.ones((128, 128), dtype=np.uint16),
        _header("Light", exposure=60.0, observed_at="2026-01-01T20:00:00Z"),
    )
    red = tmp_path / "geometry-red.fits"
    blue = tmp_path / f"geometry-{mutation}.fits"
    FakeSolver().solve(SimpleNamespace(input_path=str(source), output_path=str(red)))
    FakeSolver().solve(SimpleNamespace(input_path=str(source), output_path=str(blue)))
    with fits.open(blue, mode="update", memmap=False) as hdul:
        header = hdul[0].header
        scale = abs(float(header["CD2_2"]))
        if mutation == "rotate90":
            header["CD1_1"] = 0.0
            header["CD1_2"] = -scale
            header["CD2_1"] = -scale
            header["CD2_2"] = 0.0
        elif mutation == "mirror":
            header["CD1_1"] = scale
        else:
            header["CTYPE1"] = "RA---TAN-SIP"
            header["CTYPE2"] = "DEC--TAN-SIP"
            header["A_ORDER"] = 2
            header["B_ORDER"] = 2
            header["A_2_0"] = 1e-3
            header["A_1_1"] = 0.0
            header["A_0_2"] = 0.0
            header["B_2_0"] = 0.0
            header["B_1_1"] = 0.0
            header["B_0_2"] = -1e-3
        hdul.flush(output_verify="exception")

    validation = _validate_cross_filter_wcs({"R": red, "B": blue})

    assert validation.valid is False
    assert validation.code == "CROSS_FILTER_WCS_MISMATCH"
    comparisons = validation.diagnostics["comparisons"]
    comparison = comparisons["R"] if "R" in comparisons else comparisons["B"]
    assert comparison["tolerancePixels"] == 0.05


@pytest.mark.skipif(load_native_kernels() is None, reason="native kernel library is not built in this checkout")
def test_drizzle_branch_produces_coverage_then_solves_the_drizzled_master(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    output = tmp_path / "drizzle-solved"
    solver = FakeSolver()

    result = run_e2e(
        _request(synthetic_project, output, mode=IntegrationMode.DRIZZLE),
        solver_backends=(solver,),
    )

    assert result.success is True
    assert len(solver.inputs) == 1
    assert "drizzle_unsolved.fits" in solver.inputs[0]
    coverage = json.loads((output / "coverage" / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["mode"] == "drizzle"
    assert coverage["options"]["scale"] == 2
    assert coverage["options"]["kernel"] == "square"
    drizzle_receipt = coverage["filters"]["R"]
    assert drizzle_receipt["status"] == "succeeded"
    assert drizzle_receipt["backend"]["id"] == "native-cpu-drizzle-v1"
    assert drizzle_receipt["statistics"]["inputFrames"] == 8
    assert drizzle_receipt["statistics"]["coverageFraction"] >= 0.90
    assert all(item["rejectionMask"] for item in drizzle_receipt["inputs"])
    assert drizzle_receipt["coverageGate"]["status"] == "PASS"
    assert drizzle_receipt["science"]["dither"]["distinctPhaseBins"] >= 1
    assert coverage["sampling"]["status"] in {"PASS_UNDERSAMPLED", "WELL_SAMPLED", "UNKNOWN_SAMPLING"}
    assert coverage["sampling"]["advisory"] is True
    assert (output / "receipts" / "drizzle_R.json").is_file()
    with fits.open(result.product_paths[0]) as hdul:
        assert [item.name for item in hdul] == ["SCI", "WHT", "COVERAGE"]
        assert hdul[0].header["CTYPE1"] == "RA---TAN"
        assert hdul[0].header["OAFDRZSC"] == 2
        assert hdul["COVERAGE"].data.shape == (256, 256)
        science = np.asarray(hdul[0].data, dtype=np.float64)
        weights = np.asarray(hdul["WHT"].data, dtype=np.float64)
        finite = np.isfinite(science)
        assert finite.mean() >= 0.90
        assert np.all(weights[finite] > 0)
        assert np.all(weights[~finite] == 0)


def test_sampling_evidence_is_advisory_for_twox_data() -> None:
    metadata = SimpleNamespace(header={})
    unknown = SimpleNamespace(
        features=SimpleNamespace(median_fwhm_native_pixels=None, nina_hfr_pixels=None),
        metadata=metadata,
    )
    evidence = _drizzle_sampling_evidence((unknown,), DrizzleOptions(scale=2))
    assert evidence["status"] == "UNKNOWN_SAMPLING" and evidence["advisory"] is True

    well_sampled = SimpleNamespace(
        features=SimpleNamespace(median_fwhm_native_pixels=3.4, nina_hfr_pixels=None),
        metadata=metadata,
    )
    evidence = _drizzle_sampling_evidence((well_sampled,), DrizzleOptions(scale=2))
    assert evidence["status"] == "WELL_SAMPLED"
    assert "recommendation" in evidence

    undersampled = SimpleNamespace(
        features=SimpleNamespace(median_fwhm_native_pixels=1.8, nina_hfr_pixels=0.9),
        metadata=metadata,
    )
    evidence = _drizzle_sampling_evidence((undersampled,), DrizzleOptions(scale=2))
    assert evidence["status"] == "PASS_UNDERSAMPLED"
    assert _drizzle_sampling_evidence((undersampled,), DrizzleOptions(scale=1))["status"] == "NOT_APPLICABLE"


def test_drizzle_options_expose_production_defaults() -> None:
    options = DrizzleOptions()

    assert (options.scale, options.pixfrac, options.kernel) == (2, 0.9, "square")
    assert options.minimum_coverage_fraction == 0.90
    assert options.maximum_null_fraction == 0.10
    assert options.minimum_distinct_dither_phases == 3
    assert options.minimum_dither_phase_separation_pixels == 0.15
    assert options.minimum_dither_span_pixels == 0.35
    assert options.maximum_fwhm_for_upsampling_pixels == 3.0
    assert options.rejection_minimum_frames == 3
    options.validate()


def test_projective_registration_is_never_silently_truncated_for_ordinary_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    import ufwbpp_registration

    path = str(synthetic_project["lights"][0])
    projective = np.asarray(
        [[1.0, 0.0, 0.2], [0.0, 1.0, -0.1], [1.0e-5, -2.0e-5, 1.0]],
        dtype=np.float64,
    )
    transform = SimpleNamespace(
        path=path,
        filter_name="R",
        accepted=True,
        full_matrix=projective,
        reason=None,
        match_count=50,
        inlier_count=48,
        inlier_ratio=0.96,
        rms_preview_px=0.1,
        rms_full_px=0.1,
        warp_pearson=0.99,
        warp_valid_fraction=0.95,
    )
    fake_run = SimpleNamespace(transforms=(transform,))
    monkeypatch.setattr(ufwbpp_registration, "run_registration", lambda *args, **kwargs: fake_run)

    with pytest.raises(E2EError, match="REGISTRATION_PROJECTIVE_UNSUPPORTED"):
        _register_lights(
            (synthetic_project["lights"][0],),
            calibration_plan=None,
            detection=None,
            registration=SimpleNamespace(refine_full_centroids=True),
            workers=1,
            allow_projective=False,
        )



def test_default_e2e_registration_uses_full_resolution_affine_for_ordinary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ufwbpp_registration

    captured: dict[str, object] = {}

    def fail_after_capture(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(ufwbpp_registration, "run_registration", fail_after_capture)
    with pytest.raises(E2EError, match="captured"):
        _register_lights(
            (),
            SimpleNamespace(),
            detection=None,
            registration=None,
            workers=1,
            allow_projective=False,
        )
    selected = captured["registration"]
    assert selected.refine_full_centroids is True
    assert selected.full_transform_model == "affine"


def test_a_transform_resting_on_hot_pixels_fails_registration_with_its_numbers(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    import ufwbpp_registration

    def transform(path: Path, **fields: object) -> SimpleNamespace:
        return SimpleNamespace(
            path=str(path),
            filter_name="R",
            accepted=True,
            full_matrix=np.eye(3),
            reason=None,
            match_count=540,
            inlier_count=531,
            inlier_ratio=0.5,
            rms_preview_px=0.1,
            rms_full_px=0.1,
            warp_pearson=0.05,
            warp_valid_fraction=0.99,
            **fields,
        )

    # The refinement leaves the catalog model in place and says why.
    refused = transform(
        synthetic_project["lights"][0],
        transform_model="similarity",
        full_refine_evidence={
            "status": "REJECTED",
            "reason": "CONTROL_POINTS_SENSOR_FIXED",
            "finalInlierCount": 531,
            "sensorFixedControlPoints": {
                "check": "same-pixel-core-fraction-v1",
                "coreFractionMinimum": 0.75,
                "inlierCount": 529,
                "remainingObjects": "ELSEWHERE",
                "remainingConsensusInlierCount": 459,
                "remainingConsensusOffsetPreviewPixels": 7.26,
                "remainingConsensusOffsetMaximumPreviewPixels": 2.5,
            },
        },
    )
    gated = transform(
        synthetic_project["lights"][1],
        transform_model="similarity",
        full_refine_evidence={
            "status": "REJECTED",
            "reason": "FULL_MODEL_GATE_FAILED",
            "gateReasons": ["INLIER_RATIO_LOW"],
            "commonCentroidCount": 80,
            "finalInlierRatio": 0.2125,
        },
    )
    registration = SimpleNamespace(refine_full_centroids=True, full_transform_model="projective")
    for first, expected in (
        (
            refused,
            "529 of 531 control points are single pixels on the same sensor pixel of this Light and of the "
            "reference (hot pixels, not stars), and 459 other catalog objects register 7.3 preview pixels away "
            "(limit 2.5); calibrate the Lights with a matching Dark",
        ),
        (gated, "FULL_MODEL_GATE_FAILED (gateReasons=[INLIER_RATIO_LOW], commonCentroidCount=80, finalInlierRatio=0.2125)"),
    ):
        other = gated if first is refused else refused
        monkeypatch.setattr(
            ufwbpp_registration,
            "run_registration",
            lambda *args, first=first, other=other, **kwargs: SimpleNamespace(transforms=(first, other)),
        )
        with pytest.raises(E2EError) as raised:
            _register_lights(
                synthetic_project["lights"][:2],
                calibration_plan=None,
                detection=None,
                registration=registration,
                workers=1,
                allow_projective=True,
            )
        assert raised.value.code == "REGISTRATION_FULL_REFINEMENT_REJECTED"
        assert raised.value.path == first.path
        assert expected in str(raised.value)


@pytest.mark.parametrize("use_raw_flats", [True, False])
def test_standard_public_project_with_pi_masters_and_missing_metadata(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
    use_raw_flats: bool,
) -> None:
    """Exercise public recipe -> shared masters -> real panel pixels -> fixture solve."""
    from lightframeqc.xisf import XISF
    from ufwbpp.inventory import inventory_project
    from ufwbpp.recipe import Recipe
    from ufwbpp.runtime import build_e2e_request
    from ufwbpp.workflows.project import ProjectE2ERequest, run_project_e2e
    from ufwbpp.calibration.preflight import inspect_calibration

    paths: list[Path] = []
    for index, source in enumerate(synthetic_project["lights"][:8]):
        header = fits.getheader(source)
        del header["BAYERPAT"]
        paths.append(_write(tmp_path / "raw" / f"light-{index}.fits", fits.getdata(source), header))
    source_bias = np.median(np.stack([fits.getdata(path).astype(np.float64) for path in synthetic_project["biases"]]), axis=0)
    source_dark = np.median(np.stack([fits.getdata(path).astype(np.float64) for path in synthetic_project["darks"]]), axis=0)
    master_specs = [("Master Dark", "dark", source_dark / 65535.0, 60.0)]
    if use_raw_flats:
        master_specs.append(("Master Bias", "bias", source_bias / 65535.0, 0.001))
        # Enough raw Flats for the default robust integration policy, with
        # absent temperature metadata carried into the generated master.
        for index in range(20):
            source = synthetic_project["flats"][index % 3]
            header = fits.getheader(source)
            del header["CCD-TEMP"]
            del header["BAYERPAT"]
            header["COMMENT"] = str(index)
            paths.append(_write(tmp_path / "raw" / f"flat-{index}.fits", fits.getdata(source), header))
    else:
        flat = np.median(np.stack([fits.getdata(path).astype(np.float64) for path in synthetic_project["flats"]]), axis=0) - source_bias
        master_specs.append(("Master Flat", "flat", flat / np.median(flat), 2.0))
    for role, label, pixels, exposure in master_specs:
        path = tmp_path / f"{label}.xisf"
        XISF.write(str(path), pixels.astype(np.float32)[:, :, None], image_metadata={
            "id": "integration", "imageType": role.replace(" ", ""),
            "FITSKeywords": {
                "IMAGETYP": [{"value": repr(role), "comment": ""}],
                "EXPTIME": [{"value": str(exposure), "comment": ""}],
                "FILTER": [{"value": "'R'", "comment": ""}],
                "XBINNING": [{"value": "1", "comment": ""}],
                "YBINNING": [{"value": "1", "comment": ""}],
            },
        })
        paths.append(path)
    before = {path: _content_sha256(path) for path in paths}
    recipe = Recipe.from_dict({"calibration": {"workflow": "mono-standard-v1", "bias": "OPTIONAL"}})
    preflight = inspect_calibration([str(path) for path in paths], recipe)
    assert preflight["calibrationReady"], preflight["issues"]
    inventory = inventory_project(paths)
    base = build_e2e_request(inventory, recipe, tmp_path / "result", workers=2,
        ra_hint_degrees=150, dec_hint_degrees=20, field_of_view_degrees=3, search_radius_degrees=5)
    result = run_project_e2e(ProjectE2ERequest(inventory, base, base.output_directory), solver_backends=(FakeSolver(),))
    assert result.success, result.code
    assert len(result.passed_light_paths) == 8
    assert {path: _content_sha256(path) for path in paths} == before
    receipt = json.loads(Path(result.receipt_path).read_text())
    assert receipt["calibrationPolicy"]["workflow"] == "mono-standard-v1"
