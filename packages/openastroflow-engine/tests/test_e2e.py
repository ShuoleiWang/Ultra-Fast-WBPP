from __future__ import annotations

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
REGISTRATION_SOURCE = Path(__file__).resolve().parents[3] / "engine" / "native" / "python"
if str(REGISTRATION_SOURCE) not in sys.path:
    sys.path.insert(0, str(REGISTRATION_SOURCE))

from openastroflow_engine.drizzle_execution import (
    DrizzleExecutionRequest,
    DrizzleFrameInput,
    DrizzleProvider,
    execute_drizzle,
)
from openastroflow_engine.e2e import (
    DrizzleOptions,
    E2ERequest,
    E2EState,
    IntegrationMode,
    ProgressEvent,
    ReviewApproval,
    run_e2e,
)
from openastroflow_engine.e2e import (
    E2EError,
    _SolverHints,
    _build_registration_masters,
    _build_drizzle_rejection_masks,
    _drizzle_output_to_input_matrix,
    _drizzle_sampling_evidence,
    _inferred_solver_hints,
    _register_lights,
    _share_safe_receipt_core,
    _share_safe_string,
    _solve_one,
    _SourceIdentity,
    _unify_same_grid_solutions,
    _validate_cross_filter_wcs,
)
from openastroflow_engine.solver import (
    AstrometricQuality,
    SolutionKind,
    SolverResult,
    SolverIndexArtifact,
    SolverStatus,
    WcsParity,
)
from openastroflow_engine.pixel_pipeline import (
    MasterMetadataOverride,
    PipelineParameters,
    RawFrameMetadataOverride,
    run_portable_pipeline,
)
from openastroflow_engine.project_e2e import _build_shared_calibration


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
                "path": str(tmp_path / "build" / "libopenastroflow_native.dylib"),
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
    assert receipt["nativeLibrary"]["path"] == "local-redacted/libopenastroflow_native.dylib"
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


class NumpyPointAccumulator:
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
        self._sum = np.zeros(out_shape, dtype=np.float64)
        self._weight = np.zeros(out_shape, dtype=np.float32)

    @property
    def out_wht(self) -> np.ndarray:
        return self._weight

    @property
    def out_img(self) -> np.ndarray:
        output = np.full(self._weight.shape, np.nan, dtype=np.float32)
        selected = self._weight > 0
        output[selected] = (self._sum[selected] / self._weight[selected]).astype(np.float32)
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
        del exptime, pixfrac, pixel_scale_ratio
        assert in_units == "cps"
        x = np.floor(pixmap[..., 0] + 0.5).astype(np.int64)
        y = np.floor(pixmap[..., 1] + 0.5).astype(np.int64)
        weights = np.asarray(weight_map, dtype=np.float64) * wht_scale
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
        np.add.at(self._sum, (y[valid], x[valid]), data[valid] * weights[valid])
        np.add.at(self._weight, (y[valid], x[valid]), weights[valid].astype(np.float32))


def _drizzle_provider() -> DrizzleProvider:
    return DrizzleProvider("numpy-e2e-oracle", "test-v1", NumpyPointAccumulator)


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
        drizzle=DrizzleOptions(scale=2, pixfrac=1.0, kernel="point", tile_rows=64),
    )


def _all_source_bytes(project: dict[str, tuple[Path, ...]]) -> dict[Path, bytes]:
    return {path: path.read_bytes() for values in project.values() for path in values}


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def test_solver_hints_derive_nina_fov_and_ignore_conflicting_user_constraints(
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

    hints = _inferred_solver_hints(request, frames)

    assert hints.ra_degrees == pytest.approx(282.14, abs=1e-6)
    assert hints.dec_degrees == pytest.approx(-7.40)
    assert hints.field_of_view_degrees == pytest.approx(1.122, abs=0.002)
    assert hints.evidence["coordinates"]["consistency"] == "CONFLICT"
    assert hints.evidence["coordinates"]["selection"] == "metadata-consensus"
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

    hints = _inferred_solver_hints(request, (frame,))

    assert hints.ra_degrees == 282.15
    assert hints.dec_degrees == -7.41
    assert hints.field_of_view_degrees == 1.12
    assert hints.evidence["coordinates"]["consistency"] == "CONSISTENT"
    assert hints.evidence["fieldOfView"]["consistency"] == "CONSISTENT"


@pytest.mark.skipif(
    os.environ.get("OAF_RUN_REAL_METAL") != "1",
    reason="opt-in test requires a real Apple Metal device",
)
def test_real_m3_pro_e2e_receipt_proves_metal_production_path(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    output = tmp_path / "real-metal-e2e"
    result = run_e2e(
        _request(synthetic_project, output),
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
    import openastroflow_engine.e2e as e2e_module
    import openastroflow_engine.pixel_pipeline as pixel_module

    e2e_master_integrations: list[str] = []
    e2e_master_sources: list[str] = []
    pixel_master_integrations: list[str] = []
    captured: dict[str, Any] = {}
    original_e2e_integrate = e2e_module.integrate_expressions
    original_pixel_integrate = pixel_module.integrate_expressions
    original_pixel_run = e2e_module._run_portable_pipeline_fits

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

    monkeypatch.setattr(e2e_module, "integrate_expressions", count_e2e_integrations)
    monkeypatch.setattr(pixel_module, "integrate_expressions", count_pixel_integrations)
    monkeypatch.setattr(e2e_module, "_run_portable_pipeline_fits", capture_pixel_run)

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
    import openastroflow_engine.e2e as e2e_module

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
    original = e2e_module._run_portable_pipeline_fits

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
        e2e_module, "_run_portable_pipeline_fits", tamper_before_consume
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
    import openastroflow_engine.e2e as e2e_module
    import openastroflow_engine.pixel_pipeline as pixel_module

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
    original_e2e_hash = e2e_module._sha256
    original_pixel_hash = pixel_module._hash_file
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

    monkeypatch.setattr(e2e_module, "_sha256", count_e2e)
    monkeypatch.setattr(pixel_module, "_hash_file", count_pixel)
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
    from openastroflow_registration import DetectionConfig, analyze_frames

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
    plan, receipt = _build_registration_masters(
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

    assert set(plan.dark_paths) == {60.0, 120.0}
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
    from xisf import XISF
    from openastroflow_engine.inventory import inventory_project
    from openastroflow_engine.project_e2e import ProjectE2ERequest, run_project_e2e

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


def test_single_admitted_light_stops_before_calibration_and_preserves_qc(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openastroflow_engine.e2e as e2e_module
    from lightframeqc.models import GateDisposition

    original_gate = e2e_module.evaluate_quality_gate

    def one_admitted(results, measurements, policy):
        original_gate(results, measurements, policy)
        for index, result in enumerate(results):
            result.quality_gate.disposition = GateDisposition.PASS if index == 0 else GateDisposition.REVIEW

    def calibration_must_not_start(*args, **kwargs):
        pytest.fail("Calibration must not start when only one Light is admitted")

    monkeypatch.setattr(e2e_module, "evaluate_quality_gate", one_admitted)
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


def test_solver_failure_blocks_final_publication_and_retains_unsolved_evidence(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    output = tmp_path / "solver-miss"

    result = run_e2e(
        _request(synthetic_project, output),
        solver_backends=(FailingSolver(),),
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


def test_drizzle_branch_produces_coverage_then_solves_the_drizzled_master(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    output = tmp_path / "drizzle-solved"
    solver = FakeSolver()

    result = run_e2e(
        _request(synthetic_project, output, mode=IntegrationMode.DRIZZLE),
        solver_backends=(solver,),
        drizzle_provider=_drizzle_provider(),
    )

    assert result.success is True
    assert len(solver.inputs) == 1
    assert "drizzle_unsolved.fits" in solver.inputs[0]
    coverage = json.loads((output / "coverage" / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["mode"] == "drizzle"
    assert coverage["options"]["scale"] == 2
    drizzle_receipt = coverage["filters"]["R"]
    assert drizzle_receipt["statistics"]["coverageFraction"] >= 0.90
    assert drizzle_receipt["statistics"]["nullPixelFraction"] <= 0.10
    assert drizzle_receipt["statistics"]["rejectionMasksProvided"] == 8
    assert drizzle_receipt["statistics"]["allFramesHaveRejectionMasks"] is True
    assert drizzle_receipt["scienceGate"]["dither"]["distinctPhaseCount"] >= 3
    assert drizzle_receipt["scienceGate"]["dither"]["spanXPixels"] >= 0.35
    assert drizzle_receipt["scienceGate"]["dither"]["spanYPixels"] >= 0.35
    assert coverage["sampling"]["status"] == "PASS_UNDERSAMPLED"
    mask_directory = output / "coverage" / "rejection-masks" / "R"
    assert len(tuple(mask_directory.glob("*_rejection.fits"))) == 8
    mask_manifest = json.loads(
        (mask_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(mask_manifest["frames"]) == 8
    assert mask_manifest["totalRejectedPixels"] == 0
    assert (output / "receipts" / "drizzle_R.json").is_file()
    with fits.open(result.product_paths[0]) as hdul:
        assert [item.name for item in hdul] == ["SCI", "WHT", "COVERAGE"]
        assert hdul[0].header["CTYPE1"] == "RA---TAN"
        assert hdul["COVERAGE"].data.shape == (256, 256)


def test_sampling_gate_blocks_well_sampled_and_reviews_unknown_twox_data() -> None:
    metadata = SimpleNamespace(header={})
    unknown = SimpleNamespace(
        features=SimpleNamespace(
            median_fwhm_native_pixels=None,
            nina_hfr_pixels=None,
        ),
        metadata=metadata,
    )
    with pytest.raises(E2EError) as unknown_error:
        _drizzle_sampling_evidence((unknown,), DrizzleOptions(scale=2))
    assert unknown_error.value.code == "DRIZZLE_SAMPLING_REVIEW_REQUIRED"

    well_sampled = SimpleNamespace(
        features=SimpleNamespace(
            median_fwhm_native_pixels=3.4,
            nina_hfr_pixels=None,
        ),
        metadata=metadata,
    )
    with pytest.raises(E2EError) as sampled_error:
        _drizzle_sampling_evidence((well_sampled,), DrizzleOptions(scale=2))
    assert sampled_error.value.code == "DRIZZLE_UPSCALE_NOT_RECOMMENDED"

    conflicting = SimpleNamespace(
        features=SimpleNamespace(
            median_fwhm_native_pixels=2.0,
            nina_hfr_pixels=2.0,
        ),
        metadata=metadata,
    )
    with pytest.raises(E2EError) as conflicting_error:
        _drizzle_sampling_evidence((conflicting,), DrizzleOptions(scale=2))
    assert conflicting_error.value.code == "DRIZZLE_SAMPLING_REVIEW_REQUIRED"


def test_drizzle_options_expose_fail_closed_production_defaults() -> None:
    options = DrizzleOptions()

    assert options.minimum_coverage_fraction == 0.90
    assert options.maximum_null_fraction == 0.10
    assert options.minimum_distinct_dither_phases == 3
    assert options.minimum_dither_phase_separation_pixels == 0.15
    assert options.minimum_dither_span_pixels == 0.35
    assert options.maximum_fwhm_for_upsampling_pixels == 3.0
    assert options.rejection_minimum_frames == 3
    options.validate()


def test_tiled_mad_masks_reject_a_real_per_frame_outlier(
    tmp_path: Path,
) -> None:
    calibrated: dict[str, Path] = {}
    transforms: dict[str, tuple[tuple[float, float, float], ...]] = {}
    analyses: dict[str, tuple[int, Any]] = {}
    paths: list[Path] = []
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for index in range(3):
        source = tmp_path / "sources" / f"light-{index}.fits"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"source-{index}".encode("ascii"))
        path = source.resolve(strict=True)
        calibrated_path = tmp_path / "calibrated" / f"light-{index}.fits"
        calibrated_path.parent.mkdir(parents=True, exist_ok=True)
        values = np.full((8, 8), 10.0, dtype=np.float32)
        if index == 2:
            values[3, 4] = 1000.0
        fits.writeto(calibrated_path, values, overwrite=False)
        canonical = str(path)
        paths.append(path)
        calibrated[canonical] = calibrated_path
        transforms[canonical] = identity
        analyses[canonical] = (
            index,
            SimpleNamespace(total_integrated_flux=1000.0),
        )

    mask_paths, manifest = _build_drizzle_rejection_masks(
        calibrated=calibrated,
        transforms=transforms,
        paths=tuple(paths),
        analysis_by_path=analyses,
        reference_shape=(8, 8),
        directory=tmp_path / "masks",
        options=DrizzleOptions(scale=1, tile_rows=3),
        source_exposure_seconds={str(path): 60.0 for path in paths},
    )

    with fits.open(mask_paths[str(paths[2])]) as hdul:
        assert hdul["MASK"].data[3, 4] == 1
    assert manifest["totalRejectedPixels"] >= 1
    frames = tuple(
        DrizzleFrameInput(
            calibrated_path=str(calibrated[str(path)]),
            output_to_input_projective=identity,
            rejection_mask_path=str(mask_paths[str(path)]),
            rejection_mask_hdu="MASK",
        )
        for path in paths
    )
    drizzle_request = DrizzleExecutionRequest(
        frames=frames,
        output_path=str(tmp_path / "drizzle" / "master.fits"),
        receipt_path=str(tmp_path / "drizzle" / "receipt.json"),
        output_shape=(8, 8),
        scale=1,
        pixfrac=1.0,
        kernel="point",
        tile_rows=3,
        minimum_distinct_dither_phases=1,
        minimum_dither_span_pixels=0.0,
    )
    result = execute_drizzle(drizzle_request, provider=_drizzle_provider())

    assert result.completed is True
    with fits.open(result.output_path) as hdul:
        assert hdul["SCI"].data[3, 4] == pytest.approx(10.0)
    assert result.receipt is not None
    assert result.receipt["statistics"]["rejectionMaskPixels"] >= 1
    assert result.receipt["inputs"][2]["rejection"]["maskRejectedPixels"] >= 1


def test_drizzle_rejection_does_not_double_apply_mixed_exposure_scaling(
    tmp_path: Path,
) -> None:
    calibrated: dict[str, Path] = {}
    transforms: dict[str, tuple[tuple[float, float, float], ...]] = {}
    analyses: dict[str, tuple[int, Any]] = {}
    exposures: dict[str, float] = {}
    paths: list[Path] = []
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    for index, exposure in enumerate((60.0, 300.0, 300.0)):
        source = tmp_path / "mixed-sources" / f"light-{index}.fits"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"source-{index}".encode("ascii"))
        path = source.resolve(strict=True)
        calibrated_path = tmp_path / "mixed-calibrated" / f"light-{index}.fits"
        calibrated_path.parent.mkdir(parents=True, exist_ok=True)
        y, x = np.indices((8, 8), dtype=np.float32)
        fits.writeto(calibrated_path, 10.0 + x + 2.0 * y, overwrite=False)
        canonical = str(path)
        paths.append(path)
        calibrated[canonical] = calibrated_path
        transforms[canonical] = identity
        analyses[canonical] = (
            index,
            SimpleNamespace(total_integrated_flux=1000.0 * exposure),
        )
        exposures[canonical] = exposure

    mask_paths, manifest = _build_drizzle_rejection_masks(
        calibrated=calibrated,
        transforms=transforms,
        paths=tuple(paths),
        analysis_by_path=analyses,
        reference_shape=(8, 8),
        directory=tmp_path / "mixed-masks",
        options=DrizzleOptions(scale=1, tile_rows=3),
        source_exposure_seconds=exposures,
    )

    assert manifest["totalRejectedPixels"] == 0
    assert [item["photometricScale"] for item in manifest["frames"]] == [1.0, 1.0, 1.0]
    for path in paths:
        with fits.open(mask_paths[str(path)]) as hdul:
            assert int(np.count_nonzero(hdul["MASK"].data)) == 0


def test_projective_registration_is_never_silently_truncated_for_ordinary_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_project: dict[str, tuple[Path, ...]],
) -> None:
    import openastroflow_registration

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
    monkeypatch.setattr(openastroflow_registration, "run_registration", lambda *args, **kwargs: fake_run)

    with pytest.raises(E2EError, match="REGISTRATION_PROJECTIVE_UNSUPPORTED"):
        _register_lights(
            (synthetic_project["lights"][0],),
            calibration_plan=None,
            detection=None,
            registration=SimpleNamespace(refine_full_centroids=True),
            workers=1,
            allow_projective=False,
        )

    # The drizzle bridge preserves the same full homography and composes only
    # the requested output scale.  Verify the mapping on an arbitrary point.
    output_to_input = np.asarray(
        _drizzle_output_to_input_matrix(projective, scale=2), dtype=np.float64
    )
    source = np.asarray([41.5, 73.25, 1.0])
    reference = projective @ source
    reference /= reference[2]
    high_resolution_output = np.asarray(
        [2.0 * reference[0], 2.0 * reference[1], 1.0]
    )
    recovered = output_to_input @ high_resolution_output
    recovered /= recovered[2]
    np.testing.assert_allclose(recovered, source, rtol=0.0, atol=1e-10)
    assert not np.allclose(output_to_input[2], (0.0, 0.0, 1.0))


def test_default_e2e_registration_uses_full_resolution_affine_for_ordinary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openastroflow_registration

    captured: dict[str, object] = {}

    def fail_after_capture(*args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(openastroflow_registration, "run_registration", fail_after_capture)
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


@pytest.mark.parametrize("use_raw_flats", [True, False])
def test_standard_public_project_with_pi_masters_and_missing_metadata(
    tmp_path: Path,
    synthetic_project: dict[str, tuple[Path, ...]],
    use_raw_flats: bool,
) -> None:
    """Exercise public recipe -> shared masters -> real panel pixels -> fixture solve."""
    from xisf import XISF
    from openastroflow_engine.inventory import inventory_project
    from openastroflow_engine.recipe import Recipe
    from openastroflow_engine.runtime import build_e2e_request
    from openastroflow_engine.project_e2e import ProjectE2ERequest, run_project_e2e
    from openastroflow_engine.calibration_preflight import inspect_calibration

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
        ra_hint_degrees=150, dec_hint_degrees=20, field_of_view_degrees=3, search_radius_degrees=5,
        requested_hardware_profile="portable-cpu")
    result = run_project_e2e(ProjectE2ERequest(inventory, base, base.output_directory), solver_backends=(FakeSolver(),))
    assert result.success, result.code
    assert len(result.passed_light_paths) == 8
    assert {path: _content_sha256(path) for path in paths} == before
    receipt = json.loads(Path(result.receipt_path).read_text())
    assert receipt["calibrationPolicy"]["workflow"] == "mono-standard-v1"
