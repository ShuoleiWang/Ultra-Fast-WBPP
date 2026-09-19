"""Inject known defects into real Light frames and check the selection policy.

Copies a set of real raw Lights, injects one synthetic defect per chosen
frame (thin cloud, patchy cloud, dew, defocus, occlusion, trailing) plus
benign controls (uniform dimming, pointing shift, brighter sky), runs the E2E
pipeline with the requested selection policy and the counterfactual oracle,
and tabulates per frame: quality-gate disposition, selection action and
confidence, and the oracle verdict.  Recall = defects that were excluded or
down-weighted; false positives = benign frames that lost weight.

    .venv/bin/python benchmarks/selection_defect_injection.py \
        --lights '/path/DATE_0322/*300.00s*.fits' \
        --master-bias masterBias.xisf --master-dark masterDark.xisf \
        [--flats '/path/DATE_0322/*FLAT*.fits' | --master-flats '/path/masterFlat_*FILTER-R*.xisf'] \
        --policy unattended-v1 --output benchmarks/results/<name>.json \
        --scratch $SCRATCH/defect-injection

The astrometric solve is faked (the repository's FakeSolver), so the run
measures screening, registration, normalization and integration only.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import glob
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable

from astropy.io import fits
import numpy as np
from scipy import ndimage

REPOSITORY = Path(__file__).resolve().parents[1]
for relative in (
    "packages/openastroflow-engine/src",
    "packages/light-frame-qc/src",
    "engine/native/python",
    "packages/openastroflow-engine/tests",
):
    candidate = REPOSITORY / relative
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from openastroflow_engine.calibration_policy import MONO_STANDARD  # noqa: E402
from openastroflow_engine.e2e import E2ERequest, IntegrationMode, run_e2e  # noqa: E402
from openastroflow_engine.pixel_pipeline import PipelineParameters  # noqa: E402
from openastroflow_engine.selection import SelectionParameters  # noqa: E402
from openastroflow_engine.solver import (  # noqa: E402
    AstrometricQuality,
    SolutionKind,
    SolverIndexArtifact,
    SolverResult,
    SolverStatus,
    WcsParity,
)
from test_e2e import FakeSolver  # noqa: E402


class HintFakeSolver(FakeSolver):
    """A fake solver that places the WCS at the request's own hints.

    Real N.I.N.A. frames carry coordinates, so the E2E verifies the solved
    centre and scale against the hints it derived from the headers; the
    repository FakeSolver's fixed centre would fail that gate.
    """

    def solve(self, request: Any) -> SolverResult:
        self.inputs.append(request.input_path)
        ra = float(request.ra_hint_degrees) if request.ra_hint_degrees is not None else self.center_ra_degrees
        dec = float(request.dec_hint_degrees) if request.dec_hint_degrees is not None else 20.0
        field = (
            float(request.field_of_view_degrees)
            if request.field_of_view_degrees
            else self.field_width_degrees
        )
        with fits.open(request.input_path, mode="readonly", memmap=False) as hdul:
            image_hdu = next(item for item in hdul if item.data is not None and item.data.ndim == 2)
            height, width = image_hdu.data.shape
            primary = hdul[0].header
            primary["CTYPE1"] = "RA---TAN"
            primary["CTYPE2"] = "DEC--TAN"
            primary["CRPIX1"] = (width + 1.0) / 2.0
            primary["CRPIX2"] = (height + 1.0) / 2.0
            primary["CRVAL1"] = ra
            primary["CRVAL2"] = dec
            pixel_scale_degrees = field / (width - 1)
            primary["CD1_1"] = -pixel_scale_degrees
            primary["CD1_2"] = 0.0
            primary["CD2_1"] = 0.0
            primary["CD2_2"] = pixel_scale_degrees
            primary["OAFSTATE"] = "SOLVED"
            hdul.writeto(request.output_path, overwrite=False, checksum=True)
            solved_header = primary.copy()
        quality = AstrometricQuality(
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
                "astrometricQuality": quality.serializable(),
            },
        )

Injector = Callable[[np.ndarray, float, float, np.random.Generator], np.ndarray]


def _pedestal_and_sky(image: np.ndarray) -> tuple[float, float]:
    pedestal = float(np.percentile(image, 0.5))
    sky = float(np.median(image))
    return pedestal, sky


def _line_kernel(length: int, angle_degrees: float) -> np.ndarray:
    size = length + 4
    kernel = np.zeros((size, size))
    centre = size // 2
    theta = np.deg2rad(angle_degrees)
    for t in np.linspace(-length / 2, length / 2, length * 4):
        y = int(round(centre + t * np.sin(theta)))
        x = int(round(centre + t * np.cos(theta)))
        kernel[y, x] += 1.0
    return kernel / kernel.sum()


def _disk_kernel(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = ((yy * yy + xx * xx) <= radius * radius).astype(float)
    return kernel / kernel.sum()


def thin_cloud(image, pedestal, sky, rng):
    signal = image - pedestal
    return pedestal + signal * 0.65 + 0.35 * (sky - pedestal)


def patchy_cloud(image, pedestal, sky, rng):
    height, width = image.shape
    yy, xx = np.indices(image.shape, dtype=np.float64)
    loss = 0.55 * np.exp(-(((yy - height * 0.25) ** 2 + (xx - width * 0.2) ** 2) / (2 * (0.18 * width) ** 2)))
    transparency = 1.0 - loss
    signal = image - pedestal
    return pedestal + signal * transparency + loss * (sky - pedestal) * 0.8


def dew(image, pedestal, sky, rng):
    signal = image - pedestal
    core = ndimage.gaussian_filter(signal, 2.2)
    halo = ndimage.gaussian_filter(signal, 14.0)
    return pedestal + 0.85 * core + 0.15 * halo


def defocus(image, pedestal, sky, rng):
    signal = image - pedestal
    return pedestal + ndimage.convolve(signal, _disk_kernel(5), mode="nearest")


def occlusion(image, pedestal, sky, rng):
    height, width = image.shape
    mask = np.zeros(image.shape)
    mask[: int(height * 0.5), : int(width * 0.5)] = 1.0  # 25% of the area
    mask = ndimage.gaussian_filter(mask, 25.0)
    blocked = pedestal + 0.15 * (sky - pedestal) + rng.normal(0, 3.0, image.shape)
    return image * (1 - mask) + blocked * mask


def trailing(image, pedestal, sky, rng):
    signal = image - pedestal
    return pedestal + ndimage.convolve(signal, _line_kernel(14, 25.0), mode="nearest")


def benign_dim(image, pedestal, sky, rng):
    return pedestal + (image - pedestal) * 0.85


def benign_shift(image, pedestal, sky, rng):
    shifted = np.roll(np.roll(image, 30, axis=0), 40, axis=1)
    shifted[:30, :] = sky
    shifted[:, :40] = sky
    return shifted


def benign_bright_sky(image, pedestal, sky, rng):
    return image + 0.25 * (sky - pedestal)


CASES: dict[str, tuple[str, Injector]] = {
    "thin_cloud": ("defect", thin_cloud),
    "patchy_cloud": ("defect", patchy_cloud),
    "dew": ("defect", dew),
    "defocus": ("defect", defocus),
    "occlusion": ("defect", occlusion),
    "trailing": ("defect", trailing),
    "benign_dim": ("benign", benign_dim),
    "benign_shift": ("benign", benign_shift),
    "benign_bright_sky": ("benign", benign_bright_sky),
}


def inject(source: Path, destination: Path, case: str | None, rng: np.random.Generator) -> dict[str, Any]:
    if case is None:
        shutil.copyfile(source, destination)
        return {"case": None, "kind": "clean"}
    kind, injector = CASES[case]
    with fits.open(source, memmap=False) as hdul:
        header = hdul[0].header.copy()
        image = np.asarray(hdul[0].data, dtype=np.float64)
    pedestal, sky = _pedestal_and_sky(image)
    modified = injector(image, pedestal, sky, rng)
    data = np.clip(np.rint(modified), 0, 65535).astype(np.uint16)
    header["HISTORY"] = f"ultra-fast-wbpp defect injection: {case}"
    fits.PrimaryHDU(data=data, header=header).writeto(destination, overwrite=False)
    return {"case": case, "kind": kind, "pedestal": pedestal, "sky": sky}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lights", required=True, help="glob of raw Light frames (one filter)")
    parser.add_argument("--flats", default=None, help="glob of raw Flat frames (optional)")
    parser.add_argument("--master-flats", default=None, help="glob of supplied MasterFlat files (optional, e.g. the filter's masterFlat_*.xisf)")
    parser.add_argument("--master-bias", required=True)
    parser.add_argument("--master-dark", required=True)
    parser.add_argument("--count", type=int, default=12, help="number of Lights to use")
    parser.add_argument("--policy", default="unattended-v1", choices=("unattended-v1", "include-all"))
    parser.add_argument("--priority", default="balanced")
    parser.add_argument("--region-weights", dest="region_weights", action="store_true", default=None, help="enable per-frame region weight maps (the policy default)")
    parser.add_argument("--no-region-weights", dest="region_weights", action="store_false", help="disable per-frame region weight maps")
    parser.add_argument("--max-passes", type=int, default=None, help="cap the counterfactual integration passes (diagnostics)")
    parser.add_argument("--cases", default=",".join(CASES), help="comma-separated case names to inject, one per frame from index 6")
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"{output} already exists")
    lights = sorted(Path(item) for item in glob.glob(args.lights))
    if len(lights) < args.count:
        parser.error(f"only {len(lights)} Lights match; need {args.count}")
    lights = lights[: args.count]
    cases = [item for item in args.cases.split(",") if item]
    unknown = sorted(set(cases) - set(CASES))
    if unknown:
        parser.error(f"unknown cases: {unknown}")
    if 6 + len(cases) > len(lights):
        parser.error(f"{len(cases)} cases need at least {6 + len(cases)} Lights (six stay clean)")
    scratch = Path(args.scratch).resolve()
    injected_dir = scratch / "lights"
    injected_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    assignments: dict[str, dict[str, Any]] = {}
    injected_paths: list[str] = []
    started = time.perf_counter()
    for index, source in enumerate(lights):
        case = cases[index - 6] if index >= 6 and index - 6 < len(cases) else None
        destination = injected_dir / source.name
        if destination.exists():
            destination.unlink()
        # Published receipts must not carry a local filesystem layout, so the
        # report records file names only (unique within a run); the absolute
        # destination path stays in memory as the join key.
        assignments[str(destination.resolve())] = {"index": index, "source": source.name, **inject(source, destination, case, rng)}
        injected_paths.append(str(destination.resolve()))
    injection_seconds = time.perf_counter() - started

    request = E2ERequest(
        light_files=tuple(injected_paths),
        flat_files=tuple(sorted(glob.glob(args.flats))) if args.flats else (),
        bias_files=(),
        master_bias_files=(str(Path(args.master_bias).resolve()),),
        master_dark_files=(str(Path(args.master_dark).resolve()),),
        master_flat_files=(
            tuple(str(Path(item).resolve()) for item in sorted(glob.glob(args.master_flats)))
            if args.master_flats
            else ()
        ),
        output_directory=str(scratch / f"run-{args.policy}"),
        integration_mode=IntegrationMode.ORDINARY,
        workers=args.workers,
        pipeline_parameters=replace(PipelineParameters(), calibration_workflow=MONO_STANDARD),
        selection=SelectionParameters(
            policy=args.policy,
            priority=args.priority,
            **({} if args.region_weights is None else {"region_weights": bool(args.region_weights)}),
            **({"max_integration_passes": int(args.max_passes)} if args.max_passes else {}),
        ),
        # Hints are derived from the N.I.N.A. headers by the engine; none are
        # imposed here so the fake solver follows the frames.
    )
    run_started = time.perf_counter()
    result = run_e2e(request, solver_backends=(HintFakeSolver(),))
    run_seconds = time.perf_counter() - run_started
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "ultra-fast-wbpp-selection-defect-injection-v1",
        "recordedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "policy": args.policy,
        "priority": args.priority,
        "success": result.success,
        "code": result.code,
        "message": getattr(result, "message", None),
        "injectionSeconds": injection_seconds,
        "runSeconds": run_seconds,
        "frames": [],
    }
    if result.success:
        root = Path(result.output_directory)
        manifest = json.loads((root / "qc" / "manifest.json").read_text(encoding="utf-8"))
        selection = json.loads((root / "qc" / "selection.json").read_text(encoding="utf-8"))
        # Published receipts carry share-safe source paths (source/<id>/<name>),
        # so frames are joined on their file name (unique within the run).
        gate = {Path(item["path"]).name: (item.get("qualityGate") or {}) for item in manifest["frames"]}
        decisions = {Path(item["path"]).name: item for item in selection["frames"]}
        oracle: dict[str, dict[str, Any]] = {}
        for group in (selection.get("counterfactual") or {}).values():
            for frame in group["frames"]:
                oracle[Path(frame["path"]).name] = frame
        parameters = selection["parameters"]
        for path, assignment in assignments.items():
            name = Path(path).name
            decision = decisions.get(name, {})
            frame_oracle = oracle.get(name)
            verdict = None
            if frame_oracle is not None:
                depth_ci = frame_oracle.get("deltaDepthCi") or [None, None]
                background_ci = frame_oracle.get("deltaBackgroundCi") or [None, None]
                depth_beneficial = depth_ci[1] is not None and depth_ci[1] < -parameters["beneficialDepthMag"]
                background_harmful = (
                    background_ci[0] is not None and background_ci[0] > parameters["harmfulBackgroundSigma"]
                )
                # The policy's rule: background structure only counts against a
                # frame that does not deepen the master.
                if (depth_ci[0] is not None and depth_ci[0] > parameters["harmfulDepthMag"]) or (
                    background_harmful and not depth_beneficial
                ):
                    verdict = "HARMFUL"
                elif depth_beneficial:
                    verdict = "BENEFICIAL"
                else:
                    verdict = "NEUTRAL"
            flagged = decision.get("action") == "EXCLUDE" or (decision.get("confidence", 1.0) < 1.0)
            report["frames"].append(
                {
                    **assignment,
                    "name": name,
                    "gate": gate.get(name, {}).get("disposition"),
                    "gateCodes": [item["code"] for item in gate.get(name, {}).get("evidence", []) if item["severity"] in ("REVIEW", "ERROR", "HARD_FAIL")],
                    "action": decision.get("action"),
                    "confidence": decision.get("confidence"),
                    "reasons": [item["code"] for item in decision.get("reasons", [])],
                    "oracle": verdict,
                    "deltaDepthMag": frame_oracle.get("deltaDepthMag") if frame_oracle else None,
                    "deltaDepthCi": frame_oracle.get("deltaDepthCi") if frame_oracle else None,
                    "deltaBackgroundSigma": frame_oracle.get("deltaBackgroundSigma") if frame_oracle else None,
                    "deltaBackgroundCi": frame_oracle.get("deltaBackgroundCi") if frame_oracle else None,
                    "deltaFwhmPx": frame_oracle.get("deltaFwhmPx") if frame_oracle else None,
                    "psfFactor": decision.get("psfFactor"),
                    "weightMultiplier": decision.get("weightMultiplier"),
                    "suggestion": decision.get("suggestion"),
                    "flagged": flagged,
                }
            )
        defects = [item for item in report["frames"] if item["kind"] == "defect"]
        benign = [item for item in report["frames"] if item["kind"] == "benign"]
        clean = [item for item in report["frames"] if item["kind"] == "clean"]
        report["summary"] = {
            "defectRecall": sum(1 for item in defects if item["flagged"]) / max(1, len(defects)),
            "defectOracleHarmful": sum(1 for item in defects if item["oracle"] == "HARMFUL"),
            "benignFalsePositives": sum(1 for item in benign if item["flagged"]),
            "cleanFalsePositives": sum(1 for item in clean if item["flagged"]),
            "defects": len(defects),
            "benign": len(benign),
            "clean": len(clean),
        }
        print(f"{'frame':6s} {'case':18s} {'gate':9s} {'action':8s} {'conf':5s} {'oracle':10s} dDepth   dBg")
        for item in sorted(report["frames"], key=lambda row: row["index"]):
            print(
                f"{item['index']:6d} {str(item['case']):18s} {str(item['gate']):9s} {str(item['action']):8s} "
                f"{item['confidence'] if item['confidence'] is not None else '-':5} {str(item['oracle']):10s} "
                f"{item['deltaDepthMag'] if item['deltaDepthMag'] is None else round(item['deltaDepthMag'], 4)!s:8} "
                f"{item['deltaBackgroundSigma'] if item['deltaBackgroundSigma'] is None else round(item['deltaBackgroundSigma'], 3)}"
            )
        print("summary:", json.dumps(report["summary"]))
    else:
        print("run failed:", result.code, getattr(result, "message", ""))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print("wrote", output)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
