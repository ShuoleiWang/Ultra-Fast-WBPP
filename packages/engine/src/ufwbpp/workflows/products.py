"""The candidate products of a run (ordinary masters, drizzle, proper coadds) and how each is solved."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Mapping, Sequence

from astropy.io import fits
import numpy as np

from ..integrity import sha256_digest
from ..solvers.base import SolverBackend, validate_wcs_header
from ..stacking.drizzle_native import DrizzleGroupRequest, drizzle_group, verify_drizzle_receipt
from ..stacking.integration import CalibrationError, read_frame_info
from .common import _emit, _relativize_solver_attempts, _safe_token
from .contracts import IntegrationMode, ProgressStage, E2EError, DrizzleOptions, E2ERequest, ProgressCallback
from .solve import _read_image_header, _wcs_grid_disagreement, _solve_one, _SolverHints, _WCS_CARD_PATTERN


def _drizzle_candidates(
    *,
    staging: Path,
    work: Path,
    pipeline_result: Any,
    options: DrizzleOptions,
    sampling_evidence: Mapping[str, Any],
    threads: int | None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Drizzle every filter group from the ordinary integration's products.

    The pixel pipeline hands over, per group, the calibrated (unregistered)
    Lights, their registration matrices, the normalization coefficients, the
    integration weights and the per-sample rejection masks; the native drizzle
    reproduces the group's integration on the finer grid from exactly those.
    """

    groups = getattr(pipeline_result, "drizzle_groups", None) or {}
    if not groups:
        raise E2EError(
            "DRIZZLE_INPUTS_MISSING",
            "the pixel pipeline did not capture drizzle inputs for any filter group",
        )
    candidates: dict[str, Path] = {}
    receipts: dict[str, Any] = {}
    for filter_name, group in sorted(groups.items()):
        token = _safe_token(filter_name)
        target_dir = work / "drizzle" / token
        target_dir.mkdir(parents=True, exist_ok=False)
        # A colour channel group of a Bayer filter is a Bayer drizzle: the
        # calibrated mosaics' own samples of that colour are dropped, with the
        # channel group's normalization, weights and rejection.
        request = DrizzleGroupRequest(
            frames=group.frames,
            reference_shape=tuple(int(value) for value in group.reference_shape),
            output_path=str(target_dir / f"master_light_{token}_drizzle_unsolved.fits"),
            receipt_path=str(target_dir / "receipt.json"),
            scale=options.scale,
            pixfrac=options.pixfrac,
            kernel=options.kernel,
            cfa_pattern=group.cfa_pattern,
            channel=group.channel,
            metadata={**dict(group.metadata), "OAFCROP": "NONE"},
            max_accumulator_bytes=options.max_working_set_bytes,
            threads=threads,
            durable=False,
        )
        try:
            result = drizzle_group(request)
            verified = verify_drizzle_receipt(result.receipt_path)
        except CalibrationError as error:
            raise E2EError(error.code, str(error), path=error.path) from error
        statistics = verified.get("statistics", {})
        coverage_status = "PASS"
        if float(statistics.get("coverageFraction", 0.0)) < options.minimum_coverage_fraction:
            coverage_status = "LOW_COVERAGE"
        candidates[filter_name] = Path(result.output_path).resolve(strict=True)
        receipts[filter_name] = {
            **dict(verified),
            "coverageGate": {
                "status": coverage_status,
                "minimumCoverageFraction": float(options.minimum_coverage_fraction),
                "observedCoverageFraction": float(statistics.get("coverageFraction", 0.0)),
                "advisory": True,
            },
        }
    return candidates, {
        "mode": IntegrationMode.DRIZZLE.value,
        "options": options.serializable(),
        "sampling": dict(sampling_evidence),
        "filters": receipts,
        "localNormalizationEvidence": {"status": "NOT_APPLICABLE", "groups": {}},
    }


def _ordinary_candidates(
    pipeline_paths: Sequence[str],
    pipeline_root: Path,
    staging: Path,
    *,
    minimum_coverage_fraction: float = 0.90,
    maximum_null_fraction: float = 0.10,
) -> tuple[dict[str, Path], dict[str, Any]]:
    pipeline_receipt = json.loads((pipeline_root / "receipt.json").read_text(encoding="utf-8"))
    integration_groups = pipeline_receipt.get("statistics", {}).get("integrationGroups", {})
    candidates: dict[str, Path] = {}
    coverage: dict[str, Any] = {}
    for value in pipeline_paths:
        path = Path(value).resolve(strict=True)
        info = read_frame_info(path)
        filter_name = info.filter_name
        if filter_name in candidates:
            raise E2EError("FILTER_OUTPUT_DUPLICATE", f"multiple masters for filter {filter_name}")
        candidates[filter_name] = path
        group = integration_groups.get(filter_name, {})
        promoted_maps: dict[str, str] = {}
        token = _safe_token(filter_name)
        raw_maps = group.get("integration", {}).get("maps", {})
        required_maps = {
            "acceptedSampleCount",
            "coverageFraction",
            "rejectionCount",
        }
        if not isinstance(raw_maps, Mapping) or set(raw_maps) != required_maps:
            raise E2EError(
                "ORDINARY_INTEGRATION_MAPS_MISSING",
                f"ordinary integration lacks required maps for filter {filter_name}",
            )
        for map_name in sorted(required_maps):
            relative = raw_maps[map_name]
            if not isinstance(relative, str):
                raise E2EError(
                    "PIPELINE_RECEIPT_INVALID", "integration map path is not a string"
                )
            source = (pipeline_root / relative).resolve(strict=True)
            try:
                source.relative_to(pipeline_root.resolve(strict=True))
            except ValueError as error:
                raise E2EError(
                    "PIPELINE_RECEIPT_INVALID", "integration map escapes pipeline root"
                ) from error
            destination = staging / "coverage" / f"{token}_{map_name}.fits"
            try:
                with source.open("rb") as input_stream, destination.open("xb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream, length=4 * 1024 * 1024)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
            except FileExistsError as error:
                raise E2EError(
                    "OUTPUT_EXISTS", "refusing to replace promoted integration map",
                    path=str(destination),
                ) from error
            if sha256_digest(source) != sha256_digest(destination):
                raise E2EError(
                    "ARTIFACT_HASH_MISMATCH",
                    "promoted integration map differs from its source",
                    path=str(destination),
                )
            promoted_maps[map_name] = str(destination.relative_to(staging))
        coverage_path = staging / promoted_maps["coverageFraction"]
        with fits.open(coverage_path, mode="readonly", memmap=True, checksum=True) as hdul:
            coverage_data = np.asarray(hdul[0].data, dtype=np.float32)
            finite_coverage = np.isfinite(coverage_data)
            supported_fraction = float(
                np.count_nonzero(finite_coverage & (coverage_data > 0))
                / coverage_data.size
            )
        with fits.open(path, mode="readonly", memmap=True, checksum=True) as hdul:
            # Keep only the finite mask: an array of the memory map itself
            # would outlive the ``with`` block and, on Windows, keep the
            # master locked while the run later moves it.
            finite_master = np.isfinite(hdul[0].data)
        null_fraction = float(
            1.0 - np.count_nonzero(finite_master) / finite_master.size
        )
        if (
            supported_fraction < minimum_coverage_fraction
            or null_fraction > maximum_null_fraction
        ):
            raise E2EError(
                "ORDINARY_COVERAGE_GATE_FAILED",
                f"filter {filter_name} supported fraction {supported_fraction:.6g}, "
                f"null fraction {null_fraction:.6g}; require >= {minimum_coverage_fraction:.6g} "
                f"and <= {maximum_null_fraction:.6g}",
            )
        coverage[filter_name] = {
            "mode": IntegrationMode.ORDINARY.value,
            "cropTopLeftBottomRightExclusive": group.get("crop"),
            "masterStatistics": group.get("masterStatistics"),
            "mapStatistics": group.get("mapStatistics"),
            "maps": promoted_maps,
            "weights": group.get("integration", {}).get("weightComponents"),
            "execution": group.get("integration", {}).get("execution"),
            "sourceReceipt": "receipts/pixel-pipeline.json",
            "qualityGate": {
                "status": "PASS",
                "supportedFraction": supported_fraction,
                "minimumSupportedFraction": minimum_coverage_fraction,
                "nullFraction": null_fraction,
                "maximumNullFraction": maximum_null_fraction,
                "zeroAcceptedPixelsRemainNaN": True,
            },
        }
    return candidates, {"mode": IntegrationMode.ORDINARY.value, "filters": coverage}


def _promote_proper_coadds(
    *,
    proper_paths: Mapping[str, str],
    solved_products: Mapping[str, Path],
    products_dir: Path,
    staging: Path,
) -> tuple[list[Path], dict[str, Any]]:
    """Publish each filter's proper coadd beside its solved master.

    The coadd is produced from the same registered frames, on the same
    reference grid, and cropped to the same rectangle as the ordinary master,
    so the master's independently verified WCS describes it exactly.  The
    solution is copied rather than re-solved, and the product says so:
    ``OAFWCS = 'INHERITED'`` with ``OAFWCSIN`` naming the master it came from.
    The ordinary master remains the run's primary product.
    """

    promoted: list[Path] = []
    records: dict[str, Any] = {}
    for filter_name in sorted(proper_paths):
        solved = solved_products.get(filter_name)
        if solved is None:
            raise E2EError(
                "PROPER_COADD_UNSOLVED_GRID",
                f"filter {filter_name} has a proper coadd but no solved master to inherit from",
            )
        source = Path(proper_paths[filter_name]).resolve(strict=True)
        token = _safe_token(filter_name)
        destination = products_dir / token / f"{token}.proper.fits"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with source.open("rb") as input_stream, destination.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=4 * 1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
        except FileExistsError as error:
            raise E2EError(
                "OUTPUT_EXISTS", "refusing to replace a promoted proper coadd",
                path=str(destination),
            ) from error
        solved_header, solved_shape = _read_image_header(solved)
        _, coadd_shape = _read_image_header(destination)
        if coadd_shape != solved_shape:
            raise E2EError(
                "PROPER_COADD_GRID_MISMATCH",
                f"{filter_name} proper coadd is {coadd_shape}, its solved master {solved_shape}",
                path=str(destination),
            )
        adopted_cards = [
            card for card in solved_header.cards if _WCS_CARD_PATTERN.match(card.keyword)
        ]
        try:
            with fits.open(
                destination,
                mode="update",
                memmap=True,
                do_not_scale_image_data=True,
                uint=False,
                checksum=False,
            ) as hdul:
                header = hdul[0].header
                insert_at = header.index("EXTEND") + 1 if "EXTEND" in header else 5
                for offset, card in enumerate(adopted_cards):
                    header.insert(insert_at + offset, card)
                header["OAFSTATE"] = ("SOLVED", "Carries a verified same-grid solution")
                header["OAFWCS"] = ("INHERITED", "Copied from the same-grid solved master")
                header["OAFWCSIN"] = (solved.name, "Master whose verified WCS was copied")
                header.add_history(
                    "Ultra-Fast WBPP: proper coadd (ZOGY); WCS copied from the "
                    f"independently verified solve of {solved.name} on the identical grid"
                )
                for hdu in hdul:
                    if "CHECKSUM" in hdu.header or "DATASUM" in hdu.header:
                        hdu.add_checksum(override_datasum=True)
                hdul.flush(output_verify="exception")
            with destination.open("r+b") as stream:
                os.fsync(stream.fileno())
        except Exception as error:
            raise E2EError(
                "PROPER_COADD_WCS_COPY_FAILED", str(error), path=str(destination)
            ) from error
        after_header, after_shape = _read_image_header(destination)
        validation = validate_wcs_header(after_header, image_shape=after_shape)
        residual = _wcs_grid_disagreement(solved_header, after_header, after_shape)
        if not validation.valid or residual is None or residual > 1e-6:
            raise E2EError(
                "PROPER_COADD_WCS_INVALID",
                "the promoted proper coadd does not carry the master's solution exactly",
                path=str(destination),
            )
        promoted.append(destination)
        records[filter_name] = {
            "output": str(destination.relative_to(staging)),
            "wcsSource": str(solved.relative_to(staging)),
            "wcsProvenance": "INHERITED_SAME_GRID",
            "wcsGridDisagreementPixels": residual,
            "wcsValidation": validation.serializable(),
            "sha256": sha256_digest(destination),
            "primaryProduct": False,
        }
    return promoted, records


def _solve_candidates(
    candidates: Mapping[str, Path],
    *,
    products_dir: Path,
    staging: Path,
    backends: Sequence[SolverBackend],
    hints: _SolverHints,
    request: E2ERequest,
    progress: ProgressCallback | None,
) -> tuple[dict[str, Any], list[Path], dict[str, Path], bool]:
    """Solve every filter's master.

    Every filter solves in its own staging directory against read-only
    backends, so the filters run concurrently; records keep filter order.
    Returns the solver records, the staged solved products, the solved
    product of each filter and whether every filter solved.
    """

    ordered_candidates = sorted(candidates.items())
    solve_targets: dict[str, Path] = {}
    for filter_name, _candidate in ordered_candidates:
        token = _safe_token(filter_name)
        filter_dir = products_dir / token
        filter_dir.mkdir(parents=True, exist_ok=False)
        solve_targets[filter_name] = filter_dir / f"master_light_{token}_wcs.fits"
    progress_lock = threading.Lock()
    completed = 0

    def solve_filter(item: tuple[str, Path]) -> tuple[str, bool, list[dict[str, Any]]]:
        nonlocal completed
        filter_name, candidate = item
        solved, attempts = _solve_one(
            input_path=candidate,
            output_path=solve_targets[filter_name],
            backends=backends,
            hints=hints,
            min_matches=request.min_matches,
            max_rms_arcsec=request.max_rms_arcsec,
        )
        with progress_lock:
            completed += 1
            _emit(
                progress,
                ProgressStage.ASTROMETRY,
                "progress",
                f"{filter_name}: {'SOLVED' if solved else 'UNSOLVED'}",
                current=completed,
                total=len(candidates),
            )
        return filter_name, solved, attempts

    solve_workers = max(1, min(request.workers, len(ordered_candidates)))
    if solve_workers == 1:
        outcomes = [solve_filter(item) for item in ordered_candidates]
    else:
        with ThreadPoolExecutor(max_workers=solve_workers, thread_name_prefix="ufwbpp-solve") as pool:
            outcomes = list(pool.map(solve_filter, ordered_candidates))
    records: dict[str, Any] = {}
    staged_products: list[Path] = []
    solved_products: dict[str, Path] = {}
    all_solved = True
    for filter_name, solved, attempts in outcomes:
        solved_path = solve_targets[filter_name]
        records[filter_name] = {
            "status": "SOLVED" if solved else "UNSOLVED",
            "input": str(candidates[filter_name].relative_to(staging)),
            "output": str(solved_path.relative_to(staging)) if solved else None,
            "attempts": _relativize_solver_attempts(attempts, staging),
        }
        all_solved = all_solved and solved
        if solved:
            staged_products.append(solved_path)
            solved_products[filter_name] = solved_path
    return records, staged_products, solved_products, all_solved
