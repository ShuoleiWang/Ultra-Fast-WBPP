"""Portable CPU raw-to-linear-master pipeline.

The pipeline is deliberately not an astrometric solver or drizzle engine. Its
published masters are marked ``UNSOLVED_WORKING`` and contain no fabricated WCS.
All pixel operations use vertical slices and all input FITS files stay read-only.
"""


from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence, Callable

from .. import platform as platform_services
from ..calibration.inputs import TrustedGeneratedCalibrationSet, validate_trusted_generated_calibration_set
from ..errors import RuntimeConfigurationError
from ..hardware import detect_hardware
from ..image_io.xisf import convert_xisf_to_fits
from ..native_kernels import describe_native_kernels
from ..path_budget import STAGING_SUFFIX, check_output_path_budget
from ..performance_profile import select_execution_tuning
from ..platform import NoReplaceError, remove_tree
from .crop import _shared_auto_crop
from .groups import _group_concurrency, _GroupProducts, _integrate_group, _MetalSession, _NormalizationFits
from .integration import CalibrationError
from .lights import _calibrate_and_register_lights, _plan_light_jobs
from .masters import _build_calibration_masters
from .normalization import StellarScaleHint
from .parameters import OUTPUT_STATE, PIPELINE_VERSION, PipelineParameters, PipelineResult, PixelTransform
from .records import _canonical_inputs, _content_lineage_sha256, _verify_source_identities
from .run_plan import (
    _plan_run,
    _resolve_quality_weights,
    _resolve_transforms,
    _RunLedger,
    _RunPlan,
    _StagingDirs,
)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Create-only directory publication through the platform service layer."""

    try:
        platform_services.current().rename_directory_no_replace(source, destination)
    except NoReplaceError as error:
        if error.code == "OUTPUT_EXISTS":
            raise CalibrationError(
                "OUTPUT_EXISTS", "refusing to overwrite output directory", path=str(destination)
            ) from error
        raise CalibrationError(error.code, error.message) from error


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _trusted_reuse_receipt(
    plan: _RunPlan, trusted: TrustedGeneratedCalibrationSet | None
) -> dict[str, Any]:
    if trusted is None:
        return {"status": "NOT_USED"}
    raw_source_provenance = [
        dict(item) for item in plan.source_records if item["role"] in {"BIAS", "DARK", "FLAT"}
    ]
    return {
        "status": "REUSED_E2E_GENERATED_MASTER",
        "sourceContentManifestSha256": _content_lineage_sha256(raw_source_provenance),
        "privateExecutionManifestBound": True,
        "upstreamRegistrationCalibrationReceiptSha256": trusted.upstream_receipt_sha256,
        "calibrationApplied": False,
        "doubleBiasSubtraction": False,
        "originalRawSourceProvenance": raw_source_provenance,
        "masters": [
            {
                "role": item.role,
                "sha256": item.sha256,
                "sizeBytes": item.size_bytes,
                **({"biasIncluded": item.bias_included} if item.role == "MASTER_DARK" else {}),
                **({"applicationScale": item.application_scale} if item.role == "MASTER_FLAT" else {}),
            }
            for item in (trusted.master_bias, *trusted.master_darks, *trusted.master_flats)
            if item is not None
        ],
    }


def run_portable_pipeline_fits(
    *,
    bias_files: Iterable[str | os.PathLike[str]] = (),
    dark_files: Iterable[str | os.PathLike[str]] = (),
    flat_files: Iterable[str | os.PathLike[str]] = (),
    master_bias_file: str | os.PathLike[str] | None = None,
    master_dark_files: Iterable[str | os.PathLike[str]] = (),
    master_flat_files: Iterable[str | os.PathLike[str]] = (),
    light_files: Iterable[str | os.PathLike[str]],
    output_directory: str | os.PathLike[str],
    transforms: Mapping[str, PixelTransform | Sequence[Sequence[float]]] | None = None,
    quality_weights: Mapping[str, float] | None = None,
    stellar_scale_hints: Mapping[str, StellarScaleHint] | None = None,
    parameters: PipelineParameters | None = None,
    _source_aliases: Mapping[str, Path] | None = None,
    _xisf_conversions: Sequence[Mapping[str, Any]] = (),
    _trusted_generated_calibration: TrustedGeneratedCalibrationSet | None = None,
    _source_identity_seed: Mapping[str, tuple[str, Mapping[str, int]]] | None = None,
    _integration_tile_observers: Callable[[str, Sequence[str]], Any] | None = None,
    region_weight_maps: Mapping[str, Any] | None = None,
    _staging_stem: str | None = None,
) -> PipelineResult:
    """Run raw or pre-integrated calibration through unsolved linear masters.

    The stages: plan (validate and group every input), calibration masters,
    per-Light calibrate-and-register jobs, a common crop, then per output
    group normalization, rejection/integration, crop, maps and preview; the
    receipt and one no-replace publication of the staging directory close
    the run.

    ``_staging_stem`` names the transient staging directory beside the output
    (``.<stem>.<8>.stage``); the E2E run passes a short stem because that
    directory is the deepest level of its layout (see ``path_budget``).

    ``_integration_tile_observers(filter_name, ordered_light_paths)`` may
    return a tile observer for that group's ordinary integration (selection
    counterfactual); observers never change the products.

    ``region_weight_maps`` binds Lights (by path) to their selection region
    weight maps (``RegionWeightMap``: node values in normalized reference
    coordinates); a bound Light's samples are weighted by the map during the
    ordinary weighted mean.  Lights without a map keep unit sample weights.

    ``_source_identity_seed`` maps canonical original paths to a content
    digest and the stat identity that digest was captured with.  Seeded
    sources are not rehashed; a stat mismatch against the seed still fails
    closed as ``SOURCE_CHANGED``.

    A calibration profile may use raw frames or a supplied master, never both
    for the same bias/filter/exposure identity.  Supplied masters are opened
    read-only and reused directly; they are never integrated or calibrated a
    second time.
    """

    parameters = parameters or PipelineParameters()
    parameters.validate()
    plan = _plan_run(
        bias_files=bias_files,
        dark_files=dark_files,
        flat_files=flat_files,
        master_bias_file=master_bias_file,
        master_dark_files=master_dark_files,
        master_flat_files=master_flat_files,
        light_files=light_files,
        output_directory=output_directory,
        transforms=transforms,
        quality_weights=quality_weights,
        stellar_scale_hints=stellar_scale_hints,
        region_weight_maps=region_weight_maps,
        parameters=parameters,
        source_aliases=_source_aliases,
        trusted_generated_calibration=_trusted_generated_calibration,
        source_identity_seed=_source_identity_seed,
    )
    output = plan.output
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{_staging_stem or output.name}.", suffix=STAGING_SUFFIX, dir=output.parent
        )
    )
    published = False
    metal = _MetalSession()
    fits: _NormalizationFits | None = None
    try:
        dirs = _StagingDirs.create(staging)
        ledger = _RunLedger(staging)
        masters = _build_calibration_masters(plan, dirs, parameters, ledger)
        hardware_profile = detect_hardware()
        execution_tuning = select_execution_tuning(hardware_profile)
        light_jobs, calibrated_details = _plan_light_jobs(plan, dirs, parameters, masters)
        lights = _calibrate_and_register_lights(
            plan, dirs, light_jobs, calibrated_details, parameters, execution_tuning, ledger
        )
        if parameters.ordinary_integration_backend != "portable-cpu":
            metal.open(parameters)

        # Every group of this run was registered onto the same reference grid.
        # Cropping each master to the rectangle that is valid in all groups
        # keeps the masters of different filters on one identical pixel grid,
        # as WBPP's autocrop does, so LRGB composition never resamples them.
        # A single-filter run keeps exactly its own crop.
        crop_started = time.perf_counter()
        shared_crop, group_crops = _shared_auto_crop(
            plan.light_groups,
            plan.light_info,
            plan.transforms,
            enabled=parameters.auto_crop,
            max_memory_bytes=parameters.registration_memory_bytes,
            resampler=parameters.registration_resampler,
        )
        stage_timing: dict[str, Any] = {
            "fusedCalibrateWarp": round(lights.wall_seconds, 3),
            "autoCrop": round(time.perf_counter() - crop_started, 3),
            "groups": {},
        }
        ordered_groups = sorted(plan.output_groups.items())
        concurrency = _group_concurrency(parameters, execution_tuning, len(ordered_groups))
        fits = _NormalizationFits(
            plan, lights.by_group, parameters, execution_tuning, ordered_groups,
            prefetch=concurrency == 1,
        )
        # Observers are created here, in group order, so a caller that keeps
        # them sees the groups in that order however the groups are scheduled.
        observers = {
            group_name: (
                _integration_tile_observers(group_name, [str(path) for path in paths])
                if _integration_tile_observers is not None
                else None
            )
            for group_name, paths in ordered_groups
        }
        # Each group records its artifacts in its own ledger; the run ledger
        # takes them in group order, so the receipt does not depend on which
        # group finished first.
        group_ledgers = {group_name: _RunLedger(staging) for group_name, _ in ordered_groups}

        def integrate(position: int) -> _GroupProducts:
            group_name, paths = ordered_groups[position]
            if concurrency == 1:
                fits.prefetch_after(position)
            return _integrate_group(
                plan,
                dirs,
                group_name,
                paths,
                parameters=parameters,
                lights=lights,
                fits=fits,
                metal=metal,
                hardware_profile=hardware_profile,
                execution_tuning=execution_tuning,
                shared_crop=shared_crop,
                group_crops=group_crops,
                tile_observer=observers[group_name],
                ledger=group_ledgers[group_name],
            )

        if concurrency == 1:
            results = [integrate(position) for position in range(len(ordered_groups))]
        else:
            # Largest groups start first so the last group to finish is a
            # small one; every group's products are independent of the order.
            schedule = sorted(
                range(len(ordered_groups)),
                key=lambda position: (-len(ordered_groups[position][1]), position),
            )
            fits.prefetch_all(schedule, concurrent=concurrency)
            with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="ufwbpp-group") as pool:
                futures = {position: pool.submit(integrate, position) for position in schedule}
            results = [futures[position].result() for position in range(len(ordered_groups))]
        products: dict[str, _GroupProducts] = {}
        for (group_name, _paths), product in zip(ordered_groups, results, strict=True):
            products[group_name] = product
            ledger.artifacts.extend(group_ledgers[group_name].artifacts)
            stage_timing["groups"][group_name] = {
                key: round(value, 3) for key, value in product.timing.items()
            }
        fits.shutdown()
        remove_tree(dirs.work)
        _verify_source_identities(plan.source_identities)
        if _trusted_generated_calibration is not None:
            # Recheck source stat identities and rehash the generated masters
            # and upstream receipt after all consumers finish.  The enclosing
            # E2E publication gate deliberately performs the second full hash
            # of every original source; intermediate receipt lookups reuse this
            # run's path/stat-bound digest instead of rereading large inputs.
            validate_trusted_generated_calibration_set(
                _trusted_generated_calibration,
                source_groups=plan.source_groups,
                source_aliases=plan.source_aliases,
                identity_cache=plan.source_identity_cache,
            )
        receipt = {
            "schemaVersion": 1,
            "pipelineVersion": PIPELINE_VERSION,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "state": OUTPUT_STATE,
            "parameters": parameters.serializable(),
            "inputs": plan.source_records,
            "pixelNumericDomains": plan.pixel_numeric_domains,
            "trustedGeneratedCalibration": _trusted_reuse_receipt(plan, _trusted_generated_calibration),
            "pixelInputStaging": {
                "xisfPolicy": parameters.xisf_decode.serializable(),
                "conversions": [dict(item) for item in _xisf_conversions],
                "privateStagingRemovedAfterRun": True,
            },
            "masterMetadataOverrides": [
                {**item.serializable(), "status": "APPLIED_CONTENT_BOUND_DECLARATION"}
                for item in parameters.master_metadata_overrides
            ],
            "outputs": ledger.artifacts,
            "registration": lights.records,
            "statistics": {
                "calibration": ledger.stage_statistics,
                "registration": lights.execution,
                "integrationGroups": {name: product.record for name, product in products.items()},
                "integrationGroupConcurrency": concurrency,
                "timingSeconds": stage_timing,
            },
            # Platform facts behind the execution choices: what the machine
            # is, which tuning table row ran, and which native library (if
            # any) produced the kernel results.  None of them changes pixels.
            "platform": {
                "hardware": hardware_profile.serializable(),
                "tuning": execution_tuning.serializable(),
                "nativeKernels": describe_native_kernels(),
            },
            "astrometry": {
                "status": "UNSOLVED",
                "wcsValidated": False,
                "message": "No solver was invoked; every linear master remains UNSOLVED_WORKING.",
            },
            "drizzle": {
                "status": "NOT_RUN",
                "message": "This portable ordinary-integration pipeline does not run drizzle.",
            },
        }
        _write_receipt(staging / "receipt.json", receipt)
        _rename_directory_no_replace(staging, output)
        published = True
        with suppress(OSError):
            platform_services.current().fsync_directory(output.parent)

        def published_path(path: Path | str) -> str:
            return str(output / Path(path).relative_to(staging))

        return PipelineResult(
            output_directory=str(output),
            receipt_path=str(output / "receipt.json"),
            state=OUTPUT_STATE,
            master_light_paths=tuple(published_path(product.master_light) for product in products.values()),
            preview_paths=tuple(published_path(product.preview) for product in products.values()),
            drizzle_groups={
                name: replace(
                    product.drizzle,
                    frames=tuple(
                        replace(frame, calibrated_path=published_path(frame.calibrated_path))
                        for frame in product.drizzle.frames
                    ),
                )
                for name, product in products.items()
                if product.drizzle is not None
            },
            proper_coadd_paths={
                name: published_path(product.proper_coadd)
                for name, product in products.items()
                if product.proper_coadd is not None
            },
        )
    finally:
        if fits is not None:
            fits.shutdown(cancel=True)
        metal.close()
        if not published:
            remove_tree(staging)


def _rekey_for_staged_lights(
    originals: tuple[Path, ...],
    staged_by_original: Mapping[Path, Path],
    transforms: Mapping[str, PixelTransform | Sequence[Sequence[float]]] | None,
    quality_weights: Mapping[str, float] | None,
    stellar_scale_hints: Mapping[str, StellarScaleHint] | None,
) -> tuple[
    Mapping[str, PixelTransform | Sequence[Sequence[float]]] | None,
    Mapping[str, float] | None,
    Mapping[str, StellarScaleHint] | None,
]:
    resolved_transforms = _resolve_transforms(originals, transforms)
    resolved_weights = _resolve_quality_weights(originals, quality_weights)
    staged_hints: Mapping[str, StellarScaleHint] | None = None
    if stellar_scale_hints is not None:
        by_path = {
            Path(key).expanduser().resolve(strict=True): value
            for key, value in stellar_scale_hints.items()
        }
        staged_values: dict[str, StellarScaleHint] = {}
        for original in originals:
            hint = by_path.get(original)
            if hint is None:
                continue
            reference = Path(hint.reference_path).expanduser().resolve(strict=True)
            if reference not in staged_by_original:
                raise CalibrationError(
                    "STELLAR_SCALE_HINT_INPUT_MISMATCH",
                    "stellar scale reference is not an admitted Light",
                    path=str(reference),
                )
            staged_source = staged_by_original[original]
            staged_values[str(staged_source)] = replace(
                hint,
                source_path=str(staged_source),
                reference_path=str(staged_by_original[reference]),
            )
        staged_hints = staged_values
    return (
        {
            str(staged_by_original[path]): transform
            for path, transform in resolved_transforms.items()
        },
        {
            str(staged_by_original[path]): weight
            for path, weight in resolved_weights.items()
        },
        staged_hints,
    )


def run_portable_pipeline(
    *,
    bias_files: Iterable[str | os.PathLike[str]] = (),
    dark_files: Iterable[str | os.PathLike[str]] = (),
    flat_files: Iterable[str | os.PathLike[str]] = (),
    master_bias_file: str | os.PathLike[str] | None = None,
    master_dark_files: Iterable[str | os.PathLike[str]] = (),
    master_flat_files: Iterable[str | os.PathLike[str]] = (),
    light_files: Iterable[str | os.PathLike[str]],
    output_directory: str | os.PathLike[str],
    transforms: Mapping[str, PixelTransform | Sequence[Sequence[float]]] | None = None,
    quality_weights: Mapping[str, float] | None = None,
    stellar_scale_hints: Mapping[str, StellarScaleHint] | None = None,
    parameters: PipelineParameters | None = None,
) -> PipelineResult:
    """Run the portable pipeline with private, content-bound XISF staging."""

    parameters = parameters or PipelineParameters()
    parameters.validate()
    output = Path(output_directory).expanduser().resolve(strict=False)
    if output.exists() or os.path.lexists(output):
        raise CalibrationError("OUTPUT_EXISTS", "output directory must be new", path=str(output))
    light_paths = tuple(light_files)
    try:
        check_output_path_budget(
            output, light_count=len(light_paths), light_paths=light_paths, layout="pixels"
        )
    except RuntimeConfigurationError as error:
        raise CalibrationError(error.code, str(error), path=str(output)) from error
    output.parent.mkdir(parents=True, exist_ok=True)
    originals = {
        "BIAS": _canonical_inputs(bias_files, "Bias", required=False),
        "DARK": _canonical_inputs(dark_files, "Dark", required=False),
        "FLAT": _canonical_inputs(flat_files, "Flat", required=False),
        "MASTER_BIAS": _canonical_inputs(
            () if master_bias_file is None else (master_bias_file,),
            "MasterBias",
            required=False,
        ),
        "MASTER_DARK": _canonical_inputs(master_dark_files, "MasterDark", required=False),
        "MASTER_FLAT": _canonical_inputs(master_flat_files, "MasterFlat", required=False),
        "LIGHT": _canonical_inputs(light_paths, "Light", required=True),
    }
    all_originals = [path for paths in originals.values() for path in paths]
    if len({os.path.normcase(str(path)) for path in all_originals}) != len(all_originals):
        raise CalibrationError("INPUT_ROLE_OVERLAP", "one source appears in multiple roles")
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.xisf-pixels-",
        dir=output.parent,
    ) as private_name:
        private = Path(private_name)
        staged_by_original: dict[Path, Path] = {}
        aliases: dict[str, Path] = {}
        conversions: list[dict[str, Any]] = []
        sequence = 0
        for role, paths in originals.items():
            for original in paths:
                sequence += 1
                if original.suffix.casefold() == ".xisf":
                    staged = private / f"{sequence:06d}_{role.casefold()}.fits"
                    receipt = convert_xisf_to_fits(
                        original,
                        staged,
                        policy=parameters.xisf_decode,
                    )
                    conversions.append({"role": role, **receipt.serializable()})
                else:
                    staged = original
                staged_by_original[original] = staged
                aliases[str(staged)] = original
        staged = {
            role: tuple(staged_by_original[path] for path in paths)
            for role, paths in originals.items()
        }
        staged_transforms, staged_weights, staged_stellar_scale_hints = (
            _rekey_for_staged_lights(
                originals["LIGHT"],
                staged_by_original,
                transforms,
                quality_weights,
                stellar_scale_hints,
            )
        )
        return run_portable_pipeline_fits(
            bias_files=staged["BIAS"],
            dark_files=staged["DARK"],
            flat_files=staged["FLAT"],
            master_bias_file=(staged["MASTER_BIAS"][0] if staged["MASTER_BIAS"] else None),
            master_dark_files=staged["MASTER_DARK"],
            master_flat_files=staged["MASTER_FLAT"],
            light_files=staged["LIGHT"],
            output_directory=output,
            transforms=staged_transforms,
            quality_weights=staged_weights,
            stellar_scale_hints=staged_stellar_scale_hints,
            parameters=parameters,
            _source_aliases=aliases,
            _xisf_conversions=conversions,
        )


__all__ = [
    "PixelTransform",
    "AffineTransform",
    "MasterMetadataOverride",
    "OUTPUT_STATE",
    "PIPELINE_VERSION",
    "PipelineParameters",
    "PipelineResult",
    "run_portable_pipeline",
]
