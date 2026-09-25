"""Planning one pixel run: frame metadata, calibration and output grouping, the per-frame hints, and the run's staging directories and ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from lightframeqc.cfa import CFA_PATTERNS, CHANNEL_NAMES, normalize_pattern as normalize_cfa_pattern

from ..calibration.inputs import (
    TrustedGeneratedMaster,
    TrustedGeneratedCalibrationSet,
    apply_master_metadata_overrides,
    apply_raw_frame_metadata_overrides,
    trust_private_xisf_numeric_domains,
    master_dark_bias_semantics,
    assert_compatible,
    validate_trusted_generated_calibration_set,
    source_identity,
    find_dark,
    SourceIdentityCache,
)
from ..calibration.policy import (
    MONO_STANDARD,
    apply_mono_workflow,
    cfa_for_workflow,
    can_omit_bias,
    conflicting_profile_fields,
)
from .integration import CalibrationError, FrameInfo, PixelStatistics
from .normalization import StellarScaleHint
from .parameters import PipelineParameters, PixelTransform
from .records import (
    _artifact_record,
    _canonical_inputs,
    _path_receipt_reference,
    _pixel_numeric_domain_records,
    _read_infos,
    _safe_token,
    _source_records,
    require_filter,
)


def _resolve_transforms(
    light_paths: tuple[Path, ...],
    transforms: Mapping[str, PixelTransform | Sequence[Sequence[float]]] | None,
) -> dict[Path, PixelTransform]:
    if transforms is None:
        return {path: PixelTransform.identity() for path in light_paths}
    if not all(isinstance(key, str) and key for key in transforms):
        raise CalibrationError("TRANSFORM_KEY_INVALID", "transform keys must be strings")
    basename_counts: dict[str, int] = {}
    for path in light_paths:
        basename_counts[path.name] = basename_counts.get(path.name, 0) + 1
    used: set[str] = set()
    result: dict[Path, PixelTransform] = {}
    # Callers may key transforms by any spelling of a Light path; resolve
    # every key once so symlinked temporary roots and relative paths match
    # the canonical Light list exactly as quality weights already do.
    canonical_keys: dict[str, list[str]] = {}
    for key in transforms:
        try:
            canonical = str(Path(key).expanduser().resolve(strict=True))
        except OSError:
            continue
        canonical_keys.setdefault(os.path.normcase(canonical), []).append(key)
    for path in light_paths:
        candidates = [str(path), os.path.normcase(str(path))]
        if basename_counts[path.name] == 1:
            candidates.append(path.name)
        candidates.extend(canonical_keys.get(os.path.normcase(str(path)), ()))
        matching = [key for key in candidates if key in transforms]
        matching = list(dict.fromkeys(matching))
        if len(matching) > 1:
            values = [PixelTransform.from_value(transforms[key]) for key in matching]
            matrices = [value.validated_matrix() for value in values]
            if not all(np.array_equal(matrices[0], matrix) for matrix in matrices[1:]):
                raise CalibrationError(
                    "TRANSFORM_KEY_CONFLICT", "multiple transform keys disagree", path=str(path)
                )
        if matching:
            key = matching[0]
            used.update(matching)
            result[path] = PixelTransform.from_value(transforms[key])
        else:
            result[path] = PixelTransform.identity()
    unused = sorted(set(transforms) - used)
    if unused:
        raise CalibrationError(
            "TRANSFORM_INPUT_UNKNOWN",
            f"transform keys do not match any Light: {', '.join(unused)}",
        )
    return result


def _resolve_quality_weights(
    light_paths: tuple[Path, ...],
    weights: Mapping[str, float] | None,
) -> dict[Path, float]:
    if weights is None:
        return {path: 1.0 for path in light_paths}
    if not all(isinstance(key, str) and key for key in weights):
        raise CalibrationError(
            "QUALITY_WEIGHT_KEY_INVALID", "quality weight keys must be non-empty strings"
        )
    canonical = {os.path.normcase(str(path)): path for path in light_paths}
    result: dict[Path, float] = {}
    for key, raw_value in weights.items():
        try:
            path = Path(key).expanduser().resolve(strict=True)
            value = float(raw_value)
        except (OSError, TypeError, ValueError) as error:
            raise CalibrationError(
                "QUALITY_WEIGHT_INVALID",
                "quality weights must bind existing Lights to finite positive numbers",
                path=key,
            ) from error
        canonical_key = os.path.normcase(str(path))
        if canonical_key not in canonical:
            raise CalibrationError(
                "QUALITY_WEIGHT_INPUT_UNKNOWN",
                "quality weight does not bind an integration Light",
                path=str(path),
            )
        if not math.isfinite(value) or value <= 0:
            raise CalibrationError(
                "QUALITY_WEIGHT_INVALID",
                "quality weights must be finite and positive",
                path=str(path),
            )
        bound = canonical[canonical_key]
        if bound in result:
            raise CalibrationError(
                "QUALITY_WEIGHT_KEY_CONFLICT",
                "multiple quality weight keys bind the same Light",
                path=str(path),
            )
        result[bound] = value
    missing = [str(path) for path in light_paths if path not in result]
    if missing:
        raise CalibrationError(
            "QUALITY_WEIGHT_SET_INCOMPLETE",
            "quality weights must bind every admitted Light: " + ", ".join(missing),
        )
    return result


def _resolve_region_weight_maps(
    light_paths: tuple[Path, ...],
    maps: Mapping[str, Any] | None,
) -> dict[Path, Any]:
    """Bind optional region weight maps to admitted Lights (unknown keys fail)."""

    if not maps:
        return {}
    canonical = {os.path.normcase(str(path)): path for path in light_paths}
    result: dict[Path, Any] = {}
    for key, region_map in maps.items():
        if not isinstance(key, str) or not key:
            raise CalibrationError(
                "REGION_WEIGHT_KEY_INVALID", "region weight map keys must be non-empty strings"
            )
        try:
            path = Path(key).expanduser().resolve(strict=True)
        except OSError as error:
            raise CalibrationError(
                "REGION_WEIGHT_INPUT_UNKNOWN",
                "region weight map does not bind an existing Light",
                path=key,
            ) from error
        bound = canonical.get(os.path.normcase(str(path)))
        if bound is None:
            raise CalibrationError(
                "REGION_WEIGHT_INPUT_UNKNOWN",
                "region weight map does not bind an integration Light",
                path=str(path),
            )
        if not hasattr(region_map, "nodes") or not hasattr(region_map, "pixel_nodes"):
            raise CalibrationError(
                "REGION_WEIGHT_MAP_INVALID",
                "region weight maps must provide nodes and pixel_nodes()",
                path=str(path),
            )
        if bound in result:
            raise CalibrationError(
                "REGION_WEIGHT_KEY_CONFLICT",
                "multiple region weight map keys bind the same Light",
                path=str(path),
            )
        result[bound] = region_map
    return result


def _resolve_stellar_scale_hints(
    light_paths: tuple[Path, ...],
    light_info: Mapping[Path, FrameInfo],
    hints: Mapping[str, StellarScaleHint] | None,
    source_aliases: Mapping[str, Path],
    identity_cache: SourceIdentityCache,
) -> dict[Path, StellarScaleHint | None]:
    if hints is None:
        return {path: None for path in light_paths}
    if not all(isinstance(key, str) and key for key in hints):
        raise CalibrationError(
            "STELLAR_SCALE_HINT_KEY_INVALID",
            "stellar scale hint keys must be non-empty paths",
        )
    canonical = {os.path.normcase(str(path)): path for path in light_paths}
    result: dict[Path, StellarScaleHint | None] = {}
    for key, hint in hints.items():
        if not isinstance(hint, StellarScaleHint):
            raise CalibrationError(
                "STELLAR_SCALE_HINT_INVALID",
                "stellar scale hints must use the typed identity-bound contract",
                path=key,
            )
        try:
            source = Path(key).expanduser().resolve(strict=True)
            hinted_source = Path(hint.source_path).expanduser().resolve(strict=True)
            reference = Path(hint.reference_path).expanduser().resolve(strict=True)
        except OSError as error:
            raise CalibrationError(
                "STELLAR_SCALE_HINT_INVALID",
                "stellar scale hint paths must identify current Light inputs",
                path=key,
            ) from error
        source_key = os.path.normcase(str(source))
        reference_key = os.path.normcase(str(reference))
        if (
            source != hinted_source
            or source_key not in canonical
            or reference_key not in canonical
        ):
            raise CalibrationError(
                "STELLAR_SCALE_HINT_INPUT_MISMATCH",
                "stellar scale hint source/reference is outside the admitted Light set",
                path=key,
            )
        bound_source = canonical[source_key]
        bound_reference = canonical[reference_key]
        if bound_source in result:
            raise CalibrationError(
                "STELLAR_SCALE_HINT_KEY_CONFLICT",
                "multiple hints bind the same Light",
                path=key,
            )
        _, source_sha256, _ = source_identity(
            bound_source, source_aliases, identity_cache
        )
        _, reference_sha256, _ = source_identity(
            bound_reference, source_aliases, identity_cache
        )
        if (
            hint.source_sha256 != source_sha256
            or hint.reference_sha256 != reference_sha256
        ):
            raise CalibrationError(
                "STELLAR_SCALE_HINT_IDENTITY_MISMATCH",
                "stellar scale hint is not bound to the current source/reference bytes",
                path=key,
            )
        source_filter = light_info[bound_source].filter_name
        reference_filter = light_info[bound_reference].filter_name
        if (
            hint.filter_name != source_filter
            or source_filter != reference_filter
        ):
            raise CalibrationError(
                "STELLAR_SCALE_HINT_FILTER_MISMATCH",
                "stellar scale hints cannot cross optical filters",
                path=key,
            )
        if hint.status not in {
            "REFERENCE_IDENTITY",
            "STELLAR_SCALE_ACCEPTED",
            "STELLAR_SCALE_UNAVAILABLE",
        }:
            raise CalibrationError(
                "STELLAR_SCALE_HINT_STATUS_INVALID",
                "stellar scale hint has an unknown status",
                path=key,
            )
        if hint.status == "REFERENCE_IDENTITY":
            if bound_source != bound_reference or hint.scale != 1.0:
                raise CalibrationError(
                    "STELLAR_SCALE_HINT_REFERENCE_INVALID",
                    "reference hint must bind itself with scale 1",
                    path=key,
                )
        elif hint.status == "STELLAR_SCALE_ACCEPTED":
            if (
                hint.scale is None
                or not math.isfinite(hint.scale)
                or hint.scale <= 0
            ):
                raise CalibrationError(
                    "STELLAR_SCALE_HINT_INVALID",
                    "accepted stellar scale must be finite and positive",
                    path=key,
                )
        elif hint.scale is not None:
            raise CalibrationError(
                "STELLAR_SCALE_HINT_INVALID",
                "unavailable stellar scale must not carry a numeric scale",
                path=key,
            )
        result[bound_source] = hint
    missing = [str(path) for path in light_paths if path not in result]
    if missing:
        raise CalibrationError(
            "STELLAR_SCALE_HINT_SET_INCOMPLETE",
            "stellar scale hints must bind every admitted Light: "
            + ", ".join(missing),
        )
    return result


@dataclass(frozen=True)
class _RunPlan:
    """What a run decides before it writes a pixel.

    Canonical inputs, their frame metadata after overrides, the calibration
    and output groups, each Light's transform/weight/hint bindings and the
    provenance records the receipt reports.  Building it validates every
    input combination, so the stages after it only execute.
    """

    output: Path
    lights: tuple[Path, ...]
    biases: tuple[Path, ...]
    master_biases: tuple[Path, ...]
    source_aliases: dict[str, Path]
    source_groups: tuple[tuple[str, tuple[Path, ...]], ...]
    source_identity_cache: SourceIdentityCache
    trusted_generated: dict[str, Any] | None
    bias_info: dict[Path, FrameInfo]
    dark_info: dict[Path, FrameInfo]
    flat_info: dict[Path, FrameInfo]
    master_dark_info: dict[Path, FrameInfo]
    master_flat_info: dict[Path, FrameInfo]
    light_info: dict[Path, FrameInfo]
    supplied_dark_bias_included: dict[Path, bool]
    reference_bias: FrameInfo
    flat_groups: dict[str, list[Path]]
    supplied_flats: dict[str, Path]
    light_groups: dict[str, list[Path]]
    reference_exposures: dict[str, float]
    light_domain_references: dict[str, FrameInfo]
    # Output (integration) groups: a mono filter is its own group; a Bayer
    # filter becomes the colour channel groups R, G and B.
    output_groups: dict[str, list[Path]]
    group_filter: dict[str, str]
    group_channel: dict[str, int | None]
    group_cfa_pattern: dict[str, str | None]
    light_cfa_pattern: dict[str, str | None]
    dark_groups: dict[float, list[Path]]
    supplied_darks: dict[float, Path]
    trusted_bias: TrustedGeneratedMaster | None
    trusted_darks_by_exposure: dict[float, TrustedGeneratedMaster]
    trusted_flats_by_filter: dict[str, TrustedGeneratedMaster]
    transforms: dict[Path, PixelTransform]
    quality_weights: dict[Path, float]
    region_weight_maps: dict[Path, Any]
    stellar_scale_hints: dict[Path, StellarScaleHint | None]
    source_records: list[dict[str, Any]]
    source_identities: dict[str, dict[str, int]]
    pixel_numeric_domains: list[dict[str, Any]]

    def display_path(self, path: Path) -> Path:
        return self.source_aliases.get(str(path), path)

    @property
    def trusted_by_path(self) -> Mapping[str, TrustedGeneratedMaster] | None:
        return self.trusted_generated["byPath"] if self.trusted_generated is not None else None

    def receipt_reference(self, staging: Path, path: Path) -> dict[str, Any]:
        return _path_receipt_reference(
            staging, path, self.source_aliases, self.trusted_by_path, self.source_identity_cache
        )

    def dark_reference(self, exposure: float) -> FrameInfo:
        """The frame metadata that stands for the master dark of ``exposure``."""

        if exposure in self.dark_groups:
            return self.dark_info[self.dark_groups[exposure][0]]
        return self.master_dark_info[self.supplied_darks[exposure]]

    def dark_bias_included(self, exposure: float, subtract_path: Path) -> bool:
        if exposure in self.trusted_darks_by_exposure:
            return bool(self.trusted_darks_by_exposure[exposure].bias_included)
        if exposure in self.dark_groups:
            return True
        return self.supplied_dark_bias_included[subtract_path]


def _plan_run(
    *,
    bias_files: Iterable[str | os.PathLike[str]],
    dark_files: Iterable[str | os.PathLike[str]],
    flat_files: Iterable[str | os.PathLike[str]],
    master_bias_file: str | os.PathLike[str] | None,
    master_dark_files: Iterable[str | os.PathLike[str]],
    master_flat_files: Iterable[str | os.PathLike[str]],
    light_files: Iterable[str | os.PathLike[str]],
    output_directory: str | os.PathLike[str],
    transforms: Mapping[str, PixelTransform | Sequence[Sequence[float]]] | None,
    quality_weights: Mapping[str, float] | None,
    stellar_scale_hints: Mapping[str, StellarScaleHint] | None,
    region_weight_maps: Mapping[str, Any] | None,
    parameters: PipelineParameters,
    source_aliases: Mapping[str, Path] | None,
    trusted_generated_calibration: TrustedGeneratedCalibrationSet | None,
    source_identity_seed: Mapping[str, tuple[str, Mapping[str, int]]] | None,
) -> _RunPlan:
    workflow = parameters.calibration_workflow
    biases = _canonical_inputs(bias_files, "Bias", required=False)
    darks = _canonical_inputs(dark_files, "Dark", required=False)
    flats = _canonical_inputs(flat_files, "Flat", required=False)
    master_biases = _canonical_inputs(
        (() if master_bias_file is None else (master_bias_file,)),
        "MasterBias",
        required=False,
    )
    master_darks_input = _canonical_inputs(master_dark_files, "MasterDark", required=False)
    master_flats_input = _canonical_inputs(master_flat_files, "MasterFlat", required=False)
    lights = _canonical_inputs(light_files, "Light", required=True)
    if (biases and master_biases) or (not biases and not master_biases and workflow != MONO_STANDARD):
        raise CalibrationError(
            "BIAS_SOURCE_AMBIGUOUS",
            "supply exactly one Bias source mode: raw Bias frames or one MasterBias",
        )
    all_paths = (
        *biases,
        *darks,
        *flats,
        *master_biases,
        *master_darks_input,
        *master_flats_input,
        *lights,
    )
    if len({os.path.normcase(str(path)) for path in all_paths}) != len(all_paths):
        raise CalibrationError("INPUT_ROLE_OVERLAP", "one source appears in multiple roles")

    aliases = dict(source_aliases or {})
    source_groups = (
        ("BIAS", biases),
        ("DARK", darks),
        ("FLAT", flats),
        ("MASTER_BIAS", master_biases),
        ("MASTER_DARK", master_darks_input),
        ("MASTER_FLAT", master_flats_input),
        ("LIGHT", lights),
    )
    identity_cache: SourceIdentityCache = {}
    for seed_path, (seed_digest, seed_identity) in (source_identity_seed or {}).items():
        seed_key = os.path.normcase(str(Path(seed_path).expanduser().resolve(strict=True)))
        identity_cache[seed_key] = (str(seed_digest), dict(seed_identity))
    trusted_generated: dict[str, Any] | None = None
    if trusted_generated_calibration is not None:
        trusted_generated = validate_trusted_generated_calibration_set(
            trusted_generated_calibration,
            source_groups=source_groups,
            source_aliases=aliases,
            identity_cache=identity_cache,
        )
        original_keys = {os.path.normcase(str(path)) for path in all_paths}
        if original_keys.intersection(trusted_generated["byPath"]):
            raise CalibrationError(
                "TRUSTED_GENERATED_MASTER_INPUT_OVERLAP",
                "E2E-generated masters cannot be presented as public input files",
            )

    output = Path(output_directory).expanduser().resolve(strict=False)
    if output.exists() or os.path.lexists(output):
        raise CalibrationError("OUTPUT_EXISTS", "output directory must be new", path=str(output))
    output.parent.mkdir(parents=True, exist_ok=True)

    (
        bias_info,
        dark_info,
        flat_info,
        master_bias_info,
        master_dark_info,
        master_flat_info,
        light_info,
    ) = trust_private_xisf_numeric_domains(
        (
            _read_infos(biases, "BIAS"),
            _read_infos(darks, "DARK"),
            _read_infos(flats, "FLAT"),
            _read_infos(master_biases, "MASTER_BIAS"),
            _read_infos(master_darks_input, "MASTER_DARK"),
            _read_infos(master_flats_input, "MASTER_FLAT"),
            _read_infos(lights, "LIGHT"),
        ),
        aliases,
    )
    bias_info, dark_info, flat_info, light_info = apply_raw_frame_metadata_overrides(
        (bias_info, dark_info, flat_info, light_info),
        parameters.raw_frame_metadata_overrides,
        dict(aliases),
        identity_cache,
    )
    master_bias_info, master_dark_info, master_flat_info = apply_master_metadata_overrides(
        (master_bias_info, master_dark_info, master_flat_info),
        parameters.master_metadata_overrides,
        dict(aliases),
        identity_cache,
    )
    supplied_dark_bias_included = master_dark_bias_semantics(
        master_darks_input,
        parameters.master_metadata_overrides,
        dict(aliases),
        identity_cache,
        workflow=workflow,
    )
    for group in (bias_info, dark_info, flat_info, master_bias_info, master_dark_info, master_flat_info, light_info):
        for path, info in group.items():
            group[path] = apply_mono_workflow(info, workflow)
    reference_bias = (
        bias_info[biases[0]] if biases else master_bias_info[master_biases[0]] if master_biases else light_info[lights[0]]
    )
    profile_infos = [*bias_info.values(), *master_bias_info.values(), *dark_info.values(), *flat_info.values(), *master_dark_info.values(), *master_flat_info.values(), *light_info.values()]
    conflicts = conflicting_profile_fields(profile_infos, workflow)
    if conflicts:
        raise CalibrationError("CALIBRATION_PROFILE_MISMATCH", "Conflicting known acquisition metadata: " + ", ".join(conflicts))
    if not biases and not master_biases and not can_omit_bias(
        (*light_info.values(), *flat_info.values()),
        [*((info, True) for info in dark_info.values()), *((info, supplied_dark_bias_included[path]) for path, info in master_dark_info.items())],
        workflow,
    ):
        raise CalibrationError("BIAS_REQUIRED_FOR_CALIBRATION", "Bias is required unless every Light and raw Flat has a matching Dark that includes Bias.")
    for info in (*bias_info.values(), *master_bias_info.values()):
        assert_compatible(reference_bias, info, workflow=workflow)
    for info in (
        *dark_info.values(),
        *flat_info.values(),
        *master_dark_info.values(),
        *master_flat_info.values(),
        *light_info.values(),
    ):
        assert_compatible(reference_bias, info, workflow=workflow)

    flat_groups, supplied_flats, light_groups = _group_flats_and_lights(flat_info, master_flat_info, light_info, workflow)
    reference_exposures = {
        filter_name: min(float(light_info[path].exposure_seconds) for path in paths)
        for filter_name, paths in light_groups.items()
    }
    light_domain_references = {
        filter_name: light_info[paths[0]] for filter_name, paths in light_groups.items()
    }
    output_groups, group_filter, group_channel, group_cfa_pattern, light_cfa_pattern = _output_groups(
        light_groups, light_info, workflow
    )
    for filter_name, paths in flat_groups.items():
        reference = flat_info[paths[0]]
        for path in paths[1:]:
            assert_compatible(reference, flat_info[path], compare_filter=True, workflow=workflow)
        if filter_name in supplied_flats:
            raise CalibrationError(
                "FLAT_SOURCE_AMBIGUOUS",
                f"filter {filter_name} has both raw Flats and a supplied MasterFlat",
            )
    missing_flats = sorted(set(light_groups) - set(flat_groups) - set(supplied_flats))
    if missing_flats:
        raise CalibrationError(
            "MASTER_FLAT_MISSING",
            f"no raw Flat group or MasterFlat for Light filters: {', '.join(missing_flats)}",
        )
    tokens: dict[str, str] = {}
    for filter_name in {*flat_groups, *supplied_flats, *light_groups, *output_groups}:
        token = _safe_token(filter_name)
        if token in tokens and tokens[token] != filter_name:
            raise CalibrationError(
                "FILTER_FILENAME_COLLISION",
                f"filters {tokens[token]!r} and {filter_name!r} share output token {token}",
            )
        tokens[token] = filter_name
    dark_groups, supplied_darks = _group_darks(dark_info, master_dark_info, light_info)
    trusted_bias, trusted_darks_by_exposure, trusted_flats_by_filter = _trusted_generated_coverage(
        trusted_generated,
        has_raw_bias=bool(biases),
        reference_bias=reference_bias,
        dark_info=dark_info,
        dark_groups=dark_groups,
        flat_info=flat_info,
        flat_groups=flat_groups,
        parameters=parameters,
    )
    resolved_transforms = _resolve_transforms(lights, transforms)
    resolved_quality_weights = _resolve_quality_weights(lights, quality_weights)
    resolved_region_weight_maps = _resolve_region_weight_maps(lights, region_weight_maps)
    resolved_stellar_scale_hints = _resolve_stellar_scale_hints(
        lights, light_info, stellar_scale_hints, aliases, identity_cache
    )
    source_records, source_identities = _source_records(source_groups, aliases, identity_cache)
    pixel_numeric_domains = _pixel_numeric_domain_records(
        (
            ("BIAS", bias_info),
            ("DARK", dark_info),
            ("FLAT", flat_info),
            ("MASTER_BIAS", master_bias_info),
            ("MASTER_DARK", master_dark_info),
            ("MASTER_FLAT", master_flat_info),
            ("LIGHT", light_info),
        ),
        aliases,
        identity_cache,
    )
    return _RunPlan(
        output=output,
        lights=lights,
        biases=biases,
        master_biases=master_biases,
        source_aliases=aliases,
        source_groups=source_groups,
        source_identity_cache=identity_cache,
        trusted_generated=trusted_generated,
        bias_info=bias_info,
        dark_info=dark_info,
        flat_info=flat_info,
        master_dark_info=master_dark_info,
        master_flat_info=master_flat_info,
        light_info=light_info,
        supplied_dark_bias_included=supplied_dark_bias_included,
        reference_bias=reference_bias,
        flat_groups=flat_groups,
        supplied_flats=supplied_flats,
        light_groups=light_groups,
        reference_exposures=reference_exposures,
        light_domain_references=light_domain_references,
        output_groups=output_groups,
        group_filter=group_filter,
        group_channel=group_channel,
        group_cfa_pattern=group_cfa_pattern,
        light_cfa_pattern=light_cfa_pattern,
        dark_groups=dark_groups,
        supplied_darks=supplied_darks,
        trusted_bias=trusted_bias,
        trusted_darks_by_exposure=trusted_darks_by_exposure,
        trusted_flats_by_filter=trusted_flats_by_filter,
        transforms=resolved_transforms,
        quality_weights=resolved_quality_weights,
        region_weight_maps=resolved_region_weight_maps,
        stellar_scale_hints=resolved_stellar_scale_hints,
        source_records=source_records,
        source_identities=source_identities,
        pixel_numeric_domains=pixel_numeric_domains,
    )


def _group_flats_and_lights(
    flat_info: Mapping[Path, FrameInfo],
    master_flat_info: Mapping[Path, FrameInfo],
    light_info: Mapping[Path, FrameInfo],
    workflow: str,
) -> tuple[dict[str, list[Path]], dict[str, Path], dict[str, list[Path]]]:
    """Raw Flat groups, supplied MasterFlats and Light groups, all by filter."""

    flat_groups: dict[str, list[Path]] = {}
    for path, info in flat_info.items():
        flat_groups.setdefault(require_filter(info), []).append(path)
    supplied_flats: dict[str, Path] = {}
    for path, info in master_flat_info.items():
        filter_name = require_filter(info)
        if filter_name in supplied_flats:
            raise CalibrationError(
                "MASTER_FLAT_AMBIGUOUS",
                f"multiple supplied MasterFlats match filter {filter_name}",
            )
        supplied_flats[filter_name] = path
    light_groups: dict[str, list[Path]] = {}
    for path, info in light_info.items():
        if info.exposure_seconds is None or info.exposure_seconds <= 0:
            raise CalibrationError(
                "LIGHT_EXPOSURE_UNKNOWN",
                "Light requires positive EXPTIME",
                path=str(path),
            )
        light_groups.setdefault(require_filter(info), []).append(path)
    for filter_name, paths in light_groups.items():
        reference = light_info[paths[0]]
        for path in paths[1:]:
            assert_compatible(
                reference,
                light_info[path],
                compare_filter=True,
                compare_target=True,
                workflow=workflow,
            )
    return flat_groups, supplied_flats, light_groups


def _output_groups(
    light_groups: Mapping[str, list[Path]],
    light_info: Mapping[Path, FrameInfo],
    workflow: str,
) -> tuple[
    dict[str, list[Path]],
    dict[str, str],
    dict[str, int | None],
    dict[str, str | None],
    dict[str, str | None],
]:
    """Integration groups: a mono filter group is its own output group; a
    Bayer filter group is debayered into the colour channel groups R, G and
    B, each holding every Light of the filter, so the rest of the pipeline
    treats a colour channel exactly like a filter."""

    output_groups: dict[str, list[Path]] = {}
    group_filter: dict[str, str] = {}
    group_channel: dict[str, int | None] = {}
    group_cfa_pattern: dict[str, str | None] = {}
    light_cfa_pattern: dict[str, str | None] = {}
    for filter_name, paths in light_groups.items():
        pattern = normalize_cfa_pattern(cfa_for_workflow(light_info[paths[0]].cfa_pattern, workflow))
        if pattern in CFA_PATTERNS:
            light_cfa_pattern[filter_name] = pattern
            for channel, channel_name in enumerate(CHANNEL_NAMES):
                if channel_name in output_groups:
                    raise CalibrationError(
                        "CFA_CHANNEL_GROUP_COLLISION",
                        f"colour channel {channel_name} of Bayer filter {filter_name!r} collides with "
                        f"filter or channel group {group_filter[channel_name]!r}; one run integrates one "
                        "Bayer filter and no mono R/G/B filters alongside it",
                    )
                output_groups[channel_name] = list(paths)
                group_filter[channel_name] = filter_name
                group_channel[channel_name] = channel
                group_cfa_pattern[channel_name] = pattern
        else:
            if pattern not in {"NONE", "UNKNOWN", "UNSPECIFIED", ""}:
                raise CalibrationError(
                    "CFA_PATTERN_UNSUPPORTED",
                    f"Bayer pattern {pattern!r} of filter {filter_name!r} is not supported "
                    f"(supported: {', '.join(sorted(CFA_PATTERNS))})",
                )
            light_cfa_pattern[filter_name] = None
            if filter_name in output_groups:
                raise CalibrationError(
                    "CFA_CHANNEL_GROUP_COLLISION",
                    f"filter {filter_name!r} collides with a colour channel group of Bayer filter "
                    f"{group_filter[filter_name]!r}",
                )
            output_groups[filter_name] = list(paths)
            group_filter[filter_name] = filter_name
            group_channel[filter_name] = None
            group_cfa_pattern[filter_name] = None
    return output_groups, group_filter, group_channel, group_cfa_pattern, light_cfa_pattern


def _group_darks(
    dark_info: Mapping[Path, FrameInfo],
    master_dark_info: Mapping[Path, FrameInfo],
    light_info: Mapping[Path, FrameInfo],
) -> tuple[dict[float, list[Path]], dict[float, Path]]:
    """Raw Dark groups and supplied MasterDarks by exposure; every Light
    needs an exact exposure match once any Dark is supplied."""

    dark_groups: dict[float, list[Path]] = {}
    for path, info in dark_info.items():
        if info.exposure_seconds is None or info.exposure_seconds <= 0:
            raise CalibrationError(
                "DARK_EXPOSURE_UNKNOWN", "Dark requires positive EXPTIME", path=str(path)
            )
        dark_groups.setdefault(info.exposure_seconds, []).append(path)
    supplied_darks: dict[float, Path] = {}
    for path, info in master_dark_info.items():
        if info.exposure_seconds is None or info.exposure_seconds <= 0:
            raise CalibrationError(
                "DARK_EXPOSURE_UNKNOWN",
                "MasterDark requires positive EXPTIME",
                path=str(path),
            )
        if find_dark(info.exposure_seconds, supplied_darks) is not None:
            raise CalibrationError(
                "MASTER_DARK_AMBIGUOUS",
                "multiple supplied MasterDarks have the same exposure",
                path=str(path),
            )
        if find_dark(info.exposure_seconds, {value: Path() for value in dark_groups}) is not None:
            raise CalibrationError(
                "DARK_SOURCE_AMBIGUOUS",
                "an exposure has both raw Darks and a supplied MasterDark",
                path=str(path),
            )
        supplied_darks[info.exposure_seconds] = path
    available_dark_exposures = {
        **{value: Path() for value in dark_groups},
        **supplied_darks,
    }
    if available_dark_exposures:
        for path, info in light_info.items():
            if find_dark(info.exposure_seconds, available_dark_exposures) is None:
                raise CalibrationError(
                    "DARK_EXPOSURE_MISMATCH",
                    "no exact raw Dark or MasterDark exposure matches this Light",
                    path=str(path),
                )
    return dark_groups, supplied_darks


def _trusted_generated_coverage(
    trusted_generated: Mapping[str, Any] | None,
    *,
    has_raw_bias: bool,
    reference_bias: FrameInfo,
    dark_info: Mapping[Path, FrameInfo],
    dark_groups: Mapping[float, list[Path]],
    flat_info: Mapping[Path, FrameInfo],
    flat_groups: Mapping[str, list[Path]],
    parameters: PipelineParameters,
) -> tuple[
    TrustedGeneratedMaster | None,
    dict[float, TrustedGeneratedMaster],
    dict[str, TrustedGeneratedMaster],
]:
    """Map E2E-generated masters onto the raw calibration groups they replace."""

    trusted_bias = trusted_generated["bias"] if trusted_generated is not None else None
    trusted_darks_by_exposure: dict[float, TrustedGeneratedMaster] = {}
    trusted_flats_by_filter: dict[str, TrustedGeneratedMaster] = {}
    if trusted_generated is None:
        return trusted_bias, trusted_darks_by_exposure, trusted_flats_by_filter
    workflow = parameters.calibration_workflow
    if has_raw_bias != (trusted_bias is not None):
        raise CalibrationError(
            "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
            "generated MasterBias coverage does not match raw Bias provenance",
        )
    for item in trusted_generated["darks"]:
        exposure = item.frame_info.exposure_seconds
        if exposure is None or exposure <= 0 or find_dark(
            exposure, {value: Path() for value in trusted_darks_by_exposure}
        ) is not None:
            raise CalibrationError(
                "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
                "generated MasterDark exposures are invalid or ambiguous",
                path=item.path,
            )
        trusted_darks_by_exposure[float(exposure)] = item
    for item in trusted_generated["flats"]:
        filter_name = require_filter(item.frame_info)
        if filter_name in trusted_flats_by_filter:
            raise CalibrationError(
                "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
                "generated MasterFlat filters are ambiguous",
                path=item.path,
            )
        trusted_flats_by_filter[filter_name] = item
    if set(trusted_darks_by_exposure) != set(dark_groups):
        raise CalibrationError(
            "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
            "generated MasterDark coverage does not match raw Dark provenance",
        )
    if set(trusted_flats_by_filter) != set(flat_groups):
        raise CalibrationError(
            "TRUSTED_GENERATED_CALIBRATION_COVERAGE_MISMATCH",
            "generated MasterFlat coverage does not match raw Flat provenance",
        )
    if trusted_bias is not None:
        assert_compatible(reference_bias, trusted_bias.frame_info, workflow=workflow)
    for exposure, item in trusted_darks_by_exposure.items():
        assert_compatible(
            dark_info[dark_groups[exposure][0]],
            item.frame_info,
            compare_exposure=True,
            compare_temperature=True,
            temperature_tolerance_celsius=parameters.dark_temperature_tolerance_celsius,
            workflow=workflow,
        )
    for filter_name, item in trusted_flats_by_filter.items():
        assert_compatible(
            flat_info[flat_groups[filter_name][0]],
            item.frame_info,
            compare_filter=True,
            workflow=workflow,
        )
    return trusted_bias, trusted_darks_by_exposure, trusted_flats_by_filter


@dataclass(frozen=True)
class _StagingDirs:
    root: Path
    masters: Path
    calibrated: Path
    registered: Path
    coverage: Path
    previews: Path
    work: Path

    @classmethod
    def create(cls, root: Path) -> _StagingDirs:
        dirs = cls(
            root=root,
            masters=root / "masters",
            calibrated=root / "calibrated",
            registered=root / "registered",
            coverage=root / "coverage",
            previews=root / "previews",
            work=root / ".work",
        )
        for directory in (
            dirs.masters,
            dirs.calibrated,
            dirs.registered,
            dirs.coverage,
            dirs.previews,
            dirs.work,
        ):
            directory.mkdir()
        return dirs


@dataclass
class _RunLedger:
    """The evidence a run accumulates for its receipt, in production order."""

    staging: Path
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    stage_statistics: dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        path: Path,
        kind: str,
        *,
        statistics: PixelStatistics | None = None,
        details: Mapping[str, Any] | None = None,
        sha256: str | None = None,
    ) -> dict[str, Any]:
        record = _artifact_record(
            self.staging, path, kind, statistics=statistics, details=details, sha256=sha256
        )
        self.artifacts.append(record)
        return record
