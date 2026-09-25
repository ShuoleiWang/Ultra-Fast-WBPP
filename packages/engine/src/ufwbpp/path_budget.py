"""Output layout names and the projected path length of a run.

Windows without the ``LongPathsEnabled`` policy refuses paths beyond 259
characters (``MAX_PATH``), and a project run nests its work three staging
levels deep::

    <output.parent>/.<output.name>.<8>.pstage/            project staging
        details/runs/.<TARGET>.<8>.stage/                 one run per target
            work/.pp.<8>.stage/                           pixel pipeline
                registered/.00001_<light stem>.fits.partial

The failure would surface only when the deepest temporary is opened, long
after the run started, as ``ERROR_PATH_NOT_FOUND`` disguised as a missing
file.  This module owns the fixed parts of those names so that the modules
creating the directories and the projection below cannot drift apart, and it
turns the projection into a fail-closed check before any pixel work.

Every fixed part is deliberately short: the eight random characters that
``tempfile`` appends and the Light's own file stem are what the budget has
to leave room for.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Iterable, Literal

from .errors import RuntimeConfigurationError
from .platform import PathLimit, current
from .platform.windows import LONG_PATHS_REGISTRY_KEY


# ``tempfile.mkdtemp``/``mkstemp`` insert eight random characters between the
# prefix and the suffix of every staging name.
RANDOM_NAME_LENGTH = 8
# ``<output.parent>/.<output.name>.<8>.pstage`` (``workflows.project.run_project_e2e``).
PROJECT_STAGING_SUFFIX = ".pstage"
# ``<output.parent>/.<output.name>.<8>.stage`` (``workflows.single_target.run_e2e``
# and ``stacking.pipeline.run_portable_pipeline_fits``).
STAGING_SUFFIX = ".stage"
# The E2E run names its pipeline root ``work/pixel-pipeline`` but stages it as
# ``work/.pp.<8>.stage``: the root's name stays readable in receipts while the
# transient directory, the deepest level of the layout, spends no budget on it.
PIXEL_PIPELINE_DIRECTORY = "pixel-pipeline"
PIXEL_PIPELINE_STAGING_STEM = "pp"
# Fixed directory names of the layout that the projection walks through.
DETAILS_DIRECTORY = "details"
RUNS_DIRECTORY = "runs"
WORK_DIRECTORY = "work"
# ``image_io.fits.temporary_output`` (``mkstemp``): ``.<name>.<8>.partial``.
TEMPORARY_SUFFIX = ".partial"

Layout = Literal["project", "run", "pixels"]

_PLACEHOLDER = "x" * RANDOM_NAME_LENGTH


def name_token(value: str) -> str:
    """Upper-case ``[A-Z0-9_]`` token used for filter, channel group and
    target directory names (empty when nothing survives; callers raise)."""

    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def target_key(value: str) -> str:
    """Lower-case alphanumeric key of a science target (``workflows.project``)."""

    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def light_stem(path: str | os.PathLike[str]) -> str:
    """The stem the pixel pipeline gives a Light's calibrated/registered files."""

    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(path).stem).strip("_") or "light"


def staging_directory_name(name: str, *, suffix: str = STAGING_SUFFIX) -> str:
    """``.<name>.<8 random>.<suffix>`` as ``tempfile.mkdtemp`` composes it."""

    return f".{name}.{_PLACEHOLDER}{suffix}"


def frame_index_width(light_count: int) -> int:
    """Digits of the ``{index:05d}`` prefix of calibrated/registered frames."""

    return max(5, len(str(max(1, int(light_count)))))


def _longest(values: Iterable[str], fallback: str) -> str:
    longest = fallback
    for value in values:
        if len(value) > len(longest):
            longest = value
    return longest


def _pixel_staging_candidates(index_width: int, stem: str, filter_token: str) -> list[str]:
    """Deepest names inside a pixel-pipeline staging directory.

    Calibrated Lights are written through ``mkstemp`` temporaries, registered
    Lights through ``.<name>.partial`` (with a one-letter colour-channel
    suffix for Bayer Lights) and the integration keeps its rejection maps in
    ``.work``.  The projection keeps every variant because the parameters that
    select them (materialized calibrated Lights, CFA) are decided later than
    this check.
    """

    index = "0" * index_width
    return [
        f"calibrated/.{index}_{stem}.fits.{_PLACEHOLDER}{TEMPORARY_SUFFIX}",
        f"registered/.{index}_{stem}_R.fits{TEMPORARY_SUFFIX}",
        f".work/.rejection_count_{filter_token}.fits.{_PLACEHOLDER}{TEMPORARY_SUFFIX}",
    ]


def _run_candidates(index_width: int, stem: str, filter_token: str) -> list[str]:
    """Deepest names inside an E2E run's staging directory."""

    pixel_staging = staging_directory_name(PIXEL_PIPELINE_STAGING_STEM)
    candidates = [
        f"{WORK_DIRECTORY}/{pixel_staging}/{name}"
        for name in _pixel_staging_candidates(index_width, stem, filter_token)
    ]
    candidates.append(
        f"{WORK_DIRECTORY}/drizzle/{filter_token}/.master_light_{filter_token}"
        f"_drizzle_unsolved.fits{TEMPORARY_SUFFIX}"
    )
    candidates.append(
        f"{WORK_DIRECTORY}/registration-calibration/.master_flat_{filter_token}"
        f".fits.{_PLACEHOLDER}{TEMPORARY_SUFFIX}"
    )
    # ``lightframeqc.measure._thumbnail_name``: token, index, stem (64 max), digest.
    candidates.append(f"qc/thumbnails/{'x' * 10}-{'0' * 6}-{stem[:64]}-{'x' * 10}.png")
    return candidates


def projected_maximum_path(
    output: str | os.PathLike[str],
    *,
    targets: Iterable[str] = (),
    filters: Iterable[str] = (),
    light_count: int = 1,
    light_paths: Iterable[str | os.PathLike[str]] = (),
    layout: Layout = "project",
) -> tuple[int, str]:
    """The longest path a run into ``output`` may create, and that path.

    ``targets`` and ``filters`` are display names (their directory tokens are
    derived here with the pipeline's own rules); ``light_paths`` supply the
    Light file stems that the calibrated and registered frame names carry.
    XISF Lights are staged under short numbered names first, so only FITS
    stems count.  ``layout`` selects the entry point: a multi-target
    ``project`` (``run_project_e2e``), a single ``run`` (``run_e2e``) or the
    bare ``pixels`` pipeline (``run_portable_pipeline``).  The result is a
    conservative maximum over the variants a run may choose later (CFA
    groups, drizzle, local normalization), never a shorter typical case.
    """

    resolved = Path(output).expanduser().resolve(strict=False)
    stems: list[str] = []
    for path in light_paths:
        name = Path(path).name.casefold()
        if name.endswith((".fit", ".fits", ".fts")):
            stems.append(light_stem(path))
        else:
            stems.append("0" * 6 + "_light")
    stem = _longest(stems, "light")
    filter_token = _longest((name_token(name) for name in filters), "R")
    index_width = frame_index_width(light_count)

    if layout == "pixels":
        staging = staging_directory_name(resolved.name)
        relative = [
            f"{staging}/{name}"
            for name in _pixel_staging_candidates(index_width, stem, filter_token)
        ]
    elif layout == "run":
        staging = staging_directory_name(resolved.name)
        relative = [f"{staging}/{name}" for name in _run_candidates(index_width, stem, filter_token)]
    elif layout == "project":
        target_token = _longest((name_token(target_key(name)) for name in targets), "T")
        run_staging = staging_directory_name(target_token)
        staging = staging_directory_name(resolved.name, suffix=PROJECT_STAGING_SUFFIX)
        relative = [
            f"{staging}/{DETAILS_DIRECTORY}/{RUNS_DIRECTORY}/{run_staging}/{name}"
            for name in _run_candidates(index_width, stem, filter_token)
        ]
    else:
        raise ValueError(f"unknown layout {layout!r}")

    deepest = max((resolved.parent / candidate for candidate in relative), key=lambda path: len(str(path)))
    return len(str(deepest)), str(deepest)


def check_output_path_budget(
    output: str | os.PathLike[str],
    *,
    targets: Iterable[str] = (),
    filters: Iterable[str] = (),
    light_count: int = 1,
    light_paths: Iterable[str | os.PathLike[str]] = (),
    layout: Layout = "project",
    limit: PathLimit | None = None,
) -> tuple[int, str]:
    """Refuse an output directory whose run would overflow the path limit.

    Returns the projection.  Raises
    ``RuntimeConfigurationError("OUTPUT_PATH_TOO_LONG", ...)`` naming the
    projected length, the limit and both remedies: a shorter output
    directory (with the length that still fits) or the long-path policy.
    ``limit`` defaults to the running platform's; tests inject it.
    """

    resolved = Path(output).expanduser().resolve(strict=False)
    targets = tuple(targets)
    filters = tuple(filters)
    light_paths = tuple(light_paths)
    projected, deepest = projected_maximum_path(
        resolved,
        targets=targets,
        filters=filters,
        light_count=light_count,
        light_paths=light_paths,
        layout=layout,
    )
    if limit is None:
        limit = current().path_limit()
    if limit.max_characters is None or projected <= limit.max_characters:
        return projected, deepest
    budget = limit.max_characters - (projected - len(str(resolved)))
    policy = (
        "long paths are disabled"
        if limit.long_paths_enabled is False
        else "the long-path policy could not be read"
        if limit.long_paths_enabled is None
        else "even with long paths enabled"
    )
    raise RuntimeConfigurationError(
        "OUTPUT_PATH_TOO_LONG",
        f"the run would create paths of up to {projected} characters (for example "
        f"{deepest}) but this Windows installation accepts at most {limit.max_characters} "
        f"({policy}); choose a shorter output directory (at most {budget} characters, the "
        f"chosen one has {len(str(resolved))}) or enable long paths (set "
        f"{LONG_PATHS_REGISTRY_KEY} to 1 as an administrator and sign in again)",
    )


__all__ = [
    "DETAILS_DIRECTORY",
    "PIXEL_PIPELINE_DIRECTORY",
    "PIXEL_PIPELINE_STAGING_STEM",
    "PROJECT_STAGING_SUFFIX",
    "RANDOM_NAME_LENGTH",
    "RUNS_DIRECTORY",
    "STAGING_SUFFIX",
    "TEMPORARY_SUFFIX",
    "WORK_DIRECTORY",
    "check_output_path_budget",
    "frame_index_width",
    "light_stem",
    "name_token",
    "projected_maximum_path",
    "staging_directory_name",
    "target_key",
]
