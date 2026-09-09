"""Black-box contract tests for the ``prepare-wbpp`` CLI.

These tests intentionally exercise the public command rather than private
planner helpers.  The preparation implementation is landing independently, so
the suite is allowed to be red while that interface is unavailable.  Once the
command exists, every assertion below is an acceptance boundary: planning is
read-only, applying publishes a WBPP-ready tree, and neither path may mutate an
input frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from astropy.io import fits
import numpy as np
import pytest

from lightframeqc import cli


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FILTERS = ("R", "G", "B")
TARGETS = ("NGC 7000", "M31")


@dataclass(frozen=True)
class FileSnapshot:
    sha256: str
    size: int
    mtime_ns: int
    mode: int
    device: int
    inode: int


@dataclass(frozen=True)
class PrepareFixture:
    root: Path
    input_directory: Path
    lights: tuple[Path, ...]
    automatic_flats: tuple[Path, ...]
    non_lights: tuple[Path, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(64 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(path: Path) -> FileSnapshot:
    status = path.stat()
    return FileSnapshot(
        sha256=_sha256(path),
        size=status.st_size,
        mtime_ns=status.st_mtime_ns,
        mode=status.st_mode,
        device=status.st_dev,
        inode=status.st_ino,
    )


def _snapshot_many(paths: Iterable[Path]) -> dict[Path, FileSnapshot]:
    return {path.resolve(): _snapshot(path) for path in sorted(paths)}


def _snapshot_tree(root: Path) -> dict[str, FileSnapshot]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): _snapshot(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _set_fixed_mtime(path: Path, index: int) -> None:
    # Distinct nanosecond values catch accidental touch/copy-back operations.
    timestamp = 1_750_000_000_000_000_000 + index * 1_000_003
    os.utime(path, ns=(timestamp, timestamp))


def _star_field(*, target_index: int, frame_index: int) -> np.ndarray:
    height = width = 128
    yy, xx = np.mgrid[:height, :width]
    rng = np.random.default_rng(50_000 + 100 * target_index + frame_index)
    image = rng.normal(1_000.0, 2.0, size=(height, width))
    center_rng = np.random.default_rng(42_000 + target_index)
    centers: list[tuple[float, float]] = []
    while len(centers) < 72:
        candidate = tuple(center_rng.uniform(7.0, 121.0, size=2))
        if all(
            (candidate[0] - x0) ** 2 + (candidate[1] - y0) ** 2 >= 6.0**2
            for x0, y0 in centers
        ):
            centers.append(candidate)
    dither_x = 0.25 * (frame_index % 2)
    dither_y = 0.20 * ((frame_index // 2) % 2)
    for star_index, (x0, y0) in enumerate(centers):
        sigma = 1.35 + 0.05 * (star_index % 3)
        amplitude = (650.0 + 25.0 * star_index) * (0.94 + 0.02 * frame_index)
        image += amplitude * np.exp(
            -(
                (xx - x0 - dither_x) ** 2
                + (yy - y0 - dither_y) ** 2
            )
            / (2.0 * sigma**2)
        )
    return image.astype(np.float32)


def _base_header(
    hdu: fits.PrimaryHDU,
    *,
    image_type: str,
    filter_name: str,
    target: str,
    sequence: int,
) -> None:
    header = hdu.header
    header["IMAGETYP"] = image_type
    header["EXPOSURE"] = 300.0 if image_type == "LIGHT" else 1.0
    header["EXPTIME"] = 300.0 if image_type == "LIGHT" else 1.0
    header["DATE-OBS"] = f"2026-08-20T16:{sequence:02d}:00.000"
    header["XBINNING"] = 1
    header["YBINNING"] = 1
    header["GAIN"] = 0
    header["OFFSET"] = 30
    header["XPIXSZ"] = 3.76
    header["YPIXSZ"] = 3.76
    header["INSTRUME"] = "QHY268M"
    header["CAMERAID"] = "QHY268M-contract-fixture"
    header["READOUTM"] = "High Gain 2CMS"
    header["BAYERPAT"] = "NONE"
    header["TELESCOP"] = "NewTon"
    header["FOCALLEN"] = 1_200.0
    header["FILTER"] = filter_name
    header["OBJECT"] = target
    header["RA"] = 282.0 + sequence / 1_000.0
    header["DEC"] = -7.0
    header["CENTALT"] = 55.0
    header["CENTAZ"] = 210.0
    header["AIRMASS"] = 1.22
    header["ROWORDER"] = "TOP-DOWN"
    header["SWCREATE"] = "N.I.N.A. 3.1.2.9001 (x64)"


def _write_light(
    path: Path,
    *,
    target: str,
    filter_name: str,
    target_index: int,
    sequence: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(
        _star_field(target_index=target_index, frame_index=sequence)
    )
    _base_header(
        hdu,
        image_type="LIGHT",
        filter_name=filter_name,
        target=target,
        sequence=sequence,
    )
    hdu.writeto(path)


def _write_master_flat(
    path: Path,
    *,
    filter_name: str,
    variant: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:128, :128]
    pixels = (
        24_000.0
        + 9.0 * xx
        + 5.0 * yy
        + variant * (100.0 + ((xx + yy) % 7))
    ).astype(np.float32)
    hdu = fits.PrimaryHDU(pixels)
    _base_header(
        hdu,
        image_type="Master Flat",
        filter_name=filter_name,
        target="FlatWizard",
        sequence=variant,
    )
    hdu.writeto(path)


def _write_non_light(path: Path, image_type: str, *, sequence: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(np.full((128, 128), 1_000 + sequence, dtype=np.float32))
    _base_header(
        hdu,
        image_type=image_type,
        filter_name="R",
        target="CALIBRATION",
        sequence=sequence,
    )
    hdu.writeto(path)


@pytest.fixture
def prepare_fixture(tmp_path: Path) -> PrepareFixture:
    # Inputs are rooted at root/captures/LIGHT.  Automatic flats are top-level
    # files exactly two ancestors above that directory.
    root = tmp_path / "source-root"
    input_directory = root / "captures" / "LIGHT"
    lights: list[Path] = []
    for target_index, target in enumerate(TARGETS):
        target_token = target.lower().replace(" ", "-")
        for filter_index, filter_name in enumerate(FILTERS):
            for sequence in range(8):
                path = (
                    input_directory
                    / f"night-{target_index + 1}"
                    / filter_name
                    / (
                        f"{target_token}_300.00s_{filter_name}_0_"
                        f"2026-08-{20 + target_index:02d}_"
                        f"{sequence:02d}-00-00_-10.00C_"
                        f"HFR2.40_Focus17100_{sequence:05d}.fits"
                    )
                )
                _write_light(
                    path,
                    target=target,
                    filter_name=filter_name,
                    target_index=target_index,
                    sequence=sequence + filter_index * 10,
                )
                lights.append(path)

    automatic_flats: list[Path] = []
    for filter_index, filter_name in enumerate(FILTERS):
        path = (
            root
            / f"masterFlat_BIN-1_128x128_FILTER-{filter_name}_mono.fits"
        )
        _write_master_flat(path, filter_name=filter_name)
        automatic_flats.append(path)

    non_lights = (
        input_directory / "mixed" / "raw-flat.fits",
        input_directory / "mixed" / "dark.fits",
        input_directory / "mixed" / "bias.fits",
        input_directory / "mixed" / "master-light.fits",
    )
    for index, (path, role) in enumerate(
        zip(non_lights, ("FLAT", "DARK", "BIAS", "MASTER LIGHT"), strict=True)
    ):
        _write_non_light(path, role, sequence=40 + index)
    (input_directory / "mixed" / "notes.txt").write_text(
        "not an astronomical frame\n", encoding="utf-8"
    )

    all_fits = [*lights, *automatic_flats, *non_lights]
    for index, path in enumerate(sorted(all_fits)):
        _set_fixed_mtime(path, index)
    return PrepareFixture(
        root=root,
        input_directory=input_directory,
        lights=tuple(lights),
        automatic_flats=tuple(automatic_flats),
        non_lights=non_lights,
    )


def _invoke(arguments: list[str]) -> int:
    """Return an argparse exit code as an ordinary result for assertions."""

    try:
        return cli.main(arguments)
    except SystemExit as error:
        return int(error.code) if isinstance(error.code, int) else 1


def _base_arguments(
    fixture: PrepareFixture,
    *,
    destination: Path,
    report: Path,
) -> list[str]:
    return [
        "prepare-wbpp",
        str(fixture.input_directory),
        "-o",
        str(destination),
        "--report-output",
        str(report),
        "--config",
        str(PACKAGE_ROOT / "default-config.json"),
        "--workers",
        "1",
        "--no-thumbnails",
    ]


def _load_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant emitted: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    assert isinstance(value, dict)
    return value


def _root_manifest(destination: Path) -> Path:
    manifests = sorted(destination.glob("*.json"))
    assert len(manifests) == 1, (
        "an applied WBPP tree must contain exactly one root JSON manifest, "
        f"found {manifests}"
    )
    return manifests[0]


def test_prepare_plan_only_writes_reports_filters_non_lights_and_never_creates_dest(
    prepare_fixture: PrepareFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "wbpp-ready"
    report = tmp_path / "prepare-report"
    adjudication = tmp_path / "adjudication.json"
    adjudication.write_text(
        json.dumps({"schemaVersion": 1, "records": []}), encoding="utf-8"
    )
    sources = [
        *prepare_fixture.lights,
        *prepare_fixture.automatic_flats,
        *prepare_fixture.non_lights,
    ]
    before = _snapshot_many(sources)

    return_code = _invoke(
        _base_arguments(
            prepare_fixture, destination=destination, report=report
        )
        + ["--adjudication", str(adjudication)]
    )

    captured = capsys.readouterr()
    assert return_code == 0, captured.err
    assert not destination.exists(), "plan-only mode must not create DEST"
    results_path = report / "results.json"
    plan_path = report / "prepare-plan.json"
    assert results_path.is_file()
    assert plan_path.is_file()

    results = _load_json(results_path)
    frames = results.get("frames")
    assert isinstance(frames, list)
    assert len(frames) == len(prepare_fixture.lights)
    assert {Path(frame["path"]).resolve() for frame in frames} == {
        path.resolve() for path in prepare_fixture.lights
    }
    assert all(frame["metadata"]["role"] == "LIGHT" for frame in frames)
    serialized_results = json.dumps(results, sort_keys=True)
    assert all(str(path.resolve()) not in serialized_results for path in prepare_fixture.non_lights)

    plan = _load_json(plan_path)
    assert plan.get("schemaVersion") == 1
    assert plan.get("kind") == "light-frame-qc.prepare-wbpp"
    assert plan.get("layoutId") == "per-target-light-flat-v1"
    assert isinstance(plan.get("planId"), str) and str(plan["planId"]).startswith(
        "sha256:"
    )
    entries = plan.get("entries")
    assert isinstance(entries, list)
    assert sum(entry.get("action") == "COPY_LIGHT" for entry in entries) == len(
        prepare_fixture.lights
    )
    assert sum(
        entry.get("action") == "COPY_MASTER_FLAT" for entry in entries
    ) == len(TARGETS) * len(FILTERS)
    for entry in entries:
        identity = entry.get("sourceIdentity")
        assert isinstance(identity, dict)
        assert set(identity) == {"sha256", "sizeBytes", "mtimeNs", "device", "inode"}
        assert len(identity["sha256"]) == 64
    serialized_plan = json.dumps(plan, sort_keys=True)
    for target in TARGETS:
        assert target in serialized_plan
    for filter_name in FILTERS:
        assert filter_name in serialized_plan
    assert _snapshot_many(sources) == before


def test_prepare_apply_writes_two_target_rgb_tree_manifest_and_byte_exact_copies(
    prepare_fixture: PrepareFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "wbpp-ready"
    report = tmp_path / "prepare-report"
    sources = [
        *prepare_fixture.lights,
        *prepare_fixture.automatic_flats,
        *prepare_fixture.non_lights,
    ]
    before = _snapshot_many(sources)

    plan_return_code = _invoke(
        _base_arguments(prepare_fixture, destination=destination, report=report)
    )
    plan_capture = capsys.readouterr()
    assert plan_return_code == 0, plan_capture.err
    return_code = _invoke(
        ["apply-wbpp-plan", str(report / "prepare-plan.json")]
    )

    captured = capsys.readouterr()
    assert return_code == 0, captured.err
    assert destination.is_dir()
    manifest_path = _root_manifest(destination)
    manifest = _load_json(manifest_path)
    assert manifest.get("schemaVersion") == 1
    entries = manifest.get("entries")
    assert isinstance(entries, list)
    target_slugs = sorted(
        {
            entry["targetSlug"]
            for entry in entries
            if entry.get("role") == "LIGHT"
        }
    )
    assert len(target_slugs) == len(TARGETS)
    assert all(
        isinstance(slug, str)
        and slug not in {"", ".", ".."}
        and "/" not in slug
        and "\\" not in slug
        for slug in target_slugs
    )

    for slug in target_slugs:
        target_directory = destination / slug
        assert target_directory.is_dir()
        for filter_name in FILTERS:
            light_directory = target_directory / "LIGHT" / filter_name
            flat_directory = target_directory / "FLAT" / filter_name
            assert light_directory.is_dir()
            assert flat_directory.is_dir()
            assert len(list(light_directory.glob("*.fits"))) == 8
            assert len(list(flat_directory.glob("*.fits"))) == 1

    manifest_text = json.dumps(manifest, sort_keys=True)
    assert all(slug in manifest_text for slug in target_slugs)
    assert all(filter_name in manifest_text for filter_name in FILTERS)
    assert not any(path.is_symlink() for path in destination.rglob("*"))

    # Light basenames are preserved and every copy is byte-identical without
    # aliasing its source inode.
    for source in prepare_fixture.lights:
        matches = list(destination.glob(f"*/LIGHT/*/{source.name}"))
        assert len(matches) == 1
        copied = matches[0]
        assert _sha256(copied) == _sha256(source)
        assert (copied.stat().st_dev, copied.stat().st_ino) != (
            source.stat().st_dev,
            source.stat().st_ino,
        )

    for source in prepare_fixture.automatic_flats:
        matches = list(destination.glob(f"*/FLAT/*/{source.name}"))
        assert len(matches) == len(TARGETS)
        assert all(_sha256(copied) == _sha256(source) for copied in matches)

    assert _snapshot_many(sources) == before


@pytest.mark.parametrize(
    ("case", "diagnostic"),
    [
        ("missing", "MISSING_MASTER_FLAT"),
        ("ambiguous", "AMBIGUOUS_MASTER_FLAT"),
    ],
)
def test_prepare_missing_or_ambiguous_flat_fails_without_publishing_dest(
    prepare_fixture: PrepareFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    case: str,
    diagnostic: str,
) -> None:
    if case == "missing":
        removed = next(
            path
            for path in prepare_fixture.automatic_flats
            if "FILTER-B" in path.name
        )
        removed.unlink()
        # This candidate is three ancestors above INPUT and must not be found:
        # INPUT=source-root/captures/LIGHT -> captures (1), source-root (2),
        # tmp_path (3).
        too_far = tmp_path / "masterFlat_TOO_FAR_FILTER-B_mono.fits"
        _write_master_flat(too_far, filter_name="B")
        _set_fixed_mtime(too_far, 998)
    else:
        conflicting = (
            prepare_fixture.root
            / "masterFlat_SECOND_BIN-1_128x128_FILTER-R_mono.fits"
        )
        _write_master_flat(conflicting, filter_name="R", variant=1)
        _set_fixed_mtime(conflicting, 999)

    destination = tmp_path / "wbpp-ready"
    report = tmp_path / "prepare-report"
    remaining_sources = sorted(
        path for path in tmp_path.rglob("*.fits") if path.is_file()
    )
    before = _snapshot_many(remaining_sources)

    return_code = _invoke(
        _base_arguments(
            prepare_fixture, destination=destination, report=report
        )
        + ["--apply"]
    )

    captured = capsys.readouterr()
    assert return_code != 0
    assert diagnostic in captured.err
    assert not destination.exists()
    assert _snapshot_many(remaining_sources) == before
    assert not list(tmp_path.glob(".wbpp-ready*"))


def test_prepare_explicit_library_and_master_file_complete_missing_auto_flats(
    prepare_fixture: PrepareFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    green_auto = next(
        path for path in prepare_fixture.automatic_flats if "FILTER-G" in path.name
    )
    blue_auto = next(
        path for path in prepare_fixture.automatic_flats if "FILTER-B" in path.name
    )
    green_auto.unlink()
    blue_auto.unlink()

    library = tmp_path / "flat-library"
    green_library = library / "masterFlat_FILTER-G.fits"
    explicit_blue = tmp_path / "explicit" / "masterFlat_FILTER-B.fits"
    _write_master_flat(green_library, filter_name="G")
    _write_master_flat(explicit_blue, filter_name="B")
    _set_fixed_mtime(green_library, 700)
    _set_fixed_mtime(explicit_blue, 701)

    destination = tmp_path / "wbpp-ready"
    report = tmp_path / "prepare-report"
    return_code = _invoke(
        _base_arguments(
            prepare_fixture, destination=destination, report=report
        )
        + [
            "--flat-library",
            str(library),
            "--master-flat",
            str(explicit_blue),
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0, captured.err
    assert not destination.exists()
    assert (report / "prepare-plan.json").is_file()
    plan = _load_json(report / "prepare-plan.json")
    source_paths = {entry["sourcePath"] for entry in plan["entries"]}
    assert str(green_library.resolve()) in source_paths
    assert str(explicit_blue.resolve()) in source_paths


def test_prepare_hash_bound_reject_adjudication_removes_only_that_keep_frame(
    prepare_fixture: PrepareFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rejected = prepare_fixture.lights[0]
    adjudication = tmp_path / "adjudication.json"
    adjudication.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "records": [
                    {
                        "path": str(rejected.resolve()),
                        "sha256": _sha256(rejected),
                        "action": "REJECT",
                        "reason": "contract-test manual rejection",
                        "reviewer": "pytest",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "wbpp-ready"
    report = tmp_path / "prepare-report"

    return_code = _invoke(
        _base_arguments(
            prepare_fixture, destination=destination, report=report
        )
        + ["--adjudication", str(adjudication), "--apply"]
    )

    captured = capsys.readouterr()
    assert return_code == 0, captured.err
    assert not list(destination.rglob(rejected.name))
    assert sum(
        1
        for path in destination.glob("*/LIGHT/*/*.fits")
        if path.is_file()
    ) == len(prepare_fixture.lights) - 1
    manifest = _load_json(_root_manifest(destination))
    rejected_entry = next(
        entry for entry in manifest["entries"] if entry["sourcePath"] == str(rejected.resolve())
    )
    assert rejected_entry["adjudication"]["reason"] == "contract-test manual rejection"


def test_explicit_master_flat_overrides_automatic_same_profile(
    prepare_fixture: PrepareFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    explicit = tmp_path / "explicit" / "replacement-R.fits"
    _write_master_flat(explicit, filter_name="R", variant=7)
    destination = tmp_path / "wbpp-ready"
    report = tmp_path / "prepare-report"

    return_code = _invoke(
        _base_arguments(prepare_fixture, destination=destination, report=report)
        + ["--master-flat", str(explicit)]
    )

    captured = capsys.readouterr()
    assert return_code == 0, captured.err
    plan = _load_json(report / "prepare-plan.json")
    red_flats = [
        entry
        for entry in plan["entries"]
        if entry.get("action") == "COPY_MASTER_FLAT" and entry.get("filter") == "R"
    ]
    assert len(red_flats) == len(TARGETS)
    assert {entry["sourcePath"] for entry in red_flats} == {str(explicit.resolve())}


def test_prepare_reapplying_identical_plan_is_noop_and_never_overwrites_dest(
    prepare_fixture: PrepareFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "wbpp-ready"
    report = tmp_path / "prepare-report"
    arguments = _base_arguments(prepare_fixture, destination=destination, report=report)

    plan_return_code = _invoke(arguments)
    plan_capture = capsys.readouterr()
    assert plan_return_code == 0, plan_capture.err
    apply_arguments = ["apply-wbpp-plan", str(report / "prepare-plan.json")]
    first_return_code = _invoke(apply_arguments)
    first_capture = capsys.readouterr()
    assert first_return_code == 0, first_capture.err
    before = _snapshot_tree(destination)
    assert before

    second_return_code = _invoke(apply_arguments)
    second_capture = capsys.readouterr()

    assert second_return_code == 0, second_capture.err
    assert _snapshot_tree(destination) == before
    assert any(
        marker in second_capture.out.upper()
        for marker in ("NO_OP", "ALREADY_COMPLETE")
    )


def test_prepare_measurement_failure_stays_unassessable_manifest_only(
    prepare_fixture: PrepareFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tiny = prepare_fixture.input_directory / "night-1" / "R" / "tiny-light.fits"
    hdu = fits.PrimaryHDU(np.ones((8, 8), dtype=np.float32))
    _base_header(
        hdu,
        image_type="LIGHT",
        filter_name="R",
        target="NGC 7000",
        sequence=59,
    )
    hdu.writeto(tiny)
    destination = tmp_path / "wbpp-ready"
    report = tmp_path / "prepare-report"

    return_code = _invoke(
        _base_arguments(prepare_fixture, destination=destination, report=report)
    )

    captured = capsys.readouterr()
    assert return_code == 0, captured.err
    results = _load_json(report / "results.json")
    tiny_result = next(frame for frame in results["frames"] if frame["path"] == str(tiny.resolve()))
    assert tiny_result["decision"] == "UNASSESSABLE"
    assert tiny_result["metadata"]["role"] == "LIGHT"
    assert tiny_result["sourceIdentity"]["sha256"] == _sha256(tiny)
    plan = _load_json(report / "prepare-plan.json")
    tiny_entry = next(entry for entry in plan["entries"] if entry["sourcePath"] == str(tiny.resolve()))
    assert tiny_entry["action"] == "MANIFEST_ONLY"
    assert not destination.exists()


def test_prepare_does_not_measure_or_plan_path_only_light_hint(
    prepare_fixture: PrepareFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hinted = prepare_fixture.input_directory / "night-1" / "R" / "path-only.fits"
    _write_light(
        hinted,
        target="NGC 7000",
        filter_name="R",
        target_index=0,
        sequence=60,
    )
    with fits.open(hinted, mode="update") as hdul:
        del hdul[0].header["IMAGETYP"]
        hdul.flush()
    destination = tmp_path / "wbpp-ready"
    report = tmp_path / "prepare-report"

    return_code = _invoke(
        _base_arguments(prepare_fixture, destination=destination, report=report)
    )

    captured = capsys.readouterr()
    assert return_code == 0, captured.err
    assert str(hinted.resolve()) not in (report / "results.json").read_text(encoding="utf-8")
    assert str(hinted.resolve()) not in (report / "prepare-plan.json").read_text(encoding="utf-8")


def test_apply_existing_plan_rejects_changed_qc_report(
    prepare_fixture: PrepareFixture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "wbpp-ready"
    report = tmp_path / "prepare-report"
    assert _invoke(
        _base_arguments(prepare_fixture, destination=destination, report=report)
    ) == 0
    capsys.readouterr()
    results = report / "results.json"
    results.write_text(results.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    return_code = _invoke(["apply-wbpp-plan", str(report / "prepare-plan.json")])

    captured = capsys.readouterr()
    assert return_code != 0
    assert "QC_REPORT_MISMATCH" in captured.err
    assert not destination.exists()
