from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from dataclasses import replace

from astropy.io import fits
import numpy as np
import pytest

import ufwbpp.astap_backend as astap_backend_module
import ufwbpp.astrometry_net_backend as astrometry_backend_module
from ufwbpp.astap_backend import (
    AstapSolverBackend,
    discover_astap,
    probe_astap,
    verify_execution_receipt,
    verify_solver_execution_result,
)
from ufwbpp.astrometry_net_backend import (
    AstrometryNetSolveProfile,
    AstrometryNetSolverBackend,
    discover_astrometry_config,
    discover_astrometry_net,
    probe_astrometry_net,
)
from ufwbpp.catalogs import verify_catalog
from ufwbpp.solver import SolveRequest, SolverStatus, validate_solver_result


# Upper bounds for the fake solver processes, not expectations: the tests
# assert outcomes. A cold interpreter importing astropy on a hosted Windows
# runner has needed more than 2 s, which turned an expected failure code into
# SOLVER_TIMEOUT. The timeout test sets its own short limit.
FAKE_SOLVER_TIMEOUT_SECONDS = 30.0

@pytest.fixture(autouse=True)
def isolate_fake_solver_catalog_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake solvers use explicit catalog fixtures, never a developer's catalog.

    Keep the directly imported discovery function unchanged for its dedicated
    environment-discovery test. Only backend construction is isolated here.
    """
    monkeypatch.setattr(
        astrometry_backend_module,
        "discover_astrometry_config",
        lambda config_path=None, **kwargs: discover_astrometry_config(
            config_path, environment={}
        ),
    )


FAKE_SOLVER = r'''from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

backend = os.environ["FAKE_BACKEND"]
mode = os.environ.get("FAKE_MODE", "success")
args = sys.argv[1:]

if backend == "astap" and args == ["-help"]:
    if mode == "missing-capability":
        print("ASTAP version 2026.07.30 -f")
    elif mode == "no-sip":
        print("ASTAP version 2026.07.30 -f -o -wcs -ra -spd -fov -r")
    else:
        print("ASTAP version 2026.07.30 -f -o -wcs -ra -spd -fov -r -sip")
    raise SystemExit(0)
if backend == "astrometry" and args == ["--version"]:
    if mode == "version-unsupported":
        print("unrecognized option --version", file=sys.stderr)
        raise SystemExit(2)
    print("astrometry.net version 0.97")
    raise SystemExit(0)
if backend == "astrometry" and args == ["--help"]:
    if mode == "missing-capability":
        print("solve-field --dir --out")
    else:
        print("solve-field --dir --out --wcs --new-fits --solved --corr --match --no-verify --no-plots --config --downsample --objs --depth --pixel-error")
    raise SystemExit(0)

log = os.environ.get("FAKE_ARGV_LOG")
if log:
    Path(log).write_text(json.dumps(args), encoding="utf-8")
if mode == "timeout":
    time.sleep(20)
if mode == "exit":
    print("synthetic failure", file=sys.stderr)
    raise SystemExit(17)
if mode == "log-flood":
    sys.stdout.buffer.write(b"x" * (2 * 1024 * 1024))
    sys.stderr.buffer.write(b"y" * (2 * 1024 * 1024))
if mode == "lingering-child":
    marker = os.environ["FAKE_CHILD_MARKER"]
    trigger = os.environ["FAKE_CHILD_TRIGGER"]
    subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import pathlib,sys,time\n"
                "trigger=pathlib.Path(sys.argv[1])\n"
                "deadline=time.monotonic()+5\n"
                "while not trigger.exists() and time.monotonic()<deadline:\n"
                "    time.sleep(.01)\n"
                "if trigger.exists():\n"
                "    pathlib.Path(sys.argv[2]).write_text('escaped')\n"
            ),
            trigger,
            marker,
        ],
    )
if mode == "self-drift":
    with Path(__file__).open("a", encoding="utf-8") as handle:
        handle.write("\n# changed by fake solver\n")
if mode == "catalog-replace":
    catalog_path = Path(os.environ["FAKE_CATALOG_PATH"])
    replacement = catalog_path.with_suffix(".replacement")
    replacement.write_bytes(catalog_path.read_bytes())
    os.replace(replacement, catalog_path)

from astropy.io import fits
import numpy as np

def value(option: str) -> str:
    return args[args.index(option) + 1]

def solution_header(crval1: float = 270.0):
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRPIX1"] = 20.0
    header["CRPIX2"] = 15.0
    header["CRVAL1"] = crval1
    header["CRVAL2"] = -15.0
    header["CD1_1"] = -0.001
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.001
    return header

if backend == "astap":
    input_path = Path(value("-f"))
    output_base = Path(value("-o"))
    if mode == "stage-drift":
        with input_path.open("ab") as handle:
            handle.write(b"drift")
    wcs_path = output_base.with_suffix(".wcs")
    ini_path = output_base.with_suffix(".ini")
    if mode == "symlink-output":
        external = Path(os.environ["FAKE_EXTERNAL_WCS"])
        solution_header().tofile(external, overwrite=True)
        wcs_path.symlink_to(external)
    elif mode == "corrupt-output":
        wcs_path.write_bytes(b"not a fits header")
    else:
        header = solution_header()
        header_file = os.environ.get("FAKE_WCS_HEADER")
        if header_file:
            # Catalog-verification tests supply the WCS their synthetic
            # master was rendered with.
            header = fits.Header.fromtextfile(header_file)
        header.tofile(wcs_path, overwrite=True)
    solved = "F" if mode == "seed-only" else "T"
    ini_path.write_text(f"PLTSOLVD={solved}\n", encoding="ascii")
    raise SystemExit(0)

from astropy.wcs import WCS

if backend == "astrometry" and mode == "hint-miss" and any(
    option in args for option in ("--ra", "--dec", "--scale-units")
):
    print("synthetic stale hint miss", file=sys.stderr)
    raise SystemExit(1)

input_path = Path(args[-1])
wcs_path = Path(value("--wcs"))
new_path = Path(value("--new-fits"))
solved_path = Path(value("--solved"))
corr_path = Path(value("--corr"))
match_path = Path(value("--match"))
if mode == "stage-drift":
    with input_path.open("ab") as handle:
        handle.write(b"drift")
base_crval = 210.0 if mode == "wrong-sky" else 270.0
solution_header(base_crval).tofile(wcs_path, overwrite=True)
with fits.open(input_path, memmap=False) as hdul:
    data = np.asarray(hdul[0].data)
    if mode == "geometry-drift":
        data = data[:-1, :]
    header = hdul[0].header.copy()
header.update(solution_header(271.0 if mode == "output-drift" else base_crval))
fits.writeto(new_path, data, header, overwrite=False)
count = 8 if mode == "low-matches" else 20
field_x = np.linspace(4.0, 36.0, count)
field_y = np.linspace(4.0, 26.0, count)
residual = 1.0 if mode == "high-rms" else 0.05
catalog_pixels = np.column_stack((field_x + residual, field_y - residual))
catalog_world = WCS(solution_header(base_crval)).celestial.all_pix2world(catalog_pixels, 1)
columns = [
    fits.Column(name="field_x", format="D", array=field_x),
    fits.Column(name="field_y", format="D", array=field_y),
    fits.Column(name="index_ra", format="D", array=catalog_world[:, 0]),
    fits.Column(name="index_dec", format="D", array=catalog_world[:, 1]),
    fits.Column(name="field_id", format="K", array=np.arange(count, dtype=np.int64)),
    fits.Column(name="index_id", format="K", array=np.arange(1000, 1000 + count, dtype=np.int64)),
]
fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(corr_path)
match_columns = [
    fits.Column(name="DIMQUADS", format="B", array=np.asarray([4], dtype=np.uint8)),
    fits.Column(name="CRVAL", format="2D", array=np.asarray([[base_crval, -15.0]])),
    fits.Column(name="CRPIX", format="2D", array=np.asarray([[20.0, 15.0]])),
    fits.Column(name="CD", format="4D", array=np.asarray([[-0.001, 0.0, 0.0, 0.001]])),
    fits.Column(name="WCS_VALID", format="L", array=np.asarray([True])),
    fits.Column(name="INDEXID", format="I", array=np.asarray([4206], dtype=np.int16)),
    fits.Column(name="HEALPIX", format="I", array=np.asarray([17], dtype=np.int16)),
    fits.Column(name="HPNSIDE", format="I", array=np.asarray([32], dtype=np.int16)),
    fits.Column(name="PARITY", format="L", array=np.asarray([mode == "parity-drift"])),
    fits.Column(
        name="NMATCH",
        format="J",
        array=np.asarray(
            [count + (1 if mode == "match-count-drift" else 0)],
            dtype=np.int32,
        ),
    ),
    fits.Column(name="LOGODDS", format="E", array=np.asarray([100.0], dtype=np.float32)),
]
fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(match_columns)]).writeto(match_path)
if mode == "seed-only":
    solved_path.write_bytes(b"\x00")
elif mode == "malicious-marker":
    solved_path.write_bytes(b"\x01junk")
else:
    solved_path.write_bytes(b"\x01")
'''


def write_fake_solver(path: Path) -> Path:
    path.write_text(FAKE_SOLVER, encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o755)
    return path


def write_input(path: Path) -> Path:
    header = fits.Header()
    header["OBJCTRA"] = "18:00:00"
    header["OBJCTDEC"] = "-15:00:00"
    header["CRVAL1"] = 270.0  # Deliberately only a pointing seed, not a WCS.
    header["CRVAL2"] = -15.0
    fits.writeto(path, np.arange(1200, dtype=np.uint16).reshape(30, 40), header)
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def managed_catalog(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    payload = b"fake-managed-index-4206"
    root = tmp_path / "managed-catalog"
    manifests = tmp_path / "managed-manifests"
    root.mkdir()
    manifests.mkdir()
    (root / "index-4206.fits").write_bytes(payload)
    manifest = {
        "schemaVersion": 1,
        "catalogId": "fake-managed-4206",
        "provider": "Fake solver test provider",
        "version": "fixture-v1",
        "redistributionStatus": "user-download-required",
        "license": "fixture only",
        "citation": "fixture citation",
        "providerTerms": {
            "acceptanceId": "fake-managed-terms-v1",
            "url": "https://provider.invalid/terms",
            "summary": "Synthetic test fixture only.",
            "licenseStatus": "confirmed",
            "requiresExplicitAcceptance": True,
        },
        "allowedDownloadOrigins": ["https://provider.invalid"],
        "artifacts": [
            {
                "artifactId": "index-4206.fits",
                "url": "https://provider.invalid/index-4206.fits",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "sizeBytes": len(payload),
                "installScope": "user",
                "scale": 6,
            }
        ],
    }
    (manifests / "fake-managed-4206.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    verified = verify_catalog(
        "fake-managed-4206",
        catalog_root=root,
        manifest_dir=manifests,
        write_configuration=True,
    )
    assert verified["ok"] is True
    return root / "astrometry.cfg", manifests, verified


def fake_star_database(tmp_path: Path) -> Path:
    """ASTAP database files beside nothing in particular; the probe needs them."""

    directory = tmp_path / "astap-database"
    directory.mkdir(exist_ok=True)
    (directory / "d50_0101.1476").write_bytes(b"fake star database area")
    return directory


def backend_for(kind: str, script: Path, tmp_path: Path, mode: str = "success"):
    environment = {"FAKE_BACKEND": kind, "FAKE_MODE": mode}
    common = dict(
        executable=sys.executable,
        executable_args=(str(script),),
        environment=environment,
        staging_root=tmp_path,
        timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        probe_timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
    )
    # These tests exercise the process contract; the managed-catalog gate has
    # its own tests (test_catalog_correspondence.py), so it is relaxed here.
    if kind == "astap":
        return AstapSolverBackend(
            **common, require_managed_catalog=False, star_database_dir=fake_star_database(tmp_path)
        )
    return AstrometryNetSolverBackend(**common, require_managed_catalog=False)


@pytest.mark.parametrize("kind", ["astap", "astrometry"])
def test_probe_finds_version_and_required_capabilities(tmp_path: Path, kind: str) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    environment = {"FAKE_BACKEND": kind, "FAKE_MODE": "success"}
    if kind == "astap":
        assert discover_astap(sys.executable) == str(Path(sys.executable).absolute())
        without_database = probe_astap(
            sys.executable,
            executable_args=(str(script),),
            environment=environment,
        )
        assert without_database.available is True
        assert without_database.execution_ready is False
        assert without_database.error_code == "STAR_DATABASE_MISSING"
        assert without_database.evidence["starDatabase"]["found"] is False
        probe = probe_astap(
            sys.executable,
            executable_args=(str(script),),
            environment=environment,
            star_database_dir=fake_star_database(tmp_path),
        )
        assert probe.version == "2026.07.30"
        assert probe.evidence["starDatabase"] == {
            "found": True,
            "source": "configured",
            "path": str(fake_star_database(tmp_path).absolute()),
            "families": ["d50"],
            "fileCount": 1,
        }
        assert "path" not in probe.serializable()["evidence"]["starDatabase"]
        assert "catalog-correspondence-quality-v1" in probe.capabilities
    else:
        assert discover_astrometry_net(sys.executable) == str(Path(sys.executable).absolute())
        probe = probe_astrometry_net(
            sys.executable,
            executable_args=(str(script),),
            environment=environment,
        )
        assert probe.version == "0.97"
    assert probe.available is True
    assert probe.execution_ready is True
    assert "provenance-receipt-v1" in probe.capabilities


def test_astrometry_probe_allows_upstream_build_without_version_option(tmp_path: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    probe = probe_astrometry_net(
        sys.executable,
        executable_args=(str(script),),
        environment={"FAKE_BACKEND": "astrometry", "FAKE_MODE": "version-unsupported"},
    )

    assert probe.available is True
    assert probe.execution_ready is True
    assert probe.version == "unknown"
    assert probe.evidence["versionProcess"]["exitCode"] == 2


def test_astrometry_profile_is_solve_only_and_uses_exact_bounded_arguments(
    tmp_path: Path,
) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    config = tmp_path / "astrometry.cfg"
    config.write_text("inparallel\n", encoding="ascii")
    argv_log = tmp_path / "profile-argv.json"
    backend = AstrometryNetSolverBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        config_path=config,
        solve_profile=AstrometryNetSolveProfile(downsample=4),
        environment={
            "FAKE_BACKEND": "astrometry",
            "FAKE_MODE": "success",
            "FAKE_ARGV_LOG": str(argv_log),
        },
        staging_root=tmp_path,
        timeout_seconds=20.0,
        probe_timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        require_managed_catalog=False,
    )

    for probe_process in (
        backend.probe.evidence["versionProcess"],
        backend.probe.evidence["helpProcess"],
    ):
        assert "argv" not in probe_process
        assert probe_process["argumentsSha256"].startswith("sha256:")

    result = backend.solve(
        SolveRequest(
            str(source),
            str(tmp_path / "solved.fits"),
            ra_hint_degrees=270.0,
            dec_hint_degrees=-15.0,
            field_of_view_degrees=2.0,
            search_radius_degrees=5.0,
        )
    )

    assert result.status is SolverStatus.SOLVED
    argv = json.loads(argv_log.read_text(encoding="utf-8"))
    for option, value in (
        ("--config", str(config)),
        ("--downsample", "4"),
        ("--objs", "500"),
        ("--depth", "10-500"),
        ("--pixel-error", "2"),
    ):
        assert argv[argv.index(option) + 1] == value
    assert result.evidence["config"]["sha256"].startswith("sha256:")
    assert "path" not in result.evidence["config"]
    assert result.evidence["attempts"][0]["plan"]["timeoutSeconds"] == 5.0


@pytest.mark.skipif(os.name == "nt", reason="the bundled solver runtime is macOS-only")
def test_bundled_solver_reads_fits_directly_without_the_python_type_sniffer(tmp_path: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    bundle = tmp_path / "bundle"
    (bundle / "bin").mkdir(parents=True)
    (bundle / "runtime.json").write_text("{}", encoding="utf-8")
    solver = bundle / "bin" / "solve-field"
    solver.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    solver.chmod(0o755)
    argv_log = tmp_path / "argv.json"
    for bundled in (False, True):
        environment = {"FAKE_BACKEND": "astrometry", "FAKE_MODE": "success", "FAKE_ARGV_LOG": str(argv_log)}
        if bundled:
            environment["UFWBPP_BUNDLED_ASTROMETRY_ROOT"] = str(bundle)
        backend = AstrometryNetSolverBackend(
            executable=str(solver),
            environment=environment,
            staging_root=tmp_path,
            timeout_seconds=20.0,
            probe_timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
            require_managed_catalog=False,
        )
        result = backend.solve(SolveRequest(str(source), str(tmp_path / f"solved-{bundled}.fits")))
        assert result.status is SolverStatus.SOLVED
        assert ("--fits-image" in json.loads(argv_log.read_text(encoding="utf-8"))) is bundled


def test_stale_hints_cannot_lock_astrometry_out_of_unconstrained_fallback(
    tmp_path: Path,
) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    backend = AstrometryNetSolverBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        solve_profile=AstrometryNetSolveProfile(downsample=4),
        environment={"FAKE_BACKEND": "astrometry", "FAKE_MODE": "hint-miss"},
        staging_root=tmp_path,
        timeout_seconds=20.0,
        probe_timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        require_managed_catalog=False,
    )

    result = backend.solve(
        SolveRequest(
            str(source),
            str(tmp_path / "solved.fits"),
            ra_hint_degrees=12.0,
            dec_hint_degrees=55.0,
            field_of_view_degrees=20.0,
            search_radius_degrees=1.0,
        )
    )

    assert result.status is SolverStatus.SOLVED
    assert len(result.evidence["attempts"]) == 2
    first, second = result.evidence["attempts"]
    assert first["failureCode"] == "SOLVER_EXIT_NONZERO"
    assert first["plan"]["includeHints"] is True
    assert second["plan"]["includeHints"] is False
    assert second["backendConfirmed"] is True
    assert "argv" not in second["process"]
    assert sum(item["plan"]["timeoutSeconds"] for item in result.evidence["attempts"]) == 20.0


def test_astrometry_profile_validation_and_adaptive_downsampling() -> None:
    profile = AstrometryNetSolveProfile()
    assert profile.adaptive_downsample((4176, 6252)) == 4
    assert profile.adaptive_downsample((2000, 3000)) == 2
    assert profile.adaptive_downsample((1200, 1600)) == 1
    with pytest.raises(ValueError, match="source_limit"):
        AstrometryNetSolveProfile(source_limit=0).validate()


def test_small_astrometry_budget_keeps_one_hinted_attempt() -> None:
    # A budget too small for two process starts runs one constrained attempt
    # with all of it instead of pretending a fallback was tried. The fake-solver
    # tests use a generous process bound, so the branch is pinned here.
    request = SolveRequest(
        "light.fits",
        "solved.fits",
        ra_hint_degrees=270.0,
        dec_hint_degrees=-15.0,
        field_of_view_degrees=2.0,
    )
    profile = AstrometryNetSolveProfile()

    (only,) = astrometry_backend_module._solve_attempts(profile, request, (30, 40), 2.0)
    assert (only.name, only.include_hints, only.timeout_seconds) == ("adaptive-hinted", True, 2.0)

    hinted, fallback = astrometry_backend_module._solve_attempts(profile, request, (30, 40), 20.0)
    assert (hinted.name, fallback.name) == ("adaptive-hinted", "conservative-unconstrained-fallback")
    assert hinted.timeout_seconds + fallback.timeout_seconds == 20.0


def test_astrometry_config_discovery_uses_app_environment_without_probe_args(
    tmp_path: Path,
) -> None:
    config = tmp_path / "catalog" / "astrometry.cfg"
    config.parent.mkdir()
    config.write_text("inparallel\n", encoding="ascii")
    environment = {"UFWBPP_ASTROMETRY_CONFIG": str(config)}

    assert discover_astrometry_config(environment=environment) == str(config.absolute())
    assert discover_astrometry_config(
        environment={"HOME": str(tmp_path / "empty-home")}
    ) is None


def test_strict_astrometry_binds_match_index_to_managed_bytes(tmp_path: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    config, manifests, installed = managed_catalog(tmp_path)
    backend = AstrometryNetSolverBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        config_path=config,
        catalog_manifest_dir=manifests,
        environment={"FAKE_BACKEND": "astrometry", "FAKE_MODE": "success"},
        staging_root=tmp_path,
        timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        probe_timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
    )

    result = backend.solve(SolveRequest(str(source), str(tmp_path / "managed-solved.fits")))

    assert result.status is SolverStatus.SOLVED
    assert result.astrometric_quality is not None
    quality = result.astrometric_quality
    assert quality.catalog_managed is True
    assert quality.installed_set_identity == installed["installedSet"]["installedSetIdentity"]
    assert quality.catalog_manifest_sha256 == installed["manifestSha256"]
    assert [item.index_id for item in quality.index_artifacts] == ["4206"]
    assert quality.index_artifacts[0].sha256 == hashlib.sha256(
        b"fake-managed-index-4206"
    ).hexdigest()
    assert validate_solver_result(
        result,
        require_scientific_evidence=True,
        min_matches=12,
        max_rms_arcsec=2.0,
    ).valid
    serialized = json.dumps(result.evidence, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert '"argv"' not in serialized
    assert "stdoutTail" not in serialized
    assert "stderrTail" not in serialized


def test_strict_astrometry_rejects_unmanaged_and_replaced_indexes(tmp_path: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    unmanaged = AstrometryNetSolverBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        environment={"FAKE_BACKEND": "astrometry", "FAKE_MODE": "success"},
        staging_root=tmp_path,
        timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        probe_timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
    )
    rejected = unmanaged.solve(
        SolveRequest(str(source), str(tmp_path / "unmanaged-solved.fits"))
    )
    assert rejected.status is SolverStatus.FAILED
    assert rejected.evidence["failureCode"] == "CATALOG_CONFIG_UNMANAGED"
    assert not (tmp_path / "unmanaged-solved.fits").exists()

    config, manifests, _ = managed_catalog(tmp_path)
    catalog_path = config.parent / "index-4206.fits"
    replaced = AstrometryNetSolverBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        config_path=config,
        catalog_manifest_dir=manifests,
        environment={
            "FAKE_BACKEND": "astrometry",
            "FAKE_MODE": "catalog-replace",
            "FAKE_CATALOG_PATH": str(catalog_path),
        },
        staging_root=tmp_path,
        timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        probe_timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
    ).solve(SolveRequest(str(source), str(tmp_path / "replaced-solved.fits")))
    assert replaced.status is SolverStatus.FAILED
    assert replaced.evidence["failureCode"] == "CATALOG_FILE_CHANGED"
    assert not (tmp_path / "replaced-solved.fits").exists()


def test_unmanaged_astrometry_result_is_diagnostic_not_final(tmp_path: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    result = backend_for("astrometry", script, tmp_path).solve(
        SolveRequest(str(source), str(tmp_path / "diagnostic-solved.fits"))
    )
    assert result.status is SolverStatus.SOLVED
    assert result.astrometric_quality is not None
    assert result.astrometric_quality.catalog_managed is False
    validation = validate_solver_result(
        result,
        require_scientific_evidence=True,
        min_matches=12,
        max_rms_arcsec=2.0,
    )
    assert validation.code == "SOLVER_MANAGED_CATALOG_REQUIRED"


def test_large_solver_logs_are_streamed_with_a_bounded_tail(tmp_path: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    backend = backend_for("astap", script, tmp_path, "log-flood")

    result = backend.solve(SolveRequest(str(source), str(tmp_path / "solved.fits")))

    assert result.status == SolverStatus.SOLVED
    process = result.evidence["process"]
    assert process["stdoutBytes"] == 2 * 1024 * 1024
    assert process["stderrBytes"] == 2 * 1024 * 1024
    assert "stdoutTail" not in process
    assert "stderrTail" not in process
    assert process["stdoutSha256"].startswith("sha256:")
    assert process["stderrSha256"].startswith("sha256:")


def test_optional_local_solver_logs_are_separate_and_path_redacted(tmp_path: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    log_root = tmp_path / "private-diagnostics"
    backend = AstapSolverBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        environment={"FAKE_BACKEND": "astap", "FAKE_MODE": "success"},
        staging_root=tmp_path,
        diagnostic_log_root=log_root,
        timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        probe_timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        require_managed_catalog=False,
        star_database_dir=fake_star_database(tmp_path),
    )
    result = backend.solve(SolveRequest(str(source), str(tmp_path / "solved.fits")))

    assert result.status is SolverStatus.SOLVED
    logs = sorted(log_root.glob("*.json"))
    assert len(logs) >= 2  # capability probe plus solve
    for path in logs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rendered = json.dumps(payload, sort_keys=True)
        assert str(tmp_path) not in rendered
        assert "stdoutTail" in payload and "stderrTail" in payload
        assert "argv" in payload
    shareable = json.dumps(result.evidence, sort_keys=True)
    assert "stdoutTail" not in shareable
    assert "stderrTail" not in shareable
    assert '"argv"' not in shareable


def test_posix_process_group_cannot_leave_a_mutating_child(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("POSIX process-group behavior")
    import time

    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    escaped_marker = tmp_path / "escaped.txt"
    child_trigger = tmp_path / "allow-child-mutation"
    backend = AstapSolverBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        environment={
            "FAKE_BACKEND": "astap",
            "FAKE_MODE": "lingering-child",
            "FAKE_CHILD_MARKER": str(escaped_marker),
            "FAKE_CHILD_TRIGGER": str(child_trigger),
        },
        staging_root=tmp_path,
        timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        require_managed_catalog=False,
        star_database_dir=fake_star_database(tmp_path),
    )

    result = backend.solve(SolveRequest(str(source), str(tmp_path / "solved.fits")))
    child_trigger.write_text("go", encoding="ascii")
    time.sleep(0.4)

    assert result.status == SolverStatus.SOLVED
    assert not escaped_marker.exists()


@pytest.mark.parametrize("kind", ["astap", "astrometry"])
def test_success_isolated_input_and_verifiable_receipt(tmp_path: Path, kind: str) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light;not-a-shell-command.fits")
    original = digest(source)
    argv_log = tmp_path / "argv.json"
    environment = {
        "FAKE_BACKEND": kind,
        "FAKE_MODE": "success",
        "FAKE_ARGV_LOG": str(argv_log),
    }
    common = dict(
        executable=sys.executable,
        executable_args=(str(script),),
        environment=environment,
        staging_root=tmp_path,
        timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        probe_timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
    )
    backend = (
        AstapSolverBackend(
            **common, require_managed_catalog=False, star_database_dir=fake_star_database(tmp_path)
        )
        if kind == "astap"
        else AstrometryNetSolverBackend(**common, require_managed_catalog=False)
    )
    output = tmp_path / "result" / "solved.fits"
    result = backend.solve(
        SolveRequest(
            str(source),
            str(output),
            ra_hint_degrees=270.0,
            dec_hint_degrees=-15.0,
            field_of_view_degrees=2.0,
            search_radius_degrees=5.0,
        )
    )

    assert result.status == SolverStatus.SOLVED
    assert validate_solver_result(result).valid is True
    assert verify_execution_receipt(result.evidence) is True
    assert verify_solver_execution_result(result) is True
    assert output.is_file()
    assert digest(source) == original
    argv = json.loads(argv_log.read_text(encoding="utf-8"))
    assert str(source) not in argv
    assert "-update" not in argv
    if kind == "astap":
        # ASTAP's -fov is the field height: the width hint (2.0 deg) scaled by
        # the 30x40 input's aspect ratio (29/39), RA in hours, Dec as SPD.
        assert argv[argv.index("-fov") + 1] == f"{2.0 * 29 / 39:.12g}"
        assert argv[argv.index("-ra") + 1] == f"{270.0 / 15.0:.12g}"
        assert argv[argv.index("-spd") + 1] == f"{75.0:.12g}"
    assert result.evidence["process"]["shell"] is False
    assert result.evidence["source"]["sha256"] == f"sha256:{original}"
    assert result.evidence["outputs"]["published"]["sha256"].startswith("sha256:")
    if kind == "astrometry":
        assert result.astrometric_quality is not None
        assert result.astrometric_quality.matched_stars == 20
        assert result.evidence["astrometricQuality"] == result.astrometric_quality.serializable()
        assert result.evidence["outputs"]["corr"]["sha256"] == (
            "sha256:" + result.astrometric_quality.correspondence_sha256
        )
        assert result.astrometric_quality.index_identities == (
            "astrometry.net:index:4206:healpix:17:hpnside:32",
        )
        assert result.evidence["outputs"]["match"]["sha256"].startswith("sha256:")
        assert validate_solver_result(
            result,
            require_scientific_evidence=True,
            min_matches=12,
            max_rms_arcsec=2.0,
            require_managed_catalog=False,
        ).valid
        assert validate_solver_result(
            result,
            require_scientific_evidence=True,
            min_matches=12,
            max_rms_arcsec=2.0,
        ).code == "SOLVER_MANAGED_CATALOG_REQUIRED"
        drifted_quality = replace(
            result.astrometric_quality,
            matched_stars=result.astrometric_quality.matched_stars + 1,
        )
        assert verify_solver_execution_result(
            replace(result, astrometric_quality=drifted_quality)
        ) is False
    else:
        # Without a managed index set the ASTAP adapter cannot recompute
        # correspondences, so its solution stays diagnostic only.
        assert result.evidence["adapterVersion"] == "astap-process-v4"
        assert result.astrometric_quality is None
        assert result.evidence["astrometricQuality"]["status"] == "UNAVAILABLE"
        assert result.evidence["catalogPreflight"] == {"managed": False, "reason": "NO_MANAGED_CONFIG"}
        assert "-d" in argv and argv[argv.index("-d") + 1] == str(fake_star_database(tmp_path))
        assert "-sip" in argv
        assert result.evidence["solveOptions"] == {"sipRequested": True, "starDatabaseDirectoryConfigured": True}
        assert validate_solver_result(
            result,
            require_scientific_evidence=True,
            min_matches=12,
            max_rms_arcsec=2.0,
        ).code == "SOLVER_QUALITY_EVIDENCE_MISSING"
    tampered = dict(result.evidence)
    tampered["outcome"] = "FAILED"
    assert verify_execution_receipt(tampered) is False
    tampered_header = result.header.copy()
    tampered_header["CRVAL1"] = float(tampered_header["CRVAL1"]) + 0.1
    assert verify_solver_execution_result(replace(result, header=tampered_header)) is False


@pytest.mark.parametrize(
    ("mode", "code"),
    (
        ("low-matches", "SOLVER_MATCH_COUNT_BELOW_MINIMUM"),
        ("high-rms", "SOLVER_RMS_ABOVE_MAXIMUM"),
    ),
)
def test_astrometry_correspondence_quality_thresholds_fail_closed(
    tmp_path: Path,
    mode: str,
    code: str,
) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    backend = backend_for("astrometry", script, tmp_path, mode)

    result = backend.solve(SolveRequest(str(source), str(tmp_path / "solved.fits")))

    assert result.status is SolverStatus.SOLVED
    assert verify_solver_execution_result(result)
    validation = validate_solver_result(
        result,
        require_scientific_evidence=True,
        min_matches=12,
        max_rms_arcsec=2.0,
        require_managed_catalog=False,
    )
    assert validation.valid is False
    assert validation.code == code


@pytest.mark.parametrize("kind", ["astap", "astrometry"])
@pytest.mark.parametrize(
    ("mode", "code"),
    (("exit", "SOLVER_EXIT_NONZERO"), ("timeout", "SOLVER_TIMEOUT")),
)
def test_exit_and_timeout_fail_closed(tmp_path: Path, kind: str, mode: str, code: str) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    backend = backend_for(kind, script, tmp_path, mode)
    backend.timeout_seconds = 0.15 if mode == "timeout" else FAKE_SOLVER_TIMEOUT_SECONDS
    output = tmp_path / "solved.fits"

    result = backend.solve(SolveRequest(str(source), str(output)))

    assert result.status == SolverStatus.FAILED
    assert result.evidence["failureCode"] == code
    assert result.backend_confirmed is False
    assert not output.exists()
    assert verify_execution_receipt(result.evidence) is True


@pytest.mark.parametrize("kind", ["astap", "astrometry"])
def test_valid_looking_seed_without_backend_marker_is_rejected(tmp_path: Path, kind: str) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    backend = backend_for(kind, script, tmp_path, "seed-only")

    result = backend.solve(SolveRequest(str(source), str(tmp_path / "solved.fits")))

    assert result.status == SolverStatus.FAILED
    assert result.evidence["failureCode"] == "BACKEND_CONFIRMATION_MISSING"
    assert result.backend_confirmed is False


@pytest.mark.parametrize(
    ("mode", "code"),
    (
        ("output-drift", "SOLVER_OUTPUT_DRIFT"),
        ("geometry-drift", "OUTPUT_GEOMETRY_DRIFT"),
        ("malicious-marker", "BACKEND_CONFIRMATION_MISSING"),
        ("match-count-drift", "CORRESPONDENCE_MATCH_COUNT_MISMATCH"),
        ("parity-drift", "MATCH_PARITY_MISMATCH"),
    ),
)
def test_astrometry_output_drift_and_malicious_marker_fail_closed(
    tmp_path: Path, mode: str, code: str
) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    backend = backend_for("astrometry", script, tmp_path, mode)

    result = backend.solve(SolveRequest(str(source), str(tmp_path / "solved.fits")))

    assert result.status == SolverStatus.FAILED
    assert result.evidence["failureCode"] == code
    assert not (tmp_path / "solved.fits").exists()


@pytest.mark.parametrize("kind", ["astap", "astrometry"])
def test_staged_input_or_executable_drift_fails_closed(tmp_path: Path, kind: str) -> None:
    source = write_input(tmp_path / "light.fits")
    for mode, code in (("stage-drift", "STAGED_INPUT_DRIFT"), ("self-drift", "EXECUTABLE_DRIFT")):
        script = write_fake_solver(tmp_path / f"fake_{mode}.py")
        backend = backend_for(kind, script, tmp_path, mode)
        result = backend.solve(SolveRequest(str(source), str(tmp_path / f"{mode}.fits")))
        assert result.status == SolverStatus.FAILED
        assert result.evidence["failureCode"] == code
        assert not (tmp_path / f"{mode}.fits").exists()


def test_astap_sip_is_requested_only_when_the_cli_advertises_it(tmp_path: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    argv_log = tmp_path / "argv.json"
    older_cli = AstapSolverBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        environment={"FAKE_BACKEND": "astap", "FAKE_MODE": "no-sip", "FAKE_ARGV_LOG": str(argv_log)},
        staging_root=tmp_path,
        timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        probe_timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        require_managed_catalog=False,
        star_database_dir=fake_star_database(tmp_path),
    )
    assert older_cli.descriptor.execution_ready is True
    assert older_cli.sip_requested is False
    assert older_cli.descriptor.metadata["solveOptions"] == {
        "sipPolynomial": True,
        "sipRequested": False,
        "starDatabaseDirectoryConfigured": True,
    }
    result = older_cli.solve(SolveRequest(str(source), str(tmp_path / "no-sip.fits")))
    assert result.status is SolverStatus.SOLVED
    assert "-sip" not in json.loads(argv_log.read_text(encoding="utf-8"))
    assert result.evidence["solveOptions"]["sipRequested"] is False

    disabled = AstapSolverBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        environment={"FAKE_BACKEND": "astap", "FAKE_MODE": "success", "FAKE_ARGV_LOG": str(argv_log)},
        staging_root=tmp_path,
        timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        probe_timeout_seconds=FAKE_SOLVER_TIMEOUT_SECONDS,
        require_managed_catalog=False,
        star_database_dir=fake_star_database(tmp_path),
        sip_polynomial=False,
    )
    assert "-sip" in disabled.probe.evidence["supportedOptionalOptions"]
    assert disabled.sip_requested is False
    result = disabled.solve(SolveRequest(str(source), str(tmp_path / "sip-disabled.fits")))
    assert result.status is SolverStatus.SOLVED
    assert "-sip" not in json.loads(argv_log.read_text(encoding="utf-8"))


def test_astap_rejects_symlinked_solver_artifact(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("ordinary Windows users may not have symlink privileges")
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    external = tmp_path / "outside.wcs"
    backend = AstapSolverBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        environment={
            "FAKE_BACKEND": "astap",
            "FAKE_MODE": "symlink-output",
            "FAKE_EXTERNAL_WCS": str(external),
        },
        staging_root=tmp_path,
        require_managed_catalog=False,
        star_database_dir=fake_star_database(tmp_path),
    )

    result = backend.solve(SolveRequest(str(source), str(tmp_path / "solved.fits")))

    assert result.status == SolverStatus.FAILED
    assert result.evidence["failureCode"] == "ARTIFACT_NOT_REGULAR"


@pytest.mark.parametrize("kind", ["astap", "astrometry"])
def test_missing_probe_capability_disables_execution(tmp_path: Path, kind: str) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    backend = backend_for(kind, script, tmp_path, "missing-capability")
    source = write_input(tmp_path / "light.fits")

    assert backend.descriptor.available is True
    assert backend.descriptor.execution_ready is False
    result = backend.solve(SolveRequest(str(source), str(tmp_path / "solved.fits")))
    assert result.status == SolverStatus.UNAVAILABLE
    assert result.evidence["failureCode"] == "CAPABILITY_PROBE_FAILED"


@pytest.mark.parametrize("kind", ["astap", "astrometry"])
def test_adapter_never_replaces_existing_output(tmp_path: Path, kind: str) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    output = tmp_path / "solved.fits"
    output.write_bytes(b"owner data")
    backend = backend_for(kind, script, tmp_path)

    result = backend.solve(SolveRequest(str(source), str(output)))

    assert result.status == SolverStatus.FAILED
    assert result.evidence["failureCode"] == "OUTPUT_EXISTS"
    assert output.read_bytes() == b"owner data"


def test_failed_publication_rolls_back_only_its_new_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_input(tmp_path / "light.fits")
    output = tmp_path / "solved.fits"
    backend = backend_for("astap", script, tmp_path)

    def fail_sync(path: Path) -> None:
        raise astap_backend_module.SolverExecutionError(
            "DIRECTORY_SYNC_FAILED", f"synthetic failure for {path}"
        )

    from ufwbpp.solvers import process
    monkeypatch.setattr(process, "fsync_directory", fail_sync)
    result = backend.solve(SolveRequest(str(source), str(output)))

    assert result.status == SolverStatus.FAILED
    assert result.evidence["failureCode"] == "DIRECTORY_SYNC_FAILED"
    assert not output.exists()


@pytest.mark.skipif(
    os.environ.get("UFWBPP_RUN_REAL_ASTROMETRY") != "1",
    reason="opt-in test requires a local real FITS fixture and Astrometry.net indexes",
)
def test_real_astrometry_fixture_passes_public_adapter_quality_gate(tmp_path: Path) -> None:
    fixture_value = os.environ.get("UFWBPP_REAL_ASTROMETRY_FIXTURE")
    config_value = os.environ.get("UFWBPP_ASTROMETRY_CONFIG")
    if not fixture_value or not config_value:
        pytest.skip("set UFWBPP_REAL_ASTROMETRY_FIXTURE and UFWBPP_ASTROMETRY_CONFIG")
    fixture = Path(fixture_value)
    config = Path(config_value)
    if not fixture.is_file() or not config.is_file():
        pytest.skip("the configured real fixture or catalog config is unavailable")

    backend = AstrometryNetSolverBackend(config_path=config, timeout_seconds=120.0)
    result = backend.solve(
        SolveRequest(
            str(fixture),
            str(tmp_path / "real-solved.fits"),
            ra_hint_degrees=282.1473,
            dec_hint_degrees=-7.4011,
            field_of_view_degrees=1.11,
            search_radius_degrees=3.0,
        )
    )

    assert result.status is SolverStatus.SOLVED
    assert result.evidence["solveProfile"]["adaptiveDownsample"] == 4
    assert verify_execution_receipt(result.evidence)
    assert verify_solver_execution_result(result)
    assert validate_solver_result(
        result,
        require_scientific_evidence=True,
        min_matches=12,
        max_rms_arcsec=2.0,
    ).valid


def test_solver_process_path_starts_with_the_solver_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A GUI launched from the Finder inherits launchd's minimal ``PATH``;
    ``solve-field`` still has to find its helpers (``pnmfile``, ``image2pnm``,
    ``astrometry-engine``) next to itself, through a symlink or not."""

    real_dir = tmp_path / "Cellar" / "astrometry-net" / "bin"
    real_dir.mkdir(parents=True)
    real = real_dir / "solve-field"
    real.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    real.chmod(0o755)
    link_dir = tmp_path / "bin"
    link_dir.mkdir()
    link = link_dir / "solve-field"
    link.symlink_to(real)
    monkeypatch.setenv("PATH", f"/usr/bin{os.pathsep}/bin{os.pathsep}{link_dir}")

    runtime = astap_backend_module.SolverProcessRuntime(str(link), environment={"FAKE": "1"})
    entries = runtime.environment["PATH"].split(os.pathsep)
    assert entries[:2] == [str(link_dir.absolute()), str(real_dir.resolve())]
    # The inherited entries follow, once each; the solver's own directory is not repeated.
    assert entries[2:] == ["/usr/bin", "/bin"]
    assert runtime.environment["FAKE"] == "1" and runtime.environment["LC_ALL"] == "C"

    direct = astap_backend_module.SolverProcessRuntime(str(real), environment=None)
    assert direct.environment["PATH"].split(os.pathsep)[:3] == [str(real_dir.resolve()), "/usr/bin", "/bin"]
