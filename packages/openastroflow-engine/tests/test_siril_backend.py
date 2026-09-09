from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

from astropy.io import fits
import numpy as np
import pytest

from openastroflow_engine.siril_backend import (
    SirilBackend,
    SirilWorkflowRequest,
    build_siril_script,
    discover_siril_cli,
    probe_siril_cli,
    verify_siril_receipt,
)
from openastroflow_engine.solver import validate_wcs_header


FAKE_SIRIL = r'''from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time

args = sys.argv[1:]
mode = os.environ.get("FAKE_SIRIL_MODE", "success")
version = os.environ.get("FAKE_SIRIL_VERSION", "1.4.4")

if args == ["--version"]:
    print(f"Siril {version}")
    raise SystemExit(0)
if args == ["--help"]:
    if mode == "missing-cli-option":
        print("siril-cli --directory -d")
    else:
        print("siril-cli -d, --directory DIRECTORY -s, --script SCRIPT")
    raise SystemExit(0)

argv_log = os.environ.get("FAKE_SIRIL_ARGV_LOG")
if argv_log:
    Path(argv_log).write_text(json.dumps(args), encoding="utf-8")

def value(option: str) -> str:
    return args[args.index(option) + 1]

work = Path(value("-d"))
script_path = Path(value("-s"))
script = script_path.read_text(encoding="utf-8")
script_log = os.environ.get("FAKE_SIRIL_SCRIPT_LOG")
if script_log:
    Path(script_log).write_text(script, encoding="utf-8")

if mode == "timeout":
    time.sleep(20)
if mode == "exit":
    print("Error in line 9 ('register'): synthetic failure", file=sys.stderr)
    raise SystemExit(17)
if mode == "log-failure":
    print("Script execution failed.")
    raise SystemExit(0)
if mode == "no-confirmation":
    print("processing complete")

if mode == "stage-drift":
    staged = next((work / "light").glob("*.fits"))
    staged.chmod(0o600)
    with staged.open("ab") as stream:
        stream.write(b"drift")

from astropy.io import fits
import numpy as np

header = fits.Header()
header["OBJECT"] = "synthetic"
is_solve = "platesolve" in script
if is_solve and mode != "seed-only":
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRPIX1"] = 4.0
    header["CRPIX2"] = 3.0
    header["CRVAL1"] = 270.0
    header["CRVAL2"] = -15.0
    header["CD1_1"] = -0.001
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.001
elif is_solve:
    # Pointing seeds deliberately look coordinate-like but are not a WCS.
    header["CRVAL1"] = 270.0
    header["CRVAL2"] = -15.0

destination = work / "output" / ("solved.fit" if is_solve else "master.fit")
if mode == "symlink-output":
    external = Path(os.environ["FAKE_SIRIL_EXTERNAL"])
    fits.writeto(external, np.ones((6, 8), dtype=np.float32), header)
    destination.symlink_to(external)
else:
    fits.writeto(destination, np.ones((6, 8), dtype=np.float32), header)

if mode != "no-confirmation":
    print("Script execution finished successfully.")
'''


def write_fake_siril(path: Path) -> Path:
    path.write_text(FAKE_SIRIL, encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o755)
    return path


def write_frame(path: Path, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["IMAGETYP"] = "LIGHT"
    header["CRVAL1"] = 270.0  # A seed, intentionally not a complete WCS.
    header["CRVAL2"] = -15.0
    fits.writeto(path, np.full((6, 8), value, dtype=np.float32), header)
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames(tmp_path: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for role_index, role in enumerate(("light", "flat", "dark", "bias"), start=1):
        result[role] = tuple(
            str(write_frame(tmp_path / "source" / role / f"{role}-{index}.fits", role_index + index))
            for index in range(2)
        )
    return result


def backend(
    tmp_path: Path,
    script: Path,
    mode: str = "success",
    **environment: str,
) -> SirilBackend:
    return SirilBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        environment={"FAKE_SIRIL_MODE": mode, **environment},
        staging_root=tmp_path / "staging",
        timeout_seconds=2.0,
        probe_timeout_seconds=2.0,
    )


def request(tmp_path: Path, source: dict[str, tuple[str, ...]], **changes: object) -> SirilWorkflowRequest:
    values: dict[str, object] = {
        "light_files": source["light"],
        "flat_files": source["flat"],
        "dark_files": source["dark"],
        "bias_files": source["bias"],
        "output_path": str(tmp_path / "result" / "master.fits"),
        "workers": 3,
    }
    values.update(changes)
    return SirilWorkflowRequest(**values)  # type: ignore[arg-type]


def test_probe_is_strictly_certified_for_1_4_4(tmp_path: Path) -> None:
    fake = write_fake_siril(tmp_path / "fake_siril.py")

    assert discover_siril_cli(sys.executable) == str(Path(sys.executable).absolute())
    ready = probe_siril_cli(
        sys.executable,
        executable_args=(str(fake),),
        environment={"FAKE_SIRIL_VERSION": "1.4.4"},
    )
    unsupported = probe_siril_cli(
        sys.executable,
        executable_args=(str(fake),),
        environment={"FAKE_SIRIL_VERSION": "1.4.3"},
    )

    assert ready.available is True
    assert ready.execution_ready is True
    assert ready.version == "1.4.4"
    assert "drizzle-hst-registration" in ready.capabilities
    assert "plate-solve-local-astrometry-net" in ready.capabilities
    assert unsupported.available is True
    assert unsupported.execution_ready is False
    assert unsupported.error_code == "VERSION_UNSUPPORTED"
    assert unsupported.capabilities == ()


def test_probe_missing_script_switch_is_not_execution_ready(tmp_path: Path) -> None:
    fake = write_fake_siril(tmp_path / "fake_siril.py")
    probe = probe_siril_cli(
        sys.executable,
        executable_args=(str(fake),),
        environment={"FAKE_SIRIL_MODE": "missing-cli-option"},
    )

    assert probe.available is True
    assert probe.execution_ready is False
    assert probe.error_code == "CAPABILITY_PROBE_FAILED"
    assert probe.capabilities == ()


def test_script_covers_all_calibration_roles_registration_and_integration(tmp_path: Path) -> None:
    source = frames(tmp_path)
    script = build_siril_script(request(tmp_path, source))

    assert "requires 1.4.4 1.4.5" in script
    assert "stack bias rej w 3 3 -nonorm" in script
    assert "calibrate flat -bias=../masters/master_bias" in script
    assert "stack pp_flat rej w 3 3 -norm=mul" in script
    assert "stack dark rej w 3 3 -nonorm" in script
    assert "calibrate light -dark=../masters/master_dark -flat=../masters/master_flat" in script
    assert "register pp_light" in script
    assert "stack r_pp_light rej w 3 3 -norm=addscale" in script


def test_success_uses_fixed_stage_names_no_shell_and_identity_bound_receipt(tmp_path: Path) -> None:
    fake = write_fake_siril(tmp_path / "fake_siril.py")
    source = frames(tmp_path)
    injected = Path(source["light"][0]).with_name(
        "light & platesolve -force ; touch OWNED.fits"
    )
    Path(source["light"][0]).rename(injected)
    source["light"] = (str(injected), source["light"][1])
    before = {path: digest(Path(path)) for values in source.values() for path in values}
    argv_log = tmp_path / "argv.json"
    script_log = tmp_path / "script.ssf"
    worker = backend(
        tmp_path,
        fake,
        FAKE_SIRIL_ARGV_LOG=str(argv_log),
        FAKE_SIRIL_SCRIPT_LOG=str(script_log),
    )

    result = worker.run(request(tmp_path, source))

    assert result.success is True
    assert result.code == "SIRIL_WORKFLOW_SUCCEEDED"
    output = Path(result.output_path or "")
    assert output.is_file()
    assert Path(result.receipt_path or "").is_file()
    assert verify_siril_receipt(result.receipt) is True
    assert result.receipt["process"]["shell"] is False
    assert result.receipt["script"]["content"] == script_log.read_text(encoding="utf-8")
    assert "touch OWNED" not in result.receipt["script"]["content"]
    assert "platesolve" not in result.receipt["script"]["content"]
    assert not (tmp_path / "OWNED").exists()
    argv = json.loads(argv_log.read_text(encoding="utf-8"))
    assert str(injected) not in argv
    assert all(digest(Path(path)) == value for path, value in before.items())
    tampered = dict(result.receipt)
    tampered["code"] = "FORGED"
    assert verify_siril_receipt(tampered) is False


def test_drizzle_and_plate_solve_use_only_documented_1_4_4_syntax_and_validate_wcs(tmp_path: Path) -> None:
    fake = write_fake_siril(tmp_path / "fake_siril.py")
    source = frames(tmp_path)
    script_log = tmp_path / "script.ssf"
    worker = backend(tmp_path, fake, FAKE_SIRIL_SCRIPT_LOG=str(script_log))
    workflow = request(
        tmp_path,
        source,
        drizzle_scale=2,
        drizzle_pixfrac=0.8,
        drizzle_kernel="square",
        plate_solve=True,
        ra_hint_degrees=270.0,
        dec_hint_degrees=-15.0,
        focal_length_mm=500.0,
        pixel_size_microns=3.76,
        search_radius_degrees=5.0,
    )

    result = worker.run(workflow)

    assert result.success is True
    script = script_log.read_text(encoding="utf-8")
    assert "register pp_light -drizzle -scale=2 -pixfrac=0.8 -kernel=square" in script
    assert (
        "platesolve -force -noflip -localasnet 270,-15 -focal=500 -pixelsize=3.76 -radius=5"
        in script
    )
    with fits.open(result.output_path or "", memmap=False) as hdul:
        validation = validate_wcs_header(hdul[0].header, image_shape=hdul[0].data.shape[-2:])
    assert validation.valid is True
    assert result.receipt["wcsValidation"]["valid"] is True


def test_seed_coordinates_without_a_real_wcs_are_not_success(tmp_path: Path) -> None:
    fake = write_fake_siril(tmp_path / "fake_siril.py")
    source = frames(tmp_path)
    worker = backend(tmp_path, fake, "seed-only")
    workflow = request(tmp_path, source, plate_solve=True)

    result = worker.run(workflow)

    assert result.success is False
    assert result.code == "WCS_VALIDATION_FAILED"
    assert not Path(workflow.output_path).exists()


@pytest.mark.parametrize(
    ("mode", "code"),
    (
        ("exit", "SIRIL_EXIT_NONZERO"),
        ("log-failure", "SIRIL_LOG_FAILURE"),
        ("no-confirmation", "SIRIL_CONFIRMATION_MISSING"),
    ),
)
def test_process_or_log_failure_never_publishes(
    tmp_path: Path, mode: str, code: str
) -> None:
    fake = write_fake_siril(tmp_path / "fake_siril.py")
    source = frames(tmp_path)
    worker = backend(tmp_path, fake, mode)
    workflow = request(tmp_path, source)

    result = worker.run(workflow)

    assert result.success is False
    assert result.code == code
    assert not Path(workflow.output_path).exists()


def test_timeout_kills_worker_and_cleans_private_work_directory(tmp_path: Path) -> None:
    fake = write_fake_siril(tmp_path / "fake_siril.py")
    source = frames(tmp_path)
    staging = tmp_path / "staging"
    worker = backend(tmp_path, fake, "timeout")
    worker.timeout_seconds = 0.1
    workflow = request(tmp_path, source)

    result = worker.run(workflow)

    assert result.success is False
    assert result.code == "SIRIL_TIMEOUT"
    assert not Path(workflow.output_path).exists()
    assert staging.is_dir()
    assert list(staging.iterdir()) == []


def test_missing_light_and_existing_output_fail_before_process(tmp_path: Path) -> None:
    fake = write_fake_siril(tmp_path / "fake_siril.py")
    worker = backend(tmp_path, fake)
    missing = SirilWorkflowRequest((), str(tmp_path / "missing.fits"))

    missing_result = worker.run(missing)
    assert missing_result.success is False
    assert missing_result.code == "LIGHT_INPUT_MISSING"

    source = frames(tmp_path)
    workflow = request(tmp_path, source)
    output = Path(workflow.output_path)
    output.parent.mkdir(parents=True)
    output.write_bytes(b"do not replace")
    existing_result = worker.run(workflow)
    assert existing_result.success is False
    assert existing_result.code == "OUTPUT_EXISTS"
    assert output.read_bytes() == b"do not replace"


def test_source_and_siril_output_symlinks_are_rejected(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("symlink creation is privilege-dependent on Windows")
    fake = write_fake_siril(tmp_path / "fake_siril.py")
    source = frames(tmp_path)
    real = Path(source["light"][0])
    link = real.with_name("linked-light.fits")
    link.symlink_to(real)
    linked_source = dict(source)
    linked_source["light"] = (str(link), source["light"][1])
    worker = backend(tmp_path, fake)

    linked_result = worker.run(request(tmp_path, linked_source))
    assert linked_result.success is False
    assert linked_result.code == "INPUT_SYMLINK_FORBIDDEN"

    external = tmp_path / "external.fits"
    malicious = backend(
        tmp_path,
        fake,
        "symlink-output",
        FAKE_SIRIL_EXTERNAL=str(external),
    )
    workflow = request(tmp_path, source)
    malicious_result = malicious.run(workflow)
    assert malicious_result.success is False
    assert malicious_result.code == "ARTIFACT_NOT_REGULAR"
    assert external.is_file()
    assert not Path(workflow.output_path).exists()


def test_staged_input_mutation_is_detected_and_originals_remain_unchanged(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("fake chmod behavior is POSIX-specific")
    fake = write_fake_siril(tmp_path / "fake_siril.py")
    source = frames(tmp_path)
    originals = {path: digest(Path(path)) for values in source.values() for path in values}
    worker = backend(tmp_path, fake, "stage-drift")
    workflow = request(tmp_path, source)

    result = worker.run(workflow)

    assert result.success is False
    assert result.code == "STAGED_INPUT_DRIFT"
    assert all(digest(Path(path)) == value for path, value in originals.items())
    assert not Path(workflow.output_path).exists()


def test_invalid_options_never_become_script_tokens(tmp_path: Path) -> None:
    source = frames(tmp_path)
    base = request(tmp_path, source)

    for invalid in (
        replace(base, drizzle_kernel="square\nplatesolve -force"),
        replace(base, drizzle_pixfrac=float("nan")),
        replace(base, ra_hint_degrees=270.0, dec_hint_degrees=None),
    ):
        with pytest.raises(Exception):
            build_siril_script(invalid)
