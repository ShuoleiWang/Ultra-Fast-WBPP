from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_public_tree import inspect  # noqa: E402


def test_clean_tree(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "safe.py").write_text("print('safe')\n", encoding="utf-8")
    assert inspect(tmp_path) == []


def test_rejects_private_paths_raw_data_and_symlinks(tmp_path: Path) -> None:
    private_home = "/" + "Users/private/Astro/frame.fits"
    (tmp_path / "source.py").write_text(
        f'SOURCE = "{private_home}"\n', encoding="utf-8"
    )
    (tmp_path / "frame.fits").write_bytes(b"not a fixture")
    (tmp_path / "target.txt").write_text("safe\n", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(tmp_path / "target.txt")
    codes = {finding.code for finding in inspect(tmp_path)}
    assert {"MAC_HOME_PATH", "RAW_DATA", "SYMLINK"} <= codes


def test_ignored_build_and_explicit_fixture_are_allowed(tmp_path: Path) -> None:
    (tmp_path / "build").mkdir()
    private_home = "/" + "Users/private/ignored"
    (tmp_path / "build" / "secret.py").write_text(
        f'PATH = "{private_home}"\n', encoding="utf-8"
    )
    fixture = tmp_path / "tests" / "fixtures" / "tiny.fits"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"synthetic fixture")
    assert inspect(tmp_path) == []


def test_git_indexed_file_is_checked_even_below_local_state_directory(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    private = tmp_path / ".openastroflow" / "catalogs" / "forced.fits"
    private.parent.mkdir(parents=True)
    private.write_bytes(b"catalog")
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "add",
            "-f",
            ".openastroflow/catalogs/forced.fits",
        ],
        check=True,
    )
    assert any(finding.code == "RAW_DATA" for finding in inspect(tmp_path))


def test_runtime_binary_is_never_publishable_source(tmp_path: Path) -> None:
    binary = tmp_path / "package" / "native" / "libworker.dylib"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"Mach-O placeholder")
    assert any(finding.code == "BINARY_RUNTIME" for finding in inspect(tmp_path))


def test_git_audit_includes_new_source_but_not_ignored_outputs(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", ".gitignore"], check=True)
    private_home = "/" + "home/private/acquisition"
    (tmp_path / "new.py").write_text(f'PATH = "{private_home}"\n', encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "generated.py").write_text(
        f'PATH = "{private_home}"\n', encoding="utf-8"
    )
    findings = inspect(tmp_path)
    assert [(finding.code, finding.path) for finding in findings] == [
        ("LINUX_HOME_PATH", "new.py")
    ]
