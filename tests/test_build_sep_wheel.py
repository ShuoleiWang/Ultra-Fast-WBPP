from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

import pytest

from scripts import build_sep_wheel as sep_build


# Digests of the six files after the checked-in patch is applied to the pinned
# sdist. They change only when the patch or the sdist pin changes, which must
# be a deliberate, reviewed edit of this table.
EXPECTED_PATCHED_DIGESTS = {
    "PKG-INFO": "1977c611d2a164c81327ab0e94678161344f7360122a825fbeadedd7d8d4e8f8",
    "src/deblend.c": "44e094c71263f0bf92747c8ebc08273c2acea6dd86a0b28c88f66dfcbac027a9",
    "src/extract.c": "2177593878cf5b383010e31b2ad87e04c970f8b0affad9302601907deca4b9b7",
    "src/extract.h": "798375262c5c67e64ceac131d542bb571d3d426d5a176587c80763f92ff6e04e",
    "src/lutz.c": "0e84321f1bd29764f01a1b5ccbeac3c4167456c4a403bf4a00778651a7108855",
    "src/sepcore.h": "35696c219c0f6dd0c93822525322477dc6b317f986222037808b69eabd084ad8",
}


SYNTHETIC_PATCH = b"""Preamble prose that applier must ignore.

diff --git a/src/one.c b/src/one.c
--- a/src/one.c
+++ b/src/one.c
@@ -1,5 +1,6 @@
 alpha
-beta
+BETA
+gamma-inserted
 gamma

 delta
@@ -9,3 +10,3 @@
 eight
--- nine
+++ NINE
 ten
\\ No newline at end of file
diff --git a/two.txt b/two.txt
--- a/two.txt
+++ b/two.txt
@@ -1 +1 @@
-old
+new
"""

ONE_C_BEFORE = b"alpha\nbeta\ngamma\n\ndelta\nfive\nsix\nseven\neight\n-- nine\nten"
ONE_C_AFTER = b"alpha\nBETA\ngamma-inserted\ngamma\n\ndelta\nfive\nsix\nseven\neight\n++ NINE\nten"


def test_parse_patch_reads_hunks_prefixes_and_end_of_file_markers() -> None:
    patches = sep_build.parse_patch(SYNTHETIC_PATCH)

    assert [item.path for item in patches] == ["src/one.c", "two.txt"]
    first, second = patches
    assert [(hunk.old_start, hunk.old_count, hunk.new_start, hunk.new_count) for hunk in first.hunks] == [
        (1, 5, 1, 6),
        (9, 3, 10, 3),
    ]
    # A removed line that itself starts with "-- " is hunk content, not a header.
    assert first.hunks[1].lines[1] == ("-", b"-- nine\n")
    assert first.hunks[1].lines[-1] == (" ", b"ten")
    assert second.hunks[0].lines == [("-", b"old\n"), ("+", b"new\n")]


def test_apply_file_patch_is_exact_and_reports_mismatch_or_reapplication() -> None:
    patches = sep_build.parse_patch(SYNTHETIC_PATCH)
    one, two = patches

    assert sep_build.apply_file_patch(ONE_C_BEFORE, one) == ONE_C_AFTER
    assert sep_build.apply_file_patch(b"old\n", two) == b"new\n"

    with pytest.raises(sep_build.SepBuildError, match="already applied"):
        sep_build.apply_file_patch(ONE_C_AFTER, one)
    with pytest.raises(sep_build.SepBuildError, match="does not match"):
        sep_build.apply_file_patch(ONE_C_BEFORE.replace(b"beta", b"drifted"), one)
    with pytest.raises(sep_build.SepBuildError, match="past the end"):
        sep_build.apply_file_patch(b"alpha\nbeta\n", one)


@pytest.mark.parametrize(
    "text, message",
    (
        (b"--- /dev/null\n+++ b/new.c\n@@ -0,0 +1 @@\n+x\n", "creates or deletes"),
        (b"--- a/x.c\n+++ b/y.c\n@@ -1 +1 @@\n-a\n+b\n", "renames"),
        (b"--- a/../x.c\n+++ b/../x.c\n@@ -1 +1 @@\n-a\n+b\n", "unsafe path"),
        (b"--- a/x.c\n+++ b/x.c\n@@ -1,2 +1,2 @@\n-a\n+b\n", "truncated"),
        (b"--- a/x.c\n+++ b/x.c\n@@ -1 +1 @@\n-a\n+b\n+c\n", "more lines than its header"),
        (b"@@ -1 +1 @@\n-a\n+b\n", "before any file header"),
        (b"just prose\n", "no hunks"),
    ),
)
def test_parse_patch_rejects_unsupported_or_malformed_input(text: bytes, message: str) -> None:
    with pytest.raises(sep_build.SepBuildError, match=message):
        sep_build.parse_patch(text)


def test_apply_patch_writes_nothing_unless_every_file_applies(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.c").write_bytes(ONE_C_BEFORE)
    (tmp_path / "two.txt").write_bytes(b"unexpected\n")

    with pytest.raises(sep_build.SepBuildError, match="two.txt does not match"):
        sep_build.apply_patch(tmp_path, SYNTHETIC_PATCH)
    assert (tmp_path / "src" / "one.c").read_bytes() == ONE_C_BEFORE

    (tmp_path / "two.txt").write_bytes(b"old\n")
    digests = sep_build.apply_patch(tmp_path, SYNTHETIC_PATCH)
    assert (tmp_path / "src" / "one.c").read_bytes() == ONE_C_AFTER
    assert digests == {
        "src/one.c": hashlib.sha256(ONE_C_AFTER).hexdigest(),
        "two.txt": hashlib.sha256(b"new\n").hexdigest(),
    }


def test_checked_in_patch_targets_exactly_the_declared_files() -> None:
    patches = sep_build.parse_patch(sep_build.PATCH_PATH.read_bytes())

    assert sorted(item.path for item in patches) == sorted(sep_build.PATCHED_FILES)
    for item in patches:
        for hunk in item.hunks:
            assert sum(kind in " -" for kind, _ in hunk.lines) == hunk.old_count
            assert sum(kind in " +" for kind, _ in hunk.lines) == hunk.new_count
    text = sep_build.PATCH_PATH.read_text(encoding="utf-8")
    assert "+Version: 1.4.1+ufwbpp.1" in text
    assert "+int sep_rand_r(unsigned int * seed)" in text
    assert "-#define rand_r(SEED)" in text
    assert "+  QCALLOC(buffers->start, int64_t, stacksize, status);" in text
    assert "+  QCALLOC(ctx->son, short, deblend_nthresh * nsonmax * NBRANCH, status);" in text
    assert "+  int64_t nobjalloc;" in text


def test_verify_sdist_fails_closed_on_size_or_digest(tmp_path: Path) -> None:
    wrong_size = tmp_path / "sep-1.4.1.tar.gz"
    wrong_size.write_bytes(b"x" * 10)
    with pytest.raises(sep_build.SepBuildError, match="size mismatch"):
        sep_build.verify_sdist(wrong_size)

    wrong_digest = tmp_path / "other.tar.gz"
    wrong_digest.write_bytes(b"y" * sep_build.SDIST_SIZE)
    with pytest.raises(sep_build.SepBuildError, match="SHA-256 mismatch"):
        sep_build.verify_sdist(wrong_digest)
    with pytest.raises(sep_build.SepBuildError, match="missing"):
        sep_build.verify_sdist(tmp_path / "absent.tar.gz")


def _tar_with(members: list[tuple[str, bytes | None]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            if payload is None:
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                tar.addfile(info)
            else:
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


@pytest.mark.parametrize(
    "members, message",
    (
        ([("sep-1.4.1/../escape", b"x")], "unsafe path"),
        ([("/abs/file", b"x")], "unsafe path"),
        ([("other-1.0/PKG-INFO", b"x")], "unsafe path"),
        ([("sep-1.4.1/link", None)], "not a file or directory"),
    ),
)
def test_extract_sdist_validates_members_before_touching_the_disk(
    tmp_path: Path, members: list[tuple[str, bytes | None]], message: str
) -> None:
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(_tar_with(members))

    with pytest.raises(sep_build.SepBuildError, match=message):
        sep_build.extract_sdist(archive, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_build_and_install_commands_keep_isolation_and_no_deps(tmp_path: Path) -> None:
    wheel_command = sep_build.wheel_command(tmp_path / "src", tmp_path / "wheels", python="py")
    assert wheel_command == [
        "py", "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(tmp_path / "wheels"), str(tmp_path / "src")
    ]
    assert "--no-build-isolation" not in wheel_command
    install_command = sep_build.install_command(tmp_path / "sep.whl", python="py")
    assert install_command == ["py", "-m", "pip", "install", "--force-reinstall", "--no-deps", str(tmp_path / "sep.whl")]


@pytest.fixture(scope="session")
def pinned_sdist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The pinned sdist, from UFWBPP_SEP_SDIST when set, else fetched from PyPI."""

    override = os.environ.get("UFWBPP_SEP_SDIST")
    if override:
        return Path(override)
    destination = tmp_path_factory.mktemp("sep-sdist") / sep_build.SDIST_FILENAME
    try:
        return sep_build.download_sdist(destination, timeout=60.0)
    except sep_build.SepBuildError as error:
        if "cannot download" in str(error):
            pytest.skip(f"PyPI is not reachable: {error}")
        raise


def test_patch_applies_cleanly_to_the_pinned_sdist(tmp_path: Path, pinned_sdist: Path) -> None:
    assert sep_build.verify_sdist(pinned_sdist) == sep_build.SDIST_SHA256

    source_root, digests = sep_build.prepare_patched_source(pinned_sdist, tmp_path / "work")

    assert digests == EXPECTED_PATCHED_DIGESTS
    pkg_info = (source_root / "PKG-INFO").read_text(encoding="utf-8")
    assert "\nVersion: 1.4.1+ufwbpp.1\n" in pkg_info
    sepcore = (source_root / "src" / "sepcore.h").read_text(encoding="utf-8")
    assert "#define rand_r(SEED) sep_rand_r(SEED)" in sepcore
    # Applying twice must be detected, never silently doubled.
    with pytest.raises(sep_build.SepBuildError, match="already applied"):
        sep_build.apply_patch(source_root, sep_build.PATCH_PATH.read_bytes())


def test_check_mode_reports_without_building(
    tmp_path: Path, pinned_sdist: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "report.json"

    code = sep_build.main(
        ["--check", "--work-dir", str(tmp_path / "work"), "--sdist", str(pinned_sdist), "--report", str(report_path)]
    )

    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["mode"] == "check"
    assert report["patchedVersion"] == "1.4.1+ufwbpp.1"
    assert report["patchedFiles"] == EXPECTED_PATCHED_DIGESTS
    assert report["sdist"]["sha256"] == sep_build.SDIST_SHA256
    assert "wheel" not in report
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert not (tmp_path / "work" / "wheels").exists()


WHEEL_NAME = f"sep-{sep_build.PATCHED_VERSION}-cp312-cp312-win_amd64.whl"


def test_reusable_wheel_requires_the_recorded_source_interpreter_and_bytes(tmp_path: Path) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = wheels / WHEEL_NAME
    wheel.write_bytes(b"wheel bytes")
    interpreter = {"implementation": "CPython", "version": "3.12.10", "platform": "win-amd64"}

    def provenance(**changes: object) -> dict:
        fields = {"patch_sha256": "a" * 64, "patched_files": EXPECTED_PATCHED_DIGESTS, "interpreter": interpreter}
        fields.update(changes)
        return sep_build.wheel_provenance(**fields)

    assert sep_build.reusable_wheel(wheels, provenance()) is None  # no record yet
    sep_build.record_wheel_provenance(wheels, provenance(), wheel)
    assert sep_build.reusable_wheel(wheels, provenance()) == wheel
    assert sep_build.reusable_wheel(wheels, provenance(patch_sha256="b" * 64)) is None
    assert sep_build.reusable_wheel(
        wheels, provenance(patched_files={**EXPECTED_PATCHED_DIGESTS, "src/lutz.c": "0" * 64})
    ) is None
    assert sep_build.reusable_wheel(wheels, provenance(interpreter={**interpreter, "version": "3.12.11"})) is None
    (wheels / "sep-1.4.1-cp312-cp312-win_amd64.whl").write_bytes(b"another wheel")
    assert sep_build.reusable_wheel(wheels, provenance()) is None  # not exactly one wheel
    (wheels / "sep-1.4.1-cp312-cp312-win_amd64.whl").unlink()
    wheel.write_bytes(b"replaced bytes")
    assert sep_build.reusable_wheel(wheels, provenance()) is None


def test_reuse_wheel_skips_only_the_compile(
    tmp_path: Path, pinned_sdist: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds: list[Path] = []
    commands: list[list[str]] = []
    proofs: list[str] = []

    def fake_build(source_root: Path, wheel_dir: Path, *, python: str) -> Path:
        wheel_dir.mkdir(parents=True, exist_ok=True)
        wheel = wheel_dir / WHEEL_NAME
        wheel.write_bytes(b"compiled %d" % len(builds))
        builds.append(source_root)
        return wheel

    def fake_prove(*, python: str) -> dict:
        proofs.append(python)
        return {"version": sep_build.PATCHED_VERSION, "deterministic": True}

    monkeypatch.setattr(sep_build, "build_wheel", fake_build)
    monkeypatch.setattr(sep_build, "_run", lambda command, cwd=None: commands.append(list(command)) or 0.0)
    monkeypatch.setattr(sep_build, "prove_installation", fake_prove)
    work = tmp_path / "work"
    common = ["--install", "--work-dir", str(work), "--sdist", str(pinned_sdist)]

    reports = []
    for index, extra in enumerate((["--reuse-wheel"], ["--reuse-wheel"], [])):
        report_path = tmp_path / f"report-{index}.json"
        assert sep_build.main([*common, *extra, "--report", str(report_path)]) == 0
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))

    # The second run reuses the recorded wheel; without --reuse-wheel the
    # compile always runs and the record no longer describes the wheel.
    assert [report["wheel"]["reused"] for report in reports] == [False, True, False]
    assert len(builds) == 2
    assert not (work / "wheels" / sep_build.PROVENANCE_NAME).exists()
    # Every run still verified and patched the sdist, reinstalled and proved.
    assert all(report["patchedFiles"] == EXPECTED_PATCHED_DIGESTS for report in reports)
    assert len(commands) == 3 and all("--force-reinstall" in command for command in commands)
    assert len(proofs) == 3
