#!/usr/bin/env python3
"""Build (and optionally install) the patched SEP wheel from the pinned sdist.

SEP 1.4.1 as published on PyPI is not deterministic on Windows: MSVC has no
``rand_r()`` and the upstream shim maps it to ``rand()``, whose per-thread state
the per-extraction seed reset never touches, so deblended faint pixels are
assigned differently from call to call (object counts and fluxes vary on
identical input).  It also grows its object and pixel lists by one object per
detection, which the Windows heap turns into quadratic copying.  The patch in
``packaging/patches/sep-1.4.1-zero-initialised-buffers.patch`` fixes both,
zero-initialises the Lutz/deblending buffers, and stamps the build
``1.4.1+oaf.1`` so ``sep.__version__`` identifies it.

    python scripts/build_sep_wheel.py --check      # download, verify, apply, no build
    python scripts/build_sep_wheel.py              # ... and build the wheel
    python scripts/build_sep_wheel.py --install    # ... and pip install it

The sdist is fetched from PyPI and accepted only when its SHA-256 and size
match the pins below; a mismatch is a hard failure and the bytes are discarded.
The patch is applied by a strict unified-diff applier (exact context, no fuzz,
bytes in and out) so the result does not depend on a ``patch``/``git`` binary or
on line-ending conversion.  The wheel is built with ``pip wheel --no-deps`` in
an isolated build environment (the sdist pins its own setuptools/Cython/numpy
build requirements); on Windows this needs the MSVC toolchain that the native
kernels already require.  ``--install`` reinstalls the wheel with
``--force-reinstall --no-deps`` and proves, in a fresh interpreter, that
``sep.__version__`` and the distribution metadata report the patched version
and that repeated deblending extractions of one crowded field are identical.

macOS and Linux keep the PyPI wheel for now; only Windows builds install this.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Sequence
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO_ROOT / "packaging" / "patches" / "sep-1.4.1-zero-initialised-buffers.patch"

SEP_VERSION = "1.4.1"
PATCHED_VERSION = "1.4.1+oaf.1"
SDIST_FILENAME = "sep-1.4.1.tar.gz"
SDIST_URL = (
    "https://files.pythonhosted.org/packages/a5/34/"
    "14537815638e3878209ad14aa51099a5ec73aca7ef517ad49b2b85072241/sep-1.4.1.tar.gz"
)
SDIST_SHA256 = "a0c8324ab66ee716080472cb212e4303e6c3e33b0d43763075b20d4cab793b25"
SDIST_SIZE = 569714
SDIST_ROOT = "sep-1.4.1"
DEFAULT_WORK_DIR = REPO_ROOT / "build" / "sep-wheel"

# Files the checked-in patch touches; the report records their digests so a
# build machine's tree can be compared with CI's.
PATCHED_FILES = (
    "PKG-INFO",
    "src/deblend.c",
    "src/extract.c",
    "src/extract.h",
    "src/lutz.c",
    "src/sepcore.h",
)


class SepBuildError(RuntimeError):
    """A fail-closed gate of the patched SEP build did not pass."""


# --------------------------------------------------------------------------
# sdist acquisition
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sdist(path: Path) -> str:
    """Return the sdist digest, raising unless size and SHA-256 match the pins."""

    if path.is_symlink() or not path.is_file():
        raise SepBuildError(f"sdist is missing or not a regular file: {path}")
    size = path.stat().st_size
    if size != SDIST_SIZE:
        raise SepBuildError(
            f"sdist size mismatch for {path.name}: expected {SDIST_SIZE} bytes, found {size}"
        )
    digest = _sha256(path)
    if digest != SDIST_SHA256:
        raise SepBuildError(
            f"sdist SHA-256 mismatch for {path.name}: expected {SDIST_SHA256}, found {digest}"
        )
    return digest


def download_sdist(destination: Path, *, url: str = SDIST_URL, timeout: float = 120.0) -> Path:
    """Fetch the pinned sdist to ``destination`` unless a verified copy exists.

    The download lands in a temporary file beside the destination and is
    renamed into place only after the digest check; a corrupt or tampered
    download therefore never becomes reusable.
    """

    if destination.exists():
        try:
            verify_sdist(destination)
            return destination
        except SepBuildError:
            # A stale or partial file cannot be trusted; refetch it.
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "openastroflow-build-sep-wheel/1"})
    handle, temporary_name = tempfile.mkstemp(prefix=".sep-sdist-", dir=str(destination.parent))
    temporary = Path(temporary_name)
    try:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, os.fdopen(handle, "wb") as stream:
                handle = -1
                shutil.copyfileobj(response, stream, 1 << 16)
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise SepBuildError(f"cannot download the SEP sdist from {url}: {error}") from error
        verify_sdist(temporary)
        temporary.replace(destination)
    except Exception:
        if handle >= 0:
            os.close(handle)
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def extract_sdist(archive: Path, destination: Path) -> Path:
    """Extract the verified sdist into ``destination`` and return the source root.

    Members are validated before extraction: regular files and directories
    only, relative paths without traversal, all below the single expected
    top-level directory.  ``destination`` must not exist yet.
    """

    if destination.exists() or destination.is_symlink():
        raise SepBuildError(f"extraction directory must be new: {destination}")
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                portable = PurePosixPath(member.name)
                parts = portable.parts
                if (
                    not parts
                    or portable.is_absolute()
                    or any(part in {"", ".", ".."} for part in parts)
                    or parts[0] != SDIST_ROOT
                    or "\\" in member.name
                ):
                    raise SepBuildError(f"sdist member has an unsafe path: {member.name!r}")
                if not (member.isreg() or member.isdir()):
                    raise SepBuildError(f"sdist member is not a file or directory: {member.name!r}")
            destination.mkdir(parents=True)
            extract_filter = getattr(tarfile, "data_filter", None)
            if extract_filter is not None:
                tar.extractall(destination, members=members, filter=extract_filter)
            else:  # pragma: no cover - Python < 3.11.4 without extraction filters
                tar.extractall(destination, members=members)
    except (tarfile.TarError, OSError, ValueError) as error:
        raise SepBuildError(f"cannot extract the SEP sdist: {error}") from error
    source_root = destination / SDIST_ROOT
    if not source_root.is_dir():
        raise SepBuildError("sdist did not contain the expected source root")
    return source_root


# --------------------------------------------------------------------------
# strict unified-diff applier
# --------------------------------------------------------------------------


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[tuple[str, bytes]] = field(default_factory=list)


@dataclass
class FilePatch:
    path: str
    hunks: list[Hunk] = field(default_factory=list)


_HUNK_HEADER = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _strip_prefix(name: bytes, prefix: bytes) -> str:
    text = name.split(b"\t", 1)[0].strip()
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return text.decode("utf-8")


def parse_patch(text: bytes) -> list[FilePatch]:
    """Parse a unified diff with ``a/``/``b/`` prefixes into file patches.

    Leading prose (the patch preamble) and ``diff --git`` lines are ignored, as
    ``patch(1)`` and ``git apply`` ignore them.  File renames, deletions,
    creations and binary hunks are rejected: the SEP patch only edits files.
    """

    patches: list[FilePatch] = []
    lines = text.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    index = 0
    current: FilePatch | None = None
    hunk: Hunk | None = None
    remaining_old = remaining_new = 0
    while index < len(lines):
        line = lines[index]
        if hunk is not None and line.startswith(b"\\"):
            # "\ No newline at end of file" qualifies the previous hunk line,
            # which may also be the last line of a completed hunk.
            if not hunk.lines:
                raise SepBuildError(f"misplaced end-of-file marker in {current.path}")
            kind, content = hunk.lines[-1]
            hunk.lines[-1] = (kind, content.rstrip(b"\n"))
            index += 1
            continue
        if hunk is not None and (remaining_old or remaining_new):
            # Inside a hunk every line is content, including one that happens
            # to start with "--- " (a removed line beginning with "-- ").
            if not line:
                # Some tools strip the trailing space of an empty context line.
                kind, content = " ", b"\n"
            else:
                kind, content = chr(line[0]), line[1:] + b"\n"
            if kind == " ":
                remaining_old -= 1
                remaining_new -= 1
            elif kind == "-":
                remaining_old -= 1
            elif kind == "+":
                remaining_new -= 1
            else:
                raise SepBuildError(f"unexpected line inside a hunk of {current.path}: {line[:40]!r}")
            if remaining_old < 0 or remaining_new < 0:
                raise SepBuildError(f"hunk in {current.path} has more lines than its header declares")
            hunk.lines.append((kind, content))
            index += 1
            continue
        if line.startswith(b"--- ") and index + 1 < len(lines) and lines[index + 1].startswith(b"+++ "):
            old_name = _strip_prefix(line[4:], b"a/")
            new_name = _strip_prefix(lines[index + 1][4:], b"b/")
            if old_name == "/dev/null" or new_name == "/dev/null":
                raise SepBuildError("patch creates or deletes a file; only edits are supported")
            if old_name != new_name:
                raise SepBuildError(f"patch renames {old_name!r} to {new_name!r}; renames are not supported")
            portable = PurePosixPath(old_name)
            if portable.is_absolute() or any(part in {"", ".", ".."} for part in portable.parts):
                raise SepBuildError(f"patch targets an unsafe path: {old_name!r}")
            current = FilePatch(path=old_name)
            patches.append(current)
            hunk = None
            index += 2
            continue
        match = _HUNK_HEADER.match(line)
        if match:
            if current is None:
                raise SepBuildError("hunk header before any file header")
            hunk = Hunk(
                old_start=int(match.group(1)),
                old_count=int(match.group(2)) if match.group(2) is not None else 1,
                new_start=int(match.group(3)),
                new_count=int(match.group(4)) if match.group(4) is not None else 1,
            )
            remaining_old, remaining_new = hunk.old_count, hunk.new_count
            current.hunks.append(hunk)
            index += 1
            continue
        if hunk is not None and line[:1] in (b"+", b"-", b" "):
            # Content after a hunk consumed its declared line counts.
            raise SepBuildError(f"hunk in {current.path} has more lines than its header declares")
        # Preamble text, "diff --git" and "index" lines are ignored.
        index += 1
    if hunk is not None and (remaining_old or remaining_new):
        raise SepBuildError(f"hunk in {current.path if current else '?'} is truncated")
    if not patches or any(not item.hunks for item in patches):
        raise SepBuildError("patch contains no hunks")
    return patches


def _split_lines(data: bytes) -> list[bytes]:
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
        return [line + b"\n" for line in lines]
    return [line + b"\n" for line in lines[:-1]] + [lines[-1]]


def apply_file_patch(original: bytes, patch: FilePatch) -> bytes:
    """Apply every hunk strictly at its declared position; no fuzz, no offsets.

    Prior hunks shift later positions by their net line delta, exactly as in
    the file the hunk headers describe.  A hunk whose old lines do not match
    fails the whole build: the pinned sdist has one known content, so any
    mismatch means the patch and the archive disagree.
    """

    lines = _split_lines(original)
    result: list[bytes] = []
    cursor = 0  # index into `lines` of the next unconsumed original line
    for number, hunk in enumerate(patch.hunks, start=1):
        old_lines = [content for kind, content in hunk.lines if kind in " -"]
        new_lines = [content for kind, content in hunk.lines if kind in " +"]
        if len(old_lines) != hunk.old_count or len(new_lines) != hunk.new_count:
            raise SepBuildError(f"hunk {number} of {patch.path} disagrees with its header counts")
        start = hunk.old_start - 1 if hunk.old_count else hunk.old_start
        if start < cursor:
            raise SepBuildError(f"hunk {number} of {patch.path} overlaps the previous hunk")
        if start + len(old_lines) > len(lines):
            raise SepBuildError(f"hunk {number} of {patch.path} extends past the end of the file")
        if lines[start : start + len(old_lines)] != old_lines:
            if lines[start : start + len(new_lines)] == new_lines:
                raise SepBuildError(f"hunk {number} of {patch.path} is already applied")
            raise SepBuildError(f"hunk {number} of {patch.path} does not match the pinned source")
        result.extend(lines[cursor:start])
        result.extend(new_lines)
        cursor = start + len(old_lines)
    result.extend(lines[cursor:])
    return b"".join(result)


def apply_patch(source_root: Path, patch_text: bytes) -> dict[str, str]:
    """Apply the patch below ``source_root``; return SHA-256 of each patched file."""

    digests: dict[str, str] = {}
    patched: list[tuple[Path, bytes]] = []
    for file_patch in parse_patch(patch_text):
        target = source_root / file_patch.path
        if target.is_symlink() or not target.is_file():
            raise SepBuildError(f"patched file is missing from the sdist: {file_patch.path}")
        patched.append((target, apply_file_patch(target.read_bytes(), file_patch)))
    # Write only after every file applied cleanly so a failure leaves the tree
    # pristine and diagnosable.
    for target, content in patched:
        target.write_bytes(content)
        digests[target.relative_to(source_root).as_posix()] = hashlib.sha256(content).hexdigest()
    return digests


def prepare_patched_source(sdist: Path, work_dir: Path, *, patch_path: Path = PATCH_PATH) -> tuple[Path, dict[str, str]]:
    """Verify, extract and patch the sdist under ``work_dir/source``."""

    verify_sdist(sdist)
    if patch_path.is_symlink() or not patch_path.is_file():
        raise SepBuildError(f"patch file is missing: {patch_path}")
    source_parent = work_dir / "source"
    if source_parent.exists():
        shutil.rmtree(source_parent)
    source_root = extract_sdist(sdist, source_parent)
    digests = apply_patch(source_root, patch_path.read_bytes())
    if set(digests) != set(PATCHED_FILES):
        raise SepBuildError(
            "patch touched an unexpected set of files: " + ", ".join(sorted(digests))
        )
    version_line = (source_root / "PKG-INFO").read_text(encoding="utf-8").splitlines()
    if f"Version: {PATCHED_VERSION}" not in version_line:
        raise SepBuildError("patched PKG-INFO does not declare the patched version")
    return source_root, digests


# --------------------------------------------------------------------------
# wheel build, install and proof
# --------------------------------------------------------------------------


def _run(command: Sequence[str], *, cwd: Path | None = None) -> float:
    print("+ " + " ".join(command), flush=True)
    started = time.perf_counter()
    completed = subprocess.run(list(command), cwd=None if cwd is None else str(cwd), check=False)
    if completed.returncode != 0:
        raise SepBuildError(f"command exited with {completed.returncode}: {command[0]} ...")
    return time.perf_counter() - started


def wheel_command(source_root: Path, wheel_dir: Path, *, python: str) -> list[str]:
    # Build isolation stays on: the sdist pins setuptools<72.2, Cython, numpy
    # and setuptools_scm for its own build, which the runtime venv need not
    # carry. --no-deps keeps pip from resolving anything but sep itself.
    return [python, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheel_dir), str(source_root)]


def install_command(wheel: Path, *, python: str) -> list[str]:
    return [python, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)]


def build_wheel(source_root: Path, wheel_dir: Path, *, python: str = sys.executable) -> Path:
    wheel_dir.mkdir(parents=True, exist_ok=True)
    # Only earlier sep wheels are removed, never the directory itself, so a
    # shared --wheel-dir cannot be wiped by mistake.
    for stale in wheel_dir.glob("sep-*.whl"):
        stale.unlink()
    _run(wheel_command(source_root, wheel_dir, python=python), cwd=REPO_ROOT)
    wheels = sorted(wheel_dir.glob("sep-*.whl"))
    expected_prefix = f"sep-{PATCHED_VERSION}-"
    if len(wheels) != 1 or not wheels[0].name.startswith(expected_prefix):
        names = ", ".join(path.name for path in wheels) or "none"
        raise SepBuildError(f"expected exactly one {expected_prefix}*.whl, found: {names}")
    return wheels[0]


_PROOF_SCRIPT = r"""
import importlib.metadata, json, sys
import numpy as np
import sep

rng = np.random.default_rng(7)
height, width = 384, 512
image = rng.normal(100.0, 4.0, size=(height, width))
yy, xx = np.mgrid[0:height, 0:width]
centres = rng.uniform(8, width - 8, 400), rng.uniform(8, height - 8, 400)
for x0, y0, amplitude in zip(*centres, rng.lognormal(6.0, 1.0, 400)):
    image += amplitude * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 1.8 ** 2))
    # A close companion forces deblending, which is the code path the patch fixes.
    image += 0.7 * amplitude * np.exp(-((xx - x0 - 4.5) ** 2 + (yy - y0 - 1.5) ** 2) / (2 * 1.8 ** 2))
image = image.astype(np.float32)
background = sep.Background(image)
data = image - background
reference = None
identical = True
for _ in range(4):
    junk = [np.ones(int(size)) for size in rng.integers(1000, 100000, 16)]
    del junk
    objects, segmentation = sep.extract(data, 1.5, err=background.globalrms, segmentation_map=True)
    payload = (objects.tobytes(), segmentation.tobytes())
    if reference is None:
        reference = payload
    elif payload != reference:
        identical = False
print(json.dumps({
    "version": sep.__version__,
    "metadata": importlib.metadata.version("sep"),
    "deterministic": identical,
    "objects": int(len(objects)),
    # SEP_OBJ_MERGED (bit 0): objects that came out of the deblender, i.e.
    # the code path whose randomness the patch makes reproducible.
    "deblended": int(((objects["flag"] & 1) != 0).sum()),
}))
"""


def prove_installation(*, python: str = sys.executable, timeout: float = 300.0) -> dict[str, Any]:
    """Import the installed sep in a fresh interpreter and check version + determinism."""

    try:
        completed = subprocess.run(
            [python, "-c", _PROOF_SCRIPT],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            cwd=str(REPO_ROOT),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SepBuildError(f"installed sep proof did not run: {error}") from error
    if completed.returncode != 0:
        raise SepBuildError("installed sep proof failed: " + completed.stderr.strip()[-1500:])
    try:
        facts = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise SepBuildError("installed sep proof emitted no JSON") from error
    if facts.get("version") != PATCHED_VERSION or facts.get("metadata") != PATCHED_VERSION:
        raise SepBuildError(
            f"installed sep reports {facts.get('version')!r}/{facts.get('metadata')!r}, "
            f"expected {PATCHED_VERSION}"
        )
    if not isinstance(facts.get("deblended"), int) or facts["deblended"] <= 0:
        raise SepBuildError("installed sep proof field did not exercise the deblender")
    if facts.get("deterministic") is not True:
        raise SepBuildError("installed sep produced different catalogs for identical input")
    return facts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="scratch directory (default: build/sep-wheel)")
    parser.add_argument("--sdist", type=Path, default=None, help="use this local copy of the pinned sdist instead of downloading")
    parser.add_argument("--wheel-dir", type=Path, default=None, help="where to put the wheel (default: <work-dir>/wheels)")
    parser.add_argument("--python", default=sys.executable, help="interpreter whose pip builds/installs the wheel")
    parser.add_argument("--check", action="store_true", help="download, verify and patch only; do not build")
    parser.add_argument("--install", action="store_true", help="pip install --force-reinstall --no-deps the built wheel and prove it")
    parser.add_argument("--report", type=Path, default=None, help="write a JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    work_dir = arguments.work_dir.resolve()
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "openastroflow-patched-sep-build",
        "sdist": {"fileName": SDIST_FILENAME, "sha256": SDIST_SHA256, "sizeBytes": SDIST_SIZE, "url": SDIST_URL},
        "patch": {"path": PATCH_PATH.relative_to(REPO_ROOT).as_posix()},
        "upstreamVersion": SEP_VERSION,
        "patchedVersion": PATCHED_VERSION,
        "mode": "check" if arguments.check else ("install" if arguments.install else "wheel"),
    }
    try:
        if arguments.check and arguments.install:
            raise SepBuildError("--check and --install are mutually exclusive")
        work_dir.mkdir(parents=True, exist_ok=True)
        report["patch"]["sha256"] = _sha256(PATCH_PATH)
        sdist = arguments.sdist.resolve() if arguments.sdist is not None else download_sdist(work_dir / SDIST_FILENAME)
        source_root, digests = prepare_patched_source(sdist, work_dir)
        report["patchedFiles"] = digests
        if not arguments.check:
            wheel_dir = (arguments.wheel_dir or work_dir / "wheels").resolve()
            wheel = build_wheel(source_root, wheel_dir, python=arguments.python)
            report["wheel"] = {"path": str(wheel), "fileName": wheel.name, "sha256": _sha256(wheel), "sizeBytes": wheel.stat().st_size}
            if arguments.install:
                _run(install_command(wheel, python=arguments.python), cwd=REPO_ROOT)
                report["installed"] = prove_installation(python=arguments.python)
        report["ok"] = True
    except SepBuildError as error:
        report["ok"] = False
        report["error"] = str(error)
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PATCHED_FILES",
    "PATCHED_VERSION",
    "PATCH_PATH",
    "SDIST_SHA256",
    "SDIST_SIZE",
    "SDIST_URL",
    "SEP_VERSION",
    "SepBuildError",
    "apply_file_patch",
    "apply_patch",
    "build_wheel",
    "download_sdist",
    "extract_sdist",
    "main",
    "parse_patch",
    "prepare_patched_source",
    "prove_installation",
    "verify_sdist",
    "wheel_command",
    "install_command",
]
