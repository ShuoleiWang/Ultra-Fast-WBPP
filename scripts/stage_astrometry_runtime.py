#!/usr/bin/env python3
"""Create-only stage a self-contained Astrometry.net solver for a demo app.

A self-contained macOS demo build ships ``solve-field`` and the managed index
set inside the app, so a new Mac plate-solves without Homebrew or a catalog
download.  This script:

- fetches the pinned macOS 14 (``arm64_sonoma``) Homebrew bottles of
  astrometry-net and its three libraries and checks their SHA-256;
- relocates ``solve-field``, ``astrometry-engine`` and the libraries to
  ``@rpath`` beside each other and re-signs them ad hoc;
- writes ``removelines`` and ``uniformize`` wrappers that run upstream's two
  Python helpers inside the frozen engine (``ufwbpp.solvers.astrometry_helpers``);
- copies the index files of a checked catalog manifest, hash-verified, which
  the engine installs into the user's catalog directory on first use;
- records everything in ``runtime.json``.

Astrometry.net is GPL-3.0-or-later as distributed (it bundles GSL) and the
index files' redistribution terms are unresolved, so this is a demo-only
build step; the release pipeline does not call it (see docs/licensing.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Sequence
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = "aarch64-apple-darwin"
BOTTLE_TAG = "arm64_sonoma"
MAXIMUM_DEPLOYMENT_TARGET = (14, 0)
BOTTLES = {
    "astrometry-net": ("0.97", "ed304036c98a2e5b34afd683447e9ec731da0a932f4efd4583948ffa912cd0ab"),
    "gsl": ("2.8", "b5cd011cc1f8ac606487224628d21247cbe290b4a035f844ab016088c82bbdf7"),
    "wcslib": ("8.9", "0203ae796a2bed7337e97124bb2cd6323de06d98d6bc9d60069facb2b96c3ff3"),
    "cfitsio": ("4.7.0", "86813032566ed2d9b4ef2b7c0af87b3083c10ce5366e7dbfcfc787a3cc8e700c"),
}
# (bottle, member inside the bottle, staged path)
PROGRAMS = (
    ("astrometry-net", "bin/solve-field", "bin/solve-field"),
    ("astrometry-net", "bin/astrometry-engine", "bin/astrometry-engine"),
)
LIBRARIES = (
    ("gsl", "lib/libgsl.28.dylib", "lib/libgsl.28.dylib"),
    ("gsl", "lib/libgslcblas.0.dylib", "lib/libgslcblas.0.dylib"),
    ("wcslib", "lib/libwcs.8.dylib", "lib/libwcs.8.dylib"),
    ("cfitsio", "lib/libcfitsio.10.dylib", "lib/libcfitsio.10.dylib"),
)
NOTICES = (
    ("astrometry-net", "LICENSE", "legal/astrometry-net-LICENSE"),
    ("gsl", "COPYING", "legal/gsl-COPYING"),
    ("wcslib", "COPYING", "legal/wcslib-COPYING"),
    ("wcslib", "COPYING.LESSER", "legal/wcslib-COPYING.LESSER"),
)
HELPERS = ("removelines", "uniformize")
WRAPPER = """#!/bin/sh
# solve-field's Python helper {tool!r}, run by the bundled Ultra-Fast WBPP engine
# (ufwbpp.solvers.astrometry_helpers), so no Python installation is needed.
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P) || exit 70
exec "$here/{engine}" __astrometry-helper-v1 {tool} "$@"
"""


class StageError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(*command: str) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise StageError(f"{' '.join(command)} failed: {result.stderr.strip()}")
    return result.stdout


def fetch_bottles(cache: Path) -> dict[str, Path]:
    cache.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for formula, (_, digest) in BOTTLES.items():
        path = cache / f"{formula}.{BOTTLE_TAG}.tar.gz"
        if not path.exists():
            request = urllib.request.Request(
                f"https://ghcr.io/v2/homebrew/core/{formula}/blobs/sha256:{digest}",
                headers={"Authorization": "Bearer QQ=="},
            )
            partial = path.with_suffix(".partial")
            with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            partial.rename(path)
        if _sha256(path) != digest:
            raise StageError(f"{path.name} does not match its pinned SHA-256")
        paths[formula] = path
    return paths


def _extract(bottle: Path, formula: str, member: str, destination: Path) -> None:
    version = BOTTLES[formula][0]
    with tarfile.open(bottle) as archive:
        name = f"{formula}/{version}/{member}"
        for _ in range(8):
            info = archive.getmember(name)
            if not info.issym():
                break
            name = str(Path(name).parent / info.linkname)
        else:
            raise StageError(f"symlink loop at {member} in {bottle.name}")
        if not info.isfile():
            raise StageError(f"{member} is not a regular file in {bottle.name}")
        stream = archive.extractfile(info)
        assert stream is not None
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as output:
            shutil.copyfileobj(stream, output)
    destination.chmod(0o755 if member.startswith("bin/") else 0o644)


def _relocate(path: Path, *, program: bool) -> None:
    names = [line.split(" (")[0].strip() for line in _run("otool", "-L", str(path)).splitlines()[1:]]
    arguments: list[str] = []
    for name in names:
        if name.startswith("@@HOMEBREW_"):
            arguments += ["-change", name, f"@rpath/{Path(name).name}"]
    if program:
        arguments += ["-add_rpath", "@executable_path/../lib"]
    else:
        arguments += ["-id", f"@rpath/{path.name}"]
    _run("install_name_tool", *arguments, str(path))
    _run("codesign", "--force", "--sign", "-", "--timestamp=none", str(path))


def _check_macho(path: Path) -> None:
    links = _run("otool", "-L", str(path))
    if "@@HOMEBREW" in links or "/opt/homebrew" in links or "/usr/local/" in links:
        raise StageError(f"{path.name} still links outside the bundle:\n{links}")
    commands = _run("otool", "-l", str(path)).split()
    minimum = commands[commands.index("minos") + 1]
    if tuple(int(part) for part in minimum.split(".")[:2]) > MAXIMUM_DEPLOYMENT_TARGET:
        raise StageError(f"{path.name} requires macOS {minimum}")
    _run("codesign", "--verify", "--strict", str(path))


def _catalog(catalog_id: str, source: Path) -> Any:
    sys.path.insert(0, str(REPO_ROOT / "packages" / "engine" / "src"))
    from ufwbpp.solvers.catalogs import get_catalog_manifest

    manifest = get_catalog_manifest(catalog_id)
    for artifact in manifest.artifacts:
        path = source / artifact.artifact_id
        if not path.is_file() or path.stat().st_size != artifact.size_bytes or _sha256(path) != artifact.sha256:
            raise StageError(f"{path} is missing or does not match the checked manifest")
    return manifest


def stage(output: Path, cache: Path, catalog_dir: Path, catalog_id: str) -> dict[str, Any]:
    if output.exists():
        raise StageError(f"refusing to replace existing output: {output}")
    manifest = _catalog(catalog_id, catalog_dir)
    bottles = fetch_bottles(cache)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for formula, member, relative in PROGRAMS + LIBRARIES:
            path = staging / relative
            _extract(bottles[formula], formula, member, path)
            _relocate(path, program=(formula, member, relative) in PROGRAMS)
            _check_macho(path)
        for formula, member, relative in NOTICES:
            _extract(bottles[formula], formula, member, staging / relative)
        engine = f"../../ufwbpp-engine/ufwbpp-engine-{TARGET}/ufwbpp-engine-{TARGET}"
        for tool in HELPERS:
            wrapper = staging / "bin" / tool
            wrapper.write_text(WRAPPER.format(tool=tool, engine=engine), encoding="utf-8")
            wrapper.chmod(0o755)
        index = staging / "index"
        index.mkdir()
        for artifact in manifest.artifacts:
            source, target = catalog_dir / artifact.artifact_id, index / artifact.artifact_id
            # An APFS clone costs no space; any other volume gets a copy.
            if subprocess.run(["cp", "-c", str(source), str(target)], capture_output=True).returncode != 0:
                shutil.copyfile(source, target)
        files = sorted(
            (path for path in staging.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(staging).as_posix(),
        )
        record = {
            "schemaVersion": 1,
            "kind": "ultra-fast-wbpp-bundled-astrometry-runtime",
            "targetTriple": TARGET,
            "bottles": [
                {"formula": formula, "version": version, "bottleTag": BOTTLE_TAG, "sha256": digest}
                for formula, (version, digest) in BOTTLES.items()
            ],
            "catalog": {
                "catalogId": manifest.catalog_id,
                "manifestSha256": manifest.manifest_sha256,
                "relativePath": "index",
            },
            "files": [
                {
                    "path": path.relative_to(staging).as_posix(),
                    "sizeBytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in files
            ],
        }
        (staging / "runtime.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.chmod(0o755)
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bottle-cache", type=Path, default=REPO_ROOT / "build" / "astrometry-bottles")
    parser.add_argument("--catalog-dir", type=Path, help="installed managed index directory (default: the engine's)")
    parser.add_argument("--catalog-id", default="astrometry-net-4107-4112")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        parser.error("the bundled solver runtime is staged on macOS only")
    catalog_dir = args.catalog_dir
    if catalog_dir is None:
        sys.path.insert(0, str(REPO_ROOT / "packages" / "engine" / "src"))
        from ufwbpp.solvers.catalogs import default_catalog_root

        catalog_dir = default_catalog_root()
    try:
        record = stage(args.output.absolute(), args.bottle_cache.absolute(), catalog_dir, args.catalog_id)
    except (StageError, OSError, KeyError) as error:
        sys.stderr.write(f"stage_astrometry_runtime: {error}\n")
        return 1
    total = sum(item["sizeBytes"] for item in record["files"])
    sys.stdout.write(f"staged {len(record['files'])} files ({total / 2**20:.0f} MiB) at {args.output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
