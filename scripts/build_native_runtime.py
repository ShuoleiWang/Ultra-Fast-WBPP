#!/usr/bin/env python3
"""Build, test and install the native kernel library for the running host.

One implementation for every platform (CI, release, developer machines):

    python scripts/build_native_runtime.py --build-dir build/native-release

configures ``native`` as a Release build with the strict flags that the
CMake project applies (``/W4 /WX /fp:strict`` on MSVC, ``-Werror -fno-fast-math
-ffp-contract=off`` elsewhere), builds it, runs ``ctest`` and installs the
shared library into ``packages/engine/src/ufwbpp/
native`` where ``ufwbpp.native_kernels`` finds it.  Apple Metal is
enabled on macOS and disabled elsewhere unless ``--metal`` says otherwise.

The optional ``--report`` JSON records the installed library's path, SHA-256,
the compiler and generator, so a build machine's toolchain can be quoted in
receipts and reports.

On Windows the installed DLL's PE import table is parsed (``scripts/
pe_imports.py``) and recorded as ``library.imports``/``library.crtLinkage``.
The CMake project links the Visual C++ runtime statically, so the DLL must not
import ``MSVCP140.dll``/``VCRUNTIME140*.dll``: a dynamic CRT is only present on
machines with the matching redistributable, and the frozen engine cannot rely
on that.  ``--require-static-crt`` (the default on Windows; CI passes it
explicitly) turns such an import into a build failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

try:
    from scripts.pe_imports import (
        PEFormatError,
        crt_linkage,
        dynamic_crt_imports,
        is_pe_image,
        read_pe_imports,
        ucrt_imports,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from pe_imports import (
        PEFormatError,
        crt_linkage,
        dynamic_crt_imports,
        is_pe_image,
        read_pe_imports,
        ucrt_imports,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = REPO_ROOT / "native"
INSTALL_PREFIX = REPO_ROOT / "packages" / "engine" / "src"
RUNTIME_DIR = INSTALL_PREFIX / "ufwbpp" / "native"
LIBRARY_NAMES = (
    "libufwbpp_native.dylib",
    "libufwbpp_native.so",
    "ufwbpp_native.dll",
)


def host_library_name(sys_platform: str = sys.platform) -> str:
    if sys_platform == "darwin":
        return "libufwbpp_native.dylib"
    if sys_platform == "win32":
        return "ufwbpp_native.dll"
    return "libufwbpp_native.so"


def metal_enabled(choice: str, sys_platform: str = sys.platform) -> bool:
    if choice == "on":
        return True
    if choice == "off":
        return False
    return sys_platform == "darwin"


def configure_command(
    build_dir: Path,
    *,
    metal: bool,
    tests: bool,
    generator: str | None,
    architecture: str | None,
    cmake: str = "cmake",
) -> list[str]:
    command = [cmake, "-S", str(NATIVE_SOURCE), "-B", str(build_dir)]
    if generator:
        command.extend(["-G", generator])
    if architecture:
        command.extend(["-A", architecture])
    command.extend(
        [
            f"-DUFWBPP_BUILD_TESTS={'ON' if tests else 'OFF'}",
            f"-DUFWBPP_ENABLE_METAL={'ON' if metal else 'OFF'}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    return command


def build_command(build_dir: Path, *, jobs: int | None, cmake: str = "cmake") -> list[str]:
    command = [cmake, "--build", str(build_dir), "--config", "Release", "--parallel"]
    if jobs:
        command.append(str(jobs))
    return command


def test_command(build_dir: Path, *, ctest: str = "ctest") -> list[str]:
    return [ctest, "--test-dir", str(build_dir), "-C", "Release", "--output-on-failure"]


def install_command(build_dir: Path, *, prefix: Path = INSTALL_PREFIX, cmake: str = "cmake") -> list[str]:
    return [cmake, "--install", str(build_dir), "--config", "Release", "--prefix", str(prefix)]


def _run(command: Sequence[str], *, cwd: Path = REPO_ROOT) -> float:
    print("+ " + " ".join(command), flush=True)
    started = time.monotonic()
    subprocess.run(list(command), check=True, cwd=cwd)
    return time.monotonic() - started


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cmake_cache(build_dir: Path) -> dict[str, str]:
    """CMake cache entries plus the compiler facts CMake keeps beside the cache.

    ``CMAKE_CXX_COMPILER_ID``/``_VERSION`` live in
    ``CMakeFiles/<cmake-version>/CMakeCXXCompiler.cmake`` as ``set()`` calls,
    not in ``CMakeCache.txt``.
    """

    cache = build_dir / "CMakeCache.txt"
    values: dict[str, str] = {}
    if not cache.is_file():
        return values
    pattern = re.compile(r"^([A-Za-z0-9_\-]+):[A-Z]+=(.*)$")
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            values[match.group(1)] = match.group(2)
    setter = re.compile(r'^set\((CMAKE_CXX_COMPILER_ID|CMAKE_CXX_COMPILER_VERSION) "([^"]*)"\)$')
    for compiler_file in sorted((build_dir / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake")):
        for line in compiler_file.read_text(encoding="utf-8", errors="replace").splitlines():
            match = setter.match(line.strip())
            if match:
                values.setdefault(match.group(1), match.group(2))
    return values


def require_static_crt(choice: bool | None, sys_platform: str = sys.platform) -> bool:
    """Resolve the ``--require-static-crt`` tri-state: default on for Windows."""

    if choice is None:
        return sys_platform == "win32"
    return choice


def library_import_facts(library: Path) -> dict[str, Any] | None:
    """PE import facts for a Windows DLL; ``None`` for Mach-O/ELF libraries.

    The record is written into the build report so a receipt can quote what the
    kernel library will load at run time, and ``static_crt_violation`` decides
    whether that closure is acceptable for a redistributable build.
    """

    if not is_pe_image(library):
        return None
    facts = read_pe_imports(library)
    names = facts.all_imports
    return {
        "format": facts.format,
        "machine": facts.machineName,
        "isDll": facts.isDll,
        "imports": list(facts.imports),
        "delayImports": list(facts.delayImports),
        "crtLinkage": crt_linkage(names),
        "dynamicCrtImports": list(dynamic_crt_imports(names)),
        "ucrtImports": list(ucrt_imports(names)),
    }


def static_crt_violation(import_facts: dict[str, Any] | None) -> str | None:
    """Explain why the DLL cannot ship, or ``None`` when the CRT is static."""

    if import_facts is None:
        return None
    offenders = import_facts["dynamicCrtImports"]
    if not offenders:
        return None
    return (
        "the native kernel DLL imports the dynamic Visual C++ runtime ("
        + ", ".join(offenders)
        + "); it must be built with MSVC_RUNTIME_LIBRARY=MultiThreaded (/MT) so the "
        "frozen engine runs on machines without the VC++ redistributable. Rebuild "
        "from a clean build directory after checking native/CMakeLists.txt."
    )


def remove_stale_libraries(runtime_dir: Path | None = None) -> list[str]:
    # Resolved at call time: a default bound at import time would ignore a
    # redirected RUNTIME_DIR and remove the library of the real package tree.
    runtime_dir = RUNTIME_DIR if runtime_dir is None else runtime_dir
    removed: list[str] = []
    for name in LIBRARY_NAMES:
        candidate = runtime_dir / name
        if candidate.exists() or candidate.is_symlink():
            candidate.unlink()
            removed.append(name)
    return removed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build" / "native-release")
    parser.add_argument("--metal", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--generator", default=None, help="CMake generator (default: CMake's choice)")
    parser.add_argument("--architecture", default=None, help="CMake -A value for multi-config generators (e.g. x64)")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--skip-tests", action="store_true", help="build without ctest")
    parser.add_argument("--no-install", action="store_true", help="do not install into the Python package")
    parser.add_argument("--report", type=Path, default=None, help="write a JSON build report")
    parser.add_argument(
        "--require-static-crt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "fail when the installed Windows DLL imports MSVCP140/VCRUNTIME140 "
            "(default: on for Windows hosts, not applicable elsewhere)"
        ),
    )
    parser.add_argument("--cmake", default=shutil.which("cmake") or "cmake")
    parser.add_argument("--ctest", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    build_dir = arguments.build_dir.resolve()
    metal = metal_enabled(arguments.metal)
    ctest = arguments.ctest or str(Path(arguments.cmake).with_name("ctest" + Path(arguments.cmake).suffix))
    if not Path(ctest).exists():
        ctest = shutil.which("ctest") or "ctest"
    timings: dict[str, float] = {}
    timings["configure"] = _run(
        configure_command(
            build_dir,
            metal=metal,
            tests=not arguments.skip_tests,
            generator=arguments.generator,
            architecture=arguments.architecture,
            cmake=arguments.cmake,
        )
    )
    timings["build"] = _run(build_command(build_dir, jobs=arguments.jobs, cmake=arguments.cmake))
    if not arguments.skip_tests:
        timings["ctest"] = _run(test_command(build_dir, ctest=ctest))
    installed: Path | None = None
    if not arguments.no_install:
        removed = remove_stale_libraries(RUNTIME_DIR)
        if removed:
            print("removed stale runtime libraries: " + ", ".join(removed), flush=True)
        timings["install"] = _run(install_command(build_dir, cmake=arguments.cmake))
        installed = RUNTIME_DIR / host_library_name()
        if not installed.is_file():
            print(f"error: install did not produce {installed}", file=sys.stderr)
            return 1
    cache = cmake_cache(build_dir)
    report = {
        "schemaVersion": 1,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "buildDir": str(build_dir),
        "generator": cache.get("CMAKE_GENERATOR"),
        "compilerId": cache.get("CMAKE_CXX_COMPILER_ID"),
        "compilerVersion": cache.get("CMAKE_CXX_COMPILER_VERSION"),
        "compiler": cache.get("CMAKE_CXX_COMPILER"),
        "metal": metal,
        "tests": not arguments.skip_tests,
        "library": None,
        "timingsSeconds": {key: round(value, 3) for key, value in timings.items()},
    }
    violation: str | None = None
    if installed is not None:
        report["library"] = {
            "path": str(installed),
            "fileName": installed.name,
            "sizeBytes": installed.stat().st_size,
            "sha256": "sha256:" + _sha256(installed),
        }
        try:
            import_facts = library_import_facts(installed)
        except PEFormatError as error:
            print(f"error: cannot parse the installed DLL's import table: {error}", file=sys.stderr)
            return 1
        if import_facts is not None:
            report["library"].update(import_facts)
            if require_static_crt(arguments.require_static_crt):
                violation = static_crt_violation(import_facts)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text, flush=True)
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(text + "\n", encoding="utf-8")
    if violation is not None:
        # The report is written first so the failing import list is preserved.
        print("error: " + violation, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
