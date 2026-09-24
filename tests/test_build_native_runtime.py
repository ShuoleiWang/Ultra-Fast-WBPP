from __future__ import annotations

import json
from pathlib import Path

from pe_fixtures import build_pe_image
from scripts import build_native_runtime as native_build


def test_host_library_name_follows_the_platform_service_layer() -> None:
    assert native_build.host_library_name("darwin") == "libufwbpp_native.dylib"
    assert native_build.host_library_name("win32") == "ufwbpp_native.dll"
    assert native_build.host_library_name("linux") == "libufwbpp_native.so"


def test_metal_defaults_to_the_host_and_is_overridable() -> None:
    assert native_build.metal_enabled("auto", "darwin") is True
    assert native_build.metal_enabled("auto", "win32") is False
    assert native_build.metal_enabled("auto", "linux") is False
    assert native_build.metal_enabled("on", "win32") is True
    assert native_build.metal_enabled("off", "darwin") is False


def test_commands_are_release_strict_and_explicit(tmp_path: Path) -> None:
    build_dir = tmp_path / "b"
    configure = native_build.configure_command(
        build_dir, metal=False, tests=True, generator="Ninja", architecture=None, cmake="cmake"
    )
    assert configure[:5] == ["cmake", "-S", str(native_build.NATIVE_SOURCE), "-B", str(build_dir)]
    assert "-G" in configure and "Ninja" in configure
    assert "-A" not in configure
    assert "-DUFWBPP_BUILD_TESTS=ON" in configure
    assert "-DUFWBPP_ENABLE_METAL=OFF" in configure
    assert "-DCMAKE_BUILD_TYPE=Release" in configure
    vs = native_build.configure_command(
        build_dir, metal=False, tests=False, generator=None, architecture="x64"
    )
    assert vs[vs.index("-A") + 1] == "x64"
    assert "-DUFWBPP_BUILD_TESTS=OFF" in vs
    build = native_build.build_command(build_dir, jobs=4)
    assert build == ["cmake", "--build", str(build_dir), "--config", "Release", "--parallel", "4"]
    assert native_build.test_command(build_dir)[-3:] == ["-C", "Release", "--output-on-failure"]
    install = native_build.install_command(build_dir, prefix=tmp_path)
    assert install[-2:] == ["--prefix", str(tmp_path)]


def test_cmake_cache_parser_and_stale_library_removal(tmp_path: Path) -> None:
    (tmp_path / "CMakeCache.txt").write_text(
        "# comment\nCMAKE_CXX_COMPILER_ID:STRING=MSVC\nCMAKE_GENERATOR:INTERNAL=Visual Studio 17 2022\nbad line\n",
        encoding="utf-8",
    )
    compiler_dir = tmp_path / "CMakeFiles" / "4.4.3"
    compiler_dir.mkdir(parents=True)
    (compiler_dir / "CMakeCXXCompiler.cmake").write_text(
        'set(CMAKE_CXX_COMPILER "cl.exe")\nset(CMAKE_CXX_COMPILER_ID "MSVC")\n'
        'set(CMAKE_CXX_COMPILER_VERSION "19.44.35229.0")\n',
        encoding="utf-8",
    )
    cache = native_build.cmake_cache(tmp_path)
    assert cache["CMAKE_CXX_COMPILER_ID"] == "MSVC"
    assert cache["CMAKE_CXX_COMPILER_VERSION"] == "19.44.35229.0"
    assert cache["CMAKE_GENERATOR"] == "Visual Studio 17 2022"
    assert native_build.cmake_cache(tmp_path / "missing") == {}
    runtime = tmp_path / "native"
    runtime.mkdir()
    (runtime / "ufwbpp_native.dll").write_bytes(b"x")
    (runtime / "README.md").write_text("keep", encoding="utf-8")
    assert native_build.remove_stale_libraries(runtime) == ["ufwbpp_native.dll"]
    assert (runtime / "README.md").exists()
    assert native_build.remove_stale_libraries(runtime) == []


def test_static_crt_requirement_defaults_to_windows_hosts() -> None:
    assert native_build.require_static_crt(None, "win32") is True
    assert native_build.require_static_crt(None, "darwin") is False
    assert native_build.require_static_crt(None, "linux") is False
    assert native_build.require_static_crt(False, "win32") is False
    assert native_build.require_static_crt(True, "darwin") is True


def test_library_import_facts_record_the_dll_closure_and_crt_linkage(tmp_path: Path) -> None:
    dynamic = tmp_path / "dynamic.dll"
    dynamic.write_bytes(
        build_pe_image(
            imports=(
                "KERNEL32.dll",
                "MSVCP140.dll",
                "VCRUNTIME140.dll",
                "VCRUNTIME140_1.dll",
                "api-ms-win-crt-runtime-l1-1-0.dll",
                "api-ms-win-crt-math-l1-1-0.dll",
            ),
            delay_imports=("ADVAPI32.dll",),
        )
    )
    static = tmp_path / "static.dll"
    static.write_bytes(build_pe_image(imports=("KERNEL32.dll",)))
    dylib = tmp_path / "libufwbpp_native.dylib"
    dylib.write_bytes(b"\xcf\xfa\xed\xfe" + b"\0" * 64)

    facts = native_build.library_import_facts(dynamic)
    assert facts is not None
    assert facts["format"] == "PE32+"
    assert facts["machine"] == "x86_64"
    assert facts["isDll"] is True
    assert facts["crtLinkage"] == "dynamic"
    assert facts["dynamicCrtImports"] == ["MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll"]
    assert facts["ucrtImports"] == [
        "api-ms-win-crt-math-l1-1-0.dll",
        "api-ms-win-crt-runtime-l1-1-0.dll",
    ]
    assert facts["delayImports"] == ["ADVAPI32.dll"]
    violation = native_build.static_crt_violation(facts)
    assert violation is not None
    assert "MSVCP140.dll, VCRUNTIME140.dll, VCRUNTIME140_1.dll" in violation
    assert "MultiThreaded" in violation

    static_facts = native_build.library_import_facts(static)
    assert static_facts is not None
    assert static_facts["crtLinkage"] == "static"
    assert static_facts["imports"] == ["KERNEL32.dll"]
    assert native_build.static_crt_violation(static_facts) is None
    # Mach-O/ELF libraries have no PE import table and no CRT verdict.
    assert native_build.library_import_facts(dylib) is None
    assert native_build.static_crt_violation(None) is None


def _last_json(capsys) -> dict:
    """The build script prints commands first and the JSON report last."""

    output = capsys.readouterr().out
    start = output.index("{\n")
    return json.loads(output[start:])


def _fake_build_chain(monkeypatch, runtime_dir: Path, dll_bytes: bytes) -> list[list[str]]:
    """Replace cmake/ctest with a recorder; the install step drops the DLL."""

    commands: list[list[str]] = []

    def run(command, *, cwd=None):
        commands.append(list(command))
        if "--install" in command:
            runtime_dir.mkdir(parents=True, exist_ok=True)
            (runtime_dir / "ufwbpp_native.dll").write_bytes(dll_bytes)
        return 0.01

    monkeypatch.setattr(native_build, "_run", run)
    monkeypatch.setattr(native_build, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(native_build, "host_library_name", lambda sys_platform=None: "ufwbpp_native.dll")
    return commands


def test_windows_build_report_records_imports_and_fails_on_dynamic_crt(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    runtime_dir = tmp_path / "native"
    report_path = tmp_path / "report.json"
    dynamic = build_pe_image(imports=("KERNEL32.dll", "VCRUNTIME140.dll", "MSVCP140.dll"))
    commands = _fake_build_chain(monkeypatch, runtime_dir, dynamic)

    code = native_build.main(
        ["--build-dir", str(tmp_path / "b"), "--skip-tests", "--report", str(report_path), "--require-static-crt"]
    )

    assert code == 1
    captured = capsys.readouterr()
    assert "dynamic Visual C++ runtime (MSVCP140.dll, VCRUNTIME140.dll)" in captured.err
    # The report is still written so the offending import list is preserved.
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["library"]["crtLinkage"] == "dynamic"
    assert report["library"]["imports"] == ["KERNEL32.dll", "MSVCP140.dll", "VCRUNTIME140.dll"]
    assert report["library"]["dynamicCrtImports"] == ["MSVCP140.dll", "VCRUNTIME140.dll"]
    assert report["library"]["sha256"].startswith("sha256:")
    assert [command[1] for command in commands] == ["-S", "--build", "--install"]

    # Without the gate the same DLL is only recorded, never rejected.
    assert native_build.main(["--build-dir", str(tmp_path / "b"), "--skip-tests", "--no-require-static-crt"]) == 0
    recorded = _last_json(capsys)
    assert recorded["library"]["crtLinkage"] == "dynamic"


def test_windows_build_report_marks_static_crt_as_compliant(tmp_path: Path, monkeypatch, capsys) -> None:
    runtime_dir = tmp_path / "native"
    static = build_pe_image(imports=("KERNEL32.dll",))
    _fake_build_chain(monkeypatch, runtime_dir, static)

    assert native_build.main(["--build-dir", str(tmp_path / "b"), "--skip-tests", "--require-static-crt"]) == 0
    report = _last_json(capsys)
    assert report["library"]["crtLinkage"] == "static"
    assert report["library"]["imports"] == ["KERNEL32.dll"]
    assert report["library"]["dynamicCrtImports"] == []
    assert report["library"]["machine"] == "x86_64"


def test_install_prefix_is_the_engine_source_root() -> None:
    # The library is installed next to the package that loads it; a wrong
    # prefix installs it where no import ever looks.
    assert (native_build.INSTALL_PREFIX / "ufwbpp" / "native_kernels.py").is_file()
    assert native_build.RUNTIME_DIR == native_build.INSTALL_PREFIX / "ufwbpp" / "native"
