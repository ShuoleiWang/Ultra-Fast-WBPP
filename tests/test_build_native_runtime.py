from __future__ import annotations

from pathlib import Path

from scripts import build_native_runtime as native_build


def test_host_library_name_follows_the_platform_service_layer() -> None:
    assert native_build.host_library_name("darwin") == "libopenastroflow_native.dylib"
    assert native_build.host_library_name("win32") == "openastroflow_native.dll"
    assert native_build.host_library_name("linux") == "libopenastroflow_native.so"


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
    assert "-DOAF_BUILD_TESTS=ON" in configure
    assert "-DOAF_ENABLE_METAL=OFF" in configure
    assert "-DCMAKE_BUILD_TYPE=Release" in configure
    vs = native_build.configure_command(
        build_dir, metal=False, tests=False, generator=None, architecture="x64"
    )
    assert vs[vs.index("-A") + 1] == "x64"
    assert "-DOAF_BUILD_TESTS=OFF" in vs
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
    (runtime / "openastroflow_native.dll").write_bytes(b"x")
    (runtime / "README.md").write_text("keep", encoding="utf-8")
    assert native_build.remove_stale_libraries(runtime) == ["openastroflow_native.dll"]
    assert (runtime / "README.md").exists()
    assert native_build.remove_stale_libraries(runtime) == []
