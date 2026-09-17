# Windows build machine scripts

`bootstrap.ps1` installs the developer toolchain for a Windows x86-64 build or
test machine with winget (Python 3.12, CMake, Ninja, Visual Studio 2022 Build
Tools with the C++ workload, uv; `-WithRust` adds rustup) and writes a JSON
report of every tool and version it found or installed. Run it from an elevated
PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\bootstrap.ps1
```

Then, in a new shell (package installs do not update the calling shell's
PATH):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ./packages/light-frame-qc[test] -e ./engine/native/python[test] -e ./packages/openastroflow-engine[test,all]
.venv\Scripts\python scripts\build_native_runtime.py --build-dir build\native-release
.venv\Scripts\python -m pytest -q packages/light-frame-qc/tests engine/native/python/tests packages/openastroflow-engine/tests tests
```

`build_native_runtime.py` uses CMake's Visual Studio generator, which locates
the Build Tools through `vswhere` and needs no developer prompt. The native
kernels are compiled with `/W4 /WX /fp:strict` and the differential tests
require them to be value-identical to the NumPy reference on Windows.
