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
require them to be value-identical to the NumPy reference on Windows. The DLL
links the C runtime statically; on Windows the script parses its PE import
table, records `library.crtLinkage` and the import list in the build report,
and fails when `MSVCP140.dll`/`VCRUNTIME140*.dll` are imported
(`--require-static-crt`, on by default on Windows).

## Patched SEP wheel

The PyPI `sep` 1.4.1 wheel is not deterministic on MSVC builds and its
extraction time grows super-linearly with the object count. Windows machines
install the patched build before running the tests (CI does the same):

```powershell
.venv\Scripts\python scripts\build_sep_wheel.py --install
.venv\Scripts\python -c "import sep; print(sep.__version__)"   # 1.4.1+oaf.1
```

The script downloads the hash-pinned sdist, applies
`packaging/patches/sep-1.4.1-zero-initialised-buffers.patch`, builds the wheel
with MSVC (build isolation on) and proves in a fresh interpreter that repeated
deblending extractions are identical. `--check` only downloads and patches.

## Installer gate

`attest-installed-msi.ps1` is the release gate the workflow runs on the MSI:
silent install, layout check, PE import-closure and launch attestation of the
installed worker tree, silent uninstall, no leftovers. It runs on a laptop
from an elevated shell against a locally built bundle:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\attest-installed-msi.ps1 `
  -Msi "target\release\bundle\msi\Ultra-Fast WBPP_0.1.0_x64_en-US.msi" `
  -Target x86_64-pc-windows-msvc `
  -Python .venv\Scripts\python.exe `
  -Output build\bundle-attestations\openastroflow-worker-x86_64-pc-windows-msvc.bundled.manifest.json
```

Add `-KeepInstalled` to leave the application installed for a GUI run.
