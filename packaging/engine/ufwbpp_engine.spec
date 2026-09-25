# -*- mode: python ; coding: utf-8 -*-
"""One-directory worker spec used by ``scripts/build_engine_sidecar.py``.

Run the build script rather than invoking this file directly. The script owns
target validation, preflight smoke tests, create-only publication, and the
release manifest.
"""

from pathlib import Path
import os
import runpy
import sys

from PyInstaller.utils.hooks import collect_data_files, copy_metadata


spec_dir = Path(SPECPATH).resolve()
repo_root = spec_dir.parents[1]
policy = runpy.run_path(str(spec_dir / "resource_policy.py"))
filter_entries = policy["filter_pyinstaller_entries"]
required_metadata_distributions = policy["REQUIRED_METADATA_DISTRIBUTIONS"]
checked_catalog_resources = policy["CHECKED_CATALOG_RESOURCES"]
CATALOG_MANIFEST_DIRECTORY = policy["CATALOG_MANIFEST_DIRECTORY"]

launcher = spec_dir / "launcher.py"
search_paths = [
    repo_root / "packages" / "engine" / "src",
    repo_root / "packages" / "light-frame-qc" / "src",
    repo_root / "packages" / "registration" / "src",
]

datas = filter_entries(
    collect_data_files(
        "ufwbpp",
        includes=["native/*", "native/**/*", "py.typed"],
    )
)
binaries = []
if sys.platform == "win32":
    # The native kernel DLL is a real binary on Windows: collecting it as a
    # binary lets PyInstaller walk its import table like every other DLL in
    # the tree instead of copying it blindly as data. The DLL links the C
    # runtime statically (native/CMakeLists.txt, enforced by
    # scripts/build_native_runtime.py --require-static-crt), so that walk finds
    # only Windows system DLLs; the release attestation re-checks the whole
    # frozen tree's PE import closure against a clean machine's DLL set.
    native_dlls = [
        entry for entry in datas if entry[0].lower().endswith(".dll") and entry[1].replace("\\", "/").startswith("ufwbpp/native")
    ]
    datas = [entry for entry in datas if entry not in native_dlls]
    binaries.extend(native_dlls)
datas.extend(
    filter_entries(
        [
            (str(repo_root / "packages" / "engine" / "src" / CATALOG_MANIFEST_DIRECTORY / name), CATALOG_MANIFEST_DIRECTORY)
            for name in checked_catalog_resources
        ]
    )
)
# The launcher import graph owns ordinary modules. Keep this list limited to
# real runtime imports performed through importlib or image-format registries;
# broad collect_all/collect_submodules calls made the first launch validate
# hundreds of Astropy/Pillow modules that the product can never execute.
hidden_imports = [
    "reproject.mosaicking",
    "reproject.interpolation.high_level",
    "shapely",
    "numpy.lib.recfunctions",
    "PIL.PngImagePlugin",
    "PIL.TiffImagePlugin",
]
excluded_imports = [
    # The headless worker never renders WCSAxes. On Windows, importing this
    # optional UI package during binary-dependency analysis raises a pytest
    # Skip when Matplotlib is intentionally absent.
    "astropy.visualization",
    "astropy.visualization.wcsaxes",
]
for distribution_name in required_metadata_distributions:
    datas.extend(filter_entries(copy_metadata(distribution_name)))

# Stable ordering makes the analysis input auditable and avoids accidental
# platform-dependent ordering in TOCs. PyInstaller itself still determines the
# final bootloader representation for the current host.
datas = sorted(set(datas), key=lambda item: (item[1], item[0]))
binaries = sorted(set(binaries), key=lambda item: (item[1], item[0]))
hidden_imports = sorted(set(hidden_imports))
excluded_imports = sorted(set(excluded_imports))

worker_name = os.environ.get("UFWBPP_WORKER_BASENAME")
if not worker_name or "/" in worker_name or "\\" in worker_name:
    raise SystemExit("UFWBPP_WORKER_BASENAME must be a portable file basename")

analysis = Analysis(
    [str(launcher)],
    pathex=[str(path) for path in search_paths],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[str(spec_dir / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_imports,
    noarchive=False,
    optimize=1,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name=worker_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name=worker_name,
)
