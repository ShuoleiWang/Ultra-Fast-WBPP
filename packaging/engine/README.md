# Engine sidecar packaging

The desktop app launches a target-suffixed, one-directory frozen engine (`ufwbpp-engine`). Build it
from a native Python environment for the target operating system; PyInstaller
does not cross-compile between macOS and Windows.

```bash
python -m pip install -r packaging/engine/requirements-build.txt
python scripts/build_engine_sidecar.py \
  --target aarch64-apple-darwin \
  --output-dir build/sidecars
```

An Apple Silicon release that declares macOS 14 support must be built from
native libraries whose Mach-O deployment targets are also macOS 14 or older.
The builder never resolves a mutable `latest` dependency. Materialize the
Homebrew OCI layout pinned by
[`macos14-runtime-libraries-v1.json`](macos14-runtime-libraries-v1.json), then
pass that immutable input to the sidecar builder:

```bash
python scripts/fetch_macos14_runtime_libraries.py \
  --output build/macos14-runtime-libraries
python scripts/build_engine_sidecar.py \
  --target aarch64-apple-darwin \
  --output-dir build/sidecars-macos14 \
  --macos14-bottle-root build/macos14-runtime-libraries
```

The fetcher starts only at the policy's HTTPS `ghcr.io` endpoints, requests
each manifest by SHA-256 digest, and uses anonymous bearer tokens scoped to one
exact repository. At most three HTTPS redirects are accepted, only to the
explicit GitHub-container/Azure-blob host allowlist; `Authorization` is removed
before every cross-origin request. The final bytes remain trusted only after
their pinned size and SHA-256 pass. The output is create-only, and extraction
reads only the three named regular-file members instead of unpacking the bottle.

The policy fixes the registry, repositories, Sonoma bottle tag, formula
version, source revision, license, OCI manifest/config/blob SHA-256 and size,
and each extracted dylib's SHA-256 and size. The builder accepts exactly `libcrypto.3.dylib`,
`libssl.3.dylib`, and `libmpdec.4.dylib`; it performs normal install-name/rpath
relocation and ad-hoc signing, never deployment-command patching. It then
launches a hidden frozen-runtime smoke that imports `ssl`, `decimal`, and
`ctypes`, records Python/OpenSSL/mpdecimal versions, and requires the pinned
ABI before publishing the create-only tree. A path-free provenance record is
embedded in `_internal` and covered by the runtime manifest.

On Apple Silicon, omitting `--target` selects `aarch64-apple-darwin`. Windows
x64 selects `x86_64-pc-windows-msvc`. A requested target must equal the target
of the running Python interpreter.

Before PyInstaller runs, the builder runs the source launcher's
`doctor --json` self-report (engine version, pixel executors ready, native
kernel facts) and the repository public-tree audit. It then freezes `ufwbpp`,
`lightframeqc`, `ufwbpp_registration`, `astropy`, `reproject`, and `shapely`,
omitting tests, downloaded catalogs, raw astronomical frames, and user data.
The frozen runtime must pass the same self-report, report the same engine
version and load its native kernels before it can be published.

The executable is the `ultra-fast-wbpp` command line; the desktop runs one
command per operation (`doctor`, `inventory`, `calibration-check`,
`quality-check`, `blink-measure`, `run-project`, `catalog`). Without a
command it prints its usage and exits with status 2.

Successful output contains one create-only runtime tree and one manifest:

```text
ufwbpp-engine-aarch64-apple-darwin/
├── ufwbpp-engine-aarch64-apple-darwin
└── _internal/
ufwbpp-engine-aarch64-apple-darwin.manifest.json
```

Windows adds `.exe` to the entry point, while the directory and manifest keep
the same target-triple stem. The v3 manifest records every relative file,
directory and safe in-tree symlink, every file SHA-256 and size, executable
bits, and a deterministic whole-tree digest. It also records the engine version and
exact Python/PyInstaller/scientific package versions. It contains no build
timestamp, source path, catalog location, or user-data path. Its checked-in
contract is [`sidecar-manifest-v3.schema.json`](sidecar-manifest-v3.schema.json).

`scripts/stage_tauri_sidecar.py` verifies the complete tree again and stages it
create-only under `apps/desktop/src-tauri/resources/ufwbpp-engine/`.
Tauri copies that directory to application Resources. The GUI launches files
in place; there is no per-launch extraction and users do not install Python.

The builder never replaces an existing target. Delete or archive an obsolete
release artifact deliberately before rebuilding it; there is no `--force`
escape hatch.

`make desktop-build` performs the pinned fetch automatically on macOS and
passes the resulting root to the sidecar builder. Windows builds do neither;
they continue through the platform-native build path.
