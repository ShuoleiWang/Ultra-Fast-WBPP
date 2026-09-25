# Building from source

This guide takes a fresh clone to a Mac app you can run or hand to someone else. There are three things you can build:

| Build | Command | What the person who runs it needs |
|---|---|---|
| Engine only (command line) | `make bootstrap` | the clone and its `.venv` |
| Desktop app | `make desktop-build-macos-prerelease` | Apple Silicon, macOS 14+, and a plate solver they install themselves ([solver setup](recipes/offline-solver-catalogs.md)) |
| Self-contained demo app | `make desktop-build-macos-demo` | Apple Silicon and macOS 14+. Nothing else: `solve-field` and its index set are inside the app |

The demo app is for private demonstrations only. It bundles Astrometry.net, which is GPL-3.0-or-later as distributed, and index files whose redistribution terms are unresolved. Do not publish it ([licensing](licensing.md)). Windows builds are covered in [windows.md](windows.md#developing-on-windows). The demo build is macOS-only.

## What the build machine needs

- A Mac with Apple Silicon on macOS 14 or later, with about 8 GB of free disk and network access to PyPI, npm, crates.io and `ghcr.io`.
- Xcode, or at least its Command Line Tools (`xcode-select --install`). The steps below were followed in a fresh clone on an M3 Pro with Xcode 27 on macOS 27. With the download caches already warm, bootstrap took 61 s and the first demo build 5 min 12 s. A first build on a new machine also downloads the Python wheels, npm packages and Rust crates.
- [Homebrew](https://brew.sh), then:

  ```bash
  brew install python@3.12 cmake node rust
  ```

  The app bundles a Python runtime, which the release build freezes from Homebrew's `python@3.12`. A pinned macOS 14 compatibility layer replaces that interpreter's OpenSSL and mpdecimal libraries so the app also runs on macOS 14. The python.org installer's Python cannot take that layer, because it bundles its own OpenSSL. Other versions are untested; Homebrew's default `python3` is 3.14 at the time of writing. Rust 1.88 or newer and Node.js 22 or newer are required; `rustup` works as well as Homebrew's `rust`.

## 1. Clone and bootstrap

```bash
git clone https://github.com/ShuoleiWang/Ultra-Fast-WBPP.git
cd Ultra-Fast-WBPP
make bootstrap BOOTSTRAP_PYTHON=python3.12
```

`make bootstrap` creates `.venv` from that interpreter and installs the three Python packages in editable mode with the packaging tools. It then runs `npm ci` for the desktop and builds, tests and installs the optimized native kernels (`make native-release-install`). `make test` runs every suite (Python, Rust, frontend and native). Check the result with `.venv/bin/ultra-fast-wbpp doctor`.

## 2. Build the desktop app

```bash
CARGO_PROFILE_RELEASE_STRIP=none make desktop-build-macos-prerelease
```

This chain builds, tests and installs the Release native kernels. It then fetches the pinned macOS 14 runtime libraries and freezes the engine with PyInstaller. After that it stages the engine for Tauri and builds an ad-hoc signed app at `target/release/bundle/macos/Ultra-Fast WBPP.app`. `CARGO_PROFILE_RELEASE_STRIP=none` is required on macOS 27 with Xcode 27, where a stripped build fails with `error[E0463]`, and is harmless elsewhere.

People who run this app set up a plate solver themselves: Astrometry.net `solve-field`, plus the index set from the app's solver panel.

## 3. Build the self-contained demo app

The demo build copies the index set from your own catalog directory, so install it once. It is 333.5 MiB from `data.astrometry.net`. `catalog list` prints each catalog's terms and acceptance ID. Read the terms of `astrometry-net-4107-4112`, then pass its acceptance ID:

```bash
.venv/bin/ultra-fast-wbpp catalog list
.venv/bin/ultra-fast-wbpp catalog install astrometry-net-4107-4112 --accept-provider-terms astrometry-net-index-data-2026-09
```

The files go to `~/.ultra-fast-wbpp/catalogs/astrometry-net`, or to `~/.openastroflow/catalogs/astrometry-net` when only that older directory exists. Then build:

```bash
CARGO_PROFILE_RELEASE_STRIP=none make desktop-build-macos-demo
```

This is the desktop build plus one staging step, `scripts/stage_astrometry_runtime.py`, which fills `apps/desktop/src-tauri/resources/astrometry-net` (339 MiB):

- It downloads the macOS 14 (`arm64_sonoma`) Homebrew bottles of astrometry-net 0.97, GSL, WCSLIB and CFITSIO into `build/astrometry-bottles` and checks each against a pinned SHA-256. The bottles on your own Mac usually require a newer macOS.
- It relocates `solve-field`, `astrometry-engine` and the four libraries so they load from inside the app, and re-signs them.
- Upstream's two Python helpers, `removelines` and `uniformize`, become small wrappers that run inside the bundled engine.
- It copies the six index files after checking them against the catalog manifest.

The app is about 500 MB. On first launch the engine installs the bundled index set into the user's catalog directory. The copy is an APFS clone, so it takes no extra space when the app sits in Applications. The run then goes through the same hash checks and receipt as a download. On the NGC 7331 reference data the demo app's masters are bit-identical to those of a Homebrew-solver install (see the [changelog](../CHANGELOG.md)).

## 4. Check the result

```bash
.venv/bin/python scripts/attest_bundled_runtime.py \
  --app "target/release/bundle/macos/Ultra-Fast WBPP.app" \
  --target aarch64-apple-darwin --output "$(mktemp -d)/attestation.json"
codesign --verify --deep --strict "target/release/bundle/macos/Ultra-Fast WBPP.app"
```

The attestation checks the frozen engine's identity, signature, legal files and launch time. For the demo app, you can also check that the bundled engine solves with nothing from Homebrew and no installed catalog. The sandbox below hides Homebrew, and the empty `HOME` has no catalog:

```bash
engine="target/release/bundle/macos/Ultra-Fast WBPP.app/Contents/Resources/resources/ufwbpp-engine/ufwbpp-engine-aarch64-apple-darwin/ufwbpp-engine-aarch64-apple-darwin"
sandbox-exec -p '(version 1)(allow default)(deny file-read* (subpath "/opt/homebrew"))(deny file-read* (subpath "/usr/local"))' \
  env -i HOME="$(mktemp -d)" PATH=/usr/bin:/bin "$engine" doctor --json | grep -o '"solverReady": *[a-z]*'
```

It prints `"solverReady": true`. The same sandbox and `run-project` give a full run on a real data set.

## 5. Share the app

```bash
ditto -c -k --sequesterRsrc --keepParent "target/release/bundle/macos/Ultra-Fast WBPP.app" Ultra-Fast-WBPP-macOS-arm64.zip
```

The demo app compresses to about 300 MB. The recipient needs Apple Silicon and macOS 14 or later, and moves the app to Applications. The app is ad-hoc signed and not notarized, so macOS refuses the first launch. They open System Settings → Privacy & Security and choose **Open Anyway**, once. Or they run `xattr -dr com.apple.quarantine "/Applications/Ultra-Fast WBPP.app"`. Only Developer ID signing plus notarization removes that step ([release process](release-process.md#signing-and-publication)).

## Rebuilding

The staged intermediates are create-only, so a second build refuses to replace them (`OUTPUT_EXISTS` or a `refusing to replace …` message). Delete them before rebuilding. They are build outputs only:

```bash
rm -rf build/sidecars/ufwbpp-engine-aarch64-apple-darwin build/sidecars/ufwbpp-engine-aarch64-apple-darwin.manifest.json \
  apps/desktop/src-tauri/resources/ufwbpp-engine apps/desktop/src-tauri/resources/astrometry-net
```

The downloaded bottles in `build/astrometry-bottles` and `build/macos14-runtime-libraries` are kept and re-verified, and the Rust build is incremental. A rebuild takes a few minutes. `make clean` removes these intermediates too, but also the rest of `build/`, including the downloaded bottles.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `error[E0463]: can't find crate` while Tauri builds | macOS 27 / Xcode 27: prefix the build with `CARGO_PROFILE_RELEASE_STRIP=none` |
| The engine freeze fails its runtime smoke (OpenSSL or libmpdec version) | `.venv` was not created from Homebrew's `python@3.12`: `rm -rf .venv` and bootstrap again with `BOOTSTRAP_PYTHON=python3.12` |
| `OUTPUT_EXISTS` or `refusing to replace …` | Delete the intermediates listed under [Rebuilding](#rebuilding) |
| `stage_astrometry_runtime: … does not match the checked manifest` | The index set is missing or incomplete: `ultra-fast-wbpp catalog verify astrometry-net-4107-4112` shows which file |
| Runs are several times slower than expected | The unoptimized kernels are installed: `make native-release-install` |
| The first solve after a new build takes a few seconds longer | macOS scans newly signed binaries the first time they run; later runs are normal |
