# Ultra-Fast WBPP

**Turn a night—or several nights—of astrophotography into calibrated, aligned, stacked images.**

A local Mac application for monochrome astrophotography: drop in Light folders and calibration frames, review questionable exposures, then produce linear FITS masters and previews. Original files stay read-only.

[简体中文](docs/README.zh-CN.md) · [User guide](docs/recipes/automatic-screening.md) · [Validation](docs/validation-matrix.md) · [Contributing](CONTRIBUTING.md)

![Ultra-Fast WBPP: real native application showing completed B, G, L, R and RGB products](assets/branding/ultra-fast-wbpp-results.jpg)

*A completed run in the native macOS app. [Import view](assets/branding/ultra-fast-wbpp-table.jpg) · [Screening view](assets/branding/ultra-fast-wbpp-screening.jpg) (synthetic test frames).*

## What it does

- **Import together:** N.I.N.A. folders from different nights, FITS/mono XISF files, raw Flat/Dark/Bias frames, and existing calibration masters.
- **Review before processing:** metadata grouping, calibration matching, and checks for severe focus, tracking, cloud, obstruction, and field problems. Questionable frames remain excluded until reviewed; definite failures stay excluded.
- **Process locally:** build calibration masters, calibrate Lights, register across rotations and meridian flips, reject outlier pixels and detected transient trails, integrate, and crop to common coverage.
- **Export:** linear monochrome and RGB/LRGB FITS, inspection previews, quality evidence, and processing records. The desktop verifies final sky coordinates before showing success.

## Status and requirements

**Alpha, under active development. There is no public installer yet.** The current target is Apple Silicon on macOS 14+. Packaged real-data runs have been checked on an M3 Pro; other Macs and macOS 14 hardware still need acceptance testing. Local builds are ad-hoc signed, not Developer ID signed or notarized.

The main workflow is **monochrome**. OSC/Bayer processing is not supported. Drizzle, LocalNormalization, RGB/LRGB, and mosaics need broader real-data validation. LocalNormalization is not a guarantee of gradient-free output; stacked images may still need background modeling in your image editor. Windows has development/test targets, but no supported installer.

Final astrometric solving requires a separately installed **Astrometry.net `solve-field`** and local indexes. Follow [solver setup](docs/recipes/offline-solver-catalogs.md) and use **Recheck setup** in the app. Allow disk space for full-resolution intermediate frames.

## Quick start from source

On macOS, install Python 3.11+, Rust 1.88+, Node.js 22+, CMake 3.28+, and the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/) including Xcode. From this checkout:

```bash
make bootstrap
make desktop-dev
```

1. **Import** all Light and calibration folders. Check detected frame types, filters, and calibration matches.
2. **Review** the Light quality results and inspect questionable frames.
3. **Process** into a new output location. Open the FITS masters and previews after completion.

The interface supports English and 简体中文. Existing output directories are never replaced. See [N.I.N.A. inputs](docs/recipes/nina-mono.md) and [calibration conventions](docs/recipes/calibration.md) for missing metadata and master reuse.

To build a local `.app` with its Python runtime bundled:

```bash
make desktop-build-macos-prerelease
```

This also downloads pinned macOS runtime build dependencies. Signing, DMG creation, and release checks are documented in [release packaging](docs/release-process.md). Installed bundles do not need user Python, Node.js, or Rust.

The same engine is available from the command line:

```bash
.venv/bin/ultra-fast-wbpp run /data/lights /data/calibration --recipe docs/recipes/mono-standard.json --output /data/new-result --progress-json
```

## Development

| Directory | Responsibility |
|---|---|
| `apps/desktop` | React interface and Tauri desktop bridge |
| `crates/app-core` | Project state and execution contracts |
| `packages/openastroflow-engine` | Calibration, processing pipeline, and CLI |
| `packages/light-frame-qc` | Light measurements and quality decisions |
| `engine/native` | Registration support and C++/Metal acceleration |

Run `make test` for Python, Rust, frontend, and native tests; `make check` also runs Rust formatting and lint checks. `make demo` opens a labelled browser preview without scientific processing. See [architecture](docs/architecture.md) and the [desktop development guide](apps/desktop/README.md).

Performance depends on the workload and hardware; CPU and Metal acceleration do not imply every stage uses the GPU. [Validation records](docs/validation-matrix.md) state tested cases and limits. This independent implementation does **not** claim algorithmic or pixel equivalence to PixInsight/WBPP.

## License and attribution

Original project code is [MIT licensed](LICENSE). Redistribution, including commercial redistribution, must preserve the copyright and license notice. [NOTICE](NOTICE) provides suggested attribution.

Third-party components retain their own licenses. The current bundled worker includes GPL-3.0-licensed `xisf` and cannot be distributed as an MIT-only bundle; see [licensing](docs/licensing.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

The project is not affiliated with PixInsight, Pleiades Astrophoto, N.I.N.A., or Astrometry.net, and contains no PixInsight/PCL source or binaries.
