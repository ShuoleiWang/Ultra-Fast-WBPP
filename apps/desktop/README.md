# Desktop application

The Tauri 2 + React client for [Ultra-Fast WBPP](../../README.md). A release bundle embeds the interface and Python scientific worker; it runs locally without a browser server or a user-installed Python environment.

## Development

Install Python 3.11+, Rust 1.88+, Node.js 22+, CMake 3.28+, and the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/). From the repository root:

```bash
make bootstrap
make desktop-dev
```

Debug builds find the repository `.venv/bin/openastroflow-engine` (`Scripts/openastroflow-engine.exe` on Windows), or an executable selected by `OPENASTROFLOW_ENGINE_EXECUTABLE`. Release builds use the attested bundled worker. Worker invocation uses argument arrays, not a shell or a `PATH` search.

For interface-only work, run `make demo`. The browser preview is explicitly marked **DEMO**; native file access and real processing are disabled.

## Code map

| Location | Responsibility |
|---|---|
| `src/App.tsx` | Window shell: toolbar, sidebar, content views, inspector |
| `src/Toolbar.tsx`, `src/Sidebar.tsx`, `src/Inspector.tsx` | The macOS-style chrome around the content ([design](../../docs/gui-redesign-plan.md)) |
| `src/views.tsx` | Import, screening, processing and result views, launch bar, master forms, solver setup |
| `src/useWorkflow.ts` | Import, review, processing state, progress, and timer |
| `src/bridge.ts` | Typed frontend calls and native events |
| `src/FrameInventory.tsx` | Input groups, manual type hints and calibration status |
| `src/i18n.ts` | English and Simplified Chinese interface text |
| `src/icons.tsx` | SF Symbols-style line icons and the brand mark |
| `src/demoAutopilot.ts` | `?demo=` stages for documentation screenshots (browser demo only) |
| `scripts/brand_icon.py`, `scripts/screenshots.py` | Reproducible application icon and README screenshots |
| `src-tauri/src/project.rs` | Project requests, worker execution, and result verification |
| `src-tauri/src/sidecar.rs` | Worker discovery, integrity, and launch |
| `src-tauri/src/platform/` | Process-tree control (`ManagedChild`: process groups on POSIX, Job Objects on Windows) and hardware profiles |

The standard workflow is monochrome. Missing optional master metadata remains unknown; known conflicts still block calibration, and explicitly tagged Bayer data is unsupported. Advanced overrides are bound to source content. See [calibration conventions](../../docs/recipes/calibration.md).

Every worker process is started through `platform::ManagedChild` with UTF-8 stdio. On macOS and Linux the worker leads its own process group; on Windows it runs inside a Job Object with kill-on-close, so cancelling a run or quitting the application also stops the worker pools and solver processes it started. The worker's progress stream is decoded leniently: a stray console byte never ends progress reporting.

**Review** runs the real Light quality gate. `PASS` frames are admitted; `REVIEW` frames are excluded by default and can be approved only in the supported single-panel workflow; `HARD_FAIL` frames cannot be approved. **Process** repeats the required checks and writes into a new destination. Before showing success, Rust verifies the published receipt, product paths, sizes, hashes, and final astrometry evidence.

## Checks

Run from the repository root:

```bash
npm --prefix apps/desktop test
npm --prefix apps/desktop run build
cargo test -p openastroflow-desktop --lib --locked
cargo fmt --all -- --check
cargo clippy -p openastroflow-desktop --all-targets --no-deps -- -D warnings
```

Frontend tests cover import and calibration, review decisions, cancellation, progress/timing, the platform solver routes, and final-result display. Rust tests cover worker protocol, process-tree termination, and result verification; the Job Object tests run only on Windows (`cargo test -p openastroflow-desktop platform::windows::tests`). These checks do not replace installed-app or real-data acceptance; consult the [validation matrix](../../docs/validation-matrix.md).

## Packaging

```bash
make desktop-build-macos-prerelease
```

This builds/tests the Release native library, packages the Python worker, and creates an ad-hoc signed macOS `.app`. It does not create a notarized public release. DMG creation, dependency inventories, signing, and platform acceptance are covered by the [release process](../../docs/release-process.md).

The active desktop targets are Apple Silicon and Windows x86-64. Packaged real-data runs have been exercised on an M3 Pro. On Windows the primary solver is ASTAP (`astap_cli.exe` with a D20 or larger star database), whose solutions the engine verifies against the managed Astrometry.net indexes; solve-field stays the primary solver on macOS, and the recipe's solver backend is `auto`. Result previews travel as data URLs, so output folders on other drives or shares display without widening the asset-protocol scope. Real macOS 14 hardware, broader Mac coverage, the Windows installer attestation and the retained Windows real-data acceptance are covered by the release process; other Windows architectures (ARM64) are refused by the shell.
