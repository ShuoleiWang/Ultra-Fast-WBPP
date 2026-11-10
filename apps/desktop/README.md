# Desktop application

The Tauri 2 + React client for [Ultra-Fast WBPP](../../README.md). A release bundle embeds the interface and the frozen Python engine; it runs locally without a browser server or a user-installed Python environment.

## Development

Install Python 3.11+, Rust 1.88+, Node.js 22+, CMake 3.28+, and the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/). From the repository root:

```bash
make bootstrap BOOTSTRAP_PYTHON=python3.12
make desktop-dev
```

Debug builds find the repository `.venv/bin/ultra-fast-wbpp` (`Scripts/ultra-fast-wbpp.exe` on Windows), or an executable selected by `UFWBPP_ENGINE_EXECUTABLE`. Release builds use the attested bundled engine. Engine invocation uses argument arrays, not a shell or a `PATH` search.

For interface-only work, run `make demo`. The browser preview is explicitly marked **DEMO**; native file access and real processing are disabled.

## Code map

| Location | Responsibility |
|---|---|
| `src/App.tsx` | Window shell: toolbar, sidebar, content views, inspector |
| `src/Toolbar.tsx`, `src/Sidebar.tsx`, `src/Inspector.tsx` | The macOS-style chrome around the content ([design](../../docs/gui-redesign-plan.md)) |
| `src/views/` | One file per content view: `ImportView`, `ScreeningView` (legacy screening review), `RunView`, `ResultView`; `LaunchSettings.tsx` (launch bar and settings), `MasterForm.tsx` (master forms, solver setup) and `common.tsx` (frame tiles, stage labels, alerts shared with `FrameInventory` and the toolbar) |
| `src/BlinkView.tsx` | Blink & select: the channel step rail, the shared-viewport stage with compare mode, the chronological filmstrip with night headers, paint-gated playback, keyboard decisions, drop night / undo, mandatory channel confirmation, the launch bar in blink mode ([recipe](../../docs/recipes/blink-screening.md)) |
| `src/useWorkflow.ts` | Import, Blink session/decisions/review progress, diagnostic evidence, processing state, progress, and timer; `startRun` requires automatic screening or a completed Blink review |
| `src/workflow/model.ts` | UI-independent decisions: panel matrix, master overrides, `startBlockers` (the one list behind the start button and the launch bar) |
| `src/bridge.ts` | Typed frontend calls and native events (`blinkMeasure`, `loadBlinkPreview` among them) |
| `src/FrameInventory.tsx` | Input groups, manual type hints and calibration status |
| `src/i18n.ts` | English and Simplified Chinese interface text |
| `src/icons.tsx` | SF Symbols-style line icons and the brand mark |
| `src/demoAutopilot.ts` | `?demo=` stages for documentation screenshots (browser demo only) |
| `scripts/brand_icon.py`, `scripts/screenshots.py` | Reproducible application icon and README screenshots |
| `src-tauri/src/project.rs`, `project/` | Project requests (including the validated `selection`), engine execution, bounded blink preview loading, and result verification (including the astrometric receipt in `project/astrometry.rs`) |
| `src-tauri/src/sidecar.rs`, `sidecar/` | Engine discovery and bundle integrity, the `doctor --json` capability probe, and the `inventory` / `calibration-check` / `quality-check` / `blink-measure` calls with their transport budgets |
| `src-tauri/src/platform/` | Process-tree control (`ManagedChild`: process groups on POSIX, Job Objects on Windows), hardware profiles, and `paths.rs` (`plain_path`: the plain spelling of a Windows path) |

The standard workflow is monochrome. Missing optional master metadata remains unknown; a Bias, Dark or Flat of another size, binning or colour never calibrates a Light, other differences are paired as WBPP pairs them and reported as warnings, and Bayer (one-shot-colour) Lights are processed as R/G/B colour channel groups ([recipe](../../docs/recipes/osc-cfa.md)). Advanced overrides are bound to source content. See [calibration conventions](../../docs/recipes/calibration.md).

Every engine process is started through `platform::ManagedChild` with UTF-8 stdio. On macOS and Linux the engine leads its own process group and is stopped in two steps (SIGTERM, on which a run kills the solver processes it started, each in its own session; then SIGKILL for the group); on Windows it runs inside a Job Object with kill-on-close. Either way, cancelling a run or quitting the application also stops the worker pools and solver processes it started. The engine's progress stream is decoded leniently: a stray console byte never ends progress reporting.

A run that is cancelled, killed or crashes leaves its staging tree beside the output (`.<output name>.<8>.pstage`, about 12 bytes per pixel per Light). When the engine is reaped without a result, the run's thread removes the staging directories of that output which were not there when the run started: only real directories (never a link) directly in the canonical output parent, and never the output itself or its `<output>.unsolved` evidence. Quitting waits up to 10 s for that cleanup.

The engine reports itself unavailable while no solver passes its self-check, so *Recheck setup* and a finished catalog install read `doctor --json` again; Start becomes available without a relaunch. An error the engine reports before it has a result (one JSON line on its diagnostic stream) reaches the interface with its own code and message, for runs, inspections and catalog installs. Paths typed or dropped in the interface are passed after `--` and must be absolute, so no string can become an engine option. On Windows every path from `canonicalize()` is turned back from the verbatim `\\?\` form into the plain one where that names the same file, before it goes to the engine or back to the interface, and an import of more separately selected files than one command line holds (32,000 UTF-16 units) is refused with a message; neither has run on Windows hardware.

**Automatic screening** is on by default: the run request carries `selection.policy: unattended-v2` and no selection, so processing starts without a review, and the result page lists every Light the policy left out (`AUTO_DROP` with its `SEL_*` codes) and the channels worth a look in Blink ([automatic screening](../../docs/recipes/automatic-screening.md)). The checkbox sits in the launch bar and is remembered; the controller refuses an automatic request that also carries a selection or a review proof.

**Blink & select** is required before processing when *Automatic screening* is unchecked; opening Blink from the launch bar unchecks it. The engine measures Lights, chooses a reference per channel and renders aligned, normalized previews with a shared stretch into a create-only cache session. Imported Master Flats, exposure-matched Master Darks and Master Bias are forwarded for preview calibration. Filmstrip transport remains bounded at 200 KB per image / 32 MB overall, with on-demand loading beyond that budget; zoom images are loaded on demand with a 2 MB limit. The native WebKit view paints decoded images to canvas; unavailable previews have explicit placeholders/retry and require an explicit drop if they cannot be viewed. Algorithm flags are advice, every frame starts pending with KEEP selected, and Start requires every frame viewed plus each channel confirmed. The controller binds review to the manifest digest, exact frame/channel sets, explicit decisions and current Light paths. Decision edits invalidate the affected confirmation; Light reimport resets the session. CLI admission policy is unchanged.

**Review** (legacy, optional) runs the real Light quality gate and shows its evidence as a read-only table; it approves nothing. Admission comes from automatic screening (`unattended-v2`) or from the Blink decisions, sent as the run's selection: `startRun` always sends an empty `reviewSelections`, and the native controller starts a run only with automatic screening or a completed Blink review, and no review selections. **Process** repeats the required checks and writes into a new destination. Before showing success, Rust verifies the published receipt, product paths, sizes, hashes, and final astrometry evidence.

## Checks

Run from the repository root:

```bash
npm --prefix apps/desktop run format:check
npm --prefix apps/desktop test
npm --prefix apps/desktop run build
cargo test -p ultra-fast-wbpp-desktop --lib --locked
cargo fmt --all -- --check
cargo clippy -p ultra-fast-wbpp-desktop --all-targets --no-deps -- -D warnings
```

The frontend has 140 Vitest tests and the desktop crate 77 Rust tests (2 ignored). Frontend tests cover mandatory review, channel confirmations, paint-gated playback, loading failures/retry, decisions/undo, selection transport, reimport invalidation, calibration and solver readiness, cancellation and run timing. Rust tests cover the complete-review contract and stale/incomplete rejection, sidecar request/preview transport and result verification. jsdom canvas stubs verify control flow only; they do not establish pixel rendering. Native macOS WebKit was separately checked with the 97-Light NGC 6822 session for main previews, thumbnails and reference comparison. Windows installed-app rendering remains unvalidated; see the [validation matrix](../../docs/validation-matrix.md).

## Packaging

```bash
make desktop-build-macos-prerelease
```

This builds/tests the Release native library, packages the Python engine, and creates an ad-hoc signed macOS `.app`. It does not create a notarized public release. DMG creation, dependency inventories, signing, and platform acceptance are covered by the [release process](../../docs/release-process.md). `make desktop-build-macos-demo` builds a self-contained demo app with `solve-field` and the index set inside ([building from source](../../docs/building-from-source.md)).

The active desktop targets are Apple Silicon and Windows x86-64. Packaged real-data runs have been exercised on an M3 Pro. On Windows the primary solver is ASTAP (`astap_cli.exe` with a D20 or larger star database), whose solutions the engine verifies against the managed Astrometry.net indexes; solve-field stays the primary solver on macOS, and the recipe's solver backend is `auto`. Result previews travel as data URLs, so output folders on other drives or shares display without widening the asset-protocol scope. A launch on real macOS 15 hardware (the declared minimum), broader Mac coverage, the Windows installer attestation and the retained Windows real-data acceptance are covered by the release process; other Windows architectures (ARM64) are refused by the shell.

## Complementary Blink displays

The desktop explicitly requests `blink-complementary-display-v2`. The main canvas switches between star detail and full field; the inspector (or an inline panel when hidden) shows relative signal/noise, a signed background difference and corresponding native-pixel crops. Shape crops match stellar amplitude only when the peak is measurable; the separate signal mode keeps a shared curve. Enlarged crops remain original pixel samples. Both the main field and available background view must paint before a frame counts as viewed. The worker reports unsupported native-crop formats and missing photometry/registration explicitly. Missing complete preview calibration is shown, and a Flat alone is not applied as complete calibration. These displays never change science pixels.

Inspector layout: filename and metadata use a spaced header; calibration disclosures grow with wrapped text. Check both 260 px and 300 px sidebars with long filenames, missing calibration, expanded error details and both languages in a browser and native WebKit; DOM-only component tests do not verify text geometry.
