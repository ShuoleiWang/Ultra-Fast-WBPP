# AGENTS.md — working in this repository

This is the contributor guide for coding agents and for people. OpenAI Codex reads it automatically; Claude Code reads [`CLAUDE.md`](CLAUDE.md), which states the same rules in short form and adds session notes. Keep the two consistent when you change either. Human-facing setup notes are in [`CONTRIBUTING.md`](CONTRIBUTING.md); the documentation map is [`docs/README.md`](docs/README.md).

## What this project is

Ultra-Fast WBPP is a desktop and headless preprocessing pipeline for astrophotography: quality-gated Light selection, calibration, registration, normalization, rejection and integration, optional drizzle, plate solving with verification, and receipted publication of linear masters. It is a Python engine (`packages/engine`, `packages/light-frame-qc`, `packages/registration`) with multithreaded C++ kernels and an audited Metal path (`engine/native`), driven by a Tauri 2 + React desktop (`apps/desktop`) that runs the engine's command line, one process per operation. Supported platforms are macOS 14+ on Apple Silicon and Windows 10 22H2 / 11 x64. The project is alpha.

Naming: the product is "Ultra-Fast WBPP" and its command line is `ultra-fast-wbpp`. Code identifiers use the short token `ufwbpp` (the `ufwbpp` and `ufwbpp_registration` Python packages, `ufwbpp_native_*` C symbols, `UFWBPP_*` environment variables, the `ufwbpp-engine` sidecar); names a user sees spell the product out (`ultra-fast-wbpp-desktop`, `~/.ultra-fast-wbpp`, `io.github.shuoleiwang.ultra-fast-wbpp`). Three things keep the project's former name `openastroflow` on purpose, because users' files carry them: the `OAF*` FITS keyword namespace of the masters, the text of the content-addressed catalog manifests, and an existing data root (`~/.openastroflow`, `%LOCALAPPDATA%\OpenAstroFlow`), which is used in place when present.

It is an independent implementation with no PixInsight/PCL source or binaries. Masters are judged against the PixInsight WBPP masters of the same data with the quantitative standard in [`docs/master-evaluation-standard.md`](docs/master-evaluation-standard.md), never by eye. Read [`docs/features.md`](docs/features.md) for what the project does and the evidence behind each claim.

## Rules that are not negotiable

1. **Inputs are read-only, outputs are create-only.** The engine never modifies a source file or an earlier result; every run publishes into a directory that did not exist (`OUTPUT_EXISTS` otherwise, through the platform layer's no-replace rename), and intermediates live under the output volume, never in a system temp directory.
2. **Fail closed.** No product without its receipt; no `SOLVED` WCS without an engine-verified solution (a header or N.I.N.A. hint is a `SEED`, never promoted); the desktop shows "done" only after Rust re-verified receipt, hashes and final sky coordinates. Do not add "best effort" paths that publish partial science.
3. **Never commit** acquisition data (FITS/XISF/XDRZ), benchmark or evaluation result files, absolute home-directory paths (a macOS or Windows user folder in any string), API keys or signing material, proprietary application code, or third-party catalogs. `scripts/check_public_tree.py .` and `scripts/check_local_links.py .` run in CI and locally as `make source-check`. Fixtures are tiny and synthetic under `tests/fixtures` (and `packages/**/tests/fixtures`). Receipts written by harnesses must be share-safe (basenames, not paths).
4. **Scientific changes are contracts.** State the mathematical contract, keep the NumPy/CPU reference implementation, hold the native kernel value-identical to it with a differential test (`packages/engine/tests/test_native_kernels.py`, `engine/native/tests`), give the algorithm a versioned identifier that receipts record, and keep the previous identifier reproducible. When a change may legitimately move pixels (a different numerical evaluation), state and check what it may do with [`tools/validation/master_tolerance_gate.py`](tools/validation/master_tolerance_gate.py). Never loosen a scientific threshold to make a test pass. Discuss algorithm changes in an issue first.
5. **Performance changes are proven, not claimed.** A pipeline change is verified on a real project by comparing the SHA-256 of the solved masters with a baseline built from the base commit on the same OS (bit-identical), or by the tolerance gate plus unchanged evaluator statuses when pixels may move; report wall time before and after. A change that costs run time needs a strong reason.
6. **Defaults stay reproducible.** `selection.policy: legacy-gate` is the default and reproduces the historical admission exactly; kernel ABIs `v1`/`v2` remain callable; `include-all` is diagnostic. New behaviour goes behind a recipe field with a documented default.
7. **Receipts name what ran.** Algorithm ids, kernel ids, backend, selection policy, platform facts and their sources (`platform.hardware`, `platform.tuning`, `platform.nativeKernels`, `execution.sourceExtraction`) are part of every receipt. A fact that was not measured is reported as `fallback`/`unavailable`, not invented.
8. **Desktop contracts.** `apps/desktop/src/useWorkflow.ts` (React state and lifecycle), `workflow/model.ts` (pure decisions such as `startBlockers`), `bridge.ts` (typed commands/events) and the tests in `App.test.tsx`, `NativeApp.test.tsx`, `BlinkView.test.tsx` and `workflow/model.test.ts` are contracts. Add tests for new behaviour; do not rewrite existing assertions to fit a visual change. Every user-visible string exists in both languages in `i18n.ts`. The browser demo (`make demo`, `?demo=…`) is visibly labelled DEMO and never touches native files.
9. **Documentation.** English is canonical; `README.md` and `docs/README.zh-CN.md` mirror each other and must be updated together. User-visible changes get a `CHANGELOG.md` entry under *Unreleased*. Links must resolve; numbers cite where they were measured; validation pages state what is *not* validated as clearly as what is.
10. **Git.** Conventional Commit subjects (`feat(engine): …`, `perf(trace): …`, `fix: …`). One logical commit per pull request (fold fixes in with `git commit --amend` and `git push --force-with-lease`). `main` is protected: pull request plus the `CI gate` status. Commit messages and pull request text carry no AI attribution (no co-author trailers or "generated with" footers); the maintainer is the author. Do not push, tag, merge or create releases unless the maintainer asked for that specific action.

## Repository map

| Path | What lives there | Notes for changes |
|---|---|---|
| `packages/engine/src/ufwbpp/` | The engine: `cli.py` (every command), `workflows/` (`project.py` multi-target/filter projects and colour products, `single_target.py` one field, `solve.py`, `contracts.py`), `pixel_pipeline.py` (plan → calibration masters → fused calibrate+warp → per-group normalization/rejection/integration), `calibration.py`, `global_normalization.py`, `residual_background.py`, `transient_rejection.py` (fast-Radon trail corridors), `drizzle_native.py`, `selection/` (unattended policy, guards, counterfactual, region weight maps), `blink_session.py`, `native_kernels.py` (ctypes bridge), `integrity.py` (canonical JSON and SHA-256 forms), `platform/` (OS services), `astap_backend.py` / `astrometry_net_backend.py` / `catalog_correspondence.py` (solvers and verification), `recipe.py`, `runtime.py`, `planning.py` | Keep receipt schemas stable; the orchestration functions are split into named stages, keep new work in a stage. `reference/` holds NumPy oracles used by tests. |
| `packages/light-frame-qc/src/lightframeqc/` | Light measurement, the quality gate and Blink flags: `measure.py`, `analysis.py`, `quality_gate.py`, `blink_flags.py`, `blink_reference.py`, `nightly_statistics.py`, `native_psf.py` (native-resolution star stamps), `cfa.py` (Bayer), `source_extraction.py` (SEP determinism self-test), `registration.py` (QC-scale registration), `content_hash.py`; its own `light-frame-qc analyze` command | Its README is in Chinese. Measurements carry provenance fields; check them before trusting a number. |
| `engine/native/` | C++20 kernels (`PortableKernels.cpp`: Lanczos-3 warp, MAD rejection, masked mean; `DrizzleKernel.cpp`, `DebayerKernel.cpp`, `Lanczos3Table.cpp`, `FusedIntegration.cpp`), Metal integration (`MetalFusedIntegration.mm`), C ABI (`c_api.cpp`), tests | Operation-for-operation reproductions of the NumPy reference; `-fno-fast-math -ffp-contract=off`, MSVC `/fp:strict`. New kernels need a Python reference, a differential test and a thread-count invariance test. |
| `packages/registration/` | Registration library (`ufwbpp-registration`) | Portable reference path for detection, transform estimation, warping, crop and weights. |
| `apps/desktop/` | React 19 interface (`src/`, formatted with Prettier) and Tauri Rust bridge (`src-tauri/src/`: `project/` run lifecycle and verification, `sidecar/` engine discovery, `doctor --json` capability probe, inspection and blink, `catalog.rs` index management, `platform/` process trees) | See `apps/desktop/README.md` for the code map and checks. |
| `tools/validation/` | `evaluate_masters.py` (vs PixInsight masters), `master_tolerance_gate.py` | The acceptance tools of the evaluation standard. |
| `benchmarks/` | `trace_run.py` (real-run tracer), `native_kernels_pipeline.py`, `e2e_stage_timings.py`, `selection_defect_injection.py`, `selection_oracle_report.py`; `results/` holds only the original checked-in reports | New result files are never committed. |
| `validation/` | Dated, share-safe evidence receipts | Historical; add new receipts only when share-safe and referenced from `docs/validation-matrix.md`. |
| `scripts/` | Build and release tooling: `build_native_runtime.py`, `build_engine_sidecar.py`, `stage_tauri_sidecar.py`, `attest_bundled_runtime.py`, `build_sep_wheel.py`, `check_public_tree.py`, `check_local_links.py`, `windows/` | Tested by `tests/`. |
| `packaging/` | Engine sidecar spec (`packaging/engine`) and the SEP determinism patch | |
| `docs/` | Architecture, features, evaluation standard, recipes, platforms, validation, design records | Index in `docs/README.md`. |
| `tests/` | Repository-level tests (build scripts, release wiring, checkers) | |

Sizes for orientation: about 57 k lines of Python, 9 k lines of C++/Metal (with tests), 1 070 Python tests, 55 Rust tests, 77 Vitest tests, three C++ test binaries.

## Setup and commands

Prerequisites: Python 3.11+, Rust 1.88+, Node.js 22+, CMake 3.28+, the Tauri prerequisites (Xcode on macOS; VS 2022 Build Tools on Windows, installed by `scripts/windows/bootstrap.ps1`).

```bash
make bootstrap          # .venv with the three editable packages, npm ci, test build of the kernels
make desktop-dev        # Tauri dev window (uses .venv's engine)
make demo               # browser demo of the interface, no data
make test               # python-test rust-test frontend-test native-test
make check              # source-check + cargo fmt/clippy + make test
```

The individual suites, as CI runs them (a pull request runs the jobs its changed paths select, see `scripts/ci_changed_areas.py`; pushes to `main` run all of them):

```bash
.venv/bin/python scripts/build_native_runtime.py --build-dir build/native-release   # Release kernels: build, ctest, install
.venv/bin/python -m pytest -q packages/light-frame-qc/tests packages/registration/tests packages/engine/tests tests
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
npm --prefix apps/desktop run format:check && npm --prefix apps/desktop test && npm --prefix apps/desktop run build
.venv/bin/python scripts/check_public_tree.py . && .venv/bin/python scripts/check_local_links.py .
```

The unoptimized test build with Metal enabled is `make native-test` (`cmake -S engine/native -B build/native -DUFWBPP_BUILD_TESTS=ON -DUFWBPP_ENABLE_METAL=ON`, build, `ctest`).

Native kernels for real runs: `make native-release-install` (or `python scripts/build_native_runtime.py --build-dir build/native-release`, the chain CI uses) builds `build/native-release` with `CMAKE_BUILD_TYPE=Release`, runs its ctest and installs the library into `packages/engine/src/ufwbpp/native/`. `build/native` (from `make native-build`) is the unoptimized test configuration: run `ctest` there, never install it.

Engine CLI: `ultra-fast-wbpp doctor --json` (hardware, kernels, solver readiness; the desktop's capability probe), `inventory`, `calibration-check`, `quality-check`, `blink-measure`, `plan`, `run <lights…> <calibration…> --output <new dir> --recipe <json> --progress-json`, `run-project` (multi-target/filter projects; the desktop's route, `--request-json` for scripted callers) and `catalog`. Recipes are versioned JSON contracts (`docs/recipes/`); the desktop uses `mono-standard-v1`.

Useful environment variables: `UFWBPP_DISABLE_NATIVE_KERNELS=1` (NumPy reference path), `UFWBPP_NATIVE_LIBRARY=<path>` (explicit kernel library), `UFWBPP_QC_CACHE_DIR=off` (no measurement cache, for comparisons), `LIGHTFRAMEQC_PARALLELISM=threads`, `LIGHTFRAMEQC_FITS_READER=memmap`, `UFWBPP_ASTAP=<astap_cli>`, `UFWBPP_ENGINE_EXECUTABLE=<engine>` (desktop debug builds).

Desktop bundle on macOS: `make desktop-build-macos-prerelease` (release kernels → PyInstaller engine → staged sidecar → Tauri). Verify a bundle with `scripts/attest_bundled_runtime.py --app <app> --target aarch64-apple-darwin --output <json>`. On macOS 27 / Xcode 27 build with `CARGO_PROFILE_RELEASE_STRIP=none` in front of the npm command (see gotchas).

## Before you finish a change

- Run the suites your change touches, and `make source-check` always. A scientific or native change also runs the native ctest and the differential Python tests; a desktop change runs Vitest, `tsc` (through `npm run build`), `cargo test -p ultra-fast-wbpp-desktop --lib --locked`, fmt and clippy.
- A pipeline change is reproduced on a real project (the maintainer keeps a private 61-Light, four-night L/R/G/B NGC 7331 set; ask for the recipe) and reported with: wall time before/after, the four master hashes against a same-OS baseline or the tolerance-gate result, and `tools/validation/evaluate_masters.py` PASS/WARN/FAIL per filter when quality could move. Keep the real-data outputs and reports out of the repository.
- A quality-control change is checked for silent failures: look at the provenance fields of the new measurement (for example `fwhmSource`, error keys) on real frames, not only at the summary numbers.
- Update `CHANGELOG.md`, the relevant `docs/` page and both READMEs when the user-visible behaviour or a headline number changed.
- Fill the pull request template honestly: what was validated, what was skipped.

## Gotchas that have cost time

- **Wrong kernel library installed.** Installing the `build/native` (unoptimized) library makes every kernel 6–7× slower and a project run takes minutes instead of ~76 s. Check `describe_native_kernels()['loaded']` and the library size (release ≈ 225 KB, debug ≈ 640 KB on macOS); reinstall from `build/native-release`.
- **Bit-identity baselines are per OS.** An OS upgrade changes libm/Accelerate rounding, so master hashes from a previous OS do not match; rebuild the baseline from a `git worktree` of the base commit on the same OS, with its own kernel library and `PYTHONPATH` pointing at all three packages (they are editable installs of the main tree). Windows and macOS masters differ legitimately; compare across platforms only with the tolerance gate.
- **macOS 27 / Xcode 27 release builds** fail with `E0463 can't find crate` because dyld rejects the stripped proc-macro dylibs; build with `CARGO_PROFILE_RELEASE_STRIP=none`. `cargo clean` does not help.
- **QC measurements can fail silently.** The native PSF path once returned nothing on scaled 16-bit FITS for a whole round while the summary looked plausible; read the provenance fields.
- **Two reference frames.** QC grids (transparency, region maps) are in the QC reference frame; registered pixels are in the pipeline reference frame. A meridian flip between the two is a 180° rotation; transform spatial QC products (`RegionWeightMap.transformed`) before applying them.
- **Windows determinism.** The PyPI SEP wheel is non-deterministic on MSVC; use the patched `sep 1.4.1+ufwbpp.1` (`scripts/build_sep_wheel.py --install`), otherwise runs differ and receipts carry `SEP_NONDETERMINISTIC`.
- **Solver PATH.** A Finder-launched app inherits launchd's minimal `PATH`; the solver runtime prepends the solver's own directories so `solve-field` finds its helpers. Keep that when touching `SolverProcessRuntime`.
- **Spawned pools.** Scripts that use the spawn start method need an `if __name__ == "__main__":` guard, or every worker re-runs the script.
- **ETXTBSY on Linux.** The sidecar launcher retries a "text file busy" exec; the CI test that caught it stays.
- **Metal is not tested on hosted CI.** GitHub's macOS runners expose no Metal device: `ctest` reports `native-core-metal-differential` as skipped and the C-ABI Metal checks return early, so a green CI proves only that the Metal path compiles. Verify a change to `engine/native/metal` or the Metal executor with `make native-test` on Apple Silicon.
- **Shell details.** Quote globs under zsh; use absolute paths when several shells run in parallel; run `tsc` as `./node_modules/.bin/tsc` inside `apps/desktop` (a bare `npx tsc` elsewhere installs an unrelated package).
- **Real runs are long-ish.** A full project run is about 76 s on an M3 Pro and 4.5 min on the validation laptop; `evaluate_masters.py` takes about two minutes per 26 MP filter pair; the defect-injection harness re-integrates several times. Plan runs, don't poll them.

## Evidence and data

- Real acquisition data is private and never enters the repository; docs cite it as "the reference data set" with frame counts, sensor geometry and machine. Numbers in the README come from `CHANGELOG.md`, `docs/windows.md`, `docs/recipes/*.md`, `benchmarks/README.md` and `docs/validation-matrix.md`; when you improve a headline number, update those first and the READMEs from them.
- `docs/validation-matrix.md` is the honest register of what is and is not validated. Do not turn an unchecked row into a claim.
- The maintainer's working language is Chinese; several design records in `docs/` are Chinese and marked as such in the docs index. Product documentation stays English with the mirrored `docs/README.zh-CN.md`.

## Module layout

`workflows/project.py` runs a project (targets, filters, shared calibration,
colour products and mosaics) and calls `workflows/single_target.py` once per
target; `workflows/solve.py` owns the final solve and WCS verification and
`workflows/contracts.py` the request, selection and progress types. Both run
functions and `pixel_pipeline._run_portable_pipeline_fits` read as a sequence
of named stages (`_validated_sources`, `_screen_lights`, `_plan_run`,
`_build_calibration_masters`, `_integrate_group`, ...); a new step belongs in
a stage, not inline. Use `calibration_inputs.py` for content-bound master
metadata, `image_io/fits.py` for FITS primitives, `solvers/process.py` for
shared solver execution, `publication.py` for create-only colour/mosaic
publication and `integrity.py` for the byte forms digests are computed over.
The desktop controller is split into `project/` and `sidecar/` modules; its
command names and the UI bridge contract stay stable.

LocalNormalization is retired: old enabled recipes fail explicitly, disabled
legacy fields remain readable, and stellar/background normalization is unchanged.
Do not restore an unvalidated alternative behind a GUI checkbox. Counterfactual
v2 uses fixed global-row statistics and masks NaNs before region weighting;
its algorithm ID distinguishes it from historical v1 receipts. The evaluator
must report unmeasured standard gates and never certify their absence as PASS.
