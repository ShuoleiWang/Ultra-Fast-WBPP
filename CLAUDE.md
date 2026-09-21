# CLAUDE.md

Guide for Claude Code sessions in this repository. [`AGENTS.md`](AGENTS.md) is the long-form contributor guide (repository map, every command, the full gotcha list); read it before touching scientific code, the native kernels, the desktop contracts or release tooling. This file states the rules that matter in every session.

## The project in one paragraph

Ultra-Fast WBPP preprocesses astrophotography Lights into verified, plate-solved linear masters: quality-gated selection, calibration, registration, normalization, rejection/integration, optional drizzle, verified plate solving, receipted publication. Python engine (`packages/openastroflow-engine`, `packages/light-frame-qc`), C++/Metal kernels (`engine/native`), Tauri 2 + React desktop (`apps/desktop`), Rust contracts (`crates/app-core`). macOS Apple Silicon and Windows x64. Alpha. Independent implementation, no PixInsight code; masters are compared with PixInsight WBPP masters by the standard in `docs/master-evaluation-standard.md`. What the project does and the evidence for it: [`docs/features.md`](docs/features.md).

## Working agreement

- **Local by default.** Edit locally; commit, push, open or merge a pull request only when the maintainer asks for that action. When a PR is requested it is one commit (amend + `--force-with-lease`), Conventional Commit subject, template filled honestly, and no AI attribution anywhere in the commit or PR text (no co-author trailers, no "generated with" footer).
- **Nothing private in the tree.** No acquisition data, no benchmark/evaluation result files, no absolute home paths, no secrets. Run `.venv/bin/python scripts/check_public_tree.py .` and `.venv/bin/python scripts/check_local_links.py .` before handing over. Harness receipts record basenames, not paths.
- **Sources read-only, outputs create-only, fail closed.** Never add a path that publishes unverified science or overwrites anything.
- **Prove, don't claim.** A pipeline change is verified on the real reference project: wall time before/after, the four solved-master hashes against a same-OS baseline (bit-identical) or `benchmarks/master_tolerance_gate.py` when pixels may move, and `benchmarks/evaluate_masters.py` PASS/WARN/FAIL per filter when quality could move. Report per-filter numbers with confidence intervals; state what was skipped.
- **Performance is a feature.** A change that costs measurable run time needs a strong reason; the native kernels stay value-identical to the NumPy reference; defaults (`legacy-gate`, kernel ids) stay reproducible.
- **Minimum user effort** is the product goal for the desktop: judge UX changes by operations saved; keep `useWorkflow.ts`, `bridge.ts` and the existing test assertions as contracts; both languages in `i18n.ts`.
- **Delegate the long-running work.** Full runs, builds, benchmark sweeps, surveys and CI watching go to subagents with a precise brief (inputs, expected outputs, the provenance fields to report). Design, core code and review stay in the main session. Don't poll: a project run is ~76 s on the M3 Pro, the evaluator ~2 min per filter pair.
- **Disk hygiene.** Delete only this project's own temporary outputs (superseded run directories, aborted `*.project-staging` dirs, scratchpad files); never unrelated files.
- **Reports.** Lead with the outcome, numbers per filter, provenance stated, honest about failures. The maintainer writes Chinese; answer in the language they used. Documentation stays English with the mirrored `docs/README.zh-CN.md`, updated together with `README.md` and `CHANGELOG.md`.

## Commands

```bash
make bootstrap && make test            # first time; python + rust + frontend + native tests
.venv/bin/python -m pytest -q packages/light-frame-qc/tests engine/native/python/tests packages/openastroflow-engine/tests tests
cargo fmt --all -- --check && cargo clippy --workspace --all-targets --locked -- -D warnings && cargo test --workspace --locked
npm --prefix apps/desktop test && npm --prefix apps/desktop run build
make native-release-install            # Release kernels into the engine package (the only library to install)
make source-check                      # public-tree and link checks
make desktop-dev | make demo | CARGO_PROFILE_RELEASE_STRIP=none make desktop-build-macos-prerelease
```

`ultra-fast-wbpp doctor --json` shows hardware, kernels and solver readiness; `run` / `run-project … --output <new dir> --recipe <json> --progress-json` run the engine headless. `OPENASTROFLOW_DISABLE_NATIVE_KERNELS=1` selects the NumPy path; `OPENASTROFLOW_QC_CACHE_DIR=off` disables the measurement cache for comparisons.

## Gotchas (details in AGENTS.md)

- `build/native` is the unoptimized test build; install only from `build/native-release`. A run that is 6–7× too slow has the debug library installed.
- Master hashes are per OS; rebuild the baseline from a worktree of the base commit after an OS upgrade. Windows vs macOS: tolerance gate only.
- macOS 27 / Xcode 27: release builds need `CARGO_PROFILE_RELEASE_STRIP=none`.
- QC measurements can fail silently: check provenance fields (`fwhmSource`, error keys) on real frames.
- QC grids live in the QC reference frame, registered pixels in the pipeline reference frame; meridian flips are 180° apart — transform spatial QC products.
- Windows needs the patched deterministic SEP build; unpatched runs differ (`SEP_NONDETERMINISTIC`).
- Spawn-pool scripts need `if __name__ == "__main__":`; quote zsh globs; absolute paths in parallel shells; `./node_modules/.bin/tsc` inside `apps/desktop`.

## Where to look

`docs/README.md` (documentation map) · `docs/architecture.md` (execution path, boundaries, science) · `docs/features.md` (advantages and evidence) · `docs/validation-matrix.md` (what is and is not validated) · `docs/recipes/` (user guides) · `benchmarks/README.md` (measurement tools) · `apps/desktop/README.md` (desktop code map) · `engine/native/README.md` (kernel contracts) · `CHANGELOG.md` (what changed, with numbers).
