# Documentation

Start with the [project README](../README.md) (also in [简体中文](README.zh-CN.md)). This page maps every document in the repository and says what kind of document it is: a guide you follow, a standard you implement against, a validated fact register, or a dated record that is kept as evidence of how a decision was reached.

## Start here

| Page | What it is |
|---|---|
| [features.md](features.md) | Why Ultra-Fast WBPP: the advantages over PixInsight WBPP, the noteworthy features and the evidence behind each claim, plus what the project does not do |
| [architecture.md](architecture.md) | How a run executes: desktop → Rust → Python engine → native kernels; data and correctness boundaries; the scientific stages (legacy gate, explicit blink selection, unattended selection); performance; platform layer |
| [recipes/README.md](recipes/README.md) | Which recipe to start from for your data, and the built-in contracts |
| [building-from-source.md](building-from-source.md) | Build the engine, the desktop app or a self-contained demo app from a clone; check the result and share it |
| [validation-matrix.md](validation-matrix.md) | The register of what is validated, on which hardware, and what is explicitly not |
| [evidence/](evidence/) | The dated, share-safe receipts and benchmark reports the validation matrix and the benchmarks page cite; new results are not committed |
| [../CONTRIBUTING.md](../CONTRIBUTING.md), [../AGENTS.md](../AGENTS.md), [../CLAUDE.md](../CLAUDE.md) | How to develop here: setup, rules, checks. `AGENTS.md` is the full guide (read by Codex), `CLAUDE.md` the Claude Code session guide |

## User guides (recipes)

| Page | Use it for |
|---|---|
| [recipes/nina-mono.md](recipes/nina-mono.md) | N.I.N.A. mono folders through the standard workflow; approving REVIEW frames |
| [recipes/calibration.md](recipes/calibration.md) | Raw Flat/Dark/Bias versus supplied masters, Dark bias semantics, overrides |
| [recipes/blink-screening.md](recipes/blink-screening.md) | Blink-style screening: the flags and their thresholds, the reference frame per channel, the normalised previews, the selection file, `blink-measure` and `--selection`, what a run records, the limits |
| [blink-display-redesign.md](blink-display-redesign.md) | Experimental display-only redesign: complementary star/background views, native-pixel crops, real-image comparison and remaining human validation |
| [recipes/automatic-screening.md](recipes/automatic-screening.md) | Mixed-night import, the legacy gate (what it measures, how it decides, why it cannot see a uniformly bad night) and the optional review |
| [recipes/drizzle.md](recipes/drizzle.md) | 1×–4× native drizzle: options, products, measured quality |
| [recipes/osc-cfa.md](recipes/osc-cfa.md) | One-shot-colour (Bayer) Lights: pipeline and the synthetic validation |
| [recipes/advanced-algorithms.md](recipes/advanced-algorithms.md) | The two opt-in algorithms, both off by default: ZOGY proper coaddition as an additional product, and the robust IRLS combination |
| [recipes/astrometry.md](recipes/astrometry.md), [recipes/offline-solver-catalogs.md](recipes/offline-solver-catalogs.md) | `SEED` versus `SOLVED`, installing the solver and index set |
| [recipes/normalization-and-xisf.md](recipes/normalization-and-xisf.md) | XISF inputs, supported stellar/background normalization and retired LN compatibility |
| [recipes/project-mosaic-rgb.md](recipes/project-mosaic-rgb.md) | Multi-panel projects and RGB/LRGB products |
| [windows.md](windows.md) | Windows x64: requirements, the ASTAP route, Windows-specific behaviour, what is validated |
| [hardware.md](hardware.md) | Apple Silicon profiles and tuning, what `performance-validated` means |

## Standards and design (implement against these)

| Page | What it fixes |
|---|---|
| [master-evaluation-standard.md](master-evaluation-standard.md) | How a master is judged against the PixInsight WBPP master of the same data (metrics, tolerances, verdicts); implemented by `tools/validation/evaluate_masters.py` |
| [../benchmarks/README.md](../benchmarks/README.md), [../tools/validation/README.md](../tools/validation/README.md) | The measurement tools (tracer, kernel and stage benchmarks) with the retained numbers, and the acceptance tools (evaluator, tolerance gate) |
| [../native/README.md](../native/README.md) | Native kernel contracts: value identity with the NumPy reference, parity gates, thread invariance |
| [../packages/engine/docs/backend-contracts.md](../packages/engine/docs/backend-contracts.md) | Backend truth flags (`available`, `executionReady`) and what `doctor` reports; the desktop's command-line route is documented in architecture.md |
| [../apps/desktop/README.md](../apps/desktop/README.md) | Desktop code map and checks |
| [../packages/engine/README.md](../packages/engine/README.md), [../packages/light-frame-qc/README.md](../packages/light-frame-qc/README.md) (Chinese), [../packages/registration/README.md](../packages/registration/README.md) | The engine CLI, the quality-control package, the registration library |
| [../packaging/engine/README.md](../packaging/engine/README.md), [../scripts/windows/README.md](../scripts/windows/README.md) | Engine sidecar packaging and the Windows build machine |

## Design records (dated; kept as evidence, Chinese unless noted)

| Page | Status |
|---|---|
| [frame-selection-plan.md](frame-selection-plan.md) | Analysis and design of unattended selection (2026-09-19). Implemented; the current behaviour is described in [architecture.md](architecture.md#unattended-light-selection) |
| [frame-selection-implementation.md](frame-selection-implementation.md) | Implementation specification and the round-by-round real-data results (§8) that fixed the thresholds |
| [gui-redesign-plan.md](gui-redesign-plan.md) | The desktop redesign: diagnosis, design and the implementation record (§8) |
| [release-readiness.md](release-readiness.md) | Release review of 2026-09-09 (English): what was fixed, what blocks a stable binary release |
| [security-audit.md](security-audit.md) | Dependency and secret audit snapshot of 2026-09-05 (English) |

## Project

| Page | |
|---|---|
| [../CHANGELOG.md](../CHANGELOG.md) | Every user-visible change with its numbers; *Unreleased* is the current state of `main` |
| [release-process.md](release-process.md) | Release checklist and packaging steps |
| [licensing.md](licensing.md), [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), [../LICENSE](../LICENSE), [../NOTICE](../NOTICE) | MIT for original code; third-party components and attribution |
| [../SECURITY.md](../SECURITY.md), [../CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md), [../CITATION.cff](../CITATION.cff) | Reporting, conduct, citation |
| [../assets/branding/README.md](../assets/branding/README.md) | Icon, mark and screenshots, and the scripts that render them |

Conventions for this directory: English is canonical and `README.zh-CN.md` mirrors the project README; a number in a page names where it was measured; a page that records a decision keeps its date and is not silently rewritten; what is *not* validated is stated next to what is.
