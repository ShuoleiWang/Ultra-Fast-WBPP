# Validation matrix

Support, compatibility, implementation, and optimization are separate claims. The receipts below are dated local evidence, primarily from September 1–6, 2026; they are **not a test report for the latest source tree**. Later fixes and private runs do not retroactively update these receipts. Hosted workflows count as current evidence only after they pass on the published commit, and never substitute for scientific hardware validation or release signing.

Before tagging a candidate, regenerate the source-test summary and platform evidence from that exact commit. Report optional hardware skips explicitly, retain private data outside Git, and publish only sanitized summaries. Until those checks are complete, the correct status is alpha; in particular, gradient-free output and general PixInsight/WBPP equivalence are not established.

## Standard master workflow update

The [2026-09-06 standard-master receipt](../validation/standard-mono-masters-20260906.json) records the new desktop conventions, 591 passing Python tests and two public synthetic pixel E2E cases. All actual RGB calibration groups match without manual overrides; L remains missing a Flat. A real 19-Light packaged attempt passed master preparation but stopped after QC admitted only one Light. No full real-image success or rejected-frame accuracy is claimed for that attempt. See [calibration behavior](recipes/calibration.md).

## Latest input and screening update

The [input/screening receipt](../validation/input-screening-calibration-20260905.json) records the compact native table UI, metadata-only calibration checks, and final packaged-worker/GUI screening. Synthetic pixels gave 20 normal PASS and six excluded defects; a denser wrong-field reference test preserved 10 normal images and excluded the wrong field. The final package also retained all 38 real NINA Lights from four capture dates and matched their raw Flats and supplied masters. New fixes cover sparse-field cloud evidence, whole-night FWHM degradation without HFR, bounded reference support, and calibration diagnostics. See [workflow and limits](recipes/automatic-screening.md). This update did not repeat the full raw-to-WCS run below.

## Historical packaged candidate: independent project-entry recheck

The [2026-09-05 packaged-project receipt](../validation/macos-arm64-packaged-project-20260905.json) records a fresh full-filter run through the actual bundled worker's `run-project` entry point: 38 B Lights, 20 raw Flats and two supplied normalized-XISF masters, SOLVED in 503.996 seconds on this M3 Pro. All 38 Lights were admitted; real Metal ran with CPU numerical parity, and the final WCS used 55 matches at 1.263 arcsec RMS. Independent final-FITS checks found no nonfinite pixels. The read-only DMG launched its GUI and engine; all 289 Mach-O deployment targets are macOS 14.0 or lower. Native launch and the worker full run were verified separately, not by driving an entire real dataset through GUI clicks.

This recheck fixed physical FITS decoding, flat-response validity, nonfinite/low-coverage rejection, camera/readout text matching, and supplied-XISF identity preservation through shared project calibration. The Python baseline passed 553 tests with 3 opt-in skips; the two subsequent project-entry fixes passed 29 planning/CLI and 3 shared-project regressions. React passed 24 tests, Rust 82, and native Release 4 including real Metal.

A separately aligned comparison measured Pearson 0.998992100 and affine NRMSE 4.322%. These are measured differences, not a new PixInsight-equivalence acceptance or a reuse of the earlier 25-check pass. The public-entry elapsed time includes different orchestration and concurrent DMG packaging, so it is not a controlled speed comparison. The source implementation remained stable throughout the run.

## Earlier retained evidence

The following rows describe previous receipts and are historical evidence. The independent recheck above identifies exactly what was rerun on September 5.

| Area | Evidence | Status |
|---|---|---|
| Light Quality Gate | 170 local unit/regression tests plus a separate 370-frame retained plan-only audit | Implemented; the real frames are private and are not in Git |
| `prepare-wbpp` publication I/O | Deterministic fixture reduced full-content reads from 83,980,800 to 33,592,320 bytes | 60.0% reduction with unchanged create-only/hash gate; [receipt](../packages/light-frame-qc/benchmarks/results/prepare-apply-io-m3-pro-20260901.json) |
| Complete Python suites | 548 passed and 3 explicit opt-in skips across Quality Gate, registration/scientific-engine, and release/packaging collections | Passing locally; skips require a real Metal device or managed Astrometry.net fixture and are not converted into support claims |
| Rust control plane | 53 app-core and 29 desktop tests including runtime discovery/tamper, duplicate-run prevention, language-event stability, and exit-time process-group cleanup | Passing locally; hosted macOS/Windows/Linux-development jobs are configured but pending the first authorized public commit |
| React GUI | 19 component/interaction tests plus a 1280×840 native macOS light-interface review | Native commands are wired to the real controller; browser mode remains a visibly labelled development-only preview |
| Portable native math | Sandboxed CMake run passed CPU C ABI, calibration and image-I/O tests and explicitly skipped the unavailable Metal device test; a separate unsandboxed M3 Pro native differential passed | Hosted native C++ jobs are configured for macOS, Windows and Linux; this is not Windows scientific E2E or packaging acceptance |
| Apple Metal | Native CPU↔Metal differential passed on the real M3 Pro; the production adapter's 96-frame, three-tile/three-NaN case sampled first/middle/last tiles and matched portable CPU science plus accepted/coverage/rejection maps; forced-Release 6252×4176×96 kernel benchmark used one warmup and five measured repetitions | Passing on this Apple M3 Pro; [execution-path evidence](../benchmarks/results/m3-pro-metal-e2e-validation-20260901.json) and [current repeatable kernel evidence](../benchmarks/results/m3-pro-integration-96-v2-20260905.json). The v2 candidate is from a dirty worktree and must be regenerated from the clean release commit |
| Real raw mono E2E | 38 B Lights + 20 raw Flats + supplied normalized-XISF MasterBias/MasterDark; 466.116 s; 38/38 QC PASS; full-resolution Lanczos-3; tuned Metal parity PASS; 55-match/1.263-arcsecond managed WCS; independent PixInsight comparison 25/25 PASS, Pearson 0.999115, affine NRMSE 4.05% | PASS for this private QHY268M/M3 Pro fixture only; [sanitized evidence](../validation/m3-pro-dunpai3-full-b-20260905.json) contains no source paths, content hashes, or pixels and makes no WBPP-equivalence claim |
| XISF pixel bridge and numeric domains | Uncompressed/zlib/lz4/zstd fixtures, XML/DTD and size/working-set failures, unique-science/auxiliary-map selection, one sanitized real MasterFlat decode, and the full-B UInt16-Light to normalized-Float32-XISF Bias/Dark 65535:1 application oracle | Passing for the bounded mono bridge; [real decode receipt](../validation/xisf-real-masterflat-20260901.json) and [full-B evidence](../validation/m3-pro-dunpai3-full-b-20260905.json). RGB/multichannel and ambiguous containers are blocked |
| Global normalization | Real 38-frame B validation bound 37/37 stellar-scale hints; median/P90 relative scale error 1.56%/3.90%; one guarded additive grid and 36 scalar fallbacks; photometry and background checks passed | Accepted for one observing condition; not PixInsight-equivalent and retained multi-condition acceptance is pending |
| LocalNormalization | CPU synthetic extended-scene/gradient oracle, underconstrained-model failure, ordinary/Drizzle wiring, and retained scale/offset/residual evidence | Opt-in and fail-closed; not algorithmically equivalent to PixInsight LocalNormalization; no retained real-data acceptance |
| Drizzle CPU | Synthetic science/weight/coverage, sampling/dither/rejection gates, and STScI 2.2 adapter tests | Implemented as optional backend; retained real-data Drizzle validation pending |
| RGB/LRGB and mosaic | Synthetic four-panel × RGB execution exercises one-time shared raw-Dark calibration, panel solve/reprojection/final solve/alignment/product contracts | Shared raw-Dark reuse and `PROPAGATED_VERIFIED` reference-WCS provenance regressions pass; no retained real multi-panel/color acceptance |
| Solver adapters | Fake-executable failure/timeout/drift/unmanaged/ambiguous tests plus two sanitized M3 Pro managed-index executions | Full-B E2E passed at 55 matches/1.263 arcsec; standalone [adapter benchmark](../benchmarks/results/m3-pro-astrometry-net-real-fixture-20260901.json) passed at 56 matches/1.404 arcsec. Neither proves universal sky/scale coverage; ASTAP is diagnostic-only |
| Desktop bundle | Current `Ultra-Fast WBPP.app`: 428-file / 162,877,826-byte immutable worker; all 289 arm64 Mach-O slices at macOS 14.0 or lower with 0 unresolved load edges; pinned Sonoma OpenSSL/mpdecimal ABI smoke; three byte-attested legal resources; ad-hoc hardened-runtime signature; read-only mounted DMG; mounted GUI direct launch with 0 listening TCP sockets; 1.395-second mounted worker handshake; frozen real-FITS QC preview smoke | Local arm64 binary audit [passes](../validation/macos-arm64-renamed-bundle-20260905.json). Real macOS 14 hardware launch is still pending. It is not Developer ID signed/notarized, commit-bound, a packaged-app-initiated real-data E2E, a complete-SBOM release, or a Windows package |

## Apple Silicon release matrix

| Profile | Required evidence | Current evidence |
|---|---|---|
| `portable-cpu` | Python/Rust protocol and synthetic E2E on x86-64 Linux plus CPU differential gates | Hosted Linux jobs are configured; current local CPU suites pass, but the first authorized public run is pending |
| `generic-arm64-cpu` | Build and synthetic E2E on Apple Silicon; no Metal required | Local M3 Pro CPU path passing |
| `generic-apple-metal` | Capability-driven dispatch and CPU differential on at least one M-series runner | Local M3 Pro passing |
| `m3-pro-tuned` | Retained M3 Pro scientific and performance reports | Forced-Release 96-frame 6252×4176 kernel benchmark at 4.410 s median (4.372–5.397 s; 568.29 MP-frame/s), 96-frame production-adapter three-tile NaN/map differential, and the private 38-frame raw-to-solved Lanczos-3 validation with 25/25 independent reference checks |
| Non-M3 M-series compatibility | Generic profile on at least one M1/M2/M4-class machine before a stable support claim | Pending external/CI hardware evidence |

The application may run a generic arm64/Metal path on an unbenchmarked M-series device, but the UI and manifest distinguish `compatible-generic` from `performance-validated`.

## Windows release matrix

The hosted Windows matrix is configured for Python 3.11/3.12, Rust app-core/Tauri commands, explicit `taskkill /T` process-tree tests, and native C++ CPU C ABI tests; the current uncommitted candidate has not produced a hosted run. These remain interface, compilation, and synthetic-test targets only: no packaged Windows scientific E2E, real raw E2E, reparse-point/handle-identity acceptance, installed-runtime attestation, installer smoke, signing, or representative Windows hardware evidence exists. Windows GPU support cannot be inferred from the interface or CI configuration.

## Scientific 1.0 gate

Define the supported scope before applying this gate. Unsupported or explicitly experimental features may stay outside a mono macOS release; they must not be advertised as validated stable features. The requirements below apply to each feature and platform included in that release.

- Mono L/R/G/B and arbitrary narrowband filters across at least two cameras and two independent datasets.
- If OSC/CFA is introduced: calibration, registration, CFA Drizzle and color reconstruction with independent validation. The current mono-only release excludes this scope.
- Raw and supplied-master Bias/Dark/Flat combinations, exposure/temperature/readout mismatch failures, and no double bias subtraction.
- Ordinary and Drizzle integration with science, weight, coverage/context, rejection and null-pixel evidence.
- Final WCS after the last geometry-changing stage, including SIP or a documented bounded linear-model residual.
- Crash/cancel/recovery, warm-cache invalidation, source drift, low disk, permission failure, and existing-destination tests.
- M3 Pro performance profile plus at least one non-M3 Apple Silicon compatibility run. Windows x64 CPU E2E is required before a Windows release, not before a macOS-only release.

No unchecked row is converted into a marketing claim.
