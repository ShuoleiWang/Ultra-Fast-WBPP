# Changelog

All notable changes are documented here. The format follows Keep a Changelog and the project uses Semantic Versioning after the first stable release.

## [Unreleased]

### Fixed

- Use standard mono/master conventions in the desktop without per-file metadata approval; preserve unrecorded values, remove zero/Light-derived defaults, and retain sparse overrides only in advanced settings. Known conflicts and required calibration dependencies remain enforced.

- Enable native title-bar dragging and keep controls interactive; expose content-bound mono and Master confirmations directly on Import, with clearer unverified-calibration diagnostics.

- Allow individual Raw/Master Flat, Dark and Bias batches to be appended without requiring Lights in that same import; full-project and per-file validation remain enforced.

- Preserve supplied XISF master identities through shared project calibration so child panels can validate the original metadata declarations.

- Match declared master-camera/readout metadata with FITS headers independent of capitalization, while rejecting unknown or different acquisition modes.

- Preserve faint physical values when decoding unsigned 32/64-bit FITS storage before converting to float32.
- Exclude non-positive flat responses and infinite integration samples; apply rejection minimums per pixel so sparse coverage does not trigger unsupported clipping.
- Restore automatic role detection between imports and freeze inputs during quality review or processing.
- Recheck solver installation without discarding the project, clear stale readiness on failure, and explain missing inputs beside the disabled start button.

### Changed

- License original project code under standard MIT; document commercial attribution and the separate GPL obligations of the current XISF worker. Earlier releases retain their original license.
- Refresh public documentation with a real GUI screenshot, correct implemented architecture and platform boundaries, and keep tag-triggered binary releases in draft until review.

- Renamed the user-facing project and desktop application to Ultra-Fast WBPP. The existing `openastroflow` module, worker, protocol, schema, receipt kinds, and `OAF*` FITS keywords remain unchanged for compatibility and evidence continuity; user-visible FITS `ORIGIN`/`HISTORY` branding changes, so newly written whole-file hashes are expected to differ.
- Preserved the prior WiX UpgradeCode. Because the macOS app filename changed, users of the withdrawn OpenAstroFlow alpha must remove only `/Applications/OpenAstroFlow.app` before installing the renamed app; the compatibility data directory, managed catalogs, and user outputs must be retained.
- Made ordinary-integration MAD rejection independent of memory- and hardware-selected tile sizes, unified CPU/Metal rejection decisions, and aligned NaN rejection-map semantics.
- Reused content-bound registration-calibration masters in single-field ordinary E2E runs, removed the second XISF conversion, and bound all pixel work to verified private input snapshots. On the retained M3 Pro raw-B fixture, two final runs improved from a 106.895-second corrected baseline to 85.391/84.070 seconds with exact final FITS, map, transform, weight, and WCS parity.
- Reduced the desktop workflow to `Import → Review → Process`, made English the fresh-install default with a persistent Simplified Chinese option, connected Review to the real per-Light Quality Gate, and added content-bound visual approval only for safe single-panel REVIEW cases.
- Removed repeated runtime/catalog hashing on every role change, made language changes event-safe, blocked duplicate starts while a task is launching or running, paginated PASS evidence, scoped provider links, and terminated every managed process group on application exit.
- Added explicit FITS/XISF pixel numeric-domain reconciliation. UInt16 Lights and normalized Float32 PixInsight MasterBias/MasterDark inputs now use the recorded 65535:1 additive scale in both registration calibration and final pixels; ambiguous Float FITS masters require a SHA-bound user declaration.
- Replaced production bilinear registration with normalized, domain-bounded 6×6 Lanczos-3 while keeping the reference identity-exact. The complete 38-Light QHY268M B run passed all 25 independent PixInsight-reference checks in 466.116 seconds on the retained M3 Pro.
- Moved ordinary global normalization to content-bound same-filter stellar scale plus a guarded low-frequency additive grid, applied mosaic seam gain/offset corrections before coaddition, and retained explicit non-equivalence and real-data validation boundaries.
- Embedded canonical GPL, notice, and third-party-notice files in the desktop bundle, added byte-for-byte legal-resource attestation, and made local macOS prerelease builds use the same Tauri entry point as release CI.

## [0.1.0-alpha.1] - 2026-09-01

### Added

- Clean, independent OpenAstroFlow public monorepo and GPL-3.0-or-later licensing.
- Five-step Tauri/React native desktop GUI connected to the real `run-project --request-json` controller, with file/folder selection, role and mono-CFA confirmation, Quality Gate, recipe configuration, process-tree cancellation, catalog installation progress, and a Rust-revalidated WCS result gate. The browser build remains an explicitly labelled non-executing demo.
- Rust app-core with versioned NDJSON workers, recipes, artifact receipts, solver/drizzle publication gates, safe paths, and no-replace publication contracts.
- Portable Python N.I.N.A. inventory and planning worker with generic Apple M-series, measured M3 Pro tuned, and unvalidated Windows CPU interface profiles.
- Independent Light Frame QC 0.3.0 package and fail-closed WBPP-ready preparation path.
- Raw or supplied-master Bias/Dark/Flat calibration, multi-exposure registration, registration-quality/noise weighted ordinary integration, per-pixel coverage/rejection evidence, and final managed-catalog Astrometry.net validation.
- Portable C++20 calibration/integration/TAN math library, stable C ABI, POSIX FITS/XISF codec, Apple Metal worker, and strict CPU↔Metal differential tests.
- Portable tiled Drizzle execution with registration-derived dither gates, per-frame MAD/sigma rejection masks, 90% coverage enforcement, and tamper-evident artifact receipts.
- Bounded XISF-to-private-FITS pixel bridge and an opt-in conservative LocalNormalization stage that explicitly does not claim PixInsight algorithmic equivalence.
- Mono RGB/LRGB and solved-panel mosaic modules with synthetic four-panel × RGB execution coverage, one-time shared raw-Dark calibration, fresh per-filter mosaic solves, and explicit `PROPAGATED_VERIFIED` post-reprojection WCS provenance.
- A manifest-attested PyInstaller onedir runtime embedded under Tauri Resources. The final local arm64 DMG passed ad-hoc hardened-runtime signing, deep code-sign verification, DMG verification, runtime-tree attestation, and sub-two-second cold startup.
- Explicit offline Astrometry.net catalog selection, terms acknowledgement, resumable verified download, immutable installed-set receipts, and selected-index byte binding. Catalog redistribution terms remain unresolved and no indexes are bundled.
- Sanitized M3 Pro real raw-B E2E evidence: 8/8 Lights accepted, tuned Metal integration, 106.4-second wall time, and a final 57-match/1.377-arcsecond managed Astrometry.net solution. Raw data and output pixels are not redistributed.
- User recipes and architecture, calibration, Drizzle, astrometry, Apple Silicon, Windows, security, and release documentation.
- Public hosted Python 3.11/3.12, Rust, native C++, React, Apple Silicon Metal, CodeQL, and dependency-review gates across macOS, Windows, and Linux-development targets.
- Commit-bound release metadata, checksums that cover the metadata itself, GitHub build-provenance attestations, and read-only mounted-DMG runtime verification for tagged macOS arm64 prereleases.

### Security

- Raw astronomy files, catalogs, credentials, signing material, local paths, and private application artifacts are excluded by default.
- Original frames remain read-only and final results require owned staging, checksummed receipts, and no-replace publication.
- Unix cancellation verifies an isolated child process group before signalling it and fails safely to direct-child termination if isolation is uncertain.

### Known pre-1.0 limits

- OSC/CFA calibration, CFA-preserving Drizzle, Debayer and color reconstruction are blocked; Bayer data is never processed as mono.
- Drizzle, LocalNormalization, RGB/LRGB, and multi-panel mosaic have synthetic evidence only. Real-data acceptance remains a release gate for all four.
- Astrometry.net is a required user-installed external solver with separately installed managed indexes; there is no built-in solver. ASTAP is diagnostic-only for the strict final gate.
- Checkpoint/resumable pixel execution is not claimed. Cancellation is safe but a rerun creates a new output directory.
- Only the 36 GiB M3 Pro has retained performance and real scientific E2E evidence. Other M-series machines use the generic capability profile but are not performance-validated.
- Hosted Windows Python 3.11/3.12, Rust, and native C++ jobs pass, but real scientific E2E, release-grade handle/reparse-point validation, installed-runtime attestation, public installer smoke, and signing have not completed.
- The local immutable onedir bundle/attestation gate passes on arm64 macOS. The prerelease remains ad-hoc signed and is not a stable download; Developer ID notarization and Windows Authenticode are required for stable releases.
- Existing real-data evidence remains private and is represented only by sanitized receipts; it does not establish broad camera, filter, sky, scale, or catalog parity.
