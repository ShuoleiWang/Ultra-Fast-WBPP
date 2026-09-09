# Release readiness review — 2026-09-09

This is a local source review, not a stable-release certification. The repository
can be presented as an alpha project after the candidate's checks pass. Existing
local desktop acceptance is useful evidence, but is not a current multi-platform
CI result or a validation of every algorithm/input combination.

## Fixed in this review

- Standard MIT for original project code, consistent package/UI metadata,
  commercial redistribution attribution guidance, and explicit GPL XISF worker
  boundaries. Third-party notices are retained and bundled notice validation
  covers the new licensing guide and preserved license texts.
- Concise English/Chinese READMEs with a real completed GUI screenshot, source
  quick start, platform requirements and explicit scientific limitations.
- Tag-triggered releases remain drafts for human review. Source checks include
  new nonignored files and Markdown image targets; secret scanning no longer
  exempts entire resource/build paths.
- Removed confirmed unused imports, corrected transient-rejection documentation,
  and tested LocalNormalization option forwarding in both directions. The option
  defaults to off; no evidence of a lost frontend option was found.
- Corrected architecture, release and security documents to distinguish current
  implementation from planned scheduling, recovery and updater capabilities.

## Required before a stable binary release

| Priority | Gap | Acceptance criterion |
| --- | --- | --- |
| High | Reproducible Python environments | Target-specific, hash-locked runtime/build dependencies; clean-checkout builds on each advertised platform. Broad version ranges and one local environment are insufficient. |
| High | Binary licensing and supply-chain inventory | Exact component SBOM, complete notices/license texts, and GPL/LGPL corresponding-source/relinking obligations reviewed for each artifact. The current XISF worker is not MIT-only. |
| High | Independent scientific regression evidence | Redistributable fixtures covering raw/supplied masters, mixed nights, meridian flips, rejected clouds/trails, sparse groups, and supported normalization/color modes; compare pixel/flux/geometry evidence, not only successful receipts. |
| High | Background/reference selection quality | Retained data show gradients, normalization fallbacks and a borderline normalization reference. Evaluate reference selection and background behavior on representative scenes before claiming automatic finished images. Do not conflate this with proof of QC misclassification. |
| High | Supported platform packaging | Developer ID signing/notarization and clean-Mac install/uninstall/run acceptance for the claimed macOS targets. Windows needs its own evidence before being advertised as supported. |
| Medium | Onboarding and failure recovery | New-user solver/index setup, readable actionable errors, and documented recovery of preserved failed outputs. Current success-page acceptance does not establish a seamless first install. |
| Medium | Repository operations | Enable private vulnerability reporting, protected branches and required CI in the actual GitHub repository. These require maintainer setup; this local review did not change remote settings. |

## Maintainability work to do incrementally

`e2e.py`, `pixel_pipeline.py` and `project_e2e.py` are large orchestration modules.
Split them along input/calibration planning, registration, integration and
publication boundaries, preserving public imports and receipt schemas. Start
with existing differential tests before moving scientific code.

`App.tsx` combines several large workflow views and dense JSX. Extract import,
review, processing and result views behind the existing hook/bridge contract;
keep the demo visibly separate. Broad mechanical reformatting or splitting was
not included in this review because it would obscure the active scientific fixes.

Static reference checks did not establish additional private functions as safe
to delete. CLI dispatch, protocol handlers, exported APIs, platform adapters and
opt-in native paths are not dead merely because a local run did not call them.

## Validation boundary

Run `make source-check` and the appropriate suites in `make check` against the
exact candidate commit before public release. Record skips (especially real
Metal/solver/data tests) separately from passes. This review did not push,
create a release, rebuild the installed app or rerun the full real-data workflow.

Local checks performed during this review:

| Check | Result / limit |
| --- | --- |
| Python full default suite | 724 passed, 3 skipped; one fake-solver test exposed developer-catalog leakage. |
| Solver regression after isolation fix | 36 passed, 1 opt-in skipped. Only the affected test module was rerun; production solver validation was not relaxed. |
| Frontend | 52 tests passed and production frontend build passed. |
| Rust | Formatting, offline locked Clippy, and workspace tests passed; 95 tests passed and 1 opt-in test ignored. |
| Native | Incremental Release build passed; 3 CTest cases passed and the real Metal differential case was skipped in this environment. |
| Public source checks | No public-tree or local-link findings; README image links included. |
| Secret scan | No findings in the examined 8-commit history or candidate source snapshot. This is detection coverage, not proof of absence. |

No full astronomical dataset or newly packaged desktop binary was executed as
part of this source-review pass. Installed local binaries retain their previous
build until rebuilt from these changes.

## Continuation verification — 2026-09-09

The previous main task was idle and its three review subtasks were confirmed stopped before this continuation wrote files. Existing changes were retained. A focused recheck passed 83 tests with one opt-in solver test skipped, covering public-tree/link checks, release collection, bundled-runtime attestation, worker packaging, desktop bundle contracts, and solver execution. Both public source checks reported zero findings. These are current local results; the broader suite results above are retained results from the earlier review, not rerun claims.

This continuation corrected Zarr's MIT classification, explicitly identified the LGPL-2.1 GEOS libraries bundled with Shapely wheels, and added the historical CC BY 4.0 attribution for Contributor Covenant 2.1. The recorded Git author and retained Astroalign notice were checked; the limits of copyright provenance are documented in [licensing](licensing.md). Exact binary inventories, target-specific locks, signing and clean-machine acceptance remain release gates. No commit, push, publication, or application rebuild was performed.

## Follow-up image interoperability observation

A separate image-processing run found that PixInsight 1.9.4 interpreted the PC/CDELT WCS representation in one retained solved FITS master as 3600 arcseconds per pixel, although Astropy read approximately 0.647 arcseconds per pixel. The exported master also lacked the observation epoch needed for current Gaia-based processing. The processing copy was independently solved in PixInsight using acquisition metadata; the original master was preserved. Before claiming direct PixInsight astrometry/SPCC interoperability, add a consumer-level check of equivalent CD-matrix serialization and acquisition-epoch preservation. Existing structural and Astropy WCS checks do not establish that interoperability. This observation has not been fixed in the production exporter during the documentation review.
