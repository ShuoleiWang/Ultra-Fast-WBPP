# Architecture

Ultra-Fast WBPP has a React/Tauri desktop, a Python processing engine, and native
CPU/Metal kernels. This page describes the current implementation. Roadmap items
are tracked separately in the [release review](release-readiness.md).

## The desktop execution path

```text
React App → useWorkflow → typed bridge commands
                       ↓
Tauri Rust: import, settings, process lifecycle, progress, result validation
                       ↓  run-project --request-json … --progress-json
Python CLI → project_e2e → e2e → pixel_pipeline
                       ↓
FITS/XISF calibration · registration · normalization · rejection/integration
                       ↓
external solve-field → channel alignment/crop → RGB/LRGB → result receipts
                       ↓
Rust validates returned files and sky-coordinate evidence → GUI completion
```

- [`apps/desktop/src/useWorkflow.ts`](../apps/desktop/src/useWorkflow.ts) owns UI
  state, run timing, option selection and cancellation state. The browser demo
  is explicitly separate from native execution.
- [`apps/desktop/src-tauri/src/project.rs`](../apps/desktop/src-tauri/src/project.rs)
  validates the request, starts and tracks the Python child process, forwards
  progress, and validates completion. Native filesystem access is in Rust; the
  web view does not read astronomical files itself.
- [`project_e2e.py`](../packages/openastroflow-engine/src/openastroflow_engine/project_e2e.py)
  groups channels, shares calibration masters, runs one multi-filter E2E run
  per target, aligns final channels, crops their common support, and creates
  color products.
- [`e2e.py`](../packages/openastroflow-engine/src/openastroflow_engine/e2e.py) and
  [`pixel_pipeline.py`](../packages/openastroflow-engine/src/openastroflow_engine/pixel_pipeline.py)
  orchestrate the scientific work. Python owns most scheduling, image I/O and
  registration; the pipeline is not a fully native C++ or GPU implementation.
- [`crates/app-core`](../crates/app-core/README.md) supplies typed project,
  validation and execution contracts. Its abstractions are not a claim that
  every desktop operation already runs through a durable task scheduler.

## Data and correctness boundaries

Original inputs are read-only. The engine binds processing evidence to input
identities and writes intermediate data into owned staging directories. It
uses no-replace publication for new result directories; supplied images and
previous results are not overwritten.

The **Python project runner** materializes the completed result directory and
returns its receipt. Rust then reopens and verifies the returned products before
showing success. A failure of this final desktop check can therefore leave a
result directory on disk without a successful GUI result. File existence alone
is never a completion signal. Do not describe this as a Rust-owned atomic
transaction over the entire scientific run.

The project CLI writes a bounded result JSON to stdout and progress events to
stderr. A separate versioned NDJSON worker interface supports controller/worker
contracts; it is not the transport used for every desktop operation. See
[protocol](../protocol/README.md) and [backend contracts](../packages/openastroflow-engine/docs/backend-contracts.md).

## Scientific work

1. Inventory frames and interpret calibration metadata.
2. Measure Light quality and exclude `HARD_FAIL` and unapproved `REVIEW` frames (or, under an unattended `selection.policy`, decide and weight every frame from the same evidence).
3. Build raw calibration masters or validate supplied masters; calibrate Lights.
4. Estimate transforms and register accepted frames. Every filter of a
   target registers onto one reference frame with a projective model fitted
   to core-weighted star centroids, so frames of other nights or hour angles
   (tilt, differential refraction) and asymmetric PSFs do not leave
   filter-dependent offsets. Exact identity/integer half-turns copy pixels;
   general rotations and dithers use one Lanczos warp.
5. Normalize, reject inconsistent samples and detected transient trails
   (each frame's residual against the temporal median is integrated along
   every line of every dyadic length with a fast Radon transform, so faint
   satellites are found from their whole length and only that frame's
   samples inside the corridor are dropped), and integrate. Frames are matched to the frame with the flattest large-scale
   background of acceptable quality; when the sky level varies enough across
   the group, the sky-proportional part of each frame's background (residual
   flat-field structure) is separated from the object by regression in the
   sensor frame (through each frame's registration transform, so a meridian
   flip does not move it) and removed before matching, and the master's
   background tilt is moved to the flattest non-negative mix of the frames'
   own tilts, so gradients of different nights cancel where they disagree.
   The filter masters of one run are cropped to one common rectangle so
   they share their pixel grid. With drizzle enabled, each group is then
   integrated a second time on a 1x-4x grid by the native drizzle kernel
   from the same calibrated frames, matrices, normalization, weights and
   per-sample rejection decisions (see `docs/recipes/drizzle.md`).
6. Solve every channel product independently; masters that share a grid
   verify each other's solutions at solver precision and then carry one of
   them, so the project copies them onto the reference grid without any
   resampling. Only mosaics of separately solved panels are reprojected. The
   actual common finite footprint is cropped before RGB/LRGB and previews.
7. Record input, algorithm, output and solve evidence; validate the result.

Quality screening, relative normalization and background-gradient removal are
separate responsibilities. A passing Light can contain a smooth sky gradient.
LocalNormalization is optional and is not equivalent to PixInsight's algorithm;
unsafe local models retain explicit fallback evidence. Final images may still
need background modeling. Scientific thresholds must not change merely to make
an acceptance test pass.

### Unattended Light selection

[`openastroflow_engine.selection`](../packages/openastroflow-engine/src/openastroflow_engine/selection/)
turns the Light Frame QC evidence into per-frame decisions without a human
review gate when a recipe sets `selection.policy` to `unattended-v1` (the
default `legacy-gate` keeps the historical PASS-only admission). Guards
exclude frames the pipeline cannot use (hard gate failures, failed
registration, transparency below the normalization floor, extreme extinction);
REVIEW codes that only mean "not enough evidence" keep the frame at reduced
weight; defect codes enter a gray zone whose PSF cut-offs follow the chosen
priority (depth, balanced, resolution). The PSF features come from Light
Frame QC's native-resolution star stamps (`lightframeqc.native_psf`:
half-flux radius, FWHM, wing fraction) and fall back to the preview FWHM when
too few stars qualify. The decision's confidence, times a PSF factor for the
chosen priority, scales the frame's registration quality weight. During
integration a tile observer computes each frame's leave-one-out counterfactual
on 64-row statistics tiles (block-noise depth, second-order background
residual, FWHM proxy, tile bootstrap) so `qc/selection.json` records whether
the master would be better without the frame. A frame the counterfactual
confirms harmful (enough tiles, interval beyond the threshold, outlier against
its group) is removed and the group is integrated again, at most three passes
and within the soft-exclusion guard; normalization references and the
two-frame panel minimum are never removed, and the receipt's `reintegration`
block lists what was removed and why the rest was kept. The pixels of the
final pass are the product; earlier passes are deleted. With
`selection.regionWeights` the QC grid of a frame with a blocked or dimmed
region becomes a per-frame weight map (`selection/region.py`) that the
weighted mean applies sample by sample (frame expressions carry the node grid;
the native masked-mean V2 kernel and the NumPy reference agree bit for bit),
so the clean part of a partly occluded or partly clouded frame is used and the
rest contributes nothing; the coverage map then reports the effective weight
fraction. See
[frame-selection-plan.md](frame-selection-plan.md) and
[frame-selection-implementation.md](frame-selection-implementation.md).

## Performance

Work is tiled and bounded by configured memory budgets. These estimates do not
cap process RSS or the OS file cache.

The three hot loops of the ordinary pipeline run in multithreaded native CPU
kernels (`engine/native/src/PortableKernels.cpp`, bound through
[`native_kernels.py`](../packages/openastroflow-engine/src/openastroflow_engine/native_kernels.py)):
the Lanczos-3 registration warp, the full-stack median/MAD rejection decision
(v2: the noise part of each pixel's scale is the pooled MAD of its row window
and every frame is judged against its own noise, so small stacks no longer clip
good samples where the per-pixel MAD is low by chance), and the weighted
reduction. Frame weights are the inverse variance of 4x4 block means, which is
insensitive to the sub-pixel phase of the resampling. Each kernel reproduces the NumPy reference
arithmetic operation for operation, so the two paths publish identical pixels;
the NumPy path remains the portable fallback and every receipt names the kernel
that ran. Calibration and registration are fused: a Light is decoded once,
calibrated in memory against once-decoded masters, warped directly, and only the
registered frame is written (streaming SHA-256, no reread). The number of Lights
in flight follows the registration memory budget; each warp receives the
remaining CPU share. Sources are read in place; the E2E path keeps one inventory
hash and one final publication hash per original.

The Metal path accelerates only the weighted integration kernel while CPU code
prepares tiles, masks and statistics, so `auto` ordinary integration now selects
the native CPU kernels whenever they are loaded: the measured M-series cost is
the rejection statistics, and the Metal weighted mean's host-side normalization,
buffer copies and parity reruns exceed its few milliseconds of GPU time. An
explicit `generic-apple-metal` or `m3-pro-tuned` request still runs the audited
Metal path, and every receipt names the backend and kernels that ran. FITS
intermediates are written through buffered sequential writes; only published
products are fsynced.

Statistics passes that sample the images (integration statistics, the
group-wide rejection sigma floor, global-normalization pairing) gather all
sampled rows of a frame in one read and evaluate the per-coordinate medians in
one vectorized pass that reproduces NumPy's `nanmedian` arithmetic; the
per-star registration refinement, the residual-background cell statistics and
the transient binning pass are likewise batched, and the last submission round
of fused warps spreads over the idle cores. These are pure overhead removals:
every replacement is held to its one-row, one-star or one-frame reference by a
differential test. (Light Frame QC group
analysis stays sequential: its astroalign bootstrap is GIL-bound Python, and
threads measured slower.) See the scoped, reproducible measurements in
[benchmarks](../benchmarks/README.md).

## Platform and solver support

Apple Silicon/macOS is the current desktop target. Windows (x86-64, Windows
10/11) is being brought up in phases (see [windows.md](windows.md)): the native
CPU kernels build with MSVC (`/W4 /WX /fp:strict`), the Python CI job builds and
installs them on Ubuntu, macOS and Windows before the tests so the
value-identical differential tests run against the real library on every
runner, and the sidecar builder refuses a frozen worker that cannot load them.
Windows still needs independent packaged scientific acceptance before it is a
support claim.

Operating-system differences live in the platform service layer
`openastroflow_engine.platform` (`current()` selects the `darwin`, `windows` or
`linux` services behind one `PlatformServices` protocol). It reports the facts
that tuning and receipts consume: physical/available memory (`sysconf`,
`GlobalMemoryStatusEx`), the CPU topology (physical/performance/efficiency
cores, SMT, from `sysctl`, `GetLogicalProcessorInformationEx` or
`/proc/cpuinfo`), the native library file name, and later the filesystem and
process primitives that are still duplicated across modules. Every probe is a
pure function of its raw input, so each platform's parser is tested on every
host, and `detect_hardware()` never claims a value it did not measure (an
injected host reports memory as `unavailable`; a failed probe reports
`fallback`). `HardwareProfile` and `ExecutionTuning` carry these facts, and the
tuning itself is a table keyed by platform, CPU family (`APPLE_M`, `X86_64`,
`GENERIC`), memory band and core count; the native library adds the
instruction-set facts of the machine (`oaf_native_cpu_features_v1`) and its own
SHA-256, all of which the `doctor` command, execution plans and the pipeline
receipt's `platform` block record. None of these facts changes a pixel: tile
sizes, memory budgets and thread counts are held result-invariant by the
differential tests, and the native `ParallelRange` claims fixed-size chunks
from an atomic counter so hybrid or throttled cores never change results, only
timing. Keep remaining platform-specific code behind these services and the
existing filesystem/process adapters.

A separately installed Astrometry.net `solve-field` and checked local indexes
satisfy the current final solve contract. ASTAP and Siril adapters are optional
and must not be described as equivalent final-validation backends. No online
image-upload solver fallback is implemented. Executable licenses and catalog
redistribution permissions are separate; see [licensing](licensing.md).
