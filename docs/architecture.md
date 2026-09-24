# Architecture

Ultra-Fast WBPP has a React/Tauri desktop, a Python processing engine, and native
CPU/Metal kernels. This page describes the current implementation. Roadmap items
are tracked separately in the [release review](release-readiness.md).

## The desktop execution path

```text
React App → useWorkflow (state) + workflow/model (pure decisions) → typed bridge commands
                       ↓
Tauri Rust: import, settings, process lifecycle, progress, result validation
                       ↓  one engine command per operation:
                       ↓  doctor --json · inventory · calibration-check · quality-check
                       ↓  blink-measure · run-project --request-json … --progress-json · catalog
Python CLI → workflows.project → workflows.single_target → pixel_pipeline
                       ↓
FITS/XISF calibration · registration · normalization · rejection/integration
                       ↓
external solve-field/ASTAP → channel alignment/crop → RGB/LRGB → result receipts
                       ↓
Rust validates returned files and sky-coordinate evidence → GUI completion
```

- [`apps/desktop/src/useWorkflow.ts`](../apps/desktop/src/useWorkflow.ts) owns UI
  state, run timing, option selection and cancellation state;
  [`workflow/model.ts`](../apps/desktop/src/workflow/model.ts) holds the pure
  decisions (for example `startBlockers`, the one list behind both the start
  button and the launch bar's reasons). The browser demo is explicitly
  separate from native execution.
- [`apps/desktop/src-tauri/src/project/`](../apps/desktop/src-tauri/src/project)
  validates the request, starts and tracks the engine process, forwards
  progress, and validates completion;
  [`sidecar/`](../apps/desktop/src-tauri/src/sidecar) discovers the bundled
  engine, reads its `doctor --json` self-report for the capability panel and
  runs the inspection and Blink commands. Native filesystem access is in Rust;
  the web view does not read astronomical files itself.
- [`workflows/project.py`](../packages/engine/src/ufwbpp/workflows/project.py)
  groups channels, shares calibration masters, runs one multi-filter run per
  target, aligns final channels, crops their common support, and creates
  colour products.
- [`workflows/single_target.py`](../packages/engine/src/ufwbpp/workflows/single_target.py)
  runs one field as named phases (source validation, screening, registration
  calibration, registration, integration passes, solving, publication);
  [`pixel_pipeline.py`](../packages/engine/src/ufwbpp/pixel_pipeline.py)
  plans a run, builds the calibration masters, calibrates and warps every
  Light and integrates each output group. Python owns most scheduling, image
  I/O and registration; the pipeline is not a fully native C++ or GPU
  implementation.

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

Every command writes a bounded result JSON to stdout, progress events to stderr
(`--progress-json`) and, on failure, a structured error with a stable `code`.
Digests of receipts, selections and installed catalog sets are computed over
the byte forms in [`integrity.py`](../packages/engine/src/ufwbpp/integrity.py);
backend truth flags are described in
[backend contracts](../packages/engine/docs/backend-contracts.md).

## Scientific work

1. Inventory frames and interpret calibration metadata.
2. Measure Light quality (1/4-scale previews, SEP detection, native-resolution
   PSF stamps, the QC reference and registration, spatial features) and run the
   quality gate; then admit frames by one of three routes. The *legacy gate*
   (default of headless runs without an explicit selection) excludes `HARD_FAIL` and unapproved `REVIEW` frames. An unattended
   `selection.policy` decides and weights every frame from the same evidence.
   An explicit selection (`selection-v1`, policy `explicit-v1`, made in the
   desktop's blink view or written by hand and passed as `--selection` or the
   request's `selection` block) names the kept Lights by content digest: the
   engine's `blink-measure` command (`blink_session.py`) had computed
   per-channel flags with absolute cross-night criteria
   (`lightframeqc.blink_flags`, policy `blink-flags-v1`), a reference frame
   per channel (`lightframeqc.blink_reference`, a PSF-signal-weight proxy)
   and registered, normalised, shared-stretch previews at 1/8 and 1/4 scale
   (`blink_previews.py`) into a create-only session directory under the
   platform cache root. The run recomputes the gate, the flags and the
   reference from the same functions, replaces the gate's admitted set with
   the selection (a KEEP on an unregistrable frame fails closed), writes
   `qc/blink.json` and records every drop and override in the receipt. The
   blink reference is not the pipeline's registration or normalization
   reference: those rules are part of the validated numerics, so an identical
   admitted set gives bit-identical products whichever route admitted it
   ([blink recipe](recipes/blink-screening.md), [legacy gate](recipes/automatic-screening.md)).
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
The supported normalization uses stellar scale and a guarded additive background grid;
unsafe grids retain explicit scalar-fallback evidence. The old LocalNormalization branch has been retired. Final images may still
need background modeling. Scientific thresholds must not change merely to make
an acceptance test pass.

### Unattended Light selection

[`ufwbpp.selection`](../packages/engine/src/ufwbpp/selection/)
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
[`native_kernels.py`](../packages/engine/src/ufwbpp/native_kernels.py)):
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

Apple Silicon/macOS and Windows x64 (Windows 10 22H2 / 11) are the desktop
targets (see [windows.md](windows.md)). The native CPU kernels build with MSVC
(`/W4 /WX /fp:strict`, static C runtime) and the Python CI job builds and
installs them on Ubuntu, macOS and Windows before the tests, so the
value-identical differential tests run against the real library on every
runner; the sidecar builder refuses a frozen engine that cannot load them and
the Windows bundle attestation refuses a DLL import the installed tree does not
provide. Operating-system differences live in `ufwbpp.platform`
(memory, topology, path limits, environment-name resolution, file-lifecycle
retries, process trees); on Windows the final solve is ASTAP verified by the
engine against the managed Astrometry.net index stars, with the same evidence
and gates as `solve-field`.

Operating-system differences live in the platform service layer
`ufwbpp.platform` (`current()` selects the `darwin`, `windows` or
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
instruction-set facts of the machine (`ufwbpp_native_cpu_features_v1`) and its own
SHA-256, all of which the `doctor` command, execution plans and the pipeline
receipt's `platform` block record. None of these facts changes a pixel: tile
sizes, memory budgets and thread counts are held result-invariant by the
differential tests, and the native `ParallelRange` claims fixed-size chunks
from an atomic counter so hybrid or throttled cores never change results, only
timing. Keep remaining platform-specific code behind these services and the
existing filesystem/process adapters.

A separately installed Astrometry.net `solve-field` with checked local indexes
satisfies the final solve contract, and so does ASTAP (`astap_cli` with its own
star database) when the engine verifies its solution against the managed index
stars, which is the Windows route ([windows.md](windows.md)); `backend: auto`
prefers `solve-field` where both are installed. Without the managed index set
ASTAP stays diagnostic-only. No online
image-upload solver fallback is implemented. Executable licenses and catalog
redistribution permissions are separate; see [licensing](licensing.md).

## Module ownership

- `workflows/contracts.py` owns run requests, explicit selections and progress.
  `workflows/project.py` and `workflows/single_target.py` coordinate runs;
  `workflows/solve.py` owns final solve/geometry verification.
- `pixel_pipeline.py` is a sequence of stages: `_plan_run` (validate and group
  every input into a `_RunPlan`), `_build_calibration_masters`,
  `_plan_light_jobs`, `_calibrate_and_register_lights`, `_integrate_group` per
  output group, then the receipt and one no-replace publication.
- `calibration_inputs.py` owns content-bound metadata and generated-master reuse.
  `image_io/fits.py` owns FITS reading/writing and sampling; `calibration.py`
  retains expression evaluation and ordinary integration. `PixelTransform`
  accepts affine and projective matrices.
- `integrity.py` owns canonical JSON and SHA-256 forms; `solvers/process.py`
  owns shared external-process and publication evidence; `publication.py`
  shares colour/mosaic create-only primitives.
- `drizzle.py` describes capabilities; `drizzle_native.integrate_drizzle_group`
  executes the native group contract.
- The registration library lives under `packages/registration/src`.
  `engine/native` contains only native kernels, their ABI, tests and benchmarks.
- Desktop Rust `sidecar/` separates bundle discovery, inspection and blink;
  `project/` separates requests, previews, completion checks, execution and the
  astrometric receipt it re-validates. `workflow/model.ts` holds UI-independent
  decisions; `useWorkflow.ts` owns React lifecycle and state.
- Scientific acceptance commands live under `tools/validation`; performance
  tools stay in `benchmarks`.

The main desktop command remains `start_project` → `run-project` (result JSON
on stdout, progress on stderr). The persisted identifiers users' files depend on
are unchanged: the `OAF*` FITS keywords, the catalog manifests and an existing
data root under the project's former name.
