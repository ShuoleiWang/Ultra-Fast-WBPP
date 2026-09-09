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
  groups channels, shares calibration masters, runs panels, aligns final
  channels, crops their common support, and creates color products.
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
2. Measure Light quality and exclude `HARD_FAIL` and unapproved `REVIEW` frames.
3. Build raw calibration masters or validate supplied masters; calibrate Lights.
4. Estimate transforms and register accepted frames. Exact identity/integer
   half-turns copy pixels; general rotations and dithers use one Lanczos warp.
5. Normalize, reject inconsistent samples and detected transient trails, and
   integrate. Drizzle follows a separate coverage/rejection contract.
6. Solve channel products, align them onto a common sky grid, and crop the
   actual common finite footprint before RGB/LRGB and previews.
7. Record input, algorithm, output and solve evidence; validate the result.

Quality screening, relative normalization and background-gradient removal are
separate responsibilities. A passing Light can contain a smooth sky gradient.
LocalNormalization is optional and is not equivalent to PixInsight's algorithm;
unsafe local models retain explicit fallback evidence. Final images may still
need background modeling. Scientific thresholds must not change merely to make
an acceptance test pass.

## Performance

Work is tiled and bounded by configured memory budgets. Registration uses up to
eight threads where the hardware profile permits, dividing one shared scratch
budget between them. Lower-memory profiles keep fewer workers. These estimates
do not cap process RSS or the OS file cache.

Metal currently accelerates the weighted integration kernel, while CPU code
prepares tiles, masks and statistics. It does not accelerate the entire pipeline;
low average GPU utilization is therefore possible. See the scoped, reproducible
measurements in [benchmarks](../benchmarks/README.md).

## Platform and solver support

Apple Silicon/macOS is the current desktop target. Windows has development and
interface tests, but needs independent packaged scientific acceptance. Keep
platform-specific code behind the existing filesystem/process adapters.

A separately installed Astrometry.net `solve-field` and checked local indexes
satisfy the current final solve contract. ASTAP and Siril adapters are optional
and must not be described as equivalent final-validation backends. No online
image-upload solver fallback is implemented. Executable licenses and catalog
redistribution permissions are separate; see [licensing](licensing.md).
