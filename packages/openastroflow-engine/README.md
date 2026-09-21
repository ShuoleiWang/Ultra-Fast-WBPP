# Ultra-Fast WBPP Engine

`ultra-fast-wbpp` is the primary headless CLI for the typed control-plane and portable execution package used by the Ultra-Fast WBPP GUI and automation worker. The legacy `openastroflow-engine` command remains a command-name compatibility alias and reports the current Ultra-Fast WBPP product identity. The package inventories NINA FITS/XISF projects, validates recipes and backend capabilities, runs the quality-gated mono E2E path, probes runtime hardware capabilities, and exchanges stable NDJSON messages.

Implemented execution paths remain fail-closed: a drizzle result is a second integration of the ordinary integration's own inputs (calibrated frames, registration matrices, normalization, weights and per-sample rejection) on a 1x–4x grid through the native kernel, with science, weight and coverage extensions, a verified receipt, and a newly solved final WCS. An RA/Dec hint or seeded WCS is never reported as an astrometric solution. Unsupported recipe/backend combinations remain blocked during planning.

## Install for development

From this monorepo, install the sibling QC package first and then the engine:

```bash
python -m pip install -e ../light-frame-qc
python -m pip install -e '.[test]'
pytest
```

Published installations will use the declared `light-frame-qc>=0.3.0` dependency.

Release packaging first installs the CMake native target into `src/openastroflow_engine/native`, then builds the wheel. Because the ctypes runtime is a native binary, wheels are deliberately tagged by platform (for example `py3-none-macosx_14_0_arm64`) rather than incorrectly published as `py3-none-any`. The bundled shader is embedded in that library, so an installed macOS wheel does not depend on the source checkout or a separate `.metal` file.

## CLI recipes

Inspect the machine and backend seams:

```bash
ultra-fast-wbpp doctor
ultra-fast-wbpp doctor --json
```

Recursively inventory one or more NINA acquisition folders without decoding their pixel arrays:

```bash
ultra-fast-wbpp inventory ~/Astro/NINA/Target-1 ~/Astro/Calibration --output inventory.json
```

Build an auditable plan from the default recipe:

```bash
ultra-fast-wbpp plan ~/Astro/NINA/Target-1 ~/Astro/Calibration --output plan.json
```

Run the complete ordinary path into a brand-new directory. The command
recursively identifies Light/Flat/Dark/Bias and MasterBias/MasterDark/MasterFlat roles, excludes unapproved REVIEW and all HARD_FAIL Lights,
calibrates, registers, integrates by filter, solves every master, validates the
new WCS, renders previews, and publishes atomically:

```bash
ultra-fast-wbpp run \
  ~/Astro/NINA/Target-1 ~/Astro/Calibration \
  --output ~/Astro/Results/target-1-run-001 \
  --mode ordinary \
  --solver-chain astrometry-net \
  --progress-json
```

`--output` must not exist. Missing solver executables fail before pixel work;
missing index/catalog coverage fails the individual solve and publishes only a
separate `.unsolved` evidence directory. It never promotes a pointing hint or
inherited header to `SOLVED`.

Astrometry.net and its indexes are user-installed external components, not a built-in solver. The strict final gate requires the app-managed index/config receipt. ASTAP (`astap_cli` with its own star database) satisfies the same gate when the engine verifies its solution against the managed index stars, which is the Windows route (`backend: auto` prefers `solve-field` where both are installed); without the managed index set ASTAP is diagnostic-only.

Enable drizzle in the recipe contract (scale 1–4, drop shrink, square/circular/gaussian/point kernel). Execution requires the native kernel library:

```bash
ultra-fast-wbpp run \
  ~/Astro/NINA/Target-1 ~/Astro/Calibration \
  --output ~/Astro/Results/target-1-drizzle-001 \
  --mode drizzle --drizzle-scale 2 --drop-shrink 0.9 --drizzle-kernel square
```

A complete recipe can be supplied with `--recipe recipe.json`. CLI flags override the corresponding recipe fields. The E2E implementation requires a raw Bias group or exactly one compatible MasterBias, and a raw Flat group or compatible MasterFlat for every Light filter. Dark is optional unless the recipe marks it required; when any Dark source is supplied, every Light exposure must have an exact, temperature-compatible raw Dark group or MasterDark. The default astrometric policy is required and cannot be silently downgraded.

Copy [`examples/default-recipe.json`](examples/default-recipe.json) for a conservative starting point. Supplied masters are content-hashed, role-checked, and reused read-only without another integration or bias subtraction. A raw group and supplied master may coexist only for disjoint filter/exposure profiles; ambiguous matches block planning. [`examples/drizzle-2x-recipe.json`](examples/drizzle-2x-recipe.json) enables a 2× drizzle through the native kernel (`backend: auto`, drop shrink 0.9, square kernel); sampling, dither and coverage evidence is recorded as advisory.

Ordinary integration combines the registration quality weight with the independently measured noise weight for each admitted Light. The accepted-sample count, accepted/total coverage fraction, and robust-rejection count are written as real per-pixel FITS maps and promoted beside `coverage.json`; receipts bind their paths, statistics, and checksums.

Every non-reference ordinary Light uses a full-resolution refined transform and the default normalized 6×6 `lanczos-3-clamped` resampler. Complete finite kernel support is required, the declared numeric domain and existing local extrema bound new interpolation excursions, and the reference frame remains an identity-exact copy. The resampler, support-aware crop, and per-frame evidence are recorded in the pixel-pipeline receipt. The legacy bilinear option is retained only for explicit diagnostics and is not the product default.

FITS and supported XISF pixel inputs share the same calibration path. XISF is first decoded through a size-, XML-, codec-, and working-set-bounded private FITS staging bridge; inventory support alone never implies executable pixel support. Optional LocalNormalization is disabled by default and uses a conservative registered-coordinate scale/offset grid with correlation, coefficient, coverage, and residual gates for both ordinary and Drizzle integration. See the root recipe guide for the exact fail-closed boundary and content-bound metadata overrides for legacy masters.

REVIEW is fail-closed by default. A manual exception must name the exact Light content SHA-256, the canonical Quality Gate policy digest, and the complete approval request digest emitted in `qc/manifest.json`. Any source, calibration set, QC policy, integration, registration, drizzle, or solver-quality request change invalidates the approval. HARD_FAIL cannot be manually admitted through this mechanism; see the N.I.N.A. recipe guide for the two-run review workflow.

## Worker protocol

`openastroflow-worker` implements the checked-in Rust app-core protocol v1.
stdin and stdout are independent, ordered NDJSON streams. Each direction starts
with a canonical `handshake` envelope at sequence `0`; the controller then sends
an immutable `plan` followed by `execute`. The worker stores the role-bound plan,
runs the same E2E function as the CLI, streams `progress`, and emits real
stage/artifact receipts or a structured `error`.

The GUI must not reproduce role binding or manifest hashing. Generate its plan
envelope with the same engine/sidecar binary:

```bash
ultra-fast-wbpp controller-plan \
  ~/Astro/NINA/Target-1 ~/Astro/Calibration \
  --mode ordinary --session-id session-42 --output controller-plan.json
```

The frozen sidecar accepts `controller-plan ...`; with no arguments it remains a
long-running worker. The obsolete `{"type":"plan","inputs":[...]}` format is
rejected because it bypasses canonical Project/Recipe and input-manifest
identity.

The complete message contract is documented in [`docs/worker-protocol.md`](docs/worker-protocol.md); backend truth boundaries are documented in [`docs/backend-contracts.md`](docs/backend-contracts.md).

## Hardware policy

- Every Apple-silicon Mac (`arm64`/`aarch64`) targets the generic CPU profile; Metal is advertised only after a real native executor/device/ABI probe succeeds.
- Apple M3 Pro receives the separate `apple-m3-pro-tuned-v1` optimization profile.
- Other M1/M2/M3/M4 and future M-series variants remain on the safe generic profile unless a measured tuning profile is added; they are `compatible-generic`, not performance-validated by the current evidence.
- Windows x64 runs the `windows-cpu` profile (validated on one Ryzen 7 5800H laptop and in hosted CI; see `docs/windows.md`). Its descriptor is intentionally structured so future GPU backends can be added without changing recipes; there is no GPU compute on Windows and the installer is not yet signed.

Hardware capability is not the same as an installed pixel executor. `doctor`
reports callable/process probes separately. `endToEndExecutableReady` means the
pipeline and a solver binary can run; `catalogCoverageVerified` remains false
until a specific solve proves compatible local index/catalog coverage, so the
static `endToEndReady` aggregate is intentionally conservative.

Final ordinary Light integration accepts `PipelineParameters(ordinary_integration_backend=...)` with `auto`, `portable-cpu`, `generic-apple-metal`, or `m3-pro-tuned`. `auto` selects the generic Metal path on Apple Silicon and the tuned path only on an identified M3 Pro with sufficient memory. The native library is discovered from the installed package, `OPENASTROFLOW_NATIVE_LIBRARY`, or a source-tree build; a missing/incompatible library, unavailable Metal device, resource rejection, or failed CPU↔GPU numerical gate reruns the complete group on CPU and records the reason. No fallback truncates input frames.

For stacks through 512 frames, CPU computes one full-stack median/MAD rejection mask per tile and Metal performs one weighted reduction over all frames. Partial-stack means are never combined. The mandatory normalized-Float32 parity limits remain `maxAbs <= 2e-6` and `RMSE <= 2e-7`; high-dynamic-range tiles are scaled before the ABI call and restored only after the gate passes. Receipts record the selected backend, device, profile, tile rows, active CPU workers, in-flight buffers, native-library digest, `fastMath=false`, frame count, normalization domain, and finite-mask/rejection-count/max-absolute/RMSE evidence.

`IntegrationParameters.max_memory_bytes` keeps its historical 256 MiB default as the portable-CPU budget. On an accelerator that unchanged default means “use the capability-derived integration budget” (up to 4 GiB on the validated 36 GiB M3 Pro). Supplying any non-default value makes it an explicit hard cap; Metal reduces CPU preparation concurrency and tile rows to honor it, or records a full CPU fallback if even one row cannot fit.

## Astrometric fail-closed rule

`validate_solver_result()` requires all three of the following:

1. the backend returned `SOLVED`;
2. the backend marked the result as a newly solved solution, not a seed or unchecked inherited header;
3. the resulting celestial WCS contains a finite, non-singular transform and survives Astropy round-trip validation.

Object coordinates, `CRVAL` values, or a backend process exit code alone never satisfy this contract.
