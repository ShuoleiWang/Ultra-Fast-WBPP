# Backend contracts and truth boundaries

The engine describes each backend with independent `available` and `executionReady` flags:

- `available` means the provider or executable can be discovered on this machine.
- `executionReady` means Ultra-Fast WBPP has a tested adapter that can execute the stage and verify its output.

QC and the portable FITS calibration/registration/integration composition are
probed at their actual callable entry points. STScI Drizzle is ready only when
its required array API is present. ASTAP, astrometry.net, and Siril descriptors
come from bounded real process probes, including required command-line options;
a PATH match alone is insufficient.

Solver `executionReady` still does not promise catalog coverage for every sky
position and scale. `doctor.endToEndExecutableReady` reports runnable adapters,
while `catalogCoverageVerified` remains false until the exact image solve
produces verified correspondence/index evidence.

## Required stage chain

```text
inventory
   -> raw-Light quality gate
   -> calibration (raw Flat + Bias required by this executor; Dark recipe-controlled)
   -> registration + transform provenance
   -> integration + rejection evidence
   -> optional drizzle
   -> required astrometric solver
   -> fail-closed WCS validator
   -> publishable solved master
```

The mono FITS chain above is connected to `run_e2e()` and the public `run`
command. Plans now report these stages as `READY` or `BLOCKED` from real probes;
they are never labeled contract-only. XISF pixel execution, prebuilt calibration
masters, CFA Drizzle, RGB combination, and mosaic execution remain explicit
fail-closed gaps in this path.

## Adding a Windows backend

A Windows executor implements `Backend`, advertises `DeviceKind.CPU`, `DIRECTML`, or `CUDA`, and registers a `BackendDescriptor` for its stage. Recipes contain logical stage options and stable backend IDs rather than macOS-specific process details, so adding Windows execution does not require a recipe schema change.

## Solver success

Solver success is conjunctive: backend status `SOLVED`, `SolutionKind.SOLVED`, positive backend confirmation, and a valid celestial WCS. Seeds and inherited coordinates do not meet this contract. The validator checks celestial axes, reference pixels/coordinates, a finite non-singular linear transform, projected pixel scale, and pixel/world round-trip stability.
