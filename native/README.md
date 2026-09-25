# Native scientific engine

This directory contains the independent C++20 scientific kernels and their Apple Metal counterparts. It has no PixInsight/PCL dependency.

The portable target `Ufwbpp::NativeMath` provides tiled image geometry, the fused normalization/rejection/weighted integration and the portable kernels the Python engine calls through the C ABI. `Ufwbpp::NativeMetal` implements the same integration contracts on Apple GPUs and is tested differentially against the CPU implementation. Everything in this tree is reachable from the engine: image I/O and astrometric solving live in Python (`ufwbpp.image_io` and the solver backends in `ufwbpp.solvers`), and native code that the engine did not call has been removed rather than kept compiled.

```bash
cmake -S native -B build/native -DUFWBPP_BUILD_TESTS=ON
cmake --build build/native --parallel
ctest --test-dir build/native --output-on-failure
```

`python scripts/build_native_runtime.py --build-dir build/native-release` is the
one cross-platform Release chain (configure, build, `ctest`, install into
`packages/engine/src/ufwbpp/native`) that CI, the
release workflow and Windows machines use; it writes a JSON report with the
compiler and the installed library's SHA-256. On Windows the supported compiler
is MSVC 2022 (`/W4 /WX /fp:strict`, static C runtime `/MT`; the Visual Studio generator needs no
environment setup, Ninja needs a developer prompt), and the frozen engine
collects the DLL as a PyInstaller binary; the bundle attestation refuses any
import that neither the installed tree nor the operating system provides.

Strict builds disable fast-math and floating-point contraction. An accelerator result must match finite masks and rejection counts and remain within the committed numerical gate before it can satisfy a product recipe.

The production parity gate is evaluated in a documented normalized Float32 domain: `maxAbs <= 2e-6` and `RMSE <= 2e-7`. High-dynamic-range tiles are scaled into that domain before the ABI call, the Metal masked kernel uses compensated summation, and the output is restored to the original linear scale only after the gate passes. Every integration deterministically samples the first, middle, and last tile (deduplicated for short images), requires exact finite-mask and count agreement for each, and records per-tile plus aggregate worst-case evidence. Amplitude-dependent threshold relaxation is not allowed.

The build embeds the audited Metal source into the native library and runtime-compiles it with safe math (`fastMathEnabled=NO`, or `MTLMathModeSafe` on newer macOS). The stable C ABI owns an opaque reusable executor, so Python never owns Objective-C objects. A canonical external source path remains available for development differential tests. Release packaging records the native-library SHA-256 and retains a receipt-visible CPU fallback on every Apple Silicon Mac.

The portable kernels in `src/PortableKernels.cpp` (`WarpLanczos3Clamped`,
`MadRejectionMask`, `MaskedWeightedMean`) are multithreaded C++ implementations
of the Python engine's NumPy reference arithmetic. They are deliberately
operation-for-operation reproductions: the same Float32/Float64 intermediate
types, evaluation order, `nanmedian` even-count semantics, and NaN policy, so
`packages/engine/tests/test_native_kernels.py` can require
value-identical pixels, masks, and counts from both paths (only the sign of an
exact zero may differ). They are exposed through the C ABI as
`ufwbpp_native_cpu_warp_lanczos3_v2` (v1 keeps the affine-only layout),
`ufwbpp_native_cpu_mad_rejection_v2` (v1 keeps the per-pixel MAD layout; v2 adds
the row-pooled MAD and per-frame noise factors of the rejection scale model,
and reproduces v1 exactly when neither is requested),
`ufwbpp_native_cpu_masked_mean_v1`/`_v2` (v2 applies per-sample region weight
maps), `ufwbpp_native_cpu_tile_offsets_v1`, `ufwbpp_native_cpu_radon_peaks_v1`
(the transient-trail line search), `ufwbpp_native_cpu_drizzle_v1`,
`ufwbpp_native_cpu_debayer_bilinear_v1`, `ufwbpp_native_cpu_add_offset_grid_v1`
(the bilinear normalization offset grid added to integration rows; kernel id
`native-cpu-offset-grid-v1`) and `ufwbpp_native_lanczos3_table_v1` (the
deterministic weight table behind registration kernel v3), are compiled with `-fno-fast-math
-ffp-contract=off`, and are covered by `tests/PortableKernelTests.cpp`.

On arm64 the warp evaluates four output pixels per NEON lane group: each lane
runs its pixel's scalar operations in the scalar order (the Float64 table
weights, the 36 products and their running Float32 sum, the support bounds
as `std::min`/`std::max` selects, the clamp), so the output is the scalar
output bit for bit, the sign of zero included. When every tap weight is
nonzero, a finite sum proves every sample was taken, and the bounds then come
from `vminq`/`vmaxq` unless one of them is a zero; any other window reruns the
exact lane route, and windows that need the edge clamp run the scalar code.
`TestWarpMatchesTheScalarReferenceBitForBit` holds the three routes to the
scalar reference on NaN, infinite, signed-zero and negative samples, integer
shifts, projective maps, row tails and thread counts. Other architectures run
the scalar code.
Each kernel splits its range with `ParallelRange`: worker threads (and the
calling thread) claim fixed-size chunks from an atomic counter (8 warp rows,
4096 rejection pixels, 8192 reduction pixels, one normalization tile), so a
slow core never holds the tail of a range while the others idle; every item
is processed exactly once and its result never depends on the thread or chunk
that ran it, which the thread-count invariance tests assert on ranges that end
in a partial chunk.

`ufwbpp_native_cpu_features_v1` (`src/CpuFeatures.cpp`) reports the compiled
architecture, the instruction-set extensions the running OS can use (cpuid plus
the XSAVE state check on x86-64: `sse4.2`, `avx`, `avx2`, `fma`, `avx512*`;
`neon` on arm64) and the cpuid brand string. It is report-only evidence for
receipts and `doctor`; no kernel selects code by it.

Ordinary integration uses a two-stage exact full-stack path. CPU code computes the same per-sample median/MAD rejection decision used by the portable integrator; `fused_masked_weighted_integration` then reduces every frame in one Metal request. It never averages partial batches. The product policy admits up to 512 frames; resource/capability failure falls back to CPU with `inputFramesTruncated=false` and a reason in the receipt. The older native linear-fit rejection kernel remains limited to 64 frames because it uses private per-thread sorting storage and is not used for larger ordinary stacks.

The synthetic Metal throughput benchmark is opt-in and refuses to replace its JSON output:

```bash
cmake -S native -B build/native-bench \
  -DUFWBPP_BUILD_TESTS=OFF -DUFWBPP_BUILD_BENCHMARKS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/native-bench --parallel
build/native-bench/ufwbpp_integration_benchmark \
  --metal-source "$PWD/native/metal/FusedIntegration.metal" \
  --output "$PWD/build/benchmarks/m3-pro-integration.json" \
  --width 6252 --height 4176 --frames 96 --tile-rows 64 \
  --warmup-runs 1 --repetitions 5
```

Configuration now fails unless a single-configuration build is `Release`, and a Debug multi-configuration binary refuses to run. The report uses the median of five measured runs after one warmup, retains every wall-time sample, verifies identical output hashes and complete sample accounting across runs, and records build type, compiler, Git commit/dirty state, and Metal-source SHA-256. It measures synthetic frame preparation plus the fused Metal kernel; it is not an end-to-end calibration/registration/Drizzle benchmark and labels that boundary in the report.
