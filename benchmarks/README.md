# Benchmarks

Benchmarks are performance evidence with an explicit scientific and I/O boundary; they are not feature-completeness claims.

[`m3-pro-astrometry-net-real-fixture-20260901.json`](results/m3-pro-astrometry-net-real-fixture-20260901.json) records a real retained master solved through the public Astrometry.net adapter in 5.451 seconds wall time. The adapter verified the binary solved marker, `.wcs`/`.new` agreement, 56 unique `.corr` correspondences, a 1.404-arcsecond RMS, and pre/post stat+SHA binding from `.match` INDEXID 4107 to a managed installed-set receipt, checked manifest, and concrete index file. The source image is not redistributable and is not in this repository; the report deliberately contains neither pixel data, source hash, private filesystem path, exact argv, nor raw process logs.

[`m3-pro-metal-e2e-validation-20260901.json`](results/m3-pro-metal-e2e-validation-20260901.json) is execution-path evidence rather than a throughput benchmark. It records a real M3 Pro run through the Python adapter, opaque C ABI, embedded safe-math Metal shader, 96-frame full-stack masked integration, CPU/Metal numerical gate, and a complete synthetic E2E publication with a verified final WCS.

[`m3-pro-dunpai3-full-b-20260905.json`](../validation/m3-pro-dunpai3-full-b-20260905.json) is the current full-E2E evidence: 38 private QHY268M B Lights at 6252×4176, 20 raw Flats, supplied normalized-XISF MasterBias/MasterDark, full-resolution Lanczos-3 registration, global normalization, tuned Metal integration, and a managed final solve completed in 466.116 seconds. All 25 independent PixInsight-reference checks passed. This is one private fixture and not a universal latency or WBPP-equivalence claim.

[`m3-pro-real-raw-b-optimized-e2e-20260902.json`](../validation/m3-pro-real-raw-b-optimized-e2e-20260902.json) is a historical 8-Light optimization fixture. A 106.895-second corrected baseline rebuilt calibration masters twice; two content-verified snapshot+master-reuse runs took 85.391 and 84.070 seconds with exact candidate parity. It predates the numeric-domain and Lanczos-3 accuracy fixes and is retained only to document that optimization, not as the current product baseline.

The two checked-in `openastroflow-metal-integration-benchmark-v1` kernel reports below are historical one-shot observations. They do not record build configuration, compiler/source provenance, warmup, or repeated samples, and their 40-frame versus 96-frame throughput differs by almost 10×. Do not compare or cite them as a current release baseline.

## M3 Pro fused integration baseline

[`m3-pro-integration-96-v2-20260905.json`](results/m3-pro-integration-96-v2-20260905.json) is the repeatable kernel baseline from the forced-Release procedure in [`engine/native/README.md`](../engine/native/README.md). On the 36 GiB, 12-core M3 Pro it ran one warmup plus five measured 6252×4176×96-frame repetitions and required stable output plus complete sample accounting. Median wall time was 4.410 seconds, the measured range was 4.372–5.397 seconds, and median throughput was 568.29 megapixel-frames/s. The report records every sample, compiler, source commit/dirty state, tracked-diff SHA-256, untracked-tree SHA-256/count, and Metal-source SHA-256. It excludes FITS I/O, calibration, rejection-mask generation, registration, solving and publication, so it is not comparable to the 466.116-second full E2E. The candidate is an uncommitted dirty worktree; release evidence must be regenerated from the eventual clean first commit.

[`m3-pro-integration-96-20260901.json`](results/m3-pro-integration-96-20260901.json) is a historical exact 96-frame run of the 6252×4176, 64-row profile. It processed 2,506,401,792 pixel-frame samples in 4.246 s inside the benchmark boundary (2.189 s synthetic preparation, 2.043 s Metal-call wall time, 0.728 s reported GPU execution), for 590.27 megapixel-frames/s. It excludes the production FITS/QC/calibration/registration and CPU MAD-mask stages listed below and, because it predates the v2 provenance/repetition contract above, is feasibility evidence rather than a release benchmark.

[`m3-pro-integration-20260901.json`](results/m3-pro-integration-20260901.json) was produced by a historical 2026-09-01 source snapshot on an Apple M3 Pro with a 6252×4176 mono geometry, 40 synthetic frames, and 64-row tiles.

- 1,044,334,080 pixel-frame samples.
- 17.634 s measured loop wall time.
- 10.013 s synthetic CPU sample/mask preparation.
- 1.639 s Metal-call wall time and 0.812 s reported GPU execution.
- 59.22 megapixel-frames/s end-to-end within this synthetic loop.
- 42.32 s linear projection to 96 frames, below the committed 105 s kernel feasibility threshold.
- Metal recommended working set: 30,150,672,384 bytes.

The report excludes FITS/XISF I/O, source hashing, calibration, star detection, registration, LocalNormalization model generation, rejection-mask generation, Drizzle, astrometric solving, runtime shader compilation, preview rendering, and publication. It uses synthetic grids and masks, so it validates the fused kernel/tile profile rather than predicting a complete user run.

Reproduce it with the opt-in CMake benchmark command documented in [`engine/native/README.md`](../engine/native/README.md). The output path must not exist.
