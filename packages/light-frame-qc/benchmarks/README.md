# `prepare-wbpp` apply I/O benchmark

This microbenchmark creates temporary FITS files, builds a real signed-by-digest prepare plan, applies it, and counts every full-content identity read plus every copy stream. It does not inspect user data, retain generated images, or print temporary paths.

Run from the Ultra-Fast WBPP repository root:

```bash
PYTHONPATH=packages/light-frame-qc/src \
  .venv/bin/python packages/light-frame-qc/benchmarks/prepare_apply_io.py
```

The fixed fixture has two accepted 2048 x 1024 Light frames in different targets, one manifest-only review frame, and one MasterFlat reused into both targets. There are four create-only published image files. The deterministic acceptance metric is content-read bytes; wall time is included only as a local smoke signal because filesystem cache and storage hardware dominate this small run.

Evidence captured on 2026-09-01 is in [`results/prepare-apply-io-m3-pro-20260901.json`](results/prepare-apply-io-m3-pro-20260901.json). The before trace was taken immediately before the streaming-copy change against the same fixture. It read every source once for preflight, read copied sources again through `copy2`, and hashed every target three times. The after trace hashes no source in a separate pass, reads only copied sources, hashes every private-staging target once, and uses the resulting digest/inode/stat snapshot after the atomic rename.

The measured content-read reduction is 60.0% (83,980,800 to 33,592,320 bytes). Per copied Light, apply source reads fall from two to one and target full hashes fall from three to one. The excluded frame falls from one full apply-time hash to a policy-explicit stat-only check. A reused MasterFlat still has one necessary copy read per target, while its source digest calculation is reused after the first identity-bound pass; every target remains independently created, fsynced, and hashed.
