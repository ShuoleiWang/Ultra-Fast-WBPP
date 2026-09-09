# NINA multi-panel mosaic to solved RGB/LRGB

Use this recipe when NINA `OBJECT` values identify panels such as `dunpai1`, `dunpai2`, `dunpai3`, and `dunpai4`.

> **Pre-1.0 boundary:** this is a development recipe with synthetic four-panel × RGB coverage only. Synthetic shared-calibration, seam-correction, and post-reprojection WCS-provenance regressions pass; retained real multi-panel/color acceptance remains pending. Do not treat the current module as an unattended scientific support claim.

```bash
ultra-fast-wbpp run-project \
  /data/dunpai1 /data/dunpai2 /data/dunpai3 /data/dunpai4 /data/calibration \
  --output /data/Ultra-Fast-WBPP-Shield-final --progress-json
```

GUI and scripted callers should prefer `run-project --request-json request.json`. The v1 object contains `sources[]` (`sourceId`, `hostPath`, `expectedRole`, `recursive`), `outputDirectory`, the complete Recipe v1 object, optional `solverHints`, and `execution.workers`. Header/path role evidence must agree with every `expectedRole`.

The required transaction contract is:

1. Hash every source and classify READY Lights by scientific `target × filter`.
2. Integrate raw Bias/Dark/Flat groups once into a shared calibration library, or reuse supplied masters. Every supplied MasterDark requires a SHA-bound `biasIncluded` boolean in `masterMetadataOverrides`.
3. Run QC, shared-master calibration, registration, integration/Drizzle, and a managed-catalog solve independently for every panel.
4. For each multi-panel filter, reproject only SOLVED panels, fit a bounded gain/offset relation on exact pairwise overlaps, propagate a deterministic correction from reference panel 0 across the connected overlap graph, and apply those corrections before coaddition. Corrected overlaps are checked again without fitting away residual offsets; a disconnected, extreme-gain, or seam-failing graph stops the run. Backend `match_background` is disabled to prevent an unaudited second normalization. The working mosaic remains `NEEDS_FINAL_SOLVE` with `OAFWCS=PROPAGATED`.
5. Plate-solve every working mosaic again. Propagated WCS never satisfies the final gate.
6. Align independently solved filters to one solved reference grid. R/G/B creates linear RGB FITS and color-preserving 16-bit TIFF/PNG. L, when present, participates in LRGB. Missing channels publish clearly labelled solved mono only.
7. Re-hash the original sources and publish the complete output with one create-only atomic rename. Failure publishes only `<output>.unsolved`.

The outer receipt includes a share-safe panel matrix, hashes of shared-calibration and child receipts, mosaic/final-solve/alignment evidence, and `finalProducts.guiArtifacts[]` with relative path, SHA-256, size, final gate, and managed astrometric quality. Published JSON contains no absolute host path or inode/device/mtime values.

v1 is mono-camera only. A Light whose CFA pattern is not `NONE` fails before pixel work; Bayer-as-mono processing is never called CFA Drizzle.
