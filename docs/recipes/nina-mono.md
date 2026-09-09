# N.I.N.A. mono preprocessing

This guide's retained real-data evidence is one 38-frame B-channel mono run. R/G/B/L are processed as independent mono groups. Synthetic shared-calibration, seam-correction, and post-reprojection WCS-provenance regressions pass; retained real multi-panel/color acceptance remains pending.

## Inputs

Drag the acquisition root into the Import page. Ultra-Fast WBPP reads FITS/XISF metadata and groups by target/field, camera, geometry, binning, gain, offset, readout mode, exposure, filter, temperature/session, and CFA state. Different filters are never compared by raw brightness.

For observing-night QC, N.I.N.A. `DATE-LOC` is preferred over its UTC `DATE-OBS`; a timezone-less `DATE-LOC` is treated as the observatory's local wall clock, so one evening is not split at a UTC-derived boundary. If a non-N.I.N.A. dataset has only UTC timestamps, set `observing_timezone` in the Quality Gate configuration to an IANA zone such as `Asia/Shanghai` or a fixed offset such as `+08:00`. The selected value is included in the policy digest, so changing it invalidates earlier REVIEW approvals.

Some mono-camera N.I.N.A. files omit `BAYERPAT`. The desktop explicitly selects the standard monochrome workflow, so those files need no per-file declaration. An explicit Bayer pattern still blocks this mono-only pipeline. The selected convention is recorded in the recipe and receipts rather than written into original files.

The optional `strict-v1` CLI workflow retains SHA-bound per-source mono declarations for existing strict recipes. They are no longer the normal desktop interaction.

The recommended minimum is eight Light frames per target/filter group and three same-night peers for unattended Quality Gate approval. Smaller groups remain reviewable but cannot silently pass.

Ordinary registration refines every non-reference transform on full-resolution stellar centroids and then uses normalized, domain-bounded 6×6 Lanczos-3 resampling. A sample is published only when its complete kernel support is present and finite; the identity reference is copied exactly without interpolation. The receipt records the chosen resampler and support-aware crop. This is interpolation, not post-processing sharpening.

A uniform whole-night zeropoint shift below 0.50 mag is not a defect by itself when registration, spatial transparency, source retention, background, noise, focus, trailing, and occlusion evidence are otherwise clean. Such a shift can arise from season, target angle, or throughput and is retained for the downstream global normalization and quality weighting. A shift at or above 0.50 mag still requires review; independent cloud evidence can still produce a hard failure regardless of this night-level threshold.

## Calibration matching

- Bias matches camera, geometry, binning, gain, offset, and readout mode.
- Dark additionally matches exposure within the recipe tolerance and temperature within the configured window.
- Flat additionally matches filter and CFA state. A master Flat is never selected by filename alone.
- Ambiguous compatible masters fail planning instead of choosing the newest file.

## Output

Each target/filter produces a calibrated linear master FITS, coverage and rejection evidence, an auto-stretched review preview, and a required verified celestial WCS. L/R/G/B channel combination is a separate output role and never destroys the individual masters.

For ordinary integration, `coverage/coverage.json` points to three per-filter FITS maps produced by the same rejection decisions used for the master: accepted sample count, accepted/total coverage fraction, and rejection count. Registration quality weights are combined with noise weights before the pixel reduction; the receipt records all three vectors instead of merely displaying an unused score.

Before ordinary rejection/integration, the default global normalization measures an audited multiplicative scale from robust same-filter matched-star aperture ratios and explicitly removes the source/reference exposure ratio already handled by calibration. With that scale fixed, it fits only an additive sky surface from paired low/mid-intensity pixels on the common registered grid. Bright stars and bright target cores are excluded; background covariance never sets stellar scale. The additive surface uses 128-pixel nodes with a seven-node Gaussian smoothing sigma (about 896 pixels), checkerboard holdout validation, and sky-relative amplitude/roughness gates. An already-flat or underconstrained field falls back to one scalar offset; an excessive correction fails closed. The private 38-frame B run passed stellar-scale, photometry, and background checks; multi-condition retained acceptance is still pending, so this must not be described as PixInsight ImageIntegration equivalence.

## Explicit REVIEW approval

The native GUI runs the real read-only Quality Gate before recipe execution and lists every Light's evidence. For exactly one target × filter panel, an operator can approve a `REVIEW` row in place. The GUI sends the preflight source SHA-256 and policy digest; the trusted runtime recomputes the complete request digest and repeats QC, so changed bytes/policy/request context fail closed and the source must still be uniquely classified `REVIEW`. Multi-panel GUI runs cannot safely replay that context across panel subruns, so their REVIEW rows stay explicitly excluded. `HARD_FAIL` is never promotable.

The headless two-run workflow remains available for one exact E2E source set. Run once to a new output directory and inspect its thumbnail/evidence. Read that frame's `sha256` value from the output receipt's `sources` array, place it in `sourceSha256`, and copy `gatePolicyDigest` plus `manualReviewApprovals.requestDigest` from `qc/manifest.json` into a new recipe:

```json
{
  "reviewApprovals": [
    {
      "sourceSha256": "sha256:<64 lowercase hex>",
      "gatePolicyDigest": "sha256:<64 lowercase hex>",
      "requestDigest": "sha256:<64 lowercase hex>"
    }
  ]
}
```

Rerun to another new output directory. The approval is accepted only when the SHA-256 identifies exactly one current REVIEW Light and the entire source/calibration/request context is unchanged. PASS and HARD_FAIL frames cannot be admitted with a REVIEW approval. Changing a source file, calibration selection, QC policy, integration/registration/drizzle settings, or solver quality policy produces a drift error before publication. A `run-project` request with explicit approvals is limited to one target × filter panel; split larger projects or let REVIEW remain excluded.
