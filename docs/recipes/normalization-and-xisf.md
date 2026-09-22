# XISF pixels and stellar/background normalization

> **Pre-1.0 boundary:** one retained real mono MasterFlat validates the bounded XISF decode bridge. Global normalization passed one private 38-frame B run but still lacks retained multi-condition acceptance. The former optional LocalNormalization implementation has been retired. The supported normalization remains stellar scaling plus guarded spatial background correction; it does not claim PixInsight algorithmic equivalence.

## XISF execution boundary

Ultra-Fast WBPP inventories FITS and XISF headers, but inventory success is not pixel-execution success. Before calibration, each XISF input is independently content-hashed and decoded into a private Float32 FITS staging image on the output filesystem. The source is opened read-only, rechecked after conversion, and the private staging directory is removed after the run.

The production decoder accepts a unique two-dimensional mono/CFA science image stored as an attachment, inline or embedded block. A PixInsight container may also contain explicitly identified `RejectionMapLow`, `RejectionMapHigh`, `RejectionMap`, `WeightMap`, or `CoverageMap` auxiliary images; these maps are recorded and ignored. Multiple science images, unknown extra images, RGB/multichannel storage, complex samples, malformed geometry, attachment escapes, unsupported codecs, and any image or working set above the configured limits fail closed.

The v2 bridge preserves and audits XISF `sampleFormat` and `bounds`. Float calibration images must explicitly declare the normalized `0:1` interval; malformed, reversed, zero-width, missing, or non-unit Float bounds fail closed. Unsigned integer XISF samples use the `SENSOR_CODE` domain with the corresponding full-code normalized-unit scale. The private staged FITS header carries this conversion evidence, but only a staging file bound to the original XISF source identity is trusted; an external FITS file cannot authorize its own units by copying these private keywords.

Uncompressed attachments are streamed by rows. zlib, lz4, and zstd attachments are decoded only after decoded-size, compression-ratio, and peak-working-set checks. zlib uses an output-bounded stream decoder. The XML header is size-bounded before parsing and any `DOCTYPE` or `ENTITY` declaration is rejected. `COMMENT`, `HISTORY`, and `PixInsight:ProcessingHistory` are never copied to staging FITS or public validation evidence because they can contain private source paths.

The sanitized real-file validation record is [validation/xisf-real-masterflat-20260901.json](../../validation/xisf-real-masterflat-20260901.json). It contains geometry and numeric decode checks only—no source path, content hash, raw metadata, or processing history.

## Content-bound metadata override for old masters

Some reusable PixInsight masters lack camera gain, offset, sensor temperature, or readout mode even though their pixel data and basic camera/filter metadata are valid. Ultra-Fast WBPP does not guess these fields from a filename and does not relax compatibility globally. Add a complete declaration bound to the exact source SHA-256:

```json
{
  "calibration": {
    "masterMetadataOverrides": [
      {
        "sourceSha256": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "camera": "QHY268M",
        "gain": 100,
        "offset": 50,
        "binning": [1, 1],
        "filter": "R",
        "cfaPattern": "NONE",
        "readoutMode": "Mode 1",
        "temperatureCelsius": -10,
        "exposureSeconds": 1,
        "numericDomain": "NORMALIZED_UNIT",
        "normalizedUnitScale": 1
      }
    ]
  }
}
```

The digest must identify exactly one supplied MasterBias, MasterDark, or MasterFlat in the current request. `numericDomain` and `normalizedUnitScale` are required together only for an additive Float FITS MasterBias/MasterDark whose storage has no independently trustworthy unit declaration; use `NORMALIZED_UNIT`/`1` for normalized samples or `SENSOR_CODE` with the physical full-code value for code-domain samples. XISF `0:1` bounds and unsigned integer FITS storage endpoints are resolved automatically. The override is included in the recipe/request digest and receipt. A changed or duplicated source fails before compatibility matching.

## Conservative global normalization

Ordinary integration defaults to `reference = stellarScale * target + additiveOffset(x,y)`. The multiplicative scale comes only from robust, same-filter matched-star aperture flux ratios measured during registration; the raw aperture ratio is explicitly converted into the already exposure-normalized pixel domain, so mixed 30/60-second Lights are not corrected twice. Paired low/mid-intensity pixels in the common registered footprint are then used only for the additive model, excluding the brightest 30 percent independently in both frames and clipping compact residuals. Background covariance never determines multiplicative scale. The additive model uses 128-pixel nodes smoothed by seven nodes (896-pixel sigma), so it is deliberately limited to very low spatial frequencies; unlike the scalar stellar scale, it can change broad background structure and is guarded rather than described as morphology-invariant.

A stellar scale is applied only with enough unique same-filter matches, enough valid aperture measurements, robust outlier rejection, a finite bounded ratio, exact source/reference SHA-256 bindings, and the same reference chosen by the integration quality policy. If stellar scale is unavailable, the default policy keeps scale exactly 1. The additive grid requires at least 70 percent valid cells, checkerboard holdout validation with no MAD/span regression and at least 10 percent improvement, and sky-relative P05-P95, min-max, and first-difference limits. An already-flat, underconstrained, or excessive grid is rejected; normalization keeps the independently accepted stellar scale (or the configured unit-scale fallback) and the finite scalar offset already fitted from paired samples. The rejected spatial grid is never applied or clipped into range. For excessive grids, the receipt records the measured amplitude/span/neighbor ratios, reference sky, unchanged limits, and `GLOBAL_NORMALIZATION_OFFSET_GRID_SPAN_UNSAFE` as the scalar fallback reason. This may leave a background gradient for later processing; it does not certify gradient removal. Insufficient global samples, invalid scalar estimates, invalid scale identities, and forbidden scale fallbacks remain fatal. Every scale hint, exposure correction, source/reference identity, accepted grid value/digest, coefficient, fallback reason, sample count, quantile policy, holdout result, and residual metric is embedded in the registration and pixel-pipeline receipts. This is a new pre-1.0 path and needs retained real multi-condition oracle validation before any PixInsight-parity claim.

## Retired LocalNormalization option

The former `paired-background-grid-v1` implementation is no longer a product
option. It had only synthetic validation, replaced the better-tested default,
and could not carry its local multiplicative grid into native drizzle. Its
source remains reproducible at the pre-removal Git revision.

Omitting `localNormalization`, or supplying `{"enabled": false}`, keeps the
same default pixels. A recipe with `{"enabled": true}` fails explicitly with
`LOCAL_NORMALIZATION_REMOVED`; it is never silently executed as another algorithm.
Existing receipts remain readable. The disabled legacy recipe/receipt fields
and hardware metadata are retained for compatibility, not as active features.
