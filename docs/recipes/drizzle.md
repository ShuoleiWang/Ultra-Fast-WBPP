# Drizzle recipe

Drizzle reconstructs an output grid from calibrated input samples, subpixel registration mappings, weights, and rejection masks. It is not merely a larger resize and does not replace calibration or registration.

> **Pre-1.0 boundary:** the executor and scientific gates have synthetic/adapter coverage, but there is no retained real-data Drizzle acceptance yet. Use Balanced ordinary integration for unattended science until the release matrix closes that gate.

## Recommended use

Use 2× Drizzle for genuinely undersampled mono data with enough subpixel dither positions and output coverage. Start with a square kernel and a pixfrac near `1 / output_scale`, then inspect coverage and correlated-noise evidence. Well-sampled data normally stays on the Balanced recipe because 2× increases compute, storage, and noise without recovering real detail.

OSC/CFA input is fail-closed in v1. A future CFA-preserving recipe must retain Bayer phase through calibration, mapping, rejection, and color reconstruction; Bayer-as-mono processing or Debayer-then-upscale is never labeled CFA Drizzle.

## Mandatory evidence

A successful drizzle result includes output scale, kernel, pixfrac, per-frame mapping identity, input/variance weights, rejection mask identity, science image, weight image, context or coverage image, null-pixel fraction, and coverage percentiles. Coverage below the recipe threshold fails or offers an explicit ordinary-integration re-plan; it never silently produces a sparse final master.

The production gate currently requires all of the following:

- at least three registration-derived subpixel phases, separated by at least 0.15 native pixel;
- phase span of at least 0.35 native pixel on both detector axes;
- at least 90% non-null output coverage and at most 10% null pixels;
- measured QC sampling for 2×/3× output. Unknown or conflicting QC-FWHM/N.I.N.A.-HFR evidence is `REVIEW`, not presumed undersampled, while a consensus median native stellar FWHM of 3.0 pixels or more blocks upsampling as not recommended.

Before drizzle, the E2E runner independently registers each calibrated frame into bounded native-resolution tiles. It computes a cross-frame median/MAD model, applies a six-sigma decision with measured-noise, signal, and local-gradient floors, and projects each frame's decisions back into that frame's detector coordinates. Every resulting Rice-compressed FITS rejection mask is retained, hashed, counted, and passed explicitly as a `DrizzleFrameInput`. A run with no mask is recorded as having no rejection-mask evidence; generic invalid or out-of-bounds pixels are not mislabeled as statistically rejected samples.

The executor writes no success artifact until the coverage gate passes. The E2E promotion boundary then independently re-hashes the FITS output, recomputes coverage from `WHT` and `COVERAGE`, verifies the canonical receipt, and checks every retained rejection-mask identity and count. Changed coverage or receipt data blocks final publication.

The first portable numerical oracle is the BSD-licensed STScI `drizzle` implementation. The Metal backend must pass differential science/weight/context tests against that CPU path before the GUI enables it. Final drizzle geometry is plate-solved after reconstruction.

Ultra-Fast WBPP does not consume PixInsight `.xdrz` files. Dense or projective registration maps use Ultra-Fast WBPP's documented output-to-input convention, and the final reconstructed master is solved again instead of inheriting an unverified seed WCS.
