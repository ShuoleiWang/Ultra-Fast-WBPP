# Advanced algorithms (opt-in)

Two algorithms that are **off by default** and that a recipe has to ask for by name: ZOGY proper coaddition, which adds one more product per filter, and a robust IRLS combination, which changes how the ordinary master's accepted samples are averaged. A recipe that mentions neither is the recipe of every earlier release, digest included, and produces byte-identical masters.

Neither is a replacement for the default path. The ordinary rejection/integration master stays the primary product of every run.

## Proper coaddition (Zackay & Ofek 2017)

```json
{
  "properCoaddition": {
    "enabled": false,
    "outlierHandling": "reuse-rejection",
    "apodizationPixels": 64
  }
}
```

| Recipe field | Values | Default | Meaning |
|---|---|---|---|
| `properCoaddition.enabled` | `true`/`false` | `false` | Adds `<FILTER>.proper.fits` beside each solved master (in a project run under `details/runs/<target>/products/<FILTER>/`; the desktop's result list shows only the project's masters) |
| `properCoaddition.outlierHandling` | `reuse-rejection`, `none` | `reuse-rejection` | Whether the ordinary integration's per-pixel rejection is reused before the transform |
| `properCoaddition.apodizationPixels` | `0`–`512` | `64` | Width of the mirrored guard band around the frame, in pixels |

`properCoaddition.enabled` together with `drizzle.enabled` is refused: the drizzled master lives on a finer grid, so the proper coadd would have no same-grid solved master to inherit a verified WCS from.

### What it computes

Every frame is modelled as `M_j = F_j · T ⊗ P_j + ε_j` with background noise `σ_j`. From the same registered, normalized frames the ordinary integration consumes, per filter group:

```
R̂   = Σ_j (F_j / σ_j²) conj(P̂_j) M̂_j  /  sqrt( Σ_j (F_j² / σ_j²) |P̂_j|² )
P̂_R = sqrt( Σ_j (F_j² / σ_j²) |P̂_j|² ) / F_R
F_R = sqrt( Σ_j F_j² / σ_j² )
```

`R` is the coadd; the matched-filter score image `S` is deliberately not published, because it is a detection statistic and not an image. As written, `R` has unit noise variance, so the published product is `R / F_R + sky`: the same photometric units as the normalized frames and the ordinary master, with the ideal stacked background noise `1 / F_R`. The receipt records the sky that was added back and every per-frame quantity that went into the sums.

`P̂_R` is the weighted quadratic mean of the frames' transfer functions, and Cauchy–Schwarz makes it at least as large as the ordinary weighted mean's linear average at every frequency. For frames with white noise the proper coadd therefore has the same background noise as an optimally weighted mean and a sharper effective PSF: the gain is entirely in point-source concentration, never in the background. Registered frames are not white — the resampling correlates neighbouring pixels — and the whitening division then raises the coadd's per-pixel noise (2.2–4.5 % on the reference project), which is why that project shows no net point-source gain.

**Flux scales.** After global normalization every frame carries the reference's photometric scale, so `F_j` is one and the same constant for all of them, and each frame's transparency has moved into its own `σ_j` — which is exactly where the weight `F_j / σ_j²` needs it. Substituting `M_j → a_j M_j`, `σ_j → a_j σ_j`, `F_j → F_ref` reproduces the raw-unit weights term by term, so this is the model's own weighting and not a simplification. Without global normalization the run has no measured transparency to use and equal flux scales are assumed; the receipt says which of the two applied (`fluxScaleSource`).

**Noise.** `σ_j` is the robust sigma of adjacent-column differences of the registered frame, which measures pixel noise rather than the scene. Resampling makes that estimate depend on a frame's sub-pixel shift (an unshifted frame shows more adjacent-pixel noise than a smoothed one); weighting by the ordinary integration's block-mean noise instead, which that phase does not bias, was measured on the reference project and moved the product by less than the measurements' own uncertainty in either direction, so the simpler estimate stays.

**PSF.** `P_j` is measured, never assumed. Local maxima above 30 σ are detected on the sky-subtracted frame; a star counts as isolated only when no other detection, the brightest included, lies inside its stamp box, and the brightest 5 % are then dropped as likely saturated; stamps that touch a pixel without data are skipped; up to 300 stars are extracted as 31×31 stamps, each is local-background subtracted, centroided, cubic-shifted onto the stamp centre and normalized to unit sum, and the stamps are combined with a three-pass 3 σ clip. Only when fewer than eight usable stars survive does a frame fall back to a circular Moffat (β = 2.5) of the group's median measured FWHM; the receipt records `psfSource`, `psfStars` and `psfFwhmPixels` for every frame.

**Outliers.** ZOGY assumes Gaussian noise, and a satellite trail or an unstable hot pixel is neither. With `reuse-rejection` (the default) the ordinary integration's per-pixel accepted-sample masks are reused: a sample the normal path rejected is replaced by that pixel's surviving robust mean — the ordinary master's own value, which is already on the frames' photometric scale — before the transform. The receipt counts the replaced samples per frame. `outlierHandling: "none"` skips this and is only there to make the difference measurable; it is not a recommended setting.

**Edges.** Registered frames have NaN borders and the FFT is periodic. The sky is subtracted first, so the invalid border is filled with the data's own mean (zero) and is not a step. The frame is then placed inside a guard band of `apodizationPixels` on every side, the band is filled by mirroring the frame outward and faded to zero with a half-cosine, and the whole grid is rounded up to the next fast FFT length (which also keeps a large prime factor out of the transform). Every published pixel therefore keeps unit weight — an earlier design that tapered the frame itself attenuated the outermost tens of columns of the product, because the run's common crop is only a few tens of pixels wide. What the real product shows: no ringing, corner background medians within 0.7 σ of the field background and no worse than the ordinary master's, the 8-pixel border median within 0.2 σ, and no NaNs.

**Memory and speed.** One frame at a time is transformed into two accumulators (a complex64 half-spectrum and its float32 squared modulus) with `scipy.fft` and the tuning row's worker count. Reading, cleaning and PSF measurement — most of a frame's cost, and a function of that frame alone — run up to three frames ahead on their own threads, as many as the memory budget holds; the transforms and the accumulation stay in frame order and a starless frame's fallback is resolved in that order, so the product is the same bit for bit however many frames were prepared ahead (`framesPreparedAhead` in the receipt). The stage estimates its own working set and refuses with `PROPER_COADD_MEMORY` rather than paging when even one frame at a time does not fit the tuning row's integration memory budget.

### The product

`products/<filter>/<filter>.proper.fits` beside the solved master (in a project run, `details/runs/<target>/products/<filter>/`), a linear Float32 image on exactly the ordinary master's grid and crop. Because the grid is identical, it carries the master's independently verified WCS rather than being solved again: `OAFWCS = 'INHERITED'` with `OAFWCSIN` naming the master and a HISTORY card saying so, and the promotion re-validates the header and checks that it maps the shared grid to within 1e-6 pixels of the master's solution. It is published create-only through the same atomic path as every other product and listed in `integration.properCoaddition` of the run receipt, with `primaryProduct: false`.

Header cards: `OAFPCOAD` (algorithm id), `OAFPCFR` (`F_R`), `OAFPCFWH` (the coadd PSF's FWHM in pixels), `OAFPCSKY` (the sky added back), `OAFPCAPO` (apodization), `OAFPCREP` (replaced samples), `OAFPCOUT` (outlier handling).

### How to judge it

The standard master evaluator compares two images that are supposed to share a PSF; the proper coadd deliberately does not. Read its noise and photometry families, and read its PSF family as a description, not a verdict. The number that matters for this product is point-source SNR at a fixed aperture and the background noise, measured on the same stars in both images — with the noise measured where it is not assumed white (the scatter of empty apertures and of single background pixels). Registered frames carry noise correlated by the resampling, the ordinary master inherits it, and the coadd partly whitens it, so a sigma from adjacent-pixel differences favours the coadd.

## Robust IRLS combination

```json
{ "integration": { "combination": "sigma-clip-v2" } }
```

| Recipe field | Values | Default | Meaning |
|---|---|---|---|
| `integration.combination` | `sigma-clip-v2`, `irls-huber` | `sigma-clip-v2` | How a pixel's accepted samples are combined |

`sigma-clip-v2` is the shipped behaviour: MAD rejection with the v2 scale model, then the inverse-variance weighted mean of the surviving samples. Setting it explicitly is the same as leaving the block out, digest included.

`irls-huber` keeps the rejection and the weights and replaces the final average with four iteratively reweighted least-squares passes. Each pass forms the standardized residual `r = (x − μ) / s` against the rejection scale model's own per-pixel sigma `s` — the same `s` the `sigma_clip` threshold is built from — multiplies each frame's weight by Huber's `u(r) = min(1, k/|r|)` with `k = 1.345` (95 % asymptotic efficiency at the normal) and recomputes the weighted mean. The starting point is the ordinary weighted mean, so the first residuals are measured against today's answer.

The iteration count is fixed and every reduction is a single `numpy` sum over the frame axis, so the result does not depend on the thread count, the tile height or the machine. Selecting `irls-huber` forces the portable CPU integration backend, since the Metal reduction has no IRLS path; the receipt records that as the `fallbackReason`. The reducer is the NumPy reference, and it recomputes the per-pixel scale that the native rejection kernel does not return, so the integration stage costs measurably more; the receipt reports it under `timingSeconds.combination`.

What it buys: a sample that survived the 4 σ clip but still sits in the tail is smoothly downweighted instead of counting in full. What it does not buy: anything the rejection already removed, and nothing at all when the samples are Gaussian — Huber's weight is 1 for every residual inside `k`.

## Measured behaviour

On the NGC 7331 reference project (62 admitted Lights, L/R/G/B, M3 Pro):

| | Proper coaddition | IRLS |
|---|---|---|
| Wall time | 77.9 s and 76.1 s against 65.5 s and 61.4 s for interleaved default runs (about +13 s) | 252.7 s against 65.9 s (×3.8) |
| Ordinary masters | byte-identical | replaced (that is the point) |
| Background noise | per-pixel noise 2.2–4.5 % higher than the ordinary master's | +0.9 % on L, flat elsewhere |
| Point sources | peak SNR −1.7 % to +0.7 %; aperture SNR within ±1.5 % at r = 2–4 px (empty-aperture noise) | — |
| Against the PixInsight master | indistinguishable from the default (G_8 inside its confidence interval in all four filters); zero FAIL | worse on noise gain and depth in three filters of four |

The proper coadd shows no measurable point-source gain on this project, whose frames differ little in FWHM; the model's gain grows with the spread of the frames' PSFs, which is what the option is there to measure. IRLS behaves exactly as designed on Gaussian samples (1.9 % efficiency cost) but moves the real master by 1.3–2.2 σ, because real per-pixel sample sets across registered frames carry legitimate spread — resampling on steep gradients, normalization residuals, per-frame PSF differences — that Huber downweights.

[The validation matrix](../validation-matrix.md) records the full numbers and what is and is not validated.

## Notes

Neither option is enabled by any built-in contract. The desktop offers proper coaddition as a checkbox under advanced options; IRLS is a recipe option only, because it measured worse than the default. Leave them off unless you are measuring the difference: the default path is the one with retained real-data acceptance.
