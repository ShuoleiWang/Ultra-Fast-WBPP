# Blink display redesign: complementary diagnostic views

Status: implemented locally in the desktop as `blink-complementary-display-v2`. Pixel functions live in `blink_diagnostics.py`, session rendering in `blink_diagnostic_render.py`, and original-pixel regions in `blink_native_crops.py`. The desktop requests v2 explicitly through `previews.displayAlgorithm`; the headless default remains `shared-stretch-v1`, preserving the previous display. This changes review previews only, not selection decisions or the calibration/registration/normalization/integration of science products. The read-only comparison tool uses the same pixel and crop implementations.

## Goal and observed failure

The observer needs to distinguish lost information (faint stars, usable signal, star shape) from potentially tolerable background differences. One normalized image cannot make both independent effects obvious: matching stellar flux can hide transmission loss, while removing a frame's background can hide clouds and gradients. Therefore the default diagnostic composition must show both star detail and background differences, with a full-field check and original-pixel crops available at the same time.

The initial inspection used the existing 97-Light, six-night, four-filter reference campaign and its content-bound Blink manifest. The old session supplied Master Flats without a Dark/Bias pedestal. Its RGB reference stretches used a global MAD of approximately 40 ADU; after an exposure-matched Master Dark plus Flat, the same references have global MAD around 4–5 ADU. This difference is a calibration effect and must not be credited to the new display curve. A severely clouded L frame had a stellar gain of 12.82 and approximately 28% near-white pixels in the old session; re-rendering the old algorithm with full calibration still leaves approximately 38% near-white. Normalizing a dimmed frame upward does not restore the missing information.

The study has separate columns for (1) the historical session, (2) the previous display algorithm with the same complete calibration and reference used by the candidate, and (3) the candidate. Thus calibration improvements are not confused with display improvements. Images, reports and private paths stay outside Git. The recorded measurements below are observations from one campaign, not thresholds fitted to its historical decisions or a classifier acceptance score.

## Inputs and reference selection

1. Work on calibrated linear previews. The controlled study requires an explicitly selected matching master flat and exposure-matched master dark, verifies Light SHA-256 against the original measurement manifest, and verifies all source identities remain unchanged. When complete supplied masters are unavailable, the desktop shows that preview calibration is incomplete; raw calibration frames are still combined by the processing pipeline, not silently treated as already-built masters. a flat-only preview must not claim complete calibration.
2. Keep channels separate by the existing target/filter/acquisition geometry/exposure identity. Registration must already be verified on the same preview grid. Missing registration/photometry yields an explicit unavailable comparison, never a neutral-looking “good” frame.
3. Choose a display reference only from the existing measured, registrable candidate set. Rank with `Q = (T / (sigma_local * FWHM_native))² * (1-ellipticity) * min(1, sourceRatio)`, using calibrated local noise rather than global sky variation. Resolve ties by stable frame index. Record rule `calibrated-local-noise-psf-v2`. On this campaign the four winners remain the existing references; no benefit from reference replacement is claimed.
4. Freeze reference, curve and scales for the entire channel. A production manual reference change must rebuild its diagnostics and invalidate that channel's confirmation. Reference candidacy remains a measured recommendation; it cannot certify a defect-free frame.

## Noise estimator

Estimate noise before registration interpolation. On disjoint 2×2 blocks, use `h = (p00-p01-p10+p11)/2` and `sigma_local = 1.4826 * median(abs(h-median(h)))`. For independent equal-variance pixel noise this has the original variance, while a constant or planar background cancels. The actual camera, resampling and spatial correlations mean this is a display scale estimate, not a calibrated uncertainty or a statistical significance. Constant/quantized or insufficient finite data must report unavailable instead of inventing a zero-noise reference.

A synthetic test adds a strong sky plane and requires the estimate to remain unchanged. This directly checks the failure of using global MAD as if it were only pixel noise. No frame is divided by its own noise for display: otherwise noisy and clean exposures would look similarly smooth.

## Simultaneous views

### Star detail: preserve attenuation and morphology

Build a coarse background surface `B_i(x,y)` from robust medians in 48×48 preview cells, with bilinear interpolation between supported cells. Display `I_i-B_i` using the reference noise scale and a single monotone curve for the channel. Do **not** multiply by the stellar flux-matching gain, normalize the frame's variance, denoise, sharpen or PSF-match it. Cloud-dimmed stars stay dim, and additional noise stays visible.

The curve maps zero to 22% display level. For `z = (I_i-B_i)/sigma_ref`, positive values use `u = asinh(z/2)` and `0.22 + 0.78*u/(u+1.5)`; negative values use `0.22*exp(z/3)`. It has no finite hard white clipping point. The constants are fixed display choices, not trained defect thresholds. Highlight compression still exists; a low white-pixel fraction is a clipping check, not proof of better classification.

The background split also removes some extended astronomical structure. This view is explicitly a star-detail aid and must never be presented alone as a faithful full-field image. The adjacent background comparison and full-field mode are part of the contract.

### Background differences: keep spatial defects visible

With measured stellar gain `g_i = T_ref/T_i`, compute robust cell medians of the registered difference `g_i*I_i-I_ref`, then subtract **one scalar median only**. Do not subtract a fitted plane, polynomial, local normalization map or the per-frame spatial residual. Fixed astronomical structure cancels to the extent allowed by alignment, photometry and PSF differences; differential gradients and cloud structure remain.

Use the same signed display scale for all frames: grey near zero, orange positive, blue negative, with amplitude `2/pi * atan(abs(residual/sigma_ref)/2)`. Cells with less than half finite coverage stay unknown/dark. This is in reference preview-noise units, not “sigma significance” of a cell. The map is not an automatic quality decision: residual stars, reference defects, extended structure, optical changes and registration error can contribute.

Global attenuation is deliberately compensated only in this background panel. Its loss of information remains visible in the star-detail view and is stated numerically. Never silently substitute `g_i=1` for missing photometry in a comparative map.

### Full field and original-pixel star crops

The full-field mode removes only the frame's scalar median and uses the same reference curve. It preserves the complete spatial background for checking whether the star-detail view has hidden relevant morphology.

A nine-region atlas selects moderately bright reference peaks in nine field regions and reads corresponding 80×80 original-pixel patches. It avoids using only saturated stars to judge morphology. Coordinates invert the measured preview transform and then use the reader's actual preview block size: `(p+0.5)*blockSize-0.5`. No interpolation, denoising or sharpening is applied to the crop; a 180-degree flip uses a pixel-preserving rotation. Two crop modes are explicit: a shared-curve signal view preserves dimming, while a morphology view scales each star by its central peak estimate and uses the same square-root curve. The latter compares shape only and cannot represent retained signal. A crop with a peak below ten times its locally estimated noise stays unavailable, not normalized noise masquerading as a star. This threshold is a prototype display choice, not a rejection rule. The study supports mono FITS crops only. Native XISF/CFA crops need a separate decoder contract before production use. Residual small rotations are intentionally not interpolated away and should be apparent to the observer.

## Human decisions

Keep three distinct questions visible:

| Question | Evidence | Interpretation |
|---|---|---|
| Can the current pipeline use this frame? | Decode, verified registration, coverage | Report the specific technical limitation; “this pipeline cannot register it” is not proof no method could recover it. |
| How much information was lost? | Relative stellar signal, local noise ratio, noise after matching stellar signal, original-pixel star shapes | Lower signal or poorer noise is not by itself a mandatory rejection; an observer can retain a weak but structurally sound exposure. |
| Is the field spatially damaged? | Background difference, full-field check, nine-region morphology | Patchy losses, gradients, trailing and localized obstruction need visual judgment. Color intensity alone is not a rejection threshold. |

Display `relativeSignal = 1/g`, `relativeNoise = sigma_i/sigma_ref` and `matchedSignalNoise = g*sigma_i/sigma_ref` separately. Do not turn them into a single opaque “badness” score. The last number is a relative noise diagnostic, not a predicted integration weight or independent-exposure SNR guarantee. Sky changes, airmass and legitimate observing conditions can change these values too.

Mandatory channel review and explicit confirmation remain in place. A v2 frame is counted as viewed only after both its main field and available background comparison have painted. Decode failures support retry or explicit drop; unavailable comparison data are labelled. Algorithm hints do not pre-drop frames. Historical human decisions in the study are hidden by default, and are used only for after-the-fact inspection, never reference selection or curve fitting. A future “tentative keep” label should remain an explicit human annotation attached to KEEP, rather than silently changing admission policy.

## Local evidence and remaining acceptance

The study rendered all 97 existing frames, verified their content digests and left the inputs unchanged. All four display references stayed unchanged when ranked with local noise. Representative observations:

| Frame within the study (zero-based index) | Historical human choice | Relative stellar signal | Matched-signal noise | Background P95-P5 / reference noise |
|---|---|---:|---:|---:|
| L 10, thin-cloud case | DROP | 0.470 | 2.39 | 5.69 |
| L 18, moonlit night | DROP | 0.841 | 2.27 | 11.38 |
| L 40, severe loss | DROP | 0.078 | 25.14 | 181.11 |
| R 92, weaker but retained | KEEP | 0.517 | 2.12 | 0.93 |

These values explain why equalized brightness alone is inadequate: L 10 and R 92 have similar stellar attenuation, but their background differences and the observer's choices differ. The prototype exposes those dimensions; it does not infer that the historical labels are universally correct or that it has achieved higher human accuracy.

Synthetic contracts cover cloud attenuation/noise remaining visible, a gradient surviving the background comparison, fixed nebulosity cancelling under an exact photometric model, masks staying unknown, missing alignment/photometry disabling the map, no finite white clipping, candidate reference selection, original-pixel crop coordinates, and morphology visibility without inventing stars from noise. Existing Blink preview tests remain unchanged.

For broader acceptance, perform a blind A/B review of held-out exposures with the observer, including faint nebulosity, different sensors, OSC, bad/missing flats, saturation, moderate registration errors and frames from a whole uniformly poor channel. Record missed defects, false rejections and review time. Current evidence is one mono campaign plus synthetic contracts; it does not establish automatic rejection accuracy, improved final masters or general WBPP equivalence. This is a local desktop build; broader scientific and observer-accuracy claims remain unvalidated.

## Sources and boundaries

[Siril's display/stretch documentation](https://siril.readthedocs.io/en/stable/processing/stretching.html) distinguishes screen display from modifying linear pixels; this prototype uses that same separation. [Siril's background-extraction documentation](https://siril.readthedocs.io/en/stable/processing/background.html) describes how a fitted background removes spatial trends. Here that operation is restricted to the labelled detail panel, while the background difference retains those trends. The complementary views, transfer curve and reference ranking above are this project's experimental design, not claims endorsed by those sources.
