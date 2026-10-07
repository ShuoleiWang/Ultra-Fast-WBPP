# Mosaics: several panels, nights and filters on one canvas

Use this recipe when a project covers more sky than one frame: panels named by N.I.N.A. (`OBJECT` values such as `M 31 Panel 3`), WBPP `PANEL_3/` folders, or Lights of one target that simply point at different parts of the sky. Panels may differ in filters, in the number of nights and in how each night was calibrated.

> **Pre-1.0 boundary:** canvas mosaics are validated on synthetic data only: a two-panel run through the real pipeline with optical distortion, and a six-panel set with a bright galaxy crossing an overlap. No real mosaic has been processed yet. The design and its open phases are in [mosaic-plan.md](../mosaic-plan.md) (Chinese).

```bash
ultra-fast-wbpp run-project \
  /data/M31_Panel_1 /data/M31_Panel_2 /data/M31_Panel_3 /data/calibration \
  --output /data/M31-mosaic --progress-json
```

## What a run does

1. **Panels.** Lights become panels by their `OBJECT` name, by a WBPP `PANEL` keyword in their path, or, when a target's Lights point at groups of the sky farther apart than 0.35 of a field (header `RA`/`DEC`, `FOCALLEN`, `XPIXSZ`), by those pointing groups (`M31 PANEL 1`, `M31 PANEL 2`, north first).
2. **Canvas.** One Light of every panel is plate-solved. The canvas is one gnomonic projection shared by every panel and filter:
   - its tangent point is the centroid of the panels' combined footprint;
   - its axes follow the median panel orientation, so the canvas is as small as the panels allow;
   - its pixel scale is the panels' scale; panels of different cameras take the finest scale, with a warning;
   - beyond 10° it is stereographic.

   Panels that do not overlap into one mosaic, or a canvas above 2^30 pixels, stop the run before any panel is processed.
3. **Panels on the canvas.** Each panel runs the ordinary pipeline: screening, per-night calibration, registration, normalization and rejection.
   - Its registration reference is solved and refined into TAN+SIP on catalog stars (at least 30 matches). Pure TAN solutions leave the optical distortion in the frame corners, which are exactly where panels overlap.
   - Every Light is then resampled once, from its calibrated pixels straight onto its panel's window of the canvas. The window is an integer offset of the canvas lattice, so no master is ever resampled again and every filter of every panel shares one pixel grid.
   - A panel master's WCS is the canvas window. It is verified against catalog stars, not solved blind.
4. **Matching.** For each filter, the panels are matched in their overlaps.
   - **Scale.** One scale per panel comes from the stars both panels measured. Each panel's aperture follows its own seeing (2.5 FWHM), each star's background is the median of its own annulus, and a robust flux-flux fit absorbs the additive part. All overlaps are solved at once instead of panel by panel.
   - **Background.** One plane per panel comes from the binned overlap differences, keeping the area-weighted consensus sky. Sky shared by the panels, such as a halo or IFN, is never removed. Bins where the scale's uncertainty times the local brightness could reach a tenth of the noise are left out; on a bright galaxy they give an independent scale check instead.
5. **Blending.** Every pixel is the inverse-variance weighted mean of the panels covering it. A panel fades in over at most 256 px (0.4 of its narrowest overlap) instead of starting with a step. The blend writes the mosaic and its NOISE, COVERAGE and MASK planes.
6. **Gates.** A FAIL stops publication; WARN is recorded.

   | Gate | PASS | WARN |
   |---|---|---|
   | Corrected overlap-star flux ratio, per overlap | within 0.3 % | within 1 % |
   | Overlap-star offset RMS between panels | at most 0.10 px | at most 0.25 px |
   | Residual background difference | at most 0.1 bin σ | at most 0.3 bin σ |
   | Flux conservation of overlap stars in the mosaic | 0.5 % | 2 % |
   | Canvas WCS, verified on catalog stars in every panel's window | required | — |

7. **Colour.** The filters' mosaics share the canvas, so RGB/LRGB composition never resamples a channel. Channels without coverage are NaN there.

## The published directory

```
M31-mosaic/
  L.fits  R.fits  G.fits  B.fits   # one mosaic per filter, on the shared canvas
  LRGB.fits                        # linear colour cube (RGB.fits without L)
  previews/
  receipt.json
  details/                         # shared-calibration/, runs/<panel>/, mosaics/<filter>/, color/
```

`details/mosaics/<filter>/receipt.json` records every panel's scale and plane, every overlap's star count, ratio and astrometric offset, the background residuals, the extended-structure scale checks, flux conservation, the blend statistics and each window's catalog verification. `details/runs/<panel>/receipts/canvas.json` records the panel's reference solution, its distortion fit (order, matched stars, residuals at the edges and in the middle) and its canvas window.

## Not available yet

- Drizzle onto the canvas.
- Seam routing around bright stars and cores, and a two-scale blend.
- Short and long exposure classes merged per pixel (HDR) and saturation maps.
- A bundle adjustment of all panels' distortion.
- Selection region weight maps and proper coaddition on canvas windows; these combinations are refused.

A Bayer (one-shot-colour) Light set is one target's R, G and B panels: the pixel pipeline debayers each Light into the three channel groups (see [OSC / CFA](osc-cfa.md)); Bayer-as-mono processing never happens, and a Bayer set cannot share a target with mono R/G/B Lights.
