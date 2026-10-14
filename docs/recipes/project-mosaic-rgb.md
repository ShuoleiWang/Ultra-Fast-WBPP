# Mosaics: several panels, nights and filters on one canvas

Use this recipe when a project covers more sky than one frame: panels named by N.I.N.A. (`OBJECT` values such as `M 31 Panel 3`), WBPP `PANEL_3/` folders, or Lights of one target that simply point at different parts of the sky. Panels may differ in filters, in the number of nights and in how each night was calibrated.

> **Pre-1.0 boundary:** canvas mosaics are validated on two real M31 projects and on synthetic data.
> - **Real:** a 2×2 M31 mosaic (QHY268M on a 12-inch f/4 Newtonian, mono L/R/G/B, 286 Lights over seven nights, 276 admitted by default) publishes with the Tycho-2 set alone (B PASS, L/R/G WARN on their diagonal corner overlaps, canvas WCS 0.54–1.59″, overlaps within 0.022 FWHM) and with the Gaia tiles (Astrometry.net 5205; overlaps within 0.021 FWHM). A five-panel M31 mosaic (479 Lights; one panel on four cloudy nights and both sides of the pier) publishes with the Gaia tiles, every overlap within 0.039 FWHM.
> - **Synthetic only:** one-shot-colour (Bayer) mosaics and photometric components, run through the real pipeline or the assembly.
>
> The design and its open phases are in [mosaic-plan.md](../mosaic-plan.md) (Chinese).

```bash
ultra-fast-wbpp run-project \
  /data/M31_Panel_1 /data/M31_Panel_2 /data/M31_Panel_3 /data/calibration \
  --output /data/M31-mosaic --progress-json
```

## What a run does

1. **Panels.** Lights become panels by their `OBJECT` name, by a WBPP `PANEL` keyword in their path, or, when a target's Lights point at groups of the sky farther apart than 0.35 of a field (header `RA`/`DEC`, `FOCALLEN`, `XPIXSZ`), by those pointing groups (`M31 PANEL 1`, `M31 PANEL 2`, north first).
2. **Canvas.** One Light of every panel is plate-solved. The canvas is one gnomonic projection shared by every panel and filter:
   - its tangent point is the centroid of the panels' combined footprint;
   - its axes follow the median panel orientation modulo a half turn, so the canvas is as small as the panels allow and panels shot after a meridian flip (180° from the others) fill their windows like the rest;
   - its pixel scale is the panels' scale; panels of different cameras take the finest scale, with a warning;
   - beyond 10° it is stereographic.

   Whether two panels overlap is decided on their footprint polygons on the canvas. Panels that do not overlap into one mosaic, or a canvas above 2^30 pixels, stop the run before any panel is processed.
3. **Panels on the canvas.** Each panel runs the ordinary pipeline: screening, per-night calibration, registration, normalization and rejection.
   - Its registration reference is solved and its stars are measured against the catalog (at least 30 matches). The panel's run then waits until every panel is measured.
   - The TAN+SIP solutions of all panels are fitted at once: the catalog stars hold each panel on the sky, and the stars two panels share in their overlap hold them to each other. The distortion is one polynomial for all panels or one per panel, as the BIC prefers. Pure TAN solutions leave the optical distortion in the frame corners, which are exactly where panels overlap; a panel fitted alone on a sparse catalog leaves the overlaps pixels apart.
   - Each Light is resampled through the window's lattice shifted by its own residual distortion, fitted on the stars of its registration. The filter's lateral colour, a meridian flip and another night's optics each leave a smooth shift that the Light's registration matrix cannot absorb; a panel that stacked them unshifted would put a star where none of its Lights does. A Light whose stars do not cover the frame (a passing cloud) keeps only the part of its field those stars determine.
   - A Gaia-density catalog tightens every panel's fit: the Gaia DR2 tiles `astrometry-net-5205-lite`, installed from the command line beside the Tycho-2 set (see [offline solver catalogs](offline-solver-catalogs.md); the desktop app cannot install them yet). With them each M31 panel fits on about 180 stars at 0.11–0.13″. The Tycho-2 set holds about 46 stars per square degree, about 40 in each 1.1° M31 panel: enough for the joint fit, not for a longer focal length. A mosaic whose catalog holds fewer than 30 stars in a panel's frame is refused while the canvas is planned, before any panel runs (`MOSAIC_CATALOG_TOO_SPARSE`, naming the install command with its terms acceptance; the survey's counts per panel are in the failure receipt), and so is a mosaic without a managed catalog (`MOSAIC_CATALOG_UNAVAILABLE`).
   - Every Light is then resampled once, from its calibrated pixels straight onto its panel's window of the canvas. The window is an integer offset of the canvas lattice, so no master is ever resampled again and every filter of every panel shares one pixel grid.
   - A panel master's WCS is the canvas window. It is verified against catalog stars, not solved blind.
4. **Matching.** For each filter, the panels are matched in their overlaps.
   - **Gain.** Each panel's gain comes from the stars both panels measured. Each panel's aperture follows its own seeing (2.5 FWHM), each star's background is the median of its own annulus, and a robust flux-flux fit absorbs the additive part. The fit is made in up to six bins along each overlap, and all overlaps are solved at once instead of panel by panel.
   - **Components.** Overlaps with at least 8 common stars and a positive flux-flux slope link the panels into a graph. When it falls into several connected components, each component is matched, blended and gated on its own with its own gauge: the largest is the filter's mosaic on the whole canvas box, the others are written as components on their own bounding box (an integer window of the same canvas, see the published directory), and the filter is WARN (`MOSAIC_PHOTOMETRIC_COMPONENTS`). A FAIL in any component fails the filter, after every component was gated. A bright galaxy across a starless overlap does not link it: a regression of its bins panel on panel is biased by any difference in sky gradient along the galaxy's flank.
   - The gains of a component are one scale per panel, or a scale times a linear gradient per panel when that lowers the BIC of the whole component (`gainModel`; each panel's gradient is `gainGradient`, per half of the component's box). The scales are relative to the reference panel, the one with the most flux per second, whose gain is 1 at the component's centre. A master inherits such a gradient from frames whose transparency changed across the field: thin cloud, or differential extinction at high airmass. On the M31 test mosaic the masters carried gradients of up to 1.3 % per half panel. One scale per panel left the diagonal corner overlaps 1.1–2.0 % off in B, G and L; the gradients bring them within 0.7 %. A gradient common to all panels cannot be seen in the overlaps, so the gradients keep the panels' consensus.
   - **Background.** One quadratic per panel (six terms) comes from the binned overlap differences, keeping the area-weighted consensus sky. Sky shared by the panels, such as a halo or IFN, is never removed. A strip along a panel's edge barely measures the panel's curvature across it, so the curvature terms carry a prior of a tenth of the pixel noise across the panel: the overlaps decide where they see the curvature, and curvature they cannot see stays at the consensus. Bins where the scale's uncertainty times the local brightness could reach a tenth of the noise are left out; on a bright galaxy they give an independent scale check instead.
5. **Blending.** Every pixel is the inverse-variance weighted mean of the panels covering it. A panel fades in over at most 256 px (0.4 of its narrowest overlap) instead of starting with a step.
   - Bright stars and galaxy cores in an overlap are the exception. Panels of different seeing hold different profiles of them, and their mean would match neither. Such a blob takes its small scales from the panel with the larger weight at its peak, while its large scales stay averaged.
   - A blob grows from every peak above 50 σ over the region where the panels differ, and it fades into the blend over 8 px. MASK bit 4 marks it.
   - The blend writes the mosaic and its NOISE, COVERAGE and MASK planes.
6. **Gates.** Every filter is assembled and gated; a FAIL in any of them stops publication, and the receipt holds the gates of all. WARN is recorded.

   | Gate | PASS | WARN |
   |---|---|---|
   | Corrected overlap-star flux ratio, per overlap | within 0.3 %, and within 1 % at 2 standard errors | otherwise, unless beyond 1 % by more than 2 standard errors (FAIL) |
   | Systematic overlap-star offset between panels, in the overlap's FWHM | at most 0.05 | at most 0.10 |
   | Residual background difference, in the blended mosaic's pixel noise | at most 0.1 σ | at most 0.3 σ |
   | Flux conservation of overlap stars in the mosaic | 0.5 % | 2 % |
   | Photometric network, χ² per degree of freedom | at most 9 | above 9 |
   | Overlaps without a scale (too few common stars) | none | any |
   | Canvas WCS, verified on catalog stars in every panel's window | required | — |
   | Photometric components | one | several (each gated on its own) |

   A corner overlap holds a few dozen stars and measures its ratio to about 0.7 %, so its point estimate alone cannot separate a 1 % mismatch from noise: it fails only when it lies beyond 1 % by more than two standard errors, and it passes only when it is known to be within 1 % at two standard errors. A gate with nothing to measure (a component of one panel has no overlap) is NOT_MEASURED, never PASS, and does not count toward the verdict.

   The residual background is the mean difference left in an overlap's background bins, against the per-pixel noise of the inverse-variance blend of the two panels. Since the blend fades panels in over up to 256 px, such a difference becomes a smooth ramp. The receipt also records it in bin noise and as a fraction of the sky.

   The systematic offset is the RMS of the median star offset in up to eight equal-count bins along the overlap, less the noise of those medians, which comes from the star scatter within the bins (a scatter over the whole overlap would also hold the variation between the bins that is being measured). The plain RMS of the per-star offsets is recorded as well, but it measures centroid noise and blended stars rather than registration: on a real M31 panel pair it read 1.02 px where the systematic offset was 0.17 px.

7. **Colour.** The filters' mosaics share the canvas, so RGB/LRGB composition never resamples a channel.
   - A filter shot on only some panels covers those panels and is NaN elsewhere.
   - The colour cube keeps only the pixels where the R, G and B mosaics all have data; L may cover more. A filter's other photometric components are not part of the cube.

## The published directory

```
M31-mosaic/
  L.fits  R.fits  G.fits  B.fits   # one mosaic per filter, on the shared canvas
  LRGB.fits                        # linear colour cube (RGB.fits without L)
  previews/
  receipt.json
  details/                         # shared-calibration/, runs/<panel>/, mosaics/<filter>/, color/
```

`details/mosaics/<filter>/receipt.json` records every panel's scale (with its error relative to the reference), gain gradient and quadratic, every overlap's star count, ratio and astrometric offset, the overlaps without a scale and why, the background residuals (NOT_MEASURED with the reason where an overlap has too few bins), the extended-structure scale checks, flux conservation, the blend statistics and each window's catalog verification. A filter with several photometric components lists them under `components` and keeps each one in `details/mosaics/<filter>/component<k>/` as `mosaic_<F>_component<k>.fits` (with its NOISE, COVERAGE and MASK planes) and its own `receipt.json`; the project receipt lists them under `execution.mosaics.canvas.photometricComponents`. `details/runs/<panel>/receipts/canvas.json` records the panel's reference solution; its solution from the joint fit (order, matched stars, residuals at the edges and in the middle, overlap stars, and its own catalog-only fit under `ownFit`); the whole joint fit under `jointFit`; the residual shift fields of its Lights per filter under `frameFields` (each Light's own under `shiftField` in the pixel pipeline receipt); and its canvas window. The project receipt holds the joint fit under `execution.canvas.astrometry`.

## Not available yet

- Drizzle onto the canvas.
- Proper coaddition on canvas windows; it is refused. Selection region weight maps work on canvas windows: a partly clouded Light keeps its clean part.
- A panel with fewer than 30 catalog stars in its frame, placed by its overlaps alone; such a mosaic is refused before any panel runs.

A Bayer (one-shot-colour) Light set is one target's R, G and B panels: the pixel pipeline debayers each Light into the three channel groups (see [OSC / CFA](osc-cfa.md)); Bayer-as-mono processing never happens, and a Bayer set cannot share a target with mono R/G/B Lights.
