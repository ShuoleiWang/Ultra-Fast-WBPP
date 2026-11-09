# Recipes

A recipe is a versioned scientific contract, not a loose collection of GUI toggles. It records compatible frame roles, calibration rules, Quality Gate policy, registration model, normalization, rejection, integration/drizzle parameters, crop policy, solver gate, output roles, backend capability requirements, and resource limits.

## Choose a path

| Input and goal | Start here | Pre-1.0 evidence boundary |
|---|---|---|
| One or more mono N.I.N.A. folders, ordinary integration | [N.I.N.A. mono](nina-mono.md) then [calibration](calibration.md) | Real L/R/G/B projects on one M3 Pro ([validation matrix](../validation-matrix.md)): NGC 7331 (61 Lights, four nights), NGC 6822 (97 Lights, six nights), SDSS J1004+4112, RCS2 J0327-1326 and M31; the first retained acceptance was a 38-frame B-channel run |
| Undersampled mono with measured dithers | [Drizzle](drizzle.md) | Native 1×–4× drizzle of the ordinary integration's inputs; NGC 7331 2× real-data run recorded |
| Strong gradients/transparency changes | [XISF and normalization](normalization-and-xisf.md) | Stellar scale and guarded background correction; no PixInsight-equivalence claim |
| Required final celestial WCS | [Astrometry](astrometry.md) then [offline catalogs](offline-solver-catalogs.md) | Real managed Astrometry.net solve; solver/indexes are user-installed and no field is guaranteed before it solves |
| Several panels, nights and filters of one mosaic | [Mosaics](project-mosaic-rgb.md) | Synthetic sets through the real pipeline (optical distortion, lateral colour, Bayer, 2× canvas drizzle, a bright galaxy across an overlap) and real M31 mosaics: a 2×2 that publishes with the Tycho-2 set or the Gaia tiles, five panels, one of them cloudy, and six panels (713 Lights) |
| One-shot colour (Bayer) | [OSC / CFA](osc-cfa.md) | Synthetic RGGB set built from real mono Lights: channel photometry exact, 1× debayer widens stars 5–10 %, 2× Bayer drizzle recovers full resolution; no real OSC data yet |

## Calibration contracts and options

The only named contracts are the two calibration workflows of `calibration.workflow`:

- **`mono-standard-v1`** — the desktop's: calibration paired by WBPP's rules, every compromise reported as a warning ([calibration](calibration.md)). On the command line, pass [mono-standard.json](mono-standard.json) with `--recipe`.
- **`strict-v1`** — the default of a recipe that names no workflow, kept for older command-line recipes: positive evidence for the full acquisition profile, exact Dark exposures and no ambiguous choice.

Everything else follows from the inputs or from a recipe field; none of it is a named preset:

- **Ordinary integration** of mono L/R/G/B or narrowband data is the default. Every filter, Ha/OIII/SII or any other name, is measured and normalized on its own, and filters are never compared by brightness.
- **Drizzle** — `drizzle.enabled` with a 1×–4× `scale`, a `kernel` (square, circular, gaussian, point) and `dropShrink`; it uses the integration's own weights, normalization and rejection masks and writes science/weight/coverage products; sampling and coverage evidence is advisory.
- **Mosaic** — Lights of several panels plan one canvas for every panel and filter; each panel's Lights are integrated once onto its window, the panels are matched by overlap-star photometry and background polynomials solved for all overlaps at once, blended by inverse variance, and the canvas WCS is verified on catalog stars ([mosaics](project-mosaic-rgb.md)).
- **One-shot colour** — Bayer Lights are calibrated as mosaics with per-channel flat scaling, debayered into R/G/B channel groups and, with drizzle, Bayer-drizzled from the mosaic samples, under the same recipe as mono data ([OSC / CFA](osc-cfa.md)).

## Opt-in extras

[Advanced algorithms](advanced-algorithms.md) covers two options a recipe must turn on: ZOGY proper coaddition, which adds one more product per filter next to the ordinary master, and the robust IRLS combination of the accepted samples. Both are off by default and a recipe that omits them is unchanged, digest included. It also describes `SuperL.fits`, the super luminance every project with two or more qualifying broadband channels publishes beside them unless `superLuminance.enabled` is false, and `color.luminance`, which makes it the LRGB product's luminance.

The desktop does not choose a recipe from metadata: it always builds a `mono-standard-v1` recipe from its advanced options (drizzle, proper coaddition, super luminance). Any change produces a new canonical recipe digest and invalidates only affected downstream artifacts.

For the complete product sequence, return to the [main README](../../README.md); a [Simplified Chinese quick start](../README.zh-CN.md) is also available, and the [documentation index](../README.md) maps every page.
