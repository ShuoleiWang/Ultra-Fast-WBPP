# Recipes

A recipe is a versioned scientific contract, not a loose collection of GUI toggles. It records compatible frame roles, calibration rules, Quality Gate policy, registration model, normalization, rejection, integration/drizzle parameters, crop policy, solver gate, output roles, backend capability requirements, and resource limits.

## Choose a path

| Input and goal | Start here | Pre-1.0 evidence boundary |
|---|---|---|
| One or more mono N.I.N.A. folders, ordinary integration | [N.I.N.A. mono](nina-mono.md) then [calibration](calibration.md) | 38-frame private B-channel real-data acceptance on one M3 Pro/QHY268M fixture |
| Undersampled mono with measured dithers | [Drizzle](drizzle.md) | Native 1×–4× drizzle of the ordinary integration's inputs; NGC 7331 2× real-data run recorded |
| Strong gradients/transparency changes | [XISF and LocalNormalization](xisf-and-local-normalization.md) | Conservative opt-in synthetic validation; not PixInsight-equivalent and no retained real-data acceptance |
| Required final celestial WCS | [Astrometry](astrometry.md) then [offline catalogs](offline-solver-catalogs.md) | Real managed Astrometry.net solve; solver/indexes are user-installed and no field is guaranteed before it solves |
| Four mono-camera panels with R/G/B or L/R/G/B | [Four-panel RGB/LRGB](project-mosaic-rgb.md) | Synthetic 4-panel execution; shared raw-Dark reuse and `PROPAGATED_VERIFIED` post-reprojection WCS provenance regressions pass; no retained real-data mosaic acceptance yet |
| One-shot colour (Bayer) | [OSC / CFA](osc-cfa.md) | Synthetic RGGB set built from real mono Lights: channel photometry exact, 1× debayer widens stars 5–10 %, 2× Bayer drizzle recovers full resolution; no real OSC data yet |

## Built-in contracts

- **Balanced mono** — ordinary weighted integration for mono L/R/G/B or narrowband data. This is the default when sampling and dithering do not justify Drizzle.
- **Drizzle mono** — 1×–4× output scale with an explicit kernel (square, circular, gaussian, point), drop shrink, the integration's own weights, normalization and rejection masks, and science/weight/coverage products; sampling and coverage evidence is advisory.
- **Narrowband** — per-filter measurement/normalization without comparing brightness between filters; designed for Ha/OIII/SII and arbitrary named filters.
- **Mosaic panel** — solves panel masters independently, reprojects same-filter panels, solves each resulting mosaic again, then aligns solved filters for color output.
- **One-shot colour** — Bayer Lights are calibrated as mosaics with per-channel flat scaling, debayered into R/G/B channel groups and, with drizzle, Bayer-drizzled from the mosaic samples; the recipe is the Balanced or Drizzle recipe of the mono case.

The GUI recommends a recipe from metadata but never silently changes one after planning. Any change produces a new canonical recipe digest and invalidates only affected downstream artifacts.

For the complete product sequence, return to the [main README](../../README.md); a [Simplified Chinese quick start](../README.zh-CN.md) is also available, and the [documentation index](../README.md) maps every page.
