# Drizzle recipe

Drizzle reconstructs an output grid from calibrated input samples, subpixel registration mappings, weights, and rejection masks. It is not merely a larger resize and does not replace calibration or registration.

## What the drizzle uses

The drizzle of a filter group is a second integration of the very same inputs the ordinary integration used, on a grid `scale` times finer than the reference frame:

- the calibrated, **unregistered** Lights and their registration matrices (the same full-resolution matrices the Lanczos-3 registration used);
- the group's global normalization coefficients: the multiplicative scale, the additive offset and the additive offset grid of every frame, applied in the integration's Float32 arithmetic at the registered position;
- the integration weights of every frame (noise weights, selection weights, region weight maps);
- the per-sample rejection decisions of the ordinary integration (MAD rejection, transient corridors). The accepted-sample masks live on the reference grid; a pixel is dropped only when the mask accepts it at the pixel's registered position, which is how PixInsight's DrizzleIntegration reads ImageIntegration's rejection maps.

Nothing is re-estimated on the drizzled grid, so the drizzled master is the ordinary master's exact photometric twin (flux ratio 0.993–0.994 in a 4-native-pixel aperture on the NGC 7331 data set, same sky level, geometry aligned to 0.02 output pixels) with a finer sampling.

## Options

| Recipe field | Values | Default | Meaning |
|---|---|---|---|
| `drizzle.enabled` | `true`/`false` | `false` | Adds the drizzle stage after the ordinary integration; the drizzled masters are the run's products |
| `drizzle.scale` | `1`, `2`, `3`, `4` | `2` | Output pixels per reference pixel. `1` is a drizzle onto the reference grid itself (no interpolation, exact drop areas) |
| `drizzle.dropShrink` | `0.1`–`1.0` | `0.9` | Drop shrink (pixfrac): the side of the square drop in input pixels. PixInsight WBPP uses 0.9 for mono and 1.0 for CFA |
| `drizzle.kernel` | `square`, `circular`, `gaussian`, `point` | `square` | Drop shape. Square and circular drops are exact area overlaps (polygon clipping, disc–rectangle area); the Gaussian kernel has FWHM equal to the drop width and carries the square drop's area; point drops the whole pixel on one output pixel |
| `drizzle.cfaDrizzle` | `true`/`false` | `false` | Bayer drizzle: three colour planes dropped from the mosaic's own pixels (`RGGB`, `BGGR`, `GRBG`, `GBRG`). Requires CFA Lights, which the mono pipeline does not admit yet |
| `drizzle.backend` | `auto`, `native-drizzle` | `auto` | The multithreaded native kernel is the only backend |

The desktop exposes the same scale, kernel and drop-shrink controls under *Advanced options*; the CLI takes `--mode drizzle --drizzle-scale N --drop-shrink F --drizzle-kernel K`.

## Products and evidence

Each filter's drizzled master is one multi-extension FITS: `SCI` (the weighted mean; NaN where no drop landed), `WHT` (the accumulated drop weight) and `COVERAGE` (number of contributing frames per pixel). The header records `OAFDRZ`, `OAFDRZSC` (scale), `OAFDRZPF` (drop shrink), `OAFDRZKN` (kernel), `OAFNFRM` and, for a Bayer drizzle, `OAFDRZCF`. The receipt `receipts/drizzle_<filter>.json` lists every input with its transform, weight, normalization and mask provenance, the dither phases, coverage and weight percentiles, the null-pixel fraction and the stage timing; its identifier is the hash of its content and the E2E runner verifies artifact and receipt again before promotion.

Sampling (QC FWHM), dither and coverage evidence is recorded as **advisory** (`coverageGate.advisory`, `sampling`): a well-sampled field or a small dither set is reported, not blocked, because the drizzle reuses a rejection that was already validated on the reference grid and its result is still a valid integration.

The drizzled masters of one run share their grid by construction (`scale` times the reference grid). As for ordinary masters, every filter is solved on its own, the independent solutions verify the shared grid at solver precision (tolerances are expressed in native pixels and scaled by `scale`) and the lowest-RMS solution is written to every master, so the channels combine without resampling.

## Performance and quality (NGC 7331, 61 × 26 MP Lights, L 26 / R 11 / G 11 / B 13, M3 Pro)

- Drizzle stage 2× square, drop shrink 0.9: 0.68 s per 26 MP frame (the next frame is read while the current one is dropped), 50 s for the four filters including the 1 GB products; the whole project run takes 178 s against 97 s without drizzle (the rest is the solve, alignment and colour products of four times as many pixels).
- Against the ordinary Lanczos-3 master in native-pixel units: half-light radius 2.5–4.4 % smaller (L 1.96 vs 2.01 px, R 1.90 vs 1.97, G 2.11 vs 2.21, B 2.07 vs 2.15), effective noise over 4 × 4 native pixels 6–7 % lower, star flux ratio 0.993–0.994, coverage 99.95–100 %. Block-averaged back to the reference grid the stars have the same ellipticity as the ordinary master.
- A comparison with PixInsight's DrizzleIntegration output on the same data is still to be recorded; PixInsight's WBPP is run manually for that.

## Notes

Use drizzle for undersampled data with enough distinct subpixel dither positions; well-sampled data gains resolution only marginally and pays four times the storage. Start with the square kernel and a drop shrink of 0.9, then inspect `COVERAGE` and the correlated-noise evidence.

Ultra-Fast WBPP does not consume PixInsight `.xdrz` files. The final drizzled master is solved again instead of inheriting a seed WCS.
