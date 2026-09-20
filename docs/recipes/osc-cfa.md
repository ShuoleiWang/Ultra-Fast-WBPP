# One-shot-colour (Bayer / OSC) recipe

A Light whose header names a Bayer pattern (`BAYERPAT`, `CFAPAT`, or the PixInsight
`PCL:CFASourcePattern` property; `RGGB`, `BGGR`, `GRBG` and `GBRG` are supported) is
processed as one-shot colour. No recipe switch is needed: the same Balanced and
Drizzle recipes apply, and a mono run of the same target is unchanged.

## What happens

1. **Screening and registration measure a luminance.** Block-mean previews of a
   mosaic use an even block size, so every preview pixel averages the same
   number of red, green and blue samples; the native PSF stamps and the
   full-resolution registration refinement read a bilinear-debayered luminance.
   Star shapes and transforms are therefore those of the sensor, in the
   mosaic's own pixel coordinates.
2. **Calibration stays in the mosaic domain.** Bias and dark are subtracted
   pixel by pixel. The master flat is applied with **separate scaling factors
   per colour channel** (each Bayer channel is divided by the flat normalized
   to its own channel median, PixInsight's "separate CFA flat scaling
   factors"), so the flat panel's colour does not tint the frame and the
   sensor's channel response is preserved. Hot pixels are replaced by the median
   of their eight same-colour neighbours.
3. **Debayer, then three channel groups.** The calibrated mosaic is debayered
   (bilinear, same-colour neighbours) into R, G and B planes; each plane is
   registered with the Light's transform into a colour channel group named
   `R`, `G` or `B`. From there a channel group is an ordinary filter group:
   global normalization, rejection, weights, region weight maps, the shared
   auto-crop and the masters are the same code. The masters carry `OAFCFA`
   (pattern), `OAFCFACH` (channel) and `OAFCFAF` (the Lights' own filter name).
   The project layer turns one Bayer Light set into the panels `R`, `G`, `B`
   of its target and builds the RGB product as for a mono R/G/B run.
4. **Bayer drizzle.** With `drizzle.enabled`, each channel group is drizzled
   from the calibrated **mosaics** themselves: only the samples of that colour
   are dropped, with the channel group's normalization coefficients, weights
   and rejection masks (receipt `recipe.cfaPattern`/`cfaChannel`, header
   `OAFDRZCF`/`OAFDRZCH`). Nothing is interpolated, so with enough dithers the
   drizzled channels reach the sensor's resolution.

Masters (bias, dark, flat) must describe the same Bayer pattern as the Lights;
masters built by the run inherit it (`BAYERPAT`). Frames with an unknown CFA
state still need the hash-bound confirmation of the strict workflow, and a
run integrates one Bayer filter per target — mono `R`/`G`/`B` filters of the
same target would collide with the channel groups (`CFA_CHANNEL_FILTER_COLLISION`).

## Validation (synthetic RGGB set from real mono Lights)

The 26 NGC 7331 L Lights (26 MP, four dithered nights with a meridian flip) were
turned into RGGB mosaics with channel gains R 0.80 / G 1.00 / B 0.65, a
colour-tinted master flat and the real master dark/bias, and compared with the
mono L master of the same frames (WCS-matched stars):

| | R | G | B |
|---|---|---|---|
| sky vs mono × gain | 259.8 / 259.7 | 324.6 / 324.6 | 210.9 / 211.0 |
| star flux (r = 4 px) vs mono × gain, 1× debayered | 0.973 | 0.987 | 0.972 |
| half-light radius vs mono, 1× debayered | 1.096 | 1.048 | 1.097 |
| star flux vs mono × gain, 2× Bayer drizzle | 0.995 | 0.996 | 0.995 |
| half-light radius vs mono, 2× Bayer drizzle | 0.972 | 0.969 | 0.973 |

The 1× channel masters are photometrically the mono master times the channel
gain; bilinear debayering widens stars by 5 % (G) to 10 % (R, B) and the
apertures lose 1–3 % accordingly. The 2× Bayer drizzle recovers the sensor's
resolution in every channel (3 % *sharper* than the Lanczos-3 mono master, as
the mono drizzle is) with exact photometry. The project run takes 79 s (1×,
with the native debayer kernel) and 127 s (2× Bayer drizzle) on the M3 Pro. Real one-shot-colour data has not been
processed yet; the synthetic set exercises every code path but not a real
colour filter array's crosstalk or a real OSC flat.
