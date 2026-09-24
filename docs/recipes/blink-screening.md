# Blink-style screening: flags, a reference per channel, your decisions

> **Pre-1.0 boundary:** the flags and the reference rule were fixed on one six-night L/R/G/B campaign (NGC 6822: 97 Lights, L 44 / R 18 / G 17 / B 18) against the maintainer's own blink decisions in PixInsight, and are kept honest by a share-safe regression fixture built from that campaign's measurements. They pre-mark frames; they do not decide. The pipeline numerics are untouched: an identical admitted set gives bit-identical products whether it was admitted by the legacy gate or by an explicit selection.

## What it is and why it exists

The [legacy gate](automatic-screening.md) judges every Light against its own night: a frame is a defect when it departs from its night's peers. That model cannot see a night that is uniformly bad. On the NGC 6822 campaign the L channel had 44 Lights from three nights: 18 on a dark night, 23 on a moonlit night with a raw sky 1.9–2.4 times the dark night's, 52–58 % of its detected stars and 0.22–0.34 mag of extinction, and 3 on a night with a changing sky. Every moonlit frame was self-consistent, so the gate admitted 19 of the 23 (only the four that also had cloud were held or failed), and 39 L frames were integrated. The moonlit frames carry the same relative sky tilt as the dark ones (about −10 % of the sky level across the width) at 2.7–4× the sky, so once normalised to the reference's flux each contributed roughly four times the gradient amplitude of a dark-night frame; 19 of them carried about a quarter of the master's weight, and the luminance master showed a bright band across one corner and a blob in the opposite one. Measured with a 64-pixel box-median quadratic surface in units of the per-pixel sky noise, that master's gradient span was **6.80 σ** (corner spread 6.07 σ). Our pipeline on the 17 L frames the maintainer had blinked by hand (the legacy gate held one of them, so 16 were integrated) gave **3.28 σ / 4.59 σ**, and PixInsight WBPP on the same 17 frames (with LocalNormalization) gave **3.96 σ / 5.36 σ**: the flatness of the two is alike, and the gradient came from the admission, not from calibration, normalization or integration (the flat check and the R/G/B masters, all within 0.3–0.9 σ of PixInsight's, are recorded in the [validation matrix](../validation-matrix.md)).

Blink-style screening replaces the silent decision with a visible one:

1. Every Light is measured once, as the quality check already does (1/4-scale preview, SEP star detection, native-resolution PSF, the QC reference and registration, the spatial features, the unchanged quality gate).
2. *Flags* are computed per channel from those measurements. A flag records an advisory default and a reason; the GUI initializes every frame as pending review with KEEP selected, and the flags include the cross-night, absolute criteria the gate deliberately does not use.
3. A *reference frame* is chosen per channel, and every frame of the channel is registered and photometrically normalised to it, so you blink through frames that differ only where the sky did.
4. You decide frame by frame (keyboard, playback, "drop this night", undo). Your decisions go into the run as an explicit *selection*, recorded in the receipt with every override.

## The desktop flow

Import and the calibration check are unchanged. **Blink & select (N Lights)** measures the current Lights and opens a paused, chronological review. Processing is disabled until every frame has been displayed and every channel explicitly confirmed. Confirmation advances to the next unfinished channel. Changing a decision invalidates that channel's confirmation; importing Lights or measuring again resets the review. The optional measurement table remains read-only diagnostic evidence and cannot approve frames or bypass Blink. The headless CLI retains its separate legacy and unattended policies.

Only a decoded main image painted in the visible view counts as reviewed; prefetched images and thumbnails do not. Native WebKit previews are drawn into a canvas to avoid blank transformed IMG layers. Playback waits for the current image, pauses when the document is hidden, and stops at the channel end. Failed previews show a retry action and cannot silently count as viewed; explicitly dropping an unavailable frame acknowledges it. The native controller verifies the manifest digest, exact reviewed-frame and confirmed-channel sets, imported Light paths and explicit decisions before accepting Start. This is a workflow guard, not proof of the observer's attention.

The blink view:

- **Channel chips** (`L · 16/44 viewed`, with a check after confirmation), one per channel (target × filter × camera geometry × exposure bucket); keys `1`–`4` switch.
- **Stage**: the current frame at 1/8 scale in a viewport shared by every frame of the channel (all previews share the reference's grid and size), wheel to zoom at the cursor, drag to pan, double-click to fit; the 1/4-scale image replaces it when zoomed past 1.2×. The overlay shows name, night, decision and the flag chips with their values (`Sky ×2.43`, `Stars 55 %`, `Ext 0.23 mag`), and marks the reference. *Compare with reference* splits the stage reference | current with the same viewport; holding `C` A/B-blinks the two.
- **Filmstrip**: thumbnails sorted chronologically, with an optional flagged-first order, grouped by night with a header row (`2026-08-20 · 23 frames · sky ×2.2 · 0 kept`) that carries **Drop night** / **Keep night** buttons. A tile shows the decision colour bar, flag dots and the reference star.
- **Metrics** (inspector, or a strip under the stage): sky and sky ratio, stars and ratio, extinction, native FWHM and ratio, ellipticity, registration RMS and matches, overlap, background shape, score rank and z, every flag with its message and threshold, the gate disposition and codes, and the notes.
- **Launch bar**: kept / total per channel, *Start* with the selection, blockers for incomplete review or any channel with fewer than two kept Lights, *Back*.

| Key | Action |
|---|---|
| `←` `→`, `Home`, `End` | Step through the filmstrip order |
| `Space`, `K`, `D` | Toggle keep/drop, keep and next, drop and next |
| `F`, `Shift+F` | Next / previous flagged frame |
| `R` | Jump to the reference |
| `C` (hold) | Compare with the reference (A/B while held) |
| `P`, `[`, `]`, `Esc` | Play / pause at 2–8 frames per second, slower, faster, stop |
| `N` | Drop the current frame's whole night (undoable) |
| `Z`, `Cmd/Ctrl+Z` | Undo (up to 100 steps) |
| `+`, `−`, `0` | Zoom in, out, fit |
| `1`–`4` | Channel |

Playback starts only on request. *Kept only* is available after all frames in the current channel have been viewed; it cannot skip unseen frames during the initial review. Playback pauses when you step by hand.

## Complementary review displays

Desktop sessions use `previews.displayAlgorithm: "blink-complementary-display-v2"`; standalone `blink-measure` retains `shared-stretch-v1` unless explicitly requested. The new main star-detail view preserves attenuation and noise at a common reference scale, accompanied by a background-difference panel that does not fit away frame-to-frame gradients. Full field remains available for checking extended structure. Original-pixel crops have separate signal and amplitude-matched shape modes, with low-signal regions left unavailable. See [the mathematical contract and validation limits](../blink-display-redesign.md). Display algorithm and reference rule are recorded in the session manifest, whose digest binds the GUI selection. The later run still reports its independent QC/science reference; display-reference selection does not replace it.

## Flags

Flags are computed per channel from values that already exist per frame after measurement, analysis and the gate. Two severities:

- **EXCLUDE** → strong quality warning. The manifest retains advisory `defaultDecision: DROP` for reproducibility; the desktop does not apply it automatically.
- **ATTENTION** → quality hint (amber), also subject to human review.

The gate's evidence-insufficiency codes (`GATE_INSUFFICIENT_COHORT`, `GATE_INSUFFICIENT_NIGHT_BASELINE`, `GATE_NIGHT_UNRESOLVED`, `GATE_MORPHOLOGY_SAMPLE_REVIEW`, `GATE_SOURCE_COUNT_MISSING`, `GATE_FINITE_FRACTION_REVIEW`, `GATE_DYNAMIC_RANGE_MISSING`, `GATE_REFERENCE_NOT_CONNECTED`) become *notes*, not flags: they say the gate could not judge, not that the frame is bad. The single-frame B night of the campaign, which the maintainer kept, is the case.

### Channel statistics

Computed once per channel:

- **Clean set**: frames with extinction < 0.35 mag, a source ratio ≥ 0.60, a gate disposition other than HARD_FAIL and a successful registration. Extinction is the channel's airmass-corrected `extra_extinction_mag`, falling back to the nightly extinction residual when that is not available.
- **skyClean**: the median raw sky (`image_median` of the preview) over the clean set, defined only when the clean set has at least three frames.
- **sourcesBest**: the 90th percentile of the detected star count over the non-HARD_FAIL frames (the maximum when there are fewer than ten).
- **fwhmBest**: the 10th percentile of the native PSF FWHM (preview FWHM as fallback) over the non-HARD_FAIL frames.

### Absolute, cross-night criteria

| Flag | Value | ATTENTION | EXCLUDE | Why these thresholds |
|---|---|---|---|---|
| `BLINK_SKY_BRIGHT` | sky / skyClean | ≥ 1.6 | ≥ 1.6 **and** source ratio ≤ 0.60 (combined rule) | The moonlit L night sat at 1.89–2.43 with source ratios 0.52–0.58, its four cloud frames at 2.0–2.9; the dark night never exceeded 1.17, the changing-sky night 0.87–0.91, and no R/G/B frame exceeded 1.09. The largest within-night variation of a clean night was 1.33 (moonset), so 1.6 keeps a margin on its own. The second condition exists because a brighter but transparent night is not a defect: it stays ATTENTION, and only a bright sky *with* lost stars receives the EXCLUDE advisory. |
| `BLINK_SOURCES_LOW` | stars / sourcesBest | ≤ 0.60 | ≤ 0.45 | Moonlit frames 0.52–0.58 (attention; excluded through the combined rule); the dark night 0.81–1.02 except one frame at 0.57 that the maintainer also dropped; cloud frames 0.00–0.35 (excluded). In R, thick cloud gave 0.27 / 0.15 / 0.01 (excluded) and light cloud 0.57 / 0.55 (attention, kept by the maintainer); in G four frames at 0.06–0.43 are excluded; in B one frame at 0.56 is attention and was dropped by hand. |
| `BLINK_EXTINCTION` | extinction (mag) | ≥ 0.50 | ≥ 1.00 | Light cloud the maintainer kept measured 0.54–0.91 mag; frames dropped by hand measured 0.76–0.85 (attention) and 1.20–2.80 (excluded). The moonlit night's 0.22–0.34 mag does not reach the flag; the sky flag catches it. The dark night stayed ≤ 0.18. |
| `BLINK_BACKGROUND_SHAPE` | P95 − P5, over the outer cells (the central 40 % holds the target and is excluded), of the frame's normalised background-difference grid minus the median grid of the frames that share its meridian-flip orientation. The QC analysis already differences each frame's 16 × 16 SEP background, normalised to unit cell spread, against the reference's; comparing with the frame's own flip family (registered, not HARD_FAIL, extinction below 0.60 mag, at least three members) keeps vignetting, which rotates with the camera at a flip, from posing as a shape change. The result is a shape difference invariant to sky level and vignetting; a family with fewer than three members has no statistic | ≥ 0.50 | never alone | Dark-night frames 0.03–0.14 and moonlit frames 0.13–0.18: the same shape as the reference, consistent with the −10 % tilt common to every frame. The four moonlit cloud frames 1.16–1.35 and the changing-sky night 0.58–1.19 differ in shape; one late dark-night frame whose tilt reversed sign measured 1.02. Shape alone never excludes: it marks what to look at. |
| `BLINK_GRADIENT_AMPLITUDE` (only when master flats are supplied to `blink-measure`) | the frame's flat-corrected gradient amplitude (P99 − P1 of a quadratic surface over the outer cells of the 16 × 16 background grid) scaled to the reference's flux and divided by the reference's amplitude | ≥ 2.0 | ≥ 3.0 | On the calibrated frames the moonlit night measured 87–129 ADU against 26–44 ADU on the dark night, i.e. 3.0–4.5× after flux scaling, the dark night 0.6–1.5× and the changing-sky night ≈ 1.0×. Without a flat the raw-preview amplitude is proportional to the sky level and redundant with the sky flag, so it is not computed. |
| `BLINK_FWHM_WIDE` | native FWHM / fwhmBest | ≥ 1.30 | ≥ 1.60 | L best ≈ 4.07 px: normal frames ≤ 1.24, a cloud frame 1.36 (attention), the HARD_FAIL frame 1.73 (excluded); G frames at 1.27 that were kept do not reach the flag. 1.60 is the `balanced` night cut-off of the unattended policy. |
| `BLINK_STARS_ELONGATED` | median ellipticity | ≥ 0.30 (the gate's `trailing_review_median`) | through the gate's hard trailing codes | Nothing in the campaign lies between the two levels; the one strongly elongated G frame (≈ 0.4) is attention here and excluded by its star count. |
| `BLINK_UNREGISTRABLE` | registration failed | — | always | Three frames of the campaign (one per L, R and G). The engine refuses a KEEP on these (`SELECTION_UNREGISTRABLE`). |
| `BLINK_FEW_STARS` | fewer than 20 stars | — | always | One L frame with six detections. |

**Combined rule:** only `BLINK_SKY_BRIGHT` together with `BLINK_SOURCES_LOW` at the attention level escalates to EXCLUDE; both flags then carry `"combined": true`. No other pair escalates. In particular a source ratio of 0.45–0.60 with an extinction of 0.5–1.0 mag stays ATTENTION: on the campaign that pair would have pre-dropped three light-cloud frames the maintainer kept for the three it would have caught, and that band is the observer's call.

### Gate evidence mapped to flags

Applied after the absolute criteria; a frame already carrying the equivalent absolute flag does not get a duplicate.

| Gate code | Flag | Severity |
|---|---|---|
| `GATE_COHERENT_TRAILING_HARD`, `GATE_FRAGMENTED_TRAILING_HARD` | `BLINK_TRAILING` | EXCLUDE |
| `GATE_TRAILING_REVIEW` | `BLINK_TRAILING` | ATTENTION |
| `GATE_FOCUS_SEEING_REVIEW`, `GATE_NIGHT_FOCUS_SHIFT_REVIEW` | `BLINK_FOCUS` | ATTENTION |
| `GATE_OCCLUSION_HARD` / `GATE_OCCLUSION_REVIEW` | `BLINK_OBSTRUCTION` | EXCLUDE / ATTENTION |
| `GATE_SPATIAL_DIMMING_STRONG` / `GATE_SPATIAL_DIMMING_REVIEW` | `BLINK_CLOUD_PATCHY` | EXCLUDE / ATTENTION |
| `GATE_MULTI_FAMILY_CLOUD_HARD` | `BLINK_CLOUD_THICK` | EXCLUDE |
| `GATE_SOURCE_RETENTION_STRONG` / `GATE_SOURCE_RETENTION_REVIEW` (nightly) | `BLINK_SOURCES_LOW` (only if not already set) | ATTENTION |
| `GATE_TEMPORAL_EXTINCTION_*`, `GATE_NIGHT_ZEROPOINT_SHIFT_REVIEW` | covered by `BLINK_EXTINCTION` | — |
| `GATE_COMMON_FOOTPRINT_REVIEW` | `BLINK_FIELD_MISMATCH` (overlap below 0.5: wrong field or a large offset) | ATTENTION |
| `GATE_REGISTRATION_REVIEW` with a successful registration | `BLINK_REGISTRATION_WEAK` | ATTENTION |
| `GATE_BACKGROUND_STRONG` / `GATE_BACKGROUND_REVIEW`, `GATE_NOISE_REVIEW` | `BLINK_NIGHT_OUTLIER` | ATTENTION |
| `GATE_MEASUREMENT_FAILED`, `GATE_IDENTITY_*`, `GATE_FINITE_FRACTION_HARD`, `GATE_NEAR_CONSTANT_IMAGE`, `GATE_NOT_A_LIGHT_FRAME` | `BLINK_UNMEASURABLE` | EXCLUDE |

Blink flags never use the gate's nightly z-scores (which were unstable on nights of one to three frames); `BLINK_NIGHT_OUTLIER` comes from the gate codes only.

### The result on the campaign

Every EXCLUDE agrees with the maintainer's drop except one G frame with light cloud (extinction 1.27 mag, source ratio 0.43) that had been kept "for now" — one click restores it. No dark-night L frame is EXCLUDE, and only one carries an ATTENTION flag (the one the maintainer also dropped). All 23 moonlit L frames are EXCLUDE and the night header offers *Drop night*. The R/G/B light-cloud frames the maintainer kept are ATTENTION, default keep. The thresholds live in a frozen policy object (`blink-flags-v1`) with a canonical digest recorded in every manifest, selection and receipt, so a changed threshold is visible.

## The reference frame per channel

The reference is the frame the others are registered and normalised to for blinking, and the one *Compare* shows. It is chosen by a PSF-signal-weight proxy (`psf-signal-weight-proxy-v1`): for frame *i* of channel *c*,

```
S_i = (T_i² / (1.4826 · σ'_i)²) · (F_c / F_i)² · (1 − e_i) · min(1, N_i / N_c)

F_c = 10th percentile of the native FWHM over the candidate frames
N_c = 90th percentile of the star count over the frames without an EXCLUDE flag
```

where *T* is the frame's transparency ratio to the QC reference, *σ'* the robust noise of its preview (MAD, raw ADU, which includes the sky's photon noise), *F* its native PSF FWHM (preview FWHM as fallback), *e* its median ellipticity and *N* its star count. In words: the signal power of a unit-flux star over the background noise power, divided by the PSF area (a wider PSF spreads the same flux), times the roundness, times the detection completeness. A moonlit frame loses on all of *T*, *σ'* and *N* at once; on the campaign's L channel the best dark-night frame scores about forty times any moonlit frame.

Candidates are frames with no EXCLUDE and no ATTENTION flag (if that leaves nothing, ATTENTION is allowed; if still nothing, every frame), a successful registration with at least 30 matched stars and an RMS of at most 1.5 px, an overlap of at least 0.9 with the QC reference, and finite *T* and *F*. Ties (in order): larger *S*; smaller distance of the frame's registration translation to the channel's median translation, so the reference is central in the dither pattern and the common footprint is largest; earlier observation time; path. The manifest stores each frame's `log10 S`, its z-score over the channel and its rank; the interface shows rank and z, never the raw number.

The blink reference is not copied into the run as its registration or normalization reference; the pipeline keeps its own rules (the normalization reference is the flattest low-sky frame among the best quarter by quality, `stellar-scale-hint-reference`; the registration reference is chosen at registration). What the blink evidence does change is the candidate set: a frame that carries any blink flag never anchors a group — neither as the geometric reference nor as the normalization reference — while unflagged admitted frames exist. The old rule's preference for the lowest sky otherwise picks exactly the cloud-dimmed or changing-sky frames the flags mark, and the master inherits the reference's background: on the NGC 6822 luminance the flag defaults alone left a 6.8 σ large-scale structure with such a reference and give 2.3 σ with the restriction. Products of an identical admitted set stay bit-identical for a given candidate set; where the restriction moves a reference the pixels move, so the change was checked with the master evaluator on the reference project (luminance: every family PASS/WARN as before, effective-noise gain and high-order background residuals slightly better). The run records the blink reference next to the pipeline's references (`qualityControl.blink.referenceBySource`, `pipelineReferences`) so both are auditable.

## Previews

- **One read per frame.** The measurement pass that already reads every Light at a long edge of 2048 px keeps the 1/4-scale linear preview and its 2 × 2 block mean (1/8 scale); no second decode.
- **Registered to the reference** with the similarity transform the QC analysis estimated (no new star extraction). Frames without a transform are shown unregistered and flagged `BLINK_UNREGISTRABLE`; meridian flips come out of the same transform. The uncovered area is black and the covered fraction is recorded as `coverage`.
- **Photometrically normalised** with a linear model from the measurements: `x' = (x − sky_i) · g_i + sky_r`, `g_i = T_r / T_i` (1 when either transparency is unknown), where `sky` is the preview's median. The pair (`skyOffset`, `fluxScale`) is recorded per frame.
- **One shared screen transfer per channel**, from the reference's normalised 1/8 preview: `σ_r = 1.4826 · MAD`, shadows clip `black = sky_r − 2.8 σ_r`, highlight clip `white = sky_r + 1000 σ_r`, and a midtone `m` solved so the reference sky lands exactly on the target 0.25:

  `x = clip((x' − black) / (white − black), 0, 1)`, `y = MTF(m, x) = ((m − 1) x) / ((2m − 1) x − m)`, `m = x₀ (t − 1) / (2 t x₀ − t − x₀)` for `x₀ = (sky_r − black) / (white − black)`.

  This is the PixInsight-style auto-stretch. Every channel's background sits at the same grey whatever its sky level, and the bright end stays graded instead of clipping: sky + 10 σ renders at 0.61 and sky + 200 σ at 0.96, so a thick cloud is a shaped blob rather than the same white as the galaxy. The parameters are in `channels[].stretch` as `mode: "stf"`, `shadowsClip`, `midtone`, `target`, `black`, `white`, `skyReference`, `sigmaReference`. The highlight clip only decides where the far highlights saturate; the midtone fixes the curve around the background.
- **A second, harder filmstrip image per frame** at target 0.45 (`previews.filmstripHard`), rendered from the same array in memory, so the interface can offer a contrast toggle without re-rendering. It costs about 0.2 s for a 97-frame session.
- **Scales and formats**: filmstrip and stage at 1/8 (782 × 522 for a 6252 × 4176 sensor), grayscale JPEG; zoom at 1/4 (1563 × 1044), 8-bit PNG loaded on demand and cached. The desktop caps inline previews (200 KB each, 32 MB in total); frames beyond the budget are fetched on demand.
- **Where**: a create-only session directory under the platform cache root (`~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows), never inside a source folder: `Ultra-Fast-WBPP/blink-sessions/<digest of the Light list>-<timestamp>/` with `manifest.json`, `filmstrip/`, `filmstrip-hard/`, `zoom/`. The desktop keeps at most three sessions (older ones are removed before a new measurement) and removes them at *Clear*.

## The selection file (`selection-v1`)

The desktop writes one decision per Light; a command-line user can write the file by hand. Digests are the Lights' content hashes as the blink manifest (`frames[].sourceSha256`), a run receipt (`sources[].sha256`) or `qc/manifest.json` print them; `origin.blinkManifestSha256` is the digest of the session's `manifest.json` file.

```json
{
  "schemaVersion": 1,
  "kind": "ultra-fast-wbpp-selection",
  "policy": "explicit-v1",
  "origin": {
    "sessionId": "0d7c2f1a9b3e4c5d-20260922-101530",
    "blinkManifestSha256": "sha256:3f9c0b7a1e2d4c6f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f",
    "flagsPolicyDigest": "sha256:a1b2c3d4e5f60718293a4b5c6d7e8f9012345678abcdef0123456789abcdef01",
    "createdAt": "2026-09-22T10:21:44+08:00"
  },
  "undecided": "ERROR",
  "decisions": [
    {
      "sourceSha256": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "decision": "KEEP",
      "defaultDecision": "KEEP",
      "flags": []
    },
    {
      "sourceSha256": "sha256:fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
      "decision": "DROP",
      "defaultDecision": "DROP",
      "flags": ["BLINK_SKY_BRIGHT", "BLINK_SOURCES_LOW"]
    },
    {
      "sourceSha256": "sha256:00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
      "decision": "KEEP",
      "defaultDecision": "DROP",
      "flags": ["BLINK_SKY_BRIGHT"],
      "note": "user: kept, faint gradient acceptable"
    },
    {
      "sourceSha256": "sha256:ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100",
      "decision": "DROP",
      "defaultDecision": "KEEP",
      "flags": ["BLINK_EXTINCTION"],
      "note": "user: dropped, cloud on the galaxy"
    }
  ]
}
```

Rules: digests are unique, lowercase `sha256:` values; `decision` is `KEEP` or `DROP`; `undecided` (`ERROR`, `DROP` or `KEEP`) says what happens to a Light in the run that has no decision — the desktop always sends every Light, so it sends `ERROR`; at most 10 000 entries; `origin`, `defaultDecision`, `flags` and `note` are optional and recorded, not enforced, so a hand-written file with digests and decisions is valid. The canonical JSON digest of the file is the `selectionDigest` in the receipts.

## Command line

Measure and flag a set of Lights (the request file is private, like the quality-check request; the manifest is written to the session directory and to stdout):

```bash
.venv/bin/ultra-fast-wbpp blink-measure --request-json blink-request.json --compact
```

```json
{
  "schemaVersion": 1,
  "lightPaths": ["/data/night1/L/light-001.fits", "/data/night2/L/light-002.fits"],
  "sessionDirectory": "/data/blink/session-001",
  "workers": 8,
  "previews": {"filmstripScale": 8, "zoomScale": 4, "filmstripFormat": "jpeg", "jpegQuality": 85},
  "masterFlats": [{"filter": "L", "path": "/data/masters/masterFlat_L.xisf"}]
}
```

`masterFlats` is optional and enables `BLINK_GRADIENT_AMPLITUDE`; `workers` is optional (the hardware default applies). The session directory must not exist (`BLINK_SESSION_EXISTS`); other request errors are `BLINK_REQUEST_INVALID`, `BLINK_INPUT_NOT_LIGHT`, `BLINK_NO_LIGHTS` and `BLINK_PREVIEW_FAILED`.

Run with a selection (positional mode), or put the same object under a top-level `"selection"` key of a `--request-json` project request:

```bash
.venv/bin/ultra-fast-wbpp run-project /data/night1 /data/night2 /data/masters \
    --recipe docs/recipes/mono-standard.json --selection selection.json \
    --output /data/new-result --progress-json
```

A supplied selection sets the effective selection policy to `explicit-v1`; a recipe that asks for `unattended-v1` or `include-all` at the same time, or a request that also carries `reviewSelections` / `reviewApprovals`, is refused (`SELECTION_POLICY_CONFLICT`). Without a selection, recipes keep today's behaviour: `legacy-gate` by default, `unattended-v1` as an option. Selection errors are stable codes: `SELECTION_INVALID`, `SELECTION_SOURCE_UNKNOWN` (a digest that is no Light of the request), `SELECTION_SOURCE_AMBIGUOUS` (a digest that belongs to more than one target run), `SELECTION_INCOMPLETE` (a Light without a decision while `undecided` is `ERROR`), `SELECTION_UNREGISTRABLE` (a KEEP on a frame the quality pass could not register — the run fails closed, for the same reason as `REVIEW_APPROVAL_UNREGISTRABLE`). A KEEP on a HARD_FAIL trailing or obstruction frame is honoured and counted as `overriddenGateHardFail`; the observer decides, and nothing is silent. Every target × filter panel still needs at least two kept Lights (`QC_INSUFFICIENT_LIGHTS`).

## What a run records

The measurement, analysis and gate run again inside the run, unchanged, and `qc/manifest.json` is written as before; the explicit selection then replaces the gate's admitted set. The measurement is not recomputed from scratch: each frame's complete measurement is cached under its SHA-256, the QC configuration and a fingerprint of the implementation (`measurementCache` in both manifests counts `hits`/`misses`/`writes`), so a run after a blink session reuses it and, because the restored measurements are identical, the group analysis cache hits as well. On the 97-frame NGC 6822 campaign that is 9.1 s → 5.6 s of measurement and 25.0 s → 0.6 s of analysis, with bit-identical masters. `UFWBPP_QC_CACHE_DIR=off` disables both caches.

- **Run receipt**: `qualityControl.selectionPolicy: "explicit-v1"`; `qualityControl.selection` with the policy, the `selectionDigest`, the `origin`, the `flagsPolicyDigest`, the counts (`keep`, `drop`, `overriddenExcludeFlags`, `overriddenGateHardFail`, `undecided`) and one record per frame (`sourceSha256`, `decision`, `defaultDecision`, `flags`, `gateDisposition`); `qualityControl.blink` with the path of `qc/blink.json`, the blink reference per channel (`referenceBySource`) and the pipeline's own registration and normalization references (`pipelineReferences`).
- **`qc/blink.json`**: the flags, the reference, the scores and the night summaries, computed by the same functions as the session, so a result is self-describing without the session directory. It is share-safe (basenames and `source/<id>` like the QC manifest, no absolute paths). Previews are not written by default; `qc.blinkPreviews: true` in the request's `execution` block writes them under `qc/blink/` for command-line users who want them in the result.
- **Screening summary** (`execution.screening` of the project receipt, unchanged in shape, plus `selectionPolicy`): every DROP is listed with `reason: "USER_DROP"` and its flag codes, and every KEEP that overrode an EXCLUDE flag with `reason: "USER_KEEP_OVERRIDE"`, so the result page shows both.

## Keeping mildly gradient-affected frames

Nothing changes in the pipeline when you keep a few frames with a stronger sky gradient: the global normalization's additive grid and its flattest-convex-combination low-order target (receipt `lowOrderTarget`) remove the plane, and the integration weights down-weight the noisier frames. The flags (`BLINK_SKY_BRIGHT`, `BLINK_BACKGROUND_SHAPE`) make you aware; the decision is yours. The former LocalNormalization option has been retired; the supported normalization is described in [the normalization guide](normalization-and-xisf.md).

## Limits

- **A moonlit night is only visible when a darker clean night exists in the channel.** `skyClean` needs at least three clean frames; a channel with a single hazy night has nothing darker to compare with and gets no sky flag (the extinction, source and shape flags still apply).
- **Gradient amplitude needs flats.** Without master flats the raw-preview gradient is proportional to the sky level and is not computed; the sky flag carries that information.
- **Previews are 1/8 and 1/4 scale**, the filmstrip a quality-85 JPEG. They are for blinking and for spotting gradients, cloud and trails, not for judging single pixels; the run reads the original frames.
- **Flags are suggestions, not human decisions.** Thresholds were fixed on one campaign; the light-cloud band (source ratio 0.45–0.60, extinction 0.5–1.0 mag) is left to the observer on purpose.
- **The reference is for blinking.** The run's registration and normalization references follow the pipeline's own rules (above).
- **The legacy gate still runs** and its dispositions are recorded; an explicit selection replaces its admitted set, it does not change its evidence.

## Regression

The flags and the reference rule are held by `packages/light-frame-qc/tests/test_blink_ngc6822_regression.py` on a share-safe fixture built from the campaign's measurements (per frame: the flag inputs, night, filter, a synthetic id and the basename; no digests, no paths, no coordinates) with the assertions listed above (all 23 moonlit L frames DROP, no dark-night frame EXCLUDE, the reference a dark-night frame with extinction below 0.1 mag, the night summary marking the moonlit night for dropping). The end-to-end replay (three explicit selections through `run-project` with hash comparison of the masters) is recorded in the [validation matrix](../validation-matrix.md). See also the [legacy gate](automatic-screening.md) and the [architecture](../architecture.md#scientific-work).
