# Master light evaluation standard

How an Ultra-Fast WBPP master is judged against the PixInsight WBPP master of the same raw data. Implemented by [`tools/validation/evaluate_masters.py`](../tools/validation/evaluate_masters.py); the script covers the verdict metrics of every family below (the §9 attribution against a truth reference and the §3.6 LSB metric are not yet implemented and are reported as WARN/INFO).


Version 1.0. Answers, per filter, with numbers and PASS/WARN/FAIL: "Is O at least as good as P in every
visually and scientifically relevant aspect, and better in SNR?" Implementable with numpy/scipy/astropy/sep.

## 0. Inputs, notation, conventions
- Inputs: two linear Float32 mono FITS of the same raw data and filter: `P` (PixInsight, ~0..1 units) and
  `O` (ours, ADU). Optional: the other filters' masters of both pipelines (for §7.2), and the registered
  calibrated single frames of either pipeline (for "truth" references, §9). Typical: 6252x4176 px, FWHM ~4 px.
- Notation: `T` maps O-grid coordinates to P-grid coordinates. Photometric model `P ≈ a·O + b`;
  `O' = a·O + b` is O expressed in P units, `σ_O' = a·σ_O`. `σ1` = per-pixel sky noise (§3.1),
  `σ_ref = σ1,P` (P's noise, in P units) is THE yardstick for every background/gradient amplitude of BOTH
  images. Never express amplitudes as "% of sky" (pedestal-dependent). `MADN(x) = 1.4826·median|x − median x|`.
- Coordinates: sep convention (0-based, integer = pixel centre). Random seed 0 everywhere. Runtime budget:
  < 3 min per filter pair on an M3 Pro (§10).
- Every metric is reported as: value_P, value_O (in P units when dimensional), signed difference `d`
  (positive = O worse), tolerance `τ`, 95% bootstrap CI of d when defined, status. Summaries: global
  (median), 3x3 zones (§2.6), and 4 edge bands (outer 5% rows/columns of each image's own footprint).

## 1. Decision rule (used by every metric)
- `d` = difference in the "worse for O" direction (defined per metric), `τ` = tolerance, `[lo, hi]` = 95%
  bootstrap CI of d (B = 500 resamples; paired star metrics resample matched star pairs; tile metrics
  resample tiles; background-lattice metrics have negligible statistical error and use point estimates).
  `PASS` if hi ≤ τ; `FAIL` if lo > τ; `WARN` otherwise. Metrics without a CI: PASS if d ≤ τ, else FAIL/WARN
  as stated per metric (attribution-dependent ones become WARN when attribution is impossible, §9).
- "Better" gates (SNR): a gain G with CI [Glo, Ghi]: BETTER if Glo > 1 + μ, μ = 0.02.

## 2. Common frame: footprint, detection, matching, alignment, photometric scale
### 2.1 Sanitise and footprint
- Replace non-finite by NaN. Valid footprint `V_i`: finite pixels, excluding border rows/columns that are
  entirely NaN, exactly 0, or constant (each is counted and reported, §6.5). Erode `V_i` by 8 px
  (`scipy.ndimage.binary_erosion`, iterations=8). Report the bounding box and the pixel count of `V_i`.
### 2.2 Star detection (identical relative parameters on each native grid)
- Pass 1: `bkg = sep.Background(img, bw=bh=64, fw=fh=3)`; `sub = img − bkg.back()`; `sep.extract(sub,
  thresh=10, err=bkg.globalrms, segmentation_map=True)` → star mask `M0` = segmentation ≠ 0, dilated 3 px.
- Pass 2: `bkg = sep.Background(img, mask=M0, bw=bh=64, fw=fh=3)`; `sub = img − bkg.back()`;
  `sep.extract(sub, thresh=5.0, err=bkg.rms(), minarea=5, filter_kernel=default 3x3, filter_type='matched',
  deblend_nthresh=32, deblend_cont=0.005, clean=True, segmentation_map=True)`.
- Per source: centroid `sep.winpos(sub, x, y, sig=0.85·r50_med)` (r50_med from the 200 brightest
  unsaturated sources, first pass with sep x,y); `F10 = sep.sum_circle(sub, x, y, 10.0, bkgann=(15,22),
  subpix=5)` (local annulus sky); `F4`, `F1.5`, `F8` with the same annulus; `peak` = max of `sub` in the 3x3
  around the rounded centroid; `r50 = sep.flux_radius(sub, x, y, 12.0, 0.5, normflux=F12, subpix=5)`;
  sep `flag, a, b, theta`. Star mask `M_i` = segmentation ≠ 0 dilated 3 px, plus discs of r=25 px around
  sources with SNR ≥ 1000 and r=60 px around saturated sources (wings).
### 2.3 Saturation level and clean-star selection (per image; a star is clean only if clean in BOTH)
- `sat_i` = max of the 3x3-median-filtered image over `V_i` (rejects lone hot pixels; for P this is ≤ 1.0).
  Flat-top test: ≥ 3 pixels within 2% of `peak` in the 5x5 core → saturated.
- Clean: S1 `peak < 0.5·sat_i` and not flat-topped; S2 isolated: no other detection within 12 px, none with
  flux > 10% of F10 within 20 px (cKDTree); S3 extract `flag & 0x03 == 0` (MERGED|TRUNC), aperture
  `flag & 0x30 == 0` (APER_TRUNC|APER_HASMASKED) and `a/b < 2`; S4 stellar: `|r50 − median r50(B1..B2)|
  < 0.25·median`; S5 ≥ 32 px inside `V` on both grids; S6 `SNR_P ≥ 10` where `SNR = F4/σ_F4` (σ_F4 from
  empty apertures, §3.4).
- Brightness bins by `SNR_P`: B1 ≥ 1000, B2 [300,1000), B3 [100,300), B4 [30,100), B5 [10,30). PSF metrics
  use B1–B3 (merge adjacent bins if N < 50); photometry uses all; faint-star SNR uses B5.
### 2.4 Geometric alignment (star matching; no astroalign)
1. Take the 300 brightest unsaturated, unflagged stars of each image. For each star form triangles with its
   6 nearest neighbours; invariant = (b/a, c/a) with sides a ≥ b ≥ c, plus handedness sign. Match
   invariants with a cKDTree (tolerance 0.01); each triangle match votes for 3 star pairs; keep pairs with
   ≥ 3 votes.
2. RANSAC similarity transform (scale, rotation, tx, ty; 2-pair samples, inlier radius 1.0 px, 2000
   iterations) → LSQ affine (6 params) on inliers.
3. Match the full catalogues through the affine (mutual nearest neighbour, radius 1.5 px) → refit affine;
   if residual RMS > 0.05 px, fit a projective (8 params) and then a 2nd-order polynomial (12 params); keep
   the simplest model whose RMS is within 10% of the best. Final one-to-one match radius 1.0 px.
4. Report: model, parameters, N_matched, residual RMS and the 3x3-zone median residual vectors (a zone-
   dependent residual pattern is itself a distortion/registration diagnostic for §7). Gate: RMS ≤ 0.10 px
   and N_matched ≥ 200, else the report is INCONCLUSIVE for all star-paired and difference-map metrics.
- Common analysis region `Ω` = `V_P ∩ T(V_O)` on the P grid (also report `|Ω| / min(|V_P|,|T(V_O)|)`;
  gate ≥ 0.8). Native-grid measurements in O use `T^-1(Ω)`.
### 2.5 Photometric scale and pedestal
- `a = exp(median(ln F10_P − ln F10_O))` over clean matched stars with SNR_P ≥ 100 (annulus-subtracted
  fluxes, hence independent of b). Report MADN of the ln-ratio.
- `b = median over star-free tiles of (B_P(c) − a·B_O(T^-1 c))` where `B` are the large-box background
  maps of §5.1 and `c` are tile centres in Ω. Report a, b, and the §7.1 linearity slope s; if |s−1| > 0.01, a
  is brightness-dependent and the report says so (a stays defined at SNR ≥ 100).
### 2.6 Zones and resampling policy
- Zones: 3x3 equal rectangles of the bounding box of Ω (P grid). Stars/tiles measured on the O grid are
  assigned to zones by mapping their centres through T. Edge bands: outer 5% rows (top, bottom) and columns
  (left, right) of each image's OWN footprint (per-image metrics), reported separately from the zones.
- Why resampling biases noise/PSF: interpolation is a linear filter with kernel weights w summing to 1.
  White-noise variance is multiplied by Σw² < 1 for non-integer shifts (bilinear at half-pixel: 0.25 →
  σ halves; cubic/Lanczos-3 at half-pixel: ≈0.6 → σ −23%), lag-1 correlation appears (ρ1 up to ~0.5), and
  the PSF is convolved with the kernel (bilinear at half-pixel adds 0.25 px² variance per axis: FWHM 4.00
  → 4.17 px (+4%); cubic/Lanczos-3 ≈ +0.5% but add ringing). The bias depends on the sub-pixel phase,
  which varies across the field when T contains rotation, so it is spatially structured.
- Policy: (i) PSF, noise, autocorrelation, depth, halos, hot pixels, saturation, edge metrics are measured
  on each image's NATIVE grid (alignment is used only to pair stars/tiles/zones). (ii) Only pixel-level
  difference maps use resampling: `O'_T` = O' resampled onto the P grid with `scipy.ndimage.map_coordinates
  (order=3, mode='constant', cval=nan)`; `D = P − O'_T`. D is used for large-scale background differences
  (insensitive to interpolation), trail/artifact attribution and the STF visual diff — never for noise or
  FWHM. (iii) If a symmetric variant is wanted, resample BOTH with the same kernel onto the mid-way grid
  (T^½ each) so both carry approximately equal bias; still never derive σ1 or FWHM from resampled data.
  Binned noise at b ≥ 4 px (§3.3) is nearly invariant to interpolation and is the fair SNR yardstick.

## 3. Family 2 — Noise, SNR, depth (native grids, P units)
### 3.1 Per-pixel noise σ1
- Tiles: 64x64 px, inside eroded `V_i`, star-mask fraction ≤ 30%. Per tile: LSQ plane on unmasked pixels,
  residual r, `σ_tile = MADN(r)`. `σ1 = median(σ_tile)`; zone and edge-band medians; IQR/median across tiles
  (noise non-uniformity, LN artifacts); Spearman correlation of σ_tile² with tile median background (photon
  noise expects a positive correlation; a flat σ across a strong gradient suggests rescaling artefacts).
- Sensitivity: ~4000 tiles → statistical error 0.2%; systematic (masking, plane removal) ~1%. Report also
  `bkg.globalrms` as a cross-check. `σ1` is interpolation-dependent (§2.6): reported, not used for verdicts.
### 3.2 Noise autocorrelation
- On the same tile residuals, for lag k = 1..3 along x and y and the (1,1) diagonal, using unmasked pairs:
  `ρ_k = 1 − MADN(Δ_k)² / (2·σ_tile²)`, `Δ_k = r[i] − r[i+k]` (robust: Var(Δ_k) = 2σ²(1−ρ_k)). Median over
  tiles. Expected for Lanczos-3-registered integrations: ρ1 ≈ 0.2–0.4, ρ3 ≈ 0. ρ1 > 0.6 means over-smoothing
  or a soft interpolator; negative ρ1 means sharpening/ringing. Report; no verdict (see §3.3).
### 3.3 Effective (binned) noise — the fair noise metric
- For b ∈ {2, 4, 8, 16}: fill masked pixels with the tile plane value (r = 0), block-average b×b, drop
  blocks with > 25% masked pixels, re-tile (64 px native for b ≤ 4, 256 px native for b ≥ 8), plane-remove,
  `σ_b = median MADN`. Effective noise `σ_eff(b) = b·σ_b`; correlation factor `R(b) = σ_eff(b)/σ1` (=1 for
  white noise; typically 1.2–1.7 at b = 8 for resampled stacks; it plateaus once b exceeds the kernel
  support, which is why σ_eff(b ≥ 4) does not reward interpolation tricks while σ1 does).
- Noise gain `G_b = σ_eff,P(b) / (a·σ_eff,O(b))`. Because a is fitted from star fluxes, G_b is exactly the
  signal-normalised SNR ratio for a source of scale b (SNR_O/SNR_P for the same true source). Headline
  `G_8` (≈ 2 FWHM), supporting `G_4`, `G_16`; `G_1` reported only. CI by tile bootstrap.
- Verdict (SNR gate): BETTER if G_8,lo > 1.02 and G_4,lo > 1.00. NOT WORSE under §1 with d = 1 − G_8,
  τ = 0.02 (PASS iff G_8,lo ≥ 0.98; FAIL iff G_8,hi < 0.98; WARN between). A gain
  must be accompanied by FWHM PASS (§4) and flux preservation (§3.5, §6.1) — otherwise it is smoothing or
  signal loss and the SNR gate is voided (status WARN with reason).
### 3.4 Photometric depth (empty-aperture noise)
- 3000 random positions in Ω (drawn on the P grid, mapped by T^-1 for O), rejecting apertures overlapping
  the star mask. `F_empty = sep.sum_circle(sub, x, y, 4.0, bkgann=(8,12))`; `σ_F4 = MADN(F_empty)`
  (automatically includes correlation and local-sky error). Same with r = 1.5, 8, 10 px for §2.3/§6.
- `F_lim(5σ) = 5·σ_F4`; `Δm = −2.5·log10(a·σ_F4,O / σ_F4,P)` (positive = O deeper). §1 with d = −Δm,
  τ = 0.02 (not worse); BETTER if Δm_lo ≥ +0.02. CI by bootstrap over apertures. Must agree with G_8 within ±0.03 mag.
### 3.5 Faint-star SNR and flux preservation
- Matched clean B5 (and B4) stars: `SNR_i = F4_i/σ_F4,i`; `ρ_SNR = median(SNR_O/SNR_P)` (paired); must
  equal G_8 within ±0.03, else the noise model or the flux scale is inconsistent (WARN).
- `q_faint = median(a·F10_O/F10_P)` in B5: τ |q_faint − 1| ≤ 0.02. Loss (< 0.98) = over-rejection or
  background over-subtraction of faint sources in O; > 1.02 → the same in P (attribute via §9).
### 3.6 Extended low-surface-brightness (LSB) SNR
- On the P grid: `S_P = blockmean_8(P − B_P^1024)` where `B^1024` is a star-masked `sep.Background`
  with bw=bh=1024, fw=fh=3 (preserves ≲1000 px objects); `σ8_P = σ_eff,P(8)/8`. LSB mask: blocks with
  `2·σ8_P ≤ S_P ≤ 10·σ8_P`, ≥ 16 native px from the (block-max propagated) star mask; require ≥ 500 blocks,
  else metric N/A. Map the mask through T^-1 to O (block centres) and compute `S_O'` identically on O.
- `signal_i = median S_i` over the mask, `SNR_LSB,i = signal_i/σ8_i` (P units). Flux ratio
  `q_LSB = signal_O'/signal_P`: τ |q_LSB − 1| ≤ 0.05 (sensitive to residual background differences of
  ~0.1σ1: report the correlation of (S_O' − S_P) with ΔB from §5.2; if |corr| > 0.5 the deviation is a
  background-model difference, not a flux loss). `SNR_LSB` ratio: consistent with G_8 (±0.05).

## 4. Family 1 — Resolution / PSF (native grids, same matched clean stars, per bin B1–B3, per zone)
- FWHM_hlr = 2·r50 (exact for Gaussian: r50 = FWHM/2). Measured with `sep.flux_radius` (§2.2).
- FWHM_mom: windowed second moments (SExtractor XWIN-style): Gaussian window σ_w = 2·r50 (wider than the
  winpos window on purpose, to limit truncation bias), 5 iterations, on `sub` in a 25x25 stamp with local
  annulus sky; moments `M = [[x2, xy],[xy, y2]]`; deconvolve the window:
  `M_true = (M^-1 − I/σ_w²)^-1`; `FWHM_mom = 2.3548·sqrt((λ1+λ2)/2)`; ellipticity `e = (λ1−λ2)/(λ1+λ2)`,
  e-vector `(e1, e2) = e·(cos 2θ, sin 2θ)`.
- Sharpness (core concentration): `C = F1.5/F10` (less phase-sensitive than peak/flux; for FWHM 4 px,
  C ≈ 0.32). Also `peak/F10` as an informational value.
- Estimators are paired: per star compute `Δ = (FWHM_O − FWHM_P)/FWHM_P`; report median per bin and zone.
  With ≥ 100 stars/bin the median has SE ≈ 1.25·5%/√N ≈ 0.6% → sensitivity ~1.5% (95%).
- τ: FWHM (both estimators, bins B1–B3): d = median Δ, τ = +0.02 (0.08 px at 4 px); worst zone τ = +0.04.
  Ellipticity: d = median(e_O − e_P), τ = +0.01. Coherent mean e-vector per zone |⟨e⟩|: ≤ 0.03 absolute
  (INFO unless O exceeds P's by > 0.01 → WARN). Sharpness: d = median (C_P − C_O)/C_P, τ = 0.03.
- Selection pitfalls: saturated cores flatten r50 (excluded by S1); blends inflate moments (S2/S3); galaxies
  (S4); faint bins bias r50 low with noise (hence B1–B3 only); phase differences between grids average
  out over ≥ 100 stars, never compare a single star.

## 5. Family 3 — Background flatness (P units, yardstick σ_ref = σ1,P)
### 5.1 Background models
- `B_i = sep.Background(img_i, mask=M_i, bw=bh=256, fw=fh=5).back()` on each native grid; `B_O' = a·B_O + b`
  sampled on the P grid on a 32-px lattice through T (bilinear; the maps are smooth). All following
  statistics are computed on the lattice restricted to Ω (≈ 25k points; estimator noise ≈ 0.005σ1).
- Low-order surface `P2(B)`: 2-D polynomial of total degree 2 (6 coefficients), LSQ with 3 rounds of
  3-MADN clipping. High-order residual `H_i = B_i − P2(B_i)`.
### 5.2 Metrics (each in units of σ_ref)
- Gradient carried: `PtP_B,i = p99 − p1 of (B_i − median)` — INFO only: it is mostly the LN/registration
  reference's own sky gradient, legitimately different between pipelines (different references).
- High-order structure: `RMS_H,i`, `PtP_H,i = p99 − p1 of H_i` (robust peak-to-peak). Real nebulosity
  enters both images equally (same σ_ref yardstick, so lower noise in O is not penalised).
- Difference: `ΔB = B_P − B_O'`; `ΔH = H_P − H_O'`. `PtP_ΔB,lo` (P2 part of ΔB: reference-choice effect,
  INFO), `RMS_ΔH`, `PtP_ΔH` (structure present in one master only → needs attribution, §9).
- Axis-aligned residual power (artefact signature: banding, tiles, edge roll-off are axis-aligned; nebulosity
  is not): row profile `p_y = median_x H_i`, column profile `p_x`; remove a 1-D quadratic; report
  `RMS_axis,i = sqrt(RMS(p_x)² + RMS(p_y)²)` and `max|p|`.
- Edge roll-off: for each of the 4 bands of each image's own footprint, `E_band = median over band of
  (B_i − P2_interior)` where `P2_interior` is fitted on the inner 90%; sign kept (negative = roll-off).
### 5.3 Visibility criterion and tolerances
- After PI's auto-STF (§5.4) the display slope at the sky level is ≈ 3/(16·2.8) ≈ 0.067 display units per
  σ1, i.e. ≈ 17 8-bit levels per σ1. A large-area (≫ 32 px) luminance step of ≥ 5 levels (≈ 2%, the Weber
  threshold at that display level) is perceptible → `k = 5/17 ≈ 0.3`. Rule: a background structure of
  amplitude A (in σ_ref, at scale s px) is visible iff `A > max(0.3, 3·R(s)/s)`; for the 256-px background
  scale the second term is negligible, so k = 0.3 σ_ref.
- τ (d = O − P unless stated): `RMS_H`: τ = 0.05; `PtP_H`: PASS if PtP_H,O ≤ max(PtP_H,P + 0.10, 0.30);
  `RMS_axis`: τ = 0.05 and `max|p|_O ≤ max(max|p|_P + 0.10, 0.30)`; `E_band`: PASS if
  |E_O| ≤ max(|E_P| + 0.10, 0.30) for all 4 bands; `PtP_ΔH < 0.30` ⇒ the two backgrounds are visually
  indistinguishable (PASS regardless of the above); otherwise attribute via §9: FAIL if attributed to O,
  WARN if inconclusive. PtP_B and PtP_ΔB,lo: INFO (WARN if PtP_B,O > 2·PtP_B,P and > 3σ_ref).
### 5.4 STF-invariant comparison (PixInsight ScreenTransferFunction auto-stretch)
- `MTF(m, x) = (m − 1)·x / ((2m − 1)·x − m)`, x ∈ [0,1]. With image median `med` and `madn = 1.4826·MAD`
  (star-free pixels, over Ω): `c0 = max(0, med − 2.8·madn)`, `c1 = 1` (P units), `m = MTF(0.25, (med − c0)
  /(c1 − c0))` (this maps the sky to 0.25 because MTF is symmetric in m and its output), and the stretched
  image `S = MTF(m, clip((x − c0)/(c1 − c0), 0, 1))`.
- Variant A (common STF): compute (c0, m) from P; apply to P and to O' (P units). Variant B (own STF): compute
  (c0, m) per image on its own scale (any c1 ≥ the image maximum: for sky-level values the stretch slope
  is 3/(16·(med − c0)), independent of c1, so the result is invariant to a and b by construction). Both
  variants must agree to within 1 level for the sky region; otherwise a/b are wrong.
- Metrics on S (display units, 255 levels): zone medians of S over star-free pixels → spread
  `Z_i = 255·(max − min)` over the 9 zones; visible-structure fraction `f_vis,i` = fraction of 32-px blocks
  of (S − P2(S)) with |value| > 0.02 (≈ 5 levels), star-masked; sky width `w_i = MADN(S)` (INFO, equals
  noise in display units). τ: `Z_O ≤ Z_P + 2` levels; `f_vis,O ≤ f_vis,P + 0.02`.

## 6. Family 4 — Artefacts (native grids unless stated)
### 6.1 Rejection over-aggressiveness (core clipping)
- Core fraction `c = F1.5/F10` per clean star; per image self-consistency `Δc_i = median c(B1) − median
  c(B3)` (expected ≈ 0; < −0.03 ⇒ bright cores clipped by pixel rejection in that image — no truth needed).
  Paired: `d = median (c_P − c_O)/c_P` in B1, τ = 0.03. Cross-check with the flux-ratio droop of §7.1.
### 6.2 Hot/cold pixel residuals
- `z = (sub − med8(sub))/σ1` with `med8` = median of the 8 neighbours (`ndimage.median_filter` with a 3x3
  footprint whose centre is False), star-masked. Spike: `|z| > 6`, 8-connected component size ≤ 2, and the
  neighbours' max |sub| < 0.3·|sub_pixel| (rejects PSF-like bumps). Counts `n_hot`, `n_cold` per Mpx of Ω.
  τ: `n_O ≤ 1.2·n_P + 2` per Mpx; absolute WARN if n_O > 20 per Mpx.
### 6.3 Satellite/airplane trail residuals (discrete Radon on the 4x-binned residual)
- `r4 = blockmean_4((img − B_i)·(1 − M_i)) / σ_4`; weights `w4 = blockmean_4(1 − M_i)`. For θ = 0..179° in
  1° steps: `ndimage.rotate(r4, θ, order=1, reshape=True)` and the same for w4; column sums `p_θ(u)`,
  `n_θ(u) = Σ_v w4_θ(u, v)`; `z_θ(u) = p_θ(u)/sqrt(n_θ(u))`, then standardise z by its global MADN
  (absorbs residual correlation). Trail = local maximum with `z ≥ 8`, width ≤ 3 binned px, `n ≥ 100` binned
  px (≥ 400 native px long). Report (θ, u, z, length) per detection. Also run on D (P grid): a trail present
  in only one master appears with a definite sign → attribution. τ: no trail in O without a counterpart
  (within 2°, 5 binned px) in P → otherwise FAIL; N_trail,O ≤ N_trail,P.
### 6.4 Halos and ringing around bright stars
- 60 brightest clean stars with `0.1·sat < peak < 0.5·sat`; radial profile of `sub` (sky from the
  [30, 40] px annulus median) in 0.5-px annuli to r = 24 px, normalised by the Gaussian-equivalent peak
  `peak_eq = F10 / (2π·(r50/1.1774)²)`; median-stack the 60 profiles. Ringing depth
  `D_peak = min over r ∈ [1.25, 4]·FWHM of the normalised stack`; significance on the un-normalised
  stack (P units): `D_sig = min profile(r) / (1.25·σ1·R(4)/sqrt(60·n_r))`, n_r = pixels per annulus.
  Wing/halo fraction `W = 1 − F(r = 2·FWHM)/F10`.
- τ: `D_peak,O ≥ D_peak,P − 0.002` (not > 0.2% of peak deeper); `D_sig,O ≥ −3` unless `D_sig,P < −3`;
  `W_O ≤ W_P + 0.01`. Also the difference of the stacked profiles in [0.5, 3]·FWHM (INFO plot).
### 6.5 Cosmetic, border and NaN checks (per image, own footprint)
- Counts of NaN/Inf/exact-zero pixels inside the bounding box of `V_i`; number of all-zero/NaN/constant
  border rows and columns; `σ1` in each edge band vs interior: `σ_band/σ_interior`. τ: NaN/Inf = 0; zero
  rows/cols = 0 (≤ P's → WARN, > P's → FAIL); `σ_band/σ_interior ≤ 1.10` (a noisier border = too little
  overlap left in by autocrop).

## 7. Families 5 and 6 — Photometric fidelity and geometry
### 7.1 Linearity, saturation, flat-field residual
- Linearity: matched clean stars B1–B4; Theil–Sen (`scipy.stats.theilslopes`) of `log10 F10_P` vs
  `log10 F10_O` → slope s, intercept c. Per-bin ratio `q_bin = median(a·F10_O/F10_P)` for B1..B5. Scatter:
  per bin `MADN(log10 F10_P − s·log10 F10_O − c)` vs expected `median sqrt((σ_F10,P/F_P)² + (σ_F10,O/F_O)²)
  /ln10`; excess `X = observed/expected`. τ: |s − 1| ≤ 0.01; |q_bin − 1| ≤ 0.01 for B1–B4 (B5: 0.02, §3.5);
  X ≤ 1.5 (WARN ≤ 2, FAIL beyond). A droop of q at B1 with Δc (§6.1) negative in one image identifies the
  clipping image.
- Saturation: `sat_i`, flat-top counts `n_sat,i`; in P units `sat_O' = a·sat_O + b`. τ: `sat_O' ≥ 0.98·sat_P`;
  `n_sat,O ≤ 1.05·n_sat,P + 2`. P clips at 1.0 (`n(peak ≥ 0.999)` reported).
- Flat/normalisation residual: `q_i = a·F10_O/F10_P − 1` for clean stars SNR_P ≥ 100; robust P2 fit of q vs
  (x_P, y_P); `PtP_q` of the surface over Ω (percent) and zone medians. A position-dependent ratio means a
  multiplicative field error (flat residual or LN scale field) in at least one image. τ: PtP_q ≤ 2% PASS;
  2–5% WARN; > 5% FAIL only if attributed to O (§9), else WARN. Sensitivity ~0.5% with ≥ 500 stars.
### 7.2 Geometry
- Registration accuracy proxy. If registered (or raw calibrated) single frames are available: measure
  `FWHM_frame` with the same r50 estimator on the same bins; `FWHM_stack,expected = sqrt(Σ w_f FWHM_f² / Σ w_f)`
  with the pipeline's weights (or uniform); registration blur `ε_i = sqrt(max(FWHM_i² − FWHM_expected², 0))
  /2.3548` px (RMS per-axis). τ: ε_O ≤ ε_P + 0.05 px; absolute ε ≤ 0.15 px PASS, ≤ 0.3 WARN. Without frames:
  the coherent e-vector per zone (§4) and the zone pattern of the alignment residuals (§2.4).
- Cross-channel alignment (within each pipeline, all filters registered to one reference so T = identity):
  for pairs (L,R), (L,G), (L,B), (R,G), (G,B): matched clean stars (mutual NN within 1 px, SNR ≥ 50 in both),
  offsets `(dx, dy)` from winpos centroids; `median offset`, robust `RMS = sqrt(MADN(dx)² + MADN(dy)²)`,
  3x3-zone median offsets, and a similarity fit (scale−1, rotation) as INFO (differential refraction/filter
  thickness). Justification of 0.1 px: for σ_PSF = 1.7 px the max slope is 0.36·peak/px, so 0.1 px shifts a
  pixel by 3.6% of the peak — the colour-fringe visibility limit on bright stars. τ: |global median| ≤ 0.05,
  RMS ≤ 0.10, every |zone median| ≤ 0.10 px (WARN ≤ 0.20, FAIL beyond) and O ≤ P + 0.02 px for each.

## 8. Family 7 — Verdict table and overall rules
- Table rows (one per metric, per filter): family, metric, P, O', d, τ, CI, status, note. Metrics that take
  part in the verdict are exactly: FWHM_hlr, FWHM_mom, e, C (family 1); G_4, G_8, Δm, q_faint, q_LSB
  (family 2); RMS_H, PtP_H, RMS_axis, E_band, PtP_ΔH, Z, f_vis (family 3); Δc/d_c, n_hot/n_cold, trails,
  D_peak/D_sig/W, NaN/zero/edge-σ (family 4); s, q_bin, X, sat, PtP_q (family 5); ε or ⟨e⟩, cross-channel
  (family 6). Everything else is INFO.
- Gate 0 (validity): alignment RMS ≤ 0.10 px, N_matched ≥ 200, |Ω| ratio ≥ 0.8, |s − 1| ≤ 0.02, STF
  variants agree → else INCONCLUSIVE (report values anyway).
- NOT VISUALLY WORSE ⇔ no FAIL in families 1, 3, 4, 6 and ≤ 3 WARN across them.
- SCIENTIFICALLY CONSISTENT ⇔ no FAIL in family 5 (WARNs listed).
- BETTER IN SNR ⇔ G_8,lo > 1.02 and G_4,lo > 1.00 and Δm_lo ≥ +0.02 and q_faint, q_LSB PASS and FWHM PASS
  (gain not bought by smoothing or signal loss). The CI comes from the tile bootstrap; the margin exceeds the
  ~0.5% statistical and ~1% systematic uncertainty of σ_eff.
- Overall: BETTER if all three hold; EQUIVALENT if the first two hold and the G_8 CI contains 1 (or
  |G_8 − 1| ≤ 0.02); WORSE if any FAIL; INCONCLUSIVE otherwise. Per-filter verdicts, then the run verdict is
  the worst per-filter verdict; cross-channel (§7.2) is a run-level row.
- Trend score (for tracking across versions): margin `m_j = clip((τ_j − d_j)/τ_j, −2, 2)` per verdict
  metric; report `min_j m_j` (worst) and `median_j m_j`. Not used for PASS/FAIL.

## 9. Pitfalls and attribution ("different" ≠ "worse")
- Common star list: never compare per-image catalogues at different thresholds; all star statistics use the
  matched clean list of §2.3, bins defined by SNR_P for both images.
- Pedestal/scale: only σ_ref (background amplitudes) or star flux (photometry) are yardsticks; a and b are
  reported and every dimensional number is in P units.
- Saturation: P clips at 1.0 and its integration may flatten cores below 1.0; S1 uses 0.5·sat and the
  flat-top test so both images exclude the same stars; saturation itself is scored in §7.1 only.
- Gradients vs real nebulosity: per-image absolute measures use H (P2 removed) and the same σ_ref, so
  common real structure cancels in the comparison; PtP_ΔH isolates structure that only one master has; the
  low-order difference is attributed to the reference choice (INFO). Lower noise in O never inflates its
  background metrics because the yardstick is P's σ.
- Different footprints: all comparisons in Ω; per-image edge metrics are reported on each own footprint.
- Interpolation: never use σ1, ρ1 or resampled data for verdicts (§2.6); the fair quantities are σ_eff(b≥4),
  empty-aperture depth, and native-grid FWHM.
- Attribution when the masters differ (which one is closer to the truth?), in order of preference:
  1. Truth reference T1: plain median (no rejection, no weights) of ≥ 20 registered calibrated single frames
     from either pipeline, photometrically matched to each master exactly as in §2.5. Compare the high-order
     backgrounds `H_master − H_T1`, the flux-ratio fields `q(master, T1)` and trail maps; the master with the
     larger residual is the flawed one. T1 is noisier (~1.25×) and unrejected, so it is used only for
     background, flat-field and artefact attribution, never for noise/PSF.
  2. Split halves T2: integrate odd and even frames separately with our pipeline; structure present in both
     halves is real (or a deterministic flaw), structure in one half only is a rejection/trail artefact; the
     half-difference also gives an empirical σ_eff(b) to validate §3.3.
  3. Physical priors without extra data: (a) lower σ_eff(b ≥ 4) with preserved fluxes and FWHM is better,
     unconditionally; (b) smaller FWHM with equal flux and no ringing is better; (c) axis-aligned or
     rectangular background structure (RMS_axis, block edges at 64/256-px multiples) is an artefact;
     (d) fewer spikes/trails is better; (e) the image whose own peak/flux or core-fraction relation is not
     self-consistent across brightness is the one clipping cores; (f) a flux-ratio field PtP_q > 2% cannot
     be attributed without T1 → WARN.

## 10. Implementation notes

The current evaluator does not implement every gate in this standard. It lists
common-footprint area, own/common STF agreement and LSB flux as unmeasured,
and therefore does not certify EQUIVALENT/BETTER. Failed measured gates still
produce WORSE. See [tool coverage](../tools/validation/README.md).
- Script `evaluate_masters.py --ours O_L.fits --pi P_L.fits [--ours-rgb O_R,O_G,O_B --pi-rgb ...]
  [--frames DIR] --out report/`. Outputs `report/<filter>.json` (every value, CI, status, a, b, T, star
  counts) and `report/summary.md` (the §8 table). Save the matched star table and the trail list as CSV.
- Cost plan (26 Mpx, float32; float64 only inside tiles/stamps): two `sep.Background` + `sep.extract` per
  image ≈ 3 s each; tile statistics via reshaping to (ny, 64, nx, 64) blocks, plane fits with a fixed design
  matrix per tile (`lstsq` on ~4000 tiles ≈ 1 s); binning by reshape-mean; aperture photometry on ≤ 10k
  stars ≈ 1 s; `map_coordinates` resampling ≈ 8 s; 180 rotations of the 1.6 Mpx `r4` ≈ 5 s per image;
  bootstrap on precomputed per-star/per-tile arrays ≈ 2 s. Total ≈ 60–90 s per filter pair.
- Fix all random draws (empty apertures, RANSAC, bootstrap) with seed 0; log every parameter (tile size,
  box sizes, thresholds, radii) in the JSON so results are reproducible across versions.
