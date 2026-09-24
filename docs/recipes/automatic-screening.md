# Mixed-night import, screening and calibration

Drop acquisition folders together. The All / Light / Flat / Dark / Bias tabs are views of the same input set; changing a tab does not force the next import to that frame type. The table keeps Raw and Master separate and groups actual acquisition metadata and capture dates. Capture dates are not a claim about the observing-night boundary; Light QC uses its own observing-night/airmass analysis.

The desktop uses `mono-standard-v1`. Calibration matching starts after import. External-master camera settings which were not recorded remain unknown; they are neither copied from Lights nor replaced with zero. Known conflicting acquisition settings, incompatible geometry/filter/exposure and genuine missing dependencies remain blockers. Missing Bayer metadata follows the selected monochrome workflow (a hash-bound confirmation can declare such frames as colour); Lights with a declared Bayer pattern are processed as one-shot colour ([recipe](osc-cfa.md)). Standard MasterDark inputs retain Bias by default, while explicit header declarations or an advanced override can identify a bias-subtracted Dark. File identities are checked internally without asking the user to approve hashes.

Advanced settings are for unusual calibration inputs. An override can change just one recorded field, Dark bias semantics or numeric units; blank fields mean no override. Light screening can run independently of a missing calibration file; final processing waits for required matches.

## Automatic Light screening (the legacy gate)

The quality gate described here is the *legacy gate*: the automatic admission that runs when a run carries no explicit selection. It remains the default of the command line (`selection.policy: legacy-gate`) and its evidence appears in the desktop's optional diagnostic table. The desktop requires [blink-style screening](blink-screening.md): the same measurements, plus per-channel flags with absolute cross-night criteria, a reference frame per channel and normalised previews, and the observer's frame-by-frame decisions sent to the run as an explicit selection. The gate's nightly-relative model cannot see a uniformly bad night — a moonlit or hazy night whose frames are all alike looks self-consistent to it — and that is the reason the blink flow exists; on the six-night NGC 6822 campaign it admitted 19 of the 23 frames of such a night and the luminance master carried the gradient (the numbers are on the [blink page](blink-screening.md)).

Screening is part of every run: the quality-control stage measures and gates every Light, the excluded frames are listed with the result (name, disposition, the gate's reasons and a bounded preview, from the receipt's `execution.screening` and the run's `qc/review/` previews), and nothing has to be clicked before starting. The desktop's screening review is optional: it invokes the same pixel measurement, stellar registration and quality-gate implementation used by scientific execution, so its verdicts match the run's, and it is where a REVIEW frame can be inspected and approved before processing. It measures images rather than trusting a filename's HFR alone. With an explicit selection the gate still runs and records its dispositions in `qc/manifest.json`; the selection replaces its admitted set, and every override is listed in the receipt.

Measurement and per-frame analysis run in spawned worker processes (`lightframeqc.parallel`; `LIGHTFRAMEQC_PARALLELISM=threads` forces threads). Every frame is computed by the same function either way, so the executor never changes a value; on eight cores the 67-frame NGC 7331 screening dropped from 22 s to 10 s.

Star candidates need at least three above-threshold pixels in SEP's unfiltered image (`minimum_source_support_pixels`), in addition to the existing five-pixel filtered detection area. This prevents a single hot pixel expanded by the detection kernel from behaving like a star. The original detections remain available for fragmented-trail analysis. Distributed, aligned chains of bright split detections provide independent trailing evidence before reference selection; their fragments cannot win the reference ranking merely by inflating source counts. A reference must also have independent geometric support.

Spatial transparency comparisons restore each frame's global photometric scale before constructing the shared bright envelope. Each candidate's global offset is then removed again for the local-dimming metric. This keeps an uneven, faint frame from contaminating the clear-sky baseline while preserving the separate global-transparency and airmass checks. These scientific changes use quality-policy evidence revision 3, invalidating earlier review approvals.

Nightly z-scores are bounded on tiny nights. The background, noise and focus anomalies compare a frame with its same-night peers as a robust z-score; on nights of one to three frames the peer spread is a few counts and a harmless difference became a z of 25 or −47. The scale of the z-score is now never smaller than 2 % of the night's centre, and a night with fewer than five peers also uses at least the whole cohort's spread; on the NGC 6822 campaign a −12.8 became −0.9 and a 25.3 became 2.1. This changes gate evidence, so the policy's evidence revision is 4: REVIEW approvals recorded under revision 3 are invalid and must be made again. The blink flags never use these z-scores; their night-outlier flag is taken from the gate codes only.

Review and production runs share a disposable cache of complete measured groups. Images are still read and content-checked, and quality gates and approvals are evaluated again. Identical measured cohorts can reuse stellar analysis; changed inputs, group membership, configuration, implementation or numerical-library versions invalidate that entry. The cache uses bounded JSON and falls back to calculation if unavailable. Set `UFWBPP_QC_CACHE_DIR=off` to disable it for comparisons. Diagnostic reports record measurement, analysis and gate times separately.

The astroalign 2.6.2 bootstrap reuses repeated star-coordinate and triangle-pair calculations while preserving its search, fits and tolerances; other versions use the public implementation. One seeded synthetic failed-match benchmark improved from 9.65 s to 5.12 s. This is a bootstrap microbenchmark, not a full-project speed claim.

| Evidence | Default handling |
|---|---|
| Strong coherent trailing / lost guiding | Exclude; strong supported cases are HARD_FAIL |
| Severe defocus / whole-night PSF degradation | Exclude for review; image FWHM also works when NINA HFR is absent |
| A different star field / failed geometric agreement | Exclude for review; a denser wrong frame needs independent support before it can serve as the reference |
| Thick or spatially variable cloud | Exclude when photometric/spatial evidence supports it; sparse fields now use a sufficiently populated coarse grid |
| Spatial obstruction | Exclude based on missing stars, boundary and background evidence |
| Normal brightness changes across nights or airmass | Brightness alone is not a hard rejection; legitimate controls are retained |

Without an explicit selection, PASS participates in processing. REVIEW is excluded by default and can be explicitly approved after inspecting its preview; approval binds the current file, request and policy. HARD_FAIL cannot be approved. No input is moved or deleted. Screening is not guaranteed to distinguish every real cloud, focus or pointing condition; evidence-poor batches may require review instead of an automatic pass, and a uniformly bad night passes it — blink the channel, or supply a [selection](blink-screening.md#the-selection-file-selection-v1).

Every target/filter panel needs at least two admitted Lights for registration. The desktop shows admitted counts before starting; execution checks them again immediately after screening. Insufficient-frame failures retain their per-frame QC manifest and previews, and report the affected panel and count instead of a generic registration error.

The 2026-09-05 pixel regressions cover 20 normal cross-night images plus six injected defects (trailing, defocus, wrong field, thick cloud, obstruction and patchy cloud). A separate regression covers whole-night defocus without HFR, and another uses a denser wrong field against a normal majority. These are reproducible synthetic scenes, not measured recall on a labeled real-world defect collection.

## Raw inputs become masters

| Input | Processing |
|---|---|
| Raw Bias | Robust integration into MasterBias |
| Raw Dark | Group by exposure and integrate, retaining its bias contribution |
| Raw Flat | Subtract matching exposure/temperature FlatDark when available, otherwise Bias; normalize individual Flats and robustly integrate |
| Existing Master | Validate identity, acquisition metadata and numeric-domain declarations, then reuse |

A raw MasterDark includes bias: subtracting it from a Light must not subtract Bias again. For an explicitly bias-subtracted MasterDark, the Light requires both Bias and Dark subtraction. A separate Bias is unnecessary when matching bias-inclusive Darks cover all raw targets; otherwise the engine still requires the necessary Bias subtraction. For example, Light 100 ADU, Bias 10 ADU and a bias-inclusive Dark 30 ADU gives 70 ADU before Flat division; a bias-subtracted Dark of 20 ADU gives the same `100 − 10 − 20 = 70` result.

Dark matching uses exposure and a temperature window; the current policy permits up to 3°C difference. Camera, gain, offset, geometry, binning, CFA and readout mode must agree; Flats additionally match filter. Unrecorded optional master acquisition fields do not block the standard workflow. Ambiguous Float FITS pixel units remain an actionable error; normal bounded XISF and integer FITS units are automatic. Unusual bias-subtracted Darks can be declared in advanced settings.

## Current limits

Meridian flips are handled by the estimated source-to-reference transform, not by a separate image-rotation pass or by assuming that a mount-side tag proves an exact half-turn. Exact 180-degree transforms with integer translation use bounded pixel-copy reversal, with only floating-point roundoff tolerated across the full image corners. Real angular deviations, shear and subpixel dithering continue through a single configured resampling pass. The crop mask and recorded interpolation method follow the same classification. Ordinary warps retain bounded multi-frame CPU parallelism. The current projective pixel registration is not a claim of equivalence to PixInsight's optional thin-plate-spline distortion correction.

The desktop runtime clock starts when processing is requested, includes native launch time, and excludes Blink review and diagnostic inspection. It shows HH:MM:SS while running and retains total time after completion, failure or confirmed cancellation. Channel changes do not reset it; a rejected cancellation does not imply that processing has stopped.

- Same-profile Lights from multiple dates can share compatible calibration inputs. The retained private 38-Light B fixture spans four capture dates and matches 20 Raw Flats and supplied Bias/Dark masters.
- A general library spanning different cameras, gains, readout settings or incompatible temperatures is not automatically partitioned into per-Light calibration choices. Unsupported mixtures are reported before execution.
- Same-filter Raw Flats are combined into one MasterFlat. Date alone cannot establish matching dust, rotation or optical-path state; choose a compatible calibration session when those changed. There is no automatic optical-state inference or per-session Flat mapping yet.
- Missing Bias/Flat temperature metadata is not fabricated or required by the standard workflow. Unknown Dark temperature is reported as unverified; a known incompatible temperature still blocks matching.
- Existing CLI recipes default to `strict-v1` for compatibility. Use [mono-standard.json](mono-standard.json) with `--recipe` to request the same conventions as the desktop. This is not a claim of PixInsight algorithmic equivalence, and this implementation still requires compatible Dark exposure rather than silently adding Dark scaling.

See [blink-style screening](blink-screening.md), [calibration details](calibration.md), [N.I.N.A. grouping](nina-mono.md), and the [validation matrix](../validation-matrix.md).
