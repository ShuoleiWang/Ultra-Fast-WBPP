# Mixed-night import, screening and calibration

Drop acquisition folders together. The All / Light / Flat / Dark / Bias tabs are views of the same input set; changing a tab does not force the next import to that frame type. The table keeps Raw and Master separate and groups actual acquisition metadata and capture dates. Capture dates are not a claim about the observing-night boundary; Light QC uses its own observing-night/airmass analysis.

The desktop uses `mono-standard-v1`. Calibration matching starts after import. External-master camera settings which were not recorded remain unknown; they are neither copied from Lights nor replaced with zero. Known conflicting acquisition settings, incompatible geometry/filter/exposure and genuine missing dependencies remain blockers. Missing Bayer metadata follows the selected monochrome workflow; explicitly marked Bayer data is still blocked. Standard MasterDark inputs retain Bias by default, while explicit header declarations or an advanced override can identify a bias-subtracted Dark. File identities are checked internally without asking the user to approve hashes.

Advanced settings are for unusual calibration inputs. An override can change just one recorded field, Dark bias semantics or numeric units; blank fields mean no override. Light screening can run independently of a missing calibration file; final processing waits for required matches.

## Automatic Light screening

The desktop invokes the same pixel measurement, stellar registration and quality-gate implementation used by scientific execution. It measures images rather than trusting a filename's HFR alone.

Star candidates need at least three above-threshold pixels in SEP's unfiltered image (`minimum_source_support_pixels`), in addition to the existing five-pixel filtered detection area. This prevents a single hot pixel expanded by the detection kernel from behaving like a star. The original detections remain available for fragmented-trail analysis. Distributed, aligned chains of bright split detections provide independent trailing evidence before reference selection; their fragments cannot win the reference ranking merely by inflating source counts. A reference must also have independent geometric support.

Spatial transparency comparisons restore each frame's global photometric scale before constructing the shared bright envelope. Each candidate's global offset is then removed again for the local-dimming metric. This keeps an uneven, faint frame from contaminating the clear-sky baseline while preserving the separate global-transparency and airmass checks. These scientific changes use quality-policy evidence revision 3, invalidating earlier review approvals.

Review and production runs share a disposable cache of complete measured groups. Images are still read and content-checked, and quality gates and approvals are evaluated again. Identical measured cohorts can reuse stellar analysis; changed inputs, group membership, configuration, implementation or numerical-library versions invalidate that entry. The cache uses bounded JSON and falls back to calculation if unavailable. Set `OPENASTROFLOW_QC_CACHE_DIR=off` to disable it for comparisons. Diagnostic reports record measurement, analysis and gate times separately.

The astroalign 2.6.2 bootstrap reuses repeated star-coordinate and triangle-pair calculations while preserving its search, fits and tolerances; other versions use the public implementation. One seeded synthetic failed-match benchmark improved from 9.65 s to 5.12 s. This is a bootstrap microbenchmark, not a full-project speed claim.

| Evidence | Default handling |
|---|---|
| Strong coherent trailing / lost guiding | Exclude; strong supported cases are HARD_FAIL |
| Severe defocus / whole-night PSF degradation | Exclude for review; image FWHM also works when NINA HFR is absent |
| A different star field / failed geometric agreement | Exclude for review; a denser wrong frame needs independent support before it can serve as the reference |
| Thick or spatially variable cloud | Exclude when photometric/spatial evidence supports it; sparse fields now use a sufficiently populated coarse grid |
| Spatial obstruction | Exclude based on missing stars, boundary and background evidence |
| Normal brightness changes across nights or airmass | Brightness alone is not a hard rejection; legitimate controls are retained |

PASS participates in processing. REVIEW is excluded by default and can be explicitly approved after inspecting its preview; approval binds the current file, request and policy. HARD_FAIL cannot be approved. No input is moved or deleted. Screening is not guaranteed to distinguish every real cloud, focus or pointing condition; evidence-poor or uniformly bad batches may require review instead of an automatic pass.

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

Meridian flips are handled by the estimated source-to-reference transform, not by a separate image-rotation pass or by assuming that a mount-side tag proves an exact half-turn. Exact 180-degree transforms with integer translation use bounded pixel-copy reversal, with only floating-point roundoff tolerated across the full image corners. Real angular deviations, shear and subpixel dithering continue through a single configured resampling pass. The crop mask and recorded interpolation method follow the same classification. Ordinary warps retain bounded multi-frame CPU parallelism. The current affine pixel registration is not a claim of equivalence to PixInsight's optional thin-plate-spline distortion correction.

The desktop runtime clock starts when processing is requested, includes native launch time, and excludes the separate Review screening step. It shows HH:MM:SS while running and retains total time after completion, failure or confirmed cancellation. Channel changes do not reset it; a rejected cancellation does not imply that processing has stopped.

- Same-profile Lights from multiple dates can share compatible calibration inputs. The retained private 38-Light B fixture spans four capture dates and matches 20 Raw Flats and supplied Bias/Dark masters.
- A general library spanning different cameras, gains, readout settings or incompatible temperatures is not automatically partitioned into per-Light calibration choices. Unsupported mixtures are reported before execution.
- Same-filter Raw Flats are combined into one MasterFlat. Date alone cannot establish matching dust, rotation or optical-path state; choose a compatible calibration session when those changed. There is no automatic optical-state inference or per-session Flat mapping yet.
- Missing Bias/Flat temperature metadata is not fabricated or required by the standard workflow. Unknown Dark temperature is reported as unverified; a known incompatible temperature still blocks matching.
- Existing CLI recipes default to `strict-v1` for compatibility. Use [mono-standard.json](mono-standard.json) with `--recipe` to request the same conventions as the desktop. This is not a claim of PixInsight algorithmic equivalence, and this implementation still requires compatible Dark exposure rather than silently adding Dark scaling.

See [calibration details](calibration.md), [N.I.N.A. grouping](nina-mono.md), and the [validation matrix](../validation-matrix.md).
