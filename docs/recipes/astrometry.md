# Astrometric solving

Ultra-Fast WBPP distinguishes an acquisition `SEED` from a verified `SOLVED` WCS. N.I.N.A. RA/Dec, focal length, pixel size, and orientation narrow the search, but they do not authorize final publication.

## Automatic N.I.N.A. hint handling

For every accepted light group, Ultra-Fast WBPP computes the expected horizontal field width from the actual image width and the N.I.N.A./FITS optical metadata:

`FOV = 2 × atan((widthPixels × effectivePixelSizeMicrons / 1000) / (2 × focalLengthMm))`

`FOCALLEN`/`FOCAL`/`FOCALLENGTH` and `XPIXSZ`/`YPIXSZ` (with the `PIXSIZE*` fallbacks) are supported. The result is calculated for each light and reduced to a robust median. The recorded assumption is that `XPIXSZ` and `YPIXSZ` describe the effective pixels of the stored, possibly binned image. Acquisition RA/Dec values are also reduced to a circular sky-coordinate consensus.

An explicit GUI/CLI hint is compared with that acquisition evidence. A field-width ratio in `0.7–1.3` and a center within the requested search radius are considered consistent. When they conflict, the frame-derived consensus wins and the rejected value, ratio/separation, selection, and provenance are written into the E2E receipt. A stale user hint therefore cannot silently constrain the solver to the wrong scale or sky position.

## Bounded Astrometry.net profile

The default large-frame profile uses `--downsample 4 --objs 500 --depth 10-500 --pixel-error 2` when the longest image axis is at least 5000 pixels. It adapts to downsample 2 for 2500–4999 pixels and 1 for smaller images. These are solve-only arguments: executable discovery and capability probes remain isolated `--version` and `--help` calls. A required final solve accepts only the catalog-manager generated config covered by the same installed-set receipt as its indexes. The config and every managed index are hash/stat checked before execution; the `.match` `INDEXID` is then bound to exact index bytes and the snapshot is verified again before publication.

The headless runtime discovers an app-managed config through the compatibility variable `UFWBPP_ASTROMETRY_CONFIG` (or `ASTROMETRY_NET_CONFIG`), then the standard Ultra-Fast WBPP data directories. Embedders can pass `config_path` and an immutable `AstrometryNetSolveProfile` directly. Arbitrary command strings are not accepted as a profile, and the config path is never added to probe arguments.

Use the [offline catalog recipe](offline-solver-catalogs.md) to inspect compatible scales, explicitly accept the versioned provider notice, download/resume checked artifacts, generate `astrometry.cfg`, and create an immutable installed-set receipt. Nothing is fetched automatically by a solve.

When a center or scale hint exists, its first attempt receives 25% of the total timeout, capped at 30 seconds. If that attempt times out, exits without solving, or produces no exact solved marker, the remaining budget is used for a more conservative attempt with gentler downsampling, up to 1000 sources/depth, and no center or scale constraints. The process-group timeout is still the hard boundary; fallback does not extend the configured total budget.

For a roughly 1.11-degree field, install a user-managed Astrometry.net index set covering that scale; indexes 4107–4112 were present in the M3 Pro validation environment, and the accepted solution used index 4107. One index file does not prove complete coverage. See the [sanitized real-fixture validation](../evidence/m3-pro-astrometry-net-real-fixture-20260901.json) for the measured adapter and quality-gate result. Astrometry.net's `.match` `INDEXID` identifies a logical index; the strict adapter binds it to the installed-set identity, checked manifest, relative index filename, exact size, and SHA-256.

The strict final backends are a user-installed local Astrometry.net `solve-field` plus app-managed checked indexes, and ASTAP (`astap_cli`, with its own star database) whose solution the engine verifies against the same managed indexes: the engine detects stars on the final master, projects the index stars through ASTAP's WCS, matches them one to one and records the residuals, the matched count, the parity, the correspondence table and the byte-bound index identities exactly as the astrometry.net adapter does. The recipe thresholds apply to both; `backend: auto` prefers `solve-field` when both are present. There is no built-in solver and no online upload fallback, and without the managed index set ASTAP stays diagnostic-only.

Discovering an executable and validating its CLI options proves only that the adapter can invoke it. ASTAP still needs a star database covering the requested field and scale; Astrometry.net still needs compatible index files. Coverage becomes proven only when that specific final image produces the required solved marker, WCS artifacts, and validated receipt.

A solution receipt records backend/version, managed installed-set/manifest/index identities, correspondence identity, matched-star count, RMS pixels/arcseconds, scale, parity, center, search seed, and corner/center round-trip checks. Exact argv, absolute source/config/staging paths, and raw solver output are excluded from the shareable receipt. Set `UFWBPP_SOLVER_DIAGNOSTIC_DIR` only when local troubleshooting logs are needed; those separate JSON logs contain redacted argv/output tails and must not be attached to a shared project without review. Recipe thresholds are evaluated by the engine, and the desktop re-validates the solution receipt, rather than trusting an external process exit code. ASTAP's solution satisfies a strict required final solve only through the engine-side verification against the managed index stars described above; without the managed index set its result is diagnostic-only.

Crop, Drizzle, RGB combination, and mosaic projection change geometry. The last geometry-changing stage must therefore be followed by a new solve or by a validated propagation whose residual gate is at least as strict. A failed solver leaves an explicitly `UNSOLVED` work artifact and the job incomplete; it never emits a green final-result card.

## Catalog refit of the published solution

Astrometry.net reports its solution with a SIP order of 2. A fast corrected optical train has cubic distortion that order 2 cannot follow: on a 12-inch f/4 Newtonian with a coma corrector (1.1° field, 0.65″/px), the solver's WCS of an M31 master is off from Gaia DR3 by 1.53″ median and 2.64″ at the 90th percentile.

After every filter of a run has solved (and the same-grid solve is shared), the engine refits the grid's solution on the managed catalog. It uses the TAN+SIP fit a mosaic panel uses, with the order chosen by the BIC up to 3, on the L master when there is one and otherwise on the first filter by name. The refit replaces the solver's solution only when both of these hold:

- its residual on held-out stars (two-fold cross-validation: each half of the matched stars against the fit on the other half) is at most 0.8 of the solver's residual on the same stars, and within the 2.0″ gate;
- every master of the grid passes the engine's catalog verification with it, at the run's own match and RMS thresholds.

That verification is then recorded as each master's accepted solve. The headers carry `OAFWCSRF` (the SIP order) and `OAFWCSRA` (the refit's catalog RMS in arcseconds); the receipt's `astrometry.catalogRefinement` records the residuals, the matched stars and the catalog identity, or why the solver's solution was kept. Pixel values are untouched. Set `"catalogRefit": false` in the recipe's `solver` block to publish the solver's solution unchanged.

Install `astrometry-net-5205-lite` (the Gaia DR2 LITE index 5205, 48 HEALPix tiles, 587 MiB) beside the Tycho-2 set to make the refit work; see the [offline catalog recipe](offline-solver-catalogs.md). The refit needs a catalog dense enough to measure the distortion: at least 60 catalog stars in the field, twice the 30 matches it needs, since about half survive saturation, isolation and matching. The checked Tycho-2 set 4107–4112 holds about 50 stars in that M31 field, so the refit is skipped after a 0.3 s catalog query and the solver's solution is published as before. With a Gaia-density catalog the same master refits to order 3 on 1218 stars in 3.9 s; against the independent half of Gaia, the median residual is 0.058″ and the 90th percentile 0.22″. With the 5205 tiles installed, the managed catalog itself gives order 3 on 123 stars, 0.069″ median and 0.21″ at the 90th percentile against Gaia DR3.
