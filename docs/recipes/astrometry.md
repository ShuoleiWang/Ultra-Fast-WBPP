# Astrometric solving

Ultra-Fast WBPP distinguishes an acquisition `SEED` from a verified `SOLVED` WCS. N.I.N.A. RA/Dec, focal length, pixel size, and orientation narrow the search, but they do not authorize final publication.

## Automatic N.I.N.A. hint handling

For every accepted light group, Ultra-Fast WBPP computes the expected horizontal field width from the actual image width and the N.I.N.A./FITS optical metadata:

`FOV = 2 × atan((widthPixels × effectivePixelSizeMicrons / 1000) / (2 × focalLengthMm))`

`FOCALLEN`/`FOCAL`/`FOCALLENGTH` and `XPIXSZ`/`YPIXSZ` (with the `PIXSIZE*` fallbacks) are supported. The result is calculated for each light and reduced to a robust median. The recorded assumption is that `XPIXSZ` and `YPIXSZ` describe the effective pixels of the stored, possibly binned image. Acquisition RA/Dec values are also reduced to a circular sky-coordinate consensus.

An explicit GUI/CLI hint is compared with that acquisition evidence. A field-width ratio in `0.7–1.3` and a center within the requested search radius are considered consistent. When they conflict, the frame-derived consensus wins and the rejected value, ratio/separation, selection, and provenance are written into the E2E receipt. A stale user hint therefore cannot silently constrain the solver to the wrong scale or sky position.

## Bounded Astrometry.net profile

The default large-frame profile uses `--downsample 4 --objs 500 --depth 10-500 --pixel-error 2` when the longest image axis is at least 5000 pixels. It adapts to downsample 2 for 2500–4999 pixels and 1 for smaller images. These are solve-only arguments: executable discovery and capability probes remain isolated `--version` and `--help` calls. A required final solve accepts only the catalog-manager generated config covered by the same installed-set receipt as its indexes. The config and every managed index are hash/stat checked before execution; the `.match` `INDEXID` is then bound to exact index bytes and the snapshot is verified again before publication.

The headless runtime discovers an app-managed config through the compatibility variable `OPENASTROFLOW_ASTROMETRY_CONFIG` (or `ASTROMETRY_NET_CONFIG`), then the standard Ultra-Fast WBPP data directories. Embedders can pass `config_path` and an immutable `AstrometryNetSolveProfile` directly. Arbitrary command strings are not accepted as a profile, and the config path is never added to probe arguments.

Use the [offline catalog recipe](offline-solver-catalogs.md) to inspect compatible scales, explicitly accept the versioned provider notice, download/resume checked artifacts, generate `astrometry.cfg`, and create an immutable installed-set receipt. Nothing is fetched automatically by a solve.

When a center or scale hint exists, its first attempt receives 25% of the total timeout, capped at 30 seconds. If that attempt times out, exits without solving, or produces no exact solved marker, the remaining budget is used for a more conservative attempt with gentler downsampling, up to 1000 sources/depth, and no center or scale constraints. The process-group timeout is still the hard boundary; fallback does not extend the configured total budget.

For a roughly 1.11-degree field, install a user-managed Astrometry.net index set covering that scale; indexes 4107–4112 were present in the M3 Pro validation environment, and the accepted solution used index 4107. One index file does not prove complete coverage. See the [sanitized real-fixture validation](../../benchmarks/results/m3-pro-astrometry-net-real-fixture-20260901.json) for the measured adapter and quality-gate result. Astrometry.net's `.match` `INDEXID` identifies a logical index; the strict adapter binds it to the installed-set identity, checked manifest, relative index filename, exact size, and SHA-256.

The strict final backends are a user-installed local Astrometry.net `solve-field` plus app-managed checked indexes, and ASTAP (`astap_cli`, with its own star database) whose solution the engine verifies against the same managed indexes: the engine detects stars on the final master, projects the index stars through ASTAP's WCS, matches them one to one and records the residuals, the matched count, the parity, the correspondence table and the byte-bound index identities exactly as the astrometry.net adapter does. The recipe thresholds apply to both; `backend: auto` prefers `solve-field` when both are present. There is no built-in solver and no online upload fallback, and without the managed index set ASTAP stays diagnostic-only.

Discovering an executable and validating its CLI options proves only that the adapter can invoke it. ASTAP still needs a star database covering the requested field and scale; Astrometry.net still needs compatible index files. Coverage becomes proven only when that specific final image produces the required solved marker, WCS artifacts, and validated receipt.

A solution receipt records backend/version, managed installed-set/manifest/index identities, correspondence identity, matched-star count, RMS pixels/arcseconds, scale, parity, center, search seed, and corner/center round-trip checks. Exact argv, absolute source/config/staging paths, and raw solver output are excluded from the shareable receipt. Set `OPENASTROFLOW_SOLVER_DIAGNOSTIC_DIR` only when local troubleshooting logs are needed; those separate JSON logs contain redacted argv/output tails and must not be attached to a shared project without review. Recipe thresholds are evaluated by app-core rather than trusting an external process exit code. ASTAP currently has no independently verifiable catalog-byte/correspondence contract, so its result is diagnostic and cannot satisfy a strict required final solve.

Crop, Drizzle, RGB combination, and mosaic projection change geometry. The last geometry-changing stage must therefore be followed by a new solve or by a validated propagation whose residual gate is at least as strict. A failed solver leaves an explicitly `UNSOLVED` work artifact and the job incomplete; it never emits a green final-result card.
