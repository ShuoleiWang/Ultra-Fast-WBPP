# Offline solver catalogs

Astrometry.net's executable and its indexes are separate prerequisites. Installing `solve-field` proves that the program can start; it does not prove that the angular scale of a new image is covered. Ultra-Fast WBPP never downloads indexes during startup, import, planning, or a processing run.

## 1. Inspect the checked choices

```bash
ultra-fast-wbpp catalog list
```

The output includes the provider-terms URL, versioned acceptance ID, exact storage, quad scale, derived image-FOV range, and current installed-by-size count. The complete checked 4107–4112 Tycho-2 set is 349,692,480 bytes (333.5 MiB). Keep at least twice that free while downloading the whole set because a resumable temporary file and a verified final file can briefly coexist.

Astrometry.net's [official guide](https://astrometry.net/doc/readme.html#getting-index-files) recommends indexes whose quads span roughly 10%–100% of image width. It gives 4107–4109 as the worked selection for a one-degree image. Ultra-Fast WBPP can apply this rule to a known horizontal FOV:

```bash
ultra-fast-wbpp catalog install astrometry-net-4107-4112 \
  --field-of-view 1.0 \
  --accept-provider-terms astrometry-net-index-data-2026-09 \
  --progress-json
```

Use the acceptance ID printed by your installed version of `catalog list`, not one copied indefinitely from this page. The ID changes only when the provider notice itself changes. The checked v1 manifest and its `astrometry-net-index-data-2026-09` acceptance ID retain their original OpenAstroFlow-era wording as compatibility and evidence identities; a product-brand change does not silently invalidate an already accepted notice or an installed-set receipt. Supplying the ID means that you reviewed the linked provider notice and accept responsibility for the direct provider download; it is not a license grant from Ultra-Fast WBPP. The current provider documentation says index files have separate conditions but does not give Ultra-Fast WBPP a sound basis to claim redistribution rights, so the files remain unbundled and the manifest says `provider-specific-unresolved`.

To install every checked wide-field scale instead, omit `--field-of-view`. To select exact files, repeat `--artifact index-4108.fits`. `--artifact` and `--field-of-view` are mutually exclusive.

For the catalog refit of a solved field's distortion (and for mosaic panels), install the Gaia DR2 LITE index 5205 as well. It has 48 HEALPix tiles, 587 MiB, and about six times the stars of index 4107. The desktop offers it under *Advanced options and solver setup* once the Tycho-2 set is installed; from the command line:

```bash
ultra-fast-wbpp catalog install astrometry-net-5205-lite \
  --accept-provider-terms astrometry-net-index-data-2026-09 \
  --progress-json
```

## 2. Resume or verify

An interrupted command can be repeated unchanged. Ultra-Fast WBPP resumes only when the partial file's manifest digest, exact URL, expected size and SHA-256 still match, and a strong HTTP ETag binds the `Range`/`If-Range` response to the same provider object. A strict `206 Content-Range`, unchanged ETag and exact total are required. Without a strong ETag, or if a provider ignores the range request and returns `200`, the application safely restarts its own temporary file instead of appending potentially incompatible bytes.

Every artifact URL must be HTTPS and match a manifest origin allowlist. Redirects are refused rather than followed, URL credentials and IP/localhost origins are rejected, and normal TLS hostname verification protects the named provider origin. `Content-Length` is treated only as an early consistency check; a streaming hard byte limit and final SHA-256 remain authoritative. No index is published until its exact byte count and SHA-256 pass. Publication is create-only; an existing wrong file is reported and never overwritten. To hash-check an existing directory without writing anything:

```bash
ultra-fast-wbpp catalog verify astrometry-net-4107-4112
```

If you downloaded the checked files yourself and want Ultra-Fast WBPP to generate the solver config and receipt only after all hashes pass:

```bash
ultra-fast-wbpp catalog verify astrometry-net-4107-4112 --configure
```

The default managed directory is `~/.ultra-fast-wbpp/catalogs/astrometry-net` on macOS/Linux and `%LOCALAPPDATA%\Ultra-Fast-WBPP\catalogs\astrometry-net` on Windows. An installation made under the project's former name (`~/.openastroflow`, `%LOCALAPPDATA%\OpenAstroFlow`) is used in place while the new directory does not exist; the engine never moves it. `UFWBPP_DATA_DIR` or `--catalog-dir` selects another location. The generated `astrometry.cfg` is discovered automatically; `UFWBPP_ASTROMETRY_CONFIG` selects a specific config.

## 3. Confirm runtime readiness

```bash
ultra-fast-wbpp catalog doctor
ultra-fast-wbpp doctor
```

The first command re-hashes checked index files (each command is a new process, which reads every byte) and validates the generated config. The second probes the complete engine. Neither command claims universal sky/scale coverage: coverage for a final master becomes proven only when that image solves and passes the match/RMS/WCS gates.

Alongside `astrometry.cfg`, installation creates an `installed-set-<catalog>-<identity>.json`. It contains only relative index paths plus their sizes and SHA-256 values, the checked manifest digest, config digest, citation, and an `installedSetIdentity`. Only the config digest ever changes, when a moved folder is re-pointed ([below](#moving-to-another-computer)); the index entries and the identity stay the same. Engine integrations can bind a solver's `.match` index evidence to real local files without trusting `INDEXID` alone:

```python
from ufwbpp.catalogs import installed_set_identity_for_solver_indexes

reference = installed_set_identity_for_solver_indexes(
    ["astrometry.net:index:4107:healpix:-1:hpnside:0"]
)
```

The strict Astrometry.net adapter (`astrometry-net-process-v5`) now performs this binding automatically. Before starting `solve-field`, it verifies the selected generated config, every installed-set receipt, and every index file reachable from that config, and refuses a config that names another folder than the one it is in (`CATALOG_CONFIG_RELOCATED`; `catalog doctor` re-points it, [below](#moving-to-another-computer)). After `.match` selects an `INDEXID`, it calls the binding API, which reads the selected index files in full again, requires one unambiguous concrete `index-<INDEXID>.fits`, and then compares the complete preflight snapshot (config, receipts and every index's size, SHA-256 and stat identity) with the files before publication.

An index file is hashed in full the first time a process uses it. Later uses in the same process, such as the next solve or verification of a run, take a fresh `lstat` and reuse that digest only while the device, inode, size, modification time and change time are unchanged and the file had last changed at least 2 s before it was hashed, so a later write carries a newer timestamp even on a filesystem with 2 s resolution. The config and the receipts are read in full every time. The one change the stat check cannot see is a rewrite in place that keeps the size and restores both timestamps; on Windows, where the change time is the creation time, restoring the modification time is enough. The index files a solution is bound to are never taken from the stat check. The scientific receipt records `installedSetIdentity`, checked manifest SHA-256, and each selected index's relative name, size, and SHA-256; it never records the local absolute `receiptPath`.

An unmanaged config/index directory can be used only by an explicitly diagnostic adapter instance. Its WCS and correspondence metrics may be inspected, but `catalogManaged=false` cannot satisfy a required final solve or the desktop publication gate. Replacing an index with even a byte-identical new file during solving is treated as drift.

## 4. Removal is plan-only

```bash
ultra-fast-wbpp catalog remove-plan astrometry-net-4107-4112
```

This prints exact candidate paths and reclaimable bytes but never removes anything. Artifacts shared with another checked manifest, `astrometry.cfg`, and unrelated indexes are retained. Deletion is intentionally left to a later GUI flow with a separate confirmation and shared-reference check.

## Moving to another computer

A copied catalog folder works on another computer without a new download. Install the solver program there (`brew install astrometry-net` on macOS, ASTAP with a star database on Windows), copy the whole catalog folder to the same place under that computer's data root (the default directory above, or under `UFWBPP_DATA_DIR`), then open the app or run `ultra-fast-wbpp catalog doctor`. Before the `doctor`, `run` and `run-project` commands and `catalog list`, `doctor`, `verify` and `install`, the engine finds a generated `astrometry.cfg` that still names the old folder, writes it again for the new one and re-binds the installed-set receipts to it; the index files are not touched and are still verified against their receipts before every solve. Each re-pointing is reported in one line on stderr. An `astrometry.cfg` edited by hand is never changed, and a failure is one line on stderr that leaves the solver unconfigured.

## Sources and scope

- [Astrometry.net index selection and configuration guide](https://astrometry.net/doc/readme.html#getting-index-files)
- [Official 4100-series directory and exact byte counts](https://data.astrometry.net/4100/)
- [Official 4100-series Tycho-2 build notes](https://data.astrometry.net/4100/README)
- [Provider-published MD5 checksums](https://data.astrometry.net/4100/md5sums.txt)
- [Astrometry.net paper and citation](https://arxiv.org/abs/0910.2233)

Ultra-Fast WBPP uses SHA-256 as its publication identity; the provider MD5 is retained only as an additional provenance cross-check. No real network download runs in the normal test suite.
