# Third-party notices

Original Ultra-Fast WBPP code is licensed under MIT. The Python runtime no longer imports any GPL-licensed library: XISF containers are read and written by the project's own `lightframeqc.xisf` module (MIT). See [licensing and attribution](docs/licensing.md). Binary distributions must reproduce the complete notices and license texts for the exact dependency lockfiles and bundled artifacts used by that release; this source-level summary is not a substitute for the generated release notice bundle. No signed stable binary release exists yet, and the final onedir bundle/SBOM/license evidence remains a pre-release gate.

## Python scientific stack

- NumPy — BSD-3-Clause and bundled component notices.
- SciPy — BSD-3-Clause and bundled component notices.
- Astropy — BSD-3-Clause.
- SEP — LGPL-3.0-or-later. The LGPLv3 incorporates the GPLv3 by reference; the repository therefore keeps [LICENSES/GPL-3.0.txt](LICENSES/GPL-3.0.txt) for that notice. Windows bundles carry a modified build (`1.4.1+ufwbpp.1`); the modification is the patch under [packaging/patches](packaging/patches), applied to the unmodified 1.4.1 source distribution by `scripts/build_sep_wheel.py`.
- scikit-image — BSD family and MIT component notices.
- Astroalign — MIT. The adapted triangle bootstrap retains its upstream copyright and permission notice; a complete copy is in [LICENSES/astroalign-MIT.txt](LICENSES/astroalign-MIT.txt).
- Pillow — MIT-CMU.
- `reproject`, Dask, and Shapely — BSD-3-Clause; Zarr — MIT. Shapely wheels also bundle GEOS under LGPL-2.1. These optional mosaic components and their transitive dependencies retain their own terms; inventory the exact distributed artifacts and include their complete notices rather than treating the whole dependency closure as BSD-licensed.
- PyInstaller bootloader and build tooling — GPL-2.0-or-later with the PyInstaller bootloader exception, which allows the frozen engine to be distributed under the project's own terms.
- OpenSSL 3 — Apache-2.0. The macOS 14 arm64 compatibility overlay is bound to an exact official Homebrew bottle manifest and blob digest.
- mpdecimal — BSD-2-Clause. The macOS 14 arm64 compatibility overlay is bound to an exact official Homebrew bottle manifest and blob digest.

## Desktop and control plane

- Tauri — Apache-2.0 and MIT dual license.
- React — MIT.
- Rust crates and npm packages — see the release SBOM and lockfiles.

## Optional external workers

ASTAP (MPL-2.0) and Astrometry.net (GPL-3.0-or-later) are separate optional programs. They are not copied into the source tree. A release may bundle one only after a dedicated binary-and-data license audit.

Catalog databases and index files are not included in the source repository or current bundle plan. Each downloadable catalog has its own manifest with provider, source URL, exact version, SHA-256, license status, citation, and installed scope. A program license does not automatically grant redistribution rights for its catalog data. Astrometry.net index redistribution terms remain unresolved for Ultra-Fast WBPP, so the catalog manager requires an explicit versioned provider-notice acknowledgement and performs a direct checked download into the user's managed data directory; this acknowledgement is not a license grant.

## Community documents

The Code of Conduct is adapted from Contributor Covenant 2.1 under CC BY 4.0 and retains its upstream attribution, adaptation notice, and license references in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). The original-code MIT license does not replace upstream document licenses.

## PixInsight independence

Despite its name, Ultra-Fast WBPP is not PixInsight WeightedBatchPreprocessing and does not contain or redistribute PixInsight, PCL, WeightedBatchPreprocessing, ImageSolver, their scripts, source code, icons, binaries, or XPSD catalog data. PixInsight may be used by individual developers under their own licenses as an external scientific comparison oracle. Ultra-Fast WBPP is not affiliated with or endorsed by PixInsight or Pleiades Astrophoto, and it does not claim algorithmic or pixel equivalence with PixInsight/WBPP.

Retained real Light/Flat/Dark/Bias, MasterFlat and integrated-master fixtures used for local validation are user data and are not redistributed. Public validation receipts deliberately omit source paths, source hashes, pixels, processing histories and raw solver logs.
