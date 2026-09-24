# Licensing and attribution

## Original project code

Original Ultra-Fast WBPP source code, documentation, and project-owned artwork
are licensed under the standard [MIT License](../LICENSE). Existing third-party
notices, including the Astroalign-derived triangle bootstrap, remain in place.
The previous GPL licensing of earlier revisions is part of the project history;
this change does not revoke licenses already granted for those revisions.

Commercial use, modification, and redistribution of the MIT code are permitted.
If you distribute copies or substantial portions, include the original copyright
notice and the complete MIT permission notice. An application's third-party
notices or legal information are customary places to include them.
[NOTICE](../NOTICE) provides suggested attribution wording; it is not an extra
advertising requirement. Using this software to process images does not, by
itself, require credit or a watermark on those images. Research citation is
appreciated, not an additional condition of the MIT license.

## Provenance review boundary

The local continuation review found one author and committer across the eight visible commits and no additional co-author trailers. The identified Astroalign adaptation retains its upstream MIT notice. This supports the recorded provenance but does not independently prove ownership of initial imports or every uncommitted change. Before publishing the relicensed candidate, the copyright holder must confirm authority to relicense all project-owned portions; third-party code and documents retain their original terms.

## No GPL library in the bundled engine

Earlier versions imported the GPL-3.0 [`xisf`](https://github.com/sergio-dr/xisf)
package to read XISF containers, which made the combined worker a GPLv3
distribution. XISF files are now read and written by the project's own
`lightframeqc.xisf` module (MIT; validated value-for-value against the
previous reader on PixInsight-written masters, calibrated and registered
frames and on every codec/shuffle/sample-format combination), so the Python
runtime imports no GPL-licensed code.

SEP carries LGPL-3.0-or-later requirements, which the frozen engine satisfies as
a separately replaceable shared library with its notices preserved; the LGPLv3
refers to the GPLv3 text, which the repository keeps for that reason. Windows
bundles ship a modified SEP (`1.4.1+ufwbpp.1`): the modification is the patch in
[`packaging/patches`](../packaging/patches) applied to the unmodified 1.4.1
source distribution by `scripts/build_sep_wheel.py`, which is how the
corresponding source of that build is provided. Separately
installed solvers and catalog data retain their own terms. See
[third-party notices](../THIRD_PARTY_NOTICES.md). The repository includes the
[GPLv3 text](../LICENSES/GPL-3.0.txt) and the
[Astroalign MIT notice](../LICENSES/astroalign-MIT.txt); these are not a complete
license bundle for every transitive binary dependency.

Before publicly distributing a binary, its release must include an inventory
of the exact bundled components, their complete license texts and notices,
and applicable corresponding source/build material. An SBOM, a source archive
of this repository alone, or changing the top-level LICENSE is not sufficient.
See the [release checklist](release-process.md).

Authoritative references: [OSI MIT text](https://opensource.org/license/mit),
[GNU license compatibility guidance](https://www.gnu.org/licenses/license-compatibility.en.html),
and the [XISF 1.0 specification](https://pixinsight.com/doc/docs/XISF-1.0-spec/XISF-1.0-spec.html).
