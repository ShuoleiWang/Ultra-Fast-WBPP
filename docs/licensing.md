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

## The current bundled worker is not MIT-only

The Python packages currently require [`xisf`](https://github.com/sergio-dr/xisf),
whose upstream license is GPL-3.0. The frozen scientific worker imports and
bundles that library. The project's MIT license does not relicense `xisf` or
remove the obligations of distributing that combined worker.

Treat redistribution of the current combined worker as GPLv3 distribution:
preserve applicable notices and provide the complete corresponding source and
build instructions for the distributed version. MIT portions remain available
under MIT, while the combined program must satisfy the GPL. Do not advertise
the current installer or scientific stack as an exclusively permissive product.
A permissive-only distribution would require replacing GPL dependencies and
auditing the resulting dependency tree; that work has not been completed.

SEP also carries LGPL-3.0-or-later requirements. Separately installed solvers
and catalog data retain their own terms. See [third-party notices](../THIRD_PARTY_NOTICES.md).
The repository includes the [GPLv3 text](../LICENSES/GPL-3.0.txt) and the
[Astroalign MIT notice](../LICENSES/astroalign-MIT.txt); these are not a complete
license bundle for every transitive binary dependency.

Before publicly distributing a binary, its release must include an inventory
of the exact bundled components, their complete license texts and notices,
and applicable corresponding source/build material. An SBOM, a source archive
of this repository alone, or changing the top-level LICENSE is not sufficient.
See the [release checklist](release-process.md).

Authoritative references: [OSI MIT text](https://opensource.org/license/mit),
[GNU license compatibility guidance](https://www.gnu.org/licenses/license-compatibility.en.html),
and [upstream XISF](https://github.com/sergio-dr/xisf).
