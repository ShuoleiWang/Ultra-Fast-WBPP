# Security policy

## Supported versions

This project is in the 0.x development stage. Security fixes target `main` and the latest prerelease; older prereleases do not have a maintenance guarantee. There is no stable support series yet.

## Reporting a vulnerability

Use the repository's **Security → Report a vulnerability** form when private vulnerability reporting is enabled. Maintainers must enable this channel before publishing a release. If the form is unavailable, open an issue requesting a private contact without including exploit details or affected private files.

Include the affected version or commit, operating system, minimal reproduction, expected impact, and whether original frames or existing destinations can be modified. Do not disclose path traversal, file overwrite, unsafe archive extraction, command injection, or signature/integrity bypass details in a public issue before coordinated disclosure.

## Security model

Ultra-Fast WBPP treats acquisition frames as immutable. A user-selected external solver or drizzle executable runs with that user's ordinary operating-system authority and is therefore inside the user's trust boundary; its output files, status markers, and reported metadata are not trusted and are independently validated. Process groups, a reduced environment, private staging, timeouts, and no-shell argument vectors are containment measures, not an operating-system filesystem or network sandbox. Ultra-Fast WBPP does not claim to prevent a malicious executable from reading user files or using the network.

The scientific sidecar is trusted application code, but its receipts are still checked for schema, file identity and result validity by the controller. Release candidates package it in a manifest-attested immutable directory. Ad-hoc macOS signing checks integrity; it does not establish a publisher identity or provide notarization. Stable distribution additionally requires platform signing, a complete bundle inventory and third-party notices; see the [release gates](docs/release-process.md).

Final publication creates a new destination and refuses to replace existing results. Shareable solver receipts omit absolute paths, exact command arguments and raw process output, but local run diagnostics may contain acquisition paths and source identities: review them before sharing. Catalog downloads require a versioned manifest with source, size, hash and provider terms. There is no online solver or image-upload backend.

Astrometry.net indexes are not bundled because the provider's redistribution terms have not been resolved for this project. The catalog manager downloads only after the user views the provider notice and submits the exact versioned acceptance ID, and it verifies byte count and SHA-256 before create-only installation. This acknowledgement is not a license grant from Ultra-Fast WBPP.

No software can replace backups. Keep original Light/Flat/Dark/Bias files on separate storage before running any preprocessing application.
