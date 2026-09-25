# Dependency and secret audit

Snapshot date: 2026-09-05. This is a local uncommitted-candidate audit; hosted checks remain pending until an authorized public commit exists.

## Source and secrets

`gitleaks 8.30.1` scanned 13.68 MB of the current source tree and separately scanned all eight local commits (3.46 MB), with generated runtime, catalog, build, dependency and virtual-environment paths excluded. Both scans found no leaks.

The repository additionally rejects raw FITS/XISF/XDRZ/XNML data, `.env`, keys, signing identities, catalogs, local caches, and build artifacts by default. Before a release, the staged Git index is scanned again so an ignored working-tree file cannot affect the result.

## JavaScript

The online `npm audit --audit-level=high` check reported zero vulnerabilities for `apps/desktop/package-lock.json`; an offline production-only check also reported zero.

## Python

The online `pip-audit 2.10.1 --local` check reported no known vulnerabilities. The three editable first-party distributions (`light-frame-qc`, `ufwbpp`, and `ufwbpp-registration`) are not published on PyPI and were explicitly skipped by the index-backed service, so their source tests, public-tree scan, review, and complete bundle SBOM remain separate gates.

## Rust

`cargo audit 0.22.2` loaded 1,235 RustSec advisories, scanned 487 locked crates, and found no vulnerability that causes a nonzero audit result. It reported 17 warning-only advisories for unmaintained GTK3/UNIC crates and one `glib 0.18.5` unsound iterator advisory (`RUSTSEC-2024-0429`). These packages are in Tauri/Wry's Linux WebKitGTK dependency graph, not the macOS WebKit or Windows WebView2 release targets.

Ultra-Fast WBPP therefore does not publish a Linux desktop bundle while that dependency chain remains. Linux is a source-build/development CI target only. macOS and Windows releases still rerun target-specific dependency audits, and the warning is not treated as a blanket waiver for future Tauri versions.

The repository files configure Dependabot, dependency review, CodeQL, and least-privilege workflow permissions. GitHub Dependency Graph, vulnerability alerts, secret scanning, push protection, and hosted results become evidence only after they are enabled and pass on the authorized public repository. The `glib 0.18.5` warning is documented as a target-specific risk because Linux desktop bundles are unsupported; it must be reconsidered before any Linux release or when Tauri moves to a patched GTK dependency line.

## macOS frozen-worker entitlement

The release design uses a PyInstaller onedir scientific worker embedded under application Resources so no code is extracted on launch. The build manifest and post-sign bundle attestation have distinct tree digests; the latter must be computed from the actual `.app`. Release discovery ignores environment/current/adjacent executable overrides and verifies every bundled entry, size, SHA-256, executable bit, total count/bytes, absence of extras, and the full tree digest before launch. The root `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` are mapped directly into `Resources/legal` and the attestation requires exact byte identity. Ad-hoc prereleases retain only `com.apple.security.cs.disable-library-validation`: ad-hoc code has no Developer-ID Team ID, so macOS cannot establish a common library-validation identity for the bundled Python/native libraries. The app does not request JIT, unsigned executable memory, network server, automation, or debugger entitlements. A stable release may remove this entitlement only after recursively signing the entire runtime with its protected identity and passing a signed-app launch gate. Current bundle measurements live only in the [bundle validation receipt](evidence/macos-arm64-renamed-bundle-20260905.json), which must be regenerated after every scientific or packaging change.

## Remaining release gates

- Generate and verify complete per-platform Python/PyInstaller locks with exact versions and SHA-256 hashes on native macOS arm64 and Windows x64 builders; stable release jobs must use `--require-hashes --only-binary=:all:` with no fallback to the broad development resolver.
- Generate CycloneDX/SPDX SBOMs for Rust, npm, Python, native libraries, optional workers, and catalogs.
- Reproduce complete third-party license texts from the exact binary environment.
- Verify macOS signing/notarization and Windows installer signing identities are provided only through protected release secrets.
- Run CodeQL, dependency review, staged gitleaks, package audits, and artifact checksum/signature verification on the release commit.
