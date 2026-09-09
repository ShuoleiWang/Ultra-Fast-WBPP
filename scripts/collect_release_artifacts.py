#!/usr/bin/env python3
"""Collect a host Tauri bundle and sidecar attestations with SHA-256 sums."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Sequence


class ReleaseCollectionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_roots(repository: Path) -> tuple[Path, ...]:
    return (
        repository / "target" / "release" / "bundle",
        repository / "apps" / "desktop" / "src-tauri" / "target" / "release" / "bundle",
    )


def _product_identity(repository: Path) -> tuple[str, str]:
    try:
        package = json.loads(
            (repository / "apps" / "desktop" / "package.json").read_text(
                encoding="utf-8"
            )
        )
        tauri = json.loads(
            (repository / "apps" / "desktop" / "src-tauri" / "tauri.conf.json").read_text(
                encoding="utf-8"
            )
        )
        product_name = tauri["productName"]
        version = package["version"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ReleaseCollectionError("desktop product metadata is invalid") from error
    if (
        not isinstance(product_name, str)
        or product_name != "Ultra-Fast WBPP"
        or not isinstance(version, str)
        or re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+-]*", version) is None
    ):
        raise ReleaseCollectionError("desktop product identity is non-canonical")
    return product_name, version


def _bundle_files(repository: Path, target: str) -> list[Path]:
    suffixes = {".dmg", ".app", ".msi", ".exe", ".nsis", ".deb", ".rpm", ".appimage"}
    populated: list[tuple[Path, list[Path]]] = []
    for root in _bundle_roots(repository):
        if not root.is_dir():
            continue
        candidates: list[Path] = []
        for path in root.rglob("*"):
            if (path.is_file() or path.is_symlink()) and (
                path.suffix.casefold() in suffixes
                or any(part.casefold().endswith(".app") for part in path.parts)
            ):
                # Individual files inside an .app are not release artifacts;
                # macOS distribution is the generated DMG.
                if any(part.casefold().endswith(".app") for part in path.parts):
                    continue
                candidates.append(path)
        if candidates:
            populated.append(
                (root, sorted(set(candidates), key=lambda path: str(path)))
            )
    if not populated:
        return []
    if len(populated) != 1:
        raise ReleaseCollectionError(
            "multiple Tauri bundle roots contain distributable artifacts"
        )
    root, candidates = populated[0]
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            raise ReleaseCollectionError(
                f"bundle artifact is not a regular non-symlink file: {path.name}"
            )
    product_name, version = _product_identity(repository)
    if target.endswith("apple-darwin"):
        expected = root / "dmg" / f"Ultra-Fast-WBPP_{version}_aarch64.dmg"
        if candidates != [expected]:
            names = ", ".join(path.name for path in candidates)
            raise ReleaseCollectionError(
                "macOS bundle tree must contain only the exact renamed DMG; found: "
                + names
            )
        return [expected]
    if target.endswith("windows-msvc"):
        msi = [
            path
            for path in candidates
            if path.parent == root / "msi" and path.suffix.casefold() == ".msi"
        ]
        nsis = [
            path
            for path in candidates
            if path.parent == root / "nsis" and path.suffix.casefold() == ".exe"
        ]
        selected = [*msi, *nsis]
        if (
            len(msi) != 1
            or len(nsis) != 1
            or set(selected) != set(candidates)
            or any("openastroflow" in path.name.casefold() for path in selected)
            or any(version not in path.name for path in selected)
        ):
            names = ", ".join(path.name for path in candidates)
            raise ReleaseCollectionError(
                "Windows bundle tree must contain one MSI and one NSIS installer "
                "for the renamed product; found: " + names
            )
        return sorted(selected, key=lambda path: str(path))
    brand_tokens = {
        product_name.casefold(),
        product_name.casefold().replace(" ", "-"),
        product_name.casefold().replace(" ", "_"),
    }
    selected = [
        path
        for path in candidates
        if path.suffix.casefold() in {".deb", ".rpm", ".appimage"}
        and version in path.name
        and any(token in path.name.casefold() for token in brand_tokens)
    ]
    if set(selected) != set(candidates):
        names = ", ".join(path.name for path in candidates)
        raise ReleaseCollectionError(
            "bundle tree contains stale, wrong-target, or foreign artifacts: " + names
        )
    return selected


def collect(
    repository: Path,
    output: Path,
    target: str,
    *,
    sidecar_root: Path | None = None,
    attestation_root: Path | None = None,
    source_commit: str | None = None,
) -> tuple[Path, ...]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"release output must be new: {output}")
    bundles = _bundle_files(repository, target)
    if not bundles:
        raise ReleaseCollectionError("Tauri emitted no distributable bundle")
    sidecar_root = sidecar_root or repository / "build" / "sidecars"
    sidecar_manifest = sidecar_root / f"openastroflow-worker-{target}.manifest.json"
    if not sidecar_manifest.is_file() or sidecar_manifest.is_symlink():
        raise ReleaseCollectionError("sidecar manifest is missing")
    try:
        manifest = json.loads(sidecar_manifest.read_text(encoding="utf-8"))
        runtime_name = manifest["runtime"]["directoryName"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ReleaseCollectionError("sidecar manifest is invalid") from error
    sidecar_runtime = sidecar_root / runtime_name
    if not sidecar_runtime.is_dir() or sidecar_runtime.is_symlink():
        raise ReleaseCollectionError("sidecar runtime tree is missing")
    attestation_root = attestation_root or repository / "build" / "bundle-attestations"
    attestation_source = attestation_root / f"openastroflow-worker-{target}.bundled.manifest.json"
    if attestation_source.exists() and (
        attestation_source.is_symlink() or not attestation_source.is_file()
    ):
        raise ReleaseCollectionError("bundled runtime attestation is not a regular file")
    if source_commit is not None and not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source_commit):
        raise ReleaseCollectionError("source commit must be a lowercase Git object ID")

    output.mkdir(parents=True, exist_ok=False)
    copied: list[Path] = []
    try:
        sources = [*bundles, sidecar_manifest]
        if attestation_source.is_file():
            sources.append(attestation_source)
        for source in sources:
            destination = output / source.name
            if destination.exists():
                raise ReleaseCollectionError(f"duplicate release basename: {source.name}")
            shutil.copy2(source, destination)
            copied.append(destination)
        metadata = output / f"release-metadata-{target}.json"
        bundled_attestation = None
        if attestation_source.is_file():
            try:
                bundled_attestation = json.loads(
                    attestation_source.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ReleaseCollectionError("bundled runtime attestation is invalid") from error
        signature = (
            bundled_attestation.get("signature")
            if isinstance(bundled_attestation, dict)
            else None
        )
        metadata.write_text(
            json.dumps(
                {
                    "schemaVersion": 2,
                    "targetTriple": target,
                    "sourceCommit": source_commit,
                    "sourceCommitBound": source_commit is not None,
                    "signed": bool(
                        isinstance(signature, dict)
                        and signature.get("verified") is True
                        and signature.get("mode") == "developer-id"
                    ),
                    "releaseChannel": "prerelease",
                    "bundledRuntimeAttestation": (
                        {
                            "fileName": attestation_source.name,
                            "sha256": _sha256(attestation_source),
                            "treeSha256": bundled_attestation["runtime"]["treeSha256"],
                            "signature": signature,
                            "launch": bundled_attestation["launch"],
                        }
                        if isinstance(bundled_attestation, dict)
                        else None
                    ),
                    "artifacts": [
                        {"fileName": path.name, "sha256": _sha256(path), "sizeBytes": path.stat().st_size}
                        for path in copied
                    ],
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        copied.append(metadata)
        sums = output / f"SHA256SUMS-{target}"
        sums.write_text(
            "".join(f"{_sha256(path)}  {path.name}\n" for path in sorted(copied)),
            encoding="ascii",
        )
        copied.append(sums)
    except Exception:
        shutil.rmtree(output)
        raise
    return tuple(copied)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path)
    parser.add_argument("--attestation-root", type=Path)
    parser.add_argument("--source-commit")
    arguments = parser.parse_args(argv)
    try:
        files = collect(
            Path.cwd(),
            arguments.output,
            arguments.target,
            sidecar_root=arguments.sidecar_root,
            attestation_root=arguments.attestation_root,
            source_commit=arguments.source_commit,
        )
        print(json.dumps({"ok": True, "files": [path.name for path in files]}, sort_keys=True))
        return 0
    except (FileExistsError, OSError, ReleaseCollectionError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ReleaseCollectionError", "collect", "main"]
