from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.collect_release_artifacts import ReleaseCollectionError, collect


def _source_tree(root: Path) -> None:
    desktop = root / "apps" / "desktop"
    tauri = desktop / "src-tauri"
    tauri.mkdir(parents=True, exist_ok=True)
    (desktop / "package.json").write_text('{"version":"0.1.0"}', encoding="utf-8")
    (tauri / "tauri.conf.json").write_text(
        '{"productName":"Ultra-Fast WBPP"}', encoding="utf-8"
    )
    bundle = root / "target" / "release" / "bundle" / "dmg"
    bundle.mkdir(parents=True)
    (bundle / "Ultra-Fast-WBPP_0.1.0_aarch64.dmg").write_bytes(b"dmg")
    sidecars = root / "build" / "sidecars"
    sidecars.mkdir(parents=True)
    artifact = sidecars / "openastroflow-worker-aarch64-apple-darwin"
    artifact.mkdir()
    (sidecars / f"{artifact.name}.manifest.json").write_text(
        json.dumps(
            {
                "runtime": {
                    "directoryName": artifact.name,
                }
            }
        ),
        encoding="utf-8",
    )


def test_collects_bundle_manifest_and_deterministic_sums(tmp_path: Path) -> None:
    _source_tree(tmp_path)
    output = tmp_path / "release"
    source_commit = "a" * 40
    files = collect(
        tmp_path,
        output,
        "aarch64-apple-darwin",
        source_commit=source_commit,
    )
    names = {path.name for path in files}
    assert names == {
        "Ultra-Fast-WBPP_0.1.0_aarch64.dmg",
        "openastroflow-worker-aarch64-apple-darwin.manifest.json",
        "SHA256SUMS-aarch64-apple-darwin",
        "release-metadata-aarch64-apple-darwin.json",
    }
    sums = (output / "SHA256SUMS-aarch64-apple-darwin").read_text(encoding="ascii")
    assert "Ultra-Fast-WBPP_0.1.0_aarch64.dmg" in sums
    assert "release-metadata-aarch64-apple-darwin.json" in sums
    metadata = json.loads(
        (output / "release-metadata-aarch64-apple-darwin.json").read_text()
    )
    assert metadata["schemaVersion"] == 2
    assert metadata["signed"] is False
    assert metadata["sourceCommit"] == source_commit
    assert metadata["sourceCommitBound"] is True


def test_fails_closed_without_bundle_and_never_replaces_output(tmp_path: Path) -> None:
    desktop = tmp_path / "apps" / "desktop"
    tauri = desktop / "src-tauri"
    tauri.mkdir(parents=True)
    (desktop / "package.json").write_text('{"version":"0.1.0"}', encoding="utf-8")
    (tauri / "tauri.conf.json").write_text(
        '{"productName":"Ultra-Fast WBPP"}', encoding="utf-8"
    )
    with pytest.raises(ReleaseCollectionError, match="no distributable"):
        collect(tmp_path, tmp_path / "out", "aarch64-apple-darwin")
    _source_tree(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(FileExistsError):
        collect(tmp_path, output, "aarch64-apple-darwin")
    with pytest.raises(ReleaseCollectionError, match="source commit"):
        collect(
            tmp_path,
            tmp_path / "bad-commit",
            "aarch64-apple-darwin",
            source_commit="not-a-git-object",
        )


@pytest.mark.parametrize(
    "foreign_name",
    (
        "OpenAstroFlow_0.1.0_aarch64.dmg",
        "Ultra-Fast-WBPP_0.0.9_aarch64.dmg",
        "Ultra-Fast-WBPP_0.1.0_x64-setup.exe",
    ),
)
def test_rejects_stale_foreign_or_wrong_target_bundle(
    tmp_path: Path, foreign_name: str
) -> None:
    _source_tree(tmp_path)
    foreign = tmp_path / "target" / "release" / "bundle" / "dmg" / foreign_name
    foreign.write_bytes(b"foreign")

    with pytest.raises(ReleaseCollectionError, match="exact renamed DMG"):
        collect(tmp_path, tmp_path / "release", "aarch64-apple-darwin")


def test_collects_one_windows_msi_and_one_nsis_without_guessing_names(
    tmp_path: Path,
) -> None:
    _source_tree(tmp_path)
    bundle = tmp_path / "target" / "release" / "bundle"
    (bundle / "dmg" / "Ultra-Fast-WBPP_0.1.0_aarch64.dmg").unlink()
    msi = bundle / "msi" / "Ultra-Fast WBPP_0.1.0_x64_en-US.msi"
    nsis = bundle / "nsis" / "Ultra-Fast WBPP_0.1.0_x64-setup.exe"
    msi.parent.mkdir()
    nsis.parent.mkdir()
    msi.write_bytes(b"msi")
    nsis.write_bytes(b"nsis")
    sidecars = tmp_path / "build" / "sidecars"
    runtime = sidecars / "openastroflow-worker-x86_64-pc-windows-msvc"
    runtime.mkdir()
    (sidecars / f"{runtime.name}.manifest.json").write_text(
        json.dumps({"runtime": {"directoryName": runtime.name}}), encoding="utf-8"
    )

    files = collect(tmp_path, tmp_path / "windows-release", "x86_64-pc-windows-msvc")
    names = {path.name for path in files}
    assert msi.name in names
    assert nsis.name in names


def test_rejects_ambiguous_or_old_windows_installers(tmp_path: Path) -> None:
    _source_tree(tmp_path)
    bundle = tmp_path / "target" / "release" / "bundle"
    (bundle / "dmg" / "Ultra-Fast-WBPP_0.1.0_aarch64.dmg").unlink()
    msi = bundle / "msi"
    nsis = bundle / "nsis"
    msi.mkdir()
    nsis.mkdir()
    (msi / "Ultra-Fast WBPP_0.1.0_x64_en-US.msi").write_bytes(b"msi")
    (nsis / "OpenAstroFlow_0.1.0_x64-setup.exe").write_bytes(b"old")

    with pytest.raises(ReleaseCollectionError, match="renamed product"):
        collect(tmp_path, tmp_path / "windows-release", "x86_64-pc-windows-msvc")


def test_collects_post_sign_runtime_attestation_into_release_metadata(tmp_path: Path) -> None:
    _source_tree(tmp_path)
    attestations = tmp_path / "build" / "bundle-attestations"
    attestations.mkdir()
    attestation = attestations / "openastroflow-worker-aarch64-apple-darwin.bundled.manifest.json"
    attestation.write_text(
        json.dumps(
            {
                "runtime": {"treeSha256": "a" * 64},
                "signature": {
                    "verified": True,
                    "mode": "ad-hoc",
                    "hardenedRuntime": True,
                },
                "launch": {
                    "versionSeconds": 1.0,
                    "handshakeSeconds": 1.1,
                    "catalogListSeconds": 1.2,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "release-with-attestation"
    files = collect(tmp_path, output, "aarch64-apple-darwin")
    assert attestation.name in {path.name for path in files}
    metadata = json.loads(
        (output / "release-metadata-aarch64-apple-darwin.json").read_text()
    )
    assert metadata["signed"] is False
    assert metadata["bundledRuntimeAttestation"]["treeSha256"] == "a" * 64
    assert metadata["bundledRuntimeAttestation"]["signature"]["mode"] == "ad-hoc"
