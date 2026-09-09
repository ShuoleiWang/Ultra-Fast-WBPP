from __future__ import annotations

import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
TAURI_ROOT = REPOSITORY / "apps" / "desktop" / "src-tauri"


def test_legal_resources_are_mapped_from_canonical_root_files() -> None:
    config = json.loads((TAURI_ROOT / "tauri.conf.json").read_text(encoding="utf-8"))
    bundle = config["bundle"]
    expected = {
        "../../../LICENSE": "legal/LICENSE",
        "../../../NOTICE": "legal/NOTICE",
        "../../../THIRD_PARTY_NOTICES.md": "legal/THIRD_PARTY_NOTICES.md",
    }

    assert isinstance(bundle["resources"], dict)
    for source, destination in expected.items():
        assert bundle["resources"][source] == destination
        source_path = TAURI_ROOT / source
        assert not source_path.is_symlink()
        resolved_source = source_path.resolve(strict=True)
        assert resolved_source == REPOSITORY / Path(source).name
        assert resolved_source.is_file()

    assert bundle["licenseFile"] == "../../../LICENSE"
    assert (TAURI_ROOT / bundle["licenseFile"]).resolve(strict=True) == REPOSITORY / "LICENSE"
    assert bundle["resources"]["resources/openastroflow-worker"] == (
        "resources/openastroflow-worker"
    )


def test_macos_local_and_ci_builds_share_one_prerelease_script() -> None:
    package = json.loads(
        (REPOSITORY / "apps" / "desktop" / "package.json").read_text(encoding="utf-8")
    )
    command = package["scripts"]["tauri:build:macos-prerelease"]

    assert command == (
        "tauri build --bundles app --config "
        "src-tauri/tauri.prerelease.conf.json"
    )
    makefile = (REPOSITORY / "Makefile").read_text(encoding="utf-8")
    workflow = (REPOSITORY / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert "DESKTOP_BUILD_SCRIPT := tauri:build:macos-prerelease" in makefile
    assert "npm run tauri:build:macos-prerelease" in workflow


def test_desktop_sidecar_uses_one_fresh_release_native_chain_locally_and_in_ci() -> None:
    makefile = (REPOSITORY / "Makefile").read_text(encoding="utf-8")
    workflow = (REPOSITORY / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "native-release-configure:" in makefile
    assert "native-release-build: native-release-configure" in makefile
    assert "native-release-test: native-release-build" in makefile
    assert "native-release-install: native-release-test" in makefile
    assert "desktop-sidecar: native-release-install" in makefile
    assert "-DCMAKE_BUILD_TYPE=Release" in makefile
    assert "--build $(NATIVE_RELEASE_BUILD) --config Release" in makefile
    assert "ctest --test-dir $(NATIVE_RELEASE_BUILD) -C Release" in makefile

    install_body = makefile.split("native-release-install: native-release-test", 1)[1]
    install_body = install_body.split("\ndesktop-sidecar:", 1)[0]
    removal = install_body.index("$(CMAKE) -E rm -f")
    installation = install_body.index("$(CMAKE) --install")
    assert removal < installation
    for library_name in (
        "libopenastroflow_native.dylib",
        "libopenastroflow_native.so",
        "openastroflow_native.dll",
    ):
        assert f"$(NATIVE_RUNTIME_DIR)/{library_name}" in install_body

    assert "run: make native-release-install" in workflow
    assert "cmake -S engine/native -B build/native-release" not in workflow


def test_tag_workflow_only_prepares_a_draft_release() -> None:
    workflow = (REPOSITORY / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    release_command = workflow.split('gh release create "${GITHUB_REF_NAME}"', 1)[1]
    assert "--draft" in release_command
    assert "--prerelease" in release_command
    assert "--verify-tag" in release_command
