from __future__ import annotations

import json
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
TAURI_ROOT = REPOSITORY / "apps" / "desktop" / "src-tauri"
TAURI_CLI_SCHEMA = (
    REPOSITORY / "apps" / "desktop" / "node_modules" / "@tauri-apps" / "cli" / "config.schema.json"
)


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

    # CI builds the same fresh Release chain through the one cross-platform
    # script (configure -> build -> ctest -> install), never ad-hoc cmake lines.
    assert "python scripts/build_native_runtime.py" in workflow
    assert "--build-dir build/native-release" in workflow
    assert "cmake -S engine/native -B build/native-release" not in workflow
    script = (REPOSITORY / "scripts" / "build_native_runtime.py").read_text(encoding="utf-8")
    assert '"-DCMAKE_BUILD_TYPE=Release"' in script
    assert '"--config", "Release"' in script
    assert '"-C", "Release", "--output-on-failure"' in script
    assert "remove_stale_libraries(RUNTIME_DIR)" in script
    for library_name in (
        "libopenastroflow_native.dylib",
        "libopenastroflow_native.so",
        "openastroflow_native.dll",
    ):
        assert f'"{library_name}"' in script
    ci_workflow = (REPOSITORY / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python scripts/build_native_runtime.py" in ci_workflow


def test_tag_workflow_only_prepares_a_draft_release() -> None:
    workflow = (REPOSITORY / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    release_command = workflow.split('gh release create "${GITHUB_REF_NAME}"', 1)[1]
    assert "--draft" in release_command
    assert "--prerelease" in release_command
    assert "--verify-tag" in release_command


def test_windows_installers_embed_webview2_and_install_per_user_without_downgrades() -> None:
    config = json.loads((TAURI_ROOT / "tauri.conf.json").read_text(encoding="utf-8"))
    windows = config["bundle"]["windows"]

    # A clean Windows machine may lack the WebView2 runtime; the embedded
    # bootstrapper installs it silently instead of depending on a download
    # inside the installer session.
    assert windows["webviewInstallMode"] == {"type": "embedBootstrapper", "silent": True}
    # Older builds must not replace a newer installation (MajorUpgrade rules).
    assert windows["allowDowngrades"] is False
    # The WiX upgrade code is the product identity across every MSI version.
    assert windows["wix"] == {"upgradeCode": "c9df5382-651e-5356-937a-98ee5d0f3e43"}
    # WiX stays single-language: the artifact collector expects exactly one MSI.
    assert "language" not in windows["wix"]
    assert windows["nsis"]["installMode"] == "currentUser"
    assert windows["nsis"]["languages"] == ["English", "SimpChinese"]
    assert windows["nsis"]["displayLanguageSelector"] is True
    assert config["bundle"]["targets"] == "all"


def test_tauri_config_validates_against_the_installed_cli_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    if not TAURI_CLI_SCHEMA.is_file():
        pytest.skip("run `npm --prefix apps/desktop ci` to install the Tauri CLI schema")
    schema = json.loads(TAURI_CLI_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.validators.validator_for(schema)(schema)

    def merged(base: dict, overlay: dict) -> dict:
        # `tauri build --config` applies the overlay as a JSON merge patch:
        # objects merge recursively, every other value is replaced.
        result = dict(base)
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merged(result[key], value)
            else:
                result[key] = value
        return result

    base = json.loads((TAURI_ROOT / "tauri.conf.json").read_text(encoding="utf-8"))
    prerelease = json.loads((TAURI_ROOT / "tauri.prerelease.conf.json").read_text(encoding="utf-8"))
    for name, config in (("tauri.conf.json", base), ("tauri.prerelease.conf.json", merged(base, prerelease))):
        errors = [
            f"{'/'.join(str(part) for part in error.path)}: {error.message}"
            for error in validator.iter_errors(config)
        ]
        assert errors == [], name
