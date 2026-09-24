from __future__ import annotations

from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
import pytest

from ufwbpp.catalogs import (
    CatalogError,
    DownloadResponse,
    catalog_doctor,
    catalog_list,
    get_catalog_manifest,
    install_catalog,
    installed_set_identity_for_solver_indexes,
    installed_set_snapshot_for_solver_config,
    load_catalog_manifests,
    recommended_artifacts,
    remove_catalog_plan,
    verified_installed_set_identities,
    verify_catalog,
    verify_installed_set_snapshot,
)
from ufwbpp.cli import main


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_MANIFEST_DIR = REPO_ROOT / "resources" / "catalogs"


def _fixture_manifest(directory: Path, payload: bytes = b"checked-catalog-bytes") -> tuple[Path, str]:
    directory.mkdir(parents=True)
    acceptance_id = "fake-provider-terms-v1"
    manifest = {
        "schemaVersion": 1,
        "catalogId": "fake-astrometry-4107",
        "provider": "Fake provider",
        "version": "fixture-v1",
        "bundled": False,
        "redistributionStatus": "user-download-required",
        "license": "fixture only",
        "citation": "fixture citation",
        "providerTerms": {
            "acceptanceId": acceptance_id,
            "url": "https://provider.invalid/terms",
            "summary": "Review the fake provider terms before this fixture download.",
            "licenseStatus": "provider-specific-unresolved",
            "requiresExplicitAcceptance": True,
        },
        "allowedDownloadOrigins": ["https://provider.invalid"],
        "artifacts": [
            {
                "artifactId": "index-4107.fits",
                "url": "https://provider.invalid/index-4107.fits",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "sizeBytes": len(payload),
                "installScope": "user",
                "scale": 7,
                "quadScaleArcminutes": {"minimum": 22, "maximum": 30},
                "recommendedImageFieldOfViewDegrees": {
                    "minimum": 22 / 60,
                    "maximum": 5,
                    "basis": "fixture",
                },
            }
        ],
    }
    path = directory / "fake-v1.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, acceptance_id


class FakeTransport:
    def __init__(self, payload: bytes, *, first_limit: int | None = None, ignore_range: bool = False) -> None:
        self.payload = payload
        self.first_limit = first_limit
        self.ignore_range = ignore_range
        self.calls: list[int] = []

    def open(
        self,
        url: str,
        *,
        start: int,
        timeout_seconds: float,
        if_range: str | None = None,
    ) -> DownloadResponse:
        assert url.startswith("https://")
        assert timeout_seconds > 0
        self.calls.append(start)
        effective_start = 0 if self.ignore_range else start
        status = 200 if effective_start == 0 else 206
        body = self.payload[effective_start:]
        if self.first_limit is not None and len(self.calls) == 1:
            body = body[: self.first_limit]
        headers: dict[str, str] = {
            "content-length": str(len(self.payload) - effective_start),
            "etag": '"fixture-v1"',
        }
        if status == 206:
            assert if_range == '"fixture-v1"'
            headers["content-range"] = (
                f"bytes {effective_start}-{len(self.payload) - 1}/{len(self.payload)}"
            )
        return DownloadResponse(
            status=status,
            headers=headers,
            stream=BytesIO(body),
            final_url=url,
        )


class RedirectTransport(FakeTransport):
    def open(self, url: str, **kwargs: Any) -> DownloadResponse:
        response = super().open(url, **kwargs)
        return DownloadResponse(
            status=response.status,
            headers=response.headers,
            stream=response.stream,
            final_url="https://other.invalid/index-4107.fits",
        )


class ChangedEtagTransport(FakeTransport):
    def open(self, url: str, **kwargs: Any) -> DownloadResponse:
        response = super().open(url, **kwargs)
        return DownloadResponse(
            status=response.status,
            headers={**response.headers, "etag": '"fixture-v2"'},
            stream=response.stream,
            final_url=response.final_url,
        )


class OversizeTransport(FakeTransport):
    def open(self, url: str, **kwargs: Any) -> DownloadResponse:
        response = super().open(url, **kwargs)
        return DownloadResponse(
            status=response.status,
            # Lie that the body still has the checked size. The streaming hard
            # limit, rather than Content-Length, must stop publication.
            headers=response.headers,
            stream=BytesIO(self.payload + b"unexpected-extra-byte"),
            final_url=response.final_url,
        )


def test_checked_catalog_manifests_validate_and_bind_official_4107_4112() -> None:
    schema = json.loads((OFFICIAL_MANIFEST_DIR / "catalog-manifest-v1.schema.json").read_text())
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    manifests = load_catalog_manifests(OFFICIAL_MANIFEST_DIR)
    assert {item.catalog_id for item in manifests} >= {
        "astrometry-net-4107-4112",
        "astrometry-net-4108",
        "astap-external",
    }
    for path in OFFICIAL_MANIFEST_DIR.glob("*.json"):
        if path.name.endswith(".schema.json"):
            continue
        validator.validate(json.loads(path.read_text()))

    wide = get_catalog_manifest("astrometry-net-4107-4112", OFFICIAL_MANIFEST_DIR)
    assert [item.scale for item in wide.artifacts] == [7, 8, 9, 10, 11, 12]
    assert [item.size_bytes for item in wide.artifacts] == [
        164_995_200,
        94_550_400,
        49_772_160,
        24_871_680,
        10_206_720,
        5_296_320,
    ]
    assert wide.total_size_bytes == 349_692_480
    assert wide.license_status == "provider-specific-unresolved"
    assert wide.requires_explicit_acceptance is True


def test_recommended_scales_follow_provider_ten_to_one_hundred_percent_rule() -> None:
    wide = get_catalog_manifest("astrometry-net-4107-4112", OFFICIAL_MANIFEST_DIR)
    assert [item.scale for item in recommended_artifacts(wide, 1.0)] == [7, 8, 9]
    assert [item.scale for item in recommended_artifacts(wide, 2.0)] == [7, 8, 9, 10, 11]
    with pytest.raises(CatalogError, match="no checked scale"):
        recommended_artifacts(wide, 0.05)


def test_install_requires_exact_versioned_terms_id_before_any_download(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _, acceptance_id = _fixture_manifest(manifests)
    transport = FakeTransport(b"checked-catalog-bytes")
    for supplied in (None, "yes", f"{acceptance_id}-old"):
        with pytest.raises(CatalogError) as failure:
            install_catalog(
                "fake-astrometry-4107",
                accepted_terms_id=supplied,
                catalog_root=tmp_path / "catalogs",
                manifest_dir=manifests,
                transport=transport,
            )
        assert failure.value.code == "CATALOG_TERMS_NOT_ACCEPTED"
    assert transport.calls == []
    assert not (tmp_path / "catalogs").exists()


def test_install_is_create_only_and_emits_config_and_hash_bound_receipt(tmp_path: Path) -> None:
    payload = b"checked-catalog-bytes"
    manifests = tmp_path / "manifests"
    _, acceptance_id = _fixture_manifest(manifests, payload)
    root = tmp_path / "catalogs"
    events: list[dict[str, Any]] = []
    transport = FakeTransport(payload)
    installed = install_catalog(
        "fake-astrometry-4107",
        accepted_terms_id=acceptance_id,
        catalog_root=root,
        manifest_dir=manifests,
        transport=transport,
        progress=lambda event: events.append(dict(event)),
    )
    assert installed["ok"] is True
    assert transport.calls == [0]
    assert (root / "index-4107.fits").read_bytes() == payload
    assert (root / "astrometry.cfg").read_text().splitlines()[1] == f"add_path {root}"
    receipt = installed["installedSet"]
    assert receipt["artifacts"] == [
        {
            "artifactId": "index-4107.fits",
            "relativePath": "index-4107.fits",
            "sourceUrl": "https://provider.invalid/index-4107.fits",
            "sizeBytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    assert len(receipt["manifestSha256"]) == 64
    assert len(receipt["installedSetIdentity"]) == 64
    assert Path(receipt["receiptPath"]).parent == root
    receipt_value = json.loads(Path(receipt["receiptPath"]).read_text())
    receipt_schema = json.loads(
        (OFFICIAL_MANIFEST_DIR / "installed-set-v1.schema.json").read_text()
    )
    Draft202012Validator(receipt_schema).validate(receipt_value)
    assert receipt_value["transportEvidence"] == [
        {
            "artifactId": "index-4107.fits",
            "entityTag": '"fixture-v1"',
            "method": "provider-download",
            "sourceUrl": "https://provider.invalid/index-4107.fits",
        }
    ]
    assert events[-1]["downloadedBytes"] == len(payload)

    # Repeating an identical install verifies the existing target and receipt;
    # it never opens the network or replaces the destination.
    second_transport = FakeTransport(payload)
    repeated = install_catalog(
        "fake-astrometry-4107",
        accepted_terms_id=acceptance_id,
        catalog_root=root,
        manifest_dir=manifests,
        transport=second_transport,
    )
    assert second_transport.calls == []
    assert repeated["artifacts"][0]["status"] == "ALREADY_INSTALLED"
    assert repeated["installedSet"]["installedSetIdentity"] == receipt["installedSetIdentity"]


def test_interrupted_download_resumes_with_a_valid_range(tmp_path: Path) -> None:
    payload = b"0123456789abcdef"
    manifests = tmp_path / "manifests"
    _, acceptance_id = _fixture_manifest(manifests, payload)
    root = tmp_path / "catalogs"
    interrupted = FakeTransport(payload, first_limit=5)
    with pytest.raises(CatalogError) as failure:
        install_catalog(
            "fake-astrometry-4107",
            accepted_terms_id=acceptance_id,
            catalog_root=root,
            manifest_dir=manifests,
            transport=interrupted,
        )
    assert failure.value.code == "CATALOG_DOWNLOAD_INCOMPLETE"
    partials = list((root / ".downloads").glob("*.partial"))
    assert len(partials) == 1
    assert partials[0].read_bytes() == payload[:5]

    resumed = FakeTransport(payload)
    result = install_catalog(
        "fake-astrometry-4107",
        accepted_terms_id=acceptance_id,
        catalog_root=root,
        manifest_dir=manifests,
        transport=resumed,
    )
    assert resumed.calls == [5]
    assert result["ok"] is True
    assert (root / "index-4107.fits").read_bytes() == payload
    assert not list((root / ".downloads").glob("*.partial"))


def test_provider_that_ignores_range_restarts_without_appending(tmp_path: Path) -> None:
    payload = b"0123456789abcdef"
    manifests = tmp_path / "manifests"
    _, acceptance_id = _fixture_manifest(manifests, payload)
    root = tmp_path / "catalogs"
    with pytest.raises(CatalogError):
        install_catalog(
            "fake-astrometry-4107",
            accepted_terms_id=acceptance_id,
            catalog_root=root,
            manifest_dir=manifests,
            transport=FakeTransport(payload, first_limit=3),
        )
    transport = FakeTransport(payload, ignore_range=True)
    install_catalog(
        "fake-astrometry-4107",
        accepted_terms_id=acceptance_id,
        catalog_root=root,
        manifest_dir=manifests,
        transport=transport,
    )
    assert transport.calls == [3]
    assert (root / "index-4107.fits").read_bytes() == payload


def test_hash_failure_never_publishes_and_preserves_forensic_partial(tmp_path: Path) -> None:
    expected = b"expected bytes"
    received = b"corrupt! bytes"
    assert len(expected) == len(received)
    manifests = tmp_path / "manifests"
    _, acceptance_id = _fixture_manifest(manifests, expected)
    root = tmp_path / "catalogs"
    with pytest.raises(CatalogError) as failure:
        install_catalog(
            "fake-astrometry-4107",
            accepted_terms_id=acceptance_id,
            catalog_root=root,
            manifest_dir=manifests,
            transport=FakeTransport(received),
        )
    assert failure.value.code == "CATALOG_HASH_MISMATCH"
    assert not (root / "index-4107.fits").exists()
    assert len(list((root / ".downloads").glob("*.invalid-*"))) == 1


def test_exact_manifest_url_is_enforced_and_stream_size_is_hard_limited(tmp_path: Path) -> None:
    payload = b"checked-catalog-bytes"
    manifests = tmp_path / "manifests"
    _, acceptance_id = _fixture_manifest(manifests, payload)
    for transport, expected_code in (
        (RedirectTransport(payload), "CATALOG_REDIRECT_FORBIDDEN"),
        (OversizeTransport(payload), "CATALOG_DOWNLOAD_OVERSIZE"),
    ):
        root = tmp_path / expected_code
        with pytest.raises(CatalogError) as failure:
            install_catalog(
                "fake-astrometry-4107",
                accepted_terms_id=acceptance_id,
                catalog_root=root,
                manifest_dir=manifests,
                transport=transport,
            )
        assert failure.value.code == expected_code
        assert not (root / "index-4107.fits").exists()


def test_resume_range_requires_the_same_strong_etag(tmp_path: Path) -> None:
    payload = b"0123456789abcdef"
    manifests = tmp_path / "manifests"
    _, acceptance_id = _fixture_manifest(manifests, payload)
    root = tmp_path / "catalogs"
    with pytest.raises(CatalogError):
        install_catalog(
            "fake-astrometry-4107",
            accepted_terms_id=acceptance_id,
            catalog_root=root,
            manifest_dir=manifests,
            transport=FakeTransport(payload, first_limit=5),
        )
    with pytest.raises(CatalogError) as failure:
        install_catalog(
            "fake-astrometry-4107",
            accepted_terms_id=acceptance_id,
            catalog_root=root,
            manifest_dir=manifests,
            transport=ChangedEtagTransport(payload),
        )
    assert failure.value.code == "CATALOG_RANGE_INVALID"
    assert not (root / "index-4107.fits").exists()


def test_verify_can_explicitly_configure_preexisting_checked_file(tmp_path: Path) -> None:
    payload = b"checked-catalog-bytes"
    manifests = tmp_path / "manifests"
    _fixture_manifest(manifests, payload)
    root = tmp_path / "catalogs"
    root.mkdir()
    (root / "index-4107.fits").write_bytes(payload)
    read_only = verify_catalog(
        "fake-astrometry-4107", catalog_root=root, manifest_dir=manifests
    )
    assert read_only["ok"] is True
    assert read_only["installedSet"] is None
    assert not (root / "astrometry.cfg").exists()

    configured = verify_catalog(
        "fake-astrometry-4107",
        catalog_root=root,
        manifest_dir=manifests,
        write_configuration=True,
    )
    assert configured["ok"] is True
    assert (root / "astrometry.cfg").is_file()
    assert configured["installedSet"]["artifacts"][0]["relativePath"] == "index-4107.fits"


def test_configure_adopts_semantically_identical_existing_config_without_replacing_it(
    tmp_path: Path,
) -> None:
    payload = b"checked-catalog-bytes"
    manifests = tmp_path / "manifests"
    _fixture_manifest(manifests, payload)
    root = tmp_path / "catalogs"
    root.mkdir()
    (root / "index-4107.fits").write_bytes(payload)
    existing = f"add_path {root}\nautoindex\ninparallel\n".encode()
    config = root / "astrometry.cfg"
    config.write_bytes(existing)
    inode = config.stat().st_ino
    configured = verify_catalog(
        "fake-astrometry-4107",
        catalog_root=root,
        manifest_dir=manifests,
        write_configuration=True,
    )
    assert configured["ok"] is True
    assert config.read_bytes() == existing
    assert config.stat().st_ino == inode
    assert configured["installedSet"]["config"]["sha256"] == hashlib.sha256(existing).hexdigest()


def test_installed_set_api_binds_solver_index_to_actual_verified_bytes(tmp_path: Path) -> None:
    payload = b"checked-catalog-bytes"
    manifests = tmp_path / "manifests"
    _, acceptance_id = _fixture_manifest(manifests, payload)
    root = tmp_path / "catalogs"
    installed = install_catalog(
        "fake-astrometry-4107",
        accepted_terms_id=acceptance_id,
        catalog_root=root,
        manifest_dir=manifests,
        transport=FakeTransport(payload),
    )
    receipts = verified_installed_set_identities(
        catalog_root=root, manifest_dir=manifests
    )
    assert len(receipts) == 1
    reference = installed_set_identity_for_solver_indexes(
        ("astrometry.net:index:4107:healpix:-1:hpnside:0",),
        catalog_root=root,
        manifest_dir=manifests,
    )
    assert reference["installedSetIdentity"] == installed["installedSet"]["installedSetIdentity"]
    assert reference["artifacts"][0]["sha256"] == hashlib.sha256(payload).hexdigest()

    # Config or index mutation invalidates the immutable receipt and therefore
    # the solver bind.
    config = root / "astrometry.cfg"
    original_config = config.read_bytes()
    config.write_bytes(original_config + b"# changed\n")
    with pytest.raises(CatalogError) as config_failure:
        installed_set_identity_for_solver_indexes(
            ("astrometry.net:index:4107:healpix:-1:hpnside:0",),
            catalog_root=root,
            manifest_dir=manifests,
        )
    assert config_failure.value.code == "CATALOG_INSTALLED_SET_UNBOUND"
    config.write_bytes(original_config)

    (root / "index-4107.fits").write_bytes(b"changed-catalog-bytes")
    with pytest.raises(CatalogError) as failure:
        installed_set_identity_for_solver_indexes(
            ("astrometry.net:index:4107:healpix:-1:hpnside:0",),
            catalog_root=root,
            manifest_dir=manifests,
        )
    assert failure.value.code == "CATALOG_INSTALLED_SET_UNBOUND"


def test_solver_catalog_snapshot_detects_byte_identical_replacement(tmp_path: Path) -> None:
    payload = b"checked-catalog-bytes"
    manifests = tmp_path / "manifests"
    _, acceptance_id = _fixture_manifest(manifests, payload)
    root = tmp_path / "catalogs"
    install_catalog(
        "fake-astrometry-4107",
        accepted_terms_id=acceptance_id,
        catalog_root=root,
        manifest_dir=manifests,
        transport=FakeTransport(payload),
    )
    snapshot = installed_set_snapshot_for_solver_config(
        root / "astrometry.cfg", manifest_dir=manifests
    )
    replacement = root / "replacement.fits"
    replacement.write_bytes(payload)
    os.replace(replacement, root / "index-4107.fits")
    with pytest.raises(CatalogError) as failure:
        verify_installed_set_snapshot(snapshot, manifest_dir=manifests)
    assert failure.value.code == "CATALOG_FILE_CHANGED"


def test_equally_specific_installed_sets_are_ambiguous(tmp_path: Path) -> None:
    payload = b"checked-catalog-bytes"
    manifests = tmp_path / "manifests"
    path, acceptance_id = _fixture_manifest(manifests, payload)
    root = tmp_path / "catalogs"
    install_catalog(
        "fake-astrometry-4107",
        accepted_terms_id=acceptance_id,
        catalog_root=root,
        manifest_dir=manifests,
        transport=FakeTransport(payload),
    )
    second = json.loads(path.read_text(encoding="utf-8"))
    second["catalogId"] = "second-astrometry-4107"
    second["provider"] = "Second fake provider"
    second["providerTerms"]["acceptanceId"] = "second-provider-terms-v1"
    (manifests / "second-v1.json").write_text(json.dumps(second), encoding="utf-8")
    verified = verify_catalog(
        "second-astrometry-4107",
        catalog_root=root,
        manifest_dir=manifests,
        write_configuration=True,
    )
    assert verified["ok"] is True
    with pytest.raises(CatalogError) as failure:
        installed_set_identity_for_solver_indexes(
            ("astrometry.net:index:4107:healpix:-1:hpnside:0",),
            catalog_root=root,
            manifest_dir=manifests,
        )
    assert failure.value.code == "CATALOG_INDEX_AMBIGUOUS"


def test_remove_plan_is_read_only_and_retains_shared_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "catalogs"
    root.mkdir()
    shared = root / "index-4108.fits"
    shared.write_bytes(b"not a real index")
    plan = remove_catalog_plan(
        "astrometry-net-4108",
        catalog_root=root,
        manifest_dir=OFFICIAL_MANIFEST_DIR,
    )
    assert plan["operation"] == "REMOVE_PLAN_ONLY"
    assert plan["executed"] is False
    assert plan["paths"] == []
    assert plan["retainedSharedPaths"][0]["sharedWithCatalogs"] == [
        "astrometry-net-4107-4112"
    ]
    assert shared.exists()


def test_path_traversal_and_symlink_targets_are_rejected(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    path, acceptance_id = _fixture_manifest(manifests)
    value = json.loads(path.read_text())
    value["artifacts"][0]["artifactId"] = "../escape.fits"
    path.write_text(json.dumps(value))
    with pytest.raises(CatalogError) as failure:
        load_catalog_manifests(manifests)
    assert failure.value.code == "CATALOG_PATH_UNSAFE"

    if hasattr(Path, "symlink_to"):
        safe_manifests = tmp_path / "safe-manifests"
        _fixture_manifest(safe_manifests)
        root = tmp_path / "catalogs"
        root.mkdir()
        external = tmp_path / "external"
        external.write_bytes(b"do not touch")
        (root / "index-4107.fits").symlink_to(external)
        with pytest.raises(CatalogError) as symlink_failure:
            install_catalog(
                "fake-astrometry-4107",
                accepted_terms_id=acceptance_id,
                catalog_root=root,
                manifest_dir=safe_manifests,
                transport=FakeTransport(b"checked-catalog-bytes"),
            )
        assert symlink_failure.value.code == "CATALOG_FILE_UNSAFE"
        assert external.read_bytes() == b"do not touch"


def test_localhost_ip_and_non_allowlisted_origins_are_rejected(tmp_path: Path) -> None:
    for origin, url in (
        ("https://127.0.0.1", "https://127.0.0.1/index-4107.fits"),
        ("https://provider.invalid", "https://other.invalid/index-4107.fits"),
    ):
        manifests = tmp_path / hashlib.sha256(url.encode()).hexdigest()[:8]
        path, _ = _fixture_manifest(manifests)
        value = json.loads(path.read_text())
        value["allowedDownloadOrigins"] = [origin]
        value["artifacts"][0]["url"] = url
        path.write_text(json.dumps(value))
        with pytest.raises(CatalogError) as failure:
            load_catalog_manifests(manifests)
        assert failure.value.code == "CATALOG_MANIFEST_INVALID"


def test_catalog_list_and_doctor_do_not_download_or_create_directories(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    _fixture_manifest(manifests)
    root = tmp_path / "absent"
    listing = catalog_list(catalog_root=root, manifest_dir=manifests)
    assert listing["catalogs"][0]["fullyInstalledBySize"] is False
    doctor = catalog_doctor(catalog_root=root, manifest_dir=manifests)
    assert doctor["ok"] is False
    assert "No catalog directory" in doctor["message"]
    assert not root.exists()


def test_catalog_cli_exposes_machine_readable_seam_and_stable_terms_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifests = tmp_path / "manifests"
    _fixture_manifest(manifests)
    root = tmp_path / "catalogs"
    assert main(
        [
            "catalog",
            "list",
            "--manifest-dir",
            str(manifests),
            "--catalog-dir",
            str(root),
            "--json",
        ]
    ) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["catalogs"][0]["providerTerms"]["acceptanceId"] == "fake-provider-terms-v1"

    assert main(
        [
            "catalog",
            "install",
            "fake-astrometry-4107",
            "--manifest-dir",
            str(manifests),
            "--catalog-dir",
            str(root),
        ]
    ) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"]["code"] == "CATALOG_TERMS_NOT_ACCEPTED"
    assert not root.exists()
