from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from typing import Mapping
from urllib.parse import urlencode

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_macos14_runtime_libraries.py"
SPEC = importlib.util.spec_from_file_location("fetch_macos14_runtime_libraries", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fetcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fetcher
SPEC.loader.exec_module(fetcher)


class FakeTransport:
    def __init__(self, routes: Mapping[str, list[fetcher.HttpResponse]]) -> None:
        self.routes = {url: list(responses) for url, responses in routes.items()}
        self.requests: list[tuple[str, dict[str, str]]] = []

    def request(
        self,
        url: str,
        headers: Mapping[str, str],
        *,
        maximum_bytes: int,
        timeout_seconds: float,
    ) -> fetcher.HttpResponse:
        assert maximum_bytes > 0
        assert timeout_seconds > 0
        self.requests.append((url, dict(headers)))
        try:
            response = self.routes[url].pop(0)
        except (KeyError, IndexError) as error:
            raise AssertionError(f"unexpected request: {url}") from error
        assert response.url == url
        return response


def _response(
    url: str,
    body: bytes = b"",
    *,
    status: int = 200,
    headers: Mapping[str, str] | None = None,
) -> fetcher.HttpResponse:
    return fetcher.HttpResponse(status=status, url=url, headers=headers or {}, body=body)


def _gzip_tar(
    files: Mapping[str, bytes],
    *,
    extra_members: list[tarfile.TarInfo] | None = None,
) -> bytes:
    payload = BytesIO()
    with gzip.GzipFile(fileobj=payload, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, content in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                member.mode = 0o644
                member.mtime = 0
                archive.addfile(member, BytesIO(content))
            for member in extra_members or []:
                content = b"ignored" if member.isfile() else b""
                member.size = len(content)
                member.mtime = 0
                archive.addfile(member, BytesIO(content) if content else None)
    return payload.getvalue()


def _synthetic_policy(
    tmp_path: Path,
    *,
    exact_member_as_symlink: bool = False,
) -> tuple[Path, dict[str, object], dict[str, list[fetcher.HttpResponse]]]:
    policy = json.loads(fetcher.MACOS14_RUNTIME_POLICY.read_text(encoding="utf-8"))
    routes: dict[str, list[fetcher.HttpResponse]] = {}
    source = policy["ociSource"]
    library_bytes = {
        "libcrypto.3.dylib": b"synthetic crypto dylib",
        "libssl.3.dylib": b"synthetic ssl dylib",
        "libmpdec.4.dylib": b"synthetic mpdecimal dylib",
    }
    for package in policy["packages"]:
        oci = package["oci"]
        archive_files: dict[str, bytes] = {}
        extras: list[tarfile.TarInfo] = []
        for library in package["libraries"]:
            content = library_bytes[library["destinationName"]]
            library["sourceSizeBytes"] = len(content)
            library["sourceSha256"] = hashlib.sha256(content).hexdigest()
            if exact_member_as_symlink and not extras:
                symlink = tarfile.TarInfo(library["archiveMemberPath"])
                symlink.type = tarfile.SYMTYPE
                symlink.linkname = "../../outside"
                extras.append(symlink)
            else:
                archive_files[library["archiveMemberPath"]] = content
        traversal = tarfile.TarInfo("../../must-not-extract")
        extras.append(traversal)
        layer = _gzip_tar(archive_files, extra_members=extras)
        config = json.dumps(
            {
                "architecture": "arm64",
                "os": "darwin",
                "os.version": "macOS 14.8",
                "rootfs": {"type": "layers", "diff_ids": ["sha256:" + "1" * 64]},
            },
            separators=(",", ":"),
        ).encode()
        oci["configSha256"] = hashlib.sha256(config).hexdigest()
        oci["configSizeBytes"] = len(config)
        oci["blobSha256"] = hashlib.sha256(layer).hexdigest()
        oci["blobSizeBytes"] = len(layer)
        layout = "openssl" if package["formula"] == "openssl@3" else "mpdecimal"
        oci["configRelativePath"] = f"{layout}/{oci['configSha256']}"
        oci["blobRelativePath"] = f"{layout}/{oci['blobSha256']}"
        manifest = json.dumps(
            {
                "schemaVersion": 2,
                "config": {
                    "mediaType": oci["configMediaType"],
                    "digest": f"sha256:{oci['configSha256']}",
                    "size": oci["configSizeBytes"],
                },
                "layers": [
                    {
                        "mediaType": oci["layerMediaType"],
                        "digest": f"sha256:{oci['blobSha256']}",
                        "size": oci["blobSizeBytes"],
                    }
                ],
                "annotations": {
                    "org.opencontainers.image.version": package["version"],
                    "org.opencontainers.image.ref.name": (
                        f"{package['version']}.{package['bottleTag']}"
                    ),
                    "org.opencontainers.image.revision": package["sourceRevision"],
                    "org.opencontainers.image.licenses": package["license"],
                    "sh.brew.bottle.digest": oci["blobSha256"],
                },
            },
            separators=(",", ":"),
        ).encode()
        oci["manifestSha256"] = hashlib.sha256(manifest).hexdigest()
        oci["manifestSizeBytes"] = len(manifest)
        oci["manifestDigest"] = f"sha256:{oci['manifestSha256']}"

        query = urlencode(
            {
                "service": source["tokenService"],
                "scope": f"repository:{oci['repository']}:pull",
            }
        )
        token_url = f"{source['tokenUrl']}?{query}"
        token = f"token-{layout}"
        routes[token_url] = [
            _response(
                token_url,
                json.dumps({"token": token}).encode(),
                headers={"Content-Type": "application/json"},
            )
        ]
        base = f"{source['apiBaseUrl']}/{oci['repository']}"
        manifest_url = f"{base}/manifests/{oci['manifestDigest']}"
        config_url = f"{base}/blobs/sha256:{oci['configSha256']}"
        layer_url = f"{base}/blobs/sha256:{oci['blobSha256']}"
        routes[manifest_url] = [
            _response(
                manifest_url,
                manifest,
                headers={
                    "Content-Type": oci["manifestMediaType"],
                    "Docker-Content-Digest": oci["manifestDigest"],
                },
            )
        ]
        routes[config_url] = [_response(config_url, config)]
        routes[layer_url] = [_response(layer_url, layer)]

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return policy_path, policy, routes


def test_materializes_only_pinned_regular_libraries_and_raw_oci(tmp_path: Path) -> None:
    policy_path, policy, routes = _synthetic_policy(tmp_path)
    transport = FakeTransport(routes)
    output = tmp_path / "bottles"

    summary = fetcher.materialize_runtime_libraries(
        output, policy_path=policy_path, transport=transport
    )

    assert summary["targetTriple"] == "aarch64-apple-darwin"
    assert len(summary["packages"]) == 2
    assert not (tmp_path.parent / "must-not-extract").exists()
    assert not any(path.is_symlink() for path in output.rglob("*"))
    for package in policy["packages"]:
        oci = package["oci"]
        assert (
            output / oci["manifestRelativePath"]
        ).stat().st_size == oci["manifestSizeBytes"]
        assert (
            output / oci["configRelativePath"]
        ).stat().st_size == oci["configSizeBytes"]
        assert (output / oci["blobRelativePath"]).stat().st_size == oci["blobSizeBytes"]
        for library in package["libraries"]:
            extracted_hash = hashlib.sha256(
                (output / library["sourceRelativePath"]).read_bytes()
            ).hexdigest()
            assert extracted_hash == library["sourceSha256"]
    requested_urls = [url for url, _headers in transport.requests]
    assert all("latest" not in url.casefold() for url in requested_urls)
    assert all(
        "/manifests/sha256:" in url
        for url in requested_urls
        if "/manifests/" in url
    )


def test_existing_output_is_never_touched_or_requested(tmp_path: Path) -> None:
    policy_path, _policy, routes = _synthetic_policy(tmp_path)
    output = tmp_path / "bottles"
    output.mkdir()
    marker = output / "owned-by-user"
    marker.write_text("keep", encoding="utf-8")
    transport = FakeTransport(routes)

    with pytest.raises(fetcher.RuntimeLibraryFetchError, match="refusing to replace"):
        fetcher.materialize_runtime_libraries(
            output, policy_path=policy_path, transport=transport
        )

    assert marker.read_text(encoding="utf-8") == "keep"
    assert transport.requests == []


def test_hash_failure_removes_only_new_partial_root(tmp_path: Path) -> None:
    policy_path, policy, routes = _synthetic_policy(tmp_path)
    first = policy["packages"][0]["oci"]
    layer_url = (
        f"{policy['ociSource']['apiBaseUrl']}/{first['repository']}"
        f"/blobs/sha256:{first['blobSha256']}"
    )
    routes[layer_url] = [_response(layer_url, b"wrong")]
    output = tmp_path / "bottles"

    with pytest.raises(fetcher.RuntimeLibraryFetchError, match="pinned bytes"):
        fetcher.materialize_runtime_libraries(
            output, policy_path=policy_path, transport=FakeTransport(routes)
        )

    assert not output.exists()


def test_manifest_reference_mismatch_fails_before_blob_fetch(tmp_path: Path) -> None:
    policy_path, policy, routes = _synthetic_policy(tmp_path)
    first = policy["packages"][0]["oci"]
    manifest_url = (
        f"{policy['ociSource']['apiBaseUrl']}/{first['repository']}"
        f"/manifests/{first['manifestDigest']}"
    )
    manifest = json.loads(routes[manifest_url][0].body)
    manifest["layers"][0]["digest"] = "sha256:" + "0" * 64
    altered = json.dumps(manifest, separators=(",", ":")).encode()
    first["manifestSha256"] = hashlib.sha256(altered).hexdigest()
    first["manifestSizeBytes"] = len(altered)
    first["manifestDigest"] = f"sha256:{first['manifestSha256']}"
    new_manifest_url = (
        f"{policy['ociSource']['apiBaseUrl']}/{first['repository']}"
        f"/manifests/{first['manifestDigest']}"
    )
    routes.pop(manifest_url)
    routes[new_manifest_url] = [
        _response(
            new_manifest_url,
            altered,
            headers={
                "Content-Type": first["manifestMediaType"],
                "Docker-Content-Digest": first["manifestDigest"],
            },
        )
    ]
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    transport = FakeTransport(routes)

    with pytest.raises(fetcher.RuntimeLibraryFetchError, match="references"):
        fetcher.materialize_runtime_libraries(
            tmp_path / "bottles", policy_path=policy_path, transport=transport
        )

    assert not any("/blobs/" in url for url, _headers in transport.requests)


def test_exact_library_symlink_is_rejected_without_extraction(tmp_path: Path) -> None:
    policy_path, _policy, routes = _synthetic_policy(tmp_path, exact_member_as_symlink=True)
    output = tmp_path / "bottles"

    with pytest.raises(fetcher.RuntimeLibraryFetchError, match="linked"):
        fetcher.materialize_runtime_libraries(
            output, policy_path=policy_path, transport=FakeTransport(routes)
        )

    assert not output.exists()


def _source_policy() -> dict[str, object]:
    return fetcher.load_macos14_runtime_policy()["ociSource"]


def test_cross_origin_redirect_drops_bearer_authorization() -> None:
    source = _source_policy()
    initial = "https://ghcr.io/v2/homebrew/core/mpdecimal/blobs/sha256:" + "a" * 64
    redirected = "https://pkg-containers.githubusercontent.com/object?signature=pinned-by-hash"
    transport = FakeTransport(
        {
            initial: [_response(initial, status=307, headers={"Location": redirected})],
            redirected: [_response(redirected, b"payload")],
        }
    )

    response = fetcher._request_with_redirects(
        transport,
        initial,
        {"Authorization": "Bearer do-not-leak", "Accept-Encoding": "identity"},
        source=source,
        maximum_bytes=100,
        timeout_seconds=1,
    )

    assert response.body == b"payload"
    assert transport.requests[0][1]["Authorization"] == "Bearer do-not-leak"
    assert "Authorization" not in transport.requests[1][1]


@pytest.mark.parametrize(
    "location",
    [
        "http://pkg-containers.githubusercontent.com/object",
        "https://evil.example/object",
        "https://blob.core.windows.net/object",
    ],
)
def test_http_unknown_and_bare_suffix_redirects_are_rejected(location: str) -> None:
    source = _source_policy()
    initial = "https://ghcr.io/v2/object"
    transport = FakeTransport(
        {initial: [_response(initial, status=307, headers={"Location": location})]}
    )

    with pytest.raises(fetcher.RuntimeLibraryFetchError, match="HTTPS host boundary"):
        fetcher._request_with_redirects(
            transport,
            initial,
            {"Authorization": "Bearer secret"},
            source=source,
            maximum_bytes=100,
            timeout_seconds=1,
        )

    assert len(transport.requests) == 1


def test_redirect_loop_is_rejected_and_auth_never_returns() -> None:
    source = _source_policy()
    initial = "https://ghcr.io/v2/object"
    redirected = "https://account.blob.core.windows.net/object?sig=value"
    transport = FakeTransport(
        {
            initial: [_response(initial, status=302, headers={"Location": redirected})],
            redirected: [_response(redirected, status=307, headers={"Location": initial})],
        }
    )

    with pytest.raises(fetcher.RuntimeLibraryFetchError, match="loop"):
        fetcher._request_with_redirects(
            transport,
            initial,
            {"Authorization": "Bearer secret"},
            source=source,
            maximum_bytes=100,
            timeout_seconds=1,
        )

    assert "Authorization" not in transport.requests[1][1]


def test_fourth_redirect_is_rejected_without_requesting_its_target() -> None:
    source = _source_policy()
    urls = [
        "https://ghcr.io/v2/object",
        "https://pkg-containers.githubusercontent.com/one",
        "https://account.blob.core.windows.net/two",
        "https://ghcr.io/v2/three",
        "https://pkg-containers.githubusercontent.com/four",
    ]
    transport = FakeTransport(
        {
            url: [_response(url, status=307, headers={"Location": urls[index + 1]})]
            for index, url in enumerate(urls[:-1])
        }
    )

    with pytest.raises(fetcher.RuntimeLibraryFetchError, match="limit"):
        fetcher._request_with_redirects(
            transport,
            urls[0],
            {"Authorization": "Bearer secret"},
            source=source,
            maximum_bytes=100,
            timeout_seconds=1,
        )

    assert [url for url, _headers in transport.requests] == urls[:4]


def test_policy_rejects_mutable_or_unreviewed_network_boundaries(tmp_path: Path) -> None:
    original = json.loads(fetcher.MACOS14_RUNTIME_POLICY.read_text(encoding="utf-8"))
    mutations = []
    unsafe_api = deepcopy(original)
    unsafe_api["ociSource"]["apiBaseUrl"] = "http://ghcr.io/v2"
    mutations.append(unsafe_api)
    unsafe_redirect = deepcopy(original)
    unsafe_redirect["ociSource"]["allowedRedirectHosts"] = ["example.com"]
    mutations.append(unsafe_redirect)
    mutable_repository = deepcopy(original)
    mutable_repository["packages"][0]["oci"]["repository"] = "homebrew/core/latest"
    mutations.append(mutable_repository)
    inconsistent_digest = deepcopy(original)
    inconsistent_digest["packages"][0]["oci"]["manifestDigest"] = "sha256:" + "0" * 64
    mutations.append(inconsistent_digest)

    for index, payload in enumerate(mutations):
        policy_path = tmp_path / f"unsafe-{index}.json"
        policy_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(fetcher.SidecarBuildError):
            fetcher.load_macos14_runtime_policy(policy_path)


def test_malformed_token_fails_without_publishing_output(tmp_path: Path) -> None:
    policy_path, policy, routes = _synthetic_policy(tmp_path)
    source = policy["ociSource"]
    first_repository = policy["packages"][0]["oci"]["repository"]
    query = urlencode(
        {
            "service": source["tokenService"],
            "scope": f"repository:{first_repository}:pull",
        }
    )
    token_url = f"{source['tokenUrl']}?{query}"
    routes[token_url] = [
        _response(
            token_url,
            b'{"token":"contains whitespace"}',
            headers={"Content-Type": "application/json"},
        )
    ]
    output = tmp_path / "bottles"

    with pytest.raises(fetcher.RuntimeLibraryFetchError, match="unsafe"):
        fetcher.materialize_runtime_libraries(
            output, policy_path=policy_path, transport=FakeTransport(routes)
        )

    assert not output.exists()
