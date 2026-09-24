#!/usr/bin/env python3
"""Materialize the exact Homebrew Sonoma OCI inputs used by macOS packaging.

The checked-in policy is the trust root.  This helper never resolves tags,
versions, or ``latest`` aliases: it requests one manifest by digest, verifies
every downloaded byte, and extracts only the explicitly pinned regular dylibs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_engine_sidecar import (  # noqa: E402
    MACOS14_RUNTIME_POLICY,
    SidecarBuildError,
    _verify_oci_package,
    load_macos14_runtime_policy,
)


TOKEN_MAX_BYTES = 64 * 1024
MAX_TAR_MEMBERS = 100_000
USER_AGENT = "Ultra-Fast-WBPP-macOS-runtime-fetch/1"
REDIRECT_STATUSES = frozenset({302, 307})


class RuntimeLibraryFetchError(RuntimeError):
    """The pinned OCI layout could not be materialized safely."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class HttpResponse:
    status: int
    url: str
    headers: Mapping[str, str]
    body: bytes


class HttpTransport(Protocol):
    def request(
        self,
        url: str,
        headers: Mapping[str, str],
        *,
        maximum_bytes: int,
        timeout_seconds: float,
    ) -> HttpResponse: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(  # type: ignore[no-untyped-def]
        self, req, fp, code, msg, headers, newurl
    ):
        return None


class UrllibTransport:
    """One-request HTTPS transport with automatic redirects disabled."""

    def __init__(self) -> None:
        self._opener = build_opener(_NoRedirect())

    def request(
        self,
        url: str,
        headers: Mapping[str, str],
        *,
        maximum_bytes: int,
        timeout_seconds: float,
    ) -> HttpResponse:
        request = Request(url, headers=dict(headers), method="GET")
        try:
            try:
                response = self._opener.open(request, timeout=timeout_seconds)
            except HTTPError as error:
                response = error
            with response:
                normalized_headers = {
                    str(key).casefold(): str(value)
                    for key, value in response.headers.items()
                }
                status = int(response.status)
                if status in REDIRECT_STATUSES:
                    return HttpResponse(
                        status=status,
                        url=str(response.geturl()),
                        headers=normalized_headers,
                        body=b"",
                    )
                content_encoding = normalized_headers.get("content-encoding", "identity")
                if content_encoding.casefold() not in {"", "identity"}:
                    raise RuntimeLibraryFetchError(
                        "HTTP_CONTENT_ENCODING_INVALID",
                        "registry response was transformed by content encoding",
                    )
                content_length = normalized_headers.get("content-length")
                if content_length is not None:
                    try:
                        announced_size = int(content_length)
                    except ValueError as error:
                        raise RuntimeLibraryFetchError(
                            "HTTP_RESPONSE_INVALID", "response Content-Length is malformed"
                        ) from error
                    if announced_size < 0 or announced_size > maximum_bytes:
                        raise RuntimeLibraryFetchError(
                            "HTTP_RESPONSE_TOO_LARGE", "registry response exceeds its pinned size"
                        )
                body = response.read(maximum_bytes + 1)
                if len(body) > maximum_bytes:
                    raise RuntimeLibraryFetchError(
                        "HTTP_RESPONSE_TOO_LARGE", "registry response exceeds its pinned size"
                    )
                return HttpResponse(
                    status=status,
                    url=str(response.geturl()),
                    headers=normalized_headers,
                    body=body,
                )
        except RuntimeLibraryFetchError:
            raise
        except (OSError, URLError) as error:
            raise RuntimeLibraryFetchError(
                "HTTP_FETCH_FAILED", f"HTTPS registry request failed: {type(error).__name__}"
            ) from error


def _header(response: HttpResponse, name: str) -> str | None:
    wanted = name.casefold()
    for key, value in response.headers.items():
        if str(key).casefold() == wanted:
            return str(value)
    return None


def _validated_https_url(url: str, *, allowed_hosts: set[str]) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise RuntimeLibraryFetchError("HTTP_URL_INVALID", "registry URL is malformed") from error
    host = (parsed.hostname or "").casefold()
    if (
        parsed.scheme != "https"
        or not host
        or host not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or bool(parsed.fragment)
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for character in host)
    ):
        raise RuntimeLibraryFetchError(
            "HTTP_URL_FORBIDDEN", "registry request escaped the reviewed HTTPS host boundary"
        )
    return host


def _redirect_hosts(source: Mapping[str, Any], host: str) -> set[str]:
    allowed = {source["registry"], *source["allowedRedirectHosts"]}
    for suffix in source["allowedRedirectHostSuffixes"]:
        if host.endswith(suffix) and host != suffix[1:]:
            allowed.add(host)
    return allowed


def _request_with_redirects(
    transport: HttpTransport,
    initial_url: str,
    headers: Mapping[str, str],
    *,
    source: Mapping[str, Any],
    maximum_bytes: int,
    timeout_seconds: float,
) -> HttpResponse:
    registry = source["registry"]
    _validated_https_url(initial_url, allowed_hosts={registry})
    current_url = initial_url
    current_headers = dict(headers)
    visited: set[str] = set()
    for redirect_count in range(source["maximumRedirects"] + 1):
        if current_url in visited:
            raise RuntimeLibraryFetchError("HTTP_REDIRECT_LOOP", "registry redirect loop detected")
        visited.add(current_url)
        current_host = (urlsplit(current_url).hostname or "").casefold()
        _validated_https_url(
            current_url, allowed_hosts=_redirect_hosts(source, current_host)
        )
        response = transport.request(
            current_url,
            current_headers,
            maximum_bytes=maximum_bytes,
            timeout_seconds=timeout_seconds,
        )
        if response.url != current_url:
            raise RuntimeLibraryFetchError(
                "HTTP_TRANSPORT_REDIRECTED",
                "HTTP transport followed a redirect outside the audited redirect loop",
            )
        if response.status not in REDIRECT_STATUSES:
            return response
        if redirect_count == source["maximumRedirects"]:
            raise RuntimeLibraryFetchError(
                "HTTP_REDIRECT_LIMIT", "registry redirect limit exceeded"
            )
        location = _header(response, "location")
        if (
            location is None
            or not location
            or len(location) > 8192
            or "\r" in location
            or "\n" in location
        ):
            raise RuntimeLibraryFetchError(
                "HTTP_REDIRECT_INVALID", "registry redirect Location is missing or malformed"
            )
        next_url = urljoin(current_url, location)
        next_host = (urlsplit(next_url).hostname or "").casefold()
        _validated_https_url(next_url, allowed_hosts=_redirect_hosts(source, next_host))
        if next_host != current_host:
            current_headers = {
                key: value
                for key, value in current_headers.items()
                if key.casefold() != "authorization"
            }
        current_url = next_url
    raise AssertionError("unreachable redirect state")


def _require_status_ok(response: HttpResponse) -> None:
    if response.status != 200:
        raise RuntimeLibraryFetchError(
            "HTTP_STATUS_INVALID", f"registry returned HTTP {response.status}"
        )


def _fetch_bearer_token(
    transport: HttpTransport,
    source: Mapping[str, Any],
    repository: str,
    *,
    timeout_seconds: float,
) -> str:
    query = urlencode(
        {
            "service": source["tokenService"],
            "scope": f"repository:{repository}:pull",
        }
    )
    response = _request_with_redirects(
        transport,
        f"{source['tokenUrl']}?{query}",
        {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": USER_AGENT,
        },
        source=source,
        maximum_bytes=TOKEN_MAX_BYTES,
        timeout_seconds=timeout_seconds,
    )
    _require_status_ok(response)
    content_type = (_header(response, "content-type") or "").split(";", 1)[0].strip()
    if content_type != "application/json":
        raise RuntimeLibraryFetchError(
            "TOKEN_RESPONSE_INVALID", "GHCR token response is not JSON"
        )
    try:
        payload = json.loads(response.body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeLibraryFetchError(
            "TOKEN_RESPONSE_INVALID", "GHCR token response is malformed"
        ) from error
    token = payload.get("token") if isinstance(payload, dict) else None
    if (
        not isinstance(token, str)
        or not token
        or len(token) > 16_384
        or any(character.isspace() or ord(character) < 0x21 for character in token)
    ):
        raise RuntimeLibraryFetchError(
            "TOKEN_RESPONSE_INVALID", "GHCR bearer token is missing or unsafe"
        )
    return token


def _download_pinned(
    transport: HttpTransport,
    url: str,
    headers: Mapping[str, str],
    *,
    source: Mapping[str, Any],
    expected_sha256: str,
    expected_size: int,
    timeout_seconds: float,
) -> HttpResponse:
    response = _request_with_redirects(
        transport,
        url,
        headers,
        source=source,
        maximum_bytes=expected_size,
        timeout_seconds=timeout_seconds,
    )
    _require_status_ok(response)
    if (
        len(response.body) != expected_size
        or hashlib.sha256(response.body).hexdigest() != expected_sha256
    ):
        raise RuntimeLibraryFetchError(
            "OCI_BYTES_MISMATCH", "downloaded OCI object differs from its pinned bytes"
        )
    return response


def _validate_manifest(package: Mapping[str, Any], raw: bytes) -> None:
    oci = package["oci"]
    try:
        manifest = json.loads(raw.decode("utf-8"))
        config = manifest["config"]
        layers = manifest["layers"]
        annotations = manifest["annotations"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeLibraryFetchError(
            "OCI_MANIFEST_INVALID", "OCI manifest is malformed"
        ) from error
    if (
        manifest.get("schemaVersion") != 2
        or not isinstance(config, dict)
        or config.get("mediaType") != oci["configMediaType"]
        or config.get("digest") != f"sha256:{oci['configSha256']}"
        or config.get("size") != oci["configSizeBytes"]
        or not isinstance(layers, list)
        or len(layers) != 1
        or not isinstance(layers[0], dict)
        or layers[0].get("mediaType") != oci["layerMediaType"]
        or layers[0].get("digest") != f"sha256:{oci['blobSha256']}"
        or layers[0].get("size") != oci["blobSizeBytes"]
        or not isinstance(annotations, dict)
        or annotations.get("org.opencontainers.image.version") != package["version"]
        or annotations.get("org.opencontainers.image.ref.name")
        != f"{package['version']}.{package['bottleTag']}"
        or annotations.get("org.opencontainers.image.revision") != package["sourceRevision"]
        or annotations.get("org.opencontainers.image.licenses") != package["license"]
        or annotations.get("sh.brew.bottle.digest") != oci["blobSha256"]
    ):
        raise RuntimeLibraryFetchError(
            "OCI_MANIFEST_INVALID", "OCI manifest references or identity differ from policy"
        )


def _validate_config(raw: bytes) -> None:
    try:
        config = json.loads(raw.decode("utf-8"))
        rootfs = config["rootfs"]
        diff_ids = rootfs["diff_ids"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeLibraryFetchError("OCI_CONFIG_INVALID", "OCI config is malformed") from error
    if (
        config.get("architecture") != "arm64"
        or config.get("os") != "darwin"
        or not isinstance(config.get("os.version"), str)
        or not config["os.version"].startswith("macOS 14.")
        or rootfs.get("type") != "layers"
        or not isinstance(diff_ids, list)
        or len(diff_ids) != 1
        or not isinstance(diff_ids[0], str)
        or len(diff_ids[0]) != 71
        or not diff_ids[0].startswith("sha256:")
    ):
        raise RuntimeLibraryFetchError(
            "OCI_CONFIG_INVALID", "OCI config is not an arm64 macOS 14 single-layer image"
        )


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeLibraryFetchError("OUTPUT_PATH_INVALID", "policy output path is unsafe")
    return path


def _write_create_only(root: Path, relative: str, payload: bytes) -> Path:
    member = _safe_relative(relative)
    destination = root.joinpath(*member.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    return destination


def _extract_exact_libraries(
    root: Path, package: Mapping[str, Any], blob: bytes
) -> list[dict[str, Any]]:
    requested = {
        library["archiveMemberPath"]: library for library in package["libraries"]
    }
    found: set[str] = set()
    records: list[dict[str, Any]] = []
    try:
        with tarfile.open(fileobj=BytesIO(blob), mode="r:gz") as archive:
            for member_index, member in enumerate(archive):
                if member_index >= MAX_TAR_MEMBERS:
                    raise RuntimeLibraryFetchError(
                        "OCI_ARCHIVE_INVALID", "OCI layer contains too many members"
                    )
                library = requested.get(member.name)
                if library is None:
                    continue
                if (
                    member.name in found
                    or not member.isfile()
                    or member.size != library["sourceSizeBytes"]
                ):
                    raise RuntimeLibraryFetchError(
                        "OCI_ARCHIVE_INVALID",
                        "pinned library is duplicated, linked, or has the wrong size",
                    )
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeLibraryFetchError(
                        "OCI_ARCHIVE_INVALID", "pinned library cannot be read"
                    )
                payload = stream.read(library["sourceSizeBytes"] + 1)
                if (
                    len(payload) != library["sourceSizeBytes"]
                    or hashlib.sha256(payload).hexdigest() != library["sourceSha256"]
                ):
                    raise RuntimeLibraryFetchError(
                        "OCI_LIBRARY_MISMATCH", "extracted library differs from policy"
                    )
                _write_create_only(root, library["sourceRelativePath"], payload)
                found.add(member.name)
                records.append(
                    {
                        "archiveMemberPath": member.name,
                        "destinationName": library["destinationName"],
                        "sha256": library["sourceSha256"],
                        "sizeBytes": library["sourceSizeBytes"],
                    }
                )
    except RuntimeLibraryFetchError:
        raise
    except (OSError, tarfile.TarError) as error:
        raise RuntimeLibraryFetchError(
            "OCI_ARCHIVE_INVALID", "OCI layer is not a readable gzip tar archive"
        ) from error
    if found != set(requested):
        raise RuntimeLibraryFetchError(
            "OCI_ARCHIVE_INVALID", "OCI layer is missing a pinned library"
        )
    return sorted(records, key=lambda item: item["destinationName"])


def materialize_runtime_libraries(
    output: Path,
    *,
    policy_path: Path = MACOS14_RUNTIME_POLICY,
    transport: HttpTransport | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Create a verified bottle root exactly once and return a path-free summary."""

    if timeout_seconds <= 0 or timeout_seconds > 300:
        raise RuntimeLibraryFetchError(
            "ARGUMENT_INVALID", "timeout must be greater than zero and at most 300 seconds"
        )
    try:
        policy = load_macos14_runtime_policy(policy_path)
    except SidecarBuildError as error:
        raise RuntimeLibraryFetchError(error.code, str(error)) from error
    destination = output.expanduser()
    if destination.name in {"", ".", ".."}:
        raise RuntimeLibraryFetchError("OUTPUT_PATH_INVALID", "output root is unsafe")
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeLibraryFetchError(
            "OUTPUT_PARENT_INVALID", "output parent must be an existing real directory"
        )
    if destination.exists() or destination.is_symlink():
        raise RuntimeLibraryFetchError(
            "OUTPUT_EXISTS", "refusing to replace an existing runtime-library root"
        )
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as error:
        raise RuntimeLibraryFetchError(
            "OUTPUT_EXISTS", "refusing to replace an existing runtime-library root"
        ) from error
    client = transport or UrllibTransport()
    package_records: list[dict[str, Any]] = []
    try:
        source = policy["ociSource"]
        for package in policy["packages"]:
            oci = package["oci"]
            token = _fetch_bearer_token(
                client, source, oci["repository"], timeout_seconds=timeout_seconds
            )
            authorization = f"Bearer {token}"
            base = f"{source['apiBaseUrl']}/{oci['repository']}"
            manifest = _download_pinned(
                client,
                f"{base}/manifests/{oci['manifestDigest']}",
                {
                    "Accept": oci["manifestMediaType"],
                    "Accept-Encoding": "identity",
                    "Authorization": authorization,
                    "User-Agent": USER_AGENT,
                },
                source=source,
                expected_sha256=oci["manifestSha256"],
                expected_size=oci["manifestSizeBytes"],
                timeout_seconds=timeout_seconds,
            )
            content_type = (_header(manifest, "content-type") or "").split(";", 1)[0].strip()
            if (
                content_type != oci["manifestMediaType"]
                or _header(manifest, "docker-content-digest") != oci["manifestDigest"]
            ):
                raise RuntimeLibraryFetchError(
                    "OCI_MANIFEST_HEADERS_INVALID",
                    "GHCR manifest headers differ from the pinned digest/media type",
                )
            _validate_manifest(package, manifest.body)
            config = _download_pinned(
                client,
                f"{base}/blobs/sha256:{oci['configSha256']}",
                {
                    "Accept-Encoding": "identity",
                    "Authorization": authorization,
                    "User-Agent": USER_AGENT,
                },
                source=source,
                expected_sha256=oci["configSha256"],
                expected_size=oci["configSizeBytes"],
                timeout_seconds=timeout_seconds,
            )
            _validate_config(config.body)
            layer = _download_pinned(
                client,
                f"{base}/blobs/sha256:{oci['blobSha256']}",
                {
                    "Accept-Encoding": "identity",
                    "Authorization": authorization,
                    "User-Agent": USER_AGENT,
                },
                source=source,
                expected_sha256=oci["blobSha256"],
                expected_size=oci["blobSizeBytes"],
                timeout_seconds=timeout_seconds,
            )
            _write_create_only(destination, oci["manifestRelativePath"], manifest.body)
            _write_create_only(destination, oci["configRelativePath"], config.body)
            _write_create_only(destination, oci["blobRelativePath"], layer.body)
            libraries = _extract_exact_libraries(destination, package, layer.body)
            try:
                _verify_oci_package(destination, package)
            except SidecarBuildError as error:
                raise RuntimeLibraryFetchError(error.code, str(error)) from error
            package_records.append(
                {
                    "formula": package["formula"],
                    "version": package["version"],
                    "repository": oci["repository"],
                    "manifestDigest": oci["manifestDigest"],
                    "blobSha256": oci["blobSha256"],
                    "libraries": libraries,
                }
            )
        return {
            "schemaVersion": 1,
            "kind": "ultra-fast-wbpp-macos-runtime-library-fetch",
            "targetTriple": policy["targetTriple"],
            "policySha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "packages": package_records,
        }
    except BaseException:
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="create-only bottle root")
    parser.add_argument("--policy", type=Path, default=MACOS14_RUNTIME_POLICY)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = materialize_runtime_libraries(
            arguments.output,
            policy_path=arguments.policy,
            timeout_seconds=arguments.timeout_seconds,
        )
    except RuntimeLibraryFetchError as error:
        print(
            json.dumps({"ok": False, "code": error.code, "error": str(error)}),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"ok": True, **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HttpResponse",
    "HttpTransport",
    "RuntimeLibraryFetchError",
    "UrllibTransport",
    "main",
    "materialize_runtime_libraries",
]
