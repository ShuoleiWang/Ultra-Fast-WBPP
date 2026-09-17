"""Auditable, user-initiated management of offline plate-solver catalogs.

Catalog files are never bundled or downloaded implicitly.  A checked manifest
binds every provider URL to an exact byte count and SHA-256.  Successful files
are published create-only, and immutable installed-set receipts let a solver
bind an Astrometry.net INDEXID to the actual bytes used later.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path, PurePath
import re
import stat
import sys
import tempfile
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from . import platform as platform_services


_CHUNK_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CATALOG_ID = re.compile(r"^[a-z0-9][a-z0-9.-]+$")
_SOLVER_INDEX_ID = re.compile(r"^astrometry\.net:index:(\d+):healpix:[^:]+:hpnside:[^:]+$")


class CatalogError(RuntimeError):
    """Stable catalog-management failure for CLI and future GUI callers."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CatalogArtifact:
    artifact_id: str
    url: str | None
    sha256: str | None
    size_bytes: int | None
    install_scope: str
    scale: int | None = None
    quad_min_arcminutes: float | None = None
    quad_max_arcminutes: float | None = None
    fov_min_degrees: float | None = None
    fov_max_degrees: float | None = None

    def serializable(self) -> dict[str, Any]:
        return {
            "artifactId": self.artifact_id,
            "url": self.url,
            "sha256": self.sha256,
            "sizeBytes": self.size_bytes,
            "installScope": self.install_scope,
            "scale": self.scale,
            "quadScaleArcminutes": (
                {"minimum": self.quad_min_arcminutes, "maximum": self.quad_max_arcminutes}
                if self.quad_min_arcminutes is not None
                else None
            ),
            "recommendedImageFieldOfViewDegrees": (
                {"minimum": self.fov_min_degrees, "maximum": self.fov_max_degrees}
                if self.fov_min_degrees is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class CatalogManifest:
    catalog_id: str
    provider: str
    version: str
    redistribution_status: str
    license: str
    citation: str
    acceptance_id: str
    terms_url: str
    terms_summary: str
    license_status: str
    requires_explicit_acceptance: bool
    allowed_download_origins: tuple[str, ...]
    artifacts: tuple[CatalogArtifact, ...]
    path: Path
    manifest_sha256: str

    @property
    def total_size_bytes(self) -> int:
        return sum(item.size_bytes or 0 for item in self.artifacts)

    def serializable(self) -> dict[str, Any]:
        return {
            "catalogId": self.catalog_id,
            "provider": self.provider,
            "version": self.version,
            "redistributionStatus": self.redistribution_status,
            "license": self.license,
            "citation": self.citation,
            "providerTerms": {
                "acceptanceId": self.acceptance_id,
                "url": self.terms_url,
                "summary": self.terms_summary,
                "licenseStatus": self.license_status,
                "requiresExplicitAcceptance": self.requires_explicit_acceptance,
            },
            "manifestSha256": self.manifest_sha256,
            "allowedDownloadOrigins": list(self.allowed_download_origins),
            "totalSizeBytes": self.total_size_bytes,
            "artifacts": [item.serializable() for item in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class DownloadResponse:
    status: int
    headers: Mapping[str, str]
    stream: BinaryIO
    final_url: str

    def close(self) -> None:
        self.stream.close()


class DownloadTransport(Protocol):
    def open(
        self,
        url: str,
        *,
        start: int,
        timeout_seconds: float,
        if_range: str | None = None,
    ) -> DownloadResponse:
        """Open a full or ranged HTTPS response."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        raise CatalogError(
            "CATALOG_REDIRECT_FORBIDDEN",
            f"catalog manifests bind an exact URL; HTTP {code} redirect is refused",
        )


class UrllibDownloadTransport:
    """Small standard-library HTTPS transport with redirect validation."""

    def open(
        self,
        url: str,
        *,
        start: int,
        timeout_seconds: float,
        if_range: str | None = None,
    ) -> DownloadResponse:
        headers = {"Accept-Encoding": "identity", "User-Agent": "Ultra-Fast-WBPP/0.1 catalog-installer"}
        if start:
            headers["Range"] = f"bytes={start}-"
            if if_range is None:
                raise CatalogError("CATALOG_RANGE_UNBOUND", "a resumed range request needs a strong ETag")
            headers["If-Range"] = if_range
        response = build_opener(_RejectRedirects()).open(
            Request(url, headers=headers), timeout=timeout_seconds
        )
        final_url = response.geturl()
        if final_url != url:
            response.close()
            raise CatalogError("CATALOG_REDIRECT_FORBIDDEN", "catalog response URL differs from its manifest")
        return DownloadResponse(
            status=int(getattr(response, "status", response.getcode())),
            headers={str(key).lower(): str(value) for key, value in response.headers.items()},
            stream=response,
            final_url=final_url,
        )


ProgressCallback = Callable[[Mapping[str, Any]], None]


@contextmanager
def _catalog_lock(root: Path) -> Iterable[None]:
    """Serialize writers without deleting a potentially live lock file."""

    lock_path = root / ".catalog-manager.lock"
    _regular_file(lock_path)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    locked = False
    try:
        if os.name == "posix":
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise CatalogError("CATALOG_INSTALL_BUSY", "another catalog writer is active") from error
            locked = True
        elif os.name == "nt":  # pragma: no cover - exercised by Windows CI
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise CatalogError("CATALOG_INSTALL_BUSY", "another catalog writer is active") from error
            locked = True
        else:  # pragma: no cover - only POSIX and Windows are supported
            raise CatalogError("CATALOG_LOCK_UNSUPPORTED", f"unsupported lock platform: {os.name}")
        yield
    finally:
        if locked:
            if os.name == "posix":
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
            elif os.name == "nt":  # pragma: no cover - exercised by Windows CI
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        os.close(descriptor)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_artifact_name(value: object) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise CatalogError("CATALOG_MANIFEST_INVALID", "artifactId must be a non-empty filename")
    if PurePath(value).name != value or "/" in value or "\\" in value or "\x00" in value:
        raise CatalogError("CATALOG_PATH_UNSAFE", f"unsafe artifactId: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise CatalogError("CATALOG_PATH_UNSAFE", f"non-portable artifactId: {value!r}")
    return value


def _download_origin(value: str) -> str:
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError as error:
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"invalid download URL: {value!r}") from error
    host = parsed.hostname
    if (
        parsed.scheme.lower() != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.fragment
    ):
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"unsafe download URL: {value!r}")
    lowered = host.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith(".localhost"):
        raise CatalogError("CATALOG_MANIFEST_INVALID", "localhost catalog origins are forbidden")
    try:
        ipaddress.ip_address(lowered)
    except ValueError:
        pass
    else:
        raise CatalogError("CATALOG_MANIFEST_INVALID", "IP-literal catalog origins are forbidden")
    rendered_host = f"[{lowered}]" if ":" in lowered else lowered
    return f"https://{rendered_host}"


def _parse_manifest(path: Path) -> CatalogManifest:
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise CatalogError("CATALOG_MANIFEST_UNSAFE", f"manifest is not a regular file: {path}")
        raw = path.read_bytes()
        value = json.loads(raw)
    except CatalogError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"cannot read {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schemaVersion") != 1:
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"{path} is not a version-1 manifest")
    catalog_id = value.get("catalogId")
    if not isinstance(catalog_id, str) or not _CATALOG_ID.fullmatch(catalog_id):
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"invalid catalogId in {path}")
    terms = value.get("providerTerms")
    if not isinstance(terms, dict):
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"{catalog_id} has no providerTerms")
    terms_url = terms.get("url")
    if not isinstance(terms_url, str) or urlparse(terms_url).scheme.lower() != "https":
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"{catalog_id} terms URL must use HTTPS")
    raw_artifacts = value.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"{catalog_id} has no artifacts")
    raw_origins = value.get("allowedDownloadOrigins")
    if not isinstance(raw_origins, list) or any(not isinstance(item, str) for item in raw_origins):
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"{catalog_id} has no download origin allowlist")
    allowed_origins_list: list[str] = []
    for item in raw_origins:
        parsed_origin = urlparse(item)
        if parsed_origin.path not in ("", "/") or parsed_origin.query or parsed_origin.fragment:
            raise CatalogError("CATALOG_MANIFEST_INVALID", f"download origin must not contain a path: {item}")
        allowed_origins_list.append(_download_origin(item))
    allowed_origins = tuple(allowed_origins_list)
    if len(set(allowed_origins)) != len(allowed_origins):
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"{catalog_id} has duplicate download origins")
    artifacts: list[CatalogArtifact] = []
    names: set[str] = set()
    for item in raw_artifacts:
        if not isinstance(item, dict):
            raise CatalogError("CATALOG_MANIFEST_INVALID", f"{catalog_id} has a malformed artifact")
        artifact_id = _validate_artifact_name(item.get("artifactId"))
        if artifact_id in names:
            raise CatalogError("CATALOG_MANIFEST_INVALID", f"duplicate artifact {artifact_id}")
        names.add(artifact_id)
        url = item.get("url")
        sha256 = item.get("sha256")
        size_bytes = item.get("sizeBytes")
        if url is not None:
            if not isinstance(url, str):
                raise CatalogError("CATALOG_MANIFEST_INVALID", f"{artifact_id} URL must use HTTPS")
            if _download_origin(url) not in allowed_origins:
                raise CatalogError(
                    "CATALOG_MANIFEST_INVALID",
                    f"{artifact_id} URL is outside allowedDownloadOrigins",
                )
            if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
                raise CatalogError("CATALOG_MANIFEST_INVALID", f"{artifact_id} has no exact SHA-256")
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 1:
                raise CatalogError("CATALOG_MANIFEST_INVALID", f"{artifact_id} has no exact byte count")
        scale = item.get("scale")
        quad = item.get("quadScaleArcminutes")
        fov = item.get("recommendedImageFieldOfViewDegrees")
        if quad is not None:
            if (
                not isinstance(quad, dict)
                or isinstance(quad.get("minimum"), bool)
                or isinstance(quad.get("maximum"), bool)
                or not isinstance(quad.get("minimum"), (int, float))
                or not isinstance(quad.get("maximum"), (int, float))
                or not math.isfinite(float(quad["minimum"]))
                or not math.isfinite(float(quad["maximum"]))
                or not 0 < float(quad["minimum"]) < float(quad["maximum"])
            ):
                raise CatalogError("CATALOG_MANIFEST_INVALID", f"{artifact_id} has invalid quad scale")
        if fov is not None:
            if (
                not isinstance(fov, dict)
                or isinstance(fov.get("minimum"), bool)
                or isinstance(fov.get("maximum"), bool)
                or not isinstance(fov.get("minimum"), (int, float))
                or not isinstance(fov.get("maximum"), (int, float))
                or not math.isfinite(float(fov["minimum"]))
                or not math.isfinite(float(fov["maximum"]))
                or not 0 < float(fov["minimum"]) < float(fov["maximum"])
            ):
                raise CatalogError("CATALOG_MANIFEST_INVALID", f"{artifact_id} has invalid FOV coverage")
        artifacts.append(
            CatalogArtifact(
                artifact_id=artifact_id,
                url=url if isinstance(url, str) else None,
                sha256=sha256 if isinstance(sha256, str) else None,
                size_bytes=size_bytes if isinstance(size_bytes, int) and not isinstance(size_bytes, bool) else None,
                install_scope=str(item.get("installScope", "external")),
                scale=scale if isinstance(scale, int) and not isinstance(scale, bool) else None,
                quad_min_arcminutes=float(quad["minimum"]) if isinstance(quad, dict) else None,
                quad_max_arcminutes=float(quad["maximum"]) if isinstance(quad, dict) else None,
                fov_min_degrees=float(fov["minimum"]) if isinstance(fov, dict) else None,
                fov_max_degrees=float(fov["maximum"]) if isinstance(fov, dict) else None,
            )
        )
    fields = ("provider", "version", "redistributionStatus", "license", "citation")
    if any(not isinstance(value.get(field), str) or not value[field] for field in fields):
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"{catalog_id} has missing metadata")
    acceptance_id = terms.get("acceptanceId")
    if not isinstance(acceptance_id, str) or not _CATALOG_ID.fullmatch(acceptance_id):
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"{catalog_id} has invalid acceptanceId")
    summary = terms.get("summary")
    license_status = terms.get("licenseStatus")
    if not isinstance(summary, str) or not summary.strip():
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"{catalog_id} has no provider terms summary")
    if license_status not in {"confirmed", "provider-specific-unresolved", "review-required"}:
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"{catalog_id} has invalid license status")
    if not isinstance(terms.get("requiresExplicitAcceptance"), bool):
        raise CatalogError("CATALOG_MANIFEST_INVALID", f"{catalog_id} has invalid acceptance policy")
    return CatalogManifest(
        catalog_id=catalog_id,
        provider=value["provider"],
        version=value["version"],
        redistribution_status=value["redistributionStatus"],
        license=value["license"],
        citation=value["citation"],
        acceptance_id=acceptance_id,
        terms_url=terms_url,
        terms_summary=summary,
        license_status=license_status,
        requires_explicit_acceptance=terms.get("requiresExplicitAcceptance") is True,
        allowed_download_origins=allowed_origins,
        artifacts=tuple(artifacts),
        path=path,
        manifest_sha256=_sha256_bytes(raw),
    )


def manifest_search_paths(
    manifest_dir: str | os.PathLike[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    if manifest_dir is not None:
        return (Path(manifest_dir).expanduser().absolute(),)
    env = os.environ if environment is None else environment
    paths: list[Path] = []
    configured = env.get("OPENASTROFLOW_CATALOG_MANIFEST_DIR")
    if configured:
        paths.append(Path(configured).expanduser().absolute())
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        paths.append(Path(bundle_root) / "resources" / "catalogs")
    paths.extend(
        (
            Path(__file__).resolve().parent / "catalog_manifests",
            Path(sys.prefix) / "share" / "openastroflow" / "catalogs",
            Path(__file__).resolve().parents[4] / "resources" / "catalogs",
            Path(sys.executable).resolve().parent / "resources" / "catalogs",
        )
    )
    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return tuple(unique)


def load_catalog_manifests(
    manifest_dir: str | os.PathLike[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[CatalogManifest, ...]:
    manifests: dict[str, CatalogManifest] = {}
    found_directory = False
    for directory in manifest_search_paths(manifest_dir, environment=environment):
        if not directory.is_dir():
            continue
        found_directory = True
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".schema.json"):
                continue
            manifest = _parse_manifest(path)
            previous = manifests.get(manifest.catalog_id)
            if previous is not None and previous.manifest_sha256 != manifest.manifest_sha256:
                raise CatalogError(
                    "CATALOG_MANIFEST_CONFLICT",
                    f"catalog {manifest.catalog_id} has conflicting checked manifests",
                )
            manifests.setdefault(manifest.catalog_id, manifest)
    if not found_directory:
        raise CatalogError("CATALOG_MANIFESTS_MISSING", "no catalog manifest directory is available")
    return tuple(sorted(manifests.values(), key=lambda item: item.catalog_id))


def get_catalog_manifest(
    catalog_id: str,
    manifest_dir: str | os.PathLike[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> CatalogManifest:
    for manifest in load_catalog_manifests(manifest_dir, environment=environment):
        if manifest.catalog_id == catalog_id:
            return manifest
    raise CatalogError("CATALOG_UNKNOWN", f"unknown catalog: {catalog_id}")


def default_catalog_root(
    *,
    environment: Mapping[str, str] | None = None,
    home: str | os.PathLike[str] | None = None,
) -> Path:
    base = platform_services.current().data_root(environment=environment, home=home)
    return (base / "catalogs" / "astrometry-net").absolute()


def _ensure_secure_directory(path: Path, *, create: bool) -> Path:
    path = path.expanduser().absolute()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise CatalogError("CATALOG_DIRECTORY_MISSING", f"catalog directory does not exist: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CatalogError("CATALOG_DIRECTORY_UNSAFE", f"catalog directory is not a real directory: {path}")
    return path


def _regular_file(path: Path) -> os.stat_result | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CatalogError("CATALOG_FILE_UNSAFE", f"expected a non-symlink regular file: {path}")
    return metadata


def _hash_regular_file(path: Path, *, expected_size: int | None = None) -> tuple[int, str]:
    before = _regular_file(path)
    if before is None:
        raise CatalogError("CATALOG_ARTIFACT_MISSING", f"missing catalog artifact: {path.name}")
    if expected_size is not None and before.st_size != expected_size:
        raise CatalogError(
            "CATALOG_SIZE_MISMATCH",
            f"{path.name} is {before.st_size} bytes, expected {expected_size}",
        )
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            while block := stream.read(_CHUNK_BYTES):
                digest.update(block)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or not stat.S_ISREG(after.st_mode):
        raise CatalogError("CATALOG_FILE_CHANGED", f"catalog artifact changed while hashing: {path.name}")
    return before.st_size, digest.hexdigest()


def _hash_regular_file_snapshot(
    path: Path,
    *,
    expected_size: int | None = None,
) -> dict[str, Any]:
    """Hash one regular file while binding the digest to its file identity.

    ``_hash_regular_file`` is sufficient for installation checks.  A solver
    needs a stronger before/after token as well: replacing an index with an
    identical byte-for-byte copy while ``solve-field`` is running is still
    catalog drift.  This helper verifies the descriptor and directory entry
    refer to the same file before and after hashing and returns the stat token
    used by the strict solver adapter.
    """

    before = _regular_file(path)
    if before is None:
        raise CatalogError("CATALOG_ARTIFACT_MISSING", f"missing catalog artifact: {path.name}")
    if expected_size is not None and before.st_size != expected_size:
        raise CatalogError(
            "CATALOG_SIZE_MISMATCH",
            f"{path.name} is {before.st_size} bytes, expected {expected_size}",
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        opened_identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        if before_identity != opened_identity or not stat.S_ISREG(opened.st_mode):
            raise CatalogError("CATALOG_FILE_CHANGED", f"catalog artifact changed before hashing: {path.name}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            while block := stream.read(_CHUNK_BYTES):
                digest.update(block)
            after_descriptor = os.fstat(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = path.lstat()
    after_descriptor_identity = (
        after_descriptor.st_dev,
        after_descriptor.st_ino,
        after_descriptor.st_size,
        after_descriptor.st_mtime_ns,
    )
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if (
        before_identity != after_descriptor_identity
        or before_identity != after_identity
        or not stat.S_ISREG(after.st_mode)
    ):
        raise CatalogError("CATALOG_FILE_CHANGED", f"catalog artifact changed while hashing: {path.name}")
    return {
        "sizeBytes": before.st_size,
        "sha256": digest.hexdigest(),
        "statIdentity": {
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtimeNs": before.st_mtime_ns,
            "sizeBytes": before.st_size,
        },
    }


def recommended_artifacts(manifest: CatalogManifest, field_of_view_degrees: float) -> tuple[CatalogArtifact, ...]:
    if (
        isinstance(field_of_view_degrees, bool)
        or not isinstance(field_of_view_degrees, (int, float))
        or not math.isfinite(float(field_of_view_degrees))
        or not 0 < field_of_view_degrees <= 360
    ):
        raise CatalogError("CATALOG_FOV_INVALID", "field of view must be in (0, 360] degrees")
    image_arcminutes = field_of_view_degrees * 60.0
    desired_min = image_arcminutes * 0.10
    desired_max = image_arcminutes
    selected = tuple(
        artifact
        for artifact in manifest.artifacts
        if artifact.quad_min_arcminutes is not None
        and artifact.quad_max_arcminutes is not None
        and artifact.quad_max_arcminutes >= desired_min
        # A quad equal to the complete image width leaves no matching margin;
        # the provider's worked 1-degree example likewise stops at 4109 rather
        # than including 4110's 60-arcminute lower boundary.
        and artifact.quad_min_arcminutes < desired_max
    )
    if not selected:
        raise CatalogError(
            "CATALOG_FOV_UNCOVERED",
            f"{manifest.catalog_id} has no checked scale for a {field_of_view_degrees:g}-degree field",
        )
    return selected


def _selected_artifacts(
    manifest: CatalogManifest,
    artifact_ids: Sequence[str] | None,
    field_of_view_degrees: float | None,
) -> tuple[CatalogArtifact, ...]:
    if artifact_ids and field_of_view_degrees is not None:
        raise CatalogError("CATALOG_SELECTION_CONFLICT", "use artifact IDs or field of view, not both")
    if field_of_view_degrees is not None:
        return recommended_artifacts(manifest, field_of_view_degrees)
    if not artifact_ids:
        return manifest.artifacts
    wanted = {_validate_artifact_name(item) for item in artifact_ids}
    selected = tuple(item for item in manifest.artifacts if item.artifact_id in wanted)
    missing = sorted(wanted - {item.artifact_id for item in selected})
    if missing:
        raise CatalogError("CATALOG_ARTIFACT_UNKNOWN", f"unknown artifacts: {', '.join(missing)}")
    return selected


def _atomic_create(path: Path, payload: bytes) -> None:
    if _regular_file(path) is not None:
        existing = path.read_bytes()
        if existing == payload:
            return
        raise CatalogError("CATALOG_PUBLICATION_EXISTS", f"refusing to replace existing file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise CatalogError("CATALOG_PUBLICATION_RACE", f"destination appeared during publication: {path}") from error
        temporary.unlink()
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def _resume_paths(root: Path, artifact: CatalogArtifact) -> tuple[Path, Path]:
    assert artifact.sha256 is not None
    staging = _ensure_secure_directory(root / ".downloads", create=True)
    stem = f".{artifact.artifact_id}.{artifact.sha256[:16]}"
    return staging / f"{stem}.partial", staging / f"{stem}.json"


def _strong_etag(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped.startswith("W/") or not re.fullmatch(r'"[^"\r\n]+"', stripped):
        return None
    return stripped


def _replace_owned_file(path: Path, payload: bytes) -> None:
    if _regular_file(path) is None:
        raise CatalogError("CATALOG_RESUME_STATE_INVALID", f"missing owned state file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_resume_state(
    state_path: Path, manifest: CatalogManifest, artifact: CatalogArtifact
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "schemaVersion": 1,
        "catalogId": manifest.catalog_id,
        "manifestSha256": manifest.manifest_sha256,
        "artifactId": artifact.artifact_id,
        "url": artifact.url,
        "sizeBytes": artifact.size_bytes,
        "sha256": artifact.sha256,
        "entityTag": None,
    }
    payload = _canonical_json(expected) + b"\n"
    existing = _regular_file(state_path)
    if existing is None:
        _atomic_create(state_path, payload)
        return expected
    try:
        actual = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError("CATALOG_RESUME_STATE_INVALID", f"invalid resume state: {state_path}") from error
    if not isinstance(actual, dict) or any(
        actual.get(key) != value for key, value in expected.items() if key != "entityTag"
    ):
        raise CatalogError("CATALOG_RESUME_STATE_MISMATCH", f"resume state does not match {artifact.artifact_id}")
    entity_tag = actual.get("entityTag")
    if entity_tag is not None and _strong_etag(entity_tag) != entity_tag:
        raise CatalogError("CATALOG_RESUME_STATE_INVALID", f"resume state has an invalid ETag: {state_path}")
    return actual


def _set_resume_etag(state_path: Path, state: Mapping[str, Any], entity_tag: str | None) -> dict[str, Any]:
    updated = {**state, "entityTag": entity_tag}
    _replace_owned_file(state_path, _canonical_json(updated) + b"\n")
    return updated


def _open_partial(path: Path, *, truncate: bool) -> BinaryIO:
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    flags |= os.O_TRUNC if truncate else os.O_APPEND
    descriptor = os.open(path, flags, 0o600)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise CatalogError("CATALOG_FILE_UNSAFE", f"resume target is not a regular file: {path}")
    return os.fdopen(descriptor, "wb" if truncate else "ab")


def _parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if match is None:
        return None
    return tuple(int(item) for item in match.groups())  # type: ignore[return-value]


def _download_artifact(
    manifest: CatalogManifest,
    artifact: CatalogArtifact,
    root: Path,
    *,
    transport: DownloadTransport,
    timeout_seconds: float,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    if artifact.url is None or artifact.sha256 is None or artifact.size_bytes is None:
        raise CatalogError("CATALOG_NOT_DOWNLOADABLE", f"{artifact.artifact_id} has no immutable HTTPS artifact")
    destination = root / artifact.artifact_id
    existing = _regular_file(destination)
    if existing is not None:
        size, digest = _hash_regular_file(destination, expected_size=artifact.size_bytes)
        if digest != artifact.sha256:
            raise CatalogError("CATALOG_HASH_MISMATCH", f"existing {artifact.artifact_id} has the wrong SHA-256")
        return {
            "artifactId": artifact.artifact_id,
            "status": "ALREADY_INSTALLED",
            "sizeBytes": size,
            "sha256": digest,
            "sourceUrl": artifact.url,
            "entityTag": None,
        }

    partial, state_path = _resume_paths(root, artifact)
    resume_state = _write_resume_state(state_path, manifest, artifact)
    partial_metadata = _regular_file(partial)
    offset = 0 if partial_metadata is None else partial_metadata.st_size
    if offset > artifact.size_bytes:
        raise CatalogError("CATALOG_RESUME_OVERSIZE", f"partial file is larger than {artifact.artifact_id}")
    resume_etag = _strong_etag(resume_state.get("entityTag"))
    if 0 < offset < artifact.size_bytes and resume_etag is None:
        # Without a strong validator, a byte range could splice two versions.
        # Keep the partial until the new full response opens successfully, then
        # truncate it under the catalog writer lock.
        offset = 0
    if progress:
        progress({"event": "catalog-download", "artifactId": artifact.artifact_id, "downloadedBytes": offset, "sizeBytes": artifact.size_bytes})
    if offset < artifact.size_bytes:
        try:
            response = transport.open(
                artifact.url,
                start=offset,
                timeout_seconds=timeout_seconds,
                if_range=resume_etag if offset else None,
            )
        except CatalogError:
            raise
        except Exception as error:
            raise CatalogError(
                "CATALOG_DOWNLOAD_FAILED",
                f"could not download {artifact.artifact_id}: {type(error).__name__}: {error}",
            ) from error
        try:
            if response.final_url != artifact.url:
                raise CatalogError("CATALOG_REDIRECT_FORBIDDEN", "catalog response URL differs from its manifest")
            response_etag = _strong_etag(response.headers.get("etag"))
            append = offset > 0 and response.status == 206
            if append:
                content_range = _parse_content_range(response.headers.get("content-range"))
                if (
                    response_etag != resume_etag
                    or content_range is None
                    or content_range[0] != offset
                    or content_range[1] != artifact.size_bytes - 1
                    or content_range[2] != artifact.size_bytes
                ):
                    raise CatalogError("CATALOG_RANGE_INVALID", f"provider returned an invalid range for {artifact.artifact_id}")
            elif response.status == 200:
                offset = 0
                resume_state = _set_resume_etag(state_path, resume_state, response_etag)
            else:
                raise CatalogError("CATALOG_HTTP_STATUS", f"provider returned HTTP {response.status} for {artifact.artifact_id}")
            content_length = response.headers.get("content-length")
            expected_remaining = artifact.size_bytes - offset
            if content_length is not None:
                try:
                    actual_length = int(content_length)
                except ValueError as error:
                    raise CatalogError("CATALOG_HTTP_LENGTH_INVALID", "provider returned an invalid Content-Length") from error
                if actual_length != expected_remaining:
                    raise CatalogError(
                        "CATALOG_HTTP_LENGTH_MISMATCH",
                        f"provider announced {actual_length} bytes, expected {expected_remaining}",
                    )
            downloaded = offset
            with _open_partial(partial, truncate=not append) as output:
                while block := response.stream.read(_CHUNK_BYTES):
                    downloaded += len(block)
                    if downloaded > artifact.size_bytes:
                        raise CatalogError("CATALOG_DOWNLOAD_OVERSIZE", f"provider exceeded size for {artifact.artifact_id}")
                    output.write(block)
                    if progress:
                        progress({"event": "catalog-download", "artifactId": artifact.artifact_id, "downloadedBytes": downloaded, "sizeBytes": artifact.size_bytes})
                output.flush()
                os.fsync(output.fileno())
            if downloaded != artifact.size_bytes:
                raise CatalogError(
                    "CATALOG_DOWNLOAD_INCOMPLETE",
                    f"downloaded {downloaded} of {artifact.size_bytes} bytes for {artifact.artifact_id}; rerun to resume",
                )
        finally:
            response.close()
    size, digest = _hash_regular_file(partial, expected_size=artifact.size_bytes)
    if digest != artifact.sha256:
        invalid = partial.with_name(f"{partial.name}.invalid-{digest[:12]}")
        try:
            os.link(partial, invalid, follow_symlinks=False)
            partial.unlink()
            state_path.unlink()
        except FileExistsError:
            # A previous forensic copy is already present; leave the current
            # partial untouched rather than overwriting either file.
            pass
        raise CatalogError("CATALOG_HASH_MISMATCH", f"downloaded {artifact.artifact_id} failed SHA-256 verification")
    try:
        os.link(partial, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise CatalogError("CATALOG_PUBLICATION_RACE", f"destination appeared during publication: {destination}") from error
    partial.unlink()
    state_path.unlink()
    try:
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass
    return {
        "artifactId": artifact.artifact_id,
        "status": "INSTALLED",
        "sizeBytes": size,
        "sha256": digest,
        "sourceUrl": artifact.url,
        "entityTag": resume_state.get("entityTag"),
    }


def _config_payload(root: Path) -> bytes:
    root_text = str(root)
    if "\n" in root_text or "\r" in root_text or "\x00" in root_text:
        raise CatalogError("CATALOG_CONFIG_PATH_UNSAFE", "catalog path cannot be represented in astrometry.cfg")
    return (
        "# Generated by Ultra-Fast WBPP catalog manager; do not edit in place.\n"
        f"add_path {root_text}\n"
        "autoindex\n"
        "inparallel\n"
    ).encode("utf-8")


def _compatible_config(payload: bytes, root: Path) -> bool:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return False
    commands = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return commands == [f"add_path {root}", "autoindex", "inparallel"]


def write_astrometry_config(root: str | os.PathLike[str]) -> dict[str, Any]:
    catalog_root = _ensure_secure_directory(Path(root), create=True)
    path = catalog_root / "astrometry.cfg"
    payload = _config_payload(catalog_root)
    existing = _regular_file(path)
    if existing is not None:
        actual = path.read_bytes()
        if not _compatible_config(actual, catalog_root):
            raise CatalogError(
                "CATALOG_CONFIG_CONFLICT",
                f"existing astrometry.cfg is not the exact safe catalog configuration: {path}",
            )
        return {"path": str(path), "sha256": _sha256_bytes(actual), "sizeBytes": len(actual)}
    _atomic_create(path, payload)
    return {"path": str(path), "sha256": _sha256_bytes(payload), "sizeBytes": len(payload)}


def _installed_set_core(
    manifest: CatalogManifest,
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sources = {item.artifact_id: item.url for item in manifest.artifacts}
    return {
        "schemaVersion": 1,
        "catalogId": manifest.catalog_id,
        "manifestSha256": manifest.manifest_sha256,
        "artifacts": [
            {
                "artifactId": str(item["artifactId"]),
                "relativePath": str(item["artifactId"]),
                "sourceUrl": sources[str(item["artifactId"])],
                "sizeBytes": int(item["sizeBytes"]),
                "sha256": str(item["sha256"]),
            }
            for item in sorted(artifacts, key=lambda value: str(value["artifactId"]))
        ],
    }


def write_installed_set_receipt(
    manifest: CatalogManifest,
    root: str | os.PathLike[str],
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    catalog_root = _ensure_secure_directory(Path(root), create=False)
    expected = {item.artifact_id: item for item in manifest.artifacts}
    verified: list[dict[str, Any]] = []
    transport_evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for supplied in artifacts:
        artifact_id = _validate_artifact_name(supplied.get("artifactId"))
        artifact = expected.get(artifact_id)
        if artifact is None or artifact_id in seen:
            raise CatalogError("CATALOG_RECEIPT_ARTIFACT_INVALID", f"invalid receipt artifact: {artifact_id}")
        seen.add(artifact_id)
        if supplied.get("sizeBytes") != artifact.size_bytes or supplied.get("sha256") != artifact.sha256:
            raise CatalogError("CATALOG_RECEIPT_ARTIFACT_INVALID", f"receipt metadata disagrees with {artifact_id}")
        size, digest = _hash_regular_file(catalog_root / artifact_id, expected_size=artifact.size_bytes)
        if digest != artifact.sha256:
            raise CatalogError("CATALOG_HASH_MISMATCH", f"{artifact_id} changed before receipt publication")
        verified.append({"artifactId": artifact_id, "sizeBytes": size, "sha256": digest})
        transport_evidence.append(
            {
                "artifactId": artifact_id,
                "sourceUrl": artifact.url,
                "entityTag": supplied.get("entityTag") if _strong_etag(supplied.get("entityTag")) else None,
                "method": (
                    "provider-download"
                    if supplied.get("status") == "INSTALLED"
                    else "existing-file-verification"
                ),
            }
        )
    if not verified:
        raise CatalogError("CATALOG_RECEIPT_EMPTY", "an installed-set receipt needs at least one artifact")
    config = write_astrometry_config(catalog_root)
    core = _installed_set_core(manifest, verified)
    identity = _sha256_bytes(_canonical_json(core))
    receipt = {
        **core,
        "installedSetIdentity": identity,
        "config": {
            "relativePath": "astrometry.cfg",
            "sizeBytes": config["sizeBytes"],
            "sha256": config["sha256"],
        },
        "provider": manifest.provider,
        "version": manifest.version,
        "citation": manifest.citation,
        "licenseStatus": manifest.license_status,
        "acceptanceId": manifest.acceptance_id,
        "transportEvidence": sorted(transport_evidence, key=lambda item: item["artifactId"]),
    }
    path = catalog_root / f"installed-set-{manifest.catalog_id}-{identity[:16]}.json"
    if _regular_file(path) is not None:
        try:
            existing_receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CatalogError("CATALOG_RECEIPT_INVALID", f"cannot read existing receipt: {path}") from error
        if (
            not isinstance(existing_receipt, dict)
            or existing_receipt.get("installedSetIdentity") != identity
            or existing_receipt.get("manifestSha256") != manifest.manifest_sha256
            or existing_receipt.get("artifacts") != core["artifacts"]
        ):
            raise CatalogError("CATALOG_RECEIPT_CONFLICT", f"existing receipt does not match its identity: {path}")
        return {**existing_receipt, "receiptPath": str(path)}
    _atomic_create(path, _canonical_json(receipt) + b"\n")
    return {**receipt, "receiptPath": str(path)}


def install_catalog(
    catalog_id: str,
    *,
    accepted_terms_id: str | None,
    catalog_root: str | os.PathLike[str] | None = None,
    manifest_dir: str | os.PathLike[str] | None = None,
    artifact_ids: Sequence[str] | None = None,
    field_of_view_degrees: float | None = None,
    transport: DownloadTransport | None = None,
    timeout_seconds: float = 60.0,
    progress: ProgressCallback | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    manifest = get_catalog_manifest(catalog_id, manifest_dir, environment=environment)
    if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise CatalogError("CATALOG_TIMEOUT_INVALID", "download timeout must be finite and positive")
    if manifest.requires_explicit_acceptance and accepted_terms_id != manifest.acceptance_id:
        raise CatalogError(
            "CATALOG_TERMS_NOT_ACCEPTED",
            f"review {manifest.terms_url} and pass the exact acceptance ID {manifest.acceptance_id!r}",
        )
    if manifest.redistribution_status not in {"approved", "user-download-required"}:
        raise CatalogError("CATALOG_DOWNLOAD_FORBIDDEN", f"{catalog_id} is not approved for managed download")
    selected = _selected_artifacts(manifest, artifact_ids, field_of_view_degrees)
    if any(item.install_scope != "user" or item.url is None for item in selected):
        raise CatalogError("CATALOG_NOT_DOWNLOADABLE", f"{catalog_id} contains external-only artifacts")
    root = _ensure_secure_directory(
        Path(catalog_root) if catalog_root is not None else default_catalog_root(environment=environment),
        create=True,
    )
    client = transport or UrllibDownloadTransport()
    with _catalog_lock(root):
        results = [
            _download_artifact(
                manifest,
                artifact,
                root,
                transport=client,
                timeout_seconds=timeout_seconds,
                progress=progress,
            )
            for artifact in selected
        ]
        receipt = write_installed_set_receipt(manifest, root, results)
    return {
        "ok": True,
        "catalogId": catalog_id,
        "catalogRoot": str(root),
        "termsAcceptance": {
            "acceptanceId": manifest.acceptance_id,
            "termsUrl": manifest.terms_url,
            "licenseStatus": manifest.license_status,
            "explicit": True,
        },
        "artifacts": results,
        "installedSet": receipt,
    }


def verify_catalog(
    catalog_id: str,
    *,
    catalog_root: str | os.PathLike[str] | None = None,
    manifest_dir: str | os.PathLike[str] | None = None,
    artifact_ids: Sequence[str] | None = None,
    field_of_view_degrees: float | None = None,
    write_configuration: bool = False,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    manifest = get_catalog_manifest(catalog_id, manifest_dir, environment=environment)
    selected = _selected_artifacts(manifest, artifact_ids, field_of_view_degrees)
    root = _ensure_secure_directory(
        Path(catalog_root) if catalog_root is not None else default_catalog_root(environment=environment),
        create=False,
    )
    guard = _catalog_lock(root) if write_configuration else nullcontext()
    with guard:
        results: list[dict[str, Any]] = []
        for artifact in selected:
            path = root / artifact.artifact_id
            try:
                size, digest = _hash_regular_file(path, expected_size=artifact.size_bytes)
                verified = digest == artifact.sha256
                error = None if verified else "SHA256_MISMATCH"
            except CatalogError as failure:
                size, digest, verified, error = 0, None, False, failure.code
            results.append(
                {
                    "artifactId": artifact.artifact_id,
                    "path": str(path),
                    "sizeBytes": size,
                    "sha256": digest,
                    "verified": verified,
                    "error": error,
                }
            )
        complete = bool(results) and all(item["verified"] for item in results)
        receipt = None
        if complete and write_configuration:
            receipt = write_installed_set_receipt(manifest, root, results)
    return {
        "ok": complete,
        "catalogId": catalog_id,
        "catalogRoot": str(root),
        "manifestSha256": manifest.manifest_sha256,
        "artifacts": results,
        "installedSet": receipt,
    }


def catalog_list(
    *,
    catalog_root: str | os.PathLike[str] | None = None,
    manifest_dir: str | os.PathLike[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(catalog_root) if catalog_root is not None else default_catalog_root(environment=environment)
    entries: list[dict[str, Any]] = []
    for manifest in load_catalog_manifests(manifest_dir, environment=environment):
        installed = 0
        for artifact in manifest.artifacts:
            try:
                metadata = _regular_file(root / artifact.artifact_id)
                if metadata is not None and artifact.size_bytes == metadata.st_size:
                    installed += 1
            except CatalogError:
                pass
        entries.append(
            {
                **manifest.serializable(),
                "installedArtifactsBySize": installed,
                "artifactCount": len(manifest.artifacts),
                "fullyInstalledBySize": installed == len(manifest.artifacts),
            }
        )
    return {"schemaVersion": 1, "catalogRoot": str(root.absolute()), "catalogs": entries}


def catalog_doctor(
    *,
    catalog_root: str | os.PathLike[str] | None = None,
    manifest_dir: str | os.PathLike[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(catalog_root) if catalog_root is not None else default_catalog_root(environment=environment)
    if not root.exists():
        return {
            "schemaVersion": 1,
            "ok": False,
            "catalogRoot": str(root.absolute()),
            "config": {"present": False, "valid": False},
            "catalogs": [],
            "installedSetBindingReady": False,
            "installedSets": [],
            "message": "No catalog directory exists; use catalog install with explicit provider-term acceptance.",
        }
    root = _ensure_secure_directory(root, create=False)
    config = root / "astrometry.cfg"
    config_valid = False
    try:
        config_valid = _regular_file(config) is not None and _compatible_config(config.read_bytes(), root)
    except CatalogError:
        config_valid = False
    reports: list[dict[str, Any]] = []
    for manifest in load_catalog_manifests(manifest_dir, environment=environment):
        if any(item.url is not None for item in manifest.artifacts):
            reports.append(
                verify_catalog(
                    manifest.catalog_id,
                    catalog_root=root,
                    manifest_dir=manifest_dir,
                    environment=environment,
                )
            )
    receipts = verified_installed_set_identities(
        catalog_root=root,
        manifest_dir=manifest_dir,
        environment=environment,
    )
    return {
        "schemaVersion": 1,
        "ok": config_valid and any(report["ok"] for report in reports),
        "catalogRoot": str(root),
        "config": {"path": str(config), "present": config.exists(), "valid": config_valid},
        "catalogs": reports,
        "installedSetBindingReady": bool(receipts),
        "installedSets": [
            {
                "catalogId": receipt["catalogId"],
                "installedSetIdentity": receipt["installedSetIdentity"],
                "artifactCount": len(receipt["artifacts"]),
            }
            for receipt in receipts
        ],
        "message": "At least one checked catalog set and the generated solver config are ready."
        if config_valid and any(report["ok"] for report in reports)
        else "Catalog/index coverage is incomplete or the solver config is not generated.",
    }


def remove_catalog_plan(
    catalog_id: str,
    *,
    catalog_root: str | os.PathLike[str] | None = None,
    manifest_dir: str | os.PathLike[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    manifest = get_catalog_manifest(catalog_id, manifest_dir, environment=environment)
    root = Path(catalog_root) if catalog_root is not None else default_catalog_root(environment=environment)
    all_manifests = load_catalog_manifests(manifest_dir, environment=environment)
    shared_by = {
        artifact.artifact_id: sorted(
            other.catalog_id
            for other in all_manifests
            if other.catalog_id != catalog_id
            for other_artifact in other.artifacts
            if other_artifact.artifact_id == artifact.artifact_id
        )
        for artifact in manifest.artifacts
    }
    candidates: list[dict[str, Any]] = []
    retained_shared: list[dict[str, Any]] = []
    reclaimable = 0
    for artifact in manifest.artifacts:
        path = root / artifact.artifact_id
        metadata = _regular_file(path)
        if metadata is not None:
            if shared_by[artifact.artifact_id]:
                retained_shared.append(
                    {
                        "path": str(path.absolute()),
                        "kind": "catalog-artifact",
                        "sizeBytes": metadata.st_size,
                        "sharedWithCatalogs": shared_by[artifact.artifact_id],
                    }
                )
            else:
                candidates.append({"path": str(path.absolute()), "kind": "catalog-artifact", "sizeBytes": metadata.st_size})
                reclaimable += metadata.st_size
    if root.exists():
        for receipt in sorted(root.glob(f"installed-set-{catalog_id}-*.json")):
            metadata = _regular_file(receipt)
            if metadata is not None:
                candidates.append({"path": str(receipt.absolute()), "kind": "installed-set-receipt", "sizeBytes": metadata.st_size})
                reclaimable += metadata.st_size
    return {
        "schemaVersion": 1,
        "operation": "REMOVE_PLAN_ONLY",
        "executed": False,
        "catalogId": catalog_id,
        "catalogRoot": str(root.absolute()),
        "reclaimableBytes": reclaimable,
        "paths": candidates,
        "retainedSharedPaths": retained_shared,
        "note": "No files were removed. Shared astrometry.cfg and artifacts from other manifests are intentionally excluded.",
    }


def _verified_receipts(
    *,
    catalog_root: Path,
    manifest_dir: str | os.PathLike[str] | None,
    environment: Mapping[str, str] | None,
) -> tuple[dict[str, Any], ...]:
    manifests = {item.catalog_id: item for item in load_catalog_manifests(manifest_dir, environment=environment)}
    receipts: list[dict[str, Any]] = []
    if not catalog_root.exists():
        return ()
    root = _ensure_secure_directory(catalog_root, create=False)
    for path in sorted(root.glob("installed-set-*.json")):
        try:
            _regular_file(path)
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("schemaVersion") != 1:
                continue
            manifest = manifests.get(value.get("catalogId"))
            if manifest is None or value.get("manifestSha256") != manifest.manifest_sha256:
                continue
            if (
                value.get("provider") != manifest.provider
                or value.get("version") != manifest.version
                or value.get("citation") != manifest.citation
                or value.get("licenseStatus") != manifest.license_status
                or value.get("acceptanceId") != manifest.acceptance_id
            ):
                continue
            artifacts = value.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                continue
            core = _installed_set_core(manifest, artifacts)
            identity = _sha256_bytes(_canonical_json(core))
            if value.get("installedSetIdentity") != identity:
                continue
            config = value.get("config")
            if (
                not isinstance(config, dict)
                or config.get("relativePath") != "astrometry.cfg"
                or isinstance(config.get("sizeBytes"), bool)
                or not isinstance(config.get("sizeBytes"), int)
                or not isinstance(config.get("sha256"), str)
            ):
                continue
            config_size, config_digest = _hash_regular_file(
                root / "astrometry.cfg", expected_size=config["sizeBytes"]
            )
            if config_size != config["sizeBytes"] or config_digest != config["sha256"]:
                continue
            transport_evidence = value.get("transportEvidence")
            if not isinstance(transport_evidence, list) or len(transport_evidence) != len(artifacts):
                continue
            evidence_by_id = {
                item.get("artifactId"): item
                for item in transport_evidence
                if isinstance(item, dict)
            }
            valid = True
            for item in artifacts:
                if not isinstance(item, dict):
                    valid = False
                    break
                artifact_id = _validate_artifact_name(item.get("artifactId"))
                if item.get("relativePath") != artifact_id:
                    valid = False
                    break
                size, digest = _hash_regular_file(root / artifact_id, expected_size=int(item["sizeBytes"]))
                if digest != item.get("sha256") or size != item.get("sizeBytes"):
                    valid = False
                    break
                evidence = evidence_by_id.get(artifact_id)
                expected_artifact = next(
                    (candidate for candidate in manifest.artifacts if candidate.artifact_id == artifact_id),
                    None,
                )
                if (
                    expected_artifact is None
                    or not isinstance(evidence, dict)
                    or evidence.get("sourceUrl") != expected_artifact.url
                    or evidence.get("method")
                    not in {"provider-download", "existing-file-verification"}
                    or (
                        evidence.get("entityTag") is not None
                        and _strong_etag(evidence.get("entityTag")) != evidence.get("entityTag")
                    )
                ):
                    valid = False
                    break
            if valid:
                receipts.append({**value, "receiptPath": str(path)})
        except (CatalogError, OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return tuple(receipts)


def verified_installed_set_identities(
    *,
    catalog_root: str | os.PathLike[str] | None = None,
    manifest_dir: str | os.PathLike[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    root = Path(catalog_root) if catalog_root is not None else default_catalog_root(environment=environment)
    return _verified_receipts(catalog_root=root, manifest_dir=manifest_dir, environment=environment)


def installed_set_snapshot_for_solver_config(
    config_path: str | os.PathLike[str],
    *,
    manifest_dir: str | os.PathLike[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify and snapshot every managed index reachable through one config.

    Strict solving intentionally accepts only the catalog-manager generated
    ``astrometry.cfg`` located beside immutable installed-set receipts.  The
    returned paths and stat tokens are internal runtime state; callers must not
    copy them into a shareable scientific receipt.
    """

    supplied = Path(config_path).expanduser().absolute()
    config_metadata = _regular_file(supplied)
    if config_metadata is None:
        raise CatalogError("CATALOG_CONFIG_UNMANAGED", "the solver config is missing")
    try:
        resolved = supplied.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CatalogError("CATALOG_CONFIG_UNMANAGED", "the solver config cannot be resolved") from error
    root = _ensure_secure_directory(resolved.parent, create=False)
    expected_config = root / "astrometry.cfg"
    if resolved != expected_config or supplied.is_symlink():
        raise CatalogError(
            "CATALOG_CONFIG_UNMANAGED",
            "strict solving requires the catalog-manager astrometry.cfg beside its receipts",
        )
    receipts = _verified_receipts(
        catalog_root=root,
        manifest_dir=manifest_dir,
        environment=environment,
    )
    if not receipts:
        raise CatalogError(
            "CATALOG_INSTALLED_SET_UNBOUND",
            "the solver config is not covered by a verified installed-set receipt",
        )
    config_snapshot = _hash_regular_file_snapshot(expected_config, expected_size=config_metadata.st_size)
    artifacts: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        receipt_config = receipt["config"]
        if (
            receipt_config.get("relativePath") != "astrometry.cfg"
            or receipt_config.get("sizeBytes") != config_snapshot["sizeBytes"]
            or receipt_config.get("sha256") != config_snapshot["sha256"]
        ):
            raise CatalogError(
                "CATALOG_CONFIG_UNBOUND",
                "the solver config does not match its installed-set receipt",
            )
        for artifact in receipt["artifacts"]:
            artifact_id = _validate_artifact_name(artifact.get("artifactId"))
            snapshot = _hash_regular_file_snapshot(
                root / artifact_id,
                expected_size=int(artifact["sizeBytes"]),
            )
            if snapshot["sha256"] != artifact.get("sha256"):
                raise CatalogError(
                    "CATALOG_HASH_MISMATCH",
                    f"{artifact_id} does not match its installed-set receipt",
                )
            current = {
                "artifactId": artifact_id,
                "relativeName": artifact_id,
                "sizeBytes": snapshot["sizeBytes"],
                "sha256": snapshot["sha256"],
                "statIdentity": snapshot["statIdentity"],
            }
            previous = artifacts.get(artifact_id)
            if previous is not None and previous != current:
                raise CatalogError(
                    "CATALOG_INDEX_AMBIGUOUS",
                    f"conflicting installed-set receipts describe {artifact_id}",
                )
            artifacts[artifact_id] = current
    return {
        "catalogRoot": str(root),
        "configPath": str(expected_config),
        "config": {
            "relativeName": "astrometry.cfg",
            **config_snapshot,
        },
        "receipts": [
            {
                "catalogId": receipt["catalogId"],
                "installedSetIdentity": receipt["installedSetIdentity"],
                "manifestSha256": receipt["manifestSha256"],
                "artifactIds": sorted(item["artifactId"] for item in receipt["artifacts"]),
            }
            for receipt in receipts
        ],
        "artifacts": [artifacts[name] for name in sorted(artifacts)],
    }


def verify_installed_set_snapshot(
    snapshot: Mapping[str, Any],
    *,
    manifest_dir: str | os.PathLike[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Re-hash a pre-solve snapshot and reject any stat or byte drift."""

    config_path = snapshot.get("configPath")
    if not isinstance(config_path, str):
        raise CatalogError("CATALOG_SNAPSHOT_INVALID", "catalog snapshot has no config path")
    current = installed_set_snapshot_for_solver_config(
        config_path,
        manifest_dir=manifest_dir,
        environment=environment,
    )
    for field in ("config", "receipts", "artifacts"):
        if current.get(field) != snapshot.get(field):
            raise CatalogError(
                "CATALOG_FILE_CHANGED",
                "the managed solver catalog changed during solving",
            )
    return current


def installed_set_identity_for_solver_indexes(
    index_identities: Iterable[str],
    *,
    catalog_root: str | os.PathLike[str] | None = None,
    manifest_dir: str | os.PathLike[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    required: set[str] = set()
    for value in index_identities:
        match = _SOLVER_INDEX_ID.fullmatch(value)
        if match is None:
            raise CatalogError("CATALOG_SOLVER_INDEX_INVALID", f"unsupported solver index identity: {value}")
        required.add(f"index-{match.group(1)}.fits")
    if not required:
        raise CatalogError("CATALOG_SOLVER_INDEX_MISSING", "no solver index identities were supplied")
    candidates: list[dict[str, Any]] = []
    for receipt in verified_installed_set_identities(
        catalog_root=catalog_root,
        manifest_dir=manifest_dir,
        environment=environment,
    ):
        available = {item["artifactId"] for item in receipt["artifacts"]}
        if required <= available:
            candidates.append(receipt)
    if not candidates:
        raise CatalogError(
            "CATALOG_INSTALLED_SET_UNBOUND",
            f"no verified installed-set receipt covers: {', '.join(sorted(required))}",
        )
    minimum_size = min(len(item["artifacts"]) for item in candidates)
    smallest = [item for item in candidates if len(item["artifacts"]) == minimum_size]
    if len({item["installedSetIdentity"] for item in smallest}) != 1:
        raise CatalogError(
            "CATALOG_INDEX_AMBIGUOUS",
            "multiple equally specific installed-set receipts cover the solver index",
        )
    selected = min(smallest, key=lambda item: item["installedSetIdentity"])
    selected_artifacts = [item for item in selected["artifacts"] if item["artifactId"] in required]
    index_artifacts = []
    for item in selected_artifacts:
        match = re.fullmatch(r"index-(\d+)\.fits", item["artifactId"])
        if match is None:
            raise CatalogError(
                "CATALOG_SOLVER_INDEX_INVALID",
                f"installed artifact cannot be bound to INDEXID: {item['artifactId']}",
            )
        index_artifacts.append(
            {
                "indexId": match.group(1),
                "relativeName": item["relativePath"],
                "sizeBytes": item["sizeBytes"],
                "sha256": item["sha256"],
                "manifestSha256": selected["manifestSha256"],
                "installedSetIdentity": selected["installedSetIdentity"],
            }
        )
    return {
        "installedSetIdentity": selected["installedSetIdentity"],
        "receiptPath": selected["receiptPath"],
        "catalogId": selected["catalogId"],
        "manifestSha256": selected["manifestSha256"],
        "config": dict(selected["config"]),
        "solverIndexes": sorted(index_identities),
        "artifacts": selected_artifacts,
        "indexArtifacts": sorted(index_artifacts, key=lambda item: item["indexId"]),
    }


__all__ = [
    "CatalogArtifact",
    "CatalogError",
    "CatalogManifest",
    "DownloadResponse",
    "DownloadTransport",
    "UrllibDownloadTransport",
    "catalog_doctor",
    "catalog_list",
    "default_catalog_root",
    "get_catalog_manifest",
    "install_catalog",
    "installed_set_identity_for_solver_indexes",
    "installed_set_snapshot_for_solver_config",
    "load_catalog_manifests",
    "manifest_search_paths",
    "recommended_artifacts",
    "remove_catalog_plan",
    "verified_installed_set_identities",
    "verify_catalog",
    "verify_installed_set_snapshot",
    "write_astrometry_config",
    "write_installed_set_receipt",
]
