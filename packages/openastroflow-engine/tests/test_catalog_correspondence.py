"""Engine-side catalog correspondence: index decoding, matching, evidence shape.

The fixtures build a synthetic Astrometry.net index (the real kd-tree byte
layout: ``ENDIAN`` marker, ``u32`` star triples scaled by a range table, and a
tag-along magnitude table) plus a synthetic master whose stars sit exactly at
the projected positions of a known TAN WCS, so the expected residuals are
zero and any pixel-convention slip shows up as a 1.4 px RMS.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pytest

import openastroflow_engine.astrometry_net_backend as astrometry_backend_module
import openastroflow_engine.cli as cli_module
from openastroflow_engine.astap_backend import (
    AstapSolverBackend,
    verify_execution_receipt,
    verify_solver_execution_result,
)
from openastroflow_engine.astrometry_net_backend import (
    AstrometryNetSolverBackend,
    discover_astrometry_config,
)
from openastroflow_engine.backends import BackendDescriptor, BackendRegistry, DeviceKind, StageKind
from openastroflow_engine.catalog_correspondence import (
    CatalogCorrespondenceError,
    CorrespondenceParameters,
    IndexSummary,
    field_geometry,
    match_mutual_nearest,
    rank_indexes,
    read_index_stars,
    read_index_summary,
    verify_solution,
)
from openastroflow_engine.catalogs import verify_catalog
from openastroflow_engine.planning import solver_backend_science_ready
from openastroflow_engine.runtime import RuntimeConfigurationError, select_solver_chain
from openastroflow_engine.solver import (
    DeclarativeSolverBackend,
    SolveRequest,
    WcsParity,
    validate_astrometric_quality,
    validate_solver_result,
)

from test_solver_execution import write_fake_solver


@pytest.fixture(autouse=True)
def isolate_fake_solver_catalog_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake solvers use explicit catalog fixtures, never a developer's catalog."""

    monkeypatch.setattr(
        astrometry_backend_module,
        "discover_astrometry_config",
        lambda config_path=None, **kwargs: discover_astrometry_config(config_path, environment={}),
    )


FIELD_WIDTH = 400
FIELD_HEIGHT = 300
FIELD_RA = 339.2
FIELD_DEC = 34.4
PIXEL_SCALE_ARCSEC = 1.0
INDEX_ID = 4107
INDEX_IDENTITY = f"astrometry.net:index:{INDEX_ID}:healpix:-1:hpnside:0"
STAR_SIGMA_PX = 1.3


def synthetic_wcs(
    *,
    crpix_offset: tuple[float, float] = (0.0, 0.0),
    positive_parity: bool = False,
) -> fits.Header:
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRPIX1"] = (FIELD_WIDTH + 1) / 2.0 + crpix_offset[0]
    header["CRPIX2"] = (FIELD_HEIGHT + 1) / 2.0 + crpix_offset[1]
    header["CRVAL1"] = FIELD_RA
    header["CRVAL2"] = FIELD_DEC
    scale = PIXEL_SCALE_ARCSEC / 3600.0
    header["CD1_1"] = scale if positive_parity else -scale
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = scale
    return header


def _unit(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    ra_r = np.deg2rad(ra)
    dec_r = np.deg2rad(dec)
    return np.column_stack((np.cos(dec_r) * np.cos(ra_r), np.cos(dec_r) * np.sin(ra_r), np.sin(dec_r)))


def _raw_table(name: str, payload: bytes, row_bytes: int, row_count: int, cards: dict[str, object] | None = None) -> bytes:
    header = fits.Header()
    header["XTENSION"] = "BINTABLE"
    header["BITPIX"] = 8
    header["NAXIS"] = 2
    header["NAXIS1"] = row_bytes
    header["NAXIS2"] = row_count
    header["PCOUNT"] = 0
    header["GCOUNT"] = 1
    header["TFIELDS"] = 1
    header["TFORM1"] = f"{row_bytes}A"
    header["TTYPE1"] = name
    for key, value in (cards or {}).items():
        header[key] = value
    body = payload + b"\0" * ((-len(payload)) % 2880)
    return header.tostring(sep="", endcard=True, padding=True).encode("ascii") + body


def write_synthetic_index(
    path: Path,
    ra: np.ndarray,
    dec: np.ndarray,
    magnitudes: np.ndarray,
    *,
    index_id: int = INDEX_ID,
    endian: str = "little",
    scale_lower_radians: float = 0.0064,
    scale_upper_radians: float = 0.0087,
    healpix: int = -1,
    hpnside: int = 0,
    data_type: str = "u32",
) -> Path:
    """Write an index file with the byte layout of the 4100-series star kd-tree."""

    order = "<" if endian == "little" else ">"
    marker = "04:03:02:01" if endian == "little" else "01:02:03:04"
    xyz = _unit(np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64))
    count = int(xyz.shape[0])
    primary = fits.Header()
    primary["SIMPLE"] = True
    primary["BITPIX"] = 8
    primary["NAXIS"] = 0
    primary["EXTEND"] = True
    primary["INDEXID"] = index_id
    primary["HEALPIX"] = healpix
    primary["HPNSIDE"] = hpnside
    primary["SCALE_L"] = scale_lower_radians
    primary["SCALE_U"] = scale_upper_radians
    primary["NSTARS"] = count
    primary["ALLSKY"] = healpix < 0
    blocks = [primary.tostring(sep="", endcard=True, padding=True).encode("ascii")]
    # A decoy tree of the same shape family ensures the reader selects tables
    # by column name rather than by position.
    blocks.append(_raw_table("kdtree_header_codes", b"", 0, 0, {"ENDIAN": marker, "KDT_DATA": "u16"}))
    if data_type == "u32":
        scale = 2147483647.0
        minimum = np.asarray([-1.0, -1.0, -1.0])
        encoded = np.round((xyz - minimum) * scale).astype(f"{order}u4")
        rows = encoded.tobytes()
        row_bytes = 12
    else:
        rows = xyz.astype(f"{order}f8").tobytes()
        row_bytes = 24
    blocks.append(
        _raw_table(
            "kdtree_header_stars",
            b"",
            0,
            0,
            {
                "ENDIAN": marker,
                "KDT_NAME": "stars",
                "KDT_NDAT": count,
                "KDT_NDIM": 3,
                "KDT_EXT": "double",
                "KDT_INT": "u32",
                "KDT_DATA": data_type,
            },
        )
    )
    scaling = np.asarray([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 2147483647.0], dtype=f"{order}f8")
    blocks.append(_raw_table("kdtree_range_stars", scaling.tobytes(), 8, 7))
    blocks.append(_raw_table("kdtree_data_stars", rows, row_bytes, count))
    tag = fits.Header()
    tag["XTENSION"] = "BINTABLE"
    tag["BITPIX"] = 8
    tag["NAXIS"] = 2
    tag["NAXIS1"] = 16
    tag["NAXIS2"] = count
    tag["PCOUNT"] = 0
    tag["GCOUNT"] = 1
    tag["TFIELDS"] = 4
    for position, name in enumerate(("MAG_BT", "MAG_VT", "MAG_HP", "MAG"), start=1):
        tag[f"TFORM{position}"] = "1E"
        tag[f"TTYPE{position}"] = name
    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    tag_rows = np.column_stack((magnitudes + 0.4, magnitudes, np.zeros(count), magnitudes)).astype(">f4").tobytes()
    blocks.append(tag.tostring(sep="", endcard=True, padding=True).encode("ascii") + tag_rows + b"\0" * ((-len(tag_rows)) % 2880))
    path.write_bytes(b"".join(blocks))
    return path


def synthetic_field(seed: int = 7, *, in_field_stars: int = 40) -> dict[str, np.ndarray]:
    """Catalog positions (1-based pixels and sky) for the synthetic master.

    Stars sit on a jittered 40 px grid: the detector drops blended sources,
    and the bright Tycho-2 stars this check relies on are isolated in real
    fields, so the fixture must not manufacture blends.
    """

    rng = np.random.default_rng(seed)
    header = synthetic_wcs()
    celestial = WCS(header).celestial
    grid = np.asarray(
        [(x, y) for y in range(40, FIELD_HEIGHT - 20, 40) for x in range(40, FIELD_WIDTH - 20, 40)],
        dtype=np.float64,
    )
    cells = grid[rng.permutation(grid.shape[0])] + rng.uniform(-4.0, 4.0, grid.shape)
    assert cells.shape[0] >= in_field_stars + 14
    inside = cells[:in_field_stars]
    decoys = cells[in_field_stars : in_field_stars + 14]
    # Catalog stars just outside the image (inside the search cone) and far
    # away (outside it) must neither match nor be counted as field stars.
    # Within 0.6 x diagonal (300 px) of the centre but left of the image edge.
    nearby = np.column_stack((rng.uniform(-40.0, -20.0, 8), rng.uniform(100.0, 200.0, 8)))
    far = np.column_stack((rng.uniform(3000.0, 3200.0, 5), rng.uniform(3000.0, 3200.0, 5)))
    pixels = np.vstack((inside, nearby, far))
    world = celestial.all_pix2world(pixels, 1)
    magnitudes = rng.uniform(9.0, 12.4, pixels.shape[0])
    return {
        "inside_pixels": inside,
        "all_pixels": pixels,
        "ra": world[:, 0],
        "dec": world[:, 1],
        "magnitudes": magnitudes,
        "decoy_pixels": decoys,
    }


def render_master(
    star_pixels: np.ndarray,
    *,
    seed: int = 11,
    amplitude: float = 4000.0,
) -> np.ndarray:
    """Gaussian stars at one-based FITS positions on a flat noisy background."""

    rng = np.random.default_rng(seed)
    image = 100.0 + rng.normal(0.0, 1.0, (FIELD_HEIGHT, FIELD_WIDTH))
    yy, xx = np.mgrid[0:FIELD_HEIGHT, 0:FIELD_WIDTH]
    for x, y in np.asarray(star_pixels, dtype=np.float64):
        # FITS pixel (x, y) is array index (y - 1, x - 1).
        image += amplitude * np.exp(-(((xx - (x - 1.0)) ** 2 + (yy - (y - 1.0)) ** 2) / (2.0 * STAR_SIGMA_PX**2)))
    return image.astype(np.float32)


def write_master(path: Path, image: np.ndarray) -> Path:
    header = fits.Header()
    header["OBJECT"] = "synthetic"
    header["OBJCTRA"] = "22:36:48"
    header["OBJCTDEC"] = "+34:24:00"
    fits.writeto(path, image, header, overwrite=False)
    return path


def managed_index_catalog(tmp_path: Path, index_path: Path) -> tuple[Path, Path, dict[str, object]]:
    """Adopt a synthetic index into a managed installed set with a receipt."""

    root = index_path.parent
    manifests = tmp_path / "managed-manifests"
    manifests.mkdir()
    payload = index_path.read_bytes()
    manifest = {
        "schemaVersion": 1,
        "catalogId": f"fake-managed-{INDEX_ID}",
        "provider": "Fake index provider",
        "version": "fixture-v1",
        "redistributionStatus": "user-download-required",
        "license": "fixture only",
        "citation": "fixture citation",
        "providerTerms": {
            "acceptanceId": "fake-managed-terms-v1",
            "url": "https://provider.invalid/terms",
            "summary": "Synthetic test fixture only.",
            "licenseStatus": "confirmed",
            "requiresExplicitAcceptance": True,
        },
        "allowedDownloadOrigins": ["https://provider.invalid"],
        "artifacts": [
            {
                "artifactId": index_path.name,
                "url": f"https://provider.invalid/{index_path.name}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "sizeBytes": len(payload),
                "installScope": "user",
                "scale": 7,
            }
        ],
    }
    (manifests / f"fake-managed-{INDEX_ID}.json").write_text(json.dumps(manifest), encoding="utf-8")
    verified = verify_catalog(
        f"fake-managed-{INDEX_ID}",
        catalog_root=root,
        manifest_dir=manifests,
        write_configuration=True,
    )
    assert verified["ok"] is True
    return root / "astrometry.cfg", manifests, verified


def fake_star_database(tmp_path: Path) -> Path:
    directory = tmp_path / "astap-database"
    directory.mkdir(exist_ok=True)
    for area in ("0101", "0102", "0201"):
        (directory / f"d20_{area}.1476").write_bytes(b"fake star database area")
    return directory


@pytest.fixture
def field() -> dict[str, np.ndarray]:
    return synthetic_field()


@pytest.fixture
def index_root(tmp_path: Path, field: dict[str, np.ndarray]) -> Path:
    root = tmp_path / "managed-catalog"
    root.mkdir()
    write_synthetic_index(root / f"index-{INDEX_ID}.fits", field["ra"], field["dec"], field["magnitudes"])
    return root


def test_index_reader_decodes_little_and_big_endian_trees_identically(tmp_path: Path, field: dict[str, np.ndarray]) -> None:
    little = write_synthetic_index(tmp_path / "index-4107.fits", field["ra"], field["dec"], field["magnitudes"])
    big_dir = tmp_path / "big"
    big_dir.mkdir()
    big = write_synthetic_index(big_dir / "index-4107.fits", field["ra"], field["dec"], field["magnitudes"], endian="big")
    double_dir = tmp_path / "double"
    double_dir.mkdir()
    doubles = write_synthetic_index(
        double_dir / "index-4107.fits", field["ra"], field["dec"], field["magnitudes"], data_type="double"
    )

    summary = read_index_summary(little)
    assert summary.identity == INDEX_IDENTITY
    assert summary.star_count == field["ra"].shape[0]
    assert summary.serializable()["quadScaleArcminutes"][0] == pytest.approx(22.0, abs=0.1)

    results = {}
    for name, path in (("little", little), ("big", big), ("double", doubles)):
        stars = read_index_stars(
            path,
            centre_ra_degrees=FIELD_RA,
            centre_dec_degrees=FIELD_DEC,
            radius_degrees=0.2,
        )
        results[name] = stars
    assert results["little"].endian == "little"
    assert results["big"].endian == "big"
    assert results["little"].count == field["ra"].shape[0] - 5  # the five far decoys are outside the cone
    for name in ("big", "double"):
        assert np.array_equal(results[name].rows, results["little"].rows)
        assert np.allclose(results[name].ra_degrees, results["little"].ra_degrees, atol=1e-7)
        assert np.allclose(results[name].dec_degrees, results["little"].dec_degrees, atol=1e-7)
    # u32 quantisation of the unit sphere is far below a milliarcsecond.
    rows = results["little"].rows
    assert np.allclose(results["little"].ra_degrees, field["ra"][rows] % 360.0, atol=1e-7)
    assert np.allclose(results["little"].dec_degrees, field["dec"][rows], atol=1e-7)
    assert results["little"].magnitudes is not None
    assert np.allclose(results["little"].magnitudes, field["magnitudes"][rows], atol=1e-5)


def test_index_reader_rejects_unknown_endian_and_mismatched_identity(tmp_path: Path, field: dict[str, np.ndarray]) -> None:
    path = write_synthetic_index(tmp_path / "index-4107.fits", field["ra"], field["dec"], field["magnitudes"])
    raw = path.read_bytes()
    corrupted = raw.replace(b"'04:03:02:01'", b"'01:03:02:04'")
    assert corrupted != raw
    (tmp_path / "bad").mkdir()
    bad = tmp_path / "bad" / "index-4107.fits"
    bad.write_bytes(corrupted)
    with pytest.raises(CatalogCorrespondenceError) as failure:
        read_index_stars(bad, centre_ra_degrees=FIELD_RA, centre_dec_degrees=FIELD_DEC, radius_degrees=0.2)
    assert failure.value.code == "CATALOG_INDEX_INVALID"

    (tmp_path / "renamed").mkdir()
    renamed = tmp_path / "renamed" / "index-4108.fits"
    renamed.write_bytes(raw)
    with pytest.raises(CatalogCorrespondenceError) as mismatch:
        read_index_summary(renamed)
    assert mismatch.value.code == "CATALOG_INDEX_INVALID"

    with pytest.raises(CatalogCorrespondenceError) as changed:
        read_index_stars(
            path,
            centre_ra_degrees=FIELD_RA,
            centre_dec_degrees=FIELD_DEC,
            radius_degrees=0.2,
            expected_stat={"device": -1, "inode": -1, "mtimeNs": 0, "sizeBytes": 0},
        )
    assert changed.value.code == "CATALOG_FILE_CHANGED"


def test_rank_indexes_prefers_fitting_then_densest() -> None:
    geometry = field_geometry(synthetic_wcs(), (FIELD_HEIGHT, FIELD_WIDTH))
    assert geometry.width_degrees == pytest.approx(FIELD_WIDTH / 3600.0, rel=1e-3)
    assert geometry.diagonal_degrees == pytest.approx(np.hypot(FIELD_WIDTH, FIELD_HEIGHT) / 3600.0, rel=1e-3)
    assert geometry.centre_ra_degrees == pytest.approx(FIELD_RA, abs=1e-6)

    def summary(index_id: int, lower_arcmin: float, upper_arcmin: float, stars: int) -> IndexSummary:
        return IndexSummary(
            f"index-{index_id}.fits", str(index_id), -1, 0,
            np.deg2rad(lower_arcmin / 60.0), np.deg2rad(upper_arcmin / 60.0), stars,
        )

    ranked = rank_indexes(
        (
            summary(4109, 42, 60, 700_000),
            summary(4203, 2, 2.8, 5_000_000),
            summary(4107, 22, 30, 1_900_000),
            summary(4205, 5.6, 8, 3_000_000),
        ),
        geometry,
    )
    # Only 4203 and 4205 quads (< 6.7 arcmin field width) fit; densest first.
    assert [item.index_id for item in ranked] == ["4203", "4205", "4107", "4109"]


def test_mutual_nearest_matching_is_one_to_one_and_bounded() -> None:
    detected = np.asarray([[10.0, 10.0], [50.0, 50.0], [50.5, 50.0], [90.0, 90.0]])
    predicted = np.asarray([[10.2, 9.9], [50.2, 50.0], [200.0, 200.0], [90.0, 96.0]])
    rows, catalog, distances = match_mutual_nearest(detected, predicted, 3.0)
    # (50, 50) wins its catalog star; (50.5, 50) loses it and has no other
    # partner; (90, 90) is 6 px from its nearest, outside the radius.
    assert rows.tolist() == [0, 1]
    assert catalog.tolist() == [0, 1]
    assert distances[0] == pytest.approx(np.hypot(0.2, 0.1))
    empty_rows, empty_catalog, _ = match_mutual_nearest(detected[:0], predicted, 3.0)
    assert empty_rows.size == 0 and empty_catalog.size == 0


def test_verify_solution_recovers_exact_matches_and_evidence_shape(tmp_path: Path, field: dict[str, np.ndarray], index_root: Path) -> None:
    image = render_master(np.vstack((field["inside_pixels"], field["decoy_pixels"])))
    artifacts = [{"relativeName": f"index-{INDEX_ID}.fits", "artifactId": f"index-{INDEX_ID}.fits"}]
    verification = verify_solution(
        image=image,
        wcs_header=synthetic_wcs(),
        image_shape=(FIELD_HEIGHT, FIELD_WIDTH),
        catalog_root=index_root,
        index_artifacts=artifacts,
        artifact_dir=tmp_path,
    )
    quality = verification.quality
    assert quality.matched_stars == field["inside_pixels"].shape[0]
    assert quality.rms_pixels < 0.05
    assert quality.rms_arcsec == pytest.approx(quality.rms_pixels * PIXEL_SCALE_ARCSEC, rel=0.05)
    assert quality.parity is WcsParity.NEGATIVE
    assert quality.index_identities == (INDEX_IDENTITY,)
    assert quality.catalog_managed is False
    payload = verification.correspondence_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == quality.correspondence_sha256
    table = json.loads(payload)
    assert table["coordinateOrigin"] == 1
    assert table["indexIdentities"] == [INDEX_IDENTITY]
    assert len(table["matches"]) == quality.matched_stars
    assert [row["indexId"] for row in table["matches"]] == sorted(row["indexId"] for row in table["matches"])
    assert all(row["indexMag"] is not None for row in table["matches"])
    match = json.loads(verification.match_path.read_text(encoding="utf-8"))
    assert match["indexIdentities"] == [INDEX_IDENTITY]
    assert match["indexSelection"][0]["selected"] is True
    assert match["nmatch"] == quality.matched_stars
    diagnostics = verification.correspondence_diagnostics
    assert diagnostics["uniqueOneToOneMatches"] == quality.matched_stars
    assert diagnostics["catalogStarsInsideImage"] == field["inside_pixels"].shape[0]
    assert diagnostics["catalogStarsWithinSearchRadius"] == field["ra"].shape[0] - 5
    assert diagnostics["detectedStars"] >= field["inside_pixels"].shape[0] + field["decoy_pixels"].shape[0] - 2
    assert diagnostics["matchRadiusPixels"] == pytest.approx(max(3.0, 2.0 * diagnostics["medianFwhmPixels"]))
    assert verification.match_diagnostics["indexIdentities"] == [INDEX_IDENTITY]
    assert verification.match_diagnostics["validatedParity"] == "NEGATIVE"
    # The catalog identity is the digest of the matched rows, like the .corr route.
    rows = sorted(
        (str(row["indexId"]), format(float(row["indexRa"]), ".17g"), format(float(row["indexDec"]), ".17g"))
        for row in table["matches"]
    )
    expected_identity = hashlib.sha256(
        json.dumps(
            {"indexes": [INDEX_IDENTITY], "matchedCatalogRows": rows},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert quality.catalog_identity == expected_identity


def test_verify_solution_reports_positive_parity_from_the_wcs(tmp_path: Path, field: dict[str, np.ndarray]) -> None:
    header = synthetic_wcs(positive_parity=True)
    celestial = WCS(header).celestial
    world = celestial.all_pix2world(field["inside_pixels"], 1)
    root = tmp_path / "catalog"
    root.mkdir()
    write_synthetic_index(root / f"index-{INDEX_ID}.fits", world[:, 0], world[:, 1], field["magnitudes"][: world.shape[0]])
    verification = verify_solution(
        image=render_master(field["inside_pixels"]),
        wcs_header=header,
        image_shape=(FIELD_HEIGHT, FIELD_WIDTH),
        catalog_root=root,
        index_artifacts=[{"relativeName": f"index-{INDEX_ID}.fits"}],
        artifact_dir=tmp_path,
    )
    assert verification.quality.parity is WcsParity.POSITIVE
    assert verification.quality.matched_stars == field["inside_pixels"].shape[0]


@pytest.mark.parametrize(
    ("scenario", "code"),
    (
        ("shifted-wcs", "CORRESPONDENCE_UNIQUE_MATCHES_MISSING"),
        ("too-few-stars", "CORRESPONDENCE_DETECTION_INSUFFICIENT"),
        ("field-uncovered", "CATALOG_FIELD_UNCOVERED"),
        ("colour-cube", "CORRESPONDENCE_INPUT_UNSUPPORTED"),
    ),
)
def test_verify_solution_fails_closed(tmp_path: Path, field: dict[str, np.ndarray], index_root: Path, scenario: str, code: str) -> None:
    image = render_master(np.vstack((field["inside_pixels"], field["decoy_pixels"])))
    header = synthetic_wcs()
    if scenario == "shifted-wcs":
        # Half the fixture's star pitch: every catalog star is projected at
        # least 16 px from any image star, far beyond the match radius, so a
        # plausible but wrong solution cannot be rescued by chance neighbours.
        header = synthetic_wcs(crpix_offset=(20.0, 20.0))
    elif scenario == "too-few-stars":
        image = render_master(field["inside_pixels"][:2])
    elif scenario == "field-uncovered":
        header["CRVAL1"] = (FIELD_RA + 30.0) % 360.0
    elif scenario == "colour-cube":
        image = np.stack((image, image, image))
    with pytest.raises(CatalogCorrespondenceError) as failure:
        verify_solution(
            image=image,
            wcs_header=header,
            image_shape=(FIELD_HEIGHT, FIELD_WIDTH),
            catalog_root=index_root,
            index_artifacts=[{"relativeName": f"index-{INDEX_ID}.fits"}],
            artifact_dir=tmp_path,
        )
    assert failure.value.code == code
    assert not (tmp_path / "correspondence.json").exists()


def test_correspondence_parameters_validate_bounds() -> None:
    CorrespondenceParameters().validate()
    with pytest.raises(ValueError, match="minimum_matches"):
        CorrespondenceParameters(minimum_matches=2).validate()
    with pytest.raises(ValueError, match="detection_sigma"):
        CorrespondenceParameters(detection_sigma=0.5).validate()
    with pytest.raises(ValueError, match="match_radius_fwhm_factor"):
        CorrespondenceParameters(match_radius_fwhm_factor=0.0).validate()


def _astap_backend(tmp_path: Path, script: Path, header: fits.Header, *, mode: str = "success", **overrides):
    header_file = tmp_path / f"{mode}-solution-header.txt"
    if header_file.exists():
        header_file.unlink()
    header.totextfile(header_file)
    environment = {"FAKE_BACKEND": "astap", "FAKE_MODE": mode, "FAKE_WCS_HEADER": str(header_file)}
    environment.update(overrides.pop("environment", {}))
    options = dict(
        executable=sys.executable,
        executable_args=(str(script),),
        environment=environment,
        star_database_dir=fake_star_database(tmp_path),
        staging_root=tmp_path,
        timeout_seconds=20.0,
        probe_timeout_seconds=5.0,
    )
    options.update(overrides)
    return AstapSolverBackend(**options)


def _solve_request(source: Path, output: Path) -> SolveRequest:
    return SolveRequest(str(source), str(output), ra_hint_degrees=FIELD_RA, dec_hint_degrees=FIELD_DEC, field_of_view_degrees=0.12, search_radius_degrees=5.0)


def test_strict_astap_solution_carries_managed_catalog_evidence(tmp_path: Path, field: dict[str, np.ndarray], index_root: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_master(tmp_path / "master.fits", render_master(np.vstack((field["inside_pixels"], field["decoy_pixels"]))))
    config, manifests, installed = managed_index_catalog(tmp_path, index_root / f"index-{INDEX_ID}.fits")
    backend = _astap_backend(tmp_path, script, synthetic_wcs(), config_path=config, catalog_manifest_dir=manifests)
    descriptor = backend.descriptor
    assert descriptor.execution_ready is True
    assert "catalog-correspondence-quality-v1" in descriptor.capabilities
    assert descriptor.metadata["scientificQualityEvidence"]["status"] == "recomputed"
    assert descriptor.metadata["scientificQualityEvidence"]["strictE2EBehavior"] == "final-gate"
    assert descriptor.metadata["runtimePrerequisites"]["starDatabaseDiscovered"] is True
    assert descriptor.metadata["runtimePrerequisites"]["configDiscovered"] is True
    assert descriptor.metadata["probe"]["evidence"]["starDatabase"]["families"] == ["d20"]
    assert "path" not in descriptor.metadata["probe"]["evidence"]["starDatabase"]
    assert solver_backend_science_ready(backend) is True

    output = tmp_path / "solved" / "master-solved.fits"
    result = backend.solve(_solve_request(source, output))

    assert result.status.value == "SOLVED", result.error
    assert result.evidence["adapterVersion"] == "astap-process-v4"
    quality = result.astrometric_quality
    assert quality is not None
    assert quality.matched_stars == field["inside_pixels"].shape[0]
    assert quality.rms_pixels < 0.05
    assert quality.parity is WcsParity.NEGATIVE
    assert quality.index_identities == (INDEX_IDENTITY,)
    assert quality.catalog_managed is True
    assert quality.installed_set_identity == installed["installedSet"]["installedSetIdentity"]
    assert quality.catalog_manifest_sha256 == installed["manifestSha256"]
    assert [item.index_id for item in quality.index_artifacts] == [str(INDEX_ID)]
    assert quality.index_artifacts[0].sha256 == hashlib.sha256((index_root / f"index-{INDEX_ID}.fits").read_bytes()).hexdigest()
    # Receipt shape mirrors the astrometry.net adapter so the shared
    # verifier and the E2E gate accept it unchanged.
    evidence = result.evidence
    assert evidence["astrometricQuality"] == quality.serializable()
    assert evidence["outputs"]["corr"]["sha256"] == f"sha256:{quality.correspondence_sha256}"
    assert evidence["outputs"]["match"]["sha256"].startswith("sha256:")
    assert evidence["matchDiagnostics"]["indexIdentities"] == [INDEX_IDENTITY]
    assert evidence["correspondenceDiagnostics"]["uniqueOneToOneMatches"] == quality.matched_stars
    assert evidence["catalogBinding"]["catalogManaged"] is True
    assert evidence["catalogBinding"]["installedSetIdentity"] == quality.installed_set_identity
    assert evidence["catalogPreflight"]["managed"] is True
    assert evidence["config"]["sha256"].startswith("sha256:")
    assert evidence["verification"]["method"] == "engine-catalog-correspondence-v1"
    assert evidence["backendConfirmation"] == "PLTSOLVD=T"
    assert verify_execution_receipt(evidence) is True
    assert verify_solver_execution_result(result) is True
    assert validate_solver_result(result, require_scientific_evidence=True, min_matches=12, max_rms_arcsec=2.0).valid
    assert validate_astrometric_quality(result, require_managed_catalog=True).valid
    assert output.is_file()
    published = fits.getheader(output)
    assert published["CTYPE1"] == "RA---TAN"
    serialized = json.dumps(evidence, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert '"argv"' not in serialized
    assert "stdoutTail" not in serialized
    # A tampered match count no longer verifies against the receipt.
    drifted = replace(quality, matched_stars=quality.matched_stars + 1)
    assert verify_solver_execution_result(replace(result, astrometric_quality=drifted)) is False


def test_strict_astap_fails_closed_on_wrong_sky_position(tmp_path: Path, field: dict[str, np.ndarray], index_root: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_master(tmp_path / "master.fits", render_master(field["inside_pixels"]))
    config, manifests, _ = managed_index_catalog(tmp_path, index_root / f"index-{INDEX_ID}.fits")
    backend = _astap_backend(
        tmp_path, script, synthetic_wcs(crpix_offset=(20.0, 20.0)), config_path=config, catalog_manifest_dir=manifests
    )
    output = tmp_path / "shifted-solved.fits"
    result = backend.solve(_solve_request(source, output))
    assert result.status.value == "FAILED"
    assert result.evidence["failureCode"] == "CORRESPONDENCE_UNIQUE_MATCHES_MISSING"
    assert result.evidence["wcsValidation"]["valid"] is True
    assert not output.exists()


def test_strict_astap_fails_closed_with_too_few_detected_stars(tmp_path: Path, field: dict[str, np.ndarray], index_root: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_master(tmp_path / "sparse.fits", render_master(field["inside_pixels"][:2]))
    config, manifests, _ = managed_index_catalog(tmp_path, index_root / f"index-{INDEX_ID}.fits")
    backend = _astap_backend(tmp_path, script, synthetic_wcs(), config_path=config, catalog_manifest_dir=manifests)
    output = tmp_path / "sparse-solved.fits"
    result = backend.solve(_solve_request(source, output))
    assert result.status.value == "FAILED"
    assert result.evidence["failureCode"] == "CORRESPONDENCE_DETECTION_INSUFFICIENT"
    assert not output.exists()


def test_strict_astap_requires_a_managed_catalog_config(tmp_path: Path, field: dict[str, np.ndarray]) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_master(tmp_path / "master.fits", render_master(field["inside_pixels"]))
    backend = _astap_backend(tmp_path, script, synthetic_wcs())
    assert backend.descriptor.metadata["runtimePrerequisites"]["configDiscovered"] is False
    output = tmp_path / "unmanaged-solved.fits"
    result = backend.solve(_solve_request(source, output))
    assert result.status.value == "FAILED"
    assert result.evidence["failureCode"] == "CATALOG_CONFIG_UNMANAGED"
    assert not output.exists()


def test_lenient_astap_without_catalog_is_diagnostic_only(tmp_path: Path, field: dict[str, np.ndarray]) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_master(tmp_path / "master.fits", render_master(field["inside_pixels"]))
    backend = _astap_backend(tmp_path, script, synthetic_wcs(), require_managed_catalog=False)
    output = tmp_path / "diagnostic-solved.fits"
    result = backend.solve(_solve_request(source, output))
    assert result.status.value == "SOLVED", result.error
    assert result.astrometric_quality is None
    assert result.evidence["astrometricQuality"]["status"] == "UNAVAILABLE"
    assert result.evidence["catalogPreflight"] == {"managed": False, "reason": "NO_MANAGED_CONFIG"}
    assert result.evidence["verification"]["method"] is None
    assert validate_solver_result(result, require_scientific_evidence=True, min_matches=12, max_rms_arcsec=2.0).code == "SOLVER_QUALITY_EVIDENCE_MISSING"


def test_strict_astap_detects_index_replaced_after_the_snapshot(tmp_path: Path, field: dict[str, np.ndarray], index_root: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    source = write_master(tmp_path / "master.fits", render_master(field["inside_pixels"]))
    index_path = index_root / f"index-{INDEX_ID}.fits"
    config, manifests, _ = managed_index_catalog(tmp_path, index_path)
    backend = _astap_backend(
        tmp_path,
        script,
        synthetic_wcs(),
        mode="catalog-replace",
        environment={"FAKE_CATALOG_PATH": str(index_path)},
        config_path=config,
        catalog_manifest_dir=manifests,
    )
    output = tmp_path / "replaced-solved.fits"
    result = backend.solve(_solve_request(source, output))
    assert result.status.value == "FAILED"
    assert result.evidence["failureCode"] == "CATALOG_FILE_CHANGED"
    assert not output.exists()


def test_astap_without_star_database_is_not_execution_ready(tmp_path: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    backend = AstapSolverBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        environment={"FAKE_BACKEND": "astap", "FAKE_MODE": "success"},
        staging_root=tmp_path,
        timeout_seconds=5.0,
        probe_timeout_seconds=5.0,
        require_managed_catalog=False,
    )
    descriptor = backend.descriptor
    assert descriptor.available is True
    assert descriptor.execution_ready is False
    assert backend.probe.error_code == "STAR_DATABASE_MISSING"
    assert descriptor.metadata["runtimePrerequisites"]["starDatabaseDiscovered"] is False
    assert descriptor.capabilities == ()
    assert solver_backend_science_ready(backend) is False
    result = backend.solve(_solve_request(tmp_path / "missing.fits", tmp_path / "out.fits"))
    assert result.status.value == "UNAVAILABLE"
    assert result.evidence["failureCode"] == "STAR_DATABASE_MISSING"


def test_solver_chain_accepts_science_ready_astap_and_keeps_the_default_order(tmp_path: Path, field: dict[str, np.ndarray], index_root: Path) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    config, manifests, _ = managed_index_catalog(tmp_path, index_root / f"index-{INDEX_ID}.fits")
    astap = _astap_backend(tmp_path, script, synthetic_wcs(), config_path=config, catalog_manifest_dir=manifests)
    astrometry = AstrometryNetSolverBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        environment={"FAKE_BACKEND": "astrometry", "FAKE_MODE": "success"},
        staging_root=tmp_path,
        timeout_seconds=5.0,
        probe_timeout_seconds=5.0,
        require_managed_catalog=False,
    )
    native = DeclarativeSolverBackend(
        BackendDescriptor(
            backend_id="native",
            stage=StageKind.SOLVER,
            display_name="native",
            version="protocol-v1",
            available=False,
            execution_ready=False,
            devices=(DeviceKind.CPU,),
            capabilities=(),
            reason="not bundled",
        )
    )
    registry = BackendRegistry([astap, native, astrometry])
    chain = select_solver_chain(registry, "auto")
    assert [backend.backend_id for backend in chain] == ["astrometry-net", "astap"]
    assert [backend.backend_id for backend in select_solver_chain(registry, "astap")] == ["astap"]

    without_database = AstapSolverBackend(
        executable=sys.executable,
        executable_args=(str(script),),
        environment={"FAKE_BACKEND": "astap", "FAKE_MODE": "success"},
        staging_root=tmp_path,
        timeout_seconds=5.0,
        probe_timeout_seconds=5.0,
        config_path=config,
        catalog_manifest_dir=manifests,
    )
    with pytest.raises(RuntimeConfigurationError) as failure:
        select_solver_chain(BackendRegistry([without_database, native]), "astap")
    assert failure.value.code == "SOLVER_UNAVAILABLE"
    assert "star database" in str(failure.value)


def test_doctor_reports_per_backend_science_readiness(tmp_path: Path, field: dict[str, np.ndarray], index_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = write_fake_solver(tmp_path / "fake_solver.py")
    config, manifests, _ = managed_index_catalog(tmp_path, index_root / f"index-{INDEX_ID}.fits")
    astap = _astap_backend(tmp_path, script, synthetic_wcs(), config_path=config, catalog_manifest_dir=manifests)
    native = DeclarativeSolverBackend(
        BackendDescriptor(
            backend_id="native",
            stage=StageKind.SOLVER,
            display_name="native",
            version="protocol-v1",
            available=False,
            execution_ready=False,
            devices=(DeviceKind.CPU,),
            capabilities=(),
            reason="not bundled",
        )
    )
    monkeypatch.setattr(cli_module, "default_registry", lambda: BackendRegistry([astap, native]))
    payload = cli_module.doctor_payload()
    by_id = {backend["backendId"]: backend for backend in payload["backends"]}
    assert by_id["astap"]["scienceReady"] is True
    assert by_id["astap"]["executionReady"] is True
    assert by_id["native"]["scienceReady"] is False
    assert payload["status"]["solverReady"] is True
    assert payload["status"]["solverExecutableReady"] is True
    assert payload["status"]["catalogCoverageVerified"] is False
