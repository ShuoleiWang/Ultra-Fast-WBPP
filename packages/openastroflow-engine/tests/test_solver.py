from __future__ import annotations

from dataclasses import replace

from openastroflow_engine.solver import (
    AstrometricQuality,
    SolutionKind,
    SolverResult,
    SolverBackend,
    SolverIndexArtifact,
    SolverStatus,
    WcsParity,
    canonical_wcs_sha256,
    solver_backends,
    validate_solver_result,
    validate_wcs_header,
)


def solved_header() -> dict[str, object]:
    return {
        "NAXIS": 2,
        "NAXIS1": 100,
        "NAXIS2": 80,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRPIX1": 50.0,
        "CRPIX2": 40.0,
        "CRVAL1": 270.0,
        "CRVAL2": -15.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }


def test_pointing_seed_is_not_a_solution() -> None:
    validation = validate_wcs_header(
        {
            "NAXIS1": 100,
            "NAXIS2": 80,
            "OBJCTRA": "18:00:00",
            "OBJCTDEC": "-15:00:00",
            "CRVAL1": 270.0,
            "CRVAL2": -15.0,
        }
    )
    assert validation.valid is False
    assert validation.code == "WCS_MISSING_CELESTIAL_AXES"


def test_structurally_and_numerically_valid_wcs_passes() -> None:
    validation = validate_wcs_header(solved_header())
    assert validation.valid is True
    assert validation.code == "WCS_VALID"
    assert validation.diagnostics["roundtripMaxPixels"] < 0.05


def test_singular_wcs_fails_closed() -> None:
    header = solved_header()
    header["CDELT1"] = 0.0
    validation = validate_wcs_header(header)
    assert validation.valid is False
    assert validation.code == "WCS_LINEAR_TRANSFORM_SINGULAR"


def test_invalid_external_image_shape_fails_without_throwing() -> None:
    header = solved_header()
    header.pop("NAXIS1")
    header.pop("NAXIS2")
    validation = validate_wcs_header(header, image_shape=(80, True))
    assert validation.valid is False
    assert validation.code == "WCS_IMAGE_GEOMETRY_INVALID"


def test_backend_must_confirm_newly_solved_result() -> None:
    inherited = SolverResult(
        backend_id="astap",
        status=SolverStatus.SOLVED,
        solution_kind=SolutionKind.EXISTING,
        backend_confirmed=True,
        header=solved_header(),
    )
    unconfirmed = SolverResult(
        backend_id="astap",
        status=SolverStatus.SOLVED,
        solution_kind=SolutionKind.SOLVED,
        backend_confirmed=False,
        header=solved_header(),
    )
    confirmed = SolverResult(
        backend_id="astap",
        status=SolverStatus.SOLVED,
        solution_kind=SolutionKind.SOLVED,
        backend_confirmed=True,
        header=solved_header(),
        evidence={"matchedStars": 42},
    )

    assert validate_solver_result(inherited).code == "SOLVER_RESULT_IS_NOT_NEW_SOLUTION"
    assert validate_solver_result(unconfirmed).code == "SOLVER_CONFIRMATION_MISSING"
    assert validate_solver_result(confirmed).valid is True


def test_catalog_quality_is_structured_and_validated_against_wcs() -> None:
    quality = AstrometricQuality(
        matched_stars=20,
        rms_pixels=0.2,
        rms_arcsec=0.72,
        parity=WcsParity.NEGATIVE,
        catalog_identity="1" * 64,
        index_identities=("astrometry.net:index:4206:healpix:17:hpnside:32",),
        correspondence_sha256="2" * 64,
        catalog_managed=True,
        installed_set_identity="3" * 64,
        catalog_manifest_sha256="4" * 64,
        index_artifacts=(
            SolverIndexArtifact(
                index_id="4206",
                relative_name="index-4206.fits",
                size_bytes=94_550_400,
                sha256="5" * 64,
                manifest_sha256="4" * 64,
                installed_set_identity="3" * 64,
            ),
        ),
    )
    result = SolverResult(
        backend_id="astrometry-net",
        status=SolverStatus.SOLVED,
        solution_kind=SolutionKind.SOLVED,
        backend_confirmed=True,
        header=solved_header(),
        image_shape=(80, 100),
        astrometric_quality=quality,
    )

    validation = validate_solver_result(
        result,
        require_scientific_evidence=True,
        min_matches=12,
        max_rms_arcsec=2.0,
    )
    assert validation.valid is True
    assert result.serializable()["astrometricQuality"]["matchedStars"] == 20

    wrong_parity = SolverResult(
        backend_id=result.backend_id,
        status=result.status,
        solution_kind=result.solution_kind,
        backend_confirmed=result.backend_confirmed,
        header=result.header,
        image_shape=result.image_shape,
        astrometric_quality=replace(quality, parity=WcsParity.POSITIVE),
    )
    assert validate_solver_result(wrong_parity).code == "SOLVER_PARITY_MISMATCH"


def test_canonical_wcs_digest_ignores_non_wcs_cards_and_binds_geometry() -> None:
    header = solved_header()
    digest = canonical_wcs_sha256(header)
    with_camera_card = {**header, "EXPTIME": 60.0}
    changed_wcs = {**header, "CRVAL1": 271.0}

    assert len(digest) == 64
    assert canonical_wcs_sha256(with_camera_card) == digest
    assert canonical_wcs_sha256(changed_wcs) != digest


def test_all_three_solver_provider_protocols_are_registered() -> None:
    backends = {backend.descriptor.backend_id: backend for backend in solver_backends()}
    assert set(backends) == {"astap", "native", "astrometry-net"}
    assert backends["native"].descriptor.execution_ready is False
    assert all(
        not backend.descriptor.execution_ready or backend.descriptor.available
        for backend in backends.values()
    )
    if backends["astrometry-net"].descriptor.execution_ready:
        assert (
            "catalog-correspondence-quality-v1"
            in backends["astrometry-net"].descriptor.capabilities
        )
    assert all(isinstance(backend, SolverBackend) for backend in backends.values())
