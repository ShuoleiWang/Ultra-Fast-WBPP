"""The two opt-in advanced algorithms: proper coaddition and robust IRLS.

Both are off by default, and the first thing every test here establishes is
that a default recipe, a default parameter record and a default integration
are exactly what they were before the options existed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ufwbpp.calibration import (
    DEFAULT_COMBINATION,
    FrameExpression,
    IntegrationParameters,
    integrate_expressions,
)
from ufwbpp.image_io.fits import CalibrationError, FitsFloatWriter, FitsFrame
from ufwbpp.proper_coaddition import (
    PROPER_COADD_ALGORITHM_ID,
    ProperCoadditionParameters,
    measure_frame_psf,
    proper_coadd_group,
    working_set_bytes,
)
from ufwbpp.recipe import Recipe, RecipeError


HEIGHT = 320
WIDTH = 384
SKY = 200.0


def _write(path: Path, values: np.ndarray) -> Path:
    with FitsFloatWriter(path, values.shape, {}, durable=False) as writer:
        writer.write_rows(0, np.asarray(values, dtype=np.float32))
    return path


def _read(path: Path) -> np.ndarray:
    with FitsFrame(path) as frame:
        return np.asarray(frame.read_rows(0, frame.shape[0]), dtype=np.float64)


def _star_field(rng: np.random.Generator, fwhm: float, noise: float, stars) -> np.ndarray:
    sigma = fwhm / 2.3548
    rows = np.arange(HEIGHT)[:, None]
    columns = np.arange(WIDTH)[None, :]
    image = np.full((HEIGHT, WIDTH), SKY, dtype=np.float64)
    for x, y, flux in stars:
        r2 = (rows - y) ** 2 + (columns - x) ** 2
        image += flux / (2 * math.pi * sigma**2) * np.exp(-r2 / (2 * sigma**2))
    return (image + rng.normal(0.0, noise, image.shape)).astype(np.float32)


def _group(tmp_path: Path, frames: int = 6, *, seed: int = 11):
    rng = np.random.default_rng(seed)
    # Stars on a jittered grid: far enough apart that the PSF stamps do not
    # overlap, which is what the measurement needs.
    stars = [
        (
            float(48 + 48 * column + rng.uniform(-6, 6)),
            float(48 + 48 * row + rng.uniform(-6, 6)),
            float(10 ** rng.uniform(3.4, 4.2)),
        )
        for row in range((HEIGHT - 80) // 48)
        for column in range((WIDTH - 80) // 48)
    ]
    paths = []
    for index in range(frames):
        values = _star_field(rng, 2.8 + 0.25 * index, 4.0 + 0.5 * index, stars)
        paths.append(_write(tmp_path / f"frame{index}.fits", values))
    return [FrameExpression(str(path)) for path in paths]


# --------------------------------------------------------------------------
# Defaults are unchanged
# --------------------------------------------------------------------------


def test_default_recipe_does_not_serialize_the_new_blocks() -> None:
    serialized = Recipe.from_dict({}).serializable()
    assert "properCoaddition" not in serialized
    assert "integration" not in serialized


def test_default_integration_parameters_do_not_serialize_a_combination() -> None:
    assert IntegrationParameters().combination == DEFAULT_COMBINATION
    assert "combination" not in IntegrationParameters().serializable()


def test_explicit_default_combination_is_still_omitted() -> None:
    recipe = Recipe.from_dict({"integration": {"combination": "sigma-clip-v2"}})
    assert "integration" not in recipe.serializable()


# --------------------------------------------------------------------------
# Recipe contract
# --------------------------------------------------------------------------


def test_proper_coaddition_block_round_trips() -> None:
    recipe = Recipe.from_dict(
        {"properCoaddition": {"enabled": True, "apodizationPixels": 32}}
    )
    assert recipe.proper_coaddition.enabled
    assert recipe.proper_coaddition.outlier_handling == "reuse-rejection"
    assert recipe.proper_coaddition.apodization_pixels == 32
    assert Recipe.from_dict(recipe.serializable()) == recipe


def test_proper_coaddition_refuses_unknown_keys_and_bad_values() -> None:
    with pytest.raises(RecipeError, match="unknown keys"):
        Recipe.from_dict({"properCoaddition": {"enabled": True, "nope": 1}})
    with pytest.raises(RecipeError, match="outlierHandling"):
        Recipe.from_dict({"properCoaddition": {"outlierHandling": "ignore"}})
    with pytest.raises(RecipeError, match="apodizationPixels"):
        Recipe.from_dict({"properCoaddition": {"apodizationPixels": -1}})


def test_proper_coaddition_and_drizzle_are_refused_together() -> None:
    with pytest.raises(RecipeError, match="drizzle"):
        Recipe.from_dict(
            {"properCoaddition": {"enabled": True}, "drizzle": {"enabled": True}}
        )


def test_combination_must_be_known() -> None:
    with pytest.raises(RecipeError, match="combination"):
        Recipe.from_dict({"integration": {"combination": "median"}})
    recipe = Recipe.from_dict({"integration": {"combination": "irls-huber"}})
    assert recipe.serializable()["integration"] == {"combination": "irls-huber"}


# --------------------------------------------------------------------------
# Robust IRLS combination
# --------------------------------------------------------------------------


def test_irls_is_recorded_and_moves_the_master_towards_the_robust_centre(
    tmp_path: Path,
) -> None:
    expressions = _group(tmp_path, frames=7)
    ordinary = integrate_expressions(
        expressions, tmp_path / "ordinary.fits", durable=False
    )
    robust = integrate_expressions(
        expressions,
        tmp_path / "robust.fits",
        parameters=IntegrationParameters(combination="irls-huber"),
        durable=False,
    )
    combination = robust.execution["combination"]
    assert combination["rule"] == "irls-huber"
    assert combination["iterations"] == 4
    assert combination["tuningConstant"] == pytest.approx(1.345)
    assert "combination" not in ordinary.execution
    first = _read(tmp_path / "ordinary.fits")
    second = _read(tmp_path / "robust.fits")
    # The same image, but not the same pixels: IRLS downweights the samples
    # that sigma clipping kept and the background stays where it was.
    assert not np.array_equal(first, second)
    assert float(np.nanmedian(second - first)) == pytest.approx(0.0, abs=0.5)
    assert np.nanstd(second - first) < 5.0


def test_irls_downweights_a_contaminated_sample(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    base = np.full((HEIGHT, WIDTH), SKY, dtype=np.float32)
    paths = []
    for index in range(7):
        values = base + rng.normal(0.0, 1.0, base.shape).astype(np.float32)
        if index == 0:
            # Just inside the 4 sigma clip, so ordinary integration keeps it.
            values[40, 40] = np.float32(SKY + 3.2)
        paths.append(_write(tmp_path / f"c{index}.fits", values))
    expressions = [FrameExpression(str(path)) for path in paths]
    integrate_expressions(expressions, tmp_path / "plain.fits", durable=False)
    integrate_expressions(
        expressions,
        tmp_path / "huber.fits",
        parameters=IntegrationParameters(combination="irls-huber"),
        durable=False,
    )
    plain = _read(tmp_path / "plain.fits")[40, 40]
    huber = _read(tmp_path / "huber.fits")[40, 40]
    assert abs(huber - SKY) < abs(plain - SKY)


def test_irls_keeps_gaussian_efficiency(tmp_path: Path) -> None:
    """On samples that are only noise, Huber costs a few percent, not more.

    ``k = 1.345`` is chosen for 95 % asymptotic efficiency at the normal, so
    a master built from pure Gaussian frames must not get measurably noisier.
    """

    rng = np.random.default_rng(1)
    paths = [
        _write(
            tmp_path / f"g{index}.fits",
            (SKY + rng.normal(0.0, 20.0, (HEIGHT, WIDTH))).astype(np.float32),
        )
        for index in range(20)
    ]
    expressions = [FrameExpression(str(path)) for path in paths]
    integrate_expressions(expressions, tmp_path / "mean.fits", durable=False)
    integrate_expressions(
        expressions,
        tmp_path / "huber.fits",
        parameters=IntegrationParameters(combination="irls-huber"),
        durable=False,
    )
    mean = _read(tmp_path / "mean.fits")
    huber = _read(tmp_path / "huber.fits")
    assert float(np.std(huber)) < 1.06 * float(np.std(mean))
    assert abs(float(np.median(huber - mean))) < 0.05 * float(np.std(mean))


def test_default_combination_is_reproducible(tmp_path: Path) -> None:
    expressions = _group(tmp_path, frames=5)
    first = integrate_expressions(expressions, tmp_path / "a.fits", durable=False)
    second = integrate_expressions(expressions, tmp_path / "b.fits", durable=False)
    assert first.output_sha256 == second.output_sha256


def test_irls_is_reproducible(tmp_path: Path) -> None:
    expressions = _group(tmp_path, frames=5)
    parameters = IntegrationParameters(combination="irls-huber")
    first = integrate_expressions(
        expressions, tmp_path / "a.fits", parameters=parameters, durable=False
    )
    second = integrate_expressions(
        expressions, tmp_path / "b.fits", parameters=parameters, durable=False
    )
    assert first.output_sha256 == second.output_sha256


# --------------------------------------------------------------------------
# Proper coaddition
# --------------------------------------------------------------------------


def _coadd(tmp_path: Path, expressions, **kwargs):
    integration = integrate_expressions(
        expressions, tmp_path / "master.fits", durable=False
    )
    return integration, proper_coadd_group(
        expressions,
        tmp_path / "proper.fits",
        master_path=tmp_path / "master.fits",
        shape=(HEIGHT, WIDTH),
        flux_scales=[1.0] * len(expressions),
        parameters=ProperCoadditionParameters(enabled=True, **kwargs.pop("options", {})),
        max_memory_bytes=1 << 30,
        workers=2,
        durable=False,
        **kwargs,
    )


def test_proper_coadd_keeps_the_background_and_concentrates_the_stars(
    tmp_path: Path,
) -> None:
    expressions = _group(tmp_path, frames=8)
    _, result = _coadd(
        tmp_path,
        expressions,
        accepted_bits=None,
        options={"outlier_handling": "none", "apodization_pixels": 16},
    )
    master = _read(tmp_path / "master.fits")
    proper = _read(tmp_path / "proper.fits")
    interior = (slice(48, HEIGHT - 48), slice(48, WIDTH - 48))
    # Same photometric units: the background sits where the ordinary master's
    # background sits, and the total flux above it is conserved.
    assert float(np.median(proper[interior])) == pytest.approx(
        float(np.median(master[interior])), abs=1.0
    )
    master_flux = float(np.sum(master[interior] - np.median(master[interior])))
    proper_flux = float(np.sum(proper[interior] - np.median(proper[interior])))
    assert proper_flux == pytest.approx(master_flux, rel=0.05)
    # The coadd PSF is sharper than any linear average can be, so the same
    # flux sits in fewer pixels: the star cores are brighter.  The median of
    # the 40 brightest pixels is the stable form of that statement.
    def cores(image: np.ndarray) -> float:
        values = np.sort(image[interior].ravel())[-40:]
        return float(np.median(values) - np.median(image[interior]))

    assert cores(proper) > cores(master)
    assert result.serializable()["algorithm"] == PROPER_COADD_ALGORITHM_ID
    assert result.coadd_fwhm_pixels > 0
    assert result.flux_scale_norm > 0
    # Nothing worth keeping was dropped from the transform.
    assert result.floored_spectrum_fraction < 1e-3
    assert all(item["psfSource"] == "measured-star-stack" for item in result.frames)


def test_proper_coadd_does_not_attenuate_its_own_border(tmp_path: Path) -> None:
    """The apodization lives in the guard band, never in the published pixels.

    The run's common crop is only a few tens of pixels wide, so a taper applied
    to the frame itself would reach into the product and quietly darken its
    outermost columns.
    """

    expressions = _group(tmp_path, frames=6, seed=23)
    _, result = _coadd(
        tmp_path,
        expressions,
        accepted_bits=None,
        options={"outlier_handling": "none", "apodization_pixels": 64},
    )
    proper = _read(tmp_path / "proper.fits")

    def robust_sigma(block: np.ndarray) -> float:
        d = np.diff(block, axis=1).ravel()
        return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2.0))

    inside = robust_sigma(proper[HEIGHT // 2 : HEIGHT // 2 + 8])
    for name, strip in (
        ("top", proper[:8]),
        ("bottom", proper[-8:]),
        ("left", proper[:, :8].T),
        ("right", proper[:, -8:].T),
    ):
        edge = robust_sigma(strip)
        # A taper applied to the frame would leave these strips at a few
        # percent of the field's noise; the guard band leaves them intact.
        assert 0.7 * inside < edge < 1.4 * inside, name
    assert result.padded_shape[0] >= HEIGHT + 128
    assert result.padded_shape[1] >= WIDTH + 128


def test_proper_coadd_replaces_the_samples_the_rejection_removed(
    tmp_path: Path,
) -> None:
    expressions = _group(tmp_path, frames=6, seed=5)
    accepted = [
        np.packbits(np.ones((HEIGHT, WIDTH), dtype=bool), axis=1) for _ in expressions
    ]
    # One frame carries a satellite trail; its samples are marked rejected.
    trail = np.zeros((HEIGHT, WIDTH), dtype=bool)
    trail[160, 40:340] = True
    accepted[0] = np.packbits(~trail, axis=1)
    contaminated = _read(Path(expressions[0].source_path))
    contaminated[trail] += 20000.0
    _write(tmp_path / "trail.fits", contaminated)
    expressions[0] = FrameExpression(str(tmp_path / "trail.fits"))
    _, result = _coadd(tmp_path, expressions, accepted_bits=accepted)
    assert result.replaced_samples == int(np.count_nonzero(trail))
    proper = _read(tmp_path / "proper.fits")
    # The trail did not survive into the transform: the row it occupied is
    # statistically the same as its neighbours.
    row = proper[160, 60:320]
    neighbours = proper[150, 60:320]
    assert abs(float(np.median(row)) - float(np.median(neighbours))) < 5.0


def test_proper_coadd_requires_masks_for_reuse_rejection(tmp_path: Path) -> None:
    expressions = _group(tmp_path, frames=4)
    with pytest.raises(CalibrationError, match="PROPER_COADD_REJECTION_UNAVAILABLE"):
        _coadd(tmp_path, expressions, accepted_bits=None)


def test_proper_coadd_refuses_to_page(tmp_path: Path) -> None:
    expressions = _group(tmp_path, frames=4)
    integrate_expressions(expressions, tmp_path / "master.fits", durable=False)
    with pytest.raises(CalibrationError, match="PROPER_COADD_MEMORY"):
        proper_coadd_group(
            expressions,
            tmp_path / "proper.fits",
            master_path=tmp_path / "master.fits",
            shape=(HEIGHT, WIDTH),
            flux_scales=[1.0] * len(expressions),
            accepted_bits=None,
            parameters=ProperCoadditionParameters(
                enabled=True, outlier_handling="none"
            ),
            max_memory_bytes=1024,
            durable=False,
        )


def test_working_set_is_reported_and_bounded() -> None:
    small = working_set_bytes((HEIGHT, WIDTH))
    large = working_set_bytes((4176, 6252))
    assert small < large
    # The full reference frame stays well inside the M-series tuning budget.
    assert large < 2 * 1024**3


def test_psf_falls_back_to_a_moffat_without_stars() -> None:
    flat = np.zeros((64, 64), dtype=np.float32)
    psf = measure_frame_psf(flat, 1.0, fallback_fwhm=4.0)
    assert psf.source.startswith("moffat")
    assert psf.star_count == 0
    assert psf.kernel.shape == (31, 31)
    assert float(np.sum(psf.kernel)) == pytest.approx(1.0, rel=1e-5)
    assert psf.fwhm_pixels == pytest.approx(4.0)


def test_measured_psf_tracks_the_frame_it_came_from() -> None:
    rng = np.random.default_rng(19)
    stars = [
        (float(rng.uniform(40, 460)), float(rng.uniform(40, 460)), 40000.0)
        for _ in range(60)
    ]
    sigma = 3.6 / 2.3548
    rows = np.arange(500)[:, None]
    columns = np.arange(500)[None, :]
    image = np.zeros((500, 500), dtype=np.float64)
    for x, y, flux in stars:
        r2 = (rows - y) ** 2 + (columns - x) ** 2
        image += flux / (2 * math.pi * sigma**2) * np.exp(-r2 / (2 * sigma**2))
    image += rng.normal(0.0, 3.0, image.shape)
    psf = measure_frame_psf(image.astype(np.float32), 3.0)
    assert psf.source == "measured-star-stack"
    assert psf.star_count >= 8
    assert psf.fwhm_pixels == pytest.approx(3.6, abs=0.45)


def test_psf_stars_are_isolated_from_the_dropped_bright_stars() -> None:
    from ufwbpp.proper_coaddition import PSF_STAMP_RADIUS, _psf_star_indices

    rng = np.random.default_rng(5)
    # 40 detections, brightest first: the first two are dropped as likely
    # saturated, and a faint star sits inside the stamp box of each of them.
    rows = [100, 400] + [int(value) for value in rng.integers(700, 5000, 38)]
    columns = [100, 400] + [int(value) for value in rng.integers(700, 5000, 38)]
    rows[20], columns[20] = 108, 110
    rows[21], columns[21] = 395, 420
    selected = _psf_star_indices(
        np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64), PSF_STAMP_RADIUS
    )
    assert 0 not in selected and 1 not in selected
    assert 20 not in selected and 21 not in selected
    assert selected.size > 0


def test_mask_recorder_reports_unobserved_rows() -> None:
    from types import SimpleNamespace

    from ufwbpp.pixel_pipeline import _RejectionMaskRecorder

    recorder = _RejectionMaskRecorder(2, (6, 10))
    recorder(SimpleNamespace(first_row=0, accepted=np.ones((2, 4, 10), dtype=bool)))
    assert not recorder.complete
    recorder(SimpleNamespace(first_row=4, accepted=np.ones((2, 2, 10), dtype=bool)))
    assert recorder.complete


def test_frames_prepared_ahead_give_the_sequential_coadd_bit_for_bit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from ufwbpp import proper_coaddition

    expressions = _group(tmp_path, frames=5)
    # A starless frame in the middle falls back to the Moffat of the median
    # FWHM measured on the frames before it, whatever thread measured them.
    blank = _write(
        tmp_path / "blank.fits",
        np.random.default_rng(3).normal(SKY, 5.0, (HEIGHT, WIDTH)).astype(np.float32),
    )
    expressions.insert(2, FrameExpression(str(blank)))
    digests = []
    records = []
    for ahead in (1, 3):
        monkeypatch.setattr(proper_coaddition, "PREPARE_AHEAD", ahead)
        run = tmp_path / f"ahead{ahead}"
        run.mkdir()
        _, result = _coadd(
            run,
            expressions,
            accepted_bits=None,
            options={"outlier_handling": "none", "apodization_pixels": 16},
        )
        digests.append(hashlib.sha256((run / "proper.fits").read_bytes()).hexdigest())
        records.append([(item["psfSource"], item["psfFwhmPixels"]) for item in result.frames])
    assert digests[0] == digests[1]
    assert records[0] == records[1]
    assert records[0][2][0].startswith("moffat")
    measured = [fwhm for source, fwhm in records[0][:2]]
    assert records[0][2][1] == pytest.approx(float(np.median(measured)))
