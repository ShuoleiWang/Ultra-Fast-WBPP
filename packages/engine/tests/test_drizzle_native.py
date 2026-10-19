"""Native drizzle: kernel properties through the Python bridge and the group
runner's products (FITS extensions, receipt, CFA planes, verification)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from ufwbpp import native_kernels
from ufwbpp.stacking.drizzle_native import (
    CFA_PATTERNS,
    DRIZZLE_ALGORITHM,
    SUPPORTED_KERNELS,
    SUPPORTED_SCALES,
    DrizzleError,
    DrizzleFrame,
    DrizzleGroupRequest,
    drizzle_group,
    verify_drizzle_receipt,
)

KERNELS = native_kernels.load_native_kernels()
requires_native = pytest.mark.skipif(
    KERNELS is None or not hasattr(KERNELS, "drizzle_band"),
    reason="native drizzle kernel library not available",
)

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _translation(dx: float, dy: float) -> tuple[tuple[float, float, float], ...]:
    return ((1.0, 0.0, dx), (0.0, 1.0, dy), (0.0, 0.0, 1.0))


def _band(
    source: np.ndarray,
    *,
    forward: np.ndarray,
    scale: int,
    kernel: str = "square",
    pixfrac: float = 1.0,
    threads: int = 1,
    output_shape: tuple[int, int] | None = None,
    **extra,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = source.shape
    shape = output_shape or (height * scale, width * scale)
    total = np.zeros(shape, dtype=np.float64)
    weight = np.zeros(shape, dtype=np.float64)
    touched = np.zeros(shape, dtype=np.uint8)
    KERNELS.drizzle_band(
        source,
        source_row0=0,
        forward=forward,
        scale=scale,
        pixfrac=pixfrac,
        kernel=kernel,
        output_sum=total,
        output_weight=weight,
        threads=threads,
        output_touched=touched,
        **extra,
    )
    return total, weight, touched


@requires_native
def test_identity_unit_drop_copies_pixels_and_skips_nan() -> None:
    rng = np.random.default_rng(3)
    source = rng.uniform(10.0, 20.0, (13, 17)).astype(np.float32)
    source[4, 6] = np.nan
    total, weight, touched = _band(source, forward=np.eye(3), scale=1)
    finite = np.isfinite(source)
    assert np.allclose(weight[finite], 1.0, atol=1e-12)
    assert np.allclose(total[finite], source[finite], atol=1e-6)
    assert weight[4, 6] == 0.0 and touched[4, 6] == 0
    assert np.array_equal(touched, finite.astype(np.uint8))


@requires_native
@pytest.mark.parametrize("through", ["matrix", "lattice"])
@pytest.mark.parametrize("kernel", SUPPORTED_KERNELS)
def test_kernels_are_thread_invariant_and_conserve_dropped_weight(kernel: str, through: str) -> None:
    rng = np.random.default_rng(5)
    source = rng.uniform(100.0, 200.0, (21, 19)).astype(np.float32)
    angle = 0.2
    scale = 2
    # Rotation and translation that keep every drop inside a padded output.
    registration = np.array(
        [[np.cos(angle), -np.sin(angle), 6.0], [np.sin(angle), np.cos(angle), 3.0], [0.0, 0.0, 1.0]]
    )
    forward = np.diag([scale, scale, 1.0]) @ registration
    shape = (scale * (21 + 12), scale * (19 + 12))
    pixfrac = 0.8
    extra: dict = {}
    if through == "lattice":
        # A mosaic canvas: the registration into the reference, then a
        # lattice of output coordinates (here the plain scale).
        if not getattr(KERNELS, "has_lattice_drizzle", False):
            pytest.skip("native lattice drizzle not available")
        nodes_y, nodes_x = np.mgrid[:9, :9].astype(np.float64) * 8.0 - 16.0
        extra = {"lattice_u": scale * nodes_x, "lattice_v": scale * nodes_y, "lattice_origin": (-16.0, -16.0), "lattice_spacing": 8.0}
        matrix = _band(source, forward=forward, scale=scale, kernel=kernel, pixfrac=pixfrac, output_shape=shape)
        forward = registration
    serial = _band(source, forward=forward, scale=scale, kernel=kernel, pixfrac=pixfrac, output_shape=shape, **extra)
    if through == "lattice":
        assert np.allclose(serial[0], matrix[0], rtol=1e-9, atol=1e-9)
        assert np.allclose(serial[1], matrix[1], rtol=1e-9, atol=1e-12)
    for threads in (2, 5, 16):
        threaded = _band(
            source, forward=forward, scale=scale, kernel=kernel, pixfrac=pixfrac, threads=threads, output_shape=shape, **extra
        )
        assert np.array_equal(serial[0], threaded[0]) and np.array_equal(serial[1], threaded[1])
    pixels = source.size
    drop_area = (pixfrac * scale) ** 2
    expected = {
        "square": pixels * drop_area,
        "gaussian": pixels * drop_area,
        "circular": pixels * drop_area * np.pi / 4.0,
        "point": float(pixels),
    }[kernel]
    assert serial[1].sum() == pytest.approx(expected, rel=1e-6)
    # Flux (sum of value*weight) follows the same conservation.
    per_pixel = {"point": 1.0}.get(kernel, drop_area * (np.pi / 4.0 if kernel == "circular" else 1.0))
    assert serial[0].sum() == pytest.approx(float(source.astype(np.float64).sum()) * per_pixel, rel=1e-6)


@requires_native
def test_mask_normalization_weight_grid_and_cfa_channel_selection() -> None:
    source = np.arange(1, 1 + 8 * 8, dtype=np.float32).reshape(8, 8)
    mask = np.ones((8, 8), dtype=np.uint8)
    mask[2, 3] = 0
    total, weight, _ = _band(
        source,
        forward=np.eye(3),
        scale=1,
        mask=mask,
        normalization_scale=2.0,
        normalization_offset=-1.0,
        frame_weight=0.5,
    )
    assert weight[2, 3] == 0.0
    expected = np.where(mask.astype(bool), (source * np.float32(2.0)) + np.float32(-1.0), 0.0)
    assert np.allclose(total, 0.5 * expected, atol=1e-6)
    # The offset grid adds its bilinear value at the reference position.
    grid = np.array([[10.0, 30.0], [10.0, 30.0]])
    total_grid, _, _ = _band(
        source,
        forward=np.eye(3),
        scale=1,
        grid=grid,
        grid_x_nodes=np.array([0.0, 7.0]),
        grid_y_nodes=np.array([0.0, 7.0]),
    )
    column_offsets = 10.0 + 20.0 * np.arange(8) / 7.0
    assert np.allclose(total_grid, source + column_offsets[None, :], atol=1e-4)
    # Region weights multiply the frame weight; a zero node zeroes its corner.
    weight_grid = np.array([[0.0, 1.0], [1.0, 1.0]])
    _, weighted, _ = _band(
        source,
        forward=np.eye(3),
        scale=1,
        weight_grid=weight_grid,
        weight_grid_x_nodes=np.array([0.0, 7.0]),
        weight_grid_y_nodes=np.array([0.0, 7.0]),
    )
    assert weighted[0, 0] == 0.0 and weighted[7, 7] == pytest.approx(1.0)
    assert weighted[0, 7] == pytest.approx(1.0) and weighted[3, 0] == pytest.approx(3.0 / 7.0, abs=1e-9)
    # CFA planes partition the mosaic by the 2x2 pattern.
    planes = []
    for channel in range(3):
        _, plane_weight, _ = _band(
            source, forward=np.eye(3), scale=1, cfa_pattern=CFA_PATTERNS["RGGB"], channel=channel
        )
        planes.append(plane_weight)
    assert np.allclose(sum(planes), 1.0)
    assert planes[0][0, 0] == 1.0 and planes[0][0, 1] == 0.0 and planes[2][1, 1] == 1.0
    assert planes[1].sum() == 32.0 and planes[0].sum() == 16.0 and planes[2].sum() == 16.0


@requires_native
def test_bridge_rejects_bad_accumulators() -> None:
    source = np.ones((4, 4), dtype=np.float32)
    total = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        KERNELS.drizzle_band(
            source, source_row0=0, forward=np.eye(3), scale=1, pixfrac=1.0, kernel="square",
            output_sum=total, output_weight=total,
        )
    good = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ValueError):
        KERNELS.drizzle_band(
            source, source_row0=0, forward=np.eye(3), scale=1, pixfrac=1.0, kernel="lanczos",
            output_sum=good, output_weight=good.copy(),
        )
    with pytest.raises(ValueError):
        KERNELS.drizzle_band(
            source, source_row0=0, forward=np.eye(3), scale=1, pixfrac=1.0, kernel="square",
            output_sum=good, output_weight=good.copy(), output_touched=np.zeros((4, 4), dtype=np.int16),
        )
    with pytest.raises(native_kernels.NativeKernelError):
        KERNELS.drizzle_band(
            source, source_row0=0, forward=np.eye(3), scale=9, pixfrac=1.0, kernel="square",
            output_sum=good, output_weight=good.copy(),
        )


def _write_light(path: Path, values: np.ndarray) -> str:
    header = fits.Header()
    header["IMAGETYP"] = "Light"
    header["EXPTIME"] = 60.0
    fits.PrimaryHDU(np.asarray(values, dtype=np.float32), header=header).writeto(path)
    return str(path)


def _frames(tmp_path: Path, shape: tuple[int, int] = (24, 32)) -> list[DrizzleFrame]:
    rng = np.random.default_rng(11)
    sky = rng.uniform(100.0, 110.0, shape).astype(np.float32)
    frames = []
    # Three dithered exposures of one field; the third carries a 5x scale so
    # the normalization must undo it, and a cosmic ray the mask rejects.
    for index, (dx, dy, scale_factor) in enumerate(((0.0, 0.0, 1.0), (0.5, 0.25, 1.0), (1.25, 0.75, 5.0))):
        values = (sky * scale_factor).astype(np.float32)
        mask_bits = None
        if index == 2:
            values[10, 12] = 1.0e6
            accepted = np.ones(shape, dtype=np.uint8)
            accepted[10 + int(round(dy)), 12 + int(round(dx))] = 0
            mask_bits = np.packbits(accepted, axis=1)
        frames.append(
            DrizzleFrame(
                calibrated_path=_write_light(tmp_path / f"light_{index}.fits", values),
                # NINA's default names carry a degree sign; the receipt is UTF-8.
                source_path=f"/raw/light_{index}_-10.00°C.fits",
                input_to_reference=_translation(dx, dy),
                weight=1.0 if index else 2.0,
                exposure_seconds=60.0,
                normalization_scale=1.0 / scale_factor,
                accepted_mask_bits=mask_bits,
            )
        )
    return frames


@requires_native
def test_drizzle_group_writes_verified_products(tmp_path: Path) -> None:
    frames = _frames(tmp_path)
    request = DrizzleGroupRequest(
        frames=tuple(frames),
        reference_shape=(24, 32),
        output_path=str(tmp_path / "out" / "master_drizzle.fits"),
        receipt_path=str(tmp_path / "out" / "receipt.json"),
        scale=2,
        pixfrac=0.9,
        kernel="square",
        metadata={"FILTER": "L", "OAFSTATE": "UNSOLVED"},
        durable=False,
    )
    result = drizzle_group(request)
    receipt = verify_drizzle_receipt(result.receipt_path)
    assert receipt["algorithm"] == DRIZZLE_ALGORITHM
    assert receipt["recipe"] == {
        "scale": 2, "pixfrac": 0.9, "kernel": "square", "cfaPattern": None, "cfaChannel": None,
        "inputUnits": "normalized-integration-frame",
    }
    assert receipt["geometry"]["outputHeight"] == 48 and receipt["geometry"]["outputWidth"] == 64
    assert receipt["statistics"]["inputFrames"] == 3
    assert receipt["statistics"]["acceptedInputPixels"] == 3 * 24 * 32 - 1
    assert len(receipt["statistics"]["frameSeconds"]) == 3
    assert receipt["inputs"][2]["rejectionMask"] is True and receipt["inputs"][0]["rejectionMask"] is False
    # memmap=False: the artifact is overwritten below, which Windows refuses
    # while a NumPy view still maps the file.
    with fits.open(result.output_path, memmap=False) as hdul:
        assert [hdu.name for hdu in hdul] == ["SCI", "WHT", "COVERAGE"]
        science = np.asarray(hdul["SCI"].data, dtype=np.float64)
        weights = np.asarray(hdul["WHT"].data, dtype=np.float64)
        coverage = np.asarray(hdul["COVERAGE"].data)
        header = hdul[0].header
    assert header["OAFDRZ"] == "NATIVE" and header["OAFDRZSC"] == 2 and header["OAFDRZKN"] == "square"
    assert header["OAFNFRM"] == 3 and header["FILTER"] == "L" and header["OAFSTATE"] == "UNSOLVED"
    assert science.shape == (48, 64) and coverage.dtype.kind == "i" and coverage.dtype.itemsize == 2
    interior = coverage[6:42, 6:58]
    # Every interior pixel sees all three frames except where the rejected
    # cosmic-ray pixel of frame 2 was not dropped (one 0.9-pixel drop at 2x
    # touches at most a 3x3 block).
    assert interior.min() == 2 and interior.max() == 3
    assert 1 <= np.count_nonzero(interior == 2) <= 9
    hole = np.argwhere(interior == 2) + np.array([6, 6])
    assert np.all(np.abs(hole[:, 0] - 21.5) <= 1.5) and np.all(np.abs(hole[:, 1] - 26.5) <= 1.5)
    # The cosmic ray was rejected through the mask: the normalized mean stays
    # at sky level everywhere the frames overlap.
    covered = np.isfinite(science) & (coverage >= 3)
    assert 95.0 < np.nanmedian(science[covered]) < 115.0
    assert float(np.nanmax(science[covered])) < 200.0
    assert np.all(weights[covered] > 0.0)
    assert np.all(np.isnan(science[coverage == 0]))
    assert receipt["statistics"]["coverageFraction"] == pytest.approx(float((coverage > 0).mean()))
    assert receipt["artifact"]["extensions"] == ["SCI", "WHT", "COVERAGE"]
    # Create-only publication.
    with pytest.raises(DrizzleError) as error:
        drizzle_group(request)
    assert error.value.code == "OUTPUT_EXISTS"
    # A changed artifact fails verification.
    Path(result.output_path).write_bytes(b"corrupted")
    with pytest.raises(DrizzleError) as changed:
        verify_drizzle_receipt(result.receipt_path)
    assert changed.value.code == "DRIZZLE_ARTIFACT_CHANGED"


@requires_native
def test_scale_one_identity_drizzle_reproduces_the_weighted_mean(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    shape = (16, 20)
    values = [rng.uniform(50.0, 60.0, shape).astype(np.float32) for _ in range(3)]
    weights = (1.0, 2.0, 0.5)
    frames = tuple(
        DrizzleFrame(
            calibrated_path=_write_light(tmp_path / f"f{index}.fits", value),
            source_path=f"/raw/f{index}.fits",
            input_to_reference=IDENTITY,
            weight=weight,
            exposure_seconds=30.0,
        )
        for index, (value, weight) in enumerate(zip(values, weights, strict=True))
    )
    request = DrizzleGroupRequest(
        frames=frames,
        reference_shape=shape,
        output_path=str(tmp_path / "one.fits"),
        receipt_path=str(tmp_path / "one.json"),
        scale=1,
        pixfrac=1.0,
        kernel="square",
        durable=False,
    )
    drizzle_group(request)
    science = np.asarray(fits.getdata(request.output_path), dtype=np.float64)
    expected = sum(w * v.astype(np.float64) for w, v in zip(weights, values, strict=True)) / sum(weights)
    assert np.allclose(science, expected, atol=1e-4)


@requires_native
def test_drizzle_onto_a_canvas_window_drops_through_its_forward_lattice(tmp_path: Path) -> None:
    """Frames of their own shape drizzled onto a larger canvas window whose
    forward lattice is a pure shift: the output is the matrix drizzle moved
    by the shift, the rejection mask is read at the window position, and
    the receipt names the lattice kernel and the window."""

    if not getattr(KERNELS, "has_lattice_drizzle", False):
        pytest.skip("native lattice drizzle not available")
    from ufwbpp.stacking.parameters import OutputGrid

    frames = _frames(tmp_path)
    frame_shape = (24, 32)
    window_shape = (40, 50)
    shift = (7.0, 5.0)  # reference pixel (x, y) lands on window pixel (x + 7, y + 5)
    scale = 2
    node_y, node_x = np.mgrid[:6, :6].astype(np.float64) * 16.0 - 16.0
    grid = OutputGrid(
        width=window_shape[1],
        height=window_shape[0],
        spacing=16,
        reference_x=np.asarray(node_x[:4, :5] - shift[0] + 16.0),
        reference_y=np.asarray(node_y[:4, :5] - shift[1] + 16.0),
        canvas_origin=(100, 200),
        wcs={"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": 10.0, "CRVAL2": 20.0, "CRPIX1": 1.0, "CRPIX2": 1.0},
        forward_x=np.asarray(node_x + shift[0]),
        forward_y=np.asarray(node_y + shift[1]),
        forward_origin=(-16.0, -16.0),
    )
    # The masks of the ordinary integration live on the window: move frame 2's
    # rejected pixel there.
    moved = []
    for frame in frames:
        bits = frame.accepted_mask_bits
        if bits is not None:
            accepted = np.ones(window_shape, dtype=np.uint8)
            rejected = np.argwhere(np.unpackbits(bits, axis=1, count=frame_shape[1]) == 0)
            for row, column in rejected:
                accepted[row + int(shift[1]), column + int(shift[0])] = 0
            bits = np.packbits(accepted, axis=1)
        moved.append(DrizzleFrame(**{**{f: getattr(frame, f) for f in DrizzleFrame.__slots__}, "accepted_mask_bits": bits}))
    on_window = DrizzleGroupRequest(
        frames=tuple(moved),
        reference_shape=window_shape,
        output_path=str(tmp_path / "window" / "drizzle.fits"),
        receipt_path=str(tmp_path / "window" / "receipt.json"),
        scale=scale,
        pixfrac=0.9,
        output_grid=grid,
        durable=False,
    )
    result = drizzle_group(on_window)
    receipt = verify_drizzle_receipt(result.receipt_path)
    assert receipt["backend"]["id"] == "native-cpu-drizzle-v2-lattice"
    assert receipt["geometry"]["outputHeight"] == scale * window_shape[0]
    assert receipt["geometry"]["canvasWindow"]["forwardLattice"] == {"origin": [-16.0, -16.0], "nodes": [6, 6]}
    assert receipt["geometry"]["outputPixelCentre"] == "output index = scale * canvas window index"
    # The same frames straight onto the reference grid.
    plain = DrizzleGroupRequest(
        frames=tuple(frames),
        reference_shape=frame_shape,
        output_path=str(tmp_path / "plain" / "drizzle.fits"),
        receipt_path=str(tmp_path / "plain" / "receipt.json"),
        scale=scale,
        pixfrac=0.9,
        durable=False,
    )
    drizzle_group(plain)
    with fits.open(on_window.output_path, memmap=False) as hdul:
        window_science = np.asarray(hdul["SCI"].data, dtype=np.float64)
        window_coverage = np.asarray(hdul["COVERAGE"].data)
    with fits.open(plain.output_path, memmap=False) as hdul:
        plain_science = np.asarray(hdul["SCI"].data, dtype=np.float64)
        plain_coverage = np.asarray(hdul["COVERAGE"].data)
    # Inside the reference grid's edges (the plain output clips the drops of
    # dithered frames there, the window holds them) the two agree.
    dy, dx = scale * int(shift[1]), scale * int(shift[0])
    edge = 2 * scale
    rows, columns = plain_science.shape
    moved_science = window_science[dy + edge : dy + rows - edge, dx + edge : dx + columns - edge]
    moved_coverage = window_coverage[dy + edge : dy + rows - edge, dx + edge : dx + columns - edge]
    plain_science = plain_science[edge:-edge, edge:-edge]
    plain_coverage = plain_coverage[edge:-edge, edge:-edge]
    assert np.array_equal(moved_coverage, plain_coverage)
    assert np.array_equal(np.isnan(moved_science), np.isnan(plain_science))
    finite = np.isfinite(plain_science)
    assert np.allclose(moved_science[finite], plain_science[finite], rtol=1e-6, atol=1e-3)
    assert np.count_nonzero(plain_coverage == 2) >= 1 and np.count_nonzero(plain_coverage == 3) > 0.9 * plain_coverage.size
    assert np.all(window_coverage[: dy - edge] == 0) and np.all(window_coverage[:, : dx - edge] == 0)
    # A window without its forward lattice, or of another shape, is refused,
    # by the drizzle and by the pipeline parameters that capture its inputs.
    from ufwbpp.stacking.parameters import PipelineParameters

    bare = OutputGrid(**{f: getattr(grid, f) for f in ("width", "height", "spacing", "reference_x", "reference_y", "canvas_origin", "wcs")})
    for wrong in (dict(output_grid=bare), dict(output_grid=grid, reference_shape=(41, 50))):
        with pytest.raises(DrizzleError) as error:
            DrizzleGroupRequest(**{**{f: getattr(plain, f) for f in DrizzleGroupRequest.__slots__}, **wrong}).validate()
        assert error.value.code == "DRIZZLE_GEOMETRY_INVALID"
    PipelineParameters(output_grid=grid, capture_drizzle_inputs=True).validate()
    with pytest.raises(ValueError, match="forward lattice"):
        PipelineParameters(output_grid=bare, capture_drizzle_inputs=True).validate()
    with pytest.raises(ValueError, match="forward lattice"):
        PipelineParameters(output_grid=grid, channel_grids=(("B", None, bare),), capture_drizzle_inputs=True).validate()


@requires_native
def test_cfa_drizzle_produces_three_planes_from_the_mosaic(tmp_path: Path) -> None:
    shape = (16, 16)
    yy, xx = np.indices(shape)
    # RGGB mosaic with a distinct constant per colour channel.
    mosaic = np.where((yy % 2 == 0) & (xx % 2 == 0), 10.0, np.where((yy % 2 == 1) & (xx % 2 == 1), 30.0, 20.0)).astype(np.float32)
    frames = tuple(
        DrizzleFrame(
            calibrated_path=_write_light(tmp_path / f"cfa{index}.fits", mosaic),
            source_path=f"/raw/cfa{index}.fits",
            input_to_reference=_translation(*offset),
            weight=1.0,
            exposure_seconds=30.0,
        )
        for index, offset in enumerate(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    )
    request = DrizzleGroupRequest(
        frames=frames,
        reference_shape=shape,
        output_path=str(tmp_path / "cfa.fits"),
        receipt_path=str(tmp_path / "cfa.json"),
        scale=1,
        pixfrac=1.0,
        kernel="square",
        cfa_pattern="rggb",
        durable=False,
    )
    result = drizzle_group(request)
    receipt = verify_drizzle_receipt(result.receipt_path)
    assert receipt["recipe"]["cfaPattern"] == "RGGB" and receipt["geometry"]["channels"] == 3
    with fits.open(result.output_path) as hdul:
        science = np.asarray(hdul["SCI"].data, dtype=np.float64)
        assert hdul[0].header["OAFDRZCF"] == "RGGB"
    assert science.shape == (3, 16, 16)
    # With whole-pixel dithers in both axes every colour reaches every pixel.
    inner = science[:, 2:14, 2:14]
    assert np.allclose(inner[0], 10.0) and np.allclose(inner[1], 20.0) and np.allclose(inner[2], 30.0)


def test_request_validation_reports_specific_codes(tmp_path: Path) -> None:
    frame = DrizzleFrame(
        calibrated_path=str(tmp_path / "missing.fits"),
        source_path="/raw/missing.fits",
        input_to_reference=IDENTITY,
        weight=1.0,
        exposure_seconds=1.0,
    )
    base = dict(
        frames=(frame,), reference_shape=(4, 4), output_path=str(tmp_path / "o.fits"),
        receipt_path=str(tmp_path / "o.json"),
    )
    cases = [
        (dict(base, frames=()), "DRIZZLE_NO_INPUTS"),
        (dict(base, scale=5), "DRIZZLE_SCALE_INVALID"),
        (dict(base, pixfrac=0.0), "DRIZZLE_PIXFRAC_INVALID"),
        (dict(base, kernel="lanczos"), "DRIZZLE_KERNEL_INVALID"),
        (dict(base, cfa_pattern="XYZW"), "DRIZZLE_CFA_PATTERN_INVALID"),
        (dict(base, max_accumulator_bytes=1), "DRIZZLE_MEMORY_BUDGET_TOO_SMALL"),
        (dict(base, frames=(DrizzleFrame(frame.calibrated_path, frame.source_path, ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 1.0, 1.0),)), "DRIZZLE_TRANSFORM_INVALID"),
        (dict(base, frames=(DrizzleFrame(frame.calibrated_path, frame.source_path, IDENTITY, -1.0, 1.0),)), "DRIZZLE_WEIGHT_INVALID"),
        (dict(base, frames=(DrizzleFrame(frame.calibrated_path, frame.source_path, IDENTITY, 1.0, 0.0),)), "DRIZZLE_EXPOSURE_INVALID"),
        (dict(base, frames=(DrizzleFrame(frame.calibrated_path, frame.source_path, IDENTITY, 1.0, 1.0, offset_grid=((1.0,),), offset_grid_x=(0.0, 1.0), offset_grid_y=(0.0,)),)), "DRIZZLE_GRID_INVALID"),
        (dict(base, frames=(DrizzleFrame(frame.calibrated_path, frame.source_path, IDENTITY, 1.0, 1.0, accepted_mask_bits=np.zeros((3, 1), dtype=np.uint8)),)), "DRIZZLE_MASK_INVALID"),
    ]
    for kwargs, code in cases:
        with pytest.raises(DrizzleError) as error:
            DrizzleGroupRequest(**kwargs).validate()
        assert error.value.code == code, code
    assert SUPPORTED_SCALES == (1, 2, 3, 4)
    assert set(SUPPORTED_KERNELS) == {"square", "circular", "gaussian", "point"}
    assert set(CFA_PATTERNS) == {"RGGB", "BGGR", "GRBG", "GBRG"}


def test_receipt_verification_rejects_tampered_content(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"stage": "drizzle", "status": "failed"}), encoding="ascii")
    with pytest.raises(DrizzleError) as error:
        verify_drizzle_receipt(path)
    assert error.value.code == "DRIZZLE_RECEIPT_INVALID"


def test_drizzle_discovery_does_not_advertise_a_fake_executor():
    from ufwbpp.stacking.drizzle import drizzle_backends, DrizzleCapabilityProvider
    provider = drizzle_backends()[0]
    assert isinstance(provider, DrizzleCapabilityProvider)
    assert not hasattr(provider, "drizzle")


@requires_native
def test_distorted_square_drops_match_the_numpy_polygon_reference() -> None:
    if not getattr(KERNELS, "has_lattice_drizzle", False):
        pytest.skip("native lattice drizzle not available")
    from ufwbpp.reference.drizzle_numpy import square_lattice_drizzle

    source = np.arange(24, dtype=np.float32).reshape(4, 6) / np.float32(24)
    source[1, 2] = np.nan
    registration = np.array([[1.0, 0.04, 0.31], [-0.03, 1.0, 0.23], [0.001, -0.002, 1.0]])
    y, x = np.mgrid[:6, :7].astype(np.float64) * 2.0 - 2.0
    u = 1.6 * x + 1.5 + 0.012 * x * y
    v = 1.6 * y + 1.5 + 0.009 * x * x
    extra = dict(lattice_u=u, lattice_v=v, lattice_origin=(-2.0, -2.0), lattice_spacing=2.0)
    actual = _band(source, forward=registration, scale=2, pixfrac=0.9, output_shape=(10, 13), threads=3, **extra)
    expected = square_lattice_drizzle(source, registration, u, v, (-2.0, -2.0), 2.0, (10, 13), pixfrac=0.9)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(actual[2], expected[1] > 0)
