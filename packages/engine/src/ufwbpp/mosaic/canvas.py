"""The mosaic canvas: one projection and pixel lattice for every panel and filter.

The canvas is an exact TAN projection (STG once the mosaic spans more than
``STG_EXTENT_DEGREES``) without distortion terms.  Its tangent point is the
centroid of the union of the panels' footprints, its axes follow the median
panel orientation modulo a half turn (the smallest canvas for a mosaic shot
with one camera angle, whether or not some panels went through a meridian
flip) and its pixel scale is the panels' scale at their centres.  Every
panel run resamples its Lights onto an integer-offset window of this lattice
(:func:`window_grid`), or drizzles them onto it through the window's forward
lattice (:func:`forward_lattice`), so all panels and filters share one pixel
grid and the mosaic and colour products never resample a master again.  Whether two
panels overlap is decided on their footprint polygons on the canvas, not on
their bounding boxes (:func:`panels_connected`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from numpy.typing import NDArray

from ..stacking.parameters import OutputGrid


CANVAS_ALGORITHM = "mosaic-canvas-v2"
# Lattice node spacing of a canvas window (a power of two: exact fractions).
LATTICE_SPACING = 16
# Beyond this span a gnomonic canvas stretches its edges by more than 1 %.
STG_EXTENT_DEGREES = 10.0
# Panels whose centre pixel scales differ by more than this fraction are
# taken as different cameras: the canvas takes the finest scale.
PIXEL_SCALE_SPREAD = 0.02
CANVAS_MARGIN_PIXELS = 64
WINDOW_MARGIN_PIXELS = 2
# A lattice node farther outside the reference frame than this fraction of
# the frame is not extrapolated through its distortion polynomial (NaN).
LATTICE_REFERENCE_MARGIN = 0.25
# Grid of the union-footprint centroid, along the longer axis.
_CENTROID_GRID = 256
_BOUNDARY_SAMPLES = 17


class CanvasError(RuntimeError):
    """A canvas that cannot be built honestly (fail closed)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def celestial_wcs(header: Mapping[str, Any]) -> WCS:
    """The celestial WCS of header cards (TAN, STG or TAN-SIP)."""

    cards = fits.Header()
    for key, value in header.items():
        if value is not None:
            cards[str(key)] = value
    return WCS(cards, relax=True).celestial


def _unit_vectors(ra: NDArray[np.float64], dec: NDArray[np.float64]) -> NDArray[np.float64]:
    ra_radians = np.deg2rad(ra)
    dec_radians = np.deg2rad(dec)
    cos_dec = np.cos(dec_radians)
    return np.column_stack(
        (cos_dec * np.cos(ra_radians), cos_dec * np.sin(ra_radians), np.sin(dec_radians))
    )


def _radec(vector: NDArray[np.float64]) -> tuple[float, float]:
    x, y, z = (float(value) for value in vector / np.linalg.norm(vector))
    return math.degrees(math.atan2(y, x)) % 360.0, math.degrees(math.asin(max(-1.0, min(1.0, z))))


def _plane(crval: tuple[float, float], projection: str = "TAN") -> WCS:
    """Projection-plane coordinates (degrees) about ``crval``: FITS pixel
    coordinates of a WCS with CRPIX (0, 0) and the identity CD matrix."""

    plane = WCS(naxis=2)
    plane.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
    plane.wcs.crval = [float(crval[0]), float(crval[1])]
    plane.wcs.crpix = [0.0, 0.0]
    plane.wcs.cd = np.eye(2)
    return plane


def rotation_scale_parity(cd: NDArray[np.float64]) -> tuple[float, float, int]:
    """Decompose a CD matrix as ``scale * R(angle) * diag(parity, 1)``.

    ``parity`` is -1 for the ordinary sky orientation (east left when north
    is up) and +1 for a mirrored image; ``angle`` is in radians.
    """

    determinant = float(np.linalg.det(cd))
    parity = -1 if determinant < 0.0 else 1
    rotation = cd @ np.diag((float(parity), 1.0))
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0])), math.sqrt(abs(determinant)), parity


def _orientation_median(angles: Sequence[float]) -> float:
    """The median orientation of the panels' axes, modulo a half turn: a
    panel shot after a meridian flip lies 180 degrees from the others and
    has the same axes.  The representative nearest north-up is returned, in
    (-pi/2, pi/2]."""

    first = angles[0]
    unwrapped = [first + math.remainder(angle - first, math.pi) for angle in angles]
    return math.remainder(float(np.median(unwrapped)), math.pi)


@dataclass(frozen=True)
class CanvasProjection:
    """The canvas lattice on the sky.

    Canvas pixel ``(0, 0)`` (zero-based) is FITS pixel ``(1, 1)``; ``crpix``
    is the FITS pixel of the tangent point.  A window whose zero-based
    canvas offset is ``origin`` carries CRPIX ``crpix - origin``.
    """

    projection: str
    crval: tuple[float, float]
    crpix: tuple[float, float]
    cd: tuple[tuple[float, float], tuple[float, float]]

    def header(self, origin: tuple[int, int] = (0, 0)) -> dict[str, Any]:
        return {
            "WCSAXES": 2,
            "CTYPE1": f"RA---{self.projection}",
            "CTYPE2": f"DEC--{self.projection}",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRVAL1": float(self.crval[0]),
            "CRVAL2": float(self.crval[1]),
            "CRPIX1": float(self.crpix[0] - origin[0]),
            "CRPIX2": float(self.crpix[1] - origin[1]),
            "CD1_1": float(self.cd[0][0]),
            "CD1_2": float(self.cd[0][1]),
            "CD2_1": float(self.cd[1][0]),
            "CD2_2": float(self.cd[1][1]),
            "RADESYS": "ICRS",
        }

    def wcs(self, origin: tuple[int, int] = (0, 0)) -> WCS:
        return celestial_wcs(self.header(origin))

    def scaled(self, factor: int) -> "CanvasProjection":
        """The canvas ``factor`` times finer, its pixel ``factor*i`` centred
        on pixel ``i`` of this one (the canvas of drizzled panels)."""

        return CanvasProjection(
            self.projection,
            self.crval,
            (factor * (self.crpix[0] - 1.0) + 1.0, factor * (self.crpix[1] - 1.0) + 1.0),
            ((self.cd[0][0] / factor, self.cd[0][1] / factor), (self.cd[1][0] / factor, self.cd[1][1] / factor)),
        )

    @property
    def pixel_scale_arcsec(self) -> float:
        return math.sqrt(abs(float(np.linalg.det(np.asarray(self.cd))))) * 3600.0

    def world_to_canvas(
        self, ra: NDArray[np.float64], dec: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x, y = self.wcs().wcs_world2pix(np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64), 0)
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def canvas_to_world(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ra, dec = self.wcs().wcs_pix2world(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), 0)
        return np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64)

    def serializable(self) -> dict[str, Any]:
        angle, scale, parity = rotation_scale_parity(np.asarray(self.cd, dtype=np.float64))
        return {
            "algorithm": CANVAS_ALGORITHM,
            "projection": self.projection,
            "crval": [float(self.crval[0]), float(self.crval[1])],
            "crpix": [float(self.crpix[0]), float(self.crpix[1])],
            "cd": [[float(value) for value in row] for row in self.cd],
            "pixelScaleArcsec": scale * 3600.0,
            "rotationDegrees": math.degrees(angle),
            "parity": parity,
        }


@dataclass(frozen=True)
class PanelFootprint:
    """One panel's solved frame: where its pixels lie on the sky."""

    key: str
    header: Mapping[str, Any]
    shape: tuple[int, int]

    def wcs(self) -> WCS:
        return celestial_wcs(self.header)

    def boundary_pixels(self, samples: int = _BOUNDARY_SAMPLES) -> NDArray[np.float64]:
        """Zero-based points along the outer pixel edges, counter-clockwise."""

        return frame_boundary(self.shape, samples)

    def boundary_world(self, samples: int = _BOUNDARY_SAMPLES) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        points = self.boundary_pixels(samples)
        ra, dec = self.wcs().all_pix2world(points[:, 0], points[:, 1], 0)
        return np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64)

    def centre(self) -> tuple[float, float]:
        height, width = self.shape
        ra, dec = self.wcs().all_pix2world([(width - 1) / 2.0], [(height - 1) / 2.0], 0)
        return float(ra[0]) % 360.0, float(dec[0])

    def local_cd(self) -> NDArray[np.float64]:
        """The CD matrix (degrees per pixel) of the solution at the frame
        centre, distortion included: central differences of one pixel in
        the tangent plane of the centre."""

        height, width = self.shape
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        wcs = self.wcs()
        ra, dec = wcs.all_pix2world([cx + 0.5, cx - 0.5, cx, cx], [cy, cy, cy + 0.5, cy - 0.5], 0)
        plane = _plane(self.centre())
        xi, eta = plane.wcs_world2pix(np.asarray(ra), np.asarray(dec), 1)
        return np.asarray(((xi[0] - xi[1], xi[2] - xi[3]), (eta[0] - eta[1], eta[2] - eta[3])), dtype=np.float64)


def frame_boundary(
    shape: tuple[int, int], samples: int = _BOUNDARY_SAMPLES, *, inset: float = 0.0
) -> NDArray[np.float64]:
    """Zero-based points along a frame's outer pixel edges (moved ``inset``
    pixels inwards), counter-clockwise."""

    height, width = shape
    left, right, top, bottom = -0.5 + inset, width - 0.5 - inset, -0.5 + inset, height - 0.5 - inset
    t = np.linspace(0.0, 1.0, samples, endpoint=False)
    edges = (
        np.column_stack((left + (right - left) * t, np.full(samples, top))),
        np.column_stack((np.full(samples, right), top + (bottom - top) * t)),
        np.column_stack((right - (right - left) * t, np.full(samples, bottom))),
        np.column_stack((np.full(samples, left), bottom - (bottom - top) * t)),
    )
    return np.vstack(edges)


@dataclass(frozen=True)
class CanvasPlan:
    projection: CanvasProjection
    width: int
    height: int
    panels: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...] = ()

    def serializable(self) -> dict[str, Any]:
        return {
            **self.projection.serializable(),
            "width": self.width,
            "height": self.height,
            "panels": [dict(panel) for panel in self.panels],
            "warnings": [dict(item) for item in self.warnings],
        }


def _inside_polygon(points_x: NDArray[np.float64], points_y: NDArray[np.float64], polygon: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Even-odd ray casting of many points against one polygon."""

    inside = np.zeros(points_x.shape, dtype=bool)
    xs, ys = polygon[:, 0], polygon[:, 1]
    for index in range(len(polygon)):
        x0, y0 = xs[index - 1], ys[index - 1]
        x1, y1 = xs[index], ys[index]
        if y0 == y1:
            continue
        crosses = (y0 > points_y) != (y1 > points_y)
        intersection = (x1 - x0) * (points_y - y0) / (y1 - y0) + x0
        inside ^= crosses & (points_x < intersection)
    return inside


# Points across the bounding boxes' intersection that decide whether two
# footprints share area.
_OVERLAP_SAMPLES = 128


def polygons_overlap(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    """Whether two simple polygons share area: some point of a grid over
    the intersection of their bounding boxes lies inside both.  Polygons
    that merely touch do not overlap."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x0, y0 = max(a[:, 0].min(), b[:, 0].min()), max(a[:, 1].min(), b[:, 1].min())
    x1, y1 = min(a[:, 0].max(), b[:, 0].max()), min(a[:, 1].max(), b[:, 1].max())
    if not (x1 > x0 and y1 > y0):
        return False
    steps = (np.arange(_OVERLAP_SAMPLES) + 0.5) / _OVERLAP_SAMPLES
    grid_x, grid_y = np.meshgrid(x0 + (x1 - x0) * steps, y0 + (y1 - y0) * steps)
    return bool((_inside_polygon(grid_x, grid_y, a) & _inside_polygon(grid_x, grid_y, b)).any())


def panels_connected(polygons: Mapping[str, NDArray[np.float64]]) -> bool:
    """Whether the panels' canvas footprints overlap into one mosaic."""

    keys = sorted(polygons)
    if len(keys) <= 1:
        return True
    linked = {keys[0]}
    frontier = [keys[0]]
    while frontier:
        current = polygons[frontier.pop()]
        for key in keys:
            if key not in linked and polygons_overlap(current, polygons[key]):
                linked.add(key)
                frontier.append(key)
    return len(linked) == len(keys)


def _union_centroid(footprints: Sequence[PanelFootprint]) -> tuple[float, float]:
    """Sky position of the centroid of the union of the footprints, on a
    grid in the tangent plane of their mean centre (overlaps count once)."""

    centres = np.vstack([_unit_vectors(np.asarray([ra]), np.asarray([dec])) for ra, dec in (fp.centre() for fp in footprints)])
    provisional = _radec(centres.sum(axis=0))
    plane = _plane(provisional)
    polygons = []
    for footprint in footprints:
        ra, dec = footprint.boundary_world()
        xi, eta = plane.wcs_world2pix(ra, dec, 1)
        polygons.append(np.column_stack((xi, eta)))
    stacked = np.vstack(polygons)
    if not np.all(np.isfinite(stacked)):
        raise CanvasError("MOSAIC_FOOTPRINT_INVALID", "a panel footprint does not project onto the canvas plane")
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    span = high - low
    step = float(max(span)) / _CENTROID_GRID
    xs = low[0] + step * (np.arange(int(math.ceil(span[0] / step)) + 1) + 0.5)
    ys = low[1] + step * (np.arange(int(math.ceil(span[1] / step)) + 1) + 0.5)
    grid_x, grid_y = np.meshgrid(xs, ys)
    covered = np.zeros(grid_x.shape, dtype=bool)
    for polygon in polygons:
        covered |= _inside_polygon(grid_x, grid_y, polygon)
    if not covered.any():
        raise CanvasError("MOSAIC_FOOTPRINT_INVALID", "the panel footprints cover no area")
    ra, dec = plane.wcs_pix2world([float(grid_x[covered].mean())], [float(grid_y[covered].mean())], 1)
    return float(ra[0]) % 360.0, float(dec[0])


def _extent_degrees(footprints: Sequence[PanelFootprint]) -> float:
    vectors = np.vstack([_unit_vectors(*footprint.boundary_world(5)) for footprint in footprints])
    cosine = np.clip(vectors @ vectors.T, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine.min())))


def plan_canvas(
    footprints: Sequence[PanelFootprint],
    *,
    margin: int = CANVAS_MARGIN_PIXELS,
) -> CanvasPlan:
    """Plan the canvas of a mosaic from one solved frame of every panel."""

    if not footprints:
        raise CanvasError("MOSAIC_PANELS_MISSING", "a canvas needs at least one solved panel")
    warnings: list[dict[str, Any]] = []
    decomposed = []
    for footprint in footprints:
        cd = footprint.local_cd()
        if not np.all(np.isfinite(cd)) or abs(float(np.linalg.det(cd))) <= 0.0:
            raise CanvasError("MOSAIC_FOOTPRINT_INVALID", f"panel {footprint.key} has a singular solution")
        decomposed.append(rotation_scale_parity(cd))
    parities = [parity for _, _, parity in decomposed]
    parity = max(set(parities), key=lambda value: (parities.count(value), value))
    if len(set(parities)) > 1:
        warnings.append(
            {
                "code": "MOSAIC_PANEL_PARITY_DIFFERS",
                "message": "some panels are mirrored relative to the others; the canvas takes the majority orientation",
                "panels": [footprint.key for footprint, (_, _, item) in zip(footprints, decomposed) if item != parity],
            }
        )
    angles = [angle for angle, _, item in decomposed if item == parity]
    angle = _orientation_median(angles)
    scales = np.asarray([scale for _, scale, _ in decomposed], dtype=np.float64)
    median_scale = float(np.median(scales))
    spread = float((scales.max() - scales.min()) / median_scale)
    if spread > PIXEL_SCALE_SPREAD:
        scale = float(scales.min())
        warnings.append(
            {
                "code": "MOSAIC_PIXEL_SCALES_DIFFER",
                "message": "panel pixel scales differ (different cameras or optics); the canvas takes the finest",
                "scalesArcsec": [float(value) * 3600.0 for value in scales],
            }
        )
    else:
        scale = median_scale
    extent = _extent_degrees(footprints)
    projection = "STG" if extent > STG_EXTENT_DEGREES else "TAN"
    crval = _union_centroid(footprints)
    cd = scale * np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))), dtype=np.float64
    ) @ np.diag((float(parity), 1.0))
    provisional = CanvasProjection(
        projection=projection,
        crval=crval,
        crpix=(0.0, 0.0),
        cd=((float(cd[0, 0]), float(cd[0, 1])), (float(cd[1, 0]), float(cd[1, 1]))),
    )
    boxes = []
    polygons = []
    all_x: list[NDArray[np.float64]] = []
    all_y: list[NDArray[np.float64]] = []
    for footprint in footprints:
        x, y = provisional.world_to_canvas(*footprint.boundary_world())
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise CanvasError("MOSAIC_FOOTPRINT_INVALID", f"panel {footprint.key} does not project onto the canvas")
        all_x.append(x)
        all_y.append(y)
        boxes.append((float(x.min()), float(y.min()), float(x.max()), float(y.max())))
        polygons.append(np.column_stack((x, y)))
    x0 = math.floor(min(float(item.min()) for item in all_x)) - margin
    y0 = math.floor(min(float(item.min()) for item in all_y)) - margin
    x1 = math.ceil(max(float(item.max()) for item in all_x)) + margin
    y1 = math.ceil(max(float(item.max()) for item in all_y)) + margin
    canvas = CanvasProjection(
        projection=projection,
        crval=crval,
        crpix=(float(-x0), float(-y0)),
        cd=provisional.cd,
    )
    panels = tuple(
        {
            "key": footprint.key,
            "centre": list(footprint.centre()),
            "pixelScaleArcsec": float(item_scale) * 3600.0,
            "rotationDegrees": math.degrees(item_angle),
            "parity": item_parity,
            "canvasBox": [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0],
            # The footprint on the canvas, counter-clockwise along the frame edges.
            "canvasPolygon": [[float(px - x0), float(py - y0)] for px, py in polygon],
        }
        for footprint, (item_angle, item_scale, item_parity), box, polygon in zip(footprints, decomposed, boxes, polygons)
    )
    return CanvasPlan(
        projection=canvas,
        width=int(x1 - x0 + 1),
        height=int(y1 - y0 + 1),
        panels=panels,
        warnings=tuple(warnings),
    )


def canvas_window(
    projection: CanvasProjection,
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    *,
    margin: int = WINDOW_MARGIN_PIXELS,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """The integer window ``(origin, (height, width))`` that holds every
    sky point given (the footprints of a panel's Lights)."""

    x, y = projection.world_to_canvas(ra, dec)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise CanvasError("MOSAIC_FOOTPRINT_INVALID", "a Light footprint does not project onto the canvas")
    x0 = math.floor(float(x.min())) - margin
    y0 = math.floor(float(y.min())) - margin
    x1 = math.ceil(float(x.max())) + margin
    y1 = math.ceil(float(y.max())) + margin
    return (int(x0), int(y0)), (int(y1 - y0 + 1), int(x1 - x0 + 1))


def window_grid(
    projection: CanvasProjection,
    origin: tuple[int, int],
    shape: tuple[int, int],
    reference_header: Mapping[str, Any],
    reference_shape: tuple[int, int],
    *,
    spacing: int = LATTICE_SPACING,
    forward_domain: tuple[float, float, float, float] | None = None,
) -> OutputGrid:
    """The canvas window as an output grid of a panel run: the reference
    pixel coordinates of its lattice nodes, through the canvas projection
    and the reference's own solution (distortion included).  With
    ``forward_domain`` (x0, y0, x1, y1 in reference pixels, every point the
    run's Lights reach) the grid also holds the forward map a drizzle drops
    through: the window coordinates of reference nodes covering it."""

    height, width = shape
    rows = (height - 1 + spacing - 1) // spacing + 1
    columns = (width - 1 + spacing - 1) // spacing + 1
    node_y, node_x = np.mgrid[:rows, :columns].astype(np.float64) * float(spacing)
    ra, dec = projection.canvas_to_world(node_x.ravel() + origin[0], node_y.ravel() + origin[1])
    reference = celestial_wcs(reference_header)
    if reference.sip is None:
        reference_x, reference_y = reference.wcs_world2pix(ra, dec, 0)
    else:
        reference_x, reference_y = reference.all_world2pix(
            ra, dec, 0, tolerance=1e-8, maxiter=100, adaptive=False, detect_divergence=True, quiet=True
        )
    reference_x = np.asarray(reference_x, dtype=np.float64)
    reference_y = np.asarray(reference_y, dtype=np.float64)
    reference_height, reference_width = reference_shape
    margin_x = LATTICE_REFERENCE_MARGIN * reference_width
    margin_y = LATTICE_REFERENCE_MARGIN * reference_height
    usable = (
        np.isfinite(reference_x)
        & np.isfinite(reference_y)
        & (reference_x >= -margin_x)
        & (reference_x <= reference_width - 1 + margin_x)
        & (reference_y >= -margin_y)
        & (reference_y <= reference_height - 1 + margin_y)
    )
    if usable.any() and reference.sip is not None:
        # The iterative inverse must land on the sky point it was asked for.
        check_ra, check_dec = reference.all_pix2world(reference_x[usable], reference_y[usable], 0)
        back_x, back_y = projection.world_to_canvas(np.asarray(check_ra), np.asarray(check_dec))
        expected_x = node_x.ravel()[usable] + origin[0]
        expected_y = node_y.ravel()[usable] + origin[1]
        converged = (np.abs(back_x - expected_x) < 1e-5) & (np.abs(back_y - expected_y) < 1e-5)
        indices = np.flatnonzero(usable)
        usable[indices[~converged]] = False
    reference_x[~usable] = np.nan
    reference_y[~usable] = np.nan
    forward: dict[str, Any] = {}
    if forward_domain is not None:
        x0 = math.floor(forward_domain[0] / spacing) * spacing - spacing
        y0 = math.floor(forward_domain[1] / spacing) * spacing - spacing
        forward_columns = int(math.ceil((forward_domain[2] - x0) / spacing)) + 2
        forward_rows = int(math.ceil((forward_domain[3] - y0) / spacing)) + 2
        forward = forward_lattice(
            projection, origin, reference_header, (float(x0), float(y0)), (forward_rows, forward_columns), spacing=spacing
        )
    grid = OutputGrid(
        width=int(width),
        height=int(height),
        spacing=int(spacing),
        reference_x=reference_x.reshape(rows, columns),
        reference_y=reference_y.reshape(rows, columns),
        canvas_origin=(int(origin[0]), int(origin[1])),
        wcs=projection.header(origin),
        **forward,
    )
    grid.validate()
    return grid


# Fixed-point iterations that invert a lateral-colour field (a few pixels
# across a frame, so each iteration gains three or more decimal places).
_SHIFT_INVERSE_ITERATIONS = 6


def forward_lattice(
    projection: CanvasProjection,
    origin: tuple[int, int],
    reference_header: Mapping[str, Any],
    lattice_origin: tuple[float, float],
    lattice_shape: tuple[int, int],
    *,
    spacing: int = LATTICE_SPACING,
    shift: Callable[[NDArray[np.float64], NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]]
    | None = None,
) -> dict[str, Any]:
    """The forward map a drizzle drops through, as the ``forward_*`` fields
    of an :class:`OutputGrid`: the window coordinates of the
    registration-reference points ``lattice_origin + (column, row)*spacing``
    (``lattice_shape`` is ``(rows, columns)``), through the reference's own
    solution and the canvas projection.

    With ``shift`` (a filter's lateral colour: a Light of that filter shows
    at reference point ``p + shift(p)`` what the reference shows at ``p``)
    the lattice is that filter's: its node at reference point ``q`` lands
    where the reference point ``p`` with ``p + shift(p) = q`` lands, so the
    filter's Lights drop onto the same canvas pixels the window's lattice
    resamples them onto."""

    rows, columns = lattice_shape
    node_y, node_x = np.mgrid[:rows, :columns].astype(np.float64) * float(spacing)
    reference_x = node_x.ravel() + float(lattice_origin[0])
    reference_y = node_y.ravel() + float(lattice_origin[1])
    if shift is not None:
        point_x, point_y = reference_x, reference_y
        for _ in range(_SHIFT_INVERSE_ITERATIONS):
            dx, dy = shift(point_x, point_y)
            point_x, point_y = reference_x - dx, reference_y - dy
        reference_x, reference_y = point_x, point_y
    reference = celestial_wcs(reference_header)
    ra, dec = reference.all_pix2world(reference_x, reference_y, 0)
    window_x, window_y = projection.world_to_canvas(np.asarray(ra), np.asarray(dec))
    return {
        "forward_x": np.ascontiguousarray((window_x - origin[0]).reshape(rows, columns)),
        "forward_y": np.ascontiguousarray((window_y - origin[1]).reshape(rows, columns)),
        "forward_origin": (float(lattice_origin[0]), float(lattice_origin[1])),
    }


__all__ = [
    "CANVAS_ALGORITHM",
    "LATTICE_SPACING",
    "STG_EXTENT_DEGREES",
    "CanvasError",
    "CanvasPlan",
    "CanvasProjection",
    "PanelFootprint",
    "canvas_window",
    "celestial_wcs",
    "forward_lattice",
    "frame_boundary",
    "panels_connected",
    "plan_canvas",
    "polygons_overlap",
    "rotation_scale_parity",
    "window_grid",
]
