"""Reuse triangle vertices while preserving astroalign's bootstrap search.

The coordinate-only bootstrap below is adapted from astroalign 2.6.2. The
invariants, triangle order, randomized RANSAC trials, fits and tolerances remain
upstream's. Only repeated coordinate transforms and catalog geometry are reused.
Other astroalign versions use the public implementation.
"""

# MIT License
# Copyright (c) 2016-2019 Martin Beroiz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import lru_cache

import astroalign
import numpy as np
from scipy.spatial import KDTree
from skimage.transform import SimilarityTransform


@lru_cache(maxsize=64)
def _geometry(coordinates: bytes, nearest_neighbors: int) -> tuple:
    # Bytes own the cached coordinates, so caller mutation cannot poison a hit.
    # The neighbor setting is part of the key because astroalign exposes it.
    points = np.frombuffer(coordinates, dtype=np.float64).reshape(-1, 2)
    invariants, triangles = astroalign._generate_invariants(points)
    invariants.setflags(write=False)
    triangles.setflags(write=False)
    return points, triangles, KDTree(invariants)


class _TriangleMatchTransform:
    def __init__(self, source: np.ndarray, target: np.ndarray, matches: np.ndarray):
        self._model = astroalign._MatchTransform(source, target)
        self._source = source
        pairs = np.unique(matches.reshape(-1, 2), axis=0)
        self._source_indices = pairs[:, 0]
        self._target_points = target[pairs[:, 1]]
        self._pair_indices = np.zeros((len(source), len(target)), dtype=np.int64)
        self._pair_indices[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs))

    def fit(self, data: np.ndarray) -> SimilarityTransform:
        return self._model.fit(data)

    def get_error(self, data: np.ndarray, model: SimilarityTransform) -> np.ndarray:
        # Every trial evaluates the same vertex pairs, often repeated across
        # thousands of triangles. Keep the upstream residual arithmetic while
        # transforming each control star and evaluating each pair only once.
        residuals = np.sqrt(np.sum(
            (model(self._source)[self._source_indices] - self._target_points) ** 2,
            axis=1,
        ))
        indices = self._pair_indices[data[:, :, 0], data[:, :, 1]]
        return residuals[indices].max(axis=1)


def find_catalog_transform(
    source: np.ndarray, target: np.ndarray, *, max_control_points: int
) -> SimilarityTransform:
    """Return the same bootstrap model; matched control points are not needed."""

    if getattr(astroalign, "__version__", None) != "2.6.2" or not all(
        hasattr(astroalign, name)
        for name in ("_generate_invariants", "_MatchTransform", "_ransac")
    ):
        return astroalign.find_transform(
            source, target, max_control_points=max_control_points
        )[0]

    source_points, source_triangles, source_tree = _geometry(
        np.asarray(source[:max_control_points], dtype=np.float64).tobytes(),
        astroalign.NUM_NEAREST_NEIGHBORS,
    )
    target_points, target_triangles, target_tree = _geometry(
        np.asarray(target[:max_control_points], dtype=np.float64).tobytes(),
        astroalign.NUM_NEAREST_NEIGHBORS,
    )
    neighbors = source_tree.query_ball_tree(target_tree, r=0.1)
    matches = np.array([
        list(zip(source_triangle, target_triangle))
        for source_triangle, targets in zip(source_triangles, neighbors)
        for target_triangle in target_triangles[targets]
    ])
    # Preserve upstream's empty-match failure as well as its successful path.
    model = (
        _TriangleMatchTransform(source_points, target_points, matches)
        if matches.size
        else astroalign._MatchTransform(source_points, target_points)
    )
    minimum = max(1, min(10, int(len(matches) * astroalign.MIN_MATCHES_FRACTION)))
    if (len(source_points) == 3 or len(target_points) == 3) and len(matches) == 1:
        return model.fit(matches)
    return astroalign._ransac(matches, model, astroalign.PIXEL_TOL, minimum)[0]
