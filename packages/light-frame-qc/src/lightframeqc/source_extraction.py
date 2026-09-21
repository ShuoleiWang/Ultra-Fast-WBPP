"""Determinism self-test of SEP's source extraction.

``sep.extract`` allocates its Lutz scan buffers and deblending arrays
uninitialised (``QMALLOC`` in ``lutz.c`` and ``deblend.c``) and reads some
entries before writing them.  On macOS and Linux that memory is stable, so
the same preview array always yields the same objects; on Windows the
randomised heap made six identical calls return 3141-3143 objects with
different fluxes, and the QC star counts, quality weights and every master
differed between two runs of the same data.  The engine cannot see the
heap, so it measures: two deblended extractions of a crowded synthetic
field, with allocations between them, must be identical.  The E2E receipt
records the verdict and a run on a non-deterministic build carries a
``SEP_NONDETERMINISTIC`` warning.
"""

from __future__ import annotations

from functools import cache
import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray


SELF_TEST_VERSION = "sep-extraction-self-test-v1"
SELF_TEST_SIZE = 256
SELF_TEST_SOURCES = 260
SELF_TEST_SEED = 20260921
NONDETERMINISTIC_WARNING = "SEP_NONDETERMINISTIC"


def crowded_field(
    size: int = SELF_TEST_SIZE,
    *,
    source_count: int = SELF_TEST_SOURCES,
    seed: int = SELF_TEST_SEED,
) -> NDArray[np.float32]:
    """A background-subtracted star field dense enough that sources blend.

    Gaussians of varied width and brightness are placed with a density that
    makes many of them overlap, so the deblender (the code path whose
    buffers are uninitialised) works on every extraction.  Read noise is
    added from a seeded generator; the array is exactly reproducible.
    """

    rng = np.random.default_rng(seed)
    image = rng.normal(0.0, 3.0, size=(size, size)).astype(np.float64)
    yy, xx = np.mgrid[0:size, 0:size]
    x = rng.uniform(4.0, size - 5.0, source_count)
    y = rng.uniform(4.0, size - 5.0, source_count)
    sigma = rng.uniform(1.2, 3.2, source_count)
    peak = rng.lognormal(np.log(120.0), 0.9, source_count)
    half = 12
    for index in range(source_count):
        cx, cy, width, height = float(x[index]), float(y[index]), float(sigma[index]), float(peak[index])
        x0, x1 = max(0, int(cx) - half), min(size, int(cx) + half + 1)
        y0, y1 = max(0, int(cy) - half), min(size, int(cy) + half + 1)
        window_x = xx[y0:y1, x0:x1]
        window_y = yy[y0:y1, x0:x1]
        image[y0:y1, x0:x1] += height * np.exp(
            -((window_x - cx) ** 2 + (window_y - cy) ** 2) / (2.0 * width * width)
        )
    return np.ascontiguousarray(image, dtype=np.float32)


def extraction_signature(image: NDArray[np.float32], *, sep_module: Any | None = None) -> tuple[int, str]:
    """``(object count, SHA-256 of the objects and the segmentation map)``."""

    sep = sep_module if sep_module is not None else _sep()
    objects, segmentation = sep.extract(
        image,
        3.0,
        err=3.0,
        minarea=5,
        deblend_nthresh=32,
        deblend_cont=0.005,
        clean=True,
        segmentation_map=True,
    )
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(objects).tobytes())
    digest.update(np.ascontiguousarray(segmentation).tobytes())
    return int(len(objects)), digest.hexdigest()


def heap_churn(rng: np.random.Generator, rounds: int = 6) -> None:
    """Allocate and free buffers of varied sizes between extractions.

    A fresh heap tends to hand the same block back; churn makes the next
    ``QMALLOC`` land on previously used memory, which is what exposes the
    uninitialised reads on Windows.
    """

    kept = []
    for _ in range(rounds):
        size = int(rng.integers(64 * 1024, 4 * 1024 * 1024))
        kept.append(np.full(size, 0x5A, dtype=np.uint8))
        if len(kept) > 3:
            del kept[0]
    del kept


def _sep() -> Any:
    import sep

    return sep


def extraction_self_test(*, sep_module: Any | None = None) -> dict[str, Any]:
    """Two deblended extractions of the same field must agree exactly.

    Returns ``version``, ``deterministic`` and the evidence behind it
    (``sepVersion``, the two object counts and digests).  Exceptions from
    SEP propagate: a broken extractor is not "non-deterministic".
    """

    sep = sep_module if sep_module is not None else _sep()
    field = crowded_field()
    rng = np.random.default_rng(SELF_TEST_SEED + 1)
    first = extraction_signature(field, sep_module=sep)
    heap_churn(rng)
    second = extraction_signature(field, sep_module=sep)
    deterministic = first == second
    return {
        "version": SELF_TEST_VERSION,
        "deterministic": deterministic,
        "sepVersion": str(getattr(sep, "__version__", "unknown")),
        "objectCounts": [first[0], second[0]],
        "signatures": [first[1], second[1]],
        "warnings": [] if deterministic else [NONDETERMINISTIC_WARNING],
    }


@cache
def cached_extraction_self_test() -> dict[str, Any]:
    """The self-test once per process: the verdict is a property of the
    installed SEP build, and every panel of a project run shares it."""

    return dict(extraction_self_test())


__all__ = [
    "NONDETERMINISTIC_WARNING",
    "SELF_TEST_VERSION",
    "cached_extraction_self_test",
    "crowded_field",
    "extraction_self_test",
    "extraction_signature",
    "heap_churn",
]
