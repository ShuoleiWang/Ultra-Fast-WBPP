"""Isolated exact-before/after crop benchmark; no pipeline, disk image, or GUI run."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time

import numpy as np

from ufwbpp.stacking import pipeline
from ufwbpp.stacking import crop, parameters


def baseline_histogram(heights, row_index):
    # The pre-optimization implementation, retained for reproducible comparison.
    best = None
    stack = []
    width = int(heights.size)
    for x in range(width + 1):
        height = int(heights[x]) if x < width else 0
        start = x
        while stack and stack[-1][1] > height:
            left, previous_height = stack.pop()
            candidate = (
                previous_height * (x - left), row_index - previous_height + 1,
                left, row_index + 1, x,
            )
            if best is None or candidate > best:
                best = candidate
            start = left
        if height and (not stack or stack[-1][1] < height):
            stack.append((start, height))
    return best


def scan_mask(mask, histogram):
    heights = np.zeros(mask.shape[1], dtype=np.int64)
    best = None
    for y, row in enumerate(mask):
        heights = np.where(row, heights + 1, 0)
        candidate = histogram(heights, y)
        if candidate is not None and (best is None or candidate > best):
            best = candidate
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=4176)
    parser.add_argument("--width", type=int, default=6252)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    height, width = args.height, args.width
    optimized = crop.histogram_rectangle
    angle = np.deg2rad(0.44)
    transforms = [parameters.AffineTransform.identity()]
    for theta, dx, dy in ((angle, 11.3, -7.4), (-angle, -9.2, 6.7)):
        a, b = np.cos(theta), np.sin(theta)
        cx, cy = (width - 1) / 2, (height - 1) / 2
        transforms.append(parameters.AffineTransform.from_value(
            ((a, -b, cx - a * cx + b * cy + dx),
             (b, a, cy - b * cx - a * cy + dy), (0, 0, 1))
        ))
    transforms.append(parameters.AffineTransform.from_value(
        ((-1, 0, width - 1), (0, -1, height - 1), (0, 0, 1))
    ))

    def common_crop(histogram):
        crop.histogram_rectangle = histogram
        try:
            return crop._common_valid_crop(
                (height, width), transforms, max_memory_bytes=128 << 20,
                resampler="lanczos-3-clamped",
            )
        finally:
            crop.histogram_rectangle = optimized

    finite = np.ones((height, width), dtype=bool)
    finite[:7] = False
    finite[-9:] = False
    finite[:, :5] = False
    finite[:, -8:] = False
    finite[height // 2, width // 3] = False
    rng = np.random.default_rng(780)
    holes = rng.random((128, width)) > 0.5
    cases = {
        "common-affine-footprint-including-mask-construction": common_crop,
        "finite-mask-with-internal-hole-histogram-only": lambda fn: scan_mask(finite, fn),
        "dense-random-holes-128-rows-histogram-only": lambda fn: scan_mask(holes, fn),
    }
    report = {
        "boundary": "single process, synthetic masks; no image I/O or end-to-end pipeline",
        "shape": [height, width], "repeats": args.repeat,
        "numpyVersion": np.__version__,
        "baselineFunctionSha256": hashlib.sha256(inspect.getsource(baseline_histogram).encode()).hexdigest(),
        "candidateFunctionSha256": hashlib.sha256(inspect.getsource(optimized).encode()).hexdigest(),
        "cases": {},
    }
    for name, run in cases.items():
        timings = {"before": [], "after": []}
        expected = None
        for iteration in range(args.repeat):
            order = (("before", baseline_histogram), ("after", optimized))
            if iteration % 2:
                order = order[::-1]
            for label, histogram in order:
                start = time.perf_counter()
                result = run(histogram)
                timings[label].append(time.perf_counter() - start)
                if expected is None:
                    expected = result
                assert result == expected, (name, label, result, expected)
        report["cases"][name] = {
            "seconds": timings, "exactRectangleMatch": True, "rectangle": expected,
            "speedup": statistics.median(timings["before"]) / statistics.median(timings["after"]),
        }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
