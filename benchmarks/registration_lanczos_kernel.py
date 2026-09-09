"""Small CPU-only before/after Lanczos benchmark; no pipeline or app launch.

Pass a saved pre-change calibration.py or an earlier benchmark JSON report
with --baseline-source. Reports retain the original function for replay.
Both functions read the same FITS data and require identical Float32 results.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import tempfile
import time

from astropy.io import fits
import numpy as np

from openastroflow_engine import calibration


def load_baseline(path: Path):
    source = path.read_text()
    if path.suffix == ".json":
        source = json.loads(source)["baselineFunctionSource"]
    tree = ast.parse(source)
    frame_class = next(
        (node for node in tree.body
         if isinstance(node, ast.ClassDef) and node.name == "FitsFrame"), None,
    )
    method = next(
        node for node in (frame_class.body if frame_class is not None else tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == "sample_lanczos3_clamped"
    )
    module = ast.Module(body=[method], type_ignores=[])
    namespace = dict(vars(calibration))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method.name], ast.unparse(method)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--tile-rows", type=int, default=128)
    args = parser.parse_args()
    before, before_source = load_baseline(args.baseline_source)
    after = calibration.FitsFrame.sample_lanczos3_clamped
    rng = np.random.default_rng(8531)
    pixels = rng.normal(0.15, 0.08, (args.height, args.width)).astype(np.float32)
    pixels[11::67, 19::71] = np.nan
    pixels[17::83, 23::79] = 1.4
    y, x = np.mgrid[:args.tile_rows, :args.width].astype(np.float64)
    y += (args.height - args.tile_rows) // 2
    angle = np.deg2rad(0.44)
    coordinates = {
        "fractional-translation": (x + 0.37, y + 0.49),
        "179.56-degree-affine": (
            -np.cos(angle) * x + np.sin(angle) * y + args.width - 1.37,
            -np.sin(angle) * x - np.cos(angle) * y + args.height - 0.49,
        ),
    }
    report = {
        "boundary": "single-thread warm FITS input sampling; no output I/O or pipeline",
        "sourceShape": list(pixels.shape), "tileShape": list(x.shape),
        "repeats": args.repeat, "numpyVersion": np.__version__,
        "baselineSource": str(args.baseline_source.resolve()),
        "baselineFunctionSha256": hashlib.sha256(before_source.encode()).hexdigest(),
        "baselineFunctionSource": before_source,
        "candidateFunctionSha256": hashlib.sha256(inspect.getsource(after).encode()).hexdigest(),
        "cases": {},
    }
    with tempfile.TemporaryDirectory(prefix="oaf-lanczos-kernel-") as directory:
        source = Path(directory) / "synthetic.fits"
        fits.writeto(source, pixels, fits.Header({"OAFNDOM": "NORMALIZED_TEST", "OAFNSCL": 1.0}))
        with calibration.FitsFrame(source) as frame:
            for name, (sample_x, sample_y) in coordinates.items():
                expected = before(frame, sample_x, sample_y)
                actual = after(frame, sample_x, sample_y)
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                timings = {"before": [], "after": []}
                for iteration in range(args.repeat):
                    order = (("before", before), ("after", after))
                    if iteration % 2:
                        order = order[::-1]
                    for label, function in order:
                        start = time.perf_counter()
                        function(frame, sample_x, sample_y)
                        timings[label].append(time.perf_counter() - start)
                old = statistics.median(timings["before"])
                new = statistics.median(timings["after"])
                report["cases"][name] = {
                    "bitwiseEqual": True,
                    "beforeSeconds": timings["before"],
                    "afterSeconds": timings["after"],
                    "beforeMedianSeconds": old, "afterMedianSeconds": new,
                    "speedup": old / new,
                }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
