# Ultra-Fast WBPP registration worker

This package is the portable Python reference path for bounded-preview source detection, robust transform estimation, full-resolution calibration/warping, common-footprint crop, and quality weights. It has no PixInsight/PCL runtime dependency.

The module is intentionally separate from the GUI and durable product state. The Ultra-Fast WBPP engine worker converts typed project/recipe plans into calls here, writes only to a supplied staging capability, and returns versioned artifact receipts.

```bash
python -m pip install -e './packages/light-frame-qc[test]'
python -m pip install -e './engine/native/python[test]'
PYTHONPATH=engine/native/python pytest -q engine/native/python/tests
```

The SciPy warp is the portable correctness backend. Apple Metal and future Windows GPU workers must compare against this path and the native CPU math gates before advertising an accelerated capability.
