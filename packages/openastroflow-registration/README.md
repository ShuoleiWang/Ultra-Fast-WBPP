# Ultra-Fast WBPP registration library

This package supplies production source detection, transform estimation and quality weights to the engine, plus portable calibration/warping and common-footprint reference routines. It has no PixInsight/PCL runtime dependency.

The module is intentionally separate from the GUI and durable product state. The Ultra-Fast WBPP engine worker converts typed project/recipe plans into calls here, writes only to a supplied staging capability, and returns versioned artifact receipts.

```bash
python -m pip install -e './packages/light-frame-qc[test]'
python -m pip install -e './packages/openastroflow-registration[test]'
PYTHONPATH=packages/openastroflow-registration/src pytest -q packages/openastroflow-registration/tests
```

The SciPy warp is the portable correctness backend. Apple Metal and future Windows GPU workers must compare against this path and the native CPU math gates before advertising an accelerated capability.
