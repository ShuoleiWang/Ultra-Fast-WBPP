"""Ultra-Fast WBPP engine: raw Lights to verified, plate-solved linear masters.

The command line (:mod:`.cli`) is the only process boundary; the desktop and
headless users call the same subcommands.  The main modules:

- :mod:`.workflows` coordinates a run: ``project`` (targets, filters, shared
  calibration, colour products), ``single_target`` (one field), ``solve``
  (plate solving and WCS verification) and ``contracts`` (requests, explicit
  selections, progress).
- :mod:`.pixel_pipeline` calibrates, registers, normalizes and integrates one
  run's frames; :mod:`.calibration`, :mod:`.global_normalization`,
  :mod:`.transient_rejection` and :mod:`.drizzle_native` hold the numerics.
- :mod:`.selection` and :mod:`.blink_session` decide which Lights are admitted.
- :mod:`.native_kernels` and :mod:`.metal_integration` bind the C++/Metal
  kernels, which stay value-identical to their NumPy references.

Submodules are imported on demand, so a light command such as ``inventory``
does not pay for the pixel pipeline, the solvers or SciPy at start-up.
"""

__version__ = "0.1.0"
