"""Priority-dependent PSF weight factors for admitted frames.

Registration already supplies a PSF-coherence quality weight and integration
multiplies it by an inverse-variance noise weight measured on the normalized
frames.  The unattended policy adds a third factor, ``(FWHM / median FWHM of
the group) ** (-2 p)``, where ``p`` follows the user's priority (0 for depth,
1 for balanced, 2 for resolution): a frame whose stars are 1.3x wider keeps
59% of its weight under ``balanced`` and 35% under ``resolution``.  Frames
without a FWHM measurement keep a factor of 1.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .features import FrameSelectionFeatures
from .parameters import SelectionParameters

PSF_FACTOR_FLOOR = 1.0 / 16.0
PSF_FACTOR_CEILING = 16.0


def psf_factors(
    features: Sequence[FrameSelectionFeatures], parameters: SelectionParameters
) -> dict[str, float]:
    exponent = parameters.profile.psf_exponent
    factors: dict[str, float] = {item.path: 1.0 for item in features}
    if exponent == 0.0:
        return factors
    for group_id in {item.group_id for item in features}:
        members = [item for item in features if item.group_id == group_id]
        measured = np.asarray(
            [item.fwhm_native for item in members if item.fwhm_native is not None and item.fwhm_native > 0],
            dtype=np.float64,
        )
        if measured.size < 2:
            continue
        reference = float(np.median(measured))
        if not reference > 0:
            continue
        for item in members:
            if item.fwhm_native is None or item.fwhm_native <= 0:
                continue
            factor = (item.fwhm_native / reference) ** (-2.0 * exponent)
            factors[item.path] = float(min(PSF_FACTOR_CEILING, max(PSF_FACTOR_FLOOR, factor)))
    return factors
