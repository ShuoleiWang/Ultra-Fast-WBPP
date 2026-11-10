"""Unattended Light selection: decisions, weights and counterfactual evidence.

The package turns the Light Frame QC evidence into per-frame decisions
(exclude, keep, keep with reduced weight) without a human review gate, and
measures every kept frame's marginal contribution to the master with a
leave-one-out counterfactual computed inside the integration tile loop.
"""

from .counterfactual import (
    BlockCounterfactual,
    CounterfactualReport,
    FrameCounterfactual,
    LeaveOneOutAccumulator,
    TileObservation,
)
from .features import FrameSelectionFeatures, extract_features
from .parameters import (
    PRIORITY_PROFILES,
    SELECTION_ALGORITHM,
    PriorityProfile,
    SelectionParameters,
)
from .policy import (
    SelectionDecision,
    annotate_with_counterfactual,
    confirmed_harmful,
    decide,
    exclude_confirmed,
)
from .region import RegionWeightMap, region_weight_map, region_weight_maps
from .unattended_v2 import (
    BlinkContext,
    SuspectBlock,
    V2Thresholds,
    V2_ALGORITHM,
    V2_THRESHOLDS,
    blink_context,
    confirmed_harmful_blocks,
    decide_v2,
)
from .weights import psf_factors

__all__ = [
    "BlinkContext",
    "BlockCounterfactual",
    "CounterfactualReport",
    "FrameCounterfactual",
    "FrameSelectionFeatures",
    "LeaveOneOutAccumulator",
    "PRIORITY_PROFILES",
    "PriorityProfile",
    "RegionWeightMap",
    "SELECTION_ALGORITHM",
    "SelectionDecision",
    "SelectionParameters",
    "SuspectBlock",
    "TileObservation",
    "V2Thresholds",
    "V2_ALGORITHM",
    "V2_THRESHOLDS",
    "annotate_with_counterfactual",
    "blink_context",
    "confirmed_harmful",
    "confirmed_harmful_blocks",
    "exclude_confirmed",
    "decide",
    "decide_v2",
    "extract_features",
    "psf_factors",
    "region_weight_map",
    "region_weight_maps",
]
