"""Selection policy parameters and priority profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

SELECTION_ALGORITHM = "unattended-selection-v1"

POLICIES = ("legacy-gate", "unattended-v1", "include-all")
PRIORITIES = ("depth", "balanced", "resolution")
AGGRESSIVENESS = ("conservative", "standard", "aggressive")
COUNTERFACTUAL_MODES = ("off", "analytic")


@dataclass(frozen=True, slots=True)
class PriorityProfile:
    """How the user's priority shapes weights and the PSF exclusion cut-offs."""

    priority: str
    # Exponent p of the PSF weight term FWHM^(-2p).
    psf_exponent: float
    # A frame whose FWHM exceeds the best of its night by this ratio is excluded.
    fwhm_night_exclusion_ratio: float
    # A frame whose FWHM exceeds the best of its group by this ratio is excluded.
    fwhm_group_exclusion_ratio: float


PRIORITY_PROFILES: dict[str, PriorityProfile] = {
    "depth": PriorityProfile("depth", 0.0, 2.0, 2.4),
    "balanced": PriorityProfile("balanced", 1.0, 1.6, 1.9),
    "resolution": PriorityProfile("resolution", 2.0, 1.3, 1.5),
}

# Multiplies every gray-zone exclusion threshold: conservative excludes less.
AGGRESSIVENESS_SCALE: dict[str, float] = {
    "conservative": 1.15,
    "standard": 1.0,
    "aggressive": 0.85,
}


@dataclass(frozen=True, slots=True)
class SelectionParameters:
    """Parameters of the unattended selection policy.

    ``legacy-gate`` reproduces the historical behaviour exactly (PASS admitted,
    REVIEW excluded unless approved).  ``unattended-v1`` applies the guards,
    gray-zone rules and confidence weights of the selection plan.
    ``include-all`` is a diagnostic policy that admits every frame the guards
    allow at full weight so the counterfactual can measure each one.
    """

    policy: str = "legacy-gate"
    priority: str = "balanced"
    aggressiveness: str = "standard"
    counterfactual: str = "analytic"
    # Per-frame region weight maps for partly blocked or partly clouded
    # frames (unattended policies only; legacy-gate never builds them).
    region_weights: bool = True
    # Global normalization cannot recover a frame below this stellar scale.
    transparency_floor: float = 0.50
    extra_extinction_exclusion_mag: float = 0.75
    insufficient_evidence_confidence: float = 0.50
    defect_review_confidence: float = 0.60
    # Soft (rule-based) exclusions above this fraction of a group are
    # downgraded to reduced-weight inclusion until the counterfactual confirms.
    soft_exclusion_fraction_guard: float = 0.30
    # Counterfactual confirmation thresholds (one-sided CI bounds).  The
    # background threshold is in units of the master's 8x8 block noise; a
    # frame whose weight varies across the field leaves steps of a few
    # hundredths of that noise, so only a quarter of it counts as structure
    # (about the evaluator's 0.05 pixel-noise tolerance).
    harmful_depth_mag: float = 0.005
    harmful_background_sigma: float = 0.25
    beneficial_depth_mag: float = 0.010
    harmful_fwhm_px: float = 0.10
    # A confirmation needs enough statistics tiles for the tile bootstrap and
    # the frame must stand out from its group (median + 3 MADN of the deltas).
    minimum_counterfactual_tiles: int = 16
    outlier_madn: float = 3.0
    # Whether a counterfactual-confirmed harmful frame is removed and the group
    # integrated once more without it ("exclude"), or only reported ("report").
    counterfactual_action: str = "exclude"
    # Integration passes: the first measures, each further pass removes the
    # frames confirmed harmful by the previous one (a removed frame can unmask
    # another), bounded by this count and by the soft-exclusion fraction guard.
    max_integration_passes: int = 3
    extra: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"selection.policy must be one of {POLICIES}")
        if self.priority not in PRIORITIES:
            raise ValueError(f"selection.priority must be one of {PRIORITIES}")
        if self.aggressiveness not in AGGRESSIVENESS:
            raise ValueError(f"selection.aggressiveness must be one of {AGGRESSIVENESS}")
        if self.counterfactual not in COUNTERFACTUAL_MODES:
            raise ValueError(f"selection.counterfactual must be one of {COUNTERFACTUAL_MODES}")
        for name in (
            "transparency_floor",
            "extra_extinction_exclusion_mag",
            "insufficient_evidence_confidence",
            "defect_review_confidence",
            "soft_exclusion_fraction_guard",
            "harmful_depth_mag",
            "harmful_background_sigma",
            "beneficial_depth_mag",
            "harmful_fwhm_px",
            "outlier_madn",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not value >= 0:
                raise ValueError(f"selection.{name} must be a non-negative number")
        if not 0 < self.transparency_floor <= 1:
            raise ValueError("selection.transparency_floor must be in (0, 1]")
        for name in ("insufficient_evidence_confidence", "defect_review_confidence"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"selection.{name} must be in (0, 1]")
        if not 0 <= self.soft_exclusion_fraction_guard <= 1:
            raise ValueError("selection.soft_exclusion_fraction_guard must be in [0, 1]")
        if self.counterfactual_action not in ("exclude", "report"):
            raise ValueError("selection.counterfactual_action must be 'exclude' or 'report'")
        if not isinstance(self.max_integration_passes, int) or self.max_integration_passes < 1:
            raise ValueError("selection.max_integration_passes must be a positive integer")

    @property
    def profile(self) -> PriorityProfile:
        return PRIORITY_PROFILES[self.priority]

    @property
    def threshold_scale(self) -> float:
        return AGGRESSIVENESS_SCALE[self.aggressiveness]

    @property
    def unattended(self) -> bool:
        return self.policy != "legacy-gate"

    def serializable(self) -> dict[str, Any]:
        """The recipe form: exactly the keys ``from_mapping`` accepts."""

        return {
            "policy": self.policy,
            "priority": self.priority,
            "aggressiveness": self.aggressiveness,
            "counterfactual": self.counterfactual,
            "regionWeights": self.region_weights,
            "transparencyFloor": self.transparency_floor,
            "extraExtinctionExclusionMag": self.extra_extinction_exclusion_mag,
            "insufficientEvidenceConfidence": self.insufficient_evidence_confidence,
            "defectReviewConfidence": self.defect_review_confidence,
            "softExclusionFractionGuard": self.soft_exclusion_fraction_guard,
            "counterfactualAction": self.counterfactual_action,
            "maxIntegrationPasses": self.max_integration_passes,
        }

    def describe(self) -> dict[str, Any]:
        """The receipt form: the recipe keys plus the derived thresholds."""

        return {
            "algorithm": SELECTION_ALGORITHM,
            "policy": self.policy,
            "priority": self.priority,
            "aggressiveness": self.aggressiveness,
            "counterfactual": self.counterfactual,
            "regionWeights": self.region_weights,
            "transparencyFloor": self.transparency_floor,
            "extraExtinctionExclusionMag": self.extra_extinction_exclusion_mag,
            "insufficientEvidenceConfidence": self.insufficient_evidence_confidence,
            "defectReviewConfidence": self.defect_review_confidence,
            "softExclusionFractionGuard": self.soft_exclusion_fraction_guard,
            "harmfulDepthMag": self.harmful_depth_mag,
            "harmfulBackgroundSigma": self.harmful_background_sigma,
            "beneficialDepthMag": self.beneficial_depth_mag,
            "harmfulFwhmPx": self.harmful_fwhm_px,
            "minimumCounterfactualTiles": self.minimum_counterfactual_tiles,
            "outlierMadn": self.outlier_madn,
            "counterfactualAction": self.counterfactual_action,
            "maxIntegrationPasses": self.max_integration_passes,
            "psfExponent": self.profile.psf_exponent,
            "fwhmNightExclusionRatio": self.profile.fwhm_night_exclusion_ratio * self.threshold_scale,
            "fwhmGroupExclusionRatio": self.profile.fwhm_group_exclusion_ratio * self.threshold_scale,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> SelectionParameters:
        """Build parameters from a recipe ``selection`` block (camelCase keys)."""

        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("selection must be an object")
        known = {
            "policy": "policy",
            "priority": "priority",
            "aggressiveness": "aggressiveness",
            "counterfactual": "counterfactual",
            "regionWeights": "region_weights",
            "transparencyFloor": "transparency_floor",
            "extraExtinctionExclusionMag": "extra_extinction_exclusion_mag",
            "insufficientEvidenceConfidence": "insufficient_evidence_confidence",
            "defectReviewConfidence": "defect_review_confidence",
            "softExclusionFractionGuard": "soft_exclusion_fraction_guard",
            "counterfactualAction": "counterfactual_action",
            "maxIntegrationPasses": "max_integration_passes",
        }
        unknown = sorted(set(raw) - set(known))
        if unknown:
            raise ValueError(f"selection has unknown keys: {', '.join(unknown)}")
        values: dict[str, Any] = {}
        for key, attribute in known.items():
            if key in raw:
                values[attribute] = raw[key]
        if "region_weights" in values and not isinstance(values["region_weights"], bool):
            raise ValueError("selection.regionWeights must be boolean")
        parameters = cls(**values)
        parameters.validate()
        return parameters
