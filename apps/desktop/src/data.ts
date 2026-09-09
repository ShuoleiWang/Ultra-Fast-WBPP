import type { FrameRole, Recipe, SourceSet } from "./types";

export const STEPS = [
  { id: "import", short: "01", label: "Import", caption: "Lights and calibration" },
  { id: "inspect", short: "02", label: "Review", caption: "Groups and frame quality" },
  { id: "recipe", short: "03", label: "Process", caption: "Recipe, run, result" },
] as const;

const SOURCE_META: Record<FrameRole, Pick<SourceSet, "label" | "hint" | "reuseAllowed">> = {
  LIGHT: { label: "Light", hint: "N.I.N.A. target exposures", reuseAllowed: false },
  FLAT: { label: "Flat", hint: "Match filter, gain, and optical train", reuseAllowed: true },
  DARK: { label: "Dark", hint: "Match exposure, temperature, and gain", reuseAllowed: true },
  BIAS: { label: "Bias", hint: "Match camera, gain, and readout mode", reuseAllowed: true },
  MASTER_FLAT: { label: "Master Flat", hint: "Reuse a verified integrated flat", reuseAllowed: true },
  MASTER_DARK: { label: "Master Dark", hint: "Declare whether Bias is included", reuseAllowed: true },
  MASTER_BIAS: { label: "Master Bias", hint: "Reuse with content-bound metadata", reuseAllowed: true },
};

export const emptySources = (): SourceSet[] =>
  (Object.keys(SOURCE_META) as FrameRole[]).map((role) => ({
    role,
    ...SOURCE_META[role],
    paths: [],
    fileCount: 0,
    detected: false,
    confirmed: false,
  }));

export const RECIPES: Recipe[] = [
  {
    id: "balanced",
    name: "Balanced",
    eyebrow: "RECOMMENDED",
    description: "Conservative calibration, quality weighting, and robust rejection.",
    details: ["Conservative quality gate", "Adaptive registration and rejection", "Final solve per channel"],
    estimatedScale: "Dataset dependent",
    recommended: true,
    available: true,
  },
];
