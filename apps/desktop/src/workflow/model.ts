import type {
  BlinkChannel,
  BlinkFrame,
  BlinkMeasureResponse,
  FrameRole,
  InspectedAsset,
  MasterFrameRole,
  MasterMetadataOverride,
  MasterMetadataOverrideRequest,
  OutputArtifact,
  PanelCell,
  SolverBackendStatus,
  StageProgress,
} from "../types";

export const DEMO_COUNTS: Record<FrameRole, number> = {
  LIGHT: 370,
  FLAT: 36,
  DARK: 24,
  BIAS: 64,
  MASTER_FLAT: 0,
  MASTER_DARK: 0,
  MASTER_BIAS: 0,
};
export const MASTER_ROLES = new Set<FrameRole>(["MASTER_FLAT", "MASTER_DARK", "MASTER_BIAS"]);
export const STAGE_DEFINITIONS = [
  { stageId: "prepare", name: "Preparation" },
  { stageId: "quality-control", name: "Quality Gate" },
  { stageId: "calibrate", name: "Calibration" },
  { stageId: "register", name: "Registration" },
  { stageId: "integrate", name: "Integration and rejection" },
  { stageId: "drizzle", name: "Drizzle" },
  { stageId: "solve", name: "Astrometric solve" },
  { stageId: "mosaic", name: "Mosaic verification" },
  { stageId: "alignment", name: "Channel alignment" },
  { stageId: "color", name: "Color and previews" },
  { stageId: "preview", name: "Panel previews" },
  { stageId: "verify", name: "Verification" },
  { stageId: "publish", name: "Publication" },
];
export const initialStages = (scope?: "panel" | "project"): StageProgress[] =>
  STAGE_DEFINITIONS.filter((stage) =>
    scope === "project"
      ? ["prepare", "mosaic", "alignment", "color", "verify", "publish"].includes(stage.stageId)
      : scope === "panel"
        ? !["mosaic", "alignment", "color"].includes(stage.stageId)
        : true,
  ).map((stage) => ({ ...stage, status: "WAITING", percent: 0 }));
export const DEMO_ARTIFACTS: OutputArtifact[] = [
  {
    kind: "PREVIEW",
    name: "DEMO_result.png",
    path: "/explicit-browser-demo/result.png",
    detail: "DEMO ONLY · no file was created",
  },
];
export const deduplicate = (values: string[]) => [...new Set(values)];
export const known = (value: string | undefined | null) =>
  Boolean(value?.trim() && !["UNKNOWN", "UNSPECIFIED"].includes(value.trim().toUpperCase()));
export const numberKnown = (value: number | undefined | null) => typeof value === "number" && Number.isFinite(value);
export const monoCfa = (value: string) => ["NONE", "MONO", "MONOCHROME"].includes(value.trim().toUpperCase());
export const unknownCfa = (value: string | undefined | null) =>
  !value?.trim() || ["UNKNOWN", "UNSPECIFIED"].includes(value.trim().toUpperCase());
export const BAYER_PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"];
export const bayerCfa = (value: string) => BAYER_PATTERNS.includes(value.trim().toUpperCase());
export const safeSourceId = (role: FrameRole, index: number) =>
  `${role.toLowerCase().replaceAll("_", "-")}-${String(index + 1).padStart(4, "0")}`;
export const OUTPUT_PARENT_STORAGE_KEY = "ultra-fast-wbpp.outputParent";
/** The output folder chosen last time, so a returning user only drops files and starts. */
export function storedOutputParent(): string | undefined {
  try {
    return window.localStorage.getItem(OUTPUT_PARENT_STORAGE_KEY) ?? undefined;
  } catch {
    return undefined;
  }
}
export function rememberOutputParent(value: string | undefined) {
  try {
    if (value) window.localStorage.setItem(OUTPUT_PARENT_STORAGE_KEY, value);
    else window.localStorage.removeItem(OUTPUT_PARENT_STORAGE_KEY);
  } catch {
    /* storage is a convenience only */
  }
}

export function buildMasterOverride(master: InspectedAsset, current?: MasterMetadataOverride): MasterMetadataOverride {
  if (current) return current;
  const fromString = (field: "camera" | "filter" | "cfaPattern" | "readoutMode") =>
    known(master[field]) ? master[field] : "";
  const fromNumber = (field: "gain" | "offset" | "temperatureCelsius" | "exposureSeconds") =>
    numberKnown(master[field]) ? Number(master[field]) : null;
  return {
    sourceSha256: master.sourceSha256 ?? "",
    sourcePath: master.path,
    role: master.role as MasterFrameRole,
    camera: fromString("camera"),
    gain: fromNumber("gain"),
    offset: fromNumber("offset"),
    binning: [master.binning?.[0] > 0 ? master.binning[0] : null, master.binning?.[1] > 0 ? master.binning[1] : null],
    filter: fromString("filter"),
    cfaPattern: fromString("cfaPattern"),
    readoutMode: fromString("readoutMode"),
    temperatureCelsius: fromNumber("temperatureCelsius"),
    exposureSeconds: fromNumber("exposureSeconds"),
    biasIncluded: null,
    numericDomain: null,
    normalizedUnitScale: null,
    needsMetadataOverride: false,
    confirmed: true,
  };
}

// Only explicit advanced edits are sent. Missing acquisition metadata stays missing.
export function masterOverrideRequests(items: MasterMetadataOverride[]): MasterMetadataOverrideRequest[] {
  return items
    .filter((item) => item.confirmed && item.needsMetadataOverride)
    .map((item) => {
      const result: MasterMetadataOverrideRequest = { sourceSha256: item.sourceSha256 };
      for (const field of ["camera", "filter", "cfaPattern", "readoutMode"] as const)
        if (known(item[field])) result[field] = item[field];
      for (const field of ["gain", "offset", "temperatureCelsius", "exposureSeconds"] as const)
        if (numberKnown(item[field])) result[field] = item[field]!;
      if (item.binning.every((value) => numberKnown(value) && value! > 0))
        result.binning = item.binning as [number, number];
      if (item.biasIncluded !== null) result.biasIncluded = item.biasIncluded;
      if (item.numericDomain !== null) {
        result.numericDomain = item.numericDomain;
        if (item.normalizedUnitScale !== null) result.normalizedUnitScale = item.normalizedUnitScale;
      }
      return result;
    });
}

/** Words a mosaic's panel names end with that say nothing about the object. */
export const PANEL_WORDS = /^(panel|tile|part|p|frame|field|mosaic)$/i;

/**
 * Output-folder label: the target itself, the leading words a mosaic's panels
 * share (`NGC 7000 Panel 1` + `NGC 7000 Panel 2` → `NGC 7000`), the targets
 * joined when they share none, or the project name without any target.
 */
export function runLabel(cells: Array<{ target: string }>, fallback: string): string {
  const targets = [...new Set(cells.map((cell) => cell.target.trim()).filter(Boolean))];
  if (!targets.length) return fallback;
  if (targets.length === 1) return targets[0];
  const words = targets.map((target) => target.split(/[\s_-]+/).filter(Boolean));
  const shared: string[] = [];
  for (let index = 0; index < Math.min(...words.map((list) => list.length)); index += 1) {
    const word = words[0][index];
    if (!words.every((list) => list[index].localeCompare(word, undefined, { sensitivity: "accent" }) === 0)) break;
    shared.push(word);
  }
  if (shared.length && PANEL_WORDS.test(shared[shared.length - 1])) shared.pop();
  if (shared.length) return shared.join(" ");
  return targets.length <= 3 ? targets.join(" + ") : `${targets.slice(0, 2).join(" + ")} + ${targets.length - 2} more`;
}

export function panelMatrix(
  assets: InspectedAsset[],
  admittedPaths: ReadonlySet<string> = new Set(),
): Array<PanelCell & { admittedCount: number }> {
  const groups = new Map<string, PanelCell & { admittedCount: number }>();
  for (const asset of assets.filter((item) => item.role === "LIGHT")) {
    const target = known(asset.target) ? asset.target : "UNKNOWN TARGET";
    const filter = known(asset.filter) ? asset.filter : "UNKNOWN FILTER";
    const key = `${target}\u0000${filter}`;
    const cell = groups.get(key) ?? {
      panelId: `panel-${groups.size + 1}`,
      target,
      filter,
      lightCount: 0,
      admittedCount: 0,
    };
    cell.lightCount += 1;
    if (admittedPaths.has(asset.path)) cell.admittedCount += 1;
    groups.set(key, cell);
  }
  const filterOrder = ["L", "R", "G", "B", "HA", "OIII", "SII"];
  const filterRank = (value: string) => {
    const index = filterOrder.indexOf(value.trim().toUpperCase());
    return index < 0 ? filterOrder.length : index;
  };
  return [...groups.values()].sort(
    (a, b) =>
      a.target.localeCompare(b.target) ||
      filterRank(a.filter) - filterRank(b.filter) ||
      a.filter.localeCompare(b.filter),
  );
}

export type SolverBackendId = "astrometry-net" | "astap";

/**
 * Whether the strict final gate would accept this backend's solutions: it must
 * run, and it must produce the managed-catalog correspondence evidence.  An
 * engine that predates the `scienceReady` field only ever produced that
 * evidence with solve-field, so its absence counts as ready for solve-field
 * and as not ready for ASTAP.
 */
export function solverScienceReady(backend: SolverBackendStatus | undefined): boolean {
  if (!backend?.executionReady) return false;
  if (backend.scienceReady === undefined) return backend.backendId === "astrometry-net";
  return backend.scienceReady === true;
}

export const CONTENT_DIGEST = /^sha256:[0-9a-f]{64}$/;
/** Undo depth of the blink decisions. */
export const DECISION_HISTORY_LIMIT = 100;
/** The minimum kept Lights per blink channel (the engine's `QC_INSUFFICIENT_LIGHTS` bound). */
export const MINIMUM_KEPT_PER_CHANNEL = 2;

/** A measured blink session: the manifest, where its previews live and the Lights it was measured for. */
export interface BlinkSession {
  manifest: BlinkMeasureResponse;
  sessionDirectory: string;
  inventoryKey: string;
  demo: boolean;
}
/** One blink channel with the decision counts the launch bar and chips show. */
export interface BlinkChannelSummary extends BlinkChannel {
  frames: BlinkFrame[];
  total: number;
  viewed: number;
  confirmed: boolean;
  kept: number;
  flagged: number;
  exclude: number;
  attention: number;
}

export const inventoryKeyOf = (paths: string[]) => [...paths].sort().join("\n");

/**
 * Whether a session still describes the current Lights: every imported Light
 * is in the manifest with a usable, unique content digest (the selection is
 * keyed by digest, so a duplicate or malformed one could not be sent).
 */
export function blinkSessionCovers(session: BlinkSession, lightPaths: string[]): boolean {
  if (
    inventoryKeyOf(lightPaths) !== session.inventoryKey ||
    !lightPaths.length ||
    !CONTENT_DIGEST.test(session.manifest.manifestSha256 ?? "")
  )
    return false;
  const frames = session.manifest.frames;
  const byPath = new Set(frames.map((frame) => frame.path));
  const digests = frames.map((frame) => frame.sourceSha256);
  return (
    lightPaths.every((path) => byPath.has(path)) &&
    digests.every((digest) => CONTENT_DIGEST.test(digest)) &&
    new Set(digests).size === digests.length
  );
}

export type AdmissionCell = PanelCell & { admittedCount: number };

/** One reason the run cannot start yet, in the order the launch bar lists them. */
export type StartBlocker =
  | { kind: "engine"; reason?: string }
  | { kind: "calibration" }
  | { kind: "types" }
  | { kind: "lights" }
  | { kind: "panel"; cell: AdmissionCell }
  | { kind: "solver" }
  | { kind: "output" }
  | { kind: "master" }
  | { kind: "cfa" };

export interface StartReadiness {
  engineAvailable: boolean;
  engineUnavailableReason?: string;
  calibrationReady: boolean;
  allRequiredConfirmed: boolean;
  panelCount: number;
  insufficientPanels: AdmissionCell[];
  solverSetupReady: boolean;
  outputChosen: boolean;
  masterOverridesReady: boolean;
  cfaBlocked: boolean;
}

/**
 * Every condition a native run needs besides the Blink review itself.  The
 * start button is enabled exactly when this list is empty (and no run or
 * inspection is in flight), and the launch bar shows the same list.
 */
export function startBlockers(readiness: StartReadiness): StartBlocker[] {
  const blockers: StartBlocker[] = [];
  if (!readiness.engineAvailable) blockers.push({ kind: "engine", reason: readiness.engineUnavailableReason });
  if (!readiness.calibrationReady) blockers.push({ kind: "calibration" });
  if (!readiness.allRequiredConfirmed) blockers.push({ kind: "types" });
  if (readiness.panelCount === 0) blockers.push({ kind: "lights" });
  for (const cell of readiness.insufficientPanels) blockers.push({ kind: "panel", cell });
  if (!readiness.solverSetupReady) blockers.push({ kind: "solver" });
  if (!readiness.outputChosen) blockers.push({ kind: "output" });
  if (!readiness.masterOverridesReady) blockers.push({ kind: "master" });
  if (readiness.cfaBlocked) blockers.push({ kind: "cfa" });
  return blockers;
}
