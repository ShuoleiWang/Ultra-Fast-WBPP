import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { desktopBridge, hasTauriRuntime, listenForCatalogEvents, listenForDesktopDrops, listenForPipelineEvents } from "./bridge";
import { emptySources } from "./data";
import type { Translator } from "./i18n";
import type {
  CalibrationInspection,
  CatalogCompleteEvent,
  DrizzleKernel,
  DrizzleScale,
  CatalogDoctorResponse,
  CatalogErrorEvent,
  CatalogInfo,
  CatalogListResponse,
  CatalogProgressEvent,
  FrameRole,
  GateSummary,
  InspectedAsset,
  InspectedLightQuality,
  MasterFrameRole,
  MasterMetadataOverride,
  MasterMetadataOverrideRequest,
  OutputArtifact,
  PanelCell,
  PipelineCompleteEvent,
  PipelineErrorEvent,
  PipelineProgressEvent,
  RunStatus,
  RuntimeCapabilities,
  QualityInspection,
  ScreeningSummary,
  SolverDoctorResponse,
  SourceSet,
  StageProgress,
  WorkflowStep,
} from "./types";

const DEMO_COUNTS: Record<FrameRole, number> = {
  LIGHT: 370, FLAT: 36, DARK: 24, BIAS: 64, MASTER_FLAT: 0, MASTER_DARK: 0, MASTER_BIAS: 0,
};
const MASTER_ROLES = new Set<FrameRole>(["MASTER_FLAT", "MASTER_DARK", "MASTER_BIAS"]);
const STAGE_DEFINITIONS = [
  { stageId: "prepare", name: "Preparation" },
  { stageId: "quality-control", name: "Quality Gate" },
  { stageId: "calibrate", name: "Calibration" },
  { stageId: "register", name: "Registration" },
  { stageId: "local-normalization", name: "LocalNormalization" },
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
const initialStages = (scope?: "panel" | "project"): StageProgress[] => STAGE_DEFINITIONS
  .filter((stage) => scope === "project" ? ["prepare", "mosaic", "alignment", "color", "verify", "publish"].includes(stage.stageId)
    : scope === "panel" ? !["mosaic", "alignment", "color"].includes(stage.stageId) : true)
  .map((stage) => ({ ...stage, status: "WAITING", percent: 0 }));
const DEMO_ARTIFACTS: OutputArtifact[] = [{ kind: "PREVIEW", name: "DEMO_result.png", path: "/explicit-browser-demo/result.png", detail: "DEMO ONLY · no file was created" }];
const deduplicate = (values: string[]) => [...new Set(values)];
const known = (value: string | undefined | null) => Boolean(value?.trim() && !["UNKNOWN", "UNSPECIFIED"].includes(value.trim().toUpperCase()));
const numberKnown = (value: number | undefined | null) => typeof value === "number" && Number.isFinite(value);
const monoCfa = (value: string) => ["NONE", "MONO", "MONOCHROME"].includes(value.trim().toUpperCase());
const unknownCfa = (value: string | undefined | null) => !value?.trim() || ["UNKNOWN", "UNSPECIFIED"].includes(value.trim().toUpperCase());
const BAYER_PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"];
const bayerCfa = (value: string) => BAYER_PATTERNS.includes(value.trim().toUpperCase());
const safeSourceId = (role: FrameRole, index: number) => `${role.toLowerCase().replaceAll("_", "-")}-${String(index + 1).padStart(4, "0")}`;
const OUTPUT_PARENT_STORAGE_KEY = "ultra-fast-wbpp.outputParent";
/** The output folder chosen last time, so a returning user only drops files and starts. */
function storedOutputParent(): string | undefined {
  try { return window.localStorage.getItem(OUTPUT_PARENT_STORAGE_KEY) ?? undefined; } catch { return undefined; }
}
function rememberOutputParent(value: string | undefined) {
  try { if (value) window.localStorage.setItem(OUTPUT_PARENT_STORAGE_KEY, value); else window.localStorage.removeItem(OUTPUT_PARENT_STORAGE_KEY); } catch { /* storage is a convenience only */ }
}

function buildMasterOverride(master: InspectedAsset, current?: MasterMetadataOverride): MasterMetadataOverride {
  if (current) return current;
  const fromString = (field: "camera" | "filter" | "cfaPattern" | "readoutMode") => known(master[field]) ? master[field] : "";
  const fromNumber = (field: "gain" | "offset" | "temperatureCelsius" | "exposureSeconds") => numberKnown(master[field]) ? Number(master[field]) : null;
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
function masterOverrideRequests(items: MasterMetadataOverride[]): MasterMetadataOverrideRequest[] {
  return items.filter((item) => item.confirmed && item.needsMetadataOverride).map((item) => {
    const result: MasterMetadataOverrideRequest = { sourceSha256: item.sourceSha256 };
    for (const field of ["camera", "filter", "cfaPattern", "readoutMode"] as const) if (known(item[field])) result[field] = item[field];
    for (const field of ["gain", "offset", "temperatureCelsius", "exposureSeconds"] as const) if (numberKnown(item[field])) result[field] = item[field]!;
    if (item.binning.every((value) => numberKnown(value) && value! > 0)) result.binning = item.binning as [number, number];
    if (item.biasIncluded !== null) result.biasIncluded = item.biasIncluded;
    if (item.numericDomain !== null) {
      result.numericDomain = item.numericDomain;
      if (item.normalizedUnitScale !== null) result.normalizedUnitScale = item.normalizedUnitScale;
    }
    return result;
  });
}

/** Words a mosaic's panel names end with that say nothing about the object. */
const PANEL_WORDS = /^(panel|tile|part|p|frame|field|mosaic)$/i;

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

function panelMatrix(assets: InspectedAsset[], admittedPaths: ReadonlySet<string> = new Set()): Array<PanelCell & { admittedCount: number }> {
  const groups = new Map<string, PanelCell & { admittedCount: number }>();
  for (const asset of assets.filter((item) => item.role === "LIGHT")) {
    const target = known(asset.target) ? asset.target : "UNKNOWN TARGET";
    const filter = known(asset.filter) ? asset.filter : "UNKNOWN FILTER";
    const key = `${target}\u0000${filter}`;
    const cell = groups.get(key) ?? { panelId: `panel-${groups.size + 1}`, target, filter, lightCount: 0, admittedCount: 0 };
    cell.lightCount += 1;
    if (admittedPaths.has(asset.path)) cell.admittedCount += 1;
    groups.set(key, cell);
  }
  const filterOrder = ["L", "R", "G", "B", "HA", "OIII", "SII"];
  const filterRank = (value: string) => {
    const index = filterOrder.indexOf(value.trim().toUpperCase());
    return index < 0 ? filterOrder.length : index;
  };
  return [...groups.values()].sort((a, b) => a.target.localeCompare(b.target)
    || filterRank(a.filter) - filterRank(b.filter)
    || a.filter.localeCompare(b.filter));
}

function reviewCanBeApproved(frame: InspectedLightQuality, inspection: QualityInspection | undefined, panelCount: number): boolean {
  const digest = /^sha256:[0-9a-f]{64}$/;
  return Boolean(inspection && panelCount === 1 && frame.disposition === "REVIEW"
    && frame.sourceSha256 && digest.test(frame.sourceSha256)
    && digest.test(inspection.gatePolicyDigest) && frame.previewDataUrl
    && inspection.frames.includes(frame)
    && inspection.frames.filter((item) => item.sourceSha256 === frame.sourceSha256).length === 1);
}

export function useWorkflow(t: Translator) {
  const nativeRuntime = hasTauriRuntime();
  const [step, setStep] = useState<WorkflowStep>("import");
  const [sources, setSources] = useState<SourceSet[]>(emptySources);
  const [assets, setAssets] = useState<InspectedAsset[]>([]);
  const [projectName, setProjectName] = useState("Ultra-Fast WBPP project");
  const [masterOverrides, setMasterOverrides] = useState<MasterMetadataOverride[]>([]);
  const [selectedRole, setSelectedRole] = useState<FrameRole | undefined>();
  const [isDragging, setDragging] = useState(false);
  const [gate, setGate] = useState<GateSummary>({ pass: 0, review: 0, hardFail: 0 });
  const [qualityInspection, setQualityInspection] = useState<QualityInspection>();
  const [calibrationResult, setCalibrationResult] = useState<{ key: string; report?: CalibrationInspection; error?: string }>();
  const [calibrationRevision, setCalibrationRevision] = useState(0);
  const [qualityBusy, setQualityBusy] = useState(false);
  const [qualityElapsedSeconds, setQualityElapsedSeconds] = useState(0);
  const [approvedReviewDigests, setApprovedReviewDigests] = useState<string[]>([]);
  const [capabilities, setCapabilities] = useState<RuntimeCapabilities | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus>("IDLE");
  const [runLaunchBusy, setRunLaunchBusy] = useState(false);
  const [runElapsedSeconds, setRunElapsedSeconds] = useState(0);
  const runStartedAtRef = useRef<number | undefined>(undefined);
  const runClockRef = useRef<number | undefined>(undefined);
  const [stages, setStages] = useState<StageProgress[]>(initialStages);
  const [overallProgress, setOverallProgress] = useState(0);
  const [runProgress, setRunProgress] = useState<PipelineProgressEvent>();
  const progressContextRef = useRef<string | undefined>(undefined);
  const terminalJobIdRef = useRef<string | undefined>(undefined);
  const [jobId, setJobId] = useState<string>();
  const [executionMode, setExecutionMode] = useState<"native" | "demo">("native");
  const [artifacts, setArtifacts] = useState<OutputArtifact[]>([]);
  const [screening, setScreening] = useState<ScreeningSummary>();
  const [outputParent, setOutputParentState] = useState<string | undefined>(() => nativeRuntime ? storedOutputParent() : undefined);
  const setOutputParent = (value: string | undefined) => { setOutputParentState(value); if (nativeRuntime) rememberOutputParent(value); };
  const [outputDirectory, setOutputDirectory] = useState<string>();
  const [errorMessage, setErrorMessage] = useState<string>();
  const [demoMode, setDemoMode] = useState(false);
  const [inventoryBusy, setInventoryBusy] = useState(false);
  const [drizzleEnabled, setDrizzleEnabled] = useState(false);
  // Drizzle geometry (scale, drop shrink, kernel) travels with the recipe; the
  // engine validates the same ranges, so the UI only offers valid values.
  const [drizzleScale, setDrizzleScale] = useState<DrizzleScale>(2);
  const [drizzleDropShrink, setDrizzleDropShrink] = useState(0.9);
  const [drizzleKernel, setDrizzleKernel] = useState<DrizzleKernel>("square");
  // Opt-in per session. The native recipe selects local or global normalization
  // from this flag; it must not infer the choice from a visible progress stage.
  const [localNormalizationEnabled, setLocalNormalizationEnabled] = useState(false);
  const [catalogList, setCatalogList] = useState<CatalogListResponse>();
  const [catalogDoctor, setCatalogDoctor] = useState<CatalogDoctorResponse>();
  const [solverDoctor, setSolverDoctor] = useState<SolverDoctorResponse>();
  const [catalogTermsAccepted, setCatalogTermsAccepted] = useState(false);
  const [catalogJobId, setCatalogJobId] = useState<string>();
  const [catalogProgress, setCatalogProgress] = useState<CatalogProgressEvent>();
  const [catalogStatus, setCatalogStatus] = useState<"IDLE" | "DOWNLOADING" | "VERIFYING" | "READY" | "CANCELLED" | "FAILED">("IDLE");
  const [catalogError, setCatalogError] = useState<string>();
  const [solverSetupBusy, setSolverSetupBusy] = useState(false);
  const inventoryInFlightRef = useRef(false);
  const qualityInFlightRef = useRef(false);
  const stepRef = useRef(step);
  stepRef.current = step;
  const assetsRef = useRef(assets);
  assetsRef.current = assets;
  const timerRef = useRef<number | undefined>(undefined);
  const jobIdRef = useRef<string | undefined>(undefined);
  const runLaunchInFlightRef = useRef(false);
  const catalogJobIdRef = useRef<string | undefined>(undefined);
  const cancellingRef = useRef(false);
  const catalogCancellingRef = useRef(false);
  const pendingProgressRef = useRef<PipelineProgressEvent[]>([]);
  const pendingTerminalRef = useRef<
    { kind: "complete"; event: PipelineCompleteEvent; receivedAt: number }
    | { kind: "error"; event: PipelineErrorEvent; receivedAt: number }
    | undefined
  >(undefined);
  const pendingCatalogCompleteRef = useRef<CatalogCompleteEvent | undefined>(undefined);
  const translatorRef = useRef(t);

  useEffect(() => {
    translatorRef.current = t;
  }, [t]);

  const stopRunClock = useCallback((endedAt = performance.now()) => {
    window.clearInterval(runClockRef.current);
    runClockRef.current = undefined;
    if (runStartedAtRef.current === undefined) return;
    setRunElapsedSeconds(Math.max(0, Math.floor((endedAt - runStartedAtRef.current) / 1000)));
    runStartedAtRef.current = undefined;
  }, []);

  const startRunClock = () => {
    window.clearInterval(runClockRef.current);
    const startedAt = performance.now();
    runStartedAtRef.current = startedAt;
    setRunElapsedSeconds(0);
    runClockRef.current = window.setInterval(() => setRunElapsedSeconds(Math.floor((performance.now() - startedAt) / 1000)), 1000);
  };

  useEffect(() => {
    if (!qualityBusy) return;
    const started = performance.now();
    setQualityElapsedSeconds(0);
    const timer = window.setInterval(() => setQualityElapsedSeconds(Math.floor((performance.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [qualityBusy]);

  const refreshSolverSetup = useCallback(async () => {
    if (!nativeRuntime) return;
    setSolverSetupBusy(true);
    setCatalogError(undefined);
    setSolverDoctor(undefined);
    try {
      const [listing, catalogState, solverState] = await Promise.allSettled([
        desktopBridge.catalogList(), desktopBridge.catalogDoctor(), desktopBridge.solverDoctor(),
      ]);
      setCatalogList(listing.status === "fulfilled" ? listing.value : undefined);
      setCatalogDoctor(catalogState.status === "fulfilled" ? catalogState.value : undefined);
      setSolverDoctor(solverState.status === "fulfilled" ? solverState.value : undefined);
      setCatalogStatus(catalogState.status === "fulfilled" && catalogState.value.ok ? "READY" : "IDLE");
      const failures = [listing, catalogState, solverState].flatMap((result) => result.status === "rejected" ? [String(result.reason)] : []);
      if (failures.length) throw new Error(failures.join("\n"));
    } finally { setSolverSetupBusy(false); }
  }, [nativeRuntime]);

  const importPaths = useCallback(async (paths: string[], roleHint?: FrameRole) => {
    if (!paths.length || !nativeRuntime || stepRef.current !== "import" || inventoryInFlightRef.current || qualityInFlightRef.current || runLaunchInFlightRef.current) return;
    inventoryInFlightRef.current = true;
    setInventoryBusy(true);
    setErrorMessage(undefined);
    try {
      const inspected = await desktopBridge.inspectPaths({ paths, roleHint });
      setCalibrationResult(undefined);
      setCalibrationRevision((value) => value + 1);
      setProjectName(inspected.projectName || "Ultra-Fast WBPP project");
      const priorLights = new Set(assetsRef.current.filter((asset) => asset.role === "LIGHT").map((asset) => asset.path));
      const touchesLights = inspected.sources.some((source) => source.role === "LIGHT" || source.paths.some((path) => priorLights.has(path)));
      // A calibration-only addition does not change raw Light pixel evidence.
      // Reimporting any Light still invalidates it, even at the same path.
      if (touchesLights) { setQualityInspection(undefined); setGate({ pass: 0, review: 0, hardFail: 0 }); }
      setApprovedReviewDigests([]);
      setSources((current) => {
        const next = current.map((source) => ({ ...source }));
        for (const group of inspected.sources) {
          const source = next.find((candidate) => candidate.role === group.role);
          if (!source) continue;
          source.confirmed = (!source.paths.length || source.confirmed) && !group.needsConfirmation;
          source.paths = deduplicate([...source.paths, ...group.paths]);
          source.fileCount = source.paths.length;
          source.detected = true;
        }
        return next;
      });
      setAssets((current) => {
        const byPath = new Map(current.map((asset) => [asset.path, asset]));
        for (const asset of inspected.assets ?? []) byPath.set(asset.path, asset);
        return [...byPath.values()];
      });
      setSelectedRole(undefined);
    } catch (error) {
      setErrorMessage(String(error));
    } finally {
      inventoryInFlightRef.current = false;
      setInventoryBusy(false);
    }
  }, [nativeRuntime]);

  useEffect(() => {
    const masters = assets.filter((asset) => MASTER_ROLES.has(asset.role));
    setMasterOverrides((current) => masters.map((master) => buildMasterOverride(master, current.find((item) => item.sourcePath === master.path && item.sourceSha256 === master.sourceSha256))));
  }, [assets]);

  const calibrationRequestKey = JSON.stringify({
    paths: assets.map((asset) => asset.path),
    recipe: {
      calibration: {
        workflow: "mono-standard-v1",
        bias: "OPTIONAL",
        masterMetadataOverrides: masterOverrideRequests(masterOverrides),
      },
      rawFrameMetadataOverrides: [],
    },
  });
  const calibrationCheckKey = `${calibrationRevision}:${calibrationRequestKey}`;
  const hasCalibrationInputs = assets.some((asset) => asset.role === "LIGHT");
  useEffect(() => {
    if (!nativeRuntime || !hasCalibrationInputs || inventoryBusy) return;
    let current = true;
    const timer = window.setTimeout(() => {
      desktopBridge.inspectCalibration(JSON.parse(calibrationRequestKey))
        .then((report) => { if (current) setCalibrationResult({ key: calibrationCheckKey, report }); })
        .catch((error) => { if (current) setCalibrationResult({ key: calibrationCheckKey, error: String(error) }); });
    }, 150);
    return () => { current = false; window.clearTimeout(timer); };
  }, [calibrationCheckKey, calibrationRequestKey, nativeRuntime, hasCalibrationInputs, inventoryBusy]);
  const calibrationInspection = !inventoryBusy && calibrationResult?.key === calibrationCheckKey ? calibrationResult.report : undefined;
  const calibrationError = calibrationResult?.key === calibrationCheckKey ? calibrationResult.error : undefined;
  const calibrationBusy = nativeRuntime && hasCalibrationInputs && !calibrationInspection && !calibrationError;
  const recheckCalibration = () => setCalibrationRevision((value) => value + 1);

  useEffect(() => {
    desktopBridge.getCapabilities().then(setCapabilities).catch((error) => { setErrorMessage(String(error)); setCapabilities(null); });
    if (nativeRuntime) refreshSolverSetup().catch((error) => setCatalogError(String(error)));
  }, [nativeRuntime, refreshSolverSetup]);

  useEffect(() => {
    let disposeDrop: () => void = () => undefined;
    let disposed = false;
    if (nativeRuntime) listenForDesktopDrops((paths) => void importPaths(paths, selectedRole)).then((dispose) => { if (disposed) dispose(); else disposeDrop = dispose; });
    return () => { disposed = true; disposeDrop(); };
  }, [importPaths, nativeRuntime, selectedRole]);

  const handleProgress = useCallback((event: PipelineProgressEvent) => {
    if (!jobIdRef.current) { if (runLaunchInFlightRef.current) pendingProgressRef.current.push(event); return; }
    if (event.jobId !== jobIdRef.current || event.jobId === terminalJobIdRef.current || cancellingRef.current) return;
    const context = event.scope === "panel" ? `panel:${event.panelId}` : event.scope ?? "legacy";
    const resetStages = context !== progressContextRef.current;
    progressContextRef.current = context;
    setRunProgress(event);
    const fraction = Number.isFinite(event.fraction) ? Math.max(0, Math.min(1, event.fraction)) : 0;
    setStages((current) => {
      const active = resetStages ? initialStages(event.scope) : current;
      const stageIndex = active.findIndex((stage) => stage.stageId === event.stageId);
      return active.map((stage, index) => {
        if (stageIndex < 0 || index > stageIndex) return stage;
        if (index < stageIndex) return { ...stage, status: "DONE", percent: 100 };
        return { ...stage, percent: Math.max(stage.percent, fraction * 100), status: event.state === "failed" ? "FAILED" : event.state === "succeeded" ? "DONE" : "RUNNING" };
      });
    });
    const legacyIndex = STAGE_DEFINITIONS.findIndex((stage) => stage.stageId === event.stageId);
    const overall = event.overallFraction !== undefined && Number.isFinite(event.overallFraction)
      ? event.overallFraction : legacyIndex >= 0 ? (legacyIndex + fraction) / STAGE_DEFINITIONS.length : event.stageId === undefined ? fraction : 0;
    setOverallProgress((current) => Math.max(current, Math.min(99, Math.max(0, Math.floor(overall * 100)))));
  }, []);

  const handleComplete = useCallback((event: PipelineCompleteEvent, receivedAt = performance.now()) => {
    if (!jobIdRef.current) { if (runLaunchInFlightRef.current) pendingTerminalRef.current ??= { kind: "complete", event, receivedAt }; return; }
    if (event.jobId !== jobIdRef.current || event.jobId === terminalJobIdRef.current) return;
    if (cancellingRef.current) { pendingTerminalRef.current ??= { kind: "complete", event, receivedAt }; return; }
    terminalJobIdRef.current = event.jobId;
    stopRunClock(receivedAt);
    const required = event.gate.checks.filter((check) => check.required);
    const solvedProducts = event.artifacts.filter((artifact) => artifact.kind === "SOLVED_MONO_FITS" || artifact.kind === "MASTER");
    const gateReady = event.gate.decision === "ready" && required.length > 0 && required.every((check) => check.passed);
    const solvedReady = solvedProducts.length > 0 && solvedProducts.every((artifact) => Boolean(artifact.receipt?.astrometry));
    if (!gateReady || !solvedReady) {
      setRunStatus("FAILED");
      setErrorMessage(translatorRef.current("finalGateFailed"));
      return;
    }
    setArtifacts(event.artifacts);
    setScreening(event.screening);
    setOutputDirectory(event.outputDirectory);
    setStages((current) => current.map((stage) => ({ ...stage, status: "DONE", percent: 100 })));
    setOverallProgress(100);
    setRunStatus("COMPLETED");
    setStep("result");
  }, [stopRunClock]);

  const handleError = useCallback((event: PipelineErrorEvent, receivedAt = performance.now()) => {
    if (!jobIdRef.current) { if (runLaunchInFlightRef.current) pendingTerminalRef.current ??= { kind: "error", event, receivedAt }; return; }
    if (event.jobId !== jobIdRef.current || event.jobId === terminalJobIdRef.current) return;
    if (cancellingRef.current) { pendingTerminalRef.current ??= { kind: "error", event, receivedAt }; return; }
    terminalJobIdRef.current = event.jobId;
    stopRunClock(receivedAt);
    setRunStatus("FAILED");
    setErrorMessage(`${event.code}: ${event.message}`);
  }, [stopRunClock]);

  const replayPendingRunEvents = () => {
    for (const event of pendingProgressRef.current.splice(0)) handleProgress(event);
    const pending = pendingTerminalRef.current; pendingTerminalRef.current = undefined;
    // Preserve the first terminal event and its arrival time across pending RPCs.
    if (pending?.kind === "complete") handleComplete(pending.event, pending.receivedAt);
    else if (pending?.kind === "error") handleError(pending.event, pending.receivedAt);
  };

  useEffect(() => {
    if (!nativeRuntime) return;
    let dispose: () => void = () => undefined;
    let disposed = false;
    listenForPipelineEvents({ onProgress: handleProgress, onArtifact: () => undefined, onComplete: handleComplete, onError: handleError })
      .then((value) => { if (disposed) value(); else dispose = value; });
    return () => { disposed = true; dispose(); };
  }, [handleComplete, handleError, handleProgress, nativeRuntime]);

  const finishCatalogInstall = useCallback(async (event: CatalogCompleteEvent) => {
    if (catalogJobIdRef.current && event.jobId !== catalogJobIdRef.current) return;
    setCatalogStatus("VERIFYING");
    try {
      await desktopBridge.verifyCatalog(event.catalogId, true);
      await refreshSolverSetup();
      setCatalogStatus("READY");
      setCatalogError(undefined);
    } catch (error) {
      setCatalogStatus("FAILED");
      setCatalogError(translatorRef.current("downloadVerifyFailed", { error: String(error) }));
    } finally {
      catalogJobIdRef.current = undefined;
      setCatalogJobId(undefined);
    }
  }, [refreshSolverSetup]);

  useEffect(() => {
    if (!nativeRuntime) return;
    let dispose: () => void = () => undefined;
    let disposed = false;
    listenForCatalogEvents({
      onProgress: (event) => {
        if (!catalogJobIdRef.current || event.jobId === catalogJobIdRef.current) setCatalogProgress(event);
      },
      onComplete: (event) => {
        if (catalogCancellingRef.current) return;
        if (!catalogJobIdRef.current) pendingCatalogCompleteRef.current = event;
        else void finishCatalogInstall(event);
      },
      onError: (event: CatalogErrorEvent) => {
        if (catalogCancellingRef.current) return;
        if (catalogJobIdRef.current && event.jobId !== catalogJobIdRef.current) return;
        setCatalogStatus("FAILED");
        setCatalogError(`${event.code}: ${event.message}`);
      },
    }).then((value) => { if (disposed) value(); else dispose = value; });
    return () => { disposed = true; dispose(); };
  }, [finishCatalogInstall, nativeRuntime]);

  useEffect(() => () => { window.clearInterval(timerRef.current); window.clearInterval(runClockRef.current); }, []);

  const confirmRole = (role: FrameRole) => setSources((current) => current.map((source) => source.role === role ? { ...source, confirmed: true } : source));
  const pickFiles = async (role?: FrameRole) => { try { await importPaths(await desktopBridge.pickInputFiles(role), role); } catch (error) { setErrorMessage(String(error)); } };
  const pickDirectories = async (role?: FrameRole) => { try { await importPaths(await desktopBridge.pickInputDirectories(role), role); } catch (error) { setErrorMessage(String(error)); } };
  const chooseOutputParent = async () => { try { const selected = await desktopBridge.pickOutputParent(); if (selected) setOutputParent(selected); } catch (error) { setErrorMessage(String(error)); } };
  const useOutputParentPath = (path: string) => {
    if (!nativeRuntime || !["import", "inspect"].includes(stepRef.current) || runLaunchInFlightRef.current || runStatus === "RUNNING" || runStatus === "CANCELLING") return;
    // The native start command resolves the path and requires an existing directory.
    setOutputParent(path.trim() || undefined);
  };

  const updateMasterOverride = (digest: string, patch: Partial<MasterMetadataOverride>) => setMasterOverrides((current) => current.map((item) => item.sourceSha256 === digest ? { ...item, ...patch, needsMetadataOverride: true, confirmed: false } : item));
  const resetMasterOverride = (digest: string) => setMasterOverrides((current) => current.map((item) => {
    const master = assets.find((asset) => asset.path === item.sourcePath && asset.sourceSha256 === digest);
    return item.sourceSha256 === digest && master ? buildMasterOverride(master) : item;
  }));
  const confirmMasterOverride = (digest: string) => setMasterOverrides((current) => current.map((item) => {
    if (item.sourceSha256 !== digest) return item;
    const numbersReady = [item.gain, item.offset, item.temperatureCelsius, item.exposureSeconds, ...item.binning].every((value) => value === null || Number.isFinite(value));
    const binningReady = item.binning.every((value) => value === null) || item.binning.every((value) => value !== null && Number.isInteger(value) && value > 0);
    const numericDomainReady = item.numericDomain === null || (item.normalizedUnitScale !== null && Number.isFinite(item.normalizedUnitScale) && item.normalizedUnitScale > 0);
    return { ...item, confirmed: numbersReady && binningReady && numericDomainReady };
  }));

  const loadDemo = () => {
    if (nativeRuntime) return;
    setDemoMode(true); setExecutionMode("demo"); setProjectName("Clearly labelled browser demo");
    setSources(emptySources().map((source) => ({ ...source, paths: DEMO_COUNTS[source.role] ? [`/explicit-browser-demo/${source.role.toLowerCase()}`] : [], fileCount: DEMO_COUNTS[source.role], detected: DEMO_COUNTS[source.role] > 0, confirmed: source.role !== "LIGHT" })));
    setAssets(
      Array.from({ length: 4 }, (_, panel) =>
        ["R", "G", "B", "L"].flatMap((filter) =>
          Array.from({ length: 4 }, (_, index): InspectedAsset => ({
            path: `/explicit-browser-demo/p${panel + 1}/${filter}/${index}.fit`, role: "LIGHT", width: 6248, height: 4176, channels: 1,
            filter, target: `Panel ${panel + 1}`, camera: "DEMO MONO", exposureSeconds: 180, temperatureCelsius: -10, gain: 100, offset: 50,
            binning: [1, 1], cfaPattern: "NONE", readoutMode: "DEMO",
          })),
        ),
      ).flat(),
    );
    setGate({ pass: 347, review: 23, hardFail: 0 }); setSelectedRole("LIGHT"); setErrorMessage(undefined);
  };

  const clearSources = () => {
    if (inventoryInFlightRef.current || qualityInFlightRef.current || runLaunchInFlightRef.current || runStatus === "RUNNING" || runStatus === "CANCELLING") return;
    stopRunClock(); setRunElapsedSeconds(0);
    jobIdRef.current = undefined; pendingProgressRef.current = []; pendingTerminalRef.current = undefined;
    setSources(emptySources()); setAssets([]); setMasterOverrides([]); setGate({ pass: 0, review: 0, hardFail: 0 }); setQualityInspection(undefined); setApprovedReviewDigests([]); setSelectedRole(undefined); setStep("import"); setDemoMode(false); setArtifacts([]); setScreening(undefined); setOutputDirectory(undefined); setRunStatus("IDLE"); setRunLaunchBusy(false); setErrorMessage(undefined);
  };
  const runInspection = async (force = false) => {
    if (demoMode && !nativeRuntime) { setStep("inspect"); return; }
    const lightPaths = sources.find((source) => source.role === "LIGHT")?.paths ?? [];
    if (!nativeRuntime || !lightPaths.length || inventoryInFlightRef.current || qualityInFlightRef.current) return;
    if (!force && qualityReady) { setStep("inspect"); return; }
    qualityInFlightRef.current = true;
    setQualityBusy(true); setErrorMessage(undefined); setApprovedReviewDigests([]);
    try {
      const inspection = await desktopBridge.inspectQuality(lightPaths);
      setQualityInspection(inspection);
      setGate({ pass: inspection.counts.PASS, review: inspection.counts.REVIEW, hardFail: inspection.counts.HARD_FAIL });
      setStep("inspect");
    } catch (error) { setErrorMessage(t("qcPreflightFailed", { error: String(error) })); }
    finally { qualityInFlightRef.current = false; setQualityBusy(false); }
  };
  const toggleReviewApproval = (frame: InspectedLightQuality) => {
    if (!canApproveReview(frame)) return;
    setApprovedReviewDigests((current) => current.includes(frame.sourceSha256!)
      ? current.filter((digest) => digest !== frame.sourceSha256)
      : [...current, frame.sourceSha256!]);
  };

  const startDemo = () => {
    startRunClock(); terminalJobIdRef.current = undefined; cancellingRef.current = false;
    setJobId("explicit-browser-demo"); jobIdRef.current = "explicit-browser-demo"; setExecutionMode("demo"); setRunStatus("RUNNING"); setOverallProgress(1); setStep("run");
    window.clearInterval(timerRef.current);
    timerRef.current = window.setInterval(() => setOverallProgress((current) => {
      const next = Math.min(100, current + 4); const segment = 100 / STAGE_DEFINITIONS.length;
      setStages(initialStages().map((stage, index) => { const local = Math.max(0, Math.min(100, ((next - segment * index) / segment) * 100)); return { ...stage, percent: local, status: local >= 100 ? "DONE" : local > 0 ? "RUNNING" : "WAITING" }; }));
      if (next === 100) { window.clearInterval(timerRef.current); stopRunClock(); setArtifacts(DEMO_ARTIFACTS); setRunStatus("COMPLETED"); setStep("result"); }
      return next;
    }), 200);
  };

  const startRun = async () => {
    if (!canStart) return;
    if (runLaunchInFlightRef.current || runStatus === "RUNNING" || runStatus === "CANCELLING") return;
    setErrorMessage(undefined);
    if (demoMode && !nativeRuntime) { startDemo(); return; }
    if (!nativeRuntime || !capabilities?.available || !outputParent) return;
    runLaunchInFlightRef.current = true; setRunLaunchBusy(true);
    jobIdRef.current = undefined; pendingProgressRef.current = []; pendingTerminalRef.current = undefined;
    terminalJobIdRef.current = undefined; progressContextRef.current = undefined; cancellingRef.current = false;
    setJobId(undefined); setRunProgress(undefined); setExecutionMode("native"); setArtifacts([]); setScreening(undefined); setOutputDirectory(undefined); setRunStatus("RUNNING"); setOverallProgress(0); setStages(initialStages()); setStep("run");
    // Include launch-command work, but exclude the earlier review and screening.
    startRunClock();
    try {
      let sourceIndex = 0;
      const receipt = await desktopBridge.startRun({
        sources: sources.filter((source) => source.paths.length).flatMap((source) => source.paths.map((path) => ({ sourceId: safeSourceId(source.role, sourceIndex++), role: source.role, paths: [path], recursive: false }))),
        projectName,
        runLabel: runLabel(matrix, projectName),
        recipe: { balanced: true, drizzleEnabled, drizzleScale, drizzleDropShrink, drizzleKernel, localNormalizationEnabled, solverRequired: true, calibrationWorkflow: "mono-standard-v1" },
        masterMetadataOverrides: masterOverrideRequests(masterOverrides),
        rawFrameMetadataOverrides: [],
        reviewSelections: validApprovedReviewDigests.map((sourceSha256) => ({ sourceSha256, gatePolicyDigest: qualityInspection!.gatePolicyDigest })),
        outputParentDirectory: outputParent,
      });
      if (!receipt.accepted) throw new Error(t("runNotAccepted"));
      jobIdRef.current = receipt.jobId; setJobId(receipt.jobId); setOutputDirectory(receipt.outputDirectory);
      replayPendingRunEvents();
    } catch (error) { stopRunClock(); setRunStatus("FAILED"); setErrorMessage(String(error)); }
    finally { runLaunchInFlightRef.current = false; setRunLaunchBusy(false); }
  };

  const cancelRun = async () => {
    if (!jobId || cancellingRef.current || terminalJobIdRef.current === jobId) return;
    setRunStatus("CANCELLING"); cancellingRef.current = true;
    window.clearInterval(timerRef.current);
    try {
      if (executionMode === "native") await desktopBridge.cancelRun(jobId);
      pendingTerminalRef.current = undefined;
      terminalJobIdRef.current = jobId; stopRunClock(); setRunStatus("CANCELLED");
    } catch (error) {
      // A rejected cancellation does not prove that the worker has stopped.
      cancellingRef.current = false;
      setRunStatus("RUNNING"); setErrorMessage(String(error));
      replayPendingRunEvents();
    } finally { cancellingRef.current = false; }
  };

  const recommendedCatalog = useMemo<CatalogInfo | undefined>(() => catalogList?.catalogs.find((item) => item.catalogId === "astrometry-net-4107-4112"), [catalogList]);
  const startCatalogInstall = async () => {
    if (!recommendedCatalog || !catalogTermsAccepted || catalogStatus === "DOWNLOADING") return;
    setCatalogError(undefined); setCatalogProgress(undefined); setCatalogStatus("DOWNLOADING"); catalogCancellingRef.current = false; catalogJobIdRef.current = undefined; pendingCatalogCompleteRef.current = undefined;
    try {
      const receipt = await desktopBridge.startCatalogInstall({ catalogId: recommendedCatalog.catalogId, acceptedTermsId: recommendedCatalog.providerTerms.acceptanceId });
      catalogJobIdRef.current = receipt.jobId; setCatalogJobId(receipt.jobId);
      const pending = pendingCatalogCompleteRef.current; pendingCatalogCompleteRef.current = undefined; if (pending) await finishCatalogInstall(pending);
    } catch (error) { setCatalogStatus("FAILED"); setCatalogError(String(error)); }
  };
  const cancelCatalogInstall = async () => {
    if (!catalogJobId) return;
    catalogCancellingRef.current = true;
    try { await desktopBridge.cancelCatalogInstall(catalogJobId); setCatalogStatus("CANCELLED"); setCatalogJobId(undefined); catalogJobIdRef.current = undefined; }
    catch (error) { catalogCancellingRef.current = false; setCatalogStatus("FAILED"); setCatalogError(String(error)); }
  };
  const openCatalogTerms = async (url: string) => {
    try { await desktopBridge.openProviderTerms(url); }
    catch (error) { setCatalogError(String(error)); }
  };
  const revealOutput = async (path: string) => {
    try { await desktopBridge.revealOutput(path); }
    catch (error) { setErrorMessage(String(error)); }
  };
  const recheckSolverSetup = async () => {
    if (solverSetupBusy || catalogStatus === "DOWNLOADING" || catalogStatus === "VERIFYING") return;
    try { await refreshSolverSetup(); }
    catch (error) { setCatalogError(String(error)); }
  };

  const inputBusy = inventoryBusy || qualityBusy;
  const canInspect = !inputBusy && (sources.find((source) => source.role === "LIGHT")?.fileCount ?? 0) > 0;
  const allRequiredConfirmed = sources.filter((source) => source.paths.length).every((source) => source.confirmed);
  const calibrationReady = Boolean(calibrationInspection?.calibrationReady && calibrationInspection.status === "READY");
  const masterOverridesReady = masterOverrides.every((item) => item.confirmed);
  // Bayer (one-shot-colour) frames are processed as colour channel groups;
  // only a pattern the engine does not know blocks the run.
  const cfaAssets = assets.filter((asset) => bayerCfa(asset.cfaPattern));
  const cfaBlockedAssets = assets.filter((asset) => !unknownCfa(asset.cfaPattern) && !monoCfa(asset.cfaPattern) && !bayerCfa(asset.cfaPattern));
  const cfaPattern = cfaAssets[0]?.cfaPattern.trim().toUpperCase();
  const solveField = solverDoctor?.backends.find((backend) => backend.backendId === "astrometry-net");
  const astap = solverDoctor?.backends.find((backend) => backend.backendId === "astap");
  const solverSetupReady = Boolean(catalogDoctor?.ok && solveField?.executionReady);
  const qualityReady = Boolean(qualityInspection && qualityInspection.frames.length === (sources.find((source) => source.role === "LIGHT")?.fileCount ?? 0));
  const panelCount = useMemo(() => panelMatrix(assets).length, [assets]);
  const canApproveReview = (frame: InspectedLightQuality) => reviewCanBeApproved(frame, qualityInspection, panelCount);
  const validApprovedReviewDigests = useMemo(() => approvedReviewDigests.filter((digest) => qualityInspection?.frames.some((frame) => frame.sourceSha256 === digest && reviewCanBeApproved(frame, qualityInspection, panelCount))), [approvedReviewDigests, qualityInspection, panelCount]);
  const admittedPaths = useMemo(() => new Set(qualityInspection?.frames.filter((frame) => frame.disposition === "PASS" || (frame.disposition === "REVIEW" && frame.sourceSha256 && validApprovedReviewDigests.includes(frame.sourceSha256))).map((frame) => frame.path) ?? []), [qualityInspection, validApprovedReviewDigests]);
  const matrix = useMemo(() => panelMatrix(assets, admittedPaths), [assets, admittedPaths]);
  const minimumAdmittedLights = 2;
  const insufficientQualityPanels = qualityReady ? matrix.filter((cell) => cell.admittedCount < minimumAdmittedLights) : [];
  // Screening before the run is optional: the run screens every Light itself
  // and lists the excluded frames with its result.  Before a screening, a
  // panel only needs enough Lights to register; after one, enough admitted.
  const insufficientPanels = qualityReady ? insufficientQualityPanels : matrix.filter((cell) => cell.lightCount < minimumAdmittedLights);
  const panelsReady = matrix.length > 0 && insufficientPanels.length === 0;
  const runNavigationLocked = runLaunchBusy || runStatus === "RUNNING" || runStatus === "CANCELLING";
  const canStart = demoMode && !nativeRuntime ? !runNavigationLocked : Boolean(!runNavigationLocked && !inputBusy && nativeRuntime && capabilities?.available && outputParent && calibrationReady && allRequiredConfirmed && masterOverridesReady && cfaBlockedAssets.length === 0 && solverSetupReady && panelsReady);
  const importedTotal = useMemo(() => sources.reduce((sum, source) => sum + source.fileCount, 0), [sources]);
  const firstSolved = artifacts.find((artifact) => artifact.receipt?.astrometry)?.receipt?.astrometry;

  return {
    step, setStep, sources, assets, selectedRole, setSelectedRole, isDragging, setDragging, gate, capabilities, runStatus, runLaunchBusy, runElapsedSeconds, stages, overallProgress, runProgress, executionMode, artifacts, screening,
    importPaths, confirmRole, loadDemo, clearSources, runInspection, startRun, cancelRun, canInspect, allRequiredConfirmed, importedTotal, nativeRuntime,
    browserDemoAvailable: !nativeRuntime, pickFiles, pickDirectories, chooseOutputParent, useOutputParentPath, outputParent, outputDirectory, canStart, inventoryBusy, inputBusy, errorMessage, calibrationReady,
    firstSolved, demoMode, projectName, matrix, masterOverrides, updateMasterOverride, resetMasterOverride, confirmMasterOverride, masterOverridesReady, runNavigationLocked,
    cfaBlockedAssets, cfaAssets, cfaPattern,
    qualityInspection, qualityBusy, qualityElapsedSeconds, qualityReady, approvedReviewDigests: validApprovedReviewDigests, toggleReviewApproval, canApproveReview,
    insufficientQualityPanels, insufficientPanels, minimumAdmittedLights,
    calibrationInspection, calibrationBusy, calibrationError, recheckCalibration,
    drizzleEnabled, setDrizzleEnabled, drizzleScale, setDrizzleScale, drizzleDropShrink, setDrizzleDropShrink, drizzleKernel, setDrizzleKernel, localNormalizationEnabled, setLocalNormalizationEnabled,
    catalogList, catalogDoctor, solverDoctor, recommendedCatalog, solveField, astap, solverSetupReady, catalogTermsAccepted, setCatalogTermsAccepted,
    catalogProgress, catalogStatus, catalogError, startCatalogInstall, cancelCatalogInstall, openCatalogTerms, revealOutput, solverSetupBusy, recheckSolverSetup,
  };
}
