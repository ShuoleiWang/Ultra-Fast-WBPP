import { useCallback,useEffect,useMemo,useRef,useState } from "react";
import { desktopBridge,hasTauriRuntime,listenForCatalogEvents,listenForDesktopDrops,listenForPipelineEvents } from "./bridge";
import { demoBlinkManifest } from "./demoAutopilot";
import type { Translator } from "./i18n";
import { emptySources } from "./sourceDefaults";
import type {
  BlinkDecision,
  BlinkMeasureResponse,
  CalibrationInspection,
  CatalogCompleteEvent,
  CatalogDoctorResponse,
  CatalogErrorEvent,
  CatalogInfo,
  CatalogListResponse,
  CatalogProgressEvent,
  DrizzleKernel,
  DrizzleScale,
  FrameRole,
  GateSummary,
  InspectedAsset,
  MasterMetadataOverride,
  OutputArtifact,
  PipelineCompleteEvent,
  PipelineErrorEvent,
  PipelineProgressEvent,
  QualityInspection,
  RunStatus,
  RuntimeCapabilities,
  ScreeningSummary,
  SelectionFile,
  SolverDoctorResponse,
  SourceSet,
  StageProgress,
  WorkflowStep
} from "./types";

import { CONTENT_DIGEST,DECISION_HISTORY_LIMIT,DEMO_ARTIFACTS,DEMO_COUNTS,MASTER_ROLES,MINIMUM_KEPT_PER_CHANNEL,STAGE_DEFINITIONS,bayerCfa,blinkSessionCovers,buildMasterOverride,deduplicate,initialStages,inventoryKeyOf,known,masterOverrideRequests,monoCfa,panelMatrix,rememberOutputParent,runLabel,safeSourceId,solverScienceReady,storedOutputParent,unknownCfa,type BlinkChannelSummary,type BlinkSession,type SolverBackendId } from "./workflow/model";
export { runLabel,solverScienceReady } from "./workflow/model";
export type { BlinkChannelSummary,BlinkSession,SolverBackendId } from "./workflow/model";

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
  // Blink screening: the measured session, the per-frame decisions keyed by
  // content digest, and their undo stack.  Refs mirror the two so a burst of
  // keyboard actions in one event commits in order.
  const [blinkSession, setBlinkSession] = useState<BlinkSession>();
  const blinkSessionRef = useRef<BlinkSession | undefined>(undefined);
  blinkSessionRef.current = blinkSession;
  const [blinkBusy, setBlinkBusy] = useState(false);
  const [blinkElapsedSeconds, setBlinkElapsedSeconds] = useState(0);
  const [decisions, setDecisions] = useState<Record<string, BlinkDecision>>({});
  const decisionsRef = useRef(decisions);
  decisionsRef.current = decisions;
  const [decisionHistory, setDecisionHistory] = useState<Record<string, BlinkDecision>[]>([]);
  const decisionHistoryRef = useRef(decisionHistory);
  decisionHistoryRef.current = decisionHistory;
  const [blinkChannel, setBlinkChannel] = useState<string>();
  const [viewedFrames, setViewedFrames] = useState<Record<string, true>>({});
  const viewedFramesRef = useRef<Record<string, true>>({});
  const [confirmedChannels, setConfirmedChannels] = useState<Record<string, true>>({});
  const confirmedChannelsRef = useRef<Record<string, true>>({});
  const [previewFailures, setPreviewFailures] = useState<Record<string, string>>({});
  const previewFailuresRef = useRef<Record<string, string>>({});
  const resetBlinkReview = () => {
    viewedFramesRef.current = {}; setViewedFrames({});
    confirmedChannelsRef.current = {}; setConfirmedChannels({});
    previewFailuresRef.current = {}; setPreviewFailures({});
  };
  const blinkInFlightRef = useRef(false);
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
  // The engine/desktop failure code of the current run (e.g. ASTROMETRY_REQUIRED), for the run view's failure card.
  const [runFailureCode, setRunFailureCode] = useState<string>();
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

  useEffect(() => {
    if (!blinkBusy) return;
    const started = performance.now();
    setBlinkElapsedSeconds(0);
    const timer = window.setInterval(() => setBlinkElapsedSeconds(Math.floor((performance.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [blinkBusy]);

  const clearBlinkSession = useCallback(() => {
    setBlinkSession(undefined); blinkSessionRef.current = undefined;
    decisionsRef.current = {}; setDecisions({});
    decisionHistoryRef.current = []; setDecisionHistory([]);
    setBlinkChannel(undefined);
    resetBlinkReview();
  }, []);

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
    if (!paths.length || !nativeRuntime || stepRef.current !== "import" || inventoryInFlightRef.current || qualityInFlightRef.current || blinkInFlightRef.current || runLaunchInFlightRef.current) return;
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
      // Reimporting any Light still invalidates it, even at the same path; the
      // blink session and its decisions go with it.
      if (touchesLights) { setQualityInspection(undefined); setGate({ pass: 0, review: 0, hardFail: 0 }); clearBlinkSession(); }

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
  }, [clearBlinkSession, nativeRuntime]);

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

  // A failed run stops its clock and its spinner: the stage that was running is
  // marked FAILED, the others keep their state, so the list shows where it stopped.
  const failStages = (current: StageProgress[]) => current.map((stage) => stage.status === "RUNNING" ? { ...stage, status: "FAILED" as const } : stage);

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
      setStages(failStages);
      setRunStatus("FAILED");
      setRunFailureCode("PROJECT_RESULT_GATE_BLOCKED");
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
    setStages(failStages);
    setRunStatus("FAILED");
    setRunFailureCode(event.code);
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
    if (!nativeRuntime || !["import", "inspect", "blink"].includes(stepRef.current) || runLaunchInFlightRef.current || runStatus === "RUNNING" || runStatus === "CANCELLING") return;
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
    if (inventoryInFlightRef.current || qualityInFlightRef.current || blinkInFlightRef.current || runLaunchInFlightRef.current || runStatus === "RUNNING" || runStatus === "CANCELLING") return;
    stopRunClock(); setRunElapsedSeconds(0);
    jobIdRef.current = undefined; pendingProgressRef.current = []; pendingTerminalRef.current = undefined;
    setSources(emptySources()); setAssets([]); setMasterOverrides([]); setGate({ pass: 0, review: 0, hardFail: 0 }); setQualityInspection(undefined); clearBlinkSession(); setSelectedRole(undefined); setStep("import"); setDemoMode(false); setArtifacts([]); setScreening(undefined); setOutputDirectory(undefined); setRunStatus("IDLE"); setRunLaunchBusy(false); setErrorMessage(undefined);
  };
  const runInspection = async (force = false) => {
    if (demoMode && !nativeRuntime) { setStep("inspect"); return; }
    const lightPaths = sources.find((source) => source.role === "LIGHT")?.paths ?? [];
    if (!nativeRuntime || !lightPaths.length || inventoryInFlightRef.current || qualityInFlightRef.current || blinkInFlightRef.current) return;
    if (!force && qualityReady) { setStep("inspect"); return; }
    qualityInFlightRef.current = true;
    setQualityBusy(true); setErrorMessage(undefined);
    try {
      const inspection = await desktopBridge.inspectQuality(lightPaths);
      setQualityInspection(inspection);
      setGate({ pass: inspection.counts.PASS, review: inspection.counts.REVIEW, hardFail: inspection.counts.HARD_FAIL });
      setStep("inspect");
    } catch (error) { setErrorMessage(t("qcPreflightFailed", { error: String(error) })); }
    finally { qualityInFlightRef.current = false; setQualityBusy(false); }
  };
  const installBlinkSession = (manifest: BlinkMeasureResponse, lightPaths: string[], demo: boolean) => {
    const session: BlinkSession = { manifest, sessionDirectory: manifest.sessionDirectory, inventoryKey: inventoryKeyOf(lightPaths), demo };
    blinkSessionRef.current = session; setBlinkSession(session);
    const defaults = Object.fromEntries(manifest.frames.map((frame) => [frame.sourceSha256, "KEEP" as const]));
    resetBlinkReview();
    decisionsRef.current = defaults; setDecisions(defaults);
    decisionHistoryRef.current = []; setDecisionHistory([]);
    setBlinkChannel(manifest.channels[0]?.channelId);
  };
  // Measures the Lights for blinking (previews, flags, reference) and opens
  // the view; a session that still matches the Lights is reopened instead.
  const runBlink = async (force = false) => {
    if (demoMode && !nativeRuntime) { if (!blinkSession) installBlinkSession(demoBlinkManifest(), [], true); setStep("blink"); return; }
    const lightPaths = sources.find((source) => source.role === "LIGHT")?.paths ?? [];
    if (!nativeRuntime || !lightPaths.length || inventoryInFlightRef.current || qualityInFlightRef.current || blinkInFlightRef.current) return;
    if (!force && blinkReady) { setStep("blink"); return; }
    blinkInFlightRef.current = true;
    setBlinkBusy(true); setErrorMessage(undefined);
    try {
      const masterFlats = assets.filter((asset) => asset.role === "MASTER_FLAT" && known(asset.filter)).map((asset) => ({ filter: asset.filter, path: asset.path }));
      const masterDarks = assets.filter((asset) => asset.role === "MASTER_DARK").map((asset) => ({ path: asset.path, ...(asset.exposureSeconds ? { exposureSeconds: asset.exposureSeconds } : {}) }));
      const biases = assets.filter((asset) => asset.role === "MASTER_BIAS");
      const manifest = await desktopBridge.blinkMeasure({ paths: lightPaths, masterFlats, masterDarks, ...(biases.length === 1 ? { masterBias: biases[0].path } : {}) });
      installBlinkSession(manifest, lightPaths, false);
      setStep("blink");
    } catch (error) { setErrorMessage(t("blinkFailed", { error: String(error) })); }
    finally { blinkInFlightRef.current = false; setBlinkBusy(false); }
  };
  const invalidateReviewedChannels = (changed: string[]) => {
    const ids = new Set(blinkSessionRef.current?.manifest.frames.filter((f) => changed.includes(f.sourceSha256)).map((f) => f.channelId));
    const next = Object.fromEntries(Object.entries(confirmedChannelsRef.current).filter(([id]) => !ids.has(id))) as Record<string, true>;
    confirmedChannelsRef.current = next; setConfirmedChannels(next);
  };
  const markFrameViewed = useCallback((digest: string) => {
    if (viewedFramesRef.current[digest] || !blinkSessionRef.current?.manifest.frames.some((f) => f.sourceSha256 === digest)) return;
    const next = { ...viewedFramesRef.current, [digest]: true as const };
    viewedFramesRef.current = next; setViewedFrames(next);
  }, []);
  const reportPreviewFailure = useCallback((digest: string, error: string | undefined) => {
    if (previewFailuresRef.current[digest] === error) return;
    const next = { ...previewFailuresRef.current };
    if (error) next[digest] = error; else delete next[digest];
    previewFailuresRef.current = next; setPreviewFailures(next);
  }, []);
  const confirmBlinkChannel = (channelId: string) => {
    const session = blinkSessionRef.current;
    const frames = session?.manifest.frames.filter((f) => f.channelId === channelId) ?? [];
    if (!frames.length || frames.some((f) => !viewedFramesRef.current[f.sourceSha256] || (previewFailuresRef.current[f.sourceSha256] && decisionsRef.current[f.sourceSha256] !== "DROP"))) return;
    const next = { ...confirmedChannelsRef.current, [channelId]: true as const };
    confirmedChannelsRef.current = next; setConfirmedChannels(next);
    const pending = session?.manifest.channels.find((c) => !next[c.channelId]);
    if (pending) setBlinkChannel(pending.channelId);
  };
  // Every decision change goes through here so it can be undone (bounded stack).
  const commitDecisions = (update: (current: Record<string, BlinkDecision>) => Record<string, BlinkDecision> | undefined) => {
    const current = decisionsRef.current;
    const next = update(current);
    if (!next || next === current) return;
    invalidateReviewedChannels(Object.keys(next).filter((digest) => next[digest] !== current[digest]));
    decisionsRef.current = next; setDecisions(next);
    const history = [...decisionHistoryRef.current, current].slice(-DECISION_HISTORY_LIMIT);
    decisionHistoryRef.current = history; setDecisionHistory(history);
  };
  const setDecision = (sourceSha256: string, decision: BlinkDecision) => {
    commitDecisions((current) => !(sourceSha256 in current) || current[sourceSha256] === decision ? undefined : { ...current, [sourceSha256]: decision });
    // An unreadable preview can only leave the review through an explicit drop.
    if (decision === "DROP" && previewFailuresRef.current[sourceSha256]) markFrameViewed(sourceSha256);
  };
  const toggleDecision = (sourceSha256: string) => { if (sourceSha256 in decisionsRef.current) setDecision(sourceSha256, decisionsRef.current[sourceSha256] === "KEEP" ? "DROP" : "KEEP"); };
  // A filmstrip range (shift-click): one undo step for the whole range.
  const setDecisionsBulk = (sourceSha256s: string[], decision: BlinkDecision) => commitDecisions((current) => {
    const changed = sourceSha256s.filter((digest) => digest in current && current[digest] !== decision);
    return changed.length ? { ...current, ...Object.fromEntries(changed.map((digest) => [digest, decision])) } : undefined;
  });
  const setNightDecision = (channelId: string, night: string, decision: BlinkDecision) => commitDecisions((current) => {
    const frames = (blinkSessionRef.current?.manifest.frames ?? []).filter((frame) => frame.channelId === channelId && frame.night === night && current[frame.sourceSha256] !== decision);
    return frames.length ? { ...current, ...Object.fromEntries(frames.map((frame) => [frame.sourceSha256, decision])) } : undefined;
  });
  const undo = () => {
    const history = decisionHistoryRef.current;
    if (!history.length) return;
    const previous = history[history.length - 1];
    decisionHistoryRef.current = history.slice(0, -1); setDecisionHistory(decisionHistoryRef.current);
    invalidateReviewedChannels(Object.keys(previous).filter((digest) => previous[digest] !== decisionsRef.current[digest]));
    decisionsRef.current = previous; setDecisions(previous);
  };
  const selectBlinkChannel = (channelId: string) => { if (blinkSessionRef.current?.manifest.channels.some((channel) => channel.channelId === channelId)) setBlinkChannel(channelId); };
  // One preview of the current session as a data URL; the demo answers from its bundled manifest.
  const loadBlinkPreview = useCallback(async (relativePath: string): Promise<string> => {
    const session = blinkSessionRef.current;
    if (!session) throw new Error("no blink session");
    if (session.demo) {
      const frame = session.manifest.frames.find((item) => item.previews.zoom === relativePath || item.previews.filmstrip === relativePath);
      const dataUrl = frame?.previews.zoomDataUrl ?? frame?.previews.filmstripDataUrl;
      if (!dataUrl) throw new Error(`no demo preview for ${relativePath}`);
      return dataUrl;
    }
    return desktopBridge.loadBlinkPreview(session.sessionDirectory, relativePath);
  }, []);

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
    setJobId(undefined); setRunProgress(undefined); setExecutionMode("native"); setArtifacts([]); setScreening(undefined); setOutputDirectory(undefined); setRunStatus("RUNNING"); setRunFailureCode(undefined); setOverallProgress(0); setStages(initialStages()); setStep("run");
    // Include launch-command work, but exclude the earlier review and screening.
    startRunClock();
    // A completed Blink review decides every Light explicitly.
    const selection = selectionForRun();
    try {
      let sourceIndex = 0;
      const receipt = await desktopBridge.startRun({
        sources: sources.filter((source) => source.paths.length).flatMap((source) => source.paths.map((path) => ({ sourceId: safeSourceId(source.role, sourceIndex++), role: source.role, paths: [path], recursive: false }))),
        projectName,
        runLabel: runLabel(matrix, projectName),
        recipe: { balanced: true, drizzleEnabled, drizzleScale, drizzleDropShrink, drizzleKernel, solverRequired: true, calibrationWorkflow: "mono-standard-v1" },
        masterMetadataOverrides: masterOverrideRequests(masterOverrides),
        rawFrameMetadataOverrides: [],
        reviewSelections: [],
        ...(selection ? { selection, blinkReview: {
          sessionDirectory: blinkSession!.sessionDirectory,
          manifestSha256: blinkSession!.manifest.manifestSha256!,
          reviewedSourceSha256s: Object.keys(viewedFramesRef.current),
          confirmedChannelIds: Object.keys(confirmedChannelsRef.current),
        } } : {}),
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

  const inputBusy = inventoryBusy || qualityBusy || blinkBusy;
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
  const solveFieldReady = solverScienceReady(solveField);
  const astapReady = solverScienceReady(astap);
  // solve-field has no native Windows build; there ASTAP, verified by the
  // engine against the managed indexes, is the solver users install first.
  const primarySolver: SolverBackendId = capabilities?.platform === "windows" ? "astap" : "astrometry-net";
  // Any science-ready backend satisfies the recipe's `auto` solver choice;
  // the verified offline catalog is required by both.
  const solverSetupReady = Boolean(catalogDoctor?.ok && solverDoctor?.backends.some(solverScienceReady));
  const qualityReady = Boolean(qualityInspection && qualityInspection.frames.length === (sources.find((source) => source.role === "LIGHT")?.fileCount ?? 0));
  const lightSourcePaths = sources.find((source) => source.role === "LIGHT")?.paths ?? [];
  // A session is usable while it describes exactly the current Lights (the
  // browser demo's bundled session stands in for its labelled demo files).
  const blinkReady = Boolean(blinkSession && (blinkSession.demo ? demoMode && !nativeRuntime : blinkSessionCovers(blinkSession, lightSourcePaths)));
  const blinkChannels = useMemo<BlinkChannelSummary[]>(() => (blinkSession?.manifest.channels ?? []).map((channel) => {
    const frames = blinkSession!.manifest.frames.filter((frame) => frame.channelId === channel.channelId);
    return {
      ...channel, frames, total: frames.length,
      viewed: frames.filter((f) => viewedFrames[f.sourceSha256]).length,
      confirmed: Boolean(confirmedChannels[channel.channelId]),
      kept: frames.filter((frame) => decisions[frame.sourceSha256] === "KEEP").length,
      flagged: frames.filter((frame) => frame.flags.length > 0).length,
      exclude: frames.filter((frame) => frame.defaultDecision === "DROP").length,
      attention: frames.filter((frame) => frame.defaultDecision === "KEEP" && frame.flags.length > 0).length,
    };
  }), [blinkSession, decisions, viewedFrames, confirmedChannels]);
  const blinkReviewComplete = blinkReady && blinkChannels.length > 0 && blinkChannels.every((c) => c.confirmed && c.viewed === c.total)
    && !Object.keys(previewFailures).some((digest) => decisions[digest] !== "DROP");
  const selectionForRun = (): SelectionFile | undefined => {
    if (!blinkSession || !blinkReviewComplete || !nativeRuntime) return undefined;
    const { manifest } = blinkSession;
    const digest = manifest.manifestSha256 ?? "";
    // The controller binds GUI review and decisions to this measured session.
    const origin = CONTENT_DIGEST.test(digest) ? { sessionId: manifest.sessionId, blinkManifestSha256: digest, flagsPolicyDigest: manifest.flagsPolicyDigest, createdAt: new Date().toISOString() } : undefined;
    return {
      schemaVersion: 1, kind: "ultra-fast-wbpp-selection", policy: "explicit-v1", ...(origin ? { origin } : {}), undecided: "ERROR",
      decisions: manifest.frames.map((frame) => ({ sourceSha256: frame.sourceSha256, decision: decisions[frame.sourceSha256] ?? "KEEP", defaultDecision: frame.defaultDecision, flags: frame.flags.map((item) => item.code) })),
    };
  };
  const blinkAdmittedPaths = useMemo(() => blinkReady && nativeRuntime ? new Set(blinkSession!.manifest.frames.filter((frame) => decisions[frame.sourceSha256] === "KEEP").map((frame) => frame.path)) : undefined, [blinkReady, blinkSession, decisions, nativeRuntime]);
  const admittedPaths = useMemo(() => blinkAdmittedPaths ?? new Set(lightSourcePaths), [blinkAdmittedPaths, lightSourcePaths]);
  const matrix = useMemo(() => panelMatrix(assets, admittedPaths), [assets, admittedPaths]);
  const minimumAdmittedLights = 2;
  // Blink channels with fewer kept Lights than a stack needs (the launch bar's blocker).
  const insufficientBlinkPanels = blinkReady ? blinkChannels.filter((channel) => channel.kept < MINIMUM_KEPT_PER_CHANNEL) : [];
  // Whether the admitted counts are known: blink decisions or the legacy screening.
  const admissionKnown = Boolean(blinkAdmittedPaths);
  // Before manual review, only the imported counts are known.
  const insufficientPanels = blinkAdmittedPaths ? matrix.filter((cell) => cell.admittedCount < minimumAdmittedLights || insufficientBlinkPanels.some((channel) => channel.target === cell.target && channel.filter === cell.filter))
    : matrix.filter((cell) => cell.lightCount < minimumAdmittedLights);
  const panelsReady = matrix.length > 0 && insufficientPanels.length === 0;
  const runNavigationLocked = runLaunchBusy || runStatus === "RUNNING" || runStatus === "CANCELLING";
  const canStart = demoMode && !nativeRuntime ? !runNavigationLocked : Boolean(!runNavigationLocked && !inputBusy && nativeRuntime && capabilities?.available && outputParent && calibrationReady && allRequiredConfirmed && masterOverridesReady && cfaBlockedAssets.length === 0 && solverSetupReady && panelsReady && blinkReviewComplete);
  const importedTotal = useMemo(() => sources.reduce((sum, source) => sum + source.fileCount, 0), [sources]);
  const firstSolved = artifacts.find((artifact) => artifact.receipt?.astrometry)?.receipt?.astrometry;

  return {
    step, setStep, sources, assets, selectedRole, setSelectedRole, isDragging, setDragging, gate, capabilities, runStatus, runLaunchBusy, runElapsedSeconds, stages, overallProgress, runProgress, executionMode, artifacts, screening,
    importPaths, confirmRole, loadDemo, clearSources, runInspection, startRun, cancelRun, canInspect, allRequiredConfirmed, importedTotal, nativeRuntime,
    browserDemoAvailable: !nativeRuntime, pickFiles, pickDirectories, chooseOutputParent, useOutputParentPath, outputParent, outputDirectory, canStart, inventoryBusy, inputBusy, errorMessage, runFailureCode, calibrationReady,
    firstSolved, demoMode, projectName, matrix, masterOverrides, updateMasterOverride, resetMasterOverride, confirmMasterOverride, masterOverridesReady, runNavigationLocked,
    cfaBlockedAssets, cfaAssets, cfaPattern,
    qualityInspection, qualityBusy, qualityElapsedSeconds, qualityReady,
    insufficientPanels, minimumAdmittedLights, admissionKnown,
    blinkSession, blinkBusy, blinkElapsedSeconds, blinkReady, blinkChannels, blinkChannel, selectBlinkChannel, decisions, canUndo: decisionHistory.length > 0,
    runBlink, setDecision, toggleDecision, setDecisionsBulk, setNightDecision, undo, selectionForRun,
    viewedFrames, confirmedChannels, previewFailures, markFrameViewed, reportPreviewFailure, confirmBlinkChannel, blinkReviewComplete, insufficientBlinkPanels, minimumKeptPerChannel: MINIMUM_KEPT_PER_CHANNEL, loadBlinkPreview,
    calibrationInspection, calibrationBusy, calibrationError, recheckCalibration,
    drizzleEnabled, setDrizzleEnabled, drizzleScale, setDrizzleScale, drizzleDropShrink, setDrizzleDropShrink, drizzleKernel, setDrizzleKernel,
    catalogList, catalogDoctor, solverDoctor, recommendedCatalog, solveField, astap, solveFieldReady, astapReady, primarySolver, solverSetupReady, catalogTermsAccepted, setCatalogTermsAccepted,
    catalogProgress, catalogStatus, catalogError, startCatalogInstall, cancelCatalogInstall, openCatalogTerms, revealOutput, solverSetupBusy, recheckSolverSetup,
  };
}
