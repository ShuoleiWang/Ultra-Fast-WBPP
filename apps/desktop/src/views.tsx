import { convertFileSrc } from "@tauri-apps/api/core";
import { useState } from "react";
import type { Translator } from "./i18n";
import { CalibrationGroups, FrameInventory } from "./FrameInventory";
import type { InventoryTab } from "./Sidebar";
import { dispositionLabel } from "./Inspector";
import { BrandMark, CheckIcon, CpuIcon, FileIcon, FolderIcon, PlayIcon, RevealIcon, SparkIcon, XIcon } from "./icons";
import { DRIZZLE_KERNELS, DRIZZLE_SCALES } from "./types";
import type { DrizzleKernel, DrizzleScale, GateDisposition, InspectedLightQuality, MasterMetadataOverride, OutputArtifact, OutputArtifactKind, ScreeningSummary, SolverBackendStatus } from "./types";
import type { useWorkflow } from "./useWorkflow";

export type Workflow = ReturnType<typeof useWorkflow>;
export const PASS_FRAME_BATCH = 50;

/** A deterministic star field (SVG) for hero panels; decorative only. */
export function StarField({ count = 110, seed = 3, className }: { count?: number; seed?: number; className?: string }) {
  let state = seed * 2654435761 >>> 0;
  const next = () => { state = (state * 1664525 + 1013904223) >>> 0; return state / 4294967296; };
  const stars = Array.from({ length: count }, (_, index) => { const bright = next(); return { key: index, x: next() * 100, y: next() * 60, r: 0.05 + bright * bright * 0.22, o: 0.3 + bright * 0.7 }; });
  return <svg className={className} viewBox="0 0 100 60" preserveAspectRatio="xMidYMid slice" aria-hidden="true">{stars.map((star) => <circle key={star.key} cx={star.x} cy={star.y} r={star.r} fill="#fff" opacity={star.o} />)}</svg>;
}

const FILTER_TILES: Record<string, string> = { L: "tile-l", LUM: "tile-l", R: "tile-r", RED: "tile-r", G: "tile-g", GREEN: "tile-g", B: "tile-b", BLUE: "tile-b", HA: "tile-ha", "H-ALPHA": "tile-ha", HALPHA: "tile-ha", OIII: "tile-oiii", O3: "tile-oiii", SII: "tile-sii", S2: "tile-sii" };
export const filterTileClass = (filter: string | undefined | null) => FILTER_TILES[(filter ?? "").trim().toUpperCase()] ?? "tile-default";

/** A frame tile: the preview when there is one, otherwise the filter's colour and letter. */
export function FrameTile({ filter, preview, alt, large = false }: { filter?: string | null; preview?: string | null; alt?: string; large?: boolean }) {
  const label = (filter ?? "").trim();
  if (preview) return <img className={large ? "preview" : "thumb"} src={preview} alt={alt ?? ""} />;
  return <span className={`${large ? "preview" : "thumb"} tile ${filterTileClass(filter)}`} role={alt ? "img" : undefined} aria-label={alt} aria-hidden={alt ? undefined : true}><span>{label && label.length <= 4 && label !== "UNKNOWN" ? label : ""}</span></span>;
}
export const basename = (path: string) => path.split(/[\\/]/).filter(Boolean).pop() ?? path;
const formatBytes = (bytes: number, t: Translator) => bytes ? `${(bytes / 1_000_000).toFixed(0)} MB` : t("externallyManaged");
const optionalNumber = (value: string) => value.trim() === "" ? null : Number(value);
const formatElapsed = (seconds: number) => [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60].map((value) => String(value).padStart(2, "0")).join(":");
const dispositionClass = (disposition: GateDisposition) => disposition.toLowerCase().replace("_", "-");

function artifactLabel(kind: OutputArtifactKind, t: Translator): string {
  return {
    SOLVED_MONO_FITS: t("artifactSolvedMono"), LINEAR_RGB_FITS: t("artifactLinearRgb"), RGB_PREVIEW_TIFF_16: t("artifactTiff"), RGB_PREVIEW_PNG_16: t("artifactPng"), MONO_PREVIEW_PNG: t("artifactMonoPreview"), RECEIPT: t("artifactReceipt"), REPORT: t("artifactReport"), DRIZZLE_DATA: t("artifactDrizzle"), MASTER: t("artifactMaster"), PREVIEW: t("artifactPreview"),
  }[kind];
}

export function stageLabel(stageId: string, t: Translator): string {
  return {
    "quality-control": t("stageQuality"), calibrate: t("stageCalibration"), register: t("stageRegistration"), "local-normalization": t("stageLocalNormalization"), integrate: t("stageIntegration"), drizzle: t("stageDrizzle"), solve: t("stageSolve"), mosaic: t("stageMosaic"), color: t("stageColor"), prepare: t("stagePrepare"), alignment: t("stageAlignment"), preview: t("stagePreview"), verify: t("stageVerify"), publish: t("stagePublish"),
  }[stageId] ?? stageId;
}

export function RunElapsed({ seconds, label }: { seconds: number; label: string }) {
  return <div className="run-elapsed"><span>{label}</span><output role="timer" aria-label={label}>{formatElapsed(seconds)}</output></div>;
}

export function Alert({ title, tone = "warn", children }: { title: string; tone?: "warn" | "stop" | "info"; children: React.ReactNode }) {
  return <aside className={`callout ${tone}`} role={tone === "info" ? "note" : "alert"}><span className="symbol">{tone === "info" ? "i" : "!"}</span><div><strong>{title}</strong><p>{children}</p></div></aside>;
}

/** Monochrome defaults, the CFA block and the advanced master settings. */
export function MetadataConfirmations({ workflow, t }: { workflow: Workflow; t: Translator }) {
  if (!workflow.assets.length && !workflow.masterOverrides.length) return null;
  return <div id="metadata-confirmations" className="stack" style={{ padding: 0 }}>
    {workflow.assets.length > 0 && <p className="status-note">{t("monoStandardDefaults")}</p>}
    {workflow.cfaAssets.length > 0 && <Alert tone="info" title={t("cfaDetectedTitle")}>{t("cfaDetectedBody", { count: workflow.cfaAssets.length, pattern: workflow.cfaPattern ?? "" })}</Alert>}
    {workflow.cfaBlockedAssets.length > 0 && <Alert tone="stop" title={t("cfaSelectedBlocked")}>{t("blockerCfaUnknownPattern")}</Alert>}
    {workflow.masterOverrides.length > 0 && <div className="panel"><details className="disclosure master-section"><summary><span>{t("masterTitle")}</span><small>{t("optionalOverrides")}</small></summary><div className="details-content"><p className="hint">{t("masterBody")}</p><div className="master-form-grid">{workflow.masterOverrides.map((item) => <MasterForm key={item.sourceSha256} item={item} update={(patch) => workflow.updateMasterOverride(item.sourceSha256, patch)} confirm={() => workflow.confirmMasterOverride(item.sourceSha256)} reset={() => workflow.resetMasterOverride(item.sourceSha256)} t={t} />)}</div></div></details></div>}
  </div>;
}

/** Drop shrink is typed freely and clamped to the engine's [0.1, 1] range when the field is left. */
function DropShrinkField({ value, disabled, onCommit, t }: { value: number; disabled: boolean; onCommit: (value: number) => void; t: Translator }) {
  const [text, setText] = useState(String(value));
  const commit = () => {
    const parsed = Number(text);
    const clamped = Number.isFinite(parsed) ? Math.min(1, Math.max(0.1, Math.round(parsed * 100) / 100)) : value;
    onCommit(clamped);
    setText(String(clamped));
  };
  return <div className="field-row"><label htmlFor="drizzle-drop-shrink">{t("drizzleDropShrink")}</label><input id="drizzle-drop-shrink" type="number" inputMode="decimal" min={0.1} max={1} step={0.05} value={text} disabled={disabled} onChange={(event) => setText(event.target.value)} onBlur={commit} onKeyDown={(event) => { if (event.key === "Enter") commit(); }} /></div>;
}

/** Output path entry, the advanced options with solver setup, and what still blocks a start. */
export function LaunchSettings({ mode, workflow, outputPathText, setOutputPathText, t }: { mode: "import" | "inspect"; workflow: Workflow; outputPathText: string; setOutputPathText: (value: string) => void; t: Translator }) {
  const catalogFraction = workflow.catalogProgress?.sizeBytes ? Math.min(1, workflow.catalogProgress.downloadedBytes / workflow.catalogProgress.sizeBytes) : 0;
  return <div className="stack" style={{ padding: 0 }}>
    <div className="panel">
      {workflow.nativeRuntime && <details className="disclosure"><summary><span>{t("enterOutputPath")}</span></summary><form className="details-content" onSubmit={(event) => { event.preventDefault(); workflow.useOutputParentPath(outputPathText); }}><div className="field-row"><label htmlFor="output-parent-path">{t("outputPathLabel")}</label><input id="output-parent-path" value={outputPathText} disabled={workflow.runNavigationLocked} spellCheck={false} aria-describedby="output-path-hint" onChange={(event) => setOutputPathText(event.target.value)} /><p id="output-path-hint" className="hint">{t("outputPathHint")}</p></div><div className="row-actions"><button type="submit" className="btn small" disabled={workflow.runNavigationLocked || !outputPathText.trim()}>{t("useOutputPath")}</button></div></form></details>}
      <details className="disclosure" open={!workflow.solverSetupReady && workflow.nativeRuntime}><summary><span>{t("advanced")}</span><small>Drizzle · LN · WCS</small></summary><div className="details-content">
        <label className={`toggle ${workflow.drizzleEnabled ? "enabled" : ""}`}><input type="checkbox" checked={workflow.drizzleEnabled} disabled={!workflow.capabilities?.drizzleAvailable} onChange={(event) => workflow.setDrizzleEnabled(event.target.checked)} /><span><strong>{t("drizzle")}</strong><small>{t("drizzleHint")}</small></span></label>
        {workflow.drizzleEnabled && <div className="option-grid" aria-label={t("drizzleOptions")}>
          <div className="field-row"><label htmlFor="drizzle-scale">{t("drizzleScale")}</label><select id="drizzle-scale" value={workflow.drizzleScale} disabled={workflow.runNavigationLocked} onChange={(event) => workflow.setDrizzleScale(Number(event.target.value) as DrizzleScale)}>{DRIZZLE_SCALES.map((scale) => <option key={scale} value={scale}>{scale}×</option>)}</select></div>
          <div className="field-row"><label htmlFor="drizzle-kernel">{t("drizzleKernel")}</label><select id="drizzle-kernel" value={workflow.drizzleKernel} disabled={workflow.runNavigationLocked} onChange={(event) => workflow.setDrizzleKernel(event.target.value as DrizzleKernel)}>{DRIZZLE_KERNELS.map((kernel) => <option key={kernel} value={kernel}>{t(`drizzleKernel_${kernel}` as const)}</option>)}</select></div>
          <DropShrinkField value={workflow.drizzleDropShrink} disabled={workflow.runNavigationLocked} onCommit={workflow.setDrizzleDropShrink} t={t} />
        </div>}
        <label className={`toggle ${workflow.localNormalizationEnabled ? "enabled" : ""}`}><input type="checkbox" checked={workflow.localNormalizationEnabled} onChange={(event) => workflow.setLocalNormalizationEnabled(event.target.checked)} /><span><strong>{t("localNormalization")}</strong><small>{t("localNormalizationHint")}</small></span></label>
        <section className="panel" aria-labelledby="solver-title"><div className="panel-heading"><span id="solver-title">{t("solverSetup")}</span><div className="setup-actions"><small>{workflow.solverSetupReady ? t("ready") : t("actionRequired")}</small>{workflow.nativeRuntime && <button type="button" className="btn small" disabled={workflow.solverSetupBusy || workflow.catalogStatus === "DOWNLOADING" || workflow.catalogStatus === "VERIFYING"} onClick={() => void workflow.recheckSolverSetup()}>{workflow.solverSetupBusy ? t("checkingSetup") : t("recheckSetup")}</button>}</div></div>
          <div className="panel-body">
            <SolverRows workflow={workflow} t={t} />
            {!workflow.nativeRuntime ? <p className="mock-notice">{t("browserNoSolver")}</p> : workflow.catalogDoctor?.ok ? <div className="catalog-ready"><CheckIcon /><div><strong>{t("catalogReady")}</strong><p>{workflow.catalogDoctor.catalogRoot}</p></div></div> : workflow.recommendedCatalog ? <div className="catalog-install"><div><strong>{t("catalogInstallTitle")}</strong><p className="hint">{t("catalogInstallBody")}</p></div><div className="terms-box"><p>{workflow.recommendedCatalog.providerTerms.summary}</p><a href={workflow.recommendedCatalog.providerTerms.url} onClick={(event) => { event.preventDefault(); void workflow.openCatalogTerms(workflow.recommendedCatalog!.providerTerms.url); }}>{t("viewTerms")}</a><label><input type="checkbox" checked={workflow.catalogTermsAccepted} disabled={workflow.catalogStatus === "DOWNLOADING" || workflow.catalogStatus === "VERIFYING"} onChange={(event) => workflow.setCatalogTermsAccepted(event.target.checked)} />{t("acceptTerms", { id: workflow.recommendedCatalog.providerTerms.acceptanceId })}</label></div>{workflow.catalogStatus === "DOWNLOADING" && <div className="catalog-progress" role="status"><progress max={1} value={catalogFraction} /><span>{workflow.catalogProgress ? `${workflow.catalogProgress.artifactId} · ${(catalogFraction * 100).toFixed(1)}%` : t("waitProgress")}</span><button type="button" className="btn small" onClick={() => void workflow.cancelCatalogInstall()}>{t("cancelDownload")}</button></div>}{workflow.catalogStatus === "VERIFYING" && <p role="status" className="status-note">{t("verifyingCatalog")}</p>}{workflow.catalogStatus !== "DOWNLOADING" && workflow.catalogStatus !== "VERIFYING" && <div className="row-actions"><button type="button" className="btn primary" disabled={!workflow.catalogTermsAccepted} onClick={() => void workflow.startCatalogInstall()}>{t("downloadCatalog", { size: formatBytes(workflow.recommendedCatalog.totalSizeBytes, t) })}</button></div>}</div> : <p className="error-text" role="alert">{t("noCatalogManifest")}</p>}
            {workflow.catalogError && <p className="error-text" role="alert">{workflow.catalogError}</p>}
          </div>
        </section>
        <div className="compute" aria-label={t("computeBackend")}><CpuIcon /><strong>{workflow.capabilities?.chip ?? t("detectingHardware")}</strong><span>{workflow.capabilities?.cpuBackend ?? "Portable CPU"} · {workflow.capabilities?.gpuBackend ?? "GPU probe"} · {computeLabel(workflow.capabilities, t)}</span></div>
      </div></details>
    </div>
    {!workflow.canStart && !workflow.demoMode && <div className="blockers" role="status"><strong>{t("blockers")}</strong><ul>{!workflow.capabilities?.available && <li>{workflow.capabilities?.unavailableReason ?? t("blockerEngine")}</li>}{!workflow.calibrationReady && <li>{t("blockerCalibration")}</li>}{!workflow.allRequiredConfirmed && <li>{t("blockerTypes")}</li>}{workflow.matrix.length === 0 && <li>{t("blockerLights")}</li>}{workflow.insufficientPanels.map((cell) => <li key={cell.panelId}>{t("insufficientPanelLights", { target: cell.target, filter: cell.filter, count: workflow.qualityReady ? cell.admittedCount : cell.lightCount, required: workflow.minimumAdmittedLights })}</li>)}{!workflow.solverSetupReady && <li>{t("blockerSolver")}</li>}{!workflow.outputParent && <li>{t("blockerOutput")}</li>}{!workflow.masterOverridesReady && <li>{t("blockerMaster")}</li>}{workflow.cfaBlockedAssets.length > 0 && <li>{t("blockerCfaUnknownPattern")}</li>}</ul>{workflow.qualityReady && workflow.insufficientPanels.length > 0 && mode !== "inspect" && <button type="button" className="btn small" onClick={() => workflow.setStep("inspect")}>{t("reviewLightAdmission")}</button>}</div>}
  </div>;
}

/** The validated path this build runs on, as the compute line names it. */
function computeLabel(capabilities: Workflow["capabilities"], t: Translator): string {
  if (capabilities?.platform === "browser") return t("demoNoCompute");
  if (capabilities?.platform === "windows") return capabilities.optimizationTier === "WINDOWS_X64" ? t("windowsValidated") : t("windowsUnsupported");
  return capabilities?.optimizationTier === "M3_PRO_TUNED" ? t("m3Optimized") : t("appleGeneric");
}

/**
 * The two solvers, the platform's primary one first with its install
 * instruction: solve-field on macOS, ASTAP (verified by the engine against the
 * managed indexes) on Windows, where solve-field has no native build.
 */
function SolverRows({ workflow, t }: { workflow: Workflow; t: Translator }) {
  const windows = workflow.primarySolver === "astap";
  const solveField = <SolverRow key="solve-field" name="solve-field" backend={workflow.solveField} ready={workflow.solveFieldReady} instruction={t(windows ? "solveFieldOptional" : "solveFieldInstall")} t={t} />;
  const astap = <SolverRow key="astap" name="ASTAP" backend={workflow.astap} ready={workflow.astapReady} instruction={t(windows ? "astapInstall" : "astapAlternative")} t={t} />;
  return <>{windows ? [astap, solveField] : [solveField, astap]}</>;
}

/** What the run will leave out, said before the start button is pressed. */
function ScreeningNotice({ mode, workflow, t }: { mode: "import" | "inspect"; workflow: Workflow; t: Translator }) {
  if (workflow.demoMode) return null;
  if (!workflow.qualityReady) {
    if (!workflow.canStart) return null;
    return <div className="screening-notice" role="status"><span className="symbol" aria-hidden="true">!</span><div><strong>{t("noticeUnscreenedTitle")}</strong><span>{t("noticeUnscreenedBody")}</span></div><button type="button" className="btn small" disabled={!workflow.canInspect || !workflow.allRequiredConfirmed || workflow.qualityBusy} onClick={() => void workflow.runInspection()}>{workflow.qualityBusy ? t("inspecting") : t("inspect", { count: "" })}</button></div>;
  }
  if (workflow.pendingReviewCount === 0) return null;
  return <div className="screening-notice" role="status"><span className="symbol" aria-hidden="true">!</span><div><strong>{t("noticePendingReviewTitle", { count: workflow.pendingReviewCount })}</strong><span>{t("noticePendingReviewBody", { admitted: workflow.matrix.reduce((sum, cell) => sum + cell.admittedCount, 0), total: workflow.matrix.reduce((sum, cell) => sum + cell.lightCount, 0) })}</span></div>{workflow.unapprovedApprovableCount > 0 && <button type="button" className="btn small" onClick={() => workflow.setAllReviewApprovals(true)}>{t("approveAllReview", { count: workflow.approvableReviewCount })}</button>}{mode === "import" && <button type="button" className="btn small quiet" onClick={() => workflow.setStep("inspect")}>{t("viewQualityResults")}</button>}</div>;
}

export function LaunchBar({ mode, workflow, lightCount, t }: { mode: "import" | "inspect"; workflow: Workflow; lightCount: number; t: Translator }) {
  return <footer className="launchbar">
    <ScreeningNotice mode={mode} workflow={workflow} t={t} />
    <div className="output"><FolderIcon /><div style={{ minWidth: 0 }}><small>{t("outputLabel")} · </small><strong title={workflow.outputParent}>{workflow.outputParent ?? (workflow.demoMode ? t("browserNoFiles") : t("outputNotSelected"))}</strong></div><button type="button" className="btn small" disabled={!workflow.nativeRuntime} onClick={() => void workflow.chooseOutputParent()}>{t("chooseOutput")}</button></div>
    <div className="actions">
      {mode === "inspect"
        ? <button type="button" className="btn quiet" onClick={() => workflow.setStep("import")}>{t("back")}</button>
        : <>{workflow.browserDemoAvailable && <button type="button" className="btn quiet" onClick={workflow.loadDemo}>{t("loadDemo")}</button>}<button type="button" className="btn" disabled={!workflow.canInspect || !workflow.allRequiredConfirmed || workflow.qualityBusy} onClick={() => void workflow.runInspection()}>{workflow.qualityBusy ? t("inspecting") : workflow.qualityReady ? t("viewQualityResults") : t("inspect", { count: lightCount || "" })}</button></>}
      <button type="button" className="btn primary" disabled={!workflow.canStart} onClick={() => void workflow.startRun()}><PlayIcon />{t("start")}</button>
    </div>
  </footer>;
}

export function ImportView({ workflow, t, inputPathsText, setInputPathsText, outputPathText, setOutputPathText, inventoryTab, setInventoryTab, lightCount }: {
  workflow: Workflow; t: Translator; inputPathsText: string; setInputPathsText: (value: string) => void; outputPathText: string; setOutputPathText: (value: string) => void;
  inventoryTab: InventoryTab; setInventoryTab: (tab: InventoryTab) => void; lightCount: number;
}) {
  const pathEntry = workflow.nativeRuntime && <div className="panel"><details className="disclosure"><summary><span>{t("pasteInputPaths")}</span></summary><form className="details-content" onSubmit={(event) => { event.preventDefault(); const paths = [...new Set(inputPathsText.split(/\r?\n/).map((path) => path.trim()).filter(Boolean))]; void workflow.importPaths(paths, workflow.selectedRole); }}><div className="field-row"><label htmlFor="input-paths">{t("inputPathsLabel")}</label><textarea id="input-paths" rows={4} value={inputPathsText} disabled={workflow.inputBusy} spellCheck={false} aria-describedby="input-paths-hint" onChange={(event) => setInputPathsText(event.target.value)} /><p id="input-paths-hint" className="hint">{t("inputPathsHint")}</p></div><div className="row-actions"><button type="submit" className="btn small" disabled={workflow.inputBusy || !inputPathsText.trim()}>{t("importEnteredPaths")}</button></div></form></details></div>;
  const empty = workflow.assets.length === 0;
  return <section className="view" aria-labelledby="import-title">
    {empty
      ? <div className="scroll"><div className="hero-page">
          <div className="hero">
            <StarField className="hero-sky" />
            <div className="hero-glow" aria-hidden="true" />
            <div className="hero-body">
              <span className="symbol"><BrandMark size={88} /></span>
              <h1 id="import-title">{t("importTitle")}</h1>
              <p className="hero-lead">{workflow.inventoryBusy ? t("readingMetadata") : workflow.nativeRuntime ? t("dropNative") : t("dropBrowser")}</p>
              <p className="hero-sub">{t("importDescription")}</p>
              <div className="hero-hints"><span className="hero-hint">{workflow.selectedRole ? t("nextBatch", { role: workflow.selectedRole }) : t("autoRead")}</span>{workflow.selectedRole && <button type="button" className="link" disabled={workflow.inputBusy} onClick={() => workflow.setSelectedRole(undefined)}>{t("autoDetect")}</button>}</div>
            </div>
          </div>
          <div className="stack hero-stack">
            {workflow.errorMessage && <Alert tone="stop" title={t("operationFailed")}>{workflow.errorMessage}</Alert>}
            {pathEntry}
            <FrameInventory workflow={workflow} t={t} tab={inventoryTab} setTab={setInventoryTab} />
          </div>
        </div></div>
      : <div className="scroll"><div className="stack">
          <h1 id="import-title" className="visually-hidden" style={{ position: "absolute", width: 1, height: 1, overflow: "hidden", clip: "rect(0 0 0 0)" }}>{workflow.projectName}</h1>
          {workflow.errorMessage && <Alert tone="stop" title={t("operationFailed")}>{workflow.errorMessage}</Alert>}
          <div className="row-actions"><span className="status-note">{workflow.inventoryBusy ? t("readingMetadata") : workflow.selectedRole ? t("nextBatch", { role: workflow.selectedRole }) : t("autoRead")}</span>{workflow.selectedRole && <button type="button" className="link" disabled={workflow.inputBusy} onClick={() => workflow.setSelectedRole(undefined)}>{t("autoDetect")}</button>}</div>
          <FrameInventory workflow={workflow} t={t} tab={inventoryTab} setTab={setInventoryTab} />
          {pathEntry}
          <CalibrationGroups workflow={workflow} t={t} />
          <MetadataConfirmations workflow={workflow} t={t} />
          <LaunchSettings mode="import" workflow={workflow} outputPathText={outputPathText} setOutputPathText={setOutputPathText} t={t} />
        </div></div>}
    <LaunchBar mode="import" workflow={workflow} lightCount={lightCount} t={t} />
  </section>;
}

export function ScreeningView({ workflow, t, outputPathText, setOutputPathText, selectedPath, setSelectedPath, visiblePassFrames, setVisiblePassFrames, decisionFilter, setDecisionFilter, lightCount }: {
  workflow: Workflow; t: Translator; outputPathText: string; setOutputPathText: (value: string) => void;
  selectedPath?: string; setSelectedPath: (path?: string) => void; visiblePassFrames: number; setVisiblePassFrames: (update: (current: number) => number) => void;
  decisionFilter: "ALL" | GateDisposition; setDecisionFilter: (filter: "ALL" | GateDisposition) => void; lightCount: number;
}) {
  const targets = [...new Set(workflow.matrix.map((cell) => cell.target))];
  const filters = [...new Set(workflow.matrix.map((cell) => cell.filter))];
  const frames = workflow.qualityInspection?.frames ?? [];
  const filterByPath = new Map(workflow.assets.map((asset) => [asset.path, asset.filter] as const));
  const filterOf = (path: string) => filterByPath.get(path);
  const issueFrames = frames.filter((frame) => frame.disposition !== "PASS");
  const passingFrames = frames.filter((frame) => frame.disposition === "PASS");
  const ordered = decisionFilter === "ALL" ? [...issueFrames, ...passingFrames.slice(0, visiblePassFrames)] : decisionFilter === "PASS" ? passingFrames.slice(0, visiblePassFrames) : frames.filter((frame) => frame.disposition === decisionFilter);
  const total = workflow.gate.pass + workflow.gate.review + workflow.gate.hardFail;
  const chips: Array<{ key: "ALL" | GateDisposition; label: string; count: number; dot?: string }> = [
    { key: "ALL", label: t("allFrames"), count: total }, { key: "PASS", label: t("dispositionPass"), count: workflow.gate.pass, dot: "ok" },
    { key: "REVIEW", label: t("dispositionReview"), count: workflow.gate.review, dot: "warn" }, { key: "HARD_FAIL", label: t("dispositionFail"), count: workflow.gate.hardFail, dot: "stop" },
  ];
  return <section className="view" aria-labelledby="inspect-title">
    <div className="chead">
      <h1 id="inspect-title">{t("reviewTitle")}</h1>
      <div className="chead-row">
        <div className="chips" role="group" aria-label={t("colDecision")}>{chips.map((chip) => <button type="button" key={chip.key} className="chip" aria-pressed={decisionFilter === chip.key} onClick={() => setDecisionFilter(chip.key)}>{chip.dot && <span className={`dot ${chip.dot}`} />}{chip.label} <span className="n">{chip.count}</span></button>)}</div>
        <span className="spacer" />
        {workflow.approvableReviewCount > 0 && (workflow.unapprovedApprovableCount > 0
          ? <button type="button" className="btn small" onClick={() => workflow.setAllReviewApprovals(true)}>{t("approveAllReview", { count: workflow.approvableReviewCount })}</button>
          : <button type="button" className="btn small quiet" onClick={() => workflow.setAllReviewApprovals(false)}>{t("clearReviewApprovals")}</button>)}
        <span className="muted">{workflow.demoMode ? t("demoReviewDescription") : workflow.qualityInspection ? `${t("failClosed")} · POLICY ${workflow.qualityInspection.gatePolicyDigest.slice(7, 19)}` : t("reviewDescription")}</span>
      </div>
    </div>
    <div className="scroll"><div className="stack">
      {workflow.demoMode && <Alert title={t("demoWarningTitle")}>{t("demoWarningBody")}</Alert>}
      {!workflow.demoMode && workflow.qualityInspection && <div className="panel">
        <div className="tbl-wrap"><table className="tbl"><thead><tr><th className="thumb-cell" aria-label={t("previewAlt", { name: "" }).trim()} /><th>{t("colFrame")}</th><th className="r">{t("colStars")}</th><th>{t("colConfidence")}</th><th>{t("colDecision")}</th></tr></thead><tbody>
          {ordered.map((frame) => { const approved = Boolean(frame.sourceSha256 && workflow.approvedReviewDigests.includes(frame.sourceSha256)); const canApprove = workflow.canApproveReview(frame); const unavailable = frame.registrable === false ? t("unregistrableReview") : t("previewUnavailable"); const selected = selectedPath === frame.path; return <tr key={frame.path} className={`quality-frame quality-${dispositionClass(frame.disposition)}`} aria-selected={selected} onClick={() => setSelectedPath(selected ? undefined : frame.path)}>
            <td className="thumb-cell"><FrameTile filter={filterOf(frame.path)} preview={frame.previewDataUrl} /></td>
            <td className="clip" title={frame.path}><strong className="frame-kind">{basename(frame.path)}</strong></td>
            <td className="r tnum">{frame.starCount}</td>
            <td><small>{frame.confidence}</small></td>
            <td className="dec-cell"><div className="dec">{frame.disposition === "PASS" ? <><span className="dot ok" /><span>{t("dispositionPass")}</span></> : <span className={`badge ${frame.disposition === "REVIEW" ? (approved ? "ok" : "check") : "stop"}`}>{frame.disposition === "REVIEW" && approved ? t("dispositionApproved") : dispositionLabel(frame.disposition, t)}</span>}<span className="why">{frame.evidence.length ? frame.evidence.map((item) => item.message).slice(0, 2).join("; ") : frame.disposition === "PASS" ? "" : t("noReviewEvidence")}</span>{frame.disposition === "REVIEW" && <button type="button" className="btn small" disabled={!canApprove} aria-pressed={approved} onClick={(event) => { event.stopPropagation(); workflow.toggleReviewApproval(frame); }}>{approved ? t("approvedReview") : canApprove ? t("approveReview") : unavailable}</button>}{frame.disposition === "HARD_FAIL" && <em>{t("cannotPromote")}</em>}</div></td>
          </tr>; })}
        </tbody></table></div>
        {ordered.length === 0 && <p className="table-empty">{t("inventoryEmpty")}</p>}
        {(decisionFilter === "ALL" || decisionFilter === "PASS") && passingFrames.length > visiblePassFrames && <div className="pass-pagination"><span>{t("passBatchStatus", { shown: Math.min(visiblePassFrames, passingFrames.length), total: passingFrames.length })}</span><button type="button" className="btn small" onClick={() => setVisiblePassFrames((current) => current + PASS_FRAME_BATCH)}>{t("showMorePass", { count: Math.min(PASS_FRAME_BATCH, passingFrames.length - visiblePassFrames) })}</button></div>}
      </div>}
      <section className="panel matrix-section" aria-labelledby="matrix-title"><div className="panel-heading"><span id="matrix-title">{t("panelMatrix")}</span><small>{t("matrixCounts", { targets: targets.length, filters: filters.length, panels: workflow.matrix.length })}</small></div>{workflow.matrix.length ? <div className="tbl-wrap"><table className="tbl compact"><thead><tr><th scope="col">{t("targetLabel")}</th>{filters.map((filter) => <th scope="col" key={filter}>{filter}</th>)}</tr></thead><tbody>{targets.map((target) => <tr key={target}><th scope="row">{target}</th>{filters.map((filter) => { const cell = workflow.matrix.find((item) => item.target === target && item.filter === filter); return <td key={filter} className={cell ? "matrix-present" : "matrix-empty"}>{cell ? workflow.qualityReady ? t("panelAdmittedCounts", { admitted: cell.admittedCount, total: cell.lightCount, required: workflow.minimumAdmittedLights }) : `${cell.lightCount} ${t("frames")}` : "—"}</td>; })}</tr>)}</tbody></table></div> : <p className="table-empty">{t("noGroups")}</p>}</section>
      <CalibrationGroups workflow={workflow} t={t} />
      <MetadataConfirmations workflow={workflow} t={t} />
      <LaunchSettings mode="inspect" workflow={workflow} outputPathText={outputPathText} setOutputPathText={setOutputPathText} t={t} />
    </div></div>
    <LaunchBar mode="inspect" workflow={workflow} lightCount={lightCount} t={t} />
  </section>;
}

/** The failure card of a run that failed closed: what failed, where the engine's evidence is, what to do. */
function RunFailure({ workflow, t }: { workflow: Workflow; t: Translator }) {
  const code = workflow.runFailureCode;
  const message = workflow.errorMessage ?? "";
  const detail = code && message.startsWith(`${code}: `) ? message.slice(code.length + 2) : message;
  const failedStage = workflow.stages.find((stage) => stage.status === "FAILED");
  const evidence = workflow.outputDirectory ? `${workflow.outputDirectory}.unsolved` : undefined;
  return <section className="run-failure" role="alert" aria-live="assertive">
    <div className="run-failure-head"><span className="symbol" aria-hidden="true">!</span><div><strong>{t("runFailureTitle")}</strong><span>{failedStage ? t("runFailureStage", { stage: stageLabel(failedStage.stageId, t) }) : t("runFailureNoStage")}</span></div></div>
    <p className="run-failure-detail">{code && <><code className="run-failure-code">{code}</code>: </>}{detail}</p>
    {code === "ASTROMETRY_REQUIRED" && <p className="run-failure-hint">{t("astrometryFailedHint")}</p>}
    {evidence && <p className="run-failure-hint">{t("unsolvedEvidenceHint")} <span className="run-failure-path" title={evidence}>{evidence}</span></p>}
  </section>;
}

export function RunView({ workflow, t }: { workflow: Workflow; t: Translator }) {
  const title = workflow.runStatus === "CANCELLED" ? t("cancelledTitle") : workflow.runStatus === "FAILED" ? t("failedTitle") : t("runningTitle");
  const stages = workflow.stages.filter((stage) => (workflow.drizzleEnabled || stage.stageId !== "drizzle") && (workflow.localNormalizationEnabled || stage.stageId !== "local-normalization"));
  return <section className="view" aria-labelledby="run-title">
    <div className="scroll">
      <div className="proc-head">
        <div className="ring-lg" aria-hidden="true"><svg viewBox="0 0 120 120"><circle className="ring-track" cx="60" cy="60" r="52" /><circle className={`ring-value ${workflow.runStatus === "FAILED" ? "failed" : workflow.runStatus === "COMPLETED" ? "done" : ""}`} cx="60" cy="60" r="52" pathLength="100" strokeDasharray={`${workflow.overallProgress} 100`} /></svg><div className="ring-label tnum">{workflow.overallProgress}%</div></div>
        <div><h1 id="run-title">{title}</h1><p>{t("runDescription")}</p><div className="row-actions" style={{ marginTop: 8 }}>{workflow.executionMode === "demo" && <span className="badge-demo">{t("demo")}</span>}<RunElapsed seconds={workflow.runElapsedSeconds} label={t(workflow.runStatus === "RUNNING" || workflow.runStatus === "CANCELLING" ? "runElapsed" : "runTotalTime")} /></div></div>
      </div>
      <p className="muted" role="status" style={{ padding: "0 24px" }}>{workflow.runProgress?.scope === "panel" ? t("currentPanelProgress", { index: workflow.runProgress.panelIndex ?? 0, count: workflow.runProgress.panelCount ?? 0, target: workflow.runProgress.panelTarget ?? "", filter: workflow.runProgress.panelFilter ?? "" }) : workflow.runProgress?.scope === "project" ? t("projectFinalStages") : t("overallProjectProgress")}{workflow.runProgress?.message && <small> · {workflow.runProgress.message}</small>}</p>
      {workflow.runStatus === "FAILED" && <RunFailure workflow={workflow} t={t} />}
      <ol className="stage-list">{stages.map((stage) => <li key={stage.stageId} className={stage.status.toLowerCase()}><span className="stage-index">{stage.status === "DONE" ? <CheckIcon /> : stage.status === "FAILED" ? <XIcon /> : stage.status === "RUNNING" ? <svg className="ring" viewBox="0 0 16 16" aria-hidden="true"><circle className="spin" cx="8" cy="8" r="5.9" /></svg> : <span className="pend" aria-hidden="true" />}</span><span>{stageLabel(stage.stageId, t)}</span><span className="mini-track"><span style={{ width: `${stage.percent}%` }} /></span><em>{stage.status === "FAILED" ? t("stageFailed") : `${Math.round(stage.percent)}%`}</em></li>)}</ol>
      {workflow.executionMode === "demo" && <p className="mock-notice" style={{ padding: "0 24px" }}><SparkIcon />{t("demoWarningBody")}</p>}
      {workflow.errorMessage && workflow.runStatus !== "FAILED" && <p className="error-text" role="alert" style={{ padding: "8px 24px" }}>{workflow.errorMessage}</p>}
      <div className="run-actions">{workflow.runStatus === "RUNNING" && <button type="button" className="btn" disabled={workflow.runLaunchBusy} onClick={() => void workflow.cancelRun()}>{t("cancel")}</button>}{["CANCELLED", "FAILED"].includes(workflow.runStatus) && <button type="button" className={workflow.runStatus === "FAILED" ? "btn primary" : "btn"} onClick={() => workflow.setStep("import")}>{t("returnConfig")}</button>}</div>
    </div>
  </section>;
}

const MASTER_KINDS: OutputArtifactKind[] = ["SOLVED_MONO_FITS", "LINEAR_RGB_FITS", "MASTER"];
const PREVIEW_KINDS: OutputArtifactKind[] = ["MONO_PREVIEW_PNG", "RGB_PREVIEW_PNG_16", "PREVIEW"];

function previewFor(artifact: OutputArtifact, artifacts: OutputArtifact[], native: boolean): string | undefined {
  const match = artifacts.find((candidate) => PREVIEW_KINDS.includes(candidate.kind) && (candidate.filter ?? "") === (artifact.filter ?? "") && (candidate.target ?? "") === (artifact.target ?? ""));
  if (!match) return undefined;
  // The controller carries the PNG previews as data URLs, which show from any
  // drive or share; the asset protocol (home folder and volumes only) is the
  // fallback for a preview it could not carry.
  if (match.previewDataUrl) return match.previewDataUrl;
  if (!native) return undefined;
  try { return convertFileSrc(match.path); } catch { return undefined; }
}

export function ResultView({ workflow, t }: { workflow: Workflow; t: Translator }) {
  const demo = workflow.executionMode === "demo";
  const masters = workflow.artifacts.filter((artifact) => MASTER_KINDS.includes(artifact.kind));
  const products = workflow.artifacts.filter((artifact) => !["RECEIPT", "REPORT"].includes(artifact.kind));
  const technical = workflow.artifacts.filter((artifact) => ["RECEIPT", "REPORT"].includes(artifact.kind));
  const heroPreview = masters.map((artifact) => previewFor(artifact, workflow.artifacts, workflow.nativeRuntime)).find(Boolean);
  return <section className="view" aria-labelledby="result-title">
    <div className="scroll">
      <div className="result-hero">
        {heroPreview ? <img className="result-hero-image" src={heroPreview} alt="" /> : <StarField className="result-hero-sky" seed={9} count={160} />}
        <div className="result-hero-shade" aria-hidden="true" />
        <div className="result-hero-body">
          <span className={`pill ${demo ? "warn" : ""}`}><CheckIcon />{demo ? t("resultDemo") : t("resultSolved")}</span>
          <h1 id="result-title">{demo ? t("resultDemoTitle") : t("resultTitle")}</h1>
          <p>{demo ? t("resultDemoBody") : t("resultBody")}</p>
          <dl className="hero-metrics">
            <div><dt>{t("runTotalTime")}</dt><dd><RunElapsed seconds={workflow.runElapsedSeconds} label={t("runTotalTime")} /></dd></div>
            {workflow.screening && <div className={workflow.screening.excluded > 0 ? "metric-warn" : ""}><dt>{t("screeningTitle")}</dt><dd className="tnum"><strong>{t("framesUsed", { admitted: workflow.screening.admitted, total: workflow.screening.admitted + workflow.screening.excluded })}</strong>{workflow.screening.excluded > 0 && <small> · {t("framesExcluded", { excluded: workflow.screening.excluded })}</small>}</dd></div>}
            {workflow.firstSolved ? <>
              <div><dt>{t("center")}</dt><dd className="tnum">{workflow.firstSolved.centerRaDegrees.toFixed(4)}° · {workflow.firstSolved.centerDecDegrees.toFixed(4)}°</dd></div>
              <div><dt>{t("pixelScale")}</dt><dd className="tnum">{workflow.firstSolved.pixelScaleArcsec.toFixed(3)}″ / px</dd></div>
              <div><dt>{t("solveQuality")}</dt><dd className="tnum">{workflow.firstSolved.rmsArcsec.toFixed(3)}″ RMS · {workflow.firstSolved.matchedStars} {t("starsLabel")}</dd></div>
            </> : <div><dt>WCS</dt><dd>{t("demoNoWcs")}</dd></div>}
          </dl>
        </div>
      </div>
      {masters.length > 0 && <div className="cards">{masters.map((artifact) => { const preview = previewFor(artifact, workflow.artifacts, workflow.nativeRuntime); return <article className="card" key={artifact.path}><div className={`shot tile ${preview ? "" : filterTileClass(artifact.filter)}`}>{preview ? <img src={preview} alt="" /> : <StarField seed={artifact.path.length} count={70} className="shot-sky" />}{artifact.filter && <span className="tag">{artifact.filter}</span>}</div><div className="card-body"><div className="card-title"><b title={artifact.path}>{artifact.name}</b>{artifact.target && <span>{artifact.target}</span>}</div><div className="metrics">{artifact.detail}</div></div></article>; })}</div>}
      <div className="stack">
        {workflow.screening && <ScreeningSection screening={workflow.screening} t={t} />}
        <ArtifactList title={t("products")} artifacts={products} workflow={workflow} t={t} />
        {technical.length > 0 && <div className="panel"><details className="disclosure"><summary><span>{t("technicalDetails")}</span><small>SHA-256 · JSON</small></summary><div className="details-content" style={{ padding: 0 }}><ArtifactList artifacts={technical} workflow={workflow} t={t} /></div></details></div>}
        {demo && <p className="mock-notice"><SparkIcon />{t("demoWarningBody")}</p>}
        <div className="row-actions" style={{ justifyContent: "space-between" }}><button type="button" className="btn" onClick={workflow.clearSources}>{t("startNew")}</button><button type="button" className="btn primary" disabled={!workflow.nativeRuntime || !workflow.outputDirectory} title={workflow.outputDirectory} onClick={() => workflow.outputDirectory && void workflow.revealOutput(workflow.outputDirectory)}><RevealIcon />{t("revealOutput", { name: workflow.outputDirectory ? basename(workflow.outputDirectory) : "—" })}</button></div>
      </div>
    </div>
  </section>;
}

/** The run's own screening: how many Lights went in and why the others did not. */
export function ScreeningSection({ screening, t }: { screening: ScreeningSummary; t: Translator }) {
  const decided = screening.frames;
  return <section className="panel screening-section" aria-labelledby="screening-title">
    <div className="panel-heading"><span id="screening-title">{t("screeningTitle")}</span><small>{t("screeningSummary", { admitted: screening.admitted, excluded: screening.excluded })}</small></div>
    {decided.length === 0 ? <p className="screening-clean">{t("screeningAllPassed")}</p> : <div className="tbl-wrap"><table className="tbl"><tbody>{decided.map((frame) => <tr key={`${frame.target ?? ""}/${frame.name}`} className={`quality-frame quality-${dispositionClass(frame.disposition)}`}><td className="thumb-cell">{frame.previewDataUrl ? <img className="thumb" src={frame.previewDataUrl} alt={t("previewAlt", { name: frame.name })} /> : <FrameTile />}</td><td className="clip" title={frame.name}><strong className="frame-kind">{frame.name}</strong><small>{frame.target ? `${frame.target} · ` : ""}{frame.starCount ?? "—"} {t("starsLabel")}</small></td><td className="dec-cell"><div className="dec"><span className={`badge ${frame.admitted ? "ok" : frame.disposition === "REVIEW" ? "check" : "stop"}`}>{frame.admitted ? t("screeningAdmittedReview") : dispositionLabel(frame.disposition, t)}</span><span className="why">{frame.evidence.length ? frame.evidence.slice(0, 2).join("; ") : frame.summary || t("noReviewEvidence")}</span></div></td></tr>)}</tbody></table></div>}
    {decided.some((frame) => frame.disposition === "REVIEW" && !frame.admitted) && <p className="screening-hint">{t("screeningReviewHint")}</p>}
  </section>;
}

function ArtifactList({ title, artifacts, workflow, t }: { title?: string; artifacts: Workflow["artifacts"]; workflow: Workflow; t: Translator }) {
  return <div className={title ? "panel" : ""}>{title && <div className="panel-heading"><span>{title}</span><small>{t("verifiedFiles", { count: artifacts.length })}</small></div>}<div className="artifact-list">{artifacts.map((artifact) => <article key={artifact.path}><span className={`artifact-kind kind-${artifact.kind.toLowerCase()}`}><FileIcon /></span><div style={{ minWidth: 0 }}><strong>{artifact.name}</strong><small><span>{artifactLabel(artifact.kind, t)}</span>{artifact.filter ? ` · ${artifact.target ?? ""} · ${artifact.filter}` : ""} · {artifact.detail}</small></div><button type="button" className="btn icon" disabled={!workflow.nativeRuntime} title={artifact.path} aria-label={t("showInFinder", { name: artifact.name })} onClick={() => void workflow.revealOutput(artifact.path)}><RevealIcon /></button></article>)}</div></div>;
}

export function MasterForm({ item, update, confirm, reset, t }: { item: MasterMetadataOverride; update: (patch: Partial<MasterMetadataOverride>) => void; confirm: () => void; reset: () => void; t: Translator }) {
  const numeric = (field: "gain" | "offset" | "temperatureCelsius" | "exposureSeconds", label: string) => <label><span>{label}</span><input type="number" step="any" inputMode="decimal" placeholder={t("notRecorded")} value={item[field] ?? ""} onChange={(event) => update({ [field]: optionalNumber(event.target.value) })} /></label>;
  const additiveMaster = item.role === "MASTER_BIAS" || item.role === "MASTER_DARK";
  const chooseNumericDomain = (value: string) => {
    if (value === "NORMALIZED_UNIT") update({ numericDomain: value, normalizedUnitScale: 1 });
    else if (value === "SENSOR_CODE") update({ numericDomain: value, normalizedUnitScale: 65535 });
    else update({ numericDomain: null, normalizedUnitScale: null });
  };
  return <article className={`master-form ${item.confirmed ? "confirmed" : ""}`}>
    <header><div><small>{item.role}</small><strong title={item.sourcePath}>{basename(item.sourcePath)}</strong></div><span>{item.confirmed ? item.needsMetadataOverride ? t("shaBound") : t("useFileMetadata") : t("needsConfirmation")}</span></header>
    {!item.confirmed && <p className="suggestion-note" role="status">{t("suggestion")}</p>}
    <div className="metadata-fields"><label><span>{t("cameraLabel")}</span><input value={item.camera} onChange={(event) => update({ camera: event.target.value })} /></label>{numeric("gain", t("gainLabel"))}{numeric("offset", t("offsetLabel"))}<label><span>{t("binningLabel")}</span><span className="inline-inputs"><input type="number" min="1" inputMode="numeric" placeholder={t("notRecorded")} value={item.binning[0] ?? ""} onChange={(event) => update({ binning: [optionalNumber(event.target.value), item.binning[1]] })} /><input type="number" min="1" inputMode="numeric" placeholder={t("notRecorded")} value={item.binning[1] ?? ""} onChange={(event) => update({ binning: [item.binning[0], optionalNumber(event.target.value)] })} /></span></label><label><span>{t("filterLabel")}</span><input value={item.filter} onChange={(event) => update({ filter: event.target.value })} /></label><label><span>{t("cfaLabel")}</span><input value={item.cfaPattern} onChange={(event) => update({ cfaPattern: event.target.value })} /></label><label><span>{t("readoutLabel")}</span><input value={item.readoutMode} onChange={(event) => update({ readoutMode: event.target.value })} /></label>{numeric("temperatureCelsius", t("temperatureLabel"))}{numeric("exposureSeconds", t("exposureLabel"))}</div>
    {item.role === "MASTER_DARK" && <fieldset className="bias-choice"><legend>{t("darkBiasQuestion")}</legend><label><input type="radio" name={`bias-${item.sourceSha256}`} checked={item.biasIncluded === true} onChange={() => update({ biasIncluded: true })} />{t("yesBias")}</label><label><input type="radio" name={`bias-${item.sourceSha256}`} checked={item.biasIncluded === false} onChange={() => update({ biasIncluded: false })} />{t("noBias")}</label><label><input type="radio" name={`bias-${item.sourceSha256}`} checked={item.biasIncluded === null} onChange={() => update({ biasIncluded: null })} />{t("unknownBlocked")}</label><p>{t("darkBiasHelp")}</p></fieldset>}
    {additiveMaster && <fieldset className="bias-choice unit-choice"><legend>{t("pixelUnits")}</legend><label><select aria-label={t("pixelUnits")} value={item.numericDomain ?? "AUTO"} onChange={(event) => chooseNumericDomain(event.target.value)}><option value="AUTO">{t("pixelUnitsAuto")}</option><option value="NORMALIZED_UNIT">{t("pixelUnitsNormalized")}</option><option value="SENSOR_CODE">{t("pixelUnitsSensor")}</option></select></label>{item.numericDomain === "SENSOR_CODE" && <label><span>{t("pixelUnitsScale")}</span><input type="number" step="any" className="unit-scale-input" inputMode="decimal" value={item.normalizedUnitScale ?? ""} onChange={(event) => update({ normalizedUnitScale: optionalNumber(event.target.value) })} /></label>}<p>{t("pixelUnitsHelp")}</p></fieldset>}
    <div className="row-actions"><button type="button" className="btn small" disabled={item.confirmed} onClick={confirm}>{t("confirmAndBind")}</button>{item.needsMetadataOverride && <button type="button" className="link" onClick={reset}>{t("restoreFileMetadata")}</button>}</div>
  </article>;
}

/** One solver: ready means the strict final gate accepts its solutions; an installed but unverified one also shows the engine's reason. */
function SolverRow({ name, backend, ready, instruction, t }: { name: string; backend?: SolverBackendStatus; ready: boolean; instruction: string; t: Translator }) {
  const reason = !ready && backend?.executionReady && backend.reason?.trim() ? ` ${backend.reason.trim()}` : "";
  return <article className="solver-row"><span className={`runtime-dot ${ready ? "online" : ""}`} /><div><strong>{name} {ready ? t("executable") : t("notReady")}</strong><p>{ready ? `${backend?.version ?? "?"} · ${backend?.metadata?.probe?.path ?? t("ready")}` : `${instruction}${reason}`}</p></div></article>;
}
