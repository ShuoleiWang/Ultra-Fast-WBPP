import { useState } from "react";
import type { Translator } from "../i18n";
import { CheckIcon, CpuIcon, FolderIcon, PlayIcon } from "../icons";
import { DRIZZLE_KERNELS, DRIZZLE_SCALES, type DrizzleKernel, type DrizzleScale } from "../types";
import type { StartBlocker } from "../useWorkflow";
import { MasterForm, SolverRow } from "./MasterForm";
import { Alert, formatBytes, type Workflow } from "./common";

/** Monochrome defaults, the CFA block and the advanced master settings. */
export function MetadataConfirmations({ workflow, t }: { workflow: Workflow; t: Translator }) {
  if (!workflow.assets.length && !workflow.masterOverrides.length) return null;
  return (
    <div id="metadata-confirmations" className="stack" style={{ padding: 0 }}>
      {workflow.assets.length > 0 && <p className="status-note">{t("monoStandardDefaults")}</p>}
      {workflow.cfaAssets.length > 0 && (
        <Alert tone="info" title={t("cfaDetectedTitle")}>
          {t("cfaDetectedBody", { count: workflow.cfaAssets.length, pattern: workflow.cfaPattern ?? "" })}
        </Alert>
      )}
      {workflow.cfaBlockedAssets.length > 0 && (
        <Alert tone="stop" title={t("cfaSelectedBlocked")}>
          {t("blockerCfaUnknownPattern")}
        </Alert>
      )}
      {workflow.masterOverrides.length > 0 && (
        <div className="panel">
          <details className="disclosure master-section">
            <summary>
              <span>{t("masterTitle")}</span>
              <small>{t("optionalOverrides")}</small>
            </summary>
            <div className="details-content">
              <p className="hint">{t("masterBody")}</p>
              <div className="master-form-grid">
                {workflow.masterOverrides.map((item) => (
                  <MasterForm
                    key={item.sourceSha256}
                    item={item}
                    update={(patch) => workflow.updateMasterOverride(item.sourceSha256, patch)}
                    confirm={() => workflow.confirmMasterOverride(item.sourceSha256)}
                    reset={() => workflow.resetMasterOverride(item.sourceSha256)}
                    t={t}
                  />
                ))}
              </div>
            </div>
          </details>
        </div>
      )}
    </div>
  );
}

/** Drop shrink is typed freely and clamped to the engine's [0.1, 1] range when the field is left. */
function DropShrinkField({
  value,
  disabled,
  onCommit,
  t,
}: {
  value: number;
  disabled: boolean;
  onCommit: (value: number) => void;
  t: Translator;
}) {
  const [text, setText] = useState(String(value));
  const commit = () => {
    const parsed = Number(text);
    const clamped = Number.isFinite(parsed) ? Math.min(1, Math.max(0.1, Math.round(parsed * 100) / 100)) : value;
    onCommit(clamped);
    setText(String(clamped));
  };
  return (
    <div className="field-row">
      <label htmlFor="drizzle-drop-shrink">{t("drizzleDropShrink")}</label>
      <input
        id="drizzle-drop-shrink"
        type="number"
        inputMode="decimal"
        min={0.1}
        max={1}
        step={0.05}
        value={text}
        disabled={disabled}
        onChange={(event) => setText(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === "Enter") commit();
        }}
      />
    </div>
  );
}

/** Output path entry, the advanced options with solver setup, and what still blocks a start. */
export function LaunchSettings({
  mode,
  workflow,
  outputPathText,
  setOutputPathText,
  t,
}: {
  mode: "import" | "inspect";
  workflow: Workflow;
  outputPathText: string;
  setOutputPathText: (value: string) => void;
  t: Translator;
}) {
  const catalogFraction = workflow.catalogProgress?.sizeBytes
    ? Math.min(1, workflow.catalogProgress.downloadedBytes / workflow.catalogProgress.sizeBytes)
    : 0;
  return (
    <div className="stack" style={{ padding: 0 }}>
      <div className="panel">
        {workflow.nativeRuntime && (
          <details className="disclosure">
            <summary>
              <span>{t("enterOutputPath")}</span>
            </summary>
            <form
              className="details-content"
              onSubmit={(event) => {
                event.preventDefault();
                workflow.useOutputParentPath(outputPathText);
              }}
            >
              <div className="field-row">
                <label htmlFor="output-parent-path">{t("outputPathLabel")}</label>
                <input
                  id="output-parent-path"
                  value={outputPathText}
                  disabled={workflow.runNavigationLocked}
                  spellCheck={false}
                  aria-describedby="output-path-hint"
                  onChange={(event) => setOutputPathText(event.target.value)}
                />
                <p id="output-path-hint" className="hint">
                  {t("outputPathHint")}
                </p>
              </div>
              <div className="row-actions">
                <button
                  type="submit"
                  className="btn small"
                  disabled={workflow.runNavigationLocked || !outputPathText.trim()}
                >
                  {t("useOutputPath")}
                </button>
              </div>
            </form>
          </details>
        )}
        <details className="disclosure" open={!workflow.solverSetupReady && workflow.nativeRuntime}>
          <summary>
            <span>{t("advanced")}</span>
            <small>Drizzle · LN · WCS</small>
          </summary>
          <div className="details-content">
            <label className={`toggle ${workflow.drizzleEnabled ? "enabled" : ""}`}>
              <input
                type="checkbox"
                checked={workflow.drizzleEnabled}
                disabled={!workflow.capabilities?.drizzleAvailable}
                onChange={(event) => workflow.setDrizzleEnabled(event.target.checked)}
              />
              <span>
                <strong>{t("drizzle")}</strong>
                <small>{t("drizzleHint")}</small>
              </span>
            </label>
            {workflow.drizzleEnabled && (
              <div className="option-grid" aria-label={t("drizzleOptions")}>
                <div className="field-row">
                  <label htmlFor="drizzle-scale">{t("drizzleScale")}</label>
                  <select
                    id="drizzle-scale"
                    value={workflow.drizzleScale}
                    disabled={workflow.runNavigationLocked}
                    onChange={(event) => workflow.setDrizzleScale(Number(event.target.value) as DrizzleScale)}
                  >
                    {DRIZZLE_SCALES.map((scale) => (
                      <option key={scale} value={scale}>
                        {scale}×
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field-row">
                  <label htmlFor="drizzle-kernel">{t("drizzleKernel")}</label>
                  <select
                    id="drizzle-kernel"
                    value={workflow.drizzleKernel}
                    disabled={workflow.runNavigationLocked}
                    onChange={(event) => workflow.setDrizzleKernel(event.target.value as DrizzleKernel)}
                  >
                    {DRIZZLE_KERNELS.map((kernel) => (
                      <option key={kernel} value={kernel}>
                        {t(`drizzleKernel_${kernel}` as const)}
                      </option>
                    ))}
                  </select>
                </div>
                <DropShrinkField
                  value={workflow.drizzleDropShrink}
                  disabled={workflow.runNavigationLocked}
                  onCommit={workflow.setDrizzleDropShrink}
                  t={t}
                />
              </div>
            )}
            <section className="panel" aria-labelledby="advanced-algorithms-title">
              <div className="panel-heading">
                <span id="advanced-algorithms-title">{t("advancedAlgorithms")}</span>
                <small>{t("advancedAlgorithmsBadge")}</small>
              </div>
              <div className="panel-body">
                <p className="hint">{t("advancedAlgorithmsHint")}</p>
                <label
                  className={`toggle ${workflow.properCoadditionEnabled && !workflow.drizzleEnabled ? "enabled" : ""}`}
                >
                  <input
                    type="checkbox"
                    checked={workflow.properCoadditionEnabled && !workflow.drizzleEnabled}
                    disabled={workflow.runNavigationLocked || workflow.drizzleEnabled}
                    onChange={(event) => workflow.setProperCoadditionEnabled(event.target.checked)}
                  />
                  <span>
                    <strong>{t("properCoaddition")}</strong>
                    <small>{workflow.drizzleEnabled ? t("properCoadditionDrizzle") : t("properCoadditionHint")}</small>
                  </span>
                </label>
              </div>
            </section>
            <section className="panel" aria-labelledby="solver-title">
              <div className="panel-heading">
                <span id="solver-title">{t("solverSetup")}</span>
                <div className="setup-actions">
                  <small>{workflow.solverSetupReady ? t("ready") : t("actionRequired")}</small>
                  {workflow.nativeRuntime && (
                    <button
                      type="button"
                      className="btn small"
                      disabled={
                        workflow.solverSetupBusy ||
                        workflow.catalogStatus === "DOWNLOADING" ||
                        workflow.catalogStatus === "VERIFYING"
                      }
                      onClick={() => void workflow.recheckSolverSetup()}
                    >
                      {workflow.solverSetupBusy ? t("checkingSetup") : t("recheckSetup")}
                    </button>
                  )}
                </div>
              </div>
              <div className="panel-body">
                <SolverRows workflow={workflow} t={t} />
                {!workflow.nativeRuntime ? (
                  <p className="mock-notice">{t("browserNoSolver")}</p>
                ) : workflow.catalogDoctor?.ok ? (
                  <div className="catalog-ready">
                    <CheckIcon />
                    <div>
                      <strong>{t("catalogReady")}</strong>
                      <p>{workflow.catalogDoctor.catalogRoot}</p>
                    </div>
                  </div>
                ) : workflow.recommendedCatalog ? (
                  <div className="catalog-install">
                    <div>
                      <strong>{t("catalogInstallTitle")}</strong>
                      <p className="hint">{t("catalogInstallBody")}</p>
                    </div>
                    <div className="terms-box">
                      <p>{workflow.recommendedCatalog.providerTerms.summary}</p>
                      <a
                        href={workflow.recommendedCatalog.providerTerms.url}
                        onClick={(event) => {
                          event.preventDefault();
                          void workflow.openCatalogTerms(workflow.recommendedCatalog!.providerTerms.url);
                        }}
                      >
                        {t("viewTerms")}
                      </a>
                      <label>
                        <input
                          type="checkbox"
                          checked={workflow.catalogTermsAccepted}
                          disabled={workflow.catalogStatus === "DOWNLOADING" || workflow.catalogStatus === "VERIFYING"}
                          onChange={(event) => workflow.setCatalogTermsAccepted(event.target.checked)}
                        />
                        {t("acceptTerms", { id: workflow.recommendedCatalog.providerTerms.acceptanceId })}
                      </label>
                    </div>
                    {workflow.catalogStatus === "DOWNLOADING" && (
                      <div className="catalog-progress" role="status">
                        <progress max={1} value={catalogFraction} />
                        <span>
                          {workflow.catalogProgress
                            ? `${workflow.catalogProgress.artifactId} · ${(catalogFraction * 100).toFixed(1)}%`
                            : t("waitProgress")}
                        </span>
                        <button
                          type="button"
                          className="btn small"
                          onClick={() => void workflow.cancelCatalogInstall()}
                        >
                          {t("cancelDownload")}
                        </button>
                      </div>
                    )}
                    {workflow.catalogStatus === "VERIFYING" && (
                      <p role="status" className="status-note">
                        {t("verifyingCatalog")}
                      </p>
                    )}
                    {workflow.catalogStatus !== "DOWNLOADING" && workflow.catalogStatus !== "VERIFYING" && (
                      <div className="row-actions">
                        <button
                          type="button"
                          className="btn primary"
                          disabled={!workflow.catalogTermsAccepted}
                          onClick={() => void workflow.startCatalogInstall()}
                        >
                          {t("downloadCatalog", { size: formatBytes(workflow.recommendedCatalog.totalSizeBytes, t) })}
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="error-text" role="alert">
                    {t("noCatalogManifest")}
                  </p>
                )}
                {workflow.catalogError && (
                  <p className="error-text" role="alert">
                    {workflow.catalogError}
                  </p>
                )}
              </div>
            </section>
            <div className="compute" aria-label={t("computeBackend")}>
              <CpuIcon />
              <strong>{workflow.capabilities?.chip ?? t("detectingHardware")}</strong>
              <span>
                {workflow.capabilities?.cpuBackend ?? "Portable CPU"} ·{" "}
                {workflow.capabilities?.gpuBackend ?? "GPU probe"} · {computeLabel(workflow.capabilities, t)}
              </span>
            </div>
          </div>
        </details>
      </div>
      {!workflow.canStart && !workflow.demoMode && <StartBlockerList workflow={workflow} mode={mode} t={t} />}
    </div>
  );
}

/** Why the run cannot start yet, with the one action that resolves most of them. */
function StartBlockerList({ workflow, mode, t }: { workflow: Workflow; mode: "import" | "inspect"; t: Translator }) {
  const blinkPanels = workflow.blinkReady && workflow.nativeRuntime;
  const describe = (blocker: StartBlocker): string => {
    switch (blocker.kind) {
      case "engine":
        return blocker.reason ?? t("blockerEngine");
      case "calibration":
        return t("blockerCalibration");
      case "types":
        return t("blockerTypes");
      case "lights":
        return t("blockerLights");
      case "panel": {
        const { cell } = blocker;
        return blinkPanels
          ? t("blockerBlinkChannel", {
              target: cell.target,
              filter: cell.filter,
              kept: cell.admittedCount,
              total: cell.lightCount,
              required: workflow.minimumAdmittedLights,
            })
          : t("insufficientPanelLights", {
              target: cell.target,
              filter: cell.filter,
              count: workflow.admissionKnown ? cell.admittedCount : cell.lightCount,
              required: workflow.minimumAdmittedLights,
            });
      }
      case "solver":
        return t("blockerSolver");
      case "output":
        return t("blockerOutput");
      case "master":
        return t("blockerMaster");
      case "cfa":
        return t("blockerCfaUnknownPattern");
    }
  };
  const panelsShort = workflow.insufficientPanels.length > 0;
  return (
    <div className="blockers" role="status">
      <strong>{t("blockers")}</strong>
      <ul>
        {workflow.startBlockers.map((blocker, index) => (
          <li key={blocker.kind === "panel" ? blocker.cell.panelId : `${blocker.kind}-${index}`}>
            {describe(blocker)}
          </li>
        ))}
      </ul>
      {blinkPanels && panelsShort ? (
        <button type="button" className="btn small" onClick={() => void workflow.runBlink()}>
          {t("blinkOpen")}
        </button>
      ) : (
        workflow.qualityReady &&
        panelsShort &&
        mode !== "inspect" && (
          <button type="button" className="btn small" onClick={() => workflow.setStep("inspect")}>
            {t("reviewLightAdmission")}
          </button>
        )
      )}
    </div>
  );
}

/** The validated path this build runs on, as the compute line names it. */
function computeLabel(capabilities: Workflow["capabilities"], t: Translator): string {
  if (capabilities?.platform === "browser") return t("demoNoCompute");
  if (capabilities?.platform === "windows")
    return capabilities.optimizationTier === "WINDOWS_X64" ? t("windowsValidated") : t("windowsUnsupported");
  return capabilities?.optimizationTier === "M3_PRO_TUNED" ? t("m3Optimized") : t("appleGeneric");
}

/**
 * The two solvers, the platform's primary one first with its install
 * instruction: solve-field on macOS, ASTAP (verified by the engine against the
 * managed indexes) on Windows, where solve-field has no native build.
 */
function SolverRows({ workflow, t }: { workflow: Workflow; t: Translator }) {
  const windows = workflow.primarySolver === "astap";
  const solveField = (
    <SolverRow
      key="solve-field"
      name="solve-field"
      backend={workflow.solveField}
      ready={workflow.solveFieldReady}
      instruction={t(windows ? "solveFieldOptional" : "solveFieldInstall")}
      t={t}
    />
  );
  const astap = (
    <SolverRow
      key="astap"
      name="ASTAP"
      backend={workflow.astap}
      ready={workflow.astapReady}
      instruction={t(windows ? "astapInstall" : "astapAlternative")}
      t={t}
    />
  );
  return <>{windows ? [astap, solveField] : [solveField, astap]}</>;
}

/** Manual decisions are the only GUI admission source. */
function ScreeningNotice({ workflow, t }: { workflow: Workflow; t: Translator }) {
  if (workflow.demoMode || !workflow.blinkReviewComplete) return null;
  const kept = workflow.blinkChannels.reduce((sum, channel) => sum + channel.kept, 0);
  const total = workflow.blinkChannels.reduce((sum, channel) => sum + channel.total, 0);
  return (
    <div className="screening-notice blinked" role="status">
      <span className="symbol" aria-hidden="true">
        ✓
      </span>
      <div>
        <strong>{t("noticeBlinkedTitle")}</strong>
        <span>{t("noticeBlinkedBody", { kept, total, dropped: total - kept })}</span>
      </div>
      <button
        type="button"
        className="btn small"
        disabled={workflow.inputBusy}
        onClick={() => void workflow.runBlink()}
      >
        {t("blinkOpen")}
      </button>
    </div>
  );
}

export function LaunchBar({
  mode,
  workflow,
  lightCount,
  t,
}: {
  mode: "import" | "inspect";
  workflow: Workflow;
  lightCount: number;
  t: Translator;
}) {
  // Blink is primary until every channel has been reviewed and confirmed.
  const blinkPrimary = !workflow.blinkReviewComplete;
  const blinkButton = (
    <button
      type="button"
      className={`btn ${blinkPrimary ? "primary" : ""}`}
      disabled={!workflow.canInspect || !workflow.allRequiredConfirmed || workflow.inputBusy}
      onClick={() => void workflow.runBlink()}
    >
      {workflow.blinkBusy
        ? t("blinkMeasuring", { count: lightCount, seconds: workflow.blinkElapsedSeconds })
        : workflow.blinkReady
          ? t("blinkOpen")
          : t("blinkAndSelect", { count: lightCount || "" })}
    </button>
  );
  const kept = workflow.blinkChannels.reduce((sum, channel) => sum + channel.kept, 0);
  const total = workflow.blinkChannels.reduce((sum, channel) => sum + channel.total, 0);
  const startLabel = workflow.blinkReviewComplete ? t("startSelected", { kept, total }) : t("start");
  return (
    <footer className="launchbar">
      {!workflow.blinkReviewComplete && !workflow.demoMode ? (
        <div className="screening-notice" role="status">
          <span className="symbol">!</span>
          <div>
            <strong>{t("blinkHumanReviewTitle")}</strong>
            <span>{t("blockerBlinkReview")}</span>
          </div>
        </div>
      ) : (
        <ScreeningNotice workflow={workflow} t={t} />
      )}
      <div className="output">
        <FolderIcon />
        <div style={{ minWidth: 0 }}>
          <small>{t("outputLabel")} · </small>
          <strong title={workflow.outputParent}>
            {workflow.outputParent ?? (workflow.demoMode ? t("browserNoFiles") : t("outputNotSelected"))}
          </strong>
        </div>
        <button
          type="button"
          className="btn small"
          disabled={!workflow.nativeRuntime}
          onClick={() => void workflow.chooseOutputParent()}
        >
          {t("chooseOutput")}
        </button>
      </div>
      <div className="actions">
        {mode === "inspect" ? (
          <>
            <button type="button" className="btn quiet" onClick={() => workflow.setStep("import")}>
              {t("back")}
            </button>
            {blinkButton}
          </>
        ) : (
          <>
            {workflow.browserDemoAvailable && (
              <button type="button" className="btn quiet" onClick={workflow.loadDemo}>
                {t("loadDemo")}
              </button>
            )}
            <button
              type="button"
              className="btn quiet"
              disabled={!workflow.canInspect || !workflow.allRequiredConfirmed || workflow.qualityBusy}
              onClick={() => void workflow.runInspection()}
            >
              {workflow.qualityBusy
                ? t("inspecting")
                : workflow.qualityReady
                  ? t("viewQualityResults")
                  : t("inspect", { count: lightCount || "" })}
            </button>
            {blinkButton}
          </>
        )}
        <button
          type="button"
          className={`btn ${blinkPrimary && !workflow.demoMode ? "" : "primary"}`}
          disabled={!workflow.canStart}
          onClick={() => void workflow.startRun()}
        >
          <PlayIcon />
          {startLabel}
        </button>
      </div>
    </footer>
  );
}
