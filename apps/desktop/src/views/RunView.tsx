import type { Translator } from "../i18n";
import { CheckIcon, SparkIcon, XIcon } from "../icons";
import { RunElapsed, stageLabel, type Workflow } from "./common";

/** The failure card of a run that failed closed: what failed, where the engine's evidence is, what to do. */
function RunFailure({ workflow, t }: { workflow: Workflow; t: Translator }) {
  const code = workflow.runFailureCode;
  const message = workflow.errorMessage ?? "";
  const detail = code && message.startsWith(`${code}: `) ? message.slice(code.length + 2) : message;
  const failedStage = workflow.stages.find((stage) => stage.status === "FAILED");
  const evidence = workflow.outputDirectory ? `${workflow.outputDirectory}.unsolved` : undefined;
  return (
    <section className="run-failure" role="alert" aria-live="assertive">
      <div className="run-failure-head">
        <span className="symbol" aria-hidden="true">
          !
        </span>
        <div>
          <strong>{t("runFailureTitle")}</strong>
          <span>
            {failedStage ? t("runFailureStage", { stage: stageLabel(failedStage.stageId, t) }) : t("runFailureNoStage")}
          </span>
        </div>
      </div>
      <p className="run-failure-detail">
        {code && (
          <>
            <code className="run-failure-code">{code}</code>:{" "}
          </>
        )}
        {detail}
      </p>
      {code === "ASTROMETRY_REQUIRED" && <p className="run-failure-hint">{t("astrometryFailedHint")}</p>}
      {evidence && (
        <p className="run-failure-hint">
          {t("unsolvedEvidenceHint")}{" "}
          <span className="run-failure-path" title={evidence}>
            {evidence}
          </span>
        </p>
      )}
    </section>
  );
}

export function RunView({ workflow, t }: { workflow: Workflow; t: Translator }) {
  const title =
    workflow.runStatus === "CANCELLED"
      ? t("cancelledTitle")
      : workflow.runStatus === "FAILED"
        ? t("failedTitle")
        : t("runningTitle");
  const stages = workflow.stages.filter((stage) => workflow.drizzleEnabled || stage.stageId !== "drizzle");
  return (
    <section className="view" aria-labelledby="run-title">
      <div className="scroll">
        <div className="proc-head">
          <div className="ring-lg" aria-hidden="true">
            <svg viewBox="0 0 120 120">
              <circle className="ring-track" cx="60" cy="60" r="52" />
              <circle
                className={`ring-value ${workflow.runStatus === "FAILED" ? "failed" : workflow.runStatus === "COMPLETED" ? "done" : ""}`}
                cx="60"
                cy="60"
                r="52"
                pathLength="100"
                strokeDasharray={`${workflow.overallProgress} 100`}
              />
            </svg>
            <div className="ring-label tnum">{workflow.overallProgress}%</div>
          </div>
          <div>
            <h1 id="run-title">{title}</h1>
            <p>{t("runDescription")}</p>
            <div className="row-actions" style={{ marginTop: 8 }}>
              {workflow.executionMode === "demo" && <span className="badge-demo">{t("demo")}</span>}
              <RunElapsed
                seconds={workflow.runElapsedSeconds}
                label={t(
                  workflow.runStatus === "RUNNING" || workflow.runStatus === "CANCELLING"
                    ? "runElapsed"
                    : "runTotalTime",
                )}
              />
            </div>
          </div>
        </div>
        <p className="muted" role="status" style={{ padding: "0 24px" }}>
          {workflow.runProgress?.scope === "panel"
            ? t("currentPanelProgress", {
                index: workflow.runProgress.panelIndex ?? 0,
                count: workflow.runProgress.panelCount ?? 0,
                target: workflow.runProgress.panelTarget ?? "",
                filter: workflow.runProgress.panelFilter ?? "",
              })
            : workflow.runProgress?.scope === "project"
              ? t("projectFinalStages")
              : t("overallProjectProgress")}
          {workflow.runProgress?.message && <small> · {workflow.runProgress.message}</small>}
        </p>
        {workflow.runStatus === "FAILED" && <RunFailure workflow={workflow} t={t} />}
        <ol className="stage-list">
          {stages.map((stage) => (
            <li key={stage.stageId} className={stage.status.toLowerCase()}>
              <span className="stage-index">
                {stage.status === "DONE" ? (
                  <CheckIcon />
                ) : stage.status === "FAILED" ? (
                  <XIcon />
                ) : stage.status === "RUNNING" ? (
                  <svg className="ring" viewBox="0 0 16 16" aria-hidden="true">
                    <circle className="spin" cx="8" cy="8" r="5.9" />
                  </svg>
                ) : (
                  <span className="pend" aria-hidden="true" />
                )}
              </span>
              <span>{stageLabel(stage.stageId, t)}</span>
              <span className="mini-track">
                <span style={{ width: `${stage.percent}%` }} />
              </span>
              <em>{stage.status === "FAILED" ? t("stageFailed") : `${Math.round(stage.percent)}%`}</em>
            </li>
          ))}
        </ol>
        {workflow.executionMode === "demo" && (
          <p className="mock-notice" style={{ padding: "0 24px" }}>
            <SparkIcon />
            {t("demoWarningBody")}
          </p>
        )}
        {workflow.errorMessage && workflow.runStatus !== "FAILED" && (
          <p className="error-text" role="alert" style={{ padding: "8px 24px" }}>
            {workflow.errorMessage}
          </p>
        )}
        <div className="run-actions">
          {workflow.runStatus === "RUNNING" && (
            <button
              type="button"
              className="btn"
              disabled={workflow.runLaunchBusy}
              onClick={() => void workflow.cancelRun()}
            >
              {t("cancel")}
            </button>
          )}
          {["CANCELLED", "FAILED"].includes(workflow.runStatus) && (
            <button
              type="button"
              className={workflow.runStatus === "FAILED" ? "btn primary" : "btn"}
              onClick={() => workflow.setStep("import")}
            >
              {t("returnConfig")}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
