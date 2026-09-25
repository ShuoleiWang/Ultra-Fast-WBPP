import { CalibrationGroups } from "../FrameInventory";
import { dispositionLabel } from "../Inspector";
import type { Translator } from "../i18n";
import type { GateDisposition } from "../types";
import { LaunchBar, LaunchSettings, MetadataConfirmations } from "./LaunchSettings";
import { Alert, basename, dispositionClass, FrameTile, PASS_FRAME_BATCH, type Workflow } from "./common";

export function ScreeningView({
  workflow,
  t,
  outputPathText,
  setOutputPathText,
  selectedPath,
  setSelectedPath,
  visiblePassFrames,
  setVisiblePassFrames,
  decisionFilter,
  setDecisionFilter,
  lightCount,
}: {
  workflow: Workflow;
  t: Translator;
  outputPathText: string;
  setOutputPathText: (value: string) => void;
  selectedPath?: string;
  setSelectedPath: (path?: string) => void;
  visiblePassFrames: number;
  setVisiblePassFrames: (update: (current: number) => number) => void;
  decisionFilter: "ALL" | GateDisposition;
  setDecisionFilter: (filter: "ALL" | GateDisposition) => void;
  lightCount: number;
}) {
  const targets = [...new Set(workflow.matrix.map((cell) => cell.target))];
  const filters = [...new Set(workflow.matrix.map((cell) => cell.filter))];
  const frames = workflow.qualityInspection?.frames ?? [];
  const filterByPath = new Map(workflow.assets.map((asset) => [asset.path, asset.filter] as const));
  const filterOf = (path: string) => filterByPath.get(path);
  const issueFrames = frames.filter((frame) => frame.disposition !== "PASS");
  const passingFrames = frames.filter((frame) => frame.disposition === "PASS");
  const ordered =
    decisionFilter === "ALL"
      ? [...issueFrames, ...passingFrames.slice(0, visiblePassFrames)]
      : decisionFilter === "PASS"
        ? passingFrames.slice(0, visiblePassFrames)
        : frames.filter((frame) => frame.disposition === decisionFilter);
  const total = workflow.gate.pass + workflow.gate.review + workflow.gate.hardFail;
  const chips: Array<{ key: "ALL" | GateDisposition; label: string; count: number; dot?: string }> = [
    { key: "ALL", label: t("allFrames"), count: total },
    { key: "PASS", label: t("dispositionPass"), count: workflow.gate.pass, dot: "ok" },
    { key: "REVIEW", label: t("dispositionReview"), count: workflow.gate.review, dot: "warn" },
    { key: "HARD_FAIL", label: t("dispositionFail"), count: workflow.gate.hardFail, dot: "stop" },
  ];
  return (
    <section className="view" aria-labelledby="inspect-title">
      <div className="chead">
        <h1 id="inspect-title">{t("reviewTitle")}</h1>
        <div className="chead-row">
          <div className="chips" role="group" aria-label={t("colDecision")}>
            {chips.map((chip) => (
              <button
                type="button"
                key={chip.key}
                className="chip"
                aria-pressed={decisionFilter === chip.key}
                onClick={() => setDecisionFilter(chip.key)}
              >
                {chip.dot && <span className={`dot ${chip.dot}`} />}
                {chip.label} <span className="n">{chip.count}</span>
              </button>
            ))}
          </div>
          <span className="spacer" />
          <span className="muted">
            {workflow.demoMode
              ? t("demoReviewDescription")
              : workflow.qualityInspection
                ? `${t("failClosed")} · POLICY ${workflow.qualityInspection.gatePolicyDigest.slice(7, 19)}`
                : t("reviewDescription")}
          </span>
        </div>
      </div>
      <div className="scroll">
        <div className="stack">
          {workflow.demoMode && <Alert title={t("demoWarningTitle")}>{t("demoWarningBody")}</Alert>}
          {!workflow.demoMode && workflow.qualityInspection && (
            <div className="panel">
              <div className="tbl-wrap">
                <table className="tbl">
                  <thead>
                    <tr>
                      <th className="thumb-cell" aria-label={t("previewAlt", { name: "" }).trim()} />
                      <th>{t("colFrame")}</th>
                      <th className="r">{t("colStars")}</th>
                      <th>{t("colConfidence")}</th>
                      <th>{t("colDecision")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ordered.map((frame) => {
                      const selected = selectedPath === frame.path;
                      return (
                        <tr
                          key={frame.path}
                          className={`quality-frame quality-${dispositionClass(frame.disposition)}`}
                          aria-selected={selected}
                          onClick={() => setSelectedPath(selected ? undefined : frame.path)}
                        >
                          <td className="thumb-cell">
                            <FrameTile filter={filterOf(frame.path)} preview={frame.previewDataUrl} />
                          </td>
                          <td className="clip" title={frame.path}>
                            <strong className="frame-kind">{basename(frame.path)}</strong>
                          </td>
                          <td className="r tnum">{frame.starCount}</td>
                          <td>
                            <small>{frame.confidence}</small>
                          </td>
                          <td className="dec-cell">
                            <div className="dec">
                              {frame.disposition === "PASS" ? (
                                <>
                                  <span className="dot ok" />
                                  <span>{t("dispositionPass")}</span>
                                </>
                              ) : (
                                <span className={`badge ${frame.disposition === "REVIEW" ? "check" : "stop"}`}>
                                  {dispositionLabel(frame.disposition, t)}
                                </span>
                              )}
                              <span className="why">
                                {frame.evidence.length
                                  ? frame.evidence
                                      .map((item) => item.message)
                                      .slice(0, 2)
                                      .join("; ")
                                  : frame.disposition === "PASS"
                                    ? ""
                                    : t("noReviewEvidence")}
                              </span>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {ordered.length === 0 && <p className="table-empty">{t("inventoryEmpty")}</p>}
              {(decisionFilter === "ALL" || decisionFilter === "PASS") && passingFrames.length > visiblePassFrames && (
                <div className="pass-pagination">
                  <span>
                    {t("passBatchStatus", {
                      shown: Math.min(visiblePassFrames, passingFrames.length),
                      total: passingFrames.length,
                    })}
                  </span>
                  <button
                    type="button"
                    className="btn small"
                    onClick={() => setVisiblePassFrames((current) => current + PASS_FRAME_BATCH)}
                  >
                    {t("showMorePass", { count: Math.min(PASS_FRAME_BATCH, passingFrames.length - visiblePassFrames) })}
                  </button>
                </div>
              )}
            </div>
          )}
          <section className="panel matrix-section" aria-labelledby="matrix-title">
            <div className="panel-heading">
              <span id="matrix-title">{t("panelMatrix")}</span>
              <small>
                {t("matrixCounts", {
                  targets: targets.length,
                  filters: filters.length,
                  panels: workflow.matrix.length,
                })}
              </small>
            </div>
            {workflow.matrix.length ? (
              <div className="tbl-wrap">
                <table className="tbl compact">
                  <thead>
                    <tr>
                      <th scope="col">{t("targetLabel")}</th>
                      {filters.map((filter) => (
                        <th scope="col" key={filter}>
                          {filter}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {targets.map((target) => (
                      <tr key={target}>
                        <th scope="row">{target}</th>
                        {filters.map((filter) => {
                          const cell = workflow.matrix.find((item) => item.target === target && item.filter === filter);
                          return (
                            <td key={filter} className={cell ? "matrix-present" : "matrix-empty"}>
                              {cell
                                ? workflow.admissionKnown
                                  ? t("panelAdmittedCounts", {
                                      admitted: cell.admittedCount,
                                      total: cell.lightCount,
                                      required: workflow.minimumAdmittedLights,
                                    })
                                  : `${cell.lightCount} ${t("frames")}`
                                : "—"}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="table-empty">{t("noGroups")}</p>
            )}
          </section>
          <CalibrationGroups workflow={workflow} t={t} />
          <MetadataConfirmations workflow={workflow} t={t} />
          <LaunchSettings
            mode="inspect"
            workflow={workflow}
            outputPathText={outputPathText}
            setOutputPathText={setOutputPathText}
            t={t}
          />
        </div>
      </div>
      <LaunchBar mode="inspect" workflow={workflow} lightCount={lightCount} t={t} />
    </section>
  );
}
