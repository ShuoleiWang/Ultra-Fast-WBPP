import { convertFileSrc } from "@tauri-apps/api/core";
import { dispositionLabel } from "../Inspector";
import { blinkFlagLabel, type Translator } from "../i18n";
import { CheckIcon, FileIcon, RevealIcon, SparkIcon } from "../icons";
import type { OutputArtifact, OutputArtifactKind, ScreeningSummary } from "../types";
import {
  artifactLabel,
  basename,
  dispositionClass,
  filterTileClass,
  FrameTile,
  RunElapsed,
  StarField,
  type Workflow,
} from "./common";

const MASTER_KINDS: OutputArtifactKind[] = ["SOLVED_MONO_FITS", "LINEAR_RGB_FITS", "MASTER"];
const PREVIEW_KINDS: OutputArtifactKind[] = ["MONO_PREVIEW_PNG", "RGB_PREVIEW_PNG_16", "PREVIEW"];

function previewFor(artifact: OutputArtifact, artifacts: OutputArtifact[], native: boolean): string | undefined {
  const match = artifacts.find(
    (candidate) =>
      PREVIEW_KINDS.includes(candidate.kind) &&
      (candidate.filter ?? "") === (artifact.filter ?? "") &&
      (candidate.target ?? "") === (artifact.target ?? ""),
  );
  if (!match) return undefined;
  // The controller carries the PNG previews as data URLs, which show from any
  // drive or share; the asset protocol (home folder and volumes only) is the
  // fallback for a preview it could not carry.
  if (match.previewDataUrl) return match.previewDataUrl;
  if (!native) return undefined;
  try {
    return convertFileSrc(match.path);
  } catch {
    return undefined;
  }
}

export function ResultView({ workflow, t }: { workflow: Workflow; t: Translator }) {
  const demo = workflow.executionMode === "demo";
  const masters = workflow.artifacts.filter((artifact) => MASTER_KINDS.includes(artifact.kind));
  const products = workflow.artifacts.filter((artifact) => !["RECEIPT", "REPORT"].includes(artifact.kind));
  const technical = workflow.artifacts.filter((artifact) => ["RECEIPT", "REPORT"].includes(artifact.kind));
  const heroPreview = masters
    .map((artifact) => previewFor(artifact, workflow.artifacts, workflow.nativeRuntime))
    .find(Boolean);
  return (
    <section className="view" aria-labelledby="result-title">
      <div className="scroll">
        <div className="result-hero">
          {heroPreview ? (
            <img className="result-hero-image" src={heroPreview} alt="" />
          ) : (
            <StarField className="result-hero-sky" seed={9} count={160} />
          )}
          <div className="result-hero-shade" aria-hidden="true" />
          <div className="result-hero-body">
            <span className={`pill ${demo ? "warn" : ""}`}>
              <CheckIcon />
              {demo ? t("resultDemo") : t("resultSolved")}
            </span>
            <h1 id="result-title">{demo ? t("resultDemoTitle") : t("resultTitle")}</h1>
            <p>{demo ? t("resultDemoBody") : t("resultBody")}</p>
            <dl className="hero-metrics">
              <div>
                <dt>{t("runTotalTime")}</dt>
                <dd>
                  <RunElapsed seconds={workflow.runElapsedSeconds} label={t("runTotalTime")} />
                </dd>
              </div>
              {workflow.screening && (
                <div className={workflow.screening.excluded > 0 ? "metric-warn" : ""}>
                  <dt>{t("screeningTitle")}</dt>
                  <dd className="tnum">
                    <strong>
                      {t("framesUsed", {
                        admitted: workflow.screening.admitted,
                        total: workflow.screening.admitted + workflow.screening.excluded,
                      })}
                    </strong>
                    {workflow.screening.excluded > 0 && (
                      <small> · {t("framesExcluded", { excluded: workflow.screening.excluded })}</small>
                    )}
                  </dd>
                </div>
              )}
              {workflow.firstSolved ? (
                <>
                  <div>
                    <dt>{t("center")}</dt>
                    <dd className="tnum">
                      {workflow.firstSolved.centerRaDegrees.toFixed(4)}° ·{" "}
                      {workflow.firstSolved.centerDecDegrees.toFixed(4)}°
                    </dd>
                  </div>
                  <div>
                    <dt>{t("pixelScale")}</dt>
                    <dd className="tnum">{workflow.firstSolved.pixelScaleArcsec.toFixed(3)}″ / px</dd>
                  </div>
                  <div>
                    <dt>{t("solveQuality")}</dt>
                    <dd className="tnum">
                      {workflow.firstSolved.rmsArcsec.toFixed(3)}″ RMS · {workflow.firstSolved.matchedStars}{" "}
                      {t("starsLabel")}
                    </dd>
                  </div>
                </>
              ) : (
                <div>
                  <dt>WCS</dt>
                  <dd>{t("demoNoWcs")}</dd>
                </div>
              )}
            </dl>
          </div>
        </div>
        {masters.length > 0 && (
          <div className="cards">
            {masters.map((artifact) => {
              const preview = previewFor(artifact, workflow.artifacts, workflow.nativeRuntime);
              return (
                <article className="card" key={artifact.path}>
                  <div className={`shot tile ${preview ? "" : filterTileClass(artifact.filter)}`}>
                    {preview ? (
                      <img src={preview} alt="" />
                    ) : (
                      <StarField seed={artifact.path.length} count={70} className="shot-sky" />
                    )}
                    {artifact.filter && <span className="tag">{artifact.filter}</span>}
                  </div>
                  <div className="card-body">
                    <div className="card-title">
                      <b title={artifact.path}>{artifact.name}</b>
                      {artifact.target && <span>{artifact.target}</span>}
                    </div>
                    <div className="metrics">{artifact.detail}</div>
                  </div>
                </article>
              );
            })}
          </div>
        )}
        <div className="stack">
          {workflow.screening && <ScreeningSection screening={workflow.screening} t={t} />}
          <ArtifactList title={t("products")} artifacts={products} workflow={workflow} t={t} />
          {technical.length > 0 && (
            <div className="panel">
              <details className="disclosure">
                <summary>
                  <span>{t("technicalDetails")}</span>
                  <small>SHA-256 · JSON</small>
                </summary>
                <div className="details-content" style={{ padding: 0 }}>
                  <ArtifactList artifacts={technical} workflow={workflow} t={t} />
                </div>
              </details>
            </div>
          )}
          {demo && (
            <p className="mock-notice">
              <SparkIcon />
              {t("demoWarningBody")}
            </p>
          )}
          <div className="row-actions" style={{ justifyContent: "space-between" }}>
            <button type="button" className="btn" onClick={workflow.clearSources}>
              {t("startNew")}
            </button>
            <button
              type="button"
              className="btn primary"
              disabled={!workflow.nativeRuntime || !workflow.outputDirectory}
              title={workflow.outputDirectory}
              onClick={() => workflow.outputDirectory && void workflow.revealOutput(workflow.outputDirectory)}
            >
              <RevealIcon />
              {t("revealOutput", { name: workflow.outputDirectory ? basename(workflow.outputDirectory) : "—" })}
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

/** The decision badge of a screened frame: the user's blink decision when there was one, else the gate's disposition. */
function screeningBadge(frame: ScreeningSummary["frames"][number], t: Translator) {
  if (frame.reason === "USER_DROP") return <span className="badge stop">{t("screeningUserDrop")}</span>;
  if (frame.reason === "USER_KEEP_OVERRIDE")
    return <span className="badge check">{t("screeningUserKeepOverride")}</span>;
  return (
    <span className={`badge ${frame.admitted ? "ok" : frame.disposition === "REVIEW" ? "check" : "stop"}`}>
      {frame.admitted ? t("screeningAdmittedReview") : dispositionLabel(frame.disposition, t)}
    </span>
  );
}
function screeningWhy(frame: ScreeningSummary["frames"][number], t: Translator): string {
  const flags = (frame.flags ?? []).map((code) => blinkFlagLabel(code, t));
  if (flags.length) return flags.join("; ");
  return frame.evidence.length ? frame.evidence.slice(0, 2).join("; ") : frame.summary || t("noReviewEvidence");
}

/** The run's own screening: how many Lights went in and why the others did not. */
export function ScreeningSection({ screening, t }: { screening: ScreeningSummary; t: Translator }) {
  const decided = screening.frames;
  return (
    <section className="panel screening-section" aria-labelledby="screening-title">
      <div className="panel-heading">
        <span id="screening-title">{t("screeningTitle")}</span>
        <small>{t("screeningSummary", { admitted: screening.admitted, excluded: screening.excluded })}</small>
      </div>
      {decided.length === 0 ? (
        <p className="screening-clean">{t("screeningAllPassed")}</p>
      ) : (
        <div className="tbl-wrap">
          <table className="tbl">
            <tbody>
              {decided.map((frame) => (
                <tr
                  key={`${frame.target ?? ""}/${frame.name}`}
                  className={`quality-frame quality-${dispositionClass(frame.disposition)}`}
                >
                  <td className="thumb-cell">
                    {frame.previewDataUrl ? (
                      <img className="thumb" src={frame.previewDataUrl} alt={t("previewAlt", { name: frame.name })} />
                    ) : (
                      <FrameTile />
                    )}
                  </td>
                  <td className="clip" title={frame.name}>
                    <strong className="frame-kind">{frame.name}</strong>
                    <small>
                      {frame.target ? `${frame.target} · ` : ""}
                      {frame.starCount ?? "—"} {t("starsLabel")}
                    </small>
                  </td>
                  <td className="dec-cell">
                    <div className="dec">
                      {screeningBadge(frame, t)}
                      <span className="why">{screeningWhy(frame, t)}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {decided.some((frame) => frame.disposition === "REVIEW" && !frame.admitted && !frame.reason) && (
        <p className="screening-hint">{t("screeningReviewHint")}</p>
      )}
    </section>
  );
}

function ArtifactList({
  title,
  artifacts,
  workflow,
  t,
}: {
  title?: string;
  artifacts: Workflow["artifacts"];
  workflow: Workflow;
  t: Translator;
}) {
  return (
    <div className={title ? "panel" : ""}>
      {title && (
        <div className="panel-heading">
          <span>{title}</span>
          <small>{t("verifiedFiles", { count: artifacts.length })}</small>
        </div>
      )}
      <div className="artifact-list">
        {artifacts.map((artifact) => (
          <article key={artifact.path}>
            <span className={`artifact-kind kind-${artifact.kind.toLowerCase()}`}>
              <FileIcon />
            </span>
            <div style={{ minWidth: 0 }}>
              <strong>{artifact.name}</strong>
              <small>
                <span>{artifactLabel(artifact.kind, t)}</span>
                {artifact.filter ? ` · ${artifact.target ?? ""} · ${artifact.filter}` : ""} · {artifact.detail}
              </small>
            </div>
            <button
              type="button"
              className="btn icon"
              disabled={!workflow.nativeRuntime}
              title={artifact.path}
              aria-label={t("showInFinder", { name: artifact.name })}
              onClick={() => void workflow.revealOutput(artifact.path)}
            >
              <RevealIcon />
            </button>
          </article>
        ))}
      </div>
    </div>
  );
}
