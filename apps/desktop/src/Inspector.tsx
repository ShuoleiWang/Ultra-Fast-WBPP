import type { Translator } from "./i18n";
import { BlinkFrameDetails } from "./BlinkView";
import { InputChecks } from "./FrameInventory";
import type { InspectedLightQuality } from "./types";
import type { useWorkflow } from "./useWorkflow";

type Workflow = ReturnType<typeof useWorkflow>;
const basename = (path: string) => path.split(/[\\/]/).filter(Boolean).pop() ?? path;

export function dispositionLabel(disposition: InspectedLightQuality["disposition"], t: Translator): string {
  return disposition === "PASS" ? t("dispositionPass") : disposition === "REVIEW" ? t("dispositionReview") : t("dispositionFail");
}

/** The right pane: the selected frame's evidence, then the project's input checks. */
export function Inspector({ workflow, t, frame, blinkSelectedSha }: { workflow: Workflow; t: Translator; frame?: InspectedLightQuality; blinkSelectedSha?: string }) {
  const approved = Boolean(frame?.sourceSha256 && workflow.approvedReviewDigests.includes(frame.sourceSha256));
  const blinkFrame = workflow.step === "blink" ? workflow.blinkSession?.manifest.frames.find((item) => item.sourceSha256 === blinkSelectedSha) : undefined;
  const blinkDecision = blinkFrame ? workflow.decisions[blinkFrame.sourceSha256] ?? blinkFrame.defaultDecision : undefined;
  return <aside className="inspector" aria-label={t("inspectorLabel")}>
    <div className="ins-scroll">
      {workflow.step === "blink" && (blinkFrame
        ? <div className="ins-section">
            <div className="ins-title selectable" title={blinkFrame.path}>{blinkFrame.name}</div>
            <div className="ins-sub">{blinkFrame.night} · {blinkFrame.filter}{blinkFrame.reference ? ` · ★ ${t("blinkReference")}` : ""}</div>
            {blinkFrame.previews.filmstripDataUrl ? <img className="preview" src={blinkFrame.previews.filmstripDataUrl} alt={t("previewAlt", { name: blinkFrame.name })} /> : <div className="preview" role="img" aria-label={t("previewNone")} />}
            <div className="row-actions">
              <span className={`badge ${blinkDecision === "KEEP" ? "ok" : "stop"}`}>{blinkDecision === "KEEP" ? t("blinkKept") : t("blinkDropped")}</span>
              <button type="button" className="btn small" aria-pressed={blinkDecision === "KEEP"} onClick={() => workflow.setDecision(blinkFrame.sourceSha256, "KEEP")}>{t("blinkKeep")}</button>
              <button type="button" className="btn small" aria-pressed={blinkDecision === "DROP"} onClick={() => workflow.setDecision(blinkFrame.sourceSha256, "DROP")}>{t("blinkDrop")}</button>
            </div>
            <BlinkFrameDetails frame={blinkFrame} t={t} />
          </div>
        : <p className="ins-empty">{workflow.blinkSession ? t("inspectorEmpty") : t("blinkNoSession")}</p>)}
      {workflow.step === "inspect" && (frame
        ? <div className="ins-section">
            <div className="ins-title selectable" title={frame.path}>{basename(frame.path)}</div>
            <div className="ins-sub">{frame.starCount} {t("starsLabel")} · {frame.confidence}</div>
            {frame.previewDataUrl ? <img className="preview" src={frame.previewDataUrl} alt={t("previewAlt", { name: basename(frame.path) })} /> : <div className="preview" role="img" aria-label={t("previewNone")} />}
            <dl className="frm">
              <dt>{t("colDecision")}</dt><dd><span className={`badge ${frame.disposition === "PASS" ? "ok" : frame.disposition === "REVIEW" ? (approved ? "ok" : "check") : "stop"}`}>{frame.disposition === "REVIEW" && approved ? t("dispositionApproved") : dispositionLabel(frame.disposition, t)}</span></dd>
              <dt>{t("colStars")}</dt><dd>{frame.starCount}</dd>
              <dt>{t("colConfidence")}</dt><dd>{frame.confidence}</dd>
            </dl>
            {frame.evidence.length
              ? <ul className="ev">{frame.evidence.map((item, index) => <li key={`${item.code}-${index}`}><code>{item.code}</code><span>{item.message}</span></li>)}</ul>
              : <p className={`reason ${frame.disposition === "PASS" ? "ok" : ""}`}>{frame.summary || t("noReviewEvidence")}</p>}
          </div>
        : <p className="ins-empty">{t("inspectorEmpty")}</p>)}
      <InputChecks workflow={workflow} t={t} />
    </div>
  </aside>;
}
