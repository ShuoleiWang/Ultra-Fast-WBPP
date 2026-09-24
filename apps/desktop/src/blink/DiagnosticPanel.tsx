import { useEffect, useRef, useState } from "react";
import type { Translator } from "../i18n";
import type { BlinkFrame } from "../types";
import type { useWorkflow } from "../useWorkflow";
import { PreviewCanvas } from "./PreviewCanvas";

type Workflow = ReturnType<typeof useWorkflow>;
const number = (value: number | null | undefined) =>
  value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(2)}×`;

/** Auxiliary images load on demand and never retain a previous frame's pixels. */
function DiagnosticImage({
  path,
  label,
  workflow,
  t,
  size = [252, 168],
  onPaint,
  onFailure,
  pixelated = false,
  decoderId,
}: {
  path?: string | null;
  label: string;
  workflow: Workflow;
  t: Translator;
  size?: [number, number];
  onPaint?: () => void;
  onFailure?: (error: string | undefined) => void;
  pixelated?: boolean;
  decoderId?: string;
}) {
  const [image, setImage] = useState<{ path: string; sessionId?: string; url?: string; error?: string }>();
  const [attempt, setAttempt] = useState(0);
  const callback = useRef(onFailure);
  callback.current = onFailure;
  const sessionId = workflow.blinkSession?.manifest.sessionId;
  useEffect(() => {
    let active = true;
    if (!path) return;
    setImage(undefined);
    workflow
      .loadBlinkPreview(path)
      .then((url) => {
        if (active) setImage({ path, sessionId, url });
      })
      .catch((error) => {
        if (active) {
          setImage({ path, sessionId, error: String(error) });
          callback.current?.(String(error));
        }
      });
    return () => {
      active = false;
    };
  }, [path, sessionId, attempt, workflow.loadBlinkPreview]);
  const current = image?.path === path && image?.sessionId === sessionId ? image : undefined;
  if (!path) return <div className="blink-diag-unavailable">{t("blinkDiagUnavailable")}</div>;
  return (
    <div className="blink-diag-image" style={{ aspectRatio: `${size[0]}/${size[1]}` }}>
      {current?.url ? (
        <PreviewCanvas
          key={`${sessionId}:${path}`}
          src={current.url}
          label={label}
          width={size[0]}
          height={size[1]}
          pixelated={pixelated}
          decoderId={decoderId}
          className="blink-diag-canvas"
          onPaint={() => {
            callback.current?.(undefined);
            onPaint?.();
          }}
          onError={() => {
            setImage({ path, sessionId, error: "decode" });
            callback.current?.("decode");
          }}
        />
      ) : (
        <div role="status" className="blink-diag-unavailable">
          {current?.error ? t("blinkPreviewFailedShort") : t("blinkPreviewLoading")}
          {current?.error && (
            <button className="btn small" onClick={() => setAttempt((value) => value + 1)}>
              {t("blinkPreviewRetry")}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export function DiagnosticPanel({
  workflow,
  frame,
  t,
  compact = false,
}: {
  workflow: Workflow;
  frame: BlinkFrame;
  t: Translator;
  compact?: boolean;
}) {
  const diagnostic = frame.diagnostics;
  const [crop, setCrop] = useState<"nativeShape" | "nativeSignal">("nativeShape");
  const [expanded, setExpanded] = useState(false);
  const close = useRef<HTMLButtonElement>(null);
  const closeComparison = () => {
    setExpanded(false);
    document.querySelector<HTMLElement>(".blink-view")?.focus();
  };
  useEffect(() => {
    if (expanded) close.current?.focus();
  }, [expanded]);
  useEffect(() => {
    setExpanded(false);
  }, [frame.sourceSha256]);
  if (!diagnostic) return null;
  const reference = workflow.blinkSession?.manifest.frames.find((f) => f.channelId === frame.channelId && f.reference);
  const nativeViews = (large = false) => (
    <div className={`blink-native-pair ${large ? "large" : ""}`}>
      {[reference, frame].map((item, index) => (
        <figure key={index}>
          <figcaption>{index === 0 ? t("blinkReference") : t("blinkDiagCurrent")}</figcaption>
          <DiagnosticImage
            path={item?.previews.diagnostic?.[crop]}
            label={index === 0 ? t("blinkDiagReferenceCrop") : t("blinkDiagCurrentCrop")}
            workflow={workflow}
            t={t}
            size={[252, 252]}
            pixelated
          />
        </figure>
      ))}
    </div>
  );
  return (
    <section className={`blink-diagnostics ${compact ? "compact" : ""}`} aria-label={t("blinkDiagTitle")}>
      <dl className="blink-diag-facts">
        <div>
          <dt>{t("blinkDiagSignal")}</dt>
          <dd>{number(diagnostic.relativeSignal)}</dd>
        </div>
        <div>
          <dt>{t("blinkDiagNoise")}</dt>
          <dd>{number(diagnostic.relativeNoise)}</dd>
        </div>
        <div>
          <dt>{t("blinkDiagMatchedNoise")}</dt>
          <dd>{number(diagnostic.matchedSignalNoise)}</dd>
        </div>
      </dl>
      {diagnostic.calibration !== "calibrated" && <p className="blink-diag-warning">{t("blinkDiagUncalibrated")}</p>}
      <div className="blink-background-check">
        <h3>{t("blinkDiagBackground")}</h3>
        {frame.previews.diagnostic?.background ? (
          <DiagnosticImage
            path={frame.previews.diagnostic.background}
            label={t("blinkDiagBackground")}
            workflow={workflow}
            t={t}
            pixelated
            decoderId="blink-background-decoder"
            onPaint={() => workflow.markFrameViewed(frame.sourceSha256, "background")}
            onFailure={(error) => workflow.reportPreviewFailure(frame.sourceSha256, error, "background")}
          />
        ) : (
          <p className="blink-diag-warning">{t("blinkDiagBackgroundUnavailable")}</p>
        )}
        <small>{t("blinkDiagBackgroundHint")}</small>
      </div>
      <div className="blink-native-check">
        <h3>{t("blinkDiagNative")}</h3>
        {diagnostic.nativeStatus === "ready" ? (
          <>
            <select
              value={crop}
              aria-label={t("blinkDiagNativeMode")}
              onChange={(event) => setCrop(event.target.value as typeof crop)}
            >
              <option value="nativeShape">{t("blinkDiagShapeMode")}</option>
              <option value="nativeSignal">{t("blinkDiagSignalMode")}</option>
            </select>
            {nativeViews()}
            <small>
              {crop === "nativeShape"
                ? t("blinkDiagShapeHint", { count: diagnostic.shapeRegions })
                : t("blinkDiagSignalHint")}
            </small>
            <button type="button" className="btn small" onClick={() => setExpanded(true)}>
              {t("blinkDiagExpand")}
            </button>
          </>
        ) : (
          <p className="blink-diag-warning">{t("blinkDiagNativeUnavailable")}</p>
        )}
      </div>
      {expanded && (
        <div
          className="blink-native-modal"
          role="dialog"
          aria-modal="true"
          aria-label={t("blinkDiagNative")}
          onKeyDown={(event) => {
            event.stopPropagation();
            if (event.key === "Escape") closeComparison();
            if (event.key === "Tab") {
              event.preventDefault();
              close.current?.focus();
            }
          }}
        >
          <div className="blink-native-dialog">
            <header>
              <strong>{t("blinkDiagNative")}</strong>
              <button ref={close} type="button" className="btn" onClick={closeComparison}>
                {t("blinkDiagClose")}
              </button>
            </header>
            <p>
              {crop === "nativeShape"
                ? t("blinkDiagShapeHint", { count: diagnostic.shapeRegions })
                : t("blinkDiagSignalHint")}
            </p>
            {nativeViews(true)}
          </div>
        </div>
      )}
    </section>
  );
}
