import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  type SVGProps,
  type WheelEvent as ReactWheelEvent,
} from "react";
import { blinkFlagHint, blinkFlagLabel, type Translator } from "./i18n";
import { FolderIcon, PlayIcon } from "./icons";
import { DiagnosticPanel } from "./blink/DiagnosticPanel";
import { PreviewCanvas } from "./blink/PreviewCanvas";
import type { BlinkDecision, BlinkFlag, BlinkFrame } from "./types";
import type { BlinkChannelSummary, useWorkflow } from "./useWorkflow";

type Workflow = ReturnType<typeof useWorkflow>;
type Fps = 2 | 4 | 8;
const FPS_STEPS: Fps[] = [2, 4, 8];
/** On-demand 1/4-scale images kept in memory (≈ 1–1.5 MB each). */
const ZOOM_CACHE_ENTRIES = 12;
/** On-demand 1/8-scale images beyond the controller's inline budget. */
const FILMSTRIP_CACHE_ENTRIES = 256;
/** The 1/8 image is replaced by the 1/4 one when zoomed in beyond this factor over fit. */
const ZOOM_IMAGE_FACTOR = 1.2;
const MIN_SCALE_FACTOR = 0.5;
const MAX_SCALE = 8;
/** A press of C shorter than this toggles compare mode; a longer hold is an A/B with the reference. */
const HOLD_MS = 300;

const fmt = (value: number | null | undefined, digits = 2, suffix = "") =>
  value === null || value === undefined || !Number.isFinite(value) ? "—" : `${value.toFixed(digits)}${suffix}`;
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));
const isExcluded = (frame: BlinkFrame) => frame.flags.some((flag) => flag.severity === "EXCLUDE");
/** EXCLUDE first, then ATTENTION, then clean. */
const severityRank = (frame: BlinkFrame) => (isExcluded(frame) ? 0 : frame.flags.length ? 1 : 2);
const byTime = (a: BlinkFrame, b: BlinkFrame) =>
  a.night.localeCompare(b.night) || (a.observedAt ?? "").localeCompare(b.observedAt ?? "") || a.index - b.index;
/** Filmstrip order: flagged first (EXCLUDE, ATTENTION, clean; nights and times inside), or purely chronological. */
const filmstripOrder = (frames: BlinkFrame[], chronological: boolean) =>
  chronological
    ? [...frames].sort(byTime)
    : [...frames].sort((a, b) => severityRank(a) - severityRank(b) || byTime(a, b));
/** The number a flag was set on, in the unit the user reads it in. */
function flagValueText(flag: BlinkFlag): string {
  const value = flag.value;
  if (value === null || value === undefined || !Number.isFinite(value)) return "";
  switch (flag.code) {
    case "BLINK_SKY_BRIGHT":
    case "BLINK_FWHM_WIDE":
    case "BLINK_GRADIENT_AMPLITUDE":
      return `×${value.toFixed(2)}`;
    case "BLINK_SOURCES_LOW":
      return `${Math.round(value * 100)} %`;
    case "BLINK_EXTINCTION":
      return `${value.toFixed(2)} mag`;
    case "BLINK_BACKGROUND_SHAPE":
      return `${value.toFixed(2)} σ`;
    case "BLINK_FEW_STARS":
      return String(Math.round(value));
    default:
      return value.toFixed(2);
  }
}

/** Two frames blinking: the sidebar's symbol for the view. */
export const BlinkIcon = (props: SVGProps<SVGSVGElement>) => (
  <svg
    width={16}
    height={16}
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.5}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    {...props}
  >
    <rect x="1.8" y="4.2" width="9" height="7.2" rx="1.4" />
    <path d="M5.2 4.2V3.4A1.4 1.4 0 0 1 6.6 2h6.2a1.4 1.4 0 0 1 1.4 1.4v5.8a1.4 1.4 0 0 1-1.4 1.4h-1.6" />
    <circle cx="6.3" cy="7.8" r="1" fill="currentColor" stroke="none" />
  </svg>
);

interface ViewState {
  scale: number;
  x: number;
  y: number;
}
interface Size {
  width: number;
  height: number;
}

/**
 * Bounded caches of preview data URLs, keyed by the manifest's relative path:
 * the 1/8 images beyond the controller's inline budget and the 1/4 images
 * loaded when zooming in.  Oldest entries are evicted.
 */
function usePreviewCache(loader: (relativePath: string) => Promise<string>, sessionId: string | undefined) {
  const state = useRef({
    sessionId,
    zoom: new Map<string, string>(),
    filmstrip: new Map<string, string>(),
    pending: new Set<string>(),
    errors: new Map<string, string>(),
    bypassInline: new Set<string>(),
  });
  const [version, bump] = useReducer((value: number) => value + 1, 0);
  if (state.current.sessionId !== sessionId)
    state.current = {
      sessionId,
      zoom: new Map(),
      filmstrip: new Map(),
      pending: new Set(),
      errors: new Map(),
      bypassInline: new Set(),
    };
  const get = useCallback(
    (kind: "zoom" | "filmstrip", path: string | null | undefined) => (path ? state.current[kind].get(path) : undefined),
    [],
  );
  const ensure = useCallback(
    (kind: "zoom" | "filmstrip", path: string | null | undefined) => {
      const owner = state.current;
      if (!path || owner[kind].has(path) || owner.pending.has(path) || owner.errors.has(path)) return;
      owner.pending.add(path);
      loader(path)
        .then((url) => {
          if (state.current !== owner) return;
          owner[kind].set(path, url);
          const limit = kind === "zoom" ? ZOOM_CACHE_ENTRIES : FILMSTRIP_CACHE_ENTRIES;
          while (owner[kind].size > limit) owner[kind].delete(owner[kind].keys().next().value as string);
        })
        .catch((error) => {
          if (state.current === owner) owner.errors.set(path, String(error));
        })
        .finally(() => {
          if (state.current === owner) {
            owner.pending.delete(path);
            bump();
          }
        });
    },
    [loader],
  );
  const fail = useCallback((path: string | null | undefined) => {
    if (path && !state.current.errors.has(path)) {
      state.current.errors.set(path, "decode");
      state.current.zoom.delete(path);
      state.current.filmstrip.delete(path);
      state.current.bypassInline.add(path);
      bump();
    }
  }, []);
  const retry = useCallback(
    (kind: "zoom" | "filmstrip", path: string | null | undefined) => {
      if (!path) return;
      state.current.errors.delete(path);
      state.current[kind].delete(path);
      state.current.bypassInline.add(path);
      bump();
      ensure(kind, path);
    },
    [ensure],
  );
  return useMemo(
    () => ({
      get,
      ensure,
      fail,
      retry,
      version,
      error: (path: string | null | undefined) => (path ? state.current.errors.get(path) : undefined),
      inlineAllowed: (path: string | null | undefined) => !path || !state.current.bypassInline.has(path),
    }),
    [get, ensure, fail, retry, version],
  );
}

function FilmstripImage({
  frame,
  url,
  error,
  ensure,
  onError,
  t,
}: {
  frame: BlinkFrame;
  url?: string;
  error?: string;
  ensure: () => void;
  onError: () => void;
  t: Translator;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (url || error) return;
    const element = ref.current;
    if (!element || typeof IntersectionObserver === "undefined") {
      ensure();
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) ensure();
      },
      { rootMargin: "200px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [url, error, ensure]);
  return (
    <span ref={ref} className="blink-tile-image">
      {url ? (
        <PreviewCanvas src={url} label="" width={96} height={64} onError={onError} className="blink-thumb-canvas" />
      ) : (
        <span className="blink-tile-empty" title={error ?? frame.previews.error ?? undefined}>
          {error || frame.previews.error ? t("blinkPreviewFailedShort") : t("blinkPreviewLoading")}
        </span>
      )}
    </span>
  );
}

/** The stage's size, so the fit scale follows the window and the inspector. */
function useElementSize<T extends HTMLElement>(): [RefObject<T | null>, Size] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const measure = () => {
      const rect = element.getBoundingClientRect();
      setSize((current) =>
        current.width === rect.width && current.height === rect.height
          ? current
          : { width: rect.width, height: rect.height },
      );
    };
    measure();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", measure);
      return () => window.removeEventListener("resize", measure);
    }
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, size];
}

/** The measurements, flags, gate evidence and notes of one frame (inspector section or the strip under the stage). */
export function BlinkFrameDetails({
  frame,
  t,
  compact = false,
}: {
  frame: BlinkFrame;
  t: Translator;
  compact?: boolean;
}) {
  const m = frame.metrics;
  const rows: Array<[string, string]> = [
    [t("blinkMetricSky"), `${fmt(m.sky, 0)} · ×${fmt(m.skyRatio)}`],
    [
      t("blinkMetricStars"),
      `${m.starCount ?? "—"} · ${m.sourceRatio === null ? "—" : `${Math.round(m.sourceRatio * 100)} %`}`,
    ],
    [t("blinkMetricExtinction"), fmt(m.extinctionMag, 2, " mag")],
    [t("blinkMetricFwhm"), `${fmt(m.fwhmNative)} px · ×${fmt(m.fwhmRatio)}`],
    [t("blinkMetricEllipticity"), fmt(m.ellipticity)],
    [
      t("blinkMetricRegistration"),
      frame.normalization.registered
        ? `${fmt(m.registrationRms)} px · ${m.matchedStars ?? "—"}`
        : t("blinkUnregistered"),
    ],
    [t("blinkMetricOverlap"), m.overlap === null ? "—" : `${Math.round(m.overlap * 100)} %`],
    [t("blinkMetricBackgroundShape"), fmt(m.backgroundShape, 2, " σ")],
    ...(m.gradientRatio === null ? [] : [[t("blinkMetricGradient"), `×${fmt(m.gradientRatio)}`] as [string, string]]),
    [
      t("blinkMetricScore"),
      frame.score.rank === null ? "—" : t("blinkMetricRank", { rank: frame.score.rank, z: fmt(frame.score.z, 1) }),
    ],
    [
      t("blinkMetricNormalization"),
      t("blinkNormalizationValue", {
        sky: fmt(frame.normalization.skyOffset, 0),
        scale: fmt(frame.normalization.fluxScale, 3),
      }),
    ],
    [
      t("blinkMetricGate"),
      `${frame.gate.disposition}${frame.gate.codes.length ? ` · ${frame.gate.codes.join(", ")}` : ""}`,
    ],
  ];
  return (
    <div className={`blink-details ${compact ? "compact" : ""}`}>
      <dl className="frm blink-frm">
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      {frame.flags.length > 0 && (
        <ul className="blink-flag-list" aria-label={t("blinkMetricFlags")}>
          {frame.flags.map((flag) => (
            <li key={flag.code} title={flag.message}>
              <span className={`badge ${flag.severity === "EXCLUDE" ? "stop" : "warn"}`}>
                {blinkFlagLabel(flag.code, t)} {flagValueText(flag)}
              </span>
              <span className="blink-flag-why">
                {flag.threshold !== null && flag.threshold !== undefined
                  ? `${t("blinkThreshold", { threshold: flag.threshold })} · `
                  : ""}
                {flag.severity === "EXCLUDE" ? t("blinkSeverityExclude") : t("blinkSeverityAttention")}
                {flag.combined ? ` · ${t("blinkCombinedRule")}` : ""}
              </span>
              {!compact && <span className="blink-flag-hint">{blinkFlagHint(flag.code, t)}</span>}
            </li>
          ))}
        </ul>
      )}
      {frame.notes.length > 0 && !compact && (
        <ul className="ev blink-notes" aria-label={t("blinkMetricNotes")}>
          {frame.notes.map((note, index) => (
            <li key={index}>
              <code>{t("blinkMetricNotes")}</code>
              <span>{note}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** The blink launch bar: output folder, kept counts, the blockers of the selection, back and start. */
function BlinkLaunchBar({ workflow, t }: { workflow: Workflow; t: Translator }) {
  const kept = workflow.blinkChannels.reduce((sum, channel) => sum + channel.kept, 0);
  const total = workflow.blinkChannels.reduce((sum, channel) => sum + channel.total, 0);
  const blocked = !workflow.canStart && !workflow.demoMode;
  return (
    <footer className="launchbar blink-launchbar">
      {blocked && (
        <details className="blockers blink-launch-details">
          <summary>{t("blockers")}</summary>
          <ul>
            {!workflow.blinkReviewComplete && <li>{t("blockerBlinkReview")}</li>}
            {workflow.insufficientBlinkPanels.map((channel) => (
              <li key={channel.channelId}>
                {t("blockerBlinkChannel", {
                  target: channel.target,
                  filter: channel.filter,
                  kept: channel.kept,
                  total: channel.total,
                  required: workflow.minimumKeptPerChannel,
                })}
              </li>
            ))}
            {!workflow.capabilities?.available && (
              <li>{workflow.capabilities?.unavailableReason ?? t("blockerEngine")}</li>
            )}
            {!workflow.calibrationReady && <li>{t("blockerCalibration")}</li>}
            {!workflow.allRequiredConfirmed && <li>{t("blockerTypes")}</li>}
            {!workflow.solverSetupReady && <li>{t("blockerSolver")}</li>}
            {!workflow.outputParent && <li>{t("blockerOutput")}</li>}
            {!workflow.masterOverridesReady && <li>{t("blockerMaster")}</li>}
            {workflow.cfaBlockedAssets.length > 0 && <li>{t("blockerCfaUnknownPattern")}</li>}
          </ul>
        </details>
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
        <span className="muted blink-kept-summary">
          {workflow.blinkChannels.map((channel) => `${channel.filter} ${channel.kept}/${channel.total}`).join(" · ")}
        </span>
        <button type="button" className="btn quiet" onClick={() => workflow.setStep("import")}>
          {t("back")}
        </button>
        <button
          type="button"
          className="btn primary"
          disabled={!workflow.canStart}
          onClick={() => void workflow.startRun()}
        >
          <PlayIcon />
          {workflow.blinkReady ? t("startSelected", { kept, total }) : t("start")}
        </button>
      </div>
    </footer>
  );
}

/**
 * Blink-style screening of one channel at a time: every frame registered and
 * normalised to the channel's reference, reviewed in chronological order,
 * one shared zoom/pan for all frames, playback, compare with the reference,
 * and per-frame / per-night decisions with undo.
 */
export function BlinkView({
  workflow,
  t,
  inspectorOpen,
  selectedSha,
  setSelectedSha,
}: {
  workflow: Workflow;
  t: Translator;
  inspectorOpen: boolean;
  selectedSha?: string;
  setSelectedSha: (sha: string | undefined) => void;
}) {
  const session = workflow.blinkSession;
  const channels = workflow.blinkChannels;
  const channel: BlinkChannelSummary | undefined =
    channels.find((item) => item.channelId === workflow.blinkChannel) ?? channels[0];
  const decisions = workflow.decisions;
  const [chronological, setChronological] = useState(true);
  const [keptOnly, setKeptOnly] = useState(false);
  const [fps, setFps] = useState<Fps>(2);
  // Review begins paused; playback advances only after the current image loads.
  const [playing, setPlaying] = useState(false);
  const [loadedFrame, setLoadedFrame] = useState<string>();
  const [compare, setCompare] = useState(false);
  const [displayMode, setDisplayMode] = useState<"detail" | "field">("detail");
  useEffect(() => {
    setLoadedFrame(undefined);
    setPlaying(false);
  }, [displayMode]);
  const [holdReference, setHoldReference] = useState(false);
  const holdStartedRef = useRef<number | undefined>(undefined);
  const [view, setView] = useState<ViewState>();
  const anchorRef = useRef<string | undefined>(undefined);
  const dragRef = useRef<{ pointerId: number; x: number; y: number; viewX: number; viewY: number } | undefined>(
    undefined,
  );
  const [stageRef, stageSize] = useElementSize<HTMLDivElement>();
  const containerRef = useRef<HTMLElement>(null);
  const tileRefs = useRef(new Map<string, HTMLButtonElement>());
  const cache = usePreviewCache(workflow.loadBlinkPreview, session?.manifest.sessionId);

  const order = useMemo(() => filmstripOrder(channel?.frames ?? [], chronological), [channel, chronological]);
  const groups = useMemo(() => {
    const result: Array<{ key: string; night: string; frames: BlinkFrame[] }> = [];
    for (const frame of order) {
      const last = result[result.length - 1];
      if (last && last.night === frame.night) last.frames.push(frame);
      else result.push({ key: `${frame.night}-${result.length}`, night: frame.night, frames: [frame] });
    }
    return result;
  }, [order]);
  const currentIndex = Math.max(
    0,
    order.findIndex((frame) => frame.sourceSha256 === selectedSha),
  );
  const current: BlinkFrame | undefined = order[currentIndex];
  const reference = channel?.frames.find((frame) => frame.reference);
  const latest = useRef({ order, decisions, keptOnly, currentIndex });
  latest.current = { order, decisions, keptOnly: keptOnly && channel?.viewed === channel?.total, currentIndex };

  // The keys work as soon as the view opens (the button that opened it is gone).
  useEffect(() => {
    containerRef.current?.focus({ preventScroll: true });
  }, [session?.manifest.sessionId]);
  // The inspector follows the frame on the stage.
  useEffect(() => {
    if (current && current.sourceSha256 !== selectedSha) setSelectedSha(current.sourceSha256);
  }, [current, selectedSha, setSelectedSha]);
  // A channel change resets the shared zoom (channels differ in geometry).
  useEffect(() => {
    setView(undefined);
  }, [channel?.channelId]);
  useEffect(() => {
    const element = tileRefs.current.get(current?.sourceSha256 ?? "");
    if (element && typeof element.scrollIntoView === "function")
      element.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [current?.sourceSha256]);

  const select = useCallback(
    (sha: string | undefined) => {
      if (sha) setSelectedSha(sha);
    },
    [setSelectedSha],
  );
  const nextIndex = useCallback((from: number, delta: number, skipDropped: boolean): number => {
    const { order: frames, decisions: current } = latest.current;
    if (!frames.length) return 0;
    let index = from;
    for (let step = 0; step < frames.length; step += 1) {
      index = (index + delta + frames.length) % frames.length;
      if (!skipDropped || current[frames[index].sourceSha256] === "KEEP") return index;
    }
    return from;
  }, []);
  const stepBy = useCallback(
    (delta: number, fromPlayback = false) => {
      const { order: frames, currentIndex: from, keptOnly: onlyKept } = latest.current;
      if (!fromPlayback) setPlaying(false);
      const next = nextIndex(from, delta, fromPlayback && onlyKept);
      if (fromPlayback && next <= from) {
        setPlaying(false);
        return;
      }
      select(frames[next]?.sourceSha256);
    },
    [nextIndex, select],
  );
  const jumpFlagged = useCallback(
    (delta: number) => {
      const { order: frames, currentIndex: from } = latest.current;
      setPlaying(false);
      let index = from;
      for (let step = 0; step < frames.length; step += 1) {
        index = (index + delta + frames.length) % frames.length;
        if (frames[index].flags.length && !frames[index].reference) {
          select(frames[index].sourceSha256);
          return;
        }
      }
    },
    [select],
  );

  useEffect(() => {
    if (
      !playing ||
      order.length < 2 ||
      loadedFrame !== current?.sourceSha256 ||
      !workflow.viewedFrames[current.sourceSha256] ||
      document.hidden
    )
      return;
    const timer = window.setTimeout(() => stepBy(1, true), 1000 / fps);
    return () => window.clearTimeout(timer);
  }, [fps, order.length, playing, stepBy, loadedFrame, current?.sourceSha256, workflow.viewedFrames]);
  useEffect(() => {
    const pause = () => {
      if (document.hidden) setPlaying(false);
    };
    document.addEventListener("visibilitychange", pause);
    return () => document.removeEventListener("visibilitychange", pause);
  }, []);
  // Geometry: the stage works in the 1/4-scale image's pixel grid; the 1/8
  // image is stretched into the same box, so one transform serves both.
  const imageWidth = Math.max(1, channel?.previewGeometry?.zoom?.[0] || 1563);
  const imageHeight = Math.max(1, channel?.previewGeometry?.zoom?.[1] || 1044);
  const paneWidth = Math.max(1, (stageSize.width || 960) / (compare ? 2 : 1));
  const paneHeight = Math.max(1, stageSize.height || 560);
  const fitScale = Math.min(paneWidth / imageWidth, paneHeight / imageHeight);
  const fitView = (scale: number): ViewState => ({
    scale,
    x: (paneWidth - imageWidth * scale) / 2,
    y: (paneHeight - imageHeight * scale) / 2,
  });
  const effective: ViewState = view ?? fitView(fitScale);
  const zoomFactor = effective.scale / fitScale;
  const zoomBy = (factor: number, cx = paneWidth / 2, cy = paneHeight / 2) => {
    const scale = clamp(effective.scale * factor, fitScale * MIN_SCALE_FACTOR, MAX_SCALE);
    const ratio = scale / effective.scale;
    setView({ scale, x: cx - (cx - effective.x) * ratio, y: cy - (cy - effective.y) * ratio });
  };
  const onWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    zoomBy(Math.exp(-event.deltaY * 0.0015), event.clientX - rect.left, event.clientY - rect.top);
  };
  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    dragRef.current = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      viewX: effective.x,
      viewY: effective.y,
    };
    if (typeof event.currentTarget.setPointerCapture === "function")
      event.currentTarget.setPointerCapture(event.pointerId);
  };
  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    setView({ scale: effective.scale, x: drag.viewX + event.clientX - drag.x, y: drag.viewY + event.clientY - drag.y });
  };
  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = undefined;
  };

  // Which image the stage shows: the inline 1/8 preview, or the on-demand 1/4
  // image once zoomed in (never while playing: the loop stays cheap).
  const wantZoom = !playing && zoomFactor > ZOOM_IMAGE_FACTOR;
  const filmstripUrl = useCallback(
    (frame: BlinkFrame | undefined) =>
      frame && !cache.error(frame.previews.filmstrip)
        ? (cache.get("filmstrip", frame.previews.filmstrip) ??
          (cache.inlineAllowed(frame.previews.filmstrip) ? (frame.previews.filmstripDataUrl ?? undefined) : undefined))
        : undefined,
    [cache],
  );
  const stageUrl = (frame: BlinkFrame | undefined) =>
    frame && displayMode === "field" && frame.previews.diagnostic?.field
      ? cache.get("zoom", frame.previews.diagnostic.field)
      : frame
        ? ((wantZoom ? cache.get("zoom", frame.previews.zoom) : undefined) ?? filmstripUrl(frame))
        : undefined;
  useEffect(() => {
    if (!current) return;
    if (!current.previews.filmstripDataUrl) cache.ensure("filmstrip", current.previews.filmstrip);
    if (wantZoom && displayMode === "detail") cache.ensure("zoom", current.previews.zoom);
    if (displayMode === "field") {
      cache.ensure("zoom", current.previews.diagnostic?.field);
      cache.ensure("zoom", reference?.previews.diagnostic?.field);
    }
    if (reference && !reference.previews.filmstripDataUrl) cache.ensure("filmstrip", reference.previews.filmstrip);
    if (wantZoom && displayMode === "detail" && reference && (compare || holdReference))
      cache.ensure("zoom", reference.previews.zoom);
    // Prefetch the neighbours in play order and the visible tiles around the frame.
    for (const delta of [1, 2, -1, 3, 4, 5, 6, -2, -3, -4, -5, -6]) {
      const frame = order[(currentIndex + delta + order.length) % order.length];
      if (frame && !frame.previews.filmstripDataUrl) cache.ensure("filmstrip", frame.previews.filmstrip);
    }
  }, [cache, compare, current, currentIndex, holdReference, order, reference, wantZoom, displayMode]);

  useEffect(() => {
    if (!current) return;
    const error =
      current.previews.error ??
      cache.error(displayMode === "field" ? current.previews.diagnostic?.field : current.previews.filmstrip) ??
      (!current.previews.filmstrip && !current.previews.filmstripDataUrl ? t("blinkPreviewFailed") : undefined);
    workflow.reportPreviewFailure(current.sourceSha256, error || undefined);
  }, [current, cache, workflow.reportPreviewFailure, t, displayMode]);
  const decisionOf = (frame: BlinkFrame): BlinkDecision => decisions[frame.sourceSha256] ?? "KEEP";
  const toggleRange = (frame: BlinkFrame, shift: boolean) => {
    const anchor = order.findIndex((item) => item.sourceSha256 === anchorRef.current);
    const index = order.indexOf(frame);
    if (shift && anchor >= 0 && anchor !== index) {
      const [low, high] = anchor < index ? [anchor, index] : [index, anchor];
      workflow.setDecisionsBulk(
        order.slice(low, high + 1).map((item) => item.sourceSha256),
        decisionOf(frame) === "KEEP" ? "DROP" : "KEEP",
      );
    }
    anchorRef.current = frame.sourceSha256;
    setPlaying(false);
    select(frame.sourceSha256);
  };
  const selectChannel = (index: number) => {
    const target = channels[index];
    if (target) {
      setPlaying(false);
      workflow.selectBlinkChannel(target.channelId);
      select(filmstripOrder(target.frames, chronological)[0]?.sourceSha256);
    }
  };

  const decideAndNext = (decision: BlinkDecision) => {
    if (
      !current ||
      (decision === "KEEP" &&
        (!workflow.viewedFrames[current.sourceSha256] || workflow.previewFailures[current.sourceSha256]))
    )
      return;
    workflow.setDecision(current.sourceSha256, decision);
    setPlaying(false);
    if (currentIndex + 1 < order.length) select(order[currentIndex + 1].sourceSha256);
  };
  const togglePlayback = () => {
    if (!playing && currentIndex === order.length - 1) select(order[0]?.sourceSha256);
    setPlaying((value) => !value);
  };
  const onKeyDown = (event: ReactKeyboardEvent<HTMLElement>) => {
    const target = event.target as HTMLElement;
    if (target.closest("input, select, textarea, [contenteditable=true]")) return;
    const key = event.key;
    if (event.metaKey || event.ctrlKey) {
      if (key.toLowerCase() === "z") {
        event.preventDefault();
        workflow.undo();
      }
      return;
    }
    const inButton = Boolean(target.closest("button"));
    const handled = () => event.preventDefault();
    switch (key) {
      case "ArrowRight":
        handled();
        stepBy(1);
        return;
      case "ArrowLeft":
        handled();
        stepBy(-1);
        return;
      case "Home":
        handled();
        setPlaying(false);
        select(order[0]?.sourceSha256);
        return;
      case "End":
        handled();
        setPlaying(false);
        select(order[order.length - 1]?.sourceSha256);
        return;
      case " ":
        if (inButton) return;
        handled();
        if (current) workflow.toggleDecision(current.sourceSha256);
        return;
      case "Escape":
        handled();
        setPlaying(false);
        return;
      case "[":
        handled();
        setFps((value) => FPS_STEPS[Math.max(0, FPS_STEPS.indexOf(value) - 1)]);
        return;
      case "]":
        handled();
        setFps((value) => FPS_STEPS[Math.min(FPS_STEPS.length - 1, FPS_STEPS.indexOf(value) + 1)]);
        return;
      case "+":
      case "=":
        handled();
        zoomBy(1.25);
        return;
      case "-":
      case "_":
        handled();
        zoomBy(0.8);
        return;
      case "0":
        handled();
        setView(undefined);
        return;
      case "1":
      case "2":
      case "3":
      case "4":
        handled();
        selectChannel(Number(key) - 1);
        return;
      default:
        break;
    }
    switch (key.toLowerCase()) {
      case "k":
        handled();
        decideAndNext("KEEP");
        return;
      case "d":
        handled();
        decideAndNext("DROP");
        return;
      case "f":
        handled();
        jumpFlagged(event.shiftKey ? -1 : 1);
        return;
      case "r":
        handled();
        if (reference) {
          setPlaying(false);
          select(reference.sourceSha256);
        }
        return;
      case "c":
        handled();
        if (!event.repeat) {
          holdStartedRef.current = performance.now();
          setHoldReference(true);
        }
        return;
      case "p":
        handled();
        togglePlayback();
        return;
      case "n":
        handled();
        if (current && channel) workflow.setNightDecision(channel.channelId, current.night, "DROP");
        return;
      case "z":
        handled();
        workflow.undo();
        return;

      default:
        return;
    }
  };
  const onKeyUp = (event: ReactKeyboardEvent<HTMLElement>) => {
    if (event.key.toLowerCase() !== "c") return;
    const started = holdStartedRef.current;
    holdStartedRef.current = undefined;
    setHoldReference(false);
    if (started !== undefined && performance.now() - started < HOLD_MS) setCompare((value) => !value);
  };

  if (!session || !channel) {
    return (
      <section className="view" aria-labelledby="blink-title">
        <div className="chead">
          <h1 id="blink-title">{t("blinkTitle")}</h1>
        </div>
        <div className="scroll">
          <div className="stack">
            <p className="table-empty">{t("blinkNoSession")}</p>
            <div className="row-actions">
              <button type="button" className="btn quiet" onClick={() => workflow.setStep("import")}>
                {t("back")}
              </button>
            </div>
          </div>
        </div>
      </section>
    );
  }

  const alt = (frame: BlinkFrame) =>
    t("blinkStageAlt", {
      name: frame.name,
      decision: decisionOf(frame) === "KEEP" ? t("blinkKept") : t("blinkDropped"),
      flags: frame.flags.length
        ? frame.flags.map((flag) => blinkFlagLabel(flag.code, t)).join(", ")
        : t("blinkNoFlags"),
    });
  const pane = (frame: BlinkFrame | undefined, key: string) => {
    const url = stageUrl(frame);
    const currentPath =
      displayMode === "field" && frame?.previews.diagnostic?.field
        ? frame.previews.diagnostic.field
        : wantZoom && frame && cache.get("zoom", frame.previews.zoom)
          ? frame.previews.zoom
          : frame?.previews.filmstrip;
    const error = frame
      ? (frame.previews.error ??
        cache.error(currentPath) ??
        (!frame.previews.filmstrip && !frame.previews.filmstripDataUrl ? t("blinkPreviewFailed") : undefined))
      : undefined;
    const loaded = () => {
      if (!frame) return;
      if (frame === current) setLoadedFrame(frame.sourceSha256);
      if (!document.hidden) workflow.markFrameViewed(frame.sourceSha256);
    };
    return (
      <div key={key} className="blink-pane">
        {frame && url && (
          <PreviewCanvas
            key={`${frame.sourceSha256}:${displayMode}:${wantZoom ? "zoom" : "filmstrip"}`}
            src={url}
            label={alt(frame)}
            width={paneWidth}
            height={paneHeight}
            imageWidth={imageWidth}
            imageHeight={imageHeight}
            view={effective}
            onPaint={loaded}
            decoderId="blink-stage-decoder"
            className="blink-stage-canvas"
            onError={() => {
              setLoadedFrame(undefined);
              cache.fail(currentPath);
            }}
          />
        )}
        {!url && (
          <div className="blink-load-state" role="status">
            <strong>{error || frame?.previews.error ? t("blinkPreviewFailed") : t("blinkPreviewLoading")}</strong>
            {frame && (error || frame.previews.error) && (
              <>
                <span>{t("blinkPreviewFailureHint")}</span>
                <button
                  type="button"
                  className="btn small"
                  onClick={() => cache.retry(displayMode === "field" ? "zoom" : "filmstrip", currentPath)}
                >
                  {t("blinkPreviewRetry")}
                </button>
              </>
            )}
          </div>
        )}
        {frame && (
          <div className="blink-overlay">
            <span className="blink-name selectable" title={frame.path}>
              {frame.name}
            </span>
            <span className="blink-sub">
              {frame.night} · {t("blinkFrameOf", { index: order.indexOf(frame) + 1, count: order.length })}
            </span>
            <span className="blink-chips">
              <span
                className={`badge ${workflow.viewedFrames[frame.sourceSha256] ? (decisionOf(frame) === "KEEP" ? "ok" : "stop") : "neutral"}`}
              >
                {!workflow.viewedFrames[frame.sourceSha256]
                  ? t("blinkUnreviewed")
                  : decisionOf(frame) === "KEEP"
                    ? t("blinkKept")
                    : t("blinkDropped")}
              </span>
              {frame.reference && <span className="badge neutral">★ {t("blinkReference")}</span>}
              {frame.flags.map((flag) => (
                <span
                  key={flag.code}
                  className={`badge ${flag.severity === "EXCLUDE" ? "stop" : "warn"}`}
                  title={`${flag.message}${blinkFlagHint(flag.code, t) ? `\n${blinkFlagHint(flag.code, t)}` : ""}`}
                >
                  {blinkFlagLabel(flag.code, t)} {flagValueText(flag)}
                </span>
              ))}
              {!frame.normalization.registered && <span className="badge check">{t("blinkUnregistered")}</span>}
            </span>
          </div>
        )}
      </div>
    );
  };
  const showReference = holdReference && reference && !compare;
  const keptTotal = channels.reduce((sum, item) => sum + item.kept, 0);
  const frameTotal = channels.reduce((sum, item) => sum + item.total, 0);

  return (
    <section
      ref={containerRef}
      className="view blink-view"
      aria-labelledby="blink-title"
      tabIndex={0}
      onKeyDown={onKeyDown}
      onKeyUp={onKeyUp}
    >
      <div className="chead">
        <h1 id="blink-title">
          {t("blinkTitle")}{" "}
          <span className="muted">{t("blinkKeptSummary", { kept: keptTotal, total: frameTotal })}</span>
        </h1>
        <div className="chead-row">
          <div className="chips" role="group" aria-label={t("blinkChannelLabel")}>
            {channels.map((item, index) => (
              <button
                type="button"
                key={item.channelId}
                className="chip"
                aria-pressed={item.channelId === channel.channelId}
                title={`${item.target} · ${item.filter}`}
                onClick={() => selectChannel(index)}
              >
                <span
                  className={`dot ${item.kept < workflow.minimumKeptPerChannel ? "stop" : item.flagged ? "warn" : "ok"}`}
                />
                {item.filter} · {t("blinkViewedCount", { viewed: item.viewed, total: item.total })}
                {item.confirmed ? " ✓" : ""}
              </button>
            ))}
          </div>
          <span className="spacer" />
          <div className="blink-controls">
            <button
              type="button"
              className="btn small"
              aria-pressed={playing}
              aria-label={playing ? t("blinkPause") : t("blinkPlay")}
              title={playing ? t("blinkPause") : t("blinkPlay")}
              onClick={togglePlayback}
            >
              {playing ? "⏸" : "▶"}
            </button>
            <select
              className="blink-fps"
              aria-label={t("blinkFpsLabel")}
              value={fps}
              onChange={(event) => setFps(Number(event.target.value) as Fps)}
            >
              {FPS_STEPS.map((value) => (
                <option key={value} value={value}>
                  {value} fps
                </option>
              ))}
            </select>
            <label className="blink-check">
              <input
                type="checkbox"
                disabled={channel.viewed !== channel.total}
                checked={keptOnly}
                onChange={(event) => setKeptOnly(event.target.checked)}
              />
              {t("blinkKeptOnly")}
            </label>
            {current?.diagnostics && (
              <select
                value={displayMode}
                aria-label={t("blinkDiagDisplayMode")}
                className="blink-fps"
                onChange={(event) => setDisplayMode(event.target.value as typeof displayMode)}
              >
                <option value="detail">{t("blinkDiagDetailMode")}</option>
                <option value="field">{t("blinkDiagFieldMode")}</option>
              </select>
            )}
            <button
              type="button"
              className="btn small"
              aria-pressed={compare}
              onClick={() => setCompare((value) => !value)}
            >
              {t("blinkCompare")}
            </button>
            <button
              type="button"
              className="btn small"
              aria-pressed={chronological}
              title={t("blinkOrderLabel")}
              onClick={() => setChronological((value) => !value)}
            >
              {chronological ? t("blinkChronological") : t("blinkFlaggedFirst")}
            </button>

            <button type="button" className="btn small" disabled={!workflow.canUndo} onClick={() => workflow.undo()}>
              {t("blinkUndo")}
            </button>
            <span className="seg blink-zoom" role="group" aria-label={t("blinkZoomLabel")}>
              <button type="button" aria-pressed={view === undefined} onClick={() => setView(undefined)}>
                {t("blinkZoomFit")}
              </button>
              <button
                type="button"
                aria-pressed={view !== undefined && Math.abs(effective.scale - 1) < 1e-6}
                onClick={() => setView(fitView(1))}
              >
                {t("blinkZoomOne")}
              </button>
              <button
                type="button"
                aria-pressed={view !== undefined && Math.abs(effective.scale - 2) < 1e-6}
                onClick={() => setView(fitView(2))}
              >
                {t("blinkZoomTwo")}
              </button>
            </span>
          </div>
        </div>
      </div>
      <div className="blink-review-progress">
        <div>
          <strong title={t("blinkHumanReviewHint")}>{t("blinkHumanReviewTitle")}</strong>
          <small title={reference?.name}>
            {t(current?.diagnostics ? "blinkDiagReference" : "blinkReferenceNormalization", {
              filter: channel.filter,
              name: reference?.name ?? "—",
            })}
          </small>
        </div>
        <button
          type="button"
          className="btn small"
          disabled={channel.viewed === channel.total}
          onClick={() => {
            setPlaying(false);
            select(order.find((f) => !workflow.viewedFrames[f.sourceSha256])?.sourceSha256);
          }}
        >
          {t("blinkNextUnseen")}
        </button>
        <button
          type="button"
          className="btn primary"
          disabled={
            channel.viewed !== channel.total ||
            channel.frames.some((f) => workflow.previewFailures[f.sourceSha256] && decisionOf(f) !== "DROP")
          }
          onClick={() => {
            setPlaying(false);
            workflow.confirmBlinkChannel(channel.channelId);
          }}
        >
          {channel.confirmed ? t("blinkChannelConfirmed") : t("blinkConfirmChannel", { filter: channel.filter })}
        </button>
      </div>
      {current?.diagnostics && (
        <p className="blink-display-explanation">
          {displayMode === "detail" ? t("blinkDiagDetailHint") : t("blinkDiagFieldHint")}
        </p>
      )}
      <div className="blink-body">
        <div className="blink-visual-area">
          <div className="blink-main-column">
            <div
              ref={stageRef}
              className={`blink-stage ${compare ? "split" : ""}`}
              onWheel={onWheel}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
              onDoubleClick={() => setView(undefined)}
            >
              {compare && pane(reference, "reference")}
              {pane(showReference ? reference : current, "current")}
            </div>
            <div className="blink-decide-bar">
              <button type="button" className="btn" onClick={() => stepBy(-1)}>
                {t("blinkPrevious")}
              </button>
              <button
                type="button"
                className="btn primary"
                disabled={
                  !current ||
                  !workflow.viewedFrames[current.sourceSha256] ||
                  Boolean(workflow.previewFailures[current.sourceSha256])
                }
                onClick={() => decideAndNext("KEEP")}
              >
                {t("blinkKeepNext")}
              </button>
              <button type="button" className="btn" disabled={!current} onClick={() => decideAndNext("DROP")}>
                {t("blinkDropNext")}
              </button>
              <button type="button" className="btn" onClick={() => stepBy(1)}>
                {t("blinkNext")}
              </button>
              <span className="muted">{t("blinkViewedCount", { viewed: channel.viewed, total: channel.total })}</span>
            </div>
          </div>
          {!inspectorOpen && current?.diagnostics && (
            <DiagnosticPanel workflow={workflow} frame={current} t={t} compact />
          )}
        </div>
        {!inspectorOpen && current && !current.diagnostics && (
          <div className="blink-metrics" aria-label={t("blinkMetricsLabel")}>
            <BlinkFrameDetails frame={current} t={t} compact />
          </div>
        )}
        <div className="blink-filmstrip" role="listbox" aria-label={t("blinkFilmstrip")}>
          {groups.map((group) => {
            const night = channel.nights.find((item) => item.night === group.night);
            const keptInNight = channel.frames.filter(
              (frame) => frame.night === group.night && decisionOf(frame) === "KEEP",
            ).length;
            const nightTotal =
              night?.frameCount ?? channel.frames.filter((frame) => frame.night === group.night).length;
            const header =
              night?.skyRatio !== null && night?.skyRatio !== undefined
                ? t("blinkNightHeader", {
                    night: group.night,
                    count: nightTotal,
                    sky: night.skyRatio.toFixed(2),
                    kept: keptInNight,
                  })
                : t("blinkNightHeaderNoSky", { night: group.night, count: nightTotal, kept: keptInNight });
            return (
              <div key={group.key} className={`blink-night ${night?.defaultDropNight ? "default-drop" : ""}`}>
                <div className="blink-night-head">
                  <span
                    className="blink-night-title"
                    title={night?.defaultDropNight ? t("blinkNightDefaultDrop") : undefined}
                  >
                    {header}
                  </span>
                  <button
                    type="button"
                    className="btn small quiet"
                    onClick={() => workflow.setNightDecision(channel.channelId, group.night, "DROP")}
                  >
                    {t("blinkDropNight")}
                  </button>
                  <button
                    type="button"
                    className="btn small quiet"
                    onClick={() => workflow.setNightDecision(channel.channelId, group.night, "KEEP")}
                  >
                    {t("blinkKeepNight")}
                  </button>
                </div>
                <div className="blink-tiles">
                  {group.frames.map((frame) => {
                    const kept = decisionOf(frame) === "KEEP";
                    const selected = frame.sourceSha256 === current?.sourceSha256;
                    const url = filmstripUrl(frame);
                    return (
                      <button
                        type="button"
                        key={frame.sourceSha256}
                        ref={(element) => {
                          if (element) tileRefs.current.set(frame.sourceSha256, element);
                          else tileRefs.current.delete(frame.sourceSha256);
                        }}
                        className={`blink-tile ${workflow.viewedFrames[frame.sourceSha256] ? "viewed" : "unviewed"} ${kept ? "keep" : "drop"} ${isExcluded(frame) ? "exclude" : frame.flags.length ? "attention" : ""}`}
                        aria-label={frame.name}
                        aria-pressed={kept}
                        aria-current={selected ? "true" : undefined}
                        title={`${frame.name}\n${frame.flags.map((flag) => `${blinkFlagLabel(flag.code, t)} ${flagValueText(flag)}`).join(", ") || t("blinkNoFlags")}`}
                        onClick={(event) => toggleRange(frame, event.shiftKey)}
                      >
                        <FilmstripImage
                          frame={frame}
                          url={url}
                          error={cache.error(frame.previews.filmstrip)}
                          ensure={() => cache.ensure("filmstrip", frame.previews.filmstrip)}
                          onError={() => cache.fail(frame.previews.filmstrip)}
                          t={t}
                        />
                        <span className="blink-bar" aria-hidden="true" />
                        {frame.flags.length > 0 && (
                          <span className="blink-dots" aria-hidden="true">
                            {frame.flags.map((flag) => (
                              <span
                                key={flag.code}
                                className={`dot ${flag.severity === "EXCLUDE" ? "stop" : "warn"}`}
                              />
                            ))}
                          </span>
                        )}
                        {frame.reference && (
                          <span className="blink-star" aria-hidden="true">
                            ★
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
        <details className="blink-keys">
          <summary>{t("blinkKeyboard")}</summary>
          <p>{t("blinkKeysNavigate")}</p>
          <p>{t("blinkKeysDecide")}</p>
          <p>{t("blinkKeysView")}</p>
        </details>
      </div>
      <BlinkLaunchBar workflow={workflow} t={t} />
    </section>
  );
}
