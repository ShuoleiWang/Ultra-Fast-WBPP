import type { MessageKey, Translator } from "../i18n";
import type { GateDisposition, OutputArtifact, OutputArtifactKind } from "../types";
import type { useWorkflow } from "../useWorkflow";

export type Workflow = ReturnType<typeof useWorkflow>;
export const PASS_FRAME_BATCH = 50;

/** A deterministic star field (SVG) for hero panels; decorative only. */
export function StarField({ count = 110, seed = 3, className }: { count?: number; seed?: number; className?: string }) {
  let state = (seed * 2654435761) >>> 0;
  const next = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const stars = Array.from({ length: count }, (_, index) => {
    const bright = next();
    return { key: index, x: next() * 100, y: next() * 60, r: 0.05 + bright * bright * 0.22, o: 0.3 + bright * 0.7 };
  });
  return (
    <svg className={className} viewBox="0 0 100 60" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      {stars.map((star) => (
        <circle key={star.key} cx={star.x} cy={star.y} r={star.r} fill="#fff" opacity={star.o} />
      ))}
    </svg>
  );
}

const FILTER_TILES: Record<string, string> = {
  L: "tile-l",
  LUM: "tile-l",
  R: "tile-r",
  RED: "tile-r",
  G: "tile-g",
  GREEN: "tile-g",
  B: "tile-b",
  BLUE: "tile-b",
  HA: "tile-ha",
  "H-ALPHA": "tile-ha",
  HALPHA: "tile-ha",
  OIII: "tile-oiii",
  O3: "tile-oiii",
  SII: "tile-sii",
  S2: "tile-sii",
};
export const filterTileClass = (filter: string | undefined | null) =>
  FILTER_TILES[(filter ?? "").trim().toUpperCase()] ?? "tile-default";

/** A frame tile: the preview when there is one, otherwise the filter's colour and letter. */
export function FrameTile({
  filter,
  preview,
  alt,
  large = false,
}: {
  filter?: string | null;
  preview?: string | null;
  alt?: string;
  large?: boolean;
}) {
  const label = (filter ?? "").trim();
  if (preview) return <img className={large ? "preview" : "thumb"} src={preview} alt={alt ?? ""} />;
  return (
    <span
      className={`${large ? "preview" : "thumb"} tile ${filterTileClass(filter)}`}
      role={alt ? "img" : undefined}
      aria-label={alt}
      aria-hidden={alt ? undefined : true}
    >
      <span>{label && label.length <= 4 && label !== "UNKNOWN" ? label : ""}</span>
    </span>
  );
}
/** A filter name as the interface shows it: the engine's `NOFILTER` reads "no filter". */
export const filterName = (filter: string, t: Translator) =>
  filter.trim().toUpperCase() === "NOFILTER" ? t("noFilter") : filter;
export const basename = (path: string) => path.split(/[\\/]/).filter(Boolean).pop() ?? path;
export const formatBytes = (bytes: number, t: Translator) =>
  bytes ? `${(bytes / 1_000_000).toFixed(0)} MB` : t("externallyManaged");
export const optionalNumber = (value: string) => (value.trim() === "" ? null : Number(value));
const formatElapsed = (seconds: number) =>
  [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60]
    .map((value) => String(value).padStart(2, "0"))
    .join(":");
export const dispositionClass = (disposition: GateDisposition) => disposition.toLowerCase().replace("_", "-");

export function artifactLabel(kind: OutputArtifactKind, t: Translator): string {
  return (
    {
      SOLVED_MONO_FITS: t("artifactSolvedMono"),
      LINEAR_RGB_FITS: t("artifactLinearRgb"),
      RGB_PREVIEW_TIFF_16: t("artifactTiff"),
      RGB_PREVIEW_PNG_16: t("artifactPng"),
      MONO_PREVIEW_PNG: t("artifactMonoPreview"),
      RECEIPT: t("artifactReceipt"),
      REPORT: t("artifactReport"),
      DRIZZLE_DATA: t("artifactDrizzle"),
      MASTER: t("artifactMaster"),
      PREVIEW: t("artifactPreview"),
    }[kind] ?? kind
  );
}

/**
 * A product's one-line description, written from its kind and solve evidence so
 * it follows the interface language; the controller's own text is the fallback
 * (other kinds, and the browser demo's labelled placeholder).
 */
export function artifactDetail(artifact: OutputArtifact, t: Translator): string {
  const solved = artifact.receipt?.astrometry;
  if (solved) return t("artifactDetailSolved", { rms: solved.rmsArcsec.toFixed(3), stars: solved.matchedStars });
  if (!artifact.receipt) return artifact.detail;
  switch (artifact.kind) {
    case "RGB_PREVIEW_TIFF_16":
    case "RGB_PREVIEW_PNG_16":
      return t("artifactDetailDisplayPreview");
    case "MONO_PREVIEW_PNG":
      return t("artifactDetailMonoPreview");
    case "RECEIPT":
      return t("artifactDetailReceipt");
    default:
      return artifact.detail;
  }
}

/** The engine blocker: a localised label in front of the controller's own reason. */
export const engineBlockerText = (reason: string | undefined, t: Translator) =>
  reason ? t("blockerEngineReason", { reason }) : t("blockerEngine");

export function stageLabel(stageId: string, t: Translator): string {
  return (
    {
      "quality-control": t("stageQuality"),
      calibrate: t("stageCalibration"),
      register: t("stageRegistration"),
      integrate: t("stageIntegration"),
      drizzle: t("stageDrizzle"),
      solve: t("stageSolve"),
      mosaic: t("stageMosaic"),
      color: t("stageColor"),
      prepare: t("stagePrepare"),
      alignment: t("stageAlignment"),
      preview: t("stagePreview"),
      verify: t("stageVerify"),
      publish: t("stagePublish"),
    }[stageId] ?? stageId
  );
}

export function RunElapsed({ seconds, label }: { seconds: number; label: string }) {
  return (
    <div className="run-elapsed">
      <span>{label}</span>
      <output role="timer" aria-label={label}>
        {formatElapsed(seconds)}
      </output>
    </div>
  );
}

export function Alert({
  title,
  tone = "warn",
  children,
}: {
  title: string;
  tone?: "warn" | "stop" | "info";
  children: React.ReactNode;
}) {
  return (
    <aside className={`callout ${tone}`} role={tone === "info" ? "note" : "alert"}>
      <span className="symbol">{tone === "info" ? "i" : "!"}</span>
      <div>
        <strong>{title}</strong>
        <p>{children}</p>
      </div>
    </aside>
  );
}

/** The engine and controller failure codes with a localised, actionable sentence. */
const FAILURE_HINTS: Record<string, MessageKey> = {
  ASTROMETRY_REQUIRED: "astrometryFailedHint",
  MOSAIC_CATALOG_TOO_SPARSE: "mosaicCatalogHint",
  OUTPUT_EXISTS: "failureHint_OUTPUT_EXISTS",
  OUTPUT_PATH_TOO_LONG: "failureHint_OUTPUT_PATH_TOO_LONG",
  SOLVER_UNAVAILABLE: "failureHint_SOLVER_UNAVAILABLE",
  NO_FLATS: "failureHint_NO_FLATS",
  FLAT_MATCH_MISSING: "failureHint_FLAT_MATCH_MISSING",
  DARK_MATCH_MISSING: "failureHint_DARK_MATCH_MISSING",
  BIAS_MATCH_MISSING: "failureHint_BIAS_MATCH_MISSING",
  PUBLICATION_FAILED: "failureHint_PUBLICATION_FAILED",
  ATOMIC_PUBLICATION_FAILED: "failureHint_PUBLICATION_FAILED",
  REGISTRATION_FAILED: "failureHint_REGISTRATION_FAILED",
  IO_ERROR: "failureHint_IO_ERROR",
  PLAN_NOT_EXECUTION_READY: "failureHint_PLAN_NOT_EXECUTION_READY",
  PROJECT_RESULT_GATE_BLOCKED: "failureHint_PROJECT_RESULT_GATE_BLOCKED",
  UNSUPPORTED_FORMAT: "importRefusedNoFrames",
  NO_SUPPORTED_FRAMES: "importRefusedNoFrames",
};
/** The controller's import refusals, which carry no code of their own. */
const IMPORT_REFUSALS: Array<[RegExp, MessageKey]> = [
  [/^inventory contains blocking frame errors:/, "importRefusedFrameErrors"],
  [/^frame is not ready:/, "importRefusedNotReady"],
  [/^desktop import does not accept role /, "importRefusedRole"],
  [/^no supported astronomy frames were found/, "importRefusedNoFrames"],
  [/^import takes absolute file or folder paths/, "importRefusedRelativePath"],
  [/separately selected files are too many for one import on Windows/, "importRefusedTooMany"],
];
const DISK_FULL = /\[Errno 28\]|No space left on device|ENOSPC|not enough space on the disk/i;
const CODED = /^([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+): ([\s\S]*)$/;

export interface FailureExplanation {
  /** The failure code, from the event or from a leading `CODE: ` of the text. */
  code?: string;
  /** The localised sentence of what to do, when the failure has one. */
  hint?: string;
  /** The engine's or controller's own text, without the code. */
  detail: string;
}

/**
 * What a failure means for the user: the localised sentence of a known code
 * or controller refusal (a full disk whatever the code), and the original
 * text, which stays visible beside it.  Unknown failures keep their own text.
 */
export function explainFailure(text: string, t: Translator, eventCode?: string): FailureExplanation {
  let detail = text.replace(/^Error: /, "");
  let code = eventCode;
  if (code && detail.startsWith(`${code}: `)) detail = detail.slice(code.length + 2);
  else if (!code) {
    const coded = CODED.exec(detail);
    if (coded) [, code, detail] = coded;
  }
  const hintKey = DISK_FULL.test(detail)
    ? "failureHint_DISK_FULL"
    : ((code && FAILURE_HINTS[code]) ?? IMPORT_REFUSALS.find(([pattern]) => pattern.test(detail))?.[1]);
  return { code, hint: hintKey ? t(hintKey) : undefined, detail };
}

/** An error message of one or more lines, each with its localised sentence where there is one. */
export function FailureLines({ message, t }: { message: string; t: Translator }) {
  const lines = message.split("\n").filter((line) => line.trim());
  return (
    <>
      {lines.map((line, index) => {
        const { hint } = explainFailure(line, t);
        return (
          <span key={index} className="failure-line">
            {hint && <span className="failure-hint">{hint}</span>}
            {hint ? <small className="failure-detail">{line.replace(/^Error: /, "")}</small> : line}
          </span>
        );
      })}
    </>
  );
}
