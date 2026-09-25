import type { Translator } from "../i18n";
import type { GateDisposition, OutputArtifactKind } from "../types";
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

export function stageLabel(stageId: string, t: Translator): string {
  return (
    {
      "quality-control": t("stageQuality"),
      calibrate: t("stageCalibration"),
      register: t("stageRegistration"),
      "local-normalization": t("stageLocalNormalization"),
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
