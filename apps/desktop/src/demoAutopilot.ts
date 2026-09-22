import { useEffect, useRef } from "react";
import { hasTauriRuntime } from "./bridge";
import type { BlinkChannel, BlinkFlag, BlinkFrame, BlinkMeasureResponse, BlinkNightSummary } from "./types";
import type { useWorkflow } from "./useWorkflow";

type Workflow = ReturnType<typeof useWorkflow>;
export type DemoStage = "frames" | "review" | "blink" | "run" | "result";

/** `?mac=1` reserves the traffic-light area in browser builds (documentation screenshots only). */
export function requestedMacChrome(): boolean {
  if (hasTauriRuntime() || typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).get("mac") === "1";
}

/** The `?demo=` stage requested by the URL, browser builds only (documentation screenshots). */
export function requestedDemoStage(): DemoStage | undefined {
  if (hasTauriRuntime() || typeof window === "undefined") return undefined;
  const value = new URLSearchParams(window.location.search).get("demo");
  return value === "frames" || value === "review" || value === "blink" || value === "run" || value === "result" ? value : undefined;
}

/**
 * Drives the clearly labelled browser demo to a stage without clicks, so the
 * documentation screenshots come from the real components.  It never runs
 * inside the desktop app and never touches files.
 */
export function useDemoAutopilot(workflow: Workflow, stage: DemoStage | undefined) {
  const phase = useRef(0);
  useEffect(() => {
    if (!stage) return;
    if (phase.current === 0 && workflow.importedTotal === 0) { phase.current = 1; workflow.loadDemo(); return; }
    if (phase.current === 1 && workflow.sources.some((source) => source.detected && !source.confirmed)) {
      phase.current = 2;
      for (const source of workflow.sources) if (source.detected && !source.confirmed) workflow.confirmRole(source.role);
      return;
    }
    if (phase.current === 2 && stage === "blink" && workflow.allRequiredConfirmed && workflow.step === "import") { phase.current = 3; void workflow.runBlink(); return; }
    if (phase.current === 2 && stage !== "frames" && workflow.allRequiredConfirmed && workflow.step === "import") { phase.current = 3; void workflow.runInspection(); return; }
    if (phase.current === 3 && (stage === "run" || stage === "result") && workflow.step === "inspect" && workflow.canStart) { phase.current = 4; void workflow.startRun(); return; }
  }, [stage, workflow]);
}

/* ── The bundled blink manifest of the browser demo ─────────────────────── */

const DEMO_ROOT = "/explicit-browser-demo/blink";
const digest = (seed: string) => {
  // A stable, obviously synthetic content digest per frame (the demo hashes nothing).
  let h = 2166136261;
  for (const char of seed) { h ^= char.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0; }
  return `sha256:${h.toString(16).padStart(8, "0").repeat(8)}`;
};

interface DemoFrameSpec {
  time: string; night: string; sky: number; stars: number; extinction: number; transparency: number; fwhm: number; ellipticity: number;
  shape: number; flags: BlinkFlag[]; notes?: string[]; gate?: BlinkFrame["gate"]; registered?: boolean; reference?: boolean; rank: number; z: number;
}

const flag = (code: string, severity: BlinkFlag["severity"], value: number | null, threshold: number | null, message: string, combined = false): BlinkFlag => ({ code, severity, value, threshold, combined, message });

/** A deterministic star field per channel; the frames share it because the previews are registered. */
function starField(seed: number, count: number) {
  let state = seed * 2654435761 >>> 0;
  const next = () => { state = (state * 1664525 + 1013904223) >>> 0; return state / 4294967296; };
  return Array.from({ length: count }, () => { const bright = next(); return { x: 8 + next() * 375, y: 8 + next() * 245, r: 0.5 + bright * bright * 2.2, o: 0.35 + bright * 0.65 }; });
}

/** A 391×261 SVG preview: sky level, gradient and star brightness follow the frame's numbers. */
function svgPreview(stars: ReturnType<typeof starField>, sky: number, skyClean: number, transparency: number, registered: boolean, gradient: number): string {
  const level = Math.min(150, Math.round(38 + 42 * Math.log2(Math.max(0.5, sky / skyClean) + 0.5)));
  const grey = (value: number) => `rgb(${value},${value},${Math.min(255, value + 6)})`;
  const corner = Math.min(210, Math.round(level + gradient * 70));
  const shift = registered ? "" : ` transform="translate(23 -14) rotate(2 195 130)"`;
  const visible = stars.filter((star) => star.o * transparency > 0.3);
  const body = visible.map((star) => `<circle cx="${star.x.toFixed(1)}" cy="${star.y.toFixed(1)}" r="${(star.r * Math.sqrt(transparency)).toFixed(2)}" fill="#fff" opacity="${Math.min(1, star.o * transparency).toFixed(2)}"/>`).join("");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 391 261"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${grey(level)}"/><stop offset="1" stop-color="${grey(corner)}"/></linearGradient></defs><rect width="391" height="261" fill="url(#g)"/><g${shift}>${body}</g></svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

function buildChannel(channelId: string, filter: string, seed: number, skyClean: number, sourcesBest: number, fwhmBest: number, specs: DemoFrameSpec[], startIndex: number): { channel: BlinkChannel; frames: BlinkFrame[] } {
  const stars = starField(seed, 44);
  const frames = specs.map((spec, offset): BlinkFrame => {
    const index = startIndex + offset;
    const stem = `NGC 6822_300.00s_${filter}_${spec.night}_${spec.time}`;
    const registered = spec.registered !== false;
    const skyRatio = spec.sky / skyClean;
    const preview = svgPreview(stars, spec.sky, skyClean, spec.transparency, registered, spec.shape);
    const defaultDecision = spec.flags.some((item) => item.severity === "EXCLUDE") ? "DROP" : "KEEP";
    return {
      index, channelId, filter, target: "NGC 6822", night: spec.night,
      path: `${DEMO_ROOT}/${filter}/${stem}.fits`, name: `${stem}.fits`, sourceSha256: digest(stem),
      observedAt: `${spec.night}T${spec.time.replaceAll("-", ":")}`, airmass: 1.3,
      reference: spec.reference === true, defaultDecision,
      flags: spec.flags, notes: spec.notes ?? [],
      gate: spec.gate ?? { disposition: "PASS", codes: [] },
      metrics: {
        sky: spec.sky, skyRatio: Number(skyRatio.toFixed(2)), starCount: spec.stars, sourceRatio: Number((spec.stars / sourcesBest).toFixed(2)), extinctionMag: spec.extinction,
        transparency: spec.transparency, fwhmNative: spec.fwhm, fwhmRatio: Number((spec.fwhm / fwhmBest).toFixed(2)), ellipticity: spec.ellipticity, eccentricity: Number(Math.sqrt(1 - (1 - spec.ellipticity) ** 2).toFixed(2)),
        registrationRms: registered ? 0.21 : null, matchedStars: registered ? Math.round(spec.stars * 0.33) : null, overlap: registered ? 0.99 : null, backgroundShape: spec.shape, gradientRatio: null,
      },
      score: { log10: Number((-3.4 - spec.z * 0.5).toFixed(2)), z: spec.z, rank: spec.rank },
      previews: { filmstrip: `filmstrip/${String(index).padStart(4, "0")}-${filter}-${stem}.jpg`, zoom: `zoom/${String(index).padStart(4, "0")}-${filter}-${stem}.png`, coverage: registered ? 0.99 : 1, filmstripDataUrl: preview, zoomDataUrl: preview },
      transformToReference: registered ? [[1, 0, Number((offset * 0.7 - 2).toFixed(2))], [0, 1, Number((1.1 - offset * 0.4).toFixed(2))]] : null,
      normalization: { skyOffset: spec.sky, fluxScale: Number((1 / spec.transparency).toFixed(3)), registered },
    };
  });
  const nights = [...new Set(specs.map((spec) => spec.night))].map((night): BlinkNightSummary => {
    const own = frames.filter((frame) => frame.night === night);
    const median = (values: number[]) => { const sorted = [...values].sort((a, b) => a - b); return sorted[Math.floor(sorted.length / 2)]; };
    const exclude = own.filter((frame) => frame.defaultDecision === "DROP").length;
    return {
      night, frameCount: own.length, medianSky: median(own.map((frame) => frame.metrics.sky ?? 0)), skyRatio: Number((median(own.map((frame) => frame.metrics.skyRatio ?? 0))).toFixed(2)),
      medianSourceRatio: Number(median(own.map((frame) => frame.metrics.sourceRatio ?? 0)).toFixed(2)), medianExtinction: Number(median(own.map((frame) => frame.metrics.extinctionMag ?? 0)).toFixed(2)),
      exclude, attention: own.filter((frame) => frame.defaultDecision === "KEEP" && frame.flags.length > 0).length, defaultDropNight: exclude === own.length,
    };
  });
  const reference = frames.find((frame) => frame.reference);
  const channel: BlinkChannel = {
    channelId, target: "NGC 6822", filter, frameCount: frames.length,
    reference: reference ? { index: reference.index, sourceSha256: reference.sourceSha256, rule: "psf-signal-weight-proxy-v1" } : null,
    statistics: { skyClean, cleanCount: frames.filter((frame) => frame.flags.length === 0).length, sourcesBest, fwhmBest },
    stretch: { black: skyClean - 2.5 * 43.3, white: skyClean + 10 * 43.3, softness: 4, skyReference: skyClean, sigmaReference: 43.3 },
    previewGeometry: { filmstrip: [782, 522], zoom: [1563, 1044], sourceShape: [4176, 6252] },
    nights,
  };
  return { channel, frames };
}

const MOONLIT = (sky: number, stars: number): BlinkFlag[] => [
  flag("BLINK_SKY_BRIGHT", "EXCLUDE", Number((sky / 1090).toFixed(2)), 1.6, `Sky ${(sky / 1090).toFixed(2)}× the channel's clean-sky level and ${Math.round(100 * stars / 5680)} % of its stars: moonlit or hazy night`, true),
  flag("BLINK_SOURCES_LOW", "ATTENTION", Number((stars / 5680).toFixed(2)), 0.6, `${Math.round(100 * stars / 5680)} % of the channel's best star count`, true),
];

/**
 * The blink manifest of the browser demo: two channels of the NGC 6822 case
 * (a moonlit night pre-dropped, light cloud frames flagged for attention, one
 * reference per channel) with small vector previews.  Numbers are illustrative.
 */
export function demoBlinkManifest(): BlinkMeasureResponse {
  const luminance = buildChannel("group-demo-l", "L", 3, 1090, 5680, 4.07, [
    { time: "22-36-34", night: "2026-08-17", sky: 1275, stars: 3238, extinction: 0.18, transparency: 0.93, fwhm: 4.6, ellipticity: 0.1, shape: 0.11, rank: 7, z: -0.9, flags: [flag("BLINK_SOURCES_LOW", "ATTENTION", 0.57, 0.6, "57 % of the channel's best star count")] },
    { time: "22-41-36", night: "2026-08-17", sky: 1180, stars: 5210, extinction: 0.09, transparency: 0.98, fwhm: 4.3, ellipticity: 0.08, shape: 0.06, rank: 4, z: 0.6, flags: [] },
    { time: "22-46-38", night: "2026-08-17", sky: 1110, stars: 5460, extinction: 0.05, transparency: 0.99, fwhm: 4.2, ellipticity: 0.08, shape: 0.05, rank: 3, z: 0.8, flags: [] },
    { time: "22-51-40", night: "2026-08-17", sky: 1060, stars: 5590, extinction: 0.03, transparency: 1, fwhm: 4.4, ellipticity: 0.09, shape: 0.04, rank: 2, z: 0.9, flags: [] },
    { time: "22-57-01", night: "2026-08-17", sky: 1010, stars: 5768, extinction: 0.02, transparency: 1, fwhm: 4.32, ellipticity: 0.07, shape: 0.03, rank: 1, z: 1.1, flags: [], reference: true },
    { time: "23-02-03", night: "2026-08-17", sky: 980, stars: 5610, extinction: 0.04, transparency: 0.99, fwhm: 4.5, ellipticity: 0.09, shape: 0.05, rank: 5, z: 0.7, flags: [] },
    { time: "23-20-20", night: "2026-08-17", sky: 960, stars: 5440, extinction: 0.06, transparency: 0.98, fwhm: 4.5, ellipticity: 0.1, shape: 1.02, rank: 6, z: 0.5, flags: [flag("BLINK_BACKGROUND_SHAPE", "ATTENTION", 1.02, 0.5, "Background shape differs from the reference by 1.02 σ (P95−P5 of the outer cells)")] },
    { time: "21-15-27", night: "2026-08-20", sky: 2647, stars: 2947, extinction: 0.23, transparency: 0.84, fwhm: 4.47, ellipticity: 0.09, shape: 0.13, rank: 10, z: -3.1, flags: MOONLIT(2647, 2947) },
    { time: "21-20-29", night: "2026-08-20", sky: 2510, stars: 3120, extinction: 0.25, transparency: 0.85, fwhm: 4.5, ellipticity: 0.09, shape: 0.15, rank: 9, z: -3, flags: MOONLIT(2510, 3120) },
    { time: "21-25-31", night: "2026-08-20", sky: 2360, stars: 3040, extinction: 0.28, transparency: 0.84, fwhm: 4.6, ellipticity: 0.1, shape: 0.14, rank: 11, z: -3.2, flags: MOONLIT(2360, 3040) },
    { time: "21-30-33", night: "2026-08-20", sky: 2210, stars: 3300, extinction: 0.22, transparency: 0.86, fwhm: 4.4, ellipticity: 0.09, shape: 0.16, rank: 8, z: -2.9, flags: MOONLIT(2210, 3300) },
    { time: "23-28-04", night: "2026-08-20", sky: 3150, stars: 6, extinction: 2.8, transparency: 0.08, fwhm: 7.1, ellipticity: 0.3, shape: 1.35, rank: 14, z: -6.2, registered: false,
      flags: [flag("BLINK_UNREGISTRABLE", "EXCLUDE", null, null, "No registration transform was found"), flag("BLINK_FEW_STARS", "EXCLUDE", 6, 20, "6 stars detected"), flag("BLINK_CLOUD_THICK", "EXCLUDE", null, null, "Multi-family cloud evidence (gate HARD_FAIL)")],
      gate: { disposition: "HARD_FAIL", codes: ["GATE_MULTI_FAMILY_CLOUD_HARD", "GATE_REGISTRATION_FAILED"] } },
    { time: "23-46-06", night: "2026-09-09", sky: 985, stars: 4710, extinction: 0.67, transparency: 0.54, fwhm: 4.9, ellipticity: 0.11, shape: 0.58, rank: 12, z: -1.4,
      flags: [flag("BLINK_EXTINCTION", "ATTENTION", 0.67, 0.5, "0.67 mag of extra extinction: light cloud"), flag("BLINK_BACKGROUND_SHAPE", "ATTENTION", 0.58, 0.5, "Background shape differs from the reference by 0.58 σ")],
      notes: ["GATE_INSUFFICIENT_NIGHT_BASELINE: the night has too few frames for a nightly baseline"] },
    { time: "23-56-09", night: "2026-09-09", sky: 950, stars: 5120, extinction: 0.21, transparency: 0.82, fwhm: 4.6, ellipticity: 0.1, shape: 0.31, rank: 13, z: -0.6, flags: [], notes: ["GATE_INSUFFICIENT_NIGHT_BASELINE: the night has too few frames for a nightly baseline"] },
  ], 0);
  const red = buildChannel("group-demo-r", "R", 7, 612, 4120, 4.4, [
    { time: "21-30-19", night: "2026-09-06", sky: 640, stars: 2350, extinction: 0.54, transparency: 0.61, fwhm: 4.9, ellipticity: 0.1, shape: 0.22, rank: 6, z: -1.2,
      flags: [flag("BLINK_EXTINCTION", "ATTENTION", 0.54, 0.5, "0.54 mag of extra extinction: light cloud"), flag("BLINK_SOURCES_LOW", "ATTENTION", 0.57, 0.6, "57 % of the channel's best star count")] },
    { time: "21-37-32", night: "2026-09-06", sky: 620, stars: 3980, extinction: 0.12, transparency: 0.9, fwhm: 4.5, ellipticity: 0.09, shape: 0.08, rank: 3, z: 0.5, flags: [] },
    { time: "21-44-45", night: "2026-09-06", sky: 605, stars: 4120, extinction: 0.04, transparency: 1, fwhm: 4.4, ellipticity: 0.08, shape: 0.05, rank: 1, z: 1, flags: [], reference: true },
    { time: "21-51-58", night: "2026-09-06", sky: 612, stars: 4050, extinction: 0.06, transparency: 0.98, fwhm: 4.5, ellipticity: 0.08, shape: 0.06, rank: 2, z: 0.8, flags: [] },
    { time: "23-32-53", night: "2026-09-06", sky: 690, stars: 620, extinction: 1.9, transparency: 0.17, fwhm: 6.4, ellipticity: 0.24, shape: 1.16, rank: 8, z: -5.1, registered: false,
      flags: [flag("BLINK_SOURCES_LOW", "EXCLUDE", 0.15, 0.45, "15 % of the channel's best star count"), flag("BLINK_EXTINCTION", "EXCLUDE", 1.9, 1, "1.90 mag of extra extinction: thick cloud"), flag("BLINK_UNREGISTRABLE", "EXCLUDE", null, null, "No registration transform was found")],
      gate: { disposition: "HARD_FAIL", codes: ["GATE_MULTI_FAMILY_CLOUD_HARD"] } },
    { time: "21-25-17", night: "2026-09-08", sky: 601, stars: 3900, extinction: 0.1, transparency: 0.92, fwhm: 4.6, ellipticity: 0.09, shape: 0.09, rank: 4, z: 0.3, flags: [] },
    { time: "21-32-30", night: "2026-09-08", sky: 598, stars: 3850, extinction: 0.11, transparency: 0.91, fwhm: 4.7, ellipticity: 0.1, shape: 0.1, rank: 5, z: 0.2, flags: [] },
    { time: "21-39-43", night: "2026-09-08", sky: 604, stars: 3810, extinction: 0.13, transparency: 0.9, fwhm: 4.7, ellipticity: 0.1, shape: 0.11, rank: 7, z: 0.1, flags: [] },
  ], 14);
  const frames = [...luminance.frames, ...red.frames];
  return {
    schemaVersion: 1, kind: "blink-manifest-v1", sessionId: "demo-blink-session", sessionDirectory: `${DEMO_ROOT}/session`, createdAt: "2026-09-22T00:00:00Z", engineVersion: "demo",
    gatePolicyDigest: `sha256:${"7".repeat(64)}`, flagsPolicyDigest: `sha256:${"b".repeat(64)}`, flagsPolicy: { version: "blink-flags-v1", skyBrightAttention: 1.6, sourcesLowAttention: 0.6, sourcesLowExclude: 0.45, extinctionAttention: 0.5, extinctionExclude: 1 },
    inventorySha256: `sha256:${"1".repeat(64)}`, timings: { measurementSeconds: 0, analysisSeconds: 0, gateSeconds: 0, flagsSeconds: 0, previewSeconds: 0 },
    counts: { frames: frames.length, exclude: frames.filter((frame) => frame.defaultDecision === "DROP").length, attention: frames.filter((frame) => frame.defaultDecision === "KEEP" && frame.flags.length > 0).length, clean: frames.filter((frame) => frame.flags.length === 0).length },
    channels: [luminance.channel, red.channel], frames,
    manifestSha256: `sha256:${"d".repeat(64)}`,
  };
}
