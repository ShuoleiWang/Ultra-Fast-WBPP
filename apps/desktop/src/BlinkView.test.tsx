import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BlinkMeasureRequest, BlinkMeasureResponse, InspectedAsset, PipelineEventHandlers, RunRequest } from "./types";

const native = vi.hoisted(() => ({
  runtime: true,
  handlers: undefined as PipelineEventHandlers | undefined,
  inspectPaths: vi.fn(),
  inspectCalibration: vi.fn(),
  blinkMeasure: vi.fn(),
  loadBlinkPreview: vi.fn(),
  startRun: vi.fn(),
  inspectQuality: vi.fn(),
}));

vi.mock("./bridge", () => ({
  hasTauriRuntime: () => native.runtime,
  desktopBridge: {
    getCapabilities: vi.fn(async () => ({ platform: "macos", chip: "Apple M3 Pro", cpuBackend: "Native CPU execution", gpuBackend: "Metal execution", optimizationTier: "M3_PRO_TUNED", available: true, drizzleAvailable: true, solverAvailable: true, runtimeVersion: "0.1.0" })),
    inspectPaths: (...args: unknown[]) => native.inspectPaths(...args),
    inspectCalibration: (...args: unknown[]) => native.inspectCalibration(...args),
    inspectQuality: (...args: unknown[]) => native.inspectQuality(...args),
    blinkMeasure: (...args: unknown[]) => native.blinkMeasure(...args),
    loadBlinkPreview: (...args: unknown[]) => native.loadBlinkPreview(...args),
    hashSources: vi.fn(async () => ({ entries: [] })),
    pickInputFiles: vi.fn(async () => []), pickInputDirectories: vi.fn(async () => ["/data/NGC 6822"]), pickOutputParent: vi.fn(async () => "/results/deep sky"),
    startRun: (...args: unknown[]) => native.startRun(...args), cancelRun: vi.fn(async () => undefined),
    catalogList: vi.fn(async () => ({ schemaVersion: 1, catalogRoot: "/catalogs", catalogs: [] })),
    catalogDoctor: vi.fn(async () => ({ schemaVersion: 1, ok: true, catalogRoot: "/catalogs", config: { present: true, valid: true }, installedSetBindingReady: true, message: "ready" })),
    solverDoctor: vi.fn(async () => ({ schemaVersion: 1, engineVersion: "0.1.0", backends: [{ backendId: "astrometry-net", displayName: "solve-field", version: "0.97", available: true, executionReady: true, metadata: { probe: { path: "/opt/homebrew/bin/solve-field", version: "0.97", executionReady: true } } }] })),
    startCatalogInstall: vi.fn(), cancelCatalogInstall: vi.fn(), verifyCatalog: vi.fn(), openProviderTerms: vi.fn(), revealOutput: vi.fn(async () => undefined),
  },
  listenForDesktopDrops: vi.fn(async () => () => undefined),
  listenForPipelineEvents: vi.fn(async (handlers: PipelineEventHandlers) => { native.handlers = handlers; return () => undefined; }),
  listenForCatalogEvents: vi.fn(async () => () => undefined),
}));

import App from "./App";
import { demoBlinkManifest } from "./demoAutopilot";

const manifest = (): BlinkMeasureResponse => demoBlinkManifest();
const lightPaths = () => manifest().frames.map((frame) => frame.path);
const L = (time: string) => `NGC 6822_300.00s_L_${time}.fits`;
const R = (time: string) => `NGC 6822_300.00s_R_${time}.fits`;

const asset = (path: string, filter: string, role: InspectedAsset["role"] = "LIGHT"): InspectedAsset => ({
  path, role, width: 6252, height: 4176, channels: 1, filter, target: role === "LIGHT" ? "NGC 6822" : "UNKNOWN", camera: "QHY268M",
  exposureSeconds: role === "LIGHT" ? 300 : 1, temperatureCelsius: -10, gain: 26, offset: 30, binning: [1, 1], cfaPattern: "NONE", readoutMode: "Mode 1", sourceSha256: null,
});
const inventory = () => {
  const assets = [
    ...manifest().frames.map((frame) => asset(frame.path, frame.filter)),
    asset("/data/calibration/flat L.fits", "L", "FLAT"), asset("/data/calibration/flat R.fits", "R", "FLAT"), asset("/data/calibration/dark.fits", "L", "DARK"), asset("/data/calibration/bias.fits", "L", "BIAS"),
  ];
  return { projectName: "NGC 6822", totalFiles: assets.length, assets, sources: assets.map((item) => ({ role: item.role, paths: [item.path], fileCount: 1, confidence: 1, needsConfirmation: false })) };
};
const solvedArtifact = { kind: "SOLVED_MONO_FITS" as const, name: "NGC 6822_L.fits", path: "/results/deep sky/run/NGC 6822_L.fits", detail: "final WCS", filter: "L", target: "NGC 6822", receipt: { artifactId: "final-l", relativePath: "NGC 6822_L.fits", sha256: "a".repeat(64), sizeBytes: 4096, astrometry: {
  referenceFrame: "ICRS", projection: "TAN", centerRaDegrees: 296.2, centerDecDegrees: -14.8, pixelScaleArcsec: 0.78, rotationDegrees: 0, rmsPixels: 0.3, rmsArcsec: 0.24, matchedStars: 120, parity: "POSITIVE" as const, catalogIdentity: "b".repeat(64),
  indexIdentities: ["astrometry.net:index:4108"], correspondenceSha256: "c".repeat(64), catalogManaged: true as const, installedSetIdentity: "d".repeat(64), catalogManifestSha256: "e".repeat(64), indexArtifacts: [], wcsSha256: "1".repeat(64),
} } };
const readyChecks = ["final-project-products-present", "final-project-receipts-valid", "final-project-astrometry-validated"].map((code) => ({ code, required: true, passed: true, artifactIds: ["final-l"], message: "ready" }));

const view = () => screen.getByRole("region", { name: /Blink and select/ });
const stage = () => within(document.querySelector(".blink-stage") as HTMLElement);
const tiles = () => screen.getAllByRole("button", { name: /^NGC 6822_300\.00s_/ });
const currentTile = () => tiles().find((tile) => tile.getAttribute("aria-current") === "true")!;
const chip = (filter: string) => screen.getByRole("button", { name: new RegExp(`^${filter} \\d+/\\d+ · \\d+ flagged$`) });
const key = (name: string, init: Record<string, unknown> = {}) => { fireEvent.keyDown(view(), { key: name, ...init }); fireEvent.keyUp(view(), { key: name, ...init }); };

async function importLights() {
  await userEvent.click(screen.getByRole("button", { name: /Choose folder$/ }));
  await waitFor(() => expect(native.inspectCalibration).toHaveBeenCalled());
}
async function reachBlink() {
  await importLights();
  await userEvent.click(await screen.findByRole("button", { name: "Blink & select (22 Lights)" }));
  await screen.findByRole("heading", { name: /Blink and select/ });
}

beforeEach(() => {
  native.runtime = true; native.handlers = undefined;
  window.localStorage.clear();
  // Reduced motion keeps playback off until a test asks for it, so the current frame is deterministic.
  vi.spyOn(window, "matchMedia").mockImplementation((query: string) => ({ matches: query.includes("reduced-motion"), media: query, onchange: null, addListener: () => undefined, removeListener: () => undefined, addEventListener: () => undefined, removeEventListener: () => undefined, dispatchEvent: () => false }));
  native.inspectPaths.mockReset().mockImplementation(async () => inventory());
  native.inspectCalibration.mockReset().mockResolvedValue({ schemaVersion: 1, status: "READY", calibrationReady: true, groups: [], issues: [] });
  native.blinkMeasure.mockReset().mockImplementation(async () => manifest());
  native.loadBlinkPreview.mockReset().mockImplementation(async (_directory: string, relativePath: string) => `data:image/png;base64,${Buffer.from(relativePath).toString("base64")}`);
  native.startRun.mockReset().mockResolvedValue({ jobId: "run-1", accepted: true, executionMode: "native", outputDirectory: "/results/deep sky/run" });
  native.inspectQuality.mockReset();
});
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); window.history.replaceState({}, "", "/"); });

describe("blink screening", () => {
  it("measures the Lights, opens the view flagged-first with night headers, counts and a reference", async () => {
    let finish!: (value: BlinkMeasureResponse) => void;
    native.blinkMeasure.mockImplementationOnce(() => new Promise<BlinkMeasureResponse>((resolve) => { finish = resolve; }));
    render(<App />);
    await importLights();
    // No session: the primary action is the blink, the start says it screens automatically.
    expect(screen.getByRole("button", { name: "Blink & select (22 Lights)" })).toHaveClass("primary");
    expect(screen.getByRole("button", { name: /Start processing \(automatic screening\)/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Screen 22 Lights first/ })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Blink & select (22 Lights)" }));
    expect(native.blinkMeasure).toHaveBeenCalledWith({ paths: lightPaths(), masterFlats: [] } satisfies BlinkMeasureRequest);
    // The toolbar and the disabled button both say what is being measured.
    expect(await screen.findAllByText(/Measuring 22 Lights for blink · \d+ s/)).toHaveLength(2);
    await act(async () => finish(manifest()));
    await screen.findByRole("heading", { name: /Blink and select/ });
    expect(chip("L")).toHaveTextContent("L 9/14 · 8 flagged");
    expect(chip("R")).toHaveTextContent("R 7/8 · 2 flagged");
    expect(chip("L")).toHaveAttribute("aria-pressed", "true");
    // EXCLUDE first (the moonlit night, chronological inside), then ATTENTION, then clean.
    const names = tiles().map((tile) => tile.getAttribute("aria-label"));
    expect(names.slice(0, 5)).toEqual(["2026-08-20_21-15-27", "2026-08-20_21-20-29", "2026-08-20_21-25-31", "2026-08-20_21-30-33", "2026-08-20_23-28-04"].map(L));
    expect(names.slice(5, 8)).toEqual([L("2026-08-17_22-36-34"), L("2026-08-17_23-20-20"), L("2026-09-09_23-46-06")]);
    expect(names).toHaveLength(14);
    expect(tiles()[0]).toHaveAttribute("aria-current", "true");
    expect(tiles()[0]).toHaveAttribute("aria-pressed", "false");
    expect(tiles()[5]).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("2026-08-20 · 5 frames · sky ×2.30 · 0 kept")).toBeInTheDocument();
    expect(screen.getAllByText(/^2026-08-17 · 7 frames · sky ×/).length).toBeGreaterThan(0);
    // The stage names the frame, its decision and its flags (localised labels; the engine text is the tooltip).
    const image = stage().getByRole("img", { name: `${L("2026-08-20_21-15-27")} · Dropped · Bright sky, Few stars detected` });
    expect(image).toHaveAttribute("src", expect.stringMatching(/^data:image\/svg\+xml/));
    expect(stage().getByTitle(/moonlit or hazy night/)).toHaveTextContent("Bright sky ×2.43");
    // Sidebar row and inspector follow the session and the current frame.
    const navigation = screen.getByRole("navigation", { name: "Workflow" });
    expect(within(navigation).getByRole("button", { name: /Blink/ })).toHaveTextContent("16/22");
    const inspector = screen.getByRole("complementary", { name: "Inspector" });
    expect(within(inspector).getByText(L("2026-08-20_21-15-27"))).toBeInTheDocument();
    // The localised label, threshold and explanation are shown; the engine's English message is the tooltip.
    const flagRow = within(inspector).getByTitle("Sky 2.43× the channel's clean-sky level and 52 % of its stars: moonlit or hazy night");
    expect(flagRow).toHaveTextContent("Bright sky ×2.43");
    expect(flagRow).toHaveTextContent("threshold 1.6 · EXCLUDE · dropped by default · combined rule");
    expect(flagRow).toHaveTextContent("Sky level relative to the channel's clean-sky level");
    expect(within(inspector).getByText(/GATE_INSUFFICIENT|PASS/)).toBeInTheDocument();
    await userEvent.click(within(inspector).getByRole("button", { name: "Keep" }));
    expect(chip("L")).toHaveTextContent("L 10/14 · 8 flagged");
    expect(tiles()[0]).toHaveAttribute("aria-pressed", "true");
  });

  it("steps, decides and jumps with the keyboard, and undoes in order", async () => {
    render(<App />);
    await reachBlink();
    key("ArrowRight");
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-08-20_21-20-29"));
    key(" ");
    expect(currentTile()).toHaveAttribute("aria-pressed", "true");
    expect(chip("L")).toHaveTextContent("L 10/14");
    key("d");
    expect(currentTile()).toHaveAttribute("aria-pressed", "false");
    key("k");
    expect(currentTile()).toHaveAttribute("aria-pressed", "true");
    key("ArrowLeft"); key("ArrowLeft");
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-09-09_23-56-09"));
    key("Home");
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-08-20_21-15-27"));
    key("End");
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-09-09_23-56-09"));
    key("f");
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-08-20_21-15-27"));
    key("F", { shiftKey: true });
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-09-09_23-46-06"));
    key("r");
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-08-17_22-57-01"));
    expect(stage().getByRole("img", { name: `${L("2026-08-17_22-57-01")} · Kept · no flags` })).toBeInTheDocument();
    // Undo restores the three decisions in reverse order; the button follows the stack.
    expect(screen.getByRole("button", { name: "Undo" })).toBeEnabled();
    key("z");
    expect(tiles()[1]).toHaveAttribute("aria-pressed", "false");
    key("z", { metaKey: true });
    expect(tiles()[1]).toHaveAttribute("aria-pressed", "true");
    key("z");
    expect(tiles()[1]).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: "Undo" })).toBeDisabled();
    expect(chip("L")).toHaveTextContent("L 9/14");
    // Keys typed into a control are left alone.
    fireEvent.keyDown(screen.getByRole("combobox", { name: "Frames per second" }), { key: "d" });
    expect(currentTile()).toHaveAttribute("aria-pressed", "true");
  });

  it("plays through the channel at the chosen rate, skips dropped frames when asked, and pauses on a manual step", async () => {
    render(<App />);
    await reachBlink();
    vi.useFakeTimers();
    expect(screen.getByRole("button", { name: "Play" })).toHaveAttribute("aria-pressed", "false");
    key("p");
    expect(screen.getByRole("button", { name: "Pause" })).toHaveAttribute("aria-pressed", "true");
    await act(async () => { vi.advanceTimersByTime(500); });
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-08-20_21-20-29"));
    await act(async () => { vi.advanceTimersByTime(500); });
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-08-20_21-25-31"));
    key("]");
    await act(async () => { vi.advanceTimersByTime(250); });
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-08-20_21-30-33"));
    key("Escape");
    await act(async () => { vi.advanceTimersByTime(2_000); });
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-08-20_21-30-33"));
    // Kept only: playback skips the dropped moonlit frames and wraps.
    fireEvent.click(screen.getByRole("checkbox", { name: "Kept only" }));
    key("["); key("p");
    await act(async () => { vi.advanceTimersByTime(500); });
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-08-17_22-36-34"));
    key("End");
    expect(screen.getByRole("button", { name: "Play" })).toHaveAttribute("aria-pressed", "false");
    await act(async () => { vi.advanceTimersByTime(2_000); });
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-09-09_23-56-09"));
    key("p");
    await act(async () => { vi.advanceTimersByTime(500); });
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-08-17_22-36-34"));
  });

  it("starts playing on open unless the system prefers reduced motion", async () => {
    vi.spyOn(window, "matchMedia").mockImplementation((query: string) => ({ matches: false, media: query, onchange: null, addListener: () => undefined, removeListener: () => undefined, addEventListener: () => undefined, removeEventListener: () => undefined, dispatchEvent: () => false }));
    render(<App />);
    await reachBlink();
    expect(screen.getByRole("button", { name: "Pause" })).toHaveAttribute("aria-pressed", "true");
  });

  it("drops or keeps a whole night, applies the flags again, and reorders chronologically", async () => {
    render(<App />);
    await reachBlink();
    key("r");
    key("n");
    expect(chip("L")).toHaveTextContent("L 2/14");
    expect(screen.getAllByText(/^2026-08-17 · 7 frames · sky ×\d+\.\d+ · 0 kept$/).length).toBeGreaterThan(0);
    key("z");
    expect(chip("L")).toHaveTextContent("L 9/14");
    await userEvent.click(within(screen.getByText("2026-08-20 · 5 frames · sky ×2.30 · 0 kept").closest(".blink-night")!).getByRole("button", { name: "Keep night" }));
    expect(chip("L")).toHaveTextContent("L 14/14");
    expect(screen.getByText("2026-08-20 · 5 frames · sky ×2.30 · 5 kept")).toBeInTheDocument();
    await userEvent.click(within(screen.getByText("2026-08-20 · 5 frames · sky ×2.30 · 5 kept").closest(".blink-night")!).getByRole("button", { name: "Drop night" }));
    expect(chip("L")).toHaveTextContent("L 9/14");
    // The reference is the current frame: Space drops it, A restores the flag defaults.
    key(" ");
    expect(chip("L")).toHaveTextContent("L 8/14");
    key("a");
    expect(chip("L")).toHaveTextContent("L 9/14");
    key("z");
    expect(chip("L")).toHaveTextContent("L 8/14");
    await userEvent.click(screen.getByRole("button", { name: "Apply flags" }));
    expect(chip("L")).toHaveTextContent("L 9/14");
    // Chronological order keeps the night headers but puts the nights in time order.
    await userEvent.click(screen.getByRole("button", { name: "Flagged first" }));
    expect(screen.getByRole("button", { name: "Chronological" })).toHaveAttribute("aria-pressed", "true");
    expect(tiles()[0]).toHaveAttribute("aria-label", L("2026-08-17_22-36-34"));
    expect(tiles()[7]).toHaveAttribute("aria-label", L("2026-08-20_21-15-27"));
    expect(screen.getAllByText(/^2026-08-17 · 7 frames/)).toHaveLength(1);
    // Shift-click applies one decision to the range, as a single undo step.
    await userEvent.click(tiles()[0]);
    fireEvent.click(tiles()[3], { shiftKey: true });
    expect(chip("L")).toHaveTextContent("L 5/14");
    key("z");
    expect(chip("L")).toHaveTextContent("L 9/14");
  });

  it("switches channels with the chips and the number keys and shares the zoom while stepping", async () => {
    render(<App />);
    await reachBlink();
    const transform = () => (document.querySelector(".blink-canvas") as HTMLElement).style.transform;
    const fit = transform();
    expect(screen.getByRole("button", { name: "Fit" })).toHaveAttribute("aria-pressed", "true");
    key("+");
    const zoomed = transform();
    expect(zoomed).not.toBe(fit);
    key("ArrowRight");
    expect(transform()).toBe(zoomed);
    key("2");
    expect(chip("R")).toHaveAttribute("aria-pressed", "true");
    expect(tiles()[0]).toHaveAttribute("aria-label", R("2026-09-06_23-32-53"));
    expect(transform()).toBe(fit);
    await userEvent.click(chip("L"));
    expect(chip("L")).toHaveAttribute("aria-pressed", "true");
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-08-20_21-15-27"));
    await userEvent.click(screen.getByRole("button", { name: "2×" }));
    expect(screen.getByRole("button", { name: "2×" })).toHaveAttribute("aria-pressed", "true");
    key("0");
    expect(transform()).toBe(fit);
  });

  it("compares with the reference, side by side or held with C", async () => {
    render(<App />);
    await reachBlink();
    expect(document.querySelectorAll(".blink-pane")).toHaveLength(1);
    await userEvent.click(screen.getByRole("button", { name: "Compare with reference" }));
    const panes = document.querySelectorAll(".blink-pane");
    expect(panes).toHaveLength(2);
    expect(within(panes[0] as HTMLElement).getByRole("img").getAttribute("alt")).toContain(L("2026-08-17_22-57-01"));
    expect(within(panes[1] as HTMLElement).getByRole("img").getAttribute("alt")).toContain(L("2026-08-20_21-15-27"));
    expect((panes[0].querySelector(".blink-canvas") as HTMLElement).style.transform).toBe((panes[1].querySelector(".blink-canvas") as HTMLElement).style.transform);
    key("c");
    expect(document.querySelectorAll(".blink-pane")).toHaveLength(1);
    fireEvent.keyDown(view(), { key: "c" });
    expect(stage().getByRole("img").getAttribute("alt")).toContain(L("2026-08-17_22-57-01"));
    fireEvent.keyDown(view(), { key: "c", repeat: true });
    await new Promise((resolve) => setTimeout(resolve, 320));
    fireEvent.keyUp(view(), { key: "c" });
    expect(document.querySelectorAll(".blink-pane")).toHaveLength(1);
    expect(stage().getByRole("img").getAttribute("alt")).toContain(L("2026-08-20_21-15-27"));
  });

  it("blocks the start while a channel keeps fewer than two Lights and names it in the chosen language", async () => {
    window.localStorage.setItem("ultra-fast-wbpp.language", "zh-CN");
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await waitFor(() => expect(native.inspectCalibration).toHaveBeenCalled());
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    await userEvent.click(await screen.findByRole("button", { name: "闪视筛片（22 张 Light）" }));
    await screen.findByRole("heading", { name: /闪视筛片/ });
    const region = screen.getByRole("region", { name: /闪视筛片/ });
    await waitFor(() => expect(screen.getByRole("button", { name: /开始处理 · 采用 16\/22 张 Light/ })).toBeEnabled());
    const press = (name: string) => { fireEvent.keyDown(region, { key: name }); fireEvent.keyUp(region, { key: name }); };
    press("2"); press("End"); press("n");
    expect(screen.getByRole("button", { name: /^R 4\/8/ })).toBeInTheDocument();
    for (let step = 0; step < 5; step += 1) { press("ArrowLeft"); press("d"); }
    expect(screen.getByRole("button", { name: /^R 1\/8/ })).toBeInTheDocument();
    expect(screen.getByText("NGC 6822 × R：保留 1/8 张 Light，至少需要保留 2 张。")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
    expect(native.startRun).not.toHaveBeenCalled();
    press("k");
    expect(screen.getByRole("button", { name: /开始处理 · 采用 11\/22 张 Light/ })).toBeEnabled();
    // The import page's blocker list says the same thing while the channel is short.
    press("d");
    await userEvent.click(screen.getByRole("button", { name: "返回" }));
    expect(screen.getByText("NGC 6822 × R：保留 1/8 张 Light，至少需要保留 2 张。")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "打开闪视筛片" }).length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: /开始处理 · 采用 10\/22 张 Light/ })).toBeDisabled();
  });

  it("sends the selection with one decision per Light and shows the user's reasons on the result page", async () => {
    render(<App />);
    await reachBlink();
    await userEvent.click(screen.getByRole("button", { name: "Choose output folder" }));
    key(" ");
    expect(chip("L")).toHaveTextContent("L 10/14");
    key("r"); key("d");
    await waitFor(() => expect(screen.getByRole("button", { name: /Start processing · 16 of 22 Lights/ })).toBeEnabled());
    await userEvent.click(screen.getByRole("button", { name: /Start processing · 16 of 22 Lights/ }));
    await screen.findByRole("heading", { name: "Creating verified products" });
    expect(native.startRun).toHaveBeenCalledTimes(1);
    const request = native.startRun.mock.calls[0][0] as RunRequest;
    expect(request.reviewSelections).toEqual([]);
    expect(request.selection).toMatchObject({ schemaVersion: 1, kind: "ultra-fast-wbpp-selection", policy: "explicit-v1", undecided: "ERROR", origin: { sessionId: "demo-blink-session", blinkManifestSha256: `sha256:${"d".repeat(64)}`, flagsPolicyDigest: `sha256:${"b".repeat(64)}` } });
    expect(request.selection!.origin!.createdAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    const frames = manifest().frames;
    expect(request.selection!.decisions.map((item) => item.sourceSha256)).toEqual(frames.map((frame) => frame.sourceSha256));
    const restored = request.selection!.decisions.find((item) => item.sourceSha256 === frames.find((frame) => frame.name === L("2026-08-20_21-15-27"))!.sourceSha256)!;
    expect(restored).toEqual({ sourceSha256: restored.sourceSha256, decision: "KEEP", defaultDecision: "DROP", flags: ["BLINK_SKY_BRIGHT", "BLINK_SOURCES_LOW"] });
    const dropped = request.selection!.decisions.find((item) => item.sourceSha256 === frames.find((frame) => frame.name === L("2026-08-17_22-57-01"))!.sourceSha256)!;
    expect(dropped).toEqual({ sourceSha256: dropped.sourceSha256, decision: "DROP", defaultDecision: "KEEP", flags: [] });
    expect(request.selection!.decisions.filter((item) => item.decision === "KEEP")).toHaveLength(16);
    await act(async () => native.handlers?.onComplete({
      jobId: "run-1", outputDirectory: "/results/deep sky/NGC6822_2026-09-22_0100", artifacts: [solvedArtifact], gate: { decision: "ready", checks: readyChecks },
      screening: { admitted: 16, excluded: 6, counts: { PASS: 20, REVIEW: 0, HARD_FAIL: 2 }, frames: [
        { name: L("2026-08-17_22-57-01"), target: "NGC 6822", disposition: "PASS", admitted: false, summary: "", evidence: [], starCount: 5768, reason: "USER_DROP", flags: [] },
        { name: L("2026-08-20_21-15-27"), target: "NGC 6822", disposition: "PASS", admitted: true, summary: "", evidence: [], starCount: 2947, reason: "USER_KEEP_OVERRIDE", flags: ["BLINK_SKY_BRIGHT", "BLINK_SOURCES_LOW"] },
        { name: L("2026-08-20_23-28-04"), target: "NGC 6822", disposition: "HARD_FAIL", admitted: false, summary: "clouds", evidence: ["cloud"], starCount: 6, reason: "USER_DROP", flags: ["BLINK_UNREGISTRABLE", "BLINK_FEW_STARS"] },
      ] },
    }));
    const section = await screen.findByRole("region", { name: "Light screening" });
    expect(within(section).getByText("16 stacked · 6 excluded")).toBeInTheDocument();
    expect(within(section).getAllByText("Dropped by you")).toHaveLength(2);
    expect(within(section).getByText("Kept by you despite a flag")).toBeInTheDocument();
    expect(within(section).getByText("Bright sky; Few stars detected")).toBeInTheDocument();
    expect(within(section).getByText("Cannot be registered; Almost no stars")).toBeInTheDocument();
  });

  it("keeps the session for a calibration-only addition and invalidates it when Lights are imported again", async () => {
    render(<App />);
    await reachBlink();
    key(" ");
    await userEvent.click(screen.getByRole("button", { name: "Back" }));
    // The launch bar and its notice both reopen the session.
    const openButtons = () => screen.getAllByRole("button", { name: "Open blink selection" });
    expect(openButtons()).toHaveLength(2);
    expect(openButtons()[1]).not.toHaveClass("primary");
    expect(screen.getByRole("button", { name: /Start processing · 17 of 22 Lights/ })).toHaveClass("primary");
    expect(screen.getByText("Your blink decisions will be used").closest(".screening-notice")).toHaveTextContent("17 of 22 Lights kept · 5 dropped");
    expect(screen.getByRole("navigation", { name: "Workflow" })).toHaveTextContent("17/22");
    const extra = asset("/data/calibration/master flat L.xisf", "L", "MASTER_FLAT");
    native.inspectPaths.mockResolvedValueOnce({ projectName: "NGC 6822", totalFiles: 1, assets: [extra], sources: [{ role: extra.role, paths: [extra.path], fileCount: 1, confidence: 1, needsConfirmation: false }] });
    await userEvent.click(screen.getByRole("button", { name: /Choose folder$/ }));
    expect(await screen.findAllByRole("button", { name: "Open blink selection" })).toHaveLength(2);
    await userEvent.click(openButtons()[1]);
    await screen.findByRole("heading", { name: /Blink and select/ });
    expect(native.blinkMeasure).toHaveBeenCalledTimes(1);
    expect(chip("L")).toHaveTextContent("L 10/14");
    // The master flat travels with the next measurement request.
    await userEvent.click(screen.getByRole("button", { name: "Back" }));
    await userEvent.click(screen.getByRole("button", { name: /Choose folder$/ }));
    expect(await screen.findByRole("button", { name: "Blink & select (22 Lights)" })).toHaveClass("primary");
    expect(screen.getByRole("button", { name: /Start processing \(automatic screening\)/ })).toBeInTheDocument();
    expect(screen.queryByText("Your blink decisions will be used")).not.toBeInTheDocument();
    expect(within(screen.getByRole("navigation", { name: "Workflow" })).queryByRole("button", { name: /Blink/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Blink & select (22 Lights)" }));
    await screen.findByRole("heading", { name: /Blink and select/ });
    expect(native.blinkMeasure).toHaveBeenCalledTimes(2);
    expect(native.blinkMeasure).toHaveBeenLastCalledWith({ paths: lightPaths(), masterFlats: [{ filter: "L", path: extra.path }] });
    expect(chip("L")).toHaveTextContent("L 9/14");
  });

  it("loads the zoom image on demand when zoomed in and keeps the filmstrip image while playing", async () => {
    render(<App />);
    await reachBlink();
    key("+"); key("+");
    const first = manifest().frames.find((frame) => frame.name === L("2026-08-20_21-15-27"))!;
    await waitFor(() => expect(native.loadBlinkPreview).toHaveBeenCalledWith(manifest().sessionDirectory, first.previews.zoom));
    await waitFor(() => expect(stage().getByRole("img", { name: new RegExp(L("2026-08-20_21-15-27")) })).toHaveAttribute("src", expect.stringMatching(/^data:image\/png;base64,/)));
    key("p");
    expect(stage().getByRole("img", { name: new RegExp(L("2026-08-20_21-15-27")) })).toHaveAttribute("src", expect.stringMatching(/^data:image\/svg\+xml/));
  });

  it("reports a failed measurement without leaving the import page", async () => {
    native.blinkMeasure.mockRejectedValueOnce(new Error("BLINK_NO_LIGHTS: no Light frames"));
    render(<App />);
    await importLights();
    await userEvent.click(screen.getByRole("button", { name: "Blink & select (22 Lights)" }));
    expect(await screen.findByText("Blink measurement failed: Error: BLINK_NO_LIGHTS: no Light frames")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Blink & select (22 Lights)" })).toBeEnabled();
    expect(screen.queryByRole("heading", { name: /Blink and select/ })).not.toBeInTheDocument();
  });
});

describe("browser demo", () => {
  it("opens the bundled blink session for ?demo=blink without a native runtime", async () => {
    native.runtime = false;
    window.history.replaceState({}, "", "/?demo=blink");
    render(<App />);
    expect(await screen.findByRole("heading", { name: /Blink and select/ })).toBeInTheDocument();
    expect(native.blinkMeasure).not.toHaveBeenCalled();
    expect(chip("L")).toHaveTextContent("L 9/14 · 8 flagged");
    expect(tiles()).toHaveLength(14);
    expect(screen.getByRole("button", { name: /Start processing · 16 of 22 Lights/ })).toBeEnabled();
    key("+"); key("+");
    await waitFor(() => expect(stage().getByRole("img", { name: new RegExp(L("2026-08-20_21-15-27")) })).toHaveAttribute("src", expect.stringMatching(/^data:image\/svg\+xml/)));
    expect(native.loadBlinkPreview).not.toHaveBeenCalled();
  });
});
