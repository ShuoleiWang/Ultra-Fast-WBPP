import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  BlinkMeasureRequest,
  BlinkMeasureResponse,
  InspectedAsset,
  PipelineEventHandlers,
  RunRequest,
} from "./types";

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
    getCapabilities: vi.fn(async () => ({
      platform: "macos",
      chip: "Apple M3 Pro",
      cpuBackend: "Native CPU execution",
      gpuBackend: "Metal execution",
      optimizationTier: "M3_PRO_TUNED",
      available: true,
      drizzleAvailable: true,
      solverAvailable: true,
      runtimeVersion: "0.1.0",
    })),
    inspectPaths: (...args: unknown[]) => native.inspectPaths(...args),
    inspectCalibration: (...args: unknown[]) => native.inspectCalibration(...args),
    inspectQuality: (...args: unknown[]) => native.inspectQuality(...args),
    blinkMeasure: (...args: unknown[]) => native.blinkMeasure(...args),
    loadBlinkPreview: (...args: unknown[]) => native.loadBlinkPreview(...args),
    hashSources: vi.fn(async () => ({ entries: [] })),
    pickInputFiles: vi.fn(async () => []),
    pickInputDirectories: vi.fn(async () => ["/data/NGC 6822"]),
    pickOutputParent: vi.fn(async () => "/results/deep sky"),
    startRun: (...args: unknown[]) => native.startRun(...args),
    cancelRun: vi.fn(async () => undefined),
    catalogList: vi.fn(async () => ({ schemaVersion: 1, catalogRoot: "/catalogs", catalogs: [] })),
    catalogDoctor: vi.fn(async () => ({
      schemaVersion: 1,
      ok: true,
      catalogRoot: "/catalogs",
      config: { present: true, valid: true },
      installedSetBindingReady: true,
      message: "ready",
    })),
    solverDoctor: vi.fn(async () => ({
      schemaVersion: 1,
      engineVersion: "0.1.0",
      backends: [
        {
          backendId: "astrometry-net",
          displayName: "solve-field",
          version: "0.97",
          available: true,
          executionReady: true,
          metadata: { probe: { path: "/opt/homebrew/bin/solve-field", version: "0.97", executionReady: true } },
        },
      ],
    })),
    startCatalogInstall: vi.fn(),
    cancelCatalogInstall: vi.fn(),
    verifyCatalog: vi.fn(),
    openProviderTerms: vi.fn(),
    revealOutput: vi.fn(async () => undefined),
  },
  listenForDesktopDrops: vi.fn(async () => () => undefined),
  listenForPipelineEvents: vi.fn(async (handlers: PipelineEventHandlers) => {
    native.handlers = handlers;
    return () => undefined;
  }),
  listenForCatalogEvents: vi.fn(async () => () => undefined),
}));

import App from "./App";
import { mockPreviewCanvas, paintStage, reviewChannel } from "./test/blink";
import { demoBlinkManifest } from "./demoAutopilot";

const manifest = (): BlinkMeasureResponse => demoBlinkManifest();
const lightPaths = () => manifest().frames.map((frame) => frame.path);
const L = (time: string) => `NGC 6822_300.00s_L_${time}.fits`;

const asset = (path: string, filter: string, role: InspectedAsset["role"] = "LIGHT"): InspectedAsset => ({
  path,
  role,
  width: 6252,
  height: 4176,
  channels: 1,
  filter,
  target: role === "LIGHT" ? "NGC 6822" : "UNKNOWN",
  camera: "QHY268M",
  exposureSeconds: role === "LIGHT" ? 300 : 1,
  temperatureCelsius: -10,
  gain: 26,
  offset: 30,
  binning: [1, 1],
  cfaPattern: "NONE",
  readoutMode: "Mode 1",
  sourceSha256: null,
});
const inventory = () => {
  const assets = [
    ...manifest().frames.map((frame) => asset(frame.path, frame.filter)),
    asset("/data/calibration/flat L.fits", "L", "FLAT"),
    asset("/data/calibration/flat R.fits", "R", "FLAT"),
    asset("/data/calibration/dark.fits", "L", "DARK"),
    asset("/data/calibration/bias.fits", "L", "BIAS"),
  ];
  return {
    projectName: "NGC 6822",
    totalFiles: assets.length,
    assets,
    sources: assets.map((item) => ({
      role: item.role,
      paths: [item.path],
      fileCount: 1,
      confidence: 1,
      needsConfirmation: false,
    })),
  };
};
const solvedArtifact = {
  kind: "SOLVED_MONO_FITS" as const,
  name: "NGC 6822_L.fits",
  path: "/results/deep sky/run/NGC 6822_L.fits",
  detail: "final WCS",
  filter: "L",
  target: "NGC 6822",
  receipt: {
    artifactId: "final-l",
    relativePath: "NGC 6822_L.fits",
    sha256: "a".repeat(64),
    sizeBytes: 4096,
    astrometry: {
      referenceFrame: "ICRS",
      projection: "TAN",
      centerRaDegrees: 296.2,
      centerDecDegrees: -14.8,
      pixelScaleArcsec: 0.78,
      rotationDegrees: 0,
      rmsPixels: 0.3,
      rmsArcsec: 0.24,
      matchedStars: 120,
      parity: "POSITIVE" as const,
      catalogIdentity: "b".repeat(64),
      indexIdentities: ["astrometry.net:index:4108"],
      correspondenceSha256: "c".repeat(64),
      catalogManaged: true as const,
      installedSetIdentity: "d".repeat(64),
      catalogManifestSha256: "e".repeat(64),
      indexArtifacts: [],
      wcsSha256: "1".repeat(64),
    },
  },
};
const readyChecks = [
  "final-project-products-present",
  "final-project-receipts-valid",
  "final-project-astrometry-validated",
].map((code) => ({ code, required: true, passed: true, artifactIds: ["final-l"], message: "ready" }));

const view = () => screen.getByRole("region", { name: /Blink and select/ });
const stage = () => within(document.querySelector(".blink-stage") as HTMLElement);
const tiles = () => screen.getAllByRole("button", { name: /^NGC 6822_300\.00s_/ });
const currentTile = () => tiles().find((tile) => tile.getAttribute("aria-current") === "true")!;
const chip = (filter: string) => screen.getByRole("button", { name: new RegExp(`^${filter} · \\d+/\\d+ viewed`) });
const key = (name: string, init: Record<string, unknown> = {}) => {
  fireEvent.keyDown(view(), { key: name, ...init });
  fireEvent.keyUp(view(), { key: name, ...init });
};

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
  mockPreviewCanvas();
  native.runtime = true;
  native.handlers = undefined;
  window.localStorage.clear();
  // Reduced motion keeps playback off until a test asks for it, so the current frame is deterministic.
  vi.spyOn(window, "matchMedia").mockImplementation((query: string) => ({
    matches: query.includes("reduced-motion"),
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false,
  }));
  native.inspectPaths.mockReset().mockImplementation(async () => inventory());
  native.inspectCalibration
    .mockReset()
    .mockResolvedValue({ schemaVersion: 1, status: "READY", calibrationReady: true, groups: [], issues: [] });
  native.blinkMeasure.mockReset().mockImplementation(async () => manifest());
  native.loadBlinkPreview
    .mockReset()
    .mockImplementation(
      async (_directory: string, relativePath: string) =>
        `data:image/png;base64,${Buffer.from(relativePath).toString("base64")}`,
    );
  native.startRun.mockReset().mockResolvedValue({
    jobId: "run-1",
    accepted: true,
    executionMode: "native",
    outputDirectory: "/results/deep sky/run",
  });
  native.inspectQuality.mockReset();
});
afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  window.history.replaceState({}, "", "/");
});

describe("human Blink screening", () => {
  it("requires review, opens paused in time order, and treats flags as advice", async () => {
    render(<App />);
    await importLights();
    expect(screen.getByRole("button", { name: /^Start processing/ })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Blink & select (22 Lights)" }));
    await screen.findByRole("heading", { name: /Blink and select/ });
    expect(native.blinkMeasure).toHaveBeenCalledWith({
      paths: lightPaths(),
      masterFlats: [],
      masterDarks: [],
    } satisfies BlinkMeasureRequest);
    expect(chip("L")).toHaveTextContent("0/14 viewed");
    expect(chip("R")).toHaveTextContent("0/8 viewed");
    expect(screen.getByRole("button", { name: "Play" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByRole("button", { name: "Apply flags" })).not.toBeInTheDocument();
    expect(tiles()[0]).toHaveAttribute("aria-label", L("2026-08-17_22-36-34"));
    expect(tiles().every((tile) => tile.getAttribute("aria-pressed") === "true")).toBe(true);
    expect(screen.getByRole("checkbox", { name: "Kept only" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Confirm L/ })).toBeDisabled();
    await paintStage();
    expect(chip("L")).toHaveTextContent("1/14 viewed");
    expect(screen.getByRole("button", { name: /Confirm L/ })).toBeDisabled();
    expect(stage().getByRole("img")).toHaveAttribute("aria-label", expect.stringContaining(L("2026-08-17_22-36-34")));
    await userEvent.click(screen.getByRole("button", { name: "Chronological" }));
    expect(tiles()[0]).toHaveAttribute("aria-label", L("2026-08-20_21-15-27"));
    key("Home");
    expect(stage().getByTitle(/moonlit or hazy night/)).toHaveTextContent("Bright sky ×2.43");
    expect(screen.getAllByText(/suggestion only/).length).toBeGreaterThan(0);
  });

  it("cannot bypass the channel gate with bulk decisions, then confirms channels in order", async () => {
    render(<App />);
    await reachBlink();
    await userEvent.click(screen.getByRole("button", { name: "Choose output folder" }));
    fireEvent.click(screen.getAllByRole("button", { name: "Keep night" })[0]);
    expect(chip("L")).toHaveTextContent("0/14 viewed");
    expect(screen.getByRole("button", { name: /Start processing/ })).toBeDisabled();
    await reviewChannel(/Confirm L/);
    expect(chip("L")).toHaveTextContent("14/14 viewed ✓");
    expect(chip("R")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /Start processing/ })).toBeDisabled();
    await reviewChannel(/Confirm R/);
    expect(screen.getByRole("button", { name: /Start processing/ })).toBeEnabled();
    key(" ");
    expect(screen.getByRole("button", { name: /Start processing/ })).toBeDisabled();
    expect(chip("R")).not.toHaveTextContent("✓");
    key("z");
    expect(screen.getByRole("button", { name: /Start processing/ })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /Confirm R/ }));
    expect(screen.getByRole("button", { name: /Start processing/ })).toBeEnabled();
  });

  it("steps, decides with K/D, jumps to flags/reference, and undoes whole-night/range changes", async () => {
    render(<App />);
    await reachBlink();
    const first = currentTile();
    await paintStage();
    key("d");
    expect(first).toHaveAttribute("aria-pressed", "false");
    expect(currentTile()).not.toBe(first);
    await paintStage();
    key("k");
    expect(currentTile()).toBe(tiles()[2]);
    key("Home");
    key("ArrowLeft");
    expect(currentTile()).toBe(tiles().at(-1));
    key("r");
    expect(currentTile()).toHaveAttribute("aria-label", L("2026-08-17_22-57-01"));
    key("f");
    expect(currentTile()).not.toHaveAttribute("aria-label", L("2026-08-17_22-57-01"));
    key("z");
    expect(first).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getAllByRole("button", { name: "Drop night" })[0]);
    expect(
      tiles()
        .slice(0, 7)
        .every((tile) => tile.getAttribute("aria-pressed") === "false"),
    ).toBe(true);
    key("z", { metaKey: true });
    expect(tiles().every((tile) => tile.getAttribute("aria-pressed") === "true")).toBe(true);
    fireEvent.click(tiles()[0]);
    fireEvent.click(tiles()[3], { shiftKey: true });
    expect(
      tiles()
        .slice(0, 4)
        .every((tile) => tile.getAttribute("aria-pressed") === "false"),
    ).toBe(true);
    key("z");
    expect(tiles().every((tile) => tile.getAttribute("aria-pressed") === "true")).toBe(true);
    fireEvent.keyDown(screen.getByRole("combobox", { name: "Frames per second" }), { key: "d" });
    expect(currentTile()).toHaveAttribute("aria-pressed", "true");
  });

  it("waits for a painted image before advancing playback and stops at the channel end", async () => {
    render(<App />);
    await reachBlink();
    vi.useFakeTimers();
    key("p");
    await act(async () => {
      vi.advanceTimersByTime(2_000);
    });
    expect(currentTile()).toBe(tiles()[0]);
    const image = screen.getByTestId("blink-stage-decoder");
    Object.defineProperties(image, {
      complete: { value: true },
      naturalWidth: { value: 782 },
      naturalHeight: { value: 522 },
    });
    fireEvent.load(image);
    await act(async () => {
      vi.advanceTimersByTime(20);
    });
    await act(async () => {
      vi.advanceTimersByTime(500);
    });
    expect(currentTile()).toBe(tiles()[1]);
    await act(async () => {
      vi.advanceTimersByTime(2_000);
    });
    expect(currentTile()).toBe(tiles()[1]);
    key("End");
    expect(screen.getByRole("button", { name: "Play" })).toBeInTheDocument();
    vi.useRealTimers();
    await paintStage();
    // Restart from the beginning, without treating prefetched thumbnails as viewed.
    key("p");
    expect(currentTile()).toBe(tiles()[0]);
    key("Escape");
    expect(chip("L")).toHaveTextContent("2/14 viewed");
  });

  it("does not count hidden previews and marks the frame when the view becomes visible", async () => {
    const hidden = vi.spyOn(document, "hidden", "get").mockReturnValue(true);
    render(<App />);
    await reachBlink();
    await paintStage();
    expect(chip("L")).toHaveTextContent("0/14 viewed");
    hidden.mockReturnValue(false);
    fireEvent(document, new Event("visibilitychange"));
    await waitFor(() => expect(chip("L")).toHaveTextContent("1/14 viewed"));
  });

  it("shows failed previews, retries them, and never counts a decode error as viewed", async () => {
    render(<App />);
    await reachBlink();
    fireEvent.error(screen.getByTestId("blink-stage-decoder"));
    expect(await screen.findByText("This preview could not be displayed")).toBeInTheDocument();
    expect(chip("L")).toHaveTextContent("0/14 viewed");
    expect(screen.getByRole("button", { name: /Keep and next/ })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Retry preview" }));
    await screen.findByTestId("blink-stage-decoder");
    await paintStage();
    expect(chip("L")).toHaveTextContent("1/14 viewed");
    key("ArrowRight");
    fireEvent.error(screen.getByTestId("blink-stage-decoder"));
    await screen.findByText("This preview could not be displayed");
    key("d");
    expect(chip("L")).toHaveTextContent("2/14 viewed");
    expect(tiles()[1]).toHaveAttribute("aria-pressed", "false");
  });

  it("shares zoom/pan in reference comparison and loads detailed images on demand", async () => {
    const context = mockPreviewCanvas();
    render(<App />);
    await reachBlink();
    await paintStage();
    key("+");
    key("+");
    await waitFor(() =>
      expect(native.loadBlinkPreview).toHaveBeenCalledWith(
        manifest().sessionDirectory,
        manifest().frames[0].previews.zoom,
      ),
    );
    await paintStage();
    const scale = context.scale.mock.lastCall;
    key("ArrowRight");
    await paintStage();
    expect(context.scale.mock.lastCall).toEqual(scale);
    fireEvent.click(screen.getByRole("button", { name: "Compare with reference" }));
    await paintStage();
    expect(document.querySelectorAll(".blink-pane")).toHaveLength(2);
    expect(stage().getAllByRole("img")[0]).toHaveAttribute(
      "aria-label",
      expect.stringContaining(L("2026-08-17_22-57-01")),
    );
    expect(context.scale.mock.calls.at(-1)).toEqual(context.scale.mock.calls.at(-2));
    key("2");
    expect(chip("R")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Fit" })).toHaveAttribute("aria-pressed", "true");
    key("1");
    expect(chip("L")).toHaveAttribute("aria-pressed", "true");
  });

  it("blocks a reviewed channel with fewer than two kept Lights", async () => {
    render(<App />);
    await reachBlink();
    await userEvent.click(screen.getByRole("button", { name: "Choose output folder" }));
    await reviewChannel(/Confirm L/);
    await reviewChannel(/Confirm R/);
    for (const tile of tiles().slice(1)) {
      fireEvent.click(tile);
      key(" ");
    }
    fireEvent.click(screen.getByRole("button", { name: /Confirm R/ }));
    expect(screen.getByText("NGC 6822 × R: 1 of 8 Lights kept; at least 2 must be kept.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Start processing/ })).toBeDisabled();
  });

  it("sends the exact manual decisions and completed review identity to the controller", async () => {
    render(<App />);
    await reachBlink();
    await paintStage();
    key("d");
    await reviewChannel(/Confirm L/);
    await reviewChannel(/Confirm R/);
    await userEvent.click(screen.getByRole("button", { name: "Choose output folder" }));
    await userEvent.click(screen.getByRole("button", { name: /Start processing · 21 of 22 Lights/ }));
    expect(native.startRun).toHaveBeenCalledTimes(1);
    const request = native.startRun.mock.calls[0][0] as RunRequest;
    expect(request.reviewSelections).toEqual([]);
    expect(request.selection).toMatchObject({
      policy: "explicit-v1",
      undecided: "ERROR",
      origin: { blinkManifestSha256: manifest().manifestSha256 },
    });
    expect(request.selection!.decisions[0].decision).toBe("DROP");
    expect(request.selection!.decisions.filter((item) => item.decision === "KEEP")).toHaveLength(21);
    expect(
      request.selection!.decisions.some((item) => item.defaultDecision === "DROP" && item.decision === "KEEP"),
    ).toBe(true);
    expect(new Set(request.blinkReview!.reviewedSourceSha256s)).toEqual(
      new Set(manifest().frames.map((frame) => frame.sourceSha256)),
    );
    expect(request.blinkReview!.confirmedChannelIds).toEqual(manifest().channels.map((channel) => channel.channelId));
    await act(async () =>
      native.handlers?.onComplete({
        jobId: "run-1",
        outputDirectory: "/results/deep sky/run",
        artifacts: [solvedArtifact],
        gate: { decision: "ready", checks: readyChecks },
        screening: {
          admitted: 21,
          excluded: 1,
          counts: { PASS: 20, REVIEW: 0, HARD_FAIL: 2 },
          frames: [
            {
              name: L("2026-08-17_22-36-34"),
              target: "NGC 6822",
              disposition: "PASS",
              admitted: false,
              summary: "",
              evidence: [],
              starCount: 1200,
              reason: "USER_DROP",
              flags: [],
            },
          ],
        },
      }),
    );
    expect(await screen.findByText("Dropped by you")).toBeInTheDocument();
  });

  it("retains decisions on calibration additions, resets on Light reimport, and forwards all masters", async () => {
    render(<App />);
    await reachBlink();
    await paintStage();
    key(" ");
    await userEvent.click(screen.getByRole("button", { name: "Back" }));
    const extra = [
      asset("/data/calibration/master flat L.xisf", "L", "MASTER_FLAT"),
      { ...asset("/data/calibration/master dark.fit", "L", "MASTER_DARK"), exposureSeconds: 300 },
      asset("/data/calibration/master bias.fit", "L", "MASTER_BIAS"),
    ];
    native.inspectPaths.mockResolvedValueOnce({
      projectName: "NGC 6822",
      totalFiles: extra.length,
      assets: extra,
      sources: extra.map((a) => ({
        role: a.role,
        paths: [a.path],
        fileCount: 1,
        confidence: 1,
        needsConfirmation: false,
      })),
    });
    await userEvent.click(screen.getByRole("button", { name: /Choose folder$/ }));
    await userEvent.click(screen.getAllByRole("button", { name: "Open blink selection" })[0]);
    expect(native.blinkMeasure).toHaveBeenCalledTimes(1);
    expect(tiles()[0]).toHaveAttribute("aria-pressed", "false");
    await userEvent.click(screen.getByRole("button", { name: "Back" }));
    await importLights();
    await userEvent.click(screen.getByRole("button", { name: "Blink & select (22 Lights)" }));
    await screen.findByRole("heading", { name: /Blink and select/ });
    expect(chip("L")).toHaveTextContent("0/14 viewed");
    expect(tiles()[0]).toHaveAttribute("aria-pressed", "true");
    expect(native.blinkMeasure).toHaveBeenLastCalledWith({
      paths: lightPaths(),
      masterFlats: [{ filter: "L", path: extra[0].path }],
      masterDarks: [{ path: extra[1].path, exposureSeconds: 300 }],
      masterBias: extra[2].path,
    });
  });

  it("requires both complementary displays, handles background failures and offers native shape comparison", async () => {
    const measured = manifest();
    for (const frame of measured.frames) {
      frame.previews.diagnostic = {
        field: `diagnostic/${frame.index}-field.png`,
        background: `diagnostic/${frame.index}-background.png`,
        nativeShape: `diagnostic/${frame.index}-shape.png`,
        nativeSignal: `diagnostic/${frame.index}-signal.png`,
      };
      frame.diagnostics = {
        algorithm: "blink-complementary-display-v2",
        calibration: "calibrated",
        relativeSignal: 0.5,
        relativeNoise: 1.2,
        matchedSignalNoise: 2.4,
        backgroundStatus: "ready",
        backgroundSpan: 2,
        nativeStatus: "ready",
        shapeRegions: 6,
      };
    }
    native.blinkMeasure.mockResolvedValueOnce(measured);
    render(<App />);
    await reachBlink();
    await paintStage();
    expect(chip("L")).toHaveTextContent("0/14 viewed");
    const background = await screen.findByTestId("blink-background-decoder");
    fireEvent.error(background);
    expect(screen.getByRole("button", { name: /Keep and next/ })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Retry preview" }));
    const restored = await screen.findByTestId("blink-background-decoder");
    Object.defineProperties(restored, {
      complete: { value: true },
      naturalWidth: { value: 33 },
      naturalHeight: { value: 22 },
    });
    fireEvent.load(restored);
    await waitFor(() => expect(chip("L")).toHaveTextContent("1/14 viewed"));
    expect(screen.getByRole("button", { name: /Keep and next/ })).toBeEnabled();
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Blink display" }), "field");
    await waitFor(() =>
      expect(native.loadBlinkPreview).toHaveBeenCalledWith(
        measured.sessionDirectory,
        measured.frames[0].previews.diagnostic!.field,
      ),
    );
    await userEvent.click(screen.getByRole("button", { name: "Enlarge star comparison" }));
    const dialog = screen.getByRole("dialog", { name: "Original-pixel star regions" });
    expect(dialog).toHaveTextContent("6/9 measurable regions");
    fireEvent.keyDown(dialog, { key: "d" });
    expect(tiles()[0]).toHaveAttribute("aria-pressed", "true");
    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("reports a failed measurement without leaving the import page", async () => {
    native.blinkMeasure.mockRejectedValueOnce(new Error("BLINK_NO_LIGHTS: no Light frames"));
    render(<App />);
    await importLights();
    await userEvent.click(screen.getByRole("button", { name: "Blink & select (22 Lights)" }));
    expect(
      await screen.findByText("Blink measurement failed: Error: BLINK_NO_LIGHTS: no Light frames"),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Start processing/ })).toBeDisabled();
  });
});

describe("browser demo", () => {
  it("opens the bundled session without native calls", async () => {
    native.runtime = false;
    window.history.replaceState({}, "", "/?demo=blink");
    render(<App />);
    await screen.findByRole("heading", { name: /Blink and select/ });
    expect(native.blinkMeasure).not.toHaveBeenCalled();
    expect(tiles()).toHaveLength(14);
    await paintStage();
    expect(chip("L")).toHaveTextContent("1/14 viewed");
  });
});
