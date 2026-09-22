import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { CatalogEventHandlers, InspectedAsset, PipelineEventHandlers } from "./types";

const native = vi.hoisted(() => ({
  handlers: undefined as PipelineEventHandlers | undefined,
  catalogHandlers: undefined as CatalogEventHandlers | undefined,
  inspectError: undefined as Error | undefined,
  dropHandler: undefined as ((paths: string[]) => void) | undefined,
  inspectPaths: vi.fn(),
  inspectCalibration: vi.fn(),
  qualityDisposition: "PASS" as "PASS" | "REVIEW" | "HARD_FAIL",
  reviewFirst: false,
  cfa: false,
  unknownCfa: false,
  useMasterDark: false,
  catalogReady: true,
  catalogInspectionError: undefined as Error | undefined,
  solverReady: true,
  solverError: undefined as Error | undefined,
  platform: "macos" as "macos" | "windows",
  astapReady: false,
  astapScienceReady: false,
  blinkMeasure: vi.fn(),
  startRun: vi.fn(),
  cancelRun: vi.fn(),
  startCatalogInstall: vi.fn(),
  cancelCatalogInstall: vi.fn(),
  verifyCatalog: vi.fn(),
  openProviderTerms: vi.fn(),
  hashSources: vi.fn(),
  inspectQuality: vi.fn(),
  revealOutput: vi.fn(),
  capabilityCalls: 0,
  solverSetupCalls: 0,
  dropListenerCalls: 0,
  pipelineListenerCalls: 0,
  catalogListenerCalls: 0,
  lightCount: 1,
}));

const lightAsset = (path: string, filter = "R"): InspectedAsset => ({
  path, role: "LIGHT", width: 6248, height: 4176, channels: 1, filter, target: "盾牌座 Panel 1", camera: "QHY268M",
  exposureSeconds: 180, temperatureCelsius: -10, gain: 100, offset: 50, binning: [1, 1], cfaPattern: native.cfa ? "RGGB" : native.unknownCfa ? "UNKNOWN" : "NONE", readoutMode: "Mode 1",
  sourceSha256: null,
});

const inventory = () => {
  const assets: InspectedAsset[] = [
    ...Array.from({ length: native.lightCount }, (_, index) => lightAsset(`/数据/盾牌座/亮场 ${String(index + 1).padStart(2, "0")}.fit`)),
    { ...lightAsset("/数据/校准/平场 R.fit"), role: "FLAT", target: "UNKNOWN", exposureSeconds: 1 },
    { ...lightAsset("/数据/校准/暗场.fit"), role: "DARK", target: "UNKNOWN" },
    { ...lightAsset("/数据/校准/偏置.fit"), role: "BIAS", target: "UNKNOWN", exposureSeconds: 0 },
  ];
  if (native.useMasterDark) {
    assets.splice(assets.findIndex((asset) => asset.role === "DARK"), 1, {
      ...lightAsset("/数据/校准/旧 MasterDark.fit"), role: "MASTER_DARK", target: "UNKNOWN", camera: "UNKNOWN", gain: null,
      offset: null, filter: "UNKNOWN", cfaPattern: "UNKNOWN", readoutMode: "UNKNOWN", temperatureCelsius: null, exposureSeconds: null,
      sourceSha256: `sha256:${"8".repeat(64)}`,
    });
  }
  return {
    projectName: "盾牌座 马赛克", totalFiles: assets.length, assets,
    sources: assets.map((asset) => ({ role: asset.role, paths: [asset.path], fileCount: 1, confidence: 1, needsConfirmation: false })),
  };
};

const catalogListing = {
  schemaVersion: 1 as const,
  catalogRoot: "/Users/example/测试/.openastroflow/catalogs/astrometry-net",
  catalogs: [{
    catalogId: "astrometry-net-4107-4112", provider: "Astrometry.net", version: "checked", totalSizeBytes: 349_692_480,
    artifactCount: 6, installedArtifactsBySize: 0, fullyInstalledBySize: false, allowedDownloadOrigins: ["https://data.astrometry.net"], artifacts: [],
    providerTerms: { acceptanceId: "astrometry-net-index-data-2026-09", url: "https://astrometry.net/doc/readme.html#getting-index-files", summary: "Provider terms must be reviewed.", licenseStatus: "provider-specific-unresolved", requiresExplicitAcceptance: true },
  }],
};

vi.mock("./bridge", () => ({
  hasTauriRuntime: () => true,
  desktopBridge: {
    getCapabilities: vi.fn(async () => {
      native.capabilityCalls += 1;
      if (native.platform === "windows") return { platform: "windows", chip: "AMD Ryzen 7 5800H with Radeon Graphics", cpuBackend: "Native CPU execution", gpuBackend: "GPU acceleration not used", optimizationTier: "WINDOWS_X64", available: true, drizzleAvailable: true, solverAvailable: true, runtimeVersion: "0.1.0" };
      return { platform: "macos", chip: "Apple M3 Pro", cpuBackend: "Native CPU execution", gpuBackend: "Metal execution", optimizationTier: "M3_PRO_TUNED", available: true, drizzleAvailable: true, solverAvailable: true, runtimeVersion: "0.1.0" };
    }),
    blinkMeasure: (...args: unknown[]) => native.blinkMeasure(...args),
    loadBlinkPreview: vi.fn(async () => "data:image/png;base64,fixture"),
    inspectPaths: (...args: unknown[]) => native.inspectPaths(...args),
    inspectCalibration: (...args: unknown[]) => native.inspectCalibration(...args),
    inspectQuality: (...args: unknown[]) => native.inspectQuality(...args),
    hashSources: (...args: unknown[]) => native.hashSources(...args),
    pickInputFiles: vi.fn(async () => []), pickInputDirectories: vi.fn(async () => ["/数据/盾牌座 会话"]), pickOutputParent: vi.fn(async () => "/结果/深空 输出"),
    startRun: (...args: unknown[]) => native.startRun(...args), cancelRun: (...args: unknown[]) => native.cancelRun(...args),
    catalogList: vi.fn(async () => { native.solverSetupCalls += 1; return catalogListing; }),
    catalogDoctor: vi.fn(async () => { native.solverSetupCalls += 1; if (native.catalogInspectionError) throw native.catalogInspectionError; return { schemaVersion: 1, ok: native.catalogReady, catalogRoot: catalogListing.catalogRoot, config: { present: native.catalogReady, valid: native.catalogReady }, installedSetBindingReady: native.catalogReady, message: native.catalogReady ? "ready" : "missing" }; }),
    solverDoctor: vi.fn(async () => { native.solverSetupCalls += 1; if (native.solverError) throw native.solverError; return { schemaVersion: 1, engineVersion: "0.1.0", backends: [
      { backendId: "astrometry-net", displayName: "solve-field", version: "0.97", available: native.solverReady, executionReady: native.solverReady, metadata: { probe: { path: "/opt/homebrew/bin/solve-field", version: "0.97", executionReady: native.solverReady } } },
      native.astapReady
        ? { backendId: "astap", displayName: "ASTAP", version: "2026.09.01", available: true, executionReady: true, scienceReady: native.astapScienceReady, reason: native.astapScienceReady ? null : "STAR_DATABASE_MISSING: no D20 or larger star database beside astap_cli.exe", metadata: { probe: { path: "C:\\Program Files\\astap\\astap_cli.exe", version: "2026.09.01", executionReady: true } } }
        : { backendId: "astap", displayName: "ASTAP", version: "unavailable", available: false, executionReady: false, reason: "not installed" },
    ] }; }),
    startCatalogInstall: (...args: unknown[]) => native.startCatalogInstall(...args), cancelCatalogInstall: (...args: unknown[]) => native.cancelCatalogInstall(...args), verifyCatalog: (...args: unknown[]) => native.verifyCatalog(...args),
    openProviderTerms: (...args: unknown[]) => native.openProviderTerms(...args),
    revealOutput: (...args: unknown[]) => native.revealOutput(...args),
  },
  listenForDesktopDrops: vi.fn(async (onPaths: (paths: string[]) => void) => { native.dropListenerCalls += 1; native.dropHandler = onPaths; return () => undefined; }),
  listenForPipelineEvents: vi.fn(async (handlers: PipelineEventHandlers) => { native.pipelineListenerCalls += 1; native.handlers = handlers; return () => undefined; }),
  listenForCatalogEvents: vi.fn(async (handlers: CatalogEventHandlers) => { native.catalogListenerCalls += 1; native.catalogHandlers = handlers; return () => undefined; }),
}));

import App from "./App";
import { inventoryBlinkManifest, mockPreviewCanvas, reviewChannel } from "./test/blink";

const astrometry = {
  referenceFrame: "ICRS", projection: "TAN", centerRaDegrees: 281.123456, centerDecDegrees: -6.123456, pixelScaleArcsec: 1.42,
  rotationDegrees: 0, rmsPixels: 0.29, rmsArcsec: 0.42, matchedStars: 73, parity: "POSITIVE" as const, catalogIdentity: "b".repeat(64),
  indexIdentities: ["astrometry.net:index:4108:healpix:123:hpnside:4"], correspondenceSha256: "c".repeat(64), catalogManaged: true as const,
  installedSetIdentity: "d".repeat(64), catalogManifestSha256: "e".repeat(64), indexArtifacts: [{ indexId: "4108", relativeName: "index-4108.fits", sizeBytes: 94550400, sha256: "f".repeat(64), manifestSha256: "e".repeat(64), installedSetIdentity: "d".repeat(64) }], wcsSha256: "1".repeat(64),
};
const solvedArtifact = { kind: "SOLVED_MONO_FITS" as const, name: "盾牌座_R_mosaic.fits", path: "/结果/深空 输出/ultra-fast-wbpp-run/盾牌座_R_mosaic.fits", detail: "final WCS", filter: "R", target: "盾牌座", receipt: { artifactId: "final-r", relativePath: "盾牌座_R_mosaic.fits", sha256: "a".repeat(64), sizeBytes: 4096, astrometry } };
const readyChecks = ["final-project-products-present", "final-project-receipts-valid", "final-project-astrometry-validated"].map((code) => ({ code, required: true, passed: true, artifactIds: ["final-r"], message: "ready" }));

// Lifecycle/solver tests finish human review before exercising their own gate.
async function reviewImported() {
  const wasInspect = Boolean(screen.queryByRole("heading", { name: "检查分组与真实筛片证据" }));
  await userEvent.click(screen.getAllByRole("button", { name: /^(闪视筛片（|打开闪视筛片)/ })[0]);
  await screen.findByRole("heading", { name: /闪视筛片/ });
  const count = document.querySelectorAll('[aria-label="通道"] button').length;
  for (let index = 0; index < count; index++) await reviewChannel(/^确认 .* 通道并继续$/);
  await userEvent.click(screen.getByRole("button", { name: "返回" }));
  if (wasInspect) await userEvent.click(screen.getAllByRole("button", { name: /查看筛片结果/ })[0]);
}
async function reachRecipe(review = true) {
  native.lightCount = Math.max(2, native.lightCount);
  await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
  await waitFor(() => expect(native.inspectCalibration).toHaveBeenCalled());
  if (review) await reviewImported();
  await userEvent.click(await screen.findByRole("button", { name: /先看筛片结果（\d+ 张 Light/ }));
  await screen.findByRole("heading", { name: "检查分组与真实筛片证据" });
}
async function reachRun() {
  await reachRecipe();
  await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
  await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled()); await userEvent.click(screen.getByRole("button", { name: /开始处理/ }));
  await screen.findByRole("heading", { name: "正在生成验证后的产品" });
}
// Blink supplies decisions without requiring a second, diagnostic QC pass.
async function reachRunAfterBlink() {
  native.lightCount = Math.max(2, native.lightCount);
  await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
  await waitFor(() => expect(native.inspectCalibration).toHaveBeenCalled());
  await reviewImported();
  await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
  await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled()); await userEvent.click(screen.getByRole("button", { name: /开始处理/ }));
  await screen.findByRole("heading", { name: "正在生成验证后的产品" });
}

beforeEach(() => {
  mockPreviewCanvas();
  window.localStorage.setItem("ultra-fast-wbpp.language", "zh-CN");
  window.localStorage.removeItem("ultra-fast-wbpp.outputParent");
  native.handlers = undefined; native.catalogHandlers = undefined; native.inspectError = undefined; native.qualityDisposition = "PASS"; native.reviewFirst = false; native.cfa = false; native.unknownCfa = false; native.useMasterDark = false; native.catalogReady = true; native.catalogInspectionError = undefined;
  native.capabilityCalls = 0; native.solverSetupCalls = 0; native.dropListenerCalls = 0; native.pipelineListenerCalls = 0; native.catalogListenerCalls = 0; native.lightCount = 1;
  native.solverReady = true; native.solverError = undefined; native.dropHandler = undefined;
  native.platform = "macos"; native.astapReady = false; native.astapScienceReady = false;
  native.inspectPaths.mockReset().mockImplementation(async () => { if (native.inspectError) throw native.inspectError; return inventory(); });
  native.inspectCalibration.mockReset().mockImplementation(async ({ paths }: { paths: string[] }) => {
    const ready = paths.some((path) => path.includes("平场")) && paths.some((path) => path.includes("偏置"));
    return { schemaVersion: 1, status: ready ? "READY" : "BLOCKED", calibrationReady: ready, groups: [], issues: ready ? [] : [{ code: "FLAT_MATCH_MISSING", severity: "ERROR", message: "No compatible flat for R", paths, lightGroups: [] }] };
  });
  native.blinkMeasure.mockReset().mockImplementation(async () => inventoryBlinkManifest((await native.inspectPaths.mock.results.at(-1)!.value).assets));
  native.startRun.mockReset().mockResolvedValue({ jobId: "run-native-1", accepted: true, executionMode: "native", outputDirectory: "/结果/深空 输出/ultra-fast-wbpp-run" });
  native.cancelRun.mockReset().mockResolvedValue(undefined); native.startCatalogInstall.mockReset().mockResolvedValue({ jobId: "catalog-1", accepted: true, catalogId: "astrometry-net-4107-4112" });
  native.cancelCatalogInstall.mockReset().mockResolvedValue(undefined); native.verifyCatalog.mockReset().mockImplementation(async () => { native.catalogReady = true; return { schemaVersion: 1, ok: true }; });
  native.openProviderTerms.mockReset().mockResolvedValue(undefined);
  native.hashSources.mockReset().mockImplementation(async (paths: string[]) => ({ entries: paths.map((path) => ({ path, sourceSha256: `sha256:${(path.includes("亮场") ? "9" : path.includes("平场") ? "a" : path.includes("暗场") ? "b" : "c").repeat(64)}` })) }));
  native.inspectQuality.mockReset().mockImplementation(async (paths: string[]) => {
    const dispositions = paths.map((_, index) => native.reviewFirst && index === 0 ? "REVIEW" as const : native.qualityDisposition);
    return {
      schemaVersion: 1 as const, gatePolicyDigest: `sha256:${"7".repeat(64)}`, workers: 2,
      counts: { PASS: dispositions.filter((value) => value === "PASS").length, REVIEW: dispositions.filter((value) => value === "REVIEW").length, HARD_FAIL: dispositions.filter((value) => value === "HARD_FAIL").length },
      frames: paths.map((path, index) => { const disposition = dispositions[index]; return { path, sourceSha256: `sha256:${index === 0 ? "9".repeat(64) : index.toString(16).padStart(64, "0")}`, disposition, decision: disposition === "PASS" ? "KEEP" : "REVIEW", confidence: "HIGH", starCount: 420, summary: "real gate fixture", previewDataUrl: disposition === "PASS" ? null : "data:image/png;base64,iVBORw0KGgo=", previewSha256: disposition === "PASS" ? null : `sha256:${"6".repeat(64)}`, evidence: disposition === "PASS" ? [] : [{ code: "GATE_FIXTURE", family: "PROVENANCE", severity: disposition, message: "fixture requires review" }] }; }),
    };
  });
  native.revealOutput.mockReset().mockResolvedValue(undefined);
});

describe("native product workflow", () => {
  it("imports pasted multiline paths and submits the entered output folder through the native workflow", async () => {
    native.lightCount = 2;
    render(<App />);
    await userEvent.click(screen.getByText("粘贴文件或文件夹路径"));
    fireEvent.change(screen.getByRole("textbox", { name: "文件或文件夹路径" }), { target: { value: "  /数据/夜晚 1\r\n/数据/校准/主平场.fit\n\n/数据/夜晚 1\n" } });
    await userEvent.click(screen.getByRole("button", { name: "导入路径" }));
    expect(native.inspectPaths).toHaveBeenCalledWith({ paths: ["/数据/夜晚 1", "/数据/校准/主平场.fit"], roleHint: undefined });
    expect(native.startRun).not.toHaveBeenCalled();
    await userEvent.click(await screen.findByRole("button", { name: /先看筛片结果（2 张 Light/ }));
    await reviewImported();
    await userEvent.click(screen.getByText("输入输出文件夹路径"));
    fireEvent.change(screen.getByRole("textbox", { name: "输出文件夹路径" }), { target: { value: " /结果/深空 手动输出 " } });
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "使用此路径" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled());
    await userEvent.click(screen.getByRole("button", { name: /开始处理/ }));
    expect(native.inspectQuality).toHaveBeenCalledTimes(1);
    expect(native.startRun).toHaveBeenCalledWith(expect.objectContaining({ outputParentDirectory: "/结果/深空 手动输出", reviewSelections: [], sources: expect.arrayContaining([expect.objectContaining({ role: "LIGHT", paths: ["/数据/盾牌座/亮场 01.fit"] })]) }));
  });

  it("keeps the retired LocalNormalization option out of the run", async () => {
    render(<App />);
    await reachRecipe();
    await userEvent.click(screen.getByText("高级选项与 solver 设置"));
    expect(screen.queryByRole("checkbox", { name: /LocalNormalization/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    await userEvent.click(screen.getByRole("button", { name: /开始处理/ }));
    expect(native.startRun).toHaveBeenCalled();
    expect(native.startRun.mock.calls[0][0].recipe).not.toHaveProperty("localNormalizationEnabled");
  });

  it("reopens completed screening without rerunning, retains it for calibration additions, and invalidates reimported Lights", async () => {
    native.qualityDisposition = "REVIEW";
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await userEvent.click(await screen.findByRole("button", { name: /先看筛片结果（1 张 Light/ }));
    await screen.findByRole("heading", { name: "检查分组与真实筛片证据" });
    await userEvent.click(screen.getByRole("button", { name: "返回" }));
    expect(screen.getAllByRole("button", { name: /查看筛片结果/ })[0]).toBeEnabled();
    const extra = { ...lightAsset("/数据/校准/added-flat.fit"), role: "MASTER_FLAT" as const };
    native.inspectPaths.mockResolvedValueOnce({ projectName: "same project", totalFiles: 1, assets: [extra], sources: [{ role: extra.role, paths: [extra.path], fileCount: 1, confidence: 1, needsConfirmation: false }] });
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await userEvent.click((await screen.findAllByRole("button", { name: /查看筛片结果/ }))[0]);
    await screen.findByRole("heading", { name: "检查分组与真实筛片证据" });
    expect(native.inspectQuality).toHaveBeenCalledTimes(1);
    await userEvent.click(screen.getByRole("button", { name: "返回" }));
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await screen.findByRole("button", { name: /先看筛片结果（1 张 Light/ });
    expect(screen.queryByRole("button", { name: /查看筛片结果/ })).not.toBeInTheDocument();
  });

  it("shows an available solver even when catalog inspection fails", async () => {
    native.catalogReady = false;
    native.catalogInspectionError = new Error("catalog doctor failed");
    render(<App />);
    await reachRecipe();
    expect(screen.getByText(/solve-field.*可执行/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
  });

  it("shows real calibration blockers even when all frame types are present", async () => {
    native.inspectCalibration.mockResolvedValue({ schemaVersion: 1, status: "BLOCKED", calibrationReady: false, groups: [], issues: [{ code: "FLAT_MATCH_MISSING", severity: "ERROR", message: "Flat filter does not match R", paths: ["/数据/校准/平场 R.fit"], lightGroups: ["R"] }] });
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    expect(await screen.findByText(/已导入 FLAT/)).toBeInTheDocument();
    expect(screen.getByText("Flat filter does not match R")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
  });

  it("invalidates calibration readiness while a new check is pending or fails", async () => {
    render(<App />);
    await reachRecipe();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled();
    let rejectCheck!: (error: Error) => void;
    native.inspectCalibration.mockImplementationOnce(() => new Promise((_, reject) => { rejectCheck = reject; }));
    await userEvent.click(screen.getByRole("button", { name: "重新检查校准" }));
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
    await waitFor(() => expect(rejectCheck).toBeTypeOf("function"));
    await act(async () => rejectCheck(new Error("metadata read failed")));
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
  });

  it("keeps mixed Raw and Master tabs separate from import role hints", async () => {
    const mixed = inventory();
    mixed.assets.push({ ...lightAsset("/数据/校准/master-flat.fit"), role: "MASTER_FLAT", observedAt: "2026-09-04T21:00:00Z" });
    mixed.sources.push({ role: "MASTER_FLAT", paths: ["/数据/校准/master-flat.fit"], fileCount: 1, confidence: 1, needsConfirmation: false });
    native.inspectPaths.mockResolvedValueOnce(mixed);
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await userEvent.click(screen.getByRole("tab", { name: "Flat 2" }));
    const table = screen.getByRole("tabpanel");
    expect(within(table).getByText("Master")).toBeInTheDocument();
    expect(within(table).getByText("原始帧")).toBeInTheDocument();
    expect(within(table).getByText("2026-09-04")).toBeInTheDocument();
    expect(screen.queryByText(/下一批会按 FLAT/)).not.toBeInTheDocument();
  });

  it("returns to automatic detection after importing a single frame type", async () => {
    const first = inventory();
    first.assets = first.assets.filter((asset) => asset.role === "LIGHT");
    first.sources = first.sources.filter((source) => source.role === "LIGHT");
    first.totalFiles = 1;
    native.inspectPaths.mockResolvedValueOnce(first);
    render(<App />);
    await userEvent.click(screen.getByText("手动指定下一批类型 / 素材清单"));
    await userEvent.click(screen.getByRole("button", { name: /下一批会按 Light/ }));
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await screen.findByText("自动从元数据读取帧类型、目标、滤镜、曝光与相机。");
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    expect(native.inspectPaths).toHaveBeenNthCalledWith(1, { paths: ["/数据/盾牌座 会话"], roleHint: "LIGHT" });
    expect(native.inspectPaths).toHaveBeenNthCalledWith(2, { paths: ["/数据/盾牌座 会话"], roleHint: undefined });
  });

  it("keeps imported files unchanged during quality inspection and processing", async () => {
    native.lightCount = 2;
    const quality = await native.inspectQuality(["/数据/盾牌座/亮场 01.fit", "/数据/盾牌座/亮场 02.fit"]);
    let finishInspection!: (value: typeof quality) => void;
    native.inspectQuality.mockImplementationOnce(() => new Promise((resolve) => { finishInspection = resolve; }));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await userEvent.click(await screen.findByRole("button", { name: /先看筛片结果（2 张 Light/ }));
    expect(screen.getByRole("button", { name: /选择文件夹$/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: "清空" })).toBeDisabled();
    await act(async () => native.dropHandler?.(["/data/more-flats"]));
    expect(native.inspectPaths).toHaveBeenCalledTimes(1);
    await act(async () => finishInspection(quality));
    await reviewImported();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled()); await userEvent.click(screen.getByRole("button", { name: /开始处理/ }));
    await screen.findByRole("heading", { name: "正在生成验证后的产品" });
    await act(async () => native.dropHandler?.(["/data/other-project"]));
    expect(native.inspectPaths).toHaveBeenCalledTimes(1);
  });

  it("rechecks an installed solver without losing the imported project", async () => {
    native.solverReady = false;
    render(<App />);
    await reachRecipe();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
    native.solverReady = true;
    await userEvent.click(screen.getByRole("button", { name: "重新检测配置" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled());
    expect(native.inspectPaths).toHaveBeenCalledTimes(1);
    expect(native.solverSetupCalls).toBe(6);
  });

  it("shows a failed setup recheck and invalidates stale solver readiness", async () => {
    render(<App />);
    await reachRecipe();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled();
    native.solverError = new Error("Solver configuration could not be read");
    await userEvent.click(screen.getByRole("button", { name: "重新检测配置" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Solver configuration could not be read");
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
  });

  it("explains missing calibration when only Lights were imported", async () => {
    const onlyLights = inventory();
    onlyLights.assets = onlyLights.assets.filter((asset) => asset.role === "LIGHT");
    onlyLights.sources = onlyLights.sources.filter((source) => source.role === "LIGHT");
    onlyLights.totalFiles = 1;
    native.inspectPaths.mockResolvedValueOnce(onlyLights);
    render(<App />);
    await reachRecipe();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    expect(screen.getByText(/请完成校准检查，处理缺失/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
  });

  it("does not repeat expensive runtime probes when the selected import role changes", async () => {
    render(<App />);
    await waitFor(() => expect(native.solverSetupCalls).toBe(3));
    expect(native.capabilityCalls).toBe(1);
    expect(native.dropListenerCalls).toBe(1);
    await userEvent.click(screen.getByText("手动指定下一批类型 / 素材清单"));
    await userEvent.click(screen.getByRole("button", { name: /下一批会按 Light/ }));
    await waitFor(() => expect(native.dropListenerCalls).toBe(2));
    expect(native.capabilityCalls).toBe(1);
    expect(native.solverSetupCalls).toBe(3);
  });

  it("keeps pipeline and catalog listeners stable across language changes", async () => {
    render(<App />);
    await waitFor(() => expect(native.pipelineListenerCalls).toBe(1));
    await waitFor(() => expect(native.catalogListenerCalls).toBe(1));
    await reachRun();
    const pipelineHandlers = native.handlers;
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "语言" }), "en");
    expect(native.pipelineListenerCalls).toBe(1);
    expect(native.catalogListenerCalls).toBe(1);
    expect(native.handlers).toBe(pipelineHandlers);
    await act(async () => pipelineHandlers?.onComplete({
      jobId: "run-native-1",
      outputDirectory: "/结果/深空 输出/ultra-fast-wbpp-run",
      artifacts: [solvedArtifact],
      gate: { decision: "ready", checks: readyChecks },
    }));
    expect(await screen.findByText(/FINAL GATE · WCS SOLVED/)).toBeInTheDocument();
  });

  it("starts only once and locks workflow navigation while a project is running", async () => {
    render(<App />);
    await reachRecipe();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled()); await userEvent.dblClick(screen.getByRole("button", { name: /开始处理/ }));
    expect(await screen.findByRole("heading", { name: "正在生成验证后的产品" })).toBeInTheDocument();
    expect(native.startRun).toHaveBeenCalledTimes(1);
    expect(within(screen.getByRole("navigation", { name: "处理流程" })).getByRole("button", { name: /处理/ })).toBeDisabled();
  });

  it("renders issues first and batches passing-frame evidence", async () => {
    native.lightCount = 55;
    native.reviewFirst = true;
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await userEvent.click(await screen.findByRole("button", { name: /先看筛片结果（55 张 Light/ }));
    const initialFrames = document.querySelectorAll(".quality-frame");
    expect(initialFrames).toHaveLength(51);
    expect(initialFrames[0]).toHaveClass("quality-review");
    const more = screen.getByRole("button", { name: "再显示 4 张通过帧" });
    await userEvent.click(more);
    expect(document.querySelectorAll(".quality-frame")).toHaveLength(55);
  });

  it("starts after Blink and shows the run's own screening with the result", async () => {
    render(<App />); await reachRunAfterBlink();
    expect(native.inspectQuality).not.toHaveBeenCalled();
    expect(native.startRun).toHaveBeenCalledWith(expect.objectContaining({ reviewSelections: [], runLabel: "盾牌座 Panel 1" }));
    const preview = "data:image/png;base64,iVBORw0KGgo=";
    await act(async () => native.handlers?.onComplete({
      jobId: "run-native-1", outputDirectory: "/结果/深空 输出/盾牌座-Panel-1_2026-09-19_0010", artifacts: [solvedArtifact], gate: { decision: "ready", checks: readyChecks },
      screening: { admitted: 61, excluded: 2, counts: { PASS: 61, REVIEW: 1, HARD_FAIL: 1 }, frames: [
        { name: "亮场 07.fit", target: "盾牌座 Panel 1", disposition: "HARD_FAIL", admitted: false, summary: "clouds", evidence: ["星点数量骤降", "背景升高"], starCount: 12, previewDataUrl: preview },
        { name: "亮场 09.fit", disposition: "REVIEW", admitted: false, summary: "trail", evidence: [] },
      ] },
    }));
    expect(await screen.findByText(/最终门禁 · WCS 已解算/)).toBeInTheDocument();
    const section = screen.getByRole("region", { name: "筛片结果" });
    expect(within(section).getByText("61 张进入叠加 · 2 张排除")).toBeInTheDocument();
    expect(within(section).getByText("亮场 07.fit")).toBeInTheDocument();
    expect(within(section).getByText("星点数量骤降; 背景升高")).toBeInTheDocument();
    expect(within(section).getByRole("img", { name: "亮场 07.fit 的复核预览" })).toHaveAttribute("src", preview);
    expect(within(section).getByText("trail")).toBeInTheDocument();
    expect(within(section).getByText(/如需调整入栈选择/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "显示输出：盾牌座-Panel-1_2026-09-19_0010" })).toBeEnabled();
  });

  it("remembers the output folder for the next session and blocks a one-Light panel before any screening", async () => {
    const first = render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await waitFor(() => expect(native.inspectCalibration).toHaveBeenCalled());
    expect(screen.getByText("选择输出文件夹。")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    expect(screen.queryByText("选择输出文件夹。")).not.toBeInTheDocument();
    expect(screen.getByText("盾牌座 Panel 1 × R：当前可用 1 张 Light，至少需要 2 张。")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
    expect(window.localStorage.getItem("ultra-fast-wbpp.outputParent")).toBe("/结果/深空 输出");
    first.unmount();
    render(<App />);
    expect(screen.getByText("/结果/深空 输出")).toBeInTheDocument();
  });

  it("preserves Unicode paths and shows WCS SOLVED only after the final gate", async () => {
    render(<App />); await reachRun();
    expect(native.startRun).toHaveBeenCalledWith(expect.objectContaining({ outputParentDirectory: "/结果/深空 输出", projectName: "盾牌座 马赛克", sources: expect.arrayContaining([expect.objectContaining({ role: "LIGHT", paths: ["/数据/盾牌座/亮场 01.fit"] })]) }));
    expect(screen.queryByText(/最终门禁 · WCS 已解算/)).not.toBeInTheDocument();
    await act(async () => native.handlers?.onComplete({ jobId: "run-native-1", outputDirectory: "/结果/深空 输出/ultra-fast-wbpp-run", artifacts: [solvedArtifact, { kind: "LINEAR_RGB_FITS", name: "rgb.fits", path: "/结果/深空 输出/ultra-fast-wbpp-run/rgb.fits", detail: "linear RGB" }, { kind: "RGB_PREVIEW_TIFF_16", name: "rgb.tiff", path: "/结果/深空 输出/ultra-fast-wbpp-run/rgb.tiff", detail: "16-bit" }, { kind: "RGB_PREVIEW_PNG_16", name: "rgb.png", path: "/结果/深空 输出/ultra-fast-wbpp-run/rgb.png", detail: "16-bit" }, { kind: "RECEIPT", name: "receipt.json", path: "/结果/深空 输出/ultra-fast-wbpp-run/receipt.json", detail: "auditable" }], gate: { decision: "ready", checks: readyChecks } }));
    expect(await screen.findByText(/最终门禁 · WCS 已解算/)).toBeInTheDocument();
    expect(screen.getByText("线性 RGB FITS")).toBeInTheDocument(); expect(screen.getByText("16-bit TIFF")).toBeInTheDocument(); expect(screen.getByText("处理收据")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "在 Finder 中显示 盾牌座_R_mosaic.fits" })); expect(native.revealOutput).toHaveBeenCalledWith(solvedArtifact.path);
  });

  it("keeps overall progress across panels and final operations until verified completion", async () => {
    render(<App />); await reachRun();
    const event = { jobId: "run-native-1", stageId: "publish", state: "succeeded" as const, fraction: 1,
      overallFraction: 0.22, scope: "panel" as const, panelId: "cartwheel__b", panelTarget: "Cartwheel", panelFilter: "B", panelIndex: 1, panelCount: 4, message: "panel published" };
    await act(async () => native.handlers?.onProgress(event));
    expect(screen.getByLabelText("22%")).toBeInTheDocument();
    expect(screen.getByText(/当前分组 1\/4 · Cartwheel \/ B/)).toBeInTheDocument();
    await act(async () => native.handlers?.onProgress({ ...event, panelId: "cartwheel__g", panelIndex: 2, panelFilter: "G", stageId: "quality-control", state: "running", fraction: 0, overallFraction: 0.22 }));
    expect(screen.getByLabelText("22%")).toBeInTheDocument();
    expect(screen.getByText(/当前分组 2\/4 · Cartwheel \/ G/)).toBeInTheDocument();
    expect(screen.getByText("逐通道天文解算").closest("li")).toHaveClass("waiting");
    await act(async () => native.handlers?.onProgress({ ...event, scope: "project", stageId: "alignment", state: "running", fraction: 0.5, overallFraction: 0.86, message: "aligning channels" }));
    expect(screen.getByLabelText("86%")).toBeInTheDocument();
    expect(screen.getByText("通道对齐").closest("li")).toHaveClass("running");
    await act(async () => native.handlers?.onProgress({ ...event, scope: "project", stageId: "publish", overallFraction: 1 }));
    expect(screen.getByLabelText("99%")).toBeInTheDocument();
    expect(screen.queryByText(/最终门禁 · WCS 已解算/)).not.toBeInTheDocument();
    await act(async () => native.handlers?.onProgress({ ...event, overallFraction: 0.1 }));
    expect(screen.getByLabelText("99%")).toBeInTheDocument();
    await act(async () => native.handlers?.onComplete({ jobId: event.jobId, outputDirectory: "/结果/final", artifacts: [solvedArtifact], gate: { decision: "ready", checks: readyChecks } }));
    expect(await screen.findByText(/最终门禁 · WCS 已解算/)).toBeInTheDocument();
  });

  it("cancels the native process and never fabricates a result", async () => {
    render(<App />); await reachRun(); await userEvent.click(screen.getByRole("button", { name: "安全取消" }));
    expect(native.cancelRun).toHaveBeenCalledWith("run-native-1"); expect(await screen.findByRole("heading", { name: "运行已安全取消" })).toBeInTheDocument(); expect(screen.queryByText(/最终门禁 · WCS 已解算/)).not.toBeInTheDocument();
  });

  it("admits Bayer/OSC Lights and explains the colour channel processing", async () => {
    native.cfa = true; render(<App />); await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ })); await userEvent.click(await screen.findByRole("button", { name: /先看筛片结果（1 张 Light/ }));
    expect(screen.getByText("彩色相机（Bayer）亮场")).toBeInTheDocument(); expect(screen.getByText(/RGGB Bayer 图案/)).toBeInTheDocument();
    expect(screen.queryByText(/无法识别的 CFA 图案/)).not.toBeInTheDocument();
  });

  it("marks the noninteractive toolbar subtree as draggable without capturing language controls", () => {
    render(<App />);
    expect(document.querySelector("header.topbar")).toHaveAttribute("data-tauri-drag-region", "deep");
    expect(document.querySelector(".language-picker")).toHaveAttribute("data-tauri-drag-region", "false");
  });

  it("uses the mono workflow without per-file confirmations when BAYERPAT is missing", async () => {
    native.unknownCfa = true; render(<App />); await reachRun();
    expect(native.hashSources).not.toHaveBeenCalled();
    expect(native.inspectCalibration).toHaveBeenCalledWith(expect.objectContaining({ recipe: { calibration: { workflow: "mono-standard-v1", bias: "OPTIONAL", masterMetadataOverrides: [] }, rawFrameMetadataOverrides: [] } }));
    expect(native.startRun).toHaveBeenCalledWith(expect.objectContaining({ recipe: expect.objectContaining({ calibrationWorkflow: "mono-standard-v1" }), rawFrameMetadataOverrides: [] }));
    expect(screen.queryByText("确认这一组是单色")).not.toBeInTheDocument();
  });

  it("reuses imported masters without metadata confirmations or invented values", async () => {
    native.lightCount = 2;
    native.useMasterDark = true; render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    const advanced = (await screen.findByText("校准高级设置")).closest("details")!;
    expect(advanced).not.toHaveAttribute("open");
    expect(screen.queryByText(/SHA-256/)).not.toBeInTheDocument();
    await userEvent.click(await screen.findByText("校准高级设置"));
    const form = screen.getByText("MASTER_DARK").closest("article")!;
    expect(within(form).getByLabelText("相机")).toHaveValue("");
    expect(within(form).getByLabelText("增益")).toHaveValue(null);
    expect(within(form).getByLabelText("偏置值")).toHaveValue(null);
    expect(within(form).getByLabelText("温度 °C")).toHaveValue(null);
    expect(within(form).getByLabelText("读出模式")).toHaveValue("");
    await reviewImported();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" })); await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled()); await userEvent.click(screen.getByRole("button", { name: /开始处理/ }));
    expect(native.inspectQuality).not.toHaveBeenCalled();
    expect(native.startRun).toHaveBeenCalledWith(expect.objectContaining({ masterMetadataOverrides: [] }));
  });

  it("saves independent advanced dark semantics while blank metadata stays omitted and zero stays zero", async () => {
    native.lightCount = 2;
    native.useMasterDark = true; render(<App />); await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await userEvent.click(await screen.findByText("校准高级设置"));
    const form = screen.getByText("MASTER_DARK").closest("article")!;
    await userEvent.click(within(form).getByLabelText("否，需要另减 Bias"));
    await userEvent.type(within(form).getByLabelText("偏置值"), "30");
    await userEvent.clear(within(form).getByLabelText("偏置值"));
    await userEvent.type(within(form).getByLabelText("增益"), "0");
    expect(within(form).getByLabelText("偏置值")).toHaveValue(null);
    await userEvent.click(within(form).getByRole("button", { name: "保存修改" }));
    await waitFor(() => expect(native.inspectCalibration).toHaveBeenLastCalledWith(expect.objectContaining({ recipe: { calibration: { workflow: "mono-standard-v1", bias: "OPTIONAL", masterMetadataOverrides: [{ sourceSha256: `sha256:${"8".repeat(64)}`, binning: [1, 1], gain: 0, biasIncluded: false }] }, rawFrameMetadataOverrides: [] } })));
    await reviewImported();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" })); await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled()); await userEvent.click(screen.getByRole("button", { name: /开始处理/ }));
    expect(native.startRun).toHaveBeenCalledWith(expect.objectContaining({ masterMetadataOverrides: [{ sourceSha256: `sha256:${"8".repeat(64)}`, binning: [1, 1], gain: 0, biasIncluded: false }] }));
  });

  it("requires saving or discarding actual advanced edits before starting", async () => {
    native.lightCount = 2; native.useMasterDark = true; render(<App />); await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await waitFor(() => expect(native.inspectCalibration).toHaveBeenCalled());
    await reviewImported();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled());
    await userEvent.click(await screen.findByText("校准高级设置"));
    const form = screen.getByText("MASTER_DARK").closest("article")!;
    await userEvent.type(within(form).getByLabelText("偏置值"), "30");
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
    expect(screen.getByText("保存高级校准修改，或恢复文件设置。")).toBeInTheDocument();
    await userEvent.click(within(form).getByRole("button", { name: "恢复文件设置" }));
    expect(within(form).getByLabelText("偏置值")).toHaveValue(null);
    await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled());
  });

  it("surfaces inventory errors without claiming success", async () => {
    native.inspectError = new Error("HEADER_CONFLICT: role mismatch"); render(<App />); await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    expect(await screen.findByText(/HEADER_CONFLICT/)).toBeInTheDocument(); expect(screen.queryByText(/WCS SOLVED/)).not.toBeInTheDocument();
  });

  it("keeps diagnostic REVIEW evidence read-only and requires Blink before launch", async () => {
    native.reviewFirst = true; render(<App />); await reachRecipe(false);
    expect(screen.getByText("fixture requires review")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "人工复核后批准纳入" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
    await reviewImported();
    await userEvent.click(screen.getByRole("button", { name: /开始处理/ }));
    expect(native.startRun).toHaveBeenCalledWith(expect.objectContaining({ reviewSelections: [], selection: expect.objectContaining({ policy: "explicit-v1" }), blinkReview: expect.any(Object) }));
  });

  it("cannot start an imported project before manual channel review", async () => {
    native.lightCount = 2; render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
    expect(screen.getByText("处理前逐通道人工审片")).toBeInTheDocument();
    expect(native.inspectQuality).not.toHaveBeenCalled();
    expect(native.startRun).not.toHaveBeenCalled();
  });

  it("blocks a one-Light panel before starting and names the missing admission count", async () => {
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await userEvent.click(await screen.findByRole("button", { name: /先看筛片结果（1 张 Light/ }));
    expect(screen.getByRole("cell", { name: "1 帧" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    expect(screen.getByText("盾牌座 Panel 1 × R：当前可用 1 张 Light，至少需要 2 张。")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
    expect(native.startRun).not.toHaveBeenCalled();
  });

  it("requires two manually kept Lights in every target-filter panel", async () => {
    native.lightCount = 4;
    const mixed = inventory();
    mixed.assets.filter((asset) => asset.role === "LIGHT").forEach((asset, index) => { asset.filter = index < 2 ? "B" : "L"; });
    native.inspectPaths.mockResolvedValueOnce(mixed); render(<App />);
    await userEvent.click(screen.getByRole("button", { name: /选择文件夹$/ }));
    await userEvent.click(screen.getByRole("button", { name: /闪视筛片（4 张 Light/ }));
    await reviewChannel(/^确认 B/); await reviewChannel(/^确认 L/);
    fireEvent.click(screen.getByRole("button", { name: /^B ·/ }));
    fireEvent.click(screen.getByRole("button", { name: /丢弃并下一张/ }));
    fireEvent.click(screen.getByRole("button", { name: /^确认 B/ }));
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    expect(screen.getByText("盾牌座 Panel 1 × B：保留 1/2 张 Light，至少需要保留 2 张。")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
    expect(native.startRun).not.toHaveBeenCalled();
  });

});

describe("Windows x64 solver route", () => {
  it("starts with a science-ready ASTAP when solve-field is absent and names the validated CPU path", async () => {
    native.platform = "windows"; native.solverReady = false; native.astapReady = true; native.astapScienceReady = true;
    render(<App />);
    await reachRecipe();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled());
    expect(screen.queryByText("安装解算器（solve-field 或 ASTAP）和已校验离线星表。")).not.toBeInTheDocument();
    const rows = document.querySelectorAll(".solver-row");
    expect(rows[0]).toHaveTextContent(/ASTAP 可执行/);
    expect(rows[0]).toHaveTextContent("C:\\Program Files\\astap\\astap_cli.exe");
    expect(rows[1]).toHaveTextContent(/solve-field 未就绪/);
    expect(rows[1]).toHaveTextContent("Windows 上不需要");
    expect(screen.getByLabelText("计算后端")).toHaveTextContent("AMD Ryzen 7 5800H with Radeon Graphics");
    expect(screen.getByLabelText("计算后端")).toHaveTextContent("Native CPU execution · GPU acceleration not used · Windows x64 · 已验证的 CPU 路径");
    expect(screen.getByText("离线星表已校验并配置")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /开始处理/ }));
    expect(native.startRun).toHaveBeenCalledTimes(1);
  });

  it("keeps the solver blocker while ASTAP runs but is not science-ready", async () => {
    native.platform = "windows"; native.solverReady = false; native.astapReady = true; native.astapScienceReady = false;
    render(<App />);
    await reachRecipe();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
    expect(screen.getByText("安装解算器（solve-field 或 ASTAP）和已校验离线星表。")).toBeInTheDocument();
    const astap = document.querySelectorAll(".solver-row")[0];
    expect(astap).toHaveTextContent(/ASTAP 未就绪/);
    expect(astap).toHaveTextContent("请从 hnsky.org 安装 ASTAP（astap_cli.exe）和星表数据库（D20 或更大）");
    expect(astap).toHaveTextContent("STAR_DATABASE_MISSING");
    expect(native.startRun).not.toHaveBeenCalled();
    // The engine reports the star database installed: the same setup recheck unblocks the start.
    native.astapScienceReady = true;
    await userEvent.click(screen.getByRole("button", { name: "重新检测配置" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled());
  });

  it("never lets ASTAP satisfy the gate on an engine that does not report scienceReady", async () => {
    native.platform = "windows"; native.solverReady = false; native.astapReady = true;
    const doctor = await import("./bridge").then((module) => module.desktopBridge.solverDoctor as unknown as ReturnType<typeof vi.fn>);
    const original = doctor.getMockImplementation()!;
    doctor.mockImplementation(async () => {
      const report = await original();
      report.backends = report.backends.map((backend: { backendId: string; scienceReady?: boolean }) => { const { scienceReady: _ignored, ...rest } = backend; return rest; });
      return report;
    });
    render(<App />);
    await reachRecipe();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    expect(screen.getByRole("button", { name: /开始处理/ })).toBeDisabled();
    expect(document.querySelectorAll(".solver-row")[0]).toHaveTextContent(/ASTAP 未就绪/);
    doctor.mockImplementation(original);
  });

  it("keeps macOS on solve-field first and treats a missing scienceReady as ready for it only", async () => {
    native.astapReady = true; native.astapScienceReady = false;
    render(<App />);
    await reachRecipe();
    const rows = document.querySelectorAll(".solver-row");
    expect(rows[0]).toHaveTextContent(/solve-field 可执行/);
    expect(rows[1]).toHaveTextContent(/ASTAP 未就绪/);
    expect(rows[1]).toHaveTextContent("可选替代：ASTAP");
    expect(screen.getByLabelText("计算后端")).toHaveTextContent("M3 PRO 深度优化");
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled());
  });
});

describe("result previews", () => {
  it("prefers the controller's preview data URLs over asset-protocol paths", async () => {
    render(<App />); await reachRun();
    const monoPreview = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNgAAIAAAUAAen63NgAAAAASUVORK5CYII=";
    const rgbPreview = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGP4DwQACfsD/Wm0a9YAAAAASUVORK5CYII=";
    await act(async () => native.handlers?.onComplete({
      jobId: "run-native-1", outputDirectory: "D:\\astro\\out\\NGC7331_2026-09-21_2210",
      artifacts: [
        solvedArtifact,
        { kind: "LINEAR_RGB_FITS", name: "rgb.fits", path: "D:\\astro\\out\\NGC7331_2026-09-21_2210\\rgb.fits", detail: "linear RGB" },
        { kind: "MONO_PREVIEW_PNG", name: "盾牌座_R_mosaic.png", path: "D:\\astro\\out\\NGC7331_2026-09-21_2210\\previews\\盾牌座_R_mosaic.png", detail: "auto-stretched mono preview", filter: "R", target: "盾牌座", previewDataUrl: monoPreview },
        { kind: "RGB_PREVIEW_PNG_16", name: "rgb.png", path: "D:\\astro\\out\\NGC7331_2026-09-21_2210\\rgb.png", detail: "16-bit display preview", previewDataUrl: rgbPreview },
      ],
      gate: { decision: "ready", checks: readyChecks },
    }));
    expect(await screen.findByText(/最终门禁 · WCS 已解算/)).toBeInTheDocument();
    expect(document.querySelector(".result-hero-image")).toHaveAttribute("src", monoPreview);
    const cards = [...document.querySelectorAll(".card")];
    expect(cards).toHaveLength(2);
    expect(cards[0].querySelector(".shot img")).toHaveAttribute("src", monoPreview);
    expect(cards[1].querySelector(".shot img")).toHaveAttribute("src", rgbPreview);
    expect(screen.getByRole("button", { name: "显示输出：NGC7331_2026-09-21_2210" })).toBeEnabled();
  });

  it("falls back to no preview when the controller carried none and the asset protocol is unavailable", async () => {
    render(<App />); await reachRun();
    await act(async () => native.handlers?.onComplete({
      jobId: "run-native-1", outputDirectory: "/结果/final",
      artifacts: [solvedArtifact, { kind: "MONO_PREVIEW_PNG", name: "盾牌座_R_mosaic.png", path: "/结果/final/previews/盾牌座_R_mosaic.png", detail: "auto-stretched mono preview", filter: "R", target: "盾牌座" }],
      gate: { decision: "ready", checks: readyChecks },
    }));
    expect(await screen.findByText(/最终门禁 · WCS 已解算/)).toBeInTheDocument();
    expect(document.querySelector(".result-hero-image")).toBeNull();
    expect(document.querySelector(".card .shot img")).toBeNull();
  });
});

describe("native run elapsed time", () => {
  const completion = () => ({ jobId: "run-native-1", outputDirectory: "/结果/final", artifacts: [solvedArtifact], gate: { decision: "ready" as const, checks: readyChecks } });
  const advance = async (milliseconds: number) => act(async () => { vi.advanceTimersByTime(milliseconds); });
  const elapsed = () => screen.getByRole("timer");
  const prepare = async () => {
    render(<App />);
    await reachRecipe();
    await userEvent.click(screen.getByRole("button", { name: "选择输出文件夹" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled());
  };
  const start = async () => {
    await waitFor(() => expect(screen.getByRole("button", { name: /开始处理/ })).toBeEnabled());
    await userEvent.click(screen.getByRole("button", { name: /开始处理/ }));
  };

  beforeEach(() => vi.useFakeTimers({ toFake: ["setInterval", "clearInterval", "performance", "Date"] }));
  afterEach(() => vi.useRealTimers());

  it("counts only this run across channels and clock changes, freezes success, and resets for a new project", async () => {
    await prepare();
    await advance(120_000);
    await start();
    expect(elapsed()).toHaveTextContent("00:00:00");
    await advance(3_661_000);
    expect(elapsed()).toHaveTextContent("01:01:01");
    await act(async () => native.handlers?.onProgress({ jobId: "run-native-1", stageId: "quality-control", state: "running", fraction: 0, overallFraction: 0.22, scope: "panel", panelId: "next-g", panelFilter: "G", panelIndex: 2, panelCount: 4, message: "next channel" }));
    expect(elapsed()).toHaveTextContent("01:01:01");
    vi.setSystemTime(Date.now() - 24 * 60 * 60 * 1000);
    await advance(2_000);
    expect(elapsed()).toHaveTextContent("01:01:03");
    await act(async () => native.handlers?.onComplete(completion()));
    expect(screen.getByRole("timer", { name: "总用时" })).toHaveTextContent("01:01:03");
    await advance(60_000);
    await act(async () => native.handlers?.onError({ jobId: "run-native-1", code: "LATE_ERROR", message: "late", retryable: false }));
    expect(elapsed()).toHaveTextContent("01:01:03");
    expect(screen.getByText(/最终门禁 · WCS 已解算/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "开始新项目" }));
    expect(screen.queryByRole("timer")).not.toBeInTheDocument();
    await reachRecipe();
    await start();
    expect(elapsed()).toHaveTextContent("00:00:00");
    await advance(1_000);
    expect(elapsed()).toHaveTextContent("00:00:01");
  });

  it("includes launch waiting and preserves completion time when the receipt arrives later", async () => {
    let accept!: (value: { jobId: string; accepted: boolean; executionMode: string; outputDirectory: string }) => void;
    native.startRun.mockImplementationOnce(() => new Promise((resolve) => { accept = resolve; }));
    await prepare();
    await start();
    expect(screen.getByRole("button", { name: "安全取消" })).toBeDisabled();
    await advance(3_200);
    expect(elapsed()).toHaveTextContent("00:00:03");
    await act(async () => native.handlers?.onComplete(completion()));
    await advance(5_000);
    await act(async () => accept({ jobId: "run-native-1", accepted: true, executionMode: "native", outputDirectory: "/结果/final" }));
    expect(screen.getByRole("timer", { name: "总用时" })).toHaveTextContent("00:00:03");
    await advance(10_000);
    expect(elapsed()).toHaveTextContent("00:00:03");
  });

  it("freezes native failure and starts a retry from zero", async () => {
    await prepare();
    await start();
    await advance(4_700);
    await act(async () => native.handlers?.onError({ jobId: "run-native-1", code: "PROCESS_FAILED", message: "failed", retryable: true }));
    expect(screen.getByRole("timer", { name: "总用时" })).toHaveTextContent("00:00:04");
    await advance(10_000);
    expect(elapsed()).toHaveTextContent("00:00:04");
    await userEvent.click(screen.getByRole("button", { name: "返回导入" }));
    native.startRun.mockResolvedValueOnce({ jobId: "run-native-2", accepted: true, executionMode: "native", outputDirectory: "/结果/retry" });
    await start();
    expect(elapsed()).toHaveTextContent("00:00:00");
    await act(async () => native.handlers?.onComplete(completion()));
    await advance(1_000);
    expect(elapsed()).toHaveTextContent("00:00:01");
    expect(screen.getByRole("heading", { name: "正在生成验证后的产品" })).toBeInTheDocument();
  });

  it.each(["start error", "start rejected", "final gate"])("freezes elapsed time after %s", async (failure) => {
    let finishLaunch!: () => void;
    if (failure !== "final gate") native.startRun.mockImplementationOnce(() => new Promise((resolve, reject) => {
      finishLaunch = () => failure === "start error" ? reject(new Error("launch failed")) : resolve({ jobId: "run-native-1", accepted: false, executionMode: "native", outputDirectory: "/结果/final" });
    }));
    await prepare();
    await start();
    await advance(2_800);
    await act(async () => {
      if (failure === "final gate") native.handlers?.onComplete({ ...completion(), gate: { decision: "blocked", checks: [] } });
      else finishLaunch();
    });
    expect(screen.getByRole("heading", { name: "运行已失败关闭" })).toBeInTheDocument();
    expect(screen.getByRole("timer", { name: "总用时" })).toHaveTextContent("00:00:02");
    await advance(10_000);
    expect(elapsed()).toHaveTextContent("00:00:02");
  });

  it.each([false, true])("preserves the confirmed outcome and time across cancellation races (cancel rejected: %s)", async (cancelFails) => {
    let finishCancel!: () => void;
    native.cancelRun.mockImplementationOnce(() => new Promise((resolve, reject) => {
      finishCancel = () => cancelFails ? reject(new Error("cancel failed")) : resolve(undefined);
    }));
    await prepare();
    await start();
    await advance(4_000);
    await userEvent.click(screen.getByRole("button", { name: "安全取消" }));
    await act(async () => native.handlers?.onComplete(completion()));
    await act(async () => native.handlers?.onError({ jobId: "run-native-1", code: "LATE_ERROR", message: "late", retryable: false }));
    await advance(2_500);
    expect(elapsed()).toHaveTextContent("00:00:06");
    await act(async () => finishCancel());
    const total = cancelFails ? "00:00:04" : "00:00:06";
    expect(screen.getByRole("heading", { name: cancelFails ? "每个最终通道都已写入天文坐标" : "运行已安全取消" })).toBeInTheDocument();
    expect(screen.getByRole("timer", { name: "总用时" })).toHaveTextContent(total);
    await advance(10_000);
    await act(async () => native.handlers?.onComplete(completion()));
    expect(elapsed()).toHaveTextContent(total);
    if (cancelFails) expect(screen.getByText(/最终门禁 · WCS 已解算/)).toBeInTheDocument();
    else expect(screen.queryByText(/最终门禁 · WCS 已解算/)).not.toBeInTheDocument();
  });

  it("keeps timing and navigation locked if cancellation rejects while the worker is still running", async () => {
    native.cancelRun.mockRejectedValueOnce(new Error("cancel transport failed"));
    await prepare();
    await start();
    await advance(4_000);
    await userEvent.click(screen.getByRole("button", { name: "安全取消" }));
    expect(screen.getByRole("heading", { name: "正在生成验证后的产品" })).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("cancel transport failed");
    expect(within(screen.getByRole("navigation", { name: "处理流程" })).getByRole("button", { name: /处理/ })).toBeDisabled();
    await advance(2_000);
    expect(screen.getByRole("timer", { name: "已用时间" })).toHaveTextContent("00:00:06");
    expect(screen.getByRole("button", { name: "安全取消" })).toBeEnabled();
    await act(async () => native.handlers?.onComplete(completion()));
    await advance(10_000);
    expect(screen.getByRole("timer", { name: "总用时" })).toHaveTextContent("00:00:06");
  });

  it("uses the first native failure time when a rejected cancellation releases queued terminal events", async () => {
    let rejectCancel!: () => void;
    native.cancelRun.mockImplementationOnce(() => new Promise((_, reject) => {
      rejectCancel = () => reject(new Error("cancel transport failed"));
    }));
    await prepare();
    await start();
    await advance(4_000);
    await userEvent.click(screen.getByRole("button", { name: "安全取消" }));
    await act(async () => native.handlers?.onError({ jobId: "run-native-1", code: "PROCESS_FAILED", message: "worker failed", retryable: false }));
    await act(async () => native.handlers?.onComplete(completion()));
    await advance(2_000);
    await act(async () => rejectCancel());
    expect(screen.getByRole("heading", { name: "运行已失败关闭" })).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("PROCESS_FAILED: worker failed");
    await advance(10_000);
    expect(screen.getByRole("timer", { name: "总用时" })).toHaveTextContent("00:00:04");
  });
});

describe("native run failure", () => {
  it("stops the spinner, marks the running stage failed and shows the failure card with the evidence path", async () => {
    render(<App />); await reachRun();
    const event = { jobId: "run-native-1", state: "running" as const, fraction: 0.5, overallFraction: 0.9, scope: "project" as const, message: "publishing" };
    await act(async () => native.handlers?.onProgress({ ...event, stageId: "alignment", state: "succeeded", fraction: 1 }));
    await act(async () => native.handlers?.onProgress({ ...event, stageId: "publish" }));
    expect(screen.getByText("发布与桌面校验").closest("li")).toHaveClass("running");
    expect(document.querySelector(".stage-list .spin")).not.toBeNull();
    await act(async () => native.handlers?.onError({ jobId: "run-native-1", code: "ASTROMETRY_REQUIRED", message: "one or more filters did not produce a verified new WCS solution", retryable: false }));
    expect(screen.getByRole("heading", { name: "运行已失败关闭" })).toBeInTheDocument();
    // No stage keeps spinning; the stage that was running is marked failed and the earlier ones keep their state.
    expect(document.querySelector(".stage-list .spin")).toBeNull();
    expect(screen.getByText("发布与桌面校验").closest("li")).toHaveClass("failed");
    expect(screen.getByText("通道对齐").closest("li")).toHaveClass("done");
    const card = screen.getByRole("alert");
    expect(card).toHaveClass("run-failure");
    expect(card).toHaveTextContent("运行失败 —— 未发布任何产品");
    expect(card).toHaveTextContent("停止于：发布与桌面校验");
    expect(card).toHaveTextContent("ASTROMETRY_REQUIRED: one or more filters did not produce a verified new WCS solution");
    expect(card).toHaveTextContent("解算失败时不会保留积分后的 master");
    expect(card).toHaveTextContent("/结果/深空 输出/ultra-fast-wbpp-run.unsolved");
    expect(screen.getByRole("button", { name: "返回导入" })).toHaveClass("primary");
  });
});

describe("explicit catalog setup", () => {
  it("requires exact terms acceptance, streams fake-sidecar progress, and can cancel", async () => {
    native.catalogReady = false; render(<App />); await reachRecipe();
    const install = await screen.findByRole("button", { name: /下载并校验/ }); expect(install).toBeDisabled();
    await userEvent.click(screen.getByRole("link", { name: /查看提供方条款与索引说明/ })); expect(native.openProviderTerms).toHaveBeenCalledWith("https://astrometry.net/doc/readme.html#getting-index-files");
    await userEvent.click(screen.getByLabelText(/我已阅读提供方说明/)); expect(install).toBeEnabled(); await userEvent.click(install);
    expect(native.startCatalogInstall).toHaveBeenCalledWith({ catalogId: "astrometry-net-4107-4112", acceptedTermsId: "astrometry-net-index-data-2026-09" });
    await act(async () => native.catalogHandlers?.onProgress({ jobId: "catalog-1", catalogId: "astrometry-net-4107-4112", artifactId: "index-4108.fits", downloadedBytes: 50, sizeBytes: 100 }));
    expect(screen.getByText(/index-4108.fits · 50.0%/)).toBeInTheDocument(); await userEvent.click(screen.getByRole("button", { name: "取消下载" })); expect(native.cancelCatalogInstall).toHaveBeenCalledWith("catalog-1");
    await act(async () => native.catalogHandlers?.onError({ jobId: "catalog-1", catalogId: "astrometry-net-4107-4112", code: "CATALOG_INSTALL_FAILED", message: "terminated" })); expect(screen.queryByText(/CATALOG_INSTALL_FAILED/)).not.toBeInTheDocument();
  });

  it("verifies and configures only after download completion", async () => {
    native.catalogReady = false; render(<App />); await reachRecipe(); await userEvent.click(screen.getByLabelText(/我已阅读提供方说明/)); await userEvent.click(screen.getByRole("button", { name: /下载并校验/ }));
    await act(async () => native.catalogHandlers?.onComplete({ jobId: "catalog-1", catalogId: "astrometry-net-4107-4112", install: { schemaVersion: 1 } }));
    await waitFor(() => expect(native.verifyCatalog).toHaveBeenCalledWith("astrometry-net-4107-4112", true)); expect(await screen.findByText("离线星表已校验并配置")).toBeInTheDocument();
  });
});
