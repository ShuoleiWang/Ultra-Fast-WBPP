import { invoke } from "@tauri-apps/api/core";
import { translateCurrent } from "./i18n";
import type {
  BlinkMeasureRequest,
  BlinkMeasureResponse,
  CalibrationInspection,
  CalibrationInspectionRequest,
  CatalogDoctorResponse,
  CatalogEventHandlers,
  CatalogInstallRequest,
  CatalogJobReceipt,
  CatalogListResponse,
  FrameRole,
  InspectRequest,
  InspectResponse,
  HashSourcesResponse,
  PipelineCompleteEvent,
  PipelineErrorEvent,
  PipelineEventHandlers,
  PipelineProgressEvent,
  QualityInspection,
  RunReceipt,
  RunRequest,
  RuntimeCapabilities,
  SolverDoctorResponse,
} from "./types";

export interface DesktopBridge {
  getCapabilities(): Promise<RuntimeCapabilities>;
  inspectPaths(request: InspectRequest): Promise<InspectResponse>;
  inspectCalibration(request: CalibrationInspectionRequest): Promise<CalibrationInspection>;
  inspectQuality(paths: string[]): Promise<QualityInspection>;
  /** Measures the Lights, renders the normalised previews and returns the `blink-manifest-v1` of the new session. */
  blinkMeasure(request: BlinkMeasureRequest): Promise<BlinkMeasureResponse>;
  /** One preview of a blink session (`filmstrip/…` or `zoom/…`) as a data URL, bounded by the controller. */
  loadBlinkPreview(sessionDirectory: string, relativePath: string): Promise<string>;
  hashSources(paths: string[]): Promise<HashSourcesResponse>;
  startRun(request: RunRequest): Promise<RunReceipt>;
  cancelRun(jobId: string): Promise<void>;
  pickInputFiles(role?: FrameRole): Promise<string[]>;
  pickInputDirectories(role?: FrameRole): Promise<string[]>;
  pickOutputParent(): Promise<string | undefined>;
  catalogList(): Promise<CatalogListResponse>;
  catalogDoctor(): Promise<CatalogDoctorResponse>;
  solverDoctor(): Promise<SolverDoctorResponse>;
  startCatalogInstall(request: CatalogInstallRequest): Promise<CatalogJobReceipt>;
  cancelCatalogInstall(jobId: string): Promise<void>;
  verifyCatalog(catalogId: string, configure: boolean): Promise<CatalogDoctorResponse | Record<string, unknown>>;
  openProviderTerms(url: string): Promise<void>;
  revealOutput(path: string): Promise<void>;
}

export const hasTauriRuntime = () => typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

const browserOnlyError = () => new Error(translateCurrent("browserOnlyError"));

const mockBridge: DesktopBridge = {
  async getCapabilities() {
    return {
      platform: "browser",
      chip: "Browser demo",
      cpuBackend: "No native runtime",
      gpuBackend: "No native runtime",
      optimizationTier: "MOCK",
      available: false,
      drizzleAvailable: false,
      solverAvailable: false,
      unavailableReason: translateCurrent("browserSidecarMissing"),
    };
  },
  async inspectPaths() {
    throw browserOnlyError();
  },
  async inspectCalibration() {
    throw browserOnlyError();
  },
  async inspectQuality() {
    throw browserOnlyError();
  },
  async blinkMeasure() {
    throw browserOnlyError();
  },
  async loadBlinkPreview() {
    throw browserOnlyError();
  },
  async hashSources() {
    throw browserOnlyError();
  },
  async startRun() {
    throw browserOnlyError();
  },
  async cancelRun() {},
  async pickInputFiles() {
    throw browserOnlyError();
  },
  async pickInputDirectories() {
    throw browserOnlyError();
  },
  async pickOutputParent() {
    throw browserOnlyError();
  },
  async catalogList() {
    throw browserOnlyError();
  },
  async catalogDoctor() {
    throw browserOnlyError();
  },
  async solverDoctor() {
    throw browserOnlyError();
  },
  async startCatalogInstall() {
    throw browserOnlyError();
  },
  async cancelCatalogInstall() {},
  async verifyCatalog() {
    throw browserOnlyError();
  },
  async openProviderTerms() {
    throw browserOnlyError();
  },
  async revealOutput() {
    throw browserOnlyError();
  },
};

const normalizeSelection = (selection: string | string[] | null): string[] => {
  if (selection === null) return [];
  return Array.isArray(selection) ? selection : [selection];
};

const tauriBridge: DesktopBridge = {
  getCapabilities: () => invoke("get_capabilities"),
  inspectPaths: (request) => invoke("inspect_paths", { request }),
  inspectCalibration: (request) => invoke("inspect_calibration", { request }),
  inspectQuality: (paths) => invoke("inspect_quality", { request: { paths } }),
  blinkMeasure: (request) => invoke("blink_measure", { request }),
  async loadBlinkPreview(sessionDirectory, relativePath) {
    // The controller answers with the data URL itself or wrapped as `{ dataUrl }`.
    const result = await invoke<string | { dataUrl?: unknown }>("load_blink_preview", {
      sessionDirectory,
      relativePath,
    });
    const dataUrl =
      typeof result === "string" ? result : result && typeof result === "object" ? result.dataUrl : undefined;
    if (typeof dataUrl !== "string" || !dataUrl.startsWith("data:image/"))
      throw new Error(translateCurrent("blinkPreviewInvalid"));
    return dataUrl;
  },
  hashSources: (paths) => invoke("hash_sources", { request: { paths } }),
  startRun: (request) => invoke("start_project", { request }),
  cancelRun: (jobId) => invoke("cancel_project", { jobId }),
  async pickInputFiles() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    return normalizeSelection(
      await open({
        multiple: true,
        directory: false,
        title: translateCurrent("dialogFrames"),
        filters: [{ name: "Astronomy frames", extensions: ["fit", "fits", "fts", "xisf"] }],
      }),
    );
  },
  async pickInputDirectories() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    return normalizeSelection(
      await open({
        multiple: true,
        directory: true,
        title: translateCurrent("dialogFolders"),
      }),
    );
  },
  async pickOutputParent() {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const selected = await open({ multiple: false, directory: true, title: translateCurrent("dialogOutput") });
    return typeof selected === "string" ? selected : undefined;
  },
  catalogList: () => invoke("catalog_list"),
  catalogDoctor: () => invoke("catalog_doctor"),
  solverDoctor: () => invoke("solver_doctor"),
  startCatalogInstall: (request) => invoke("start_catalog_install", { request }),
  cancelCatalogInstall: (jobId) => invoke("cancel_catalog_install", { jobId }),
  verifyCatalog: (catalogId, configure) => invoke("catalog_verify", { catalogId, configure }),
  async openProviderTerms(value) {
    const url = new URL(value);
    if (url.protocol !== "https:" || !["astrometry.net", "www.hnsky.org"].includes(url.hostname)) {
      throw new Error(translateCurrent("providerLinkError"));
    }
    const { openUrl } = await import("@tauri-apps/plugin-opener");
    await openUrl(url.toString());
  },
  async revealOutput(path) {
    if (!path || path.includes("\0")) throw new Error(translateCurrent("outputPathError"));
    const { revealItemInDir } = await import("@tauri-apps/plugin-opener");
    await revealItemInDir(path);
  },
};

export const desktopBridge: DesktopBridge = hasTauriRuntime() ? tauriBridge : mockBridge;

export async function listenForDesktopDrops(onPaths: (paths: string[]) => void): Promise<() => void> {
  if (!hasTauriRuntime()) return () => undefined;
  // Only the drop itself is needed. `onDragDropEvent` would also subscribe to
  // enter/over/leave, and the core forwards every drag-over event (one per
  // pointer move) to a webview that listens for it, which stutters the drag.
  const { listen, TauriEvent } = await import("@tauri-apps/api/event");
  return listen<{ paths: string[] }>(TauriEvent.DRAG_DROP, ({ payload }) => {
    if (Array.isArray(payload.paths) && payload.paths.length) onPaths(payload.paths);
  });
}

export async function listenForCatalogEvents(handlers: CatalogEventHandlers): Promise<() => void> {
  if (!hasTauriRuntime()) return () => undefined;
  const { listen } = await import("@tauri-apps/api/event");
  const disposers = await Promise.all([
    listen("ufwbpp://catalog-progress", ({ payload }) =>
      handlers.onProgress(payload as Parameters<CatalogEventHandlers["onProgress"]>[0]),
    ),
    listen("ufwbpp://catalog-complete", ({ payload }) =>
      handlers.onComplete(payload as Parameters<CatalogEventHandlers["onComplete"]>[0]),
    ),
    listen("ufwbpp://catalog-error", ({ payload }) =>
      handlers.onError(payload as Parameters<CatalogEventHandlers["onError"]>[0]),
    ),
  ]);
  return () => disposers.forEach((dispose) => dispose());
}

export async function listenForPipelineEvents(handlers: PipelineEventHandlers): Promise<() => void> {
  if (!hasTauriRuntime()) return () => undefined;
  const { listen } = await import("@tauri-apps/api/event");
  const disposers = await Promise.all([
    listen<PipelineProgressEvent>("ufwbpp://pipeline-progress", ({ payload }) => handlers.onProgress(payload)),
    listen<PipelineCompleteEvent>("ufwbpp://pipeline-complete", ({ payload }) => handlers.onComplete(payload)),
    listen<PipelineErrorEvent>("ufwbpp://pipeline-error", ({ payload }) => handlers.onError(payload)),
  ]);
  return () => disposers.forEach((dispose) => dispose());
}
