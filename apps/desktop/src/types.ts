export type RawFrameRole = "LIGHT" | "FLAT" | "DARK" | "BIAS";
export type MasterFrameRole = "MASTER_FLAT" | "MASTER_DARK" | "MASTER_BIAS";
export type FrameRole = RawFrameRole | MasterFrameRole;
export type GateDisposition = "PASS" | "REVIEW" | "HARD_FAIL";
export type WorkflowStep = "import" | "inspect" | "run" | "result";
export type RecipeId = "balanced";
export type RunStatus = "IDLE" | "RUNNING" | "CANCELLING" | "CANCELLED" | "COMPLETED" | "FAILED";
export type ProgressState = "queued" | "running" | "finalizing" | "succeeded" | "failed" | "cancelled";

export interface SourceSet {
  role: FrameRole;
  label: string;
  hint: string;
  paths: string[];
  fileCount: number;
  detected: boolean;
  confirmed: boolean;
  reuseAllowed: boolean;
}

export interface GateSummary { pass: number; review: number; hardFail: number; }
export interface Recipe { id: RecipeId; name: string; eyebrow: string; description: string; details: string[]; estimatedScale: string; recommended?: boolean; available: boolean; }

export interface RuntimeCapabilities {
  platform: "macos" | "windows" | "linux" | "browser";
  chip: string;
  cpuBackend: string;
  gpuBackend: string;
  optimizationTier: "M3_PRO_TUNED" | "APPLE_SILICON" | "PORTABLE" | "MOCK";
  available: boolean;
  drizzleAvailable: boolean;
  solverAvailable: boolean;
  runtimeVersion?: string;
  unavailableReason?: string;
}

export interface InspectRequest { paths: string[]; roleHint?: FrameRole; }
export interface InspectedSource { role: FrameRole; paths: string[]; fileCount: number; confidence: number; needsConfirmation: boolean; }
export interface InspectedAsset {
  path: string;
  role: FrameRole;
  width: number;
  height: number;
  channels: number;
  filter: string;
  target: string;
  camera: string;
  exposureSeconds?: number | null;
  temperatureCelsius?: number | null;
  gain?: number | null;
  offset?: number | null;
  binning: [number, number];
  cfaPattern: string;
  readoutMode: string;
  sourceSha256?: string | null;
  observedAt?: string | null;
}
export interface InspectResponse { sources: InspectedSource[]; assets: InspectedAsset[]; totalFiles: number; projectName: string; }
export interface HashSourcesResponse { entries: Array<{ path: string; sourceSha256: string }>; }
export interface QualityEvidence { code: string; family: string; severity: string; message: string; }
export interface InspectedLightQuality {
  path: string;
  sourceSha256?: string | null;
  disposition: GateDisposition;
  decision: string;
  confidence: string;
  starCount: number;
  summary: string;
  previewDataUrl?: string | null;
  previewSha256?: string | null;
  evidence: QualityEvidence[];
}
export interface QualityInspection {
  schemaVersion: 1;
  gatePolicyDigest: string;
  workers: number;
  counts: Record<GateDisposition, number>;
  frames: InspectedLightQuality[];
}

export interface PanelCell { panelId: string; target: string; filter: string; lightCount: number; }
export interface MasterMetadataOverride {
  sourceSha256: string;
  sourcePath: string;
  role: MasterFrameRole;
  camera: string;
  gain: number | null;
  offset: number | null;
  binning: [number | null, number | null];
  filter: string;
  cfaPattern: string;
  readoutMode: string;
  temperatureCelsius: number | null;
  exposureSeconds: number | null;
  biasIncluded: boolean | null;
  numericDomain: "NORMALIZED_UNIT" | "SENSOR_CODE" | null;
  normalizedUnitScale: number | null;
  needsMetadataOverride: boolean;
  confirmed: boolean;
}
export interface MasterMetadataOverrideRequest {
  sourceSha256: string;
  camera?: string;
  gain?: number;
  offset?: number;
  binning?: [number, number];
  filter?: string;
  cfaPattern?: string;
  readoutMode?: string;
  temperatureCelsius?: number;
  exposureSeconds?: number;
  biasIncluded?: boolean;
  numericDomain?: "NORMALIZED_UNIT" | "SENSOR_CODE";
  normalizedUnitScale?: number;
}

export interface RawFrameMetadataOverride {
  sourceSha256: string;
  sourcePath: string;
  camera: string;
  filter: string;
  role: RawFrameRole;
  gain: number | null;
  offset: number | null;
  binning: [number, number];
  readoutMode: string;
  cfaPattern: "NONE" | "RGGB" | "BGGR" | "GRBG" | "GBRG" | "UNKNOWN";
  confirmed: boolean;
}

export interface ProjectRecipeOptions { balanced: true; drizzleEnabled: boolean; localNormalizationEnabled: boolean; solverRequired: true; calibrationWorkflow: "mono-standard-v1"; }
export interface ReviewApprovalSelection { sourceSha256: string; gatePolicyDigest: string; }
export interface RunSource { sourceId: string; role: FrameRole; paths: string[]; recursive: boolean; }
export interface RunRequest {
  sources: RunSource[];
  projectName: string;
  /** Target-derived label that names the output folder (`NGC 7331`); empty falls back to projectName. */
  runLabel: string;
  recipe: ProjectRecipeOptions;
  masterMetadataOverrides: MasterMetadataOverrideRequest[];
  rawFrameMetadataOverrides: Array<Pick<RawFrameMetadataOverride, "sourceSha256" | "cfaPattern">>;
  reviewSelections: ReviewApprovalSelection[];
  outputParentDirectory: string;
}
export interface CalibrationInspectionRequest {
  paths: string[];
  recipe: {
    calibration: { workflow: "mono-standard-v1"; bias: "OPTIONAL"; masterMetadataOverrides: RunRequest["masterMetadataOverrides"] };
    rawFrameMetadataOverrides: RunRequest["rawFrameMetadataOverrides"];
  };
}
export interface CalibrationInspection {
  schemaVersion: 1;
  status: "READY" | "BLOCKED";
  calibrationReady: boolean;
  groups: Array<{
    groupId: string; target: string; filter: string; lightCount: number; observedDates: string[]; status: "READY" | "BLOCKED";
    matches: Record<"FLAT" | "DARK" | "BIAS", { rawCount: number; masterCount: number }>;
  }>;
  issues: Array<{ code: string; severity: "ERROR" | "WARNING" | "INFO"; message: string; paths: string[]; lightGroups: string[] }>;
}
export interface RunReceipt { jobId: string; accepted: boolean; executionMode: "native"; outputDirectory: string; }

export interface AstrometryReceipt {
  referenceFrame: string;
  projection: string;
  centerRaDegrees: number;
  centerDecDegrees: number;
  pixelScaleArcsec: number;
  rotationDegrees: number;
  rmsPixels: number;
  rmsArcsec: number;
  matchedStars: number;
  parity: "POSITIVE" | "NEGATIVE";
  catalogIdentity: string;
  indexIdentities: string[];
  correspondenceSha256: string;
  catalogManaged: true;
  installedSetIdentity: string;
  catalogManifestSha256: string;
  indexArtifacts: Array<{ indexId: string; relativeName: string; sizeBytes: number; sha256: string; manifestSha256: string; installedSetIdentity: string }>;
  wcsSha256: string;
}
export interface ArtifactReceipt { artifactId: string; relativePath: string; sha256: string; sizeBytes: number; astrometry?: AstrometryReceipt; }
export type OutputArtifactKind = "SOLVED_MONO_FITS" | "LINEAR_RGB_FITS" | "RGB_PREVIEW_TIFF_16" | "RGB_PREVIEW_PNG_16" | "MONO_PREVIEW_PNG" | "RECEIPT" | "REPORT" | "DRIZZLE_DATA" | "MASTER" | "PREVIEW";
export interface OutputArtifact { kind: OutputArtifactKind; name: string; path: string; detail: string; filter?: string; target?: string; receipt?: ArtifactReceipt; }

export interface StageProgress { name: string; stageId: string; status: "WAITING" | "RUNNING" | "DONE" | "FAILED"; percent: number; }
export interface PipelineProgressEvent { jobId: string; stageId?: string; state: ProgressState; fraction: number; completedUnits?: number | null; totalUnits?: number | null; message: string; overallFraction?: number; scope?: "panel" | "project"; panelId?: string; panelTarget?: string; panelFilter?: string; panelIndex?: number; panelCount?: number; }
export interface PipelineArtifactEvent { jobId: string; stage: { stageId: string; kind: string; status: string }; artifact: ArtifactReceipt; }
export interface GateCheck { code: string; required: boolean; passed: boolean; artifactIds: string[]; message: string; }
export interface ResultGateReport { decision: "ready" | "blocked"; checks: GateCheck[]; }
/** One Light the run's quality gate did not pass, with the reasons and a small preview. */
export interface ScreeningFrame { name: string; target?: string; disposition: GateDisposition; admitted: boolean; summary: string; evidence: string[]; starCount?: number; previewDataUrl?: string; }
/** The run's Light screening: counts plus every frame that needed a decision. */
export interface ScreeningSummary { admitted: number; excluded: number; counts: Partial<Record<GateDisposition, number>>; frames: ScreeningFrame[]; }
export interface PipelineCompleteEvent { jobId: string; outputDirectory: string; artifacts: OutputArtifact[]; gate: ResultGateReport; screening?: ScreeningSummary; }
export interface PipelineErrorEvent { jobId: string; code: string; message: string; retryable: boolean; details?: Record<string, unknown>; }
export interface PipelineLogEvent { jobId: string; stream: "stderr"; message: string; }
export interface PipelineEventHandlers {
  onProgress(event: PipelineProgressEvent): void;
  onArtifact(event: PipelineArtifactEvent): void;
  onComplete(event: PipelineCompleteEvent): void;
  onError(event: PipelineErrorEvent): void;
  onLog?(event: PipelineLogEvent): void;
}

export interface CatalogProviderTerms { acceptanceId: string; url: string; summary: string; licenseStatus: string; requiresExplicitAcceptance: boolean; }
export interface CatalogArtifactInfo { artifactId: string; sizeBytes: number | null; installScope?: string; }
export interface CatalogInfo {
  catalogId: string;
  provider: string;
  version: string;
  totalSizeBytes: number;
  artifactCount: number;
  installedArtifactsBySize: number;
  fullyInstalledBySize: boolean;
  allowedDownloadOrigins: string[];
  providerTerms: CatalogProviderTerms;
  artifacts: CatalogArtifactInfo[];
}
export interface CatalogListResponse { schemaVersion: 1; catalogRoot: string; catalogs: CatalogInfo[]; }
export interface CatalogDoctorResponse { schemaVersion: 1; ok: boolean; catalogRoot: string; config: { path?: string; present: boolean; valid: boolean }; installedSetBindingReady: boolean; message: string; }
export interface SolverBackendStatus { backendId: string; displayName: string; version: string; available: boolean; executionReady: boolean; reason?: string | null; metadata?: { probe?: { path?: string | null; version?: string; executionReady?: boolean } }; }
export interface SolverDoctorResponse { schemaVersion: 1; engineVersion: string; backends: SolverBackendStatus[]; }

export interface CatalogInstallRequest { catalogId: string; acceptedTermsId: string; fieldOfViewDegrees?: number; }
export interface CatalogJobReceipt { jobId: string; accepted: boolean; catalogId: string; }
export interface CatalogProgressEvent { jobId: string; catalogId: string; artifactId: string; downloadedBytes: number; sizeBytes: number; }
export interface CatalogCompleteEvent { jobId: string; catalogId: string; install: Record<string, unknown>; }
export interface CatalogErrorEvent { jobId: string; catalogId: string; code: string; message: string; }
export interface CatalogEventHandlers { onProgress(event: CatalogProgressEvent): void; onComplete(event: CatalogCompleteEvent): void; onError(event: CatalogErrorEvent): void; }
