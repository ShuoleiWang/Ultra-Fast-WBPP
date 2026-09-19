import { useEffect, useRef } from "react";
import { hasTauriRuntime } from "./bridge";
import type { useWorkflow } from "./useWorkflow";

type Workflow = ReturnType<typeof useWorkflow>;
export type DemoStage = "frames" | "review" | "run" | "result";

/** `?mac=1` reserves the traffic-light area in browser builds (documentation screenshots only). */
export function requestedMacChrome(): boolean {
  if (hasTauriRuntime() || typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).get("mac") === "1";
}

/** The `?demo=` stage requested by the URL, browser builds only (documentation screenshots). */
export function requestedDemoStage(): DemoStage | undefined {
  if (hasTauriRuntime() || typeof window === "undefined") return undefined;
  const value = new URLSearchParams(window.location.search).get("demo");
  return value === "frames" || value === "review" || value === "run" || value === "result" ? value : undefined;
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
    if (phase.current === 2 && stage !== "frames" && workflow.allRequiredConfirmed && workflow.step === "import") { phase.current = 3; void workflow.runInspection(); return; }
    if (phase.current === 3 && (stage === "run" || stage === "result") && workflow.step === "inspect" && workflow.canStart) { phase.current = 4; void workflow.startRun(); return; }
  }, [stage, workflow]);
}
