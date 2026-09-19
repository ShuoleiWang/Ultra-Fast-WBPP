import type { ReactNode } from "react";
import type { Translator } from "./i18n";
import { BiasIcon, DarkIcon, FlatIcon, ImportIcon, LightIcon, ProcessIcon, ResultIcon } from "./icons";
import type { RawFrameRole, WorkflowStep } from "./types";
import type { useWorkflow } from "./useWorkflow";

type Workflow = ReturnType<typeof useWorkflow>;
export type InventoryTab = RawFrameRole | "ALL";

const userStep = (step: WorkflowStep) => step === "import" || step === "inspect" ? 0 : step === "run" ? 1 : 2;

/** Source-list sidebar: the project's three states, the imported sources and the run's results. */
export function Sidebar({ workflow, t, inventoryTab, setInventoryTab }: { workflow: Workflow; t: Translator; inventoryTab: InventoryTab; setInventoryTab: (tab: InventoryTab) => void }) {
  const currentStep = userStep(workflow.step);
  const count = (role: RawFrameRole) => workflow.sources.filter((source) => source.role === role || source.role === `MASTER_${role}`).reduce((sum, source) => sum + source.fileCount, 0);
  const masterOnly = (role: RawFrameRole) => count(role) > 0 && (workflow.sources.find((source) => source.role === role)?.fileCount ?? 0) === 0;
  const steps: Array<{ target: WorkflowStep; label: string; icon: ReactNode; state: string }> = [
    { target: "import", label: t("stepImport"), icon: <ImportIcon />, state: workflow.importedTotal > 0 ? String(workflow.importedTotal) : "" },
    { target: "run", label: t("stepProcess"), icon: <ProcessIcon />, state: workflow.step === "run" ? `${workflow.overallProgress}%` : workflow.runStatus === "COMPLETED" ? "✓" : "" },
    { target: "result", label: t("stepResult"), icon: <ResultIcon />, state: workflow.artifacts.length ? String(workflow.artifacts.length) : "" },
  ];
  const onImport = workflow.step === "import";
  const pick = (tab: InventoryTab) => { setInventoryTab(tab); if (!workflow.runNavigationLocked && workflow.step === "inspect") workflow.setStep("import"); };
  return <nav className="sidebar" aria-label={t("workflowLabel")}>
    <div className="sidebar-scroll">
      <div className="sb-head">{t("sidebarProject")}</div>
      {steps.map((item, index) => {
        const active = index === currentStep;
        const disabled = workflow.runNavigationLocked || workflow.inputBusy || index !== 0 || currentStep === 0;
        return <button type="button" key={item.target} className="sb-row" aria-current={active ? "step" : undefined} disabled={disabled} onClick={() => !disabled && workflow.setStep(item.target)}>{item.icon}<span className="sb-name">{item.label}</span><span className={`sb-state ${item.state === "✓" ? "ok" : ""}`}>{item.state}</span></button>;
      })}
      {workflow.importedTotal > 0 && <>
        <div className="sb-head">{t("sidebarSources")}</div>
        <button type="button" className="sb-row" aria-pressed={onImport && inventoryTab === "LIGHT"} onClick={() => pick("LIGHT")}><LightIcon /><span className="sb-name">Light</span><span className="sb-count">{count("LIGHT")}</span></button>
        {workflow.matrix.map((cell) => <button type="button" key={cell.panelId} className="sb-row child" aria-pressed={onImport && inventoryTab === "LIGHT"} onClick={() => pick("LIGHT")}><span className="sb-name">{cell.target} · {cell.filter}</span><span className="sb-count">{cell.lightCount}</span></button>)}
        <button type="button" className="sb-row" aria-pressed={onImport && inventoryTab === "FLAT"} onClick={() => pick("FLAT")}><FlatIcon /><span className="sb-name">Flat{masterOnly("FLAT") ? " · Master" : ""}</span><span className="sb-count">{count("FLAT")}</span></button>
        <button type="button" className="sb-row" aria-pressed={onImport && inventoryTab === "DARK"} onClick={() => pick("DARK")}><DarkIcon /><span className="sb-name">Dark{masterOnly("DARK") ? " · Master" : ""}</span><span className="sb-count">{count("DARK")}</span></button>
        <button type="button" className="sb-row" aria-pressed={onImport && inventoryTab === "BIAS"} onClick={() => pick("BIAS")}><BiasIcon /><span className="sb-name">Bias{masterOnly("BIAS") ? " · Master" : ""}</span><span className="sb-count">{count("BIAS")}</span></button>
      </>}
      {workflow.artifacts.length > 0 && <>
        <div className="sb-head">{t("sidebarResults")}</div>
        <button type="button" className="sb-row" aria-current={workflow.step === "result" ? "true" : undefined} disabled={workflow.runNavigationLocked} onClick={() => workflow.setStep("result")}><ResultIcon /><span className="sb-name">{t("sidebarRun")}</span><span className="sb-count">{workflow.artifacts.length}</span></button>
      </>}
    </div>
    <div className="sb-foot">
      <div className="engine" title={workflow.capabilities?.available ? t("engineReady") : t("engineUnavailable")}><span className={`runtime-dot ${workflow.capabilities?.available ? "online" : ""}`} /><strong>{workflow.capabilities?.chip ?? t("detectingHardware")}</strong></div>
      <details><summary>{t("legal")}</summary><p>{t("legalBody")}</p></details>
    </div>
  </nav>;
}
