import type { Translator, UiLanguage } from "./i18n";
import { BrandMark, FilesIcon, FolderIcon, InspectorIcon } from "./icons";
import type { useWorkflow } from "./useWorkflow";
import { stageLabel } from "./views";

type Workflow = ReturnType<typeof useWorkflow>;

/** The window toolbar: import actions, the document title or the run activity, language and the inspector toggle. */
export function Toolbar({ workflow, language, setLanguage, t, inspectorOpen, onToggleInspector }: {
  workflow: Workflow; language: UiLanguage; setLanguage: (value: UiLanguage) => void; t: Translator;
  inspectorOpen: boolean; onToggleInspector: () => void;
}) {
  const importDisabled = !workflow.nativeRuntime || workflow.inputBusy || workflow.runNavigationLocked;
  const showActivity = workflow.step === "run";
  const runningStage = workflow.stages.find((stage) => stage.status === "RUNNING");
  const activityText = workflow.runStatus === "CANCELLED" ? t("cancelledTitle") : workflow.runStatus === "FAILED" ? t("failedTitle")
    : workflow.runStatus === "COMPLETED" ? t("resultTitle") : runningStage ? stageLabel(runningStage.stageId, t) : t("runningTitle");
  const groups = workflow.matrix.length;
  return <header className="topbar" data-tauri-drag-region="deep">
    <div className="tb-left">
      <div className="brand" aria-label="Ultra-Fast WBPP"><span className="brand-mark"><BrandMark size={22} /></span><span>Ultra-Fast <em>WBPP</em></span></div>
      {!workflow.nativeRuntime && <span className="badge-demo">{t("browserDemo")}</span>}
      <button type="button" className="btn" aria-label={t("chooseFiles")} title={t("chooseFiles")} disabled={importDisabled} onClick={() => void workflow.pickFiles(workflow.selectedRole)}><FilesIcon /><span className="label">{t("chooseFiles")}</span></button>
      <button type="button" className="btn" aria-label={t("chooseFolder")} title={t("chooseFolder")} disabled={importDisabled} onClick={() => void workflow.pickDirectories(workflow.selectedRole)}><FolderIcon /><span className="label">{t("chooseFolder")}</span></button>
      {workflow.importedTotal > 0 && <button type="button" className="btn quiet danger" disabled={workflow.inputBusy || workflow.runNavigationLocked} title={t("clearTitle")} onClick={workflow.clearSources}>{t("clear")}</button>}
    </div>
    <div className="tb-center">
      {showActivity
        ? <div className={`activity ${workflow.runStatus === "COMPLETED" ? "done" : ""}`} role="progressbar" aria-label={`${workflow.overallProgress}%`} aria-valuenow={workflow.overallProgress} aria-valuemin={0} aria-valuemax={100}>
            <div className="bar"><span style={{ width: `${workflow.overallProgress}%` }} /></div>
            <span className="txt tnum">{workflow.overallProgress}% · {activityText}</span>
          </div>
        : <>
            <div className="doc-title">{workflow.importedTotal > 0 ? workflow.projectName : "Ultra-Fast WBPP"}</div>
            <div className="doc-sub">{workflow.importedTotal > 0 ? t("statusFrames", { count: workflow.importedTotal, groups }) : t("newProject")}</div>
          </>}
    </div>
    <div className="tb-right">
      <label className="language-picker" data-tauri-drag-region="false"><span>{t("language")}</span><select aria-label={t("language")} value={language} onChange={(event) => setLanguage(event.target.value as UiLanguage)}><option value="en">{t("english")}</option><option value="zh-CN">{t("chinese")}</option></select></label>
      <button type="button" className="btn icon" aria-pressed={inspectorOpen} aria-label={t("inspectorLabel")} title={t("inspectorLabel")} onClick={onToggleInspector}><InspectorIcon /></button>
    </div>
  </header>;
}
