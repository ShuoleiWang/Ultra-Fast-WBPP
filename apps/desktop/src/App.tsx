import { useEffect, useState } from "react";
import { BlinkView } from "./BlinkView";
import { requestedDemoStage, requestedMacChrome, useDemoAutopilot } from "./demoAutopilot";
import { useI18n } from "./i18n";
import { Inspector } from "./Inspector";
import { Sidebar, type InventoryTab } from "./Sidebar";
import { Toolbar } from "./Toolbar";
import type { GateDisposition } from "./types";
import { useWorkflow } from "./useWorkflow";
import { ImportView, PASS_FRAME_BATCH, ResultView, RunView, ScreeningView } from "./views";

/** One window, one project: toolbar, source-list sidebar, content, inspector. */
function App() {
  const { language, setLanguage, t } = useI18n();
  const workflow = useWorkflow(t);
  useDemoAutopilot(workflow, requestedDemoStage());
  const [inputPathsText, setInputPathsText] = useState("");
  const [outputPathText, setOutputPathText] = useState("");
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [inventoryTab, setInventoryTab] = useState<InventoryTab>("ALL");
  const [selectedPath, setSelectedPath] = useState<string>();
  const [visiblePassFrames, setVisiblePassFrames] = useState(PASS_FRAME_BATCH);
  const [decisionFilter, setDecisionFilter] = useState<"ALL" | GateDisposition>("ALL");
  // The frame on the blink stage (by content digest); the inspector shows the
  // same one.  A digest a new session no longer has falls back to the view's
  // first frame, so no reset is needed when the session changes.
  const [blinkSelectedSha, setBlinkSelectedSha] = useState<string>();
  useEffect(() => {
    setVisiblePassFrames(PASS_FRAME_BATCH);
    setSelectedPath(undefined);
    setDecisionFilter("ALL");
  }, [workflow.qualityInspection]);
  const lightCount = workflow.sources.find((source) => source.role === "LIGHT")?.fileCount ?? 0;
  const platform = workflow.capabilities?.platform;
  const macos =
    (workflow.nativeRuntime &&
      (platform === "macos" ||
        (platform === undefined && typeof navigator !== "undefined" && /Mac/i.test(navigator.platform)))) ||
    requestedMacChrome();
  const selectedFrame = workflow.qualityInspection?.frames.find((frame) => frame.path === selectedPath);
  const showInspector =
    inspectorOpen && (workflow.step === "import" || workflow.step === "inspect" || workflow.step === "blink");

  return (
    <div
      className={`app ${workflow.step === "blink" ? "blink-focused" : ""} ${macos ? "platform-macos" : ""} ${workflow.nativeRuntime ? "" : "browser"}`}
      onContextMenu={(event) => {
        if (!(event.target instanceof Element) || !event.target.closest("input, textarea, [contenteditable=true]"))
          event.preventDefault();
      }}
    >
      <a className="skip-link" href="#workspace">
        {t("skipWorkspace")}
      </a>
      <Toolbar
        workflow={workflow}
        language={language}
        setLanguage={setLanguage}
        t={t}
        inspectorOpen={inspectorOpen}
        onToggleInspector={() => setInspectorOpen((open) => !open)}
      />
      <div className="body">
        <Sidebar workflow={workflow} t={t} inventoryTab={inventoryTab} setInventoryTab={setInventoryTab} />
        <main
          id="workspace"
          className="content"
          onDragEnter={(event) => {
            event.preventDefault();
            workflow.setDragging(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => workflow.setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            workflow.setDragging(false);
          }}
        >
          {workflow.isDragging && (
            <div className="drop-overlay">{workflow.nativeRuntime ? t("dropNative") : t("dropBrowser")}</div>
          )}
          {workflow.step === "import" && (
            <ImportView
              workflow={workflow}
              t={t}
              inputPathsText={inputPathsText}
              setInputPathsText={setInputPathsText}
              outputPathText={outputPathText}
              setOutputPathText={setOutputPathText}
              inventoryTab={inventoryTab}
              setInventoryTab={setInventoryTab}
              lightCount={lightCount}
            />
          )}
          {workflow.step === "inspect" && (
            <ScreeningView
              workflow={workflow}
              t={t}
              outputPathText={outputPathText}
              setOutputPathText={setOutputPathText}
              selectedPath={selectedPath}
              setSelectedPath={setSelectedPath}
              visiblePassFrames={visiblePassFrames}
              setVisiblePassFrames={setVisiblePassFrames}
              decisionFilter={decisionFilter}
              setDecisionFilter={setDecisionFilter}
              lightCount={lightCount}
            />
          )}
          {workflow.step === "blink" && (
            <BlinkView
              workflow={workflow}
              t={t}
              inspectorOpen={inspectorOpen}
              selectedSha={blinkSelectedSha}
              setSelectedSha={setBlinkSelectedSha}
            />
          )}
          {workflow.step === "run" && <RunView workflow={workflow} t={t} />}
          {workflow.step === "result" && <ResultView workflow={workflow} t={t} />}
        </main>
        {showInspector && (
          <Inspector workflow={workflow} t={t} frame={selectedFrame} blinkSelectedSha={blinkSelectedSha} />
        )}
      </div>
    </div>
  );
}

export default App;
