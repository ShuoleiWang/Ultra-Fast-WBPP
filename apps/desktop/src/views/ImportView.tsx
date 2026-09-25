import { CalibrationGroups, FrameInventory } from "../FrameInventory";
import type { InventoryTab } from "../Sidebar";
import type { Translator } from "../i18n";
import { BrandMark } from "../icons";
import { LaunchBar, LaunchSettings, MetadataConfirmations } from "./LaunchSettings";
import { Alert, StarField, type Workflow } from "./common";

export function ImportView({
  workflow,
  t,
  inputPathsText,
  setInputPathsText,
  outputPathText,
  setOutputPathText,
  inventoryTab,
  setInventoryTab,
  lightCount,
}: {
  workflow: Workflow;
  t: Translator;
  inputPathsText: string;
  setInputPathsText: (value: string) => void;
  outputPathText: string;
  setOutputPathText: (value: string) => void;
  inventoryTab: InventoryTab;
  setInventoryTab: (tab: InventoryTab) => void;
  lightCount: number;
}) {
  const pathEntry = workflow.nativeRuntime && (
    <div className="panel">
      <details className="disclosure">
        <summary>
          <span>{t("pasteInputPaths")}</span>
        </summary>
        <form
          className="details-content"
          onSubmit={(event) => {
            event.preventDefault();
            const paths = [
              ...new Set(
                inputPathsText
                  .split(/\r?\n/)
                  .map((path) => path.trim())
                  .filter(Boolean),
              ),
            ];
            void workflow.importPaths(paths, workflow.selectedRole);
          }}
        >
          <div className="field-row">
            <label htmlFor="input-paths">{t("inputPathsLabel")}</label>
            <textarea
              id="input-paths"
              rows={4}
              value={inputPathsText}
              disabled={workflow.inputBusy}
              spellCheck={false}
              aria-describedby="input-paths-hint"
              onChange={(event) => setInputPathsText(event.target.value)}
            />
            <p id="input-paths-hint" className="hint">
              {t("inputPathsHint")}
            </p>
          </div>
          <div className="row-actions">
            <button type="submit" className="btn small" disabled={workflow.inputBusy || !inputPathsText.trim()}>
              {t("importEnteredPaths")}
            </button>
          </div>
        </form>
      </details>
    </div>
  );
  const empty = workflow.assets.length === 0;
  return (
    <section className="view" aria-labelledby="import-title">
      {empty ? (
        <div className="scroll">
          <div className="hero-page">
            <div className="hero">
              <StarField className="hero-sky" />
              <div className="hero-glow" aria-hidden="true" />
              <div className="hero-body">
                <span className="symbol">
                  <BrandMark size={88} />
                </span>
                <h1 id="import-title">{t("importTitle")}</h1>
                <p className="hero-lead">
                  {workflow.inventoryBusy
                    ? t("readingMetadata")
                    : workflow.nativeRuntime
                      ? t("dropNative")
                      : t("dropBrowser")}
                </p>
                <p className="hero-sub">{t("importDescription")}</p>
                <div className="hero-hints">
                  <span className="hero-hint">
                    {workflow.selectedRole ? t("nextBatch", { role: workflow.selectedRole }) : t("autoRead")}
                  </span>
                  {workflow.selectedRole && (
                    <button
                      type="button"
                      className="link"
                      disabled={workflow.inputBusy}
                      onClick={() => workflow.setSelectedRole(undefined)}
                    >
                      {t("autoDetect")}
                    </button>
                  )}
                </div>
              </div>
            </div>
            <div className="stack hero-stack">
              {workflow.errorMessage && (
                <Alert tone="stop" title={t("operationFailed")}>
                  {workflow.errorMessage}
                </Alert>
              )}
              {pathEntry}
              <FrameInventory workflow={workflow} t={t} tab={inventoryTab} setTab={setInventoryTab} />
            </div>
          </div>
        </div>
      ) : (
        <div className="scroll">
          <div className="stack">
            <h1
              id="import-title"
              className="visually-hidden"
              style={{ position: "absolute", width: 1, height: 1, overflow: "hidden", clip: "rect(0 0 0 0)" }}
            >
              {workflow.projectName}
            </h1>
            {workflow.errorMessage && (
              <Alert tone="stop" title={t("operationFailed")}>
                {workflow.errorMessage}
              </Alert>
            )}
            <div className="row-actions">
              <span className="status-note">
                {workflow.inventoryBusy
                  ? t("readingMetadata")
                  : workflow.selectedRole
                    ? t("nextBatch", { role: workflow.selectedRole })
                    : t("autoRead")}
              </span>
              {workflow.selectedRole && (
                <button
                  type="button"
                  className="link"
                  disabled={workflow.inputBusy}
                  onClick={() => workflow.setSelectedRole(undefined)}
                >
                  {t("autoDetect")}
                </button>
              )}
            </div>
            <FrameInventory workflow={workflow} t={t} tab={inventoryTab} setTab={setInventoryTab} />
            {pathEntry}
            <CalibrationGroups workflow={workflow} t={t} />
            <MetadataConfirmations workflow={workflow} t={t} />
            <LaunchSettings
              mode="import"
              workflow={workflow}
              outputPathText={outputPathText}
              setOutputPathText={setOutputPathText}
              t={t}
            />
          </div>
        </div>
      )}
      <LaunchBar mode="import" workflow={workflow} lightCount={lightCount} t={t} />
    </section>
  );
}
