import { useMemo, useState } from "react";
import type { Translator } from "./i18n";
import type { FrameRole, InspectedAsset, RawFrameRole } from "./types";
import { FrameTile } from "./views";
import type { useWorkflow } from "./useWorkflow";

type Workflow = ReturnType<typeof useWorkflow>;
type Tab = RawFrameRole | "ALL";
const ROLES: RawFrameRole[] = ["LIGHT", "FLAT", "DARK", "BIAS"];
const roleFamily = (role: FrameRole) => role.replace("MASTER_", "") as RawFrameRole;
const label = (role: string) => role[0] + role.slice(1).toLowerCase();
const value = (item: string | number | null | undefined) =>
  item === undefined || item === null || item === "" || item === "UNKNOWN" ? "—" : String(item);
const filename = (path: string) => path.split(/[\\/]/).pop() ?? path;

interface AssetGroup {
  key: string;
  asset: InspectedAsset;
  paths: string[];
  exposures: number[];
  temperatures: number[];
  gains: string[];
  offsets: string[];
  readoutModes: string[];
  dates: string[];
}

const sortedUnique = <T,>(items: T[]) =>
  [...new Set(items)].sort((a, b) =>
    typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b)),
  );
/** "300" for one value, "60 / 300" for a few, "60 … 300" for many. */
const summarize = (values: (string | number)[], separator = " / ") =>
  values.length <= 3 ? values.map(value).join(separator) : `${value(values[0])} … ${value(values[values.length - 1])}`;
export const temperatureSummary = (temperatures: number[], unrecorded: string) => {
  if (!temperatures.length) return unrecorded;
  const low = temperatures[0],
    high = temperatures[temperatures.length - 1];
  return Math.abs(high - low) < 0.05 ? `${low.toFixed(1)} °C` : `${low.toFixed(1)} … ${high.toFixed(1)} °C`;
};
const dateSummary = (dates: string[]) =>
  dates.length <= 2 ? dates.join(", ") : `${dates[0]} … ${dates[dates.length - 1]}`;

/** One row per channel: the frames of a role, target and filter on the same camera geometry.
 *  Exposure, temperature, gain/offset and capture dates vary within a channel over a
 *  multi-night campaign and are shown as ranges instead of splitting the channel. */
export function groupAssets(assets: InspectedAsset[]): AssetGroup[] {
  const grouped = new Map<string, AssetGroup>();
  for (const asset of assets) {
    const key = JSON.stringify([
      asset.role,
      asset.target,
      asset.filter,
      asset.camera,
      asset.binning,
      asset.width,
      asset.height,
      asset.cfaPattern,
    ]);
    const group = grouped.get(key) ?? {
      key,
      asset,
      paths: [],
      exposures: [],
      temperatures: [],
      gains: [],
      offsets: [],
      readoutModes: [],
      dates: [],
    };
    group.paths.push(asset.path);
    if (typeof asset.exposureSeconds === "number") group.exposures.push(asset.exposureSeconds);
    if (typeof asset.temperatureCelsius === "number") group.temperatures.push(asset.temperatureCelsius);
    group.gains.push(value(asset.gain));
    group.offsets.push(value(asset.offset));
    group.readoutModes.push(value(asset.readoutMode));
    if (asset.observedAt) group.dates.push(asset.observedAt.slice(0, 10));
    grouped.set(key, group);
  }
  return [...grouped.values()]
    .map((group) => ({
      ...group,
      exposures: sortedUnique(group.exposures),
      temperatures: sortedUnique(group.temperatures),
      gains: sortedUnique(group.gains),
      offsets: sortedUnique(group.offsets),
      readoutModes: sortedUnique(group.readoutModes),
      dates: sortedUnique(group.dates),
    }))
    .sort(
      (a, b) =>
        ROLES.indexOf(roleFamily(a.asset.role)) - ROLES.indexOf(roleFamily(b.asset.role)) ||
        a.asset.target.localeCompare(b.asset.target) ||
        a.asset.filter.localeCompare(b.asset.filter),
    );
}

/** The imported frames grouped by acquisition profile, with the manual type hints and confirmations. */
export function FrameInventory({
  workflow,
  t,
  tab: controlledTab,
  setTab: setControlledTab,
  overridesOnly = false,
}: {
  workflow: Workflow;
  t: Translator;
  tab?: Tab;
  setTab?: (tab: Tab) => void;
  overridesOnly?: boolean;
}) {
  const [localTab, setLocalTab] = useState<Tab>("ALL");
  const tab = controlledTab ?? localTab;
  const setTab = setControlledTab ?? setLocalTab;
  const [expanded, setExpanded] = useState<string>();
  const groups = useMemo(() => groupAssets(workflow.assets), [workflow.assets]);
  const visible = groups.filter((group) => tab === "ALL" || roleFamily(group.asset.role) === tab);
  const overrides = (
    <>
      <details className="disclosure inventory-overrides">
        <summary>{t("manualClassification")}</summary>
        <div className="details-content">
          <div className="source-rows">
            {workflow.sources.map((source) => (
              <div className="source-row" key={source.role}>
                <button
                  type="button"
                  className="role-selector"
                  aria-label={t("nextBatch", { role: source.label })}
                  aria-pressed={workflow.selectedRole === source.role}
                  disabled={workflow.inputBusy}
                  onClick={() =>
                    workflow.setSelectedRole(workflow.selectedRole === source.role ? undefined : source.role)
                  }
                >
                  {source.label}
                </button>
                <span>
                  {source.fileCount} {t("frames")}
                </span>
                {source.detected ? (
                  <span className="source-row-status">{source.confirmed ? t("confirmed") : t("pending")}</span>
                ) : (
                  <button
                    type="button"
                    className="link"
                    disabled={!workflow.nativeRuntime || workflow.inputBusy}
                    onClick={() => {
                      workflow.setSelectedRole(source.role);
                      void workflow.pickFiles(source.role);
                    }}
                  >
                    {t("addRole", { role: source.label })}
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      </details>
      {workflow.sources
        .filter((source) => source.detected && !source.confirmed)
        .map((source) => (
          <div className="inventory-confirmation" key={source.role}>
            <span>{t("classificationPending", { role: source.label })}</span>
            <button
              type="button"
              className="btn small"
              disabled={workflow.inputBusy}
              onClick={() => workflow.confirmRole(source.role)}
            >
              {t("confirmType")}
            </button>
          </div>
        ))}
    </>
  );
  if (overridesOnly)
    return (
      <section className="panel frame-inventory" aria-label={t("frameInventory")}>
        {overrides}
      </section>
    );
  return (
    <section className="panel frame-inventory" aria-label={t("frameInventory")}>
      <div className="panel-heading">
        <div className="seg" role="tablist" aria-label={t("frameTypes")}>
          {(["ALL", ...ROLES] as const).map((role) => (
            <button
              key={role}
              type="button"
              role="tab"
              id={`tab-${role}`}
              aria-controls="frame-table"
              aria-selected={tab === role}
              onClick={() => setTab(role)}
            >
              {role === "ALL" ? t("allFrames") : label(role)}
              <span className="count">
                {workflow.sources
                  .filter((source) => role === "ALL" || roleFamily(source.role) === role)
                  .reduce((sum, source) => sum + source.fileCount, 0)}
              </span>
            </button>
          ))}
        </div>
        <small>
          {t("inventoryGroups", { count: visible.length })} · {t("metadataGrouping")}
        </small>
      </div>
      <div className="tbl-wrap" id="frame-table" role="tabpanel" aria-labelledby={`tab-${tab}`}>
        {visible.length > 0 && (
          <table className="tbl inventory-table">
            <thead>
              <tr>
                <th className="thumb-cell" />
                <th>{t("frameType")}</th>
                <th>
                  {t("targetLabel")} / {t("filterLabel")}
                </th>
                <th className="r">{t("exposureLabel")}</th>
                <th>
                  {t("cameraLabel")} / {t("temperatureLabel")}
                </th>
                <th>{t("inventoryProfile")}</th>
                <th>{t("captureDates")}</th>
                <th className="r">{t("frames")}</th>
              </tr>
            </thead>
            <tbody>
              {visible.map(({ key, asset, paths, exposures, temperatures, gains, offsets, readoutModes, dates }) => (
                <tr key={key} aria-selected={expanded === key}>
                  <td className="thumb-cell">
                    <FrameTile
                      filter={
                        roleFamily(asset.role) === "LIGHT" || roleFamily(asset.role) === "FLAT"
                          ? asset.filter
                          : roleFamily(asset.role)[0]
                      }
                    />
                  </td>
                  <td>
                    <span className={`kind kind-${roleFamily(asset.role).toLowerCase()}`}>
                      {label(roleFamily(asset.role))}
                    </span>
                    <small>{asset.role.startsWith("MASTER_") ? t("existingMaster") : t("rawInput")}</small>
                  </td>
                  <td>
                    <strong>{value(asset.target)}</strong>
                    <small>{value(asset.filter)}</small>
                  </td>
                  <td className="r tnum">{exposures.length ? summarize(exposures) : "—"} s</td>
                  <td>
                    <strong>{value(asset.camera)}</strong>
                    <small>{temperatureSummary(temperatures, t("temperatureUnrecorded"))}</small>
                  </td>
                  <td>
                    <span className="tnum">
                      {asset.binning.join("×")} · {asset.width}×{asset.height}
                    </span>
                    <small>
                      G {summarize(gains)} · O {summarize(offsets)} · {summarize(readoutModes)}
                    </small>
                  </td>
                  <td className="tnum">
                    {dates.length ? dateSummary(dates) : "—"}
                    {dates.length > 1 && <small>{t("nightCount", { count: dates.length })}</small>}
                  </td>
                  <td className="r">
                    <button
                      type="button"
                      className="btn small"
                      aria-expanded={expanded === key}
                      aria-label={t("showGroupFiles", {
                        count: paths.length,
                        type: label(roleFamily(asset.role)),
                        target: value(asset.target),
                        filter: value(asset.filter),
                      })}
                      onClick={() => setExpanded(expanded === key ? undefined : key)}
                    >
                      {paths.length}
                      <span className="chev">{expanded === key ? "▾" : "▸"}</span>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {!visible.length && (
          <div className="table-empty">
            <strong>{t("inventoryEmpty")}</strong>
            <p>{t("inventoryEmptyHint")}</p>
          </div>
        )}
      </div>
      {expanded && visible.some((group) => group.key === expanded) && (
        <div className="inventory-files">
          {visible
            .find((group) => group.key === expanded)!
            .paths.slice(0, 100)
            .map((path) => (
              <div key={path}>
                <strong>{filename(path)}</strong>
                <span title={path}>{path}</span>
              </div>
            ))}
          {visible.find((group) => group.key === expanded)!.paths.length > 100 && <p>{t("fileListLimited")}</p>}
        </div>
      )}
      {overrides}
    </section>
  );
}

/** Calibration and screening status for the inspector pane. */
export function InputChecks({ workflow, t }: { workflow: Workflow; t: Translator }) {
  const count = (role: RawFrameRole, master: boolean) =>
    workflow.sources.find((source) => source.role === (master ? `MASTER_${role}` : role))?.fileCount ?? 0;
  const pendingMasters = workflow.masterOverrides.filter((item) => !item.confirmed).length;
  const importedRoles = new Set(ROLES.filter((role) => count(role, false) + count(role, true) > 0));
  return (
    <div className="input-checks" aria-label={t("inputChecks")}>
      <section className="ins-section">
        <h3>{t("calibrationCheck")}</h3>
        <dl className="calibration-presence">
          {(["FLAT", "DARK", "BIAS"] as const).map((role) => {
            const raw = count(role, false);
            const master = count(role, true);
            return (
              <div key={role}>
                <dt>{label(role)}</dt>
                <dd className={raw + master ? "check-present" : "check-missing"}>
                  {raw + master ? `${raw} ${t("rawInput")} · ${master} Master` : t("notImported")}
                </dd>
              </div>
            );
          })}
        </dl>
        <div className={`status-note ${workflow.calibrationReady ? "ok" : ""}`} role="status">
          {workflow.calibrationBusy
            ? t("calibrationChecking")
            : workflow.calibrationInspection
              ? workflow.calibrationReady
                ? t("calibrationMatched")
                : t("calibrationBlocked")
              : t("calibrationUnverified")}
        </div>
        {pendingMasters > 0 && (
          <div className="calibration-confirmation-guide">
            <p className="status-note">{t("calibrationConfirmationPending", { masters: pendingMasters })}</p>
            <a className="link" href="#metadata-confirmations">
              {t("openCalibrationConfirmation")}
            </a>
          </div>
        )}
        {workflow.calibrationError && (
          <p className="error-text" role="alert">
            {t("calibrationReadError", { error: workflow.calibrationError })}
          </p>
        )}
        {workflow.calibrationInspection && (
          <div className="calibration-issues">
            {compactCalibrationIssues(workflow.calibrationInspection.issues).map((issue, index) => (
              <details
                key={`${issue.code}-${index}`}
                className={`disclosure calibration-issue issue-${issue.severity.toLowerCase()}`}
                open={issue.severity === "ERROR"}
              >
                <summary>
                  <span>{calibrationIssueLabel(issue.code, t, importedRoles)}</span>
                </summary>
                <p>{issue.code === "CFA_CONFIRMATION_REQUIRED" ? t("calibrationCfaReasonBody") : issue.message}</p>
                {issue.paths.length > 0 && (
                  <small title={issue.paths.join("\n")}>
                    {t("affectedFiles")}: {issue.paths.slice(0, 3).map(filename).join(", ")}
                    {issue.paths.length > 3 ? ` (+${issue.paths.length - 3})` : ""}
                  </small>
                )}
              </details>
            ))}
          </div>
        )}
        <p className="status-note">
          {t("calibrationCheckHint")} {t("flatSessionWarning")}
        </p>
        {workflow.nativeRuntime && workflow.assets.some((asset) => asset.role === "LIGHT") && (
          <button
            type="button"
            className="link"
            disabled={workflow.calibrationBusy || workflow.inputBusy}
            onClick={workflow.recheckCalibration}
          >
            {t("calibrationRecheck")}
          </button>
        )}
      </section>
      <section className="ins-section qc-check">
        <h3>{t("automaticQuality")}</h3>
        {workflow.qualityBusy ? (
          <p className="status-note" role="status">
            {t("qualityElapsed", { count: count("LIGHT", false), seconds: workflow.qualityElapsedSeconds })}
          </p>
        ) : workflow.qualityInspection ? (
          <dl className="quality-counts">
            <div>
              <dt>{t("qcPassed")}</dt>
              <dd>{workflow.gate.pass}</dd>
            </div>
            <div>
              <dt>{t("qcReview")}</dt>
              <dd>{workflow.gate.review}</dd>
            </div>
            <div>
              <dt>{t("qcRejected")}</dt>
              <dd>{workflow.gate.hardFail}</dd>
            </div>
          </dl>
        ) : (
          <p className="status-note">{t("qualityNotMeasured")}</p>
        )}
        <p className="status-note">
          {t("qualityChecksHint")} {t("qualityExclusionHint")}
        </p>
        {workflow.qualityReady && !workflow.inputBusy && (
          <div className="row-actions">
            <button type="button" className="link" onClick={() => void workflow.runInspection()}>
              {t("viewQualityResults")}
            </button>
            <button type="button" className="link" onClick={() => void workflow.runInspection(true)}>
              {t("rerunQuality")}
            </button>
          </div>
        )}
      </section>
    </div>
  );
}

// Keep repeated per-file diagnostics compact while retaining every affected path.
function compactCalibrationIssues(issues: NonNullable<Workflow["calibrationInspection"]>["issues"]) {
  const grouped = new Map<string, (typeof issues)[number]>();
  for (const issue of issues) {
    const key = JSON.stringify([issue.code, issue.severity, issue.message]);
    const previous = grouped.get(key);
    grouped.set(
      key,
      previous
        ? {
            ...previous,
            paths: [...new Set([...previous.paths, ...issue.paths])],
            lightGroups: [...new Set([...previous.lightGroups, ...issue.lightGroups])],
          }
        : { ...issue },
    );
  }
  return [...grouped.values()];
}

function calibrationIssueLabel(code: string, t: Translator, importedRoles: Set<RawFrameRole>): string {
  const role = code.split("_")[0];
  if (/^(FLAT|DARK|BIAS)_MATCH_MISSING$/.test(code))
    return t(
      importedRoles.has(role as RawFrameRole) ? "calibrationImportedUnmatchedReason" : "calibrationMissingReason",
      { role },
    );
  if (/^(FLAT|DARK|BIAS)_MATCH_AMBIGUOUS$/.test(code)) return t("calibrationAmbiguousReason", { role });
  if (code === "CFA_CONFIRMATION_REQUIRED") return t("calibrationCfaReason");
  if (code === "DARK_TEMPERATURE_UNRECORDED") return t("darkTemperatureUnrecorded");
  if (code === "DARK_EXPOSURE_MISMATCH") return t("calibrationDarkParametersReason");
  if (code === "CALIBRATION_PROFILE_UNSUPPORTED") return t("calibrationProfileReason");
  if (["DARK_BIAS_SEMANTICS_UNDECLARED", "MASTER_DARK_BIAS_SEMANTICS_REQUIRED"].includes(code))
    return t("calibrationBiasReason");
  if (code === "CAPTURE_SESSION_NOT_VERIFIED") return t("calibrationSessionReason");
  return t("calibrationReasonGeneric", { code });
}

export function CalibrationGroups({ workflow, t }: { workflow: Workflow; t: Translator }) {
  if (!workflow.calibrationInspection?.groups.length) return null;
  return (
    <section className="panel calibration-group-section" aria-label={t("calibrationGroups")}>
      <div className="panel-heading">
        <span>{t("calibrationGroups")}</span>
        <small>{t("captureDates")}</small>
      </div>
      <div className="tbl-wrap">
        <table className="tbl compact calibration-groups">
          <thead>
            <tr>
              <th>
                {t("targetLabel")} / {t("filterLabel")}
              </th>
              <th>{t("captureDates")}</th>
              <th>Flat</th>
              <th>Dark</th>
              <th>Bias</th>
              <th>{t("calibrationCheck")}</th>
            </tr>
          </thead>
          <tbody>
            {workflow.calibrationInspection.groups.map((group) => (
              <tr key={group.groupId}>
                <th scope="row">
                  {group.target} · {group.filter}
                  <small>{group.lightCount} Light</small>
                </th>
                <td>{group.observedDates.join(", ") || "—"}</td>
                {(["FLAT", "DARK", "BIAS"] as const).map((role) => (
                  <td key={role}>
                    <span>
                      {group.matches[role].rawCount} {t("rawInput")}
                    </span>
                    <small>{group.matches[role].masterCount} Master</small>
                  </td>
                ))}
                <td className={group.status === "READY" ? "check-present" : "check-missing"}>
                  {group.status === "READY" ? t("calibrationMatched") : t("calibrationBlocked")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
