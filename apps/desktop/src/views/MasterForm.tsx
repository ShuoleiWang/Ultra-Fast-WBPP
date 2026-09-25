import type { Translator } from "../i18n";
import type { MasterMetadataOverride, SolverBackendStatus } from "../types";
import { basename, optionalNumber } from "./common";

export function MasterForm({
  item,
  update,
  confirm,
  reset,
  t,
}: {
  item: MasterMetadataOverride;
  update: (patch: Partial<MasterMetadataOverride>) => void;
  confirm: () => void;
  reset: () => void;
  t: Translator;
}) {
  const numeric = (field: "gain" | "offset" | "temperatureCelsius" | "exposureSeconds", label: string) => (
    <label>
      <span>{label}</span>
      <input
        type="number"
        step="any"
        inputMode="decimal"
        placeholder={t("notRecorded")}
        value={item[field] ?? ""}
        onChange={(event) => update({ [field]: optionalNumber(event.target.value) })}
      />
    </label>
  );
  const additiveMaster = item.role === "MASTER_BIAS" || item.role === "MASTER_DARK";
  const chooseNumericDomain = (value: string) => {
    if (value === "NORMALIZED_UNIT") update({ numericDomain: value, normalizedUnitScale: 1 });
    else if (value === "SENSOR_CODE") update({ numericDomain: value, normalizedUnitScale: 65535 });
    else update({ numericDomain: null, normalizedUnitScale: null });
  };
  return (
    <article className={`master-form ${item.confirmed ? "confirmed" : ""}`}>
      <header>
        <div>
          <small>{item.role}</small>
          <strong title={item.sourcePath}>{basename(item.sourcePath)}</strong>
        </div>
        <span>
          {item.confirmed
            ? item.needsMetadataOverride
              ? t("shaBound")
              : t("useFileMetadata")
            : t("needsConfirmation")}
        </span>
      </header>
      {!item.confirmed && (
        <p className="suggestion-note" role="status">
          {t("suggestion")}
        </p>
      )}
      <div className="metadata-fields">
        <label>
          <span>{t("cameraLabel")}</span>
          <input value={item.camera} onChange={(event) => update({ camera: event.target.value })} />
        </label>
        {numeric("gain", t("gainLabel"))}
        {numeric("offset", t("offsetLabel"))}
        <label>
          <span>{t("binningLabel")}</span>
          <span className="inline-inputs">
            <input
              type="number"
              min="1"
              inputMode="numeric"
              placeholder={t("notRecorded")}
              value={item.binning[0] ?? ""}
              onChange={(event) => update({ binning: [optionalNumber(event.target.value), item.binning[1]] })}
            />
            <input
              type="number"
              min="1"
              inputMode="numeric"
              placeholder={t("notRecorded")}
              value={item.binning[1] ?? ""}
              onChange={(event) => update({ binning: [item.binning[0], optionalNumber(event.target.value)] })}
            />
          </span>
        </label>
        <label>
          <span>{t("filterLabel")}</span>
          <input value={item.filter} onChange={(event) => update({ filter: event.target.value })} />
        </label>
        <label>
          <span>{t("cfaLabel")}</span>
          <input value={item.cfaPattern} onChange={(event) => update({ cfaPattern: event.target.value })} />
        </label>
        <label>
          <span>{t("readoutLabel")}</span>
          <input value={item.readoutMode} onChange={(event) => update({ readoutMode: event.target.value })} />
        </label>
        {numeric("temperatureCelsius", t("temperatureLabel"))}
        {numeric("exposureSeconds", t("exposureLabel"))}
      </div>
      {item.role === "MASTER_DARK" && (
        <fieldset className="bias-choice">
          <legend>{t("darkBiasQuestion")}</legend>
          <label>
            <input
              type="radio"
              name={`bias-${item.sourceSha256}`}
              checked={item.biasIncluded === true}
              onChange={() => update({ biasIncluded: true })}
            />
            {t("yesBias")}
          </label>
          <label>
            <input
              type="radio"
              name={`bias-${item.sourceSha256}`}
              checked={item.biasIncluded === false}
              onChange={() => update({ biasIncluded: false })}
            />
            {t("noBias")}
          </label>
          <label>
            <input
              type="radio"
              name={`bias-${item.sourceSha256}`}
              checked={item.biasIncluded === null}
              onChange={() => update({ biasIncluded: null })}
            />
            {t("unknownBlocked")}
          </label>
          <p>{t("darkBiasHelp")}</p>
        </fieldset>
      )}
      {additiveMaster && (
        <fieldset className="bias-choice unit-choice">
          <legend>{t("pixelUnits")}</legend>
          <label>
            <select
              aria-label={t("pixelUnits")}
              value={item.numericDomain ?? "AUTO"}
              onChange={(event) => chooseNumericDomain(event.target.value)}
            >
              <option value="AUTO">{t("pixelUnitsAuto")}</option>
              <option value="NORMALIZED_UNIT">{t("pixelUnitsNormalized")}</option>
              <option value="SENSOR_CODE">{t("pixelUnitsSensor")}</option>
            </select>
          </label>
          {item.numericDomain === "SENSOR_CODE" && (
            <label>
              <span>{t("pixelUnitsScale")}</span>
              <input
                type="number"
                step="any"
                className="unit-scale-input"
                inputMode="decimal"
                value={item.normalizedUnitScale ?? ""}
                onChange={(event) => update({ normalizedUnitScale: optionalNumber(event.target.value) })}
              />
            </label>
          )}
          <p>{t("pixelUnitsHelp")}</p>
        </fieldset>
      )}
      <div className="row-actions">
        <button type="button" className="btn small" disabled={item.confirmed} onClick={confirm}>
          {t("confirmAndBind")}
        </button>
        {item.needsMetadataOverride && (
          <button type="button" className="link" onClick={reset}>
            {t("restoreFileMetadata")}
          </button>
        )}
      </div>
    </article>
  );
}

/** One solver: ready means the strict final gate accepts its solutions; an installed but unverified one also shows the engine's reason. */
export function SolverRow({
  name,
  backend,
  ready,
  instruction,
  t,
}: {
  name: string;
  backend?: SolverBackendStatus;
  ready: boolean;
  instruction: string;
  t: Translator;
}) {
  const reason = !ready && backend?.executionReady && backend.reason?.trim() ? ` ${backend.reason.trim()}` : "";
  return (
    <article className="solver-row">
      <span className={`runtime-dot ${ready ? "online" : ""}`} />
      <div>
        <strong>
          {name} {ready ? t("executable") : t("notReady")}
        </strong>
        <p>
          {ready
            ? `${backend?.version ?? "?"} · ${backend?.metadata?.probe?.path ?? t("ready")}`
            : `${instruction}${reason}`}
        </p>
      </div>
    </article>
  );
}
