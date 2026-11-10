import { describe, expect, it } from "vitest";
import type { MessageKey } from "../i18n";
import { explainFailure, filterName } from "./common";

const t = (key: MessageKey) => key;

describe("filterName", () => {
  it("shows the engine's NOFILTER as no filter and keeps every other name", () => {
    expect(filterName("NOFILTER", t)).toBe("noFilter");
    expect(filterName("L", t)).toBe("L");
    expect(filterName("", t)).toBe("");
  });
});

describe("explainFailure", () => {
  it("takes the code from the event and strips it from the text", () => {
    expect(explainFailure("NO_FLATS: raw Flats are required", t, "NO_FLATS")).toEqual({
      code: "NO_FLATS",
      hint: "failureHint_NO_FLATS",
      detail: "raw Flats are required",
    });
  });

  it("reads a leading code from a rejected call", () => {
    expect(explainFailure("Error: OUTPUT_PATH_TOO_LONG: 290 characters", t)).toEqual({
      code: "OUTPUT_PATH_TOO_LONG",
      hint: "failureHint_OUTPUT_PATH_TOO_LONG",
      detail: "290 characters",
    });
  });

  it("recognises a full disk whatever the code", () => {
    expect(explainFailure("[Errno 28] No space left on device", t, "PROJECT_EXECUTION_FAILED").hint).toBe(
      "failureHint_DISK_FULL",
    );
  });

  it("recognises the controller's import refusals", () => {
    expect(explainFailure("desktop import does not accept role MASTER_LIGHT: /x/L.fits", t).hint).toBe(
      "importRefusedRole",
    );
    expect(explainFailure("Error: frame is not ready: /x/a.fits (UNREADABLE)", t).hint).toBe("importRefusedNotReady");
  });

  it("keeps the text of an unknown failure", () => {
    expect(explainFailure("SOMETHING_NEW: details", t)).toEqual({
      code: "SOMETHING_NEW",
      hint: undefined,
      detail: "details",
    });
    expect(explainFailure("plain words", t)).toEqual({ code: undefined, hint: undefined, detail: "plain words" });
  });
});
