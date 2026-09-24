import { describe, expect, it } from "vitest";
import { startBlockers, type StartReadiness } from "./model";

const ready: StartReadiness = {
  engineAvailable: true,
  calibrationReady: true,
  allRequiredConfirmed: true,
  panelCount: 1,
  insufficientPanels: [],
  solverSetupReady: true,
  outputChosen: true,
  masterOverridesReady: true,
  cfaBlocked: false,
};

describe("startBlockers", () => {
  it("is empty exactly when every native-run condition holds", () => {
    expect(startBlockers(ready)).toEqual([]);
  });

  it("lists every unmet condition in launch-bar order", () => {
    const cell = { panelId: "p1", target: "NGC 7331", filter: "L", lightCount: 3, admittedCount: 1 };
    const blockers = startBlockers({
      ...ready,
      engineAvailable: false,
      engineUnavailableReason: "no engine",
      outputChosen: false,
      insufficientPanels: [cell],
      cfaBlocked: true,
    });
    expect(blockers).toEqual([
      { kind: "engine", reason: "no engine" },
      { kind: "panel", cell },
      { kind: "output" },
      { kind: "cfa" },
    ]);
  });

  it("asks for Lights when nothing was imported", () => {
    expect(startBlockers({ ...ready, panelCount: 0 })).toEqual([{ kind: "lights" }]);
  });
});
