import { describe, expect, it } from "vitest";
import type { BlinkMeasureResponse, InspectedAsset } from "../types";
import { blinkSessionCovers, inventoryKeyOf, plainPathSpelling, startBlockers, type StartReadiness } from "./model";

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

describe("blinkSessionCovers", () => {
  const sha = (seed: string) => `sha256:${seed.repeat(64)}`;
  const session = (frames: Array<{ path: string; sourceSha256: string }>, lightPaths: string[]) => ({
    manifest: { manifestSha256: sha("f"), frames } as unknown as BlinkMeasureResponse,
    sessionDirectory: "/session",
    inventoryKey: inventoryKeyOf(lightPaths),
    demo: false,
  });
  const light = (path: string, sourceSha256?: string) => ({ path, role: "LIGHT", sourceSha256 }) as InspectedAsset;

  it("covers the Lights by their paths", () => {
    const paths = ["/a/1.fits", "/a/2.fits"];
    const frames = paths.map((path, index) => ({ path, sourceSha256: sha(String(index)) }));
    expect(blinkSessionCovers(session(frames, paths), paths)).toBe(true);
    expect(blinkSessionCovers(session(frames.slice(0, 1), paths), paths)).toBe(false);
  });

  it("matches a Windows path the engine kept in its verbatim form", () => {
    const paths = ["D:\\Astro\\L_0001.fits", "\\\\nas\\share\\L_0002.fits"];
    const frames = [
      { path: "\\\\?\\D:\\Astro\\L_0001.fits", sourceSha256: sha("1") },
      { path: "\\\\?\\UNC\\nas\\share\\L_0002.fits", sourceSha256: sha("2") },
    ];
    expect(blinkSessionCovers(session(frames, paths), paths)).toBe(true);
  });

  it("matches by content digest when the inventory carries one, whatever the path spelling", () => {
    const paths = ["C:\\Lights\\a.fits"];
    const frames = [{ path: "\\\\?\\c:\\lights\\a.fits", sourceSha256: sha("a") }];
    expect(blinkSessionCovers(session(frames, paths), paths, [light(paths[0], sha("a"))])).toBe(true);
    // Same path, other content: the session no longer describes this Light.
    const same = [{ path: paths[0], sourceSha256: sha("a") }];
    expect(blinkSessionCovers(session(same, paths), paths, [light(paths[0], sha("b"))])).toBe(false);
  });
});

describe("plainPathSpelling", () => {
  it("removes the verbatim prefix of drive and UNC paths only", () => {
    expect(plainPathSpelling("\\\\?\\D:\\x.fits")).toBe("D:\\x.fits");
    expect(plainPathSpelling("\\\\?\\UNC\\srv\\share\\x.fits")).toBe("\\\\srv\\share\\x.fits");
    expect(plainPathSpelling("\\\\?\\Volume{1234}\\x.fits")).toBe("\\\\?\\Volume{1234}\\x.fits");
    expect(plainPathSpelling("/Volumes/data/x.fits")).toBe("/Volumes/data/x.fits");
  });
});
