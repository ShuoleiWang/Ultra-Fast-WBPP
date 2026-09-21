import { describe, expect, it } from "vitest";
import { groupAssets, temperatureSummary } from "./FrameInventory";
import type { InspectedAsset } from "./types";

const light = (path: string, overrides: Partial<InspectedAsset> = {}): InspectedAsset => ({
  path, role: "LIGHT", width: 6252, height: 4176, channels: 1, filter: "L", target: "NGC 6822", camera: "QHY268M",
  exposureSeconds: 300, temperatureCelsius: -10, gain: 56, offset: 30, binning: [1, 1], cfaPattern: "NONE", readoutMode: "Mode 1",
  sourceSha256: null, observedAt: "2026-08-17T21:43:35Z", ...overrides,
});

describe("frame inventory grouping", () => {
  it("keeps one row per channel across nights, sensor temperatures and exposures", () => {
    const groups = groupAssets([
      light("/n1/L-1.fits", { temperatureCelsius: -9.9 }),
      light("/n1/L-2.fits", { temperatureCelsius: -10 }),
      light("/n2/L-3.fits", { observedAt: "2026-08-20T22:00:00Z", temperatureCelsius: -10 }),
      light("/n3/L-4.fits", { observedAt: "2026-09-09T23:46:06Z", temperatureCelsius: 0, exposureSeconds: 600 }),
      light("/n3/B-1.fits", { filter: "B", observedAt: "2026-09-06T21:11:50Z", temperatureCelsius: -8.1 }),
      light("/n3/B-2.fits", { filter: "B", observedAt: "2026-09-06T21:16:52Z", temperatureCelsius: -8.2 }),
      light("/flat/B.fits", { role: "MASTER_FLAT", filter: "B", target: "UNKNOWN", exposureSeconds: 1, observedAt: undefined }),
      light("/bin2/L.fits", { binning: [2, 2], width: 3126, height: 2088 }),
    ]);
    expect(groups.map((group) => [group.asset.role, group.asset.filter, group.paths.length])).toEqual([
      ["LIGHT", "B", 2], ["LIGHT", "L", 4], ["LIGHT", "L", 1], ["MASTER_FLAT", "B", 1],
    ]);
    const l = groups[1];
    expect(l.exposures).toEqual([300, 600]);
    expect(l.temperatures).toEqual([-10, -9.9, 0]);
    expect(l.dates).toEqual(["2026-08-17", "2026-08-20", "2026-09-09"]);
    expect(temperatureSummary(l.temperatures, "—")).toBe("-10.0 … 0.0 °C");
    expect(temperatureSummary(groups[0].temperatures, "—")).toBe("-8.2 … -8.1 °C");
    expect(temperatureSummary([-10, -10.04], "—")).toBe("-10.0 °C");
    expect(temperatureSummary([], "unrecorded")).toBe("unrecorded");
    // Different sensor geometry (binning) is a different channel row; the master flat stays separate.
    expect(groups[2].asset.binning).toEqual([2, 2]);
    expect(groups[3].dates).toEqual([]);
  });
});
