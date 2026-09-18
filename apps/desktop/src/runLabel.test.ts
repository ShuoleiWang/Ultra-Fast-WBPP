import { describe, expect, it } from "vitest";

import { runLabel } from "./useWorkflow";

const cells = (...targets: string[]) => targets.map((target) => ({ target }));

describe("runLabel", () => {
  it("names a single-target run after its target", () => {
    expect(runLabel(cells("NGC 7331", "NGC 7331"), "project")).toBe("NGC 7331");
  });

  it("keeps the words a mosaic's panels share and drops the panel word", () => {
    expect(runLabel(cells("NGC 7000 Panel 1", "NGC 7000 Panel 2", "ngc 7000 panel 3"), "project")).toBe("NGC 7000");
    expect(runLabel(cells("Sh2-155_P1", "Sh2-155_P2"), "project")).toBe("Sh2 155");
    expect(runLabel(cells("盾牌座 Panel 1", "盾牌座 Panel 2"), "project")).toBe("盾牌座");
  });

  it("joins unrelated targets and falls back to the project name", () => {
    expect(runLabel(cells("M81", "M82"), "project")).toBe("M81 + M82");
    expect(runLabel(cells("A", "B", "C", "D"), "project")).toBe("A + B + 2 more");
    expect(runLabel(cells("", "  "), "盾牌座 马赛克")).toBe("盾牌座 马赛克");
  });
});
