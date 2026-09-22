import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import { demoBlinkManifest } from "../demoAutopilot";
import type { BlinkMeasureResponse, InspectedAsset } from "../types";

/** jsdom has no image decoder or canvas; tests explicitly deliver decode/paint.
 * Real WebKit rendering is verified separately in the packaged application. */
export function mockPreviewCanvas() {
  const context = { setTransform: vi.fn(), fillRect: vi.fn(), translate: vi.fn(), scale: vi.fn(), drawImage: vi.fn() };
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(context as unknown as CanvasRenderingContext2D);
  return context;
}
export async function paintStage() {
  for (const image of screen.getAllByTestId("blink-stage-decoder")) {
    Object.defineProperties(image, { complete: { configurable: true, value: true }, naturalWidth: { configurable: true, value: 782 }, naturalHeight: { configurable: true, value: 522 } });
    fireEvent.load(image);
  }
  await act(async () => { if (vi.isFakeTimers()) vi.advanceTimersByTime(20); else await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve())); });
}
export async function reviewChannel(confirmName: RegExp) {
  const tiles = Array.from(document.querySelectorAll<HTMLButtonElement>(".blink-tile"));
  for (const tile of tiles) { fireEvent.click(tile); await paintStage(); }
  const confirm = screen.getByRole("button", { name: confirmName });
  await waitFor(() => { if (confirm.hasAttribute("disabled")) throw new Error("channel is not reviewed"); });
  fireEvent.click(confirm);
}
export function inventoryBlinkManifest(assets: InspectedAsset[]): BlinkMeasureResponse {
  const template = demoBlinkManifest();
  const lights = assets.filter((asset) => asset.role === "LIGHT");
  const frames = lights.map((asset, index) => ({
    ...template.frames[0], path: asset.path, name: asset.path.split("/").pop()!, filter: asset.filter, target: asset.target,
    index, channelId: `${asset.target}-${asset.filter}`, sourceSha256: `sha256:${(index + 1).toString(16).padStart(64, "0")}`,
    reference: false, flags: [], defaultDecision: "KEEP" as const,
  }));
  const channels = [...new Set(frames.map((frame) => frame.channelId))].map((channelId) => {
    const members = frames.filter((frame) => frame.channelId === channelId);
    members[0].reference = true;
    return { ...template.channels[0], channelId, target: members[0].target, filter: members[0].filter, frameCount: members.length,
      reference: { ...template.channels[0].reference!, index: members[0].index, sourceSha256: members[0].sourceSha256 },
      nights: [{ ...template.channels[0].nights[0], night: members[0].night, frameCount: members.length }],
    };
  });
  return { ...template, frames, channels };
}
