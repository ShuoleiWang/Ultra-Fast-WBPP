import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  window.localStorage.clear();
});

beforeEach(() => window.localStorage.clear());

describe("Ultra-Fast WBPP browser development mode", () => {
  it("exposes the guided workflow but disables native path pickers", async () => {
    render(<App />);
    const navigation = screen.getByRole("navigation", { name: "Workflow" });
    for (const label of ["Import", "Process", "Result"]) {
      expect(within(navigation).getByText(label)).toBeInTheDocument();
    }
    for (const role of ["All", "Light", "Flat", "Dark", "Bias"]) {
      expect(screen.getByRole("tab", { name: new RegExp(`^${role} `) })).toBeInTheDocument();
    }
    await userEvent.click(screen.getByText("Manual type hint / source inventory"));
    for (const role of ["Light", "Flat", "Dark", "Bias", "Master Flat", "Master Dark", "Master Bias"]) {
      expect(screen.getByRole("button", { name: new RegExp(`Add ${role}$`) })).toBeDisabled();
    }
    expect(screen.getByRole("button", { name: /Choose files$/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Choose folder$/ })).toBeDisabled();
    expect((await screen.findAllByText("Browser demo")).length).toBeGreaterThan(0);
    expect(screen.getByText("Legal")).toBeInTheDocument();
    expect(screen.queryByText("WCS SOLVED")).not.toBeInTheDocument();
  });

  it("requires an explicit demo action before any simulated progress", async () => {
    render(<App />);
    expect(screen.queryByText("DEMO")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Load the clearly labelled browser demo" }));
    await userEvent.click(screen.getByRole("button", { name: "Confirm type" }));
    await userEvent.click(screen.getByRole("button", { name: /Screen 370 Lights first/ }));
    expect(screen.getByRole("heading", { name: "Review groups and real frame evidence" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Target × filter matrix" })).toBeInTheDocument();
    expect(screen.getByText("347")).toBeInTheDocument();
    expect(screen.getByText("23")).toBeInTheDocument();
  });

  it("keeps browser completion unmistakably separate from solved native output", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Load the clearly labelled browser demo" }));
    await userEvent.click(screen.getByRole("button", { name: "Confirm type" }));
    await userEvent.click(screen.getByRole("button", { name: /Screen 370 Lights first/ }));
    expect(screen.getByRole("button", { name: /Start processing/ })).toBeEnabled();
    await userEvent.click(screen.getByRole("button", { name: /Start processing/ }));
    await act(async () => { vi.advanceTimersByTime(6_000); });
    expect(await screen.findByText("DEMO RESULT")).toBeInTheDocument();
    expect(screen.queryByText("WCS SOLVED")).not.toBeInTheDocument();
    expect(screen.getByText(/No master or astrometric solution was created/)).toBeInTheDocument();
    expect(screen.getByText("DEMO_result.png")).toBeInTheDocument();
  });

  it("can cancel the explicit demo without claiming native work", async () => {
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Load the clearly labelled browser demo" }));
    await userEvent.click(screen.getByRole("button", { name: "Confirm type" }));
    await userEvent.click(screen.getByRole("button", { name: /Screen 370 Lights first/ }));
    await userEvent.click(screen.getByRole("button", { name: /Start processing/ }));
    expect(await screen.findByText("DEMO")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Cancel safely" }));
    expect(await screen.findByRole("heading", { name: "Run safely cancelled" })).toBeInTheDocument();
  });

  it("defaults to English and persists a Simplified Chinese choice across remount", async () => {
    const first = render(<App />);
    expect(screen.getByRole("heading", { name: "Drop in your N.I.N.A. folders" })).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Language" }), "zh-CN");
    expect(screen.getByRole("heading", { name: "拖入 N.I.N.A. 拍摄文件夹" })).toBeInTheDocument();
    first.unmount();
    render(<App />);
    expect(screen.getByRole("heading", { name: "拖入 N.I.N.A. 拍摄文件夹" })).toBeInTheDocument();
  });

  it("falls back to English when localStorage is unavailable", () => {
    vi.spyOn(window.localStorage, "getItem").mockImplementation(() => { throw new DOMException("blocked", "SecurityError"); });
    vi.spyOn(window.localStorage, "setItem").mockImplementation(() => { throw new DOMException("blocked", "SecurityError"); });
    expect(() => render(<App />)).not.toThrow();
    expect(screen.getByRole("heading", { name: "Drop in your N.I.N.A. folders" })).toBeInTheDocument();
  });
});
