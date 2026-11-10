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
    await act(async () => {
      vi.advanceTimersByTime(6_000);
    });
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

  it("opens the browser demo in the language its address asks for", () => {
    window.history.replaceState({}, "", "/?lang=zh-CN");
    try {
      render(<App />);
      expect(screen.getByRole("heading", { name: "拖入 N.I.N.A. 拍摄文件夹" })).toBeInTheDocument();
    } finally {
      window.history.replaceState({}, "", "/");
    }
  });

  it("falls back to English when localStorage is unavailable", () => {
    vi.spyOn(window.localStorage, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    vi.spyOn(window.localStorage, "setItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    expect(() => render(<App />)).not.toThrow();
    expect(screen.getByRole("heading", { name: "Drop in your N.I.N.A. folders" })).toBeInTheDocument();
  });
});

describe("interface language without a stored choice", () => {
  const systemLanguage = (value: string) => vi.spyOn(window.navigator, "language", "get").mockReturnValue(value);
  const chinese = () => screen.queryByRole("heading", { name: "拖入 N.I.N.A. 拍摄文件夹" });
  const english = () => screen.queryByRole("heading", { name: "Drop in your N.I.N.A. folders" });

  it.each(["zh-CN", "zh-TW", "zh-Hant-HK", "zh"])("opens in Simplified Chinese on a %s system", (value) => {
    systemLanguage(value);
    render(<App />);
    expect(chinese()).toBeInTheDocument();
    expect(document.documentElement.lang).toBe("zh-CN");
  });

  it.each(["en-GB", "ja-JP", "zhx-Latn", ""])('opens in English on a "%s" system', (value) => {
    systemLanguage(value);
    render(<App />);
    expect(english()).toBeInTheDocument();
  });

  it("keeps a stored choice over the system language, in either direction", () => {
    systemLanguage("zh-CN");
    window.localStorage.setItem("ultra-fast-wbpp.language", "en");
    const first = render(<App />);
    expect(english()).toBeInTheDocument();
    first.unmount();
    systemLanguage("en-US");
    window.localStorage.setItem("ultra-fast-wbpp.language", "zh-CN");
    render(<App />);
    expect(chinese()).toBeInTheDocument();
  });

  it("lets the browser demo's address override a Chinese system language", () => {
    systemLanguage("zh-CN");
    window.history.replaceState({}, "", "/?lang=en");
    try {
      render(<App />);
      expect(english()).toBeInTheDocument();
    } finally {
      window.history.replaceState({}, "", "/");
    }
  });

  it("follows the system language when the stored choice cannot be read", () => {
    systemLanguage("zh-CN");
    vi.spyOn(window.localStorage, "getItem").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    render(<App />);
    expect(chinese()).toBeInTheDocument();
  });

  it("describes screening as automatic by default on the start page", () => {
    render(<App />);
    expect(
      screen.getByText(/Screening is automatic unless you choose to review the Lights in Blink yourself/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/review the Lights in Blink before processing/)).not.toBeInTheDocument();
  });

  it("walks the default flow in the demo: automatic screening checked, Start from the import page", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Load the clearly labelled browser demo" }));
    await userEvent.click(screen.getByRole("button", { name: "Confirm type" }));
    expect(screen.getByRole("checkbox", { name: "Automatic screening" })).toBeChecked();
    expect(screen.getByRole("button", { name: "Review in Blink instead" })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Start processing/ })).toHaveClass("primary");
    await userEvent.click(screen.getByRole("button", { name: /Start processing/ }));
    expect(await screen.findByText("DEMO")).toBeInTheDocument();
    await act(async () => {
      vi.advanceTimersByTime(6_000);
    });
    expect(await screen.findByText("DEMO RESULT")).toBeInTheDocument();
  });

  it("keeps the demo's screening choice in the demo and never stores it", async () => {
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Load the clearly labelled browser demo" }));
    await userEvent.click(screen.getByRole("button", { name: "Confirm type" }));
    const checkbox = screen.getByRole("checkbox", { name: "Automatic screening" });
    await userEvent.click(checkbox);
    expect(screen.getByRole("checkbox", { name: "Automatic screening" })).not.toBeChecked();
    expect(screen.getByText("Review every channel before processing")).toBeInTheDocument();
    expect(screen.queryByText(/stays off for later projects/)).not.toBeInTheDocument();
    expect(window.localStorage.getItem("ultra-fast-wbpp.automaticScreening")).toBeNull();
    await userEvent.click(screen.getByRole("checkbox", { name: "Automatic screening" }));
    expect(screen.getByRole("checkbox", { name: "Automatic screening" })).toBeChecked();
  });

  it("keeps the demo's Blink walk-through: reviewing in Blink unchecks automatic screening there only", async () => {
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Load the clearly labelled browser demo" }));
    await userEvent.click(screen.getByRole("button", { name: "Confirm type" }));
    await userEvent.click(screen.getByRole("button", { name: "Review in Blink instead" }));
    expect(await screen.findByRole("heading", { name: /Blink and select/ })).toBeInTheDocument();
    expect(window.localStorage.getItem("ultra-fast-wbpp.automaticScreening")).toBeNull();
  });

  it("starts the ?demo=run walk-through from the import page with automatic screening", async () => {
    window.history.replaceState({}, "", "/?demo=run");
    try {
      render(<App />);
      expect(await screen.findByRole("heading", { name: "Creating verified products" })).toBeInTheDocument();
      expect(screen.getByText("DEMO")).toBeInTheDocument();
      expect(screen.queryByRole("heading", { name: "Review groups and real frame evidence" })).not.toBeInTheDocument();
    } finally {
      window.history.replaceState({}, "", "/");
    }
  });
});
