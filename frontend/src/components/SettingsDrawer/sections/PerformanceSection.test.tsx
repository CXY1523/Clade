import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ProviderConfig, UIConfig } from "@/services/api.types";
import { PerformanceSection } from "./PerformanceSection";

const storedProvider: ProviderConfig = {
  id: "stored",
  name: "Stored Provider",
  type: "openai",
  provider_type: "openai",
  base_url: "https://example.com/v1",
  api_key: "",
  api_key_configured: true,
  models: ["gpt-test"],
};

const pendingClearProvider: ProviderConfig = {
  ...storedProvider,
  id: "pending",
  name: "Pending Clear Provider",
  api_key_clear_requested: true,
};

describe("PerformanceSection provider availability", () => {
  it("offers stored-key providers but excludes providers pending clear", () => {
    const providers = {
      stored: storedProvider,
      pending: pendingClearProvider,
    };
    const config = { providers, capability_routes: {} } satisfies UIConfig;
    const { container } = render(
      <PerformanceSection config={config} providers={providers} dispatch={vi.fn()} />
    );

    const defaultProviderSelect = screen.getAllByRole("combobox")[0];
    expect(
      within(defaultProviderSelect).getByRole("option", { name: /Stored Provider/ })
    ).toBeInTheDocument();
    expect(
      within(defaultProviderSelect).queryByRole("option", { name: /Pending Clear Provider/ })
    ).toBeNull();

    fireEvent.click(container.querySelector<HTMLElement>(".capability-item-header")!);

    const capabilityProviderSelect = screen.getAllByRole("combobox")[2];
    expect(
      within(capabilityProviderSelect).getByRole("option", { name: /Stored Provider/ })
    ).toBeInTheDocument();
    expect(
      within(capabilityProviderSelect).queryByRole("option", { name: /Pending Clear Provider/ })
    ).toBeNull();
  });
});
