import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ProviderConfig } from "@/services/api.types";
import { LocalAIEndpointControl } from "./LocalAIEndpointControl";

function provider(baseUrl: string): ProviderConfig {
  return {
    id: baseUrl,
    name: baseUrl,
    type: "openai",
    provider_type: "openai",
    base_url: baseUrl,
    models: [],
  };
}

function renderControl(
  providers: Record<string, ProviderConfig>,
  overrides: Partial<React.ComponentProps<typeof LocalAIEndpointControl>> = {}
) {
  const props: React.ComponentProps<typeof LocalAIEndpointControl> = {
    enabled: true,
    savedEnabled: false,
    providers,
    adminToken: "",
    saveError: null,
    onEnabledChange: vi.fn(),
    onAdminTokenChange: vi.fn(),
    ...overrides,
  };
  return { ...render(<LocalAIEndpointControl {...props} />), props };
}

describe("LocalAIEndpointControl", () => {
  it.each(["http://localhost:11434/v1", "http://127.0.0.1:8080/v1", "http://[::1]:11434/v1"])(
    "shows the retained-config warning for %s",
    (baseUrl) => {
      renderControl({ local: provider(baseUrl), public: provider("https://api.example.com/v1") });

      expect(screen.getByText("本地 AI 访问")).toBeInTheDocument();
      expect(screen.getByText(/配置已保留，当前被安全策略阻止/)).toBeInTheDocument();
      expect(screen.getByLabelText("管理员令牌")).toHaveAttribute("type", "password");
      expect(screen.getByText(/不放行家庭局域网/)).toBeInTheDocument();
    }
  );

  it("does not show a blocked-config warning for public-only providers", () => {
    renderControl({ public: provider("https://api.example.com/v1") });

    expect(screen.queryByText(/配置已保留，当前被安全策略阻止/)).not.toBeInTheDocument();
  });

  it("normalizes localhost. for display detection", () => {
    renderControl({ local: provider("http://localhost.:11434/v1") });

    expect(screen.getByText(/配置已保留，当前被安全策略阻止/)).toBeInTheDocument();
  });

  it("reports switch changes without mutating providers", async () => {
    const user = userEvent.setup();
    const providers = { local: provider("http://localhost:11434/v1") };
    const snapshot = structuredClone(providers);
    const onEnabledChange = vi.fn();
    renderControl(providers, { enabled: false, savedEnabled: false, onEnabledChange });

    await user.click(screen.getByRole("checkbox", { name: "允许访问本机 AI 服务" }));

    expect(onEnabledChange).toHaveBeenCalledWith(true);
    expect(providers).toEqual(snapshot);
  });

  it("shows the administrator token only while the switch differs from saved config", () => {
    const { rerender, props } = renderControl({}, { enabled: false, savedEnabled: false });

    expect(screen.queryByLabelText("管理员令牌")).not.toBeInTheDocument();

    rerender(<LocalAIEndpointControl {...props} enabled />);
    expect(screen.getByLabelText("管理员令牌")).toBeInTheDocument();
  });
});
