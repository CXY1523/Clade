import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ post: vi.fn() }));
vi.mock("./base", () => ({ http: { get: vi.fn(), post: mocks.post } }));

import { fetchProviderModels, testApiConnection, updateUIConfig } from "./config";
import type { UIConfig } from "../api.types";

const config = {
  providers: {
    main: {
      id: "main",
      name: "Main",
      type: "openai",
      provider_type: "openai" as const,
      api_key: "",
      api_key_configured: true,
      api_key_clear_requested: true,
      models: [],
    },
  },
  capability_routes: {},
} satisfies UIConfig;

describe("config API credential payloads", () => {
  beforeEach(() => mocks.post.mockReset());

  it("sends explicit provider clear requests outside the persisted config", async () => {
    mocks.post.mockResolvedValue(config);

    await updateUIConfig(config);

    expect(mocks.post).toHaveBeenCalledWith("/api/config/ui", {
      config,
      clear_provider_api_keys: ["main"],
    });
  });

  it("does not clear a stored key when its input is left empty", async () => {
    const keepStoredConfig = {
      ...config,
      providers: {
        main: {
          ...config.providers.main,
          api_key_clear_requested: false,
        },
      },
    } satisfies UIConfig;
    mocks.post.mockResolvedValue(keepStoredConfig);

    await updateUIConfig(keepStoredConfig);

    expect(mocks.post).toHaveBeenCalledWith("/api/config/ui", {
      config: keepStoredConfig,
      clear_provider_api_keys: [],
    });
  });

  it("identifies a provider when testing with its stored key", async () => {
    mocks.post.mockResolvedValue({ success: true, message: "ok" });

    await testApiConnection({
      type: "chat",
      provider_id: "main",
      base_url: "https://example.com/v1",
      api_key: "",
      model: "gpt-test",
      provider_type: "openai",
    });

    expect(mocks.post).toHaveBeenCalledWith(
      "/api/config/test-api",
      expect.objectContaining({ provider_id: "main", api_key: "" }),
      { timeout: 30000 }
    );
  });

  it("identifies a provider when fetching models with its stored key", async () => {
    mocks.post.mockResolvedValue({ success: true, message: "ok", models: [] });

    await fetchProviderModels({
      provider_id: "main",
      base_url: "https://example.com/v1",
      api_key: "",
      provider_type: "openai",
    });

    expect(mocks.post).toHaveBeenCalledWith(
      "/api/config/fetch-models",
      expect.objectContaining({ provider_id: "main", api_key: "" }),
      { timeout: 20000 }
    );
  });
});
