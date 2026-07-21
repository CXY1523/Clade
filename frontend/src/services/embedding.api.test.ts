import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}));

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    http: {
      ...actual.http,
      get: apiMocks.get,
      post: apiMocks.post,
    },
  };
});

import { embeddingApi } from "./embedding.api";

describe("Embedding API boundary", () => {
  beforeEach(() => {
    apiMocks.get.mockReset();
    apiMocks.post.mockReset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("builds taxonomy through the shared HTTP client", async () => {
    const response = {
      success: true,
      tree: {
        root: {
          name: "Life",
          latin_name: "Vita",
          children: [],
        },
      },
      stats: {
        total_species: 2,
        total_clades: 1,
        domain_count: 1,
        max_depth: 1,
        clustering_params: { threshold: 0.8 },
        turn_index: 12,
      },
      species_assignments: { "sp-alpha": ["Life"] },
    };
    apiMocks.post.mockResolvedValue(response);
    const browserFetch = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify(response)));
    vi.stubGlobal("fetch", browserFetch);

    const result = await embeddingApi.buildTaxonomy(true, { threshold: 0.8 });

    expect(result).toEqual(response);
    expect(apiMocks.post).toHaveBeenCalledWith(
      "/api/embedding/taxonomy/build",
      { rebuild: true, params: { threshold: 0.8 } },
    );
    expect(browserFetch).not.toHaveBeenCalled();
  });

  it("loads species taxonomy through the shared HTTP client", async () => {
    const response = {
      species_code: "sp-alpha",
      classification: ["Life", "Alpha"],
      related_species: ["sp-beta"],
    };
    apiMocks.get.mockResolvedValue(response);
    const browserFetch = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify(response)));
    vi.stubGlobal("fetch", browserFetch);

    const result = await embeddingApi.getSpeciesTaxonomy("sp-alpha");

    expect(result).toEqual(response);
    expect(apiMocks.get).toHaveBeenCalledWith(
      "/api/embedding/taxonomy/species/sp-alpha",
    );
    expect(browserFetch).not.toHaveBeenCalled();
  });

  it("propagates shared-client taxonomy failures", async () => {
    const error = Object.assign(new Error("Taxonomy unavailable"), {
      name: "ApiError",
      status: 503,
      statusText: "Service Unavailable",
      detail: "Taxonomy unavailable",
    });
    apiMocks.post.mockRejectedValue(error);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: error.detail }))),
    );

    await expect(embeddingApi.buildTaxonomy()).rejects.toBe(error);
  });
});
