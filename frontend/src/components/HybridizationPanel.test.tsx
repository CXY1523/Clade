import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  fetchSpeciesList: vi.fn(),
  get: vi.fn(),
}));

vi.mock("@/services/api", () => ({
  fetchSpeciesList: apiMocks.fetchSpeciesList,
  http: {
    get: apiMocks.get,
  },
}));

import { HybridizationPanel } from "./HybridizationPanel";

describe("HybridizationPanel species API boundary", () => {
  beforeEach(() => {
    apiMocks.fetchSpeciesList.mockReset().mockResolvedValue([
      {
        lineage_code: "alive-species",
        latin_name: "Species viva",
        common_name: "存活物种",
        population: 120,
        status: "alive",
        ecological_role: "herbivore",
      },
      {
        lineage_code: "extinct-species",
        latin_name: "Species extincta",
        common_name: "灭绝物种",
        population: 0,
        status: "extinct",
        ecological_role: "herbivore",
      },
    ]);
    apiMocks.get.mockReset().mockResolvedValue({ candidates: [] });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("loads the living species list through the shared API service", async () => {
    const browserFetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/species/list") {
        throw new Error("HybridizationPanel must not fetch the species list directly");
      }
      if (url === "/api/hybridization/candidates") {
        return {
          ok: true,
          json: async () => ({ candidates: [] }),
        } as Response;
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", browserFetch);
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.spyOn(console, "log").mockImplementation(() => undefined);

    render(<HybridizationPanel onClose={vi.fn()} />);

    await waitFor(() => {
      expect(apiMocks.fetchSpeciesList).toHaveBeenCalledOnce();
    });
    fireEvent.click(screen.getByRole("button", { name: /强行杂交/ }));

    expect(await screen.findByText("1 个可用物种")).toBeInTheDocument();
    const requestedUrls = browserFetch.mock.calls.map(([input]) => String(input));
    expect(requestedUrls).not.toContain("/api/species/list");
  });

  it("loads hybrid candidates through the shared HTTP client", async () => {
    const browserFetch = vi.fn().mockRejectedValue(
      new Error("HybridizationPanel must not fetch candidates directly"),
    );
    vi.stubGlobal("fetch", browserFetch);
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.spyOn(console, "log").mockImplementation(() => undefined);

    render(<HybridizationPanel onClose={vi.fn()} />);

    await waitFor(() => {
      expect(apiMocks.get).toHaveBeenCalledWith("/api/hybridization/candidates");
    });
    expect(await screen.findByText("暂无可杂交物种对")).toBeInTheDocument();
    expect(browserFetch).not.toHaveBeenCalled();
  });

  it("loads the normal hybrid preview through the shared HTTP client", async () => {
    const candidate = {
      species_a: {
        lineage_code: "alpha-a",
        common_name: "候选甲",
        latin_name: "Species alpha",
        genus_code: "alpha",
      },
      species_b: {
        lineage_code: "alpha-b",
        common_name: "候选乙",
        latin_name: "Species beta",
        genus_code: "alpha",
      },
      fertility: 0.42,
      genus: "alpha",
    };
    const previewPath =
      "/api/hybridization/preview?species_a=alpha-a&species_b=alpha-b";
    apiMocks.get.mockImplementation(async (path: string) => {
      if (path === "/api/hybridization/candidates") {
        return { candidates: [candidate], total: 1 };
      }
      if (path === previewPath) {
        return {
          can_hybridize: true,
          fertility: 0.42,
          energy_cost: 10,
          can_afford: true,
          preview: {
            lineage_code: "alpha-a×alpha-b",
            common_name: "甲乙杂交种",
            predicted_trophic_level: 2,
            combined_capabilities: [],
            parent_traits_merged: true,
          },
        };
      }
      throw new Error(`Unexpected shared HTTP request: ${path}`);
    });
    const browserFetch = vi.fn().mockRejectedValue(
      new Error("HybridizationPanel must not fetch the normal preview directly"),
    );
    vi.stubGlobal("fetch", browserFetch);
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.spyOn(console, "log").mockImplementation(() => undefined);

    render(<HybridizationPanel onClose={vi.fn()} />);

    fireEvent.click(await screen.findByText("候选甲"));

    await waitFor(() => {
      expect(apiMocks.get).toHaveBeenCalledWith(previewPath);
    });
    expect(await screen.findByText("甲乙杂交种")).toBeInTheDocument();
    expect(browserFetch).not.toHaveBeenCalled();
  });
});
