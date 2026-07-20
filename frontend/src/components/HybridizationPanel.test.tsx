import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  fetchSpeciesList: vi.fn(),
}));

vi.mock("@/services/api", () => ({
  fetchSpeciesList: apiMocks.fetchSpeciesList,
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
});
