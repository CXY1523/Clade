import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  get: vi.fn(),
}));

vi.mock("@/services/api", () => ({
  http: {
    get: apiMocks.get,
  },
}));

import { DivinePowersPanel } from "./DivinePowersPanel";

describe("DivinePowersPanel API boundary", () => {
  beforeEach(() => {
    apiMocks.get.mockReset().mockResolvedValue({
      path: null,
      available_paths: [
        {
          path: "creator",
          name: "创造之神",
          icon: "🌱",
          description: "通过创造生命塑造世界",
          passive_bonus: "创造消耗降低",
          color: "#22c55e",
          skills: ["生命萌发"],
        },
      ],
      faith: {
        total_followers: 0,
        total_faith: 0,
        faith_bonus_per_turn: 0,
        followers: [],
      },
      miracles: [],
      charging_miracle: null,
      wagers: {
        active_wagers: [],
        total_bet: 0,
        total_won: 0,
        total_lost: 0,
        net_profit: 0,
        consecutive_wins: 0,
        consecutive_losses: 0,
        faith_shaken_turns: 0,
        wager_types: [],
      },
      stats: {
        total_skills_used: 0,
        total_miracles_cast: 0,
      },
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("loads divine status through the shared HTTP client", async () => {
    const browserFetch = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/divine/status") {
        throw new Error("DivinePowersPanel must not fetch divine status directly");
      }
      if (url === "/api/divine/skills") {
        return {
          ok: true,
          json: async () => ({ skills: [] }),
        } as Response;
      }
      throw new Error(`Unexpected browser request: ${url}`);
    });
    vi.stubGlobal("fetch", browserFetch);
    vi.spyOn(console, "error").mockImplementation(() => undefined);

    render(<DivinePowersPanel onClose={vi.fn()} />);

    await waitFor(() => {
      expect(apiMocks.get).toHaveBeenCalledWith("/api/divine/status");
    });
    expect(await screen.findByText("选择你的神格")).toBeInTheDocument();
    const requestedUrls = browserFetch.mock.calls.map(([input]) => String(input));
    expect(requestedUrls).not.toContain("/api/divine/status");
  });
});
