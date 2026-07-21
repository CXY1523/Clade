import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}));

vi.mock("@/services/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/api")>();
  return {
    ...actual,
    http: {
      ...actual.http,
      get: apiMocks.get,
      post: apiMocks.post,
    },
  };
});

import { DivinePowersPanel } from "./DivinePowersPanel";

const selectedStatusResponse = {
  path: {
    path: "creator",
    name: "Creator",
    icon: "creator",
    description: "Create life",
    passive_bonus: "Lower creation cost",
    color: "#22c55e",
    skills: ["life_spark"],
    level: 1,
    experience: 10,
    next_level_exp: 100,
    unlocked_skills: ["life_spark"],
    secondary_path: null,
  },
  available_paths: null,
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
};

const skillsResponse = {
  skills: [
    {
      id: "life_spark",
      name: "Life Spark",
      path: "creator",
      description: "Create new life",
      cost: 10,
      cooldown: 1,
      unlock_level: 1,
      icon: "spark",
      unlocked: true,
      uses: 0,
      is_current_path: true,
    },
  ],
  current_path: "creator",
};

function mockSelectedPathRequests() {
  apiMocks.get.mockImplementation(async (path: string) => {
    if (path === "/api/divine/status") {
      return selectedStatusResponse;
    }
    if (path === "/api/divine/skills") {
      return skillsResponse;
    }
    throw new Error(`Unexpected shared request: ${path}`);
  });
}

describe("DivinePowersPanel API boundary", () => {
  beforeEach(() => {
    apiMocks.post.mockReset();
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

  it("loads divine skills through the shared HTTP client", async () => {
    mockSelectedPathRequests();
    const browserFetch = vi
      .fn()
      .mockRejectedValue(
        new Error("DivinePowersPanel must not fetch divine skills directly"),
      );
    vi.stubGlobal("fetch", browserFetch);
    vi.spyOn(console, "error").mockImplementation(() => undefined);

    render(<DivinePowersPanel onClose={vi.fn()} />);

    await waitFor(() => {
      expect(apiMocks.get).toHaveBeenCalledWith("/api/divine/status");
      expect(apiMocks.get).toHaveBeenCalledWith("/api/divine/skills");
    });
    expect(await screen.findByText("Life Spark")).toBeInTheDocument();
    expect(browserFetch).not.toHaveBeenCalled();
  });

  it("chooses a divine path through the shared HTTP client", async () => {
    apiMocks.post.mockResolvedValue({
      success: true,
      message: "Path chosen",
      path_info: {
        path: "creator",
        name: "Creator",
        icon: "creator",
        description: "Create life",
        passive_bonus: "Lower creation cost",
        color: "#22c55e",
        level: 1,
        experience: 0,
        next_level_exp: 100,
        unlocked_skills: ["life_spark"],
        secondary_path: null,
      },
    });
    const browserFetch = vi
      .fn()
      .mockRejectedValue(
        new Error("DivinePowersPanel must not choose a path directly"),
      );
    const alertMock = vi.fn();
    vi.stubGlobal("fetch", browserFetch);
    vi.stubGlobal("alert", alertMock);
    vi.spyOn(console, "error").mockImplementation(() => undefined);

    render(<DivinePowersPanel onClose={vi.fn()} />);

    fireEvent.click(await screen.findByText("选择此神格"));
    await waitFor(() => {
      expect(apiMocks.post).toHaveBeenCalledWith("/api/divine/path/choose", {
        path: "creator",
      });
    });
    expect(alertMock).toHaveBeenCalledWith("Path chosen");
    const statusRequests = apiMocks.get.mock.calls.filter(
      ([path]) => path === "/api/divine/status",
    );
    expect(statusRequests).toHaveLength(2);
    expect(browserFetch).not.toHaveBeenCalled();
  });

  it("shows a divine path rejection returned by the shared HTTP client", async () => {
    apiMocks.post.mockRejectedValue(
      Object.assign(new Error("Request failed: Path already chosen"), {
        name: "ApiError",
        status: 400,
        statusText: "Bad Request",
        detail: "Path already chosen",
      }),
    );
    const browserFetch = vi
      .fn()
      .mockRejectedValue(
        new Error("DivinePowersPanel must not choose a path directly"),
      );
    const alertMock = vi.fn();
    vi.stubGlobal("fetch", browserFetch);
    vi.stubGlobal("alert", alertMock);
    vi.spyOn(console, "error").mockImplementation(() => undefined);

    render(<DivinePowersPanel onClose={vi.fn()} />);

    fireEvent.click(await screen.findByText("选择此神格"));
    await waitFor(() => {
      expect(apiMocks.post).toHaveBeenCalledWith("/api/divine/path/choose", {
        path: "creator",
      });
    });
    expect(alertMock).toHaveBeenCalledWith("Path already chosen");
    expect(browserFetch).not.toHaveBeenCalled();
  });

  it("uses a divine skill through the shared HTTP client", async () => {
    mockSelectedPathRequests();
    apiMocks.post.mockResolvedValue({
      success: true,
      skill: "Life Spark",
      cost: 10,
      result: {
        effect: "executed",
        details: "Skill released",
      },
      energy_remaining: 90,
    });
    const browserFetch = vi
      .fn()
      .mockRejectedValue(
        new Error("DivinePowersPanel must not use a skill directly"),
      );
    const alertMock = vi.fn();
    const energyChanged = vi.fn();
    window.addEventListener("energy-changed", energyChanged, { once: true });
    vi.stubGlobal("fetch", browserFetch);
    vi.stubGlobal("alert", alertMock);
    vi.spyOn(console, "error").mockImplementation(() => undefined);

    render(<DivinePowersPanel onClose={vi.fn()} />);

    fireEvent.click(await screen.findByText("释放"));
    await waitFor(() => {
      expect(apiMocks.post).toHaveBeenCalledWith("/api/divine/skill/use", {
        skill_id: "life_spark",
        target: null,
      });
    });
    expect(alertMock).toHaveBeenCalledWith("Skill released");
    const statusRequests = apiMocks.get.mock.calls.filter(
      ([path]) => path === "/api/divine/status",
    );
    expect(statusRequests).toHaveLength(2);
    expect(energyChanged).toHaveBeenCalledTimes(1);
    expect(browserFetch).not.toHaveBeenCalled();
  });

  it("shows a divine skill rejection returned by the shared HTTP client", async () => {
    mockSelectedPathRequests();
    apiMocks.post.mockRejectedValue(
      Object.assign(new Error("Request failed: Not enough energy"), {
        name: "ApiError",
        status: 400,
        statusText: "Bad Request",
        detail: "Not enough energy",
      }),
    );
    const browserFetch = vi
      .fn()
      .mockRejectedValue(
        new Error("DivinePowersPanel must not use a skill directly"),
      );
    const alertMock = vi.fn();
    vi.stubGlobal("fetch", browserFetch);
    vi.stubGlobal("alert", alertMock);
    vi.spyOn(console, "error").mockImplementation(() => undefined);

    render(<DivinePowersPanel onClose={vi.fn()} />);

    fireEvent.click(await screen.findByText("释放"));
    await waitFor(() => {
      expect(apiMocks.post).toHaveBeenCalledWith("/api/divine/skill/use", {
        skill_id: "life_spark",
        target: null,
      });
    });
    expect(alertMock).toHaveBeenCalledWith("Not enough energy");
    expect(browserFetch).not.toHaveBeenCalled();
  });
});
