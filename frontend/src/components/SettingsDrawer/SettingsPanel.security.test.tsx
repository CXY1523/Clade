import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { UIConfig } from "@/services/api.types";
import { SettingsPanel } from "./SettingsPanel";

function makeConfig(overrides: Partial<UIConfig> = {}): UIConfig {
  return {
    providers: {
      main: {
        id: "main",
        name: "Main Provider",
        type: "openai",
        provider_type: "openai",
        base_url: "http://localhost:11434/v1",
        api_key: "",
        models: ["local-model"],
        selected_models: ["local-model"],
      },
    },
    capability_routes: {},
    allow_local_ai_endpoints: false,
    ...overrides,
  };
}

function renderPanel(onSave = vi.fn().mockResolvedValue(undefined), onClose = vi.fn()) {
  const config = makeConfig();
  return {
    config,
    onSave,
    onClose,
    ...render(<SettingsPanel config={config} onSave={onSave} onClose={onClose} />),
  };
}

async function changeSecuritySetting(user: ReturnType<typeof userEvent.setup>) {
  await act(async () => {
    await user.click(screen.getByRole("checkbox", { name: "允许访问本机 AI 服务" }));
  });
}

async function save(user: ReturnType<typeof userEvent.setup>) {
  await act(async () => {
    await user.click(screen.getByRole("button", { name: /保存配置/ }));
  });
}

describe("SettingsPanel local AI security lifecycle", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("saves a normal field change without token options", async () => {
    const user = userEvent.setup();
    const onSave = vi.fn().mockResolvedValue(undefined);
    renderPanel(onSave);

    const nameInput = screen.getByPlaceholderText("输入便于识别的名称");
    await act(async () => {
      await user.clear(nameInput);
      await user.type(nameInput, "Renamed Provider");
    });
    await save(user);

    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1));
    expect(onSave.mock.calls[0]).toHaveLength(1);
    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({
        providers: expect.objectContaining({
          main: expect.objectContaining({ name: "Renamed Provider" }),
        }),
      })
    );
  });

  it("blocks a security switch save without an administrator token", async () => {
    const user = userEvent.setup();
    const onSave = vi.fn().mockResolvedValue(undefined);
    renderPanel(onSave);

    await changeSecuritySetting(user);
    await save(user);

    expect(onSave).not.toHaveBeenCalled();
    expect(screen.getByText("请输入管理员令牌")).toBeInTheDocument();
  });

  it("passes the administrator token only as save options", async () => {
    const user = userEvent.setup();
    const onSave = vi.fn().mockResolvedValue(undefined);
    renderPanel(onSave);

    await changeSecuritySetting(user);
    await act(async () => {
      await user.type(screen.getByLabelText("管理员令牌"), "admin-secret");
    });
    await save(user);

    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1));
    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({ allow_local_ai_endpoints: true }),
      { adminToken: "admin-secret" }
    );
    expect(JSON.stringify(onSave.mock.calls[0][0])).not.toContain("admin-secret");
  });

  it("clears the administrator token after a successful save", async () => {
    const user = userEvent.setup();
    renderPanel();

    await changeSecuritySetting(user);
    const tokenInput = screen.getByLabelText("管理员令牌");
    await act(async () => {
      await user.type(tokenInput, "admin-secret");
    });
    await save(user);

    await waitFor(() => expect(tokenInput).toHaveValue(""));
  });

  it.each([
    [403, "admin_token_invalid", "管理员令牌不正确，请重新输入。"],
    [503, "admin_token_unconfigured", "服务端尚未配置 CLADE_ADMIN_TOKEN。"],
  ])(
    "clears the token and preserves the unsaved switch after a %i failure",
    async (status, code, guidance) => {
      const user = userEvent.setup();
      const error = Object.assign(new Error("request failed"), {
        status,
        statusText: "Save failed",
        detail: "unsafe raw detail",
        code,
      });
      const onSave = vi.fn().mockRejectedValue(error);
      renderPanel(onSave);

      await changeSecuritySetting(user);
      const tokenInput = screen.getByLabelText("管理员令牌");
      await act(async () => {
        await user.type(tokenInput, "admin-secret");
      });
      await save(user);

      await waitFor(() => expect(screen.getByText(guidance)).toBeInTheDocument());
      expect(tokenInput).toHaveValue("");
      expect(screen.getByRole("checkbox", { name: "允许访问本机 AI 服务" })).toBeChecked();
    }
  );

  it.each([
    [
      "cancel button",
      async (user: ReturnType<typeof userEvent.setup>) =>
        user.click(screen.getByRole("button", { name: "取消" })),
    ],
    [
      "overlay",
      async (user: ReturnType<typeof userEvent.setup>, container: HTMLElement) =>
        user.click(container.querySelector<HTMLElement>(".settings-panel")!),
    ],
    [
      "close button",
      async (user: ReturnType<typeof userEvent.setup>) =>
        user.click(screen.getByTitle("关闭 (Esc)")),
    ],
    ["Escape", async (user: ReturnType<typeof userEvent.setup>) => user.keyboard("{Escape}")],
  ])("clears the token through the %s close path", async (_name, close) => {
    const user = userEvent.setup();
    const { container, onClose } = renderPanel();
    await changeSecuritySetting(user);
    const tokenInput = screen.getByLabelText("管理员令牌");
    await act(async () => {
      await user.type(tokenInput, "admin-secret");
    });

    await act(async () => {
      await close(user, container);
    });

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(tokenInput).toHaveValue("");
  });

  it("never writes the token to browser storage or the serialized config", async () => {
    const user = userEvent.setup();
    const localSetItem = vi.spyOn(window.localStorage, "setItem");
    const sessionSetItem = vi.spyOn(window.sessionStorage, "setItem");
    const onSave = vi.fn().mockResolvedValue(undefined);
    renderPanel(onSave);

    await changeSecuritySetting(user);
    await act(async () => {
      await user.type(screen.getByLabelText("管理员令牌"), "admin-secret");
    });
    await save(user);

    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1));
    expect(localSetItem).not.toHaveBeenCalled();
    expect(sessionSetItem).not.toHaveBeenCalled();
    expect(JSON.stringify(onSave.mock.calls[0][0])).not.toContain("admin-secret");
  });
});
