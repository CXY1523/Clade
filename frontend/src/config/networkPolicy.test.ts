import { spawnSync } from "child_process";
import { describe, expect, it } from "vitest";

const NETWORK_ENV_KEYS = ["ALLOW_LAN_ACCESS", "BACKEND_HOST", "FRONTEND_HOST"] as const;

type NetworkEnv = Partial<Record<(typeof NETWORK_ENV_KEYS)[number], string>>;

const LOAD_CONFIG_SCRIPT = `
  import { loadConfigFromFile } from "vite";
  const loaded = await loadConfigFromFile(
    { command: "serve", mode: "test" },
    "vite.config.ts",
  );
  if (!loaded) throw new Error("Vite configuration did not load");
  const server = loaded.config.server;
  console.log(JSON.stringify({
    host: server.host,
    target: server.proxy["/api"].target,
  }));
`;

function loadServerConfig(overrides: NetworkEnv) {
  const env = { ...process.env };
  for (const key of NETWORK_ENV_KEYS) {
    delete env[key];
  }
  Object.assign(env, overrides);

  const result = spawnSync(process.execPath, ["--input-type=module", "-e", LOAD_CONFIG_SCRIPT], {
    cwd: process.cwd(),
    encoding: "utf8",
    env,
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error((result.stderr || result.stdout).trim());
  }
  return JSON.parse(result.stdout) as { host: string; target: string };
}

describe("Vite network policy", () => {
  it("rejects a non-loopback frontend host without LAN opt-in", () => {
    expect(() => loadServerConfig({ FRONTEND_HOST: "0.0.0.0" })).toThrow("ALLOW_LAN_ACCESS=true");
  });

  it("allows a non-loopback frontend host with LAN opt-in", () => {
    const config = loadServerConfig({
      ALLOW_LAN_ACCESS: "true",
      FRONTEND_HOST: "0.0.0.0",
    });

    expect(config.host).toBe("0.0.0.0");
  });

  it.each(["0.0.0.0", "::"])(
    "maps wildcard backend host %s to a loopback proxy target",
    (backendHost) => {
      const config = loadServerConfig({
        ALLOW_LAN_ACCESS: "true",
        BACKEND_HOST: backendHost,
      });

      expect(config.target).toBe("http://127.0.0.1:8022");
    }
  );

  it("uses a specific LAN backend address as the proxy target", () => {
    const config = loadServerConfig({
      ALLOW_LAN_ACCESS: "true",
      BACKEND_HOST: "192.168.1.25",
    });

    expect(config.target).toBe("http://192.168.1.25:8022");
  });
});
