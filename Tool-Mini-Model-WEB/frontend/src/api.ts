import type { AppConfig, ChatResponse, HealthResponse } from "./types";

const BASE = "/api";

async function request<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(`${resp.status}: ${detail}`);
  }
  return resp.json();
}

export const api = {
  health: () => request<HealthResponse>("/health"),

  authStatus: () => request<{ authenticated: boolean }>("/auth/status"),

  connect: (refreshToken: string) =>
    request<{ status: string; expires_at: string }>("/auth/connect", {
      method: "POST",
      body: JSON.stringify({ refresh_token: refreshToken }),
    }),

  disconnect: () => request<{ status: string }>("/auth/disconnect", { method: "POST" }),

  config: () => request<AppConfig>("/config"),

  updateConfig: (patch: Partial<AppConfig["generation"] & { enable_thinking: boolean }>) =>
    request<AppConfig>("/config", { method: "PATCH", body: JSON.stringify(patch) }),

  chat: (content: string) =>
    request<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify({ content }),
    }),

  clear: () => request<{ status: string }>("/clear", { method: "POST" }),

  history: () =>
    request<{ messages: { role: "user" | "assistant"; content: string }[] }>("/history"),
};
