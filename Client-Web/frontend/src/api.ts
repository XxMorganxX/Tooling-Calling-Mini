import type { AppConfig, ApproveRequest, ChatResponse, HealthResponse, StreamCallbacks } from "./types";

const BASE = "/api";

async function request<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    credentials: "include",
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

async function chatStream(content: string, callbacks: StreamCallbacks): Promise<void> {
  const resp = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ content }),
  });

  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    callbacks.onError(`${resp.status}: ${detail}`);
    return;
  }

  const reader = resp.body?.getReader();
  if (!reader) {
    callbacks.onError("No response body");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        currentEvent = line.slice(7).trim();
      } else if (line.startsWith("data: ") && currentEvent) {
        try {
          const data = JSON.parse(line.slice(6));
          switch (currentEvent) {
            case "thinking_token":
              callbacks.onThinkingToken(data.token);
              break;
            case "thinking_done":
              callbacks.onThinkingDone();
              break;
            case "content_token":
              callbacks.onContentToken(data.token);
              break;
            case "parsed":
              callbacks.onParsed(data);
              break;
            case "tool_execution_start":
              callbacks.onToolExecutionStart(data.tools);
              break;
            case "tool_results":
              callbacks.onToolResults(data.tool_results);
              break;
            case "done":
              callbacks.onDone(data);
              break;
            case "error":
              callbacks.onError(data.error);
              break;
          }
        } catch {
          /* malformed JSON, skip */
        }
        currentEvent = "";
      }
    }
  }
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

  chatStream,

  clear: () => request<{ status: string }>("/clear", { method: "POST" }),

  approveTraining: (data: ApproveRequest) =>
    request<{ status: string; file: string }>("/training/approve", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  history: () =>
    request<{ messages: { role: "user" | "assistant"; content: string }[] }>("/history"),
};
