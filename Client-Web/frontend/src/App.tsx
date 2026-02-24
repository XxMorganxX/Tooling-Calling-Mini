import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api } from "./api";
import type { AppConfig, Message, ModelInfo, ToolCall, ToolResult } from "./types";
import "./App.css";

/* ── Sub-components ────────────────────────────────────────────────────── */

function cleanThinking(text: string): string {
  return text.replace(/<\/?think>/g, "").replace(/^\n+/, "");
}

function ThinkingBlock({ text, isStreaming }: { text: string; isStreaming?: boolean }) {
  const [open, setOpen] = useState(false);
  const preRef = useRef<HTMLPreElement>(null);
  const userClosedRef = useRef(false);

  useEffect(() => {
    if (isStreaming && !userClosedRef.current) {
      setOpen(true);
    }
  }, [isStreaming]);

  useEffect(() => {
    if (isStreaming && preRef.current) {
      preRef.current.scrollTop = preRef.current.scrollHeight;
    }
  }, [text, isStreaming]);

  const handleToggle = useCallback((e: React.SyntheticEvent<HTMLDetailsElement>) => {
    const nowOpen = (e.target as HTMLDetailsElement).open;
    setOpen(nowOpen);
    if (!nowOpen) userClosedRef.current = true;
  }, []);

  return (
    <details className="thinking-block" open={open} onToggle={handleToggle}>
      <summary>
        Thinking
        {isStreaming && <span className="streaming-indicator" />}
      </summary>
      <pre ref={preRef}>{cleanThinking(text)}</pre>
    </details>
  );
}

function ToolCallBlock({ calls }: { calls: NonNullable<Message["toolCalls"]> }) {
  return (
    <div className="tool-calls">
      <span className="tool-calls-label">Tool Calls</span>
      {calls.map((tc, i) => (
        <div key={i} className="tool-call">
          <span className="tool-name">{tc.name}</span>
          <pre>{JSON.stringify(tc.arguments, null, 2)}</pre>
        </div>
      ))}
    </div>
  );
}

function ToolResultBlock({ results }: { results: ToolResult[] }) {
  return (
    <div className="tool-results">
      <span className="tool-results-label">Tool Results</span>
      {results.map((tr, i) => (
        <div key={i} className={`tool-result ${tr.success ? "success" : "failure"}`}>
          <div className="tool-result-header">
            <span className={`tool-result-status ${tr.success ? "success" : "failure"}`}>
              {tr.success ? "\u2713" : "\u2717"}
            </span>
            <span className="tool-name">{tr.tool_name}</span>
            <span className="tool-result-duration">{tr.duration_ms.toFixed(0)}ms</span>
          </div>
          {tr.success ? (
            <pre className="tool-result-data">{JSON.stringify(tr.result, null, 2)}</pre>
          ) : (
            <pre className="tool-result-error">{tr.error}</pre>
          )}
        </div>
      ))}
    </div>
  );
}

function UsageBar({ usage }: { usage: NonNullable<Message["usage"]> }) {
  return (
    <div className="usage">
      {usage.prompt_tokens} prompt + {usage.completion_tokens} completion &middot;{" "}
      {usage.tokens_per_second.toFixed(1)} tok/s
    </div>
  );
}

function ApproveButton({
  userPrompt,
  thinking,
  toolCalls,
}: {
  userPrompt: string;
  thinking?: string | null;
  toolCalls?: ToolCall[] | null;
}) {
  const [state, setState] = useState<"idle" | "confirm" | "saving" | "saved" | "error">("idle");

  const handleSave = async () => {
    setState("saving");
    try {
      await api.approveTraining({
        prompt: userPrompt,
        thinking: thinking ?? null,
        tool_calls: toolCalls ?? null,
      });
      setState("saved");
    } catch {
      setState("error");
    }
  };

  if (state === "saved") {
    return (
      <div className="approve-pill approved">
        <div className="approve-pill-icon saved-icon">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M20 6L9 17l-5-5" />
          </svg>
        </div>
        <span>Saved to training data</span>
      </div>
    );
  }

  if (state === "confirm") {
    return (
      <div className="approve-pill confirming">
        <div className="approve-pill-icon confirm-icon">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2" />
            <rect x="9" y="3" width="6" height="4" rx="1" />
          </svg>
        </div>
        <span className="confirm-text">Save as correct training data?</span>
        <div className="confirm-actions">
          <button className="confirm-yes" onClick={handleSave}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
              <path d="M20 6L9 17l-5-5" />
            </svg>
            Confirm
          </button>
          <button className="confirm-no" onClick={() => setState("idle")}>Cancel</button>
        </div>
      </div>
    );
  }

  if (state === "saving") {
    return (
      <div className="approve-pill saving">
        <div className="approve-spinner" />
        <span>Saving...</span>
      </div>
    );
  }

  if (state === "error") {
    return (
      <div className="approve-pill error" onClick={() => setState("idle")}>
        <div className="approve-pill-icon error-icon">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
        </div>
        <span>Failed — click to retry</span>
      </div>
    );
  }

  return (
    <button className="approve-btn" onClick={() => setState("confirm")} title="Mark as correct training data">
      <div className="approve-btn-icon">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          <path d="M20 6L9 17l-5-5" />
        </svg>
      </div>
      <span>Approve</span>
    </button>
  );
}

function cleanContent(text: string): string {
  return text
    .replace(/<\/?think>/g, "")
    .replace(/<\/?tool_call>/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function ToolExecutingIndicator({ tools }: { tools?: string[] }) {
  return (
    <div className="tool-executing">
      <div className="tool-executing-header">
        <div className="loading-dots"><span /><span /><span /></div>
        <span>Executing {tools?.length ?? 0} tool{(tools?.length ?? 0) !== 1 ? "s" : ""}...</span>
      </div>
      {tools && (
        <div className="tool-executing-list">
          {tools.map((t, i) => <span key={i} className="tool-name">{t}</span>)}
        </div>
      )}
    </div>
  );
}

function MessageBubble({ msg, userPrompt }: { msg: Message; userPrompt?: string }) {
  const isUser = msg.role === "user";
  const content = isUser ? msg.content : cleanContent(msg.content);
  const showApprove = !isUser && !msg.isStreaming && !msg.toolsExecuting && userPrompt;
  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="message-header">
        {isUser ? "You" : "Assistant"}
        {msg.isStreaming && <span className="streaming-indicator" />}
      </div>
      {msg.thinking && <ThinkingBlock text={msg.thinking} isStreaming={msg.thinkingStreaming} />}
      {msg.toolCalls && msg.toolCalls.length > 0 && <ToolCallBlock calls={msg.toolCalls} />}
      {msg.toolsExecuting && <ToolExecutingIndicator />}
      {msg.toolResults && msg.toolResults.length > 0 && <ToolResultBlock results={msg.toolResults} />}
      <div className="message-content">
        {isUser ? <p>{content}</p> : content ? <ReactMarkdown>{content}</ReactMarkdown> : null}
      </div>
      {msg.usage && <UsageBar usage={msg.usage} />}
      {showApprove && (
        <ApproveButton
          userPrompt={userPrompt}
          thinking={msg.thinking}
          toolCalls={msg.toolCalls}
        />
      )}
    </div>
  );
}

function ModelBadge({ model }: { model: ModelInfo; active?: boolean }) {
  return (
    <div className={`model-badge ${model.active ? "active" : ""}`}>
      <span className="model-badge-dot" />
      <span>{model.display_name}</span>
    </div>
  );
}

function SettingsPanel({
  config,
  onUpdate,
  onClose,
}: {
  config: AppConfig;
  onUpdate: (patch: Record<string, unknown>) => void;
  onClose: () => void;
}) {
  return (
    <div className="settings-backdrop" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>Settings</h2>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>
        <div className="settings-body">
          {config.models.length > 0 && (
            <div className="settings-section">
              <div className="settings-section-title">Model</div>
              <div className="model-list">
                {config.models.map((m) => (
                  <ModelBadge key={m.key} model={m} />
                ))}
              </div>
            </div>
          )}
          <div className="settings-section">
            <div className="settings-section-title">Generation</div>
            <label>
              <input
                type="checkbox"
                checked={config.enable_thinking}
                onChange={(e) => onUpdate({ enable_thinking: e.target.checked })}
              />
              Enable thinking
            </label>
            <label>
              Max tokens
              <input
                type="number"
                value={config.generation.max_tokens}
                min={1}
                max={4096}
                onChange={(e) => onUpdate({ max_tokens: Number(e.target.value) })}
              />
            </label>
            <label>
              Temperature
              <input
                type="number"
                value={config.generation.temperature}
                min={0}
                max={2}
                step={0.05}
                onChange={(e) => onUpdate({ temperature: Number(e.target.value) })}
              />
            </label>
            <label>
              Top-p
              <input
                type="number"
                value={config.generation.top_p}
                min={0}
                max={1}
                step={0.05}
                onChange={(e) => onUpdate({ top_p: Number(e.target.value) })}
              />
            </label>
            <label>
              Top-k
              <input
                type="number"
                value={config.generation.top_k}
                min={0}
                onChange={(e) => onUpdate({ top_k: Number(e.target.value) })}
              />
            </label>
            <label>
              Repeat penalty
              <input
                type="number"
                value={config.generation.repeat_penalty}
                min={0}
                step={0.05}
                onChange={(e) => onUpdate({ repeat_penalty: Number(e.target.value) })}
              />
            </label>
          </div>
          <div className="server-info">Server: {config.server_url}</div>
        </div>
      </div>
    </div>
  );
}

/* ── Connect screen ────────────────────────────────────────────────────── */

function ConnectScreen({
  onConnected,
  serverStatus,
}: {
  onConnected: () => void;
  serverStatus: "ok" | "degraded" | "error" | "checking";
}) {
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = token.trim();
    if (!trimmed) return;

    setError(null);
    setLoading(true);
    try {
      await api.connect(trimmed);
      onConnected();
    } catch (err: unknown) {
      const detail = err instanceof Error ? err.message : String(err);
      setError(detail);
    } finally {
      setLoading(false);
    }
  };

  const statusColor =
    serverStatus === "ok" ? "var(--green)" : serverStatus === "degraded" ? "var(--yellow)" : "var(--red)";
  const statusLabel =
    serverStatus === "checking" ? "Checking..." : serverStatus === "ok" ? "Online" : serverStatus === "degraded" ? "Degraded" : "Offline";

  return (
    <div className="connect-screen">
      <div className="connect-card">
        <div className="connect-icon">🤖</div>
        <h1>Tool Mini-Model</h1>
        <p className="connect-subtitle">Enter your refresh token to connect.</p>

        <div className="connect-status">
          <span className="status-dot" style={{ background: statusColor }} />
          <span>Server: {statusLabel}</span>
        </div>

        <form onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="Refresh token"
            disabled={loading}
            autoComplete="off"
          />
          <button type="submit" disabled={loading || !token.trim()}>
            {loading ? "Connecting..." : "Connect"}
          </button>
        </form>
        {error && <div className="connect-error">{error}</div>}
        <p className="connect-hint">The token is held in memory only and never stored to disk.</p>
      </div>
    </div>
  );
}

/* ── Main app ──────────────────────────────────────────────────────────── */

export default function App() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [serverStatus, setServerStatus] = useState<"ok" | "degraded" | "error" | "checking">("checking");

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    api.health().then((h) => setServerStatus(h.status)).catch(() => setServerStatus("error"));
    api.authStatus().then((s) => setAuthenticated(s.authenticated)).catch(() => setAuthenticated(false));
  }, []);

  useEffect(() => {
    if (authenticated) {
      api.config().then(setConfig).catch(() => {});
    }
  }, [authenticated]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const handleConnected = () => {
    setAuthenticated(true);
    setMessages([]);
    setError(null);
  };

  const handleDisconnect = async () => {
    await api.disconnect().catch(() => {});
    setAuthenticated(false);
    setMessages([]);
    setConfig(null);
    setError(null);
  };

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setLoading(true);

    const streamingIdx = { current: -1 };
    let thinkingBuf = "";
    let contentBuf = "";

    const updateStreaming = (patch: Partial<Message>) => {
      setMessages((prev) => {
        const updated = [...prev];
        if (streamingIdx.current >= 0 && streamingIdx.current < updated.length) {
          updated[streamingIdx.current] = { ...updated[streamingIdx.current], ...patch };
        }
        return updated;
      });
    };

    try {
      await api.chatStream(text, {
        onThinkingToken(token) {
          if (streamingIdx.current < 0) {
            setMessages((prev) => {
              streamingIdx.current = prev.length;
              return [...prev, { role: "assistant", content: "", thinking: token, isStreaming: true, thinkingStreaming: true }];
            });
            thinkingBuf = token;
          } else {
            thinkingBuf += token;
            updateStreaming({ thinking: thinkingBuf });
          }
        },
        onThinkingDone() {
          updateStreaming({ thinkingStreaming: false });
        },
        onContentToken(token) {
          if (streamingIdx.current < 0) {
            setMessages((prev) => {
              streamingIdx.current = prev.length;
              return [...prev, { role: "assistant", content: token, isStreaming: true }];
            });
            contentBuf = token;
          } else {
            contentBuf += token;
            updateStreaming({ content: contentBuf });
          }
        },
        onParsed(data) {
          if (streamingIdx.current < 0) {
            setMessages((prev) => {
              streamingIdx.current = prev.length;
              return [...prev, { role: "assistant", content: data.content, isStreaming: true }];
            });
          }
          updateStreaming({
            content: data.content,
            thinking: data.thinking || thinkingBuf || undefined,
            toolCalls: data.tool_calls,
            usage: data.usage,
            isStreaming: false,
          });
        },
        onToolExecutionStart() {
          updateStreaming({ toolsExecuting: true });
        },
        onToolResults(results) {
          updateStreaming({ toolResults: results, toolsExecuting: false });
        },
        onDone(data) {
          updateStreaming({
            content: data.content,
            thinking: data.thinking || thinkingBuf || undefined,
            toolCalls: data.tool_calls,
            toolResults: data.tool_results,
            usage: data.usage,
            isStreaming: false,
            thinkingStreaming: false,
            toolsExecuting: false,
          });
        },
        onError(err) {
          setError(err);
        },
      });
    } catch (err: unknown) {
      const detail = err instanceof Error ? err.message : String(err);
      setError(detail);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [input, loading]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const clearChat = async () => {
    await api.clear().catch(() => {});
    setMessages([]);
    setError(null);
  };

  const updateConfig = async (patch: Record<string, unknown>) => {
    try {
      const updated = await api.updateConfig(patch);
      setConfig(updated);
    } catch {
      /* ignore */
    }
  };

  // Still loading auth status
  if (authenticated === null) {
    return (
      <div className="app">
        <div className="empty-state" style={{ margin: "auto" }}>
          <div className="loading-dots"><span /><span /><span /></div>
        </div>
      </div>
    );
  }

  // Not connected -- show login screen
  if (!authenticated) {
    return <ConnectScreen onConnected={handleConnected} serverStatus={serverStatus} />;
  }

  const statusColor =
    serverStatus === "ok" ? "var(--green)" : serverStatus === "degraded" ? "var(--yellow)" : "var(--red)";

  return (
    <div className="app">
      <header>
        <div className="header-left">
          <h1>Tool Mini-Model</h1>
          <span className="status-dot" style={{ background: statusColor }} title={serverStatus} />
          {config?.active_model_display_name && (
            <span className="header-model-name">{config.active_model_display_name}</span>
          )}
        </div>
        <div className="header-right">
          <button className="icon-btn" onClick={clearChat} title="Clear conversation">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m-1 0v14a2 2 0 01-2 2H9a2 2 0 01-2-2V6h10z" />
            </svg>
          </button>
          <button className="icon-btn" onClick={() => setShowSettings(true)} title="Settings">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
            </svg>
          </button>
          <button className="icon-btn" onClick={handleDisconnect} title="Disconnect">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4M16 17l5-5-5-5M21 12H9" />
            </svg>
          </button>
        </div>
      </header>

      <main ref={scrollRef}>
        {messages.length === 0 && !loading && (
          <div className="empty-state">
            <div className="empty-icon">🤖</div>
            <p>Chat with {config?.active_model_display_name ?? "a fine-tuned tool-calling model"}.</p>
            <p className="empty-hint">Type a message below to get started.</p>
          </div>
        )}
        {messages.map((msg, i) => {
          const userPrompt =
            msg.role === "assistant" && i > 0 && messages[i - 1]?.role === "user"
              ? messages[i - 1].content
              : undefined;
          return <MessageBubble key={i} msg={msg} userPrompt={userPrompt} />;
        })}
        {loading && messages.length > 0 && !messages[messages.length - 1]?.isStreaming && messages[messages.length - 1]?.role !== "assistant" && (
          <div className="message assistant">
            <div className="message-header">Assistant</div>
            <div className="loading-dots">
              <span /><span /><span />
            </div>
          </div>
        )}
        {error && <div className="error-banner">{error}</div>}
      </main>

      <footer>
        <div className="input-row">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            rows={1}
            disabled={loading}
          />
          <button className="send-btn" onClick={send} disabled={loading || !input.trim()}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        </div>
      </footer>

      {showSettings && config && (
        <SettingsPanel config={config} onUpdate={updateConfig} onClose={() => setShowSettings(false)} />
      )}
    </div>
  );
}
