import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api } from "./api";
import type { AppConfig, Message } from "./types";
import "./App.css";

/* ── Sub-components ────────────────────────────────────────────────────── */

function ThinkingBlock({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="thinking-block" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>Thinking</summary>
      <pre>{text}</pre>
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

function UsageBar({ usage }: { usage: NonNullable<Message["usage"]> }) {
  return (
    <div className="usage">
      {usage.prompt_tokens} prompt + {usage.completion_tokens} completion &middot;{" "}
      {usage.tokens_per_second.toFixed(1)} tok/s
    </div>
  );
}

function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="message-header">{isUser ? "You" : "Assistant"}</div>
      {msg.thinking && <ThinkingBlock text={msg.thinking} />}
      {msg.toolCalls && msg.toolCalls.length > 0 && <ToolCallBlock calls={msg.toolCalls} />}
      <div className="message-content">
        {isUser ? <p>{msg.content}</p> : <ReactMarkdown>{msg.content}</ReactMarkdown>}
      </div>
      {msg.usage && <UsageBar usage={msg.usage} />}
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

    try {
      const resp = await api.chat(text);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: resp.content,
          thinking: resp.thinking,
          toolCalls: resp.tool_calls,
          usage: resp.usage,
        },
      ]);
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
            <p>Chat with a fine-tuned Qwen3-4B tool-calling model.</p>
            <p className="empty-hint">Type a message below to get started.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} />
        ))}
        {loading && (
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
