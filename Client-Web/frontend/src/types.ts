export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  tokens_per_second: number;
}

export interface ToolResult {
  tool_name: string;
  success: boolean;
  result: unknown;
  error: string | null;
  duration_ms: number;
}

export interface ChatResponse {
  content: string;
  thinking: string | null;
  tool_calls: ToolCall[] | null;
  tool_results: ToolResult[] | null;
  usage: Usage | null;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "error";
  llama_server?: "reachable" | "unreachable";
  detail?: string;
}

export interface ModelInfo {
  key: string;
  display_name: string;
  active: boolean;
}

export interface AppConfig {
  server_url: string;
  enable_thinking: boolean;
  generation: {
    max_tokens: number;
    temperature: number;
    top_p: number;
    top_k: number;
    min_p: number;
    repeat_penalty: number;
  };
  max_history_messages: number;
  active_model: string | null;
  active_model_display_name: string | null;
  models: ModelInfo[];
}

export interface Message {
  role: "user" | "assistant" | "tool";
  content: string;
  thinking?: string | null;
  toolCalls?: ToolCall[] | null;
  toolResults?: ToolResult[] | null;
  usage?: Usage | null;
  isStreaming?: boolean;
  thinkingStreaming?: boolean;
  toolsExecuting?: boolean;
}

export interface ApproveRequest {
  prompt: string;
  thinking: string | null;
  tool_calls: ToolCall[] | null;
}

export interface StreamCallbacks {
  onThinkingToken: (token: string) => void;
  onThinkingDone: () => void;
  onContentToken: (token: string) => void;
  onParsed: (data: ChatResponse) => void;
  onToolExecutionStart: (tools: string[]) => void;
  onToolResults: (results: ToolResult[]) => void;
  onDone: (data: ChatResponse) => void;
  onError: (error: string) => void;
}
