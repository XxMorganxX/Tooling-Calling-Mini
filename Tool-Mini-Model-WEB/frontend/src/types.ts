export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  tokens_per_second: number;
}

export interface ChatResponse {
  content: string;
  thinking: string | null;
  tool_calls: ToolCall[] | null;
  usage: Usage | null;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "error";
  llama_server?: "reachable" | "unreachable";
  detail?: string;
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
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  thinking?: string | null;
  toolCalls?: ToolCall[] | null;
  usage?: Usage | null;
}
