# Client-Web

Web-based chat interface for the tool-calling inference server. React + TypeScript frontend with a FastAPI backend proxy.

## Architecture

```
Browser (React)
    ↓ HTTP / SSE
FastAPI Backend (backend/main.py :8001)
    ↓ HTTP + X-API-Key
Inference Server (Model/server.py :8000)
```

The backend acts as a proxy: it manages sessions, stores conversation history per user, handles authentication with the inference server, and serves the built frontend as static files in production.

## Subfolders

| Folder | Purpose |
|--------|---------|
| `backend/` | FastAPI proxy server — session management, config, training data approval |
| `frontend/` | React + TypeScript + Vite chat interface |
| `data/` | Approved training samples from the web UI's approval workflow |

## Running

### Development

```bash
# Terminal 1: Backend
cd Client-Web/backend
pip install -r requirements.txt
python main.py
# → http://localhost:8001

# Terminal 2: Frontend
cd Client-Web/frontend
npm install
npm run dev
# → http://localhost:5173 (proxied to backend)
```

### Production

The built frontend (`frontend/dist/`) is served as static files by the FastAPI backend.

```bash
cd Client-Web/frontend && npm run build
cd ../backend && python main.py
# → http://localhost:8001 serves both API and frontend
```

## Backend (`backend/main.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Backend + inference server health |
| `/api/connect` | POST | Authenticate with refresh token, start session |
| `/api/disconnect` | POST | Clear session |
| `/api/chat` | POST | Send message (proxied to inference server) |
| `/api/chat/stream` | POST | Streaming chat (SSE proxy) |
| `/api/config` | GET/PUT | Read/update generation parameters |
| `/api/models` | GET | List available models |
| `/api/models/switch` | POST | Switch active model |
| `/api/approve` | POST | Approve a training data sample |

Session management uses cookies — each browser session gets its own conversation history and API key state.

## Frontend (`frontend/`)

React SPA built with Vite. Key features:

- **Streaming responses** via Server-Sent Events (SSE)
- **Thinking trace visualization** — collapsible `<details>` block that auto-expands during streaming
- **Tool call display** — shows tool names, arguments, results, and execution duration
- **Settings panel** — adjust generation parameters (temperature, top_p, etc.) and switch models
- **Training data approval** — approve good prompt/response pairs for inclusion in training data

### Key Files

| File | Purpose |
|------|---------|
| `src/App.tsx` | Main chat component — message rendering, streaming, settings |
| `src/api.ts` | API client functions (chat, stream, config, models) |
| `src/types.ts` | TypeScript interfaces for all API types |
| `src/main.tsx` | React entry point |
| `vite.config.ts` | Vite config with backend proxy |

### TypeScript Types

Defined in `src/types.ts`:
- `ChatResponse` — model response with `content`, `thinking`, `tool_calls`, `tool_results`, `usage`
- `StreamCallbacks` — SSE event handlers for `onThinkingToken`, `onContentToken`, `onToolResults`, etc.
- `AppConfig` — client-side config (server URL, generation params, model selection)
- `Message` — chat message with optional streaming state fields

## Training Data Approval (`data/`)

The web interface includes an approval workflow: when the model produces a good response, the user can approve it. Approved samples are appended to `data/approved_samples.jsonl` for later inclusion in training data.
