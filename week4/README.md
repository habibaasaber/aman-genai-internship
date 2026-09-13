# GitHub MCP Chatbot — Week 4

A **production-grade AI chatbot** that combines a streaming LLM conversational interface with live GitHub integration via the **Model Context Protocol (MCP)**. Built as a demonstration of real production engineering: reliability, observability, resilience, and graceful degradation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Chainlit UI  (port 8000)                        │
│  Login → Chat → Streaming tokens → Tool steps → Approval dialogs   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP POST + Server-Sent Events (SSE)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FastAPI Backend  (port 8001)                       │
│                                                                      │
│   POST /chat/stream          POST /chat/approve                      │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                      AI Agent Loop                            │  │
│  │  1. Non-streaming LLM call → tool decision                   │  │
│  │  2. Execute read tools immediately (no approval)              │  │
│  │  3. Write tools → pause, emit approval_needed event           │  │
│  │  4. Resume after user confirms → execute write + stream reply │  │
│  └──────────────────┬───────────────────────┬─────────────────── ┘  │
│                     │                       │                        │
│  ┌──────────────────▼────┐    ┌─────────────▼───────────────────┐   │
│  │     LLM Client        │    │   MCP Client (stdio subprocess)  │   │
│  │  ┌────────────────┐   │    │   GitHub MCP Server (Node/npx)  │   │
│  │  │  OpenAI gpt-4o │   │    │   ─ list_repositories           │   │
│  │  │   (primary)    │   │    │   ─ search_issues               │   │
│  │  └────────┬───────┘   │    │   ─ get_issue                   │   │
│  │  Retry ×3 │ backoff   │    │   ─ create_issue (write)        │   │
│  │  ┌────────▼────────┐  │    │   ─ add_issue_comment (write)   │   │
│  │  │ Gemini 1.5 Flash│  │    └─────────────────┬───────────────┘   │
│  │  │   (fallback)    │  │                      │                    │
│  │  └─────────────────┘  │               GitHub REST API             │
│  └───────────────────────┘                                           │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │              Langfuse Observability Layer                      │  │
│  │  Trace: User Request → LLM Call → Agent Decision →            │  │
│  │         MCP Tool Call → GitHub Operation → Final Response      │  │
│  │  Captures: tokens, latency, cost, retries, fallback events    │  │
│  └────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
week4/
├── docker-compose.yml          # One-command system startup
├── .env.example                # Environment variable template
├── README.md                   # This file
│
├── frontend/                   # Chainlit UI
│   ├── app.py                  # Main Chainlit app (auth, streaming, approval UI)
│   ├── chainlit.md             # Welcome page content
│   ├── .chainlit/
│   │   └── config.toml         # Chainlit configuration (auth, theme)
│   ├── requirements.txt
│   └── Dockerfile
│
├── backend/                    # FastAPI backend
│   ├── main.py                 # App entry point, lifespan hooks
│   ├── config.py               # Pydantic settings (all env vars)
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── models/
│   │   └── schemas.py          # Pydantic models (request/response/SSE)
│   ├── routers/
│   │   └── chat.py             # POST /chat/stream, POST /chat/approve
│   ├── agent/
│   │   ├── agent.py            # ReAct agent loop
│   │   └── tools.py            # Tool schemas + read/write classification
│   ├── llm/
│   │   ├── client.py           # Unified LLM client (OpenAI + Gemini via LiteLLM)
│   │   └── retry.py            # Exponential backoff retry utility
│   ├── mcp_client/
│   │   └── client.py           # GitHub MCP server subprocess manager
│   └── observability/
│       └── langfuse_client.py  # Langfuse tracing wrapper
│
└── mcp-server/
    ├── README.md               # GitHub MCP server documentation
    └── config.json             # Reference config (for inspector testing)
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in all values:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key (`sk-...`) |
| `OPENAI_MODEL` | No | OpenAI model (default: `gpt-4o`) |
| `GEMINI_API_KEY` | ✅ | Google AI Studio API key (`AIza...`) |
| `GEMINI_MODEL` | No | Gemini model (default: `gemini-1.5-flash`) |
| `MAX_RETRIES` | No | Retry attempts before fallback (default: `3`) |
| `RETRY_BASE_DELAY` | No | Base backoff delay seconds (default: `1.0`) |
| `RETRY_MAX_DELAY` | No | Max backoff delay seconds (default: `30.0`) |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | ✅ | GitHub PAT with `repo`, `read:org`, `read:user` |
| `LANGFUSE_PUBLIC_KEY` | ✅ | Langfuse public key (`pk-lf-...`) |
| `LANGFUSE_SECRET_KEY` | ✅ | Langfuse secret key (`sk-lf-...`) |
| `LANGFUSE_HOST` | No | Langfuse host (default: `https://cloud.langfuse.com`) |
| `LANGFUSE_ENABLED` | No | Enable tracing (default: `true`) |
| `CHAINLIT_AUTH_SECRET` | ✅ | Random secret for auth token signing |
| `CHAINLIT_USERS` | No | `user1:pass1,user2:pass2` pairs (default: `admin:admin123,demo:demo456`) |
| `BACKEND_URL` | No | Backend URL seen by Chainlit (default: `http://localhost:8001`) |

---

## Setup & Running

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm/npx
- Docker & Docker Compose (for Docker run)

---

### Option A — Local Run

```bash
# 1. Clone / navigate to week4
cd "week4"

# 2. Set up environment
cp .env.example .env
# Edit .env and fill in all API keys

# 3. Start the FastAPI backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8001

# 4. In a new terminal, start Chainlit
cd ../frontend
pip install -r requirements.txt
# Copy .env to frontend/ or ensure BACKEND_URL is set
cp ../.env .env
chainlit run app.py --port 8000
```

Open http://localhost:8000 and log in with `admin` / `admin123`.

---

### Option B — Docker Compose

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env and fill in all API keys

# 2. Build and start everything
docker compose up --build

# 3. Open the UI
open http://localhost:8000
```

> **Note:** First startup may take ~60 seconds as Docker pulls Node.js and the
> `@modelcontextprotocol/server-github` npm package. Subsequent starts are faster
> due to Docker layer caching.

To stop: `docker compose down`

---

## Demo Scenarios

### Scenario 1 — Normal Conversational Request

> No GitHub tools involved. Full Langfuse trace with one LLM call.

**Steps:**
1. Log in and start a chat
2. Ask: *"What is exponential backoff and why is it used in distributed systems?"*
3. Observe: the assistant answers directly, no tool steps appear in the UI
4. **Langfuse:** open your Langfuse dashboard → Traces → find the trace for this session → verify one `agent_decision` LLM generation and no tool spans

---

### Scenario 2 — GitHub MCP Workflow

> Full trace: Agent Decision → MCP Tool Call → GitHub Result → Final Response.

**Steps:**
1. Ask: *"Find open issues related to authentication in my repos"*
   - Or: *"List my repositories"*
2. Observe: a tool step appears (🔧 **Search Issues**), showing input + output
3. The agent summarises the results in natural language

**Langfuse trace shows:**
- `agent_decision` generation (model chose `search_issues` tool)
- `mcp_tool:search_issues` span with latency and GitHub output
- Final streaming generation

**Write-action demo:**
1. Ask: *"Create an issue in myusername/myrepo about the login bug"*
2. Agent shows a **draft issue** (title + body)
3. An approval dialog appears: ✅ Approve or ❌ Cancel
4. If approved → issue created on GitHub

---

### Scenario 3 — LLM Failure → Retry → Fallback

> Demonstrates retry with exponential backoff and automatic Gemini fallback.

**Steps:**

```bash
# Option A: set an invalid key before starting
OPENAI_API_KEY=invalid docker compose up

# Option B: edit .env while running
# Change OPENAI_API_KEY=invalid, then restart the backend only:
docker compose restart backend
```

1. Send any message (e.g., *"Hello"*)
2. Observe in Chainlit: retry warning toasts may appear briefly
3. A banner appears: ⚡ *Using fallback: gemini-1.5-flash*
4. The response arrives from Gemini successfully

**Langfuse trace shows:**
- `llm_retry` events for each attempt (up to MAX_RETRIES)
- `llm_fallback` event with `from_model=openai`, `to_model=gemini/gemini-1.5-flash`
- A Gemini generation completing successfully

---

### Scenario 4 — Clean Failure Handling

> Invalid GitHub repo or bad request → friendly user message, no stack trace.

**Steps:**
1. Ask: *"Get issue #1 from nonexistent-org/nonexistent-repo"*
2. Observe: the UI shows a friendly error like:
   > 🐙 GitHub error: Not Found — the repository does not exist or you don't have access.
3. No stack trace is shown in the UI

---

## Retry & Fallback Configuration

| Setting | Default | Notes |
|---------|---------|-------|
| `MAX_RETRIES` | `3` | Total extra attempts after first failure |
| `RETRY_BASE_DELAY` | `1.0s` | Delay = `base * 2^attempt` (full jitter) |
| `RETRY_MAX_DELAY` | `30.0s` | Hard cap on per-retry sleep |
| Fallback trigger | After `RetriesExhaustedError` on *retriable* errors only | Auth errors, bad requests → no fallback |

**Retriable errors:** HTTP 429, 500, 502, 503, 504; timeouts; connection resets; "overloaded" messages.

**Non-retriable errors:** HTTP 401, 403 (auth); HTTP 400 (bad request); invalid user input.

---

## Observability

All traces are visible in your [Langfuse dashboard](https://cloud.langfuse.com).

Each trace captures:
- **User / session info** — user ID, session ID, input message
- **LLM generations** — model used, prompt/completion tokens, latency, estimated cost (USD)
- **Agent decision** — which tools were selected and why
- **MCP tool spans** — tool name, input arguments, output text, latency
- **Retry events** — attempt number, error type/message
- **Fallback events** — from/to model, reason
- **Error events** — error type and message with context
- **Write approval events** — tool name, draft content, approved/rejected

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| UI | [Chainlit](https://chainlit.io) |
| Backend API | [FastAPI](https://fastapi.tiangolo.com) + [uvicorn](https://www.uvicorn.org) |
| LLM (primary) | OpenAI gpt-4o via [LiteLLM](https://github.com/BerriAI/litellm) |
| LLM (fallback) | Google Gemini 1.5 Flash via LiteLLM |
| GitHub Tools | [@modelcontextprotocol/server-github](https://github.com/modelcontextprotocol/servers/tree/main/src/github) (npm) |
| MCP SDK | [mcp](https://github.com/modelcontextprotocol/python-sdk) (Python) |
| Observability | [Langfuse](https://langfuse.com) |
| Deployment | Docker & Docker Compose |
