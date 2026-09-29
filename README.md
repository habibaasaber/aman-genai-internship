# 🤖 GenAI Internship — AMAN | AI Track

> A 4-week hands-on internship focused on building **production-grade Generative AI systems** — from LLM evaluation pipelines to agentic workflows, RAG systems, and MCP-integrated chatbots.

---

## 📋 Overview

This repository documents the work completed during my internship with the **AI Team at AMAN**. Each week tackled a distinct real-world use case in the Generative AI space, progressing from foundational LLM evaluation all the way to full-stack agentic systems with live API integrations.

| Week | Project | Key Technologies |
|------|---------|-----------------|
| [Week 1](#week-1--prompt-stress-test-suite) | Prompt Stress-Test Suite | OpenAI, Gemini, Prompt Engineering, Evaluation |
| [Week 2](#week-2--production-rag-ask-aman-hr) | Production RAG — "Ask AMAN HR" | RAG, Qdrant, BGE-m3, RAGAS, Langfuse |
| [Week 3](#week-3--automated-talent-pipeline-agent) | Automated Talent Pipeline Agent | LangGraph, Chainlit, Agentic AI |
| [Week 4](#week-4--github-mcp-chatbot) | GitHub MCP Chatbot | MCP, FastAPI, Docker, LiteLLM |

---

## Week 1 — Prompt Stress-Test Suite

**Use Case:** Compare multiple LLMs and prompting strategies on real-world business tasks to identify the best-performing model/prompt combination in a cost-efficient way.

### What Was Built
A production-ready **LLM evaluation framework** that:
- Tests different **LLMs** (OpenAI GPT models, Google Gemini) against the same set of business prompts
- Evaluates multiple **prompting strategies** (zero-shot, few-shot, chain-of-thought, etc.)
- Computes metrics including **token usage**, **cost estimation**, and **quality scores**
- Generates automated reports with visualizations (CSV, Excel, charts, Markdown)
- Uses **local JSON caching** to avoid redundant API calls and save costs

### Key Engineering Decisions
- **Abstract base classes** (`BaseLLMClient`) for clean model swapping without code changes
- **Config-driven** via `config.yaml` — no hard-coded parameters
- **Decoupled prompts** stored as template files, separate from business logic
- Full **SOLID / Clean Architecture** principles applied

### Tech Stack
`OpenAI API` · `Google Gemini API` · `Python` · `Pandas` · `Matplotlib` · `JSON Caching`

📁 [`week1/`](./week1/)

---

## Week 2 — Production RAG: "Ask AMAN HR"

**Use Case:** Build a bilingual (English + Arabic) HR Q&A chatbot that answers intern questions from AMAN's official internship guide — and prove that "Advanced RAG" outperforms "Naive RAG" using rigorous evaluation.

### What Was Built
A fully **bilingual RAG chatbot** with two competing pipelines, evaluated head-to-head:

| Feature | Naive RAG | Advanced RAG |
|---------|-----------|--------------|
| Chunking | Fixed 512-char | Smart (block + bilingual pairs) |
| Retrieval | Dense-only (cosine) | BM25 + Dense → RRF Fusion |
| Reranking | ❌ | ✅ BGE cross-encoder |
| Arabic Support | Raw query | Normalised (diacritics stripped) |
| Tracing | ❌ | ✅ Langfuse `@observe()` |

- **Embeddings:** BAAI/bge-m3 (multilingual, 1024-dim)
- **Vector Store:** Qdrant (local Docker)
- **Evaluation:** RAGAS metrics — Faithfulness, Answer Relevancy, Context Recall
- **UI:** Interactive Chainlit chatbot with pipeline selector
- **Observability:** Full Langfuse tracing per pipeline step

### Tech Stack
`Qdrant` · `BAAI/bge-m3` · `BAAI/bge-reranker-v2-m3` · `BM25` · `RAGAS` · `Langfuse` · `Chainlit` · `Google Gemini 2.0 Flash` · `Docker`

📁 [`week2/`](./week2/)

---

## Week 3 — Automated Talent Pipeline Agent

**Use Case:** Automate the initial resume screening process using an AI agent that evaluates candidates against a job description and produces structured ATS-ready payloads — eliminating manual pre-screening effort.

### What Was Built
A **LangGraph agentic pipeline** that:
1. Accepts a resume (PDF / DOCX / TXT) and a Job Description
2. Extracts structured data from both using an LLM (`CandidateExtraction`, `JobRequirements` schemas)
3. Runs through a **stateful graph** with conditional routing:
   - **Pre-screening** → knock-out checks (degree, min. experience)
   - **Skill Analysis** → scores candidate 0–100 vs. required skills
   - **Disposition** → routes to Interview / Phone Screen / Rejected
4. Generates a downloadable **ATS JSON payload**

### LangGraph Pipeline Flow
```
Resume + JD
    ↓
ingestion_prescreening
    ├── FAIL → rejection → ATS Payload
    └── PASS → skill_analysis
                  ├── score > 80 → interview → ATS Payload
                  ├── 50–80     → phone_screen → ATS Payload
                  └── < 50      → rejection → ATS Payload
```

- **Handles unstructured resumes** — story-like paragraphs, mixed formats, no fixed sections required
- **Chainlit UI** for interactive use (upload resume → paste JD → get recommendation)
- **Regex fallback extractor** when no API key is set (fully functional offline)
- **Full test suite** with pytest (routing, node, and end-to-end tests)

### Tech Stack
`LangGraph` · `Chainlit` · `PyMuPDF` · `python-docx` · `Google Gemini` · `OpenAI` · `pytest`

📁 [`week 3/`](./week%203/)

---

## Week 4 — GitHub MCP Chatbot

**Use Case:** Build a production-grade AI chatbot that can read from and write to GitHub in real-time using the **Model Context Protocol (MCP)** — with full observability, retry/fallback resilience, and human-in-the-loop approval for write actions.

### What Was Built
A **full-stack agentic system** with:
- **Chainlit frontend** (port 8000) — streaming chat UI with login, tool step display, and approval dialogs
- **FastAPI backend** (port 8001) — ReAct agent loop with SSE streaming
- **GitHub MCP Server** — live GitHub integration (list repos, search/get/create issues, add comments)
- **LiteLLM** unified client — OpenAI GPT-4o (primary) + Gemini 1.5 Flash (automatic fallback)
- **Langfuse** full observability — every token, latency, retry, fallback, and tool call traced

### Key Production Engineering Features
- 🔄 **Retry with exponential backoff** — up to 3 retries with full jitter before fallback
- ⚡ **Automatic LLM fallback** — seamless switch from OpenAI → Gemini on failure
- ✅ **Human-in-the-loop** — write actions (create issue, add comment) require user approval before execution
- 🔒 **Graceful error handling** — no stack traces exposed to users; friendly error messages
- 🐳 **Docker Compose** — one-command system startup
- 📊 **Full observability** — token cost, latency, agent decisions, MCP spans all in Langfuse

### Architecture
```
Chainlit UI (8000)
       ↓ HTTP/SSE
FastAPI Backend (8001)
   ├── ReAct Agent Loop
   ├── LLM Client (GPT-4o → Gemini fallback)
   ├── MCP Client → GitHub MCP Server (npx)
   │                    └── GitHub REST API
   └── Langfuse Tracing
```

### Tech Stack
`FastAPI` · `Chainlit` · `OpenAI GPT-4o` · `Google Gemini 1.5 Flash` · `LiteLLM` · `MCP (Model Context Protocol)` · `Langfuse` · `Docker Compose` · `GitHub REST API`

📁 [`week4/`](./week4/)

---

## 🛠️ Skills & Technologies Covered

**LLM Engineering**
- Prompt engineering & evaluation (zero-shot, few-shot, chain-of-thought)
- Multi-model comparison (OpenAI GPT series, Google Gemini)
- Cost estimation & token optimization

**RAG Systems**
- Document ingestion, chunking strategies, and embedding
- Dense + sparse retrieval with RRF fusion
- Cross-encoder reranking
- Multilingual (Arabic + English) support
- RAGAS evaluation framework

**Agentic AI**
- LangGraph stateful agents with conditional routing
- ReAct agent loops
- Model Context Protocol (MCP) integration
- Human-in-the-loop approval patterns

**Production Engineering**
- Observability & tracing (Langfuse)
- Retry logic with exponential backoff
- LLM fallback strategies
- REST API design (FastAPI)
- Containerization (Docker & Docker Compose)
- Testing (pytest)

---

## 🏢 About

This internship was completed as part of the **AMAN GenAI Program** with the **AI Track team**.

Each weekly project was designed to simulate real production engineering challenges — not just tutorials or toy demos, but systems built with clean architecture, evaluation rigor, and production reliability in mind.
