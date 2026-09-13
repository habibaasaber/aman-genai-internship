# Week 2 — Production RAG: "Ask AMAN HR"

> Bilingual (EN/AR) HR Q&A chatbot powered by two RAG pipelines, evaluated with RAGAS.

---

## Architecture

```
week2/
├── app.py                        # Chainlit UI — pipeline selector
├── config.py                     # Pydantic-settings — single source of truth
├── requirements.txt              # All dependencies
├── .env.example                  # Environment variable template
│
├── data/
│   └── aman_internship_guide_2026.docx
│
├── ingestion/
│   ├── pdf_loader.py             # PyMuPDF → List[Document] with metadata
│   ├── chunking.py               # Fixed-size (naive) + smart (advanced) chunkers
│   └── embed_store.py            # BGE-m3 encoding → Qdrant upsert
│
├── retrievers/
│   ├── naive.py                  # Dense-only Qdrant retriever
│   ├── hybrid.py                 # BM25 + dense → RRF fusion
│   └── reranker.py               # BGE cross-encoder reranker
│
├── pipelines/
│   ├── naive_pipeline.py         # End-to-end naive RAG
│   └── advanced_pipeline.py      # End-to-end advanced RAG + Langfuse tracing
│
├── evaluation/
│   ├── test_questions.json       # 9 bilingual QA pairs
│   ├── ragas_eval.py             # RAGAS evaluation runner
│   └── results.md                # Auto-generated comparison table
│
└── utils/
    ├── logger.py                 # Structlog configuration
    └── arabic.py                 # Diacritic stripping + normalisation
```

---

## Pipelines

| Feature | Naive RAG | Advanced RAG |
|---|---|---|
| **Chunking** | Fixed 512-char | Smart (block + bilingual pairs) |
| **Embedding** | BAAI/bge-m3 | BAAI/bge-m3 |
| **Retrieval** | Dense-only (cosine) | BM25 + Dense → RRF |
| **Reranking** | ❌ | ✅ BAAI/bge-reranker-v2-m3 |
| **Arabic** | Raw query | Normalised (diacritics stripped) |
| **Tracing** | ❌ | ✅ Langfuse `@observe()` |

---

## Setup

### 1. Prerequisites

```bash
# Qdrant (Docker required)
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Python 3.12+
python --version
```

### 2. Install dependencies

```bash
cd week2
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys:
# GEMINI_API_KEY, OPENAI_API_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY
```

### 4. Run the Chainlit app

```bash
chainlit run app.py
```

Ingestion runs **automatically** on first launch — it will download the BGE-m3 model
(~570 MB) and index the PDF into Qdrant. This takes 2–5 minutes on first run only.

---

## Evaluation

```bash
python evaluation/ragas_eval.py
```

Runs both pipelines on the 9 bilingual test questions and produces
`evaluation/results.md` with the RAGAS comparison table.

---

## RAGAS Metrics

| Metric | Measures |
|---|---|
| **Faithfulness** | Does the answer stay grounded in the retrieved context? |
| **Answer Relevancy** | How well does the answer address the question? |
| **Context Recall** | What fraction of the ground truth is covered by context? |

---

## Sample Questions

| # | Language | Question |
|---|---|---|
| 1 | EN | What are the official working hours for AMAN interns? |
| 2 | EN | What happens if an intern is repeatedly late? |
| 3 | EN | What is the dress code at AMAN offices? |
| 4 | EN | What tools does Track 1 use for orchestration? |
| 5 | AR | كم عدد أيام الغياب المسموح بها؟ |
| 6 | EN | When does the mid-term evaluation take place? |
| 7 | EN | How long is the HR alignment session? |
| 8 | EN | What standards must Track 2 APIs meet? |
| 9 | EN | Where must penetration tests be conducted in Track 7? |

---

## Langfuse Tracing

The Advanced pipeline instruments every step with `@observe()`:

```
advanced_rag_pipeline (root trace)
├── hybrid_retrieval  — BM25+dense candidates, RRF scores
├── reranking         — cross-encoder scores before/after
└── llm_generation    — prompt, answer preview, latency
```

View traces at [cloud.langfuse.com](https://cloud.langfuse.com).

---

## Tech Stack

- **LLM**: Google Gemini 2.0 Flash
- **Embeddings**: BAAI/bge-m3 (1024-dim, multilingual)
- **Reranker**: BAAI/bge-reranker-v2-m3 (cross-encoder)
- **Vector DB**: Qdrant (local Docker)
- **Sparse**: BM25Okapi (rank-bm25)
- **Tracing**: Langfuse
- **Evaluation**: RAGAS (judge: GPT-4o-mini)
- **UI**: Chainlit
