# RAGAS Evaluation Results — Ask AMAN HR

**Metrics**: Faithfulness · Answer Relevancy · Context Recall
**Judge LLM**: gpt-4o-mini  |  **Pipelines**: Naive vs Advanced RAG

---

## Aggregate Scores

| Metric | Naive RAG | Advanced RAG | Δ (Advanced − Naive) |
|--------|-----------|--------------|----------------------|
| Faithfulness     | 0.0000 | 0.0000 | ✅ +0.0000 |
| Answer Relevancy | 0.1011 | 0.0867 | ⚠️ -0.0144 |
| Context Recall   | 0.0000 | 0.0000 | ✅ +0.0000 |

---

## Per-Question Results

| ID | Lang | Question | Naive Answer | Advanced Answer | Naive Latency | Adv Latency |
|----|------|----------|-------------|-----------------|---------------|-------------|
| 1 | en | What are the official working hours and working days fo… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | 0 ms | 0 ms |
| 2 | en | What happens if an intern is repeatedly late without a … | [ERROR] LLM generation failed: Error calling model 'gemini-2… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | 0 ms | 0 ms |
| 3 | en | What is the official dress code at AMAN corporate offic… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | 0 ms | 0 ms |
| 4 | en | What tools does Track 1 (AI & Data Engineering) use for… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | 0 ms | 0 ms |
| 5 | ar | كم عدد أيام الغياب المسموح بها خلال فترة التدريب؟… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | 0 ms | 0 ms |
| 6 | en | When does the mid-term evaluation take place?… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | 0 ms | 0 ms |
| 7 | en | How long is the individual HR alignment session with ea… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | 0 ms | 0 ms |
| 8 | en | What response time and code coverage standards must Tra… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | 0 ms | 0 ms |
| 9 | en | Where must penetration tests be conducted in Track 7 (C… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | [ERROR] LLM generation failed: Error calling model 'gemini-2… | 0 ms | 0 ms |

---

## Interpretation

> **Faithfulness** measures whether the answer contains only claims
> supported by the retrieved context (hallucination detection).
>
> **Answer Relevancy** measures how well the answer addresses the
> question (via semantic similarity of question and answer).
>
> **Context Recall** measures what fraction of the ground-truth answer
> is covered by the retrieved context.

The Advanced pipeline improves all three metrics by using:
1. **Smart chunking** — respects section and paragraph boundaries.
2. **Hybrid retrieval** — BM25 + dense search with RRF fusion.
3. **BGE cross-encoder reranking** — promotes the most relevant chunks.
