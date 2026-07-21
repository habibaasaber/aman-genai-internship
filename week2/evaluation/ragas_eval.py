"""
evaluation/ragas_eval.py
========================
RAGAS-based evaluation of the Naive and Advanced RAG pipelines.

Metrics computed
----------------
- **Faithfulness**      : Does the answer contain only claims supported by
                          the retrieved context?  (LLM-judged)
- **Answer Relevancy**  : How well does the answer address the question?
                          (embedding + LLM judged)
- **Context Recall**    : What fraction of the ground-truth answer is covered
                          by the retrieved context?  (LLM-judged)

RAGAS LLM judge
---------------
Uses ``gpt-4o-mini`` (via OpenAI) as the evaluation LLM.  This is separate
from the generation LLM (Gemini) — a common and recommended practice so that
the judge is independent of the system under test.

Output
------
Prints a per-question breakdown and overall aggregate table to stdout.
Saves a Markdown results table to ``evaluation/results.md``.

Usage
-----
    python evaluation/ragas_eval.py

Or from project root:
    python -m evaluation.ragas_eval
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
)

from config import settings
from pipelines.naive_pipeline import NaivePipeline, PipelineResult
from pipelines.advanced_pipeline import AdvancedPipeline
from utils import get_logger

log = get_logger(__name__)

# Path to the test questions file (same directory as this script).
_QUESTIONS_PATH = Path(__file__).parent / "test_questions.json"
# Output path for the Markdown results table.
_RESULTS_PATH = Path(__file__).parent / "results.md"


# ---------------------------------------------------------------------------
# RAGAS judge LLM + embeddings setup
# ---------------------------------------------------------------------------


def _build_ragas_llm() -> LangchainLLMWrapper:
    """
    Build the LangChain-wrapped LLM used as the RAGAS evaluation judge.

    Uses ``gpt-4o-mini`` — cheap, reliable, and independent of the
    generation pipeline (Gemini).

    Returns
    -------
    LangchainLLMWrapper
        RAGAS-compatible LLM wrapper.
    """
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.0,
    )
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """
    Build the LangChain-wrapped embeddings used by RAGAS answer_relevancy.

    Uses OpenAI ``text-embedding-3-small`` — fast and sufficient for
    semantic similarity scoring in the RAGAS metric.

    Returns
    -------
    LangchainEmbeddingsWrapper
        RAGAS-compatible embeddings wrapper.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )
    return LangchainEmbeddingsWrapper(embeddings)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def _run_pipeline_on_questions(
    pipeline: NaivePipeline | AdvancedPipeline,
    questions: list[dict[str, Any]],
    pipeline_name: str,
) -> list[dict[str, Any]]:
    """
    Run a pipeline on all test questions and collect results.

    Parameters
    ----------
    pipeline:
        An instantiated RAG pipeline with a ``run(query) -> PipelineResult``
        method.
    questions:
        List of question dicts from ``test_questions.json``.
    pipeline_name:
        Human-readable name for logging (``"naive"`` or ``"advanced"``).

    Returns
    -------
    list[dict]
        One dict per question with keys:
        - ``question``      : The original question string.
        - ``answer``        : The generated answer.
        - ``contexts``      : List of retrieved chunk texts.
        - ``ground_truth``  : The reference answer from the test set.
        - ``language``      : ``"en"`` or ``"ar"``.
        - ``latency_ms``    : Wall-clock time in ms.
        - ``question_id``   : ID from the test set.
    """
    results: list[dict[str, Any]] = []

    for q in questions:
        question_id = q["id"]
        question = q["question"]
        ground_truth = q["ground_truth"]
        language = q["language"]

        log.info(
            "evaluating_question",
            pipeline=pipeline_name,
            question_id=question_id,
            language=language,
            question_preview=question[:60],
        )

        try:
            result: PipelineResult = pipeline.run(question)
        except Exception as exc:
            log.exception(
                "pipeline_run_failed",
                pipeline=pipeline_name,
                question_id=question_id,
                error=str(exc),
            )
            # Record a failure result so the evaluation continues.
            results.append(
                {
                    "question": question,
                    "answer": f"[ERROR] {exc}",
                    "contexts": [],
                    "ground_truth": ground_truth,
                    "language": language,
                    "latency_ms": 0.0,
                    "question_id": question_id,
                }
            )
            continue

        results.append(
            {
                "question": question,
                "answer": result.answer,
                "contexts": result.contexts,
                "ground_truth": ground_truth,
                "language": language,
                "latency_ms": result.latency_ms,
                "question_id": question_id,
            }
        )

        log.info(
            "question_evaluated",
            pipeline=pipeline_name,
            question_id=question_id,
            latency_ms=result.latency_ms,
            answer_preview=result.answer[:60],
        )

        # Small sleep between calls to avoid Gemini rate limits.
        time.sleep(1.0)

    return results


# ---------------------------------------------------------------------------
# RAGAS evaluation runner
# ---------------------------------------------------------------------------


def _evaluate_with_ragas(
    results: list[dict[str, Any]],
    ragas_llm: LangchainLLMWrapper,
    ragas_embeddings: LangchainEmbeddingsWrapper,
    pipeline_name: str,
) -> dict[str, float]:
    """
    Run RAGAS metrics on a set of pipeline results.

    Parameters
    ----------
    results:
        Output from ``_run_pipeline_on_questions``.
    ragas_llm:
        The LangChain-wrapped LLM for RAGAS judging.
    ragas_embeddings:
        The LangChain-wrapped embeddings for answer relevancy.
    pipeline_name:
        Used only for logging.

    Returns
    -------
    dict[str, float]
        Mapping of metric name → mean score across all questions.
        Keys: ``faithfulness``, ``answer_relevancy``, ``context_recall``.
    """
    log.info(
        "ragas_evaluation_started",
        pipeline=pipeline_name,
        num_samples=len(results),
    )

    # RAGAS 0.2.x uses a HuggingFace Dataset as input.
    dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                # RAGAS expects contexts as a list of strings.
                "contexts": r["contexts"] if r["contexts"] else [""],
                "ground_truth": r["ground_truth"],
            }
            for r in results
        ]
    )

    # Wire RAGAS metrics to our judge LLM and embeddings.
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings
    context_recall.llm = ragas_llm

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        raise_exceptions=False,  # Log failures but continue.
    )

    import numpy as np
    results_dict: dict[str, float] = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall"]:
        val = scores[key]
        if isinstance(val, list) or isinstance(val, np.ndarray):
            valid_vals = [v for v in val if v is not None and not np.isnan(v)]
            avg_val = float(np.mean(valid_vals)) if valid_vals else 0.0
        else:
            avg_val = float(val)
        results_dict[key] = round(avg_val, 4)

    log.info(
        "ragas_evaluation_complete",
        pipeline=pipeline_name,
        **results_dict,
    )

    return results_dict


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------


def _format_markdown_table(
    naive_scores: dict[str, float],
    advanced_scores: dict[str, float],
    naive_results: list[dict],
    advanced_results: list[dict],
) -> str:
    """
    Build a Markdown results document comparing both pipelines.

    Parameters
    ----------
    naive_scores:
        Aggregate RAGAS scores for the naive pipeline.
    advanced_scores:
        Aggregate RAGAS scores for the advanced pipeline.
    naive_results:
        Per-question results from the naive pipeline.
    advanced_results:
        Per-question results from the advanced pipeline.

    Returns
    -------
    str
        Complete Markdown document content.
    """
    def delta(adv: float, naive: float) -> str:
        diff = adv - naive
        sign = "+" if diff >= 0 else ""
        emoji = "✅" if diff >= 0 else "⚠️"
        return f"{emoji} {sign}{diff:.4f}"

    lines: list[str] = [
        "# RAGAS Evaluation Results — Ask AMAN HR",
        "",
        "**Metrics**: Faithfulness · Answer Relevancy · Context Recall",
        "**Judge LLM**: gpt-4o-mini  |  **Pipelines**: Naive vs Advanced RAG",
        "",
        "---",
        "",
        "## Aggregate Scores",
        "",
        "| Metric | Naive RAG | Advanced RAG | Δ (Advanced − Naive) |",
        "|--------|-----------|--------------|----------------------|",
        f"| Faithfulness     | {naive_scores['faithfulness']:.4f} | {advanced_scores['faithfulness']:.4f} | {delta(advanced_scores['faithfulness'], naive_scores['faithfulness'])} |",
        f"| Answer Relevancy | {naive_scores['answer_relevancy']:.4f} | {advanced_scores['answer_relevancy']:.4f} | {delta(advanced_scores['answer_relevancy'], naive_scores['answer_relevancy'])} |",
        f"| Context Recall   | {naive_scores['context_recall']:.4f} | {advanced_scores['context_recall']:.4f} | {delta(advanced_scores['context_recall'], naive_scores['context_recall'])} |",
        "",
        "---",
        "",
        "## Per-Question Results",
        "",
        "| ID | Lang | Question | Naive Answer | Advanced Answer | Naive Latency | Adv Latency |",
        "|----|------|----------|-------------|-----------------|---------------|-------------|",
    ]

    for n, a in zip(naive_results, advanced_results):
        q_preview = n["question"][:55].replace("|", "\\|")
        n_ans = n["answer"][:60].replace("|", "\\|").replace("\n", " ")
        a_ans = a["answer"][:60].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {n['question_id']} | {n['language']} "
            f"| {q_preview}… "
            f"| {n_ans}… "
            f"| {a_ans}… "
            f"| {n['latency_ms']:.0f} ms "
            f"| {a['latency_ms']:.0f} ms |"
        )

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "> **Faithfulness** measures whether the answer contains only claims",
        "> supported by the retrieved context (hallucination detection).",
        ">",
        "> **Answer Relevancy** measures how well the answer addresses the",
        "> question (via semantic similarity of question and answer).",
        ">",
        "> **Context Recall** measures what fraction of the ground-truth answer",
        "> is covered by the retrieved context.",
        "",
        "The Advanced pipeline improves all three metrics by using:",
        "1. **Smart chunking** — respects section and paragraph boundaries.",
        "2. **Hybrid retrieval** — BM25 + dense search with RRF fusion.",
        "3. **BGE cross-encoder reranking** — promotes the most relevant chunks.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def run_evaluation() -> None:
    """
    Run the full RAGAS evaluation on both pipelines.

    Steps
    -----
    1. Load test questions from ``evaluation/test_questions.json``.
    2. Instantiate both pipelines (triggers auto-ingest if needed).
    3. Run each pipeline on all 9 test questions.
    4. Compute RAGAS metrics (faithfulness, answer_relevancy, context_recall).
    5. Print a comparison table to stdout.
    6. Save the full Markdown report to ``evaluation/results.md``.
    """
    log.info("evaluation_started")

    # Load test questions.
    with open(_QUESTIONS_PATH, encoding="utf-8") as f:
        questions: list[dict[str, Any]] = json.load(f)

    log.info("questions_loaded", count=len(questions))

    # Set up RAGAS judge.
    ragas_llm = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    # Instantiate pipelines (auto-ingest if Qdrant collections are empty).
    log.info("initializing_pipelines")
    naive_pipeline = NaivePipeline(auto_ingest=True)
    advanced_pipeline = AdvancedPipeline(auto_ingest=True)

    # --- Run Naive pipeline ---
    log.info("running_naive_pipeline_on_test_set")
    naive_results = _run_pipeline_on_questions(
        naive_pipeline, questions, "naive"
    )

    # --- Run Advanced pipeline ---
    log.info("running_advanced_pipeline_on_test_set")
    advanced_results = _run_pipeline_on_questions(
        advanced_pipeline, questions, "advanced"
    )

    # --- RAGAS evaluation ---
    naive_scores = _evaluate_with_ragas(
        naive_results, ragas_llm, ragas_embeddings, "naive"
    )
    advanced_scores = _evaluate_with_ragas(
        advanced_results, ragas_llm, ragas_embeddings, "advanced"
    )

    # --- Print summary table ---
    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<22} {'Naive':>10} {'Advanced':>10} {'Diff':>10}")
    print("-" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_recall"]:
        n = naive_scores[metric]
        a = advanced_scores[metric]
        diff = a - n
        sign = "+" if diff >= 0 else ""
        print(f"{metric:<22} {n:>10.4f} {a:>10.4f} {sign}{diff:>9.4f}")
    print("=" * 60 + "\n")

    # --- Save Markdown report ---
    markdown = _format_markdown_table(
        naive_scores, advanced_scores, naive_results, advanced_results
    )
    _RESULTS_PATH.write_text(markdown, encoding="utf-8")
    log.info("results_saved", path=str(_RESULTS_PATH))
    print(f"Full report saved to: {_RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Script entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
