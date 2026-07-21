"""
pipelines/naive_pipeline.py
============================
End-to-end Naive RAG pipeline for the AMAN HR chatbot.

Flow
----
1. On first run, check if ``aman_naive`` Qdrant collection is populated.
   If not, automatically run ingestion (load PDF → chunk → embed → store).
2. Encode the user query with BGE-m3.
3. Retrieve top-k chunks by cosine similarity from Qdrant.
4. Format retrieved chunks into a bilingual prompt.
5. Call Gemini 2.0 Flash to generate the answer.
6. Return a ``PipelineResult`` dataclass with the answer, source documents,
   and timing metadata.

Design notes
------------
- No reranking, no BM25 — pure semantic retrieval.  This is the intentional
  baseline that the Advanced pipeline must outperform.
- The prompt template is bilingual-aware: it instructs the LLM to answer
  in the same language as the question.
- The shared ``PipelineResult`` dataclass is defined here and re-used by
  ``AdvancedPipeline`` and ``evaluation/ragas_eval.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config import settings
from ingestion.embed_store import collection_exists, ingest_naive
from retrievers.naive import NaiveRetriever
from utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """
    Structured result returned by both RAG pipelines.

    Attributes
    ----------
    answer : str
        The LLM-generated answer to the user's question.
    source_documents : list[Document]
        The retrieved (and optionally reranked) Documents used as context.
    pipeline_name : str
        Identifier of the pipeline that produced this result — ``"naive"``
        or ``"advanced"``.
    latency_ms : float
        Total wall-clock time in milliseconds from query receipt to answer.
    query : str
        The original user query (preserved for RAGAS evaluation).
    contexts : list[str]
        Flat list of ``page_content`` strings from ``source_documents``.
        Pre-computed here for convenience (RAGAS, Chainlit display).
    metadata : dict[str, Any]
        Arbitrary extra fields (e.g. retrieval scores, rerank scores).
    """

    answer: str
    source_documents: list[Document]
    pipeline_name: str
    latency_ms: float
    query: str
    contexts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Auto-populate contexts from source_documents if not provided.
        if not self.contexts and self.source_documents:
            self.contexts = [doc.page_content for doc in self.source_documents]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are AMAN HR Assistant, a helpful bilingual assistant for AMAN employees \
and interns. You answer questions strictly based on the provided context \
excerpts from the AMAN Internship Guide.

Rules:
- Answer ONLY using information present in the context below.
- If the context does not contain enough information to answer, say so clearly.
- Respond in the SAME LANGUAGE as the user's question (Arabic or English).
- Be concise and accurate. Do not add information not present in the context.
- When citing specific rules or numbers, quote them exactly as written.
"""

_CONTEXT_TEMPLATE = """\
Context excerpts from the AMAN Internship Guide:
------------------------------------------------------
{context}
------------------------------------------------------

Question: {question}

Answer:"""


def _format_context(documents: list[Document]) -> str:
    """
    Format retrieved Documents into a numbered context block for the prompt.

    Each excerpt is preceded by its source page number and section (if known)
    to help the LLM ground its answer.

    Parameters
    ----------
    documents:
        Retrieved Documents to format as context.

    Returns
    -------
    str
        Formatted multi-excerpt context string.
    """
    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section", "")
        header = f"[Excerpt {i} | Page {page}"
        if section:
            header += f" | Section: {section}"
        header += "]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# NaivePipeline
# ---------------------------------------------------------------------------


class NaivePipeline:
    """
    Naive RAG pipeline: fixed-size chunks + pure semantic retrieval.

    This is the intentional baseline.  It demonstrates the limitations of
    naive chunking and cosine-only retrieval, which the Advanced pipeline
    then improves upon.

    Parameters
    ----------
    auto_ingest:
        If True (default), automatically run ingestion if the ``aman_naive``
        Qdrant collection is not yet populated.  Set to False when you want
        to control ingestion timing explicitly (e.g. in tests).

    Attributes
    ----------
    retriever : NaiveRetriever
    llm : ChatGoogleGenerativeAI
    fallback_llm : ChatOpenAI
    """

    def __init__(self, auto_ingest: bool = True) -> None:
        self._ensure_collection(auto_ingest)

        self.retriever = NaiveRetriever()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
        )
        self.fallback_llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        log.info(
            "naive_pipeline_initialized",
            collection=settings.qdrant_naive_collection,
            llm=settings.gemini_model,
            fallback_llm=settings.openai_model,
        )

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _ensure_collection(auto_ingest: bool) -> None:
        """
        Ensure the naive Qdrant collection is populated.

        Parameters
        ----------
        auto_ingest:
            Whether to run ingestion automatically if the collection is empty.
        """
        collection = settings.qdrant_naive_collection

        if collection_exists(collection):
            log.info("naive_collection_ready", collection=collection)
            return

        if not auto_ingest:
            raise RuntimeError(
                f"Qdrant collection '{collection}' is empty. "
                "Run ingestion.embed_store.ingest_naive() first, or set "
                "auto_ingest=True."
            )

        log.info("naive_collection_empty_auto_ingesting", collection=collection)
        count = ingest_naive()
        log.info("auto_ingest_naive_complete", chunks_indexed=count)

    def _build_prompt(self, query: str, documents: list[Document]) -> list:
        """
        Assemble the chat messages list for the LLM call.

        Parameters
        ----------
        query:
            The user's question.
        documents:
            Retrieved Documents to use as context.

        Returns
        -------
        list[SystemMessage | HumanMessage]
            LangChain message list ready for ``llm.invoke()``.
        """
        context_text = _format_context(documents)
        human_text = _CONTEXT_TEMPLATE.format(
            context=context_text,
            question=query,
        )
        return [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_text),
        ]

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def run(self, query: str) -> PipelineResult:
        """
        Run the full naive RAG pipeline for a user query.

        Steps
        -----
        1. Retrieve top-k Documents by cosine similarity.
        2. Format context + assemble prompt.
        3. Call Gemini 2.0 Flash.
        4. Return a ``PipelineResult``.

        Parameters
        ----------
        query:
            The user's question in English or Arabic.

        Returns
        -------
        PipelineResult
            Structured result containing the answer, source documents,
            latency, and pipeline name.

        Raises
        ------
        ValueError
            If retrieval fails (e.g. Qdrant unavailable).
        RuntimeError
            If the LLM call fails (e.g. invalid API key, quota exceeded).
        """
        log.info("naive_pipeline_run_started", query_preview=query[:80])
        start_time = time.perf_counter()

        # Step 1 — Retrieve.
        documents = self.retriever.retrieve(query, k=settings.retrieval_top_k)

        if not documents:
            log.warning("naive_no_documents_retrieved", query=query)
            return PipelineResult(
                answer=(
                    "I could not find relevant information in the AMAN Internship "
                    "Guide to answer your question. Please rephrase or ask the HR "
                    "team directly."
                ),
                source_documents=[],
                pipeline_name="naive",
                latency_ms=0.0,
                query=query,
            )

        # Step 2 — Assemble prompt.
        messages = self._build_prompt(query, documents)

        # Step 3 — Generate answer (Gemini primary, OpenAI fallback).
        used_model = settings.gemini_model
        try:
            response = self.llm.invoke(messages)
            answer: str = response.content.strip()
        except Exception as gemini_exc:
            log.warning(
                "naive_gemini_failed_trying_openai",
                error=str(gemini_exc),
                fallback=settings.openai_model,
            )
            try:
                response = self.fallback_llm.invoke(messages)
                answer = response.content.strip()
                used_model = settings.openai_model
            except Exception as openai_exc:
                log.exception("naive_fallback_llm_also_failed", error=str(openai_exc))
                raise RuntimeError(
                    f"LLM generation failed on both Gemini and OpenAI. "
                    f"Gemini error: {gemini_exc} | OpenAI error: {openai_exc}"
                ) from openai_exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        log.info(
            "naive_pipeline_run_complete",
            latency_ms=round(elapsed_ms, 1),
            docs_used=len(documents),
            answer_len=len(answer),
        )

        return PipelineResult(
            answer=answer,
            source_documents=documents,
            pipeline_name="naive",
            latency_ms=round(elapsed_ms, 1),
            query=query,
            metadata={
                "retrieval_top_k": settings.retrieval_top_k,
                "llm_model": used_model,
            },
        )
