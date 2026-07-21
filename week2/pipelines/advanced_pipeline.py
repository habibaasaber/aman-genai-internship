"""
pipelines/advanced_pipeline.py
================================
End-to-end Advanced RAG pipeline for the AMAN HR chatbot.

Flow
----
1. On first run, check if ``aman_advanced`` Qdrant collection is populated.
   If not, automatically run smart ingestion.
2. Detect whether the user query is Arabic — log it and normalise accordingly.
3. Retrieve top-k Documents using HybridRetriever (BM25 + dense RRF).
4. Rerank the candidate list with BGEReranker (cross-encoder) → top-n.
5. Format reranked Documents into a bilingual prompt.
6. Call Gemini 2.0 Flash to generate the answer.
7. Return a ``PipelineResult`` with full provenance.

Langfuse tracing
----------------
Every step is instrumented with ``@observe()`` from the Langfuse SDK.
Traces are tagged with:
- ``pipeline``: "advanced"
- ``query_lang``: "ar" or "en"
- ``retrieval_top_k``: number of hybrid candidates
- ``reranker_top_n``: number of documents passed to the LLM

Each step (retrieval, reranking, generation) appears as a child span in
the Langfuse trace, with inputs/outputs and latency captured automatically.
"""

from __future__ import annotations

import os
import time
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langfuse.decorators import langfuse_context, observe

from config import settings
from ingestion.embed_store import collection_exists, ingest_advanced
from pipelines.naive_pipeline import (
    PipelineResult,
    _SYSTEM_PROMPT,
    _format_context,
)
from retrievers.hybrid import HybridRetriever
from retrievers.reranker import BGEReranker
from utils import get_logger, is_arabic

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Langfuse environment bootstrap
# ---------------------------------------------------------------------------
# The Langfuse SDK reads these env vars automatically.  We set them
# explicitly from our typed ``settings`` object to keep a single source of
# truth and support overrides in tests.

os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_base_url)


# ---------------------------------------------------------------------------
# AdvancedPipeline
# ---------------------------------------------------------------------------


class AdvancedPipeline:
    """
    Advanced RAG pipeline: smart chunks, hybrid retrieval, reranking,
    Langfuse tracing.

    Every public method in this class is wrapped with Langfuse ``@observe()``
    so the full trace appears in the Langfuse dashboard, broken down by step.

    Parameters
    ----------
    auto_ingest:
        If True (default), automatically run smart ingestion if the
        ``aman_advanced`` Qdrant collection is not yet populated.

    Attributes
    ----------
    retriever : HybridRetriever
    reranker : BGEReranker
    llm : ChatGoogleGenerativeAI
    fallback_llm : ChatOpenAI
    """

    def __init__(self, auto_ingest: bool = True) -> None:
        self._ensure_collection(auto_ingest)

        self.retriever = HybridRetriever()
        self.reranker = BGEReranker()
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
            "advanced_pipeline_initialized",
            collection=settings.qdrant_advanced_collection,
            llm=settings.gemini_model,
            fallback_llm=settings.openai_model,
        )

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _ensure_collection(auto_ingest: bool) -> None:
        """
        Ensure the advanced Qdrant collection is populated.

        Parameters
        ----------
        auto_ingest:
            Whether to run smart ingestion automatically if the collection
            is empty.
        """
        collection = settings.qdrant_advanced_collection

        if collection_exists(collection):
            log.info("advanced_collection_ready", collection=collection)
            return

        if not auto_ingest:
            raise RuntimeError(
                f"Qdrant collection '{collection}' is empty. "
                "Run ingestion.embed_store.ingest_advanced() first, or set "
                "auto_ingest=True."
            )

        log.info(
            "advanced_collection_empty_auto_ingesting",
            collection=collection,
        )
        count = ingest_advanced()
        log.info("auto_ingest_advanced_complete", chunks_indexed=count)

    def _build_prompt(self, query: str, documents: list[Document]) -> list:
        """
        Assemble the LangChain message list for the LLM call.

        Uses the same system prompt and context formatter as the naive
        pipeline so that the two pipelines differ only in context quality,
        not in how the LLM is instructed.

        Parameters
        ----------
        query:
            The user's question.
        documents:
            Reranked Documents to use as context.

        Returns
        -------
        list[SystemMessage | HumanMessage]
            Chat messages ready for ``llm.invoke()``.
        """
        context_text = _format_context(documents)
        human_text = (
            f"Context excerpts from the AMAN Internship Guide:\n"
            f"------------------------------------------------------\n"
            f"{context_text}\n"
            f"------------------------------------------------------\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        return [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_text),
        ]

    # ---------------------------------------------------------------------- #
    # Traced sub-steps                                                         #
    # ---------------------------------------------------------------------- #

    @observe(name="hybrid_retrieval")
    def _retrieve(self, query: str) -> list[Document]:
        """
        Hybrid BM25 + dense retrieval step (Langfuse traced).

        Parameters
        ----------
        query:
            The user's question.

        Returns
        -------
        list[Document]
            Top-k Documents from hybrid RRF fusion.
        """
        langfuse_context.update_current_observation(
            input=query,
            metadata={
                "retrieval_top_k": settings.retrieval_top_k,
                "collection": settings.qdrant_advanced_collection,
            },
        )

        documents = self.retriever.retrieve(query, k=settings.retrieval_top_k)

        langfuse_context.update_current_observation(
            output=f"Retrieved {len(documents)} documents",
            metadata={
                "top_rrf_score": (
                    documents[0].metadata.get("rrf_score")
                    if documents else None
                ),
            },
        )

        log.info(
            "advanced_retrieval_complete",
            docs_retrieved=len(documents),
        )
        return documents

    @observe(name="reranking")
    def _rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Cross-encoder reranking step (Langfuse traced).

        Parameters
        ----------
        query:
            The original user query.
        documents:
            Candidate Documents from hybrid retrieval.

        Returns
        -------
        list[Document]
            Top-n Documents re-sorted by cross-encoder score.
        """
        langfuse_context.update_current_observation(
            input=f"Reranking {len(documents)} candidates",
            metadata={
                "reranker_model": settings.reranker_model,
                "top_n": settings.reranker_top_n,
            },
        )

        reranked = self.reranker.rerank(
            query, documents, top_n=settings.reranker_top_n
        )

        langfuse_context.update_current_observation(
            output=f"Reranked to {len(reranked)} documents",
            metadata={
                "top_rerank_score": (
                    reranked[0].metadata.get("rerank_score")
                    if reranked else None
                ),
            },
        )

        log.info(
            "advanced_reranking_complete",
            docs_after_rerank=len(reranked),
        )
        return reranked

    @observe(name="llm_generation")
    def _generate(self, query: str, documents: list[Document]) -> str:
        """
        LLM answer generation step (Langfuse traced).

        Parameters
        ----------
        query:
            The user's question.
        documents:
            Reranked Documents to use as context.

        Returns
        -------
        str
            The generated answer text.

        Raises
        ------
        RuntimeError
            If the Gemini API call fails.
        """
        langfuse_context.update_current_observation(
            input=query,
            metadata={
                "llm_model": settings.gemini_model,
                "context_docs": len(documents),
                "temperature": settings.llm_temperature,
            },
        )

        messages = self._build_prompt(query, documents)

        # Try Gemini first, fall back to OpenAI on any failure.
        used_model = settings.gemini_model
        try:
            response = self.llm.invoke(messages)
            answer: str = response.content.strip()
        except Exception as gemini_exc:
            log.warning(
                "advanced_gemini_failed_trying_openai",
                error=str(gemini_exc),
                fallback=settings.openai_model,
            )
            try:
                response = self.fallback_llm.invoke(messages)
                answer = response.content.strip()
                used_model = settings.openai_model
            except Exception as openai_exc:
                log.exception("advanced_fallback_llm_also_failed", error=str(openai_exc))
                raise RuntimeError(
                    f"LLM generation failed on both Gemini and OpenAI. "
                    f"Gemini error: {gemini_exc} | OpenAI error: {openai_exc}"
                ) from openai_exc

        langfuse_context.update_current_observation(
            output=answer[:200],  # Preview in Langfuse (full answer may be long).
            metadata={"answer_length": len(answer), "llm_model": used_model},
        )

        return answer

    # ---------------------------------------------------------------------- #
    # Main entry-point                                                         #
    # ---------------------------------------------------------------------- #

    @observe(name="advanced_rag_pipeline")
    def run(self, query: str) -> PipelineResult:
        """
        Run the full advanced RAG pipeline for a user query.

        This method is the root Langfuse trace.  All three sub-steps
        (retrieval, reranking, generation) appear as child spans in the
        Langfuse UI, tagged with pipeline metadata.

        Steps
        -----
        1. Detect query language (Arabic / English).
        2. Hybrid BM25 + dense retrieval → top-k candidates.
        3. Cross-encoder reranking → top-n candidates.
        4. Gemini generation over reranked context.
        5. Return a ``PipelineResult``.

        Parameters
        ----------
        query:
            The user's question in English or Arabic.

        Returns
        -------
        PipelineResult
            Structured result with answer, source documents, scores, and
            latency.  Full trace visible in Langfuse.

        Raises
        ------
        ValueError
            If hybrid retrieval fails (e.g. Qdrant unavailable).
        RuntimeError
            If LLM generation fails (e.g. API quota exceeded).
        """
        start_time = time.perf_counter()

        # Detect query language for tracing and logging.
        query_lang = "ar" if is_arabic(query) else "en"

        log.info(
            "advanced_pipeline_run_started",
            query_preview=query[:80],
            query_lang=query_lang,
        )

        # Tag the root Langfuse trace with pipeline metadata.
        langfuse_context.update_current_trace(
            name="advanced_rag_pipeline",
            tags=["pipeline:advanced", f"lang:{query_lang}"],
            metadata={
                "pipeline": "advanced",
                "query_lang": query_lang,
                "retrieval_top_k": settings.retrieval_top_k,
                "reranker_top_n": settings.reranker_top_n,
                "llm_model": settings.gemini_model,
                "openai_fallback_model": settings.openai_model,
                "reranker_model": settings.reranker_model,
            },
            input=query,
        )

        # Step 1 — Hybrid retrieval.
        candidate_docs = self._retrieve(query)

        if not candidate_docs:
            log.warning("advanced_no_documents_retrieved", query=query)
            answer = (
                "لم أتمكن من إيجاد معلومات كافية في دليل التدريب للإجابة على سؤالك. "
                "يرجى صياغة السؤال بطريقة مختلفة أو التواصل مع فريق الموارد البشرية مباشرةً."
                if query_lang == "ar"
                else
                "I could not find relevant information in the AMAN Internship "
                "Guide to answer your question. Please rephrase or contact "
                "the HR team directly."
            )
            langfuse_context.update_current_trace(output=answer)
            return PipelineResult(
                answer=answer,
                source_documents=[],
                pipeline_name="advanced",
                latency_ms=0.0,
                query=query,
            )

        # Step 2 — Reranking.
        reranked_docs = self._rerank(query, candidate_docs)

        # Step 3 — Generation.
        answer = self._generate(query, reranked_docs)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Update the root trace output.
        langfuse_context.update_current_trace(output=answer[:200])

        log.info(
            "advanced_pipeline_run_complete",
            latency_ms=round(elapsed_ms, 1),
            query_lang=query_lang,
            docs_retrieved=len(candidate_docs),
            docs_after_rerank=len(reranked_docs),
            answer_len=len(answer),
        )

        return PipelineResult(
            answer=answer,
            source_documents=reranked_docs,
            pipeline_name="advanced",
            latency_ms=round(elapsed_ms, 1),
            query=query,
            metadata={
                "query_lang": query_lang,
                "retrieval_top_k": settings.retrieval_top_k,
                "reranker_top_n": settings.reranker_top_n,
                "llm_model": settings.gemini_model,
                "reranker_model": settings.reranker_model,
                "top_rrf_score": (
                    candidate_docs[0].metadata.get("rrf_score")
                    if candidate_docs else None
                ),
                "top_rerank_score": (
                    reranked_docs[0].metadata.get("rerank_score")
                    if reranked_docs else None
                ),
            },
        )
