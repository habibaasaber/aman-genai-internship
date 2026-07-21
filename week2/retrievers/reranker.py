"""
retrievers/reranker.py
======================
Cross-encoder reranker for the Advanced RAG pipeline.

Uses ``BAAI/bge-reranker-v2-m3`` — a multilingual cross-encoder trained to
score query-document relevance pairs.  Operates after the hybrid retriever
to promote the most relevant candidates and discard noise before generation.

Cross-encoder vs Bi-encoder
----------------------------
Bi-encoders (like bge-m3 used for retrieval) encode query and document
*independently* and compare them with cosine similarity — fast but loses
the fine-grained token-level interaction between query and document.

Cross-encoders encode the query and document *jointly* (concatenated), so
every query token can attend to every document token via full self-attention.
This produces much more accurate relevance scores at the cost of latency.

Cross-encoders are therefore impractical for first-stage retrieval over
thousands of documents but are ideal for reranking a short candidate list
(10 → 4 documents).

Model: BAAI/bge-reranker-v2-m3
- 570 M parameters, multilingual (100+ languages including Arabic).
- Input: concatenated (query, passage) text pairs.
- Output: a single relevance logit (higher = more relevant).
- HuggingFace Hub: https://huggingface.co/BAAI/bge-reranker-v2-m3

Usage
-----
    from retrievers.reranker import BGEReranker
    reranker = BGEReranker()
    top_docs = reranker.rerank(query, candidate_docs, top_n=4)
"""

from __future__ import annotations

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import settings
from utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cross-encoder model singleton
# ---------------------------------------------------------------------------

_reranker_model: CrossEncoder | None = None


def _get_reranker_model() -> CrossEncoder:
    """
    Return the shared BGE cross-encoder model instance.

    Loads ``BAAI/bge-reranker-v2-m3`` on the first call and caches it.
    GPU is used automatically if available; falls back to CPU.

    Returns
    -------
    CrossEncoder
        The loaded and ready-to-use cross-encoder reranker.
    """
    global _reranker_model

    if _reranker_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(
            "reranker_model_loading",
            model=settings.reranker_model,
            device=device,
        )
        _reranker_model = CrossEncoder(
            settings.reranker_model,
            device=device,
            # Use the model's default max_length (512 tokens).
            # BGE-reranker-v2-m3 handles truncation internally.
            max_length=512,
        )
        log.info("reranker_model_loaded", model=settings.reranker_model)

    return _reranker_model


# ---------------------------------------------------------------------------
# BGEReranker class
# ---------------------------------------------------------------------------


class BGEReranker:
    """
    Multilingual cross-encoder reranker using BAAI/bge-reranker-v2-m3.

    Takes a query and a list of candidate Documents (from hybrid retrieval)
    and returns a shorter list sorted by descending cross-encoder relevance
    score.

    Parameters
    ----------
    model_name:
        Override for the reranker model identifier.  Defaults to
        ``settings.reranker_model``.

    Attributes
    ----------
    model_name : str
        The HuggingFace model ID being used.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name: str = model_name or settings.reranker_model
        # Pre-load the model at construction time so the first query
        # doesn't pay the loading cost during user interaction.
        self._model: CrossEncoder = _get_reranker_model()

        log.info("bge_reranker_initialized", model=self.model_name)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """
        Score and re-sort candidate Documents by cross-encoder relevance.

        For each (query, document) pair the cross-encoder produces a logit
        that represents how relevant the document is to the query.  Documents
        are then sorted by descending score and the top-n are returned.

        The rerank score is stored in ``doc.metadata["rerank_score"]`` so
        the Chainlit UI can display it alongside the retrieved source.

        Parameters
        ----------
        query:
            The original user question (English or Arabic, unnormalised).
        documents:
            Candidate Documents from the hybrid retriever.  Order does not
            matter; the reranker re-scores all of them.
        top_n:
            Number of top documents to return.  Defaults to
            ``settings.reranker_top_n``.

        Returns
        -------
        list[Document]
            Top-n Documents sorted by descending cross-encoder score.
            Each Document's metadata includes ``rerank_score`` (float,
            the raw cross-encoder logit — higher means more relevant).

        Notes
        -----
        - If ``documents`` is empty or shorter than ``top_n``, all available
          documents are returned (no padding).
        - The reranker truncates input at 512 tokens internally.  Very long
          chunks may lose tail content — this is handled gracefully by the
          model (not a crash).
        - Cross-encoder scores are *not* probabilities; they are raw logits.
          Comparing scores across different queries is not meaningful.

        Examples
        --------
            reranker = BGEReranker()
            top4 = reranker.rerank("working hours?", candidate_docs, top_n=4)
            for doc in top4:
                print(doc.metadata["rerank_score"], doc.page_content[:60])
        """
        _top_n: int = top_n or settings.reranker_top_n

        if not documents:
            log.warning("reranker_no_documents", query_preview=query[:60])
            return []

        if len(documents) <= 1:
            # Nothing to rerank — return as-is.
            log.debug("reranker_single_document_skip")
            return documents[:_top_n]

        log.info(
            "reranking_started",
            query_preview=query[:80],
            candidate_count=len(documents),
            top_n=_top_n,
        )

        # Build (query, passage) pairs for the cross-encoder.
        # BGE-reranker-v2-m3 expects the query and passage concatenated with
        # a separator — the CrossEncoder class handles this internally when
        # given a list of [query, passage] pairs.
        pairs: list[list[str]] = [
            [query, doc.page_content] for doc in documents
        ]

        # Score all pairs in a single batch call.
        # ``predict`` returns a numpy array of raw logits.
        scores: list[float] = self._model.predict(
            pairs,
            show_progress_bar=False,
        ).tolist()

        # Attach rerank score to each document's metadata.
        scored_docs: list[tuple[float, Document]] = []
        for score, doc in zip(scores, documents, strict=True):
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": round(float(score), 6)},
            )
            scored_docs.append((score, doc_copy))

        # Sort by descending score.
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        top_docs: list[Document] = [doc for _, doc in scored_docs[:_top_n]]

        log.info(
            "reranking_complete",
            returned=len(top_docs),
            top_score=round(scored_docs[0][0], 4) if scored_docs else None,
            bottom_score=round(scored_docs[_top_n - 1][0], 4) if len(scored_docs) >= _top_n else None,
        )

        return top_docs
