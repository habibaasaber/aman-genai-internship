"""
retrievers/hybrid.py
====================
Hybrid BM25 + dense retriever for the Advanced RAG pipeline.

Architecture
------------
Dense leg  → BAAI/bge-m3 query embedding → Qdrant cosine search (aman_advanced)
Sparse leg → BM25Okapi index over stored chunk texts → top-k by BM25 score
Fusion     → Reciprocal Rank Fusion (RRF) merges both ranked lists

Why RRF instead of score interpolation?
----------------------------------------
BM25 scores and cosine similarities live on entirely different scales and
distributions.  Normalising them before blending introduces a free parameter
(interpolation weight α) that needs tuning per dataset.  RRF avoids this:
it only uses the *rank* of each document in each list, making it robust and
parameter-free.  Formula:
    RRF(d) = Σ  1 / (k + rank_i(d))
where k=60 is a constant that dampens high-rank documents.

Arabic handling
---------------
Before BM25 tokenisation the query is passed through ``normalise_arabic``.
Chunk texts stored in Qdrant contain a ``normalised_content`` payload field
(written by ``smart_chunks``) that was processed with the same normaliser.
This ensures diacritics-free queries match diacritised guide text correctly.

BM25 index lifecycle
--------------------
The index is built lazily on the first ``retrieve`` call by scrolling all
points from the Qdrant ``aman_advanced`` collection.  The index is cached as
an instance attribute so subsequent queries within the same process avoid
rebuilding.  Rebuilding is O(n·|V|) and takes ~1–2 seconds for a small PDF.

Usage
-----
    from retrievers.hybrid import HybridRetriever
    retriever = HybridRetriever()
    docs = retriever.retrieve("ما هي ساعات العمل؟", k=10)
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from config import settings
from ingestion.embed_store import _get_qdrant_client
from retrievers.naive import NaiveRetriever
from utils import get_logger, normalise_arabic, is_arabic

log = get_logger(__name__)

# RRF damping constant — standard value from the original 2009 Cormack paper.
_RRF_K: int = 60


def _tokenise(text: str) -> list[str]:
    """
    Tokenise text for BM25 indexing.

    For Arabic text the input is first normalised (diacritics stripped, alef
    variants unified).  Tokenisation is whitespace + punctuation splitting,
    lowercased for Latin characters.

    Parameters
    ----------
    text:
        Raw or normalised chunk/query text.

    Returns
    -------
    list[str]
        List of non-empty tokens.
    """
    # Normalise Arabic characters then split on whitespace / punctuation.
    normalised = normalise_arabic(text)
    tokens = re.split(r"[\s\u060C\u061B\u061F\u0640،؛؟!?,.\-–—:;\"'()\[\]{}]+", normalised)
    return [t.lower() for t in tokens if t]


class HybridRetriever:
    """
    Hybrid BM25 + dense retriever with Reciprocal Rank Fusion.

    Combines a lexical BM25 index with dense Qdrant semantic search and
    merges results using RRF.  Supports bilingual (EN/AR) queries without
    configuration changes.

    Parameters
    ----------
    collection_name:
        Qdrant collection to search.  Defaults to
        ``settings.qdrant_advanced_collection`` (``"aman_advanced"``).
    rrf_k:
        RRF damping constant.  Higher values reduce the influence of
        top-ranked documents; 60 is the standard default.

    Attributes
    ----------
    collection_name : str
    _dense_retriever : NaiveRetriever
        The dense search leg (repointed at the advanced collection).
    _bm25 : BM25Okapi | None
        Lazily built BM25 index.  None until first ``retrieve`` call.
    _corpus_docs : list[Document] | None
        All chunk Documents fetched from Qdrant, parallel to BM25 index.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        rrf_k: int = _RRF_K,
    ) -> None:
        self.collection_name: str = (
            collection_name or settings.qdrant_advanced_collection
        )
        self._rrf_k: int = rrf_k
        self._client: QdrantClient = _get_qdrant_client()

        # Dense retriever pointed at the advanced collection.
        self._dense_retriever = NaiveRetriever(
            collection_name=self.collection_name
        )

        # BM25 index — built lazily on first retrieve() call.
        self._bm25: BM25Okapi | None = None
        self._corpus_docs: list[Document] | None = None

        log.info(
            "hybrid_retriever_initialized",
            collection=self.collection_name,
            rrf_k=self._rrf_k,
        )

    # ---------------------------------------------------------------------- #
    # BM25 index construction                                                  #
    # ---------------------------------------------------------------------- #

    def _build_bm25_index(self) -> None:
        """
        Scroll all points from Qdrant and build the BM25 index.

        Uses ``qdrant_client.scroll`` with ``limit=1000`` pages to handle
        collections larger than a single page.  For the AMAN guide (small
        PDF) this is typically a single scroll call.

        Stores the BM25 index in ``self._bm25`` and the parallel list of
        Documents in ``self._corpus_docs`` so RRF can map BM25 ranks back
        to Document objects.

        The BM25 index uses ``normalised_content`` when available (Arabic
        chunks normalised at index time by ``smart_chunks``).  For Latin
        chunks it falls back to raw ``page_content``.
        """
        log.info("bm25_index_building", collection=self.collection_name)

        all_payloads: list[dict[str, Any]] = []
        offset = None

        # Scroll through all points in the collection.
        while True:
            results, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # We only need payloads for BM25.
            )
            for point in results:
                if point.payload:
                    all_payloads.append(dict(point.payload))

            if next_offset is None:
                break
            offset = next_offset

        log.info("bm25_corpus_fetched", total_points=len(all_payloads))

        if not all_payloads:
            raise ValueError(
                f"Collection '{self.collection_name}' is empty. "
                "Run ingest_advanced() before using HybridRetriever."
            )

        # Build parallel structures: tokenised corpus + Document list.
        tokenised_corpus: list[list[str]] = []
        corpus_docs: list[Document] = []

        for payload in all_payloads:
            page_content: str = payload.get("page_content", "")

            # Prefer pre-normalised Arabic content for BM25 matching.
            bm25_text: str = payload.get("normalised_content") or page_content

            tokenised_corpus.append(_tokenise(bm25_text))

            # Reconstruct Document (don't include page_content in metadata).
            meta = {k: v for k, v in payload.items() if k != "page_content"}
            corpus_docs.append(
                Document(page_content=page_content, metadata=meta)
            )

        self._bm25 = BM25Okapi(tokenised_corpus)
        self._corpus_docs = corpus_docs

        log.info(
            "bm25_index_built",
            corpus_size=len(corpus_docs),
            vocab_size=len(self._bm25.idf),
        )

    def _ensure_bm25(self) -> None:
        """Build the BM25 index if it has not been built yet."""
        if self._bm25 is None or self._corpus_docs is None:
            self._build_bm25_index()

    # ---------------------------------------------------------------------- #
    # RRF fusion                                                               #
    # ---------------------------------------------------------------------- #

    def _reciprocal_rank_fusion(
        self,
        dense_docs: list[Document],
        sparse_docs: list[Document],
        k: int,
    ) -> list[Document]:
        """
        Merge dense and sparse ranked lists using Reciprocal Rank Fusion.

        RRF score for document d:
            score(d) = 1/(rrf_k + rank_dense(d)) + 1/(rrf_k + rank_sparse(d))
        Documents not present in a list are assigned rank = ∞ (score contrib = 0).

        Parameters
        ----------
        dense_docs:
            Ranked list from Qdrant cosine search (rank 0 = best).
        sparse_docs:
            Ranked list from BM25 (rank 0 = best).
        k:
            Number of top fused documents to return.

        Returns
        -------
        list[Document]
            Top-k Documents ranked by descending RRF score.  Each Document's
            metadata gains a ``rrf_score`` field.
        """
        # Build identity maps: content-based key → (rank, doc).
        # We use (page, chunk_id) as the deduplication key.
        def doc_key(doc: Document) -> tuple:
            return (
                doc.metadata.get("page", -1),
                doc.metadata.get("chunk_id", -1),
            )

        rrf_scores: dict[tuple, float] = {}
        doc_map: dict[tuple, Document] = {}

        for rank, doc in enumerate(dense_docs):
            key = doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(sparse_docs):
            key = doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

        # Sort by descending RRF score.
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        results: list[Document] = []
        for key in sorted_keys[:k]:
            doc = doc_map[key]
            doc.metadata["rrf_score"] = round(rrf_scores[key], 6)
            results.append(doc)

        return results

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """
        Retrieve top-k documents using hybrid BM25 + dense search with RRF.

        Steps
        -----
        1. Build BM25 index from Qdrant payloads if not already built.
        2. Normalise the query (Arabic diacritics + variant unification).
        3. Run dense semantic search via ``NaiveRetriever.retrieve``.
        4. Run BM25 sparse search over the corpus.
        5. Fuse ranked lists with RRF and return top-k.

        Parameters
        ----------
        query:
            User question — English, Arabic, or mixed.
        k:
            Number of documents to return after fusion.  Defaults to
            ``settings.retrieval_top_k``.

        Returns
        -------
        list[Document]
            Top-k Documents from the fused ranking.  Each carries an
            ``rrf_score`` field in metadata.

        Raises
        ------
        ValueError
            If the advanced collection is empty (ingestion not run).
        """
        _k: int = k or settings.retrieval_top_k
        self._ensure_bm25()

        log.info(
            "hybrid_retrieve_started",
            query_preview=query[:80],
            k=_k,
            collection=self.collection_name,
        )

        # Normalise Arabic in the query before BM25 tokenisation.
        query_for_bm25: str = normalise_arabic(query) if is_arabic(query) else query
        query_lang: str = "ar" if is_arabic(query) else "en"

        # --- Dense leg ---
        dense_docs = self._dense_retriever.retrieve(query, k=_k)

        # --- Sparse (BM25) leg ---
        assert self._bm25 is not None
        assert self._corpus_docs is not None

        query_tokens = _tokenise(query_for_bm25)
        bm25_scores: np.ndarray = self._bm25.get_scores(query_tokens)

        # Get top-k BM25 indices by descending score.
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:_k]
        sparse_docs: list[Document] = [
            self._corpus_docs[i] for i in top_bm25_indices
            if bm25_scores[i] > 0.0  # Exclude zero-score (no token overlap) docs.
        ]

        log.info(
            "hybrid_legs_complete",
            dense_results=len(dense_docs),
            sparse_results=len(sparse_docs),
            query_lang=query_lang,
        )

        # --- RRF fusion ---
        fused_docs = self._reciprocal_rank_fusion(dense_docs, sparse_docs, k=_k)

        log.info(
            "hybrid_retrieve_complete",
            fused_results=len(fused_docs),
            top_rrf_score=fused_docs[0].metadata.get("rrf_score") if fused_docs else None,
        )

        return fused_docs

    def invalidate_bm25_cache(self) -> None:
        """
        Force a rebuild of the BM25 index on the next ``retrieve`` call.

        Call this after re-ingesting data into the advanced collection to
        ensure the index reflects the latest chunk texts.
        """
        self._bm25 = None
        self._corpus_docs = None
        log.info("bm25_cache_invalidated")
