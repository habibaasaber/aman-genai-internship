"""
retrievers/naive.py
===================
Dense semantic retriever for the Naive RAG pipeline.

Encodes the user query with ``BAAI/bge-m3`` and returns the top-k most
similar chunks from the ``aman_naive`` Qdrant collection using cosine
similarity (pure semantic search — no BM25, no reranking).

Design notes
------------
- Uses the query-side instruction ``"Represent this sentence: "`` which is
  the same prefix used at indexing time in ``embed_store._encode_chunks``.
  BGE-m3's README recommends the same instruction for both queries and
  passages for general retrieval tasks.
- The embedding model is imported from ``embed_store`` to reuse the already-
  loaded singleton — no double loading of 570 MB.
- ``retrieve`` returns ``langchain_core.documents.Document`` objects so the
  pipeline layer can treat naive and advanced retrievers identically.

Usage
-----
    from retrievers import NaiveRetriever
    retriever = NaiveRetriever()
    docs = retriever.retrieve("What are the working hours?", k=4)
"""

from __future__ import annotations

import torch
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from config import settings
from ingestion.embed_store import _get_embedding_model, _get_qdrant_client
from utils import get_logger

log = get_logger(__name__)

# BGE-m3 query prefix — must match what was used at indexing time.
_QUERY_PREFIX = "Represent this sentence: "


class NaiveRetriever:
    """
    Pure semantic (dense) retriever backed by Qdrant.

    Encodes the query with ``BAAI/bge-m3`` and performs cosine-similarity
    search against the naive pipeline's Qdrant collection.

    Parameters
    ----------
    collection_name:
        Qdrant collection to search.  Defaults to
        ``settings.qdrant_naive_collection`` (``"aman_naive"``).

    Attributes
    ----------
    collection_name : str
        The Qdrant collection this retriever searches.
    """

    def __init__(
        self,
        collection_name: str | None = None,
    ) -> None:
        self.collection_name: str = (
            collection_name or settings.qdrant_naive_collection
        )
        self._client: QdrantClient = _get_qdrant_client()
        self._model = _get_embedding_model()

        log.info(
            "naive_retriever_initialized",
            collection=self.collection_name,
        )

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _encode_query(self, query: str) -> list[float]:
        """
        Encode a query string into a normalised dense vector.

        Applies the BGE-m3 query instruction prefix and L2-normalises the
        output so cosine similarity works correctly with Qdrant's distance
        metric.

        Parameters
        ----------
        query:
            The raw user query (English or Arabic).

        Returns
        -------
        list[float]
            1024-dimensional normalised embedding vector.
        """
        prefixed = f"{_QUERY_PREFIX}{query}"
        vector = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # ``encode`` returns a numpy array; Qdrant expects a plain list.
        return vector.tolist()

    def _payload_to_document(self, payload: dict, score: float) -> Document:
        """
        Convert a Qdrant search result payload into a LangChain Document.

        Adds the retrieval ``score`` to metadata so downstream components
        (reranker, Chainlit UI) can display or use it.

        Parameters
        ----------
        payload:
            The Qdrant point payload dict (mirrors chunk metadata + page_content).
        score:
            Cosine similarity score from Qdrant (range 0–1 for normalised vectors).

        Returns
        -------
        Document
            A ``langchain_core.documents.Document`` with page_content and
            full metadata including the retrieval score.
        """
        page_content: str = payload.pop("page_content", "")
        metadata: dict = {**payload, "retrieval_score": round(float(score), 6)}
        return Document(page_content=page_content, metadata=metadata)

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """
        Retrieve the top-k most semantically similar chunks for a query.

        Parameters
        ----------
        query:
            The user's question (English or Arabic).  No preprocessing is
            applied here — the naive retriever uses the raw query text.
        k:
            Number of documents to return.  Defaults to
            ``settings.retrieval_top_k``.

        Returns
        -------
        list[Document]
            Up to ``k`` Documents sorted by descending cosine similarity.
            Each Document's metadata includes ``retrieval_score``.

        Raises
        ------
        ConnectionError
            If Qdrant is not reachable (propagated from ``_get_qdrant_client``).
        ValueError
            If the collection does not exist (not yet ingested).
        """
        _k: int = k or settings.retrieval_top_k

        log.info(
            "naive_retrieve_started",
            query_preview=query[:80],
            k=_k,
            collection=self.collection_name,
        )

        query_vector = self._encode_query(query)

        try:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=_k,
            )
            results = response.points
        except Exception as exc:
            raise ValueError(
                f"Failed to search Qdrant collection '{self.collection_name}'. "
                f"Error: {exc}"
            ) from exc

        documents: list[Document] = []
        for hit in results:
            if hit.payload is None:
                continue
            # Make a copy to avoid mutating the Qdrant response object.
            doc = self._payload_to_document(dict(hit.payload), hit.score)
            documents.append(doc)

        log.info(
            "naive_retrieve_complete",
            retrieved=len(documents),
            top_score=documents[0].metadata.get("retrieval_score") if documents else None,
        )

        return documents

    def embed_query(self, query: str) -> list[float]:
        """
        Expose query embedding as a public method.

        Used by ``retrievers/hybrid.py`` to obtain the dense vector for the
        fusion step without duplicating the encoding logic.

        Parameters
        ----------
        query:
            Raw query string.

        Returns
        -------
        list[float]
            Normalised 1024-dimensional query embedding.
        """
        return self._encode_query(query)
