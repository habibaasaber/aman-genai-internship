"""
ingestion/embed_store.py
========================
Embedding and Qdrant ingestion for both RAG pipelines.

Responsibilities
----------------
1. Load ``BAAI/bge-m3`` (multilingual dense embedding model) once and cache
   it as a module-level singleton to avoid repeated 570 MB downloads.
2. Encode chunk texts in configurable batches (CPU/GPU auto-detected).
3. Create Qdrant collections with the correct cosine-distance schema if they
   do not already exist.
4. Upsert encoded vectors with lean metadata payloads into Qdrant.
5. Expose ``ingest_naive()`` and ``ingest_advanced()`` as the public API that
   orchestrates the full load → chunk → embed → store pipeline.

Qdrant collections created
--------------------------
- ``aman_naive``    : fixed-size chunks (naive pipeline)
- ``aman_advanced`` : smart/semantic chunks (advanced pipeline)

Both collections use:
- Vector size  : 1024  (BGE-m3 output dimensionality)
- Distance     : Cosine
- On-disk      : False  (in-memory for local Docker instance)

Usage
-----
    from ingestion.embed_store import ingest_naive, ingest_advanced
    ingest_naive()      # Run once to populate aman_naive collection
    ingest_advanced()   # Run once to populate aman_advanced collection
"""

from __future__ import annotations

import uuid
from typing import Any

import torch
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import settings
from ingestion.chunking import fixed_size_chunks, smart_chunks
from ingestion.pdf_loader import load_pdf
from utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Embedding model singleton
# ---------------------------------------------------------------------------

# This variable is populated lazily on first call to ``_get_embedding_model``.
_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    """
    Return the shared ``BAAI/bge-m3`` embedding model instance.

    Loads the model on the first call and caches it for all subsequent
    calls in the same process.  GPU is used automatically if available.

    Returns
    -------
    SentenceTransformer
        The loaded and ready-to-use embedding model.
    """
    global _embedding_model

    if _embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(
            "embedding_model_loading",
            model=settings.embedding_model,
            device=device,
        )
        _embedding_model = SentenceTransformer(
            settings.embedding_model,
            device=device,
        )
        log.info("embedding_model_loaded", model=settings.embedding_model)

    return _embedding_model


# ---------------------------------------------------------------------------
# Qdrant client singleton
# ---------------------------------------------------------------------------

_qdrant_client: QdrantClient | None = None


def _get_qdrant_client() -> QdrantClient:
    """
    Return the shared Qdrant client instance.

    Creates the client on first call pointing at ``settings.qdrant_url``.
    Raises ``ConnectionError`` if Qdrant is not reachable.

    Returns
    -------
    QdrantClient
        Connected Qdrant client.
    """
    global _qdrant_client

    if _qdrant_client is None:
        log.info("qdrant_connecting", url=settings.qdrant_url)
        _qdrant_client = QdrantClient(url=settings.qdrant_url, timeout=30)
        # Validate connectivity — raises if Qdrant is unreachable.
        try:
            _qdrant_client.get_collections()
            log.info("qdrant_connected", url=settings.qdrant_url)
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to Qdrant at '{settings.qdrant_url}'. "
                "Make sure Docker is running: "
                "docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant"
            ) from exc

    return _qdrant_client


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


def _ensure_collection(collection_name: str) -> None:
    """
    Create the Qdrant collection if it does not already exist.

    Collection schema:
    - Vector size : ``settings.embedding_dim`` (1024 for BGE-m3)
    - Distance    : Cosine
    - Indexing    : HNSW (Qdrant default) with ef_construct=128

    Parameters
    ----------
    collection_name:
        The Qdrant collection name to create or verify.
    """
    client = _get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}

    if collection_name in existing:
        log.info("collection_exists", collection=collection_name)
        return

    log.info(
        "collection_creating",
        collection=collection_name,
        dim=settings.embedding_dim,
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=settings.embedding_dim,
            distance=qdrant_models.Distance.COSINE,
        ),
        hnsw_config=qdrant_models.HnswConfigDiff(
            m=16,
            ef_construct=128,
        ),
    )
    log.info("collection_created", collection=collection_name)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _encode_chunks(
    chunks: list[Document],
    batch_size: int = 32,
) -> list[list[float]]:
    """
    Encode the ``page_content`` of each chunk into a dense vector.

    BGE-m3 uses a query/passage asymmetry: passages are prefixed with
    ``"Represent this sentence: "`` for optimal retrieval performance.
    We apply this prefix during indexing so stored vectors are passage-style.
    The same prefix must NOT be used at query time (the retriever handles this).

    Parameters
    ----------
    chunks:
        List of chunk Documents to encode.
    batch_size:
        Number of chunks encoded per forward pass.  Reduce if OOM on GPU.

    Returns
    -------
    list[list[float]]
        Parallel list of embedding vectors (one per chunk).
    """
    model = _get_embedding_model()
    texts = [doc.page_content for doc in chunks]

    log.info("encoding_started", total_chunks=len(texts), batch_size=batch_size)

    all_vectors: list[list[float]] = []

    for batch_start in tqdm(
        range(0, len(texts), batch_size),
        desc="Encoding chunks",
        unit="batch",
    ):
        batch_texts = texts[batch_start : batch_start + batch_size]
        # BGE-m3 passage prefix — improves retrieval quality.
        prefixed = [f"Represent this sentence: {t}" for t in batch_texts]

        vectors = model.encode(
            prefixed,
            normalize_embeddings=True,  # Cosine similarity requires L2-normalised vectors.
            show_progress_bar=False,
        )
        all_vectors.extend(vectors.tolist())

    log.info("encoding_complete", total_vectors=len(all_vectors))
    return all_vectors


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


def _upsert_chunks(
    collection_name: str,
    chunks: list[Document],
    vectors: list[list[float]],
    upsert_batch_size: int = 128,
) -> None:
    """
    Upsert chunk vectors and metadata payloads into Qdrant.

    Each point is assigned a deterministic UUID based on its ``chunk_id``
    and collection name so that re-running ingestion is idempotent — existing
    points are overwritten rather than duplicated.

    Parameters
    ----------
    collection_name:
        Target Qdrant collection.
    chunks:
        List of chunk Documents (provides metadata payloads).
    vectors:
        Parallel list of embedding vectors from ``_encode_chunks``.
    upsert_batch_size:
        Number of points upserted per Qdrant API call.
    """
    client = _get_qdrant_client()

    points: list[qdrant_models.PointStruct] = []

    for chunk, vector in zip(chunks, vectors, strict=True):
        # Deterministic ID: prevents duplicate points on repeated ingestion.
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{collection_name}::{chunk.metadata['chunk_id']}",
            )
        )
        # Build the payload — only lean metadata fields go into Qdrant.
        payload: dict[str, Any] = {
            "page_content": chunk.page_content,
            **chunk.metadata,
        }

        points.append(
            qdrant_models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    log.info(
        "upsert_started",
        collection=collection_name,
        total_points=len(points),
        batch_size=upsert_batch_size,
    )

    for batch_start in tqdm(
        range(0, len(points), upsert_batch_size),
        desc=f"Upserting → {collection_name}",
        unit="batch",
    ):
        batch = points[batch_start : batch_start + upsert_batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    log.info(
        "upsert_complete",
        collection=collection_name,
        total_points=len(points),
    )


# ---------------------------------------------------------------------------
# Public entry-points
# ---------------------------------------------------------------------------


def ingest_naive(pdf_path: str | None = None) -> int:
    """
    Run the full naive ingestion pipeline.

    Steps
    -----
    1. Load PDF → list of page Documents.
    2. Apply fixed-size chunking.
    3. Encode chunks with BGE-m3.
    4. Upsert into the ``aman_naive`` Qdrant collection.

    Parameters
    ----------
    pdf_path:
        Override for the PDF path.  Defaults to ``settings.pdf_absolute_path``.

    Returns
    -------
    int
        Number of chunks indexed.
    """
    path = str(pdf_path or settings.pdf_absolute_path)
    log.info("ingest_naive_started", pdf_path=path)

    pages = load_pdf(path)
    chunks = fixed_size_chunks(pages)

    _ensure_collection(settings.qdrant_naive_collection)
    vectors = _encode_chunks(chunks)
    _upsert_chunks(settings.qdrant_naive_collection, chunks, vectors)

    log.info(
        "ingest_naive_complete",
        collection=settings.qdrant_naive_collection,
        chunks_indexed=len(chunks),
    )
    return len(chunks)


def ingest_advanced(pdf_path: str | None = None) -> int:
    """
    Run the full advanced ingestion pipeline.

    Steps
    -----
    1. Load PDF → list of page Documents.
    2. Apply smart (section-aware, bilingual-pair) chunking.
    3. Encode chunks with BGE-m3.
    4. Upsert into the ``aman_advanced`` Qdrant collection.

    Parameters
    ----------
    pdf_path:
        Override for the PDF path.  Defaults to ``settings.pdf_absolute_path``.

    Returns
    -------
    int
        Number of chunks indexed.
    """
    path = str(pdf_path or settings.pdf_absolute_path)
    log.info("ingest_advanced_started", pdf_path=path)

    pages = load_pdf(path)
    chunks = smart_chunks(pages)

    _ensure_collection(settings.qdrant_advanced_collection)
    vectors = _encode_chunks(chunks)
    _upsert_chunks(settings.qdrant_advanced_collection, chunks, vectors)

    log.info(
        "ingest_advanced_complete",
        collection=settings.qdrant_advanced_collection,
        chunks_indexed=len(chunks),
    )
    return len(chunks)


def collection_exists(collection_name: str) -> bool:
    """
    Check whether a Qdrant collection has been populated.

    Used by pipelines at startup to decide whether ingestion is needed
    before the first query can be answered.

    Parameters
    ----------
    collection_name:
        The Qdrant collection name to check.

    Returns
    -------
    bool
        True if the collection exists AND contains at least one point.
    """
    try:
        client = _get_qdrant_client()
        info = client.get_collection(collection_name)
        return info.points_count is not None and info.points_count > 0
    except Exception:
        return False
