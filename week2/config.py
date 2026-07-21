"""
config.py
=========
Centralised, type-safe configuration for the Week 2 RAG project.

All values are read from environment variables (or a .env file in the same
directory). Import the singleton ``settings`` object instead of calling
os.getenv() directly anywhere in the codebase.

Usage
-----
    from config import settings
    print(settings.qdrant_url)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables / .env file.

    Pydantic-settings automatically:
    - Reads from a .env file (specified by model_config).
    - Casts string env-vars to the annotated Python types.
    - Raises ValidationError with a clear message if a required var is absent.
    """

    model_config = SettingsConfigDict(
        # Look for .env in the same directory as this file (week2/).
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        # Allow extra vars in .env without raising an error.
        extra="ignore",
        # Field names are case-insensitive vs env var names.
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ #
    # LLM — Google Gemini (primary)                                        #
    # ------------------------------------------------------------------ #
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemini-2.5-flash-preview-05-20",
        description="Gemini model identifier used for answer generation",
    )

    # ------------------------------------------------------------------ #
    # LLM — OpenAI (RAGAS judge + fallback)                               #
    # ------------------------------------------------------------------ #
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model used as the RAGAS evaluation judge",
    )

    # ------------------------------------------------------------------ #
    # LLM generation parameters                                           #
    # ------------------------------------------------------------------ #
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature — 0.0 for deterministic RAG answers",
    )
    llm_max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Maximum tokens in a generated answer",
    )

    # ------------------------------------------------------------------ #
    # Langfuse — Observability                                             #
    # ------------------------------------------------------------------ #
    langfuse_secret_key: str = Field(..., description="Langfuse secret key (sk-lf-…)")
    langfuse_public_key: str = Field(..., description="Langfuse public key (pk-lf-…)")
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse server URL (cloud or self-hosted)",
    )

    # ------------------------------------------------------------------ #
    # Qdrant — Vector Store                                                #
    # ------------------------------------------------------------------ #
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant REST endpoint — run via: docker run -p 6333:6333 qdrant/qdrant",
    )
    qdrant_naive_collection: str = Field(
        default="aman_naive",
        description="Qdrant collection name for the naive pipeline",
    )
    qdrant_advanced_collection: str = Field(
        default="aman_advanced",
        description="Qdrant collection name for the advanced pipeline",
    )

    # ------------------------------------------------------------------ #
    # Embedding & Reranking Models                                         #
    # ------------------------------------------------------------------ #
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace model ID for sentence embeddings (multilingual)",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="HuggingFace cross-encoder model ID for reranking",
    )
    # BGE-m3 produces 1024-dimensional vectors.
    embedding_dim: int = Field(
        default=1024,
        description="Dimensionality of the embedding model output",
    )

    # ------------------------------------------------------------------ #
    # Naive Pipeline — Fixed-size Chunking                                 #
    # ------------------------------------------------------------------ #
    naive_chunk_size: int = Field(
        default=512,
        gt=0,
        description="Character count per chunk for the naive pipeline",
    )
    naive_chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="Character overlap between consecutive naive chunks",
    )

    # ------------------------------------------------------------------ #
    # Advanced Pipeline — Smart Chunking                                   #
    # ------------------------------------------------------------------ #
    advanced_min_chunk_size: int = Field(
        default=200,
        gt=0,
        description="Minimum paragraph size (chars) before merging with next paragraph",
    )
    advanced_max_chunk_size: int = Field(
        default=1200,
        gt=0,
        description="Maximum chunk size (chars) before a forced split",
    )

    # ------------------------------------------------------------------ #
    # Retrieval                                                             #
    # ------------------------------------------------------------------ #
    retrieval_top_k: int = Field(
        default=10,
        gt=0,
        description="Number of candidate documents fetched from the vector store",
    )
    reranker_top_n: int = Field(
        default=4,
        gt=0,
        description="Documents kept after reranking — passed to the LLM prompt",
    )

    # ------------------------------------------------------------------ #
    # Data                                                                  #
    # ------------------------------------------------------------------ #
    pdf_path: Path = Field(
        default=Path("data/aman_internship_guide_2026.pdf"),
        description="Path to the AMAN Internship Guide PDF (relative to week2/)",
    )

    # ------------------------------------------------------------------ #
    # Computed fields — derived at validation time, not from env           #
    # ------------------------------------------------------------------ #

    @computed_field  # type: ignore[misc]
    @property
    def pdf_absolute_path(self) -> Path:
        """Resolve pdf_path relative to this config file's directory."""
        base = Path(__file__).parent
        resolved = (base / self.pdf_path).resolve()
        return resolved

    @computed_field  # type: ignore[misc]
    @property
    def evaluation_questions_path(self) -> Path:
        """Absolute path to the RAGAS test questions JSON file."""
        return Path(__file__).parent / "evaluation" / "test_questions.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Using lru_cache ensures the .env file is read exactly once per process,
    even if get_settings() is called from multiple modules.

    Returns
    -------
    Settings
        The validated, fully-populated settings instance.
    """
    return Settings()


# Module-level singleton — import this directly for convenience:
#   from config import settings
settings: Settings = get_settings()
