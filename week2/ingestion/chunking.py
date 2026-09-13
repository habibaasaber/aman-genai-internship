"""
ingestion/chunking.py
=====================
Two chunking strategies for the AMAN Internship Guide RAG pipelines.

Strategy 1 — ``fixed_size_chunks``  (Naive Pipeline)
-----------------------------------------------------
Splits concatenated page text at fixed character boundaries with a
configurable overlap.  Fast and simple — serves as the baseline that the
advanced pipeline must outperform.

Strategy 2 — ``smart_chunks``  (Advanced Pipeline)
---------------------------------------------------
Works at the *block* (paragraph) level extracted by the PDF loader.
Key behaviours:
- Respects paragraph boundaries — a chunk never splits mid-sentence inside
  a paragraph.
- Detects section headings via font-size heuristics and prepends the current
  section title to every child chunk (context injection).
- Merges short paragraphs with their neighbours to avoid stub chunks.
- Keeps Arabic / English bilingual pairs together: if consecutive blocks
  alternate language, they are merged before the size limit is applied.
- Enforces a maximum chunk size by splitting oversized paragraphs at sentence
  boundaries (``。``, ``.``, ``!``, ``?``, ``\n``) as a last resort.

Public API
----------
    from ingestion.chunking import fixed_size_chunks, smart_chunks
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from utils import get_logger, is_arabic, normalise_arabic, pii_masking_service

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Font size threshold above which a block is treated as a section heading.
# Most body text in the guide is ~10–12 pt; headings are typically ≥14 pt.
_HEADING_FONT_SIZE_THRESHOLD: float = 14.0

# Sentence boundary separators used as last-resort split points for oversized
# paragraphs.  Ordered from coarsest to finest to minimise splits.
_SENTENCE_SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    ". ",
    ".\n",
    "؟ ",   # Arabic question mark
    "! ",
    "? ",
]


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _slim_metadata(
    source_meta: dict[str, Any],
    *,
    chunk_id: int,
    strategy: str,
    language: str,
    char_count: int,
    section: str = "",
) -> dict[str, Any]:
    """
    Build lean chunk-level metadata from a page-level metadata dict.

    Strips the heavy ``block_texts`` and ``block_font_sizes`` fields that
    were only needed during chunking.  Adds chunk-specific fields.

    Parameters
    ----------
    source_meta:
        The ``metadata`` dict from a page-level Document.
    chunk_id:
        Zero-based index of this chunk within the entire document.
    strategy:
        ``"naive"`` or ``"advanced"`` — identifies which pipeline produced
        this chunk.
    language:
        Detected language of the chunk content — ``"ar"`` or ``"en"``.
    char_count:
        Number of characters in the chunk content.
    section:
        The section heading that was active when this chunk was created
        (advanced pipeline only).  Empty string for naive chunks.

    Returns
    -------
    dict[str, Any]
        Lean metadata dict safe to store in Qdrant payload.
    """
    return {
        "source": source_meta.get("source", ""),
        "page": source_meta.get("page", 0),
        "page_count": source_meta.get("page_count", 0),
        "chunk_id": chunk_id,
        "chunk_strategy": strategy,
        "language": language,
        "char_count": char_count,
        "section": section,
    }


# ---------------------------------------------------------------------------
# Strategy 1 — Fixed-size chunking  (Naive Pipeline)
# ---------------------------------------------------------------------------


def fixed_size_chunks(
    pages: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split page Documents into fixed-size character chunks.

    Uses LangChain's ``RecursiveCharacterTextSplitter`` with ``\\n\\n``,
    ``\\n``, and space as separator hierarchy so it still tries to break at
    natural boundaries, but falls back to raw character splits when the text
    is too long.

    Parameters
    ----------
    pages:
        List of page-level Documents from ``ingestion.pdf_loader.load_pdf``.
    chunk_size:
        Maximum character count per chunk.  Defaults to
        ``settings.naive_chunk_size``.
    chunk_overlap:
        Character overlap between consecutive chunks.  Defaults to
        ``settings.naive_chunk_overlap``.

    Returns
    -------
    list[Document]
        Flat list of chunk Documents with lean metadata.  Each chunk carries
        the page number, source file, and a global ``chunk_id``.

    Notes
    -----
    The ``block_texts`` / ``block_font_sizes`` metadata fields added by the
    PDF loader are stripped here — they are not needed after chunking.
    """
    _chunk_size: int = chunk_size or settings.naive_chunk_size
    _chunk_overlap: int = chunk_overlap or settings.naive_chunk_overlap

    log.info(
        "fixed_size_chunking_started",
        page_count=len(pages),
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    all_chunks: list[Document] = []
    chunk_id: int = 0

    for page_doc in pages:
        page_meta: dict[str, Any] = page_doc.metadata
        page_text: str = page_doc.page_content

        if not page_text.strip():
            continue

        # Split the page text into raw string chunks.
        raw_chunks: list[str] = splitter.split_text(page_text)

        for raw_chunk in raw_chunks:
            text = raw_chunk.strip()
            if not text:
                continue
                
            text = pii_masking_service.sanitize_text(text)

            lang = "ar" if is_arabic(text) else "en"

            chunk_doc = Document(
                page_content=text,
                metadata=_slim_metadata(
                    page_meta,
                    chunk_id=chunk_id,
                    strategy="naive",
                    language=lang,
                    char_count=len(text),
                ),
            )
            all_chunks.append(chunk_doc)
            chunk_id += 1

    log.info(
        "fixed_size_chunking_complete",
        total_chunks=len(all_chunks),
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Strategy 2 — Smart chunking  (Advanced Pipeline)
# ---------------------------------------------------------------------------


def _is_heading(block_text: str, font_size: float, body_font_size: float) -> bool:
    """
    Decide whether a text block is a section heading.

    A block is treated as a heading if it satisfies ANY of:
    1. Its font size is above ``_HEADING_FONT_SIZE_THRESHOLD`` AND notably
       larger than the typical body font size.
    2. The entire block is in ALL-CAPS (common for section labels in PDFs).
    3. The block text starts with a numbering pattern like "1." or "1.1" or
       "Section 2" — common in guide documents.

    Parameters
    ----------
    block_text:
        Stripped text content of the block.
    font_size:
        Maximum font size observed in this block's spans.
    body_font_size:
        Median / typical body font size of the page, used as reference.

    Returns
    -------
    bool
        True if the block should be treated as a section heading.
    """
    # Font-size heuristic: heading must be larger than body AND above threshold.
    if font_size >= _HEADING_FONT_SIZE_THRESHOLD and font_size > body_font_size * 1.15:
        return True

    # All-caps Latin text (ignore very short tokens like "A" or "I").
    stripped = block_text.strip()
    if len(stripped) > 3 and stripped == stripped.upper() and stripped.isascii():
        return True

    # Numbered section patterns: "1.", "1.1", "Section 2", "Chapter 3" etc.
    if re.match(r"^(?:Section|Chapter|Track|Part)?\s*\d+[\.\d]*\s", stripped, re.I):
        return True

    return False


def _compute_body_font_size(font_sizes: list[float]) -> float:
    """
    Estimate the body (paragraph) font size for a page.

    Takes the median of all block font sizes.  Heading font sizes are larger
    and will be above this value, so it serves as a good baseline.

    Parameters
    ----------
    font_sizes:
        List of max-font-size values, one per block on the page.

    Returns
    -------
    float
        Median font size, or 12.0 if the list is empty.
    """
    if not font_sizes:
        return 12.0
    sorted_sizes = sorted(font_sizes)
    mid = len(sorted_sizes) // 2
    return sorted_sizes[mid]


def _split_oversized(
    text: str,
    max_size: int,
) -> list[str]:
    """
    Split a single oversized text block into smaller pieces at sentence
    boundaries.

    Tries each separator in ``_SENTENCE_SEPARATORS`` in order.  If no
    separator produces pieces small enough, falls back to hard character
    splits at ``max_size`` boundaries.

    Parameters
    ----------
    text:
        The text to split.
    max_size:
        Maximum allowed character count per output piece.

    Returns
    -------
    list[str]
        List of text pieces, each ≤ ``max_size`` characters.
    """
    if len(text) <= max_size:
        return [text]

    # Try each natural separator.
    for sep in _SENTENCE_SEPARATORS:
        parts = text.split(sep)
        if all(len(p) <= max_size for p in parts if p.strip()):
            return [p.strip() for p in parts if p.strip()]

    # Last resort: hard character split.
    return [text[i : i + max_size] for i in range(0, len(text), max_size)]


def _should_merge_bilingual(block_a: str, block_b: str) -> bool:
    """
    Decide whether two consecutive blocks form a bilingual EN/AR pair.

    Returns True when blocks are in different languages, indicating they
    likely represent the same concept translated — and should be kept together
    to avoid losing cross-lingual context during retrieval.

    Parameters
    ----------
    block_a:
        Text of the first block.
    block_b:
        Text of the second block.

    Returns
    -------
    bool
        True if the blocks are in different languages.
    """
    lang_a = "ar" if is_arabic(block_a) else "en"
    lang_b = "ar" if is_arabic(block_b) else "en"
    return lang_a != lang_b


def smart_chunks(
    pages: list[Document],
    min_chunk_size: int | None = None,
    max_chunk_size: int | None = None,
) -> list[Document]:
    """
    Split page Documents into semantically coherent chunks (Advanced Pipeline).

    Algorithm
    ---------
    For each page:
    1.  Compute the page's body font size (median of block font sizes).
    2.  Iterate over text blocks in reading order.
    3.  If a block is a heading, flush the current buffer as a chunk and
        record the heading as the active section title.
    4.  If consecutive blocks are in different languages (EN/AR pair), merge
        them before applying size constraints.
    5.  Append the block to the running buffer if adding it won't exceed
        ``max_chunk_size``.  Otherwise, flush the buffer first.
    6.  If the buffer is smaller than ``min_chunk_size``, continue accumulating
        (small paragraph merging).
    7.  After all blocks on a page, flush any remaining buffer.
    8.  Oversized individual blocks are further split at sentence boundaries.

    Each output chunk stores the active section heading in its metadata so
    that retrievers can surface the section name alongside the answer.

    Parameters
    ----------
    pages:
        List of page-level Documents from ``ingestion.pdf_loader.load_pdf``.
    min_chunk_size:
        Minimum characters before a buffer is flushed as a chunk.  Defaults
        to ``settings.advanced_min_chunk_size``.
    max_chunk_size:
        Maximum characters allowed in a single chunk.  Defaults to
        ``settings.advanced_max_chunk_size``.

    Returns
    -------
    list[Document]
        Flat list of chunk Documents with lean metadata including the
        ``section`` field for citation display.
    """
    _min_size: int = min_chunk_size or settings.advanced_min_chunk_size
    _max_size: int = max_chunk_size or settings.advanced_max_chunk_size

    log.info(
        "smart_chunking_started",
        page_count=len(pages),
        min_chunk_size=_min_size,
        max_chunk_size=_max_size,
    )

    all_chunks: list[Document] = []
    chunk_id: int = 0

    for page_doc in pages:
        page_meta: dict[str, Any] = page_doc.metadata
        block_texts: list[str] = page_meta.get("block_texts", [])
        block_font_sizes: list[float] = page_meta.get("block_font_sizes", [])

        if not block_texts:
            continue

        # Compute body font size for this page (used in heading detection).
        body_font_size: float = _compute_body_font_size(block_font_sizes)

        # Pad font_sizes list to match block_texts length (safety guard).
        while len(block_font_sizes) < len(block_texts):
            block_font_sizes.append(body_font_size)

        # Running state for the current page.
        buffer: list[str] = []
        buffer_size: int = 0
        current_section: str = ""

        def _flush_buffer() -> None:
            """Flush the current buffer as a chunk Document."""
            nonlocal buffer, buffer_size, chunk_id

            if not buffer:
                return

            raw_text = "\n\n".join(buffer).strip()
            if not raw_text:
                buffer = []
                buffer_size = 0
                return

            # Split oversized buffers at sentence boundaries.
            pieces = _split_oversized(raw_text, _max_size)

            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                    
                piece = pii_masking_service.sanitize_text(piece)

                lang = "ar" if is_arabic(piece) else "en"
                # Normalise Arabic content for BM25 indexing in the metadata.
                normalised = normalise_arabic(piece) if lang == "ar" else piece

                chunk_doc = Document(
                    page_content=piece,
                    metadata=_slim_metadata(
                        page_meta,
                        chunk_id=chunk_id,
                        strategy="advanced",
                        language=lang,
                        char_count=len(piece),
                        section=current_section,
                    ),
                )
                # Store normalised text as extra metadata for BM25 retriever.
                chunk_doc.metadata["normalised_content"] = normalised
                all_chunks.append(chunk_doc)
                chunk_id += 1

            buffer = []
            buffer_size = 0

        i = 0
        while i < len(block_texts):
            block = block_texts[i].strip()
            font_size = block_font_sizes[i]

            if not block:
                i += 1
                continue

            # Check if this block is a section heading.
            if _is_heading(block, font_size, body_font_size):
                # Flush anything accumulated under the previous heading.
                _flush_buffer()
                current_section = block
                log.debug(
                    "heading_detected",
                    page=page_meta.get("page"),
                    heading=block[:60],
                )
                i += 1
                continue

            # Bilingual pair detection: if the next block is in a different
            # language, merge both before adding to the buffer.
            if i + 1 < len(block_texts):
                next_block = block_texts[i + 1].strip()
                if next_block and _should_merge_bilingual(block, next_block):
                    block = block + "\n\n" + next_block
                    i += 1  # Skip the next block since it's merged.

            block_len = len(block)

            # If adding this block would overflow the buffer, flush first.
            if buffer_size + block_len > _max_size and buffer:
                _flush_buffer()

            buffer.append(block)
            buffer_size += block_len
            i += 1

            # Only flush if we're at or above min size to avoid tiny chunks.
            # Exception: if we're already at max, flush immediately.
            if buffer_size >= _max_size:
                _flush_buffer()

        # Flush any remaining content at the end of the page.
        _flush_buffer()

    log.info(
        "smart_chunking_complete",
        total_chunks=len(all_chunks),
    )
    return all_chunks
