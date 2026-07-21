"""
ingestion/pdf_loader.py
=======================
PDF text extraction for the AMAN Internship Guide.

Produces one ``langchain_core.documents.Document`` per page with rich
metadata so that downstream chunking, retrieval, and citation logic has
full provenance information.

Why PyMuPDF (fitz)?
-------------------
- ``get_text("dict")`` returns a structured block/span tree that preserves
  reading order for both LTR (English) and RTL (Arabic) text.
- It is significantly faster than pdfplumber or pypdf for large PDFs.
- It exposes font-size data that the smart chunker uses for heading detection.

Public API
----------
    from ingestion import load_pdf
    pages: list[Document] = load_pdf("data/aman_internship_guide_2026.pdf")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_core.documents import Document

from utils import get_logger, is_arabic

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_page_blocks(page: fitz.Page) -> list[dict[str, Any]]:
    """
    Extract text blocks from a single PDF page preserving reading order.

    Uses ``get_text("dict")`` which returns a nested structure:
        page → blocks → lines → spans
    Each block corresponds to a paragraph or heading-level text region.

    Parameters
    ----------
    page:
        A PyMuPDF ``Page`` object.

    Returns
    -------
    list[dict]
        A list of block dicts, each containing:
        - ``text``      : concatenated text of all lines in the block.
        - ``font_sizes``: list of font sizes seen in the block (for heading detection).
        - ``bbox``      : (x0, y0, x1, y1) bounding box of the block.
        - ``block_no``  : original block index within the page.
    """
    raw = page.get_text("dict", sort=True)  # sort=True enforces reading order
    blocks: list[dict[str, Any]] = []

    for block in raw.get("blocks", []):
        # Only process text blocks (type 0); skip image blocks (type 1).
        if block.get("type") != 0:
            continue

        lines_text: list[str] = []
        font_sizes: list[float] = []

        for line in block.get("lines", []):
            line_parts: list[str] = []
            for span in line.get("spans", []):
                span_text: str = span.get("text", "").strip()
                if span_text:
                    line_parts.append(span_text)
                    font_sizes.append(span.get("size", 12.0))
            if line_parts:
                lines_text.append(" ".join(line_parts))

        block_text = "\n".join(lines_text).strip()
        if not block_text:
            continue

        blocks.append(
            {
                "text": block_text,
                "font_sizes": font_sizes,
                "bbox": block.get("bbox", (0, 0, 0, 0)),
                "block_no": block.get("number", len(blocks)),
            }
        )

    return blocks


def _detect_page_language(blocks: list[dict[str, Any]]) -> str:
    """
    Detect the dominant language of a page from its text blocks.

    Concatenates all block texts and uses ``utils.is_arabic`` to decide
    whether the page is predominantly Arabic. Returns ``"ar"`` or ``"en"``.
    In practice, the AMAN guide has mixed pages — we label by majority.

    Parameters
    ----------
    blocks:
        List of block dicts as returned by ``_extract_page_blocks``.

    Returns
    -------
    str
        ``"ar"`` if Arabic characters dominate, ``"en"`` otherwise.
    """
    combined = " ".join(b["text"] for b in blocks)
    return "ar" if is_arabic(combined, threshold=0.25) else "en"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """
    Load the AMAN Internship Guide PDF and return one Document per page.

    Each Document carries:
    - ``page_content``: Full plain-text content of the page (blocks joined
      by double newlines to preserve paragraph structure).
    - ``metadata``:
        - ``source``      : Absolute path to the PDF file (str).
        - ``page``        : 1-based page number (int).
        - ``page_count``  : Total number of pages in the PDF (int).
        - ``language``    : Detected language — ``"ar"`` or ``"en"`` (str).
        - ``char_count``  : Number of characters in the page content (int).
        - ``block_texts`` : List of raw block strings for smart chunking (list[str]).
        - ``block_font_sizes``: Max font size per block for heading detection (list[float]).

    The ``block_texts`` and ``block_font_sizes`` metadata fields are consumed
    by ``ingestion/chunking.py:smart_chunks()`` and stripped before the
    Document is stored in Qdrant to keep payloads lean.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file. Can be relative (resolved from CWD) or absolute.

    Returns
    -------
    list[Document]
        One ``langchain_core.documents.Document`` per page with the metadata
        structure described above. Empty pages are excluded.

    Raises
    ------
    FileNotFoundError
        If the PDF does not exist at the given path.
    RuntimeError
        If PyMuPDF fails to open or parse the file.

    Examples
    --------
        pages = load_pdf("data/aman_internship_guide_2026.pdf")
        print(f"Loaded {len(pages)} pages")
        print(pages[0].metadata)
    """
    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at '{pdf_path}'. "
            "Ensure the file exists under week2/data/ and PDF_PATH is set correctly."
        )

    log.info("pdf_load_started", path=str(pdf_path))

    documents: list[Document] = []

    try:
        with fitz.open(pdf_path) as doc:
            page_count: int = doc.page_count
            log.info("pdf_opened", page_count=page_count)

            for page_index in range(page_count):
                page: fitz.Page = doc[page_index]
                page_number: int = page_index + 1  # 1-based

                blocks = _extract_page_blocks(page)

                if not blocks:
                    log.debug("empty_page_skipped", page=page_number)
                    continue

                # Join blocks with double newline to preserve paragraph breaks.
                page_content: str = "\n\n".join(b["text"] for b in blocks)

                language: str = _detect_page_language(blocks)

                # Max font size per block — used by smart_chunks() for heading
                # detection.  If a block has no spans, default to body size 12.
                block_font_sizes: list[float] = [
                    max(b["font_sizes"]) if b["font_sizes"] else 12.0
                    for b in blocks
                ]

                document = Document(
                    page_content=page_content,
                    metadata={
                        "source": str(pdf_path),
                        "page": page_number,
                        "page_count": page_count,
                        "language": language,
                        "char_count": len(page_content),
                        # Raw block strings — consumed by smart_chunks(), then stripped.
                        "block_texts": [b["text"] for b in blocks],
                        # Max font size per block — used for heading detection.
                        "block_font_sizes": block_font_sizes,
                    },
                )
                documents.append(document)

                log.debug(
                    "page_extracted",
                    page=page_number,
                    language=language,
                    char_count=len(page_content),
                    block_count=len(blocks),
                )

    except fitz.FileNotFoundError as exc:
        raise RuntimeError(f"PyMuPDF could not open '{pdf_path}': {exc}") from exc

    log.info(
        "pdf_load_complete",
        path=str(pdf_path),
        pages_extracted=len(documents),
        pages_skipped=page_count - len(documents),
    )

    return documents