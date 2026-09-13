"""
app/document_loader.py
Converts uploaded resume files (PDF / DOCX / TXT) to plain text.

Public API:
    load_document(file_path: str | Path) -> str

Supported extensions: .pdf, .docx, .txt
"""

from pathlib import Path


def load_document(file_path: str | Path) -> str:
    """
    Load a resume file and return its full text content.

    Dispatches to the appropriate parser based on the file extension.
    Raises ValueError for unsupported types or empty results.

    Args:
        file_path: Absolute or relative path to the resume file.

    Returns:
        Plain text extracted from the file.

    Raises:
        ValueError: If the file type is unsupported or content is empty.
        RuntimeError: If extraction fails (e.g. corrupted file).
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = _load_pdf(path)
    elif suffix == ".docx":
        text = _load_docx(path)
    elif suffix == ".txt":
        text = _load_txt(path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Please upload a .pdf, .docx, or .txt file."
        )

    text = text.strip()
    if not text:
        raise ValueError(
            "The uploaded file appears to be empty or could not be read. "
            "Please check the file and try again."
        )

    return text


# ---------------------------------------------------------------------------
# PDF loader — PyMuPDF (fitz)
# ---------------------------------------------------------------------------

def _load_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "PyMuPDF is not installed. Run: pip install PyMuPDF"
        )

    try:
        doc = fitz.open(str(path))
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()
        return "\n".join(pages_text)
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF file: {exc}") from exc


# ---------------------------------------------------------------------------
# DOCX loader — python-docx
# ---------------------------------------------------------------------------

def _load_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError(
            "python-docx is not installed. Run: pip install python-docx"
        )

    try:
        doc = Document(str(path))
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)
    except Exception as exc:
        raise RuntimeError(f"Failed to read DOCX file: {exc}") from exc


# ---------------------------------------------------------------------------
# TXT loader — plain UTF-8 read
# ---------------------------------------------------------------------------

def _load_txt(path: Path) -> str:
    """Read a plain text file, trying UTF-8 then latin-1 as fallback."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")
    except Exception as exc:
        raise RuntimeError(f"Failed to read TXT file: {exc}") from exc
