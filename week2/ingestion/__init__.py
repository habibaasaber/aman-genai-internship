"""
ingestion/__init__.py
=====================
Public API for the ingestion package.

Exports the primary entry-point functions so other modules can use:
    from ingestion import load_pdf
"""

from ingestion.pdf_loader import load_pdf

__all__ = ["load_pdf"]
