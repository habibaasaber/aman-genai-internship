"""
utils/__init__.py
=================
Public API for the utils package.

Re-exports the most commonly used helpers so other modules can do:
    from utils import get_logger, normalise_arabic
instead of importing from sub-modules directly.
"""

from utils.arabic import is_arabic, normalise_arabic, strip_diacritics
from utils.logger import get_logger
from utils.privacy import pii_masking_service

__all__ = [
    "get_logger",
    "is_arabic",
    "normalise_arabic",
    "strip_diacritics",
    "pii_masking_service",
]
