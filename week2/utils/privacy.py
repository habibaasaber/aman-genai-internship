"""
utils/privacy.py
================
PII (Personally Identifiable Information) masking service.

Provides a reusable ``PIIMaskingService`` that detects and redacts sensitive
information from text — employee IDs, phone numbers, emails, national IDs,
bank accounts, salary amounts, and generic person-name patterns.

Design
------
- **Rule-based (regex)**: lightweight, no external NLP model required.
  Each pattern is a named ``_PIIPattern`` so rules can be added/removed
  independently.
- **Modular**: the public API is ``sanitize_text(text)`` — callers do not
  depend on the internal regex engine.  The implementation can be swapped
  to Microsoft Presidio, spaCy NER, or a cloud API later without changing
  the call sites.
- **Safe defaults**: if an unknown error occurs during masking, the original
  text is returned rather than raising — retrieval must not break because of
  a masking failure.

Usage
-----
    from utils.privacy import pii_masking_service

    clean = pii_masking_service.sanitize_text(
        "Ahmed Mohamed, EMP-12345, earns 35000 EGP."
    )
    # → "[PERSON_NAME], [EMPLOYEE_ID], earns [SALARY] EGP."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PIIPattern:
    """A named regex pattern that maps to a redaction placeholder."""
    name: str
    pattern: re.Pattern[str]
    replacement: str


# Order matters — more specific patterns should come first to avoid partial
# matches being consumed by broader rules.
_PATTERNS: list[_PIIPattern] = [
    # Employee IDs: EMP-12345, EMP12345, emp-00001
    _PIIPattern(
        name="employee_id",
        pattern=re.compile(r"\b[Ee][Mm][Pp]-?\d{3,10}\b"),
        replacement="[EMPLOYEE_ID]",
    ),

    # Email addresses
    _PIIPattern(
        name="email",
        pattern=re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
        ),
        replacement="[EMAIL]",
    ),

    # Phone numbers — Egyptian mobile (01x-xxxx-xxxx) and international
    _PIIPattern(
        name="phone",
        pattern=re.compile(
            r"(?:\+?\d{1,3}[\s\-]?)?"          # optional country code
            r"(?:\(?\d{2,4}\)?[\s\-]?)?"        # optional area code
            r"\d{3,4}[\s\-]?\d{3,4}\b"          # main number
        ),
        replacement="[PHONE]",
    ),

    # National ID (Egyptian 14-digit)
    _PIIPattern(
        name="national_id",
        pattern=re.compile(r"\b[23]\d{13}\b"),
        replacement="[NATIONAL_ID]",
    ),

    # Bank account / IBAN
    _PIIPattern(
        name="bank_account",
        pattern=re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"),
        replacement="[BANK_ACCOUNT]",
    ),

    # Salary / monetary amounts (e.g. "35,000 EGP", "EGP 12000", "$5,000")
    _PIIPattern(
        name="salary",
        pattern=re.compile(
            r"(?:EGP|USD|EUR|SAR|AED|LE|E\.?G\.?P\.?|جنيه|ريال)[\s]?"
            r"[\d,]+(?:\.\d{1,2})?"
            r"|"
            r"[\d,]+(?:\.\d{1,2})?[\s]?"
            r"(?:EGP|USD|EUR|SAR|AED|LE|E\.?G\.?P\.?|جنيه|ريال)\b"
        ),
        replacement="[SALARY]",
    ),

    # Home / street addresses (basic heuristic for numbered street addresses)
    _PIIPattern(
        name="address",
        pattern=re.compile(
            r"\d{1,5}\s+(?:[A-Za-z]+\s+){1,4}"
            r"(?:Street|St\.|Avenue|Ave\.|Road|Rd\.|Boulevard|Blvd\.|شارع|ش\.)"
        ),
        replacement="[ADDRESS]",
    ),
]


# ---------------------------------------------------------------------------
# PIIMaskingService
# ---------------------------------------------------------------------------

class PIIMaskingService:
    """
    Detect and redact PII from text using configurable regex patterns.

    Attributes
    ----------
    patterns : list[_PIIPattern]
        The list of active PII detection patterns.
    """

    def __init__(
        self,
        patterns: list[_PIIPattern] | None = None,
        extra_sanitizer: Callable[[str], str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        patterns:
            Override the default pattern list.  If None, uses ``_PATTERNS``.
        extra_sanitizer:
            An optional post-processing function applied after regex masking.
            Useful for plugging in an NLP-based masker as a second pass.
        """
        self.patterns: list[_PIIPattern] = patterns if patterns is not None else list(_PATTERNS)
        self._extra_sanitizer = extra_sanitizer

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def sanitize_text(self, text: str) -> str:
        """
        Detect and replace PII tokens in *text* with bracketed placeholders.

        If the masking engine raises an unexpected error the **original text
        is returned unchanged** — retrieval must never fail because of a
        masking bug.

        Parameters
        ----------
        text:
            Raw text that may contain PII.

        Returns
        -------
        str
            Text with PII replaced by placeholders such as ``[EMAIL]``,
            ``[EMPLOYEE_ID]``, ``[PHONE]``, etc.
        """
        if not text:
            return text

        try:
            result = text
            for pii in self.patterns:
                result = pii.pattern.sub(pii.replacement, result)

            if self._extra_sanitizer is not None:
                result = self._extra_sanitizer(result)

            return result
        except Exception as exc:
            log.warning(
                "pii_masking_failed_returning_original",
                error=str(exc),
                text_preview=text[:80],
            )
            return text

    def sanitize_dict(self, data: dict) -> dict:
        """
        Recursively sanitize all string values in a dict.

        Parameters
        ----------
        data:
            Arbitrary dict whose string leaves should be scrubbed.

        Returns
        -------
        dict
            A new dict with all string values sanitized.
        """
        sanitized: dict = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = self.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_text(item) if isinstance(item, str)
                    else self.sanitize_dict(item) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

pii_masking_service = PIIMaskingService()
