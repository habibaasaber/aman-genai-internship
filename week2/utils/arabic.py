"""
utils/arabic.py
===============
Arabic text pre-processing utilities for the Week 2 RAG project.

Two levels of normalisation are provided:

1. ``strip_diacritics``  — removes tashkeel (short vowel marks) only.
   Use when you want to preserve all consonant characters but ignore vowelisation.

2. ``normalise_arabic``  — strips diacritics AND unifies common character
   variants (alef forms, teh marbuta, waw, yeh).  Use for BM25 indexing and
   query normalisation to maximise lexical recall on diacritics-free queries.

Dependencies
------------
- ``pyarabic`` (pip install pyarabic) — provides strip_tashkeel().
- ``re`` (stdlib) — for character-class substitutions.
"""

from __future__ import annotations

import re
import unicodedata

import pyarabic.araby as araby


# ---------------------------------------------------------------------------
# Unicode ranges used in the substitution patterns below
# ---------------------------------------------------------------------------

# Alef variants: Alef with Hamza above (أ U+0623), Alef with Madda (آ U+0622),
# Alef with Hamza below (إ U+0625), Alef Wasla (ٱ U+0671) → plain Alef (ا U+0627)
_ALEF_VARIANTS_PATTERN = re.compile(r"[أإآٱ]")

# Teh Marbuta (ة U+0629) → Heh (ه U+0647)
_TEH_MARBUTA_PATTERN = re.compile(r"ة")

# Alef Maqsura (ى U+0649) → Yeh (ي U+064A)
_ALEF_MAQSURA_PATTERN = re.compile(r"ى")

# Waw with Hamza above (ؤ U+0624) → Waw (و U+0648)
_WAW_HAMZA_PATTERN = re.compile(r"ؤ")

# Yeh with Hamza above (ئ U+0626) → Yeh (ي U+064A)
_YEH_HAMZA_PATTERN = re.compile(r"ئ")

# Tatweel / kashida (ـ U+0640) — decorative elongation, no semantic value
_TATWEEL_PATTERN = re.compile(r"ـ")


def strip_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (tashkeel / harakat) from ``text``.

    Diacritics include short vowel marks (fatha, kasra, damma, sukun,
    shadda, etc.) encoded in the Unicode range U+064B–U+065F.

    This function delegates to ``pyarabic.araby.strip_tashkeel``, which
    handles all standard Arabic diacritic code points.

    Parameters
    ----------
    text:
        Input string — may contain Arabic, Latin, or mixed content.
        Non-Arabic characters are passed through unchanged.

    Returns
    -------
    str
        The input string with all Arabic diacritics removed.

    Examples
    --------
    >>> strip_diacritics("مَرْحَباً بِكَ")
    'مرحبا بك'
    >>> strip_diacritics("Hello مَرْحَباً")
    'Hello مرحبا'
    """
    if not text:
        return text

    return araby.strip_tashkeel(text)


def normalise_arabic(text: str) -> str:
    """
    Apply full Arabic text normalisation for lexical retrieval.

    Performs the following transformations in order:

    1. Strip diacritics (tashkeel) via ``strip_diacritics``.
    2. Unify alef variants (أ إ آ ٱ) → plain alef (ا).
    3. Replace teh marbuta (ة) with heh (ه).
    4. Replace alef maqsura (ى) with yeh (ي).
    5. Replace waw with hamza (ؤ) with plain waw (و).
    6. Replace yeh with hamza (ئ) with plain yeh (ي).
    7. Remove tatweel / kashida (ـ).
    8. Apply Unicode NFC normalisation to handle composed forms.
    9. Collapse multiple whitespace characters into a single space.

    Parameters
    ----------
    text:
        Input string — may be fully Arabic, bilingual, or contain mixed
        scripts. Non-Arabic characters (Latin, digits, punctuation) are
        left unchanged except for whitespace normalisation.

    Returns
    -------
    str
        The fully normalised string, suitable for BM25 indexing and
        diacritics-free query matching.

    Examples
    --------
    >>> normalise_arabic("الإنتَرنشِيبُ")
    'الانترنشيب'
    >>> normalise_arabic("مَرْحَباً  بِكَ")
    'مرحبا بك'
    >>> normalise_arabic("Hello مَرْحَباً")
    'Hello مرحبا'
    """
    if not text:
        return text

    # Step 1: Remove diacritics.
    result = strip_diacritics(text)

    # Step 2: Unify alef variants → plain alef.
    result = _ALEF_VARIANTS_PATTERN.sub("ا", result)

    # Step 3: Teh marbuta → heh.
    result = _TEH_MARBUTA_PATTERN.sub("ه", result)

    # Step 4: Alef maqsura → yeh.
    result = _ALEF_MAQSURA_PATTERN.sub("ي", result)

    # Step 5: Waw with hamza → plain waw.
    result = _WAW_HAMZA_PATTERN.sub("و", result)

    # Step 6: Yeh with hamza → plain yeh.
    result = _YEH_HAMZA_PATTERN.sub("ي", result)

    # Step 7: Remove tatweel / kashida.
    result = _TATWEEL_PATTERN.sub("", result)

    # Step 8: Unicode NFC — canonical composition.
    result = unicodedata.normalize("NFC", result)

    # Step 9: Collapse multiple whitespace → single space and strip edges.
    result = re.sub(r"\s+", " ", result).strip()

    return result


def is_arabic(text: str, threshold: float = 0.3) -> bool:
    """
    Heuristically detect whether ``text`` is predominantly Arabic.

    Counts characters in the Arabic Unicode block (U+0600–U+06FF) and
    returns True if they exceed ``threshold`` fraction of all letters.

    Parameters
    ----------
    text:
        Input string.
    threshold:
        Fraction of Arabic characters required to classify as Arabic.
        Default is 0.30 (30 %).

    Returns
    -------
    bool
        True if the text is predominantly Arabic, False otherwise.

    Examples
    --------
    >>> is_arabic("ما هي ساعات العمل؟")
    True
    >>> is_arabic("What are the working hours?")
    False
    """
    if not text:
        return False

    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False

    arabic_letters = [
        ch for ch in letters if "\u0600" <= ch <= "\u06FF"
    ]

    return len(arabic_letters) / len(letters) >= threshold
