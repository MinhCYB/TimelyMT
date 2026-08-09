"""Deterministic human-readable lexical tokenization for source streaming."""

from __future__ import annotations

import re


# Letters and digits form tokens. Apostrophes and hyphens are retained only
# inside words; decimal points and common C/C++ suffixes retain lexical meaning.
_LEXICAL_TOKEN = re.compile(
    r"[^\W_]+(?:['\u2019-][^\W_]+)*(?:\.[0-9]+)?(?:\+\+|#)?",
    re.UNICODE,
)


def lexical_tokens(text: str) -> list[str]:
    """Return model-independent lexical units without subtitle punctuation cues."""
    return _LEXICAL_TOKEN.findall(text)


def alphanumeric_weight(token: str) -> int:
    """Return the simple character weight used by weighted timing allocation."""
    return max(1, sum(character.isalnum() for character in token))
