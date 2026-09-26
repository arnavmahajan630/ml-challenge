"""
src/text/phonetic.py — Phonetic skeleton and metaphone keys for transliteration robustness.

The consonant skeleton is designed to match common Indian transliteration
variants (Shree/Sri, Ganesh/Ganess, etc.) and French names with varying
accent conventions.

Skeleton algorithm (§6.2):
1. Lowercase and ASCII-fold the token.
2. Apply phonetic equivalences: ph→f, ck→k, q→k, z→s, w→v, x→ks.
3. Drop 'h' after consonants: sh→s, th→t, dh→d, bh→b, kh→k.
4. Drop vowels except the first character.
5. Collapse repeated letters.

metaphone_tokens: thin wrapper around jellyfish.metaphone per token.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List


# ---------------------------------------------------------------------------
# Skeleton algorithm
# ---------------------------------------------------------------------------

_PH_SUBS = [
    ("ph", "f"),
    ("ck", "k"),
    ("qu", "k"),
    ("q", "k"),
    ("z", "s"),
    ("w", "v"),
    ("x", "ks"),
]

_AFTER_CONS_H = {
    "sh": "s",
    "th": "t",
    "dh": "d",
    "bh": "b",
    "kh": "k",
    "gh": "g",
    "ch": "c",
}


def _ascii_fold(s: str) -> str:
    """NFKD decompose and drop combining marks → pure ASCII lower."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower()


def skeleton(token: str) -> str:
    """Compute the phonetic consonant skeleton of a single token.

    Args:
        token: A single word/token (may contain Unicode).

    Returns:
        A short ASCII string representing the phonetic skeleton.
        Returns empty string for empty input.

    Examples:
        >>> skeleton("Shree")
        'sre'
        >>> skeleton("Sri")
        'sri'
        >>> skeleton("Ganesh")
        'gns'
        >>> skeleton("Traders")
        'trdrs'
        >>> skeleton("Corporation")
        'crprtn'
    """
    if not token:
        return ""

    s = _ascii_fold(token)

    # 1. Apply phonetic substitutions (order matters)
    for pat, rep in _PH_SUBS:
        s = s.replace(pat, rep)

    # 2. Drop 'h' after specific consonants (apply longest match first)
    result = []
    i = 0
    while i < len(s):
        # Check 2-char combos
        if i + 1 < len(s) and s[i : i + 2] in _AFTER_CONS_H:
            result.append(_AFTER_CONS_H[s[i : i + 2]])
            i += 2
            continue
        result.append(s[i])
        i += 1
    s = "".join(result)

    # 3. Drop vowels except the first character
    vowels = set("aeiou")
    if not s:
        return ""
    first = s[0]
    rest = "".join(c for c in s[1:] if c not in vowels)
    s = first + rest

    # 4. Collapse consecutive repeated letters
    s = re.sub(r"(.)\1+", r"\1", s)

    return s


def skeleton_tokens(text: str) -> List[str]:
    """Compute skeleton for each whitespace-separated token in *text*.

    Args:
        text: Space-separated token string (e.g. ``name_tokens`` field).

    Returns:
        List of skeleton strings (empty skeletons filtered out).
    """
    return [sk for tok in text.split() if (sk := skeleton(tok))]


def skeleton_string(text: str) -> str:
    """Return the space-joined skeleton of all tokens in *text*."""
    return " ".join(skeleton_tokens(text))


# ---------------------------------------------------------------------------
# Metaphone
# ---------------------------------------------------------------------------

def metaphone_token(token: str) -> str:
    """Return the jellyfish metaphone code for a single token.

    Falls back to the token itself if jellyfish is not installed.
    """
    try:
        import jellyfish  # type: ignore
        return jellyfish.metaphone(_ascii_fold(token))
    except ImportError:
        return _ascii_fold(token)


def metaphone_tokens(text: str) -> List[str]:
    """Return metaphone codes for all tokens in *text*.

    Args:
        text: Space-separated token string.

    Returns:
        List of non-empty metaphone strings.
    """
    return [m for tok in text.split() if (m := metaphone_token(tok))]


def metaphone_string(text: str) -> str:
    """Return the space-joined metaphone codes for *text*."""
    return " ".join(metaphone_tokens(text))
