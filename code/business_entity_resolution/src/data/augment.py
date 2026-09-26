"""
src/data/augment.py — Label-free noise generator for format-invariance training.

This module provides a single generic augmentation function that applies
country-agnostic noise to a business record.  It is used in three places:

1. **Bi-encoder training (§7.2):** 20% of each batch = (record, noised copy)
   label-free pairs — teaches the encoder that noised variants are equivalent.

2. **Cross-encoder training (§8.2):** p=0.3 noise on one side of a pair.

3. **Test-time adaptation (§10):** self-supervised (test record, noised copy)
   pairs for domain adaptation to France.

All operations are purely generic linguistic/format knowledge — no real
business names or addresses are hard-coded.
"""
from __future__ import annotations

import random
import re
import unicodedata
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Noise operation library
# ---------------------------------------------------------------------------

# Generic landmark phrases (country-agnostic)
_LANDMARK_PHRASES = [
    "near the market",
    "opp station",
    "behind the park",
    "next to the hospital",
    "adjacent to the mall",
    "near the bank",
    "opposite the school",
]

# Common legal suffixes to drop (taken from abbrev.yaml seed content)
_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "cie",
    "ltd", "limited", "llc", "llp", "lp", "plc", "pvt", "private", "pte",
    "pty", "gmbh", "ag", "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc",
    "ste", "sté", "ei", "scop", "grp", "group", "groupe", "hldg", "holding",
    "intl", "international", "mfg", "bros", "brothers", "enterprises", "ent",
    "traders", "industries", "ind",
}


def _strip_accents(text: str) -> str:
    """Remove diacritics via NFKD decomposition."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if unicodedata.category(c) != "Mn"
    )


def _add_accent(token: str) -> str:
    """Randomly replace a vowel with an accented variant."""
    accent_map = {
        "a": ["à", "â", "ä"],
        "e": ["é", "è", "ê", "ë"],
        "i": ["î", "ï"],
        "o": ["ô", "ö"],
        "u": ["ù", "û", "ü"],
        "c": ["ç"],
    }
    chars = list(token)
    positions = [i for i, c in enumerate(chars) if c.lower() in accent_map]
    if not positions:
        return token
    pos = random.choice(positions)
    char = chars[pos].lower()
    chars[pos] = random.choice(accent_map[char])
    return "".join(chars)


def _vowel_drop(token: str) -> str:
    """Drop interior vowels (skeleton-style transliteration noise)."""
    if len(token) <= 2:
        return token
    vowels = set("aeiouAEIOU")
    first = token[0]
    rest = "".join(c for c in token[1:] if c not in vowels)
    return first + rest if rest else token


def _one_char_typo(token: str) -> str:
    """Apply one random character-level perturbation."""
    if len(token) < 2:
        return token
    ops = ["sub", "del", "transpose"]
    op = random.choice(ops)
    pos = random.randint(0, len(token) - 1)
    chars = list(token)
    if op == "sub":
        chars[pos] = random.choice("abcdefghijklmnopqrstuvwxyz")
    elif op == "del" and len(token) > 1:
        chars.pop(pos)
    elif op == "transpose" and pos < len(token) - 1:
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def _drop_legal_suffix(name: str) -> str:
    """Remove trailing legal-form tokens from a name string."""
    tokens = name.split()
    while tokens and tokens[-1].lower().rstrip(".") in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) if tokens else name


def _abbreviate_token(token: str, abbrev_map: Dict[str, List[str]]) -> str:
    """Replace a token with one of its abbreviations (if known)."""
    key = token.lower()
    if key in abbrev_map:
        return random.choice(abbrev_map[key])
    # Try reverse: abbreviation → full form
    for full, abbrevs in abbrev_map.items():
        if key in [a.lower() for a in abbrevs]:
            return full
    return token


def _drop_postcode(address: str) -> str:
    """Remove digit runs that look like postcodes (4–6 digits)."""
    return re.sub(r"\b\d{4,6}\b", "", address).strip()


def _drop_house_no(address: str) -> str:
    """Remove the first short digit run (house number) from an address."""
    return re.sub(r"^\d{1,5}[a-zA-Z]?\s*", "", address).strip()


def _shuffle_tokens(text: str) -> str:
    """Randomly shuffle word tokens."""
    tokens = text.split()
    random.shuffle(tokens)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Public augmentation function
# ---------------------------------------------------------------------------

def augment_record(
    record: Dict[str, Any],
    p: float = 0.3,
    rng: Optional[random.Random] = None,
    abbrev_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Apply a random subset of format-invariance noise ops to a record.

    Operates on the ``name_clean`` and ``addr_clean`` fields.  Other fields
    (entity_id, country, etc.) are copied unchanged.

    Args:
        record: Dict with at least ``name_clean`` and ``addr_clean`` fields.
        p: Probability of applying each individual noise operation.
        rng: Optional :class:`random.Random` instance for reproducibility.
        abbrev_map: Optional canonical abbreviation map (from abbrev.yaml).
                    Keys are full forms; values are lists of short forms.

    Returns:
        A new dict (copy) with noised ``name_clean`` / ``addr_clean``.
        Original values preserved in ``orig_name_clean`` / ``orig_addr_clean``.
    """
    if rng is None:
        rng = random

    augmented = deepcopy(record)
    augmented["orig_name_clean"] = record.get("name_clean", "")
    augmented["orig_addr_clean"] = record.get("addr_clean", "")

    name = record.get("name_clean", "") or ""
    addr = record.get("addr_clean", "") or ""

    # --- Name-level ops ---
    if rng.random() < p:
        name = _drop_legal_suffix(name)

    if rng.random() < p and abbrev_map:
        tokens = name.split()
        if tokens:
            idx = rng.randint(0, len(tokens) - 1)
            tokens[idx] = _abbreviate_token(tokens[idx], abbrev_map)
            name = " ".join(tokens)

    if rng.random() < p:
        tokens = name.split()
        if tokens:
            idx = rng.randint(0, len(tokens) - 1)
            tokens[idx] = _vowel_drop(tokens[idx])
            name = " ".join(tokens)

    if rng.random() < p:
        tokens = name.split()
        if tokens:
            idx = rng.randint(0, len(tokens) - 1)
            tokens[idx] = _one_char_typo(tokens[idx])
            name = " ".join(tokens)

    if rng.random() < p:
        name = _strip_accents(name)
    elif rng.random() < p:
        tokens = name.split()
        if tokens:
            idx = rng.randint(0, len(tokens) - 1)
            tokens[idx] = _add_accent(tokens[idx])
            name = " ".join(tokens)

    # --- Address-level ops ---
    if rng.random() < p:
        addr = _drop_postcode(addr)

    if rng.random() < p:
        addr = _drop_house_no(addr)

    if rng.random() < p and addr:
        addr = _shuffle_tokens(addr)

    if rng.random() < p:
        addr = addr + " " + rng.choice(_LANDMARK_PHRASES)

    augmented["name_clean"] = name.strip()
    augmented["addr_clean"] = addr.strip()

    return augmented


def make_label_free_pairs(
    records: List[Dict[str, Any]],
    n_pairs: int,
    p: float = 0.3,
    seed: int = 42,
    abbrev_map: Optional[Dict[str, List[str]]] = None,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Generate (original, noised) label-free positive pairs.

    Used for bi-encoder self-supervised training and TTA.

    Args:
        records: List of record dicts (each needs ``name_clean``, ``addr_clean``).
        n_pairs: How many pairs to generate.
        p: Noise probability per operation.
        seed: Random seed.
        abbrev_map: Optional abbreviation map for token abbreviation ops.

    Returns:
        List of (original_record, noised_copy) tuples.
    """
    rng = random.Random(seed)
    pairs = []
    for _ in range(n_pairs):
        rec = rng.choice(records)
        noised = augment_record(rec, p=p, rng=rng, abbrev_map=abbrev_map)
        pairs.append((rec, noised))
    return pairs
