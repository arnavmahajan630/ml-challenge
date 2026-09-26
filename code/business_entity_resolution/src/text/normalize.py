"""
src/text/normalize.py — Full text normalization pipeline for BER.

Produces, per record, the following derived fields:
    name_clean       : raw text with Unicode + whitespace cleanup only (for neural models)
    addr_clean       : same for address
    name_norm        : fully normalized name (abbrevs applied, punctuation stripped)
    name_core        : name_norm minus legal forms and stopwords
    name_legal       : "|"-joined set of legal-form tokens found
    addr_norm        : fully normalized address
    name_tokens      : space-joined word tokens of name_norm
    addr_tokens      : space-joined word tokens of addr_norm
    digits           : space-joined all digit runs from both fields
    postcode_like    : space-joined digit runs 4-6 chars + alphanumeric postal patterns
    house_no         : first short digit run in address (with suffixes: 12b, 12 bis, 12-A)
    landmark_flag    : "1" / "0"
    dba_flag         : "1" / "0"
    alias_names      : "|"-joined list of name aliases (split on DBA markers, parens)
    name_skel        : space-joined phonetic skeleton tokens of name_norm
    addr_skel        : space-joined phonetic skeleton tokens of addr_norm
    field_pattern    : int bitmask (bit0=name_digits, bit1=postcode, bit2=house_no,
                       bit3=landmark)

All operations are country-agnostic. The abbrev map is loaded from configs/abbrev.yaml.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from .phonetic import skeleton_string


# ---------------------------------------------------------------------------
# Load abbreviation map
# ---------------------------------------------------------------------------

_ABBREV_MAP: Dict[str, List[str]] = {}  # full_form → [abbrev, ...]
_ABBREV_REV: Dict[str, str] = {}        # abbrev → canonical_full

_LEGAL_FORMS: Set[str] = set()
_STOPWORDS: Set[str] = {
    "the", "and", "of", "et", "de", "du", "des", "la", "le", "les",
    "l", "d", "a", "au", "aux",
}

# Landmark / DBA indicator patterns
_LANDMARK_RE = re.compile(
    r"\b(near|nr|opp|opposite|behind|bhd|adj|adjacent|next\s+to|"
    r"in\s+front\s+of|facing|off)\b",
    re.IGNORECASE,
)
_DBA_RE = re.compile(
    r"\b(dba|d\.b\.a\.?|t/a|trading\s+as|aka|also\s+known\s+as)\b",
    re.IGNORECASE,
)
_PAREN_RE = re.compile(r"\(([^)]+)\)")

# Postal noise patterns to strip
_POSTAL_NOISE_RE = re.compile(
    r"\b(cedex\s*\d*|bp\s*\d+|po\s+box\s*\d+|p\.?o\.?\s*box\s*\d+|"
    r"pin\s*:?\s*\d{4,6}|pincode\s*:?\s*\d{4,6}|zip\s*:?\s*\d{4,6})\b",
    re.IGNORECASE,
)
# Postcode-like: 4-6 digit runs, or alphanumeric postal codes like "SW1A 2AA"
_POSTCODE_DIGIT_RE = re.compile(r"\b\d{4,6}\b")
_POSTCODE_ALPHA_RE = re.compile(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b")
# House number: leading digit run with optional suffix
_HOUSE_NO_RE = re.compile(r"^\s*(\d{1,5}(?:\s*(?:bis|ter|[a-zA-Z]))?)\b", re.IGNORECASE)


def load_abbrev_map(abbrev_yaml_path: str | Path) -> None:
    """Load the abbreviation map from configs/abbrev.yaml into module globals.

    Call once at startup before normalizing any records.
    """
    global _ABBREV_MAP, _ABBREV_REV, _LEGAL_FORMS
    path = Path(abbrev_yaml_path)
    if not path.exists():
        return

    data: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    legal_section = data.get("legal_forms", {})
    addr_section = data.get("address_words", {})
    number_section = data.get("number_words", {})
    saint_section = data.get("saints", {})

    for section in (legal_section, addr_section, number_section, saint_section):
        for canonical, variants in section.items():
            canonical_lc = canonical.lower()
            _ABBREV_MAP[canonical_lc] = [str(v).lower() for v in variants]
            for v in variants:
                _ABBREV_REV[str(v).lower()] = canonical_lc

    _LEGAL_FORMS = {k.lower() for k in legal_section}
    # Also add abbreviations of legal forms
    for canon, variants in legal_section.items():
        _LEGAL_FORMS.update(str(v).lower() for v in variants)


# ---------------------------------------------------------------------------
# Unicode / basic cleanup
# ---------------------------------------------------------------------------

def _nfkd_clean(text: str) -> str:
    """NFKD decompose, drop combining marks, normalize whitespace."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _clean(text: str) -> str:
    """Minimal clean: Unicode normalize + collapse whitespace.

    Used to produce ``name_clean`` / ``addr_clean`` for neural models.
    """
    text = _nfkd_clean(text)
    # Normalize quotes and apostrophes
    text = re.sub(r"[""«»]", '"', text)
    text = re.sub(r"[''`´]", "'", text)
    # Normalize dashes
    text = re.sub(r"[–—‐‑‒]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Full normalization
# ---------------------------------------------------------------------------

def _elision_split(text: str) -> str:
    """Split French elisions: l'atelier → l atelier."""
    return re.sub(r"\b(l|d|qu|j|m|t|s|n|c)'", r"\1 ", text, flags=re.IGNORECASE)


def _symbol_sub(text: str) -> str:
    """Replace & and + with 'and'."""
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\+", " and ", text)
    return text


def _normalize_punct(text: str) -> str:
    """Remove punctuation except inside digit runs and within abbreviations.

    'p.v.t.' → 'pvt'; '123-456' → '123-456'.
    """
    # Collapse abbreviation dots: p.v.t. → pvt, u.s.a. → usa
    text = re.sub(r"\b([a-zA-Z])\.([a-zA-Z])\.([a-zA-Z])\.?", r"\1\2\3", text)
    text = re.sub(r"\b([a-zA-Z])\.([a-zA-Z])\.?", r"\1\2", text)
    # Keep hyphens within digit runs (ZIP+4: 12345-6789)
    text = re.sub(r"(\d)-(\d)", r"\1HYPHEN\2", text)
    # Remove remaining punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.replace("HYPHEN", "-")
    return text


def _apply_abbrev(token: str) -> List[str]:
    """Return the canonical form of *token* (or itself if not found).

    For ambiguous tokens (like 'st'), returns all alternatives.
    """
    lc = token.lower()

    # Direct canonical lookup
    if lc in _ABBREV_MAP:
        # This token IS a canonical form; also keep abbreviations as alternatives
        return [lc] + _ABBREV_MAP[lc]

    # Reverse lookup: is this an abbreviation of something?
    if lc in _ABBREV_REV:
        return [_ABBREV_REV[lc], lc]

    return [lc]


_NUMBER_WORDS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
    "un": "1", "une": "1", "deux": "2", "trois": "3", "quatre": "4",
    "cinq": "5", "six": "6", "sept": "7", "huit": "8", "neuf": "9", "dix": "10",
    # Ordinals
    "first": "1", "1st": "1", "1er": "1", "1ère": "1", "premier": "1",
    "second": "2", "2nd": "2", "2e": "2", "2ème": "2", "deuxième": "2",
    "third": "3", "3rd": "3", "3e": "3", "troisième": "3",
}


def _map_number_words(text: str) -> str:
    """Map number words to digits (both directions as alternatives)."""
    for word, digit in _NUMBER_WORDS.items():
        text = re.sub(r"\b" + re.escape(word) + r"\b", digit, text, flags=re.IGNORECASE)
    return text


def normalize(text: str) -> str:
    """Full normalization pipeline for a single text field.

    Returns the normalized string (primary canonical form).
    Tokens are single-space separated.
    """
    if not text or not text.strip():
        return ""

    text = _nfkd_clean(text)
    # Normalize quotes / dashes
    text = re.sub(r"[""«»]", '"', text)
    text = re.sub(r"[''`´]", "'", text)
    text = re.sub(r"[–—‐‑‒]", "-", text)
    text = text.lower()
    text = _elision_split(text)
    text = _symbol_sub(text)
    # Strip postal noise into flags (done before tokenization)
    text = _POSTAL_NOISE_RE.sub(" ", text)
    text = _normalize_punct(text)
    text = _map_number_words(text)

    tokens = text.split()
    # Apply abbreviation canonicalization (use first/primary form only)
    norm_tokens = [_apply_abbrev(t)[0] for t in tokens]
    return " ".join(t for t in norm_tokens if t)


def extract_digits(name: str, addr: str) -> List[str]:
    """Extract all digit runs from name and address."""
    return re.findall(r"\d+", name + " " + addr)


def extract_postcode_like(addr: str) -> List[str]:
    """Extract postcode-like patterns from address."""
    digit_codes = _POSTCODE_DIGIT_RE.findall(addr)
    alpha_codes = _POSTCODE_ALPHA_RE.findall(addr.upper())
    return list(dict.fromkeys(digit_codes + [c.strip() for c in alpha_codes]))


def extract_house_no(addr_norm: str) -> str:
    """Extract the first house number (short digit run with optional suffix)."""
    m = _HOUSE_NO_RE.match(addr_norm)
    return m.group(1).strip() if m else ""


def extract_alias_names(raw_name: str) -> List[str]:
    """Split a business name on DBA markers and parenthetical expressions."""
    aliases: List[str] = []
    # Split on DBA markers
    parts = _DBA_RE.split(raw_name)
    # DBA_RE.split returns [..., marker, ..., ...]; keep non-marker parts
    for i, part in enumerate(parts):
        if not _DBA_RE.match(part.strip()):
            cleaned = part.strip().strip('",;')
            if cleaned:
                aliases.append(cleaned)

    # Also extract parenthetical expressions
    for m in _PAREN_RE.finditer(raw_name):
        content = m.group(1).strip()
        if content and len(content) > 1:
            aliases.append(content)

    # Deduplicate while preserving order
    seen: Dict[str, None] = {}
    for a in aliases:
        seen[a] = None
    return list(seen.keys())


def extract_name_core(name_norm: str, name_legal: Set[str]) -> str:
    """Strip legal forms and stopwords from normalized name."""
    tokens = name_norm.split()
    core_tokens = [
        t for t in tokens
        if t.lower() not in name_legal
        and t.lower() not in _STOPWORDS
    ]
    return " ".join(core_tokens)


# ---------------------------------------------------------------------------
# Main per-record normalization
# ---------------------------------------------------------------------------

def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single business record dict.

    Input keys expected: ``business_name``, ``business_address`` (from TSV).
    May also have ``entity_id``, ``country``, etc. — all are preserved.

    Returns a new dict with all original fields plus the derived normalized fields.
    """
    raw_name = record.get("business_name", "") or ""
    raw_addr = record.get("business_address", "") or ""

    # --- Clean (for neural models) ---
    name_clean = _clean(raw_name)
    addr_clean = _clean(raw_addr)

    # --- Full normalization ---
    name_norm_str = normalize(raw_name)
    addr_norm_str = normalize(raw_addr)

    # --- Legal form extraction ---
    name_tokens_list = name_norm_str.split()
    legal_found: Set[str] = set()
    for tok in name_tokens_list:
        if tok.lower() in _LEGAL_FORMS:
            legal_found.add(tok.lower())

    # --- Name core ---
    name_core_str = extract_name_core(name_norm_str, legal_found)

    # --- Digit / postcode / house_no extraction ---
    digits_list = extract_digits(raw_name, raw_addr)
    postcode_list = extract_postcode_like(raw_addr)
    house_no = extract_house_no(addr_norm_str)

    # --- Landmark / DBA flags ---
    landmark_flag = "1" if _LANDMARK_RE.search(raw_addr) else "0"
    dba_flag = "1" if _DBA_RE.search(raw_name) else "0"

    # --- Alias names ---
    alias_list = extract_alias_names(raw_name)

    # --- Phonetic skeletons ---
    name_skel_str = skeleton_string(name_norm_str)
    addr_skel_str = skeleton_string(addr_norm_str)

    # --- Field pattern bitmask ---
    has_name_digits = int(bool(re.search(r"\d", raw_name)))
    has_postcode = int(bool(postcode_list))
    has_house_no = int(bool(house_no))
    has_landmark = int(landmark_flag == "1")
    field_pattern = (
        has_name_digits
        | (has_postcode << 1)
        | (has_house_no << 2)
        | (has_landmark << 3)
    )

    result = dict(record)
    result.update(
        {
            "name_clean": name_clean,
            "addr_clean": addr_clean,
            "name_norm": name_norm_str,
            "name_core": name_core_str,
            "name_legal": "|".join(sorted(legal_found)),
            "addr_norm": addr_norm_str,
            "name_tokens": " ".join(name_tokens_list),
            "addr_tokens": " ".join(addr_norm_str.split()),
            "digits": " ".join(digits_list),
            "postcode_like": " ".join(postcode_list),
            "house_no": house_no,
            "landmark_flag": landmark_flag,
            "dba_flag": dba_flag,
            "alias_names": "|".join(alias_list),
            "name_skel": name_skel_str,
            "addr_skel": addr_skel_str,
            "field_pattern": str(field_pattern),
        }
    )
    return result


def normalize_dataframe(df, abbrev_yaml_path: Optional[str] = None) -> "pd.DataFrame":
    """Normalize an entire DataFrame of records.

    Loads the abbrev map from *abbrev_yaml_path* if not already loaded.
    Adds all derived columns in-place and returns the DataFrame.
    """
    import pandas as pd

    if abbrev_yaml_path and not _LEGAL_FORMS:
        load_abbrev_map(abbrev_yaml_path)

    records = df.to_dict(orient="records")
    normalized = [normalize_record(r) for r in records]
    return pd.DataFrame(normalized)
