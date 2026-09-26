"""
src/blocking/tokens.py — Exact-key and phonetic blockers: B6, B7, B8.

B6: Exact keys (shared postcode_like + name TF-IDF > 0.25;
    shared rare name token (top-5% IDF); same house_no + street token).

B7: Phonetic — shared rare name_skel token OR skeleton char TF-IDF top-K.
    Handles transliteration variants (Shree/Sri Ganesh Traders).

B8: Address-anchored — same postcode_like + same house_no + street-token
    overlap ≥ 1, regardless of name. Catches DBA/trade-name cases where
    the business name is completely different but the address is the same.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# IDF helpers
# ---------------------------------------------------------------------------

def _compute_idf(texts: List[str], min_df: int = 1) -> Dict[str, float]:
    """Compute IDF for each token across all texts.

    Returns a dict mapping token → IDF value.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer as _TF
    vect = _TF(analyzer="word", min_df=min_df, norm=None, use_idf=True, smooth_idf=True)
    vect.fit(texts)
    feature_names = vect.get_feature_names_out()
    idf_values = vect.idf_
    return dict(zip(feature_names, idf_values))


def _rare_tokens(idf_map: Dict[str, float], top_pct: float = 0.05) -> Set[str]:
    """Return the top-pct% highest-IDF tokens (rarest tokens)."""
    if not idf_map:
        return set()
    sorted_tokens = sorted(idf_map, key=lambda t: idf_map[t], reverse=True)
    cutoff = max(1, int(len(sorted_tokens) * top_pct))
    return set(sorted_tokens[:cutoff])


# ---------------------------------------------------------------------------
# B6 — Exact key blocking
# ---------------------------------------------------------------------------

class ExactKeyBlocker:
    """B6: Exact-match blocking on shared structural keys.

    Three sub-keys (OR combination):
    1. Same postcode_like + name TF-IDF cosine > 0.25.
    2. Shared rare name token (top-5% IDF).
    3. Same house_no + shared street token (non-digit addr token).
    """

    def __init__(self, cap: int = 30) -> None:
        self.cap = cap
        self._name_idf: Dict[str, float] = {}
        self._rare_name_tokens: Set[str] = set()
        self._name_vectorizer: Optional[TfidfVectorizer] = None

    def fit(self, all_df: pd.DataFrame) -> "ExactKeyBlocker":
        """Fit IDF and vectorizer on combined train+test text."""
        texts = all_df["name_tokens"].fillna("").tolist()
        self._name_idf = _compute_idf(texts)
        self._rare_name_tokens = _rare_tokens(self._name_idf, top_pct=0.05)

        self._name_vectorizer = TfidfVectorizer(
            analyzer="word", min_df=1, norm="l2", dtype=np.float32,
        )
        self._name_vectorizer.fit(texts)
        return self

    def _street_tokens(self, addr_tokens: str) -> Set[str]:
        """Extract non-digit address tokens (street words)."""
        return {t for t in addr_tokens.split() if not t.isdigit() and len(t) > 1}

    def retrieve(
        self,
        s1_df: pd.DataFrame,
        s23_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Retrieve B6 candidate pairs via exact key matching."""
        # Index S2/S3 records by their key structures
        # Key 1: postcode_like → list of s23 entity_ids
        post_index: Dict[str, List[str]] = defaultdict(list)
        # Key 2: rare name token → list of s23 entity_ids
        rare_tok_index: Dict[str, List[str]] = defaultdict(list)
        # Key 3: (house_no, street_tok) → list of s23 entity_ids
        house_street_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for _, row in s23_df.iterrows():
            eid = row["entity_id"]
            for pc in str(row.get("postcode_like", "")).split():
                if pc:
                    post_index[pc].append(eid)
            for tok in str(row.get("name_tokens", "")).split():
                if tok in self._rare_name_tokens:
                    rare_tok_index[tok].append(eid)
            hn = str(row.get("house_no", "")).strip()
            if hn:
                for st in self._street_tokens(str(row.get("addr_tokens", ""))):
                    house_street_index[(hn, st)].append(eid)

        # TF-IDF matrix for name-similarity check (Key 1)
        s23_name_mat = self._name_vectorizer.transform(
            s23_df["name_tokens"].fillna("").tolist()
        )
        s23_ids = s23_df["entity_id"].tolist()
        s23_id_to_idx = {eid: i for i, eid in enumerate(s23_ids)}

        rows = []
        for _, row in s1_df.iterrows():
            s1_id = row["entity_id"]
            candidates: Set[str] = set()

            # Key 1: shared postcode
            post_cands: Set[str] = set()
            for pc in str(row.get("postcode_like", "")).split():
                if pc:
                    post_cands.update(post_index.get(pc, []))

            # Filter by name TF-IDF cosine > 0.25
            if post_cands:
                q_name = self._name_vectorizer.transform(
                    [str(row.get("name_tokens", ""))]
                )
                for cand_id in post_cands:
                    c_idx = s23_id_to_idx.get(cand_id)
                    if c_idx is not None:
                        sim = float((q_name @ s23_name_mat[c_idx].T).toarray()[0, 0])
                        if sim > 0.25:
                            candidates.add(cand_id)

            # Key 2: shared rare name token
            for tok in str(row.get("name_tokens", "")).split():
                if tok in self._rare_name_tokens:
                    candidates.update(rare_tok_index.get(tok, []))

            # Key 3: same house_no + shared street token
            hn = str(row.get("house_no", "")).strip()
            if hn:
                for st in self._street_tokens(str(row.get("addr_tokens", ""))):
                    candidates.update(house_street_index.get((hn, st), []))

            # Cap
            candidates.discard(s1_id)
            cand_list = list(candidates)[: self.cap]
            for cand_id in cand_list:
                rows.append(
                    {
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": cand_id,
                        "b6_hit": 1,
                    }
                )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "b6_hit"]
        )


# ---------------------------------------------------------------------------
# B7 — Phonetic blocking
# ---------------------------------------------------------------------------

class PhoneticBlocker:
    """B7: Shared rare skeleton token OR skeleton char TF-IDF top-K."""

    def __init__(self, top_k: int = 20) -> None:
        self.top_k = top_k
        self._rare_skels: Set[str] = set()
        self._skel_vectorizer: Optional[TfidfVectorizer] = None
        self._skel_idf: Dict[str, float] = {}

    def fit(self, all_df: pd.DataFrame) -> "PhoneticBlocker":
        """Fit on skeleton tokens from all records."""
        skel_texts = all_df["name_skel"].fillna("").tolist()
        self._skel_idf = _compute_idf(skel_texts)
        self._rare_skels = _rare_tokens(self._skel_idf, top_pct=0.05)

        self._skel_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            max_features=100_000, norm="l2", dtype=np.float32,
        )
        self._skel_vectorizer.fit(skel_texts)
        return self

    def retrieve(
        self,
        s1_df: pd.DataFrame,
        s23_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Retrieve phonetically similar candidates."""
        # Index by rare skeleton tokens
        rare_skel_index: Dict[str, List[str]] = defaultdict(list)
        for _, row in s23_df.iterrows():
            for tok in str(row.get("name_skel", "")).split():
                if tok in self._rare_skels:
                    rare_skel_index[tok].append(row["entity_id"])

        # Skeleton char TF-IDF matrix
        s23_skel_mat = self._skel_vectorizer.transform(
            s23_df["name_skel"].fillna("").tolist()
        )
        s23_ids = s23_df["entity_id"].tolist()

        rows = []
        for _, row in s1_df.iterrows():
            s1_id = row["entity_id"]
            candidates: Set[str] = set()

            # Shared rare skeleton token
            for tok in str(row.get("name_skel", "")).split():
                if tok in self._rare_skels:
                    candidates.update(rare_skel_index.get(tok, []))

            # Skeleton char TF-IDF top-K
            q_skel = self._skel_vectorizer.transform(
                [str(row.get("name_skel", ""))]
            )
            sims = (q_skel @ s23_skel_mat.T).toarray()[0]
            top_idx = np.argsort(-sims)[: self.top_k]
            for idx in top_idx:
                if sims[idx] > 0.1:
                    candidates.add(s23_ids[idx])

            candidates.discard(s1_id)
            for cand_id in list(candidates)[: self.top_k]:
                rows.append(
                    {
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": cand_id,
                        "b7_hit": 1,
                    }
                )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "b7_hit"]
        )


# ---------------------------------------------------------------------------
# B8 — Address-anchored blocking
# ---------------------------------------------------------------------------

class AddressAnchorBlocker:
    """B8: Same postcode + same house_no + ≥1 shared street token, regardless of name."""

    def __init__(self, cap: int = 20) -> None:
        self.cap = cap

    def _street_tokens(self, addr_tokens: str) -> Set[str]:
        return {t for t in addr_tokens.split() if not t.isdigit() and len(t) > 1}

    def retrieve(
        self,
        s1_df: pd.DataFrame,
        s23_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Retrieve address-anchored candidates."""
        # Index S2/S3 by (postcode, house_no) → {street_tokens, entity_id}
        anchor_index: Dict[Tuple[str, str], List[Tuple[Set[str], str]]] = defaultdict(list)
        for _, row in s23_df.iterrows():
            hn = str(row.get("house_no", "")).strip()
            if not hn:
                continue
            for pc in str(row.get("postcode_like", "")).split():
                if pc:
                    st = self._street_tokens(str(row.get("addr_tokens", "")))
                    anchor_index[(pc, hn)].append((st, row["entity_id"]))

        rows = []
        for _, row in s1_df.iterrows():
            s1_id = row["entity_id"]
            s1_hn = str(row.get("house_no", "")).strip()
            if not s1_hn:
                continue
            s1_st = self._street_tokens(str(row.get("addr_tokens", "")))
            if not s1_st:
                continue

            candidates: Set[str] = set()
            for pc in str(row.get("postcode_like", "")).split():
                if pc:
                    for (c_st, c_id) in anchor_index.get((pc, s1_hn), []):
                        if s1_st & c_st:  # at least 1 shared street token
                            candidates.add(c_id)

            candidates.discard(s1_id)
            for cand_id in list(candidates)[: self.cap]:
                rows.append(
                    {
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": cand_id,
                        "b8_hit": 1,
                    }
                )

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "b8_hit"]
        )
