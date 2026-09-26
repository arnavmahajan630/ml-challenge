"""
src/blocking/tfidf.py — TF-IDF based candidate generation blockers.

Implements blockers B1, B2, B3:
    B1: char_wb 2–4-gram TF-IDF on name_core
    B2: char_wb TF-IDF on name_core + " " + addr_norm
    B3: word TF-IDF (IDF-weighted) on name + address tokens

All blockers run S1→S2/S3 and S2/S3→S1 (bidirectional); union is
performed in blocking/union.py.

The TF-IDF vocabulary and IDF weights are fit on train+test combined
text (no labels involved; documented in README). This is allowed per §1.6.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Vectorizer builders
# ---------------------------------------------------------------------------

def _build_char_vectorizer(
    ngram_range: Tuple[int, int] = (2, 4),
    max_features: int = 300_000,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
        norm="l2",
        dtype=np.float32,
    )


def _build_word_vectorizer(max_features: int = 200_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        max_features=max_features,
        sublinear_tf=True,
        min_df=1,
        norm="l2",
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------

def _chunked_cosine_top_k(
    query_matrix: sp.csr_matrix,
    index_matrix: sp.csr_matrix,
    top_k: int,
    chunk_size: int = 1000,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute cosine top-K retrieval in chunks to avoid OOM.

    Args:
        query_matrix: (n_queries, vocab) sparse matrix (L2-normalised).
        index_matrix: (n_index, vocab) sparse matrix (L2-normalised).
        top_k: Number of candidates to retrieve per query.
        chunk_size: Number of queries per chunk.

    Returns:
        List of (indices, scores) tuples, one per query.
        indices[i] shape: (≤top_k,); scores[i] shape: (≤top_k,)
    """
    n_queries = query_matrix.shape[0]
    n_index = index_matrix.shape[0]
    top_k = min(top_k, n_index)

    results: List[Tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        chunk = query_matrix[start:end]
        # Cosine similarity = dot product (both matrices are L2-normalised)
        sims = (chunk @ index_matrix.T).toarray()  # (chunk, n_index)
        for row in sims:
            if top_k >= len(row):
                idx = np.argsort(-row)
            else:
                idx = np.argpartition(-row, top_k)[:top_k]
                idx = idx[np.argsort(-row[idx])]
            results.append((idx[:top_k], row[idx[:top_k]]))
    return results


# ---------------------------------------------------------------------------
# Blocker class
# ---------------------------------------------------------------------------

class TFIDFBlocker:
    """TF-IDF retrieval blocker supporting B1, B2, B3.

    Args:
        blocker_id: One of "B1", "B2", "B3".
        top_k: Candidates to retrieve per query.
    """

    def __init__(self, blocker_id: str, top_k: int = 30) -> None:
        self.blocker_id = blocker_id
        self.top_k = top_k
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._is_fit = False

    def _get_text(self, df: pd.DataFrame) -> List[str]:
        """Extract the text field used by this blocker."""
        if self.blocker_id == "B1":
            return df["name_core"].tolist()
        elif self.blocker_id == "B2":
            return (df["name_core"] + " " + df["addr_norm"]).tolist()
        elif self.blocker_id == "B3":
            return (df["name_tokens"] + " " + df["addr_tokens"]).tolist()
        else:
            raise ValueError(f"Unknown blocker_id: {self.blocker_id}")

    def fit(self, all_df: pd.DataFrame) -> "TFIDFBlocker":
        """Fit the vectorizer on combined train+test text (no labels).

        Args:
            all_df: Combined DataFrame of all records (train S1/S2/S3 + test S1/S2/S3).
        """
        texts = self._get_text(all_df)
        if self.blocker_id in ("B1", "B2"):
            self._vectorizer = _build_char_vectorizer()
        else:  # B3
            self._vectorizer = _build_word_vectorizer()
        self._vectorizer.fit(texts)
        self._is_fit = True
        return self

    def retrieve(
        self,
        query_df: pd.DataFrame,
        index_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Retrieve top-K candidates for each query record.

        Args:
            query_df: DataFrame of query records (e.g. S1).
            index_df: DataFrame of index records (e.g. S2+S3).

        Returns:
            DataFrame with columns: query_entity_id, candidate_entity_id,
            score, rank, blocker_id.
        """
        assert self._is_fit, "Call fit() before retrieve()."

        q_texts = self._get_text(query_df)
        i_texts = self._get_text(index_df)

        q_mat = self._vectorizer.transform(q_texts)
        i_mat = self._vectorizer.transform(i_texts)

        results = _chunked_cosine_top_k(q_mat, i_mat, self.top_k)

        rows = []
        q_ids = query_df["entity_id"].tolist()
        i_ids = index_df["entity_id"].tolist()

        for q_idx, (cand_indices, cand_scores) in enumerate(results):
            for rank, (c_idx, score) in enumerate(zip(cand_indices, cand_scores)):
                rows.append(
                    {
                        "source1_entity_id": q_ids[q_idx],
                        "candidate_entity_id": i_ids[c_idx],
                        f"{self.blocker_id.lower()}_score": float(score),
                        f"{self.blocker_id.lower()}_rank": rank + 1,
                    }
                )

        return pd.DataFrame(rows)

    def retrieve_bidirectional(
        self,
        s1_df: pd.DataFrame,
        s23_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Retrieve in both S1→S2/S3 and S2/S3→S1 directions and union.

        Forward (S1→S2/S3) scores and reverse (S2/S3→S1) scores are
        stored in separate columns for downstream feature computation.
        """
        # Forward: S1 queries S2/S3
        fwd = self.retrieve(s1_df, s23_df)
        fwd_renamed = fwd.rename(
            columns={
                f"{self.blocker_id.lower()}_score": f"{self.blocker_id.lower()}_score",
                f"{self.blocker_id.lower()}_rank": f"{self.blocker_id.lower()}_rank",
            }
        )

        # Reverse: S2/S3 queries S1 — swap source1 and candidate
        rev = self.retrieve(s23_df, s1_df)
        rev = rev.rename(
            columns={
                "source1_entity_id": "candidate_entity_id",
                "candidate_entity_id": "source1_entity_id",
                f"{self.blocker_id.lower()}_score": f"{self.blocker_id.lower()}_rev_score",
                f"{self.blocker_id.lower()}_rank": f"{self.blocker_id.lower()}_rev_rank",
            }
        )

        # Outer-join on (source1, candidate) pair
        merged = pd.merge(
            fwd_renamed,
            rev,
            on=["source1_entity_id", "candidate_entity_id"],
            how="outer",
        )
        # Fill missing scores with 0 and missing ranks with large sentinel
        score_cols = [c for c in merged.columns if "score" in c]
        rank_cols = [c for c in merged.columns if "rank" in c]
        merged[score_cols] = merged[score_cols].fillna(0.0)
        merged[rank_cols] = merged[rank_cols].fillna(999)

        return merged


def run_tfidf_blockers(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    all_df: pd.DataFrame,
    top_k: int = 30,
    blocker_ids: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run all TF-IDF blockers and return a dict of result DataFrames.

    Args:
        s1_df: Normalized S1 DataFrame.
        s23_df: Normalized combined S2+S3 DataFrame.
        all_df: Combined all records (for fitting vocabulary).
        top_k: Candidates per query per blocker.
        blocker_ids: Which blockers to run (default: B1, B2, B3).

    Returns:
        Dict mapping blocker_id → result DataFrame.
    """
    if blocker_ids is None:
        blocker_ids = ["B1", "B2", "B3"]

    results: Dict[str, pd.DataFrame] = {}
    for bid in blocker_ids:
        blocker = TFIDFBlocker(blocker_id=bid, top_k=top_k)
        blocker.fit(all_df)
        results[bid] = blocker.retrieve_bidirectional(s1_df, s23_df)

    return results
