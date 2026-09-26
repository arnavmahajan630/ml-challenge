"""
src/text/variants.py — Mine abbreviation/variant rules from GT positive pairs.

From the training ground-truth positives, align token pairs using greedy
best Jaro-Winkler matching and count substitutions (a→b).  Pairs with
count ≥ 5 and precision ≥ 0.9 are auto-added as extra canonicalization rules.

These mined rules are:
- Fitted WITHIN FOLDS when evaluating (fit on train folds only).
- Fitted on FULL train data for test inference.
- Token→token only (no country conditioning).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Token alignment
# ---------------------------------------------------------------------------

def _jaro_winkler(a: str, b: str) -> float:
    """Jaro-Winkler similarity using rapidfuzz (MIT)."""
    try:
        from rapidfuzz.distance import JaroWinkler  # type: ignore
        return JaroWinkler.normalized_similarity(a, b)
    except ImportError:
        # Fallback: simple character overlap
        if a == b:
            return 1.0
        return 0.0


def _greedy_align(
    tokens_a: List[str],
    tokens_b: List[str],
    threshold: float = 0.9,
) -> List[Tuple[str, str]]:
    """Greedily align tokens from A to B by best Jaro-Winkler score.

    Args:
        tokens_a: Token list from side A of the pair.
        tokens_b: Token list from side B.
        threshold: Minimum JW similarity to consider a match.

    Returns:
        List of (token_a, token_b) aligned pairs (excluding identical pairs).
    """
    used_b: Set[int] = set()
    pairs: List[Tuple[str, str]] = []

    for ta in tokens_a:
        best_score = threshold - 1e-9
        best_j = -1
        for j, tb in enumerate(tokens_b):
            if j in used_b:
                continue
            score = _jaro_winkler(ta, tb)
            if score > best_score:
                best_score = score
                best_j = j

        if best_j >= 0:
            used_b.add(best_j)
            ta_norm = ta.lower()
            tb_norm = tokens_b[best_j].lower()
            if ta_norm != tb_norm:
                pairs.append((ta_norm, tb_norm))

    return pairs


# ---------------------------------------------------------------------------
# Rule mining
# ---------------------------------------------------------------------------

class VariantMiner:
    """Mines token-level substitution rules from GT positive pairs.

    Usage:
        miner = VariantMiner()
        miner.fit(s1_records, s23_records, gt_df)
        rules = miner.rules_  # dict: token_a → canonical
    """

    def __init__(self, min_count: int = 5, min_precision: float = 0.9) -> None:
        self.min_count = min_count
        self.min_precision = min_precision
        self.rules_: Dict[str, str] = {}   # token → canonical
        self._counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._total_a: Dict[str, int] = defaultdict(int)

    def fit(
        self,
        s1_df: pd.DataFrame,
        s23_df: pd.DataFrame,
        gt_df: pd.DataFrame,
    ) -> "VariantMiner":
        """Mine variant rules from GT positive pairs.

        Args:
            s1_df: S1 normalized DataFrame (must have ``entity_id``, ``name_tokens``).
            s23_df: Combined S2+S3 normalized DataFrame.
            gt_df: Ground truth with ``source1_entity_id`` and ``matched_ids_list``.

        Returns:
            self (for chaining).
        """
        s1_tokens = dict(zip(s1_df["entity_id"], s1_df["name_tokens"]))
        s23_tokens = dict(zip(s23_df["entity_id"], s23_df["name_tokens"]))

        for _, row in gt_df.iterrows():
            s1_id = row["source1_entity_id"]
            matched: List[str] = row.get("matched_ids_list", [])
            if not matched:
                continue
            ta_str = s1_tokens.get(s1_id, "")
            tokens_a = ta_str.split() if ta_str else []
            if not tokens_a:
                continue

            for s23_id in matched:
                tb_str = s23_tokens.get(s23_id, "")
                tokens_b = tb_str.split() if tb_str else []
                if not tokens_b:
                    continue

                aligned = _greedy_align(tokens_a, tokens_b)
                for ta, tb in aligned:
                    self._counts[(ta, tb)] += 1
                    self._total_a[ta] += 1
                    self._counts[(tb, ta)] += 1
                    self._total_a[tb] += 1

        # Apply thresholds and extract rules
        self.rules_ = {}
        for (ta, tb), count in self._counts.items():
            total = self._total_a.get(ta, 0)
            if total == 0:
                continue
            precision = count / total
            if count >= self.min_count and precision >= self.min_precision:
                # Canonical form = lexicographically earlier (arbitrary but stable)
                canonical = min(ta, tb)
                self.rules_[ta] = canonical
                self.rules_[tb] = canonical

        return self

    def apply(self, token: str) -> str:
        """Return the canonical form of *token* (or the token itself)."""
        return self.rules_.get(token.lower(), token.lower())

    def apply_to_text(self, text: str) -> str:
        """Apply mined rules to a space-separated token string."""
        return " ".join(self.apply(t) for t in text.split())


def fit_variant_rules(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    min_count: int = 5,
    min_precision: float = 0.9,
) -> VariantMiner:
    """Convenience wrapper: fit and return a VariantMiner."""
    miner = VariantMiner(min_count=min_count, min_precision=min_precision)
    miner.fit(s1_df, s23_df, gt_df)
    return miner
