"""
src/decode/expected_f.py — Exact-DP Expected-F0.5 decoder (§9.5).

The decoder selects the predicted set size k* ∈ {0, 1, …, n} that maximises
the expected F0.5 under a Poisson-binomial independence model.

Key components:
- Poisson-binomial DP: exact distribution over TP count given chosen probs.
- Missed-mass term: Poisson(λ_miss) added to the "outside" distribution O.
- Entity-level override variants:
    (i)  pure pair model (baseline)
    (ii) replace E[F_0] with (1-q); scale E[F_k≥1] by q / (1 - prod(1-p))
    (iii) blend: E[F_0] = w*(1-q) + (1-w)*prod(1-p)
- Tunable prob_power γ (pᵢ^γ) and empty_bonus β.
- Monte Carlo (512 samples) kept ONLY as a test oracle to verify the DP.

Appendix A math (PLAN.md):
    F(t, o; k) = 1.25·t / (k + 0.25·(t + o))   if t ≥ 1 else 0
    E[F_k]     = Σ_t Σ_o  PB_S(t) · PB_rest⊕Poisson(λ_miss)(o) · F(t, o; k)
    E[F_0]     = PB_all(0) · e^(−λ_miss)
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Poisson-Binomial DP
# ---------------------------------------------------------------------------

def poisson_binomial_dp(probs: np.ndarray) -> np.ndarray:
    """Compute the exact Poisson-binomial PMF over the given probabilities.

    Args:
        probs: Array of success probabilities p_1, …, p_n (all in [0, 1]).

    Returns:
        PMF array of length n+1 where result[k] = P(TP = k).
    """
    n = len(probs)
    dp = np.zeros(n + 1, dtype=np.float64)
    dp[0] = 1.0
    for p in probs:
        # Iterate downward to avoid double-counting
        for m in range(n, 0, -1):
            dp[m] = dp[m] * (1.0 - p) + dp[m - 1] * p
        dp[0] *= (1.0 - p)
    return dp


def poisson_pmf_trunc(lam: float, max_o: int = 10) -> np.ndarray:
    """Truncated Poisson PMF for the missed-mass term.

    Args:
        lam: Poisson rate λ_miss.
        max_o: Maximum number of missed positives to consider.

    Returns:
        Array of length max_o+1 where result[o] = P(O=o) (un-normalised tail
        is ignored; re-normalised to sum to 1).
    """
    pmf = np.array([math.exp(-lam) * lam**o / math.factorial(o) for o in range(max_o + 1)])
    pmf /= pmf.sum()
    return pmf


# ---------------------------------------------------------------------------
# Core expected-F computation
# ---------------------------------------------------------------------------

def _f_metric(t: int, o: int, k: int) -> float:
    """F0.5 given TP=t, outside positives=o, prediction size=k."""
    if t == 0:
        return 0.0
    return 1.25 * t / (k + 0.25 * (t + o))


def expected_f_k(
    prefix_pmf: np.ndarray,
    suffix_pmf: np.ndarray,
    poisson_pmf: np.ndarray,
    k: int,
) -> float:
    """Compute E[F_k] for a given k using precomputed prefix/suffix PMFs.

    Args:
        prefix_pmf: PB PMF over the k chosen candidates (TP distribution).
        suffix_pmf: PB PMF over the remaining n-k candidates.
        poisson_pmf: Truncated Poisson PMF for missed positives.
        k: Prediction set size (≥ 1).

    Returns:
        E[F_k] as a float.
    """
    # Convolve suffix PMF with Poisson PMF for total O distribution
    max_o = len(suffix_pmf) + len(poisson_pmf) - 2
    o_pmf = np.convolve(suffix_pmf, poisson_pmf)

    total = 0.0
    for t in range(1, len(prefix_pmf)):
        p_t = prefix_pmf[t]
        if p_t < 1e-15:
            continue
        for o in range(len(o_pmf)):
            p_o = o_pmf[o]
            if p_o < 1e-15:
                continue
            total += p_t * p_o * _f_metric(t, o, k)
    return total


def decode_single(
    probs: List[float],
    gamma: float = 1.0,
    beta: float = 1.0,
    lambda_miss: float = 0.0,
    entity_variant: str = "none",
    q: float = 0.5,
    w: float = 0.5,
    max_candidates: int = 50,
) -> int:
    """Select the optimal prediction set size k* for one S1 entity.

    Args:
        probs: Sorted (descending) calibrated candidate probabilities.
        gamma: Probability power for sharpening/smoothing (pᵢ → pᵢ^γ).
        beta: Empty-set bonus multiplier on E[F_0].
        lambda_miss: Poisson rate for missed positives (blocking misses).
        entity_variant: One of "none", "ii", "iii".
            "none": pure pair model.
            "ii":   replace E[F_0] with (1-q); scale E[F_k] by q/(1-∏(1-p)).
            "iii":  blend E[F_0] = w*(1-q) + (1-w)*∏(1-p).
        q: P(G ≥ 1) from the has-match classifier (used in variants ii/iii).
        w: Blend weight for variant iii.
        max_candidates: Maximum n to consider (truncate if longer).

    Returns:
        k* — the optimal predicted set size (0 means predict empty).
    """
    # Truncate and apply power
    p = np.array(sorted(probs, reverse=True)[:max_candidates], dtype=np.float64)
    p = np.clip(p ** gamma, 1e-9, 1.0 - 1e-9)
    n = len(p)

    if n == 0:
        return 0

    poisson_pmf = poisson_pmf_trunc(lambda_miss, max_o=10)

    # Build prefix PB incrementally as k grows
    # prefix_pmf[k][j] = P(TP = j | choosing top-k candidates)
    # suffix_pmf[k] = PB over p[k:]

    # Pre-compute suffix PMFs
    suffix_pmfs: List[np.ndarray] = [None] * (n + 1)  # type: ignore
    suffix_pmfs[n] = np.array([1.0])  # P(O=0) = 1 when no remaining candidates
    for i in range(n - 1, -1, -1):
        suffix_pmfs[i] = np.zeros(n - i + 1, dtype=np.float64)
        suffix_pmfs[i][0] = 1.0
        for j, pi in enumerate(p[i:]):
            for m in range(j, -1, -1):
                if m < len(suffix_pmfs[i]) - 1:
                    suffix_pmfs[i][m + 1] += suffix_pmfs[i][m] * pi
                suffix_pmfs[i][m] *= (1.0 - pi)

    # Rebuild suffix PMFs properly using DP
    suffix_pmfs_clean: List[np.ndarray] = []
    for i in range(n + 1):
        if i == n:
            suffix_pmfs_clean.append(np.array([1.0]))
        else:
            suffix_pmfs_clean.append(poisson_binomial_dp(p[i:]))

    # E[F_0]
    prod_complement = float(np.prod(1.0 - p))
    e_f0_raw = prod_complement * math.exp(-lambda_miss)

    if entity_variant == "ii":
        e_f0 = (1.0 - q) * beta
        # Scaling factor for k≥1: conditioning on "at least one match"
        denom = 1.0 - prod_complement
        scale_k = q / denom if denom > 1e-12 else 1.0
    elif entity_variant == "iii":
        e_f0 = (w * (1.0 - q) + (1.0 - w) * prod_complement) * beta
        scale_k = 1.0
    else:  # "none"
        e_f0 = e_f0_raw * beta
        scale_k = 1.0

    best_k = 0
    best_e = e_f0

    # Build prefix PMF incrementally
    prefix_pmf = np.array([1.0])  # k=0: trivially empty

    for k in range(1, n + 1):
        # Extend prefix by adding p[k-1]
        new_prefix = np.zeros(k + 1, dtype=np.float64)
        for m in range(k - 1, -1, -1):
            if m < len(prefix_pmf):
                new_prefix[m + 1] += prefix_pmf[m] * p[k - 1]
                new_prefix[m] += prefix_pmf[m] * (1.0 - p[k - 1])
        prefix_pmf = new_prefix

        e_fk = expected_f_k(
            prefix_pmf,
            suffix_pmfs_clean[k],
            poisson_pmf,
            k,
        ) * scale_k

        if e_fk > best_e:
            best_e = e_fk
            best_k = k

    return best_k


# ---------------------------------------------------------------------------
# Monte Carlo oracle (tests only)
# ---------------------------------------------------------------------------

def decode_single_mc(
    probs: List[float],
    gamma: float = 1.0,
    beta: float = 1.0,
    lambda_miss: float = 0.0,
    entity_variant: str = "none",
    q: float = 0.5,
    w: float = 0.5,
    n_samples: int = 512,
    seed: int = 42,
    max_candidates: int = 50,
) -> Tuple[int, np.ndarray]:
    """Monte Carlo version — used ONLY as a test oracle.

    Must agree with :func:`decode_single` within 0.01 on expected score.

    Returns:
        (k*, array of per-k expected F0.5 from MC).
    """
    rng = np.random.RandomState(seed)
    p = np.array(sorted(probs, reverse=True)[:max_candidates], dtype=np.float64)
    p = np.clip(p ** gamma, 1e-9, 1.0 - 1e-9)
    n = len(p)
    if n == 0:
        return 0, np.array([1.0])

    # Sample GT realisations
    gt_samples = rng.binomial(1, p, size=(n_samples, n))  # (S, n)

    # Poisson missed
    missed = rng.poisson(lambda_miss, size=n_samples)

    e_scores = np.empty(n + 1, dtype=float)
    # k=0
    no_match = (gt_samples.sum(axis=1) + missed) == 0
    if entity_variant == "ii":
        e_f0 = float(np.mean(no_match)) * (1 - q) / max(float(np.mean(no_match)), 1e-9)
        e_f0 = (1 - q) * beta
    elif entity_variant == "iii":
        e_f0 = (w * (1 - q) + (1 - w) * float(np.mean(no_match))) * beta
    else:
        e_f0 = float(np.mean(no_match)) * beta
    e_scores[0] = e_f0

    for k in range(1, n + 1):
        tp = gt_samples[:, :k].sum(axis=1)
        o = gt_samples[:, k:].sum(axis=1) + missed
        g = tp + o
        scores_k = np.where(
            g == 0,
            1.0,
            np.where(
                tp == 0,
                0.0,
                1.25 * tp / (k + 0.25 * g),
            ),
        )
        e_scores[k] = float(np.mean(scores_k))

    best_k = int(np.argmax(e_scores))
    return best_k, e_scores


# ---------------------------------------------------------------------------
# Batch decoding
# ---------------------------------------------------------------------------

def decode_batch(
    probs_list: List[List[float]],
    gamma: float = 1.0,
    beta: float = 1.0,
    lambda_miss: float = 0.0,
    entity_variant: str = "none",
    q_list: Optional[List[float]] = None,
    w: float = 0.5,
    max_candidates: int = 50,
) -> List[int]:
    """Decode a batch of S1 entities.

    Args:
        probs_list: List of probability lists (one per S1).
        q_list: Optional list of has-match probabilities (one per S1).
        Other args: same as :func:`decode_single`.

    Returns:
        List of k* values (one per S1).
    """
    if q_list is None:
        q_list = [0.5] * len(probs_list)
    return [
        decode_single(p, gamma=gamma, beta=beta, lambda_miss=lambda_miss,
                      entity_variant=entity_variant, q=q, w=w,
                      max_candidates=max_candidates)
        for p, q in zip(probs_list, q_list)
    ]


# ---------------------------------------------------------------------------
# Grid search for γ and β on OOF
# ---------------------------------------------------------------------------

def grid_search_decoder(
    probs_list: List[List[float]],
    gt_lists: List[List[str]],
    candidate_lists: List[List[str]],
    gamma_grid: Optional[List[float]] = None,
    beta_grid: Optional[List[float]] = None,
    lambda_miss_grid: Optional[List[float]] = None,
    entity_variant: str = "none",
    q_list: Optional[List[float]] = None,
) -> dict:
    """Grid search γ, β, λ_miss on OOF data.

    Args:
        probs_list: Calibrated probs per S1 (same order as gt_lists).
        gt_lists: Ground truth matched IDs per S1.
        candidate_lists: Candidate IDs per S1 (for building predictions).
        gamma_grid: Gamma values to try.
        beta_grid: Beta values to try.
        lambda_miss_grid: Lambda_miss values to try.
        entity_variant: Decoder variant ("none", "ii", "iii").
        q_list: Has-match probs per S1.

    Returns:
        Dict with best (gamma, beta, lambda_miss, oof_f05).
    """
    from .metric import macro_f05 as _mf05

    if gamma_grid is None:
        gamma_grid = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
    if beta_grid is None:
        beta_grid = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    if lambda_miss_grid is None:
        lambda_miss_grid = [0.0]
    if q_list is None:
        q_list = [0.5] * len(probs_list)

    best = {"gamma": 1.0, "beta": 1.0, "lambda_miss": 0.0, "oof_f05": -1.0}

    for gamma in gamma_grid:
        for beta in beta_grid:
            for lam in lambda_miss_grid:
                k_star_list = decode_batch(
                    probs_list, gamma=gamma, beta=beta,
                    lambda_miss=lam, entity_variant=entity_variant,
                    q_list=q_list,
                )
                preds = [
                    cands[:k] for k, cands in zip(k_star_list, candidate_lists)
                ]
                score = _mf05(gt_lists, preds)
                if score > best["oof_f05"]:
                    best = {"gamma": gamma, "beta": beta, "lambda_miss": lam, "oof_f05": score}

    return best
