"""Ranking metrics computed per query and averaged.

All functions take a DataFrame with one row per (query, product) pair, a column
holding the graded relevance gain, and a column holding the model score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dcg(gains: np.ndarray, k: int | None = None) -> float:
    """Discounted cumulative gain with the exponential (2^rel - 1) form."""
    gains = np.asarray(gains, dtype=float)
    if k is not None:
        gains = gains[:k]
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2))
    return float(np.sum((2.0**gains - 1.0) / discounts))


def ndcg(gains_in_ranked_order: np.ndarray, k: int | None = None) -> float:
    """NDCG for a single ranked list. Returns 0 when the list has no relevant items."""
    ideal = dcg(np.sort(np.asarray(gains_in_ranked_order, dtype=float))[::-1], k)
    if ideal == 0.0:
        return 0.0
    return dcg(gains_in_ranked_order, k) / ideal


def reciprocal_rank(gains_in_ranked_order: np.ndarray, threshold: float) -> float:
    """1 / rank of the first item whose gain is >= threshold, or 0 if none."""
    hits = np.flatnonzero(np.asarray(gains_in_ranked_order) >= threshold)
    return 0.0 if hits.size == 0 else 1.0 / (hits[0] + 1)


def _ranked_gains(group: pd.DataFrame, gain_col: str, score_col: str) -> np.ndarray:
    # Stable sort so ties keep input order, which makes results reproducible.
    order = np.argsort(-group[score_col].to_numpy(), kind="stable")
    return group[gain_col].to_numpy()[order]


def evaluate_ranking(
    df: pd.DataFrame,
    score_col: str,
    gain_col: str = "gain",
    query_col: str = "query_id",
    ks: tuple[int, ...] = (5, 10),
    mrr_threshold: float = 3.0,
) -> dict[str, float]:
    """Mean NDCG@k for each k, full-list NDCG, and MRR over all queries.

    MRR counts an item as relevant when its gain is >= ``mrr_threshold``
    (by default only Exact matches).
    """
    per_query: dict[str, list[float]] = {f"ndcg@{k}": [] for k in ks}
    per_query["ndcg"] = []
    per_query["mrr"] = []
    for _, group in df.groupby(query_col, sort=False):
        ranked = _ranked_gains(group, gain_col, score_col)
        for k in ks:
            per_query[f"ndcg@{k}"].append(ndcg(ranked, k))
        per_query["ndcg"].append(ndcg(ranked))
        per_query["mrr"].append(reciprocal_rank(ranked, mrr_threshold))
    return {name: float(np.mean(vals)) if vals else 0.0 for name, vals in per_query.items()}
