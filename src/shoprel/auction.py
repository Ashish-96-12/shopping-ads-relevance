"""A small sponsored-search auction that uses predicted relevance.

For each query, every candidate product is treated as an ad with an
advertiser bid. The predicted relevance is used three ways, mirroring how a
shopping ads system works:

1. **Eligibility**: ads whose predicted relevance is below a threshold are
   filtered out before the auction.
2. **Ranking**: ads are ordered by ad rank = bid x relevance^alpha.
3. **Pricing**: generalized second price (GSP). Each winner pays the
   smallest cost-per-click that would still keep its slot, which is
   ``next ad rank / own relevance^alpha``. A more relevant ad pays less.

Bids are simulated (ESCI has none), so revenue numbers are only useful for
comparing policies against each other.

Outcomes are scored with the *human* labels: expected clicks use a click
probability per ESCI label, and the share of irrelevant ads shown is what
users would actually see.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from scipy.stats import norm

# Probability a user clicks an ad in slot 1, by true label. Lower slots scale down.
CLICK_PROB = {"E": 0.30, "S": 0.15, "C": 0.06, "I": 0.01}
POSITION_DECAY = np.array([1.0, 0.7, 0.5, 0.35, 0.25, 0.2, 0.15, 0.1])


def simulate_bids(product_ids: pd.Series, mean_cpc: float = 0.8, sigma: float = 0.6) -> np.ndarray:
    """Deterministic log-normal bids per product, independent of relevance."""
    seeds = product_ids.astype(str).map(lambda s: zlib.crc32(s.encode()))
    u = (seeds.to_numpy(dtype=np.float64) + 0.5) / 2**32
    return mean_cpc * np.exp(sigma * norm.ppf(u) - sigma**2 / 2)


def run_auction(
    df: pd.DataFrame,
    relevance_col: str,
    threshold: float = 0.0,
    slots: int = 4,
    alpha: float = 1.0,
    reserve: float = 0.05,
) -> dict[str, float]:
    """Run one auction per query and return averaged outcomes.

    ``df`` needs ``query_id``, ``bid``, ``esci_label`` and ``relevance_col``
    (a predicted relevance in [0, 1]).
    """
    shown_labels: list[str] = []
    revenue = clicks = 0.0
    queries_with_ads = 0
    n_queries = df["query_id"].nunique()

    for _, g in df.groupby("query_id", sort=False):
        rel = np.clip(g[relevance_col].to_numpy(dtype=float), 1e-6, 1.0)
        bid = g["bid"].to_numpy(dtype=float)
        rank_score = bid * rel**alpha
        eligible = (rel >= threshold) & (rank_score >= reserve)
        if not eligible.any():
            continue
        idx = np.flatnonzero(eligible)
        idx = idx[np.argsort(-rank_score[idx], kind="stable")]
        winners = idx[:slots]
        queries_with_ads += 1
        labels = g["esci_label"].to_numpy()
        for pos, w in enumerate(winners):
            nxt = rank_score[idx[pos + 1]] if pos + 1 < len(idx) else reserve
            price = min(bid[w], max(nxt, reserve) / rel[w] ** alpha)
            p_click = CLICK_PROB[labels[w]] * POSITION_DECAY[min(pos, len(POSITION_DECAY) - 1)]
            clicks += p_click
            revenue += p_click * price
            shown_labels.append(labels[w])

    shown = pd.Series(shown_labels, dtype=object)
    n_shown = max(len(shown), 1)
    return {
        "coverage": queries_with_ads / n_queries,
        "ads_per_query": len(shown) / n_queries,
        "exact_share": float((shown == "E").sum() / n_shown),
        "irrelevant_share": float((shown == "I").sum() / n_shown),
        "clicks_per_1k": 1000 * clicks / n_queries,
        "revenue_per_1k": 1000 * revenue / n_queries,
    }


def compare_policies(
    test: pd.DataFrame,
    p_relevant: np.ndarray,
    bm25_score: np.ndarray,
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
    slots: int = 4,
) -> pd.DataFrame:
    """Auction outcomes for several ways of using (or ignoring) relevance."""
    df = test[["query_id", "esci_label", "product_id"]].copy()
    df["bid"] = simulate_bids(df["product_id"])
    df["no_relevance"] = 1.0
    # BM25 is not a probability; min-max scale per query so it can be thresholded.
    b = pd.Series(bm25_score, index=df.index)
    grp = b.groupby(df["query_id"])
    df["bm25_scaled"] = (b - grp.transform("min")) / (grp.transform("max") - grp.transform("min"))
    df["bm25_scaled"] = df["bm25_scaled"].fillna(1.0)
    df["model"] = p_relevant
    df["human"] = df["esci_label"].isin(["E", "S"]).astype(float)

    rows = {"bid only (no relevance)": run_auction(df, "no_relevance", slots=slots)}
    rows["BM25 score, threshold 0.5"] = run_auction(df, "bm25_scaled", 0.5, slots=slots)
    for t in thresholds:
        rows[f"relevance model, threshold {t}"] = run_auction(df, "model", t, slots=slots)
    rows["human labels (oracle)"] = run_auction(df, "human", 0.5, slots=slots)
    table = pd.DataFrame(rows).T
    table.index.name = "policy"
    return table
