"""Hand-crafted query-product features for the learning-to-rank model.

Each feature is cheap to compute and has a clear reason to exist, which
keeps the model explainable (see ``feature_importance`` in the report).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shoprel.rankers.bm25 import BM25Ranker, TfidfRanker
from shoprel.text import split_on_complement_marker, tokenize


def _overlap_features(query: str, title: str, all_text: set[str], brand: str, color: str):
    q = tokenize(query)
    q_set = set(q)
    t_set = set(tokenize(title))
    head, tail = split_on_complement_marker(title)
    head_set, tail_set = set(tokenize(head)), set(tokenize(tail))
    brand_set = set(tokenize(brand))
    color_set = set(tokenize(color))
    n = max(len(q_set), 1)
    return (
        len(q),  # query_len
        len(t_set),  # title_len
        len(q_set & t_set) / n,  # title_coverage
        len(q_set & all_text) / n,  # all_coverage
        len(q_set & t_set) / max(len(q_set | t_set), 1),  # title_jaccard
        float(bool(brand_set) and brand_set <= q_set),  # brand_in_query
        float(bool(color_set & q_set)),  # color_in_query
        len(q_set & head_set) / n,  # head_coverage
        len(q_set & tail_set) / n,  # tail_coverage (after "for ...")
        float(bool(tail)),  # has_complement_marker
        float(query.lower().strip() in title.lower()),  # exact_phrase_in_title
    )


OVERLAP_NAMES = [
    "query_len",
    "title_len",
    "title_coverage",
    "all_coverage",
    "title_jaccard",
    "brand_in_query",
    "color_in_query",
    "head_coverage",
    "tail_coverage",
    "has_complement_marker",
    "exact_phrase_in_title",
]


class FeatureBuilder:
    """Fits the text scorers on training products and builds a feature matrix."""

    def __init__(self):
        self.scorers = [
            BM25Ranker(field="title"),
            BM25Ranker(field="all"),
            BM25Ranker(field="product_brand"),
            TfidfRanker(field="title"),
        ]

    def fit(self, train: pd.DataFrame) -> FeatureBuilder:
        for scorer in self.scorers:
            scorer.fit(train)
        return self

    @property
    def feature_names(self) -> list[str]:
        names = [s.name for s in self.scorers]
        per_query = [f"{s.name}_{suffix}" for s in self.scorers[:2] for suffix in ("rank", "gap")]
        return names + per_query + OVERLAP_NAMES

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df.index)
        for scorer in self.scorers:
            feats[scorer.name] = scorer.score(df)

        # Position within the candidate list matters more than the raw score,
        # because BM25 scores are not comparable across queries.
        groups = df["query_id"]
        for scorer in self.scorers[:2]:
            col = feats[scorer.name]
            feats[f"{scorer.name}_rank"] = col.groupby(groups).rank(ascending=False, method="min")
            feats[f"{scorer.name}_gap"] = col.groupby(groups).transform("max") - col

        all_tokens = [
            set(tokenize(" ".join(row)))
            for row in df[["product_title", "product_bullet_point", "product_description"]]
            .astype(str)
            .itertuples(index=False)
        ]
        overlap = [
            _overlap_features(q, t, a, b, c)
            for q, t, a, b, c in zip(
                df["query"],
                df["product_title"],
                all_tokens,
                df["product_brand"],
                df["product_color"],
                strict=True,
            )
        ]
        feats[OVERLAP_NAMES] = np.asarray(overlap, dtype=float)
        return feats[self.feature_names]
