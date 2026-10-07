"""Rankers share one interface: ``fit(train_df)`` then ``score(df) -> np.ndarray``.

``LambdaMARTRanker`` lives in ``shoprel.rankers.ltr``; it is not re-exported
here because it depends on ``shoprel.features``, which itself uses BM25.
"""

from shoprel.rankers.bm25 import BM25, BM25Ranker, TfidfRanker

__all__ = ["BM25", "BM25Ranker", "TfidfRanker"]
