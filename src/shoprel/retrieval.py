"""First stage: match each query against the whole ad inventory.

The reranking task only scores a short candidate list. In production those
candidates come from a retrieval step that searches every eligible product.
This module builds that step with BM25 over the full catalog and measures
recall: of the products humans rated Exact for a query, how many make it
into the top k retrieved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shoprel.rankers.bm25 import BM25
from shoprel.text import full_text


class BM25Retriever:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.bm25 = BM25(k1=k1, b=b)

    def fit(self, catalog: pd.DataFrame) -> BM25Retriever:
        self.catalog = catalog.drop_duplicates("product_id").reset_index(drop=True)
        docs = full_text(self.catalog)
        self.bm25.fit(docs)
        self.doc_weights_ = self.bm25._doc_weights(docs).T.tocsc()
        return self

    def retrieve(self, queries: pd.Series, k: int = 100, batch_size: int = 256) -> list[list[str]]:
        """Top-k product ids for each query."""
        ids = self.catalog["product_id"].to_numpy()
        results: list[list[str]] = []
        for start in range(0, len(queries), batch_size):
            q = self.bm25.vectorizer.transform(queries.iloc[start : start + batch_size])
            q = q.astype(float).tocsr()
            q.data[:] = 1.0
            scores = (q.multiply(self.bm25.idf_).tocsr() @ self.doc_weights_).toarray()
            kk = min(k, scores.shape[1])
            top = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
            for row, cand in zip(scores, top, strict=True):
                cand = cand[np.argsort(-row[cand], kind="stable")]
                results.append(ids[cand[row[cand] > 0]].tolist())
        return results


def recall_at_k(
    judged: pd.DataFrame, retrieved: dict, ks: tuple[int, ...] = (10, 50, 100)
) -> dict[str, float]:
    """Mean share of each query's Exact products found in its top-k list."""
    exact = judged[judged["esci_label"] == "E"].groupby("query_id")["product_id"].apply(set)
    out = {}
    for k in ks:
        vals = [len(want & set(retrieved[q][:k])) / len(want) for q, want in exact.items()]
        out[f"recall@{k}"] = float(np.mean(vals)) if vals else 0.0
    return out


def evaluate_retrieval(
    train: pd.DataFrame, test: pd.DataFrame, ks: tuple[int, ...] = (10, 50, 100)
) -> dict[str, float]:
    """Index every product in the data (the ad inventory), retrieve for test queries."""
    catalog = pd.concat([train, test])
    retriever = BM25Retriever().fit(catalog)
    queries = test.drop_duplicates("query_id")[["query_id", "query"]]
    hits = retriever.retrieve(queries["query"], k=max(ks))
    result = recall_at_k(test, dict(zip(queries["query_id"], hits, strict=True)), ks)
    result["catalog_size"] = len(retriever.catalog)
    return result
