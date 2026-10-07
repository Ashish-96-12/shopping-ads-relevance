"""Okapi BM25 and TF-IDF scorers, vectorised with sparse matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from shoprel.text import full_text, tokenize


def _field_text(df: pd.DataFrame, field: str) -> pd.Series:
    if field == "title":
        return df["product_title"]
    if field == "all":
        return full_text(df)
    return df[field]


def _rowwise_dot(a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


class BM25:
    """Okapi BM25 over a product corpus.

    The corpus is every unique product seen in ``fit``, so IDF reflects the
    catalog, not individual candidate lists. Unseen products at score time
    are handled with the fitted vocabulary and average length.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)

    def fit(self, docs: pd.Series) -> BM25:
        tf = self.vectorizer.fit_transform(docs)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf_ = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        self.avgdl_ = float(tf.sum(axis=1).mean()) or 1.0
        return self

    def _doc_weights(self, docs: pd.Series) -> sp.csr_matrix:
        tf = self.vectorizer.transform(docs).astype(float).tocsr()
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl_)
        # Apply the saturation formula only to stored (non-zero) entries.
        row_norm = np.repeat(norm, np.diff(tf.indptr))
        tf.data = tf.data * (self.k1 + 1) / (tf.data + row_norm)
        return tf

    def score_pairs(self, queries: pd.Series, docs: pd.Series) -> np.ndarray:
        """BM25(query_i, doc_i) for each row i."""
        q = self.vectorizer.transform(queries).astype(float).tocsr()
        q.data[:] = 1.0  # each query term counts once
        q = q.multiply(self.idf_).tocsr()
        return _rowwise_dot(q, self._doc_weights(docs))


class BM25Ranker:
    """Baseline: rank candidates by BM25 on one product field."""

    def __init__(self, field: str = "all", k1: float = 1.2, b: float = 0.75):
        self.field = field
        self.bm25 = BM25(k1=k1, b=b)
        self.name = f"bm25_{field}"

    def fit(self, train: pd.DataFrame) -> BM25Ranker:
        products = train.drop_duplicates("product_id")
        self.bm25.fit(_field_text(products, self.field))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.bm25.score_pairs(df["query"], _field_text(df, self.field))


class TfidfRanker:
    """Cosine similarity of character n-gram TF-IDF vectors.

    Character n-grams give partial credit for plurals, typos and model
    numbers ("airpod" vs "airpods"), which word-level BM25 misses.
    """

    def __init__(self, field: str = "title", ngram_range: tuple[int, int] = (3, 5)):
        self.field = field
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, sublinear_tf=True, min_df=1
        )
        self.name = f"tfidf_char_{field}"

    def fit(self, train: pd.DataFrame) -> TfidfRanker:
        products = train.drop_duplicates("product_id")
        self.vectorizer.fit(pd.concat([_field_text(products, self.field), train["query"]]))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        q = self.vectorizer.transform(df["query"])
        d = self.vectorizer.transform(_field_text(df, self.field))
        return _rowwise_dot(q.tocsr(), d.tocsr())
