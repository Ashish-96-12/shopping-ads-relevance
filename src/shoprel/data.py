"""Loading and sampling the Amazon Shopping Queries (ESCI) dataset.

The dataset ships as two parquet files:

* ``shopping_queries_dataset_examples.parquet``: one row per (query, product)
  judgement with an ESCI label (Exact, Substitute, Complement, Irrelevant).
* ``shopping_queries_dataset_products.parquet``: product catalog text.

Source: https://github.com/amazon-science/esci-data (Reddy et al., 2022).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"

# Integer gains so they work both for NDCG and as LightGBM lambdarank labels.
ESCI_GAINS = {"E": 3, "S": 2, "C": 1, "I": 0}

PRODUCT_TEXT_COLS = [
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
]


def load_esci(
    data_dir: str | Path,
    locale: str = "us",
    small_version: bool = True,
) -> pd.DataFrame:
    """Read and join the ESCI examples and products for one locale.

    ``small_version`` selects Task 1 (query-product ranking), the reduced
    version of the data that is meant for reranking experiments.
    """
    data_dir = Path(data_dir)
    examples = pd.read_parquet(data_dir / EXAMPLES_FILE)
    products = pd.read_parquet(data_dir / PRODUCTS_FILE)

    examples = examples[examples["product_locale"] == locale]
    if small_version:
        examples = examples[examples["small_version"] == 1]
    products = products[products["product_locale"] == locale]

    df = examples.merge(products, on=["product_id", "product_locale"], how="left")
    return prepare(df)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns: fill missing text, add numeric ``gain``."""
    df = df.copy()
    for col in PRODUCT_TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    df["query"] = df["query"].fillna("").astype(str)
    df["gain"] = df["esci_label"].map(ESCI_GAINS).astype(int)
    return df.reset_index(drop=True)


def sample_queries(
    df: pd.DataFrame,
    n_queries: int | None,
    seed: int = 42,
    query_col: str = "query_id",
) -> pd.DataFrame:
    """Keep all judgements for a random subset of queries (keeps lists intact)."""
    if n_queries is None:
        return df
    ids = df[query_col].unique()
    if n_queries >= len(ids):
        return df
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(ids, size=n_queries, replace=False).tolist())
    return df[df[query_col].isin(keep)].reset_index(drop=True)


def train_test_split_by_query(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    query_col: str = "query_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by query so no query leaks between train and test.

    If the data carries ESCI's official ``split`` column, that is used instead.
    """
    if "split" in df.columns and set(df["split"].unique()) >= {"train", "test"}:
        train = df[df["split"] == "train"]
        test = df[df["split"] == "test"]
    else:
        ids = df[query_col].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
        n_test = max(1, int(round(len(ids) * test_size)))
        test_ids = set(ids[:n_test].tolist())
        is_test = df[query_col].isin(test_ids)
        train, test = df[~is_test], df[is_test]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def describe(df: pd.DataFrame) -> dict[str, object]:
    """Small summary used by the CLI and the README."""
    per_query = df.groupby("query_id").size()
    return {
        "queries": int(df["query_id"].nunique()),
        "pairs": int(len(df)),
        "products": int(df["product_id"].nunique()),
        "avg_candidates_per_query": round(float(per_query.mean()), 1),
        "label_share": {
            k: round(float(v), 3)
            for k, v in df["esci_label"].value_counts(normalize=True).sort_index().items()
        },
    }
