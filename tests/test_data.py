import pandas as pd

from shoprel.data import ESCI_GAINS, load_esci, prepare, sample_queries, train_test_split_by_query
from shoprel.synthetic import make_synthetic_esci


def test_synthetic_data_has_esci_columns_and_all_labels():
    df = make_synthetic_esci(n_queries=50)
    for col in ["query_id", "query", "product_id", "product_title", "esci_label", "gain"]:
        assert col in df.columns
    assert set(df["esci_label"]) == set(ESCI_GAINS)
    # every query has at least one exact match
    assert (df.groupby("query_id")["gain"].max() == 3).all()


def test_split_has_no_query_leakage():
    df = make_synthetic_esci(n_queries=50)
    train, test = train_test_split_by_query(df, test_size=0.2)
    assert not set(train["query_id"]) & set(test["query_id"])
    assert len(train) + len(test) == len(df)


def test_split_uses_official_split_column_when_present():
    df = make_synthetic_esci(n_queries=10)
    df["split"] = ["test" if q < 3 else "train" for q in df["query_id"]]
    train, test = train_test_split_by_query(df)
    assert set(test["query_id"]) == {0, 1, 2}


def test_sample_queries_keeps_whole_candidate_lists():
    df = make_synthetic_esci(n_queries=50)
    sample = sample_queries(df, 10)
    assert sample["query_id"].nunique() == 10
    full_sizes = df.groupby("query_id").size()
    assert (sample.groupby("query_id").size() == full_sizes[sample["query_id"].unique()]).all()


def test_load_esci_reads_and_joins_parquet(tmp_path):
    examples = pd.DataFrame(
        {
            "example_id": [0, 1, 2],
            "query": ["red shoes", "red shoes", "zapatos"],
            "query_id": [1, 1, 2],
            "product_id": ["A", "B", "A"],
            "product_locale": ["us", "us", "es"],
            "esci_label": ["E", "I", "E"],
            "small_version": [1, 1, 1],
            "large_version": [1, 1, 1],
            "split": ["train", "train", "train"],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["A", "B", "A"],
            "product_title": ["Red Shoes", "Mug", "Zapatos"],
            "product_description": [None, None, None],
            "product_bullet_point": [None, None, None],
            "product_brand": ["Acme", None, None],
            "product_color": ["red", None, None],
            "product_locale": ["us", "us", "es"],
        }
    )
    examples.to_parquet(tmp_path / "shopping_queries_dataset_examples.parquet")
    products.to_parquet(tmp_path / "shopping_queries_dataset_products.parquet")

    df = load_esci(tmp_path, locale="us")
    assert len(df) == 2
    assert df.set_index("product_id").loc["A", "product_title"] == "Red Shoes"
    assert df["product_description"].eq("").all()
    assert df["gain"].tolist() == [3, 0]


def test_prepare_fills_missing_text_columns():
    df = prepare(pd.DataFrame({"query": ["x"], "esci_label": ["S"], "product_title": ["t"]}))
    assert df.loc[0, "product_brand"] == ""
    assert df.loc[0, "gain"] == 2


def test_load_esci_samples_queries_before_reading_products(tmp_path):
    df = make_synthetic_esci(n_queries=30)
    examples = df[
        [
            "example_id",
            "query",
            "query_id",
            "product_id",
            "product_locale",
            "esci_label",
            "small_version",
            "large_version",
        ]
    ]
    products = df.drop_duplicates("product_id")[
        [
            "product_id",
            "product_locale",
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
            "product_color",
        ]
    ]
    examples.to_parquet(tmp_path / "shopping_queries_dataset_examples.parquet")
    products.to_parquet(tmp_path / "shopping_queries_dataset_products.parquet")

    loaded = load_esci(tmp_path, n_queries=5)
    assert loaded["query_id"].nunique() == 5
    assert loaded["product_title"].ne("").all()  # every judged product got its text
