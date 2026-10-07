import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from shoprel.auction import CLICK_PROB, compare_policies, run_auction, simulate_bids
from shoprel.data import train_test_split_by_query
from shoprel.llm_rater import LLMRater, agreement
from shoprel.relevance import RelevanceRater, rating_metrics
from shoprel.retrieval import BM25Retriever, evaluate_retrieval, recall_at_k
from shoprel.synthetic import make_synthetic_esci


@pytest.fixture(scope="module")
def split():
    df = make_synthetic_esci(n_queries=200, seed=3)
    return train_test_split_by_query(df, seed=3)


def test_relevance_model_predicts_human_ratings(split):
    train, test = split
    pred = RelevanceRater(n_estimators=100).fit(train).predict(test)
    probs = pred[["p_I", "p_C", "p_S", "p_E"]].to_numpy()
    assert np.allclose(probs.sum(1), 1.0)
    assert pred["expected_rating"].between(0, 1).all()
    m = rating_metrics(test["esci_label"], pred)
    assert m["accuracy"] > 0.5  # 4 classes, chance is ~0.25-0.3
    assert m["auc_relevant"] > 0.8


def test_retriever_finds_matching_products():
    catalog = pd.DataFrame(
        {
            "product_id": ["a", "b", "c"],
            "product_title": ["red running shoes", "coffee maker", "blue running shoes"],
            "product_brand": ["", "", ""],
            "product_color": ["", "", ""],
            "product_bullet_point": ["", "", ""],
            "product_description": ["", "", ""],
        }
    )
    hits = BM25Retriever().fit(catalog).retrieve(pd.Series(["red shoes", "espresso"]), k=2)
    assert hits[0][0] == "a"
    assert hits[1] == []  # no term overlap -> nothing retrieved


def test_recall_at_k():
    judged = pd.DataFrame(
        {"query_id": [1, 1, 2], "product_id": ["a", "b", "c"], "esci_label": ["E", "E", "E"]}
    )
    r = recall_at_k(judged, {1: ["a", "x", "b"], 2: ["y"]}, ks=(1, 3))
    assert r["recall@1"] == pytest.approx((0.5 + 0) / 2)
    assert r["recall@3"] == pytest.approx((1.0 + 0) / 2)


def test_retrieval_on_synthetic(split):
    train, test = split
    r = evaluate_retrieval(train, test, ks=(10, 100))
    assert 0 < r["recall@10"] <= r["recall@100"] <= 1


def _auction_df():
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1],
            "bid": [2.0, 1.0, 1.0],
            "rel": [0.5, 0.9, 0.2],
            "esci_label": ["S", "E", "I"],
        }
    )


def test_gsp_pricing_and_ordering():
    # ad ranks: 1.0, 0.9, 0.2 -> order A, B, C
    out = run_auction(_auction_df(), "rel", slots=1)
    # A wins slot 1 and pays next ad rank / own relevance = 0.9 / 0.5 = 1.8 (< bid 2.0)
    assert out["revenue_per_1k"] == pytest.approx(1000 * CLICK_PROB["S"] * 1.8)
    assert out["ads_per_query"] == 1


def test_eligibility_threshold_filters_low_relevance_ads():
    out = run_auction(_auction_df(), "rel", threshold=0.6, slots=4)
    assert out["ads_per_query"] == 1
    assert out["irrelevant_share"] == 0.0
    assert run_auction(_auction_df(), "rel", threshold=1.01)["coverage"] == 0.0


def test_bids_are_deterministic_and_positive():
    ids = pd.Series(["p1", "p2", "p3"])
    b1, b2 = simulate_bids(ids), simulate_bids(ids)
    assert np.array_equal(b1, b2)
    assert (b1 > 0).all()


def test_relevance_aware_auction_shows_fewer_irrelevant_ads(split):
    train, test = split
    pred = RelevanceRater(n_estimators=100).fit(train).predict(test)
    table = compare_policies(test, pred["p_relevant"].to_numpy(), np.ones(len(test)))
    bid_only = table.loc["bid only (no relevance)", "irrelevant_share"]
    model = table.loc["relevance model, threshold 0.5", "irrelevant_share"]
    assert model < bid_only
    assert table.loc["human labels (oracle)", "irrelevant_share"] == 0.0


class FakeClient:
    """Stands in for anthropic.Anthropic(); answers E when the title contains the query."""

    def __init__(self):
        self.calls = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        content = kwargs["messages"][0]["content"]
        query = content.split("\n")[0].removeprefix("Query: ")
        label = "E" if query in content.split("Product ad:")[1].lower() else "I"
        text = json.dumps({"reason": "test", "label": label})
        return SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)]
        )


def test_llm_rater_with_cache(tmp_path):
    df = pd.DataFrame(
        {
            "query": ["red mug", "red mug"],
            "product_id": ["a", "b"],
            "product_title": ["Big red mug", "Garden hose"],
            "product_brand": ["", ""],
            "product_color": ["", ""],
            "product_bullet_point": ["", ""],
            "esci_label": ["E", "I"],
        }
    )
    client = FakeClient()
    rater = LLMRater(client=client)
    cache = tmp_path / "labels.jsonl"
    labels = rater.rate(df, cache)["llm_label"]
    assert labels.tolist() == ["E", "I"]
    assert client.calls == 2
    LLMRater(client=client).rate(df, cache)  # second run is served from the cache
    assert client.calls == 2
    stats = agreement(df["esci_label"], labels)
    assert stats["accuracy"] == 1.0
