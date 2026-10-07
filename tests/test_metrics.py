import math

import pandas as pd
import pytest

from shoprel.metrics import dcg, evaluate_ranking, ndcg, reciprocal_rank


def test_dcg_matches_hand_computation():
    # (2^3-1)/log2(2) + (2^0-1)/log2(3) + (2^2-1)/log2(4)
    expected = 7 / 1 + 0 + 3 / 2
    assert dcg([3, 0, 2]) == pytest.approx(expected)


def test_ndcg_is_one_for_ideal_order_and_lower_otherwise():
    assert ndcg([3, 2, 1, 0]) == pytest.approx(1.0)
    assert ndcg([0, 1, 2, 3]) < 1.0


def test_ndcg_at_k_only_looks_at_top_k():
    assert ndcg([3, 0, 0, 2], k=1) == pytest.approx(1.0)


def test_ndcg_without_relevant_items_is_zero():
    assert ndcg([0, 0, 0]) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([0, 2, 3], threshold=3) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 0], threshold=3) == 0.0


def test_evaluate_ranking_averages_over_queries():
    df = pd.DataFrame(
        {
            "query_id": [1, 1, 2, 2],
            "gain": [3, 0, 0, 3],
            "score": [0.9, 0.1, 0.9, 0.1],  # query 1 perfect, query 2 inverted
        }
    )
    m = evaluate_ranking(df, score_col="score", ks=(1,))
    assert m["ndcg@1"] == pytest.approx(0.5)
    assert m["mrr"] == pytest.approx((1 + 0.5) / 2)
    worst = (2**3 - 1) / math.log2(3) / (2**3 - 1)
    assert m["ndcg"] == pytest.approx((1 + worst) / 2)
