import json

import pytest

from shoprel.cli import main
from shoprel.data import train_test_split_by_query
from shoprel.features import FeatureBuilder
from shoprel.pipeline import build_rankers, run_experiment
from shoprel.synthetic import make_synthetic_esci


@pytest.fixture(scope="module")
def split():
    df = make_synthetic_esci(n_queries=200, seed=1)
    return train_test_split_by_query(df, seed=1)


def test_feature_matrix_shape_and_no_nans(split):
    train, test = split
    feats = FeatureBuilder().fit(train).transform(test)
    assert feats.shape == (len(test), len(FeatureBuilder().feature_names))
    assert not feats.isna().any().any()


def test_learned_and_lexical_rankers_beat_random(split):
    train, test = split
    result = run_experiment(train, test, build_rankers(["random", "bm25_all", "lambdamart"]))
    ndcg = result.metrics["ndcg@10"]
    assert ndcg["bm25_all"] > ndcg["random"] + 0.1
    assert ndcg["lambdamart"] > ndcg["random"] + 0.1
    assert result.feature_importance is not None


def test_unknown_ranker_name_raises():
    with pytest.raises(ValueError):
        build_rankers(["nope"])


def test_cli_run_writes_report(tmp_path):
    main(
        [
            "run",
            "--synthetic",
            "--n-queries",
            "80",
            "--models",
            "bm25_title,lambdamart",
            "--out",
            str(tmp_path),
        ]
    )
    payload = json.loads((tmp_path / "results.json").read_text())
    assert set(payload["ranking"]) == {"bm25_title", "lambdamart"}
    assert (tmp_path / "results.md").read_text().startswith("# Results")
