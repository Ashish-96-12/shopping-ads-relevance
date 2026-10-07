"""Train every ranker on the same split and compare them on the same metrics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from shoprel.auction import compare_policies
from shoprel.metrics import evaluate_ranking
from shoprel.rankers import BM25Ranker, TfidfRanker
from shoprel.rankers.ltr import LambdaMARTRanker
from shoprel.relevance import RelevanceRater, rating_metrics
from shoprel.retrieval import evaluate_retrieval


def build_rankers(names: list[str]) -> list:
    """Map short names from the CLI to ranker instances."""
    factories = {
        "random": RandomRanker,
        "bm25_title": lambda: BM25Ranker(field="title"),
        "bm25_all": lambda: BM25Ranker(field="all"),
        "tfidf": lambda: TfidfRanker(field="title"),
        "lambdamart": LambdaMARTRanker,
        "relevance_model": RelevanceRater,
    }
    rankers = []
    for name in names:
        if name == "cross_encoder":
            from shoprel.rankers.cross_encoder import CrossEncoderRanker

            rankers.append(CrossEncoderRanker())
        elif name in factories:
            rankers.append(factories[name]())
        else:
            raise ValueError(
                f"Unknown ranker {name!r}. Options: {sorted(factories)} + cross_encoder"
            )
    return rankers


class RandomRanker:
    """Lower bound: shuffled candidate order."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> RandomRanker:
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return np.random.default_rng(self.seed).random(len(df))


@dataclass
class ExperimentResult:
    metrics: pd.DataFrame
    feature_importance: pd.Series | None = None
    scored_test: pd.DataFrame | None = None
    rating: dict | None = None
    retrieval: dict | None = None
    auction: pd.DataFrame | None = None
    info: dict = field(default_factory=dict)


def run_experiment(
    train: pd.DataFrame, test: pd.DataFrame, rankers: list, ads: bool = True
) -> ExperimentResult:
    """Fit and score every ranker; with ``ads`` also run retrieval and the auction."""
    rows = []
    scored = test[["query_id", "query", "product_id", "product_title", "esci_label", "gain"]].copy()
    result = ExperimentResult(metrics=pd.DataFrame(), scored_test=scored)
    for ranker in rankers:
        t0 = time.perf_counter()
        ranker.fit(train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        if isinstance(ranker, RelevanceRater):
            pred = ranker.predict(test)
            scored[ranker.name] = pred["expected_rating"].to_numpy()
            scored["p_relevant"] = pred["p_relevant"].to_numpy()
            result.rating = rating_metrics(test["esci_label"], pred)
        else:
            scored[ranker.name] = ranker.score(test)
        score_s = time.perf_counter() - t0
        metrics = evaluate_ranking(scored, score_col=ranker.name)
        rows.append({"model": ranker.name, **metrics, "fit_s": fit_s, "score_s": score_s})
        if isinstance(ranker, LambdaMARTRanker):
            result.feature_importance = ranker.feature_importance()
    result.metrics = pd.DataFrame(rows).set_index("model")

    if ads:
        result.retrieval = evaluate_retrieval(train, test)
        if "p_relevant" in scored:
            bm25 = (
                scored["bm25_all"] if "bm25_all" in scored else BM25Ranker().fit(train).score(test)
            )
            result.auction = compare_policies(
                test, scored["p_relevant"].to_numpy(), np.asarray(bm25)
            )
    return result


def write_report(result: ExperimentResult, out_dir: str | Path, title: str) -> Path:
    """Write ``results.md`` and ``results.json`` to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [c for c in result.metrics.columns if not c.endswith("_s")]

    lines = [f"# {title}", ""]
    if result.info:
        lines += ["## Data", "", "```json", json.dumps(result.info, indent=2), "```", ""]
    if result.retrieval:
        lines += [
            "## 1. Retrieval: BM25 over the whole ad inventory",
            "",
            _markdown_table(
                pd.DataFrame([result.retrieval], index=pd.Index(["bm25"], name="retriever"))
            ),
            "",
        ]
    if result.rating:
        lines += [
            "## 2. Predicting human ratings (relevance model)",
            "",
            _markdown_table(
                pd.DataFrame([result.rating], index=pd.Index(["relevance_model"], name="model"))
            ),
            "",
        ]
    lines += [
        "## 3. Ranking candidates (test queries)",
        "",
        _markdown_table(result.metrics[metric_cols]),
        "",
    ]
    if result.auction is not None:
        lines += [
            "## 4. Ads auction: eligibility, ranking and pricing",
            "",
            "Bids are simulated. Outcomes are scored with the human labels.",
            "",
            _markdown_table(result.auction),
            "",
        ]
    if result.feature_importance is not None:
        top = result.feature_importance.head(10)
        share = (top / result.feature_importance.sum()).rename("gain_share")
        share.index.name = "feature"
        lines += [
            "## LambdaMART top features (share of total gain)",
            "",
            _markdown_table(share.to_frame()),
            "",
        ]
    path = out_dir / "results.md"
    path.write_text("\n".join(lines))

    payload = {
        "info": result.info,
        "retrieval": result.retrieval,
        "rating": result.rating,
        "ranking": result.metrics.round(4).to_dict(orient="index"),
        "auction": None
        if result.auction is None
        else result.auction.round(4).to_dict(orient="index"),
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    return path


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        vals = [_fmt(v) for v in row]
        out.append("| " + " | ".join([str(idx), *vals]) + " |")
    return "\n".join(out)


def _fmt(v) -> str:
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() and abs(v) >= 2 else f"{v:.4f}"
    return str(v)
