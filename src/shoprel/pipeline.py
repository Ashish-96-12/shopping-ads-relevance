"""Train every ranker on the same split and compare them on the same metrics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from shoprel.metrics import evaluate_ranking
from shoprel.rankers import BM25Ranker, TfidfRanker
from shoprel.rankers.ltr import LambdaMARTRanker


def build_rankers(names: list[str]) -> list:
    """Map short names from the CLI to ranker instances."""
    factories = {
        "random": RandomRanker,
        "bm25_title": lambda: BM25Ranker(field="title"),
        "bm25_all": lambda: BM25Ranker(field="all"),
        "tfidf": lambda: TfidfRanker(field="title"),
        "lambdamart": LambdaMARTRanker,
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
    info: dict = field(default_factory=dict)


def run_experiment(train: pd.DataFrame, test: pd.DataFrame, rankers: list) -> ExperimentResult:
    rows = []
    scored = test[["query_id", "query", "product_id", "product_title", "esci_label", "gain"]].copy()
    importance = None
    for ranker in rankers:
        t0 = time.perf_counter()
        ranker.fit(train)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        scored[ranker.name] = ranker.score(test)
        score_s = time.perf_counter() - t0
        metrics = evaluate_ranking(scored, score_col=ranker.name)
        rows.append({"model": ranker.name, **metrics, "fit_s": fit_s, "score_s": score_s})
        if isinstance(ranker, LambdaMARTRanker):
            importance = ranker.feature_importance()
    table = pd.DataFrame(rows).set_index("model")
    return ExperimentResult(metrics=table, feature_importance=importance, scored_test=scored)


def write_report(result: ExperimentResult, out_dir: str | Path, title: str) -> Path:
    """Write ``results.md`` and ``results.json`` to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [c for c in result.metrics.columns if not c.endswith("_s")]

    lines = [f"# {title}", ""]
    if result.info:
        lines += ["## Data", "", "```json", json.dumps(result.info, indent=2), "```", ""]
    lines += ["## Results (test queries)", "", _markdown_table(result.metrics[metric_cols]), ""]
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

    payload = {"info": result.info, "metrics": result.metrics.round(4).to_dict(orient="index")}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    return path


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        out.append("| " + " | ".join([str(idx), *vals]) + " |")
    return "\n".join(out)
