"""Command line entry point: ``shoprel download | run | llm-rate``."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from shoprel import data
from shoprel.pipeline import build_rankers, run_experiment, write_report
from shoprel.synthetic import make_synthetic_esci

ESCI_BASE_URL = "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/"
DEFAULT_MODELS = "random,bm25_title,bm25_all,tfidf,lambdamart,relevance_model"


def cmd_download(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in (data.EXAMPLES_FILE, data.PRODUCTS_FILE):
        target = out / name
        if target.exists():
            print(f"skip {target} (exists)")
            continue
        print(f"downloading {name} ...")
        with urllib.request.urlopen(ESCI_BASE_URL + name) as resp, open(target, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        print(f"  -> {target} ({target.stat().st_size / 1e6:.0f} MB)")


def cmd_run(args: argparse.Namespace) -> None:
    df = _load(args)
    if args.synthetic:
        title = "Results on synthetic ESCI-style data"
    else:
        title = f"Results on Amazon ESCI ({args.locale}, Task 1)"

    train, test = data.train_test_split_by_query(df, seed=args.seed)
    info = {"train": data.describe(train), "test": data.describe(test)}
    print(f"train: {info['train']['queries']} queries / {info['train']['pairs']} pairs")
    print(f"test:  {info['test']['queries']} queries / {info['test']['pairs']} pairs")

    rankers = build_rankers([m.strip() for m in args.models.split(",") if m.strip()])
    result = run_experiment(train, test, rankers, ads=not args.no_ads)
    result.info = info
    print()
    if result.retrieval:
        print("retrieval:", {k: round(v, 4) for k, v in result.retrieval.items()})
    if result.rating:
        print("rating:   ", {k: round(v, 4) for k, v in result.rating.items()})
    print()
    print(result.metrics.round(4).to_string())
    if result.auction is not None:
        print()
        print(result.auction.round(3).to_string())
    path = write_report(result, args.out, title)
    print(f"\nreport written to {path}")


def _load(args: argparse.Namespace):
    if args.synthetic:
        return make_synthetic_esci(n_queries=args.n_queries or 400, seed=args.seed)
    return data.load_esci(
        args.data_dir, locale=args.locale, n_queries=args.n_queries, seed=args.seed
    )


def cmd_llm_rate(args: argparse.Namespace) -> None:
    from shoprel.llm_rater import LLMRater, agreement

    df = _load(args)
    # Stratify by label so the sample covers every class.
    per_label = max(1, args.n_pairs // 4)
    sample = pd.concat(
        g.sample(min(len(g), per_label), random_state=args.seed)
        for _, g in df.groupby("esci_label")
    ).reset_index(drop=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"rating {len(sample)} pairs with {args.model} ...")
    rated = LLMRater(model=args.model, effort=args.effort).rate(sample, out / "llm_labels.jsonl")
    stats = agreement(sample["esci_label"], rated["llm_label"])
    stats["model"] = args.model
    (out / "llm_agreement.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="shoprel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="download the ESCI parquet files (~1 GB)")
    p_dl.add_argument("--out", default="data/esci")
    p_dl.set_defaults(func=cmd_download)

    p_run = sub.add_parser("run", help="train and evaluate rankers")
    p_run.add_argument("--data-dir", default="data/esci")
    p_run.add_argument("--synthetic", action="store_true", help="use generated toy data")
    p_run.add_argument("--locale", default="us", choices=["us", "es", "jp"])
    p_run.add_argument(
        "--n-queries",
        type=int,
        default=None,
        help="sample this many queries (keeps it laptop-sized)",
    )
    p_run.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=f"comma separated; default {DEFAULT_MODELS}; also: cross_encoder",
    )
    p_run.add_argument("--no-ads", action="store_true", help="skip retrieval and auction")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--out", default="reports")
    p_run.set_defaults(func=cmd_run)

    p_llm = sub.add_parser("llm-rate", help="rate pairs with Claude, compare to human labels")
    p_llm.add_argument("--data-dir", default="data/esci")
    p_llm.add_argument("--synthetic", action="store_true")
    p_llm.add_argument("--locale", default="us", choices=["us", "es", "jp"])
    p_llm.add_argument("--n-queries", type=int, default=None)
    p_llm.add_argument("--n-pairs", type=int, default=200, help="pairs to rate (costs API calls)")
    p_llm.add_argument("--model", default="claude-opus-5-5")
    p_llm.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    p_llm.add_argument("--seed", type=int, default=42)
    p_llm.add_argument("--out", default="reports/llm_rater")
    p_llm.set_defaults(func=cmd_llm_rate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
