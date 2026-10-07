# Shopping Relevance Models

Ranking products for a search query, the way an e-commerce search engine does it.

Given a shopper's query like `stridex running shoes size 10` and a list of candidate products, the goal is to put the products the shopper actually wants at the top. This repo builds that ranking step on the **Amazon Shopping Queries Dataset (ESCI)** and compares a classic keyword baseline against a learned ranker.

| | |
|---|---|
| **Task** | Query-product reranking (ESCI Task 1) |
| **Data** | [Amazon ESCI](https://github.com/amazon-science/esci-data): ~130k queries, ~2.6M human relevance judgements |
| **Baselines** | Random, BM25 (title / all fields), character n-gram TF-IDF |
| **Learned model** | LightGBM LambdaMART on 19 hand-built features |
| **Optional** | Zero-shot cross-encoder (MiniLM) reranker |
| **Metrics** | NDCG@5, NDCG@10, NDCG, MRR |

## Why this is hard

ESCI labels every query-product pair as one of four classes, which become graded relevance gains:

| Label | Meaning | Example for query `iphone 13` | Gain |
|---|---|---|---|
| **E**xact | It's what was asked for | iPhone 13, 128GB | 3 |
| **S**ubstitute | Close, but misses a detail | iPhone 12 | 2 |
| **C**omplement | Goes with it, isn't it | Case for iPhone 13 | 1 |
| **I**rrelevant | Unrelated | Garden hose | 0 |

Complements are the classic trap for keyword search: "Case for iPhone 13" contains every word of the query, so BM25 often ranks it above the phone. The learned model gets features built for exactly this, such as how much of the query appears *after* "for" / "compatible with" in the title.

## How it works

```
query + candidates
      │
      ├── BM25 (title, all text, brand) ─┐
      ├── char n-gram TF-IDF ────────────┤
      ├── per-query rank and score gap ──┼──► 19 features ──► LightGBM LambdaMART ──► ranked list
      └── overlap / brand / color /      │                    (optimises NDCG)
          "for ..." complement signals ──┘
```

* **BM25** is implemented from scratch with sparse matrices (`src/shoprel/rankers/bm25.py`), with IDF fitted over the product catalog.
* **LambdaMART** uses LightGBM's `lambdarank` objective, which directly optimises NDCG over each query's candidate list.
* **Train/test split** is by query, so no query appears in both. On real ESCI the official `split` column is used.
* **Evaluation** (`src/shoprel/metrics.py`) uses exponential-gain NDCG and MRR where only Exact matches count as hits.

## Quick start

```bash
git clone https://github.com/Ashish-96-12/shopping-relevance-models.git
cd shopping-relevance-models
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Smoke test on generated data, no download needed (~10 seconds)
shoprel run --synthetic --n-queries 600 --out reports/synthetic

# 2. Real data: download ESCI (~1 GB of parquet) and run on a laptop-sized sample
shoprel download --out data/esci
shoprel run --data-dir data/esci --n-queries 5000 --out reports/esci_5k

# 3. Optional neural reranker
pip install -e ".[neural]"
shoprel run --data-dir data/esci --n-queries 2000 --models bm25_all,lambdamart,cross_encoder
```

Each run prints a results table and writes `results.md` and `results.json` to `--out`.

If `shoprel download` fails, grab the two `shopping_queries_dataset_*.parquet` files from the [ESCI repo](https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset) and put them in `data/esci/`.

## Results

### Synthetic sanity check

The repo ships a small generator (`src/shoprel/synthetic.py`) that builds ESCI-shaped data with the same traps (complements that repeat query words, substitutes that miss a brand or size, synonym queries). It's there for tests and CI, so these numbers only show the pipeline works end to end. Full output is in [`reports/synthetic/results.md`](reports/synthetic/results.md).

| Model | NDCG@5 | NDCG@10 | NDCG | MRR |
|---|---|---|---|---|
| Random | 0.485 | 0.562 | 0.747 | 0.440 |
| TF-IDF (char n-grams, title) | 0.882 | 0.908 | 0.944 | 0.896 |
| BM25 (title) | 0.927 | 0.939 | 0.968 | 0.969 |
| BM25 (all fields) | 0.927 | 0.943 | 0.967 | 0.944 |
| **LambdaMART** | **0.956** | **0.962** | **0.979** | **0.979** |

600 queries, 480 train / 120 test, about 17 candidates per query.

### Real ESCI

Run step 2 of the quick start to fill this in. The command writes the table to `reports/esci_5k/results.md`.

| Model | NDCG@5 | NDCG@10 | NDCG | MRR |
|---|---|---|---|---|
| BM25 (all fields) | | | | |
| LambdaMART | | | | |

## Project layout

```
src/shoprel/
  data.py            load + join ESCI parquet, sample queries, split by query
  synthetic.py       ESCI-shaped toy data for tests and demos
  text.py            tokenizer and "for ..." complement splitter
  features.py        19 query-product features
  metrics.py         NDCG@k, MRR
  pipeline.py        fit every ranker on the same split, write the report
  cli.py             `shoprel download` / `shoprel run`
  rankers/
    bm25.py          BM25 and TF-IDF rankers
    ltr.py           LightGBM LambdaMART
    cross_encoder.py optional sentence-transformers reranker
tests/               pytest suite (metrics, BM25, data loading, end to end)
```

## Development

```bash
pytest           # 23 tests, runs in a few seconds
ruff check .     # lint
ruff format .    # format
```

CI runs lint and tests on every push (`.github/workflows/ci.yml`).

## Ideas for next steps

* Fine-tune the cross-encoder on ESCI train pairs instead of using it zero-shot, then feed its score into LambdaMART as a feature.
* Add dense retrieval (bi-encoder + FAISS) as a first stage before reranking.
* Report metrics per ESCI label to see exactly where complements get ranked too high.
* Try the Spanish and Japanese locales (`--locale es`, `--locale jp`).

## References

* Reddy et al., *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search*, 2022. [arXiv:2206.06588](https://arxiv.org/abs/2206.06588)
* Burges, *From RankNet to LambdaRank to LambdaMART: An Overview*, 2010.
* Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009.

## License

MIT. The ESCI dataset is released by Amazon under the Apache 2.0 license.
