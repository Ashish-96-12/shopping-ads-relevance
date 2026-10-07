# Shopping Ads Relevance

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashish-96-12/shopping-ads-relevance/blob/main/notebooks/shopping_ads_relevance.ipynb)
[![CI](https://github.com/Ashish-96-12/shopping-ads-relevance/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashish-96-12/shopping-ads-relevance/actions/workflows/ci.yml)

Predicting how relevant a product ad is to a shopper's search query, and using that prediction across the whole ads pipeline: retrieval, ad eligibility, ranking and auction pricing.

This is a small, end-to-end version of what a shopping ads relevance team does. Human raters judge whether an ad matches a query. A model learns to predict those ratings. The predictions then decide which ads are allowed to show, in what order, and what each advertiser pays. An LLM rater is included too, to test how well a model like Claude can stand in for human raters.

Everything runs on the **Amazon Shopping Queries Dataset (ESCI)**, which has about 130k real queries and 2.6M human relevance judgements.

```
                 ┌──────────────────────────────────────────────┐
 query ──► 1. Retrieval ──► 2. Relevance model ──► 3. Ranking ──► 4. Auction ──► ads shown
           BM25 over the     predicts the human     LambdaMART     eligibility threshold,
           full catalog      rating (E/S/C/I)       ranker         ad rank = bid × relevance,
                                    ▲                              GSP pricing
                                    │
                         5. LLM rater (Claude) ── agreement with human raters
```

## The human ratings

Every query-product pair in ESCI carries one of four ratings:

| Rating | Meaning | Example for query `iphone 13` |
|---|---|---|
| **E**xact | It's what was asked for | iPhone 13, 128GB |
| **S**ubstitute | Close, but misses a detail | iPhone 12 |
| **C**omplement | Goes with it, isn't it | Case for iPhone 13 |
| **I**rrelevant | Unrelated | Garden hose |

Complements are the classic trap for keyword matching: "Case for iPhone 13" contains every word of the query, so a keyword system happily shows it. Several features in this project are built to catch that, like how much of the query appears *after* "for" or "compatible with" in the title.

## The five stages

| Stage | What it does | Code | Metrics |
|---|---|---|---|
| 1. Retrieval | BM25 (written from scratch with sparse matrices) searches every product in the inventory | `retrieval.py` | recall@10/50/100 of Exact products |
| 2. Relevance model | LightGBM classifier predicts the human rating from 19 query-ad features, and outputs P(relevant) and an expected rating | `relevance.py`, `features.py` | accuracy, macro F1, AUC, Spearman vs human ratings |
| 3. Ranking | LambdaMART (LightGBM `lambdarank`) orders candidates, compared with BM25, TF-IDF and random | `rankers/` | NDCG@5, NDCG@10, MRR |
| 4. Auction | Filters ads below a relevance threshold, ranks by bid × relevance, prices with generalized second price | `auction.py` | irrelevant ads shown, clicks, revenue |
| 5. LLM rater | Claude rates pairs using rater guidelines with structured output; agreement is measured against humans | `llm_rater.py` | accuracy, Cohen's kappa, confusion matrix |

### How relevance affects the auction

Each candidate is treated as an ad with a simulated bid (ESCI has no bids). For every query:

1. **Eligibility.** Ads with predicted relevance below the threshold are dropped.
2. **Ranking.** Ad rank = bid × relevance. A relevant ad with a lower bid can beat an irrelevant ad with a higher one.
3. **Pricing.** Each winner pays just enough to keep its slot: next ad rank ÷ its own relevance. More relevant ads pay less per click.

Outcomes are scored with the human labels. Clicks come from a click probability per rating and slot, so showing irrelevant ads costs clicks and revenue.

## Run it in Google Colab

The easiest way to run everything on the real dataset is the [Colab notebook](https://colab.research.google.com/github/Ashish-96-12/shopping-ads-relevance/blob/main/notebooks/shopping_ads_relevance.ipynb). Click the **Open in Colab** badge at the top, then **Runtime → Run all**. It installs the project, downloads ESCI, runs the pipeline and shows the results. The free CPU runtime is enough.

To try the Claude rater there, add your Anthropic API key as a Colab secret named `ANTHROPIC_API_KEY` (the key icon in the left sidebar) and tick `RUN_LLM_RATER` in the notebook.

## Quick start (local)

```bash
git clone https://github.com/Ashish-96-12/shopping-ads-relevance.git
cd shopping-ads-relevance
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Smoke test on generated data, no download needed (~15 seconds)
shoprel run --synthetic --n-queries 600 --out reports/synthetic

# Real data: download ESCI (~1 GB of parquet), run on a laptop-sized sample
shoprel download --out data/esci
shoprel run --data-dir data/esci --n-queries 5000 --out reports/esci_5k

# LLM rater: Claude rates 200 human-labelled pairs (uses your Anthropic API key)
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...
shoprel llm-rate --data-dir data/esci --n-queries 5000 --n-pairs 200

# Optional neural reranker
pip install -e ".[neural]"
shoprel run --data-dir data/esci --n-queries 2000 --models bm25_all,lambdamart,cross_encoder
```

Each run prints its results and writes `results.md` and `results.json` to `--out`. The LLM rater saves every answer to `llm_labels.jsonl`, so re-running doesn't pay twice. It uses `claude-opus-5-5` at low effort by default; pass `--model` to use another one.

If `shoprel download` fails, grab the two `shopping_queries_dataset_*.parquet` files from the [ESCI repo](https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset) and put them in `data/esci/`.

## Results

### Synthetic sanity check

The repo includes a small generator (`synthetic.py`) that builds ESCI-shaped data with the same traps: complements that repeat query words, substitutes that miss a brand or size, and synonym queries. It's there for tests and CI, so these numbers only show that the pipeline works end to end. The full output is in [`reports/synthetic/results.md`](reports/synthetic/results.md). There were 600 queries, split 480 for training and 120 for testing.

**Predicting human ratings**

| Accuracy (4 classes) | Macro F1 | AUC, relevant vs not | Spearman |
|---|---|---|---|
| 0.836 | 0.830 | 0.965 | 0.860 |

**Ranking**

| Model | NDCG@5 | NDCG@10 | MRR |
|---|---|---|---|
| Random | 0.485 | 0.562 | 0.440 |
| TF-IDF (char n-grams) | 0.882 | 0.908 | 0.896 |
| BM25 (all fields) | 0.927 | 0.943 | 0.944 |
| Relevance model | 0.947 | 0.956 | 0.973 |
| **LambdaMART** | **0.956** | **0.962** | **0.979** |

**Auction (4 ad slots per query)**

| Policy | Irrelevant ads shown | Clicks per 1k queries | Revenue per 1k queries |
|---|---|---|---|
| Bid only, no relevance | 25.0% | 325 | 439 |
| BM25 score, threshold 0.5 | 1.8% | 548 | 404 |
| **Relevance model, threshold 0.5** | **1.7%** | **519** | **493** |
| Human labels (upper bound) | 0.0% | 544 | 534 |

Using the relevance model cuts irrelevant ads from 25% to under 2%, raises clicks by about 60% and raises revenue by about 12% over bid-only. BM25 filtering also removes junk, but it's poorly calibrated and drops good ads too, so revenue falls below bid-only.

### Real ESCI

Run the real-data commands from the quick start to fill this in. The full tables are written to `reports/esci_5k/results.md` and `reports/llm_rater/llm_agreement.json`.

| | Result |
|---|---|
| Retrieval recall@100 | |
| Relevance model AUC | |
| LambdaMART NDCG@10 vs BM25 | |
| Irrelevant ads shown: bid only vs relevance model | |
| LLM rater agreement with humans (accuracy / kappa) | |

## Project layout

```
src/shoprel/
  data.py            load + join ESCI parquet, sample queries, split by query
  synthetic.py       ESCI-shaped toy data for tests and demos
  text.py            tokenizer and "for ..." complement splitter
  features.py        19 query-ad features (BM25, TF-IDF, overlap, brand, complement signals)
  retrieval.py       stage 1: BM25 retrieval over the whole catalog, recall@k
  relevance.py       stage 2: model that predicts the human rating
  rankers/           stage 3: BM25, TF-IDF, LambdaMART, optional cross-encoder
  auction.py         stage 4: eligibility, ad rank, GSP pricing, policy comparison
  llm_rater.py       stage 5: Claude as a relevance rater, agreement with humans
  metrics.py         NDCG@k, MRR
  pipeline.py        runs stages 1-4 on one split and writes the report
  cli.py             `shoprel download | run | llm-rate`
notebooks/
  shopping_ads_relevance.ipynb   one-click Colab run on real data
tests/               pytest suite (33 tests)
```

## Development

```bash
pytest           # 33 tests, about 10 seconds
ruff check .     # lint
ruff format .    # format
```

CI runs lint, the tests and a synthetic end-to-end run on every push (`.github/workflows/ci.yml`). The LLM rater tests use a fake client, so CI needs no API key.

## Ideas for next steps

* **Distillation:** label a large unlabelled sample with the LLM rater and train the relevance model on it, then check whether it closes the gap to the human-label model.
* Fine-tune the cross-encoder on ESCI pairs and use its score as a relevance-model feature.
* Add dense retrieval (bi-encoder + FAISS) next to BM25 and compare recall.
* Learn the eligibility threshold per query category instead of using one global value.
* Report results per rating to see exactly where complements slip through.

## References

* Reddy et al., *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search*, 2022. [arXiv:2206.06588](https://arxiv.org/abs/2206.06588)
* Edelman, Ostrovsky & Schwarz, *Internet Advertising and the Generalized Second-Price Auction*, 2007.
* Burges, *From RankNet to LambdaRank to LambdaMART: An Overview*, 2010.
* Thomas et al., *Large language models can accurately predict searcher preferences*, 2024. [arXiv:2309.10621](https://arxiv.org/abs/2309.10621)

## License

MIT. The ESCI dataset is released by Amazon under the Apache 2.0 license.
