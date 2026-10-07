# Shopping Ads Relevance

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashish-96-12/shopping-ads-relevance/blob/main/notebooks/shopping_ads_relevance.ipynb)
[![CI](https://github.com/Ashish-96-12/shopping-ads-relevance/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashish-96-12/shopping-ads-relevance/actions/workflows/ci.yml)

When you search for something on a shopping site, the ads you see are supposed to match what you actually want. A lot of the time they don't. You search for an iPhone and get a phone case, or a different phone, or something totally random.

I built this project to understand how that problem gets solved in a real ads system. The idea is pretty simple. People (human raters) look at a search query and an ad and say how good a match it is. You train a model to predict what those raters would say. Then you use that prediction everywhere: to pick which ads are allowed to show, what order they go in, and even how much the advertiser pays.

I also wanted to see if an LLM (Claude) could do the rater's job, so there's a part for that too.

Everything runs on the **Amazon Shopping Queries Dataset (ESCI)**. It has around 130k real search queries and 2.6M query-product pairs that people have already labelled.

## How it fits together

```
 query ──► 1. Retrieval ──► 2. Relevance model ──► 3. Ranking ──► 4. Auction ──► ads shown
           BM25 over the     predicts the human     LambdaMART     relevance threshold,
           full catalog      rating (E/S/C/I)       ranker         ad rank = bid × relevance,
                                    ▲                              second-price pricing
                                    │
                         5. LLM rater (Claude), checked against the human ratings
```

## What the labels mean

Every query-product pair in ESCI has one of four labels:

| Label | What it means | Example for `iphone 13` |
|---|---|---|
| **E**xact | It's what you asked for | iPhone 13, 128GB |
| **S**ubstitute | Close, but not quite | iPhone 12 |
| **C**omplement | Goes with it, but isn't it | Case for iPhone 13 |
| **I**rrelevant | Nothing to do with it | Garden hose |

The complements were the most interesting part for me. "Case for iPhone 13" has every single word from the query in it, so plain keyword search loves it and ranks it near the top. I added a few features to catch this. One of them checks how much of the query shows up *after* words like "for" or "compatible with" in the title. That's usually a sign it's an accessory, not the thing itself.

## The five parts

**1. Retrieval.** Before you can rank ads you need to find candidates. I wrote BM25 from scratch with sparse matrices and use it to search the whole product catalog. I measure recall: of the products people marked Exact, how many make it into the top 10, 50 and 100. (`retrieval.py`)

**2. Relevance model.** This is the core of the project. It's a LightGBM classifier that looks at 19 features of a query-ad pair and predicts which of the four labels a human would give it. From that I get a "probability this ad is relevant" score, which the auction uses. (`relevance.py`, `features.py`)

**3. Ranking.** A LambdaMART ranker (LightGBM's `lambdarank`) that orders the candidates for each query. I compare it against BM25, TF-IDF and random ordering, using NDCG and MRR. (`rankers/`)

**4. Ads auction.** This is where relevance actually affects what gets shown and what it costs. For each query:
- Ads with low predicted relevance get filtered out.
- The rest are ranked by bid × relevance, so a relevant ad with a smaller bid can beat an irrelevant one with a bigger bid.
- Each winner pays just enough to keep its spot (a second-price auction), and more relevant ads end up paying less per click.

ESCI doesn't have real bids, so I simulate them. Clicks come from the human labels, so showing junk ads actually costs you clicks and money in the simulation. (`auction.py`)

**5. LLM rater.** I give Claude the same kind of rating guidelines a human rater would get, ask it to label query-ad pairs, and compare its answers with the human labels (accuracy, Cohen's kappa and a confusion matrix). If it agrees well enough, you could use it to label a lot more data cheaply. (`llm_rater.py`)

## Run it in Google Colab

This is the easiest way to run everything on the real data. Click the **Open in Colab** badge at the top, then **Runtime → Run all**. It installs the project, downloads ESCI, runs the whole pipeline and shows the results. The free CPU runtime is fine.

If you want to try the Claude rater, add your Anthropic API key as a Colab secret called `ANTHROPIC_API_KEY` (it's the key icon in the left sidebar), then tick `RUN_LLM_RATER` in the notebook.

## Run it on your own machine

```bash
git clone https://github.com/Ashish-96-12/shopping-ads-relevance.git
cd shopping-ads-relevance
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# quick check on generated data, no download needed (about 15 seconds)
shoprel run --synthetic --n-queries 600 --out reports/synthetic

# real data: download ESCI (about 1 GB) and run on a sample
shoprel download --out data/esci
shoprel run --data-dir data/esci --n-queries 5000 --out reports/esci_5k

# LLM rater: Claude labels 200 pairs (needs your Anthropic API key)
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...
shoprel llm-rate --data-dir data/esci --n-queries 5000 --n-pairs 200

# optional neural reranker
pip install -e ".[neural]"
shoprel run --data-dir data/esci --n-queries 2000 --models bm25_all,lambdamart,cross_encoder
```

Every run prints the results and saves `results.md` and `results.json` to the `--out` folder. The LLM rater saves each answer to `llm_labels.jsonl`, so if you run it again it won't pay for the same pairs twice. It uses `claude-opus-5-5` at low effort by default, and you can switch models with `--model`.

If `shoprel download` doesn't work for you, grab the two `shopping_queries_dataset_*.parquet` files from the [ESCI repo](https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset) and drop them into `data/esci/`.

## Results

### On generated test data

To be upfront: these numbers are **not** from the real dataset. I wrote a small generator (`synthetic.py`) that makes fake data in the same format as ESCI, with the same tricky cases built in. It's mainly there so the tests and CI can run without downloading 1 GB. So treat these as "the pipeline works end to end", not as real performance. The full output is in [`reports/synthetic/results.md`](reports/synthetic/results.md). It used 600 queries, with 480 for training and 120 for testing.

**Predicting the human rating**

| Accuracy (4 labels) | Macro F1 | AUC (relevant vs not) | Spearman |
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
| Bid only, ignore relevance | 25.0% | 325 | 439 |
| BM25 score, threshold 0.5 | 1.8% | 548 | 404 |
| **Relevance model, threshold 0.5** | **1.7%** | **519** | **493** |
| Human labels (best case) | 0.0% | 544 | 534 |

This was the part I found most interesting. Once the relevance model is in the loop, irrelevant ads drop from 25% to under 2%, clicks go up by about 60%, and revenue goes up by about 12% compared to just picking the highest bid. BM25 also filters out the junk, but its scores aren't real probabilities, so it throws away good ads too and revenue actually ends up lower than bid-only.

### On the real ESCI data

Still to fill in once I run it on the real data (the Colab notebook does this). The full tables end up in `reports/esci_5k/results.md` and `reports/llm_rater/llm_agreement.json`.

| | Result |
|---|---|
| Retrieval recall@100 | |
| Relevance model AUC | |
| LambdaMART NDCG@10 vs BM25 | |
| Irrelevant ads shown: bid only vs relevance model | |
| LLM rater agreement with humans (accuracy / kappa) | |

## What's where

```
src/shoprel/
  data.py            loads and joins the ESCI files, samples queries, splits by query
  synthetic.py       generates fake ESCI-style data for tests and demos
  text.py            tokenizer and the "for ..." accessory check
  features.py        the 19 query-ad features
  retrieval.py       part 1: BM25 search over the whole catalog
  relevance.py       part 2: model that predicts the human rating
  rankers/           part 3: BM25, TF-IDF, LambdaMART, optional cross-encoder
  auction.py         part 4: filtering, ad rank, pricing, comparing policies
  llm_rater.py       part 5: Claude as a rater
  metrics.py         NDCG and MRR
  pipeline.py        runs parts 1 to 4 and writes the report
  cli.py             the `shoprel download | run | llm-rate` commands
notebooks/
  shopping_ads_relevance.ipynb   the Colab notebook
tests/               33 tests
```

## Tests and linting

```bash
pytest           # 33 tests, takes about 10 seconds
ruff check .     # lint
ruff format .    # format
```

GitHub Actions runs the lint, the tests and a quick end-to-end run on every push. The LLM rater tests use a fake client, so CI doesn't need an API key.

## Things I'd like to try next

- Use the LLM rater to label a big batch of unlabelled pairs, train the relevance model on those labels, and see how close it gets to the one trained on human labels.
- Fine-tune the cross-encoder on ESCI and use its score as another feature.
- Add dense retrieval (a bi-encoder with FAISS) next to BM25 and compare recall.
- Learn a separate relevance threshold for each product category instead of one global number.
- Break the results down by label to see exactly where complements still slip through.

## References

- Reddy et al., *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search*, 2022. [arXiv:2206.06588](https://arxiv.org/abs/2206.06588)
- Edelman, Ostrovsky & Schwarz, *Internet Advertising and the Generalized Second-Price Auction*, 2007.
- Burges, *From RankNet to LambdaRank to LambdaMART: An Overview*, 2010.
- Thomas et al., *Large language models can accurately predict searcher preferences*, 2024. [arXiv:2309.10621](https://arxiv.org/abs/2309.10621)

## License

MIT. The ESCI dataset is released by Amazon under the Apache 2.0 license.
