# Results on synthetic ESCI-style data

## Data

```json
{
  "train": {
    "queries": 480,
    "pairs": 8059,
    "products": 880,
    "avg_candidates_per_query": 16.8,
    "label_share": {
      "C": 0.238,
      "E": 0.216,
      "I": 0.238,
      "S": 0.308
    }
  },
  "test": {
    "queries": 120,
    "pairs": 2019,
    "products": 787,
    "avg_candidates_per_query": 16.8,
    "label_share": {
      "C": 0.236,
      "E": 0.226,
      "I": 0.241,
      "S": 0.297
    }
  }
}
```

## 1. Retrieval: BM25 over the whole ad inventory

| retriever | recall@10 | recall@50 | recall@100 | catalog_size |
|---|---|---|---|---|
| bm25 | 0.6390 | 0.9223 | 0.9550 | 880 |

## 2. Predicting human ratings (relevance model)

| model | accuracy | macro_f1 | auc_relevant | spearman |
|---|---|---|---|---|
| relevance_model | 0.8356 | 0.8300 | 0.9651 | 0.8603 |

## 3. Ranking candidates (test queries)

| model | ndcg@5 | ndcg@10 | ndcg | mrr |
|---|---|---|---|---|
| random | 0.4850 | 0.5618 | 0.7469 | 0.4397 |
| bm25_title | 0.9274 | 0.9386 | 0.9684 | 0.9694 |
| bm25_all | 0.9274 | 0.9428 | 0.9666 | 0.9437 |
| tfidf_char_title | 0.8824 | 0.9084 | 0.9442 | 0.8958 |
| lambdamart | 0.9560 | 0.9624 | 0.9794 | 0.9792 |
| relevance_model | 0.9472 | 0.9562 | 0.9767 | 0.9729 |

## 4. Ads auction: eligibility, ranking and pricing

Bids are simulated. Outcomes are scored with the human labels.

| policy | coverage | ads_per_query | exact_share | irrelevant_share | clicks_per_1k | revenue_per_1k |
|---|---|---|---|---|---|---|
| bid only (no relevance) | 1.0000 | 4 | 0.2208 | 0.2500 | 324.7792 | 438.5958 |
| BM25 score, threshold 0.5 | 1.0000 | 3.7167 | 0.5583 | 0.0179 | 547.5667 | 404.3970 |
| relevance model, threshold 0.3 | 1.0000 | 4 | 0.4125 | 0.0208 | 521.0833 | 503.9457 |
| relevance model, threshold 0.5 | 1.0000 | 3.9750 | 0.4109 | 0.0168 | 519.1167 | 492.5948 |
| relevance model, threshold 0.7 | 1.0000 | 3.9583 | 0.4063 | 0.0168 | 517.0292 | 484.1800 |
| human labels (oracle) | 1.0000 | 4 | 0.4375 | 0.0000 | 544.0000 | 534.1746 |

## LambdaMART top features (share of total gain)

| feature | gain_share |
|---|---|
| bm25_title_gap | 0.3469 |
| bm25_all_rank | 0.1468 |
| bm25_all_gap | 0.1296 |
| bm25_all | 0.0885 |
| all_coverage | 0.0750 |
| tfidf_char_title | 0.0500 |
| bm25_title_rank | 0.0360 |
| bm25_title | 0.0270 |
| tail_coverage | 0.0244 |
| head_coverage | 0.0223 |
