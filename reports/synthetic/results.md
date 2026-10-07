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

## Results (test queries)

| model | ndcg@5 | ndcg@10 | ndcg | mrr |
|---|---|---|---|---|
| random | 0.4850 | 0.5618 | 0.7469 | 0.4397 |
| bm25_title | 0.9274 | 0.9386 | 0.9684 | 0.9694 |
| bm25_all | 0.9274 | 0.9428 | 0.9666 | 0.9437 |
| tfidf_char_title | 0.8824 | 0.9084 | 0.9442 | 0.8958 |
| lambdamart | 0.9560 | 0.9624 | 0.9794 | 0.9792 |

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
