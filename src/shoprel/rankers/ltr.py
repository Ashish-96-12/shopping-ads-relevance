"""LightGBM LambdaMART ranker over hand-crafted features."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from shoprel.features import FeatureBuilder


def _group_sizes(df: pd.DataFrame) -> np.ndarray:
    # LightGBM needs rows of the same query to be contiguous.
    return df.groupby("query_id", sort=False).size().to_numpy()


class LambdaMARTRanker:
    """Learning-to-rank with LightGBM's lambdarank objective (optimises NDCG)."""

    name = "lambdamart"

    def __init__(
        self, n_estimators: int = 300, learning_rate: float = 0.05, seed: int = 42, **lgb_params
    ):
        self.features = FeatureBuilder()
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
            **lgb_params,
        )

    def fit(self, train: pd.DataFrame) -> LambdaMARTRanker:
        train = train.sort_values("query_id", kind="stable")
        self.features.fit(train)
        X = self.features.transform(train)
        self.model.fit(X, train["gain"], group=_group_sizes(train))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.features.transform(df))

    def feature_importance(self) -> pd.Series:
        """Total gain contributed by each feature, highest first."""
        booster = self.model.booster_
        imp = booster.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.features.feature_names).sort_values(ascending=False)
