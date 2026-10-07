"""Predict the human relevance rating of a (query, product ad) pair.

Ads relevance teams train models to reproduce what trained human raters
would say about a query-ad pair, then use the predicted rating downstream
(retrieval, ad eligibility, ranking, pricing). Here the human ratings are
the four ESCI labels.

The model is a LightGBM multiclass classifier over the same features as the
ranker. Its output is turned into two numbers the ads system can use:

* ``p_relevant``: P(Exact or Substitute), i.e. "would the shopper be happy to
  see this ad". Used for the eligibility threshold and in the auction.
* ``expected_rating``: E[gain] / 3, a graded score in [0, 1] for ranking.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from shoprel.features import FeatureBuilder

# Class index == ESCI gain, so the classifier's columns line up with gains.
CLASSES = ["I", "C", "S", "E"]
RELEVANT_LABELS = {"E", "S"}


class RelevanceRater:
    name = "relevance_model"

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, seed: int = 42):
        self.features = FeatureBuilder()
        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
        )

    def fit(self, train: pd.DataFrame) -> RelevanceRater:
        self.features.fit(train)
        self.model.fit(self.features.transform(train), train["gain"])
        return self

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Probability of each ESCI label, columns ``p_I, p_C, p_S, p_E``."""
        proba = self.model.predict_proba(self.features.transform(df))
        out = np.zeros((len(df), len(CLASSES)))
        out[:, self.model.classes_.astype(int)] = proba
        return pd.DataFrame(out, columns=[f"p_{c}" for c in CLASSES], index=df.index)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(df)
        gains = np.arange(len(CLASSES))
        proba["p_relevant"] = proba["p_E"] + proba["p_S"]
        proba["expected_rating"] = proba[[f"p_{c}" for c in CLASSES]].to_numpy() @ gains / 3
        proba["pred_label"] = [
            CLASSES[i] for i in proba[[f"p_{c}" for c in CLASSES]].to_numpy().argmax(1)
        ]
        return proba

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Ranker interface, so the rating model can be compared with the rankers."""
        return self.predict(df)["expected_rating"].to_numpy()


def rating_metrics(labels: pd.Series, pred: pd.DataFrame) -> dict[str, float]:
    """How well predicted ratings agree with the human ratings."""
    is_relevant = labels.isin(RELEVANT_LABELS).astype(int)
    gains = labels.map({c: i for i, c in enumerate(CLASSES)})
    return {
        "accuracy": float(accuracy_score(labels, pred["pred_label"])),
        "macro_f1": float(f1_score(labels, pred["pred_label"], average="macro")),
        "auc_relevant": float(roc_auc_score(is_relevant, pred["p_relevant"]))
        if is_relevant.nunique() == 2
        else float("nan"),
        "spearman": float(spearmanr(gains, pred["expected_rating"]).statistic),
    }
