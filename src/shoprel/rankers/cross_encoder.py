"""Optional neural reranker: a pretrained cross-encoder (zero-shot).

Needs the ``neural`` extra (``pip install -e ".[neural]"``) and downloads the
model from the Hugging Face Hub on first use. A cross-encoder reads the query
and product text together, so it can tell "case for iphone" apart from
"iphone" in a way bag-of-words features struggle to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderRanker:
    name = "cross_encoder"

    def __init__(
        self, model_name: str = DEFAULT_MODEL, batch_size: int = 64, max_length: int = 128
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                'CrossEncoderRanker needs the neural extra: pip install -e ".[neural]"'
            ) from exc
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.batch_size = batch_size

    def fit(self, train: pd.DataFrame) -> CrossEncoderRanker:
        # Zero-shot: the MS MARCO model is used as-is. Fine-tuning on ESCI
        # train pairs is a natural next step (see README).
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        docs = (df["product_title"] + ". " + df["product_brand"]).tolist()
        pairs = list(zip(df["query"].tolist(), docs, strict=True))
        return np.asarray(
            self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        )
