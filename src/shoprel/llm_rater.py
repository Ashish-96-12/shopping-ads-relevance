"""Use an LLM as a relevance rater and measure how well it agrees with humans.

Human relevance ratings are slow and expensive. A common approach is to give
an LLM the same rating guidelines the human raters use, check its agreement
against a human-labelled sample, and, if agreement is good, use it to label
more data for the smaller production model.

Needs the ``llm`` extra and an Anthropic API key::

    pip install -e ".[llm]"
    export ANTHROPIC_API_KEY=...
    shoprel llm-rate --synthetic --n-pairs 200
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

DEFAULT_MODEL = "claude-opus-5-5"
LABELS = ["E", "S", "C", "I"]

GUIDELINES = """You are a search quality rater for a shopping ads system. \
Rate how relevant a product ad is to a shopper's search query using exactly one label:

E (Exact): the product is what the query asks for, matching every detail \
the query specifies (type, brand, size, color, model).
S (Substitute): the product is the right kind of item but misses at least one \
specified detail, yet it could still serve the shopper as a replacement.
C (Complement): the product is not what was asked for but is used together with \
it (e.g. a case for a phone, filters for a coffee maker).
I (Irrelevant): the product does not meet the query in a meaningful way, \
including when it fails a central aspect of the query.

Judge from the shopper's point of view. Shared words alone do not make a product \
relevant: "case for iPhone 13" is a Complement for the query "iphone 13", not Exact."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "description": "One short sentence."},
        "label": {"type": "string", "enum": LABELS},
    },
    "required": ["reason", "label"],
    "additionalProperties": False,
}


def _product_text(row: pd.Series, max_chars: int = 600) -> str:
    parts = [f"Title: {row['product_title']}"]
    for col, name in [("product_brand", "Brand"), ("product_color", "Color")]:
        if row.get(col):
            parts.append(f"{name}: {row[col]}")
    if row.get("product_bullet_point"):
        parts.append(f"Details: {str(row['product_bullet_point'])[:max_chars]}")
    return "\n".join(parts)


class LLMRater:
    """Rates (query, product) pairs with Claude using structured output."""

    def __init__(self, model: str = DEFAULT_MODEL, client=None, effort: str = "low"):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise ImportError('LLMRater needs the llm extra: pip install -e ".[llm]"') from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort

    def rate_one(self, query: str, product: pd.Series) -> dict:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=2048,
            system=GUIDELINES,
            messages=[
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nProduct ad:\n{_product_text(product)}",
                }
            ],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
            # If a request is declined by a safety classifier, retry on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return {"label": None, "reason": "refused"}
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)

    def rate(self, df: pd.DataFrame, cache_path: str | Path | None = None) -> pd.DataFrame:
        """Rate every row, reusing answers saved in ``cache_path`` (JSONL)."""
        cache: dict[str, dict] = {}
        if cache_path and Path(cache_path).exists():
            for line in Path(cache_path).read_text().splitlines():
                rec = json.loads(line)
                cache[rec["key"]] = rec
        out = []
        fh = open(cache_path, "a") if cache_path else None  # noqa: SIM115
        try:
            for _, row in df.iterrows():
                key = f"{self.model}|{row['query']}|{row['product_id']}"
                if key not in cache:
                    rec = {"key": key, **self.rate_one(row["query"], row)}
                    cache[key] = rec
                    if fh:
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                out.append(cache[key])
        finally:
            if fh:
                fh.close()
        return pd.DataFrame(
            {"llm_label": [r["label"] for r in out], "llm_reason": [r["reason"] for r in out]},
            index=df.index,
        )


def agreement(human: pd.Series, llm: pd.Series) -> dict:
    """Agreement between human and LLM labels on the rows the LLM answered."""
    mask = llm.notna()
    h, m = human[mask], llm[mask]
    h_rel, m_rel = h.isin(["E", "S"]), m.isin(["E", "S"])
    return {
        "rated": int(mask.sum()),
        "accuracy": float(accuracy_score(h, m)),
        "cohen_kappa": float(cohen_kappa_score(h, m, labels=LABELS)),
        "relevant_vs_not_accuracy": float((h_rel == m_rel).mean()),
        "confusion (rows=human, cols=llm, order E,S,C,I)": confusion_matrix(
            h, m, labels=LABELS
        ).tolist(),
    }
