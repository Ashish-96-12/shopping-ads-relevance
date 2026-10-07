import pandas as pd
import pytest

from shoprel.rankers.bm25 import BM25
from shoprel.text import split_on_complement_marker, tokenize


def test_tokenize_keeps_model_numbers():
    assert tokenize("Galaxy S21, 15.6 inch!") == ["galaxy", "s21", "15.6", "inch"]


@pytest.mark.parametrize(
    "title, head, tail",
    [
        ("Slim Case for Galaxy S21", "slim case", "galaxy s21"),
        ("Charger compatible with Laptop", "charger", "laptop"),
        ("Galaxy S21 128GB", "galaxy s21 128gb", ""),
        ("Fortnite poster", "fortnite poster", ""),  # "for" inside a word is not a marker
    ],
)
def test_split_on_complement_marker(title, head, tail):
    assert split_on_complement_marker(title) == (head, tail)


def test_bm25_prefers_matching_and_rarer_terms():
    docs = pd.Series(["red running shoes", "blue running shoes", "red coffee mug", "black mug"])
    bm25 = BM25().fit(docs)
    scores = bm25.score_pairs(pd.Series(["red shoes"] * 4), docs)
    assert scores.argmax() == 0
    assert scores[3] == 0.0


def test_bm25_length_normalisation():
    docs = pd.Series(["phone", "phone case cover holder stand grip"])
    bm25 = BM25(b=0.75).fit(docs)
    short, long = bm25.score_pairs(pd.Series(["phone", "phone"]), docs)
    assert short > long
