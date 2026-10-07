"""Tiny text helpers shared by the rankers and feature builder."""

from __future__ import annotations

import re

import pandas as pd

_TOKEN_RE = re.compile(r"[^\W_]+(?:\.[0-9]+)?", re.UNICODE)

# Words that usually introduce what an accessory is *for*.
# "case for galaxy phone" is a complement to "galaxy phone", not a phone.
COMPLEMENT_MARKERS = ("for", "compatible with", "fits", "replacement for")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def full_text(df: pd.DataFrame) -> pd.Series:
    """All product text fields joined into one string per row."""
    return (
        df["product_title"]
        + " "
        + df["product_brand"]
        + " "
        + df["product_color"]
        + " "
        + df["product_bullet_point"]
        + " "
        + df["product_description"]
    )


def split_on_complement_marker(title: str) -> tuple[str, str]:
    """Split a title into (head, tail) at the first complement marker.

    >>> split_on_complement_marker("Slim Case for Galaxy S21 - Black")
    ('slim case', 'galaxy s21 - black')
    """
    lower = title.lower()
    best = -1
    best_len = 0
    for marker in COMPLEMENT_MARKERS:
        m = re.search(rf"\b{re.escape(marker)}\b", lower)
        if m and (best == -1 or m.start() < best):
            best, best_len = m.start(), len(m.group(0))
    if best == -1:
        return lower.strip(), ""
    return lower[:best].strip(), lower[best + best_len :].strip()
