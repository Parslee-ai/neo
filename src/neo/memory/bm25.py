"""
Minimal in-memory BM25 ranker (no external dependency).

BM25(D, Q) = Σ IDF(q_i) · tf(q_i, D)·(k1+1) / (tf(q_i, D) + k1·(1 − b + b·|D|/avgdl))

IDF(q_i) = log((N − n(q_i) + 0.5) / (n(q_i) + 0.5) + 1)

Defaults k1=1.5, b=0.75 — Robertson-Sparck-Jones canonical values that
hold up across most retrieval benchmarks.

Used by FactStore as the sparse half of a dense+sparse hybrid retrieval
(paper 2603.19935 Memori §3.3 reports +6-10 pts factoid recall when the
sparse channel catches keyword overlaps the dense embedding smooths out).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

_TOKEN_RE = re.compile(r"\w+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase word-token split. Strips punctuation, keeps unicode word chars."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """In-memory BM25 index. Build once, query many.

    Rebuild from scratch on corpus change; the caller decides when —
    cheap at Neo's scope-cap of ~850 facts.
    """

    __slots__ = ("docs", "df", "idf", "doc_lens", "avgdl", "n")

    def __init__(self, documents: Sequence[Sequence[str]]):
        self.docs: list[list[str]] = [list(d) for d in documents]
        self.n = len(self.docs)
        self.doc_lens = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_lens) / self.n) if self.n else 0.0
        self.df: Counter[str] = Counter()
        for d in self.docs:
            for term in set(d):
                self.df[term] += 1
        # Precompute IDF; +1 inside log keeps it non-negative for very
        # frequent terms (Lucene-style "BM25Plus" smoothing).
        self.idf: dict[str, float] = {
            t: math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)
            for t, df in self.df.items()
        }

    def score(self, query_terms: Sequence[str], doc_idx: int) -> float:
        """Score one document against query_terms."""
        if doc_idx < 0 or doc_idx >= self.n:
            return 0.0
        doc = self.docs[doc_idx]
        if not doc:
            return 0.0
        tf = Counter(doc)
        dl = self.doc_lens[doc_idx]
        denom_norm = K1 * (1.0 - B + B * dl / self.avgdl) if self.avgdl > 0 else K1
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self.idf.get(term, 0.0)
            score += idf * (f * (K1 + 1.0)) / (f + denom_norm)
        return score

    def scores(self, query_terms: Sequence[str]) -> list[float]:
        """Score every document in the corpus against query_terms."""
        return [self.score(query_terms, i) for i in range(self.n)]
