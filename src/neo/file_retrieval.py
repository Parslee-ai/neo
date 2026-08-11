"""Rank candidate files by BM25 over their CONTENT.

The scorer this replaces never read the file. `score_candidate` took
`(rel_path, size, prompt_tokens, git_recent, entry_points)` and content was
first opened *after* selection, only to chunk what had already been chosen.
With no content signal, relevance had to be inferred from a filename and a
byte count, so the scorer became a stack of hand-tuned additive bonuses whose
dominant term was anti-correlated with relevance:

    score -= 0.01 * size_kb          # once over 10 KB, uncapped

against a realistic positive signal of +0.6 to +2.1. A file with one keyword
hit was unrankable above 60 KB. Measured on this repo, for "fix the fact store
supersession threshold", `src/neo/memory/store.py` scored +0.60 keyword,
-0.15 depth, -1.62 size = **0.000**, and ranked 200th of 284 — the most
relevant file in the repository, unrankable, because it is large. Ground-truth
files measured 31-177 KB against a corpus median of 10 KB, because central
files are large *because* they are central.

That had already been noticed once and patched with a seven-name stem
whitelist (`core, engine, main, index, app, server, lib`), which rescued
`engine.py` at -0.13 and left `store.py` at -1.62 -- a 12x disparity between
two files of near-identical size, decided by whether someone had thought of
the name.

**The literature says the sign was wrong, not the magnitude.** This task is
IR-based bug localization. BugLocator's revised VSM (Zhou et al., ICSE 2012)
adds a logistic length function that ranks LARGER files HIGHER, because larger
files are empirically more likely to contain the defect. BM25 addresses the
same concern properly: its `b` parameter normalizes by document length against
the corpus average, which is bounded, corpus-derived, and cannot run away to
-1.62.

Two more reasons BM25 is the right instrument rather than another bonus:

- **IDF is computed from the corpus**, so a term's weight reflects how
  discriminating it actually is in THIS repository. Two earlier attempts hand-
  approximated that with a document-frequency filter and a token-length floor;
  both were measured and both lost.
- **Code-aware tokenization is what makes it beat dense retrieval.** Splitting
  identifiers on camelCase and separators while keeping both the whole
  identifier and its parts lets a query saying "user by id" reach
  `getUserById`.

Measured END-TO-END through the real CLI, `PYTHONPATH` pinned per tree, over
cases mined from git history (commit subject as query, changed non-test files
as ground truth). One table, one harness generation -- an earlier draft of this
docstring carried FIRST-PASS numbers from a harness that called
`score_candidate` directly and so never saw the re-rank, the adaptive limit or
the byte budget, and those numbers disagreed with CLAUDE.md's by enough that
`car` appeared as both 0.969 and 0.507:

    repo   cases   R@10 before -> after     MRR before -> after
    neo      60        0.078 -> 0.603        0.082 -> 0.655
    car      50        0.210 -> 0.487        0.149 -> 0.233
    quip     50        0.213 -> 0.850        0.188 -> 0.735

Neo already shipped this BM25 (`neo.memory.bm25`, Lucene-style with IDF
smoothing) and already used hybrid dense+sparse fusion for FACT retrieval.
File selection used none of it.

**Fusion is deliberately absent, provisionally.** RRF over BM25 + the existing
dense channel measured WORSE than BM25 alone (0.596 best-weighted vs 0.693
R@10): dense returns ~25 files against BM25's ~180 and is roughly half as
accurate, so equal-weight fusion lets a short low-quality list promote
mediocre hits, and no weighting recovered past BM25-only.

Two caveats on that number, both found in review and both worth respecting
before it is treated as settled. It was measured at k=10, and k=10 is where
every ranking configuration on this repo is flat — the same cutoff mistake
that produced a wrong conclusion about the re-rank being redundant. And the
chunk-allocation defect it blames (`store.py` 7% covered against
`text_budget.py` at 100%) was FIXED in the same branch, so the stated
mechanism no longer holds. Re-measure at k=3/k=5 before relying on this.
"""

import re
from typing import Optional

from neo.memory.bm25 import BM25

#: CHARACTERS read per file when building the index — `TextIOWrapper.read(n)`
#: counts code points, not bytes, so on non-ASCII source this reads more bytes
#: than the number suggests. Named for what it measures: `CLAUDE.md` already
#: records the same defect in `_render_context_files`, where a banner summed
#: `len(str)` and called the result bytes.
#:
#: Bounds cost without truncating anything a human wrote — the largest
#: hand-written file in this repo is 177 KB. Note `iter_paths` already skips
#: files over 512 KB, so this only bites the 200-512 KB band.
MAX_INDEXED_CHARS = 200_000

#: The path is indexed alongside the content, repeated, so "a file named for
#: the thing" stays real evidence without being the ONLY evidence — which is
#: precisely what the old scorer got wrong. Three is enough to break ties
#: between files whose content matches equally; it is not enough to let a
#: lucky filename outrank a file that actually discusses the subject.
PATH_TOKEN_WEIGHT = 3

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def code_tokens(text: str) -> list[str]:
    """Tokenize code or a prompt into BM25 terms.

    Emits BOTH the whole identifier and its parts: `getUserById` yields
    `getuserbyid`, `get`, `user`, `by`, `id`. Keeping both is what lets a
    prompt phrased in prose reach an identifier phrased in camelCase, while a
    query that names the identifier exactly still gets the stronger whole-token
    match. Dropping the whole identifier would lose exact matches; dropping the
    parts would lose every prose query.

    Deliberately NOT length-filtered. `db`, `os`, `fs`, `ui` and `id` are real
    identifiers, and a token-length floor was measured against this corpus and
    rejected. BM25's IDF already demotes terms that appear everywhere, which is
    the corpus-derived version of what a length floor guesses at.
    """
    out: list[str] = []
    for raw in _NON_ALNUM.split(text):
        if not raw:
            continue
        out.append(raw.lower())
        parts = [p for p in _CAMEL_BOUNDARY.split(raw) if p]
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
    return out


def _read(path: str, limit: int = MAX_INDEXED_CHARS) -> str:
    """File text for indexing, or "" for anything that is not text.

    `errors="ignore"` would happily turn a PDF into thousands of junk tokens.
    That is not merely wasted work: BM25's length normalization divides by the
    corpus average document length, so binary blobs make every real file's
    length penalty depend on how many PDFs happen to sit in the repository.
    Measured on this repo before the probe: 5 PDFs were 1.7% of documents and
    **32.9% of all tokens**, and dropping them moved `avgdl` from 2773 to 1892.
    A size term whose behaviour depends on something unrelated to relevance is
    the exact defect this module exists to remove; letting it back in through
    the corpus statistics would be the same bug by another door.
    """
    try:
        with open(path, "rb") as raw:
            head = raw.read(8192)
        if b"\x00" in head:
            return ""
        with open(path, encoding="utf-8") as handle:
            return handle.read(limit)
    except (OSError, UnicodeDecodeError):
        return ""


class FileIndex:
    """BM25 over candidate file content, built once per gather.

    Rebuilt per invocation rather than cached. The gatherer already reads every
    file it selects, and a cache keyed on content would need invalidation that
    a one-shot CLI cannot amortize. Measure before adding one.
    """

    __slots__ = ("paths", "_bm25")

    def __init__(self, candidates: list[tuple[str, str, int]]):
        """`candidates` is the `(abs_path, rel_path, size)` shape `iter_paths`
        returns."""
        self.paths: list[str] = []
        documents: list[list[str]] = []
        for abs_path, rel_path, _size in candidates:
            content = _read(abs_path)
            documents.append(
                code_tokens(rel_path) * PATH_TOKEN_WEIGHT + code_tokens(content)
            )
            self.paths.append(rel_path)
        self._bm25 = BM25(documents) if documents else None

    def scores(self, prompt: str) -> dict[str, float]:
        """Per-file BM25 score for `prompt`. Absent files scored 0."""
        if self._bm25 is None:
            return {}
        terms = code_tokens(prompt)
        if not terms:
            return {}
        raw = self._bm25.scores(terms)
        return {path: score for path, score in zip(self.paths, raw) if score > 0.0}


def normalize(scores: dict[str, float], ceiling: float = 1.0) -> dict[str, float]:
    """Scale BM25 scores into `[0, ceiling]` by the maximum in this result set.

    BM25 is unbounded and its scale varies with corpus and query length, so a
    raw score cannot be added to the fixed-size bonuses the gatherer still
    applies (`EXPLICIT_PATH_BOOST` at 10.0, git recency at 0.3). Normalizing
    per query keeps those calibrated against each other: an explicitly named
    path still outranks everything, and a tie-breaker still breaks ties rather
    than deciding the ranking.

    **Known limit: this preserves ORDER but discards EVIDENCE.** The top file
    always receives the full ceiling, whether its raw score was 20 or 4. Total
    abstention works — a query matching nothing yields `{}` and no boost at all
    — but a WEAK best match is promoted as confidently as a strong one.
    Measured raw top scores on this repo:

        on-topic   "fix the fact store supersession threshold"   14.79
        on-topic   "tree-sitter parser drops interfaces"         20.17
        vague      "make it better"                               6.33
        off-topic  "how do I bake sourdough bread"                7.48
        gibberish  "zzqx wibble frobnicate"                       0.00

    So a ~2x separation between on-topic and off-topic exists in the raw score
    and is currently thrown away. An evidence term — scaling the whole query's
    contribution by `min(1.0, top / SATURATION)` — was implemented and
    measured at SATURATION 10/15/20 and moved MRR by at most +0.010, which is
    noise. It is NOT shipped, for a reason worth stating: every prompt in the
    evaluation set is an on-topic commit subject, so the harness structurally
    cannot see the prompt class the term exists for. Adding it would be
    shipping on reasoning rather than measurement. Revisit with an eval that
    contains off-topic and vague prompts, where "return nothing" is the
    correct answer and can be scored.
    """
    if not scores:
        return {}
    top = max(scores.values())
    if top <= 0.0:
        return {}
    return {path: (score / top) * ceiling for path, score in scores.items()}


def build_index(candidates: list[tuple[str, str, int]]) -> Optional["FileIndex"]:
    """`FileIndex` for `candidates`, or None when there is nothing to index."""
    return FileIndex(candidates) if candidates else None
