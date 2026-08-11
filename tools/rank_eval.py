#!/usr/bin/env python3
"""File-selection retrieval evaluation: recall@k and MRR against ground truth.

    python tools/rank_eval.py --build-from-git . > cases.json
    python tools/rank_eval.py --eval . cases.json

Why this exists: file ranking was tuned three times against metrics that could
not judge it ("mean rank of the first source file", "count of tests and docs in
the top 10"). Both reward ANY src/ file landing early regardless of relevance,
both penalise a test file that is the correct answer, and both IMPROVE when a
file is evicted from context entirely. Recall against labels has none of those
properties.

Labels come from git history -- commit subject as the query, changed non-test
files as ground truth, keeping commits that touch 1-3 files. That is the
standard construction in the bug-localization literature and it removes the two
worst properties of a hand-written set: prompts chosen by the person evaluating
the change, and labels that are that person's opinion.

Report several k. Differences live at tight cutoffs, because per-file character
caps mean a file at rank 3 contributes far more content than the same file at
rank 9.

Limits: commit subjects are terser and better-formed than real prompts; ground
truth is what a commit CHANGED, a subset of what a developer needed to READ;
content is read at HEAD while the commit is historical.

Diagnosis and plan: ~/git/working/2026-08-10-neo-file-selection-plan.md
"""

import json
import subprocess
import sys
import os
import re

def build(repo, max_commits=400, exts=(".py",".ts",".tsx",".js",".cs",".go",".rb",".php",".java")):
    log = subprocess.run(
        ["git","-C",repo,"log","--no-merges","-n",str(max_commits),
         "--pretty=format:%H%x1f%s","--name-only"],
        capture_output=True, text=True).stdout
    cases=[]
    for block in log.split("\n\n"):
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        head=lines[0]
        if "\x1f" not in head:
            continue
        sha, subject = head.split("\x1f",1)
        files=[f for f in lines[1:] if f.endswith(exts)]
        files=[f for f in files if not re.search(r'(^|/)(tests?|spec)/|(^|/)test_|_test\.', f)]
        # A usable case: a focused commit. Huge commits have no single answer;
        # zero-file commits (docs-only) have no answer at all.
        if 1 <= len(files) <= 3 and len(subject) > 15:
            cases.append({"sha": sha, "query": subject, "files": sorted(set(files))})
    return cases


from collections import Counter  # noqa: E402
from neo.context_gatherer import (iter_paths, extract_prompt_tokens, score_candidate,  # noqa: E402
                                  get_git_recent_files)
from neo.memory.bm25 import BM25  # noqa: E402

ENTRY = {'main','app','server','index','login','auth','__init__'}
_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

def code_tokens(text):
    """Code-aware tokenization: split identifiers on camelCase and separators,
    and keep BOTH the whole identifier and its parts. The literature finds this
    is what makes BM25 beat dense retrieval on code."""
    out = []
    for raw in _SPLIT.split(text):
        if not raw:
            continue
        low = raw.lower()
        out.append(low)
        parts = [p.lower() for p in _CAMEL.split(raw) if p]
        if len(parts) > 1:
            out.extend(parts)
    return out

def load_corpus(root, cands, cap=200_000):
    docs, paths = [], []
    for abs_p, rel, size in cands:
        try:
            with open(abs_p, encoding="utf-8", errors="ignore") as f:
                content = f.read(cap)
        except OSError:
            continue
        # Path tokens are repeated so the filename still carries weight —
        # a file named for the thing is real evidence, just not the only one.
        docs.append(code_tokens(rel) * 3 + code_tokens(content))
        paths.append(rel)
    return docs, paths

def rrf(rankings, k=60):
    """Reciprocal rank fusion. Rank-based, so it needs no score normalization
    and no weight tuning between channels on different scales."""
    fused = Counter()
    for ranked in rankings:
        for i, p in enumerate(ranked):
            fused[p] += 1.0 / (k + i + 1)
    return [p for p, _ in fused.most_common()]

def evaluate(root, cases, ks=(1,3,5,10,20)):
    cands = iter_paths(root, [], [], None)
    cands = [c for c in cands if c[1].endswith((".py",".ts",".tsx",".js",".cs",".go",".rb",".php",".java"))]
    sizes = {rel: sz for _, rel, sz in cands}
    gr = get_git_recent_files(root)
    docs, paths = load_corpus(root, cands)
    bm = BM25(docs)
    try:
        from neo.index.project_index import ProjectIndex
        idx = ProjectIndex(root)
        has_idx = bool(idx.chunks)
    except Exception:
        idx, has_idx = None, False

    strategies = ["neo_current", "bm25_content", "dense", "rrf_bm25_dense"]
    hits = {s: {k: 0.0 for k in ks} for s in strategies}
    mrr = {s: 0.0 for s in strategies}
    n = 0

    for case in cases:
        want = set(case["files"])
        if not (want & set(paths)):
            continue  # ground truth no longer in the tree
        n += 1
        q = case["query"]

        toks = extract_prompt_tokens(q)
        cur = sorted(((score_candidate(r, sizes[r], toks, gr, ENTRY), r) for _, r, _ in cands), reverse=True)
        cur = [r for _, r in cur]

        qt = code_tokens(q)
        bs = bm.scores(qt)
        bmr = [paths[i] for i in sorted(range(len(paths)), key=lambda i: -bs[i])]

        if has_idx:
            best = {}
            for ch in idx.retrieve(q, 60):
                p = os.path.relpath(ch.file_path, root) if os.path.isabs(ch.file_path) else ch.file_path
                best[p] = max(best.get(p, 0.0), float(getattr(ch, "similarity", 0.0)))
            dn = [p for p, _ in sorted(best.items(), key=lambda kv: -kv[1])]
        else:
            dn = []

        ranked = {"neo_current": cur, "bm25_content": bmr, "dense": dn,
                  "rrf_bm25_dense": rrf([bmr, dn]) if dn else bmr}
        for s, r in ranked.items():
            for k in ks:
                hits[s][k] += len(want & set(r[:k])) / len(want)
            pos = next((i + 1 for i, p in enumerate(r) if p in want), None)
            mrr[s] += 1.0 / pos if pos else 0.0

    print(f"  {n} evaluable cases (of {len(cases)})\n")
    print(f"  {'strategy':<18}" + "".join(f"{'R@'+str(k):>8}" for k in ks) + f"{'MRR':>8}")
    for s in strategies:
        print(f"  {s:<18}" + "".join(f"{hits[s][k]/n:>8.3f}" for k in ks) + f"{mrr[s]/n:>8.3f}")


def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build-from-git", metavar="REPO",
                    help="mine an eval set from that repo's git history, to stdout")
    ap.add_argument("--eval", nargs=2, metavar=("REPO", "CASES"),
                    help="evaluate strategies for REPO against a cases file")
    args = ap.parse_args()

    if args.build_from_git:
        cases = build(args.build_from_git)
        print(json.dumps(cases, indent=1))
        print(f"# {len(cases)} cases", file=sys.stderr)
        return 0
    if args.eval:
        repo, casefile = args.eval
        cases = [c for c in json.load(open(casefile))
                 if not c["query"].lower().startswith("release ")]
        evaluate(repo, cases)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
