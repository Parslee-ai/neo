#!/usr/bin/env python3
"""Does neo's context selection beat naive baselines on the SAME mined cases?

Absolute R@k figures cannot answer "is neo effective" — they have no
comparator. This scores four naive rankers over the identical candidate
universe, queries and ground truth that `rank_mine_eval.py` used, so the only
thing that differs is the ranking rule.

Baselines, chosen to be fair rather than strawmen:
  random      the floor: what chance looks like on this corpus.
  filename    prompt tokens matched against the PATH. The "grep the filename"
              heuristic, and what neo's own scorer did before content BM25.
  recency     most-recently-modified first. A real heuristic developers use.
              CAVEAT: mtime-based, so editing files in the repo under
              measurement perturbs it — this harness lives in neo and
              editing it moved its own score. Weak baseline; not the
              comparator the claim rests on (that is `grep`).
  size        largest first. CLAUDE.md notes central files are large, so this
              is a genuine signal, not a joke baseline.
"""
import hashlib
import json
import random
import re
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, "/Users/mliotta/git/neo/src")
from neo.eligibility import walk, WalkPolicy  # noqa: E402

_SOURCE_EXTS = {".py",".js",".jsx",".ts",".tsx",".cs",".go",".rs",".java",
                ".rb",".swift",".kt",".c",".h",".cc",".cpp",".hpp",".m",".mm"}
_TEST_RE = re.compile(r"(^|/)(tests?|spec|__tests__)/|(^|/)test_|_test\.|\.spec\.|\.test\.")
_PREFIX = re.compile(r"^\w+(\([^)]*\))?!?:\s*")

def tokens(q):
    q = _PREFIX.sub("", q)
    return {t for t in re.split(r"[^A-Za-z0-9_]+", q.lower()) if len(t) >= 3}

def candidates(repo):
    res = walk(repo, WalkPolicy(match_globs=("**/*",)))
    out = []
    for e in res.paths:
        p = Path(e.path)
        rel = str(p.relative_to(repo))
        if p.suffix.lower() not in _SOURCE_EXTS:
            continue
        if _TEST_RE.search(rel):
            continue
        out.append(rel)
    return out

def metrics(ranked, truth, ks=(1,3,10)):
    t = set(truth)
    out = {}
    for k in ks:
        top = ranked[:k]
        hit = len(t & set(top))
        out[f"R@{k}"] = hit / len(t) if t else 0.0
        out[f"H@{k}"] = 1.0 if hit else 0.0
    rr = 0.0
    for i, p in enumerate(ranked, 1):
        if p in t:
            rr = 1.0 / i
            break
    out["RR"] = rr
    return out

def main():
    repo = sys.argv[1]
    neo_json = sys.argv[2]
    seed = 20260830
    cases = json.load(open(neo_json))["per_case"]
    cands = candidates(repo)
    stats = {}
    for p in cands:
        fp = Path(repo)/p
        try:
            st = fp.stat()
            stats[p] = (st.st_mtime, st.st_size)
        except OSError:
            stats[p] = (0, 0)

    # Content cache for the grep baseline. Read once; files are small enough
    # that this is cheaper than re-reading per case.
    body = {}
    for rel in cands:
        try:
            body[rel] = (Path(repo) / rel).read_text(
                encoding="utf-8", errors="replace").lower()
        except OSError:
            body[rel] = ""

    def grep_rank(q, rnd):
        """Count prompt-token occurrences in file CONTENT — what a developer
        does with grep, and the control that matters: it reads content, so it
        inherits the same "corpus has seen the answer" leak neo's BM25 does.
        If neo only beat content-blind baselines, the advantage could be the
        leak rather than the ranker."""
        tk = tokens(q)
        def score(rel):
            b = body[rel]
            return -sum(b.count(t) for t in tk)
        return sorted(cands, key=lambda rel: (score(rel), rel))

    rankers = {
        "random":   lambda q, rnd: rnd.sample(cands, len(cands)),
        "grep":     grep_rank,
        "filename": lambda q, rnd: sorted(cands, key=lambda p: (
                        -len(tokens(q) & set(re.split(r"[^A-Za-z0-9_]+", p.lower()))), p)),
        "recency":  lambda q, rnd: sorted(cands, key=lambda p: -stats[p][0]),
        "size":     lambda q, rnd: sorted(cands, key=lambda p: -stats[p][1]),
    }
    agg = {name: [] for name in rankers}
    agg["neo"] = []
    for c in cases:
        q, truth = c["query"], c["truth"]
        if not truth:
            continue
        agg["neo"].append(metrics(c["ranked"], truth))
        # hashlib, NOT hash(): Python salts str hashing per process
        # (PYTHONHASHSEED), so hash(q) made the random floor
        # unreproducible across runs. Caught by re-running.
        digest = hashlib.sha256(q.encode("utf-8")).hexdigest()
        rnd = random.Random(seed + int(digest[:8], 16))
        for name, fn in rankers.items():
            agg[name].append(metrics(fn(q, rnd), truth))

    n = len(agg["neo"])
    print(f"repo={repo}  cases={n}  candidate files={len(cands)}\n")
    hdr = f"{'ranker':<10} {'R@1':>6} {'R@3':>6} {'R@10':>6} {'MRR':>6} {'H@1':>6} {'H@10':>6}"
    print(hdr)
    print("-" * len(hdr))
    order = ["random","recency","size","filename","grep","neo"]
    res = {}
    for name in order:
        rows = agg[name]
        m = {k: sum(r[k] for r in rows)/len(rows) for k in rows[0]}
        res[name] = m
        star = "  <- neo" if name == "neo" else ""
        print(f"{name:<10} {m['R@1']:>6.3f} {m['R@3']:>6.3f} {m['R@10']:>6.3f} "
              f"{m['RR']:>6.3f} {m['H@1']:>6.3f} {m['H@10']:>6.3f}{star}")
    print()
    best_naive = max((n_ for n_ in order if n_!="neo"), key=lambda x: res[x]["RR"])
    print(f"best naive baseline: {best_naive} (MRR {res[best_naive]['RR']:.3f})")
    print(f"neo MRR {res['neo']['RR']:.3f}  ->  "
          f"{res['neo']['RR']/max(res[best_naive]['RR'],1e-9):.1f}x the best naive, "
          f"{res['neo']['RR']/max(res['random']['RR'],1e-9):.0f}x random")
    # paired sign test neo vs best naive
    bet = wor = 0
    for a, b in zip(agg["neo"], agg[best_naive]):
        if a["RR"] > b["RR"] + 1e-9:
            bet += 1
        elif a["RR"] < b["RR"] - 1e-9:
            wor += 1
    nn = bet + wor
    k = min(bet, wor)
    p=min(sum(comb(nn,i) for i in range(k+1))/(2**nn)*2,1.0) if nn else 1.0
    print(f"paired neo vs {best_naive}: {bet} better / {wor} worse / {n-nn} tied, sign p={p:.2e}")

main()
