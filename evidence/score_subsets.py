#!/usr/bin/env python3
"""Score the M1 runs, whole and split by query shape.

The unified-store plan asks for the concept-shaped queries reported separately
from the ones that name a path, because the front door treats them as
different stages: a named path is PINNED (stage 1) and a concept query is
RANKED (stages 3 and 4). Averaging them hides which stage moved.

The split is computed from `per_case`, not by re-mining: re-mining with a
filter would draw a second sample and the two arms would no longer be scored
on identical cases.

`shape` uses the gatherer's own `matches_explicit_path` against the repo's
TRACKED FILES, not a regex on the query and not the case's ground truth. Two
wrong definitions were tried first and both mislabelled:

- a regex on the query calls `e.g.`, `0.42.0` and `Foo.Bar` file names. The
  gatherer does not; only a token that resolves to a real file can pin.
- matching against the case's GROUND TRUTH asks a different question — "did
  the query name the answer?" — and labelled the one case in 150 whose top-10
  actually moved as concept-shaped. That case names `AGENTS.md`, `CLAUDE.md`
  and `GEMINI.md`, all real files, none of them ground truth: stage 1 fired,
  pinned two of them, and the label said it could not have.

The question that decides which stage runs is "does the query name a path this
repo has", so that is what is asked.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neo.context_gatherer import (  # noqa: E402
    extract_explicit_paths,
    matches_explicit_path,
)

EVIDENCE = os.path.dirname(os.path.abspath(__file__))
REPOS = ["neo", "aieweb", "m365dotnet"]
ARMS = ["main", "branch"]
KS = [1, 3, 10]


def tracked_files(repo_path):
    out = subprocess.run(
        ["git", "ls-files"], cwd=repo_path, capture_output=True, text=True,
        check=True, timeout=120,
    ).stdout
    return [line for line in out.splitlines() if line]


def shape(case, tracked):
    """"file-named" when the query names a path this repo actually has."""
    explicit = extract_explicit_paths(case["query"])
    if not explicit:
        return "concept"
    return (
        "file-named"
        if any(matches_explicit_path(f, explicit) for f in tracked)
        else "concept"
    )


def score(cases):
    if not cases:
        return None
    recall = {k: 0.0 for k in KS}
    hit = {k: 0.0 for k in KS}
    rr = 0.0
    for c in cases:
        truth, ranked = set(c["truth"]), c["ranked"]
        for k in KS:
            found = truth & set(ranked[:k])
            recall[k] += len(found) / len(truth)
            hit[k] += 1.0 if found else 0.0
        for i, path in enumerate(ranked, start=1):
            if path in truth:
                rr += 1.0 / i
                break
    n = len(cases)
    return {
        "n": n,
        "MRR": round(rr / n, 3),
        **{f"R@{k}": round(recall[k] / n, 3) for k in KS},
        **{f"H@{k}": round(hit[k] / n, 3) for k in KS},
    }


def main():
    rows = []
    for repo in REPOS:
        loaded = {}
        for arm in ARMS:
            path = os.path.join(EVIDENCE, f"m1_{repo}_{arm}.json")
            loaded[arm] = json.load(open(path))
        # Identical case sets are the whole basis of the comparison; assert it
        # rather than assume it. Two arms scored on different samples produce a
        # delta that describes the sample.
        queries = {a: [c["query"] for c in d["per_case"]] for a, d in loaded.items()}
        assert queries["main"] == queries["branch"], f"{repo}: case sets diverged"

        tracked = tracked_files(loaded["main"]["repo"])
        shapes = [shape(c, tracked) for c in loaded["main"]["per_case"]]
        for subset in ("all", "concept", "file-named"):
            for arm in ARMS:
                cases = [
                    c for c, sh in zip(loaded[arm]["per_case"], shapes)
                    if subset == "all" or sh == subset
                ]
                s = score(cases)
                if s:
                    rows.append({
                        "repo": repo, "subset": subset, "arm": arm,
                        "repo_head": loaded[arm]["repo_head"],
                        "tree_head": loaded[arm]["tree_head"],
                        "failed": loaded[arm]["failed_cases"], **s,
                    })

    print(json.dumps(rows, indent=2))
    hdr = f"{'repo':<12} {'subset':<11} {'arm':<7} {'n':>3} {'MRR':>6} {'R@1':>6} {'R@3':>6} {'R@10':>6} {'H@10':>6}"
    print("\n" + hdr, file=sys.stderr)
    for r in rows:
        print(
            f"{r['repo']:<12} {r['subset']:<11} {r['arm']:<7} {r['n']:>3} "
            f"{r['MRR']:>6.3f} {r['R@1']:>6.3f} {r['R@3']:>6.3f} "
            f"{r['R@10']:>6.3f} {r['H@10']:>6.3f}", file=sys.stderr,
        )


if __name__ == "__main__":
    main()
