#!/usr/bin/env python3
"""Rank-quality harness: git-mined cases, scored through the REAL CLI.

Run: `python tools/rank_mine_eval.py --repo /path/to/repo --tree /path/to/tree`

This is the harness behind the cross-repo R@k / MRR figures. It is separate
from `tools/rank_eval.py` on purpose, and the two are NOT interchangeable:

    rank_eval.py       12 hand-labelled prompts, THIS repo, recall@k only.
                       Labels are opinions, written down and arguable.
    rank_mine_eval.py  Cases mined from git history, ANY repo, R@k + MRR.
                       Ground truth is what a commit actually changed.

Quoting a number from one under the other's name is the "same label, different
instrument" failure this branch spent a review round untangling -- `car` once
appeared as both 0.969 and 0.507 for what was called the same quantity. Every
figure this tool prints carries the generation stamp in its header; keep them
together when you copy a table out.

Two traps, both hit while this work was being done, both of which produced
confident wrong numbers:

1. **The venv installs neo EDITABLE against `src/`.** Running the CLI from a
   worktree of another commit executes THIS tree's code against THAT tree's
   files. A "baseline" run that is not the baseline: it made main look like
   MRR 0.613 when its real figure is 0.082. `--tree` is therefore mandatory
   and is exported as `PYTHONPATH`; there is no default.
2. **`score_candidate` is not the pipeline.** `gather_context` re-ranks with
   `pi_boost + hist_boost + _symbol_score`, then applies an adaptive limit and
   a byte budget -- exactly the stages that decide the tight cutoffs. An
   in-process replica overstated R@10 by 0.14. This tool shells out to
   `neo --dry-run` and parses what the model would actually have been sent.
   That costs ~2 s per case and is the only honest way to run it.

**Recency contamination: reduced, NOT removed. Read this before quoting a number.**
`get_git_recent_files` feeds the scorer the files touched by the last 50
commits PLUS everything dirty in `git status --porcelain`. Mining a case from
inside that window hands the scorer its own answer key. `--skip-recent`
(default 50) starts mining below the window, which removes the direct hit --
and an earlier version of this docstring claimed that settled it. It does not,
by two paths that are routine rather than exotic:

1. **The recency signal stores paths, not commits.** A case mined at commit 80
   whose truth file was touched AGAIN in commit 10 is back in the recent set.
   Central files are edited often, so the leak concentrates on exactly the
   files most likely to be ground truth.
2. **A dirty working tree leaks regardless of history.** Editing a file puts it
   in `git status --porcelain`, and it collects the same boost. Measuring the
   branch you are editing, from the tree you are editing it in, contaminates
   the arm you care about most.

So this tool MEASURES the leak instead of asserting it away: every run reports
`contaminated_cases` (cases whose ground truth intersects the live recent set)
and refuses to stay quiet about a dirty tree. Run it on a clean checkout, and
read the absolute figures as an upper bound on any run where that count is not
zero. Both arms inherit the leak, so direction is the robust part -- but
"direction survives" is not a licence to quote the magnitudes.

**Recall@k and hit-rate@k are different questions and are both reported.**
A case whose commit changed 4 files scores 0.25 recall on one hit and 1.0 hit
rate. "Finds the correct file in the top 10 N% of the time" is the hit rate;
"R@10" in the tables is recall. They diverge most on exactly the multi-file
commits that matter, so reporting one alone invites the reader to hear the
other.

Limits, stated so the numbers are not over-read.

The population is filtered, and every filter biases the same way -- UPWARD.
Cases must have a subject of 20+ characters that is not a merge, revert, WIP
or version bump; 1 to 5 changed source files; and those files must still exist
at HEAD. That keeps small, well-described, surviving implementation changes and
discards broad refactors, test-only work, deleted and renamed-away files, and
every commit whose author wrote a poor message. Read the result as a proxy for
focused maintenance prompts, not as a fair sample of what a developer asks.

Then the metric's own validity: a commit subject is terser and better-formed
than a real prompt, and ground truth is what the commit CHANGED, which is
neither all the files needed to understand the task nor necessarily the only
relevant ones. These are change-set retrieval figures, not a measure of
developer-context quality.

It is a better instrument than a hand-labelled dozen, not a good one.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from typing import Optional

# Ground truth is source the commit changed. Tests are excluded because the
# scorer deliberately demotes them (a test file is a lexical superset of its
# subject), so counting them as answers would score the demotion as a defect.
_TEST_RE = re.compile(r"(^|/)(tests?|spec|__tests__)/|(^|/)test_|_test\.|\.spec\.|\.test\.")
_SOURCE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".cs", ".go", ".rs", ".java",
    ".rb", ".swift", ".kt", ".c", ".h", ".cc", ".cpp", ".hpp", ".m", ".mm",
}
# A subject that names no work ("bump version to 1.2.3") is not a query. The
# conventional-commit prefix is stripped FIRST: anchoring at `^` alone let
# "chore: bump version to 0.39.0" through, and it was mined as a real case.
_CONVENTIONAL_PREFIX_RE = re.compile(r"^\w+(\([^)]*\))?!?:\s*")
_SKIP_SUBJECT_RE = re.compile(
    r"^(merge|revert|bump|release|v?\d+\.\d+\.\d+|wip\b|fixup!|squash!)", re.I
)

# One selected-file line of `--dry-run` output:
#   "  src/neo/memory/store.py (lines 1-140) - 6183 bytes (score: 4.23)"
_DRY_RUN_LINE = re.compile(r"^\s{2}(\S.*?)(?:\s+\(lines\s[\d-]+\))?\s+-\s+[\d,]+\s+bytes")


def _git(repo: str, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def mine_cases(repo: str, want: int, skip_recent: int, max_files: int) -> list[dict]:
    """Walk back from HEAD~skip_recent collecting (subject, changed source) pairs.

    Deterministic: the same repo at the same HEAD yields the same cases, which
    is what lets an A/B compare two trees rather than two samples.
    """
    head_files = set(_git(repo, "ls-files").splitlines())
    log = _git(repo, "log", "--no-merges", "--format=%H\x1f%s",
               f"--skip={skip_recent}", "-n", str(want * 12))

    cases = []
    for line in log.splitlines():
        if "\x1f" not in line:
            continue
        sha, subject = line.split("\x1f", 1)
        subject = subject.strip()
        if len(subject) < 20 or _SKIP_SUBJECT_RE.match(
            _CONVENTIONAL_PREFIX_RE.sub("", subject)
        ):
            continue

        changed = _git(repo, "show", "--name-only", "--format=", sha).splitlines()
        truth = {
            f for f in changed
            if f in head_files                       # still exists to be found
            and os.path.splitext(f)[1] in _SOURCE_EXTS
            and not _TEST_RE.search(f)
        }
        # A 40-file sweep commit is one case with 40 equal answers -- noise, not
        # a query anyone would type. A 0-file case has nothing to find.
        if not 1 <= len(truth) <= max_files:
            continue

        cases.append({"sha": sha, "query": subject, "truth": sorted(truth)})
        if len(cases) >= want:
            break
    return cases


def recent_files(repo: str, window: int = 50) -> set[str]:
    """The recency set the scorer actually reads -- history AND working tree.

    Mirrors `context_gatherer.get_git_recent_files` so a run can report how much
    of its own ground truth the scorer was handed for free.
    """
    recent = set()
    for line in _git(repo, "status", "--porcelain").splitlines():
        if len(line) > 3:
            recent.add(line[3:].strip())
    recent.update(
        _git(repo, "log", f"-{window}", "--name-only", "--format=").splitlines()
    )
    return {f for f in recent if f}


def rank_files(repo: str, tree: str, query: str, timeout: int) -> Optional[list[str]]:
    """The ranked, de-duplicated file list the model would have been sent.

    Returns None for a run that FAILED and [] for one that completed and chose
    nothing. Collapsing those was the earlier behaviour and it is the failure
    mode this whole branch exists to stop: a timeout, a crash or a renamed
    heading would have scored as perfect-zero retrieval and averaged into the
    table as evidence about ranking quality.
    """
    env = {
        **os.environ,
        "PYTHONPATH": os.path.join(tree, "src"),   # trap 1; not optional
        "NEO_OBSERVER_AUTOSTART": "0",             # don't spawn a daemon per case
        "NEO_PROFILE": "off",                      # don't write metrics per case
    }
    try:
        # stderr is MERGED, not discarded: the CLI writes its progress notices
        # AND the dry-run listing to stderr, so reading stdout alone returns an
        # empty ranking for every case and scores a working pipeline at zero.
        proc = subprocess.run(
            [sys.executable, "-m", "neo.cli", "--dry-run", query],
            cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None

    # Fail closed on both counts: a non-zero exit is not an empty ranking, and
    # a missing marker means the format moved under us, not that nothing matched.
    if proc.returncode != 0 or "=== DRY RUN" not in proc.stdout:
        return None
    out = proc.stdout
    ranked: list[str] = []
    for line in out.split("=== DRY RUN", 1)[1].splitlines():
        m = _DRY_RUN_LINE.match(line)
        if m:
            path = m.group(1).strip()
            if path not in ranked:      # a file chunked twice is one file
                ranked.append(path)
    return ranked


def score(cases: list[dict], ranked_per_case: list[list[str]], ks: list[int]) -> dict:
    recall = {k: 0.0 for k in ks}
    hit = {k: 0.0 for k in ks}
    rr = 0.0
    for case, ranked in zip(cases, ranked_per_case):
        truth = set(case["truth"])
        for k in ks:
            found = truth & set(ranked[:k])
            recall[k] += len(found) / len(truth)
            hit[k] += 1.0 if found else 0.0
        for i, path in enumerate(ranked, start=1):
            if path in truth:
                rr += 1.0 / i
                break
    n = len(cases) or 1
    return {
        "cases": len(cases),
        "recall": {f"R@{k}": round(recall[k] / n, 3) for k in ks},
        "hit_rate": {f"H@{k}": round(hit[k] / n, 3) for k in ks},
        "MRR": round(rr / n, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True, help="repository to mine and rank in")
    ap.add_argument("--tree", required=True,
                    help="neo source tree to execute (exported as PYTHONPATH; "
                         "an editable venv install will otherwise run THIS tree)")
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--skip-recent", type=int, default=50,
                    help="commits to skip so ground truth stays outside the "
                         "git-recency window the scorer reads (default 50)")
    ap.add_argument("--max-truth-files", type=int, default=5)
    ap.add_argument("--drop-contaminated", action="store_true",
                    help="score only cases whose ground truth is OUTSIDE the "
                         "scorer's recent-file set. Shrinks the sample -- often "
                         "sharply, because central files are edited often and are "
                         "also the likeliest ground truth -- in exchange for "
                         "absolute figures that mean what they say")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 10])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # Mined BEFORE the contamination filter, so --drop-contaminated still gets
    # a full-sized sample to select from rather than a filtered remnant.
    over_mine = args.cases * 4 if args.drop_contaminated else args.cases
    cases = mine_cases(args.repo, over_mine, args.skip_recent, args.max_truth_files)
    if args.drop_contaminated:
        recent_now = recent_files(args.repo, args.skip_recent)
        cases = [c for c in cases if not (set(c["truth"]) & recent_now)][: args.cases]
    if not cases:
        # Naming the wrong knob is its own defect: these two emptinesses have
        # nothing to do with each other, and only one of them is a settings problem.
        if args.drop_contaminated:
            print("every mined case was dropped as contaminated: all of their "
                  "ground truth sits in the scorer's recent-file set. On a repo "
                  "whose churn concentrates in a few central files this is the "
                  "expected outcome, and it means absolute figures here cannot "
                  "be decontaminated by sampling -- raise --skip-recent to widen "
                  "the gap, or drop the flag and read the figures as bounds.",
                  file=sys.stderr)
        else:
            print("no eligible cases mined -- widen --max-truth-files or lower "
                  "--skip-recent", file=sys.stderr)
        return 1

    raw = []
    for i, case in enumerate(cases, start=1):
        raw.append(rank_files(args.repo, args.tree, case["query"], args.timeout))
        print(f"  {i}/{len(cases)}", end="\r", file=sys.stderr, flush=True)

    # A failed case is DROPPED, not scored as a zero. Scoring it would let a
    # broken subprocess masquerade as a ranking result.
    failed = sum(1 for r in raw if r is None)
    scored_cases = [c for c, r in zip(cases, raw) if r is not None]
    scored_ranks = [r for r in raw if r is not None]
    if not scored_cases:
        print("every case failed to produce a ranking -- check --tree and the "
              "CLI invocation", file=sys.stderr)
        return 1

    recent = recent_files(args.repo, args.skip_recent)
    contaminated = sum(1 for c in scored_cases if set(c["truth"]) & recent)
    dirty = bool(_git(args.repo, "status", "--porcelain").strip())

    result = score(scored_cases, scored_ranks, sorted(args.k))
    result["failed_cases"] = failed
    result["contaminated_cases"] = contaminated
    result["repo_dirty"] = dirty
    result["repo"] = args.repo
    result["tree"] = args.tree
    result["skip_recent"] = args.skip_recent

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nrepo={args.repo}  tree={args.tree}  "
              f"cases={result['cases']}  skip_recent={args.skip_recent}")
        if failed:
            print(f"  WARNING: {failed} case(s) failed (timeout or non-zero exit) "
                  f"and were DROPPED, not scored")
        if contaminated:
            print(f"  WARNING: {contaminated}/{result['cases']} case(s) have ground "
                  f"truth inside the scorer's recent-file set; absolute figures "
                  f"are an upper bound")
        if dirty:
            print("  WARNING: working tree is dirty -- every modified path is in "
                  "the recency signal. Re-run on a clean checkout before quoting.")
        print(f"  recall   {result['recall']}")
        print(f"  hit rate {result['hit_rate']}")
        print(f"  MRR      {result['MRR']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
