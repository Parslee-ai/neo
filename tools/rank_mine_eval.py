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
   MRR 0.613 against a real figure of 0.082 -- both SUPERSEDED-generation
   numbers, quoted for the size of the error and not as measurements, since
   this harness puts main's neo MRR at 0.304. `--tree` is therefore mandatory,
   is made absolute before use, is checked against what `import neo` actually
   resolves to, and is exported as `PYTHONPATH`; there is no default.
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

So the default is to switch the leaking signal OFF. `neo --dry-run` already
takes `--no-git`, which gates `git_recent` and nothing else, and
`tools/rank_eval.py` has been passing it all along. Turning it off costs a real
signal (ablated at +0.021 R@10 on the superseded harness generation -- an
unstamped figure, quoted here for scale, not as a current measurement) and buys
absolute numbers that are not partly an echo of the answer key, which is the
better trade for an A/B whose whole purpose is comparing two rankers.

`_history_boost` and the other re-rank channels stay live under `--no-git`.
That means UNGATED, not clean: `_history_boost` reads the FactStore, which past
neo runs against this same repo populated. Different provenance, same family.
CLAUDE.md measures its contribution as nil, so it is not moving these numbers,
but do not read "stays live" as "is uncontaminated".

`--with-git` re-enables it to measure the full shipped pipeline. That run is
contaminated by construction, so it also reports `contaminated_cases` (ground
truth intersecting the live recent set) and warns on a dirty tree; read its
absolutes as upper bounds. Both arms inherit the leak, so direction survives --
but "direction survives" is not a licence to quote the magnitudes.

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

**The leak that no flag removes.** Cases are queried with a commit subject and
scored against the files as they exist at HEAD -- that is, AFTER that commit
landed. The subject's terms are in the file partly because the commit put them
there, so the ranker is reading a corpus that has already seen the answer.
BugLocator, cited approvingly by the module this measures, indexes the pre-fix
snapshot for exactly this reason; this harness does not. `--no-git` does
nothing about it. Both arms inherit it, so the direction survives -- but it is
why the absolutes remain upper bounds even at the default, and it is the reason
not to read a headline R@10 as "the ranker finds the right file 74% of the
time" on prompts written before the work exists.

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


def is_skippable(subject: str) -> bool:
    """A subject that names no work is not a query.

    The prefix strip and the skip test live TOGETHER here rather than being
    composed at the call site: `chore: bump version to 0.39.0` was mined as a
    real case because `_SKIP_SUBJECT_RE` anchors at `^` and the conventional
    prefix sat in front of it. Composed at the call site, a test can only pin
    the composition it performs itself, which stays green while `mine_cases`
    drops the strip.

    Tested BOTH ways round, because some skip words are themselves valid
    conventional-commit types: `revert: the thing that broke prod` and `wip:
    still figuring this out` have their own skip word consumed as a prefix and
    read as ordinary work on the stripped subject alone. A revert's changed
    files are a removal, which is not ground truth for its own subject.
    """
    stripped = _CONVENTIONAL_PREFIX_RE.sub("", subject)
    return bool(_SKIP_SUBJECT_RE.match(subject) or _SKIP_SUBJECT_RE.match(stripped))


def is_ground_truth(path: str, head_files: set[str]) -> bool:
    """Whether a changed file counts as an answer for the case it came from."""
    return (
        path in head_files                               # still exists to be found
        and os.path.splitext(path)[1] in _SOURCE_EXTS
        and not _TEST_RE.search(path)                    # demoted by design
    )


def _git(repo: str, *args: str) -> str:
    # Timeout so a hung git (a lock, a prompting credential helper) fails the
    # run instead of parking it forever with no output.
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True,
        timeout=120,
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
        if len(subject) < 20 or is_skippable(subject):
            continue

        changed = _git(repo, "show", "--name-only", "--format=", sha).splitlines()
        truth = {f for f in changed if is_ground_truth(f, head_files)}
        # A 40-file sweep commit is one case with 40 equal answers -- noise, not
        # a query anyone would type. A 0-file case has nothing to find.
        if not 1 <= len(truth) <= max_files:
            continue

        cases.append({"sha": sha, "query": subject, "truth": sorted(truth)})
        if len(cases) >= want:
            break
    return cases


def _git_head(path: str) -> str:
    """Short HEAD, or a marker -- never an exception.

    `--tree` is not required to be a git checkout, and stamping it must not be
    able to destroy a run. Computed BEFORE the case loop for the same reason:
    a `git worktree prune`d checkout still has `src/neo`, so it passes the
    effectiveness guard, ranks all 50 cases, and only then discovers it cannot
    be stamped -- ~100 seconds of subprocess work thrown away with a traceback
    pointing at a string format instead of at the real problem.

    `rev-parse` searches ANCESTORS, so a non-git tree sitting inside a checkout
    would stamp that enclosing checkout's sha -- a confidently wrong label on
    the field whose whole point is saying which ranker ran. Compared against
    `--show-toplevel` so only the path itself counts.
    """
    try:
        top = _git(path, "rev-parse", "--show-toplevel").strip()
        if os.path.realpath(top) != os.path.realpath(path):
            return "not-a-git-checkout"
        return _git(path, "rev-parse", "HEAD").strip()[:12]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return "not-a-git-checkout"


def assert_tree_is_effective(tree: str, repo: str) -> None:
    """Prove `--tree` actually won, rather than trusting that it did.

    Mandatory is not the same as effective. The venv's editable install is a
    `.pth` naming this checkout, so a `--tree` that points at a typo, a pruned
    worktree or one directory too high does not fail -- `import neo` quietly
    falls through to the working tree, BOTH arms run the same code, and the run
    completes with a full ranking and `failed_cases: 0`. That is the 0.613
    against 0.082 error (superseded generation, see the module docstring),
    unguarded, inside the tool written to prevent it.

    Two ways this guard could check one directory while the run measured
    another, both closed here, and the second is why it takes `repo`:

    1. A RELATIVE `--tree` resolved against the harness's cwd while
       `rank_files` handed the same string to a child running with `cwd=repo`.
       Python drops a nonexistent `PYTHONPATH` entry without a word, so the
       guard stayed green while the run used the installed copy. `main`
       absolutizes both paths before calling here.
    2. The PROBE must run from `cwd=repo`, exactly as `rank_files` does.
       Python puts the cwd at `sys.path[0]`, AHEAD of `PYTHONPATH`, for both
       `-c` and `-m` -- so a repo containing a top-level `neo/` package
       shadows the named tree in the measurement while a probe run from
       anywhere else sees nothing wrong. None of neo, car or quip has such a
       directory today; a guard is the one place that is not an argument.
    """
    src = os.path.join(tree, "src")
    if not os.path.isdir(os.path.join(src, "neo")):
        raise SystemExit(f"--tree {tree!r} has no src/neo -- it would silently "
                         f"fall back to the installed copy and measure the same "
                         f"code twice")
    loaded = subprocess.run(
        [sys.executable, "-c", "import neo, sys; sys.stdout.write(neo.__file__)"],
        cwd=repo,   # resolve imports the way the measurement does; see above
        env={**os.environ, "PYTHONPATH": src}, capture_output=True, text=True,
        timeout=120,
    ).stdout.strip()
    if not loaded.startswith(os.path.realpath(src)) and \
            not loaded.startswith(src):
        raise SystemExit(f"--tree {tree!r} did not take effect: `import neo` "
                         f"resolved to {loaded!r}. Every figure from this run "
                         f"would describe that tree, not the one you named.")


def recent_files(repo: str, window: int = 50) -> set[str]:
    """The recency set the scorer actually reads -- history AND working tree.

    Mirrors `context_gatherer.get_git_recent_files` so a run can report how much
    of its own ground truth the scorer was handed for free. `window` MUST match
    that function's hardcoded `-50`; it is deliberately not wired to
    `--skip-recent`, which is a different quantity (how far back to start
    mining). Passing the latter here under-reports contamination whenever it is
    lowered -- an error in the flattering direction.
    """
    recent = set()
    for line in _git(repo, "status", "--porcelain").splitlines():
        if len(line) > 3:
            recent.add(line[3:].strip())
    recent.update(
        _git(repo, "log", f"-{window}", "--name-only", "--format=").splitlines()
    )
    return {f for f in recent if f}


def rank_files(repo: str, tree: str, query: str, timeout: int,
               use_git: bool = False) -> Optional[list[str]]:
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
        cmd = [sys.executable, "-m", "neo.cli", "--dry-run"]
        if not use_git:
            cmd.append("--no-git")     # gates git_recent only; see module docstring
        proc = subprocess.run(
            [*cmd, query],
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
    ap.add_argument("--with-git", action="store_true",
                    help="measure the full shipped pipeline, recency signal "
                         "included. Off by default because that signal is fed "
                         "the answer key: the run is contaminated by "
                         "construction and its absolutes are upper bounds")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 10])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # Absolute BEFORE anything reads them. Both are handed to children running
    # with `cwd=repo`, while the guard below resolves them from the HARNESS's
    # cwd -- so `--repo ~/git/car --tree .` checked one directory and measured
    # another, and Python drops a nonexistent PYTHONPATH entry silently, which
    # put the guard green while both arms ran the installed copy.
    args.tree = os.path.abspath(args.tree)
    args.repo = os.path.abspath(args.repo)
    assert_tree_is_effective(args.tree, args.repo)
    # Stamped up front: nothing about identifying the two inputs should be
    # discoverable only after a 50-case run has already been spent.
    repo_head, tree_head = _git_head(args.repo), _git_head(args.tree)
    cases = mine_cases(args.repo, args.cases, args.skip_recent, args.max_truth_files)
    if not cases:
        print("no eligible cases mined -- widen --max-truth-files or lower "
              "--skip-recent", file=sys.stderr)
        return 1

    raw = []
    for i, case in enumerate(cases, start=1):
        raw.append(rank_files(args.repo, args.tree, case["query"], args.timeout,
                              use_git=args.with_git))
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

    # Fail-closed on the HEADER is not enough. A per-line format change keeps
    # the banner and parses to nothing, so every case returns [] and the run
    # reports a confident R@10 0.000 with failed_cases: 0 -- "scored every case
    # zero" coming back through a different door. A live trigger exists today:
    # the CLI's second dry-run printer emits ABSOLUTE paths, which parse fine
    # and match no repo-relative ground truth. Nothing parsed anywhere is a
    # broken instrument, never a result.
    if not any(scored_ranks):
        print("the CLI ran and the DRY RUN banner was found, but NOT ONE case "
              "yielded a parsable file line -- the output format has moved. "
              "This is a parser failure, not a score of zero.", file=sys.stderr)
        return 1

    # Only meaningful when the recency signal is live; with --no-git the scorer
    # never reads it, so reporting a count would imply a leak that cannot occur.
    contaminated = dirty = None
    if args.with_git:
        recent = recent_files(args.repo)   # the scorer's window, not --skip-recent
        contaminated = sum(1 for c in scored_cases if set(c["truth"]) & recent)
        dirty = bool(_git(args.repo, "status", "--porcelain").strip())

    result = score(scored_cases, scored_ranks, sorted(args.k))
    result["failed_cases"] = failed
    result["git_recency"] = bool(args.with_git)
    result["contaminated_cases"] = contaminated
    result["repo_dirty"] = dirty
    result["repo"] = args.repo
    result["tree"] = args.tree
    result["skip_recent"] = args.skip_recent
    # --skip-recent counts back from HEAD, so two arms measured at different
    # HEADs score different case sets. Without this stamp a delta between runs
    # cannot be attributed to the change under test rather than to the window
    # having moved -- which has already happened once on this branch.
    result["repo_head"] = repo_head
    # And the TREE's revision, which is the object the two arms actually differ
    # by. `result["tree"]` is a mutable path, not a generation: it says where
    # the ranker was read from, never which ranker it was.
    result["tree_head"] = tree_head

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # The generation stamp belongs in the header people copy tables out of,
        # not only in --json.
        print(f"\nrepo={args.repo}@{result['repo_head']}  "
              f"tree={args.tree}@{result['tree_head']}  cases={result['cases']}  "
              f"skip_recent={args.skip_recent}  "
              f"git_recency={'ON' if args.with_git else 'off (--no-git)'}")
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
