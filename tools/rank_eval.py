#!/usr/bin/env python3
"""Rank-quality harness: recall@k against hand-labelled relevant files.

Run: `python tools/rank_eval.py [k ...]` from the repo root.

This exists because three separate attempts to improve file ranking were
judged with metrics that could not judge them. The first two were "mean rank
of the first source file" and "count of tests and docs in the top 10", and
both are wrong in the same way: they reward ANY `src/` file landing early
regardless of relevance, and they penalise a test file that is the correct
answer. Measured live -- for "add a new CLI subcommand for exporting facts",
one candidate change surfaced `tests/test_subcommands.py` and
`src/neo/prompt/cli.py` (both relevant) and scored WORSE than a baseline whose
top hits were `.pytest_cache/README.md`, `docs/mistakes.md` and
`construct/README.md`.

They are also blind to eviction: a file pushed below `MIN_SCORE_THRESHOLD`
leaves context entirely, and both metrics IMPROVE when that happens.

Recall against labels has neither problem. The labels are the files a
competent engineer would want in context for that prompt; they are opinions,
but they are written down and arguable, which the previous metrics were not.

**Report several k.** The interesting differences live at tight cutoffs,
because per-file character caps mean a file at rank 3 contributes far more
content than the same file at rank 9. A change can be flat at k=10 and still
be a real improvement -- that is exactly the shape of the test-file demotion
this harness was built to check.

Limits, stated so the numbers are not over-read: 12 prompts, one repository,
labels by one person, and recall ignores what else got in. It is a better
instrument than a rank mean, not a good one.
"""
import re
import subprocess
import sys

LABELLED = {
 "add retry logic to the CAR adapter":
   {"src/neo/adapters.py"},
 "fix the fact store supersession threshold":
   {"src/neo/memory/store.py"},
 "why does the observer leak memory":
   {"src/neo/memory/observer.py"},
 "add a new CLI subcommand for exporting facts":
   {"src/neo/subcommands.py", "src/neo/cli.py"},
 "the transcript ingester is dropping episodes":
   {"src/neo/memory/transcript.py"},
 "the tree-sitter parser drops interfaces":
   {"src/neo/index/language_parser.py"},
 "context assembly exceeds the token budget":
   {"src/neo/memory/context.py", "src/neo/text_budget.py"},
 "episode promotion never fires":
   {"src/neo/memory/store.py", "src/neo/memory/episodes.py"},
 "the orchestrator emits no terminal event":
   {"src/neo/events.py", "src/neo/engine.py"},
 "fix the reasoning mode decision gate":
   {"src/neo/reasoning_mode.py", "src/neo/engine.py"},
 "prompt truncation markers are missing":
   {"src/neo/text_budget.py", "src/neo/engine.py"},
 "the A2UI inspector shows a stale fact count":
   {"src/neo/a2ui.py"},
}

def selected(prompt, k=10):
    out = subprocess.run([sys.executable, "-m", "neo.cli", "--dry-run", "--no-git",
                          prompt, "--cwd", "."], capture_output=True, text=True, timeout=300)
    txt = out.stdout + out.stderr
    body = txt.split("DRY RUN", 1)[-1]
    paths = []
    for line in body.splitlines():
        m = re.match(r"^  (\S+?)(?: \(lines [\d-]+\))? - \d+ bytes", line)
        if m and m.group(1) not in paths:
            paths.append(m.group(1))
    return paths[:k]


def main() -> int:
    ks = [int(a) for a in sys.argv[1:]] or [3, 5, 10, 20]
    widest = max(ks)
    cache = {p: selected(p, widest) for p in LABELLED}

    for k in ks:
        total = 0.0
        misses = []
        for prompt, want in LABELLED.items():
            got = set(cache[prompt][:k])
            total += len(want & got) / len(want)
            if not (want & got):
                misses.append(prompt)
        print(f"  recall@{k:<3} = {total / len(LABELLED):.3f}"
              f"   prompts with no relevant file: {len(misses)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
