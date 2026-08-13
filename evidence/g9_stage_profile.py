#!/usr/bin/env python
"""Per-stage wall-clock profile of one real `neo --dry-run` invocation.

    <branch>/.venv/bin/python evidence/g9_stage_profile.py <target-repo> [prompt]

Goal 9's closing re-run of the instrumentation Goal 7 introduced (which lived in
/tmp and therefore could not be re-run by anyone reading its numbers; this one is
committed beside the numbers it produced).

No profiler. Timers are wrapped around the stage functions themselves, so the
parts sum to a wall-clock a stopwatch agrees with — a sampling profiler's
attribution would not survive being quoted against the battery's median.

Read the totals as two numbers, never one. `GATHER TOTAL` is file selection; the
fact-store rows are the memory system, which is issue #211 and out of the
unified-store plan's scope. Reporting only the sum credits or blames retrieval
for a cost it does not own.
"""

import os
import resource
import sys
import time
from collections import defaultdict

T0 = time.perf_counter()

TIMES: dict[str, float] = defaultdict(float)
CALLS: dict[str, int] = defaultdict(int)


def _wrap(owner, name, label):
    """Time every call to `owner.name`, in place. Returns silently if the
    attribute is absent so a rename shows up as a missing ROW rather than as a
    crash that loses the whole profile."""
    original = getattr(owner, name, None)
    if original is None:
        return

    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            TIMES[label] += time.perf_counter() - start
            CALLS[label] += 1

    setattr(owner, name, timed)


def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    prompt = sys.argv[2] if len(sys.argv) > 2 else (
        "How does the backend authenticate requests from the web app?"
    )

    os.environ.setdefault("NEO_OBSERVER_AUTOSTART", "0")

    t_import = time.perf_counter()
    from neo import context_gatherer, eligibility
    from neo.index import content_index as ci
    from neo.memory import store as memory_store
    TIMES["imports"] = time.perf_counter() - t_import

    _wrap(eligibility, "walk", "eligibility walk")
    _wrap(ci.ContentIndex, "refresh", "content index refresh")
    _wrap(ci.ContentIndex, "scores", "content index scores")
    _wrap(context_gatherer, "_project_index_boost", "project index boost")
    _wrap(context_gatherer, "_history_boost", "fact store history boost")
    _wrap(memory_store.FactStore, "retrieve_relevant", "fact store retrieve")
    _wrap(context_gatherer, "gather_context", "GATHER TOTAL")

    from neo.cli import main as cli_main

    sys.argv = ["neo", "--dry-run", "--quiet", "--cwd", repo, prompt]
    try:
        cli_main()
    except SystemExit:
        pass

    wall = time.perf_counter() - T0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports ru_maxrss in KiB, macOS in bytes.
    rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3

    order = [
        "imports",
        "eligibility walk",
        "content index refresh",
        "content index scores",
        "project index boost",
        "fact store history boost",
        "fact store retrieve",
        "GATHER TOTAL",
    ]
    print(f"\n=== per-stage profile: {repo} ===", file=sys.stderr)
    for label in order:
        if label not in TIMES:
            continue
        suffix = "" if label == "imports" else (
            f"  ({CALLS[label]} call{'' if CALLS[label] == 1 else 's'})")
        print(f"T {label:<40}{TIMES[label]:>7.2f}s{suffix}", file=sys.stderr)
    print(f"T {'WALL TOTAL':<40}{wall:>7.2f}s   peak rss = {rss_mb:.0f} MB",
          file=sys.stderr)

    # The two memory rows are NOT both inside GATHER TOTAL and must not be
    # subtracted from it together. `_history_boost` is a stage of the gather;
    # the engine's own `retrieve_relevant` runs after the gather returns.
    # Subtracting both from GATHER produced a negative "selection only" on the
    # first run of this script, which is the arithmetic saying so out loud.
    history = TIMES.get("fact store history boost", 0.0)
    engine_recall = TIMES.get("fact store retrieve", 0.0)
    print(f"T {'  selection only (GATHER minus history boost)':<40}"
          f"{TIMES.get('GATHER TOTAL', 0.0) - history:>7.2f}s", file=sys.stderr)
    print(f"T {'  wall excluding the memory system':<40}"
          f"{wall - history - engine_recall:>7.2f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
