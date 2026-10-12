# Measuring retrieval: delivery cap, effectiveness, measurement traps

Detailed engineering notes moved verbatim from `CLAUDE.md` (2026-10-12). `CLAUDE.md` keeps the short rules; this file keeps the reasons, measurements and footguns behind them.

- **Delivery-cap sweep** (`docs/delivery-cap-sweep-2026-08-30.md`;
  `rank_mine_eval.py --max-files N`). `--max-files` (default 30) had never been
  measured, and could not be until it was fixed: `calculate_adaptive_limit`
  returned its three broad-prompt buckets VERBATIM, so `--max-files 5` on a
  vague prompt delivered 15, and sweeping below 25 moved nothing for any prompt
  that was not highly specific.
  **The default of 30 is well chosen and must not be cut to 10.** Rescored over
  everything the model receives: recall(delivered) rises monotonically —
  neo 0.523 / 0.654 / 0.776 / **0.813** / 0.869 and m365dotnet 0.571 / 0.652 /
  0.714 / **0.741** / 0.750 at caps 5/10/20/30/50. Going 10 -> 30 buys +24%
  relative recall on neo and +14% on m365dotnet for +50% context bytes; 30 -> 50
  buys +7% / +1% for a further +28% bytes AND degrades the top of the list
  (MRR 0.763 -> 0.752, R@1 0.403 -> 0.377), so extra files dilute the ranking
  they extend. 30 sits where returns flatten without that dilution.
  **Footgun, and the reason a first pass concluded the opposite:** R@10 and MRR
  are **structurally blind to ranks 11+**, so a correct answer delivered at rank
  15 cannot move either number at any cap. They read flat from cap 10 to 50
  because they COULD NOT VARY, and the first analysis published "the knee is ~10,
  the default costs 15-50% for nothing" in v0.52.0's changelog and release notes
  before this was caught. Measured properly, **truth sits at rank 11-30 for 15%
  of neo's answer files and 11% of m365dotnet's**, so a cap of 10 would drop 23
  of 107 and 14 of 112 respectively (concretely: `src/neo/cli.py` at rank 14,
  `project_index.py` at rank 12). **A delivery question needs recall over the
  DELIVERED set, never recall@10** — using a top-k metric to size a cap larger
  than k measures nothing about the files the cap admits.
  Byte cost is **not monotonic in file count** — cap 5 spends MORE than cap 10
  (142 KB vs 131 KB) while delivering less, because `--max-bytes` is apportioned
  across whatever is admitted. Below 10 there is real loss on both metrics.
- **Effectiveness evidence** (`docs/effectiveness-evidence-2026-08-30.md`,
  `tools/rank_baseline_eval.py`). Absolute R@k figures have no comparator and
  cannot answer "is neo effective". The baseline harness scores four naive
  rankers over the IDENTICAL candidate set, queries and ground truth
  `rank_mine_eval.py` used, so only the ranking rule differs.
  **The load-bearing baseline is `grep`** — prompt-token counts over file
  CONTENT — because the mining leak (a file holds the commit's terms because
  that commit put them there) helps any content-reading ranker and does nothing
  for `size`/`recency`. Comparing only against content-blind baselines would let
  the leak masquerade as ranker quality. Measured, MRR vs grep: neo 0.778/0.584
  (1.3×), aieweb 0.729/0.511 (1.4×), m365dotnet 0.536/0.219 (2.4×); **pooled
  over 150 cases 81 better / 37 worse / 32 tied, sign p = 6.3e-05**. The margin
  GROWS with repo size (vs random 6× → 32× → 43×) because naive heuristics
  collapse as the candidate pool grows — `size` runs MRR 0.443 on neo's 99 files
  and 0.098 on m365dotnet's 2,882. The comparison is **conservative toward neo**:
  neo is scored on the ~30 files it would actually deliver while every baseline
  ranks the full set (99 / 480 / 2,882). **Footgun**: `random` must be seeded
  from `hashlib`, not `hash()` — the latter is salted per process and did not
  reproduce; and `recency` is mtime-based, so editing this harness (which lives
  in the repo it measures) moves its own score. Both are floor comparators; the
  deterministic ones reproduce byte-identically across runs. **The answer link is
  measured too**, in a constructed pre-fix repo (git-mined cases cannot show it —
  the fix is already at HEAD, so neo correctly proposes nothing): three planted
  bugs, each with a failing test defining "fixed", run with the buggy file in
  context vs `--exclude`d. **3/3 patches applied and passed with the file, 0/3
  without.** The mechanism is narrower than it looks and is recorded as such:
  arm B's fix LOGIC was right (for one bug byte-identical to the passing arm) and
  the patch was rejected because the surrounding context lines were hallucinated.
  Retrieval's measured contribution there is **groundedness, not reasoning** — on
  textbook defects priors supply the fix either way. **A second experiment closes
  that gap**: two conventions defined only in a scratch repo (a `ConfigError`
  whose `code` kwarg is required, a key rule stripping an `acme::` prefix), with
  the CONVENTION file excluded in arm B while the file to edit stays in context,
  so only project knowledge varies. **2/2 pass with the convention visible, 0/2
  without** — and the failure mode is REFUSAL, not a wrong guess: neo names the
  file it needs and says the convention "cannot be verified". Instructive
  contrast with the textbook case, where priors gave it false confidence and it
  patched against hallucinated context. **Footgun that invalidated the first
  run**: the tests defining the conventions lived IN the repo and were retrieved
  at score 2.52, so the no-context arm read the answer out of the answer key and
  "passed". A benchmark whose answer key is inside the corpus measures nothing —
  the tests now live outside the repository. **What this does NOT show**: how
  often real tasks turn on project-specific knowledge (n=2 is an existence proof,
  not a rate), and the learning loop delivering value in practice — on a live install that loop is starved, 2
  accepted outcomes in 208 episodes and both from drills.
- **Measuring retrieval changes: pin `PYTHONPATH`, and measure the REAL pipeline.**
  Two traps, both hit during the BM25 work, both producing confident wrong numbers.
  (1) The venv installs neo EDITABLE against `src/`, so running
  `.venv/bin/python -m neo.cli` from a git worktree of another commit executes THIS
  tree's code against that tree's files — a "baseline" run that is not the baseline.
  Every A/B needs `PYTHONPATH=<that tree>/src`, and `rank_mine_eval` now REFUSES to
  run unless the tree it was handed is the tree `import neo` actually resolves to —
  mandatory is not the same as effective, since the editable `.pth` silently catches
  a typo'd path and measures the working checkout twice. Measured on the superseded
  harness generation: it made main look like MRR 0.613 against a real figure of
  0.082 (the current harness puts main at 0.304 — different instrument, same trap). (2) Calling `score_candidate` directly
  measures the FIRST-PASS ranking only; `gather_context` then re-ranks with
  `pi_boost + hist_boost + _symbol_score` and applies an adaptive limit and a byte
  budget. A first-pass harness overstated R@10 by 0.14 against the real CLI. Validate
  any in-process replica against `--dry-run` output before trusting a sweep.
