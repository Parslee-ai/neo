# Pre-registration: does Neo's memory change answers? (`tools/ab_recall.py`)

Written 2026-10-24, before the first model call. Do not edit after a run;
append a dated "Deviations" section instead.

## Question
Do the lessons Neo hands over change the answer to a real prompt for the better?

## Tasks
12 prompts the user actually typed (Oct 13–24) on turns where the Claude Code
mod's recall fired (`recall_delivered` in `~/.neo/sessions/host_events.jsonl`,
joined to the session transcript). Only prompts of ≥12 words that stand alone
as questions, from repos that still exist. Fixed list, kept locally (see
Deviations).

## Arms (same model, same effort, same code context)
- **A**: `neo --dry-run --fast --no-memory` prompt (no memory).
- **B**: `neo --dry-run --fast` prompt (engine fact layer on).
- **C**: A's messages plus the text `neo recall` returns for the prompt, wrapped
  exactly as the mod's `recallInjection` wraps it (what Claude is handed).

All three prompts come from dry runs, so nothing is written to any fact store.
For each task, log whether B's file selection differs from A's (a confound to
report, not hide).

## Scoring
One judge call per task scores all three answers (order shuffled, seed 7),
1–10, on specificity to the repo, correctness and actionability. A second judge
call per task is shown the recall lesson text and asks, for each of A, B and C,
whether that answer applies the specific guidance of at least one lesson.

## Thresholds (decided now)
- **Memory changes outcomes**: mean paired diff (arm − A) ≥ +0.5, and that arm
  wins at least twice as many tasks as it loses.
- **Memory hurts**: mean paired diff ≤ −0.5.
- **Null**: anything else. Report a null as a null.
- **Lessons used**: lesson-applied rate in C minus rate in A (A is the control
  for "the model would have said it anyway"). ≥ 25 points = lessons get used.

n=12 is small; any result is directional, not proof.

## Deviations
- 2026-10-24, before any full-run result was read (only a 1-task smoke test on
  the wrong model, gpt-5.6): the quality judge is now given arm A's REPOSITORY
  CONTEXT so it can check codebase claims; failed dry runs and failed recalls
  (empty stdout) error the task instead of counting as "no lessons"; malformed
  judge fields error the task instead of scoring 0/False. Model fixed to the
  configured `gpt-6.1-sol`.
- 2026-10-24, after the run, publication only (no change to what was run):
  the 12 prompts and the results name private projects, so they moved out of
  this public repo to `~/.neo/eval/` (`ab_recall_tasks.json`,
  `ab_recall_results_2026-10-24.json`); `ab_recall.py` reads the task list
  from there (`--tasks`). Summary: B−A −0.75 (0/9/3), C−A ±0.00 (1/1/10),
  lesson applied A 12% / B 12% / C 62%, recall empty 4/12.
