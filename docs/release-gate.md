# The release gate

**Date:** 2026-08-12
**Plan of record:** [`docs/unified-store-plan.md`](unified-store-plan.md), Goal 10 (G5-inv)

A release cannot publish while any of C#, TypeScript or Python has a red LLM
round trip. Every pull request gets the free half of the same battery, with no
model call.

## Why a language axis

Three failures motivate this, and all three were invisible from the outside:

- **C# was absent from Neo's index for 8.5 months** (#158/#159). Everything
  that could have caught it was exercised against Python, and passed.
- **The gatherer selected gitignored junk** — 14 of 16 selected files on one
  measured prompt (#186).
- **Every failure printed success and exited 0.**

A single-language check would have stayed green through all of it. The
language axis is the assertion, not decoration. Matt Liotta, on the
2026-08-10 sync: real-LLM round-trip tests per language belong in the release
process — *"if that's what we have to do as part of the release process, we
want to do it because we obviously want to catch that shit."* Cost-bearing
model calls are approved **for releases**, not for every push.

## The two halves

| | Free half | Paid half |
|---|---|---|
| What | Guard-invariant battery | Per-language LLM round trip |
| File | `tests/test_selection_invariants.py` | `tests/test_release_roundtrip.py` |
| Marker | `-m invariants` | `-m roundtrip` |
| Model calls | none | one per language |
| Runs on | every push and PR | the release flow, and manual dispatch |
| Wired in | `invariants` job in `.github/workflows/ci.yml` | `language-roundtrip` job in `.github/workflows/publish.yml` |
| Blocks | the PR | the wheel — `build` needs it |

Both build the **same** three fixture repositories from
`tests/fixtures/language_repos.py`. That sharing is deliberate: a red release
gate has to be diagnosable from a free CI run, and it cannot be if the two
halves disagree about what the repository under test looks like.

## The fixtures

`build_fixture_repo(language, root)` generates a real git repository per
language — real, because the invariants are defined against `git
check-ignore` and a fake cannot be differentially compared to git. Each holds,
in its own language:

- a **target** file the prompt names by path, carrying a **sentinel** symbol
  that appears in no other file;
- **gitignored junk** in the same language, sharing the target's filename stem
  and containing its sentinel, so it competes for the same slots;
- a **duplicate** of the target under an agent-worktree layout
  (`.claude/worktrees/...`), which is *not* gitignored — the gatherer's own
  default exclusions are what must catch it;
- **bulk below the sentinel**, so the target exceeds the reasoning prompt's
  per-file character cap and the truncation-marker assertions are non-vacuous
  on the one file that matters. The sentinel sits at the top because the cut
  keeps the head.

Adding a language to `LANGUAGES` adds it to both halves at once. That is the
only ordering that cannot leave a language gated in name only.

## What the free half asserts

Per language, against `docs/unified-store-plan.md`'s guard invariants:

- **G1-inv** — no selected path is excluded by `git check-ignore` (asked of
  git, never re-derived); the planted junk is specifically absent; no path is
  selected twice; the worktree copy does not compete with the original; no two
  selected files are byte-identical.
- **G2-inv** — the prompt-named file is selected, and arrives whole rather
  than windowed.
- **G3-inv** — every cut file carries a marker and every uncut file carries
  none; the banner counts what was **sent**, not what was offered; the
  truncated-file count matches.

Several assertions guard the guard, because a battery that passes on no
evidence is worse than no battery: the planted junk is checked to be genuinely
gitignored, the target is checked to be genuinely large enough to be cut, and
the sentinel is checked to survive the cut.

## What the paid half adds

One `neo --json` subprocess per language — a subprocess, not an in-process
engine, because the thing under test is the release artifact end to end:
argument parsing, adapter resolution from the environment, gathering, the
model call, serialization.

The load-bearing assertion is that **the model names the sentinel**. The
sentinel exists in exactly one file, so it cannot be named unless that file
reached the prompt. That closes the gap the free half cannot: the battery
proves the gatherer *selected* the file; this proves the model *saw* it — the
gap C# sat inside for 8.5 months.

It also re-asserts the selection invariants on the commit being published.
"The battery passed on some other commit" is not evidence about this one.

Low confidence is **not** gated. A correct refusal is a legitimate answer and
must not block a release. Absent confidence is a different matter and does
fail — that is the serialization contract breaking.

### Reading a red round trip

A red per-language round trip reads as "that language broke", and for the
failure this gate exists to catch, that is right. Several other things can
turn it red, and the failure message names which, because sending someone to
the gatherer for a parser problem is an expensive wrong turn.

Measured during this gate's own bring-up: two consecutive local runs of the
same commit, one 21/21 green, the next red on Python with
`ValidationError: Failed to parse simulation traces: missing_start_sentinel`.
The gatherer had selected the right four files both times. The model's reply
simply did not carry neo's `<<<NEO:SCHEMA=v3:KIND=simulation>>>` sentinel.

**That is not swallowed and is not retried.** A user issuing the same command
gets the same error, so it is a genuine release blocker — and a gate that
retries until green is the same defect as a failure that exits 0. But it is
not a selection failure, so:

- `pytest -m invariants` will be **green** when the parser is the cause. That
  contrast is the fastest available diagnosis, and it costs nothing.
- The assertion message names the layer (`structured_parser.py`, the
  provider, the network, the credential) rather than leaving "neo failed".

Expect this to bite occasionally. The right response is to fix the brittleness
it exposes, not to loosen the gate.

## Running it

```bash
# Free half — every PR runs this; it costs nothing.
pytest -m invariants -v

# Paid half. The marker alone is not enough: the module skips unless the
# environment variable is set, so an ordinary `pytest` reports the round trips
# SKIPPED rather than silently omitting them.
NEO_RELEASE_ROUNDTRIP=1 pytest -m roundtrip -v
```

Locally the round trip uses whatever provider your config resolves. With
CarHost running and `inference_mode` left at `auto`, CAR routes it and no API
key is needed:

```bash
NEO_INFERENCE_MODE=auto NEO_RELEASE_ROUNDTRIP=1 pytest -m roundtrip -v
```

## Configuring CI

The `language-roundtrip` job needs one repository secret:

| Name | Kind | Required | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | secret | yes | the round trip's provider credential |
| `NEO_RELEASE_MODEL` | variable | no | overrides the model; defaults to `adapters.create_adapter`'s own Anthropic default |

**The job fails, loudly, when the secret is absent.** It does not skip. An
absent credential is a red gate, not a green one — a gate that no-ops when
unconfigured is the same class of defect as a failure that exits 0. Until the
secret is added, releases are blocked; that is the gate working.

## Dry-running the gate

`publish.yml` still carries `workflow_dispatch`, so the whole release path
including this gate can be exercised without cutting a release:

```bash
gh workflow run publish.yml --ref <branch>
gh run watch
```

`ci-check` and `language-roundtrip` run; `build` waits on both; nothing
publishes (`publish-pypi` is gated on `github.event_name == 'release'`). Note
that `publish-testpypi`'s upload step fails for an unrelated, pre-existing
reason — no Trusted Publisher is configured — which is documented in
`.claude/commands/ship-release.md` and in the workflow itself.

## Where this sits in the release flow

`/ship-release` Phase 6 creates the GitHub Release, which triggers
`publish.yml`. The gate runs there, before `build`, so the ordering is:

```
ci-check ─┐
          ├─▶ build ─▶ publish-pypi
roundtrip ─┘
```

A red language stops the chain at `build`. Nothing reaches PyPI.
