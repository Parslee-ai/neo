"""Bounded text for LM prompts, honest about what it dropped.

Every path that puts repository content, retrieved memory or a captured
failure into a prompt has to bound it. The bound is not the problem; a bound
that hides itself is. A bare slice produces text that simply stops, which the
model cannot distinguish from text that ended — so it reasons about absence
from a fragment.

Three shapes, because the right cut depends on where the information is:

- `truncate_marked` keeps the HEAD. Correct when the front is the substance:
  a source file, a problem statement, a formatted memory fact.
- `elide_middle` keeps BOTH ENDS. Correct for a traceback, where the most
  specific frame is at one end and the original cause at the other — a
  tail-cut of a 40-frame traceback keeps `File "..."` headers and discards
  `ValueError: database is locked`, which is the only line that matters.
- `shown_of` annotates an elided LIST. A list is truncated as silently as a
  string, and under a prompt that says "follow this exactly" three bullets
  read as the complete set.

Reach for the one that matches the input. Picking `truncate_marked` for a
traceback compiles, passes tests, and throws away the answer.

## On the ValueErrors

`truncate_marked` and `apportion` both reject a budget that cannot do the job.
No caller catches them, deliberately: every call site passes a module constant
against a bounded input count, so neither can fire without someone editing a
constant, and at that point a loud failure beats a silent one. The single path
where one is reachable at runtime — the deliberation panel — already treats
any failure as "fall back to the fast path", so a misconfigured constant is
logged and non-fatal rather than a traceback out of a reasoning call. That is
asserted in `TestMisconfiguredBudget`, not assumed.

## Marker dialects

This module does NOT own every truncation marker in the tree, and claiming
otherwise would be the same kind of confident-but-wrong statement the module
exists to prevent. Also live:

- `agent_context._read_doc` — `... [truncated]`, no counts
- `memory/outcomes` — `... (truncated)` on diff summaries

Those predate this seam and are not worth churn on their own. What matters is
that they are NOT nested inside these helpers: `outcomes` already truncates
`diff_summary` before `pattern_extraction` truncates it again, so a cut can
drop another cut's marker. When you add a truncation, check whether the value
arrives already cut.
"""

from __future__ import annotations

from typing import List, Optional

# The marker this module emits. `budget` bounds the CONTENT, not the returned
# string — the marker is appended on top, so a return value runs to roughly
# budget + 50 characters. Callers sizing a hard token ceiling must account for
# it; every current caller is sizing a soft prompt section and does not.
MARKER_TEMPLATE = "\n... [truncated: {dropped} of {total} characters not shown]"

# What a marker looks like regardless of the numbers in it. Tests and callers
# that need to detect a cut in already-rendered text match on this.
MARKER_PREFIX = "\n... [truncated:"

_ELIDE_TEMPLATE = "\n... [{elided} characters elided] ...\n"
# Percent of the usable budget kept from the front when no head is given.
# 40% preserves the ratio the traceback caller has always used (1600 of 4000)
# while scaling down instead of swallowing a smaller budget whole.
_ELIDE_HEAD_SHARE = 40


def truncate_marked(text: Optional[str], budget: int) -> str:
    """Return `text` with its head kept to `budget` characters, marking a cut.

    Returns the text unchanged when it fits, so the marker's presence always
    means a cut actually happened and never becomes noise the model learns to
    ignore. `None` is treated as empty.

    Raises ValueError for a non-positive budget on non-empty text: every
    caller passes a module constant, so that is a programmer error, and the
    alternatives are both bad — returning "" hides the cut, returning a bare
    marker emits 50 characters from a budget of zero.
    """
    content = text or ""
    if not content:
        # Empty at ANY budget, including negative: nothing was dropped, so
        # nothing may claim to have been. Checked before the budget guard —
        # `budget >= len(content)` is `-1 >= 0`, which is False, and fell
        # through to emit "0 of 0 characters not shown" from the one function
        # whose contract is that a marker always means a real cut.
        return ""
    if budget >= len(content):
        return content
    if budget <= 0:
        raise ValueError(
            f"truncate_marked: budget must be positive to cut {len(content)} "
            f"characters, got {budget}"
        )
    return content[:budget] + MARKER_TEMPLATE.format(
        dropped=len(content) - budget, total=len(content)
    )


def elide_middle(text: str, budget: int) -> str:
    """Trim `text` to `budget` by removing the MIDDLE, marking what was cut.

    Both ends of a traceback carry information, and which end carries *what*
    depends on its shape — which is why neither a head-cut nor a tail-cut is
    safe for the general case:

    - Plain traceback: the frames run top-down and the exception line is
      LAST. `ValueError: database is locked` is the final line, so a head-cut
      of a 40-frame trace keeps forty `File "..."` headers and discards the
      only line naming the failure.
    - Chained (`raise X from Y`): the render puts the ORIGINAL cause near the
      front and the outer exception last. A tail-cut keeps the outer
      exception — which is already recorded separately in `error_type` and
      `error_message` — and discards the cause, the one new fact.

    Elide the middle and both shapes keep what they needed. The head share is
    a fraction of the budget rather than a fixed count, because any fixed
    count exceeding the budget collapses into a head-only cut and reinstates
    the first failure above.

    The marker is not decoration: a silently truncated traceback presented as
    a whole one is the same class of defect this evidence exists to fix.
    """
    if len(text) <= budget:
        return text
    usable = max(budget - len(_ELIDE_TEMPLATE.format(elided=len(text))), 0)
    if usable <= 0:
        return _ELIDE_TEMPLATE.format(elided=len(text)).strip()
    kept_head = usable * _ELIDE_HEAD_SHARE // 100
    kept_tail = usable - kept_head
    elided = len(text) - kept_head - kept_tail
    return (
        text[:kept_head]
        + _ELIDE_TEMPLATE.format(elided=elided)
        + (text[-kept_tail:] if kept_tail else "")
    )


def shown_of(items: List, shown: int, tail: bool = False) -> str:
    """Annotate an elided list with how much of it is being shown.

    Empty string when nothing was elided, so the annotation always means an
    omission — the same contract `truncate_marked` holds for text. Returned
    with a leading space so it can sit on a header line.

    `tail=True` for a `[-n:]` slice. Which END survived is part of the
    meaning, not decoration: "Recent attempts [showing 3 of 20]" reads as a
    sample of twenty, where "[showing last 3 of 20]" says the loop has run
    twenty times and you are seeing where it is now. Only one of the seven
    call sites is a tail cut, which is exactly why the default has to be the
    head form.
    """
    if len(items) <= shown:
        return ""
    where = "last " if tail else ""
    return f" [showing {where}{shown} of {len(items)}]"


def apportion(sizes: dict, budget: int) -> dict:
    """Split `budget` across named sections by max-min fair share.

    Sections shorter than an equal share are funded in full and their unused
    capacity is redistributed to the ones that can use it, repeatedly. This
    is what a flat per-section cap gets wrong: measured on a live store, the
    deliberation context had one 33,144-character section and one 30-character
    section against a 6,000 budget, and a flat 2,000-per-section cap sent
    2,030 — leaving 66% of the budget unspent while discarding 94% of the
    memory it was meant to carry.

    Order-independent by construction, which matters because the alternative
    (cut each, concatenate, cut again) drops whichever section happens to be
    last, and drops its marker with it.

    Raises ValueError when the budget cannot seat every section with at least
    one character. Returning zeros instead would push the caller into either
    dropping sections silently — the exact defect this module exists to
    prevent, reintroduced as a guard clause — or handing a zero budget to
    `truncate_marked`, which rejects it for the same reason. Every caller
    passes a module constant against a bounded section count, so this is a
    programmer error and belongs to whoever wrote the constant.
    """
    pending = {name: size for name, size in sizes.items() if size > 0}
    if not pending:
        return {}
    if budget < len(pending):
        raise ValueError(
            f"apportion: budget {budget} cannot seat {len(pending)} sections "
            f"with at least one character each"
        )

    remaining = budget
    allocation: dict = {}
    while pending:
        share = remaining // len(pending)
        satisfied = {n: s for n, s in pending.items() if s <= share}
        if not satisfied:
            # Everyone wants more than their share — split it evenly and stop.
            allocation.update({name: share for name in pending})
            break
        allocation.update(satisfied)
        remaining -= sum(satisfied.values())
        for name in satisfied:
            pending.pop(name)
    return allocation
