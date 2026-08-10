"""`--dry-run`: show what would actually be sent to the model.

The old implementation exited in `cli.main` **before the engine was
constructed**, so it could only ever print the file-gathering result. That is
not what it claimed. `CLAUDE.md` told operators the flag "assembles the full
context (file selection, fact retrieval, constraints, four-layer assembly) and
prints what *would* be sent to the LM"; three of those four never ran. The
Execution Envelope reaches the model through seven prompt builders, retrieved
facts drive the whole memory system, and the REPOSITORY CONTEXT block carries
the truncation markers that exist specifically so a cut is visible -- and none
of it was inspectable through the tool built for inspecting it.

That mattered more than an ordinary documentation slip, because `--dry-run` is
the instrument the project tells people to use *before believing any claim
about what Neo saw*. An instrument that under-reports sends the operator
looking in the wrong place, which is the same failure as a cap that blames the
wrong knob.

**The prompt is recorded, never reconstructed.** `RecordingLM` is a real
`LMAdapter` installed in the engine's own `self.lm` slot, so what gets printed
is the exact `messages` list the provider adapter would have received. A
renderer that walked the context dict and re-assembled the prompt would be a
second implementation of the prompt builders, free to drift from the seven
real ones the moment any of them changed -- the same duplicated-rule shape
that put `EXCLUDED_DIR_NAMES` in two places and cost a release.

`DryRunComplete` derives from `BaseException`, not `Exception`, and that is
load-bearing: `NeoEngine._process_guarded` wraps the run in `except Exception`
and converts anything it catches into a `FAILED` lifecycle event. A dry run is
not a failure, and reporting one as a crash would be a third way of lying
about what happened. `finally` blocks still run, so the overseer stops and the
engine lock releases exactly as on any other path.
"""

from typing import Any, Optional

from neo.models import LMAdapter


class DryRunComplete(BaseException):
    """Control-flow signal: assembly finished, stop before inference.

    `BaseException` on purpose -- see the module docstring. It carries the
    recorded calls so the caller does not have to reach back into the engine.
    """

    def __init__(self, calls: list[dict[str, Any]]):
        super().__init__("dry run complete")
        self.calls = calls


class RecordingLM(LMAdapter):
    """An `LMAdapter` that captures one call's messages and stops the run.

    Raising on the FIRST call is deliberate. Letting the run continue would
    mean either returning a stub -- which the parser downstream would reject,
    producing a repair loop and a second call -- or making real requests,
    which is the one thing `--dry-run` promises not to do. One call is also
    all that is needed: every builder assembles from the same enriched
    context, so the first prompt is the one that answers "what did Neo see".
    """

    #: Advertised so `_decide_reasoning_mode` and any capability probe see a
    #: plausible model rather than an empty string.
    model = "dry-run"
    provider = "dry-run"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def name(self) -> str:
        return "dry-run"

    def generate(
        self,
        messages: list[dict[str, str]],
        stop: Optional[list[str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        self.calls.append({
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "stop": stop,
        })
        raise DryRunComplete(self.calls)


def render(calls: list[dict[str, Any]], gathered: Any = None) -> str:
    """Format recorded calls for the console.

    Prints every message role in full. There is no truncation here on purpose:
    this output exists so an operator can see what the model got, and a viewer
    that silently cut its own output would reproduce, one level up, the exact
    defect the `text_budget` markers were introduced to fix.
    """
    out: list[str] = []
    if not calls:
        out.append(
            "No LM call was assembled. The run ended before reaching the "
            "model -- usually a budget skip or an early return, not an empty "
            "prompt."
        )
        return "\n".join(out)

    call = calls[0]
    total = sum(len(m.get("content") or "") for m in call["messages"])
    out.append("=== DRY RUN: the exact prompt that would be sent ===")
    out.append("")
    out.append(
        f"model params: max_tokens={call['max_tokens']} "
        f"temperature={call['temperature']} "
        f"reasoning_effort={call['reasoning_effort']}"
    )
    out.append(f"{len(call['messages'])} message(s), {total:,} chars total")
    out.append("")

    for i, message in enumerate(call["messages"], 1):
        content = message.get("content") or ""
        role = message.get("role", "?")
        out.append(f"--- message {i}/{len(call['messages'])}: {role} "
                   f"({len(content):,} chars) ---")
        out.append(content)
        out.append("")

    if len(calls) > 1:  # pragma: no cover - RecordingLM stops at the first
        out.append(f"({len(calls) - 1} further call(s) recorded)")
    return "\n".join(out)
