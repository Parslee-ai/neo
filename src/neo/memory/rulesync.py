"""
Rule-file sync diagnostic — flag drift between AGENTS.md / CLAUDE.md / GEMINI.md.

When a repo is worked by more than one coding agent, each tool reads its own
rule file (`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`). Teams routinely update one
and forget the others, so the agents drift out of sync — one is told "use
pytest", another never hears it; one says "tabs", another "spaces". This is a
static cross-file analysis (no transcripts) that reports two divergence kinds:

- **gap**: a rule present in one file with no equivalent in another.
- **conflict**: two files whose rules on the same topic contradict (LM-judged).

Flag + propose only — never writes files (neo stays read-only). Reuses the same
embedding infra as the rest of memory; the LM conflict judge mirrors the
`issues.suggest_rules` pattern (resolve_adapter + _parse_json), is bounded, and
degrades gracefully (skipped, not fatal) when no adapter is available.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from neo.math_utils import batched_cosine

logger = logging.getLogger(__name__)

# Canonical tool name per rule-file basename.
_RULE_FILENAMES = {
    "AGENTS.md": "agents",
    "CLAUDE.md": "claude",
    "GEMINI.md": "gemini",
}

# Two units "say the same thing" above this cosine — used for gap alignment.
ALIGN_THRESHOLD = 0.78
# Above this, the two units are near-identical wording: in sync, never a conflict.
IDENTICAL_THRESHOLD = 0.97
# Cap LM conflict-judge calls per run (bounded cost; degrades to fewer checks).
_MAX_CONFLICT_CHECKS = 40

# Lines that are structure/prose, not rules. Headings, fences, blanks, links-only.
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_FENCE_RE = re.compile(r"^\s*```")


@dataclass
class RuleFile:
    tool: str            # "agents" | "claude" | "gemini"
    path: str
    units: list[str] = field(default_factory=list)
    embeddings: list = field(default_factory=list)  # aligned to units


@dataclass
class RuleGap:
    rule: str
    present_in: list[str]
    missing_from: list[str]

    @property
    def suggestion(self) -> str:
        files = ", ".join(f"{t.upper()}.md" for t in self.missing_from)
        return f"Add to {files}: {self.rule}"


@dataclass
class RuleConflict:
    tool_a: str
    text_a: str
    tool_b: str
    text_b: str
    explanation: str = ""

    @property
    def suggestion(self) -> str:
        return (
            f"Reconcile: {self.tool_a.upper()}.md says {self.text_a!r}; "
            f"{self.tool_b.upper()}.md says {self.text_b!r}. Pick one and update the other."
        )


@dataclass
class SyncReport:
    files: list[RuleFile] = field(default_factory=list)
    gaps: list[RuleGap] = field(default_factory=list)
    conflicts: list[RuleConflict] = field(default_factory=list)
    note: str = ""  # e.g. "single source", "all files identical"

    @property
    def in_sync(self) -> bool:
        return not self.gaps and not self.conflicts


def discover_rule_files(root: str) -> list[tuple[str, Path]]:
    """Find rule files at the repo root, case-insensitively. (tool, path) pairs."""
    base = Path(root)
    if not base.is_dir():
        return []
    by_lower = {fn.lower(): (fn, tool) for fn, tool in _RULE_FILENAMES.items()}
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()
    try:
        entries = sorted(base.iterdir(), key=lambda p: p.name)
    except OSError:
        return []
    for entry in entries:
        if not entry.is_file():
            continue
        match = by_lower.get(entry.name.lower())
        if match and match[1] not in seen:
            found.append((match[1], entry))
            seen.add(match[1])
    return found


def parse_units(text: str) -> list[str]:
    """Split a rule-file's markdown into normalized rule units.

    A unit is a bullet item plus any following *more-indented, non-bullet*
    continuation lines (wrapped text or sub-detail) folded in — so a wrapped
    bullet does not fragment into meaningless standalone lines. Headings, code
    fences (and their contents), blank lines, and top-level non-bullet prose are
    dropped: these files encode rules as bullets, and treating prose as rules is
    pure noise. Blank lines and headings flush the current unit.
    """
    units: list[str] = []
    in_fence = False
    current: Optional[str] = None
    current_indent = 0

    def flush() -> None:
        nonlocal current
        if current is not None:
            u = _normalize_unit(current)
            if len(u) >= 8:  # skip trivial fragments
                units.append(u)
        current = None

    for raw in (text or "").splitlines():
        if _FENCE_RE.match(raw):
            in_fence = not in_fence
            flush()
            continue
        if in_fence:
            continue
        if not raw.strip() or _HEADING_RE.match(raw):
            flush()
            continue
        indent = len(raw) - len(raw.lstrip())
        m = _BULLET_RE.match(raw)
        if m:
            flush()
            current = m.group(1).strip()
            current_indent = indent
        elif current is not None and indent > current_indent:
            current += " " + raw.strip()  # wrapped continuation of the bullet
        else:
            flush()  # top-level non-bullet prose: not a rule
    flush()
    return units


def _normalize_unit(s: str) -> str:
    s = re.sub(r"`([^`]*)`", r"\1", s)         # strip inline code backticks
    s = re.sub(r"\*\*([^*]*)\*\*", r"\1", s)   # strip bold
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)  # link -> text
    return re.sub(r"\s+", " ", s).strip()


def _best_sim(vec, others: list) -> float:
    """Best cosine of vec against a list of vectors (0.0 if none usable)."""
    if vec is None or not others:
        return 0.0
    sims = batched_cosine(others, vec, default=0.0)
    return max(sims) if sims else 0.0


def analyze_rule_sync(
    files: list[RuleFile],
    *,
    lm_adapter=None,
) -> SyncReport:
    """Compute gaps and (optionally LM-judged) conflicts across rule files."""
    report = SyncReport(files=files)
    present = [f for f in files if f.units]
    if len(present) < 2:
        report.note = "single rule file — nothing to cross-check" if present else "no rule files"
        return report

    # --- Gaps: a unit in one file with no aligned match in another file. ------
    # Dedup so the same rule (which may appear in several files) is reported once.
    seen_gap: set[str] = set()
    conflict_candidates: list[tuple[RuleFile, str, object, RuleFile, str, object]] = []
    for fa in present:
        for ui, unit in enumerate(fa.units):
            vec = fa.embeddings[ui] if ui < len(fa.embeddings) else None
            missing_from: list[str] = []
            for fb in present:
                if fb.tool == fa.tool:
                    continue
                best = _best_sim(vec, fb.embeddings)
                if best < ALIGN_THRESHOLD:
                    missing_from.append(fb.tool)
                elif best < IDENTICAL_THRESHOLD:
                    # aligned but not identical -> candidate contradiction
                    bj = _argmax_sim(vec, fb.embeddings)
                    if bj is not None and fa.tool < fb.tool:  # one direction only
                        conflict_candidates.append(
                            (fa, unit, vec, fb, fb.units[bj], fb.embeddings[bj])
                        )
            if missing_from:
                key = unit.lower()
                if key not in seen_gap:
                    seen_gap.add(key)
                    present_in = [fa.tool] + [
                        fb.tool for fb in present
                        if fb.tool != fa.tool and _best_sim(vec, fb.embeddings) >= ALIGN_THRESHOLD
                    ]
                    report.gaps.append(
                        RuleGap(rule=unit, present_in=present_in, missing_from=missing_from)
                    )

    # --- Conflicts: LM-judge the aligned-but-divergent candidate pairs. -------
    if lm_adapter is not None and conflict_candidates:
        seen_conf: set[tuple[str, str]] = set()
        for fa, ta, _va, fb, tb, _vb in conflict_candidates[:_MAX_CONFLICT_CHECKS]:
            sig = tuple(sorted((ta.lower(), tb.lower())))
            if sig in seen_conf:
                continue
            seen_conf.add(sig)
            verdict = _judge_conflict(ta, tb, lm_adapter)
            if verdict and verdict.get("conflict"):
                report.conflicts.append(
                    RuleConflict(
                        tool_a=fa.tool, text_a=ta,
                        tool_b=fb.tool, text_b=tb,
                        explanation=str(verdict.get("explanation") or "").strip(),
                    )
                )
    return report


def _argmax_sim(vec, others: list) -> Optional[int]:
    if vec is None or not others:
        return None
    sims = batched_cosine(others, vec, default=0.0)
    if not sims:
        return None
    best_i = max(range(len(sims)), key=lambda i: sims[i])
    return best_i


_CONFLICT_PROMPT = """Two rule files for the same codebase each state a rule on \
what appears to be the same topic. Decide whether they CONTRADICT (prescribe \
incompatible things) or are merely worded differently / compatible.

Rule A: <<A>>
Rule B: <<B>>

Respond with JSON only: {"conflict": true|false, "explanation": "<one sentence>"}"""


def _judge_conflict(text_a: str, text_b: str, lm_adapter) -> Optional[dict]:
    from neo.memory.transcript import _parse_json

    prompt = _CONFLICT_PROMPT.replace("<<A>>", text_a).replace("<<B>>", text_b)
    try:
        out = lm_adapter.generate(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
    except Exception as e:
        logger.warning("rulesync: conflict-judge LM call failed: %s", e)
        return None
    data = _parse_json(out)
    return data if isinstance(data, dict) else None


def find_rule_sync(
    store,
    *,
    root: Optional[str] = None,
    check_conflicts: bool = True,
    lm_adapter=None,
) -> SyncReport:
    """Discover rule files under ``root``, embed their units, and analyze sync.

    Read-only. ``lm_adapter`` (for conflict judging) is used only when
    ``check_conflicts`` is True; absent it, gaps are still reported.
    """
    root = root or getattr(store, "codebase_root", None) or "."
    discovered = discover_rule_files(root)

    files: list[RuleFile] = []
    contents: dict[str, str] = {}
    for tool, path in discovered:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        contents[tool] = text
        files.append(RuleFile(tool=tool, path=str(path), units=parse_units(text)))

    # Identical content (e.g. a symlinked single source) -> explicitly in sync.
    if len(contents) >= 2 and len(set(contents.values())) == 1:
        rep = SyncReport(files=files, note="all rule files are identical — in sync")
        return rep

    embed: Optional[Callable] = getattr(store, "_embed_text", None)
    if embed is not None:
        for f in files:
            f.embeddings = [_safe_embed(embed, u) for u in f.units]

    return analyze_rule_sync(
        files, lm_adapter=lm_adapter if check_conflicts else None
    )


def _safe_embed(embed: Callable, text: str):
    try:
        return embed(text)
    except Exception:
        return None
