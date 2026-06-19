"""Tests for the conversation-mined issue diagnostic (neo.memory.issues).

Model-free and deterministic: a fake embedder maps text to fixed unit vectors
and episodes are hand-built, so no Jina model loads and clustering is exact.
"""

import numpy as np

from neo.memory.issues import (
    CATEGORY_ABSENT_GUARDRAIL,
    CATEGORY_MISSING_TOOL,
    CATEGORY_VAGUE_RULE,
    Issue,
    IssueEvidence,
    detect_issues,
    find_issues,
    tag_signals,
)
from neo.memory.models import FactScope
from neo.memory.transcript import Episode

NOW = 1_000_000.0
RECENT = str(NOW - 100)
STALE = str(NOW - 40 * 86400)

_TESTS = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_LINT = np.array([0.0, 1.0, 0.0], dtype=np.float32)
_OTHER = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def fake_embed(text):
    """Keyword -> fixed vector, so 'same topic' clusters exactly."""
    t = (text or "").lower()
    if "pytest" in t or "unittest" in t or "test framework" in t:
        return _TESTS
    if "ruff" in t or "import" in t or "e402" in t:
        return _LINT
    return _OTHER


class _StaticSource:
    name = "test"
    scope = FactScope.PROJECT

    def __init__(self, episodes):
        self._episodes = list(episodes)

    def collect_episodes(self):
        return list(self._episodes)


class _FakeStore:
    codebase_root = "/tmp/repo"

    def _embed_text(self, text):
        return fake_embed(text)


def _ep(session, ask, *, errors=None, assistant=None, ts=RECENT):
    return Episode(
        session_id=session,
        anchor_uuid=session + "-a",
        last_uuid=session + "-b",
        timestamp=ts,
        ask=ask,
        assistant_text=list(assistant or []),
        errors=list(errors or []),
    )


def _signals(episodes):
    return [tag_signals(e) for e in episodes]


def _embeds(episodes):
    return [fake_embed(e.ask) for e in episodes]


# --------------------------------------------------------------------------
# tag_signals
# --------------------------------------------------------------------------


def test_tag_signals_detects_tool_error():
    ep = _ep("s1", "run ruff", errors=["E402 module level import not at top of file"])
    sig = tag_signals(ep)
    assert sig.has_tool_error is True
    assert sig.error_class and "import" in sig.error_class
    assert sig.asst_asked_clarification is False
    assert sig.has_friction is True


def test_tag_signals_detects_clarification():
    ep = _ep("s1", "fix the tests", assistant=["Sure — could you clarify which file you mean?"])
    sig = tag_signals(ep)
    assert sig.asst_asked_clarification is True
    assert sig.has_tool_error is False
    assert sig.has_friction is True


def test_tag_signals_no_friction():
    ep = _ep("s1", "add a function", assistant=["Done."])
    sig = tag_signals(ep)
    assert sig.has_friction is False


def test_tool_use_error_envelope_is_not_friction():
    # Claude Code's own tool-protocol guards are harness churn, not project issues.
    ep = _ep(
        "s1", "edit file",
        errors=["<tool_use_error>File has not been read yet. Read it first.</tool_use_error>"],
    )
    sig = tag_signals(ep)
    assert sig.has_tool_error is False
    assert sig.has_friction is False


def test_bare_exit_code_banner_is_not_friction():
    for banner in ("Exit code 2", "Process exited with code 1", "Command timed out"):
        sig = tag_signals(_ep("s1", "run it", errors=[banner]))
        assert sig.has_tool_error is False, banner


def test_non_error_command_output_is_not_friction():
    # A non-zero exit whose captured output is not an error message (a section
    # header, a diff) must not register as friction.
    ep = _ep("s1", "show deltas", errors=["=== engine.rs 540-566 (state deltas) ==="])
    sig = tag_signals(ep)
    assert sig.has_tool_error is False
    assert sig.has_friction is False


def test_code_identifier_containing_error_is_not_friction():
    # A Rust type like WorkflowError in a signature is code, not an error.
    ep = _ep("s1", "edit fn", errors=[") -> Result<StepOutcome, WorkflowError> {"])
    sig = tag_signals(ep)
    assert sig.has_tool_error is False
    assert sig.has_friction is False


def test_error_like_line_found_past_output_noise():
    # Real error buried under non-error output lines is still found.
    blob = "=== build log ===\nrunning step 3\nfatal: not a git repository"
    sig = tag_signals(_ep("s1", "build", errors=[blob]))
    assert sig.has_tool_error is True
    assert sig.error_class and "git repository" in sig.error_class


def test_real_git_merge_error_survives_filtering():
    # The genuine recurring finding from the cross-project probe must survive.
    line = "error: Your local changes to the following files would be overwritten by merge:"
    sig = tag_signals(_ep("s1", "git merge", errors=[line]))
    assert sig.has_tool_error is True
    assert sig.error_class and "overwritten by merge" in sig.error_class


def test_banner_then_substantive_line_keeps_substantive_class():
    # Codex-style: banner first line, real error after -> use the real line.
    ep = _ep("s1", "run", errors=["Process exited with code 1\nModuleNotFoundError: foo"])
    sig = tag_signals(ep)
    assert sig.has_tool_error is True
    assert sig.error_class and "modulenotfounderror" in sig.error_class


def test_harness_noise_clusters_produce_no_issues():
    eps = [
        _ep("s1", "edit a", errors=["<tool_use_error>File has not been read yet.</tool_use_error>"]),
        _ep("s2", "edit b", errors=["<tool_use_error>File has not been read yet.</tool_use_error>"]),
        _ep("s3", "edit c", errors=["<tool_use_error>File has not been read yet.</tool_use_error>"]),
    ]
    issues = detect_issues(eps, _signals(eps), _embeds(eps), min_cluster=3, now=NOW)
    assert issues == []


def test_normalize_error_groups_volatile_variants():
    a = tag_signals(_ep("s1", "x", errors=["File /a/b/c.py line 42: E402 import not at top"]))
    b = tag_signals(_ep("s2", "x", errors=["File /q/r.py line 9: E402 import not at top"]))
    assert a.error_class == b.error_class  # paths + line numbers stripped


# --------------------------------------------------------------------------
# detect_issues — gate
# --------------------------------------------------------------------------


def test_gate_rejects_small_cluster():
    eps = [
        _ep("s1", "use pytest?", assistant=["could you clarify?"]),
        _ep("s2", "pytest framework?", assistant=["could you clarify?"]),
    ]
    issues = detect_issues(eps, _signals(eps), _embeds(eps), min_cluster=3, now=NOW)
    assert issues == []


def test_gate_requires_two_distinct_sessions():
    eps = [
        _ep("s1", "use pytest?", assistant=["could you clarify?"]),
        _ep("s1", "pytest framework?", assistant=["could you clarify?"]),
        _ep("s1", "test framework choice?", assistant=["could you clarify?"]),
    ]
    issues = detect_issues(eps, _signals(eps), _embeds(eps), min_cluster=3, now=NOW)
    assert issues == []


def test_gate_requires_friction():
    # Recurring topic across sessions but no errors / no clarification -> not an issue.
    eps = [
        _ep("s1", "use pytest", assistant=["Done."]),
        _ep("s2", "pytest setup", assistant=["Done."]),
        _ep("s3", "test framework pytest", assistant=["Done."]),
    ]
    issues = detect_issues(eps, _signals(eps), _embeds(eps), min_cluster=3, now=NOW)
    assert issues == []


# --------------------------------------------------------------------------
# detect_issues — emission + categories
# --------------------------------------------------------------------------


def test_emits_vague_rule_for_clarification_cluster():
    eps = [
        _ep("s1", "use pytest?", assistant=["Could you clarify which test framework?"]),
        _ep("s2", "test framework?", assistant=["I'm not sure — please specify the framework."]),
        _ep("s3", "pytest or not?", assistant=["Could you clarify the framework you want?"]),
    ]
    issues = detect_issues(eps, _signals(eps), _embeds(eps), min_cluster=3, now=NOW)
    assert len(issues) == 1
    assert issues[0].category == CATEGORY_VAGUE_RULE
    assert issues[0].session_count == 3
    assert issues[0].member_count == 3


def test_emits_missing_tool_for_command_not_found():
    eps = [
        _ep("s1", "run ruff lint", errors=["ruff: command not found"]),
        _ep("s2", "ruff check imports", errors=["ruff: command not found"]),
        _ep("s3", "lint with ruff", errors=["ruff: command not found"]),
    ]
    issues = detect_issues(eps, _signals(eps), _embeds(eps), min_cluster=3, now=NOW)
    assert len(issues) == 1
    assert issues[0].category == CATEGORY_MISSING_TOOL


def test_emits_absent_guardrail_for_recurring_error():
    eps = [
        _ep("s1", "fix ruff import", errors=["E402 module level import not at top"]),
        _ep("s2", "ruff import order", errors=["E402 module level import not at top"]),
        _ep("s3", "imports failing ruff", errors=["E402 module level import not at top"]),
    ]
    issues = detect_issues(eps, _signals(eps), _embeds(eps), min_cluster=3, now=NOW)
    assert len(issues) == 1
    assert issues[0].category == CATEGORY_ABSENT_GUARDRAIL


def test_no_such_file_is_absent_guardrail_not_missing_tool():
    # "No such file" is a missing path, not a missing tool -> absent-guardrail.
    eps = [
        _ep("s1", "search lib", errors=["ugrep: src/lib.rs: No such file or directory"]),
        _ep("s2", "search lib", errors=["ugrep: src/lib.rs: No such file or directory"]),
        _ep("s3", "search lib", errors=["ugrep: src/lib.rs: No such file or directory"]),
    ]
    issues = detect_issues(eps, _signals(eps), _embeds(eps), min_cluster=3, now=NOW)
    assert len(issues) == 1
    assert issues[0].category == CATEGORY_ABSENT_GUARDRAIL


def test_category_precedence_tool_error_beats_clarification():
    # Mixed cluster: tool error must win over clarification (most structural).
    eps = [
        _ep("s1", "run ruff", errors=["ruff: command not found"], assistant=["could you clarify?"]),
        _ep("s2", "ruff lint", errors=["ruff: command not found"]),
        _ep("s3", "ruff imports", assistant=["could you clarify?"]),
    ]
    issues = detect_issues(eps, _signals(eps), _embeds(eps), min_cluster=3, now=NOW)
    assert len(issues) == 1
    assert issues[0].category == CATEGORY_MISSING_TOOL


# --------------------------------------------------------------------------
# evidence + scoring
# --------------------------------------------------------------------------


def test_evidence_spans_are_verbatim():
    eps = [
        _ep("s1", "run ruff", errors=["ruff: command not found"]),
        _ep("s2", "ruff lint", errors=["ruff: command not found"]),
        _ep("s3", "ruff imports", errors=["ruff: command not found"]),
    ]
    issues = detect_issues(eps, _signals(eps), _embeds(eps), min_cluster=3, now=NOW)
    assert issues
    for ev in issues[0].evidence:
        assert isinstance(ev, IssueEvidence)
        # span must be a verbatim substring of the source error
        src_ep = next(e for e in eps if e.session_id == ev.session_id)
        assert ev.span in src_ep.errors[0]
    assert len(issues[0].evidence) <= 3


def test_clarification_evidence_spans_are_verbatim():
    eps = [
        _ep("s1", "use pytest?", assistant=["First line.", "Could you clarify the test framework?"]),
        _ep("s2", "test framework?", assistant=["Please specify which framework to use."]),
        _ep("s3", "pytest?", assistant=["I'm not sure — could you clarify which one?"]),
    ]
    issues = detect_issues(eps, _signals(eps), _embeds(eps), min_cluster=3, now=NOW)
    assert len(issues) == 1
    for ev in issues[0].evidence:
        src_ep = next(e for e in eps if e.session_id == ev.session_id)
        # span must be verbatim within a single assistant turn, not a join.
        assert any(ev.span in turn for turn in src_ep.assistant_text)


def test_empty_string_errors_do_not_satisfy_friction_gate():
    # Three similar asks, but their only "error" is an empty envelope -> no friction.
    eps = [
        _ep("s1", "run ruff", errors=[""]),
        _ep("s2", "ruff lint", errors=["   "]),
        _ep("s3", "ruff imports", errors=[""]),
    ]
    sigs = _signals(eps)
    assert all(not s.has_friction for s in sigs)
    assert detect_issues(eps, sigs, _embeds(eps), min_cluster=3, now=NOW) == []


def test_confidence_ordering_recent_large_first():
    big_recent = [
        _ep(f"s{i}", "ruff import error", errors=["E402 import not at top"], ts=RECENT)
        for i in range(4)
    ]
    small_stale = [
        _ep("c1", "use pytest?", assistant=["could you clarify?"], ts=STALE),
        _ep("c2", "pytest framework?", assistant=["could you clarify?"], ts=STALE),
        _ep("c3", "test framework pytest?", assistant=["could you clarify?"], ts=STALE),
    ]
    eps = big_recent + small_stale
    issues = detect_issues(eps, _signals(eps), _embeds(eps), min_cluster=3, now=NOW)
    assert len(issues) == 2
    assert issues[0].confidence >= issues[1].confidence
    assert issues[0].category == CATEGORY_ABSENT_GUARDRAIL  # the big recent cluster


# --------------------------------------------------------------------------
# find_issues orchestrator
# --------------------------------------------------------------------------


def test_find_issues_end_to_end():
    eps = [
        _ep("s1", "use pytest?", assistant=["Could you clarify the test framework?"]),
        _ep("s2", "test framework?", assistant=["Please specify which framework."]),
        _ep("s3", "pytest or unittest?", assistant=["Could you clarify which you want?"]),
    ]
    issues = find_issues(_FakeStore(), sources=[_StaticSource(eps)], now=NOW)
    assert len(issues) == 1
    assert issues[0].category == CATEGORY_VAGUE_RULE


def test_find_issues_since_filter_excludes_old():
    eps = [
        _ep("s1", "use pytest?", assistant=["could you clarify?"], ts=STALE),
        _ep("s2", "test framework?", assistant=["could you clarify?"], ts=STALE),
        _ep("s3", "pytest?", assistant=["could you clarify?"], ts=STALE),
    ]
    # 7-day window excludes the 40-day-old episodes -> nothing to cluster.
    issues = find_issues(
        _FakeStore(), sources=[_StaticSource(eps)], since_seconds=7 * 86400, now=NOW
    )
    assert issues == []


def test_find_issues_is_read_only_never_invokes_ingester(monkeypatch):
    """find_issues must not go through the fact-admission / watermark path."""
    import neo.memory.transcript as tr

    def _boom(*args, **kwargs):
        raise AssertionError("find_issues must not construct the TranscriptIngester")

    monkeypatch.setattr(tr, "TranscriptIngester", _boom)

    eps = [
        _ep("s1", "run ruff", errors=["ruff: command not found"]),
        _ep("s2", "ruff lint", errors=["ruff: command not found"]),
        _ep("s3", "ruff imports", errors=["ruff: command not found"]),
    ]
    issues = find_issues(_FakeStore(), sources=[_StaticSource(eps)], now=NOW)
    assert len(issues) == 1  # ran to completion without touching the ingester


def test_issue_dataclass_shape():
    iss = Issue(
        title="t", category=CATEGORY_VAGUE_RULE, confidence=0.5,
        evidence=[], session_count=2, member_count=3,
    )
    assert iss.suggested_rule is None  # v1: no LM-phrased rule
