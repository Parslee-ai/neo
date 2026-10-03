"""Neo as a CAR Lattice player (`neo.lattice`).

The daemon is the system boundary here, so node tests talk to an in-memory
stand-in that records the JSON-RPC calls the real daemon would receive, and
the supervisor tests use a stand-in for the `car_runtime` module. The
WebSocket transport itself is tested against a real local WebSocket server,
because the reverse-call reply is the one piece a fake cannot vouch for.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import pytest

from neo import lattice
from neo.lattice import (
    Answerer,
    LatticeNode,
    NodeConfig,
    PeerRequest,
    format_reply,
    plan_task,
    render_output,
    split_unified_diff,
)
from neo.models import (
    CodeSuggestion,
    NeoOutput,
    OrchestratorMessage,
    PlanStep,
    StaticCheckResult,
)

DIFF = """diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2
 y = 3
diff --git a/docs/new.md b/docs/new.md
--- /dev/null
+++ b/docs/new.md
@@ -0,0 +1 @@
+hello
diff --git a/old.txt b/old.txt
--- a/old.txt\t2026-01-01 00:00:00
+++ /dev/null
@@ -1 +0,0 @@
-bye
"""


def req(body: str, kind: str = "question", **kw) -> PeerRequest:
    return PeerRequest(message_id=kw.pop("message_id", "m-1"),
                       sender=kw.pop("sender", "codex-1"), body=body, kind=kind, **kw)


def _git_init(path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return str(path)


# ---------------------------------------------------------------------------
# Classifying requests
# ---------------------------------------------------------------------------


class TestPlanTask:
    def test_an_answer_is_never_answered(self):
        # Two auto-responders replying to each other loop forever.
        assert plan_task(req("thanks!", kind="answer")).op == "ignore"

    @pytest.mark.parametrize("kind", ["ack", "status", "result", "error", "something-new"])
    def test_unknown_kinds_are_dropped_not_answered(self, kind):
        # An allowlist: a kind Neo does not know is never a model call.
        assert plan_task(req("received", kind=kind)).op == "ignore"

    def test_an_untyped_reply_is_not_a_question(self):
        assert plan_task(req("received, thanks", kind="", in_reply_to="m-0")).op == "ignore"

    def test_handoff_is_declined_with_a_pointer(self):
        task = plan_task(req("please implement the parser", kind="handoff"))
        assert task.op == "decline"
        assert "players.find" in task.reason

    def test_plain_message_kind_is_a_question(self):
        # The daemon sends kind=message for an untyped message.
        request = PeerRequest.from_params({"id": "m", "from": "x", "body": "why?", "kind": "message"})
        assert request.kind == ""
        assert plan_task(request).op == "reason"

    def test_memory_prefix_is_a_lookup(self):
        task = plan_task(req("memory: supersession threshold"))
        assert (task.op, task.prompt) == ("memory", "supersession threshold")

    def test_review_request_separates_note_from_diff(self):
        task = plan_task(req("Is this safe?\n\n" + DIFF, kind="review_request"))
        assert task.op == "review"
        assert task.prompt == "Is this safe?"
        assert task.diff.startswith("diff --git a/src/app.py")

    def test_review_request_that_is_only_a_diff(self):
        task = plan_task(req(DIFF, kind="review_request"))
        assert (task.op, task.prompt) == ("review", "")
        assert task.diff == DIFF.strip()

    def test_a_prose_dashes_line_is_not_a_diff(self):
        task = plan_task(req("Note\n--- Summary\nmore words", kind="review_request"))
        assert task.diff == "" and "more words" in task.prompt

    def test_review_request_without_a_diff_still_reviews_the_text(self):
        task = plan_task(req("review my approach: cache per tenant", kind="review_request"))
        assert task.op == "review" and task.diff == ""
        assert "cache per tenant" in task.prompt

    def test_json_body(self):
        body = json.dumps({"op": "reason", "prompt": "why does it crash", "error_trace": "KeyError: x"})
        task = plan_task(req(body))
        assert (task.op, task.prompt, task.error_trace) == ("reason", "why does it crash", "KeyError: x")

    def test_json_cannot_widen_a_review_request_into_the_learning_path(self):
        body = json.dumps({"op": "reason", "prompt": "apply this", "diff": DIFF})
        assert plan_task(req(body, kind="review_request")).op == "review"

    def test_json_review_body_with_diff(self):
        task = plan_task(req(json.dumps({"op": "review", "diff": DIFF})))
        assert task.op == "review" and task.diff == DIFF

    def test_empty_request_is_declined_with_usage(self):
        task = plan_task(req("   "))
        assert task.op == "decline" and "review_request" in task.reason


class TestSplitDiff:
    def test_multi_file_diff_keeps_each_file_whole(self):
        pairs = split_unified_diff(DIFF)
        assert [p for p, _ in pairs] == ["src/app.py", "docs/new.md", "old.txt"]
        assert pairs[0][1].startswith("diff --git a/src/app.py")
        assert "+x = 2" in pairs[0][1] and "hello" not in pairs[0][1]

    def test_rename_and_binary_sections_do_not_corrupt_their_neighbours(self):
        diff = DIFF.split("diff --git a/docs")[0] + (
            "diff --git a/old b/new\nsimilarity index 100%\nrename from old\nrename to new\n"
            "diff --git a/i.png b/i.png\nBinary files a/i.png and b/i.png differ\n"
        )
        pairs = split_unified_diff(diff)
        assert [p for p, _ in pairs] == ["src/app.py"]
        assert "rename" not in pairs[0][1] and "Binary" not in pairs[0][1]

    def test_a_form_feed_inside_a_hunk_line_does_not_split_it(self):
        diff = "--- a/c.py\n+++ b/c.py\n@@ -1 +1 @@\n-x = '\f'\n+x = 1\n"
        [(path, text)] = split_unified_diff(diff)
        assert text == diff

    def test_a_diff_of_a_patch_file_is_one_section(self):
        diff = ("diff --git a/f.patch b/f.patch\n--- a/f.patch\n+++ b/f.patch\n"
                "@@ -1,2 +1,2 @@\n---- x\n-+++ y\n+--- z\n++++ w\n")
        assert [p for p, _ in split_unified_diff(diff)] == ["f.patch"]

    def test_quoted_paths_are_unquoted(self):
        diff = ('diff --git "a/caf\\303\\251.py" "b/caf\\303\\251.py"\n'
                '--- "a/caf\\303\\251.py"\n+++ "b/caf\\303\\251.py"\n@@ -1 +1 @@\n-a\n+b\n')
        assert [p for p, _ in split_unified_diff(diff)] == ["café.py"]

    def test_an_overstated_hunk_count_does_not_swallow_the_next_file(self):
        # Hand-edited and model-written diffs miscount routinely.
        diff = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1,5 +1,5 @@\n-a\n+b\n c\n"
                "diff --git a/y.py b/y.py\n--- a/y.py\n+++ b/y.py\n@@ -1 +1 @@\n-d\n+e\n")
        pairs = split_unified_diff(diff)
        assert [p for p, _ in pairs] == ["x.py", "y.py"]
        assert "y.py" not in pairs[0][1]

    def test_an_overstated_count_before_a_headerless_next_file(self):
        diff = ("--- a/x.py\n+++ b/x.py\n@@ -1,9 +1,9 @@\n-a\n+b\n"
                "--- a/y.py\n+++ b/y.py\n@@ -1 +1 @@\n-d\n+e\n")
        assert [p for p, _ in split_unified_diff(diff)] == ["x.py", "y.py"]

    def test_text_without_headers_yields_nothing(self):
        assert split_unified_diff("@@ -1 +1 @@\n-a\n+b\n") == []


# ---------------------------------------------------------------------------
# Identity, capabilities, repository root
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_project_name_comes_from_the_remote(self, tmp_path):
        root = _git_init(tmp_path / "car2")
        subprocess.run(["git", "-C", root, "remote", "add", "origin", "git@github.com:Parslee-ai/car.git"], check=True)
        assert lattice.derive_project_name(root) == "car"

    def test_project_name_falls_back_to_the_directory(self, tmp_path):
        root = _git_init(tmp_path / "my repo!")
        assert lattice.derive_project_name(root) == "my-repo"

    def test_languages_are_advertised_only_past_a_floor(self, tmp_path):
        root = _git_init(tmp_path / "r")
        for n in range(lattice.MIN_LANGUAGE_FILES):
            (tmp_path / "r" / f"m{n}.py").write_text("x = 1\n")
        (tmp_path / "r" / "one.rs").write_text("fn main() {}\n")
        subprocess.run(["git", "-C", root, "add", "."], check=True)
        tags = lattice.capabilities_for(root)
        assert "python" in tags and "rust" not in tags
        assert set(lattice.BASE_CAPABILITIES) <= set(tags)

    def test_a_non_repo_gets_base_tags_only(self, tmp_path):
        assert lattice.capabilities_for(str(tmp_path)) == list(lattice.BASE_CAPABILITIES)

    def test_every_tag_satisfies_the_daemon_rules(self):
        # 1-64 bytes, letters/digits/space/-_.+#/, at most 32 tags.
        assert len(lattice.BASE_CAPABILITIES) + lattice.MAX_LANGUAGE_TAGS <= 32
        for tag in lattice.BASE_CAPABILITIES:
            assert 1 <= len(tag) <= 64 and not lattice._TAG_RE.search(tag)

    def test_a_linked_worktree_resolves_to_the_main_checkout(self, tmp_path):
        # The node must not be pinned to an agent worktree that gets deleted.
        main = _git_init(tmp_path / "main")
        subprocess.run(["git", "-C", main, "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-q", "--allow-empty", "-m", "init"], check=True)
        wt = tmp_path / "wt"
        subprocess.run(["git", "-C", main, "worktree", "add", "-q", str(wt)], check=True)
        assert os.path.realpath(lattice.repository_root(str(wt))) == os.path.realpath(main)

    def test_outside_a_repository_there_is_no_root(self, tmp_path):
        assert lattice.repository_root(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def _output(self, suggestions, summary="I'd change 1 thing(s) in /.", cautions=("unverified",)):
        return NeoOutput(plan=[PlanStep(description="trace it", rationale="r")], simulation_traces=[],
                         code_suggestions=suggestions, static_checks=[], next_questions=["Is x ever None?"],
                         confidence=None, notes="",
                         orchestrator=OrchestratorMessage(summary=summary, cautions=list(cautions)))

    def test_a_change_renders_cautions_summary_plan_diff_and_questions(self):
        change = CodeSuggestion(file_path="a.py", unified_diff="-x\n+y", description="fix", confidence=0.8)
        text = render_output(self._output([change], summary="Found it."))
        for expected in ("Found it.", "- unverified", "1. trace it", "### a.py (confidence 0.80)",
                         "```diff\n-x\n+y\n```", "Is x ever None?"):
            assert expected in text

    def test_cautions_come_first_so_truncation_cannot_cut_them(self):
        answer = CodeSuggestion(file_path="/", unified_diff="", description="d", confidence=0.5,
                                code_block="z" * (lattice.ANSWER_BUDGET * 2))
        body = format_reply(req("q"), render_output(self._output([answer], cautions=["check me"])))
        assert "- check me" in body
        assert len(body) < lattice.ANSWER_BUDGET + 200

    def test_an_answer_shaped_suggestion_is_rendered_as_the_answer(self):
        answer = CodeSuggestion(file_path="/", unified_diff="", description="d", confidence=0.7,
                                code_block="Because the handler is the receipt boundary.\n\nFacts used: []")
        text = render_output(self._output([answer]))
        assert "Because the handler is the receipt boundary." in text
        assert "change 1 thing" not in text and "Facts used" not in text
        assert "### /" not in text and "Plan:" not in text

    def test_a_diffless_suggestion_on_a_real_path_is_an_answer_not_a_change(self):
        answer = CodeSuggestion(file_path="/src/neo/lattice.py", unified_diff="", description="d",
                                confidence=0.5, code_block="It declines handoffs: Neo is read-only.")
        text = render_output(self._output([answer]))
        assert "(re /src/neo/lattice.py)\nIt declines handoffs" in text
        assert "change 1 thing" not in text

    def test_reply_names_the_message_it_answers(self):
        assert format_reply(req("q", message_id="abc"), "text").startswith("re: abc\n")


# ---------------------------------------------------------------------------
# Answerer against stand-in engine and store (the LM is the boundary)
# ---------------------------------------------------------------------------


class _Engine:
    def __init__(self, store=None):
        self.inputs = []
        self.fact_store = store

    def process(self, neo_input):
        self.inputs.append(neo_input)
        checks = []
        if neo_input.operating_mode.value == "verify":
            checks = [StaticCheckResult(tool_name="ruff", diagnostics=[], summary="clean", status="passed")]
        return NeoOutput(plan=[], simulation_traces=[], code_suggestions=[], static_checks=checks,
                         next_questions=[], confidence=None, notes="",
                         orchestrator=OrchestratorMessage(summary=f"{neo_input.operating_mode.value} done"))


class _Store:
    def __init__(self, facts=()):
        self.facts = list(facts)
        self.calls = []

    def retrieve_relevant(self, query, k=30, domain=None, *, record_access=True):
        self.calls.append({"k": k, "record_access": record_access})
        return list(self.facts)


def _fact(subject, body="", scope="project", **kw):
    from neo.memory.models import Fact, FactScope

    return Fact(subject=subject, body=body, scope=FactScope(scope), **kw)


class TestAnswerer:
    def test_an_explanatory_question_runs_advise_and_records_nothing(self, tmp_path):
        engine = _Engine()
        Answerer(str(tmp_path), lambda: engine).answer(plan_task(req("explain how the queue works")))
        assert engine.inputs[0].operating_mode.value == "advise"
        assert engine.inputs[0].working_directory == str(tmp_path)
        assert engine.inputs[0].requested_output == lattice.QUESTION_OUTPUT

    def test_a_bugfix_question_runs_the_learning_path(self, tmp_path):
        # Only promotable task kinds can ever be confirmed by git, so only
        # they are worth an episode.
        engine = _Engine()
        Answerer(str(tmp_path), lambda: engine).answer(
            plan_task(req("fix the crash in the parser")))
        assert engine.inputs[0].operating_mode.value == "learn"

    def test_review_runs_verify_then_advise_never_learn(self, tmp_path):
        engine = _Engine()
        text = Answerer(str(tmp_path), lambda: engine).answer(
            plan_task(req("Is this safe?\n\n" + DIFF, kind="review_request")))
        modes = [i.operating_mode.value for i in engine.inputs]
        assert modes == ["verify", "advise"]
        assert [c.file_path for c in engine.inputs[0].proposed_changes] == ["src/app.py", "docs/new.md", "old.txt"]
        assert "- ruff: passed — clean" in text
        assert "Is this safe?" in engine.inputs[1].prompt and "+x = 2" in engine.inputs[1].prompt

    def test_memory_lookup_needs_no_engine_and_stamps_nothing(self, tmp_path):
        store = _Store([_fact("supersession at 0.85", "cosine >= 0.85", tags=["transcript-derived"])])

        def no_engine():
            raise AssertionError("a memory lookup must not build the engine (it needs an LM key)")

        text = Answerer(str(tmp_path), no_engine, store_factory=lambda: store).answer(
            plan_task(req("memory: supersession")))
        assert "1. supersession at 0.85 (pattern" in text and "[transcript-derived]" in text
        # A stamp ages a fact toward demotion with no chance of a success.
        assert store.calls and all(c["record_access"] is False for c in store.calls)

    def test_memory_shows_only_this_projects_facts(self, tmp_path):
        store = _Store([_fact("mine", scope="project"), _fact("other repo", scope="global"),
                        _fact("team", scope="org")])
        text = Answerer(str(tmp_path), store_factory=lambda: store).memory("x")
        assert "mine" in text and "other repo" not in text and "team" not in text

    def test_memory_drops_repeated_facts(self, tmp_path):
        store = _Store([_fact("Secret exposure", "never commit keys")] * 2 + [_fact("Other", "b")])
        text = Answerer(str(tmp_path), store_factory=lambda: store).memory("secrets")
        assert text.count("Secret exposure") == 1 and "2. Other" in text

    def test_history_facts_show_the_commit_not_the_patch(self, tmp_path):
        store = _Store([_fact("history:abc fix", "Commit: abc\nFiles: a.py\nChanges:\n+++ b/a.py\n+y")])
        text = Answerer(str(tmp_path), store_factory=lambda: store).memory("fix")
        assert "Files: a.py" in text and "+++ b/a.py" not in text

    def test_the_engine_store_replaces_the_standalone_one(self, tmp_path):
        engine_store = _Store()
        answerer = Answerer(str(tmp_path), lambda: _Engine(engine_store), store_factory=_Store)
        answerer.memory("x")  # builds the standalone store
        answerer.answer(plan_task(req("explain it")))  # builds the engine
        answerer.memory("y")
        assert engine_store.calls and answerer._store is None


# ---------------------------------------------------------------------------
# The node against a recording daemon stand-in
# ---------------------------------------------------------------------------


class _Daemon:
    """Records what the node sends; `deliver` plays the daemon's push."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.handlers = {}
        self.connected = True
        self.fail: dict[str, BaseException] = {}
        self.auth_error: BaseException | None = None
        self.closed = 0

    def __call__(self, url):
        return self

    def on_request(self, method, handler):
        self.handlers[method] = handler

    async def connect(self, auth):
        self.calls.append(("session.auth", auth))
        if self.auth_error:
            raise self.auth_error
        self.connected = True

    async def call(self, method, params, timeout=None):
        self.calls.append((method, params))
        if method in self.fail:
            raise self.fail[method]
        return {"id": f"sent-{len(self.calls)}"} if method == "agents.message" else {}

    async def close(self):
        self.connected = False
        self.closed += 1

    async def deliver(self, **params):
        return await self.handlers["agent.peer_message"](params)

    def sent(self, method):
        return [p for m, p in self.calls if m == method]


class _Answerer:
    def __init__(self, fail=False):
        self.tasks = []
        self.fail = fail
        self.loaded = False

    def answer(self, task):
        self.tasks.append(task)
        if self.fail:
            raise ValueError("no key")
        return f"answer to {task.prompt}"


def _node(daemon, answerer=None):
    config = NodeConfig(root="/repo", project="car", agent_id="neo-lattice-abc", token="t",
                        url="ws://x", capabilities=["code-review"])
    return LatticeNode(config, answerer or _Answerer(), client_factory=daemon)


async def _started(daemon, answerer=None, maxsize=lattice.MAX_QUEUE):
    node = _node(daemon, answerer)
    node.queue = asyncio.Queue(maxsize=maxsize)
    await node.connect_once()
    return node


async def _serve(node, *messages):
    worker = asyncio.create_task(node.worker())
    acks = [await node.client.deliver(**m) for m in messages]
    await asyncio.wait_for(node.queue.join(), timeout=5)
    worker.cancel()
    return acks


def msg(id, body, kind="question", sender="mcp:abc", **kw):
    return {"id": id, "from": sender, "body": body, "kind": kind, **kw}


@pytest.mark.asyncio
async def test_node_binds_agent_identity_and_joins():
    daemon = _Daemon()
    await _started(daemon)
    assert daemon.calls[0] == ("session.auth", {"token": "t", "agent_id": "neo-lattice-abc"})
    join = daemon.sent("players.join")[0]
    assert join["project"] == "car" and join["runtime"] == "neo" and join["status"] == "available"


@pytest.mark.asyncio
async def test_question_is_acked_then_answered_with_correlation():
    daemon = _Daemon()
    node = await _started(daemon)
    [ack] = await _serve(node, msg("m-7", "why?"))
    assert ack["received"] is True and ack["action"] == "queued"
    reply = daemon.sent("agents.message")[0]
    assert reply["to"] == "mcp:abc" and reply["kind"] == "answer"
    assert reply["in_reply_to"] == "m-7"
    assert reply["body"].startswith("re: m-7\n") and "answer to why?" in reply["body"]
    assert [p["status"] for p in daemon.sent("players.join")] == ["available", "busy", "available"]


@pytest.mark.asyncio
async def test_answers_unknown_kinds_and_no_reply_produce_no_message():
    daemon = _Daemon()
    node = await _started(daemon)
    acks = await _serve(node, msg("a", "thanks", kind="answer"), msg("b", "ok", kind="ack"),
                        msg("c", "fyi", no_reply=True))
    assert [a["action"] for a in acks] == ["ignored", "ignored", "queued"]
    assert daemon.sent("agents.message") == []


@pytest.mark.asyncio
async def test_handoff_is_declined_without_running_the_engine():
    daemon = _Daemon()
    answerer = _Answerer()
    node = await _started(daemon, answerer)
    await _serve(node, msg("h", "take this", kind="handoff"))
    assert answerer.tasks == []
    assert "read-only" in daemon.sent("agents.message")[0]["body"]


@pytest.mark.asyncio
async def test_a_failed_answer_is_reported_to_the_asker():
    daemon = _Daemon()
    node = await _started(daemon, _Answerer(fail=True))
    await _serve(node, msg("q", "why?"))
    assert "Neo failed to answer: ValueError: no key" in daemon.sent("agents.message")[0]["body"]


@pytest.mark.asyncio
async def test_overflow_gets_exactly_one_busy_notice():
    daemon = _Daemon()
    node = await _started(daemon, maxsize=1)
    await node.client.deliver(**msg("1", "one", sender="x"))
    ack = await node.client.deliver(**msg("2", "two", sender="y"))
    assert ack["action"] == "busy"
    await asyncio.gather(*node._tasks)
    busy = daemon.sent("agents.message")
    assert len(busy) == 1 and busy[0]["to"] == "y" and "Try again" in busy[0]["body"]


@pytest.mark.asyncio
async def test_a_refused_reply_is_not_retried():
    daemon = _Daemon()
    daemon.fail["agents.message"] = RuntimeError("recipient gone")
    node = await _started(daemon)
    await asyncio.wait_for(node._reply(req("q"), "text"), timeout=2)
    assert len(daemon.sent("agents.message")) == 1 and daemon.connected


@pytest.mark.asyncio
async def test_a_reply_lost_to_the_transport_is_resent_after_reconnecting():
    daemon = _Daemon()
    daemon.fail["agents.message"] = ConnectionError("socket closed")
    node = await _started(daemon)
    reply = asyncio.create_task(node._reply(req("q"), "text"))
    await asyncio.sleep(0.05)
    assert daemon.connected is False  # the transport failure dropped the connection
    del daemon.fail["agents.message"]
    await node.connect_once()  # what the run loop does next
    await asyncio.wait_for(reply, timeout=2)
    assert len(daemon.sent("agents.message")) == 2


@pytest.mark.parametrize("error", [
    asyncio.TimeoutError(),
    RuntimeError("this connection no longer owns agent `x`; a newer attach superseded it"),
])
@pytest.mark.asyncio
async def test_a_silent_or_superseded_daemon_drops_the_connection(error):
    # Either way the socket stays open and nothing else would notice.
    daemon = _Daemon()
    node = await _started(daemon)
    daemon.fail["players.join"] = error
    await node.set_status("busy")
    assert daemon.connected is False


@pytest.mark.asyncio
async def test_a_failed_auth_leaves_the_client_where_the_run_loop_closes_it():
    daemon = _Daemon()
    daemon.auth_error = RuntimeError("token mismatch")
    node = _node(daemon)
    with pytest.raises(RuntimeError):
        await node.connect_once()
    # Assigned before connecting: the loop's `self.client.close()` reaches
    # THIS client, not the previous, already-closed one.
    assert node.client is daemon


@pytest.mark.asyncio
async def test_a_timed_out_reply_is_not_resent():
    # The daemon may have delivered it; resending would duplicate the answer.
    daemon = _Daemon()
    daemon.fail["agents.message"] = asyncio.TimeoutError()
    node = await _started(daemon)
    await asyncio.wait_for(node._reply(req("q"), "text"), timeout=2)
    assert len(daemon.sent("agents.message")) == 1


@pytest.mark.asyncio
async def test_answering_marks_the_node_as_in_use(monkeypatch):
    # A node serving only peers is in use even if nobody runs the neo CLI.
    touched = []
    monkeypatch.setattr(lattice, "touch_seen", touched.append)
    daemon = _Daemon()
    node = await _started(daemon)
    await _serve(node, msg("q", "why?"))
    assert touched == ["neo-lattice-abc"]


@pytest.mark.asyncio
async def test_a_decline_does_not_flap_the_status():
    daemon = _Daemon()
    node = await _started(daemon)
    await _serve(node, msg("h", "do it", kind="handoff"))
    assert [p["status"] for p in daemon.sent("players.join")] == ["available"]


@pytest.mark.asyncio
async def test_a_turned_away_request_gets_its_budget_slot_back():
    daemon = _Daemon()
    node = await _started(daemon, maxsize=1)
    await node.client.deliver(**msg("1", "one", sender="x"))
    await node.client.deliver(**msg("2", "two", sender="x"))  # busy
    assert len(node._spend["x"]) == 1


@pytest.mark.asyncio
async def test_a_stop_interrupts_the_reconnect_backoff(monkeypatch):
    daemon = _Daemon()
    daemon.auth_error = ConnectionError("daemon down")
    node = _node(daemon)
    monkeypatch.setattr(lattice, "RECONNECT_MIN_SECS", 30.0)
    run = asyncio.create_task(node.run())
    await asyncio.sleep(0.1)  # first connect failed; now in a 30 s backoff
    node._stop.set()
    await asyncio.wait_for(run, timeout=2)


@pytest.mark.asyncio
async def test_a_daemon_without_the_lattice_ends_the_node_cleanly():
    from neo.a2ui import RpcError

    daemon = _Daemon()
    daemon.fail["players.join"] = RpcError("jsonrpc players.join failed: unknown method: players.join")
    node = _node(daemon)
    await asyncio.wait_for(node.run(), timeout=5)  # returns; no reconnect loop
    assert node.unavailable is True
    assert len(daemon.sent("players.join")) == 1


@pytest.mark.asyncio
async def test_a_clean_stop_is_not_reported_as_a_dead_worker(monkeypatch):
    daemon = _Daemon()
    node = _node(daemon)
    monkeypatch.setattr(lattice, "POLL_SECS", 0.05)

    async def finishing_worker():
        node._stop.set()  # the worker finishes because it was told to stop

    node.worker = finishing_worker
    await asyncio.wait_for(node.run(), timeout=2)  # no WorkerDied


class TestBudgets:
    def test_a_sender_is_cut_off_after_the_hourly_budget(self):
        node = _node(_Daemon())
        ops = [node.admit(req("q", sender="loop"), plan_task(req("q")), now=100.0).op
               for _ in range(lattice.SENDER_BUDGET_PER_HOUR + 1)]
        assert ops[-1] == "decline" and set(ops[:-1]) == {"reason"}
        assert node.admit(req("q", sender="other"), plan_task(req("q")), now=100.0).op == "reason"
        assert node.admit(req("q", sender="loop"), plan_task(req("q")), now=100.0 + 3601).op == "reason"

    def test_memory_lookups_are_not_charged(self):
        node = _node(_Daemon())
        for _ in range(lattice.SENDER_BUDGET_PER_HOUR * 2):
            task = node.admit(req("memory: x", sender="s"), plan_task(req("memory: x")))
            assert task.op == "memory"

    def test_a_follow_up_chain_stops_at_the_depth_cap(self):
        node = _node(_Daemon())
        node._answer_depth["a0"] = lattice.MAX_FOLLOWUP_DEPTH
        follow_up = req("and then?", in_reply_to="a0")
        assert node.admit(follow_up, plan_task(follow_up)).op == "ignore"
        shallow = req("and then?", in_reply_to="a1")
        node._answer_depth["a1"] = 1
        assert node.admit(shallow, plan_task(shallow)).depth == 2


class TestRecycle:
    def test_idle_with_nothing_loaded_never_recycles(self):
        node = _node(_Daemon(), Answerer("/repo", engine_factory=object))
        node.last_activity -= lattice.IDLE_RECYCLE_SECS * 2
        assert node.should_recycle() is False

    @pytest.mark.parametrize("load", ["engine", "store"])
    def test_idle_with_memory_loaded_recycles_and_busy_does_not(self, load):
        answerer = Answerer("/repo", engine_factory=object, store_factory=object)
        if load == "engine":
            answerer.engine
        else:
            answerer._memory_store()
        node = _node(_Daemon(), answerer)
        node.last_activity -= lattice.IDLE_RECYCLE_SECS * 2
        assert node.should_recycle() is True
        node.working = True
        assert node.should_recycle() is False


@pytest.mark.asyncio
async def test_a_dead_worker_fails_the_process_instead_of_acking_forever(monkeypatch):
    daemon = _Daemon()
    node = _node(daemon)
    monkeypatch.setattr(lattice, "POLL_SECS", 0.01)

    async def broken_worker():
        raise KeyError("boom")

    node.worker = broken_worker
    with pytest.raises(lattice.WorkerDied):
        await asyncio.wait_for(node.run(), timeout=5)


# ---------------------------------------------------------------------------
# The supervised process and its registration
# ---------------------------------------------------------------------------


class TestForeground:
    def test_agent_credentials_leave_the_environment(self, monkeypatch, tmp_path):
        # car_runtime reads these too; left in place, any CAR use inside the
        # process attaches as this agent and supersedes the node's connection.
        seen = {}

        async def fake_run(self):
            seen["env"] = (os.environ.get("CAR_AGENT_ID"), os.environ.get("CAR_AGENT_TOKEN"))
            seen["config"] = (self.config.agent_id, self.config.token)

        monkeypatch.setenv("CAR_AGENT_ID", "neo-lattice-1")
        monkeypatch.setenv("CAR_AGENT_TOKEN", "secret")
        monkeypatch.setattr(lattice.LatticeNode, "run", fake_run)
        assert lattice._run_foreground(str(tmp_path), "p") == 0
        assert seen == {"env": (None, None), "config": ("neo-lattice-1", "secret")}

    def test_recycle_hands_the_credentials_to_the_new_image_only(self, monkeypatch, tmp_path):
        execs = []

        async def fake_run(self):
            self.recycle = True

        monkeypatch.setenv("CAR_AGENT_ID", "neo-lattice-1")
        monkeypatch.setenv("CAR_AGENT_TOKEN", "secret")
        monkeypatch.setattr(lattice.LatticeNode, "run", fake_run)
        monkeypatch.setattr(lattice.os, "execve", lambda path, argv, env: execs.append((argv, env)))
        lattice._run_foreground(str(tmp_path), "p")
        [(argv, env)] = execs
        assert (env["CAR_AGENT_ID"], env["CAR_AGENT_TOKEN"]) == ("neo-lattice-1", "secret")
        assert "CAR_AGENT_TOKEN" not in os.environ
        assert argv[1:] == lattice.node_argv(str(tmp_path), "p")


class _Car:
    """Stand-in for the `car_runtime` module's supervisor calls."""

    def __init__(self):
        self.agents: list[dict] = []
        self.list_error: BaseException | None = None
        self.ops: list[tuple[str, str]] = []
        self.spec: dict = {}
        self.has_lattice: bool | None = True

    def agents_list(self):
        if self.list_error:
            raise self.list_error
        return json.dumps(self.agents)

    def agents_upsert(self, spec):
        self.spec = json.loads(spec)
        self.ops.append(("upsert", self.spec["id"]))

    def agents_start(self, agent_id):
        self.ops.append(("start", agent_id))
        return json.dumps({"pid": 1})

    def agents_restart(self, agent_id):
        self.ops.append(("restart", agent_id))
        return json.dumps({"pid": 1})

    def agents_stop(self, agent_id):
        self.ops.append(("stop", agent_id))

    def agents_remove(self, agent_id):
        self.ops.append(("remove", agent_id))


@pytest.fixture
def car(monkeypatch):
    import neo.memory.observer as observer

    fake = _Car()
    monkeypatch.delenv("NEO_OBSERVER_AUTOSTART", raising=False)
    monkeypatch.setattr(observer, "_car_server_reachable", lambda *a, **k: True)
    monkeypatch.setattr(observer, "_require_car_runtime", lambda: fake)
    # The daemon is the boundary: never let a test probe the machine's real one.
    monkeypatch.setattr(lattice, "daemon_has_lattice", lambda url, **k: fake.has_lattice)
    return fake


class TestAutojoin:
    def test_registers_the_main_checkout_with_a_safe_cwd(self, tmp_path, car):
        root = _git_init(tmp_path / "repo")
        agent_id = lattice.maybe_autojoin(root)
        assert car.ops == [("upsert", agent_id), ("start", agent_id)]
        spec = car.spec
        # The repository is an argument, never the process cwd or sys.path.
        assert not os.path.realpath(spec["cwd"]).startswith(os.path.realpath(root))
        assert os.path.realpath(spec["args"][spec["args"].index("--cwd") + 1]) == os.path.realpath(root)
        if sys.version_info >= (3, 11):
            assert spec["args"][0] == "-P"
        assert spec["auto_start"] is True and spec["command"] == sys.executable

    def test_a_viable_registration_is_left_alone(self, tmp_path, car):
        root = _git_init(tmp_path / "repo")
        car.agents = [{"id": lattice.agent_id_for(root), "args": ["--cwd", root],
                       "command": sys.executable, "status": "running"}]
        assert lattice.maybe_autojoin(root) is None and car.ops == []

    def test_a_node_whose_checkout_vanished_is_re_registered(self, tmp_path, car):
        root = _git_init(tmp_path / "repo")
        agent_id = lattice.agent_id_for(root)
        car.agents = [{"id": agent_id, "args": ["--cwd", str(tmp_path / "deleted-worktree")],
                       "command": sys.executable, "status": "errored"}]
        assert lattice.maybe_autojoin(root) == agent_id
        assert car.ops == [("upsert", agent_id), ("start", agent_id)]

    def test_a_record_missing_fields_is_unknown_not_broken(self, tmp_path, car):
        # Both "broken" answers are destructive (restart this node, reap the
        # others), so a listing that omits fields must trigger neither.
        root = _git_init(tmp_path / "repo")
        car.agents = [{"id": lattice.agent_id_for(root), "status": "running"},
                      {"id": "neo-lattice-other", "status": "running"}]
        assert lattice.maybe_autojoin(root) is None and car.ops == []

    @pytest.mark.parametrize("answer", [False, None])
    def test_a_daemon_without_the_lattice_gets_no_node(self, tmp_path, car, answer):
        # Today's CarHost has no players.*: a node registered there could
        # never join and would reconnect forever.
        root = _git_init(tmp_path / "repo")
        car.has_lattice = answer
        assert lattice.maybe_autojoin(root) is None and car.ops == []

    def test_explicit_join_says_why_on_a_daemon_without_the_lattice(self, tmp_path, car):
        root = _git_init(tmp_path / "repo")
        car.has_lattice = False
        result = lattice.join(root)
        assert result["status"] == "error" and "no Lattice" in result["message"]
        assert car.ops == []

    def test_a_failed_listing_registers_nothing(self, tmp_path, car):
        root = _git_init(tmp_path / "repo")
        car.list_error = RuntimeError("handshake failed")
        assert lattice.maybe_autojoin(root) is None and car.ops == []

    def test_stale_and_orphaned_nodes_are_reaped(self, tmp_path, car):
        root = _git_init(tmp_path / "repo")
        live_root = _git_init(tmp_path / "live")
        car.agents = [
            {"id": "neo-lattice-stale", "args": ["--cwd", live_root], "command": sys.executable},
            {"id": "neo-lattice-gone", "args": ["--cwd", str(tmp_path / "gone")], "command": sys.executable},
            {"id": "neo-lattice-fresh", "args": ["--cwd", live_root], "command": sys.executable},
            {"id": "someone-else", "args": [], "command": ""},
        ]
        for agent_id, age_days in (("neo-lattice-stale", lattice.STALE_NODE_DAYS + 1), ("neo-lattice-fresh", 1)):
            marker = lattice._seen_marker(agent_id)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            past = marker.stat().st_mtime - age_days * 86400
            os.utime(marker, (past, past))
        lattice.maybe_autojoin(root)
        removed = {a for op, a in car.ops if op == "remove"}
        assert removed == {"neo-lattice-stale", "neo-lattice-gone"}

    def test_outside_a_repository(self, tmp_path, car):
        assert lattice.maybe_autojoin(str(tmp_path)) is None and car.ops == []

    def test_the_shared_autostart_opt_out(self, tmp_path, car, monkeypatch):
        root = _git_init(tmp_path / "repo")
        monkeypatch.setenv("NEO_OBSERVER_AUTOSTART", "0")
        assert lattice.maybe_autojoin(root) is None and car.ops == []

    def test_a_repository_the_user_left_stays_off_until_joined(self, tmp_path, car):
        root = _git_init(tmp_path / "repo")
        assert lattice.leave(root)["status"] == "not_joined"
        assert lattice.maybe_autojoin(root) is None and car.ops == []
        assert lattice.join(root)["status"] == "started"
        assert not lattice._left_marker(root).exists()

    def test_daemon_down_is_silent(self, tmp_path, monkeypatch):
        import neo.memory.observer as observer

        root = _git_init(tmp_path / "repo")
        monkeypatch.delenv("NEO_OBSERVER_AUTOSTART", raising=False)
        monkeypatch.setattr(observer, "_car_server_reachable", lambda *a, **k: False)
        monkeypatch.setattr(observer, "_require_car_runtime",
                            lambda: pytest.fail("reached CAR with the daemon down"))
        assert lattice.maybe_autojoin(root) is None


def test_status_does_not_trust_car_over_a_live_node(tmp_path, car):
    # Under client/daemon protocol skew agents_list reports every agent
    # stopped; the node's own pid file is the check that does not route
    # through CAR.
    root = _git_init(tmp_path / "repo")
    agent_id = lattice.agent_id_for(root)
    car.agents = [{"id": agent_id, "args": ["--cwd", root], "command": sys.executable, "status": "stopped"}]
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)",
                                "neo.lattice", "--cwd", root])
    try:
        pid_file = lattice._pid_file(agent_id)
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(f"{sleeper.pid}\n{root}\n")
        assert lattice.status(root)["status"] == "unverified"
        assert lattice.join(root)["status"] == "error"
        assert car.ops == []
    finally:
        sleeper.kill()


def test_a_reused_pid_serving_another_repository_is_not_this_node(tmp_path, car):
    # A pid file outlives a SIGKILL; its pid can come back as a different
    # repository's node, which must not pin this one as "unverified".
    root = _git_init(tmp_path / "repo")
    other = _git_init(tmp_path / "other")
    agent_id = lattice.agent_id_for(root)
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)",
                                "neo.lattice", "--cwd", other])
    try:
        pid_file = lattice._pid_file(agent_id)
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(f"{sleeper.pid}\n{root}\n")
        assert lattice.status(root)["status"] == "not_joined"
    finally:
        sleeper.kill()


# ---------------------------------------------------------------------------
# DaemonClient over a real socket
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_daemon_client_answers_a_reverse_call_over_a_real_socket():
    websockets = pytest.importorskip("websockets")
    from neo.a2ui import DaemonClient

    replies: list[dict] = []
    done = asyncio.Event()

    async def daemon(ws):
        auth = json.loads(await ws.recv())
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": auth["id"], "result": {"ok": True}}))
        # The daemon's own id space can collide with the client's (1 here):
        # the frame carries `method`, so it must be treated as a request.
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "agent.peer_message",
                                  "params": {"id": "m", "from": "x", "body": "hi"}}))
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "agent.unknown", "params": {}}))
        for _ in range(2):
            replies.append(json.loads(await ws.recv()))
        done.set()
        await ws.wait_closed()

    async with websockets.serve(daemon, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = DaemonClient(f"ws://127.0.0.1:{port}/")

        async def on_message(params):
            return {"received": True, "body": params["body"]}

        client.on_request("agent.peer_message", on_message)
        result = await client.connect({"token": "t", "agent_id": "neo-1"})
        assert result == {"ok": True}
        await asyncio.wait_for(done.wait(), timeout=5)
        await client.close()

    assert replies[0] == {"jsonrpc": "2.0", "id": 1, "result": {"received": True, "body": "hi"}}
    assert replies[1]["id"] == 2 and replies[1]["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_daemon_client_times_out_a_silent_daemon_and_closes_on_failed_auth():
    websockets = pytest.importorskip("websockets")
    from neo.a2ui import DaemonClient

    async def silent(ws):
        auth = json.loads(await ws.recv())
        if auth["params"].get("token") == "bad":
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": auth["id"],
                                      "error": {"code": -32603, "message": "auth failed"}}))
        else:
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": auth["id"], "result": {}}))
        await ws.wait_closed()  # and never answer anything else

    async with websockets.serve(silent, "127.0.0.1", 0) as server:
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/"
        client = DaemonClient(url)
        await client.connect({"token": "ok"})
        with pytest.raises(asyncio.TimeoutError):
            await client.call("players.join", {}, timeout=0.2)
        assert client._pending == {}
        await client.close()

        bad = DaemonClient(url)
        with pytest.raises(RuntimeError):
            await bad.connect({"token": "bad"})
        assert not bad.connected


@pytest.mark.parametrize("refusal,expected", [
    ("unknown method: players.list", False),  # today's CarHost
    ("players and work claims need a peer address: connect as an attached agent", True),
])
def test_the_lattice_probe_reads_the_refusal(refusal, expected):
    websockets = pytest.importorskip("websockets")
    import threading

    ready = threading.Event()
    port: list[int] = []
    stop: list = []

    async def daemon(ws):
        async for raw in ws:
            frame = json.loads(raw)
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": frame["id"],
                                      "error": {"code": -32603, "message": refusal}}))

    async def serve():
        async with websockets.serve(daemon, "127.0.0.1", 0) as server:
            port.append(server.sockets[0].getsockname()[1])
            loop = asyncio.get_running_loop()
            done = loop.create_future()
            stop.append((loop, done))
            ready.set()
            await done

    thread = threading.Thread(target=asyncio.run, args=(serve(),), daemon=True)
    thread.start()
    ready.wait(5)
    try:
        assert lattice.daemon_has_lattice(f"ws://127.0.0.1:{port[0]}/") is expected
    finally:
        loop, done = stop[0]
        loop.call_soon_threadsafe(done.set_result, None)
        thread.join(5)


def test_an_unreachable_daemon_is_unknown_not_absent():
    assert lattice.daemon_has_lattice("ws://127.0.0.1:9/", timeout=1) is None
