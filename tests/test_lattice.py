"""Neo as a CAR Lattice player (`neo.lattice`).

The daemon is the system boundary here, so node tests talk to an in-memory
stand-in that records the JSON-RPC calls the real daemon would receive. The
WebSocket transport itself is tested against a real local WebSocket server,
because the reverse-call reply is the one piece a fake cannot vouch for.
"""

from __future__ import annotations

import asyncio
import os
import json
import subprocess

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


class TestPlanTask:
    def test_an_answer_is_never_answered(self):
        # Two auto-responders replying to each other loop forever.
        assert plan_task(req("thanks!", kind="answer")).op == "ignore"

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

    def test_review_request_without_a_diff_still_reviews_the_text(self):
        task = plan_task(req("review my approach: cache per tenant", kind="review_request"))
        assert task.op == "review" and task.diff == ""
        assert "cache per tenant" in task.prompt

    def test_json_body(self):
        body = json.dumps({"op": "reason", "prompt": "why does it crash", "error_trace": "KeyError: x"})
        task = plan_task(req(body))
        assert (task.op, task.prompt, task.error_trace) == ("reason", "why does it crash", "KeyError: x")

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

    def test_text_without_headers_yields_nothing(self):
        assert split_unified_diff("@@ -1 +1 @@\n-a\n+b\n") == []


class TestIdentity:
    def _repo(self, path, remote: str | None = None):
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        if remote:
            subprocess.run(["git", "-C", str(path), "remote", "add", "origin", remote], check=True)
        return str(path)

    def test_project_name_comes_from_the_remote(self, tmp_path):
        root = self._repo(tmp_path / "car2", "git@github.com:Parslee-ai/car.git")
        assert lattice.derive_project_name(root) == "car"

    def test_project_name_falls_back_to_the_directory(self, tmp_path):
        root = self._repo(tmp_path / "my repo!")
        assert lattice.derive_project_name(root) == "my-repo"

    def test_languages_are_advertised_only_past_a_floor(self, tmp_path):
        root = self._repo(tmp_path / "r")
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


class TestRendering:
    def test_render_output_carries_summary_cautions_plan_and_diffs(self):
        output = NeoOutput(
            plan=[PlanStep(description="Guard the empty case", rationale="r")],
            simulation_traces=[],
            code_suggestions=[CodeSuggestion(file_path="a.py", unified_diff="-x\n+y",
                                             description="fix", confidence=0.8)],
            static_checks=[], next_questions=["Is x ever None?"], confidence=0.8, notes="",
            orchestrator=OrchestratorMessage(summary="Found it.", cautions=["untested"]),
        )
        text = render_output(output)
        for expected in ("Found it.", "- untested", "1. Guard the empty case",
                         "### a.py (confidence 0.80)", "```diff\n-x\n+y\n```", "Is x ever None?"):
            assert expected in text

    def test_reply_names_the_message_it_answers_and_is_bounded(self):
        body = format_reply(req("q", message_id="abc"), "z" * (lattice.ANSWER_BUDGET * 2))
        assert body.startswith("re: abc\n")
        assert len(body) < lattice.ANSWER_BUDGET + 200


# ---------------------------------------------------------------------------
# Answerer against a stand-in engine (the engine's LM is the boundary)
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


class TestAnswerer:
    def test_question_runs_learn_mode_in_the_repo(self, tmp_path):
        engine = _Engine()
        text = Answerer(str(tmp_path), lambda: engine).answer(plan_task(req("why?")))
        assert engine.inputs[0].operating_mode.value == "learn"
        assert engine.inputs[0].working_directory == str(tmp_path)
        assert text == "learn done"

    def test_memory_uses_the_engine_store_once_the_engine_exists(self, tmp_path):
        class Store:
            def retrieve_relevant(self, query, k):
                return []

        engine = _Engine(Store())
        answerer = Answerer(str(tmp_path), lambda: engine,
                            store_factory=lambda: pytest.fail("second store built"))
        answerer.answer(plan_task(req("why?")))
        assert "nothing on" in answerer.answer(plan_task(req("memory: x")))

    def test_review_runs_verify_then_advise_never_learn(self, tmp_path):
        engine = _Engine()
        text = Answerer(str(tmp_path), lambda: engine).answer(
            plan_task(req("Is this safe?\n\n" + DIFF, kind="review_request")))
        modes = [i.operating_mode.value for i in engine.inputs]
        # ADVISE, not LEARN: the patch is the peer's, not a Neo suggestion.
        assert modes == ["verify", "advise"]
        assert [c.file_path for c in engine.inputs[0].proposed_changes] == ["src/app.py", "docs/new.md", "old.txt"]
        assert "- ruff: passed — clean" in text
        assert "Is this safe?" in engine.inputs[1].prompt and "+x = 2" in engine.inputs[1].prompt

    def test_memory_lookup_makes_no_engine_call(self, tmp_path):
        from neo.memory.models import Fact, FactKind

        class Store:
            def retrieve_relevant(self, query, k):
                fact = Fact(subject="supersession at 0.85", body="cosine >= 0.85 supersedes",
                            kind=FactKind.PATTERN, tags=["transcript-derived"])
                return [fact]

        def no_engine():
            raise AssertionError("a memory lookup must not build the engine (it needs an LM key)")

        text = Answerer(str(tmp_path), no_engine, store_factory=Store).answer(
            plan_task(req("memory: supersession")))
        assert "1. supersession at 0.85 (pattern" in text and "[transcript-derived]" in text


# ---------------------------------------------------------------------------
# The node against a recording daemon stand-in
# ---------------------------------------------------------------------------


class _Daemon:
    """Records what the node sends; `deliver` plays the daemon's push."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.handlers = {}
        self.connected = True
        self.refuse: set[str] = set()

    def __call__(self, url):
        return self

    def on_request(self, method, handler):
        self.handlers[method] = handler

    async def connect(self, auth):
        self.calls.append(("session.auth", auth))

    async def call(self, method, params):
        self.calls.append((method, params))
        if method in self.refuse:
            raise RuntimeError(f"{method} refused")
        return {}

    async def close(self):
        self.connected = False

    async def deliver(self, **params):
        return await self.handlers["agent.peer_message"](params)

    def sent(self, method):
        return [p for m, p in self.calls if m == method]


class _SlowAnswerer:
    def __init__(self):
        self.tasks = []

    def answer(self, task):
        self.tasks.append(task)
        return f"answer to {task.prompt}"


def _node(daemon, answerer=None):
    config = NodeConfig(root="/repo", project="car", agent_id="neo-lattice-abc", token="t",
                        url="ws://x", capabilities=["code-review"])
    return LatticeNode(config, answerer or _SlowAnswerer(), client_factory=daemon)


async def _drain(node):
    await asyncio.wait_for(node.queue.join(), timeout=5)


@pytest.mark.asyncio
async def test_node_binds_agent_identity_and_joins():
    daemon = _Daemon()
    node = _node(daemon)
    node.queue = asyncio.Queue(maxsize=lattice.MAX_QUEUE)
    await node.connect_once()
    assert daemon.calls[0] == ("session.auth", {"token": "t", "agent_id": "neo-lattice-abc"})
    join = daemon.sent("players.join")[0]
    assert join["project"] == "car" and join["runtime"] == "neo" and join["status"] == "available"


@pytest.mark.asyncio
async def test_question_is_acked_then_answered_with_correlation():
    daemon = _Daemon()
    node = _node(daemon)
    node.queue = asyncio.Queue(maxsize=lattice.MAX_QUEUE)
    await node.connect_once()
    worker = asyncio.create_task(node.worker())
    ack = await daemon.deliver(id="m-7", **{"from": "mcp:abc"}, body="why?", kind="question")
    assert ack["received"] is True and ack["action"] == "queued"
    await _drain(node)
    worker.cancel()

    reply = daemon.sent("agents.message")[0]
    assert reply["to"] == "mcp:abc" and reply["kind"] == "answer"
    assert reply["in_reply_to"] == "m-7"
    assert reply["body"].startswith("re: m-7\n") and "answer to why?" in reply["body"]
    statuses = [p["status"] for p in daemon.sent("players.join")]
    assert statuses == ["available", "busy", "available"]


@pytest.mark.asyncio
async def test_answers_and_no_reply_produce_no_message():
    daemon = _Daemon()
    node = _node(daemon)
    node.queue = asyncio.Queue(maxsize=lattice.MAX_QUEUE)
    await node.connect_once()
    worker = asyncio.create_task(node.worker())
    ack = await daemon.deliver(id="a", **{"from": "x"}, body="thanks", kind="answer")
    assert ack["action"] == "ignored"
    await daemon.deliver(id="b", **{"from": "x"}, body="fyi", kind="question", no_reply=True)
    await _drain(node)
    worker.cancel()
    assert daemon.sent("agents.message") == []


@pytest.mark.asyncio
async def test_handoff_is_declined_without_running_the_engine():
    daemon = _Daemon()
    answerer = _SlowAnswerer()
    node = _node(daemon, answerer)
    node.queue = asyncio.Queue(maxsize=lattice.MAX_QUEUE)
    await node.connect_once()
    worker = asyncio.create_task(node.worker())
    await daemon.deliver(id="h", **{"from": "x"}, body="take this", kind="handoff")
    await _drain(node)
    worker.cancel()
    assert answerer.tasks == []
    assert "read-only" in daemon.sent("agents.message")[0]["body"]


@pytest.mark.asyncio
async def test_overflow_is_told_busy_instead_of_queued_forever():
    daemon = _Daemon()
    node = _node(daemon)
    node.queue = asyncio.Queue(maxsize=1)
    await node.connect_once()
    await daemon.deliver(id="1", **{"from": "x"}, body="one", kind="question")
    ack = await daemon.deliver(id="2", **{"from": "y"}, body="two", kind="question")
    assert ack["action"] == "busy"
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    busy = daemon.sent("agents.message")
    assert busy and busy[0]["to"] == "y" and "Try again" in busy[0]["body"]


@pytest.mark.asyncio
async def test_a_refused_reply_on_a_live_connection_is_not_retried():
    daemon = _Daemon()
    daemon.refuse.add("agents.message")
    node = _node(daemon)
    node.queue = asyncio.Queue(maxsize=lattice.MAX_QUEUE)
    await node.connect_once()
    await asyncio.wait_for(node._reply(req("q"), "text"), timeout=2)
    assert len(daemon.sent("agents.message")) == 1


# ---------------------------------------------------------------------------
# DaemonClient: daemon-to-client requests get a reply on the real transport
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
async def test_a_superseded_binding_is_dropped_so_the_node_reattaches():
    # The daemon answers every call on a superseded binding with an error while
    # the socket stays open; without closing it the node would be a ghost.
    daemon = _Daemon()
    node = _node(daemon)
    node.queue = asyncio.Queue(maxsize=lattice.MAX_QUEUE)
    await node.connect_once()

    async def superseded(method, params):
        raise RuntimeError("this connection no longer owns agent `x`; a newer attach superseded it")

    daemon.call = superseded
    await node.set_status("busy")
    assert daemon.connected is False


class TestRecycle:
    def test_idle_without_an_engine_never_recycles(self):
        node = _node(_Daemon(), Answerer("/repo", engine_factory=lambda: object()))
        node.last_activity -= lattice.IDLE_RECYCLE_SECS * 2
        assert node.should_recycle() is False

    def test_idle_with_a_loaded_engine_recycles_and_busy_does_not(self):
        answerer = Answerer("/repo", engine_factory=lambda: object())
        answerer.engine  # load it
        node = _node(_Daemon(), answerer)
        node.last_activity -= lattice.IDLE_RECYCLE_SECS * 2
        assert node.should_recycle() is True
        node.working = True
        assert node.should_recycle() is False


class TestAutojoinGates:
    """Every gate that must stop autojoin BEFORE it touches CAR."""

    @pytest.fixture
    def no_car(self, monkeypatch):
        import neo.memory.observer as observer

        def unreachable(*a, **k):
            raise AssertionError("autojoin reached CAR past a gate that should have stopped it")

        monkeypatch.setattr(observer, "_car_server_reachable", unreachable)
        monkeypatch.delenv("NEO_OBSERVER_AUTOSTART", raising=False)

    def test_outside_a_repository(self, tmp_path, no_car):
        assert lattice.maybe_autojoin(str(tmp_path)) is None

    def test_the_shared_autostart_opt_out(self, tmp_path, no_car, monkeypatch):
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        monkeypatch.setenv("NEO_OBSERVER_AUTOSTART", "0")
        assert lattice.maybe_autojoin(str(tmp_path)) is None

    def test_a_repository_the_user_left(self, tmp_path, no_car):
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        root = lattice.repository_root(str(tmp_path))
        marker = lattice._left_marker(root)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("left\n")
        assert lattice.maybe_autojoin(str(tmp_path)) is None

    def test_daemon_down_is_silent(self, tmp_path, monkeypatch):
        import neo.memory.observer as observer

        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        monkeypatch.delenv("NEO_OBSERVER_AUTOSTART", raising=False)
        monkeypatch.setattr(observer, "_car_server_reachable", lambda *a, **k: False)
        assert lattice.maybe_autojoin(str(tmp_path)) is None


def test_the_node_takes_agent_credentials_out_of_its_environment(monkeypatch, tmp_path):
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


class TestAnswerRendering:
    def _output(self, suggestions, summary="I'd change 1 thing(s) in /."):
        return NeoOutput(plan=[PlanStep(description="trace it", rationale="r")], simulation_traces=[],
                         code_suggestions=suggestions, static_checks=[], next_questions=[],
                         confidence=None, notes="",
                         orchestrator=OrchestratorMessage(summary=summary, cautions=["unverified"]))

    def test_an_answer_shaped_suggestion_is_rendered_as_the_answer(self):
        answer = CodeSuggestion(file_path="/", unified_diff="", description="d", confidence=0.7,
                                code_block="Because the handler is the receipt boundary.\n\nFacts used: []")
        text = render_output(self._output([answer]))
        assert text.startswith("Because the handler is the receipt boundary.")
        assert "change 1 thing" not in text and "Facts used" not in text
        assert "- unverified" in text  # cautions are never dropped
        assert "### /" not in text and "Plan:" not in text

    def test_a_diffless_suggestion_on_a_real_path_is_an_answer_not_a_change(self):
        answer = CodeSuggestion(file_path="/src/neo/lattice.py", unified_diff="", description="d",
                                confidence=0.5, code_block="It declines handoffs: Neo is read-only.")
        text = render_output(self._output([answer]))
        assert text.startswith("(re /src/neo/lattice.py)\nIt declines handoffs")
        assert "change 1 thing" not in text

    def test_history_facts_show_the_commit_not_the_patch(self, tmp_path):
        from neo.memory.models import Fact

        class Store:
            def retrieve_relevant(self, query, k):
                return [Fact(subject="history:abc fix", body="Commit: abc\nMessage: fix\nFiles: a.py\nChanges:\n--- a/a.py\n+++ b/a.py\n-x\n+y")]

        text = Answerer(str(tmp_path), store_factory=Store).memory("fix")
        assert "Files: a.py" in text and "+++ b/a.py" not in text

    def test_a_real_change_keeps_summary_plan_and_diff(self):
        change = CodeSuggestion(file_path="src/a.py", unified_diff="-x\n+y", description="fix", confidence=0.9)
        text = render_output(self._output([change], summary="One fix."))
        assert text.startswith("One fix.") and "Plan:" in text and "### src/a.py" in text

    def test_memory_drops_repeated_facts(self, tmp_path):
        from neo.memory.models import Fact

        class Store:
            def retrieve_relevant(self, query, k):
                twin = dict(subject="Secret exposure", body="never commit keys")
                return [Fact(**twin), Fact(**twin), Fact(subject="Other", body="b")]

        text = Answerer(str(tmp_path), store_factory=Store).memory("secrets")
        assert text.count("Secret exposure") == 1 and "2. Other" in text
