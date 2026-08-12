"""`--dry-run` must show what would actually be sent, and change nothing.

The flag used to exit in `cli.main` before the engine was constructed, so it
could only print the file-gathering result while `CLAUDE.md` told operators it
printed "file selection, fact retrieval, constraints, four-layer assembly".
Three of those four never ran. Since `--dry-run` is the instrument the project
tells people to use before believing any claim about what Neo saw, an
instrument that under-reports sends the operator to the wrong knob.

Two properties are load-bearing and pinned here:

1. What is printed is what would be SENT -- recorded off the adapter, never
   rebuilt. A renderer that walked the context dict would be a second
   implementation of the prompt builders, free to drift the moment one
   changed.
2. A dry run mutates nothing. The old implementation had that property for
   free by never reaching the engine; the new one has to earn it, because
   retrieval marks facts accessed and `detect_implicit_feedback` both mutates
   confidence and saves.
"""

import json
import subprocess
import sys

import pytest

from neo.dry_run import DryRunComplete, RecordingLM, render


class TestRecordingAdapter:
    def test_generate_records_the_messages_and_stops(self):
        lm = RecordingLM()
        messages = [{"role": "system", "content": "sys"},
                    {"role": "user", "content": "usr"}]

        with pytest.raises(DryRunComplete) as excinfo:
            lm.generate(messages, max_tokens=99, temperature=0.3)

        recorded = excinfo.value.calls[0]
        assert recorded["messages"] == messages
        assert recorded["max_tokens"] == 99
        assert recorded["temperature"] == 0.3

    def test_dry_run_complete_is_not_an_exception(self):
        """`_process_guarded` wraps the run in `except Exception` and turns
        anything it catches into a FAILED lifecycle event. A dry run is not a
        crash, and reporting it as one would be a third way of lying about
        what happened.
        """
        assert issubclass(DryRunComplete, BaseException)
        assert not issubclass(DryRunComplete, Exception)

    def test_a_broad_handler_does_not_swallow_it(self):
        """The property above, exercised rather than asserted about."""
        caught = None
        try:
            try:
                RecordingLM().generate([{"role": "user", "content": "x"}])
            except Exception as exc:  # noqa: BLE001 - the point of the test
                caught = f"swallowed by except Exception: {exc}"
        except DryRunComplete:
            caught = "propagated"
        assert caught == "propagated"


class TestRender:
    def test_renders_every_message_in_full(self):
        """No truncation in the viewer.

        This output exists so an operator can see what the model got. A viewer
        that silently cut its own output would reproduce, one level up, the
        exact defect the `text_budget` markers were introduced to fix.
        """
        body = "x" * 50_000
        out = render([{"messages": [{"role": "user", "content": body}],
                       "max_tokens": 1, "temperature": 0.0,
                       "reasoning_effort": None, "stop": None}])
        assert body in out
        assert "50,000 chars" in out

    def test_no_calls_says_so_rather_than_printing_nothing(self):
        out = render([])
        assert "No LM call was assembled" in out


def _run(args, timeout=300):
    return subprocess.run(
        [sys.executable, "-m", "neo", *args],
        capture_output=True, text=True, timeout=timeout,
    )


class TestEndToEnd:
    """Exercised through the real CLI: these are integration properties."""

    def test_dry_run_prints_the_assembled_prompt_not_just_files(self, tmp_path):
        (tmp_path / "app.py").write_text("def handler():\n    return 1\n")
        result = _run(["--dry-run", "fix the handler", "--cwd", str(tmp_path)])

        assert result.returncode == 0
        # The file list survives -- it carries the SCORE, which is what an
        # operator needs when the right file is missing.
        assert "files selected, in rank order" in result.stderr
        # ...and the four things the docs promised and did not deliver.
        assert "messages Neo hands the adapter" in result.stderr
        assert "Execution Envelope" in result.stderr
        assert "message 1/" in result.stderr

    def test_dry_run_makes_no_real_lm_call(self, tmp_path):
        """A bogus key would fail loudly against a real provider."""
        (tmp_path / "app.py").write_text("def handler():\n    return 1\n")
        result = _run(["--dry-run", "fix the handler", "--cwd", str(tmp_path)])
        assert result.returncode == 0
        assert "AuthenticationError" not in result.stderr

    def test_no_scan_does_not_crash(self, tmp_path):
        """`gathered` is only bound inside the gather branch; `--no-scan`
        skips it, and the dry-run report below still has to say something."""
        result = _run(["--dry-run", "--no-scan", "hello", "--cwd", str(tmp_path)])
        assert result.returncode == 0
        assert "NameError" not in result.stderr

    def test_json_input_dry_run_also_stops_before_inference(self, tmp_path):
        """The old early-exit lived inside the plain-text branch only, so
        `--stdin-json --dry-run` fell through and made a REAL call. The stop
        now lives at engine construction, which both modes share.
        """
        payload = json.dumps({
            "prompt": "fix the handler",
            "task_type": "bugfix",
            "context_files": [{"path": "app.py", "content": "def handler(): return 1"}],
            "working_directory": str(tmp_path),
        })
        result = subprocess.run(
            [sys.executable, "-m", "neo", "--stdin-json", "--dry-run"],
            input=payload, capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0
        assert "messages Neo hands the adapter" in result.stderr

    def test_dry_run_leaves_the_fact_files_byte_identical(self, tmp_path):
        """A coarse backstop, and labelled as one.

        This compares the fact files across two dry runs, so it catches any
        NEW writer that appears on the path. It does NOT prove the
        `detect_implicit_feedback` skip works -- that call is a no-op without
        a prior session, so on a fresh HOME it writes nothing either way.
        Measured: deleting the `not self.dry_run` guard leaves this test
        GREEN. `FactStore(read_only=True)` is the real guard -- it stops the
        writer this test cannot see, `initialize()`'s prune/demote/purge,
        which needs an AGED store to fire and so never fires on the cold
        empty one two fresh runs produce. This is a net, not a hook.
        """
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        (tmp_path / "app.py").write_text("def handler():\n    return 1\n")

        env = {**dict(__import__("os").environ), "HOME": str(home)}
        args = [sys.executable, "-m", "neo", "--dry-run", "fix the handler",
                "--cwd", str(tmp_path)]

        subprocess.run(args, capture_output=True, text=True, timeout=300, env=env)
        facts_dir = home / ".neo" / "facts"
        before = {
            p.name: p.read_bytes()
            for p in (facts_dir.glob("*.json") if facts_dir.exists() else [])
        }

        subprocess.run(args, capture_output=True, text=True, timeout=300, env=env)
        after = {
            p.name: p.read_bytes()
            for p in (facts_dir.glob("*.json") if facts_dir.exists() else [])
        }

        assert after == before, (
            "a dry run rewrote the fact store; it must assemble and report, "
            "never persist"
        )


class TestSideEffectsAreSkipped:
    """The one call before inference that mutates confidence AND saves.

    `--dry-run` used to exit in the CLI before the engine existed, so it had
    no side effects at all. Gaining them while fixing what the flag reports
    would be a poor trade: an operator runs a dry run precisely when they do
    not want to perturb the thing they are inspecting.

    Asserted with a spy rather than by diffing files, because
    `detect_implicit_feedback` is a no-op without a prior session -- the
    file-level check stays green whether the skip works or not, which makes
    it a backstop and not a guard.
    """

    def _engine(self, tmp_path, *, dry_run):
        from unittest.mock import MagicMock

        from neo.engine import NeoEngine

        engine = NeoEngine(
            lm_adapter=RecordingLM(),
            enable_persistent_memory=False,
            codebase_root=str(tmp_path),
            dry_run=dry_run,
        )
        engine.persistent_memory = MagicMock()
        engine.persistent_memory.retrieve_relevant.return_value = []
        engine.persistent_memory.build_context.return_value = None
        return engine

    def _run(self, engine, tmp_path):
        from neo.models import NeoInput

        try:
            engine.process(NeoInput(
                prompt="fix the handler",
                context_files=[],
                working_directory=str(tmp_path),
            ))
        except (DryRunComplete, Exception):  # noqa: BLE001
            # Either the dry-run stop or a downstream parse failure on the
            # RecordingLM path -- neither is what this test measures.
            pass

    def test_dry_run_skips_the_one_call_that_saves(self, tmp_path):
        engine = self._engine(tmp_path, dry_run=True)
        self._run(engine, tmp_path)
        engine.persistent_memory.detect_implicit_feedback.assert_not_called()

    def test_a_normal_run_still_makes_it(self, tmp_path):
        """The other half. Without this, deleting the whole `if` block --
        not just the `and not self.dry_run` clause -- would pass the test
        above while silently disabling implicit feedback for every run.
        """
        engine = self._engine(tmp_path, dry_run=False)
        self._run(engine, tmp_path)
        engine.persistent_memory.detect_implicit_feedback.assert_called_once()


class TestNoCredentialsRequired:
    def test_dry_run_needs_no_api_key_even_with_scanning(self, tmp_path):
        """`--dry-run` makes no call, so requiring a key to find out what Neo
        would have sent is a barrier with nothing behind it.

        `test_no_scan_dry_run_does_not_require_api_key` in
        test_cli_global_flags.py covers the `--no-scan` path and caught this
        when moving the stop into the engine pushed it past adapter
        construction. This covers the scanning path, which is the one people
        actually use.
        """
        import os

        (tmp_path / "app.py").write_text("def handler():\n    return 1\n")
        env = {**os.environ, "NEO_SKIP_UPDATE_CHECK": "1"}
        for key in ("NEO_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                    "GOOGLE_API_KEY"):
            env.pop(key, None)

        result = subprocess.run(
            [sys.executable, "-m", "neo", "--dry-run", "fix the handler",
             "--cwd", str(tmp_path)],
            capture_output=True, text=True, env=env, timeout=300,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "API key required" not in result.stdout + result.stderr
        assert "messages Neo hands the adapter" in result.stderr


class TestPanelIsSuppressed:
    """`--dry-run` must not reach `create_adapter("car", ...)`.

    The panel does not reason through `self.lm`: `_build_car_role_factory`
    builds a real CAR adapter per role and uses `self.lm` only as the fallback
    for unassigned roles. So `RecordingLM` does not intercept it. On a machine
    with `car-server` reachable -- this project's documented normal setup,
    since the observer autostarts off it -- a novel prompt under `--dry-run`
    would have run the full panel against real models, spent real money, never
    raised `DryRunComplete`, and printed ordinary output. The flag would have
    been a no-op that bills you.

    Measured before the fix: 4 `create_adapter("car", ...)` calls (planner,
    coder, critic, judge).
    """

    def test_no_car_adapter_is_built(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from neo.engine import NeoEngine
        from neo.models import NeoInput

        built = []
        reached_the_stop = False
        with patch("neo.adapters.create_adapter",
                   side_effect=lambda *a, **k: (built.append(a), MagicMock())[1]):
            engine = NeoEngine(lm_adapter=RecordingLM(),
                               codebase_root=str(tmp_path), dry_run=True)
            # Force the panel to look fully available.
            with patch.object(NeoEngine, "_car_route_capability",
                              return_value=(True, 5, lambda *a, **k: {})):
                try:
                    engine.process(NeoInput(
                        prompt="design a novel distributed consensus scheme",
                        context_files=[], working_directory=str(tmp_path)))
                except DryRunComplete:
                    reached_the_stop = True
                except Exception:  # noqa: BLE001
                    pass

        assert built == [], f"dry run built real adapters: {built}"
        # Without this, `assert built == []` also passes when `process()` dies
        # on its first line -- one unrelated regression from being green for
        # entirely the wrong reason.
        assert reached_the_stop, "the run never reached the dry-run stop"

    def test_mode_is_forced_fast(self, tmp_path):
        from unittest.mock import patch

        from neo.engine import NeoEngine
        from neo.reasoning_mode import ReasoningMode

        engine = NeoEngine(lm_adapter=RecordingLM(),
                           codebase_root=str(tmp_path), dry_run=True)
        with patch.object(NeoEngine, "_car_route_capability",
                          return_value=(True, 5, lambda *a, **k: {})):
            decision, route_fn = engine._decide_reasoning_mode({}, "hard", None)

        assert decision.mode is ReasoningMode.FAST
        assert route_fn is None

    def test_the_operator_is_told_the_panel_is_never_previewed(self, tmp_path, capsys):
        """Say it, but do not overclaim it.

        Silence would reproduce one level up the defect this flag exists to
        expose: fast-path output on a machine that would have deliberated is
        a report about a run Neo would not have made.

        The obvious version gated the note on `car_available` and announced a
        "suppressed panel" on any car-server machine -- including for the
        familiar, low-effort prompts that take the fast path regardless. That
        is the failure `shown_of` already forbids: a marker that fires when
        nothing was dropped trains the reader to ignore it. Knowing the real
        answer needs `capable_model_count` and the effort tier, i.e. a CAR
        daemon round-trip taken only to decide whether to print a string,
        under a flag that promises no calls. So the note is unconditional and
        states only what is unconditionally true.
        """
        from unittest.mock import patch

        from neo.engine import NeoEngine

        engine = NeoEngine(lm_adapter=RecordingLM(),
                           codebase_root=str(tmp_path), dry_run=True)

        for capability in ((True, 5, lambda *a, **k: {}), (False, 0, None)):
            with patch.object(NeoEngine, "_car_route_capability",
                              return_value=capability) as probe:
                engine._decide_reasoning_mode({"prompt": "x"}, "hard", None)
            note = capsys.readouterr().err
            assert "always takes the fast path" in note
            # ...and never asserts that a panel WOULD have run here.
            assert "suppressed" not in note
            probe.assert_not_called()


class TestJsonDryRunKeepsItsContract:
    """`--json` promises stdout is exactly ONE JSON document and every run
    ends with `completed` or `FAILED`. Writing the report to stderr broke both
    at once: zero documents on stdout, no terminal event, and every source
    line beginning with `{` became a counterfeit event for a host parsing the
    JSONL stream by that prefix."""

    def test_stdout_is_one_json_document_and_the_stream_terminates(self, tmp_path):
        (tmp_path / "app.py").write_text("def handler():\n    return 1\n")
        result = _run(["--dry-run", "--json", "fix the handler",
                       "--cwd", str(tmp_path)])

        assert result.returncode == 0
        payload = json.loads(result.stdout)          # raises if not exactly one
        assert payload["dry_run"] is True
        assert payload["calls"]

        events = [json.loads(line) for line in result.stderr.splitlines()
                  if line.startswith("{")]
        assert any(e.get("type") == "completed" for e in events), (
            "STARTED-then-silence is what a host cannot tell from a crash"
        )

    def test_no_prompt_content_leaks_into_the_event_stream(self, tmp_path):
        (tmp_path / "app.py").write_text('def handler():\n    return {"a": 1}\n')
        result = _run(["--dry-run", "--json", "fix the handler",
                       "--cwd", str(tmp_path)])

        for line in result.stderr.splitlines():
            if line.startswith("{"):
                json.loads(line)  # every `{`-prefixed line is a real event
