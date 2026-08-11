"""Evidence-ledger tests for versioned learning episodes."""

from __future__ import annotations

import json

from neo.engine import NeoEngine
from neo.memory.episodes import (
    EPISODE_SCHEMA_VERSION,
    LearningEpisode,
    LearningEpisodeStore,
)
from neo.memory.models import ContextResult, Fact
from neo.models import ContextFile, NeoInput
from neo.execution_context import (
    ExecutionIdentity,
    GoalSpec,
    HypothesisRecord,
    IntentSpec,
    ValidationGate,
    ValidationObservation,
)


class _CombinedLM:
    provider = "test-provider"
    model = "test-model"

    def name(self):
        return "test-provider/test-model"

    def generate(self, messages, **kwargs):
        return """<<<NEO:SCHEMA=v3:KIND=plan>>>
[{"id":"ps_1","description":"change it","rationale":"requested","dependencies":[],"schema_version":"3"}]
<<<END:plan>>>
<<<NEO:SCHEMA=v3:KIND=simulation>>>
[{"n":1,"input_data":"x","expected_output":"y","reasoning_steps":["**NO_MODIFY**"],"issues_found":[],"schema_version":"3"}]
<<<END:simulation>>>
<<<NEO:SCHEMA=v3:KIND=code>>>
[{"file_path":"src/example.py","unified_diff":"+value = 1","code_block":"value = 1","description":"set value","confidence":0.8,"tradeoffs":[],"schema_version":"3"}]
<<<END:code>>>"""


def test_partial_legacy_record_loads_conservatively(tmp_path):
    store = LearningEpisodeStore("project", base_dir=tmp_path)
    store.path.mkdir(parents=True)
    (store.path / "old.json").write_text(json.dumps({
        "episode_id": "old",
        "objective": "legacy task",
    }))

    episode = store.load("old")

    assert episode is not None
    assert episode.schema_version == EPISODE_SCHEMA_VERSION
    assert episode.final_outcome == "pending"
    assert episode.verification == []
    assert episode.operating_mode == "learn"
    assert episode.authority == {}


def test_legacy_mutation_without_state_snapshots_loads_conservatively(tmp_path):
    store = LearningEpisodeStore("project", base_dir=tmp_path)
    store.path.mkdir(parents=True)
    (store.path / "old-mutation.json").write_text(json.dumps({
        "episode_id": "old-mutation",
        "memory_mutations": [{
            "mutation_id": "mutation-1",
            "operation": "legacy_fact_write",
            "fact_id": "fact-1",
        }],
    }))

    episode = store.load("old-mutation")

    assert episode is not None
    assert episode.memory_mutations[0].before_state == {}
    assert episode.memory_mutations[0].after_state == {}


def test_partial_nested_evidence_entries_load_conservatively(tmp_path):
    """A verification/suggestion/candidate entry missing required fields must
    load with conservative defaults, not quarantine the whole episode."""
    from neo.memory.episodes import aggregate_verification_status

    store = LearningEpisodeStore("project", base_dir=tmp_path)
    store.path.mkdir(parents=True)
    (store.path / "partial.json").write_text(json.dumps({
        "episode_id": "partial",
        "verification": [{"kind": "test"}],            # no verification_id, no status
        "suggestions": [{"suggestion_id": "s1"}],       # no file_path/description/confidence
        "memory_candidates": [{"candidate_id": "c1"}],  # no suggestion_id/subject/body/kind
    }))

    episode = store.load("partial")

    assert episode is not None                          # not quarantined
    assert episode.verification[0].kind == "test"
    assert episode.verification[0].status == "unavailable"  # conservative
    # Fail-closed: a status-less verification never reads as passed.
    assert aggregate_verification_status(episode.verification) != "passed"
    assert episode.suggestions[0].suggestion_id == "s1"
    assert episode.suggestions[0].confidence == 0.0
    assert episode.memory_candidates[0].candidate_id == "c1"


def test_future_extra_fields_are_dropped(tmp_path):
    """Unknown keys from a future schema are dropped, not raised on (which would
    quarantine the record)."""
    store = LearningEpisodeStore("project", base_dir=tmp_path)
    store.path.mkdir(parents=True)
    (store.path / "future.json").write_text(json.dumps({
        "episode_id": "future",
        "suggestions": [{
            "suggestion_id": "s1", "file_path": "a.py", "description": "d",
            "confidence": 0.5, "future_field_xyz": "ignored",
        }],
        "verification": [{
            "verification_id": "v1", "kind": "test", "status": "passed",
            "another_new_field": 42,
        }],
    }))

    episode = store.load("future")

    assert episode is not None
    assert episode.suggestions[0].suggestion_id == "s1"
    assert not hasattr(episode.suggestions[0], "future_field_xyz")
    assert episode.verification[0].status == "passed"


def test_corrupt_bucket_is_bounded(tmp_path, monkeypatch):
    """Quarantined .corrupt-* files are capped, not accumulated indefinitely."""
    import neo.memory.episodes as ep_mod
    monkeypatch.setattr(ep_mod, "MAX_CORRUPT_PER_PROJECT", 3)
    store = LearningEpisodeStore("project", base_dir=tmp_path)
    store.path.mkdir(parents=True)
    for i in range(6):
        (store.path / f"bad-{i}.json").write_text("{not json")
        assert store.load(f"bad-{i}") is None  # each load quarantines
    assert len(list(store.path.glob("*.corrupt-*"))) <= 3


def test_malformed_record_is_preserved_and_skipped(tmp_path):
    store = LearningEpisodeStore("project", base_dir=tmp_path)
    store.path.mkdir(parents=True)
    target = store.path / "broken.json"
    target.write_text("{not-json")

    assert store.load("broken") is None
    assert not target.exists()
    assert list(store.path.glob("broken.json.corrupt-*"))


def test_store_is_bounded(tmp_path, monkeypatch):
    from neo.memory import episodes as episode_module

    monkeypatch.setattr(episode_module, "MAX_EPISODES_PER_PROJECT", 2)
    store = LearningEpisodeStore("project", base_dir=tmp_path)
    for i in range(3):
        store.save(LearningEpisode(episode_id=f"ep-{i}", started_at=float(i)))

    assert len(list(store.path.glob("*.json"))) == 2


def test_engine_persists_trace_without_raw_source_or_verification_claim(tmp_path, monkeypatch):
    engine = NeoEngine(
        lm_adapter=_CombinedLM(),
        enable_persistent_memory=False,
        codebase_root=str(tmp_path),
    )
    monkeypatch.setattr(engine, "_car_route_capability", lambda prompt: (False, 0, None))
    monkeypatch.setattr(engine, "_run_static_checks", lambda suggestions, constraints=None: [])
    source = "SECRET_SOURCE_CONTENT = 'local-only'"

    output = engine.process(NeoInput(
        prompt="set the example value",
        context_files=[ContextFile(path="src/example.py", content=source)],
        working_directory=str(tmp_path),
    ))

    episode_id = output.metadata["learning_episode_id"]
    episode = engine.episode_store.load(episode_id)
    assert episode is not None
    assert episode.session_id and episode.task_id
    assert episode.provider == "test-provider"
    assert episode.model == "test-model"
    assert episode.reasoning_mode == "fast"
    assert episode.suggestions[0].suggestion_id
    assert episode.suggestions[0].code_sha256
    assert episode.verification[0].status == "skipped"
    assert episode.final_outcome == "suggested_pending_downstream_outcome"

    persisted = (engine.episode_store.path / f"{episode_id}.json").read_text()
    assert source not in persisted
    assert "value = 1" not in persisted


def test_future_schema_fails_safely(tmp_path):
    store = LearningEpisodeStore("project", base_dir=tmp_path)
    store.path.mkdir(parents=True)
    (store.path / "future.json").write_text(json.dumps({
        "episode_id": "future",
        "schema_version": EPISODE_SCHEMA_VERSION + 1,
    }))

    assert store.load("future") is None


def test_legacy_static_check_kind_is_normalized():
    episode = LearningEpisode.from_dict({
        "verification": [{
            "verification_id": "v1",
            "kind": "static_check",
            "status": "passed",
        }],
    })

    assert episode.verification[0].kind == "neo_static"

    downstream = LearningEpisode.from_dict({
        "schema_version": 1,
        "verification": [{
            "verification_id": "v2",
            "kind": "downstream_outcome",
            "status": "failed",
        }],
    })
    assert downstream.verification[0].kind == "user_modification"


def test_verification_aggregate_is_fail_closed():
    from neo.memory.episodes import VerificationEvidence, aggregate_verification_status

    evidence = [
        VerificationEvidence("v1", "lint", "passed"),
        VerificationEvidence("v2", "test", "unavailable"),
    ]

    assert aggregate_verification_status(evidence) == "unavailable"
    evidence.append(VerificationEvidence("v3", "compile", "failed"))
    assert aggregate_verification_status(evidence) == "failed"
    assert aggregate_verification_status([
        VerificationEvidence("v4", "device", "waived"),
        VerificationEvidence("v5", "test", "passed"),
    ]) == "waived"


def test_retrieved_fact_score_and_context_inclusion_are_traced():
    engine = NeoEngine(lm_adapter=_CombinedLM(), enable_persistent_memory=False)
    engine.current_learning_episode = LearningEpisode()
    fact = Fact(id="fact-1", subject="Convention", body="Use typed IDs")
    context = ContextResult(valid_facts=[fact], retrieval_scores={fact.id: 0.875})

    engine._capture_retrieval_context(context, included=False)
    engine._capture_retrieval_context(context, included=True)

    evidence = engine.current_learning_episode.retrieved_facts
    assert len(evidence) == 1
    assert evidence[0].fact_id == "fact-1"
    assert evidence[0].score == 0.875
    assert evidence[0].included_in_context is True


def test_only_explicit_fact_citations_are_marked_used():
    from neo.models import CodeSuggestion, PlanStep, SimulationTrace

    engine = NeoEngine(lm_adapter=_CombinedLM(), enable_persistent_memory=False)
    engine.current_learning_episode = LearningEpisode()
    facts = [
        Fact(id="fact-used", subject="Convention", body="Use typed IDs"),
        Fact(id="fact-only-retrieved", subject="Other", body="Use UTC"),
    ]
    context = ContextResult(
        valid_facts=facts,
        retrieval_scores={fact.id: 0.8 for fact in facts},
    )
    engine._capture_retrieval_context(context, included=True)

    engine._capture_detectable_fact_use(
        [PlanStep(description="Apply convention", rationale="Use [fact:fact-used]")],
        [SimulationTrace("input", "output", [])],
        [CodeSuggestion("src/a.py", "", "Change IDs", 0.8)],
    )

    by_id = {item.fact_id: item for item in engine.current_learning_episode.retrieved_facts}
    assert by_id["fact-used"].used_in_reasoning is True
    assert by_id["fact-only-retrieved"].used_in_reasoning is False


def test_detectable_fact_use_accumulates_across_passes():
    """OR-accumulate: a citation credited in an earlier reasoning pass stays
    credited even if a later pass's artifacts don't repeat it (previously the
    second call clobbered the first's True back to False)."""
    from neo.models import CodeSuggestion, PlanStep, SimulationTrace

    engine = NeoEngine(lm_adapter=_CombinedLM(), enable_persistent_memory=False)
    engine.current_learning_episode = LearningEpisode()
    facts = [Fact(id="fact-used", subject="Convention", body="Use typed IDs")]
    context = ContextResult(valid_facts=facts, retrieval_scores={"fact-used": 0.8})
    engine._capture_retrieval_context(context, included=True)

    # Pass 1: citation present -> credited True.
    engine._capture_detectable_fact_use(
        [PlanStep(description="Apply", rationale="Use [fact:fact-used]")],
        [SimulationTrace("in", "out", [])],
        [CodeSuggestion("src/a.py", "", "Change", 0.8)],
    )
    evidence = engine.current_learning_episode.retrieved_facts[0]
    assert evidence.used_in_reasoning is True

    # Pass 2: no citation in this pass's artifacts -> must NOT reset to False.
    engine._capture_detectable_fact_use(
        [PlanStep(description="Unrelated", rationale="no citation here")],
        [SimulationTrace("in", "out", [])],
        [CodeSuggestion("src/b.py", "", "Other", 0.8)],
    )
    assert evidence.used_in_reasoning is True  # accumulated, not clobbered


def test_structured_facts_used_self_report_credits_fact():
    """T10: a 'Facts used: [id]' self-report credits the fact even when the
    inline [fact:id] marker is absent — the primary detection signal, since
    production LLMs rarely echo the machine marker."""
    from neo.models import CodeSuggestion, PlanStep, SimulationTrace

    engine = NeoEngine(lm_adapter=_CombinedLM(), enable_persistent_memory=False)
    engine.current_learning_episode = LearningEpisode()
    facts = [Fact(id="fact-x", subject="Convention", body="Use typed IDs")]
    engine._capture_retrieval_context(
        ContextResult(valid_facts=facts, retrieval_scores={"fact-x": 0.8}), included=True)
    engine._capture_detectable_fact_use(
        [PlanStep(description="Apply it", rationale="Applied. Facts used: [fact-x]")],
        [SimulationTrace("in", "out", [])],
        [CodeSuggestion("src/a.py", "", "Change", 0.8)],
    )
    assert engine.current_learning_episode.retrieved_facts[0].used_in_reasoning is True


def test_subject_overlap_credits_used_fact_without_marker():
    """T10: conservative subject overlap credits a fact the reasoning clearly
    used, and leaves an unrelated retrieved fact uncredited."""
    from neo.models import CodeSuggestion, PlanStep, SimulationTrace

    engine = NeoEngine(lm_adapter=_CombinedLM(), enable_persistent_memory=False)
    engine.current_learning_episode = LearningEpisode()
    engine._retrieved_fact_texts = {}
    facts = [
        Fact(id="used", subject="validate credentials before dispatch", body="x"),
        Fact(id="unrelated", subject="rotate encryption keys quarterly", body="y"),
    ]
    engine._capture_retrieval_context(
        ContextResult(valid_facts=facts,
                      retrieval_scores={"used": 0.8, "unrelated": 0.8}), included=True)
    engine._capture_detectable_fact_use(
        [PlanStep(description="validate credentials before dispatch",
                  rationale="validate the credentials on dispatch")],
        [SimulationTrace("in", "out", [])],
        [CodeSuggestion("src/a.py", "", "add credential validation before dispatch", 0.8)],
    )
    by_id = {e.fact_id: e for e in engine.current_learning_episode.retrieved_facts}
    assert by_id["used"].used_in_reasoning is True
    assert by_id["unrelated"].used_in_reasoning is False


def test_overlap_ignores_trivial_single_token_subject():
    """T10: overlap requires >=2 significant tokens, so a one-word subject can't
    fabricate use from an incidental word match."""
    from neo.models import CodeSuggestion, PlanStep, SimulationTrace

    engine = NeoEngine(lm_adapter=_CombinedLM(), enable_persistent_memory=False)
    engine.current_learning_episode = LearningEpisode()
    engine._retrieved_fact_texts = {}
    facts = [Fact(id="trivial", subject="Testing", body="x")]
    engine._capture_retrieval_context(
        ContextResult(valid_facts=facts, retrieval_scores={"trivial": 0.8}), included=True)
    engine._capture_detectable_fact_use(
        [PlanStep(description="testing testing", rationale="all about testing")],
        [SimulationTrace("in", "out", [])],
        [CodeSuggestion("a.py", "", "testing", 0.8)],
    )
    assert engine.current_learning_episode.retrieved_facts[0].used_in_reasoning is False


def test_failed_verification_is_associated_only_with_used_retrieval(tmp_path):
    from neo.memory.episodes import RetrievedFactEvidence
    from neo.models import StaticCheckResult

    engine = NeoEngine(
        lm_adapter=_CombinedLM(),
        enable_persistent_memory=False,
        codebase_root=str(tmp_path),
    )
    episode = LearningEpisode(project_id=engine.episode_store.project_id)
    episode.retrieved_facts = [
        RetrievedFactEvidence("used", included_in_context=True, used_in_reasoning=True),
        RetrievedFactEvidence("unused", included_in_context=True, used_in_reasoning=False),
    ]
    engine.current_learning_episode = episode

    engine._complete_learning_episode(
        code_suggestions=[],
        static_checks=[StaticCheckResult(
            tool_name="pytest",
            diagnostics=[{"severity": "error"}],
            summary="failed",
            kind="test",
            status="failed",
        )],
        reasoning_fact=None,
        simulation_facts=[],
        metadata={},
    )

    assert episode.retrieved_facts[0].outcome_association == "failure"
    assert episode.retrieved_facts[1].outcome_association == ""


def test_objective_credentials_are_redacted(tmp_path, monkeypatch):
    engine = NeoEngine(
        lm_adapter=_CombinedLM(),
        enable_persistent_memory=False,
        codebase_root=str(tmp_path),
    )
    monkeypatch.setattr(engine, "_car_route_capability", lambda prompt: (False, 0, None))
    monkeypatch.setattr(engine, "_run_static_checks", lambda suggestions, constraints=None: [])
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"

    output = engine.process(NeoInput(prompt=f"Use token {secret}"))

    episode = engine.episode_store.load(output.metadata["learning_episode_id"])
    assert episode is not None
    assert secret not in episode.objective
    assert "[REDACTED]" in episode.objective


def test_execution_envelope_is_persisted_with_explicit_inference_provenance(
    tmp_path, monkeypatch,
):
    engine = NeoEngine(
        lm_adapter=_CombinedLM(),
        enable_persistent_memory=False,
        codebase_root=str(tmp_path),
    )
    monkeypatch.setattr(engine, "_car_route_capability", lambda prompt: (False, 0, None))
    monkeypatch.setattr(engine, "_run_static_checks", lambda suggestions, constraints=None: [])

    output = engine.process(NeoInput(
        prompt="Tests still fail",
        goal=GoalSpec("All auth tests pass"),
        intent=IntentSpec("diagnose_failed_attempt"),
        current_state={"source_code": "SECRET_SOURCE_PAYLOAD"},
    ))

    episode = engine.episode_store.load(output.metadata["learning_episode_id"])
    assert episode.schema_version == EPISODE_SCHEMA_VERSION == 4
    assert episode.execution_context["goal"]["origin"] == "explicit"
    assert episode.execution_context["intent"]["origin"] == "explicit"
    assert "SECRET_SOURCE_PAYLOAD" not in json.dumps(episode.execution_context)
    assert episode.execution_context["current_state"]["source_code"]["sha256"]
    assert output.goal_assessment.status == "in_progress"


def test_v4_engine_episode_preserves_identity_gates_and_hypotheses(
    tmp_path, monkeypatch,
):
    engine = NeoEngine(
        lm_adapter=_CombinedLM(),
        enable_persistent_memory=False,
        codebase_root=str(tmp_path),
    )
    monkeypatch.setattr(engine, "_car_route_capability", lambda prompt: (False, 0, None))
    monkeypatch.setattr(engine, "_run_static_checks", lambda suggestions, constraints=None: [])

    output = engine.process(NeoInput(
        prompt="Diagnose the boundary",
        validation_gates=[ValidationGate("probe", "Run the causal probe")],
        validation_observations=[ValidationObservation(
            "probe-pass", "probe", "passed", summary="probe passed",
        )],
        hypotheses=[HypothesisRecord(
            "h1", "Duplicate injection is causal", status="confirmed",
            falsifying_test="Remove the second injection",
            supporting_observation_ids=["probe-pass"],
            public_claim_safe=True,
        )],
        execution_identity=ExecutionIdentity(
            session_id="session-external",
            goal_id="goal-commercial",
            task_id="task-deadpan",
            parent_task_id="task-commercial",
            trace_id="trace-1",
            discovery_source="user_path_review",
            repositories_touched=["studio", "runtime"],
            artifact_refs=["movie-output.mov"],
        ),
    ))

    episode = engine.episode_store.load(output.metadata["learning_episode_id"])
    assert episode.session_id == "session-external"
    assert episode.task_id == "task-deadpan"
    assert episode.parent_task_id == "task-commercial"
    assert episode.validation_gates[0].gate_id == "probe"
    assert episode.verification[0].gate_id == "probe"
    assert episode.hypotheses[0].status == "confirmed"
    assert episode.memory_candidates == []
    assert episode.artifact_refs[0] != "movie-output.mov"
    assert "movie-output.mov" not in json.dumps(episode.to_dict())
    assert output.validation_assessment.passed == 1
    assert output.strategy_assessment.decision == "stop_success"


class TestSuggestionVerifiability:
    """Promotion is gated on a git-verified outcome, which is detected by
    diffing the suggested file. A suggestion whose path doesn't resolve can
    never be verified — calling it promotable leaves the ledger accruing
    permanently-pending episodes. Measured live: only 23 of 65 recorded
    suggestions were verifiable at all.
    """

    def _engine(self, root):
        from neo.engine import NeoEngine
        eng = NeoEngine.__new__(NeoEngine)
        eng.codebase_root = str(root) if root else None
        return eng

    def _sugg(self, file_path, code="x = 1"):
        import types
        return types.SimpleNamespace(file_path=file_path, code_block=code,
                                     unified_diff="")

    def test_real_relative_file_with_code_is_verifiable(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 0")
        eng = self._engine(tmp_path)
        assert eng._suggestion_is_verifiable(self._sugg("app.py")) is True

    def test_real_absolute_file_is_verifiable(self, tmp_path):
        target = tmp_path / "app.py"
        target.write_text("x = 0")
        eng = self._engine(None)
        assert eng._suggestion_is_verifiable(self._sugg(str(target))) is True

    def test_model_invented_pseudo_path_is_not_verifiable(self, tmp_path):
        """The dominant real-world case: advisory prompts get a topical path.

        These are rejected because the invented parent directory ("review/",
        "architecture-review/") does not exist in the repo.
        """
        eng = self._engine(tmp_path)
        for bogus in ("/review/commit-840d4b625d5d.md",
                      "/architecture-review/notification-idempotency.md",
                      "/REVIEW_ONLY/no_code_change.md"):
            assert eng._suggestion_is_verifiable(self._sugg(bogus)) is False

    def test_known_limit_root_level_sentinel_is_admitted(self, tmp_path):
        """A bare-slash name at repo ROOT can't be told from a real new file.

        "/NO_CODE_PLANNING_ONLY" and "/README.md" are structurally identical:
        both normalize to a repo-root path whose parent exists. We admit them
        deliberately — under-admitting is unrecoverable because `kind` is frozen
        when the candidate is minted, so a wrongly-rejected suggestion can never
        promote; over-admitting only leaves a candidate pending, which is the
        pre-existing behavior. The >=2 verified-acceptance gate is what actually
        keeps unverified material out of the fact store.
        """
        eng = self._engine(tmp_path)
        assert eng._suggestion_is_verifiable(self._sugg("/NO_CODE_PLANNING_ONLY")) is True

    def test_bare_leading_slash_real_file_is_verifiable(self, tmp_path):
        """`/src/foo.js` under codebase_root is a REAL, verifiable file.

        Attribution normalizes bare-leading-slash paths (a common shape in
        recorded suggestions) before diffing. An independent resolver here that
        treated them as absolute rejected two genuinely promotable candidates on
        live data — this is the case that catches that class of drift.
        """
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "analyzer.js").write_text("var x = 1;")
        eng = self._engine(tmp_path)
        assert eng._suggestion_is_verifiable(self._sugg("/src/analyzer.js")) is True

    def test_plausible_new_file_in_repo_is_verifiable(self, tmp_path):
        """Proposing a NEW file is a real suggestion — once committed it shows
        up in `git log --since`, so it must stay promotable."""
        (tmp_path / "src").mkdir()
        eng = self._engine(tmp_path)
        assert eng._suggestion_is_verifiable(self._sugg("src/newmod.py")) is True

    def test_new_file_in_missing_directory_is_not_verifiable(self, tmp_path):
        eng = self._engine(tmp_path)
        assert eng._suggestion_is_verifiable(self._sugg("nope/newmod.py")) is False

    def test_new_file_outside_the_repo_is_not_verifiable(self, tmp_path):
        """`/dev/null` has an existing parent but is not the repo's business."""
        eng = self._engine(tmp_path / "repo")
        (tmp_path / "repo").mkdir()
        assert eng._suggestion_is_verifiable(self._sugg("/dev/null")) is False
        assert eng._suggestion_is_verifiable(self._sugg(str(tmp_path / "outside.py"))) is False

    def test_directory_is_not_verifiable(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        eng = self._engine(tmp_path)
        assert eng._suggestion_is_verifiable(self._sugg("pkg")) is False

    def test_real_file_without_code_is_not_verifiable(self, tmp_path):
        """Nothing to diff against — the outcome could only ever be UNVERIFIED."""
        (tmp_path / "app.py").write_text("x = 0")
        eng = self._engine(tmp_path)
        assert eng._suggestion_is_verifiable(self._sugg("app.py", code="")) is False

    def test_relative_path_without_codebase_root_is_not_verifiable(self, tmp_path):
        eng = self._engine(None)
        assert eng._suggestion_is_verifiable(self._sugg("app.py")) is False

    def test_placeholder_paths_rejected(self, tmp_path):
        eng = self._engine(tmp_path)
        for bogus in ("", "/", "N/A"):
            assert eng._suggestion_is_verifiable(self._sugg(bogus)) is False
