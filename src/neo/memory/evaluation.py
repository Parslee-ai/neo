"""Deterministic evaluation harness for Neo's evidence-learning loop.

The harness exercises the real FactStore retrieval, episode persistence,
candidate promotion, attributed regression, project scoping, and explanation
paths. It uses a fixed local hashing embedder and synthetic repository roots so
results require neither an LM, network access, CAR, nor the user's memory.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

from neo.memory.episodes import (
    ContextSelection,
    LearningEpisode,
    LearningEpisodeStore,
    MemoryCandidateEvidence,
    RetrievedFactEvidence,
    SuggestionEvidence,
    VerificationEvidence,
)
from neo.memory.explain import explain_fact
from neo.memory.models import Fact, FactKind, FactScope, Provenance, update_effectiveness
from neo.memory.outcomes import Outcome, OutcomeType
from neo.memory.store import FactStore

DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[3] / "benchmarks" / "learning_loop_v1.json"
)
REPORT_SCHEMA_VERSION = 1


class EvaluationMode(str, Enum):
    """Comparison policies required by the milestone."""

    DISABLED = "memory_disabled"
    LEGACY = "legacy_immediate_memory"
    EVIDENCE = "evidence_driven"


class DeterministicHashEmbedder:
    """Small stable local embedder for benchmark semantics, not production."""

    dimensions = 128

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
        return [token for token in normalized.split() if token]

    def embed(self, texts: list[str]):
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=np.float32)
            for token in self._tokens(text):
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:2], "big") % self.dimensions
                sign = 1.0 if digest[2] % 2 == 0 else -1.0
                vector[index] += sign
            norm = float(np.linalg.norm(vector))
            if norm:
                vector /= norm
            yield vector


@dataclass
class ScenarioResult:
    """One deterministic acceptance scenario."""

    id: str
    passed: bool
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeMetrics:
    """Required quality, safety, and cost measurements for one policy."""

    task_success_rate: float = 0.0
    constraint_adherence: float = 0.0
    retrieval_precision: float = 0.0
    harmful_memory_rate: float = 0.0
    unsupported_promotion_rate: float = 0.0
    repeat_error_rate: float = 0.0
    project_leakage_rate: float = 0.0
    latency_ms: float = 0.0
    model_calls: int = 0
    token_usage: int = 0


@dataclass
class ModeReport:
    """Metrics plus scenario evidence for a comparison mode."""

    mode: str
    policy: str
    metrics: ModeMetrics
    scenarios: list[ScenarioResult] = field(default_factory=list)


@dataclass
class LearningEvaluationReport:
    """Machine-readable benchmark report and acceptance decision."""

    benchmark_id: str
    corpus_schema_version: int
    modes: list[ModeReport]
    accepted: bool
    acceptance_failures: list[str]
    schema_version: int = REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_corpus(path: Optional[Path] = None) -> dict[str, Any]:
    """Load and minimally validate the versioned benchmark corpus."""
    target = path or DEFAULT_CORPUS_PATH
    data = json.loads(target.read_text())
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("unsupported learning benchmark corpus schema")
    if not data.get("task_families"):
        raise ValueError("learning benchmark corpus has no task families")
    return data


class LearningLoopEvaluator:
    """Run isolated comparisons through Neo's real local memory primitives."""

    def __init__(self, corpus: dict[str, Any], *, workspace: Path):
        self.corpus = corpus
        self.workspace = workspace
        self.facts_dir = workspace / "facts"
        self.episodes_dir = workspace / "episodes"
        self.repositories_dir = workspace / "repositories"
        self.embedder = DeterministicHashEmbedder()

    def _store(self, mode: EvaluationMode, project: str) -> FactStore:
        root = self.repositories_dir / mode.value / project
        root.mkdir(parents=True, exist_ok=True)
        store = FactStore(
            codebase_root=str(root),
            eager_init=False,
            facts_dir=self.facts_dir / mode.value,
            episodes_dir=self.episodes_dir / mode.value,
            emit_metrics=False,
        )
        store._embedder = self.embedder
        store._embedder_initialized = True
        return store

    def _episode_store(self, mode: EvaluationMode, store: FactStore) -> LearningEpisodeStore:
        return LearningEpisodeStore(
            store.project_id or "unscoped",
            base_dir=self.episodes_dir / mode.value,
        )

    @staticmethod
    def _apply_detected_outcome(store: FactStore, outcome: Outcome) -> None:
        store._outcome_tracker.detect_outcomes = lambda: ([outcome], {})
        store._outcome_tracker.compute_arch_delta = lambda: None
        store.detect_implicit_feedback({"prompt": outcome.suggestion_description}, [])

    def _record_evidence_family(
        self, store: FactStore, family: dict[str, Any]
    ) -> tuple[list[str], Optional[Fact]]:
        episode_store = self._episode_store(EvaluationMode.EVIDENCE, store)
        episode_ids: list[str] = []
        accepted_outcomes: list[Outcome] = []
        for index, outcome_name in enumerate(family["training_outcomes"]):
            if outcome_name == "regression":
                continue
            identity = f"{store.project_id}-{family['id']}-{index}"
            episode_id = identity
            suggestion_id = f"suggestion-{identity}"
            candidate_id = f"candidate-{identity}"
            episode = LearningEpisode(
                episode_id=episode_id,
                project_id=store.project_id,
                objective=family["query"],
                repository_revision=f"revision-{family['id']}-{index}",
            )
            episode.suggestions.append(SuggestionEvidence(
                suggestion_id=suggestion_id,
                file_path="src/service.py",
                description=family["body"],
                confidence=0.7,
            ))
            episode.memory_candidates.append(MemoryCandidateEvidence(
                candidate_id=candidate_id,
                suggestion_id=suggestion_id,
                subject=family["subject"],
                body=family["body"],
                kind=FactKind.PATTERN.value,
            ))
            verification_status = "passed" if outcome_name == "accepted" else "skipped"
            episode.verification.append(VerificationEvidence(
                verification_id=f"verification-{family['id']}-{index}",
                kind="test",
                status=verification_status,
                tool_name="deterministic_benchmark",
                summary=(
                    "benchmark verification passed"
                    if verification_status == "passed"
                    else "verification unavailable"
                ),
                repository_revision=episode.repository_revision,
            ))
            episode_store.save(episode)
            episode_ids.append(episode_id)
            outcome = Outcome(
                outcome_type=(
                    OutcomeType.ACCEPTED
                    if outcome_name == "accepted"
                    else OutcomeType.UNVERIFIED
                ),
                file_path="src/service.py",
                suggestion_description=family["body"],
                suggestion_confidence=0.7,
                suggestion_id=suggestion_id,
                learning_episode_id=episode_id,
                repository_revision=episode.repository_revision,
                candidate_id=candidate_id,
                candidate_subject=family["subject"],
                candidate_body=family["body"],
                candidate_kind=FactKind.PATTERN.value,
            )
            self._apply_detected_outcome(store, outcome)
            if outcome.outcome_type == OutcomeType.ACCEPTED:
                accepted_outcomes.append(outcome)

        promoted = next((
            fact for fact in store.entries
            if fact.subject == family["subject"] and "episode-derived" in fact.tags
        ), None)
        regression_count = family["training_outcomes"].count("regression")
        for index, accepted in enumerate(accepted_outcomes[:regression_count]):
            store.record_later_regression(
                learning_episode_id=accepted.learning_episode_id,
                suggestion_id=accepted.suggestion_id,
                summary="deterministic benchmark later regression",
                repository_revision=f"regression-revision-{index}",
            )
        return episode_ids, promoted

    def _record_legacy_family(self, store: FactStore, family: dict[str, Any]) -> Fact:
        fact = store.add_fact(
            subject=family["subject"],
            body=family["body"],
            kind=FactKind.PATTERN,
            scope=FactScope.PROJECT,
            confidence=0.5,
            tags=["legacy-generated"],
            provenance=Provenance.INFERRED,
        )
        regressions = 0
        for outcome in family["training_outcomes"]:
            if outcome == "accepted":
                fact.metadata.confidence = min(1.0, fact.metadata.confidence + 0.2)
                fact.metadata.success_count += 1
                update_effectiveness(fact, outcome="better")
            elif outcome == "unverified":
                # Historical behavior treated absence of evidence as weak success.
                fact.metadata.confidence = min(1.0, fact.metadata.confidence + 0.1)
                fact.metadata.success_count += 1
                update_effectiveness(fact, outcome="better")
            elif outcome == "regression":
                regressions += 1
                fact.metadata.confidence = max(0.1, fact.metadata.confidence - 0.2)
                update_effectiveness(fact, outcome="worse")
        if regressions:
            fact.tags.append("known-regression")
        store.save()
        return fact

    def _record_later_task_trace(
        self, store: FactStore, family: dict[str, Any], fact: Fact
    ) -> str:
        """Persist the later retrieval-to-decision proof without mutating memory."""
        context = store.build_context(family["query"], k=1)
        score = context.retrieval_scores.get(fact.id)
        episode_id = f"later-{store.project_id}-{family['id']}"
        suggestion_id = f"later-suggestion-{store.project_id}-{family['id']}"
        episode = LearningEpisode(
            episode_id=episode_id,
            project_id=store.project_id,
            objective=family["query"],
            repository_root=store.codebase_root or "",
            repository_revision=f"later-revision-{family['id']}",
            reasoning_mode="deterministic_evaluation",
            provider="none",
            model="none",
            final_outcome=OutcomeType.ACCEPTED.value,
        )
        episode.context_selection.append(ContextSelection(
            path=f"memory:{fact.id}",
            content_sha256=hashlib.sha256(fact.body.encode()).hexdigest(),
            kind="durable_memory",
        ))
        episode.retrieved_facts.append(RetrievedFactEvidence(
            fact_id=fact.id,
            score=score,
            included_in_context=True,
            used_in_reasoning=True,
            outcome_association=OutcomeType.ACCEPTED.value,
        ))
        episode.suggestions.append(SuggestionEvidence(
            suggestion_id=suggestion_id,
            file_path="src/later_service.py",
            description=f"Apply [fact:{fact.id}]: {family['body']}",
            confidence=0.0,
        ))
        episode.verification.append(VerificationEvidence(
            verification_id=f"later-verification-{family['id']}",
            kind="test",
            status="passed",
            tool_name="deterministic_benchmark",
            summary="equivalent later task followed the verified convention",
            repository_revision=episode.repository_revision,
        ))
        episode.outcome_details = {
            "suggestion_id": suggestion_id,
            "repository_revision": episode.repository_revision,
        }
        self._episode_store(EvaluationMode.EVIDENCE, store).save(episode)
        return episode_id

    @staticmethod
    def _retrieve_ids(store: FactStore, query: str) -> list[str]:
        return [fact.id for fact in store.retrieve_relevant(query, k=1)]

    def _run_mode(self, mode: EvaluationMode) -> ModeReport:
        started = time.perf_counter()
        alpha = self._store(mode, "alpha")
        unrelated: Optional[Fact] = None
        if mode is not EvaluationMode.DISABLED:
            unrelated = alpha.add_fact(
                subject="structural convention: UTC audit timestamps",
                body="Repository audit timestamps use UTC.",
                kind=FactKind.PATTERN,
                confidence=0.8,
                provenance=Provenance.STRUCTURAL,
                tags=["benchmark-control"],
            )

        facts_by_family: dict[str, Optional[Fact]] = {}
        episode_ids_by_family: dict[str, list[str]] = {}
        for family in self.corpus["task_families"]:
            if mode is EvaluationMode.DISABLED:
                facts_by_family[family["id"]] = None
                episode_ids_by_family[family["id"]] = []
            elif mode is EvaluationMode.LEGACY:
                facts_by_family[family["id"]] = self._record_legacy_family(alpha, family)
                episode_ids_by_family[family["id"]] = []
            else:
                episode_ids, fact = self._record_evidence_family(alpha, family)
                facts_by_family[family["id"]] = fact
                episode_ids_by_family[family["id"]] = episode_ids

        retrieved_by_family: dict[str, list[str]] = {}
        for family in self.corpus["task_families"]:
            retrieved_by_family[family["id"]] = (
                [] if mode is EvaluationMode.DISABLED
                else self._retrieve_ids(alpha, family["query"])
            )

        verified_families = [
            family for family in self.corpus["task_families"]
            if family["expected_later_behavior"] == "retrieve"
        ]
        verified = verified_families[0]
        unverified = next(
            family for family in self.corpus["task_families"]
            if family["expected_later_behavior"] == "do_not_trust"
        )
        regressed = next(
            family for family in self.corpus["task_families"]
            if family["expected_later_behavior"] == "roll_back"
        )
        verified_fact = facts_by_family[verified["id"]]
        unverified_fact = facts_by_family[unverified["id"]]
        regressed_fact = facts_by_family[regressed["id"]]

        verified_results = {
            family["id"]: bool(
                facts_by_family[family["id"]]
                and facts_by_family[family["id"]].is_valid
                and facts_by_family[family["id"]].id
                in retrieved_by_family[family["id"]]
            )
            for family in verified_families
        }
        verified_success = all(verified_results.values())
        later_episode_ids: dict[str, str] = {}
        if mode is EvaluationMode.EVIDENCE:
            for family in verified_families:
                fact = facts_by_family[family["id"]]
                if fact is not None and verified_results[family["id"]]:
                    later_episode_ids[family["id"]] = self._record_later_task_trace(
                        alpha, family, fact
                    )
        unverified_error = bool(
            unverified_fact
            and unverified_fact.is_valid
            and unverified_fact.id in retrieved_by_family[unverified["id"]]
        )
        regression_error = bool(
            regressed_fact
            and regressed_fact.is_valid
            and regressed_fact.id in retrieved_by_family[regressed["id"]]
        )

        eligible = [
            fact for fact in alpha.entries
            if fact.kind == FactKind.PATTERN
            and fact.metadata.provenance != Provenance.STRUCTURAL.value
        ]
        valid_eligible = [fact for fact in eligible if fact.is_valid]
        harmful = [
            fact for fact in valid_eligible
            if fact.subject in {unverified["subject"], regressed["subject"]}
        ]
        unsupported = [
            fact for fact in valid_eligible if len(set(fact.supporting_episode_ids)) < 2
        ]

        beta = self._store(mode, "beta")
        beta._embedder = self.embedder
        beta._embedder_initialized = True
        leaked_ids = [] if mode is EvaluationMode.DISABLED else self._retrieve_ids(beta, verified["query"])
        project_fact_ids = {
            fact.id for fact in alpha.entries
            if fact.scope == FactScope.PROJECT and fact.project_id == alpha.project_id
        }
        leakage = bool(project_fact_ids.intersection(leaked_ids))

        scenarios: list[ScenarioResult] = []
        if mode is EvaluationMode.EVIDENCE:
            global_before_cross_project = [
                fact for fact in alpha.entries if fact.scope == FactScope.GLOBAL
            ]
            self._record_evidence_family(beta, verified)
            global_after_cross_project = [
                fact for fact in beta.entries if fact.scope == FactScope.GLOBAL
            ]
            malformed_store = self._episode_store(mode, alpha)
            malformed_store.path.mkdir(parents=True, exist_ok=True)
            malformed_path = malformed_store.path / "malformed-eval.json"
            malformed_path.write_text("{not-json")
            old_path = malformed_store.path / "old-eval.json"
            old_path.write_text(json.dumps({"episode_id": "old-eval", "objective": "old"}))
            old_loaded = malformed_store.load("old-eval") is not None
            malformed_safe = malformed_store.load("malformed-eval") is None

            first_rank = self._retrieve_ids(self._store(mode, "alpha"), verified["query"])
            second_rank = self._retrieve_ids(self._store(mode, "alpha"), verified["query"])

            explanation = (
                explain_fact(
                    alpha.entries,
                    verified_fact.id,
                    episode_store=malformed_store,
                )
                if verified_fact is not None
                else {"model_calls": -1, "supporting_evidence": []}
            )
            sequence_explanations = {
                family["id"]: explain_fact(
                    alpha.entries,
                    facts_by_family[family["id"]].id,
                    episode_store=malformed_store,
                )
                for family in verified_families
                if facts_by_family[family["id"]] is not None
            }
            complete_sequence_chains = all(
                item["model_calls"] == 0
                and len(item["supporting_evidence"]) >= 2
                and len(item["retrieval_history"]) >= 1
                and any(row["used_in_reasoning"] for row in item["retrieval_history"])
                and len(item["mutation_history"]) >= 1
                for item in sequence_explanations.values()
            ) and len(sequence_explanations) >= 2
            scenarios = [
                ScenarioResult("verified_convention_retrieved_later", verified_success, {
                    "fact_id": verified_fact.id if verified_fact else None,
                    "retrieved_ids": retrieved_by_family[verified["id"]],
                }),
                ScenarioResult("unverified_suggestion_not_promoted", unverified_fact is None, {
                    "fact_id": unverified_fact.id if unverified_fact else None,
                }),
                ScenarioResult("attributed_failure_reduces_confidence", bool(
                    regressed_fact and regressed_fact.metadata.confidence < 0.6
                ), {
                    "confidence": regressed_fact.metadata.confidence if regressed_fact else None,
                }),
                ScenarioResult("failure_preserves_unrelated_fact", bool(
                    unrelated and unrelated.is_valid and unrelated.metadata.confidence == 0.8
                ), {"unrelated_fact_id": unrelated.id if unrelated else None}),
                ScenarioResult("contradiction_is_explicit", bool(
                    regressed_fact and len(regressed_fact.contradicting_episode_ids) == 2
                    and regressed_fact.invalidation_reason == "repeated_attributed_contradiction"
                ), {
                    "contradicting_episode_ids": (
                        regressed_fact.contradicting_episode_ids if regressed_fact else []
                    ),
                }),
                ScenarioResult("project_knowledge_does_not_leak", not leakage, {
                    "beta_retrieved_ids": leaked_ids,
                }),
                ScenarioResult("global_promotion_requires_cross_project_evidence", bool(
                    not global_before_cross_project
                    and len(global_after_cross_project) == 1
                    and len(global_after_cross_project[0].supporting_episode_ids) >= 4
                ), {
                    "before_cross_project": [
                        fact.id for fact in global_before_cross_project
                    ],
                    "after_cross_project": [
                        fact.id for fact in global_after_cross_project
                    ],
                    "support_count": (
                        len(global_after_cross_project[0].supporting_episode_ids)
                        if global_after_cross_project else 0
                    ),
                }),
                ScenarioResult("older_and_malformed_records_survive", old_loaded and malformed_safe, {
                    "old_loaded": old_loaded,
                    "malformed_preserved": bool(list(
                        malformed_store.path.glob("malformed-eval.json.corrupt-*")
                    )),
                }),
                ScenarioResult("ranking_is_deterministic", first_rank == second_rank, {
                    "first": first_rank,
                    "second": second_rank,
                }),
                ScenarioResult("provenance_explains_without_model", bool(
                    explanation["model_calls"] == 0
                    and len(explanation["supporting_evidence"]) >= 2
                ), {
                    "model_calls": explanation["model_calls"],
                    "support_count": len(explanation["supporting_evidence"]),
                }),
                ScenarioResult("two_verified_task_sequences_improve_later_tasks", bool(
                    len(verified_results) >= 2 and all(verified_results.values())
                ), {"families": verified_results}),
                ScenarioResult("two_sequence_causal_chains_are_local", complete_sequence_chains, {
                    "families": {
                        family_id: {
                            "model_calls": item["model_calls"],
                            "support_count": len(item["supporting_evidence"]),
                            "retrieval_count": len(item["retrieval_history"]),
                            "mutation_count": len(item["mutation_history"]),
                            "later_episode_id": later_episode_ids.get(family_id),
                        }
                        for family_id, item in sequence_explanations.items()
                    },
                }),
            ]

        metrics = ModeMetrics(
            task_success_rate=(
                sum(verified_results.values()) / len(verified_results)
                if verified_results else 0.0
            ),
            constraint_adherence=1.0 - ((int(unverified_error) + int(regression_error)) / 2.0),
            retrieval_precision=(
                sum(verified_results.values()) / len(verified_results)
                if verified_results else 0.0
            ),
            harmful_memory_rate=(len(harmful) / len(valid_eligible)) if valid_eligible else 0.0,
            unsupported_promotion_rate=(
                len(unsupported) / len(valid_eligible) if valid_eligible else 0.0
            ),
            repeat_error_rate=(int(unverified_error) + int(regression_error)) / 2.0,
            project_leakage_rate=1.0 if leakage else 0.0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model_calls=0,
            token_usage=0,
        )
        policies = {
            EvaluationMode.DISABLED: "No persistent memory or retrieval.",
            EvaluationMode.LEGACY: (
                "Generated suggestions become facts immediately; unverified counts as weak success."
            ),
            EvaluationMode.EVIDENCE: (
                "Episode candidates require repeated accepted evidence; attributed failures roll back."
            ),
        }
        return ModeReport(mode=mode.value, policy=policies[mode], metrics=metrics,
                          scenarios=scenarios)

    def run(self) -> LearningEvaluationReport:
        """Run all comparison modes and enforce safety plus improvement gates."""
        modes = [self._run_mode(mode) for mode in EvaluationMode]
        by_mode = {item.mode: item for item in modes}
        evidence = by_mode[EvaluationMode.EVIDENCE.value]
        baseline = by_mode[EvaluationMode.DISABLED.value]
        failures = [scenario.id for scenario in evidence.scenarios if not scenario.passed]

        thresholds = self.corpus["safety_thresholds"]
        safety_metrics = {
            "harmful_memory_rate": "harmful_memory_rate_max",
            "unsupported_promotion_rate": "unsupported_promotion_rate_max",
            "repeat_error_rate": "repeat_error_rate_max",
            "project_leakage_rate": "project_leakage_rate_max",
        }
        for metric_name, threshold_name in safety_metrics.items():
            value = getattr(evidence.metrics, metric_name)
            if value > float(thresholds[threshold_name]):
                failures.append(
                    f"{metric_name}={value:.4f} exceeds {threshold_name}="
                    f"{float(thresholds[threshold_name]):.4f}"
                )
        if evidence.metrics.latency_ms > float(thresholds["latency_ms_max"]):
            failures.append(
                f"latency_ms={evidence.metrics.latency_ms:.4f} exceeds "
                f"latency_ms_max={float(thresholds['latency_ms_max']):.4f}"
            )

        primary = self.corpus["primary_quality_metrics"]
        improved = any(
            getattr(evidence.metrics, name) > getattr(baseline.metrics, name)
            for name in primary
        )
        if not improved:
            failures.append("no primary quality metric improved over memory-disabled baseline")
        for name in primary:
            if getattr(evidence.metrics, name) < getattr(baseline.metrics, name):
                failures.append(f"primary metric degraded: {name}")
        if evidence.metrics.model_calls or evidence.metrics.token_usage:
            failures.append("deterministic benchmark unexpectedly used a model")

        return LearningEvaluationReport(
            benchmark_id=self.corpus["benchmark_id"],
            corpus_schema_version=self.corpus["schema_version"],
            modes=modes,
            accepted=not failures,
            acceptance_failures=failures,
        )


def run_learning_evaluation(
    *, corpus_path: Optional[Path] = None, workspace: Optional[Path] = None
) -> LearningEvaluationReport:
    """Convenience entry point using an isolated temporary workspace by default."""
    corpus = load_corpus(corpus_path)
    if workspace is not None:
        workspace.mkdir(parents=True, exist_ok=True)
        return LearningLoopEvaluator(corpus, workspace=workspace).run()
    with tempfile.TemporaryDirectory(prefix="neo-learning-eval-") as temp:
        return LearningLoopEvaluator(corpus, workspace=Path(temp)).run()
