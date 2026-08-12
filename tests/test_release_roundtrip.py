"""Per-language LLM round trip — the paid half of the release gate.

One real `neo --json` invocation per language, against the generated fixture
repo, through whatever provider the environment is configured for. A red
language here blocks the release; see `docs/release-gate.md` and the
`language-roundtrip` job in `.github/workflows/publish.yml`.

**These tests cost money and are release-only.** They skip unless
`NEO_RELEASE_ROUNDTRIP=1` is set, and they SKIP rather than deselect on
purpose: an ordinary `pytest` run should print three skipped round trips, so
a reader can see the gate exists. A gate nobody can see is the failure this
goal exists to prevent — C# was absent from the index for 8.5 months and
every run printed success and exited 0.

What each language asserts, end to end:

1. the invocation exits 0 and stdout is exactly one JSON document;
2. the result is non-empty and on topic — the model names the fixture's
   sentinel symbol, which appears in exactly one file, so it cannot be named
   unless that file actually reached the prompt. This is the assertion the
   free battery cannot make: the battery proves the gatherer SELECTED the
   file, this proves the model SAW it;
3. the selection invariants the battery checks still hold for this run's
   context — the language's files present, nothing gitignored, no duplicates.

Point 3 is re-asserted here rather than assumed because the release gate is
the last thing standing between a broken selection and a published wheel,
and "the battery passed on some other commit" is not evidence about this one.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures.language_repos import (
    LANGUAGES,
    FixtureRepo,
    build_fixture_repo,
    check_ignored,
)
from tests.test_selection_invariants import _gather

pytestmark = [
    pytest.mark.roundtrip,
    pytest.mark.skipif(
        os.environ.get("NEO_RELEASE_ROUNDTRIP") != "1",
        reason=(
            "release-only: costs a real LLM call per language. "
            "Set NEO_RELEASE_ROUNDTRIP=1 to run (see docs/release-gate.md)."
        ),
    ),
]

# Generous: this is one call on a release, not a per-push cost, and a gate
# that flakes on a slow provider teaches people to re-run it until it passes.
_TIMEOUT_SECONDS = 600


@pytest.fixture(scope="module")
def roundtrip_repos(tmp_path_factory) -> dict[str, FixtureRepo]:
    base = tmp_path_factory.mktemp("roundtrip_fixtures")
    return {
        language: build_fixture_repo(language, base / language)
        for language in LANGUAGES
    }


def _run_neo(fx: FixtureRepo) -> dict:
    """Invoke the installed CLI the way a user does, and parse its stdout.

    A subprocess rather than an in-process `NeoEngine`, because the thing
    under test is the release artifact's behaviour end to end: argument
    parsing, adapter resolution from the environment, gathering, the model
    call and serialization. An in-process call would skip the half of that
    which a release can actually break.

    stdout is exactly one JSON document by contract; stderr carries JSONL
    events and is surfaced verbatim on failure, because a provider error
    reported as "the assertion failed" costs an hour.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "neo", "--json", fx.prompt],
        cwd=str(fx.root),
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
    )
    detail = (
        f"\n--- exit {proc.returncode} ---"
        f"\n--- stdout ---\n{proc.stdout[-4000:]}"
        f"\n--- stderr ---\n{proc.stderr[-4000:]}"
    )
    assert proc.returncode == 0, (
        f"neo failed for {fx.language}: {_classify_failure(proc.stdout)}{detail}"
    )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:  # pragma: no cover - failure path
        raise AssertionError(
            f"stdout was not a single JSON document for {fx.language}: "
            f"{e}{detail}"
        ) from None

    # The CLI prints an error envelope and exits non-zero on failure, so a
    # zero exit carrying `error` would mean the contract itself broke.
    assert "error" not in payload, f"{fx.language}: {payload.get('error')}{detail}"
    return payload


# Error envelopes the CLI emits that say nothing about file selection, mapped
# to where to look instead. Naming the layer is the whole value: measured
# live, a run failed on `ValidationError` (the model's reply missed neo's own
# v3 start sentinel) and the default reading of a red LANGUAGE round trip
# would have sent someone to the gatherer, which was working perfectly.
_UNRELATED_TO_SELECTION = {
    "ValidationError": (
        "neo's structured parser rejected the model's reply. This is a real "
        "release blocker — a user would get the same error — but it is NOT a "
        "file-selection failure. Look at structured_parser.py, not the "
        "gatherer. `pytest -m invariants` covers selection for free and will "
        "be GREEN when this is the cause"
    ),
    "RequestTimeout": "the provider did not answer in time; not a selection failure",
    "NetworkTimeout": "the network did not reach the provider; not a selection failure",
    "ProcessingError": "an unexpected engine error; read the stderr events below",
}


def _classify_failure(stdout: str) -> str:
    """Name the layer that broke, so a red gate points somewhere real.

    A red per-language round trip reads as "the language broke", and for the
    failure this gate exists to catch that is right. For the several ways an
    invocation can fail that have nothing to do with which files were
    selected, it is an expensive wrong turn.
    """
    try:
        envelope = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return "no JSON envelope on stdout"

    kind = envelope.get("error")
    if not kind:
        return "non-zero exit with no error envelope"
    if kind.startswith("Failed to initialize LM adapter"):
        return (
            "no usable provider credential. The release job requires "
            "ANTHROPIC_API_KEY and fails rather than skips without it"
        )
    return f"{kind} — {_UNRELATED_TO_SELECTION.get(kind, 'see the envelope below')}"


@pytest.fixture(scope="module")
def roundtrip_results(roundtrip_repos) -> dict[str, dict]:
    """One invocation per language, shared by the assertions below.

    Module-scoped so the gate costs three model calls, not three per
    assertion. If a language's call fails, its assertions fail with the
    provider's own output attached.
    """
    return {
        language: _run_neo(fx) for language, fx in sorted(roundtrip_repos.items())
    }


@pytest.mark.parametrize("language", LANGUAGES)
class TestLanguageRoundTrip:
    def test_the_result_is_not_empty(self, language, roundtrip_results):
        """A run that returns nothing is a red language, not a quiet one."""
        payload = roundtrip_results[language]

        substance = (
            payload.get("plan")
            or payload.get("code_suggestions")
            or payload.get("hypotheses")
            or (payload.get("notes") or "").strip()
        )
        assert substance, (
            f"{language}: neo returned no plan, no suggestion, no hypothesis "
            f"and no notes"
        )

    def test_the_model_names_the_symbol_only_that_file_contains(
        self, language, roundtrip_repos, roundtrip_results
    ):
        """The end-to-end assertion the free battery cannot make.

        The sentinel appears in exactly one file in the fixture. The model
        can only name it if that file reached the prompt, so this closes the
        gap between "the gatherer selected it" and "the model saw it" — the
        gap C# sat inside for 8.5 months.
        """
        fx = roundtrip_repos[language]
        blob = json.dumps(roundtrip_results[language])

        assert fx.sentinel.lower() in blob.lower(), (
            f"{language}: the response never mentions {fx.sentinel}, the only "
            f"symbol in {fx.target_rel}. Either the file did not reach the "
            f"prompt or it was truncated above the symbol."
        )

    def test_confidence_is_reported(self, language, roundtrip_results):
        """`confidence` absent is a different failure from low confidence.

        Low confidence is a legitimate answer and is NOT gated here — a
        correct refusal must not block a release. Its absence means the
        serialization contract broke.
        """
        payload = roundtrip_results[language]

        assert isinstance(payload.get("confidence"), (int, float)), payload.keys()

    def test_the_orchestrator_envelope_is_present(self, language, roundtrip_results):
        """Both host adapters read this envelope; a release must not ship
        without it. See `tests/test_host_adapter_parity.py`."""
        orchestrator = roundtrip_results[language].get("orchestrator")

        assert isinstance(orchestrator, dict), orchestrator
        assert (orchestrator.get("summary") or "").strip()


@pytest.mark.parametrize("language", LANGUAGES)
class TestSelectionInvariantsHoldOnTheReleaseCommit:
    """The battery's invariants, re-asserted at the gate.

    Not redundant with `test_selection_invariants.py`: that runs on a PR,
    this runs on the commit being published. The only claim a release can
    make about its own selection behaviour is one it measured itself.
    """

    def test_the_named_file_is_selected_and_whole(self, language, roundtrip_repos):
        fx = roundtrip_repos[language]
        gathered = _gather(fx)

        target = next(
            (g for g in gathered if g.rel_path == fx.target_rel), None
        )
        assert target is not None, [g.rel_path for g in gathered]
        assert (target.content or "") == (
            (fx.root / fx.target_rel).read_text(encoding="utf-8")
        )

    def test_nothing_gitignored_is_selected(self, language, roundtrip_repos):
        fx = roundtrip_repos[language]
        rels = [g.rel_path for g in _gather(fx)]

        assert check_ignored(fx.root, rels) == []

    def test_no_duplicate_copies_are_selected(self, language, roundtrip_repos):
        fx = roundtrip_repos[language]
        rels = [g.rel_path for g in _gather(fx)]
        basename = Path(fx.target_rel).name

        assert len(rels) == len(set(rels)), rels
        assert [r for r in rels if Path(r).name == basename] == [fx.target_rel]
