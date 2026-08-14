"""The linkage that lets a verified acceptance reinforce the fact it came from.

`suggestion_fact_ids` maps each suggested file path to the fact the suggestion
applied. `detect_implicit_feedback` reinforces the linked fact on a git-verified
acceptance, and `neo memory replay-feedback` — the documented repair command for
a broken memory loop — reads nothing else.

It went permanently empty when episodes replaced immediate fact-writing:
`engine._store_reasoning` returns None on every path, and the old builder
returned `{}` for a None fact. Measured on a live install, every session after
the changeover carried zero links and no fact's success_count ever moved again.
"""

from types import SimpleNamespace

from neo.engine import NeoEngine


def _suggestion(suggestion_id, file_path):
    return SimpleNamespace(suggestion_id=suggestion_id, file_path=file_path)


def _episode(*candidates):
    return SimpleNamespace(memory_candidates=[
        SimpleNamespace(suggestion_id=sid, subject=subject)
        for sid, subject in candidates
    ])


class _Store:
    """Stands in for FactStore.find_durable_fact_for_candidate."""

    def __init__(self, durable_by_subject):
        self._durable = durable_by_subject
        self.asked = []

    def find_durable_fact_for_candidate(self, subject):
        self.asked.append(subject)
        fact_id = self._durable.get(subject)
        return SimpleNamespace(id=fact_id) if fact_id else None


def _build(fact, suggestions, episode, store):
    engine = NeoEngine.__new__(NeoEngine)
    engine.fact_store = store
    return NeoEngine._build_suggestion_fact_ids(engine, fact, suggestions, episode)


def test_links_a_suggestion_to_the_durable_fact_it_applies():
    """The whole point: a re-applied lesson has a fact to credit."""
    subject = "bugfix: guard empty path [a.py] [fp:abc]"
    store = _Store({subject: "fact-1"})

    ids = _build(None, [_suggestion("s1", "src/a.py")], _episode(("s1", subject)), store)

    assert ids == {"src/a.py": "fact-1"}


def test_no_link_before_the_lesson_is_durable():
    """First acceptances are evidence toward promotion, not reinforcement."""
    subject = "bugfix: guard empty path [a.py] [fp:abc]"
    store = _Store({})

    ids = _build(None, [_suggestion("s1", "src/a.py")], _episode(("s1", subject)), store)

    assert ids == {}
    assert store.asked == [subject], "the candidate must still be consulted"


def test_each_suggestion_resolves_against_its_own_candidate():
    """A multi-suggestion run must not credit one fact for another's path."""
    first = "bugfix: guard empty path [a.py] [fp:abc]"
    second = "algorithm: batch the scan [b.py] [fp:def]"
    store = _Store({first: "fact-1", second: "fact-2"})

    ids = _build(
        None,
        [_suggestion("s1", "src/a.py"), _suggestion("s2", "src/b.py")],
        _episode(("s1", first), ("s2", second)),
        store,
    )

    assert ids == {"src/a.py": "fact-1", "src/b.py": "fact-2"}


def test_unattributable_paths_are_never_linked():
    """A path a git diff could never name must not carry a link."""
    subject = "explanation: what does x do [/] [fp:abc]"
    store = _Store({subject: "fact-1"})

    for bad_path in ("/", "N/A", ""):
        ids = _build(
            None, [_suggestion("s1", bad_path)], _episode(("s1", subject)), store
        )
        assert ids == {}, f"{bad_path!r} must not be linked"


def test_suggestion_without_a_candidate_is_skipped():
    """No candidate means no signature to resolve; it must not guess."""
    store = _Store({"anything": "fact-1"})

    ids = _build(None, [_suggestion("s-unknown", "src/a.py")], _episode(), store)

    assert ids == {}
    assert store.asked == []


def test_a_backend_that_still_mints_a_fact_links_directly():
    """The legacy path stays intact and does not consult the episode."""
    store = _Store({})

    ids = _build(
        SimpleNamespace(id="legacy-fact"),
        [_suggestion("s1", "src/a.py"), _suggestion("s2", "src/b.py")],
        _episode(),
        store,
    )

    assert ids == {"src/a.py": "legacy-fact", "src/b.py": "legacy-fact"}
    assert store.asked == []


def test_no_episode_and_no_fact_yields_no_links():
    """Degrades quietly rather than raising on a store with no episode."""
    assert _build(None, [_suggestion("s1", "src/a.py")], None, _Store({})) == {}
    assert _build(None, [_suggestion("s1", "src/a.py")], _episode(), None) == {}
