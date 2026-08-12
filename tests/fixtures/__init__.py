"""Generated fixture repositories shared by the release gate.

The selection-invariant battery (`tests/test_selection_invariants.py`, every
PR, no LLM) and the per-language round trip
(`tests/test_release_roundtrip.py`, release only, real LLM) build the SAME
three repos from this package. That sharing is the point: a language whose
round trip is red in the release gate must be diagnosable from a free CI run,
and it cannot be if the two gates disagree about what the repo looks like.
"""
