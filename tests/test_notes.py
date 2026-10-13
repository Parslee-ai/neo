"""`neo.notes`: the notes a rule file points to, indexed for recall.

Recall may not load a model, so the observer embeds the passages and recall
only reads them; a note edited since it was embedded is never quoted."""

import numpy as np

from neo import notes
from neo.notes import chunk_note, load_index, note_files, refresh_index

MODEL = "test-model"


def _project(tmp_path):
    (tmp_path / "docs" / "notes").mkdir(parents=True)
    (tmp_path / "docs" / "notes" / "observer.md").write_text(
        "# Observer\n\n- The drain mark gates re-parsing, never the watermark mtime, "
        "because that file is rewritten after every consumed episode.\n")
    (tmp_path / "docs" / "notes" / "store.md").write_text(
        "- Supersession at cosine 0.85; pre-write dedup is canonical-signature equality, not cosine.\n")
    (tmp_path / "docs" / "single.md").write_text(
        "- A lone note named by its own path in the rule file, long enough to be a passage.\n")
    (tmp_path / "CLAUDE.md").write_text(
        "Read `docs/notes/` first, and `docs/single.md`. Not `../outside.md`, "
        "not `CLAUDE.md`, not `missing/dir/`, not `notes.py`.\n")
    return tmp_path


def _embed(text):
    v = np.zeros(4, dtype=np.float32)
    v[0 if "drain" in text else 1 if "Supersession" in text else 2] = 1.0
    return v


def test_rule_file_paths_name_the_notes(tmp_path):
    root = _project(tmp_path)
    (tmp_path.parent / "outside.md").write_text("- outside the project")
    names = [p.relative_to(root.resolve()).as_posix() for p in note_files(root)]
    assert names == ["docs/notes/observer.md", "docs/notes/store.md", "docs/single.md"]


def test_passages_are_bullets_cut_at_sentence_ends():
    long_bullet = "- " + " ".join(f"Sentence number {i} carries a reason." for i in range(80))
    passages = chunk_note(f"# Title\n\nshort\n\n{long_bullet}\n- Another bullet, with enough text to be kept as a passage of its own.\n")
    assert all(len(p) <= notes.CHUNK_CHARS for p in passages)
    assert len(passages) > 2
    assert passages[-1].startswith("- Another bullet")
    assert "short" not in passages and not any(p.startswith("#") for p in passages)


def test_index_round_trip_and_unchanged_notes_are_not_re_embedded(tmp_path):
    root = _project(tmp_path)
    assert refresh_index(root, _embed, MODEL) == 3
    loaded = load_index(root, MODEL)
    assert {f for f, _, _ in loaded} == {"docs/notes/observer.md", "docs/notes/store.md", "docs/single.md"}
    drain = next(v for f, _, v in loaded if f.endswith("observer.md"))
    assert drain.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert refresh_index(root, _embed, MODEL) == 0


def test_an_edited_note_is_not_quoted_until_re_embedded(tmp_path):
    root = _project(tmp_path)
    refresh_index(root, _embed, MODEL)
    edited = "- Supersession moved to 0.9 in this edit, which is long enough to be a passage."
    (root / "docs" / "notes" / "store.md").write_text(edited + "\n")
    assert "docs/notes/store.md" not in {f for f, _, _ in load_index(root, MODEL)}
    assert refresh_index(root, _embed, MODEL) == 1
    texts = [t for f, t, _ in load_index(root, MODEL) if f == "docs/notes/store.md"]
    assert texts == [edited]


def test_another_model_or_no_embedder_serves_nothing(tmp_path):
    root = _project(tmp_path)
    refresh_index(root, _embed, MODEL)
    assert load_index(root, "other-model") == []
    assert refresh_index(root, lambda text: None, "other-model") == 0
    assert load_index(root, MODEL), "a failed re-embed must keep the old index"


def test_an_index_of_another_layout_is_rebuilt_not_served(tmp_path):
    import json

    root = _project(tmp_path)
    refresh_index(root, _embed, MODEL)
    path = notes.index_path(root)
    old = json.loads(path.read_text())
    old["version"] = 1
    path.write_text(json.dumps(old))
    assert load_index(root, MODEL) == []
    assert refresh_index(root, _embed, MODEL) == 3
    assert load_index(root, MODEL)
