"""Tests for neo.languages — centralized language identity utilities."""

from pathlib import Path

from neo.languages import (
    EXTENSION_TO_LANGUAGE,
    KNOWN_FENCE_TAGS,
    _DISPLAY_NAMES,
    _FENCE_TAGS,
    display_name_for,
    fence_tag_for,
    language_for_path,
    normalize_language_name,
)


class TestLanguageForPath:
    def test_python_extension(self):
        assert language_for_path("foo.py") == "python"

    def test_typescript_extension(self):
        assert language_for_path("foo.ts") == "typescript"

    def test_tsx_distinct_from_typescript(self):
        assert language_for_path("foo.tsx") == "tsx"

    def test_mjs_and_cjs_map_to_javascript(self):
        assert language_for_path("foo.mjs") == "javascript"
        assert language_for_path("foo.cjs") == "javascript"

    def test_csharp_uses_tree_sitter_name(self):
        # Canonical name is `c_sharp` (tree-sitter), not `csharp` (GFM).
        assert language_for_path("foo.cs") == "c_sharp"

    def test_unknown_extension(self):
        assert language_for_path("foo.xyz") is None

    def test_path_object_accepted(self):
        # Same result whether caller passes str or Path.
        assert language_for_path(Path("foo.py")) == "python"

    def test_case_insensitive(self):
        # MAIN.JAVA → still .java → java
        assert language_for_path("MAIN.JAVA") == "java"

    def test_no_extension(self):
        assert language_for_path("Makefile") is None


class TestFenceTagFor:
    def test_canonical_language_maps(self):
        assert fence_tag_for("python") == "python"
        assert fence_tag_for("javascript") == "javascript"

    def test_csharp_normalized(self):
        # Canonical c_sharp maps to GFM csharp.
        assert fence_tag_for("c_sharp") == "csharp"
        # And csharp itself also accepted.
        assert fence_tag_for("csharp") == "csharp"

    def test_unknown_language_returns_empty(self):
        assert fence_tag_for("klingon") == ""

    def test_none_returns_empty(self):
        assert fence_tag_for(None) == ""

    def test_empty_string_returns_empty(self):
        assert fence_tag_for("") == ""

    def test_case_insensitive(self):
        assert fence_tag_for("PYTHON") == "python"


class TestDisplayNameFor:
    def test_python(self):
        assert display_name_for("python") == "Python"

    def test_csharp_display(self):
        assert display_name_for("c_sharp") == "C#"
        assert display_name_for("csharp") == "C#"

    def test_unknown_returns_input(self):
        # Preserve the input string so prompts still get a usable label.
        assert display_name_for("brainfuck") == "brainfuck"

    def test_none_returns_empty(self):
        assert display_name_for(None) == ""

    def test_case_insensitive_lookup(self):
        assert display_name_for("Python") == "Python"


class TestNormalize:
    def test_canonical_passes_through(self):
        assert normalize_language_name("python") == "python"
        assert normalize_language_name("c_sharp") == "c_sharp"

    def test_gfm_csharp_normalized_to_c_sharp(self):
        assert normalize_language_name("csharp") == "c_sharp"

    def test_hash_alias_for_csharp(self):
        assert normalize_language_name("c#") == "c_sharp"

    def test_short_aliases(self):
        # Common one-letter shortenings used by LM responses.
        assert normalize_language_name("py") == "python"
        assert normalize_language_name("js") == "javascript"
        assert normalize_language_name("ts") == "typescript"
        assert normalize_language_name("rs") == "rust"
        assert normalize_language_name("rb") == "ruby"
        assert normalize_language_name("kt") == "kotlin"

    def test_case_insensitive(self):
        assert normalize_language_name("CSHARP") == "c_sharp"
        assert normalize_language_name("Python") == "python"

    def test_none_returns_empty(self):
        assert normalize_language_name(None) == ""

    def test_unknown_passes_through_lowercased(self):
        assert normalize_language_name("Klingon") == "klingon"


class TestNormalizationFlowsThroughLookups:
    """`fence_tag_for` and `display_name_for` should accept any alias
    that `normalize_language_name` recognizes. This pins the contract
    so adding a new alias doesn't require touching three maps."""

    def test_csharp_fence(self):
        assert fence_tag_for("csharp") == "csharp"
        assert fence_tag_for("c_sharp") == "csharp"
        assert fence_tag_for("c#") == "csharp"

    def test_csharp_display(self):
        assert display_name_for("csharp") == "C#"
        assert display_name_for("c_sharp") == "C#"
        assert display_name_for("c#") == "C#"

    def test_py_alias_resolves(self):
        assert fence_tag_for("py") == "python"
        assert display_name_for("py") == "Python"


class TestMapIntegrity:
    def test_extension_keys_have_leading_dot(self):
        for ext in EXTENSION_TO_LANGUAGE:
            assert ext.startswith("."), f"{ext!r} missing leading dot"

    def test_extension_keys_are_lowercase(self):
        for ext in EXTENSION_TO_LANGUAGE:
            assert ext == ext.lower()

    def test_fence_map_keys_are_lowercase(self):
        for key in _FENCE_TAGS:
            assert key == key.lower()

    def test_display_map_keys_are_lowercase(self):
        for key in _DISPLAY_NAMES:
            assert key == key.lower()

    def test_known_fence_tags_lowercase(self):
        for tag in KNOWN_FENCE_TAGS:
            # `c#` is the one allowed non-alpha tag (intentional)
            assert tag == tag.lower()
