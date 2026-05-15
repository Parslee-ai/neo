"""Tests for neo.code_smells — high-precision anti-pattern detectors."""

from neo.code_smells import CodeSmell, format_for_prompt, scan_files
from neo.models import ContextFile


def _file(path: str, content: str) -> ContextFile:
    return ContextFile(path=path, content=content)


# ---------------------------------------------------------------------------
# TODO / FIXME / HACK / XXX markers
# ---------------------------------------------------------------------------

class TestMarkerDetection:
    def test_python_hash_todo(self):
        f = _file("a.py", "x = 1  # TODO: replace with config\n")
        smells = scan_files([f])
        assert any(s.kind == "todo" and s.severity == "info" for s in smells)
        assert smells[0].line == 1
        assert "TODO" in smells[0].message

    def test_javascript_double_slash_fixme(self):
        f = _file("a.js", "// FIXME: leaks memory under load\nconst x = 1;\n")
        smells = scan_files([f])
        assert smells[0].kind == "todo"
        assert smells[0].severity == "info"

    def test_block_comment_hack_warns(self):
        f = _file("a.c", "/* HACK: relies on undefined behavior */\nint x;\n")
        smells = scan_files([f])
        assert smells[0].kind == "todo"
        assert smells[0].severity == "warn"  # HACK is warn, not info

    def test_xxx_marker_warns(self):
        f = _file("a.py", "# XXX broken; do not ship\n")
        smells = scan_files([f])
        assert smells[0].severity == "warn"

    def test_marker_inside_identifier_not_flagged(self):
        # `update_todo_list` should NOT trigger the TODO scanner.
        f = _file("a.py", "def update_todo_list():\n    return []\n")
        smells = scan_files([f])
        # Only the stub check might fire, but not the TODO marker.
        assert not any(s.kind == "todo" for s in smells)

    def test_marker_in_string_literal_does_match(self):
        # We deliberately don't suppress markers in string literals — too
        # easy to lose real findings. This test pins that decision so a
        # future "improvement" doesn't silently drop coverage.
        # (Marker still requires a comment-prefix character, so a bare
        # string with "TODO" in it is fine.)
        f = _file("a.py", 'msg = "TODO write docs"\n')
        smells = scan_files([f])
        # No comment prefix — should NOT match.
        assert not smells


# ---------------------------------------------------------------------------
# Python stubs
# ---------------------------------------------------------------------------

class TestStubDetection:
    def test_pass_only_body(self):
        f = _file("a.py", "def stub():\n    pass\n")
        smells = scan_files([f])
        stub = next(s for s in smells if s.kind == "stub")
        assert "pass-only" in stub.message
        assert stub.severity == "warn"

    def test_ellipsis_only_body(self):
        f = _file("a.py", "def stub():\n    ...\n")
        stub = next(s for s in scan_files([f]) if s.kind == "stub")
        assert "ellipsis" in stub.message

    def test_raises_not_implemented(self):
        f = _file("a.py", "def stub():\n    raise NotImplementedError\n")
        stub = next(s for s in scan_files([f]) if s.kind == "stub")
        assert "NotImplementedError" in stub.message

    def test_raises_not_implemented_with_message(self):
        f = _file("a.py", "def stub():\n    raise NotImplementedError('not done')\n")
        stub = next(s for s in scan_files([f]) if s.kind == "stub")
        assert "NotImplementedError" in stub.message

    def test_docstring_does_not_count_as_content(self):
        # A function with only a docstring + pass IS still a stub.
        src = 'def stub():\n    """Does nothing yet."""\n    pass\n'
        smells = scan_files([_file("a.py", src)])
        assert any(s.kind == "stub" for s in smells)

    def test_real_function_not_flagged(self):
        f = _file("a.py", "def real():\n    return 42\n")
        smells = scan_files([f])
        assert not any(s.kind == "stub" for s in smells)

    def test_async_function_stub_detected(self):
        f = _file("a.py", "async def stub():\n    pass\n")
        assert any(s.kind == "stub" for s in scan_files([f]))


# ---------------------------------------------------------------------------
# Bare except / swallowed exceptions
# ---------------------------------------------------------------------------

class TestExceptDetection:
    def test_bare_except_flagged(self):
        f = _file("a.py", "try:\n    x = 1\nexcept:\n    log_it()\n")
        kinds = {s.kind for s in scan_files([f])}
        assert "bare_except" in kinds
        assert "swallowed_except" not in kinds  # body isn't silent

    def test_swallowed_typed_except(self):
        # Catches a specific exception but silently drops it.
        src = "try:\n    risky()\nexcept ValueError:\n    pass\n"
        kinds = {s.kind for s in scan_files([_file("a.py", src)])}
        assert "swallowed_except" in kinds
        assert "bare_except" not in kinds  # type IS specified

    def test_bare_and_swallowed_both_fire(self):
        # Worst case: bare `except:` AND silently dropped.
        src = "try:\n    risky()\nexcept:\n    pass\n"
        kinds = {s.kind for s in scan_files([_file("a.py", src)])}
        assert "bare_except" in kinds
        assert "swallowed_except" in kinds

    def test_ellipsis_body_is_swallowing(self):
        src = "try:\n    risky()\nexcept Exception:\n    ...\n"
        kinds = {s.kind for s in scan_files([_file("a.py", src)])}
        assert "swallowed_except" in kinds

    def test_handled_exception_not_flagged(self):
        src = (
            "try:\n"
            "    risky()\n"
            "except ValueError as e:\n"
            "    raise RuntimeError('wrapped') from e\n"
        )
        kinds = {s.kind for s in scan_files([_file("a.py", src)])}
        assert "swallowed_except" not in kinds
        assert "bare_except" not in kinds


# ---------------------------------------------------------------------------
# Empty catch blocks (JS/TS/Java/C#/C++) — tree-sitter backed
# ---------------------------------------------------------------------------

class TestSwallowedCatchDetection:
    def test_javascript_empty_catch(self):
        src = "try { doThing(); } catch (e) {}\n"
        kinds = {s.kind for s in scan_files([_file("a.js", src)])}
        assert "swallowed_catch" in kinds

    def test_typescript_empty_catch(self):
        src = "try { doThing(); } catch (e: unknown) {}\n"
        kinds = {s.kind for s in scan_files([_file("a.ts", src)])}
        assert "swallowed_catch" in kinds

    def test_tsx_empty_catch(self):
        src = "function f() { try { x(); } catch (e) {} }\n"
        kinds = {s.kind for s in scan_files([_file("a.tsx", src)])}
        assert "swallowed_catch" in kinds

    def test_java_empty_catch(self):
        src = (
            "class A {\n"
            "  void m() {\n"
            "    try { risky(); } catch (Exception e) {}\n"
            "  }\n"
            "}\n"
        )
        kinds = {s.kind for s in scan_files([_file("A.java", src)])}
        assert "swallowed_catch" in kinds

    def test_csharp_empty_catch(self):
        src = (
            "class A {\n"
            "  void M() {\n"
            "    try { Risky(); } catch (Exception e) {}\n"
            "  }\n"
            "}\n"
        )
        kinds = {s.kind for s in scan_files([_file("A.cs", src)])}
        assert "swallowed_catch" in kinds

    def test_cpp_empty_catch(self):
        src = (
            "void m() {\n"
            "  try { risky(); } catch (const std::exception& e) {}\n"
            "}\n"
        )
        kinds = {s.kind for s in scan_files([_file("a.cpp", src)])}
        assert "swallowed_catch" in kinds

    def test_comment_only_body_flagged(self):
        # Body has only a comment — structurally equivalent to an empty
        # body. The developer affirmatively chose to do nothing.
        src = "try { x(); } catch (e) { /* ignore */ }\n"
        kinds = {s.kind for s in scan_files([_file("a.js", src)])}
        assert "swallowed_catch" in kinds

    def test_non_empty_body_not_flagged(self):
        src = "try { x(); } catch (e) { console.error(e); }\n"
        kinds = {s.kind for s in scan_files([_file("a.js", src)])}
        assert "swallowed_catch" not in kinds

    def test_rethrow_not_flagged(self):
        src = "try { x(); } catch (e) { throw e; }\n"
        kinds = {s.kind for s in scan_files([_file("a.js", src)])}
        assert "swallowed_catch" not in kinds

    def test_line_number_points_to_catch(self):
        src = "function f() {\n  try {\n    x();\n  } catch (e) {}\n}\n"
        smells = [s for s in scan_files([_file("a.js", src)]) if s.kind == "swallowed_catch"]
        assert smells
        assert smells[0].line == 4

    def test_parse_error_does_not_crash(self):
        # Severely malformed source — scanner must not raise.
        src = "function f( { try { x() catch \n"
        # Should produce zero smells (or just markers) rather than crash.
        result = scan_files([_file("a.js", src)])
        assert isinstance(result, list)

    def test_parse_error_does_not_emit_false_positive(self):
        # Mid-keystroke source: a half-typed catch is structurally a
        # `catch_clause` to tree-sitter (with an ERROR descendant), but
        # we can't trust its body — skip rather than fire.
        src = "try { x(); } catch (e\n"
        kinds = {s.kind for s in scan_files([_file("a.js", src)])}
        assert "swallowed_catch" not in kinds

    def test_optional_catch_binding(self):
        # ES2019: `catch` without a parameter. Body is still the only
        # named child; must still flag when empty.
        src = "try { x(); } catch {}\n"
        kinds = {s.kind for s in scan_files([_file("a.js", src)])}
        assert "swallowed_catch" in kinds

    def test_nested_catch_only_outer_fires(self):
        # Inner catch has real handling; outer is empty. Only outer
        # should fire.
        src = (
            "try {\n"
            "  try { x(); } catch (inner) { console.error(inner); }\n"
            "} catch (outer) {}\n"
        )
        catches = [s for s in scan_files([_file("a.js", src)]) if s.kind == "swallowed_catch"]
        assert len(catches) == 1
        # The outer catch is on line 3 of this snippet.
        assert catches[0].line == 3

    # --- PHP (catch_clause, like JS/Java/C#) ---

    def test_php_empty_catch_flagged(self):
        src = "<?php\ntry { x(); } catch (Exception $e) {}\n"
        kinds = {s.kind for s in scan_files([_file("a.php", src)])}
        assert "swallowed_catch" in kinds

    def test_php_non_empty_catch_not_flagged(self):
        src = "<?php\ntry { x(); } catch (Exception $e) { log($e); }\n"
        kinds = {s.kind for s in scan_files([_file("a.php", src)])}
        assert "swallowed_catch" not in kinds

    # --- Kotlin (catch_block; empty = no `statements` child) ---

    def test_kotlin_empty_catch_flagged(self):
        src = "fun f() { try { x() } catch (e: Exception) {} }\n"
        kinds = {s.kind for s in scan_files([_file("a.kt", src)])}
        assert "swallowed_catch" in kinds

    def test_kotlin_non_empty_catch_not_flagged(self):
        src = "fun f() { try { x() } catch (e: Exception) { log(e) } }\n"
        kinds = {s.kind for s in scan_files([_file("a.kt", src)])}
        assert "swallowed_catch" not in kinds

    # --- Swift (catch_block; empty = no `statements` child) ---

    def test_swift_empty_catch_flagged(self):
        src = "func f() throws { do { try x() } catch {} }\n"
        kinds = {s.kind for s in scan_files([_file("a.swift", src)])}
        assert "swallowed_catch" in kinds

    def test_swift_non_empty_catch_not_flagged(self):
        src = "func f() throws { do { try x() } catch { print(error) } }\n"
        kinds = {s.kind for s in scan_files([_file("a.swift", src)])}
        assert "swallowed_catch" not in kinds

    # --- Ruby (rescue; empty = no `then` child) ---

    def test_ruby_empty_rescue_flagged(self):
        src = "begin\n  x = 1\nrescue StandardError => e\nend\n"
        smells = [s for s in scan_files([_file("a.rb", src)]) if s.kind == "swallowed_catch"]
        assert smells
        # Idiom label should say "rescue" not "catch" for Ruby.
        assert "rescue" in smells[0].message

    def test_ruby_non_empty_rescue_not_flagged(self):
        src = (
            "begin\n  x = 1\nrescue StandardError => e\n  log(e)\nend\n"
        )
        kinds = {s.kind for s in scan_files([_file("a.rb", src)])}
        assert "swallowed_catch" not in kinds

    # --- Go (`if err != nil { }` and variants) ---

    def test_go_empty_err_nil_check_flagged(self):
        src = (
            "package main\n"
            "func a() {\n"
            "  err := doStuff()\n"
            "  if err != nil {\n"
            "  }\n"
            "}\n"
        )
        smells = [s for s in scan_files([_file("a.go", src)]) if s.kind == "swallowed_catch"]
        assert smells
        assert "nil-check" in smells[0].message

    def test_go_err_nil_check_with_initializer_flagged(self):
        # `if err := f(); err != nil { }` — initializer form
        src = (
            "package main\n"
            "func a() {\n"
            "  if err := doStuff(); err != nil {\n"
            "  }\n"
            "}\n"
        )
        kinds = {s.kind for s in scan_files([_file("a.go", src)])}
        assert "swallowed_catch" in kinds

    def test_go_non_empty_err_check_not_flagged(self):
        src = (
            "package main\n"
            "func a() {\n"
            "  err := doStuff()\n"
            "  if err != nil {\n"
            "    log.Print(err)\n"
            "  }\n"
            "}\n"
        )
        kinds = {s.kind for s in scan_files([_file("a.go", src)])}
        assert "swallowed_catch" not in kinds

    def test_go_non_nil_condition_not_flagged(self):
        # Only nil-comparison conditions are flagged. An empty `if x > 0`
        # block isn't an error-handling smell.
        src = (
            "package main\n"
            "func a() {\n"
            "  x := 5\n"
            "  if x > 0 {\n"
            "  }\n"
            "}\n"
        )
        kinds = {s.kind for s in scan_files([_file("a.go", src)])}
        assert "swallowed_catch" not in kinds

    # --- Rust (`if let Err` and `match Err =>`) ---

    def test_rust_empty_if_let_err_flagged(self):
        src = (
            "fn a() {\n"
            "    if let Err(_) = result() {\n"
            "    }\n"
            "}\n"
        )
        smells = [s for s in scan_files([_file("a.rs", src)]) if s.kind == "swallowed_catch"]
        assert smells
        assert "if-let Err" in smells[0].message

    def test_rust_non_empty_if_let_err_not_flagged(self):
        src = (
            "fn a() {\n"
            "    if let Err(e) = result() {\n"
            "        log(e);\n"
            "    }\n"
            "}\n"
        )
        kinds = {s.kind for s in scan_files([_file("a.rs", src)])}
        assert "swallowed_catch" not in kinds

    def test_rust_if_let_ok_not_flagged(self):
        # Only Err patterns are flagged; if let Ok with an empty body is
        # unusual but isn't error swallowing.
        src = (
            "fn a() {\n"
            "    if let Ok(_) = result() {\n"
            "    }\n"
            "}\n"
        )
        kinds = {s.kind for s in scan_files([_file("a.rs", src)])}
        assert "swallowed_catch" not in kinds

    def test_rust_empty_err_match_arm_flagged(self):
        src = (
            "fn a() {\n"
            "    match result() {\n"
            "        Err(_) => {},\n"
            "        Ok(v) => use_it(v),\n"
            "    }\n"
            "}\n"
        )
        smells = [s for s in scan_files([_file("a.rs", src)]) if s.kind == "swallowed_catch"]
        assert smells
        assert "Err match arm" in smells[0].message

    def test_rust_non_empty_err_arm_not_flagged(self):
        src = (
            "fn a() {\n"
            "    match result() {\n"
            "        Err(e) => { log(e); },\n"
            "        Ok(v) => use_it(v),\n"
            "    }\n"
            "}\n"
        )
        kinds = {s.kind for s in scan_files([_file("a.rs", src)])}
        assert "swallowed_catch" not in kinds

    def test_rust_empty_ok_arm_not_flagged(self):
        # Empty Ok arm isn't an error-swallow — Ok with no body just
        # means "success requires no action". Only Err arms are flagged.
        src = (
            "fn a() {\n"
            "    match result() {\n"
            "        Ok(_) => {},\n"
            "        Err(e) => log(e),\n"
            "    }\n"
            "}\n"
        )
        kinds = {s.kind for s in scan_files([_file("a.rs", src)])}
        assert "swallowed_catch" not in kinds


# ---------------------------------------------------------------------------
# Hardcoded credentials
# ---------------------------------------------------------------------------

class TestSecretDetection:
    def test_openai_key_flagged(self):
        # Synthetic key — never a real one.
        f = _file("a.py", 'API_KEY = "sk-' + "X" * 30 + '"\n')
        smells = scan_files([f])
        secret = next(s for s in smells if s.kind == "secret")
        assert secret.severity == "high"
        assert "OpenAI" in secret.message

    def test_aws_access_key_flagged(self):
        f = _file("a.txt", "AKIAIOSFODNN7EXAMPLE\n")
        secret = next(s for s in scan_files([f]) if s.kind == "secret")
        assert "AWS" in secret.message

    def test_github_token_flagged(self):
        f = _file("a.txt", "token = ghp_" + "A" * 30 + "\n")
        assert any(s.kind == "secret" for s in scan_files([f]))

    def test_short_random_string_not_flagged(self):
        # Generic high-entropy strings should NOT trigger — we only match
        # known prefixed shapes.
        f = _file("a.py", 'token = "abc123def456"\n')
        smells = scan_files([f])
        assert not any(s.kind == "secret" for s in smells)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_syntax_error_python_falls_through_to_marker_scan(self):
        # File doesn't parse — Python detectors silently skip, but the
        # marker scan (regex-based) still fires.
        src = "def broken(:\n    # TODO fix syntax\n"
        smells = scan_files([_file("a.py", src)])
        assert any(s.kind == "todo" for s in smells)
        # Stub detection is AST-only, so it can't fire here.
        assert not any(s.kind == "stub" for s in smells)

    def test_empty_content_yields_no_findings(self):
        assert scan_files([_file("a.py", "")]) == []

    def test_nonexistent_file_extension_still_scans_markers(self):
        f = _file("README", "# TODO write docs\n")
        smells = scan_files([f])
        assert any(s.kind == "todo" for s in smells)

    def test_per_file_cap_enforced(self):
        # 20 TODOs in one file → only 8 returned (the per-file cap).
        content = "\n".join(f"# TODO item {i}" for i in range(20))
        smells = scan_files([_file("a.py", content)])
        assert len(smells) == 8


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

class TestFormatForPrompt:
    def test_empty_returns_empty_string(self):
        assert format_for_prompt([]) == ""

    def test_renders_severity_high_first(self):
        smells = [
            CodeSmell("a.py", 1, "todo", "info", "TODO: x", "# TODO: x"),
            CodeSmell("a.py", 2, "secret", "high", "OpenAI key", "..."),
            CodeSmell("a.py", 3, "stub", "warn", "stub", "..."),
        ]
        out = format_for_prompt(smells)
        # high should come first, then warn, then info
        assert out.find("[secret/high]") < out.find("[stub/warn]")
        assert out.find("[stub/warn]") < out.find("[todo/info]")

    def test_truncation_note_when_over_max(self):
        smells = [
            CodeSmell("a.py", i, "todo", "info", f"item {i}", f"# TODO {i}")
            for i in range(25)
        ]
        out = format_for_prompt(smells, max_findings=20)
        assert "+5 more findings suppressed" in out


# ---------------------------------------------------------------------------
# End-to-end on a realistic mixed file
# ---------------------------------------------------------------------------

def test_end_to_end_mixed_file():
    src = (
        "import os\n"
        "\n"
        "# TODO: load from env\n"
        "API_KEY = 'sk-" + "X" * 30 + "'\n"
        "\n"
        "def authenticate():\n"
        "    pass\n"
        "\n"
        "def call_api():\n"
        "    try:\n"
        "        do_something()\n"
        "    except:\n"
        "        pass\n"
    )
    smells = scan_files([_file("auth.py", src)])
    kinds = {s.kind for s in smells}
    # All five detectors fire on this one file.
    assert kinds == {"todo", "secret", "stub", "bare_except", "swallowed_except"}
