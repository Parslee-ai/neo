"""Centralized language-identity utilities for Neo.

Different subsystems used to maintain their own language maps — this
module is now the single source of truth for:
- file extension → canonical language name
- canonical language → GFM fence tag (for markdown rendering)
- canonical language → human-readable display name (for prompts)
- the set of fence tags we accept as "leading tag" markers when
  parsing LM responses

The canonical language IDs follow tree-sitter's naming convention
(`c_sharp`, not `csharp`), because that's the broadest existing
source of truth in this codebase. GFM tags and display names are
derived translations from canonical → presentation.

This module is intentionally pure-data — no tree-sitter imports —
so it can be loaded without any optional dependency present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


# Canonical: extension (with leading dot, lowercase) → tree-sitter
# language name. This is the broadest map; subsystems filter for the
# languages they support (e.g. code_smells skips Ruby because it
# doesn't use catch_clause).
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".cs": "c_sharp",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
}


# Non-canonical language labels that callers commonly pass in.
# Routed through `normalize_language_name` so the per-translation
# maps below stay canonical-only (no `csharp` row alongside `c_sharp`).
_LANGUAGE_ALIASES: dict[str, str] = {
    "csharp": "c_sharp",
    "c#": "c_sharp",
    "cs": "c_sharp",
    "js": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "ts": "typescript",
    "py": "python",
    "golang": "go",
    "c++": "cpp",
    "cc": "cpp",
    "rs": "rust",
    "rb": "ruby",
    "kt": "kotlin",
}


# Canonical language → GFM fence tag. Canonical keys only —
# normalization happens at the function boundary.
_FENCE_TAGS: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "tsx": "tsx",
    "java": "java",
    "c_sharp": "csharp",
    "go": "go",
    "rust": "rust",
    "c": "c",
    "cpp": "cpp",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
}


# Canonical language → human-readable display name (used in prompts
# like "Implement this in {display_name}:"). Canonical keys only.
_DISPLAY_NAMES: dict[str, str] = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "tsx": "TypeScript (TSX)",
    "java": "Java",
    "c_sharp": "C#",
    "go": "Go",
    "rust": "Rust",
    "c": "C",
    "cpp": "C++",
    "ruby": "Ruby",
    "php": "PHP",
    "swift": "Swift",
    "kotlin": "Kotlin",
}


# Tags we accept as a "leading language label" when parsing fenced LM
# responses. Larger than the canonical fence-tag map — includes common
# aliases (`py`, `js`, `rb`) and non-code tags (`json`, `yaml`, `sh`)
# we may see emitted by models. Used as a tight allowlist so generic
# code lines (`pass`, `done`, `42`) aren't mistaken for fence tags.
KNOWN_FENCE_TAGS: frozenset[str] = frozenset({
    "python", "py",
    "javascript", "js", "jsx",
    "typescript", "ts", "tsx",
    "java",
    "csharp", "cs", "c_sharp", "c#",
    "go", "golang",
    "rust", "rs",
    "c", "cpp", "c++", "cc",
    "ruby", "rb",
    "php",
    "swift",
    "kotlin", "kt",
    "sh", "bash", "zsh",
    "html", "css", "json", "yaml", "yml", "toml", "xml",
    "sql",
})


def normalize_language_name(language: Optional[str]) -> str:
    """Convert any common language label to its canonical form.

    Canonical form is the tree-sitter name (`c_sharp`, not `csharp`).
    Unknown labels are returned lowercased but otherwise unchanged.
    Returns the empty string for None or empty input.
    """
    if not language:
        return ""
    key = language.lower()
    return _LANGUAGE_ALIASES.get(key, key)


def language_for_path(path: Union[str, Path]) -> Optional[str]:
    """Resolve a file path to its canonical (tree-sitter) language name.

    Returns None for extensions we don't recognize — callers decide
    whether to ignore or fall back.
    """
    p = path if isinstance(path, Path) else Path(path)
    return EXTENSION_TO_LANGUAGE.get(p.suffix.lower())


def fence_tag_for(language: Optional[str]) -> str:
    """Map a language name (any common form) to its GFM fence tag.

    Input is normalized first so `csharp`, `c#`, and `c_sharp` all
    resolve to `csharp`. Returns the empty string when the language
    is unknown or None, so callers can write ```{fence_tag_for(lang)}
    and get a bare fence when there's no signal.
    """
    canonical = normalize_language_name(language)
    return _FENCE_TAGS.get(canonical, "")


def display_name_for(language: Optional[str]) -> str:
    """Map a language name (any common form) to its display name.

    Returns the (lowercased) input unchanged when unknown — callers
    that build prompts ("Implement this in {name}:") still get a
    pronounceable string even for languages we haven't mapped.
    Returns the empty string for None or empty input (no input ≠
    unmapped input).
    """
    canonical = normalize_language_name(language)
    if not canonical:
        return ""
    return _DISPLAY_NAMES.get(canonical, canonical)
