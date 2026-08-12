"""Three small single-language repositories, built on demand.

Each one is a real git repository with a real `.gitignore`, because the
invariants under test are defined against `git check-ignore` and a fake
cannot be differentially compared to git. Each carries, in its own language:

- a **target** file the prompt names by path, holding a sentinel symbol that
  appears nowhere else. Naming the path pins the file (G2-inv); the sentinel
  is what lets the round trip prove the MODEL saw it rather than merely that
  the gatherer selected it.
- **gitignored junk** in the same language, which `git check-ignore` excludes
  and the gatherer therefore must not select (G1-inv). C# was silently absent
  from Neo's index for 8.5 months and the gatherer selected gitignored junk;
  both failures were invisible because nothing asserted the negative.
- a **duplicate copy** of the target under an agent worktree layout. Not
  gitignored — this is the other half of G1-inv, and it is asserted on
  identity (one copy of the basename) rather than on a count.
- **bulk below the sentinel**, so the target exceeds the reasoning prompt's
  per-file character cap. That makes the truncation-marker assertion
  (G3-inv) non-vacuous on the one file that matters, and it is realistic:
  real source files exceed 3000 characters constantly. The sentinel sits at
  the top because `truncate_marked` keeps the head.

`LANGUAGES` is the parametrization both gates iterate. Adding a language here
adds it to the free CI battery and to the paid release gate at once, which is
the only ordering that cannot leave a language gated in name only.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Comfortably past `engine._CONTEXT_FILE_CHARS` (3000) so the target file is
# always cut, and cut by enough that a marker claiming a plausible-but-wrong
# amount would be visible. Not imported from `engine` on purpose: the fixture
# must stay oversized if that constant is ever raised.
_TARGET_MIN_CHARS = 12_000


@dataclass(frozen=True)
class FixtureRepo:
    """A built fixture and everything the two gates need to assert against."""

    language: str
    root: Path
    #: Repo-relative path of the file the prompt names.
    target_rel: str
    #: Symbol present only in `target_rel`. The round trip requires the model
    #: to echo it, which it cannot do unless the file reached the prompt.
    sentinel: str
    #: The prompt handed to the gatherer and to the LLM.
    prompt: str
    #: Repo-relative paths `git check-ignore` excludes. Selecting any is G1-inv.
    ignored_rels: tuple[str, ...] = field(default_factory=tuple)
    #: Repo-relative path of the second copy of the target.
    duplicate_rel: str = ""
    #: Extension that identifies this language's source, without the dot.
    ext: str = ""


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


def _pad(comment: str, unit: str, minimum: int = _TARGET_MIN_CHARS) -> str:
    """Filler that reads as code, sized to push a file past `minimum`.

    `unit` is one syntactically valid declaration for the language; it is
    repeated with a running index so the result parses and so no two lines are
    identical (identical lines would let a line-level deduplicator collapse
    the bulk and quietly un-truncate the file).
    """
    body = [comment]
    i = 0
    while sum(len(line) + 1 for line in body) < minimum:
        body.append(unit.format(i=i))
        i += 1
    return "\n".join(body) + "\n"


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _init_repo(root: Path) -> None:
    """A real repository, so `git check-ignore` is answering about real state.

    `git init` is run with `cwd=root` under conftest's ambient-git scrub; see
    `tests/test_git_env_isolation.py` for why that scrub is load-bearing.
    """
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", ".")
    _git(root, "config", "user.email", "fixtures@neo.test")
    _git(root, "config", "user.name", "Neo Fixtures")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "fixture: initial import")


# --------------------------------------------------------------------------
# Python
# --------------------------------------------------------------------------

_PY_SENTINEL = "compute_zephyr_checksum"

_PY_TARGET = f'''"""Rolling checksum over a stream of records."""

CHECKSUM_MODULUS = 65521


def {_PY_SENTINEL}(records: list[bytes]) -> int:
    """Return the rolling checksum of `records`.

    The accumulator is seeded at 1 rather than 0 so that a leading empty
    record is distinguishable from no records at all.
    """
    low = 1
    high = 0
    for record in records:
        for byte in record:
            low = (low + byte) % CHECKSUM_MODULUS
            high = (high + low) % CHECKSUM_MODULUS
    return (high << 16) | low


def verify_checksum(records: list[bytes], expected: int) -> bool:
    return {_PY_SENTINEL}(records) == expected


''' + _pad(
    "# Supporting record codecs, kept in this module for locality.",
    "\n\ndef decode_record_{i}(payload: bytes) -> bytes:\n"
    "    \"\"\"Decode a version-{i} record payload.\"\"\"\n"
    "    return payload[{i} % 4:] or payload\n",
)

_PY_OTHER = '''"""Reader that feeds the checksum routine."""

from pathlib import Path


def read_records(path: Path) -> list[bytes]:
    return [line for line in path.read_bytes().splitlines() if line]
'''

_PY_IGNORED = '''"""Generated. Do not edit; do not read."""


def compute_zephyr_checksum(records):  # noqa: F811 - stale generated copy
    raise NotImplementedError("generated stub")
'''


# --------------------------------------------------------------------------
# C#
# --------------------------------------------------------------------------

_CS_SENTINEL = "ComputeZephyrChecksum"

_CS_TARGET = f'''using System;
using System.Collections.Generic;

namespace Fixture.Checksums;

/// <summary>Rolling checksum over a stream of records.</summary>
public sealed class ChecksumService : IChecksumService
{{
    private const int ChecksumModulus = 65521;

    /// <summary>Returns the rolling checksum of <paramref name="records"/>.</summary>
    /// <remarks>
    /// The accumulator is seeded at 1 rather than 0 so a leading empty record
    /// is distinguishable from no records at all.
    /// </remarks>
    public int {_CS_SENTINEL}(IReadOnlyList<byte[]> records)
    {{
        var low = 1;
        var high = 0;
        foreach (var record in records)
        {{
            foreach (var b in record)
            {{
                low = (low + b) % ChecksumModulus;
                high = (high + low) % ChecksumModulus;
            }}
        }}

        return (high << 16) | low;
    }}

    public bool VerifyChecksum(IReadOnlyList<byte[]> records, int expected)
        => {_CS_SENTINEL}(records) == expected;
}}

''' + _pad(
    "// Supporting record codecs, kept in this file for locality.",
    "\npublic static class RecordCodec{i}\n{{\n"
    "    /// <summary>Decodes a version-{i} record payload.</summary>\n"
    "    public static byte[] Decode(byte[] payload) => payload;\n}}\n",
)

_CS_OTHER = '''using System.Collections.Generic;

namespace Fixture.Checksums;

public interface IChecksumService
{
    int ComputeZephyrChecksumContract(IReadOnlyList<byte[]> records);
}
'''

_CS_IGNORED = '''// <auto-generated /> Build output. Do not edit; do not read.
namespace Fixture.Checksums.Generated;

public static class ChecksumServiceStub
{
    public static int ComputeZephyrChecksum() => throw new System.NotImplementedException();
}
'''


# --------------------------------------------------------------------------
# TypeScript
# --------------------------------------------------------------------------

_TS_SENTINEL = "computeZephyrChecksum"

_TS_TARGET = f'''/** Rolling checksum over a stream of records. */

export const CHECKSUM_MODULUS = 65521;

/**
 * Returns the rolling checksum of `records`.
 *
 * The accumulator is seeded at 1 rather than 0 so that a leading empty record
 * is distinguishable from no records at all.
 */
export function {_TS_SENTINEL}(records: Uint8Array[]): number {{
  let low = 1;
  let high = 0;
  for (const record of records) {{
    for (const byte of record) {{
      low = (low + byte) % CHECKSUM_MODULUS;
      high = (high + low) % CHECKSUM_MODULUS;
    }}
  }}
  return (high << 16) | low;
}}

export function verifyChecksum(records: Uint8Array[], expected: number): boolean {{
  return {_TS_SENTINEL}(records) === expected;
}}

''' + _pad(
    "// Supporting record codecs, kept in this module for locality.",
    "\n/** Decodes a version-{i} record payload. */\n"
    "export function decodeRecord{i}(payload: Uint8Array): Uint8Array {{\n"
    "  return payload.subarray({i} % 4);\n}}\n",
)

_TS_OTHER = '''/** Reader that feeds the checksum routine. */

export function readRecords(raw: string): Uint8Array[] {
  return raw.split("\\n").filter(Boolean).map((line) => new TextEncoder().encode(line));
}
'''

_TS_IGNORED = '''/* Generated. Do not edit; do not read. */
export function computeZephyrChecksum(): number {
  throw new Error("generated stub");
}
'''


_SPECS: dict[str, dict] = {
    "python": {
        "ext": "py",
        "sentinel": _PY_SENTINEL,
        "target_rel": "src/checksum_service.py",
        "target": _PY_TARGET,
        "others": {"src/record_reader.py": _PY_OTHER},
        "gitignore": "/generated/\n__pycache__/\n*.pyc\n",
        "ignored": {"generated/checksum_service_stub.py": _PY_IGNORED},
    },
    "csharp": {
        "ext": "cs",
        "sentinel": _CS_SENTINEL,
        "target_rel": "src/ChecksumService.cs",
        "target": _CS_TARGET,
        "others": {"src/IChecksumService.cs": _CS_OTHER},
        "gitignore": "/generated/\nobj/\n",
        "ignored": {"generated/ChecksumServiceStub.cs": _CS_IGNORED},
    },
    "typescript": {
        "ext": "ts",
        "sentinel": _TS_SENTINEL,
        "target_rel": "src/checksumService.ts",
        "target": _TS_TARGET,
        "others": {"src/recordReader.ts": _TS_OTHER},
        "gitignore": "/generated/\nnode_modules/\n",
        "ignored": {
            "generated/checksumServiceStub.ts": _TS_IGNORED,
            "node_modules/left-pad/index.ts": _TS_IGNORED,
        },
    },
}

#: Iteration order for both gates. Sorted so a failure report reads the same
#: on every runner.
LANGUAGES: tuple[str, ...] = tuple(sorted(_SPECS))


def build_fixture_repo(language: str, root: Path) -> FixtureRepo:
    """Build the `language` fixture at `root` and describe it.

    `root` need not exist. The returned `FixtureRepo` is what both gates
    assert against; nothing else should re-derive these paths.
    """
    try:
        spec = _SPECS[language]
    except KeyError:  # pragma: no cover - a typo in a parametrize list
        raise ValueError(
            f"unknown fixture language {language!r}; known: {LANGUAGES}"
        ) from None

    target_rel = spec["target_rel"]
    _write(root, target_rel, spec["target"])
    for rel, content in spec["others"].items():
        _write(root, rel, content)
    for rel, content in spec["ignored"].items():
        _write(root, rel, content)
    _write(root, ".gitignore", spec["gitignore"])

    # A second copy of the target under an agent worktree layout. Not
    # gitignored: the gatherer's own default exclusions are what must catch
    # it, and a worktree copy competes with the original for the same slots
    # rather than merely adding noise.
    duplicate_rel = f".claude/worktrees/scratch/{target_rel}"
    _write(root, duplicate_rel, spec["target"])

    _write(
        root,
        "README.md",
        "# checksum fixture\n\nA single-language fixture repository.\n",
    )
    _init_repo(root)

    return FixtureRepo(
        language=language,
        root=root,
        target_rel=target_rel,
        sentinel=spec["sentinel"],
        prompt=(
            f"Explain what {target_rel} does and identify the off-by-one risk "
            f"in its accumulator seeding. Name the function you are describing."
        ),
        ignored_rels=tuple(spec["ignored"]),
        duplicate_rel=duplicate_rel,
        ext=spec["ext"],
    )


def check_ignored(root: Path, rel_paths: list[str]) -> list[str]:
    """Return the subset of `rel_paths` that `git check-ignore` excludes.

    The differential half of G1-inv. Asking git rather than re-implementing
    its rules is the whole point — a second implementation of gitignore is
    the defect this guards against, not the guard.

    `check-ignore` exits 1 when nothing matches, which is not an error, so the
    return code is read rather than `check=True`.
    """
    if not rel_paths:
        return []
    proc = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=root,
        input="\n".join(rel_paths) + "\n",
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"git check-ignore failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line for line in proc.stdout.splitlines() if line]
