# Project index, eligibility walk cache, persistent content index

Detailed engineering notes moved verbatim from `CLAUDE.md` (2026-10-12). `CLAUDE.md` keeps the short rules; this file keeps the reasons, measurements and footguns behind them.

- Project index (`index/project_index.py`, `index/language_parser.py`; full
  notes in `docs/tree-sitter-setup.md`). Three invariants, each of which was
  violated and each of which produced an index that could not answer a question
  about its own repository:
  1. **Budgets are apportioned, never sliced.** `_select_files` groups eligible
     files by language and hands each a share of `--max-files` proportional to
     repo composition, floor of one slot per language (`_allocate_slots`); then
     `_cap_chunks` round-robins `MAX_CHUNKS_PER_REPO` across FILES so each keeps
     a chunk before any keeps a second. Both exist because a list slice is not a
     ranking: globbing `**/*.py` before `**/*.cs` and slicing gave a .NET repo of
     4,272 C# files an index of 83 Python files (95 from an in-repo worktree) and
     zero C#, exit 0. Fixing only the file cut left `chunks[:1000]` re-creating it
     one function later — chunks arrive grouped by file and files by language, so
     the slice kept 1000 C# chunks and dropped every other language, with 37 of
     the 100 selected files contributing nothing.
     **And the order WITHIN a language is source-before-tests, then depth, then
     alphabetical** — the test key being the load-bearing one. Depth alone is a
     centrality proxy that INVERTS on the conventional Python layout: with
     `src/<pkg>/…` beside `tests/…` every test sits at depth 2 and every source
     file at depth 3, so the whole test tree sorted ahead of the whole source
     tree and `--max-files` was spent before one source file was examined.
     Measured here: 131 Python files at depth 2, and the catalog came out
     **100% tests** while the build reported success — the cap had genuinely
     bound and the report was truthful about that, but nothing said the files
     it kept were all tests (#213). This module already treats "tests outrank
     the source they test" as a failure mode — `_embed_chunks` embeds a
     structured summary rather than the raw body precisely because assertion
     strings carry a query's keywords verbatim — but that mitigation runs at
     EMBEDDING time and so never got a chance; selection had already spent the
     budget. Tests are DEMOTED, not excluded: they fill the slots source does
     not need, so a test-only repo still indexes. `is_test_path` is IMPORTED
     from `context_gatherer`, never restated — a second copy of that rule would
     agree with the first only by coincidence, and its careful cases
     (`testdata/` and `testing/` are ordinary source; `Foo.Tests/` is not) are
     exactly what a re-implementation loses.
  2. **Exclusion is two layers and `bin`/`build`/`out`/`target`/`dist`/`vendor`
     belong to neither by name.** Both layers, and the walk that applies them,
     live in `neo/eligibility.py` — the ONE eligibility module, consumed by
     `context_gatherer`, `ProjectIndex` and `architecture_metrics` alike.
     `DEFAULT_IGNORE_PATTERNS` covers what repos forget to ignore (`.worktrees`,
     `.claude/worktrees`, `node_modules`, `obj`, virtualenvs);
     `load_ignore_patterns` layers the repo's own root `.gitignore`/`.ignore` on
     top, so a repo's `!negation` can re-include what a default excluded.
     `should_ignore` only tests the path handed to it, so ancestor directories
     must be handled separately — `walk` does that by PRUNING an ignored
     directory instead of descending, which is git's own rule and the reason a
     `!` beneath an excluded directory does not fire. Matching is exact and
     case-sensitive, against directory components only. The ambiguous names stay
     out because each is real source somewhere (`src/bin/main.rs`, vendored
     trees; 254 tracked files under `bin/`+`vendor/` across three local repos)
     and the asymmetry is one-sided: over-excluding hides code permanently,
     over-including only spends slots.
     **Three more exclusion classes are NOT gitignore, and conflating them
     sends an operator to the wrong file**: `WalkPolicy` knobs (symlink
     rejection, the gatherer's 512 KB `MAX_FILE_BYTES` ceiling, extension and
     per-language glob filters) are per-consumer policy; nested `.gitignore`
     files are not read, which under-excludes and is the accepted limit; and git
     applies ignore rules only to files it does not already TRACK, so a file
     added before a rule was written stays tracked while the walk still skips it
     (four `specs/*.md` on this checkout — recorded, deliberately not fixed in a
     pure refactor, since closing it means a `git ls-files` fork on the warm
     path). Two tests hold the line: `test_eligibility_single_source.py`
     AST-scans `src/` and fails on a second definition, a second `os.walk` or a
     second exclusion list (detected by CONTENT — three sentinel directory names
     in one literal — because a copy always renames the variable);
     `test_eligibility_differential.py` diffs the walk against `git check-ignore`
     over a fixture corpus AND this checkout, and fails on any tracked file
     skipped without an ignore rule accounting for it. Both are marked
     `invariants`, so they run in the Guard-invariant battery on every PR.
     **Footgun**: the index's `excluded` count is now excluded PATHS SEEN
     (`excluded_dirs` + `excluded_files`), not files under an excluded
     directory. A pruned subtree is one path; the walk does not descend and
     therefore does not know how many files are inside. The old "200 paths"
     number was only available because the old code globbed the whole repo and
     filtered afterwards — i.e. it walked every worktree copy in order to count
     what it was about to discard.
  3. **A cap that fired must be reported.** `selection_report` carries eligible /
     selected / excluded / duplicates / chunk counts and the CLI prints them.
     `truncated` means THE CAP BOUND US and is `examined < eligible` — whether
     any candidate went unlooked-at, which the per-language iterators already
     know. Both cheaper predicates name the wrong knob: `selected < eligible`
     made dedup print "2 of 7 eligible files (capped at --max-files=1000)", and
     `selected >= max_files` did the same whenever `--max-files` landed exactly
     on the unique-file count. Raising a cap that never bound fixes nothing.
     Separately, round-robin only represents every file while chunk slots ≥
     files; `MAX_CHUNKS_PER_REPO` is fixed at 1000 while `--max-files` is not,
     so `files_with_chunks` reports the shortfall — `truncated` is False there,
     because the FILE cap genuinely was not the constraint.
     **And the corollary the first version of this report broke: never blame a
     cap for an absence it did not cause.** A selected file is missing from the
     index for one of TWO unrelated reasons, and the console must not guess
     between them. Either it produced no chunks at all — no function, class,
     interface or struct for the grammar to match, as in an empty
     `__init__.py`, an enum-only `.cs`, a type-alias-only `.ts` — which no cap
     setting changes and which therefore gets a bare statement with NO remedy
     attached; or the cap took every chunk it produced, which gets the cap
     named and `lower --max-files`. `files_producing_chunks` is measured
     BEFORE `_cap_chunks` precisely so the two stay separable; subtracting the
     post-cap `files_with_chunks` from `selected` conflates them and is what
     printed "the 1000-chunk cap is below the 25 files selected" for a build
     that kept 559 of 559 chunks, with `chunks_capped` False on the same
     report. It fired on this repo, on any repo with an `__init__.py`. A report
     that invents a cause is worse than the silence it replaced, because
     silence at least does not send the operator to the wrong knob.
  **Query footgun** (`language_parser.py`; incident detail in the tree-sitter
  doc's "Why queries break silently"): a query that fails to compile is
  indistinguishable from one that matches nothing — `_get_query` returns None
  and `parse_file` moves on; edge failures log at DEBUG. Four were broken at
  once, so TS interfaces and ALL C# inheritance edges were absent from every
  index since they shipped. Three rules follow. Compile results are cached
  INCLUDING failures (the uncached retry warned per query *per file* — 9,699
  lines in one run). C# bases need all four of `identifier`, `generic_name`,
  and `qualified_name` with either as its `name:` field — across
  class/interface/record/struct — since a narrower pattern compiles fine and
  silently drops `: Repository<Order>`, `: System.Exception` and
  `interface IX : IY`. That query is GENERATED from `_CS_BASE_TYPES` over the
  four declaration kinds rather than written out four times: the hand-copied
  version is what left `interface_declaration` uncovered, and each widening
  since has had to be applied everywhere at once or not at all. And
  `test_every_chunk_query_compiles` /
  `test_every_edge_query_compiles` prove compilation ONLY, so a new query still
  needs its own behavioural assertion.
- Persistent eligibility walk (`index/walk_cache.py`, `eligibility.DirectoryListing`).
  The #208 walk is **kept on disk too**, in the same `.neo/`, because once the
  content index stopped rebuilding it became the largest single item in a warm
  call: 4.64 s on m365dotnet to re-derive that 9,348 files are eligible, on every
  invocation, in a repository that had not changed (#210). Warm walk now 0.16 s;
  the canonical M2 battery went 15.63 s → **8.57 s** median with byte-identical
  selection on all six prompts and byte-identical `rank_mine_eval` on all three
  flagships.
  - **The cost is the pattern matching, not the filesystem, and that decides the
    whole design.** Measured before the cache existed, on m365dotnet: the full
    walk **6.85 s**; the same traversal with the per-FILE ignore test removed
    **0.80 s**; `stat` over all 9,378 admitted files **0.10 s**. So 6.05 s of a
    6.85 s walk is `should_ignore` (11,219 calls), and caching the syscalls
    would have saved almost nothing. What is stored per directory is therefore
    the VERDICTS — which subdirectories survive, which filenames survive, and
    the exclusion counts — with the directory's stamps saying whether they hold.
  - **Directory mtime is the right key for exactly one reason**: on every POSIX
    filesystem it moves when an entry is created, deleted or renamed inside that
    directory, and does NOT move when the content of a file inside it changes.
    An edit can change what a file SAYS; it can never change whether it is
    eligible.
  - **mtime alone is not enough, because mtime is forgeable.** `touch -r`,
    `tar -x`, `rsync -a` and every snapshot restore write a directory's mtime
    back to a recorded value, so a restore that adds or deletes a file can land
    on exactly the mtime the cache holds and be reported `warm` — reproduced
    with two lines of `os.utime`, found by the fresh-verifier pass. `ctime_ns`
    is stored beside it: the inode change time moves on any metadata change, no
    API restores it, and it arrives in the same `stat`. On Windows `st_ctime` is
    a CREATION time and hence constant, which makes the extra comparison a no-op
    there rather than a false invalidation.
  - **Sizes and mtimes are never remembered.** They come fresh from the `stat`
    the walk owes its callers anyway (0.10 s), because the content index uses
    them as ITS freshness stamp — serving a remembered mtime would make an
    edited file look unedited and turn one cache's staleness into another's.
    `TestStampsAreNeverCached` pins it, mutation-verified.
  - **A `.gitignore` edit invalidates by CONTENT, not by any timestamp.** No
    directory's mtime moves when a pattern file is edited, so every stored
    verdict looks current while every one of them may now be wrong. The
    signature hashes the effective pattern list (shared defaults + the repo's own
    `.gitignore`/`.ignore`), plus a `MATCHER_VERSION` for the case the patterns
    are identical and `should_ignore` is not. Cost of an edit: one full walk,
    then warm again.
  - **`os.walk(followlinks=False)` had been doing work no one had named.** The
    traversal now recurses by hand, one directory at a time, so that flag
    protects only the single read it is passed — a link to an ancestor became an
    infinite descent and a link to `/` walked the machine. The refusal to
    descend into a symlinked DIRECTORY is restated explicitly and is not gated by
    `skip_symlinks`, which is about whether a symlinked FILE is delivered.
  - **A directory modified within `RACY_WINDOW_NS` (1 s) of being read is not
    trusted.** Timestamp granularity is not always finer than the interval
    between two events — HFS+ stamps whole seconds — so a directory read at E
    with mtime M can be modified afterwards into the same bucket whenever
    `E - M` is under one tick. Git carries the same guard under the name "racily
    clean". Its test had to pre-create `.neo/` to mean anything: writing the
    cache creates that directory, which moves the repository ROOT's mtime, so
    the call after a first-ever call re-lists the root whatever the guard does —
    the first cut of that test passed with the guard deleted.
  - **`extra_ignores` is never served from a cache and never writes one.** A
    caller's patterns are appended after the repo's own and gitignore is
    last-match-wins, so a `!negation` there can re-include what a stored verdict
    excluded — the stored verdict is not a stale answer, it is an answer to a
    different question. Nothing in Neo takes that path today (`--exclude` is
    applied after the walk); the guard exists so a future caller gets a correct
    answer rather than a fast wrong one. Reported as mode `bypassed`.
  - **JSON, not SQLite — the opposite conclusion from the same question.** Every
    directory is validated on every call, so the file is read whole every time,
    which is what JSON is for and what the semantic catalog beside it already
    uses. 507 KB and ~10 ms for m365dotnet. The neighbouring content index went
    to SQLite because a query touches ten terms of a few hundred thousand; the
    access shape decides, not the size.
  - The verdicts are the IGNORE layer only, so one cache serves every consumer:
    `--exts` / `match_globs` / `max_file_bytes` / `skip_symlinks` are applied on
    top, per call. `--index` and `architecture_metrics.compute` (which runs on
    the outcome-detection path of every real invocation) go through the same
    cache, so `--index` warms the walk as well as the catalog.
  - Degradations are loud and none is fatal: a corrupt, truncated, malformed or
    foreign-signature cache is discarded with a warning and the walk runs in
    full; an unwritable `.neo/` costs a warning and nothing else. A malformed
    ENTRY discards the whole file rather than the entry, because half a cache is
    half a repository, silently. `--dry-run` names which of cold / rebuilt /
    incremental (N of M directories) / warm / bypassed happened, in the
    selected-files block and in the `--json` payload's `walk_cache` key. A cold
    walk announces itself BEFORE it starts, since a first call on a large
    repository is seconds of otherwise-silent work.
- Persistent content index (`index/content_index.py`, `index/freshness.py`). The
  BM25 corpus above is **built once and kept on disk**, in the repository's own
  `.neo/` beside the semantic catalog, not re-derived per call. Rebuilding it per
  invocation was the entire cost of a Neo call on a large repo: the canonical M2
  battery on m365dotnet (9,348 eligible files) measured a **53.47 s** median wall
  and 1.95 GB peak RSS, against 18.50 s / 1.45 GB with the store (#195). Selection
  is UNCHANGED — byte-identical selected files and rank order on all six battery
  prompts, and byte-identical `rank_mine_eval` MRR / R@k on all three flagships
  (neo 0.712, aieweb 0.728, m365dotnet 0.669, 50 cases each).
  - **SQLite, not the catalog's JSON, and the access shape is the reason.** The
    catalog is read whole (every embedding participates in every query); a keyword
    query touches ten terms out of a few hundred thousand, so a parse-whole format
    would spend the warm budget deserializing postings nothing asked for. Index
    artifacts only — postings, doc lengths, hashes, a tokenizer/schema signature —
    never a file body; delivery still reads whole files fresh from disk. Cost is
    disk: 109 MB for m365dotnet, 5 MB for neo.
  - **The cheap stamp decides what to HASH, never that a file changed.** size +
    mtime come free from the walk's existing `stat` (`EligiblePath.mtime_ns`);
    only a content-hash mismatch counts as a change, so `touch` re-stamps and
    re-tokenizes nothing and is reported as `touched`, separately from `changed`.
    `UNRECORDED = -1` disables the cheap path for a store that persists hashes
    alone — and it is checked EXPLICITLY, because the first cut stamped both the
    store and the candidate with the sentinel and `-1 == -1` reported every
    changed file as unchanged.
  - **Eligibility arrives from the #208 walker and is never recomputed here** —
    `test_the_module_does_not_walk_the_filesystem` fails on an `os.walk`/`glob` in
    the module. A file the walker stops admitting (newly gitignored, deleted) has
    its postings dropped on the same pass, so a stale index cannot answer with an
    excluded file. Correspondingly the gatherer now walks ONCE, unfiltered, and
    applies `--exts`/`--include`/`--exclude` to the result: the index is a
    property of the REPOSITORY, and letting one `--exts py` call prune it would
    force the next call to rebuild. `--exclude` no longer prunes a directory
    subtree (it excludes every file under it via `should_ignore`, which matches a
    non-final component); the names that make pruning matter are in the walker's
    shared default list.
  - **Parity is enforced, not asserted.** `K1`/`B`/IDF are imported from
    `neo.memory.bm25` and the document is the same expression `FileIndex` used, so
    `TestParity` can score one corpus both ways and compare to floating point.
    Query-term MULTIPLICITY is preserved — deduping the query before hitting the
    postings table is a silent ranking change, since the scorer this replaces
    iterated the token LIST. **A filtered call gets filtered statistics**:
    `scores(prompt, candidates)` takes N, df and avgdl from `candidates`, not
    from the whole repository. Repo-global stats were the first cut and are the
    better IR design in the abstract; they are also a re-rank, and a
    fresh-verifier pass caught it — unflagged runs matched main exactly while
    `--exts py` changed all 25 selected lines. Now byte-identical under
    `--exts`, `--exclude` and `--include` as well.
  - Degradations are loud and none is fatal, and the **except-clause ORDER is the
    whole mechanism**: `sqlite3.OperationalError` is a SUBCLASS of
    `DatabaseError`, so `except DatabaseError` written first catches "database is
    locked" and runs the corruption handler — which `os.unlink`s a perfectly good
    store. Reproduced: the index was deleted AND the peer's committed transaction
    went into an unlinked inode and vanished with no error anywhere. Two Neo
    invocations in one repo is ordinary (an editor plugin and a shell), not
    exotic. So: **locked / read-only / full** → serve this call from memory via
    `FileIndex` and say so; **corrupt** (a `DatabaseError` that is NOT an
    `OperationalError`) → delete and rebuild, detected while OPENING because the
    memory fallback is permanent and a store that merely failed to open would pin
    that repo to the full per-call rebuild forever; **tokenizer/schema** bump →
    wipe and rebuild (no per-file hash can see it — the files did not change, the
    tokenizer did). `cold`/`rebuilt` are reported separately though both read
    everything: only one means something went wrong, and the corruption flag is
    CONSUMED, or a reused instance rebuilds on every refresh forever.
  - **A warm call opens no write transaction.** Rewriting the unchanged signature
    unconditionally made every invocation a writer, so two ordinary overlapping
    calls contended on the steady-state path rather than only during a rebuild —
    which is what made the clause-order bug reachable in normal use.
  - **A file that cannot be hashed keeps its path tokens.** `chmod 000` used to
    delete it from the corpus permanently, while the per-call index still ranked
    it on its name (its content simply read as empty). Permission was withdrawn
    from the CONTENT; the name is still a real name in the repository. The empty
    hash is safe as a comparison value because stamps are keyed by path —
    `"" == ""` is only ever asked of one file against its own previous state.
  - **The vocabulary is not preloaded to answer a query.** Reading every `terms`
    row to resolve ten of them cost a few hundred thousand rows on the warm path
    this module exists to make cheap; term ids resolve per query, and the full
    map is loaded only when writing.
  - Cold build is bounded and ANNOUNCED before it starts, with progress every 250
    files — 122 s for m365dotnet's 9,348, 2.3 s for neo's 307. `--dry-run` names
    which of cold / rebuilt / incremental (N files) / warm / memory happened, in
    the selected-files block and in the `--json` payload's `content_index` key
    (`--json` implies `--quiet`, so the stderr note is suppressed on exactly the
    path a machine consumer reads).
  - **M2's 500 MB target is NOT reachable from file selection and never was.**
    Warm profile on m365dotnet: imports 1.2 s, eligibility walk 4.6 s, content
    index refresh 0.4 s + scores 0.1 s, `_history_boost` 2.9 s **and +1.26 GB**.
    The RSS is the FactStore (152 MB of JSON with 768-dim embeddings inflating to
    ~1.3 GB of Python objects) — Goal 1's 1.43 GB baseline was never the
    gatherer's. What remains of the warm wall-clock is the walker and the memory
    system, in that order.
