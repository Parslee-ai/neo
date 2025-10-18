# CONSTRUCT.md - The Construct Design Document

## Overview

**The Construct** is a semantic pattern library embedded in Neo - a curated collection of architecture and design patterns with semantic search capabilities. It serves as a knowledge base of vendor-agnostic engineering solutions, indexed using the same embedding technology that powers Neo's reasoning memory.

**Status**: Implemented in v0.7.7+
**Author**: mliotta
**Design Philosophy**: "There is no spoon" - patterns are concepts, not code

## Motivation

### The Problem

Modern software engineers face recurring challenges:
1. **Reinventing solutions**: Common problems (rate limiting, caching, circuit breakers) are solved repeatedly
2. **Vendor lock-in**: Most pattern libraries are tied to specific frameworks (Spring patterns, AWS Well-Architected)
3. **Discoverability**: Hard to find relevant patterns without knowing exact terminology
4. **Quality variance**: Blog posts and Stack Overflow answers vary wildly in quality

### The Solution

The Construct provides:
- **Curated patterns**: Reviewed for quality, completeness, and accuracy
- **Vendor-agnostic**: Describe concepts (cache, queue) not products (Redis, RabbitMQ)
- **Semantic search**: Find patterns by intent, not just keywords
- **Consequence-aware**: Document tradeoffs, failure modes, and observability

## Architecture

### Component Structure

```
/construct/
├── README.md                    # Pattern library guide
├── rate-limiting/
│   ├── token-bucket.md
│   ├── sliding-window.md
│   └── distributed-rate-limiting.md
├── caching/
│   ├── cache-aside.md
│   ├── write-through.md
│   └── cache-invalidation.md
└── <domain>/
    └── <pattern>.md

~/.neo/
├── construct_index.faiss        # FAISS index for semantic search
└── construct_metadata.json      # Pattern metadata cache
```

### Core Classes

**PatternSchema** (dataclass):
- `pattern_id`: e.g., "rate-limiting/token-bucket"
- `name`: Human-readable name
- `author`: GitHub username or full name
- `intent`, `forces`, `solution`, `consequences`, `references`: Content sections
- `embedding`: 768-dim vector (Jina Code v2)

**PatternReader**:
- `load(path)`: Parse markdown → PatternSchema
- Extracts structured sections from markdown
- Validates required fields (author, intent, forces, solution, consequences)

**PatternValidator**:
- `validate(pattern)`: Check quality constraints
- Author field mandatory
- Maximum 300 lines
- Minimum section lengths

**ConstructIndex**:
- `build_index()`: Scan /construct/, embed patterns, save to FAISS
- `load_index()`: Load pre-built FAISS index + metadata
- `search(query, top_k)`: Semantic search using embeddings
- `list_patterns(domain)`: File-based listing with optional filter
- `show_pattern(pattern_id)`: Load and display single pattern

### CLI Integration

New subcommand structure:
```bash
neo construct <action> [options]

Actions:
  list [--domain DOMAIN]          # List all patterns
  show <pattern-id>                # Display full pattern
  search <query> [--top-k K]      # Semantic search
  index [--force]                  # Build search index
```

Implementation:
- Added `subparsers` to `parse_args()` in cli.py
- Created `handle_construct(args)` function
- Wired into `main()` before other flag handlers

## Technical Decisions

### Embedding Model: Jina Code v2

**Choice**: `jinaai/jina-embeddings-v2-base-code` (768 dimensions)

**Rationale**:
- Code-optimized (trained on code + text)
- Same model used for Neo's reasoning memory (consistency)
- Local inference (no API calls, privacy-preserving)
- 768 dims (good balance of quality vs. storage)

**Alternatives considered**:
- OpenAI text-embedding-3-small (1536 dims): Requires API key, costs money
- BGE-small (384 dims): Lower quality for code
- Sentence-BERT (768 dims): Not code-optimized

### Storage: FAISS + JSON

**Choice**: FAISS index + JSON metadata

**Rationale**:
- FAISS: Fast approximate nearest neighbors (L2 distance)
- JSON: Human-readable metadata, easy to inspect/debug
- Local storage: No cloud dependencies
- <5MB total size: Easily portable

**Alternatives considered**:
- SQLite: Overkill for <100 patterns
- Pickle: Not human-readable
- Pure JSON: O(n) search, too slow

### Pattern Format: Markdown

**Choice**: Structured markdown with `##` section headers

**Rationale**:
- Human-readable and editable
- Git-friendly (diffs work well)
- No YAML parsing quirks
- Easy to preview in GitHub

**Alternatives considered**:
- YAML frontmatter: Harder to write, parsing errors
- JSON: Not human-friendly for long text
- Custom DSL: Unnecessary complexity

### Validation: Parse-time

**Choice**: Validate patterns when loading, not at commit time

**Rationale**:
- Fast feedback during development
- No CI/CD dependencies
- Patterns can be fixed without rebuilding index
- Warnings logged but don't block usage

**Alternatives considered**:
- Pre-commit hooks: Slows down commits
- CI validation: Delayed feedback
- Runtime validation: Too late

## Performance Characteristics

### Build Index
- **Target**: <5s for 100 patterns
- **Actual**: ~2s for 6 patterns (well under budget)
- **Scaling**: O(n) in pattern count, O(d) in embedding dimension
- **Bottleneck**: Embedding generation (mitigated by local model)

### Search
- **Target**: <100ms warm cache
- **Actual**: ~20-50ms after index loaded
- **Scaling**: O(log n) with FAISS index
- **Bottleneck**: Initial index load (~500ms cold start)

### List
- **Target**: <50ms
- **Actual**: ~10ms (filesystem scan)
- **Scaling**: O(n) in pattern count
- **Bottleneck**: Filesystem I/O

### Show
- **Target**: <10ms
- **Actual**: ~2-5ms (single file read)
- **Scaling**: O(1)
- **Bottleneck**: Disk I/O

## Quality Standards

### Pattern Requirements

1. **Author Attribution**: Mandatory field, visible in output
2. **Line Limit**: Maximum 300 lines (keeps patterns focused)
3. **Vendor-Agnostic**: No AWS/GCP/Azure/specific products
4. **Consequence-Aware**: Must document both benefits AND risks
5. **Observability**: Include metrics and failure modes

### Review Process

1. Contributor submits PR with new pattern
2. Automated validation runs (author present, <300 lines)
3. Maintainer review for:
   - Accuracy (is the pattern correct?)
   - Completeness (all tradeoffs documented?)
   - Clarity (understandable to intermediate engineer?)
4. Community feedback (optional)
5. Merge and rebuild index

## Usage Patterns

### Discovery Flow

1. Developer encounters problem (e.g., "API getting hammered")
2. Search semantically: `neo construct search "prevent api abuse"`
3. Review top results (token bucket, rate limiting, circuit breaker)
4. Show detailed pattern: `neo construct show rate-limiting/token-bucket`
5. Adapt solution to specific context

### Learning Flow

1. Developer wants to understand a domain
2. List patterns: `neo construct list --domain caching`
3. Read each pattern sequentially
4. Compare tradeoffs between approaches
5. Bookmark relevant patterns for future reference

### Contribution Flow

1. Developer solves a problem using a pattern
2. Check if pattern exists: `neo construct search "my solution"`
3. If not, write pattern using template
4. Submit PR with pattern file
5. Respond to review feedback
6. Pattern merged, benefits entire community

## Future Enhancements

### Phase 2: Advanced Search
- **Multi-pattern queries**: "caching AND rate limiting"
- **Filtering**: By author, date, domain, complexity
- **Ranking refinement**: Incorporate user feedback on relevance

### Phase 3: Pattern Relationships
- **Dependencies**: Pattern X requires pattern Y
- **Alternatives**: Pattern X vs. pattern Y comparison
- **Compositions**: Pattern X + Y solve problem Z

### Phase 4: Interactive Examples
- **Runnable code**: Link to example implementations
- **Visualization**: Sequence diagrams embedded in patterns
- **Failure simulations**: Chaos engineering examples

### Phase 5: Neo Integration
- **Auto-retrieval**: Neo automatically searches Construct for relevant patterns
- **Pattern application**: Neo suggests patterns in reasoning output
- **Feedback loop**: Track which patterns lead to successful solutions

## Testing Strategy

### Unit Tests (test_construct.py)

**PatternReader**:
- `test_pattern_reader_parses_author_field`: Verify author extraction
- `test_pattern_reader_rejects_missing_author`: Enforce requirement
- `test_pattern_reader_rejects_missing_sections`: Validate completeness
- `test_pattern_reader_extracts_all_sections`: End-to-end parsing

**PatternValidator**:
- `test_pattern_schema_validation`: Required fields and constraints
- `test_pattern_validation_author_required`: Author mandatory
- `test_pattern_validation_line_limit`: 300-line maximum

**ConstructIndex**:
- `test_construct_list_empty_directory`: Handle no patterns gracefully
- `test_construct_list_filtering_by_domain`: Domain filter works
- `test_construct_show_missing_pattern`: 404 handling
- `test_construct_show_malformed_yaml`: Parse error recovery
- `test_construct_search_with_zero_results`: No embedder fallback
- `test_construct_search_relevance_ordering`: Semantic ranking works
- `test_construct_index_build_performance`: <5s build time
- `test_construct_search_performance`: <100ms search time

**CLI Integration**:
- `test_construct_cli_backward_compatibility`: Existing commands unaffected
- `test_construct_list_command`: List output format
- `test_construct_show_command`: Show output format

### Integration Tests

**End-to-end flow**:
1. Add new pattern file
2. Run `neo construct index`
3. Search for pattern
4. Verify result relevance

**Upgrade path**:
1. Install older Neo version
2. Upgrade to version with Construct
3. Verify existing functionality preserved
4. Run Construct commands successfully

## Documentation

### User-Facing Docs

- **README.md**: Overview and quick start
- **/construct/README.md**: Contribution guide and pattern structure
- **CONSTRUCT.md** (this file): Design document and architecture
- **CONTRIBUTING.md**: Pattern submission process

### Developer Docs

- **src/neo/construct.py**: Inline docstrings for all classes/methods
- **tests/test_construct.py**: Test docstrings explain intent
- **Pattern template**: Example in /construct/README.md

## Migration and Compatibility

### Backward Compatibility

- No breaking changes to existing CLI
- `neo <prompt>` still works (subparser design)
- Existing config/memory unaffected
- Construct is opt-in feature

### Data Migration

- No migration needed (new feature)
- Index built on first use
- Graceful degradation if index missing

## Security Considerations

### Input Validation

- Pattern files: Trusted (only from repo)
- User queries: Sanitized before embedding
- File paths: Validated against construct directory

### Privacy

- All processing local (no external API calls)
- Embeddings generated on-device
- No telemetry or tracking

### Supply Chain

- Dependencies: fastembed, faiss-cpu (both OSS, well-maintained)
- No new network dependencies
- Pattern files reviewed before merge

## Success Metrics

### Adoption
- Patterns used in Neo reasoning (Phase 5)
- Community contributions (PR count)
- Pattern views (via analytics, if added)

### Quality
- Pattern validation pass rate (>95%)
- Community feedback scores
- Issue reports (bugs, unclear patterns)

### Performance
- Index build time (<5s for 100 patterns)
- Search latency (<100ms)
- Storage size (<5MB for 100 patterns)

## Acknowledgments

Design inspired by:
- **Design Patterns** (Gang of Four): Pattern structure
- **POSA** (Buschmann et al.): Consequences section
- **Release It!** (Nygard): Failure modes and observability
- **Neo's reasoning memory**: Semantic search architecture

Built with contributions from the Neo community.

---

**Questions or feedback?** Open an issue in the Neo repository.
