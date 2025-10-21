# Changelog

## [0.8.0] - 2025-10-21

### Added

**Release Automation**
- Added `/prepare-release` command for automated version bumping and changelog updates (#23)
- Added `/ship-release` command for complete release workflow with PR creation and PyPI publishing (#23)
- Automated version updates across pyproject.toml, __init__.py, and plugin.json

**The Construct - Semantic Pattern Discovery**
- Added semantic pattern discovery system for extracting reusable patterns from successful code (#24)
- Pattern extraction with confidence scoring and similarity-based clustering
- Integration with Neo's semantic memory for pattern recall and reuse
- Enables learning from successful implementations across projects

**Executable Artifacts & Incremental Planning**

*Grounded in recent code generation research (Liu ICLR 2023, Zhang 2023, Huang 2025, Yao NAACL 2024)*

**Executable Artifacts for CodeSuggestion**
- Added 7 optional fields to CodeSuggestion schema for actionable outputs:
  - `patch_content`: Full unified diff content (not truncated)
  - `apply_command`: Shell command to apply change (ADVISORY - validate before execution)
  - `rollback_command`: Shell command to undo change (ADVISORY)
  - `test_command`: Shell command to verify change (ADVISORY)
  - `dependencies`: Array of suggestion IDs this depends on (execution order)
  - `estimated_risk`: Enum (low/medium/high) for risk assessment
  - `blast_radius`: Float 0.0-100.0 percentage of codebase files affected (files changed / total files × 100)
- Security warnings: All command fields documented as ADVISORY ONLY (never use shell=True)
- Backward compatible: All new fields optional, schema version remains v3

**Incremental Planning for PlanStep**
- Added 8 optional fields to PlanStep schema for as-needed decomposition:
  - `preconditions[]`: Conditions that must be met before execution
  - `actions[]`: Concrete actions to perform in this step
  - `exit_criteria[]`: Success verification criteria
  - `risk`: Step-specific risk level (low/medium/high)
  - `retrieval_keys[]`: Keywords for step-scoped memory retrieval (CodeSim-style)
  - `failure_signatures[]`: Known failure patterns from past attempts (ReasoningBank)
  - `verifier_checks[]`: Validation checks (MapCoder's Solver-Critic-Verifier pattern)
  - `expanded`: Boolean tracking if step was expanded from seed plan
- Enables seed plan → expand when blocked workflow (Yao et al., NAACL 2024)
- Step-level failure learning for ReasoningBank integration (Chen et al., 2025)

**Testing & Quality**
- Added 8 comprehensive schema validation tests using jsonschema
- All tests use actual `jsonschema.validate()` (not mocks)
- Test coverage: 100% of new schema fields validated
- Tests verify enum constraints, range validation, and backward compatibility
- Code review: Linus agent ACCEPT (kernel-level quality standards met)

**Documentation**
- Enhanced README with detailed schema documentation
- Expanded Research & References section with 8 academic papers
- Added proper links to papers, GitHub repos, and datasets
- Included citation block for academic use

### Changed

**Schema Enhancements**
- `blast_radius`: Changed from integer (1-100) to float (0.0-100.0) for precision
  - Allows sub-1% impact representation (e.g., 0.5% for large codebases)
- Command field descriptions: Added security warnings about safe execution
- Schema validation: Maintained strict `additionalProperties: False` for safety

### Performance

- Schema validation overhead: <10ms per suggestion/step (O(1) constant time)
- Memory footprint: ~50 bytes per new field with default values (negligible)
- Backward compatibility: Zero impact on existing code (optional fields)

### Research References

This release implements concepts from:
- Liu et al., ICLR 2023 - Planning-guided code generation (preconditions, exit criteria)
- Zhang et al., 2023 - Self-planning workflow (+7% HumanEval improvement)
- Huang et al., 2025 - AdaCoder adaptive multi-agent framework (risk assessment)
- Islam et al., 2024 - MapCoder Solver-Critic-Verifier (verifier_checks)
- Xu et al., 2023 - CodeSim step-level retrieval (retrieval_keys)
- Yao et al., NAACL 2024 - As-needed decomposition (expanded flag, incremental planning)
- Chen et al., 2025 - ReasoningBank failure learning (failure_signatures)
- Wang et al., 2024 - Multi-agent survey (architectural foundations)

## [0.7.6] - 2025-10-14

### Fixed
- Python 3.9 compatibility: Replaced Python 3.10+ union syntax (X | Y) with Optional/Union for broader compatibility (#21)
- Added missing `source_context` field to ReasoningEntry dataclass (#20)

### Documentation
- Updated documentation files to latest standards

## [0.7.5] - 2025-10-10

### Changed
- Bumped version to 0.7.5 to match plugin version for consistency

### Fixed
- Plugin file paths: Ensured all file paths are correctly relative to the plugin root (#15)
- Plugin file paths: Fixed to be relative to repository root (#14)

### Added
- Updated plugin version to 0.7.5 and removed redundant README.md file (#13)
- Load program feature: HuggingFace dataset import (#12)
- Required YAML front matter to command files for Claude Code compatibility (#11)
- Plugin install step to README (#10)

### Changed
- Increased default max_entries from 200 to 2000 for larger memory capacity (#7)

### Fixed
- Claude Code plugin manifest schema validation errors (#9)

## [0.7.4] - 2025-10-10

### Fixed
- ImportError: Export CodeSuggestion, PlanStep, SimulationTrace, and StaticCheckResult from neo package (Fixes #5)
- Version sync: Updated __version__ in __init__.py from 0.7.0 to 0.7.4 to match pyproject.toml

### Added
- GitHub community files for open source management (#6):
  - SECURITY.md with vulnerability reporting policy
  - PR template with comprehensive checklist
  - dependabot.yml for automated dependency updates

## [0.7.0] - 2025-10-10

### Added - ReasoningBank Implementation (Phases 2-5)

*Based on ReasoningBank paper (arXiv:2509.25140v1)*

**Phase 2: Semantic Anchor Embedding**
- Implemented semantic anchor strategy: embeddings now use pattern+context only (not full reasoning)
- Reduces noise in similarity matching by focusing on WHAT+WHEN instead of HOW
- Backward compatible with existing embeddings (no re-embedding required)

**Phase 3: Systematic Failure Learning**
- Added failure root cause extraction when confidence < 0.5
- LLM-based failure analysis with heuristic fallback for reliability
- Failure patterns stored in `common_pitfalls` and surfaced in Neo output
- Tracks WHY patterns fail, not just that they failed

**Phase 4: Self-Contrast Consolidation**
- Added `problem_outcomes` tracking for contrastive learning
- Archetypal patterns (consistent winners) get +0.2 confidence boost
- Spurious patterns (lucky once, fail elsewhere) get -0.2 penalty
- Enables learning "which patterns work WHERE OTHERS FAIL"

**Phase 5: Strategy Evolution Tracking**
- Added strategy level inference: procedural, adaptive, compositional
- Difficulty-aware retrieval boosts (compositional +0.15 on hard problems)
- Procedural strategies penalized -0.10 on hard problems to prevent poor suggestions
- Zero new schema fields - pure algorithmic leverage from existing difficulty_affinity data

**Testing & Quality**
- Added 39 comprehensive tests (all passing)
- Integration test suite validates all phases working together
- Performance benchmarks: 12.3ms avg retrieval (target <100ms)
- Kernel-quality code review by Linus agent

**Documentation**
- Phase-specific documentation for each improvement (phases 2-5)
- Production readiness checklist with deployment plan
- Benchmark impact analysis and performance validation
- Linus review findings and fixes documented

### Changed

**Performance Optimizations**
- Replaced recursive DFS with iterative to eliminate RecursionError risk
- Extracted magic numbers to named class constants for tunability
- Consistent difficulty validation across all code paths

**Code Quality**
- Added named constants for all tunable parameters:
  - `AFFINITY_BONUS_WEIGHT = 0.2`
  - `CONTRASTIVE_SCALE = 0.4`
  - `STRATEGY_BOOST_HARD_COMPOSITIONAL = 0.15`
  - `CONFIDENCE_BOOST_SUCCESS = 0.1`
- Improved confidence reinforcement from ±0.02 to ±0.1 (stronger learning signals)

### Fixed
- RecursionError risk in clustering DFS (now uses iterative approach)
- Inconsistent difficulty validation (now defaults invalid values to "medium")
- Zero-vector edge case in cosine similarity (already handled, verified)

### Performance Metrics
- Retrieval latency: 12.3ms avg (87% faster than 100ms target)
- Consolidation: <50ms for 5-entry clusters
- Strategy inference: 66.7% accuracy on test cases
- Contrastive boost: ±0.4 difference (archetypal vs spurious)

### Technical Debt (Documented & Acceptable)
- O(n³) contrastive boost complexity (acceptable for <200 entries)
- Hardcoded strategy thresholds (66.7% accuracy acceptable for v1)
- Both items tracked for future optimization if needed

## [0.2.0] - 2025-09-30

### Added
- Plain text input mode with smart context gathering (CLI ergonomics like Claude Code)
- Context gathering with .gitignore-aware file discovery and git-based prioritization
- Keyword-based relevance scoring for context files
- Refactoring warnings for files >50KB (god object detection)
- Warning headers in LLM context for large files to enable specific refactoring suggestions
- Missing datasketch dependency for MinHash-based similarity detection

### Changed
- Lowered default max_bytes from 300KB to 100KB for better gpt-5-codex performance
- Strengthened size penalty: 10KB=-0.1, 50KB=-0.5, 100KB=-1.0 (favor smaller modules)
- Fixed OpenAI adapter to support gpt-5-codex /v1/responses endpoint
- Increased HTTP timeout from 60s to 300s for complex prompts

### Fixed
- Added context_gatherer module to package distribution
- OpenAI adapter now uses correct endpoint and minimal payload for gpt-5-codex

## [0.1.0] - Initial Release
