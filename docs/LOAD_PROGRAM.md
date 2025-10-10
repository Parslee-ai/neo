# Load Program - Design Document

## Matrix Metaphor

```
Operator → Trainer process that "loads" programs
Training pack → Dataset slice from HuggingFace
Neural jack → Embedding pipeline
The Construct → Local ~/.neo memory store + FAISS index
Dojo session → Evaluation pass that updates confidence
"I know kung fu" → New patterns become retrievable
```

## Concept

**Goal**: Expand Neo's memory with reusable reasoning patterns from datasets
**NOT**: Model fine-tuning, gradient updates, or weight training
**IS**: Retrieval learning - append patterns to local semantic memory

## CLI Interface

### MVP Command
```bash
neo --load-program <dataset_id> \
    --split <train|test|validation> \
    --columns <json_mapping>
```

### Examples
```bash
# Load Python basics from MBPP (recommended starter)
neo --load-program mbpp --split train --limit 1000

# Load from OpenAI HumanEval (164 coding problems)
neo --load-program openai_humaneval --split test

# Load from BigCode HumanEvalPack (multi-language)
neo --load-program bigcode/humanevalpack --split test --limit 500

# Dry run preview
neo --load-program mbpp --dry-run

# Custom column mapping
neo --load-program my_dataset --columns '{"text":"pattern","code":"solution"}'
```

### Required Parameters
- `dataset_id`: HuggingFace dataset identifier
- `--split`: Dataset split (train/test/validation)
- `--columns`: JSON mapping of dataset columns to pattern fields

### Optional Parameters
- `--limit N`: Cap at N samples (default: 1000)
- `--mode`: `append` (default) or `rebuild`
- `--dry-run`: Preview without importing
- `--dojo`: Run evaluation pass after import
- `--quiet`: Suppress progress, show only final report
- `--allow-license <SPDX>`: Override license check

## Data Flow

```
1. ACQUIRE   → Pull dataset slice from HuggingFace
2. NORMALIZE → Map rows to ReasoningEntry schema
3. DEDUPE    → Hash-based deduplication
4. EMBED     → Generate local embeddings (Jina Code v2)
5. INDEX     → Upsert into FAISS
6. DOJO      → (Optional) Evaluate on synthetic probes
7. REPORT    → Matrix quote + counts
```

## Recommended Datasets

### Out-of-Box Support (No Custom Mapping Required)

| Dataset | Size | Description | Best For |
|---------|------|-------------|----------|
| **mbpp** | 1,000 | Mostly Basic Programming Problems | Python basics, beginners |
| **openai_humaneval** | 164 | Hand-written programming problems | Function-level Python |
| **bigcode/humanevalpack** | Varies | Multi-language HumanEval variants | Cross-language patterns |
| **Muennighoff/natural-instructions** | Large | Task instructions with examples | NLP/text tasks |

### Usage
```bash
# All work without --columns flag
neo --load-program mbpp --split train --limit 1000
neo --load-program openai_humaneval --split test
neo --load-program bigcode/humanevalpack --split test
```

## Schema Mapping

### HuggingFace Row → ReasoningEntry

```python
{
  # Required mappings
  "text": entry.pattern,        # Problem/pattern description
  "code": entry.suggestion,     # Solution/code

  # Optional mappings
  "tags": entry.algorithm_type, # Algorithm category
  "difficulty": difficulty_affinity metadata
}
```

### Pre-Configured Mappings

```python
# MBPP
{"pattern": "text", "suggestion": "code"}

# OpenAI HumanEval
{"pattern": "prompt", "suggestion": "canonical_solution"}

# BigCode HumanEvalPack
{"pattern": "prompt", "suggestion": "canonical_solution"}
```

### Metadata Attached
```python
{
  "source": "hf",
  "dataset": "<dataset_id>",
  "split": "<split>",
  "imported_at": "<timestamp>",
  "lang": "<inferred>",
  "license": "<SPDX>"
}
```

## Data Hygiene

### License Check
- Query HuggingFace API for dataset license
- Block non-permissive licenses by default
- Allow override with `--allow-license`
- Store license in pattern metadata

### Deduplication
- Hash on `pattern + suggestion` fields
- Compare against existing memory via LSH (already implemented)
- Report dedupe rate in final output

### PII Guards
- Basic keyword scan for email/SSN patterns
- Warn on detection, allow override with flag
- Log suspicious patterns for review

### Quality Gates
- Pattern validity: Non-empty pattern + suggestion
- Code syntax: Optional Python AST validation
- Min confidence: Start at 0.3 (trainable via dojo)

## Output Format

### Success
```
"I know kung fu."

Loaded: 847 patterns
Deduped: 153 duplicates
Index rebuilt: 1.2s
Memory: 1247 total patterns
```

### With --dojo
```
"I know kung fu."

Loaded: 847 patterns
Dojo eval: 72% retrieval hit-rate
Confidence adjusted: 431 patterns
Index rebuilt: 1.2s
```

### Errors
```
Error: Dataset 'foo/bar' not found on HuggingFace
Error: License 'CC-BY-NC-ND' requires --allow-license flag
Error: Column mapping missing required field 'text'
```

## Implementation Plan

### Phase 1: MVP (Single Dataset)
- [ ] `src/neo/program_loader.py` - Core loader module
- [ ] Support MBPP dataset only
- [ ] Hardcoded column mapping
- [ ] Append-only mode
- [ ] Basic deduplication
- [ ] Matrix-style output

### Phase 2: Flexible Mapping
- [ ] JSON column mapping support
- [ ] Multiple dataset support (code_search_net, humaneval)
- [ ] Language detection (Python, JavaScript, etc.)
- [ ] License checking

### Phase 3: Quality & Evaluation
- [ ] Dojo evaluation pass
- [ ] PII guards
- [ ] Rebuild mode
- [ ] Quality metrics reporting

### Phase 4: Production Ready
- [ ] Comprehensive tests
- [ ] Documentation
- [ ] Error recovery
- [ ] Performance benchmarks

## File Structure

```
src/neo/
  program_loader.py      # Main loader implementation
  dataset_mappings.py    # Pre-configured dataset mappings

tests/
  test_program_loader.py # Loader tests
  test_dataset_quality.py # Quality gate tests

docs/
  LOAD_PROGRAM.md        # This file
```

## Non-Goals

- ❌ Model fine-tuning or gradient updates
- ❌ Remote state or cloud storage
- ❌ Source code mutation
- ❌ Automatic dataset discovery
- ❌ Multi-language embeddings (start with code only)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Column drift across datasets | Version mappings per dataset |
| Garbage patterns pollute retrieval | Quality gates + dedupe |
| Over-indexing generic Q&A | Focus on code-specific datasets |
| Index bloat | Auto-consolidation at 2000 entries |
| License violations | Block by default, explicit override |

## Success Metrics

- **Import speed**: <10s for 1000 patterns
- **Dedupe rate**: >20% on similar datasets
- **Retrieval quality**: +10% hit-rate after import
- **Index size**: <5MB for 2000 patterns
- **Code validity**: >95% parseable Python

## References

- ReasoningEntry schema: `src/neo/persistent_reasoning.py:76`
- Embedding pipeline: `src/neo/persistent_reasoning.py:604`
- FAISS indexing: `src/neo/persistent_reasoning.py:2078`
- CLI parsing: `src/neo/cli.py:2448`
