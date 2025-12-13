# Prompt Enhancement System - Architecture

## Overview

The Prompt Enhancement System automatically analyzes Claude Code usage patterns to identify effective prompts, track prompt evolution, and suggest improvements. It operates as a background learning system that runs on every `neo` invocation.

## Goals

1. **Learn what works** - Identify prompt patterns that lead to successful task completion
2. **Learn what doesn't** - Detect prompts that cause confusion, errors, or excessive iterations
3. **Track evolution** - Record when and why CLAUDE.md/commands change
4. **Suggest improvements** - Proactively recommend prompt enhancements
5. **Build pattern library** - Accumulate reusable effective prompt patterns

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROMPT ENHANCEMENT SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Scanner    │───▶│   Analyzer   │───▶│  Knowledge   │                  │
│  │              │    │              │    │    Base      │                  │
│  │ - History    │    │ - Effective  │    │              │                  │
│  │ - Sessions   │    │ - Signals    │    │ - Patterns   │                  │
│  │ - CLAUDE.md  │    │ - Evolution  │    │ - Evolutions │                  │
│  │ - Commands   │    │ - Patterns   │    │ - Metrics    │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Change Detector                                  │  │
│  │  - Tracks last scan timestamp per source                             │  │
│  │  - Only processes new/modified data                                  │  │
│  │  - Maintains watermarks in ~/.neo/prompt_watermarks.json             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Enhancement Engine                               │  │
│  │  - Generates improvement suggestions                                  │  │
│  │  - Auto-rewrites prompts (optional)                                  │  │
│  │  - Maintains prompt effectiveness scores                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Scanner (`src/neo/prompt/scanner.py`)

Responsible for reading Claude Code data from local files.

```python
@dataclass
class ClaudeCodeSources:
    """Paths to Claude Code data sources."""
    claude_home: Path = Path.home() / ".claude"

    @property
    def history_file(self) -> Path:
        return self.claude_home / "history.jsonl"

    @property
    def projects_dir(self) -> Path:
        return self.claude_home / "projects"

    @property
    def plans_dir(self) -> Path:
        return self.claude_home / "plans"

    @property
    def global_claude_md(self) -> Path:
        return self.claude_home / "CLAUDE.md"

    def find_project_claude_mds(self) -> list[Path]:
        """Find all project CLAUDE.md files."""
        # Search common locations
        patterns = [
            Path.home() / "git" / "*" / "CLAUDE.md",
            Path.home() / "projects" / "*" / "CLAUDE.md",
            Path.home() / "*" / "CLAUDE.md",
        ]
        # Also check .claude/commands directories
        ...

@dataclass
class ScannedPrompt:
    """A prompt extracted from Claude Code history."""
    text: str
    timestamp: datetime
    project: str
    session_id: str
    source: Literal["history", "session", "claude_md", "command"]

@dataclass
class ScannedSession:
    """A complete conversation session."""
    session_id: str
    project: str
    messages: list[dict]  # Full message history
    start_time: datetime
    end_time: datetime
    tool_calls: int
    errors: list[str]
    outcome: Optional[str]  # Inferred outcome

@dataclass
class ScannedClaudeMd:
    """A CLAUDE.md file with metadata."""
    path: Path
    content: str
    last_modified: datetime
    hash: str  # For change detection

class Scanner:
    """Scans Claude Code data sources."""

    def scan_history(self, since: datetime = None) -> list[ScannedPrompt]:
        """Scan history.jsonl for prompts."""
        ...

    def scan_sessions(self, project: str = None, since: datetime = None) -> list[ScannedSession]:
        """Scan project session files for full conversations."""
        ...

    def scan_claude_mds(self) -> list[ScannedClaudeMd]:
        """Find and scan all CLAUDE.md files."""
        ...

    def scan_commands(self) -> list[dict]:
        """Scan slash command definitions."""
        ...
```

### 2. Change Detector (`src/neo/prompt/change_detector.py`)

Tracks what has changed since the last scan to avoid reprocessing.

```python
@dataclass
class Watermark:
    """Tracks last processed position for a data source."""
    source: str  # e.g., "history", "session:project_name", "claude_md:/path"
    timestamp: datetime
    position: Optional[int]  # Line number for files
    hash: Optional[str]  # Content hash for CLAUDE.md files

class ChangeDetector:
    """Detects changes since last scan."""

    WATERMARK_FILE = Path.home() / ".neo" / "prompt_watermarks.json"

    def __init__(self):
        self.watermarks = self._load_watermarks()

    def get_changes_since_last_scan(self, scanner: Scanner) -> dict:
        """Get all changes since last scan."""
        return {
            "new_prompts": self._get_new_prompts(scanner),
            "new_sessions": self._get_new_sessions(scanner),
            "modified_claude_mds": self._get_modified_claude_mds(scanner),
            "modified_commands": self._get_modified_commands(scanner),
        }

    def _get_modified_claude_mds(self, scanner: Scanner) -> list[tuple[ScannedClaudeMd, ScannedClaudeMd]]:
        """Return (old, new) pairs for modified CLAUDE.md files."""
        # Compare hashes to detect changes
        # Return tuples of (previous_version, current_version)
        ...

    def update_watermarks(self):
        """Update watermarks after successful processing."""
        ...
```

### 3. Effectiveness Analyzer (`src/neo/prompt/analyzer.py`)

Analyzes conversations to determine prompt effectiveness using semantic signals.

```python
class EffectivenessSignal(Enum):
    """Signals that indicate prompt effectiveness."""

    # Positive signals
    TASK_COMPLETED = "task_completed"           # Task finished successfully
    SINGLE_ITERATION = "single_iteration"       # Done in one turn
    COMMIT_MADE = "commit_made"                 # Git commit created
    TESTS_PASSED = "tests_passed"               # Tests ran successfully
    TOPIC_CHANGED = "topic_changed"             # User moved to new topic (implicit success)

    # Negative signals
    IMMEDIATE_CLARIFICATION = "immediate_clarification"  # User had to clarify right away
    CLAUDE_CONFUSED = "claude_confused"         # "I'm not sure", "Could you clarify"
    MULTIPLE_RETRIES = "multiple_retries"       # Same task attempted multiple times
    ERROR_IN_RESPONSE = "error_in_response"     # Error messages in output
    ABANDONED_TASK = "abandoned_task"           # User gave up / changed topic abruptly

@dataclass
class PromptEffectivenessScore:
    """Effectiveness score for a prompt."""
    prompt_hash: str
    prompt_text: str
    score: float  # -1.0 (very ineffective) to 1.0 (very effective)
    signals: list[EffectivenessSignal]
    iterations_to_complete: int
    tool_calls: int
    sample_count: int  # How many times we've seen similar prompts
    confidence: float  # How confident we are in this score

@dataclass
class PromptPattern:
    """A reusable prompt pattern extracted from effective prompts."""
    pattern_id: str
    name: str
    description: str
    template: str  # Template with placeholders
    examples: list[str]  # Concrete examples
    effectiveness_score: float
    use_cases: list[str]
    anti_patterns: list[str]  # What NOT to do

class EffectivenessAnalyzer:
    """Analyzes prompt effectiveness using semantic signals."""

    # Patterns that indicate Claude confusion
    CONFUSION_PATTERNS = [
        r"I'm not sure",
        r"Could you clarify",
        r"I need more information",
        r"Can you provide more context",
        r"I don't understand",
        r"What do you mean by",
        r"Please specify",
    ]

    # Patterns that indicate success
    SUCCESS_PATTERNS = [
        r"Done\.",
        r"I've (completed|finished|implemented|fixed|created)",
        r"The (task|change|fix|feature) is complete",
        r"Successfully",
    ]

    def analyze_session(self, session: ScannedSession) -> list[PromptEffectivenessScore]:
        """Analyze a session to score each prompt's effectiveness."""
        scores = []
        messages = session.messages

        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue

            prompt_text = self._extract_prompt_text(msg)
            signals = self._detect_signals(messages, i)
            score = self._calculate_score(signals)

            scores.append(PromptEffectivenessScore(
                prompt_hash=self._hash_prompt(prompt_text),
                prompt_text=prompt_text,
                score=score,
                signals=signals,
                iterations_to_complete=self._count_iterations(messages, i),
                tool_calls=self._count_tool_calls(messages, i),
                sample_count=1,
                confidence=0.5,  # Single sample, low confidence
            ))

        return scores

    def _detect_signals(self, messages: list[dict], prompt_index: int) -> list[EffectivenessSignal]:
        """Detect effectiveness signals from conversation context."""
        signals = []

        # Look at the response and subsequent messages
        response = self._get_next_assistant_message(messages, prompt_index)
        next_user_msg = self._get_next_user_message(messages, prompt_index)

        # Check for confusion patterns in response
        if response and self._matches_patterns(response, self.CONFUSION_PATTERNS):
            signals.append(EffectivenessSignal.CLAUDE_CONFUSED)

        # Check for success patterns
        if response and self._matches_patterns(response, self.SUCCESS_PATTERNS):
            signals.append(EffectivenessSignal.TASK_COMPLETED)

        # Check if user had to immediately clarify
        if next_user_msg and self._is_clarification(next_user_msg, prompt_text):
            signals.append(EffectivenessSignal.IMMEDIATE_CLARIFICATION)

        # Check for commits (look for git commit in tool calls)
        if self._has_commit(messages, prompt_index):
            signals.append(EffectivenessSignal.COMMIT_MADE)

        # Check for test success
        if self._has_passing_tests(messages, prompt_index):
            signals.append(EffectivenessSignal.TESTS_PASSED)

        return signals

    def _calculate_score(self, signals: list[EffectivenessSignal]) -> float:
        """Calculate effectiveness score from signals."""
        score = 0.0

        # Positive signals
        if EffectivenessSignal.TASK_COMPLETED in signals:
            score += 0.4
        if EffectivenessSignal.SINGLE_ITERATION in signals:
            score += 0.2
        if EffectivenessSignal.COMMIT_MADE in signals:
            score += 0.2
        if EffectivenessSignal.TESTS_PASSED in signals:
            score += 0.2

        # Negative signals
        if EffectivenessSignal.CLAUDE_CONFUSED in signals:
            score -= 0.3
        if EffectivenessSignal.IMMEDIATE_CLARIFICATION in signals:
            score -= 0.2
        if EffectivenessSignal.MULTIPLE_RETRIES in signals:
            score -= 0.3
        if EffectivenessSignal.ERROR_IN_RESPONSE in signals:
            score -= 0.2

        return max(-1.0, min(1.0, score))

    def extract_patterns(self, effective_prompts: list[PromptEffectivenessScore]) -> list[PromptPattern]:
        """Extract reusable patterns from highly effective prompts."""
        # Cluster similar effective prompts
        # Abstract into templates
        # Identify common success factors
        ...
```

### 4. Evolution Tracker (`src/neo/prompt/evolution.py`)

Tracks changes to CLAUDE.md files and commands over time.

```python
@dataclass
class ClaudeMdEvolution:
    """Tracks a change to a CLAUDE.md file."""
    path: Path
    timestamp: datetime
    previous_content: str
    new_content: str
    diff: str  # Unified diff
    change_type: Literal["created", "modified", "deleted"]
    inferred_reason: Optional[str]  # Why the change was made (from context)

@dataclass
class CommandEvolution:
    """Tracks a change to a slash command."""
    command_name: str
    path: Path
    timestamp: datetime
    previous_content: str
    new_content: str
    diff: str

class EvolutionTracker:
    """Tracks evolution of CLAUDE.md files and commands."""

    EVOLUTION_FILE = Path.home() / ".neo" / "prompt_evolutions.json"

    def record_claude_md_change(self, old: ScannedClaudeMd, new: ScannedClaudeMd) -> ClaudeMdEvolution:
        """Record a change to a CLAUDE.md file."""
        diff = self._compute_diff(old.content, new.content)
        reason = self._infer_change_reason(old, new, diff)

        evolution = ClaudeMdEvolution(
            path=new.path,
            timestamp=datetime.now(),
            previous_content=old.content,
            new_content=new.content,
            diff=diff,
            change_type="modified",
            inferred_reason=reason,
        )

        self._save_evolution(evolution)
        return evolution

    def _infer_change_reason(self, old: ScannedClaudeMd, new: ScannedClaudeMd, diff: str) -> str:
        """Infer why a CLAUDE.md was changed based on recent sessions."""
        # Look at recent sessions for this project
        # Find patterns like:
        # - User repeatedly had to explain X -> Added rule about X
        # - Claude kept making mistake Y -> Added instruction to avoid Y
        # - User asked "can you do Z" multiple times -> Added capability Z
        ...

    def get_evolution_history(self, path: Path = None) -> list[ClaudeMdEvolution]:
        """Get evolution history, optionally filtered by path."""
        ...

    def suggest_improvements(self, sessions: list[ScannedSession]) -> list[dict]:
        """Suggest CLAUDE.md improvements based on session patterns."""
        suggestions = []

        # Pattern: User repeatedly clarifies same thing
        clarification_patterns = self._find_repeated_clarifications(sessions)
        for pattern in clarification_patterns:
            suggestions.append({
                "type": "add_rule",
                "target": pattern["project"],
                "suggestion": f"Add rule: {pattern['suggested_rule']}",
                "reason": f"User had to clarify '{pattern['topic']}' {pattern['count']} times",
                "confidence": pattern["confidence"],
            })

        # Pattern: Same errors keep occurring
        error_patterns = self._find_recurring_errors(sessions)
        for pattern in error_patterns:
            suggestions.append({
                "type": "add_constraint",
                "target": pattern["project"],
                "suggestion": f"Add constraint: {pattern['suggested_constraint']}",
                "reason": f"Error '{pattern['error']}' occurred {pattern['count']} times",
                "confidence": pattern["confidence"],
            })

        return suggestions
```

### 5. Prompt Knowledge Base (`src/neo/prompt/knowledge_base.py`)

Separate storage for prompt-related knowledge, distinct from neo's main memory.

```python
@dataclass
class PromptEntry:
    """An entry in the prompt knowledge base."""
    id: str
    entry_type: Literal["pattern", "score", "evolution", "suggestion"]
    data: dict
    embedding: Optional[list[float]]
    created_at: datetime
    updated_at: datetime
    project: Optional[str]  # None for global patterns

class PromptKnowledgeBase:
    """Separate knowledge base for prompt patterns and effectiveness data."""

    STORAGE_FILE = Path.home() / ".neo" / "prompt_knowledge.json"
    EMBEDDINGS_FILE = Path.home() / ".neo" / "prompt_embeddings.pkl"

    def __init__(self):
        self.entries: list[PromptEntry] = []
        self._load()

    # Pattern storage
    def add_pattern(self, pattern: PromptPattern):
        """Add an effective prompt pattern."""
        ...

    def search_patterns(self, query: str, k: int = 5) -> list[PromptPattern]:
        """Search for relevant patterns using semantic similarity."""
        ...

    # Effectiveness scores
    def update_effectiveness_score(self, score: PromptEffectivenessScore):
        """Update effectiveness score, aggregating with existing data."""
        existing = self._find_by_hash(score.prompt_hash)
        if existing:
            # Aggregate scores using weighted average
            existing.score = self._weighted_average(existing, score)
            existing.sample_count += 1
            existing.confidence = min(0.95, existing.confidence + 0.1)
        else:
            self._add_entry("score", score)

    # Evolution history
    def add_evolution(self, evolution: ClaudeMdEvolution):
        """Record a CLAUDE.md evolution."""
        ...

    def get_evolutions(self, path: Path = None) -> list[ClaudeMdEvolution]:
        """Get evolution history."""
        ...

    # Suggestions
    def add_suggestion(self, suggestion: dict):
        """Add an improvement suggestion."""
        ...

    def get_pending_suggestions(self, project: str = None) -> list[dict]:
        """Get unresolved suggestions."""
        ...

    # Statistics
    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        return {
            "total_entries": len(self.entries),
            "patterns": len([e for e in self.entries if e.entry_type == "pattern"]),
            "scores": len([e for e in self.entries if e.entry_type == "score"]),
            "evolutions": len([e for e in self.entries if e.entry_type == "evolution"]),
            "pending_suggestions": len(self.get_pending_suggestions()),
            "projects_tracked": len(set(e.project for e in self.entries if e.project)),
        }
```

### 6. Enhancement Engine (`src/neo/prompt/enhancer.py`)

Generates prompt improvements and rewrites.

```python
@dataclass
class PromptEnhancement:
    """A suggested enhancement to a prompt."""
    original: str
    enhanced: str
    improvements: list[str]  # What was improved
    expected_benefit: str
    confidence: float

class PromptEnhancer:
    """Generates prompt enhancements using learned patterns."""

    def __init__(self, knowledge_base: PromptKnowledgeBase, lm_adapter: LMAdapter):
        self.kb = knowledge_base
        self.lm = lm_adapter

    def enhance_prompt(self, prompt: str, context: dict = None) -> PromptEnhancement:
        """Enhance a prompt using learned patterns."""
        # 1. Find similar effective prompts
        similar = self.kb.search_patterns(prompt, k=3)

        # 2. Identify what's missing from this prompt
        issues = self._identify_issues(prompt, similar)

        # 3. Generate enhancement
        enhanced = self._generate_enhancement(prompt, similar, issues)

        return PromptEnhancement(
            original=prompt,
            enhanced=enhanced,
            improvements=issues,
            expected_benefit=self._estimate_benefit(issues),
            confidence=self._calculate_confidence(similar),
        )

    def _identify_issues(self, prompt: str, similar_effective: list[PromptPattern]) -> list[str]:
        """Identify what could be improved in the prompt."""
        issues = []

        # Check for vagueness
        if self._is_vague(prompt):
            issues.append("lacks_specificity")

        # Check for missing constraints
        if not self._has_constraints(prompt) and self._should_have_constraints(prompt):
            issues.append("missing_constraints")

        # Check for missing context
        if self._needs_context(prompt):
            issues.append("missing_context")

        # Compare structure to effective prompts
        structural_issues = self._compare_structure(prompt, similar_effective)
        issues.extend(structural_issues)

        return issues

    def suggest_claude_md_updates(self, project: str) -> list[dict]:
        """Suggest updates to a project's CLAUDE.md based on usage patterns."""
        # Get recent sessions for this project
        # Analyze recurring issues
        # Generate concrete suggestions
        ...

    def auto_enhance(self, prompt: str) -> str:
        """Automatically enhance a prompt without explanation."""
        enhancement = self.enhance_prompt(prompt)
        return enhancement.enhanced if enhancement.confidence > 0.7 else prompt
```

### 7. CLI Integration (`src/neo/prompt/cli.py`)

New `neo prompt` command.

```python
def register_prompt_commands(parser: argparse.ArgumentParser):
    """Register prompt-related CLI commands."""
    prompt_parser = parser.add_subparsers(dest="prompt_command")

    # neo prompt analyze
    analyze = prompt_parser.add_parser("analyze", help="Analyze prompt effectiveness")
    analyze.add_argument("--project", help="Specific project to analyze")
    analyze.add_argument("--since", help="Analyze sessions since date")

    # neo prompt enhance <prompt>
    enhance = prompt_parser.add_parser("enhance", help="Enhance a prompt")
    enhance.add_argument("prompt", nargs="?", help="Prompt to enhance (or read from stdin)")
    enhance.add_argument("--auto", action="store_true", help="Auto-apply enhancement")

    # neo prompt patterns
    patterns = prompt_parser.add_parser("patterns", help="Show effective patterns")
    patterns.add_argument("--search", help="Search for specific pattern")
    patterns.add_argument("--limit", type=int, default=10)

    # neo prompt suggest
    suggest = prompt_parser.add_parser("suggest", help="Suggest CLAUDE.md improvements")
    suggest.add_argument("--project", help="Project to analyze")
    suggest.add_argument("--apply", action="store_true", help="Apply suggestions interactively")

    # neo prompt history
    history = prompt_parser.add_parser("history", help="Show CLAUDE.md evolution history")
    history.add_argument("--path", help="Specific file to show history for")

    # neo prompt stats
    stats = prompt_parser.add_parser("stats", help="Show prompt knowledge base stats")

def handle_prompt_command(args, engine: NeoEngine):
    """Handle prompt subcommands."""
    from neo.prompt import PromptSystem

    system = PromptSystem()

    if args.prompt_command == "analyze":
        results = system.analyze(project=args.project, since=args.since)
        _print_analysis(results)

    elif args.prompt_command == "enhance":
        prompt = args.prompt or sys.stdin.read().strip()
        enhancement = system.enhance(prompt)
        if args.auto:
            print(enhancement.enhanced)
        else:
            _print_enhancement(enhancement)

    elif args.prompt_command == "patterns":
        patterns = system.get_patterns(search=args.search, limit=args.limit)
        _print_patterns(patterns)

    elif args.prompt_command == "suggest":
        suggestions = system.suggest_improvements(project=args.project)
        if args.apply:
            _apply_suggestions_interactive(suggestions)
        else:
            _print_suggestions(suggestions)

    elif args.prompt_command == "history":
        evolutions = system.get_evolution_history(path=args.path)
        _print_evolutions(evolutions)

    elif args.prompt_command == "stats":
        stats = system.get_stats()
        _print_stats(stats)
```

---

## Background Scan Integration

The system runs a lightweight scan on every `neo` invocation.

```python
# In cli.py, at the start of main()

def _background_prompt_scan():
    """Run lightweight prompt scan in background."""
    from neo.prompt import PromptSystem
    import threading

    def scan():
        try:
            system = PromptSystem()
            system.incremental_scan()  # Only process new data
        except Exception as e:
            logger.debug(f"Background prompt scan failed: {e}")

    thread = threading.Thread(target=scan, daemon=True)
    thread.start()

def main():
    # Start background scan (non-blocking)
    _background_prompt_scan()

    # Continue with normal neo execution
    ...
```

---

## Data Flow

```
┌─────────────────┐
│  neo invocation │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Background Scan │◄──── Runs in daemon thread
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Change Detector │──── Check watermarks
└────────┬────────┘
         │
         ▼ (only if changes detected)
┌─────────────────┐
│    Scanner      │──── Read new data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Analyzer      │──── Score prompts, detect patterns
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Knowledge Base     │──── Store patterns, scores, evolutions
└─────────────────────┘
```

---

## File Structure

```
src/neo/prompt/
├── __init__.py          # PromptSystem facade
├── scanner.py           # Data scanning
├── change_detector.py   # Change detection
├── analyzer.py          # Effectiveness analysis
├── evolution.py         # Evolution tracking
├── knowledge_base.py    # Storage
├── enhancer.py          # Enhancement generation
└── cli.py               # CLI commands
```

---

## Storage Schema

### `~/.neo/prompt_knowledge.json`

```json
{
  "version": "1.0",
  "entries": [
    {
      "id": "pattern_abc123",
      "entry_type": "pattern",
      "data": {
        "name": "Explicit File Reference",
        "description": "Reference specific files when asking for changes",
        "template": "In {file}, {action}",
        "examples": ["In src/main.py, add error handling to the parse function"],
        "effectiveness_score": 0.85,
        "use_cases": ["code_modification", "bugfix"],
        "anti_patterns": ["Change the code to fix the bug"]
      },
      "project": null,
      "created_at": "2025-01-15T10:30:00Z",
      "updated_at": "2025-01-15T10:30:00Z"
    },
    {
      "id": "score_def456",
      "entry_type": "score",
      "data": {
        "prompt_hash": "a1b2c3d4",
        "prompt_text": "fix the bug in login",
        "score": -0.3,
        "signals": ["immediate_clarification", "multiple_retries"],
        "iterations_to_complete": 4,
        "tool_calls": 12,
        "sample_count": 3,
        "confidence": 0.7
      },
      "project": "/Users/x/git/myapp",
      "created_at": "2025-01-14T08:00:00Z",
      "updated_at": "2025-01-15T09:00:00Z"
    },
    {
      "id": "evolution_ghi789",
      "entry_type": "evolution",
      "data": {
        "path": "/Users/x/git/myapp/CLAUDE.md",
        "change_type": "modified",
        "diff": "@@ -10,6 +10,8 @@\n+## Error Handling\n+Always use try/except for API calls",
        "inferred_reason": "User had to explain error handling requirements 5 times"
      },
      "project": "/Users/x/git/myapp",
      "created_at": "2025-01-15T11:00:00Z",
      "updated_at": "2025-01-15T11:00:00Z"
    }
  ]
}
```

### `~/.neo/prompt_watermarks.json`

```json
{
  "version": "1.0",
  "watermarks": {
    "history": {
      "timestamp": "2025-01-15T12:00:00Z",
      "position": 15234
    },
    "session:/Users/x/git/myapp": {
      "timestamp": "2025-01-15T12:00:00Z",
      "last_session": "abc-123-def"
    },
    "claude_md:/Users/x/.claude/CLAUDE.md": {
      "timestamp": "2025-01-15T12:00:00Z",
      "hash": "sha256:abcdef123456"
    }
  }
}
```

---

## Example Usage

```bash
# Analyze prompt effectiveness for a project
neo prompt analyze --project ~/git/neo

# Enhance a prompt interactively
neo prompt enhance "fix the bug"
# Output:
# Original: fix the bug
# Enhanced: Fix the authentication bug in src/auth.py where users cannot log in after password reset
# Improvements:
#   - Added specificity (which bug)
#   - Added file reference
#   - Added context (what's happening)
# Expected benefit: ~60% fewer clarification rounds

# Auto-enhance (just output the enhanced prompt)
echo "add tests" | neo prompt enhance --auto
# Output: Add unit tests for the UserService class in src/services/user.py covering the create, update, and delete methods

# Show effective prompt patterns
neo prompt patterns --search "refactor"

# Suggest CLAUDE.md improvements
neo prompt suggest --project ~/git/neo
# Output:
# Suggestions for /Users/x/git/neo/CLAUDE.md:
# 1. [HIGH] Add rule about test patterns
#    Reason: You've had to specify "use pytest" 8 times in the last week
#    Suggested addition:
#    ## Testing
#    - Use pytest for all tests
#    - Test files should be named test_*.py

# Show CLAUDE.md evolution history
neo prompt history --path ~/git/neo/CLAUDE.md

# Show stats
neo prompt stats
# Output:
# Prompt Knowledge Base Statistics
# ================================
# Total entries: 1,234
# Patterns: 45
# Effectiveness scores: 892
# Evolutions tracked: 23
# Pending suggestions: 7
# Projects tracked: 12
```

---

## Implementation Phases

### Phase 1: Foundation
- [ ] Create `src/neo/prompt/` module structure
- [ ] Implement `Scanner` for reading Claude Code data
- [ ] Implement `ChangeDetector` with watermarks
- [ ] Basic `PromptKnowledgeBase` storage

### Phase 2: Analysis
- [ ] Implement `EffectivenessAnalyzer` with signal detection
- [ ] Pattern extraction from effective prompts
- [ ] Implement `EvolutionTracker` for CLAUDE.md changes

### Phase 3: Enhancement
- [ ] Implement `PromptEnhancer` for suggestions
- [ ] CLAUDE.md improvement suggestions
- [ ] Pattern-based auto-enhancement

### Phase 4: CLI Integration
- [ ] Add `neo prompt` subcommands
- [ ] Background scan integration
- [ ] Output formatting

### Phase 5: Refinement
- [ ] Tune effectiveness signals based on real data
- [ ] Improve pattern extraction
- [ ] Add semantic embeddings for pattern search
