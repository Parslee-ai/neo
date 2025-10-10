"""
Auto-pattern extraction from self-corrections.
Neo learns prevention rules from its own mistakes.
"""

import json
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PreventionPattern:
    """A learned pattern for preventing specific bug classes."""
    bug_category: str           # "off-by-one", "logic-error", etc.
    signature_keywords: list[str]  # Words that indicate this pattern applies
    common_mistake: str         # What developers typically do wrong
    prevention_rule: str        # How to avoid the mistake
    confidence: float          # 0.0-1.0, based on how many times seen
    example_problems: list[str]  # Problems where this pattern was learned


class PatternLibrary:
    """Storage and retrieval of prevention patterns."""

    def __init__(self, storage_path: str = "~/.neo/prevention_patterns.json"):
        self.storage_path = Path(storage_path).expanduser()
        self.patterns: Dict[str, PreventionPattern] = {}
        self.load()

    def load(self):
        """Load patterns from disk."""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for key, pattern_dict in data.items():
                    self.patterns[key] = PreventionPattern(**pattern_dict)

    def save(self):
        """Save patterns to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            data = {k: asdict(v) for k, v in self.patterns.items()}
            json.dump(data, f, indent=2)

    def add_pattern(self, pattern: PreventionPattern):
        """Add or update a pattern."""
        key = f"{pattern.bug_category}_{len(pattern.signature_keywords)}"

        if key in self.patterns:
            # Update existing pattern
            existing = self.patterns[key]
            existing.confidence = (existing.confidence + pattern.confidence) / 2
            existing.example_problems.extend(pattern.example_problems)
        else:
            self.patterns[key] = pattern

        self.save()

    def get_applicable_patterns(
        self,
        problem_description: str,
        code: Optional[str] = None
    ) -> list[PreventionPattern]:
        """Find patterns that apply to this problem/code."""
        applicable = []
        text_to_search = (problem_description + " " + (code or "")).lower()

        for pattern in self.patterns.values():
            # Check if signature keywords match
            matches = sum(1 for kw in pattern.signature_keywords if kw in text_to_search)
            match_ratio = matches / len(pattern.signature_keywords) if pattern.signature_keywords else 0

            if match_ratio >= 0.5:  # At least half the keywords match
                applicable.append(pattern)

        # Sort by confidence
        return sorted(applicable, key=lambda p: p.confidence, reverse=True)


def extract_pattern_from_correction(
    problem_description: str,
    failed_code: str,
    corrected_code: str,
    bug_category: str,
    root_cause: str,
    adapter
) -> PreventionPattern:
    """
    Use LLM to analyze a failed→corrected pair and extract a reusable pattern.

    This is the 100x move: Neo learns from its own corrections.
    """

    extraction_prompt = f"""Analyze this coding mistake and extract a PREVENTION RULE:

PROBLEM TYPE:
{problem_description[:300]}

FAILED CODE:
```python
{failed_code[:500]}
```

CORRECTED CODE:
```python
{corrected_code[:500]}
```

BUG CATEGORY: {bug_category}
ROOT CAUSE: {root_cause}

Extract a reusable prevention pattern in this format:

1. Signature keywords (3-5 words that indicate this pattern applies):
2. Common mistake (one sentence, what developers typically do wrong):
3. Prevention rule (one sentence, how to avoid it):
4. Generalization (does this apply to similar problems?):

Make the prevention rule ACTIONABLE and SPECIFIC."""

    response = adapter.generate(
        [{"role": "user", "content": extraction_prompt}],
        temperature=0.0,
        max_tokens=500
    )

    # Parse response (simple parsing)
    lines = response.strip().split('\n')

    signature_keywords = []
    common_mistake = "Unknown"
    prevention_rule = "Unknown"

    for line in lines:
        if "signature" in line.lower() and ":" in line:
            # Extract keywords
            keywords_text = line.split(':', 1)[1].strip()
            signature_keywords = [
                kw.strip().strip(',[]"\'')
                for kw in keywords_text.split()
                if len(kw.strip()) > 2
            ][:5]  # Max 5 keywords
        elif "common mistake" in line.lower() and ":" in line:
            common_mistake = line.split(':', 1)[1].strip().strip('*').strip()
        elif "prevention rule" in line.lower() and ":" in line:
            prevention_rule = line.split(':', 1)[1].strip().strip('*').strip()

    return PreventionPattern(
        bug_category=bug_category,
        signature_keywords=signature_keywords,
        common_mistake=common_mistake,
        prevention_rule=prevention_rule,
        confidence=0.7,  # Initial confidence, will increase with more examples
        example_problems=[problem_description[:100]]
    )


def generate_prevention_warnings(
    problem_description: str,
    code: Optional[str],
    library: PatternLibrary
) -> str:
    """
    Generate warning text to add to LLM prompt based on applicable patterns.

    This is injected BEFORE generation to prevent bugs proactively.
    """
    patterns = library.get_applicable_patterns(problem_description, code)

    if not patterns:
        return ""

    warnings = ["\n⚠️  Common pitfalls for this problem type:"]

    for i, pattern in enumerate(patterns[:3], 1):  # Max 3 warnings
        warnings.append(
            f"{i}. {pattern.bug_category.upper()}: {pattern.prevention_rule}"
        )

    return "\n".join(warnings)


# Global library instance
_library = None

def get_library() -> PatternLibrary:
    """Get or create the global pattern library."""
    global _library
    if _library is None:
        _library = PatternLibrary()
    return _library