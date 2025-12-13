"""
Prompt Enhancement Engine for Neo.

Provides rule-based prompt enhancement to improve prompt clarity,
specificity, and effectiveness. Identifies common issues like
vague verbs, missing file references, and lack of acceptance criteria.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neo.prompt.knowledge_base import PromptKnowledgeBase
    from neo.cli import LMAdapter

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PromptEnhancement:
    """A suggested enhancement to a prompt."""

    original: str
    enhanced: str
    improvements: list[str] = field(default_factory=list)
    expected_benefit: str = ""
    confidence: float = 0.0


# ============================================================================
# Constants
# ============================================================================

# Vague verbs that typically need more specifics
VAGUE_VERBS = frozenset([
    "fix", "change", "update", "modify", "improve", "refactor",
    "adjust", "tweak", "handle", "do", "make", "work on",
    "look at", "check", "review", "clean up", "optimize",
])

# Patterns indicating file references
FILE_REFERENCE_PATTERNS = [
    r'\b[\w/.-]+\.(py|js|ts|go|rs|java|cpp|c|h|md|json|yaml|yml|toml)\b',
    r'\bin\s+[\w/.-]+\b',
    r'\bsrc/\S+',
    r'\blib/\S+',
    r'\btest[s]?/\S+',
]

# Patterns indicating specific requirements or constraints
CONSTRAINT_PATTERNS = [
    r'\bmust\b',
    r'\bshould\b',
    r'\brequire[sd]?\b',
    r'\bensure\b',
    r'\bvalidate\b',
    r'\breturn[s]?\s+\w+',
    r'\baccept[s]?\s+\w+',
    r'\bwhen\s+\w+',
    r'\bif\s+\w+',
    r'\bbefore\b',
    r'\bafter\b',
    r'\bwithout\b',
    r'\bonly\b',
]

# Patterns for acceptance criteria
ACCEPTANCE_CRITERIA_PATTERNS = [
    r'\btests?\s+(?:should\s+)?pass',
    r'\bshould\s+(return|output|produce)',
    r'\bexpect[ed]?\b',
    r'\bverif(y|ied)\b',
    r'\bconfirm\b',
    r'\bresult[s]?\s+in\b',
    r'\bpass(?:es|ing)?\s+(?:all\s+)?tests?\b',
]


# ============================================================================
# PromptEnhancer Class
# ============================================================================

class PromptEnhancer:
    """Generates prompt enhancements using learned patterns and rules."""

    def __init__(
        self,
        knowledge_base: Optional["PromptKnowledgeBase"] = None,
        lm_adapter: Optional["LMAdapter"] = None,
    ):
        """
        Initialize the prompt enhancer.

        Args:
            knowledge_base: Optional PromptKnowledgeBase for pattern lookup
            lm_adapter: Optional LM adapter for LLM-based enhancement (not used in rule-based mode)
        """
        self.kb = knowledge_base
        self.lm = lm_adapter

    def enhance_prompt(
        self,
        prompt: str,
        context: Optional[dict] = None,
    ) -> PromptEnhancement:
        """
        Enhance a prompt using learned patterns and rules.

        Args:
            prompt: The original prompt text
            context: Optional context dict with additional info (project, files, etc.)

        Returns:
            PromptEnhancement with original, enhanced prompt, and improvement details
        """
        context = context or {}

        # Find similar effective patterns if knowledge base is available
        similar_patterns: list = []
        if self.kb is not None:
            try:
                similar_patterns = self.kb.search_patterns(prompt, k=3)
            except Exception as e:
                logger.debug(f"Pattern search failed: {e}")

        # Identify issues with the prompt
        issues = self._identify_issues(prompt, similar_patterns)

        # Generate enhanced prompt
        enhanced = self._generate_enhancement(prompt, similar_patterns, issues)

        # Calculate expected benefit
        expected_benefit = self._estimate_benefit(issues)

        # Calculate confidence based on issues found and patterns matched
        confidence = self._calculate_confidence(similar_patterns, issues)

        return PromptEnhancement(
            original=prompt,
            enhanced=enhanced,
            improvements=issues,
            expected_benefit=expected_benefit,
            confidence=confidence,
        )

    def _identify_issues(
        self,
        prompt: str,
        similar_patterns: list,
    ) -> list[str]:
        """
        Identify what could be improved in the prompt.

        Args:
            prompt: The prompt text to analyze
            similar_patterns: List of similar effective patterns from knowledge base

        Returns:
            List of issue identifiers
        """
        issues = []

        # Check for vagueness
        if self._is_vague(prompt):
            issues.append("lacks_specificity")

        # Check for missing file references
        if not self._has_file_references(prompt):
            issues.append("missing_file_references")

        # Check for missing constraints
        if not self._has_constraints(prompt):
            issues.append("missing_constraints")

        # Check for missing acceptance criteria
        if not self._has_acceptance_criteria(prompt):
            issues.append("missing_acceptance_criteria")

        # Check prompt length (too short often means too vague)
        word_count = len(prompt.split())
        if word_count < 5:
            issues.append("too_brief")

        return issues

    def _is_vague(self, prompt: str) -> bool:
        """
        Check if prompt uses vague verbs without specifics.

        Args:
            prompt: The prompt text to analyze

        Returns:
            True if prompt contains vague verbs without sufficient context
        """
        prompt_lower = prompt.lower()
        words = prompt_lower.split()

        # Check for vague verbs
        has_vague_verb = False
        for verb in VAGUE_VERBS:
            # Check if verb appears as standalone word or phrase
            if verb in words or verb in prompt_lower:
                has_vague_verb = True
                break

        if not has_vague_verb:
            return False

        # Vague verb is OK if there are specific details following
        # Check for indicators of specificity
        has_specifics = (
            self._has_file_references(prompt) or
            self._has_constraints(prompt) or
            len(words) > 15  # Longer prompts tend to have more context
        )

        return not has_specifics

    def _has_file_references(self, prompt: str) -> bool:
        """
        Check if prompt contains file references.

        Args:
            prompt: The prompt text to analyze

        Returns:
            True if prompt contains file path patterns
        """
        for pattern in FILE_REFERENCE_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        return False

    def _has_constraints(self, prompt: str) -> bool:
        """
        Check if prompt specifies constraints or requirements.

        Args:
            prompt: The prompt text to analyze

        Returns:
            True if prompt contains constraint indicators
        """
        for pattern in CONSTRAINT_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        return False

    def _has_acceptance_criteria(self, prompt: str) -> bool:
        """
        Check if prompt specifies acceptance criteria.

        Args:
            prompt: The prompt text to analyze

        Returns:
            True if prompt contains acceptance criteria indicators
        """
        for pattern in ACCEPTANCE_CRITERIA_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        return False

    def _generate_enhancement(
        self,
        prompt: str,
        patterns: list,
        issues: list[str],
    ) -> str:
        """
        Generate an enhanced version of the prompt.

        Args:
            prompt: Original prompt text
            patterns: Similar effective patterns from knowledge base
            issues: List of identified issues

        Returns:
            Enhanced prompt string
        """
        # Start with the original prompt
        enhanced = prompt.strip()

        # Build suggestions based on issues
        suggestions = []

        if "missing_file_references" in issues:
            suggestions.append("[Specify target file(s)]")

        if "lacks_specificity" in issues or "too_brief" in issues:
            suggestions.append("[Add details: what specifically needs to change?]")

        if "missing_constraints" in issues:
            suggestions.append("[Add constraints: requirements, boundaries, or conditions]")

        if "missing_acceptance_criteria" in issues:
            suggestions.append("[Add acceptance criteria: how to verify success?]")

        # If we have suggestions, append them as guidance
        if suggestions:
            enhanced = f"{enhanced}\n\nConsider adding:\n" + "\n".join(f"- {s}" for s in suggestions)

        return enhanced

    def _estimate_benefit(self, issues: list[str]) -> str:
        """
        Estimate the benefit of addressing the identified issues.

        Args:
            issues: List of identified issues

        Returns:
            Human-readable benefit description
        """
        if not issues:
            return "Prompt appears well-structured"

        # Map issues to benefits
        benefit_map = {
            "lacks_specificity": "clearer intent",
            "missing_file_references": "targeted changes",
            "missing_constraints": "bounded scope",
            "missing_acceptance_criteria": "verifiable success",
            "too_brief": "more context for accurate response",
        }

        benefits = [benefit_map.get(issue, issue) for issue in issues]

        if len(benefits) == 1:
            return f"Enhancement provides {benefits[0]}"

        return f"Enhancement provides {', '.join(benefits[:-1])} and {benefits[-1]}"

    def _calculate_confidence(
        self,
        similar_patterns: list,
        issues: list[str],
    ) -> float:
        """
        Calculate confidence in the enhancement.

        Args:
            similar_patterns: Matched patterns from knowledge base
            issues: Identified issues

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.5

        # Increase confidence based on number of issues found (more issues = more room for improvement)
        if issues:
            confidence += min(0.3, len(issues) * 0.1)

        # Increase confidence if we have similar patterns to guide enhancement
        if similar_patterns:
            confidence += min(0.2, len(similar_patterns) * 0.07)

        return min(1.0, confidence)

    def suggest_claude_md_updates(self, project: str) -> list[dict]:
        """
        Suggest updates to a project's CLAUDE.md based on usage patterns.

        Args:
            project: Project path or identifier

        Returns:
            List of suggestion dictionaries with type, target, suggestion, reason, confidence
        """
        suggestions = []

        # Without knowledge base or LM, return empty suggestions
        if self.kb is None:
            logger.debug("No knowledge base available for CLAUDE.md suggestions")
            return suggestions

        # Get pending suggestions for this project from knowledge base
        try:
            pending = self.kb.get_pending_suggestions(project=project)
            suggestions.extend(pending)
        except Exception as e:
            logger.debug(f"Failed to get pending suggestions: {e}")

        return suggestions

    def auto_enhance(self, prompt: str) -> str:
        """
        Automatically enhance a prompt without explanation.

        Only returns enhanced prompt if confidence is above threshold.

        Args:
            prompt: Original prompt text

        Returns:
            Enhanced prompt if confidence > 0.7, otherwise original prompt
        """
        enhancement = self.enhance_prompt(prompt)

        if enhancement.confidence > 0.7:
            return enhancement.enhanced

        return prompt
