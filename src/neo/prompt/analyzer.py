"""
Effectiveness Analyzer for prompt evaluation.

Analyzes Claude Code sessions to determine prompt effectiveness using
semantic signals. Identifies patterns that lead to successful task
completion vs confusion and iteration.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neo.prompt.scanner import ScannedSession

logger = logging.getLogger(__name__)


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
    confidence: float  # How confident we are in this score (0.0 to 1.0)


@dataclass
class PromptPattern:
    """A reusable prompt pattern extracted from effective prompts."""

    pattern_id: str
    name: str
    description: str
    template: str  # Template with placeholders
    examples: list[str] = field(default_factory=list)  # Concrete examples
    effectiveness_score: float = 0.0
    use_cases: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)  # What NOT to do


class EffectivenessAnalyzer:
    """Analyzes prompt effectiveness using semantic signals."""

    # Patterns that indicate Claude confusion
    CONFUSION_PATTERNS: list[str] = [
        r"I'm not sure",
        r"I am not sure",
        r"Could you clarify",
        r"Can you clarify",
        r"I need more information",
        r"Can you provide more context",
        r"I don't understand",
        r"I do not understand",
        r"What do you mean by",
        r"Please specify",
        r"Could you be more specific",
        r"I'm uncertain",
        r"It's unclear",
        r"I cannot determine",
    ]

    # Patterns that indicate success
    SUCCESS_PATTERNS: list[str] = [
        r"Done\.",
        r"I've (completed|finished|implemented|fixed|created|added|updated)",
        r"I have (completed|finished|implemented|fixed|created|added|updated)",
        r"The (task|change|fix|feature|update|implementation) is complete",
        r"Successfully (completed|implemented|fixed|created|added|updated)",
        r"Changes have been (made|applied|committed)",
        r"All (tests|checks) pass",
        r"Here's the (completed|updated|fixed)",
    ]

    # Patterns that indicate errors
    ERROR_PATTERNS: list[str] = [
        r"Error:",
        r"Exception:",
        r"Traceback",
        r"failed with",
        r"cannot (find|locate|access)",
        r"does not exist",
        r"permission denied",
        r"syntax error",
        r"undefined",
        r"not found",
    ]

    # Patterns indicating git commit
    COMMIT_PATTERNS: list[str] = [
        r"git commit",
        r"committed",
        r"Created commit",
        r"\[.+\]\s+\w+",  # Git commit output format
    ]

    # Patterns indicating test success
    TEST_SUCCESS_PATTERNS: list[str] = [
        r"tests? passed",
        r"all tests",
        r"pytest.*passed",
        r"OK \(\d+ tests?\)",
        r"\d+ passed",
        r"PASSED",
    ]

    def __init__(self):
        """Initialize the analyzer with compiled regex patterns."""
        self._compiled_confusion = [
            re.compile(p, re.IGNORECASE) for p in self.CONFUSION_PATTERNS
        ]
        self._compiled_success = [
            re.compile(p, re.IGNORECASE) for p in self.SUCCESS_PATTERNS
        ]
        self._compiled_error = [
            re.compile(p, re.IGNORECASE) for p in self.ERROR_PATTERNS
        ]
        self._compiled_commit = [
            re.compile(p, re.IGNORECASE) for p in self.COMMIT_PATTERNS
        ]
        self._compiled_test_success = [
            re.compile(p, re.IGNORECASE) for p in self.TEST_SUCCESS_PATTERNS
        ]

    def analyze_session(
        self, session: "ScannedSession"
    ) -> list[PromptEffectivenessScore]:
        """
        Analyze a session to score each prompt's effectiveness.

        Args:
            session: A ScannedSession containing message history

        Returns:
            List of effectiveness scores for each user prompt
        """
        scores = []
        messages = session.messages

        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue

            prompt_text = self._extract_prompt_text(msg)
            if not prompt_text:
                continue

            signals = self._detect_signals(messages, i)
            score = self._calculate_score(signals)

            scores.append(
                PromptEffectivenessScore(
                    prompt_hash=self._hash_prompt(prompt_text),
                    prompt_text=prompt_text,
                    score=score,
                    signals=signals,
                    iterations_to_complete=self._count_iterations(messages, i),
                    tool_calls=self._count_tool_calls(messages, i),
                    sample_count=1,
                    confidence=0.5,  # Single sample, low confidence
                )
            )

        return scores

    def _detect_signals(
        self, messages: list[dict], prompt_index: int
    ) -> list[EffectivenessSignal]:
        """
        Detect effectiveness signals from conversation context.

        Args:
            messages: Full message list from session
            prompt_index: Index of the user prompt being analyzed

        Returns:
            List of detected effectiveness signals
        """
        signals = []
        prompt_msg = messages[prompt_index]
        prompt_text = self._extract_prompt_text(prompt_msg)

        # Get the assistant's response
        response = self._get_next_assistant_message(messages, prompt_index)
        response_text = self._extract_response_text(response) if response else ""

        # Get next user message (if any)
        next_user_msg = self._get_next_user_message(messages, prompt_index)
        next_user_text = self._extract_prompt_text(next_user_msg) if next_user_msg else ""

        # Check for confusion patterns in response
        if response_text and self._matches_patterns(response_text, self._compiled_confusion):
            signals.append(EffectivenessSignal.CLAUDE_CONFUSED)

        # Check for error patterns in response
        if response_text and self._matches_patterns(response_text, self._compiled_error):
            signals.append(EffectivenessSignal.ERROR_IN_RESPONSE)

        # Check for success patterns
        if response_text and self._matches_patterns(response_text, self._compiled_success):
            signals.append(EffectivenessSignal.TASK_COMPLETED)

        # Check if user had to immediately clarify
        if next_user_text and self._is_clarification(next_user_text, prompt_text):
            signals.append(EffectivenessSignal.IMMEDIATE_CLARIFICATION)

        # Check for commits
        if self._has_commit(messages, prompt_index):
            signals.append(EffectivenessSignal.COMMIT_MADE)

        # Check for test success
        if self._has_passing_tests(messages, prompt_index):
            signals.append(EffectivenessSignal.TESTS_PASSED)

        # Check for single iteration (task completed in one exchange)
        iterations = self._count_iterations(messages, prompt_index)
        if iterations == 1 and EffectivenessSignal.TASK_COMPLETED in signals:
            signals.append(EffectivenessSignal.SINGLE_ITERATION)

        # Check for multiple retries
        if iterations > 2:
            signals.append(EffectivenessSignal.MULTIPLE_RETRIES)

        # Check for topic change (potential implicit success or abandonment)
        if next_user_text and self._is_topic_change(prompt_text, next_user_text):
            if EffectivenessSignal.TASK_COMPLETED in signals:
                signals.append(EffectivenessSignal.TOPIC_CHANGED)
            elif not any(s in signals for s in [
                EffectivenessSignal.TASK_COMPLETED,
                EffectivenessSignal.COMMIT_MADE,
                EffectivenessSignal.TESTS_PASSED,
            ]):
                signals.append(EffectivenessSignal.ABANDONED_TASK)

        return signals

    def _calculate_score(self, signals: list[EffectivenessSignal]) -> float:
        """
        Calculate effectiveness score from signals.

        Args:
            signals: List of detected effectiveness signals

        Returns:
            Score between -1.0 (very ineffective) and 1.0 (very effective)
        """
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
        if EffectivenessSignal.TOPIC_CHANGED in signals:
            score += 0.1  # Small bonus for natural progression

        # Negative signals
        if EffectivenessSignal.CLAUDE_CONFUSED in signals:
            score -= 0.3
        if EffectivenessSignal.IMMEDIATE_CLARIFICATION in signals:
            score -= 0.2
        if EffectivenessSignal.MULTIPLE_RETRIES in signals:
            score -= 0.3
        if EffectivenessSignal.ERROR_IN_RESPONSE in signals:
            score -= 0.2
        if EffectivenessSignal.ABANDONED_TASK in signals:
            score -= 0.4

        return max(-1.0, min(1.0, score))

    def extract_patterns(
        self, effective_prompts: list[PromptEffectivenessScore]
    ) -> list[PromptPattern]:
        """
        Extract reusable patterns from highly effective prompts.

        Args:
            effective_prompts: List of prompts with high effectiveness scores

        Returns:
            List of extracted prompt patterns
        """
        patterns = []

        # Filter to only highly effective prompts
        high_score_prompts = [p for p in effective_prompts if p.score >= 0.6]

        if not high_score_prompts:
            return patterns

        # Group by structural similarity
        clusters = self._cluster_prompts(high_score_prompts)

        for cluster_id, cluster_prompts in clusters.items():
            if len(cluster_prompts) < 2:
                continue  # Need at least 2 examples for a pattern

            # Extract common template
            template = self._extract_template(cluster_prompts)
            if not template:
                continue

            # Determine use cases from prompts
            use_cases = self._infer_use_cases(cluster_prompts)

            # Calculate average effectiveness
            avg_score = sum(p.score for p in cluster_prompts) / len(cluster_prompts)

            pattern = PromptPattern(
                pattern_id=f"pattern_{cluster_id}",
                name=self._generate_pattern_name(template),
                description=f"Pattern extracted from {len(cluster_prompts)} effective prompts",
                template=template,
                examples=[p.prompt_text for p in cluster_prompts[:3]],
                effectiveness_score=avg_score,
                use_cases=use_cases,
                anti_patterns=self._find_anti_patterns(cluster_prompts, effective_prompts),
            )
            patterns.append(pattern)

        return patterns

    # Helper methods

    def _extract_prompt_text(self, msg: dict) -> str:
        """Extract text content from a message."""
        if not msg:
            return ""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Handle content blocks format
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            return " ".join(text_parts)
        return ""

    def _extract_response_text(self, msg: dict) -> str:
        """Extract text content from an assistant response."""
        return self._extract_prompt_text(msg)

    def _hash_prompt(self, prompt_text: str) -> str:
        """Generate a hash for the prompt text."""
        normalized = prompt_text.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _get_next_assistant_message(
        self, messages: list[dict], prompt_index: int
    ) -> Optional[dict]:
        """Get the next assistant message after a user prompt."""
        for i in range(prompt_index + 1, len(messages)):
            if messages[i].get("role") == "assistant":
                return messages[i]
            if messages[i].get("role") == "user":
                break  # No assistant response before next user message
        return None

    def _get_next_user_message(
        self, messages: list[dict], prompt_index: int
    ) -> Optional[dict]:
        """Get the next user message after a user prompt (skipping assistant messages)."""
        found_assistant = False
        for i in range(prompt_index + 1, len(messages)):
            if messages[i].get("role") == "assistant":
                found_assistant = True
            elif messages[i].get("role") == "user" and found_assistant:
                return messages[i]
        return None

    def _matches_patterns(
        self, text: str, compiled_patterns: list[re.Pattern]
    ) -> bool:
        """Check if text matches any of the compiled patterns."""
        for pattern in compiled_patterns:
            if pattern.search(text):
                return True
        return False

    def _is_clarification(self, next_msg: str, original_prompt: str) -> bool:
        """Check if the next message is a clarification of the original prompt."""
        if not next_msg or not original_prompt:
            return False

        next_lower = next_msg.lower().strip()

        # Direct clarification indicators
        clarification_indicators = [
            "i mean",
            "what i meant",
            "to clarify",
            "let me clarify",
            "sorry, i meant",
            "no, i want",
            "not that",
            "actually,",  # "actually" at start of sentence
            "specifically",
            "in other words",
            "wait,",
            "no,",
            "nope,",
        ]

        for indicator in clarification_indicators:
            if indicator in next_lower:
                return True

        # Short messages that are likely positive feedback, NOT clarifications
        positive_feedback = {
            "thanks", "thank you", "thanks!", "thank you!",
            "perfect", "perfect!", "great", "great!",
            "good", "good!", "ok", "okay", "ok!",
            "awesome", "awesome!", "nice", "nice!",
            "yes", "yes!", "yep", "yep!", "done", "done!",
            "lgtm", "ship it", "looks good", "sounds good",
        }

        # If it's a short message but matches positive feedback, NOT a clarification
        words = next_msg.split()
        if len(words) <= 5:
            # Check if it's positive feedback
            if next_lower.rstrip("!.,") in positive_feedback:
                return False
            # Check if it starts with positive feedback
            if words and words[0].lower().rstrip("!.,") in {"thanks", "thank", "perfect", "great", "good", "ok", "okay", "awesome", "nice", "yes", "yep", "done", "lgtm"}:
                return False
            # Short correction-like messages (starts with "no" or similar)
            if words and words[0].lower().rstrip(",") in {"no", "nope", "wait", "actually"}:
                return True
            # Otherwise, short messages might be clarifications if they relate to the original prompt
            # Check word overlap with original prompt
            orig_words = set(original_prompt.lower().split())
            next_words = set(next_lower.split())
            overlap = len(orig_words & next_words)
            if overlap >= 1 and len(words) <= 3:
                # Very short message with overlap suggests correction/clarification
                return True

        return False

    def _is_topic_change(self, original_prompt: str, next_prompt: str) -> bool:
        """Check if the next prompt represents a significant topic change."""
        if not original_prompt or not next_prompt:
            return False

        # Simple heuristic: check word overlap
        orig_words = set(original_prompt.lower().split())
        next_words = set(next_prompt.lower().split())

        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "to", "in", "for", "on", "it", "and", "or"}
        orig_words -= stop_words
        next_words -= stop_words

        if not orig_words or not next_words:
            return False

        overlap = len(orig_words & next_words)
        max_len = max(len(orig_words), len(next_words))

        # Low overlap suggests topic change
        return overlap / max_len < 0.2

    def _has_commit(self, messages: list[dict], prompt_index: int) -> bool:
        """Check if a git commit was made in response to this prompt."""
        # Look at messages after the prompt
        for i in range(prompt_index + 1, min(prompt_index + 10, len(messages))):
            msg = messages[i]
            if msg.get("role") == "user":
                break  # Stop at next user message

            text = self._extract_response_text(msg)
            if self._matches_patterns(text, self._compiled_commit):
                return True

            # Also check tool calls for git commit
            tool_calls = msg.get("tool_calls", [])
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    func_name = tool_call.get("function", {}).get("name", "")
                    args = tool_call.get("function", {}).get("arguments", "")
                    if "git" in func_name.lower() and "commit" in args.lower():
                        return True

        return False

    def _has_passing_tests(self, messages: list[dict], prompt_index: int) -> bool:
        """Check if tests passed in response to this prompt."""
        for i in range(prompt_index + 1, min(prompt_index + 10, len(messages))):
            msg = messages[i]
            if msg.get("role") == "user":
                break

            text = self._extract_response_text(msg)
            if self._matches_patterns(text, self._compiled_test_success):
                return True

        return False

    def _count_iterations(self, messages: list[dict], prompt_index: int) -> int:
        """
        Count how many iterations it took to complete the task.

        An iteration is defined as a user message followed by an assistant response.
        """
        iterations = 0
        i = prompt_index

        while i < len(messages):
            if messages[i].get("role") == "user":
                # Check if followed by assistant
                if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                    iterations += 1

                    # Check if task is complete at this point
                    response_text = self._extract_response_text(messages[i + 1])
                    if self._matches_patterns(response_text, self._compiled_success):
                        break

                    i += 2
                else:
                    break
            else:
                i += 1

        return max(1, iterations)

    def _count_tool_calls(self, messages: list[dict], prompt_index: int) -> int:
        """Count tool calls made in response to this prompt."""
        tool_count = 0

        for i in range(prompt_index + 1, min(prompt_index + 20, len(messages))):
            msg = messages[i]
            if msg.get("role") == "user":
                break

            tool_calls = msg.get("tool_calls", [])
            if isinstance(tool_calls, list):
                tool_count += len(tool_calls)

            # Also check for tool_use in content
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_count += 1

        return tool_count

    def _cluster_prompts(
        self, prompts: list[PromptEffectivenessScore]
    ) -> dict[str, list[PromptEffectivenessScore]]:
        """
        Cluster prompts by structural similarity.

        Uses a simple approach based on prompt structure patterns.
        """
        clusters: dict[str, list[PromptEffectivenessScore]] = {}

        for prompt in prompts:
            # Extract structural pattern
            pattern_key = self._get_structural_pattern(prompt.prompt_text)

            if pattern_key not in clusters:
                clusters[pattern_key] = []
            clusters[pattern_key].append(prompt)

        return clusters

    def _get_structural_pattern(self, text: str) -> str:
        """Extract a structural pattern key from prompt text."""
        # Simple pattern extraction based on:
        # - Starting verb/word
        # - Presence of file references
        # - Question vs command
        # - Length category

        words = text.lower().split()
        if not words:
            return "empty"

        pattern_parts = []

        # Starting word category
        action_verbs = ["add", "create", "fix", "update", "modify", "implement", "remove", "delete"]
        question_words = ["how", "what", "why", "where", "when", "can", "could", "would"]

        first_word = words[0].strip(".,!?")
        if first_word in action_verbs:
            pattern_parts.append(f"action:{first_word}")
        elif first_word in question_words:
            pattern_parts.append("question")
        else:
            pattern_parts.append("other")

        # File reference detection
        file_patterns = [r"\.[a-z]{2,4}\b", r"/[a-z_]+", r"src/", r"test"]
        has_file_ref = any(re.search(p, text.lower()) for p in file_patterns)
        if has_file_ref:
            pattern_parts.append("with_file")

        # Length category
        if len(words) < 10:
            pattern_parts.append("short")
        elif len(words) < 30:
            pattern_parts.append("medium")
        else:
            pattern_parts.append("long")

        return "_".join(pattern_parts)

    def _extract_template(
        self, prompts: list[PromptEffectivenessScore]
    ) -> Optional[str]:
        """Extract a common template from a cluster of prompts."""
        if not prompts:
            return None

        # Find common starting patterns
        texts = [p.prompt_text for p in prompts]

        # Simple approach: use the shortest prompt as the base template
        # and identify variable parts
        base_prompt = min(texts, key=len)

        # Replace specific file names, identifiers with placeholders
        template = re.sub(r"\b[a-z_]+\.(py|js|ts|go|rs|java)\b", "{file}", base_prompt)
        template = re.sub(r"\b[A-Z][a-zA-Z]+(?:Service|Controller|Handler|Manager)\b", "{component}", template)
        template = re.sub(r"def [a-z_]+", "def {function}", template)
        template = re.sub(r"class [A-Z][a-zA-Z]+", "class {class}", template)

        return template

    def _generate_pattern_name(self, template: str) -> str:
        """Generate a human-readable name for a pattern."""
        words = template.split()[:4]
        name = " ".join(words).title()
        if len(name) > 40:
            name = name[:37] + "..."
        return name

    def _infer_use_cases(
        self, prompts: list[PromptEffectivenessScore]
    ) -> list[str]:
        """Infer use cases from a cluster of prompts."""
        use_cases = set()

        keywords_to_use_cases = {
            "fix": "bugfix",
            "bug": "bugfix",
            "error": "bugfix",
            "add": "feature",
            "create": "feature",
            "implement": "feature",
            "new": "feature",
            "update": "enhancement",
            "modify": "enhancement",
            "refactor": "refactoring",
            "test": "testing",
            "tests": "testing",
            "doc": "documentation",
            "readme": "documentation",
        }

        for prompt in prompts:
            text_lower = prompt.prompt_text.lower()
            for keyword, use_case in keywords_to_use_cases.items():
                if keyword in text_lower:
                    use_cases.add(use_case)

        return list(use_cases)

    def _find_anti_patterns(
        self,
        cluster_prompts: list[PromptEffectivenessScore],
        all_prompts: list[PromptEffectivenessScore],
    ) -> list[str]:
        """Find anti-patterns by looking at similar but ineffective prompts."""
        anti_patterns = []

        # Find prompts with similar structure but low scores
        cluster_pattern = self._get_structural_pattern(cluster_prompts[0].prompt_text)

        for prompt in all_prompts:
            if prompt.score < 0.0:  # Ineffective
                prompt_pattern = self._get_structural_pattern(prompt.prompt_text)
                if prompt_pattern == cluster_pattern:
                    # Identify what makes this one ineffective
                    anti_patterns.append(
                        f"Avoid: {prompt.prompt_text[:100]}..."
                        if len(prompt.prompt_text) > 100
                        else f"Avoid: {prompt.prompt_text}"
                    )

        return anti_patterns[:3]  # Limit to 3 anti-patterns
