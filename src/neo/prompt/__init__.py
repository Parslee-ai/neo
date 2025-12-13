"""
Prompt Enhancement System for Neo.

This module provides tools for analyzing prompt effectiveness,
tracking CLAUDE.md evolution, and suggesting improvements.

The PromptSystem class is the main facade that ties together all components:
- Scanner: Reads Claude Code data from local files
- ChangeDetector: Tracks what has changed since last scan
- EffectivenessAnalyzer: Analyzes conversations to determine prompt effectiveness
- EvolutionTracker: Tracks changes to CLAUDE.md files over time
- PromptKnowledgeBase: Storage for prompt patterns and effectiveness data
- PromptEnhancer: Generates prompt improvements
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Re-export components for backwards compatibility
from neo.prompt.scanner import (
    Scanner,
    ClaudeCodeSources,
    ScannedPrompt,
    ScannedSession,
    ScannedClaudeMd,
)
from neo.prompt.change_detector import ChangeDetector, Watermark
from neo.prompt.analyzer import (
    EffectivenessSignal,
    PromptEffectivenessScore,
    PromptPattern,
    EffectivenessAnalyzer,
)
from neo.prompt.knowledge_base import PromptKnowledgeBase, PromptEntry
from neo.prompt.evolution import (
    ClaudeMdEvolution,
    CommandEvolution,
    EvolutionTracker,
)
from neo.prompt.enhancer import PromptEnhancer, PromptEnhancement


class PromptSystem:
    """
    Main facade for the Prompt Enhancement System.

    Coordinates all components to provide:
    - Incremental scanning of Claude Code data
    - Prompt effectiveness analysis
    - Pattern-based prompt enhancement
    - CLAUDE.md evolution tracking
    - Improvement suggestions
    """

    def __init__(self):
        """Initialize all prompt system components."""
        self._scanner: Optional[Scanner] = None
        self._change_detector: Optional[ChangeDetector] = None
        self._analyzer: Optional[EffectivenessAnalyzer] = None
        self._evolution_tracker: Optional[EvolutionTracker] = None
        self._knowledge_base: Optional[PromptKnowledgeBase] = None
        self._enhancer: Optional[PromptEnhancer] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize components on first use."""
        if self._initialized:
            return

        try:
            self._scanner = Scanner()
        except Exception as e:
            logger.warning(f"Scanner initialization failed: {e}")

        try:
            self._change_detector = ChangeDetector()
        except Exception as e:
            logger.warning(f"ChangeDetector initialization failed: {e}")

        try:
            self._analyzer = EffectivenessAnalyzer()
        except Exception as e:
            logger.warning(f"EffectivenessAnalyzer initialization failed: {e}")

        try:
            self._evolution_tracker = EvolutionTracker()
        except Exception as e:
            logger.warning(f"EvolutionTracker initialization failed: {e}")

        try:
            self._knowledge_base = PromptKnowledgeBase()
        except Exception as e:
            logger.warning(f"PromptKnowledgeBase initialization failed: {e}")

        try:
            # PromptEnhancer can work with or without LM adapter
            self._enhancer = PromptEnhancer(
                knowledge_base=self._knowledge_base,
                lm_adapter=None,  # Use rule-based enhancement by default
            )
        except Exception as e:
            logger.warning(f"PromptEnhancer initialization failed: {e}")

        self._initialized = True

    def incremental_scan(self) -> dict:
        """
        Scan changes since last run, analyze, and store results.

        This is designed to run on every `neo` invocation as a background task.
        It only processes new/modified data to minimize overhead.

        Returns:
            dict with scan statistics:
            - new_prompts: Number of new prompts processed
            - new_sessions: Number of new sessions analyzed
            - modified_claude_mds: Number of CLAUDE.md changes detected
            - patterns_extracted: Number of new patterns added
            - errors: List of any errors encountered
        """
        self._ensure_initialized()

        stats = {
            "new_prompts": 0,
            "new_sessions": 0,
            "modified_claude_mds": 0,
            "patterns_extracted": 0,
            "errors": [],
        }

        if not self._scanner or not self._change_detector:
            stats["errors"].append("Scanner or ChangeDetector not available")
            return stats

        try:
            # Get changes since last scan
            changes = self._change_detector.get_changes_since_last_scan(self._scanner)

            # Process new prompts
            new_prompts = changes.get("new_prompts", [])
            stats["new_prompts"] = len(new_prompts)

            # Process new sessions
            new_sessions = changes.get("new_sessions", [])
            if new_sessions and self._analyzer:
                for session in new_sessions:
                    try:
                        scores = self._analyzer.analyze_session(session)
                        if self._knowledge_base:
                            for score in scores:
                                self._knowledge_base.update_effectiveness_score(score)
                        stats["new_sessions"] += 1
                    except Exception as e:
                        stats["errors"].append(f"Session analysis failed: {e}")

            # Process CLAUDE.md changes
            modified_mds = changes.get("modified_claude_mds", [])
            if modified_mds and self._evolution_tracker and self._knowledge_base:
                for old, new in modified_mds:
                    try:
                        evolution = self._evolution_tracker.record_claude_md_change(old, new)
                        self._knowledge_base.add_evolution(evolution)
                        stats["modified_claude_mds"] += 1
                    except Exception as e:
                        stats["errors"].append(f"Evolution tracking failed: {e}")

            # Extract patterns from highly effective prompts
            if self._analyzer and self._knowledge_base:
                try:
                    effective_entries = [
                        e for e in self._knowledge_base.entries
                        if e.entry_type == "score"
                        and e.data.get("score", 0) > 0.7
                    ]
                    # Only extract patterns if we have enough samples
                    if len(effective_entries) >= 5:
                        # Convert entries to PromptEffectivenessScore objects
                        effective_scores = []
                        for entry in effective_entries:
                            score = PromptEffectivenessScore(
                                prompt_hash=entry.data.get("prompt_hash", ""),
                                prompt_text=entry.data.get("prompt_text", ""),
                                score=entry.data.get("score", 0.0),
                                signals=[
                                    EffectivenessSignal(s)
                                    for s in entry.data.get("signals", [])
                                    if s in [e.value for e in EffectivenessSignal]
                                ],
                                iterations_to_complete=entry.data.get("iterations_to_complete", 0),
                                tool_calls=entry.data.get("tool_calls", 0),
                                sample_count=entry.data.get("sample_count", 1),
                                confidence=entry.data.get("confidence", 0.5),
                            )
                            effective_scores.append(score)

                        patterns = self._analyzer.extract_patterns(effective_scores)
                        for pattern in patterns:
                            self._knowledge_base.add_pattern(pattern)
                            stats["patterns_extracted"] += 1
                except Exception as e:
                    stats["errors"].append(f"Pattern extraction failed: {e}")

            # Note: watermarks are updated inline in get_changes_since_last_scan
            # and individual getter methods

        except Exception as e:
            logger.error(f"Incremental scan failed: {e}")
            stats["errors"].append(str(e))

        return stats

    def analyze(self, project: Optional[str] = None, since: Optional[str] = None) -> dict:
        """
        Analyze prompt effectiveness for a project.

        Args:
            project: Optional project path to filter by
            since: Optional date string (ISO format) to filter from

        Returns:
            dict containing:
            - total_sessions: Number of sessions analyzed
            - total_prompts: Number of prompts scored
            - avg_effectiveness: Average effectiveness score (-1.0 to 1.0)
            - top_patterns: Most effective patterns found
            - common_issues: Most common negative signals
            - recommendations: Suggested improvements
        """
        self._ensure_initialized()

        result = {
            "total_sessions": 0,
            "total_prompts": 0,
            "avg_effectiveness": 0.0,
            "top_patterns": [],
            "common_issues": [],
            "recommendations": [],
        }

        if not self._scanner or not self._analyzer:
            logger.warning("Scanner or Analyzer not available for analysis")
            return result

        try:
            # Parse since date if provided
            since_dt = None
            if since:
                try:
                    since_dt = datetime.fromisoformat(since)
                except ValueError:
                    logger.warning(f"Invalid date format: {since}, using all data")

            # Scan sessions
            sessions = self._scanner.scan_sessions(project=project, since=since_dt)
            result["total_sessions"] = len(sessions)

            # Analyze each session
            all_scores: list[PromptEffectivenessScore] = []
            signal_counts: dict[str, int] = {}

            for session in sessions:
                scores = self._analyzer.analyze_session(session)
                all_scores.extend(scores)

                # Count signals
                for score in scores:
                    for signal in score.signals:
                        signal_name = signal.value if hasattr(signal, "value") else str(signal)
                        signal_counts[signal_name] = signal_counts.get(signal_name, 0) + 1

            result["total_prompts"] = len(all_scores)

            # Calculate average effectiveness
            if all_scores:
                result["avg_effectiveness"] = sum(s.score for s in all_scores) / len(all_scores)

            # Identify common issues (negative signals)
            negative_signals = [
                "immediate_clarification", "claude_confused", "multiple_retries",
                "error_in_response", "abandoned_task"
            ]
            result["common_issues"] = [
                {"signal": sig, "count": count}
                for sig, count in sorted(signal_counts.items(), key=lambda x: -x[1])
                if sig in negative_signals
            ][:5]

            # Get top patterns if knowledge base available
            if self._knowledge_base:
                patterns = self._knowledge_base.search_patterns("", k=5)
                result["top_patterns"] = [
                    {
                        "name": p.name if hasattr(p, "name") else str(p),
                        "score": p.effectiveness_score if hasattr(p, "effectiveness_score") else 0.0,
                    }
                    for p in patterns
                ]

            # Generate recommendations
            if result["avg_effectiveness"] < 0.3:
                result["recommendations"].append(
                    "Consider adding more specific instructions to your CLAUDE.md"
                )
            if signal_counts.get("immediate_clarification", 0) > 3:
                result["recommendations"].append(
                    "Many prompts require immediate clarification - try being more specific"
                )
            if signal_counts.get("claude_confused", 0) > 2:
                result["recommendations"].append(
                    "Claude shows confusion patterns - provide more context in prompts"
                )

        except Exception as e:
            logger.error(f"Analysis failed: {e}")

        return result

    def enhance(self, prompt: str) -> PromptEnhancement:
        """
        Enhance a prompt using learned patterns and rules.

        Args:
            prompt: The prompt text to enhance

        Returns:
            PromptEnhancement with original, enhanced version, and metadata
        """
        self._ensure_initialized()

        # Default response if enhancer not available
        default = PromptEnhancement(
            original=prompt,
            enhanced=prompt,
            improvements=[],
            expected_benefit="No enhancement available",
            confidence=0.0,
        )

        if not self._enhancer:
            logger.warning("PromptEnhancer not available")
            return default

        try:
            return self._enhancer.enhance_prompt(prompt)
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return default

    def get_patterns(self, search: Optional[str] = None, limit: int = 10) -> list:
        """
        Get effective prompt patterns.

        Args:
            search: Optional search query to filter patterns
            limit: Maximum number of patterns to return

        Returns:
            List of PromptPattern objects
        """
        self._ensure_initialized()

        if not self._knowledge_base:
            logger.warning("PromptKnowledgeBase not available")
            return []

        try:
            if search:
                return self._knowledge_base.search_patterns(search, k=limit)
            else:
                # Return all patterns sorted by effectiveness
                pattern_entries = [
                    e for e in self._knowledge_base.entries
                    if e.entry_type == "pattern"
                ]
                # Sort by effectiveness score descending
                sorted_entries = sorted(
                    pattern_entries,
                    key=lambda e: e.data.get("effectiveness_score", 0),
                    reverse=True
                )
                # Convert to PromptPattern objects
                patterns = []
                for entry in sorted_entries[:limit]:
                    data = entry.data
                    pattern = PromptPattern(
                        pattern_id=data.get("pattern_id", entry.id),
                        name=data.get("name", ""),
                        description=data.get("description", ""),
                        template=data.get("template", ""),
                        examples=data.get("examples", []),
                        effectiveness_score=data.get("effectiveness_score", 0.0),
                        use_cases=data.get("use_cases", []),
                        anti_patterns=data.get("anti_patterns", []),
                    )
                    patterns.append(pattern)
                return patterns
        except Exception as e:
            logger.error(f"Pattern retrieval failed: {e}")
            return []

    def suggest_improvements(self, project: Optional[str] = None) -> list:
        """
        Suggest CLAUDE.md improvements based on usage patterns.

        Args:
            project: Optional project path to analyze

        Returns:
            List of improvement suggestion dicts with:
            - type: Type of suggestion (add_rule, add_constraint, etc.)
            - target: Target file or project
            - suggestion: The suggested change
            - reason: Why this is suggested
            - confidence: How confident we are in this suggestion
        """
        self._ensure_initialized()

        if not self._evolution_tracker or not self._scanner:
            logger.warning("EvolutionTracker or Scanner not available")
            return []

        try:
            # Get recent sessions
            sessions = self._scanner.scan_sessions(project=project)

            # Generate suggestions
            suggestions = self._evolution_tracker.suggest_improvements(sessions)
            return suggestions

        except Exception as e:
            logger.error(f"Suggestion generation failed: {e}")
            return []

    def get_evolution_history(self, path: Optional[str] = None) -> list:
        """
        Get CLAUDE.md evolution history.

        Args:
            path: Optional specific file path to get history for

        Returns:
            List of ClaudeMdEvolution objects
        """
        self._ensure_initialized()

        if not self._evolution_tracker:
            logger.warning("EvolutionTracker not available")
            return []

        try:
            path_obj = Path(path) if path else None
            return self._evolution_tracker.get_evolution_history(path=path_obj)
        except Exception as e:
            logger.error(f"Evolution history retrieval failed: {e}")
            return []

    def get_stats(self) -> dict:
        """
        Get prompt knowledge base statistics.

        Returns:
            dict containing:
            - total_entries: Total entries in knowledge base
            - patterns: Number of prompt patterns
            - scores: Number of effectiveness scores
            - evolutions: Number of evolution records
            - pending_suggestions: Number of unresolved suggestions
            - projects_tracked: Number of distinct projects
            - components_available: Which components are initialized
        """
        self._ensure_initialized()

        stats = {
            "total_entries": 0,
            "patterns": 0,
            "scores": 0,
            "evolutions": 0,
            "pending_suggestions": 0,
            "projects_tracked": 0,
            "components_available": {
                "scanner": self._scanner is not None,
                "change_detector": self._change_detector is not None,
                "analyzer": self._analyzer is not None,
                "evolution_tracker": self._evolution_tracker is not None,
                "knowledge_base": self._knowledge_base is not None,
                "enhancer": self._enhancer is not None,
            },
        }

        if self._knowledge_base:
            try:
                kb_stats = self._knowledge_base.get_stats()
                stats.update({
                    "total_entries": kb_stats.get("total_entries", 0),
                    "patterns": kb_stats.get("patterns", 0),
                    "scores": kb_stats.get("scores", 0),
                    "evolutions": kb_stats.get("evolutions", 0),
                    "pending_suggestions": kb_stats.get("pending_suggestions", 0),
                    "projects_tracked": kb_stats.get("projects_tracked", 0),
                })
            except Exception as e:
                logger.error(f"Failed to get knowledge base stats: {e}")

        return stats


__all__ = [
    # Main facade
    "PromptSystem",
    # Scanner
    "Scanner",
    "ClaudeCodeSources",
    "ScannedPrompt",
    "ScannedSession",
    "ScannedClaudeMd",
    # Change detection
    "ChangeDetector",
    "Watermark",
    # Analyzer
    "EffectivenessSignal",
    "PromptEffectivenessScore",
    "PromptPattern",
    "EffectivenessAnalyzer",
    # Knowledge base
    "PromptKnowledgeBase",
    "PromptEntry",
    # Evolution tracking
    "ClaudeMdEvolution",
    "CommandEvolution",
    "EvolutionTracker",
    # Enhancer
    "PromptEnhancer",
    "PromptEnhancement",
]
