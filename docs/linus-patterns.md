# Linus Review Patterns
<!-- Automatically updated by capture-linus command -->
<!-- Each pattern is a lesson learned from code reviews -->

## Code Quality Standards
- Always import all types used in type hints - using Optional[Any] without importing Any breaks static type checking
- Bug fix tests must explicitly check for the bug symptoms, not just command success - validates the actual fix and prevents regressions
- Don't make redundant system calls - cache expensive operation results and reuse them instead of rescanning
- Wrap filesystem operations in try/except with fail-safe behavior - operations like stat() can fail due to permissions, missing files, or network issues
- Avoid double-negative logic in variable names - use positive, intent-revealing names that match semantic meaning (e.g., 'needs_rebuild' instead of 'not stale')
- Include full context in log messages - use relative paths instead of just filenames when multiple files might have the same name
