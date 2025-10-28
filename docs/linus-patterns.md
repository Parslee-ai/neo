# Linus Review Patterns
<!-- Automatically updated by capture-linus command -->
<!-- Each pattern is a lesson learned from code reviews -->

## Code Quality Standards
- Always import all types used in type hints - using Optional[Any] without importing Any breaks static type checking
- Bug fix tests must explicitly check for the bug symptoms, not just command success - validates the actual fix and prevents regressions
