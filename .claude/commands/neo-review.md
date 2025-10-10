---
description: Get Neo's code review with semantic analysis
argument-hint: [file-path or code-selection]
---

Use the Neo agent to perform a comprehensive code review focusing on:
- Correctness and edge cases
- Performance and algorithmic complexity
- Code quality and maintainability
- Common pitfalls and anti-patterns
- Security concerns

Target: ${ARGUMENTS:-.}

Provide Neo with the code and request detailed review with confidence scores for each finding.

IMPORTANT: Pass through Neo's raw output WITH PERSONALITY INTACT. Do NOT reformat, sanitize, or convert into a corporate summary. Neo has a distinctive voice - preserve it. Just present Neo's response directly to the user.
