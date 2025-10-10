---
description: "Get Neo's code review with semantic analysis"
---

Get Neo's code review with semantic analysis.

## Usage

```
/neo-review <file path or code description>
```

## Description

Use this command to get Neo's code review with semantic pattern matching, security analysis, and optimization suggestions.

## Examples

```
/neo-review src/api/handlers.py

/neo-review authentication module

/neo-review the payment processing code
```

## What Happens

Neo will:
1. Gather context from the specified file or module
2. Analyze code for security vulnerabilities, edge cases, and performance issues
3. Check semantic memory for similar code review patterns
4. Provide improvements with confidence scores

## Parameters

- `<target>` - File path, module name, or code description (required)

Focus areas: security, edge cases, error handling, performance
