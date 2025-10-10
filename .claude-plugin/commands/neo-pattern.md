---
description: "Extract reusable patterns with Neo"
---

Extract reusable patterns with Neo.

## Usage

```
/neo-pattern <code area or pattern type>
```

## Description

Use this command to extract reusable architectural patterns from your codebase. Neo will identify patterns, analyze their effectiveness, and suggest when to apply them.

## Examples

```
/neo-pattern repository pattern implementation

/neo-pattern error handling across the API

/neo-pattern caching strategies in the codebase
```

## What Happens

Neo will:
1. Analyze code to identify the pattern
2. Evaluate pattern effectiveness and consistency
3. Search memory for similar patterns
4. Suggest improvements or alternative patterns with confidence scores

## Parameters

- `<target>` - Code area or pattern type (required)

Specify what pattern you want to analyze or where to look for patterns
