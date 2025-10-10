---
description: "Get optimization suggestions from Neo"
---

Get optimization suggestions from Neo.

## Usage

```
/neo-optimize <file path or function name>
```

## Description

Use this command to get performance optimization recommendations from Neo using semantic analysis and past optimization patterns.

## Examples

```
/neo-optimize process_large_dataset function

/neo-optimize src/data/processor.py

/neo-optimize the search algorithm
```

## What Happens

Neo will:
1. Analyze the code for algorithmic complexity
2. Identify bottlenecks and inefficiencies
3. Search memory for similar optimization patterns
4. Suggest improvements with confidence scores

## Parameters

- `<target>` - File path or function name (required)

Optionally include performance requirements (e.g., "needs <2s for 10k records")
