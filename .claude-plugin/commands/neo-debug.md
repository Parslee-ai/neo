---
description: "Get debugging assistance from Neo"
---

Get debugging assistance from Neo.

## Usage

```
/neo-debug <error message or bug description>
```

## Description

Use this command when debugging complex issues, especially intermittent bugs or race conditions. Neo uses semantic pattern matching to identify likely root causes.

## Examples

```
/neo-debug TypeError in data processing pipeline

/neo-debug Race condition in concurrent task processor

/neo-debug Memory leak happening after 1000+ requests
```

## What Happens

Neo will:
1. Analyze the error or bug description
2. Identify likely root causes using semantic patterns
3. Search memory for similar debugging scenarios
4. Suggest debugging strategies and fixes with confidence scores

## Parameters

- `<description>` - Error message or bug description (required)

Include: error messages, frequency, reproduction steps, environment details
