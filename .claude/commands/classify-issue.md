---
name: classify_issue
pattern: /classify-issue
description: Classify issue as chore, bug, or feature
parameters:
  - name: issue_description
    description: description of the issue or link to github issue
    required: true
---

# Issue Command Selection

Based on the `Issue` below, follow the `Instructions` to select the appropriate command to execute based on the `Command Mapping`.

## Instructions

- Based on the details in the `Issue`, select the appropriate command to execute.
- Respond exclusively with '/' followed by the command to execute.
- Use the command mapping to help you decide which command to respond with.
- Ultrathink about the command to execute.

## Command Mapping

- Respond with `/chore-autonomous` if the issue is a chore.
- Respond with `/bug-autonomous` if the issue is a bug.
- Respond with `/feature-autonomous` if the issue is a feature.
- Respond with `0` if the issue isn't any of the above.

## Issue

$ARGUMENTS