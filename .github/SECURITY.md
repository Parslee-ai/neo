# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.7.x   | :white_check_mark: |
| < 0.7   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them using one of the following methods:

### Option 1: GitHub Security Advisories (Preferred)

Report a vulnerability privately using GitHub's built-in security advisory feature:
- Go to https://github.com/Parslee-ai/neo/security/advisories/new
- Fill in the details
- We will respond within 48 hours

### Option 2: Email

Email security concerns to: **hello@parslee.ai**

Please include:
- Type of vulnerability
- Full paths of affected source files
- Location of the affected code (tag/branch/commit)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability

## Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 5 business days with confirmation or questions
- **Fix Timeline**: Depends on severity
  - Critical: 1-7 days
  - High: 7-30 days
  - Medium: 30-90 days
  - Low: 90+ days

## Disclosure Policy

We follow coordinated disclosure:

1. Security report received and acknowledged
2. Issue confirmed and assessed for severity
3. Fix developed and tested
4. Security advisory published on GitHub
5. New version released with fix
6. Public disclosure after users have time to update

## Security Best Practices for Users

When using Neo:

- **API Keys**: Never commit API keys to Git. Use environment variables.
- **Updates**: Keep Neo updated to the latest version for security patches.
- **Dependencies**: Enable Dependabot alerts if forking this repository.
- **Trusted Publishing**: When publishing to PyPI, use GitHub Actions with Trusted Publishing instead of long-lived API tokens.

## Known Security Considerations

### Persistent Memory Storage

Neo stores reasoning patterns locally in `~/.neo/`:
- **Privacy**: Memory files may contain code snippets from your projects
- **Recommendation**: Review memory files before sharing your system
- **Location**: `~/.neo/reasoning_patterns.json`

### LLM Provider API Keys

Neo requires API keys for language model providers:
- **Storage**: Keys stored in environment variables or `~/.neo/config.json`
- **Protection**: Config file has restricted permissions (600)
- **Recommendation**: Use short-lived or scoped API keys when possible

## Bug Bounty

We currently do not have a formal bug bounty program. However, we deeply appreciate security researchers who report vulnerabilities responsibly and will acknowledge your contribution in our release notes (if desired).

## Contact

For general security questions: hello@parslee.ai

Thank you for helping keep Neo and our users safe!
