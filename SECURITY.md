# Security Policy

## Supported Versions

We release security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.7.x   | ✅ Current release |
| < 0.7   | ❌ No longer supported |

We recommend always using the latest version from PyPI.

## Reporting a Vulnerability

If you discover a security vulnerability in Neo, please report it responsibly:

### How to Report

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please:

1. **Create a private security advisory** on GitHub:
   - Go to https://github.com/Parslee-ai/neo/security/advisories
   - Click "Report a vulnerability"
   - Provide details about the vulnerability

2. **Or email us directly:**
   - Email: security@parslee.ai
   - Subject: "Neo Security Vulnerability Report"

### What to Include

Please include the following in your report:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if you have one)
- Your contact information (optional)

### Response Timeline

- **Initial response:** Within 7 days
- **Status update:** Every 7 days until resolved
- **Fix timeline:** Depends on severity (critical issues prioritized)

### Disclosure Policy

- We will acknowledge your report within 7 days
- We will work with you to understand and validate the issue
- We will develop and test a fix
- We will release a security update
- We will publicly disclose the vulnerability after the fix is released
- We will credit you in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices

When using Neo:

### API Key Security

- **Never commit API keys** to version control
- **Use environment variables** for API keys
- **Rotate keys regularly** if they may have been exposed
- **Limit key permissions** to minimum required access

### Local Storage

- Neo stores patterns in `~/.neo/` directory
- This includes code snippets from your projects
- Ensure proper file permissions on `~/.neo/` directory
- Consider excluding sensitive codebases from Neo's context

### Network Security

- Neo makes API calls to your configured LLM provider
- Code context is sent to the LLM for analysis
- **Do not use Neo with proprietary/sensitive code** unless your LLM provider's terms allow it
- Review your LLM provider's data handling policies

## Scope

### In Scope

Security issues we will address:

- Code execution vulnerabilities
- Arbitrary file read/write vulnerabilities
- API key leakage or exposure
- Malicious input handling
- Dependency vulnerabilities

### Out of Scope

The following are not considered security vulnerabilities:

- LLM output quality or correctness
- Performance issues or resource usage
- Feature requests
- Issues in third-party dependencies (report to those projects directly)

## Historical Security Issues

### Hardcoded Credentials (Resolved in v0.7.0)

**Status:** Fixed

Prior to v0.7.0, Neo supported Azure Cosmos DB for cloud storage. A development version inadvertently contained a hardcoded connection string. This was immediately removed.

**Current Status:**
- Neo v0.7.0+ uses local file storage only (~/.neo directory)
- No cloud credentials required
- No remote storage connections

---

Thank you for helping keep Neo secure!
