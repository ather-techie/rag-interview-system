# Security Policy

This repository is a collection of interview-preparation Markdown, a static GitHub Pages site, and lab notebooks. It runs no server and stores no user data; the quiz and assessment keep their state in the visitor's own browser.

## Reporting a vulnerability

If you find a security problem (for example, a cross-site scripting issue on the published site, a leaked credential in the repository, or a dependency with a known exploit that affects the build), please **do not open a public issue**. Email [ather.techie@gmail.com](mailto:ather.techie@gmail.com) with the details and, if possible, steps to reproduce. You should get a reply within a few days.

## Scope notes

- Answers in the interview bank are educational and not a security review of any product.
- Lab notebooks call third-party LLM APIs with a key you supply in `.env` or at a prompt. Never commit a real key; `.env` is git-ignored.
