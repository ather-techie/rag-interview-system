# 04 — Patterns (Planned)

> **Status:** Planned — not yet written. This stub describes what the section will contain so you know what's coming and where to contribute.

## What this section will cover

Reusable RAG design patterns that cut across the 12 architectures in [`02_interview_bank/`](../02_interview_bank/):

- **Composition patterns** — router + fallback, retrieve-then-verify, cascade (cheap model first), fan-out/fan-in retrieval
- **Migration patterns** — upgrading Naive → Advanced → Modular → Agentic RAG without a rewrite
- **Anti-patterns** — over-chunking, premature agentification, reranking everything, one-index-for-all-tenants

## Intended format

One file per pattern: problem → forces/trade-offs → solution diagram → when *not* to use it → related interview questions.

## In the meantime

- Architecture selection guidance: [`cheatsheets/CHEATSHEET.md`](../cheatsheets/CHEATSHEET.md) decision tree
- Production patterns: [`00_overview/system_design_principles.md`](../00_overview/system_design_principles.md)

## Contributing

Have a pattern you've used (or been asked about) in production? See [CONTRIBUTING.md](../CONTRIBUTING.md) — real-world patterns are prioritized.
