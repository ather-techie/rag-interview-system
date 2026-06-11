# 10 — Decision System (Planned)

> **Status:** Planned — not yet written. This stub describes what the section will contain so you know what's coming and where to contribute.

## What this section will cover

An interactive architecture selector: answer a series of questions about your corpus, query traffic, latency budget, and team, and get a recommended RAG architecture with justification — essentially an executable version of the cheatsheet's decision tree:

- The full decision model (inputs → weights → recommendation) documented and arguable
- Edge cases where two architectures tie, and the tiebreaker reasoning
- "Defend this choice" interview practice: each recommendation comes with the counterarguments an interviewer would raise

## Intended format

A documented decision model (markdown) plus a small script/notebook implementing it.

## In the meantime

- Static decision tree: [`cheatsheets/CHEATSHEET.md`](../cheatsheets/CHEATSHEET.md#decision-tree-which-rag-to-use)
- Classification axes behind the decisions: [`00_overview/rag_taxonomy.md`](../00_overview/rag_taxonomy.md)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md).
