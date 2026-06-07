# RAG Interview Questions & Answers

![CI](https://github.com/ather-techie/rag-interview-questions/actions/workflows/deploy.yml/badge.svg)
[![Stars](https://img.shields.io/github/stars/ather-techie/rag-interview-questions?style=social)](https://github.com/ather-techie/rag-interview-questions/stargazers)
[![License: MIT](https://img.shields.io/github/license/ather-techie/rag-interview-questions)](LICENSE)
![Last Commit](https://img.shields.io/github/last-commit/ather-techie/rag-interview-questions)
![Questions](https://img.shields.io/badge/questions-140-blue)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

A comprehensive collection of interview questions covering all major types of Retrieval-Augmented Generation (RAG) architectures.

## 🎯 Quick Reference

→ [CHEATSHEET.md](CHEATSHEET.md) — all 12 RAG types compared by mechanism, best fit, and when to avoid each.

---

## 📚 Sections

| # | Topic | Questions |
|---|-------|-----------|
| 01 | [Naive / Basic RAG](./sections/01-naive-rag.md) | 12 |
| 02 | [Advanced RAG](./sections/02-advanced-rag.md) | 12 |
| 03 | [Modular RAG](./sections/03-modular-rag.md) | 12 |
| 04 | [Agentic RAG](./sections/04-agentic-rag.md) | 12 |
| 05 | [Graph RAG](./sections/05-graph-rag.md) | 12 |
| 06 | [Corrective RAG (CRAG)](./sections/06-corrective-rag.md) | 12 |
| 07 | [Self-RAG](./sections/07-self-rag.md) | 12 |
| 08 | [Speculative RAG](./sections/08-speculative-rag.md) | 12 |
| 09 | [Multi-modal RAG](./sections/09-multimodal-rag.md) | 12 |
| 10 | [Long-context RAG](./sections/10-long-context-rag.md) | 12 |
| 11 | [Adaptive RAG](./sections/11-adaptive-rag.md) | 10 |
| 12 | [Structured / SQL RAG](./sections/12-structured-rag.md) | 10 |

**Total: 140 questions**

**Difficulty distribution: 13 Basic, 58 Intermediate, 69 Advanced**

---

## 🗺️ RAG Landscape Overview

```
Naive RAG
  └── Chunk → Embed → Store → Retrieve → Generate

Advanced RAG
  └── Query rewriting + Hybrid search + Re-ranking

Modular RAG
  └── Plug-and-play pipeline components

Agentic RAG
  └── LLM decides when/how to retrieve (ReAct, FLARE)

Graph RAG
  └── Knowledge graph for entity-aware retrieval

Corrective RAG (CRAG)
  └── Evaluates retrieval quality, falls back to web search

Self-RAG
  └── Model trained to reflect, retrieve, and critique itself

Speculative RAG
  └── Small model drafts → Large model selects best

Multi-modal RAG
  └── Retrieve across text, images, tables, audio

Long-context RAG
  └── Stuff entire docs into large context windows

Adaptive RAG
  └── Query classifier routes to no-retrieval / single-hop / multi-hop

Structured / SQL RAG
  └── Text-to-SQL generation for relational database retrieval
```

---

## 💡 How to Use

- Each section file contains **10–12 interview questions** with detailed answers
  - Sections 01–10: 12 questions each (original 10 + Q11 on cost optimization + Q12 on security/adversarial robustness)
  - Sections 11–12: 10 questions each (new RAG types)
- Questions are tagged with difficulty: `[Basic]` `[Intermediate]` `[Advanced]`
- Great for both **interviewers** (to assess candidates) and **candidates** (to prepare)

---

## 🔗 Related Topics

- Vector Databases (Pinecone, Weaviate, Chroma, FAISS)
- Embedding Models (OpenAI, BGE, E5, Cohere)
- LLM Frameworks (LangChain, LlamaIndex, Haystack)
- Evaluation Frameworks (RAGAS, TruLens, DeepEval)

---

## Contributing

This repo grows best with real-world signal. If you were asked a RAG question in an interview, **open a PR** — real questions are prioritized over synthetically generated ones.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to submit a question.

---

## Support

For issues, questions, or general feedback:

- Open an issue on [GitHub](https://github.com/ather-techie/rag-interview-questions/issues)
- Join the [Discord community](https://discord.gg/kSUE3CA9P)
- Contact: [ather.techie@gmail.com](mailto:ather.techie@gmail.com)

---

## License

[MIT](LICENSE)

---

*See [Contributing](#contributing) to add your interview experience to the repo.*
