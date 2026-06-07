# RAG Interview Questions & Answers


[![Stargazers][stars-shield]][stars-url]
[![License: MIT][license-shield]][license-url]
![Last Commit][commits-shield]
![Questions][questions-shield]
[![PRs Welcome][prs-shield]][prs-url]


<p align="center">
  <img src="assets/logos/image.png" alt="RAG Interview Questions Banner" width="800" />
</p>

A comprehensive collection of interview questions covering all major types of Retrieval-Augmented Generation (RAG) architectures.

## 🎯 Quick Reference

→ [CHEATSHEET.md](CHEATSHEET.md) — all 12 RAG types compared by mechanism, best fit, and when to avoid each.

---

## 📚 Sections

### 📖 Overview & Concepts

| # | Topic | Purpose |
|---|-------|---------|
| 00a | [Roadmap](./00_overview/roadmap.md) | RAG maturity model, skill progression, and interview prep pathway |
| 00b | [RAG Taxonomy](./00_overview/rag_taxonomy.md) | Classification framework for all 12 architectures |
| 00c | [Learning Path](./00_overview/learning_path.md) | Structured curriculum and study plans |
| 00d | [System Design Principles](./00_overview/system_design_principles.md) | Production-grade architecture patterns |
| 01a | [Embeddings](./01_concepts/embeddings.md) | Embedding models, similarity metrics, and fine-tuning |
| 01b | [Chunking Strategies](./01_concepts/chunking_strategies.md) | Document splitting and chunk optimization |
| 01c | [Vector Databases](./01_concepts/vector_databases.md) | Storage, indexing, and hybrid search |
| 01d | [Retrieval Strategies](./01_concepts/retrieval_strategies.md) | Dense, sparse, hybrid, and advanced retrieval |
| 01e | [Reranking](./01_concepts/reranking.md) | Cross-encoders and precision filtering |
| 01f | [Evaluation Metrics](./01_concepts/evaluation_metrics.md) | RAGAS, NDCG, and production monitoring |
| 01g | [Prompt Injection Risks](./01_concepts/prompt_injection_risks.md) | Security and defense strategies |

### ❓ Interview Question Banks

| # | Topic | Questions |
|---|-------|-----------|
| 02.01 | [Naive / Basic RAG](./02_interview_bank/01-naive-rag.md) | 12 |
| 02.02 | [Advanced RAG](./02_interview_bank/02-advanced-rag.md) | 12 |
| 02.03 | [Modular RAG](./02_interview_bank/03-modular-rag.md) | 12 |
| 02.04 | [Agentic RAG](./02_interview_bank/04-agentic-rag.md) | 12 |
| 02.05 | [Graph RAG](./02_interview_bank/05-graph-rag.md) | 12 |
| 02.06 | [Corrective RAG (CRAG)](./02_interview_bank/06-corrective-rag.md) | 12 |
| 02.07 | [Self-RAG](./02_interview_bank/07-self-rag.md) | 12 |
| 02.08 | [Speculative RAG](./02_interview_bank/08-speculative-rag.md) | 12 |
| 02.09 | [Multi-modal RAG](./02_interview_bank/09-multimodal-rag.md) | 12 |
| 02.10 | [Long-context RAG](./02_interview_bank/10-long-context-rag.md) | 12 |
| 02.11 | [Adaptive RAG](./02_interview_bank/11-adaptive-rag.md) | 10 |
| 02.12 | [Structured / SQL RAG](./02_interview_bank/12-structured-rag.md) | 10 |

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

**Three content types:**

1. **Overview & Concepts (00_overview/, 01_concepts/)** — Reference material, not Q&A
   - Read these first to build foundational understanding
   - Comparison tables, ASCII diagrams, code examples, and system design patterns
   - Use to answer conceptual questions and understand mechanisms deeply

2. **Interview Questions (02_interview_bank/)** — 10–12 questions per architecture
   - Each section contains interview-style Q&A with detailed answers
   - Sections 01–10: 12 questions each (original 10 + Q11 on cost optimization + Q12 on security)
   - Sections 11–12: 10 questions each (newer RAG types)
   - Questions are tagged with difficulty: `[Basic]` `[Intermediate]` `[Advanced]`

3. **CHEATSHEET (CHEATSHEET.md)** — Quick reference
   - All 12 RAG types compared in one table
   - Use during phone screens or quick prep

**Study path:**
- **1-week prep:** Start with `00_overview/learning_path.md` → pick a track → follow the schedule
- **Phone screen:** `CHEATSHEET.md` + Q1–Q5 from relevant architectures
- **System design round:** `00_overview/system_design_principles.md` + Q9–Q12 from all files
- **Deep prep:** Read `01_concepts/` files + all `02_interview_bank/` Q&A

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

<!-- Badge References -->
[stars-shield]: https://img.shields.io/github/stars/ather-techie/rag-interview-questions?style=social
[stars-url]: https://github.com/ather-techie/rag-interview-questions/stargazers
[license-shield]: https://img.shields.io/github/license/ather-techie/rag-interview-questions
[license-url]: LICENSE
[commits-shield]: https://img.shields.io/github/last-commit/ather-techie/rag-interview-questions
[questions-shield]: https://img.shields.io/badge/questions-140-blue
[prs-shield]: https://img.shields.io/badge/PRs-welcome-brightgreen
[prs-url]: CONTRIBUTING.md
