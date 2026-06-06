# RAG Interview Questions & Answers

![CI](https://github.com/ather-techie/rag-interview-questions/actions/workflows/deploy.yml/badge.svg)
[![Stars](https://img.shields.io/github/stars/ather-techie/rag-interview-questions?style=social)](https://github.com/ather-techie/rag-interview-questions/stargazers)

A comprehensive collection of interview questions covering all major types of Retrieval-Augmented Generation (RAG) architectures.

## 🎯 Quick Reference

| RAG Type | Key Differentiator | When to Use |
|----------|--------------------|-------------|
| **Naive** | Chunk → embed → retrieve → generate | Prototyping; clean, static corpora |
| **Advanced** | Query rewriting + hybrid search + re-ranking | Production search over mixed content |
| **Modular** | Swap any pipeline component independently | Custom pipelines needing flexibility |
| **Agentic** | LLM decides when/how to retrieve dynamically | Multi-step reasoning; tool-heavy workflows |
| **Graph** | Knowledge graph for entity-aware retrieval | Complex relationships; multi-hop queries |
| **Corrective (CRAG)** | Evaluates retrieval quality; falls back to web | High-stakes grounding; potentially stale KB |
| **Self-RAG** | Model reflects on and critiques its own output | Maximum accuracy; iterative refinement |
| **Speculative** | Small model drafts; large model selects best | Cost-quality tradeoff at scale |
| **Multimodal** | Cross-modal retrieval (text, image, table, audio) | Mixed-media corpora; vision-heavy tasks |
| **Long-context** | Full docs stuffed into large context window | Documents where chunking breaks coherence |

---

## 📚 Sections

| # | Topic | Questions |
|---|-------|-----------|
| 01 | [Naive / Basic RAG](./sections/01-naive-rag.md) | 10 |
| 02 | [Advanced RAG](./sections/02-advanced-rag.md) | 10 |
| 03 | [Modular RAG](./sections/03-modular-rag.md) | 10 |
| 04 | [Agentic RAG](./sections/04-agentic-rag.md) | 10 |
| 05 | [Graph RAG](./sections/05-graph-rag.md) | 10 |
| 06 | [Corrective RAG (CRAG)](./sections/06-corrective-rag.md) | 10 |
| 07 | [Self-RAG](./sections/07-self-rag.md) | 10 |
| 08 | [Speculative RAG](./sections/08-speculative-rag.md) | 10 |
| 09 | [Multi-modal RAG](./sections/09-multimodal-rag.md) | 10 |
| 10 | [Long-context RAG](./sections/10-long-context-rag.md) | 10 |

**Total: 100 questions**

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
```

---

## 💡 How to Use

- Each section file contains **10 interview questions** with detailed answers
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

This repo grows best with real-world signal. If you were recently asked a RAG question in an interview — whether you answered brilliantly or blanked — **open a PR**.

**How to contribute:**
1. Fork the repo and open the relevant `sections/XX-<type>.md` file
2. Add your question under the appropriate difficulty tag (`[Basic]`, `[Intermediate]`, `[Advanced]`)
3. Include a solid answer (or a starter and let the community refine it)
4. Open a PR with the title: `Add: "<your question text>"`

Questions from real interviews are prioritized over synthetically generated ones. Even a rough question with no answer is valuable — open the PR and we'll fill it in together.

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
