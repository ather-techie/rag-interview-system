# RAG Interview Questions & Answers

A comprehensive collection of interview questions covering all major types of Retrieval-Augmented Generation (RAG) architectures.

## 📚 Sections

| # | Topic | Questions |
|---|-------|-----------|
| 01 | [Naive / Basic RAG](./sections/01-naive-rag.md) | 5 |
| 02 | [Advanced RAG](./sections/02-advanced-rag.md) | 5 |
| 03 | [Modular RAG](./sections/03-modular-rag.md) | 5 |
| 04 | [Agentic RAG](./sections/04-agentic-rag.md) | 5 |
| 05 | [Graph RAG](./sections/05-graph-rag.md) | 5 |
| 06 | [Corrective RAG (CRAG)](./sections/06-corrective-rag.md) | 5 |
| 07 | [Self-RAG](./sections/07-self-rag.md) | 5 |
| 08 | [Speculative RAG](./sections/08-speculative-rag.md) | 5 |
| 09 | [Multi-modal RAG](./sections/09-multimodal-rag.md) | 5 |
| 10 | [Long-context RAG](./sections/10-long-context-rag.md) | 5 |

**Total: 50 questions**

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

- Each section file contains **5 interview questions** with detailed answers
- Questions are tagged with difficulty: `[Basic]` `[Intermediate]` `[Advanced]`
- Great for both **interviewers** (to assess candidates) and **candidates** (to prepare)

---

## 🔗 Related Topics

- Vector Databases (Pinecone, Weaviate, Chroma, FAISS)
- Embedding Models (OpenAI, BGE, E5, Cohere)
- LLM Frameworks (LangChain, LlamaIndex, Haystack)
- Evaluation Frameworks (RAGAS, TruLens, DeepEval)

---

*Contributions welcome — open a PR to add more questions or improve answers.*
