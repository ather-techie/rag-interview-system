# RAG Quick Reference Cheatsheet

## RAG Types at a Glance

| # | Type | Key Mechanism | Best For | Avoid When |
|---|---|---|---|---|
| 1 | **Naive RAG** | Chunk → Embed → Top-k retrieval | Prototypes, small corpora | High precision needed |
| 2 | **Advanced RAG** | HyDE + Hybrid search + Reranking | Production Q&A | Latency is critical |
| 3 | **Modular RAG** | Swappable pipeline components | Custom enterprise pipelines | Simple single-source systems |
| 4 | **Agentic RAG** | LLM-controlled multi-step retrieval | Complex reasoning, tool use | Strict latency budgets |
| 5 | **Graph RAG** | Knowledge graph traversal | Relational data, multi-hop | Simple FAQ systems |
| 6 | **Corrective RAG** | Retrieval quality evaluation + fallback | Out-of-KB or stale corpora | All queries are well-covered |
| 7 | **Self-RAG** | LLM trained with reflection tokens | High-accuracy specialized domains | General chatbots, no fine-tuning |
| 8 | **Speculative RAG** | Drafter → Verifier pattern | Cost/quality optimization | Multi-hop queries |
| 9 | **Multi-modal RAG** | Cross-modal embeddings (CLIP) | Docs with images/tables | Text-only corpora |
| 10 | **Long-context RAG** | Full documents in context | Complex documents, small corpus | Large corpora, cost-sensitive |

---

## Evaluation Metrics

| Metric | What it measures | Tool |
|---|---|---|
| Context Precision | Fraction of retrieved chunks that are relevant | RAGAS |
| Context Recall | Fraction of relevant info that was retrieved | RAGAS |
| Faithfulness | Does the answer only use the retrieved context? | RAGAS, TruLens |
| Answer Relevance | Does the answer actually address the question? | RAGAS |
| MRR | Rank of first relevant result | Custom |
| NDCG | Quality of ranked retrieval results | Custom |
| Latency P95 | 95th percentile end-to-end response time | Infrastructure |

---

## Common Tools by Layer

### Embedding Models
- `text-embedding-3-small/large` (OpenAI)
- `BGE-large-en-v1.5` (BAAI, open-source)
- `E5-mistral-7b-instruct` (Microsoft, open-source)
- `Cohere Embed v3`

### Vector Databases
- **Pinecone** — managed, production-grade
- **Weaviate** — hybrid search built-in
- **Qdrant** — fast, open-source
- **Chroma** — lightweight, local dev
- **FAISS** — library, not a server

### RAG Frameworks
- **LangChain** — general LLM orchestration
- **LlamaIndex** — data-centric RAG
- **Haystack** — modular, open-source
- **LangGraph** — stateful agentic workflows

### Evaluation Frameworks
- **RAGAS** — retrieval + generation metrics
- **TruLens** — LLM app evaluation
- **DeepEval** — unit-test style LLM eval
- **Arize Phoenix** — tracing + eval

### Rerankers
- `ms-marco-MiniLM-L-6-v2` (cross-encoder)
- `Cohere Rerank`
- `Jina Reranker v2`

---

## Decision Tree: Which RAG to Use?

```
Start
  │
  ├─ Is your corpus relational / entity-heavy?
  │     └─ YES → Graph RAG
  │
  ├─ Does your query require multiple retrieval steps?
  │     └─ YES → Agentic RAG
  │
  ├─ Does your corpus include images/tables?
  │     └─ YES → Multi-modal RAG
  │
  ├─ Is your corpus small (<50 docs) and complex?
  │     └─ YES → Long-context RAG
  │
  ├─ Is retrieval accuracy critical in a specialized domain?
  │     └─ YES → Self-RAG (if you can fine-tune)
  │
  ├─ Is your knowledge base potentially outdated/incomplete?
  │     └─ YES → Corrective RAG
  │
  ├─ Do you need cost/quality optimization at scale?
  │     └─ YES → Speculative RAG
  │
  ├─ Do you need a flexible, customizable pipeline?
  │     └─ YES → Modular RAG
  │
  ├─ Is this a production system needing good precision?
  │     └─ YES → Advanced RAG
  │
  └─ Prototyping or simple use case?
        └─ YES → Naive RAG
```
