# 03 — Modular RAG

> Treats the RAG pipeline as a set of interchangeable modules — retriever, reranker, reader, memory, router.

---

## Q1. What is Modular RAG and why is it an improvement over fixed pipelines? `[Basic]`

**Answer:**

Modular RAG decomposes the RAG system into independent, swappable components:

- **Search module** — vector search, keyword search, web search
- **Memory module** — short-term (conversation) and long-term (knowledge base)
- **Fusion module** — merges results from multiple retrievers
- **Routing module** — decides which retriever to call for a given query
- **Reader/Generator module** — the LLM that synthesizes the answer

Unlike Naive or Advanced RAG's fixed pipelines, Modular RAG lets you **add, remove, or swap** components without redesigning the whole system. This maps well to production engineering where different use cases need different retrieval strategies.

---

## Q2. How does a routing module work in Modular RAG? `[Intermediate]`

**Answer:**

A **routing module** decides which retrieval pathway to invoke based on the query type:

```
Query → Router
          ├── Structured query → SQL database retriever
          ├── Conceptual query → Vector store retriever
          ├── Recent events → Web search retriever
          └── Internal docs  → Private knowledge base retriever
```

Routing can be:
- **Rule-based** — keyword classifiers or regex patterns
- **LLM-based** — prompt the LLM to classify the query type
- **Learned** — a fine-tuned classifier

LlamaIndex's `RouterQueryEngine` and LangChain's `MultiRetrievalQAChain` are production implementations of this pattern.

---

## Q3. What is a fusion retriever and when would you use one? `[Intermediate]`

**Answer:**

A **fusion retriever** runs multiple retrievers in parallel and merges their results. For example:

1. Run dense vector search → top-10 chunks
2. Run BM25 keyword search → top-10 chunks
3. Run web search → top-5 results
4. Merge all 25 results using **Reciprocal Rank Fusion (RRF)**
5. Pass final top-5 to the LLM

**When to use it:**
- When no single retriever has full coverage
- For heterogeneous corpora (structured + unstructured data)
- When recall matters more than latency

**Trade-off:** Higher latency (parallel calls) and cost (more tokens passed to LLM).

---

## Q4. How do you handle memory in a Modular RAG system for multi-turn conversations? `[Advanced]`

**Answer:**

Two types of memory are needed:

| Memory Type | What it stores | Example |
|---|---|---|
| **Short-term** | Current conversation history | Last 5 user/assistant turns |
| **Long-term** | User preferences, past sessions | Summarized past interactions |

**Strategies:**

1. **Sliding window** — Keep the last N turns in the prompt.
2. **Summary memory** — Periodically summarize older turns with an LLM.
3. **Entity memory** — Extract and store key entities (names, preferences) as a mini knowledge base.
4. **Episodic memory** — Store past Q&A pairs in a vector DB; retrieve relevant past exchanges for context.

For production, tools like **MemGPT** or **Zep** implement long-term memory as a separate service.

---

## Q5. Compare Modular RAG to the LangChain and LlamaIndex frameworks. Which maps better? `[Advanced]`

**Answer:**

Both frameworks implement modular RAG concepts but with different philosophies:

| Aspect | LangChain | LlamaIndex |
|---|---|---|
| **Focus** | General LLM orchestration | Data-centric RAG pipelines |
| **Routing** | `MultiRetrievalQAChain` | `RouterQueryEngine` |
| **Memory** | `ConversationBufferMemory` | `ChatMemoryBuffer` |
| **Fusion** | `EnsembleRetriever` | `QueryFusionRetriever` |
| **Best for** | Complex agentic workflows | Document-heavy RAG systems |

**LlamaIndex** maps more directly to Modular RAG's architecture since it treats data ingestion, indexing, and querying as first-class modular concerns. LangChain is more flexible but requires more manual wiring. In practice, many production systems use both.
