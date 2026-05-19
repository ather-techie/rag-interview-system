# 02 — Advanced RAG

> Improves on Naive RAG with pre-retrieval, retrieval, and post-retrieval enhancements.

---

## Q1. What distinguishes Advanced RAG from Naive RAG? `[Basic]`

**Answer:**

Advanced RAG introduces improvements at three stages:

| Stage | Technique |
|---|---|
| **Pre-retrieval** | Query rewriting, query expansion, HyDE |
| **Retrieval** | Hybrid search (dense + sparse), better embeddings |
| **Post-retrieval** | Reranking, context compression, lost-in-the-middle mitigation |

The key insight is that the raw user query is often a poor retrieval signal — it may be vague, ambiguous, or phrased differently from how the documents are written.

---

## Q2. What is HyDE and when is it useful? `[Intermediate]`

**Answer:**

**HyDE (Hypothetical Document Embeddings)** is a pre-retrieval technique where:

1. The LLM is prompted to generate a *hypothetical* answer to the query.
2. That hypothetical answer (not the original query) is embedded and used for retrieval.

**Why it works:** The hypothetical answer lives in the same semantic space as real documents, so it retrieves more relevant chunks than the short, ambiguous query.

**When to use it:**
- Open-domain question answering where the query is sparse
- When the query vocabulary doesn't match the document vocabulary
- Not suitable for real-time systems with strict latency budgets (adds one LLM call)

---

## Q3. How does hybrid search improve retrieval quality? `[Intermediate]`

**Answer:**

Hybrid search combines **dense (vector) retrieval** with **sparse (BM25/keyword) retrieval**:

- **Dense retrieval** excels at semantic similarity — good for paraphrased or conceptually related queries.
- **Sparse retrieval** excels at exact keyword matching — good for named entities, product codes, or technical terms.

Results from both are merged using **Reciprocal Rank Fusion (RRF)** or a learned ranker. This outperforms either method alone, especially on out-of-domain or tail queries.

**Tools:** Elasticsearch, OpenSearch, Weaviate, and Qdrant all support hybrid search natively.

---

## Q4. What is a cross-encoder reranker and how does it differ from a bi-encoder? `[Advanced]`

**Answer:**

| | Bi-encoder | Cross-encoder |
|---|---|---|
| **How it works** | Query and doc embedded separately; similarity via dot product | Query and doc fed together into one model |
| **Speed** | Fast — embeddings pre-computed | Slow — must run per (query, doc) pair |
| **Accuracy** | Good for retrieval at scale | Higher accuracy for reranking top-k |
| **Use** | First-stage retrieval | Second-stage reranking |

The typical pipeline: bi-encoder retrieves top-100 → cross-encoder reranks to top-5. Models like `ms-marco-MiniLM` or Cohere Rerank are common cross-encoders.

---

## Q5. What is the "lost-in-the-middle" problem and how do you address it? `[Advanced]`

**Answer:**

Research shows that LLMs perform worse when relevant information appears in the **middle** of a long context — they are better at using information at the beginning or end of the prompt.

**Mitigations:**

1. **Reorder retrieved chunks** — Place the most relevant chunks at the start or end, not the middle.
2. **Context compression** — Use a smaller model to summarize or filter each retrieved chunk before passing it to the LLM.
3. **Reduce k** — Pass fewer, higher-quality chunks rather than many lower-quality ones.
4. **LongLLMLingua / Selective Context** — Prompt compression tools that remove low-information tokens from retrieved context.

Addressing this is critical for production systems where k > 5.
