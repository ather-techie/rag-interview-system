# 01 — Naive / Basic RAG

> The simplest RAG form: chunk → embed → store → retrieve → generate.

---

## Q1. What is Naive RAG and how does it work at a high level? `[Basic]`

**Answer:**

Naive RAG is the foundational form of retrieval-augmented generation, following a fixed three-step pipeline:

1. **Indexing** — Documents are split into fixed-size chunks, each chunk is embedded into a vector, and stored in a vector database.
2. **Retrieval** — At query time, the query is embedded and the top-k most similar chunks are fetched via cosine (or dot-product) similarity.
3. **Generation** — Retrieved chunks are concatenated into a prompt and passed to the LLM to produce the final answer.

It is simple to implement but suffers from poor precision, no query understanding, and context fragmentation.

---

## Q2. What are the key limitations of Naive RAG? `[Basic]`

**Answer:**

| Limitation | Description |
|---|---|
| Chunking artifacts | Fixed-size splits can cut context mid-sentence, losing coherence |
| No query understanding | Raw query may not align with how content is stored |
| Low precision | Top-k cosine search can return irrelevant chunks |
| No feedback loop | Retriever and generator are not jointly optimized |
| Context stuffing | All chunks passed verbatim — no reranking or filtering |

---

## Q3. What embedding strategies are commonly used in Naive RAG? `[Intermediate]`

**Answer:**

- **Dense embeddings** — Models like `text-embedding-ada-002`, `BGE-large`, or `E5-mistral` map text to dense float vectors.
- **Sparse embeddings** — BM25 or TF-IDF produce sparse keyword-weighted vectors, better for exact-match queries.
- **Chunk overlap** — A sliding window (e.g., 50-token overlap) reduces boundary cutoff artifacts.
- **Sentence-level chunking** — Using sentence or paragraph boundaries instead of fixed token counts preserves semantic units.

The choice of embedding model heavily influences retrieval quality; domain-specific fine-tuned embeddings often outperform general-purpose ones.

---

## Q4. How do you evaluate the quality of retrieval in a Naive RAG system? `[Intermediate]`

**Answer:**

Common retrieval evaluation metrics:

- **Context Precision** — What fraction of retrieved chunks are actually relevant?
- **Context Recall** — What fraction of all relevant information was retrieved?
- **MRR (Mean Reciprocal Rank)** — Measures how high the first relevant chunk ranks.
- **NDCG** — Weighs relevant results appearing higher in the list more heavily.
- **RAGAS** — An open-source framework that evaluates faithfulness, answer relevance, context precision, and context recall end-to-end.

A common pitfall is optimizing only for recall (retrieve everything) at the cost of precision (retrieve junk).

---

## Q5. In what scenarios would you still choose Naive RAG over more complex approaches? `[Advanced]`

**Answer:**

Naive RAG remains valid when:

1. **Prototyping / POCs** — Speed of setup outweighs retrieval quality; you can iterate later.
2. **Small, clean corpora** — A compact, well-structured knowledge base achieves high precision without reranking.
3. **Low-latency requirements** — No reranking or multi-step retrieval means faster end-to-end response.
4. **Cost constraints** — Fewer API calls and no additional reranker model reduces operational cost.
5. **Narrow-domain queries** — When all documents are highly relevant and query distribution is predictable, Naive RAG performs surprisingly well.

Always profile your specific use case before over-engineering — Naive RAG often gets you 80% of the way.
