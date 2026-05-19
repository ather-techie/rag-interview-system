# 10 — Long-context RAG

> Leverages massive context windows (100K–1M tokens) to pass entire documents without chunking, trading compute for retrieval simplicity.

---

## Q1. What is Long-context RAG and how does it differ from chunk-based RAG? `[Basic]`

**Answer:**

**Long-context RAG** exploits the large context windows of modern LLMs (Gemini 1.5 Pro: 1M tokens; Claude 3: 200K; GPT-4o: 128K) to pass entire documents — or large portions of a corpus — directly into the prompt, eliminating the need for chunking and retrieval.

| Aspect | Chunk-based RAG | Long-context RAG |
|---|---|---|
| **Chunking** | Required | Not needed |
| **Retrieval step** | Required | Reduced or eliminated |
| **Context quality** | May miss context boundaries | Full document coherence preserved |
| **Compute cost** | Low (only top-k chunks) | High (full document per query) |
| **Latency** | Low | Higher (long prefill) |
| **Best for** | Large corpora | Small-medium corpora, complex documents |

---

## Q2. What is the "needle-in-a-haystack" problem and how does it relate to Long-context RAG? `[Intermediate]`

**Answer:**

The **needle-in-a-haystack (NIAH)** test measures whether an LLM can find a specific fact ("needle") hidden within a large document ("haystack"). It evaluates:

- **Recall by position** — Does performance degrade when the needle is in the middle of the context?
- **Recall by depth** — Does performance degrade as the haystack grows longer?

**Relevance to Long-context RAG:**

Long-context RAG assumes the LLM can use all the context it's given — but NIAH tests reveal that most models have **degraded recall for information in the middle** of very long contexts (the "lost-in-the-middle" problem).

**Mitigation strategies:**
- Place the most critical documents at the **start or end** of the prompt.
- Use models specifically optimized for long context (Gemini 1.5, Claude 3 which use special attention mechanisms).
- Combine long-context with retrieval: do a coarse retrieval to narrow to 10–20 relevant documents, then pass all of them to the long-context LLM.

---

## Q3. When should you use Long-context RAG over traditional chunked RAG? `[Intermediate]`

**Answer:**

**Choose Long-context RAG when:**

1. **Document coherence matters** — Legal contracts, scientific papers, or financial reports where context spans the entire document.
2. **Small corpus** — Under ~50 documents; cheap to stuff everything in.
3. **Complex multi-part queries** — Questions that require synthesizing information from many sections of a document.
4. **Cross-document reasoning** — Comparing two contracts, finding contradictions across documents.
5. **Avoiding retrieval errors** — When chunking heuristics produce too many irrelevant or fragmented results.

**Stick with chunked RAG when:**
- Your corpus is millions of documents (context window can't scale to that).
- Latency and cost are critical — long-context inference is 10–100x more expensive per query.
- Most queries are narrow and answered by a small portion of your corpus.

---

## Q4. How do prompt compression techniques reduce the cost of Long-context RAG? `[Advanced]`

**Answer:**

Prompt compression reduces the number of tokens passed to the LLM while preserving the information needed to answer the query:

**Technique 1 — LLMLingua / LongLLMLingua:**
- Uses a small LM to score token importance for the given query.
- Drops low-importance tokens (filler words, redundant sentences).
- Achieves 3–20x compression with minimal accuracy loss.

**Technique 2 — Selective Context:**
- Computes self-information (surprisal) of each sentence.
- Removes low-surprisal (redundant) sentences.

**Technique 3 — Recomp:**
- Trains a compressor model to generate abstractive summaries of retrieved documents that are tailored to the query.

**Technique 4 — RAG-Token:**
- Rather than compressing the input, the model is trained to generate answers token-by-token where each token can attend to different retrieved passages.

**Result:** These techniques let you get the coherence benefits of long-context while reducing cost by 5–20x.

---

## Q5. Design a hybrid system that combines retrieval with long-context to handle a 10,000-document legal corpus. `[Advanced]`

**Answer:**

**The problem:** 10,000 legal documents × ~20K tokens each = 200M tokens total — far too large for any context window.

**Hybrid architecture:**

```
Stage 1 — Coarse Retrieval (fast, cheap)
  Query → BM25 + vector search → top 20 relevant documents
  [milliseconds, negligible cost]

Stage 2 — Long-context Synthesis (slow, expensive — but only for top 20)
  Top 20 docs (~400K tokens) → Claude 3 / Gemini 1.5 → Answer
  [seconds, moderate cost per query]
```

**Optimizations:**

1. **Document-level caching** — Cache the KV (key-value attention cache) of frequently accessed documents. On repeated access, skip recomputation of the document prefix.
   - Anthropic's **prompt caching** feature reduces cost by up to 90% for repeated context.
   
2. **Tiered retrieval** — Stage 1 narrows to 20 docs; Stage 2 optionally narrows to the 3 most relevant full documents for the costliest queries.

3. **Background indexing** — Run nightly BM25 + embedding updates as new documents arrive.

4. **Answer caching** — Cache answers for common query patterns (e.g., "what is the governing law clause in contract X?" is likely asked repeatedly).

**Cost estimate:** 20 docs × 20K tokens = 400K input tokens per query. At $3/1M tokens (with caching), that's ~$1.20/query — acceptable for high-value legal use cases.
