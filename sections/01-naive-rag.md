# 01 — Naive / Basic RAG

> The simplest RAG form: chunk → embed → store → retrieve → generate.

---

## Q1. What is Naive RAG and how does it work at a high level? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Naive RAG is the foundational form of retrieval-augmented generation, following a fixed three-step pipeline:

1. **Indexing** — Documents are split into fixed-size chunks, each chunk is embedded into a vector, and stored in a vector database.
2. **Retrieval** — At query time, the query is embedded and the top-k most similar chunks are fetched via cosine (or dot-product) similarity.
3. **Generation** — Retrieved chunks are concatenated into a prompt and passed to the LLM to produce the final answer.

It is simple to implement but suffers from poor precision, no query understanding, and context fragmentation.

```
[Offline Indexing]
Docs → Chunker → Embedder ──► Vector DB
                                  ▲
[Online Query]                    │
Query → Embedder → ANN Search ────┘ → Top-k chunks → LLM → Answer
```

</details>

---

## Q2. What are the key limitations of Naive RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Limitation | Description |
|---|---|
| Chunking artifacts | Fixed-size splits can cut context mid-sentence, losing coherence |
| No query understanding | Raw query may not align with how content is stored |
| Low precision | Top-k cosine search can return irrelevant chunks |
| No feedback loop | Retriever and generator are not jointly optimized |
| Context stuffing | All chunks passed verbatim — no reranking or filtering |

</details>

---

## Q3. What embedding strategies are commonly used in Naive RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

- **Dense embeddings** — Models like `text-embedding-ada-002`, `BGE-large`, or `E5-mistral` map text to dense float vectors.
- **Sparse embeddings** — BM25 or TF-IDF produce sparse keyword-weighted vectors, better for exact-match queries.
- **Chunk overlap** — A sliding window (e.g., 50-token overlap) reduces boundary cutoff artifacts.
- **Sentence-level chunking** — Using sentence or paragraph boundaries instead of fixed token counts preserves semantic units.

The choice of embedding model heavily influences retrieval quality; domain-specific fine-tuned embeddings often outperform general-purpose ones.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)
results = vectorstore.similarity_search(query, k=5)
```

</details>

---

## Q4. How do you evaluate the quality of retrieval in a Naive RAG system? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Common retrieval evaluation metrics:

- **Context Precision** — What fraction of retrieved chunks are actually relevant?
- **Context Recall** — What fraction of all relevant information was retrieved?
- **MRR (Mean Reciprocal Rank)** — Measures how high the first relevant chunk ranks.
- **NDCG** — Weighs relevant results appearing higher in the list more heavily.
- **RAGAS** — An open-source framework that evaluates faithfulness, answer relevance, context precision, and context recall end-to-end.

A common pitfall is optimizing only for recall (retrieve everything) at the cost of precision (retrieve junk).

</details>

---

## Q5. In what scenarios would you still choose Naive RAG over more complex approaches? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Naive RAG remains valid when:

1. **Prototyping / POCs** — Speed of setup outweighs retrieval quality; you can iterate later.
2. **Small, clean corpora** — A compact, well-structured knowledge base achieves high precision without reranking.
3. **Low-latency requirements** — No reranking or multi-step retrieval means faster end-to-end response.
4. **Cost constraints** — Fewer API calls and no additional reranker model reduces operational cost.
5. **Narrow-domain queries** — When all documents are highly relevant and query distribution is predictable, Naive RAG performs surprisingly well.

Always profile your specific use case before over-engineering — Naive RAG often gets you 80% of the way.

</details>

---

## Q6. How do chunking strategies affect retrieval quality? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Chunking strategy directly impacts both retrieval precision and recall:

- **Fixed-size chunks (e.g., 512 tokens)** — Efficient and predictable, but may split semantic units, losing context boundaries.
- **Sliding window overlap** — Reduces boundary artifacts by repeating context; increases index size but improves continuity.
- **Semantic/sentence-based chunking** — Preserves natural language boundaries, reducing fragmentation but adding computational cost.
- **Hierarchical chunking** — Chunk documents into paragraphs, then sub-chunks; allows retrieval at multiple granularities.

```
[Naive: Fixed 512-token chunks]
"The CEO announced..."   |  "...a $100M acquisition"  |  "The deal closes next"
                                 ^ boundary cut loses context

[With sliding window 50-token overlap]
"The CEO announced..."        |  "announced a $100M acquisition deal"     |  "acquisition deal closes next"
        └─ preserved ─────────────────────────────────────────────────────────────┘
```

A practical benchmark: measure retrieval recall vs. index size. Overlap 10-15% of chunk size often balances both.

```python
# Compare strategies
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, 
    CharacterTextSplitter
)

# Fixed chunk, no overlap
fixed = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)

# Sliding window
sliding = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

# Semantic (split by sentence, then combine up to 512 tokens)
semantic = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=512,
    chunk_overlap=50
)
```

</details>

---

## Q7. How does approximate nearest neighbor (ANN) search work in vector DBs? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Exact cosine similarity search is O(n·d) — infeasible at scale. Vector DBs use approximate indexing to trade small accuracy loss for massive speed gains.

| Technique | Data Structure | Query Time | Index Size | Best For |
|-----------|----------------|-----------|-----------|----------|
| **HNSW** | Hierarchical graph | O(log n) | +30% | General-purpose; balanced speed/memory |
| **IVF** | Inverted file clusters | O(k log n) | Compact | Very large (>10M) embeddings |
| **PQ** | Product quantization | O(1) lookup | Tiny | Mobile/edge; <1% recall loss tolerable |
| **LSH** | Hash buckets | O(1) lookup | Compact | Approximate fingerprinting |

**HNSW (Hierarchical Navigable Small World)** is the dominant production choice:

```
Level 2:     A ─── B
             │     │
Level 1:  C─ A ─── B ─── D
          │  │     │     │
Level 0:  C─ A ─── B ─── D ─── E ─── F
```

Query routing: start at top, traverse down to nearest neighbor at each level, then local search at Layer 0.

**Tuning parameters:**
- `M` (max neighbors per node): Higher M = more accurate but slower indexing.
- `ef` (search width): Higher ef = more accurate but slower queries.
- `efConstruction`: Controls indexing time.

Most vector DBs (Qdrant, Weaviate, Chroma) expose these knobs. A typical production setting: `M=16, ef=200, efConstruction=500` balances recall (~99%) and latency (<100ms).

</details>

---

## Q8. How do you handle document updates and deletions in a Naive RAG system? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Updating or deleting documents in a vector store requires careful handling:

| Strategy | Approach | Pros | Cons |
|----------|----------|------|------|
| **Full re-index** | Rebuild entire index from scratch | Simple; no stale data | Downtime; expensive for large corpora |
| **Soft delete** | Mark documents deleted; filter at query time | Zero downtime | Bloats index; requires filtering logic |
| **Versioned IDs** | Assign version tags (doc_v1, doc_v2); delete old versions | Rollback-safe; atomic | Complex ID management |
| **Lazy deletion** | Mark for deletion; compact during maintenance window | Balances simplicity & cost | Stale results until compact |
| **Hybrid: Update w/ new embed** | Delete old ID, insert new embedding with same semantic ID | Minimal downtime | Requires atomic operations |

**Best practice for production:**

1. Use soft delete with a `deleted_at` timestamp in metadata.
2. Filter deletions in query-time post-processing (cheap).
3. Periodically compact the index (off-peak) to reclaim space.
4. For critical documents, maintain a document version table (DB) separate from the vector index.

```python
# Soft-delete example with Chroma
vectorstore.delete(ids=["doc_123"])  # Soft marks as deleted

# Query-time filtering
results = vectorstore.similarity_search(query, k=10)
filtered = [r for r in results if not r.metadata.get("deleted")]
```

</details>

---

## Q9. What is semantic caching and how does it reduce Naive RAG latency? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Semantic caching stores embedding vectors of previous queries and reuses them if new queries are semantically similar, avoiding re-embedding and re-retrieval.

```
[Naive: No cache]
Query → Embed → Vector DB search → Chunks → LLM → Answer
 └─ 50ms    └─ 200ms              └─ 100ms

[With semantic cache]
Query → Embed → Cache lookup (similarity > 0.95?) ──[HIT]──> Cached chunks → LLM
                    └─ 5ms (fast)
                    └─[MISS]──> Vector DB search → Cache update
```

**Implementation:**

1. Embed the incoming query.
2. Search the cache (a smaller, in-memory vector DB or hash table).
3. If similarity > threshold (e.g., 0.95), reuse cached chunks.
4. Otherwise, perform normal retrieval and cache the result.

**Trade-offs:**

- Cache hit rate depends on query distribution (repetitive queries → high hit rate).
- Risk: cached results become stale if indexed documents change.
- Memory overhead for cache storage.

```python
from redis import Redis
import numpy as np

class SemanticCache:
    def __init__(self, threshold=0.95):
        self.cache = Redis(host='localhost')
        self.threshold = threshold
    
    def get_cached_result(self, query_embedding):
        """Check cache for semantically similar query."""
        cached_queries = self.cache.hgetall("query_embeddings")
        for cached_id, cached_vec in cached_queries.items():
            similarity = np.dot(query_embedding, np.frombuffer(cached_vec, dtype=np.float32))
            if similarity > self.threshold:
                return self.cache.get(f"result:{cached_id}")
        return None
    
    def cache_result(self, query_embedding, chunks, query_id):
        """Store query and retrieved chunks."""
        self.cache.hset("query_embeddings", query_id, query_embedding.tobytes())
        self.cache.set(f"result:{query_id}", json.dumps(chunks), ex=86400)  # 24h TTL
```

**Production systems (e.g., Anthropic's prompt caching)** use token-based caching at the LLM layer instead, which is more general but less semantic.

</details>

---

## Q10. Build an end-to-end Naive RAG system in under 50 lines of Python `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Here is a complete, runnable Naive RAG pipeline using LangChain, Chroma, and OpenAI:

```python
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Load and chunk documents
loader = TextLoader("document.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 2. Create embeddings and store in vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    chunks, 
    embeddings,
    persist_directory="./chroma_db"
)

# 3. Build RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 4. Query
response = qa_chain({"query": "What is the main topic?"})
print(response["result"])
for doc in response["source_documents"]:
    print(f"Source: {doc.metadata}")
```

**Key points:**
- `RecursiveCharacterTextSplitter` handles heterogeneous documents (code, prose, tables).
- `OpenAIEmbeddings` uses the latest text-embedding-3-small (cheaper, better than ada-002).
- `RetrievalQA` wraps the retriever + LLM; `chain_type="stuff"` concatenates all chunks into one prompt.
- Persist the vector store to disk for reuse without re-embedding.

To extend this to production: add reranking (Q2), query rewriting (Q2), or a custom retriever with filtering (Q3).

</details>
