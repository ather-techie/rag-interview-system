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

```
User Query
    │
    ▼
Pre-retrieval: Query rewriting / HyDE / expansion
    │
    ▼
Retrieval: Dense search + Sparse (BM25) → Hybrid merge (RRF)
    │
    ▼
Post-retrieval: Cross-encoder reranking → Context compression
    │
    ▼
LLM → Answer
```

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

```python
# HyDE: generate a hypothetical answer, embed it, then retrieve
hypo_doc = llm.invoke(f"Write a passage that answers: {query}")
embedding = embed_model.embed_query(hypo_doc)
results = vectorstore.similarity_search_by_vector(embedding, k=5)
```

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

---

## Q6. What is multi-query retrieval and how does it improve recall? `[Intermediate]`

**Answer:**

Multi-query retrieval generates multiple reformulations of the original query and retrieves chunks for each, then deduplicates and merges results. This combats the single-query vocabulary problem.

**Process:**

1. **Query generator** — Prompt an LLM to generate N paraphrases or related queries.
   - Original: "How do I optimize SQL queries?"
   - Paraphrases: "SQL performance tuning", "Query execution plans", "Database indexes"

2. **Parallel retrieval** — Retrieve top-k chunks for each query variant.

3. **Deduplication & merge** — Pool all results, remove duplicates by document ID, rerank.

**Benefits:**
- Higher recall: captures documents matching alternative phrasings.
- Minimal latency overhead: parallel retrieval is fast.

**Trade-off:**
- N LLM calls for generation (typically N=3-5).
- Index bloat: potentially retrieve more redundant chunks.

```python
from langchain.llms import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import Chroma

llm = ChatOpenAI(model="gpt-4o-mini")
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm,
    prompt=MULTI_QUERY_PROMPT,  # Custom template for rephrasing
)

# Retrieves with 3 query variants, dedupes, returns union
results = retriever.get_relevant_documents(query)
```

---

## Q7. How does contextual compression work and when should you use it? `[Intermediate]`

**Answer:**

Contextual compression filters retrieved chunks to extract only the relevant passages, reducing token usage and lost-in-the-middle effects.

```
[Without compression]
Retrieved chunk (512 tokens):
"Company was founded in 1995...
 [100 tokens of irrelevant background]
 Revenue grew 25% YoY...
 [100 tokens of irrelevant details]
 CEO is Jane Doe..."

[With compression]
→ "Revenue grew 25% YoY. CEO is Jane Doe."  (45 tokens)
```

**Techniques:**

1. **Extractive** — LLM selects and extracts relevant sentences.
2. **Abstractive** — LLM summarizes relevant portions.
3. **Query-aware** — Use the query to guide which parts are relevant.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
compressor = LLMChainExtractor.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    prompt=EXTRACTION_PROMPT  # Guides what to extract
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Retrieves top-10, compresses to relevant snippets
results = compression_retriever.get_relevant_documents(query)
```

**When to use:**
- When k > 5 and you want to reduce token bloat.
- When the LLM tends to get distracted by irrelevant context.
- Not recommended if you need citations to exact chunks.

---

## Q8. What is step-back prompting and how does it help RAG for complex questions? `[Advanced]`

**Answer:**

Step-back prompting is a two-phase technique where the LLM first abstracts the query to a higher-level concept, retrieves using that concept, then answers the original query with the retrieved context.

**Phase 1: Abstract the query**
- User: "How does the photosynthetic process enable plants to convert CO₂?"
- Step-back: "What is photosynthesis and its role in plant biology?"
→ Retrieve: general photosynthesis documents

**Phase 2: Retrieve using abstraction**
→ Use both original and abstracted query for retrieval.
→ Answer the original question with enriched context.

```
User Query (specific)
    │
    ├──[Retrieve]──> Top-k chunks (specific)
    │
    └──[LLM: Step back]──> Abstract query (general)
        │
        └──[Retrieve]──> Top-k chunks (general, foundational)
            │
            └──[LLM: Answer]──> Synthesize with both sets
```

```python
# Simplified step-back prompting example
abstract_prompt = """Given a question, infer the underlying topic or concept.
Question: {question}
Topic:"""

# Phase 1: Retrieve with original query
original_results = retriever.get_relevant_documents(question)

# Phase 2: Generate abstract query
abstract_query = llm.invoke(abstract_prompt.format(question=question))

# Phase 3: Retrieve with abstract query
abstract_results = retriever.get_relevant_documents(abstract_query)

# Phase 4: Merge and answer
merged_context = original_results + abstract_results
answer = llm.invoke(f"Answer: {question}\n\nContext: {merged_context}")
```

---

## Q9. How do you measure the contribution of each Advanced RAG component via ablation? `[Advanced]`

**Answer:**

Ablation studies systematically remove components to measure their individual contribution to end-to-end metrics.

| Component | Context Precision | Context Recall | RAGAS Faithfulness | Latency (ms) |
|-----------|-------------------|----------------|--------------------|--------------|
| Baseline (Naive RAG) | 0.72 | 0.58 | 0.68 | 150 |
| + HyDE | 0.74 | 0.65 | 0.70 | +120 |
| + Hybrid search | 0.81 | 0.72 | 0.75 | +50 |
| + Reranking | 0.88 | 0.74 | 0.82 | +200 |
| + Context compression | 0.87 | 0.71 | 0.80 | +80 |
| Full Advanced RAG | 0.89 | 0.76 | 0.84 | +680 |

**Interpretation:**
- Hybrid search has the largest gain in precision (+9 pp) with minimal latency cost.
- Reranking improves faithfulness most (+7 pp) but adds 200ms.
- Full system achieves +17pp precision but 4.5x latency.

**Best practice:**
1. Measure each component independently and together (order matters).
2. Use a fixed test set of 100-500 realistic queries.
3. Compute statistical significance (e.g., bootstrap confidence intervals).
4. Plot Pareto frontier: accuracy vs. latency to guide production choices.

---

## Q10. Design a production Advanced RAG pipeline with all enhancements enabled `[Advanced]`

**Answer:**

Here is a complete end-to-end Advanced RAG architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                              │
│                         │                                   │
│                    PREPROCESSING                            │
│         (spell-check, intent classification)               │
│                         │                                   │
│                PRE-RETRIEVAL ENHANCEMENTS                   │
│   ┌──────────────────────┬──────────────────────┐          │
│   │                      │                      │          │
│   ▼                      ▼                      ▼          │
│ Original Query      HyDE Generation      Multi-Query       │
│ (paraphrase)        (hypothetical doc)    (3 variants)    │
│                                                │          │
│   └──────────────────────┬──────────────────────┘          │
│                         │                                   │
│                    RETRIEVAL STAGE                          │
│   ┌──────────────────────┬──────────────────────┐          │
│   │                      │                      │          │
│   ▼                      ▼                      ▼          │
│ Dense (Vector)      Sparse (BM25)       Keyword Filter    │
│ Top-20             Top-20               Top-10            │
│   │                      │                      │          │
│   └──────────────────────┬──────────────────────┘          │
│         Reciprocal Rank Fusion (RRF) → 20 unique chunks   │
│                         │                                   │
│              POST-RETRIEVAL ENHANCEMENTS                    │
│   ┌──────────────────────┬──────────────────────┐          │
│   │                      │                      │          │
│   ▼                      ▼                      ▼          │
│ Cross-Encoder      Contextual           Reordering        │
│ Reranking          Compression          (bookend)          │
│ → Top-5            (extract saliency)   (place top at      │
│                                         start/end)        │
│                         │                                   │
│              FINAL GENERATION STAGE                         │
│ ┌──────────────────────────────────────────────────────┐  │
│ │ LLM (gpt-4o): Synthesize top-5 chunks + citations   │  │
│ └──────────────────────────────────────────────────────┘  │
│                         │                                   │
│                   FINAL ANSWER + SOURCES                    │
└─────────────────────────────────────────────────────────────┘
```

**Implementation skeleton:**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.bm25 import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and chunk
docs = TextLoader("document.txt").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50).split_documents(docs)

# Set up retrievers
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
dense_retriever = Chroma.from_documents(chunks, embeddings).as_retriever(search_kwargs={"k": 20})
sparse_retriever = BM25Retriever.from_documents(chunks)

# Hybrid search with RRF
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]  # Favor dense for semantics
)

# Multi-query wrapper
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=ensemble_retriever,
    llm=ChatOpenAI(model="gpt-4o-mini")
)

# Contextual compression
compressor = LLMChainExtractor.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini")
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_query_retriever
)

# Final QA chain with reranking
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True
)

# Query
response = qa({"query": "Your complex question here?"})
print(f"Answer: {response['result']}")
for doc in response["source_documents"]:
    print(f"Source: {doc.metadata}")
```

**Production tuning:**
- Measure latency per stage (HyDE: 120ms, hybrid search: 50ms, reranking: 200ms, compression: 80ms, generation: 500ms).
- Dynamically disable HyDE for simple queries (< 3 words); use hybrid search for all queries.
- Cache results for common questions (semantic caching, Q9 of section 01).
- Monitor reranking recall; adjust top-k if cross-encoder is dropping relevant chunks.
