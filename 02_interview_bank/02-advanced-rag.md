# 02 — Advanced RAG

> Improves on Naive RAG with pre-retrieval, retrieval, and post-retrieval enhancements.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
Query
  │
  ▼
Query Rewriter / Expander (HyDE, multi-query)
  │
  ▼
Hybrid Retriever
  ├── Sparse (BM25) ──┐
  └── Dense (Vector) ─┴──► Reciprocal Rank Fusion (RRF)
  │
  ▼
Cross-Encoder Reranker (top-100 → top-5)
  │
  ▼
Generator LLM → Answer
```

### Key Components

| Component | Responsibility |
|---|---|
| Query Rewriter / Expander | Reformulates or generates hypothetical answers (HyDE) to bridge the query-document vocabulary gap |
| Sparse Retriever (BM25) | Keyword-based retrieval; strong for exact terms, names, and codes |
| Dense Retriever | Embedding-based semantic search; strong for paraphrased/conceptual matches |
| Fusion Step | Merges sparse + dense result lists (e.g., Reciprocal Rank Fusion) into one ranked list |
| Cross-Encoder Reranker | Jointly scores (query, doc) pairs to reorder top candidates by true relevance |
| Generator | Synthesizes the final answer from the reranked, often compressed, context |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Orchestration | LangChain, LlamaIndex |
| Sparse Retrieval (BM25) | Elasticsearch, OpenSearch |
| Dense Retrieval | Pinecone, Weaviate, Qdrant |
| Reranking | Cohere Rerank, `bge-reranker`, `cross-encoder/ms-marco-MiniLM` |

---

## Q1. What distinguishes Advanced RAG from Naive RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

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

</details>

---

## Q2. What is HyDE and when is it useful? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

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

</details>

---

## Q3. How does hybrid search improve retrieval quality? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Hybrid search combines **dense (vector) retrieval** with **sparse (BM25/keyword) retrieval**:

- **Dense retrieval** excels at semantic similarity — good for paraphrased or conceptually related queries.
- **Sparse retrieval** excels at exact keyword matching — good for named entities, product codes, or technical terms.

Results from both are merged using **Reciprocal Rank Fusion (RRF)** or a learned ranker. This outperforms either method alone, especially on out-of-domain or tail queries.

**Tools:** Elasticsearch, OpenSearch, Weaviate, and Qdrant all support hybrid search natively.

</details>

---

## Q4. What is a cross-encoder reranker and how does it differ from a bi-encoder? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| | Bi-encoder | Cross-encoder |
|---|---|---|
| **How it works** | Query and doc embedded separately; similarity via dot product | Query and doc fed together into one model |
| **Speed** | Fast — embeddings pre-computed | Slow — must run per (query, doc) pair |
| **Accuracy** | Good for retrieval at scale | Higher accuracy for reranking top-k |
| **Use** | First-stage retrieval | Second-stage reranking |

The typical pipeline: bi-encoder retrieves top-100 → cross-encoder reranks to top-5. Models like `ms-marco-MiniLM` or Cohere Rerank are common cross-encoders.

</details>

---

## Q5. What is the "lost-in-the-middle" problem and how do you address it? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Research shows that LLMs perform worse when relevant information appears in the **middle** of a long context — they are better at using information at the beginning or end of the prompt.

**Mitigations:**

1. **Reorder retrieved chunks** — Place the most relevant chunks at the start or end, not the middle.
2. **Context compression** — Use a smaller model to summarize or filter each retrieved chunk before passing it to the LLM.
3. **Reduce k** — Pass fewer, higher-quality chunks rather than many lower-quality ones.
4. **LongLLMLingua / Selective Context** — Prompt compression tools that remove low-information tokens from retrieved context.

Addressing this is critical for production systems where k > 5.

</details>

---

## Q6. What is multi-query retrieval and how does it improve recall? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

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
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma

llm = ChatOpenAI(model="gpt-4o-mini")
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm,
    prompt=MULTI_QUERY_PROMPT,  # Custom template for rephrasing
)

# Retrieves with 3 query variants, dedupes, returns union
results = retriever.get_relevant_documents(query)
```

</details>

---

## Q7. How does contextual compression work and when should you use it? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

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

</details>

---

## Q8. What is step-back prompting and how does it help RAG for complex questions? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

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

</details>

---

## Q9. How do you measure the contribution of each Advanced RAG component via ablation? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

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

</details>

---

## Q10. Design a production Advanced RAG pipeline with all enhancements enabled `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

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
from langchain_community.vectorstores import Chroma
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

</details>

---

## Q11. How do you quantify and reduce the added cost of HyDE generation and cross-encoder reranking in an Advanced RAG pipeline? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Cost breakdown per component:**

| Component | Latency | Cost/Query | Frequency |
|-----------|---------|-----------|-----------|
| **HyDE generation** | 100–200ms | $0.002–0.004 | Every query |
| **Hybrid search** | 50–100ms | $0.001 | Every query |
| **Cross-encoder reranking** | 150–300ms | $0.005–0.01 | Every query |
| **LLM generation** | 500–2000ms | $0.01–0.05 | Every query |
| **Total Advanced RAG** | ~800–2600ms | $0.02–0.07 | — |

**HyDE cost optimization:**

HyDE generates hypothetical documents before retrieval, which requires a separate LLM call:

1. **Use smaller models** — Replace GPT-4 with GPT-3.5-turbo or Llama 2 7B for HyDE generation.
   - GPT-4: $0.015/1K tokens → $0.004/query (assuming 260 tokens per HyDE output).
   - GPT-3.5-turbo: $0.0005/1K tokens → $0.00013/query (60x cheaper).
   - Trade-off: ~2-5% drop in retrieval quality.

2. **Conditional HyDE** — Only run HyDE for complex queries (multi-sentence, ambiguous).
   - Simple queries (1–3 words) → skip HyDE, use direct embedding.
   - Savings: skip 30% of queries → 30% cost reduction.

3. **Cached HyDE** — Cache hypothetical documents for common query patterns.
   - E.g., "Tell me about product X" → always generates similar hypotheticals.
   - Reuse cache hits without regeneration.

**Cross-encoder reranking cost optimization:**

Reranking scores retrieved documents using a fine-tuned cross-encoder (e.g., ms-marco-MiniLM-L-6-v2).

1. **Lightweight rerankers** — Use smaller cross-encoders with less latency:
   - ms-marco-MiniLM-L-6-v2: ~50ms for 100 documents, very cheap.
   - Cohere Rerank v2: ~200ms, higher quality but higher cost.
   - Jina Reranker: ~100ms, middle ground.

2. **Rerank fewer documents** — Instead of reranking top-100 results, rerank top-20:
   - Retrieval: top-100 by BM25 + dense similarity.
   - Reranking: top-20 → top-5 by cross-encoder.
   - Latency reduction: 150–300ms → 50–100ms.
   - Quality: minimal impact (top docs are already high quality).

3. **Conditional reranking** — Skip reranking if initial retrieval is confident:
   ```python
   def should_rerank(initial_scores):
       # If top document similarity > 0.95, retrieval is confident
       # Skip reranking for this query
       if initial_scores[0] > 0.95:
           return False
       return True
   ```
   - Savings: skip ~20% of queries.

4. **Batch reranking** — Rerank multiple queries in a single batch to amortize latency.
   - Single query: 200ms.
   - Batch of 10: 200ms total (20ms per query).
   - For real-time serving, batch across concurrent requests.

**Example optimization:**

Baseline Advanced RAG:
- HyDE: GPT-4, every query → $0.004/query.
- Reranking: Cohere, top-100 → $0.007/query.
- Total additional cost: $0.011/query (vs. Naive RAG's $0.001/query).

Optimized Advanced RAG:
- HyDE: GPT-3.5-turbo, conditional (skip 30%) → $0.00009/query.
- Reranking: MiniLM, top-20, conditional (skip 20%) → $0.0003/query.
- Total additional cost: $0.00039/query (97% reduction).

**Quality impact of optimizations:**

| Configuration | Cost/Query | F1 (avg) | Latency (p95) |
|---|---|---|---|
| Baseline (HyDE + full reranking) | $0.011 | 0.85 | 2000ms |
| Optimized (cheap HyDE + lightweight reranking) | $0.001 | 0.84 | 1200ms |
| Hybrid (conditional on query complexity) | $0.004 | 0.847 | 1500ms |

The hybrid approach balances cost and quality: conditional HyDE and selective reranking lose only ~1% F1 while cutting costs by 65%.

**Monitoring and tuning:**

Track:
- HyDE cost per query (should be <$0.001 after optimization).
- Reranking cost per query (target: <$0.002).
- Quality metrics (F1, NDCG) per optimization tier.
- Latency per component (identify bottlenecks).

</details>

---

## Q12. How can adversaries craft queries that systematically evade hybrid (dense + sparse) retrievers, and how do you harden the retrieval layer? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Evasion attacks on hybrid retrievers:**

Hybrid retrieval combines dense embeddings (semantic similarity) and sparse retrieval (BM25, keyword matching). An attacker can craft queries or documents to fool one or both components.

**Attack 1: Semantic drift attack**

Attacker queries the system with a benign query, but crafts it to retrieve a specific malicious document by embedding near it:

```
Benign query: "How do I improve my running speed?"
Malicious query (attacked): "How do I improve my [RUN_FAST_TRICK] speed?"

The token [RUN_FAST_TRICK] is semantically meaningless but, when embedded,
lands near a malicious document about doping.
```

In hybrid retrieval:
- Dense embedding → malicious document is top-3 (adversarial embedding).
- BM25 → benign documents rank higher (keywords match).
- Result: malicious doc mixed into results.

**Attack 2: Keyword injection**

Attacker embeds many keywords from legitimate documents in a malicious document:

```
Malicious document:
"Running shoes, training tips, marathon, nutrition, recovery, [HIDDEN_MALWARE_URL]"

Sparse retrieval (BM25) → ranks high (many keyword matches).
Dense retrieval → ranks lower (semantic mismatch).
Hybrid score → medium rank, passes through.
```

**Attack 3: Frequency manipulation**

Attacker crafts queries with rare terms that appear disproportionately in malicious documents:

```
Legitimate corpus: 1M documents, avg keyword frequency = 5.
Malicious corpus: 100 documents, injected rare keywords like "ephemeris_correction".

User query: "What is ephemeris correction?"
BM25 → malicious docs rank higher (keyword is rare, high IDF score).
Hybrid → retrieves malicious docs.
```

**Defences:**

**1. Ensemble diversity check:**

Ensure that dense and sparse retrievers agree on top-k documents:

```python
def hybrid_retrieve_with_diversity_check(query, k=5):
    # Dense retrieval
    dense_results = dense_retriever.search(query, k=k*2)
    dense_top_ids = set(doc.id for doc, _ in dense_results[:k])
    
    # Sparse retrieval
    sparse_results = sparse_retriever.search(query, k=k*2)
    sparse_top_ids = set(doc.id for doc, _ in sparse_results[:k])
    
    # Intersection: documents both retrievers agree on
    agreement = dense_top_ids & sparse_top_ids
    
    # If agreement < 50%, flag as suspicious and lower confidence
    if len(agreement) / k < 0.5:
        log_suspicious_query(query, agreement)
    
    # Return documents both retrievers ranked high
    return [doc for doc in dense_results if doc.id in sparse_top_ids]
```

**2. Semantic coherence check:**

Verify that retrieved documents are semantically similar to each other (not scattered):

```python
def check_semantic_coherence(retrieved_docs):
    embeddings = [embed(doc.text) for doc in retrieved_docs]
    
    # Compute pairwise cosine similarity
    coherence_score = 0.0
    for i, j in itertools.combinations(range(len(embeddings)), 2):
        similarity = cosine_similarity(embeddings[i], embeddings[j])
        coherence_score += similarity
    
    coherence_score /= len(list(itertools.combinations(range(len(embeddings)), 2)))
    
    # If coherence is very low, documents are scattered → suspicious
    if coherence_score < 0.5:
        log_suspicious_retrieval(retrieved_docs)
        # Option 1: Rerank with stricter threshold
        # Option 2: Escalate to user with confidence flag
```

**3. Adversarial robustness in embedding models:**

Use embedding models fine-tuned for robustness:

- Standard: text-embedding-3-small (vulnerable to adversarial examples).
- Robust: fine-tune embeddings on adversarial pairs to reduce vulnerability.

```python
# Fine-tuning a robust embedding model
adversarial_pairs = [
    ("normal query", "adversarial query with injected tokens"),
    ("legitimate document", "document with poisoned keywords")
]

robust_model = fine_tune_embeddings(base_model, adversarial_pairs)

# Embeddings from robust_model are less susceptible to evasion
dense_results = robust_model.search(query)
```

**4. Keyword sanitization:**

Remove or downweight suspicious keywords:

```python
def sanitize_query(query):
    # Identify rare, potentially malicious keywords
    rare_keywords = [term for term in query.split() if idf(term) > threshold]
    
    if len(rare_keywords) > k:
        # Too many rare keywords → suspicious query
        log_suspicious_query(query)
        # Downweight rare keywords in BM25 scoring
        query_boosted = " ".join(
            term if idf(term) < threshold else f"~{term}"
            for term in query.split()
        )
        return query_boosted
    
    return query
```

**5. Threshold and confidence monitoring:**

Track retrieval scores and flag when sparse and dense retrievers significantly disagree:

```python
def get_hybrid_score(doc, query, dense_weight=0.5, sparse_weight=0.5):
    dense_score = dense_similarity(query, doc)
    sparse_score = bm25_score(query, doc)
    
    # Detect disagreement
    if abs(dense_score - sparse_score) > 0.4:
        # High disagreement → possibly adversarial
        confidence = 0.7  # Lower confidence
    else:
        confidence = 0.95
    
    hybrid_score = dense_weight * dense_score + sparse_weight * sparse_score
    return hybrid_score, confidence
```

**6. Periodic adversarial evaluation:**

Regularly test the retriever against known adversarial queries:

```python
adversarial_test_cases = [
    ("benign query", "crafted to retrieve malicious doc"),
    ("normal query with rare keywords", "evasion attempt"),
    ...
]

for benign, adversarial in adversarial_test_cases:
    benign_results = retrieve(benign)
    adversarial_results = retrieve(adversarial)
    
    # Measure if adversarial version retrieves malicious docs
    attack_success_rate = (
        count_malicious(adversarial_results) / 
        count_malicious(benign_results)
    )
    
    if attack_success_rate > 10%:
        # System is vulnerable; trigger retraining or threshold adjustment
        alert_security_team()
```

**Defence-in-depth:**

Combine multiple layers:
1. Ensemble checks (dense + sparse agreement).
2. Semantic coherence verification.
3. Keyword sanitization.
4. Confidence-based ranking (flag low-confidence retrievals).
5. Continuous adversarial evaluation.

An attacker must craft queries that fool multiple retrieval modalities, which is much harder than fooling a single dense or sparse retriever.

</details>

---

## Q13. What is Reciprocal Rank Fusion, and why is it used to merge sparse and dense result lists instead of averaging scores? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Reciprocal Rank Fusion (RRF) merges two ranked lists by scoring each document as `1 / (k + rank)` in each list it appears in (a constant like `k=60` is typical) and summing across lists — a document ranked #1 by both retrievers scores highest, and a document that only one retriever found still contributes something, rather than being penalized for the other retriever's silence.

The reason RRF is preferred over simply averaging raw similarity scores is that BM25 scores and cosine-similarity scores live on **incompatible scales** — BM25 scores are unbounded and corpus-dependent (they depend on term frequency statistics across the whole index), while cosine similarity is bounded between -1 and 1 and depends on the embedding model's geometry. Averaging these two directly would let whichever score happens to have a larger numeric range dominate the merge, regardless of which retriever is actually more informative for a given query. RRF sidesteps this entirely by discarding the raw scores and working only with rank position, which is directly comparable across any two ranking methods no matter how differently their underlying scores are distributed.

</details>

---

## Q14. What is the research and practical origin of the query-rewriting and reranking techniques that define Advanced RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

HyDE (Q2) comes from Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (arXiv:2212.10496, 2022), which showed that embedding an LLM-generated hypothetical answer closes the query-document vocabulary gap better than embedding the raw query, without needing any labeled relevance data to train a better retriever. Cross-encoder reranking (Q4) builds on the broader cross-encoder-vs-bi-encoder distinction established in sentence-embedding and passage-ranking research (the MS MARCO passage ranking benchmark drove much of the reranker tooling still in use, like `ms-marco-MiniLM`), applying joint query-document attention specifically as a second-stage precision filter over a cheap first-stage retriever's candidates.

Neither technique originated as part of a single "Advanced RAG" paper — the term describes a practitioner-assembled combination of separately-published pre- and post-retrieval techniques (query rewriting, hybrid search, reranking, compression) layered onto Naive RAG's pipeline, in the same way this bank documents several other "combination" architectures (e.g., LongRAG + Self-Route, #45) built from independently-discovered pieces rather than one unified research contribution.

</details>

---

## Q15. How does Advanced RAG compare to Modular RAG (#03)? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Advanced RAG is a **fixed pipeline** with a specific, named set of enhancements bolted onto Naive RAG in a specific order (query rewriting → hybrid retrieval → reranking → compression, per this file's architecture diagram) — every query flows through the same stages in the same sequence. Modular RAG (#03) generalizes this into a **library of swappable, composable components** (different retrievers, different rerankers, different routing logic) that can be assembled differently per use case, or even per query, rather than committing to one fixed enhancement sequence for the whole system.

The practical relationship: Advanced RAG is one particular, common configuration of Modular RAG's component library — a team building a Modular RAG system very often arrives at something that looks exactly like this file's architecture diagram, but Modular RAG's framing makes explicit that the specific stages, their order, and even whether a given stage runs at all, are configuration choices rather than a fixed architecture. Advanced RAG is the right mental model when a single enhancement pipeline suffices for your whole query distribution; Modular RAG is the right mental model once different query types genuinely need different pipelines.

</details>

---

## Q16. Why does dense retrieval alone struggle with exact-match queries, and how does hybrid search fix this? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A dense embedding model is trained to capture semantic *meaning*, which means it can represent "increase" and "growth" as near-identical vectors — exactly the generalization that makes it good at paraphrase matching also makes it comparatively weak at treating an exact product code, part number, or rare proper noun as something that must match precisely. Two different product SKUs sharing similar surrounding context can end up with very similar embeddings even though a user searching for one specific SKU needs the *other* one filtered out entirely, not ranked as "close enough."

BM25 (sparse/keyword) retrieval has the opposite bias: it scores documents by term-frequency statistics, so an exact rare-term match scores very highly regardless of surrounding semantic context, and it has no notion of "these two terms mean roughly the same thing" to blur its ranking the way dense retrieval can. Hybrid search (Q3) combines both specifically because their failure modes are close to complementary — dense retrieval catches the paraphrased, conceptually-related documents BM25 would miss, and BM25 catches the exact-identifier documents dense retrieval might blur together with semantically similar but factually wrong alternatives.

</details>

---

## Q17. What are the key tuning knobs for hybrid search, and how do you choose them? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Knob | Effect | Starting point |
|---|---|---|
| RRF constant `k` (Q13) | Controls how quickly a rank's contribution decays — smaller `k` weights top ranks more heavily; larger `k` flattens the contribution across a wider rank range | 60 is the standard default from the original RRF literature |
| Dense/sparse weight blend (if not using pure RRF) | Determines which retriever dominates when they disagree | Start balanced (0.5/0.5) and adjust based on evaluation split by query type (exact-match-heavy vs. paraphrase-heavy) |
| Top-k per retriever before fusion | More candidates per retriever improve the odds the right document survives to the fusion step, at higher compute cost | 20-100 per retriever is typical before fusing down to a much smaller final top-k |
| Reranker top-k (Q4) | How many fused candidates get the expensive cross-encoder treatment | 20-100 in, top-5 out is a common shape, balancing reranking's accuracy gain against its per-candidate cost |

The dense/sparse weight blend is the knob most worth tuning per-corpus rather than accepting a universal default: a corpus dominated by technical identifiers and codes (Q16) benefits from weighting sparse retrieval more heavily, while a corpus of narrative prose with few exact-match query patterns benefits from weighting dense retrieval more heavily — the right blend is an empirical question answerable via the segmented evaluation approach in Q18, not a fixed rule of thumb.

</details>

---

## Q18. How do you build your own ablation harness to measure which Advanced RAG enhancements actually help your corpus? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Q9's illustrative ablation table shows the *shape* of the exercise; running your own requires a fixed, representative query set and a harness that can toggle each component independently:

```python
def run_ablation(query_set: list[dict], configs: dict[str, callable]) -> dict:
    """configs: {"name": pipeline_fn} — each pipeline_fn implements one
    configuration (baseline, +HyDE, +hybrid, +reranking, full stack)."""
    results = {}
    for name, pipeline_fn in configs.items():
        precisions, recalls, latencies = [], [], []
        for item in query_set:
            start = time.time()
            retrieved = pipeline_fn(item["query"])
            latencies.append(time.time() - start)
            precisions.append(precision_at_k(retrieved, item["relevant_ids"]))
            recalls.append(recall_at_k(retrieved, item["relevant_ids"]))
        results[name] = {
            "precision": mean(precisions), "recall": mean(recalls),
            "p95_latency": percentile(latencies, 95),
        }
    return results
```

Run configurations cumulatively (baseline, baseline+HyDE, baseline+HyDE+hybrid, ...) rather than each enhancement in isolation, since Q9's table shows enhancements interact — hybrid search's precision gain, for instance, is measured *on top of* whatever HyDE already contributed, not independently, and a component that looks marginal in isolation can still be worth keeping if it meaningfully improves the full stack's Pareto frontier (accuracy vs. latency, Q9's closing recommendation) at an acceptable cost. Re-run this ablation whenever the corpus, embedding model, or query distribution shifts meaningfully, since Q9's specific numbers are corpus- and query-distribution-dependent and don't transfer to a different deployment unchanged.

</details>

---

## Q19. What is the characteristic failure mode of mis-applying reranking, in either direction? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Over-applying reranking** (running the expensive cross-encoder pass even when the first-stage retrieval was already precise) wastes the 150-300ms and per-candidate cost the reranking stage adds (Q11) for no accuracy benefit — Q11's own conditional-reranking optimization (skip reranking when `initial_scores[0] > 0.95`) exists precisely to detect and avoid this case. **Under-applying reranking** (skipping it, or applying it too shallowly, on queries where first-stage retrieval is genuinely ambiguous) leaves low-precision, topically-similar-but-wrong candidates in the final context with no second-stage filter to catch them — exactly the scenario reranking's accuracy gain in Q9's ablation table (+7pp faithfulness) is measuring.

**Detection:** segment production queries by first-stage retrieval confidence (top result's similarity score, or the gap between the top result and the next few) and compare downstream answer quality with reranking on vs. off within each confidence bucket — high-confidence queries should show little to no reranking benefit (confirming it's safe to skip there for cost savings, Q11), while low-confidence queries should show reranking's benefit concentrated exactly there. A system applying a uniform "always rerank" or "never rerank" policy regardless of first-stage confidence is very likely paying for reranking where it doesn't help, missing it where it matters most, or both simultaneously across different query segments.

</details>

---

## Q20. What are the limitations of Advanced RAG as a category, and when do you need Modular or Agentic RAG instead? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Advanced RAG's enhancements (Q1) are all still **single-pass**: one query rewrite, one retrieval fusion, one rerank, one generation — there is no mechanism for the system to notice mid-pipeline that the retrieved evidence is insufficient and try again, decompose a genuinely multi-hop question into sub-questions, or adapt its strategy based on query complexity. A query that needs several rounds of retrieval-then-reasoning (a multi-hop question, an aggregate synthesis across many documents) will run through Advanced RAG's fixed single-pass pipeline exactly once and produce whatever that one pass yields, regardless of whether it was enough.

**When Advanced RAG is no longer sufficient:** the practical signal is the same evaluation discipline used throughout this bank (Q18) — if a meaningful share of your query distribution needs multiple retrieval rounds to answer correctly (multi-hop questions, per Iterative Multi-Hop RAG #19), needs the system to decide *whether* to retrieve at all rather than always doing so (Adaptive RAG, #11), or needs different retrieval/reranking strategies for structurally different query types (Modular RAG, #03, or full agentic orchestration, #04), a single fixed pipeline — however well-tuned its individual stages are — has a hard ceiling that no amount of further stage-level tuning (better HyDE prompts, a stronger reranker) can push past, because the limitation is architectural (single-pass) rather than a matter of component quality.

</details>

---

## Real-World Applications

| Application | Domain | Why Advanced RAG Fits |
|---|---|---|
| Enterprise knowledge search (e.g., Notion AI, Confluence AI) | Enterprise productivity | HyDE + hybrid search surfaces relevant docs even when employees use informal language that doesn't match exact headings |
| Developer documentation copilot (e.g., Stripe Docs AI) | DevTools | Query rewriting handles abbreviations and jargon; reranking ensures the most precise API reference snippet surfaces first |
| Technical support tier-1 agent | IT / SaaS | Multi-query retrieval captures edge cases across a large ticket history; context compression keeps latency under SLA |
| Legal document research assistant | Legal / Compliance | Hybrid BM25 + dense search handles both exact statute citations and semantic "what does this clause mean?" queries |
| Healthcare patient portal Q&A | Healthcare | Reranking and deduplication prevent similar-sounding but distinct clinical terms from colliding in the context window |
