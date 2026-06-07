# 02 — Retrieval Failure

> The retrieval system fails to surface relevant context for a given query, resulting in missing or out-of-scope chunks being returned to the LLM.

---

## Q1. What is retrieval failure in a RAG system, and why is it distinct from other failure modes? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Retrieval failure occurs when the retrieval stage of RAG fails to return relevant documents/chunks for a given query. Unlike hallucination (where context exists but the LLM misuses it), retrieval failure is a **supply-side problem**: the necessary information is not provided to the LLM at all.

This is a foundational failure because:

1. **Information cannot be synthesized without being present.** If retrieval returns nothing, the LLM has no grounding and must rely on its training data (guaranteed hallucination).
2. **It cascades all downstream risks.** Poor retrieval leads directly to:
   - Hallucinations (LLM fills the gap)
   - Low answer relevance (wrong chunks = wrong answer)
   - Poor user experience (missing information the system should have)
3. **It's often invisible.** Users may not know the information exists in the knowledge base; they only see "I don't know" or a hallucinated answer.

**Key distinction from other modes:**

| Failure Mode | Stage | Root Cause | Symptom |
|---|---|---|---|
| **Retrieval Failure** | Retrieval | Missing/wrong chunks | Query returns empty or irrelevant chunks |
| **Hallucination** | LLM Generation | Ignores context | Correct chunks present but answer is false |
| **Embedding Mismatch** | Embedding | Semantic gap | Similar intent but different vocabulary |
| **Reranker Failure** | Reranking | Wrong ranking | Right chunks present but buried in ranking |

</details>

---

## Q2. What are the observable symptoms of retrieval failure? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Retrieval failures manifest in several ways, detectable through monitoring and user feedback:

| Symptom | How It Appears | Detection Method |
|---------|---|---|
| **Empty result** | Retrieval returns 0 chunks for a query that should match data | Log inspection: `retrieved_count == 0` |
| **All irrelevant chunks** | Returns 5 chunks, none match the query intent | Manual QA audit, embedding similarity score < threshold |
| **Rank mismatch** | Relevant chunks exist in the DB but ranked at position 20+ (outside top-k) | Offline evaluation: check if ground truth is in corpus but not in top-5 |
| **Semantic gap** | Query uses term "neural network", docs use "deep learning" but no overlap | Embedding similarity between query and docs is low; manual inspection finds the "same" content with different wording |
| **Vocabulary mismatch** | Query: "What is a transformer?", Relevant doc titled "Attention Is All You Need" exists but uses that term only in the abstract | Token overlap is low; BM25 retrieval fails |
| **Scope mismatch** | Query asks about Product A, retrieval returns chunks only about Product B (even though both are in corpus) | Retrieved chunk metadata (product ID, section) doesn't match query intent |
| **Temporal mismatch** | Query about recent events, retrieval returns old chunks | Chunk timestamps are before query date; no recency filtering applied |

**User-observable symptoms:**

- "I know this information is in our docs, but the AI says it doesn't know"
- "The answer it gives is about a different product"
- "The answer is outdated"
- "It answers some variations of my question but not others"

</details>

---

## Q3. What are the root causes of retrieval failure? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Retrieval failures stem from several technical and architectural causes:

### 1. Semantic Gap (Vocabulary/Concept Mismatch)

The query and documents use different terminology for the same concept:

```
Query: "How do I authenticate users with JWT?"
Document A: "Token-based authentication using JSON Web Tokens..."
Document B: "Bearer tokens for API security..."

Embedding model computes:
  cos_sim(query_embedding, doc_a_embedding) = 0.72 ✓ retrievable
  
But if embedding model is domain-unaware:
  cos_sim(query_embedding, doc_b_embedding) = 0.45 ✗ may not rank high
```

**Why it happens:**
- General-purpose embedding models (e.g., `text-embedding-3-small`) train on broad web data, not domain jargon
- Synonymy: "machine learning" vs "artificial intelligence" vs "data science"
- Acronyms: "LLM" vs "large language model"

### 2. Out-of-Vocabulary / Coverage Gaps

The knowledge base doesn't contain information about a topic, or contains it in unexpected forms:

```
Query: "Pricing for Enterprise plan"

Knowledge base contains:
- "Our standard pricing: $10/user/month"
- "Contact sales for custom enterprise agreements"
- "No docs explicitly stating 'Enterprise plan pricing'"

Result: Retrieval returns irrelevant pricing docs or nothing
```

**Root causes:**
- Documentation incomplete or out of date
- Niche topics not covered
- Information exists but under a different section/title

### 3. Query Ambiguity

A query has multiple valid interpretations; retrieval optimizes for the wrong one:

```
Query: "How do I scale?"

Could mean:
  a) Scale the database (add shards, replicas)
  b) Scale the API servers (auto-scaling groups)
  c) Scale the model training (distributed training)

Retrieval returns (a), but user needed (b).
Without context, all interpretations are equally valid.
```

### 4. Chunking Strategy Mismatch

Chunks are split at boundaries that break semantic coherence:

```
Document:
"The BERT model introduced in 2018 by Devlin et al. achieved state-of-the-art
results on many NLP benchmarks. [CHUNK BOUNDARY] The key innovation was
bidirectional context. This enabled better understanding of word meaning."

Query: "What was the key innovation of BERT?"

If chunking splits after "state-of-the-art", the answer chunk is separated
from the question-relevant content. Retrieval misses it.
```

### 5. Embedding Model Limitations

The embedding model is poorly suited for the domain:

```python
# General-purpose embeddings may struggle with:
# - Technical jargon ("cache locality", "Byzantine fault tolerance")
# - Rare/new concepts ("retrieval-augmented generation" was new ~2020)
# - Multi-lingual queries in non-English-dominant domains
# - Asymmetric query-document length (long docs vs short queries)

# Example: Medical domain
query_embedding = embed("What are the contraindications for metformin?")
doc_embedding = embed("Patient is 45M with Type 2 diabetes...")

# Generic embedding model may not capture medical relevance
similarity = cosine(query_embedding, doc_embedding)  # Low despite relevance
```

### 6. Sparse/Dense Hybrid Mismatch

Relying solely on dense embeddings when keywords matter:

```
Query: "Configure the CORS policy"
Document: "To enable cross-origin requests, set the CORS headers..."

Dense similarity: Medium (both about web policies)
BM25 similarity: High (exact match on "CORS")

Hybrid would rank high; dense-only might rank medium/low.
```

### 7. Temporal Mismatch

Query requires recent information, but index contains old data:

```
Query: "Who is the CEO of Company X as of 2024?"
Document (from 2020): "CEO is John Doe"
Document (from 2024): "CEO is Jane Smith"  ← Should rank first

Without metadata filtering or recency weighting,
both might rank equally, or old data might win.
```

### 8. Metadata and Filtering Misconfiguration

Relevant chunks exist but are filtered out:

```python
# Example: Range filtering
query = "Pricing for startups"
metadata_filter = {"plan_type": "enterprise"}  # Wrong filter!

# Result: Startup pricing chunks are filtered out,
# retrieval returns only enterprise pricing
```

</details>

---

## Q4. How do you measure and detect retrieval failure in production? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Retrieval failure detection combines offline evaluation (using ground truth) and production monitoring:

### Offline Evaluation Metrics

| Metric | Definition | Interpretation | Best For |
|--------|-----------|---|---|
| **Recall@k** | % of ground-truth relevant chunks in top-k results | Recall@5=0.8 means 80% of correct answers are in top-5 | Measuring completeness of retrieval |
| **Mean Reciprocal Rank (MRR)** | Average position of first correct result (1/rank) | MRR=0.5 means correct result is at position 2 on average | Evaluating ranking quality |
| **Normalized Discounted Cumulative Gain (NDCG@k)** | Relevance-weighted ranking quality (0–1) | NDCG@5=0.7 means ranking quality is good but imperfect | Penalizes wrong order of relevant docs |
| **Mean Average Precision (MAP)** | Precision at each position, averaged | MAP=0.6 = decent precision across ranking | Overall ranking quality |

**Concrete example:**

```python
def compute_retrieval_metrics(query, retrieved_chunks, ground_truth_chunks):
    """Evaluate retrieval quality against known relevant docs."""
    
    # retrieved_chunks = [chunk_1, chunk_3, chunk_7, chunk_2]  (ranked by relevance)
    # ground_truth_chunks = [chunk_1, chunk_2, chunk_5]  (actually relevant)
    
    # Recall@k
    recall_at_5 = len(set(retrieved_chunks[:5]) & set(ground_truth_chunks)) / len(ground_truth_chunks)
    # = 2 / 3 = 0.67
    
    # MRR
    for rank, chunk in enumerate(retrieved_chunks, 1):
        if chunk in ground_truth_chunks:
            mrr = 1 / rank
            break
    # First match at rank 1, so MRR = 1.0
    
    # NDCG@5
    dcg = 0
    for rank, chunk in enumerate(retrieved_chunks[:5], 1):
        relevance = 1 if chunk in ground_truth_chunks else 0
        dcg += relevance / math.log2(rank + 1)
    
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(6, len(ground_truth_chunks) + 1)))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return {'recall@5': recall_at_5, 'mrr': mrr, 'ndcg@5': ndcg}
```

### Benchmark Datasets

**Standard datasets with ground truth:**

| Dataset | Domain | Size | Metric |
|---------|--------|------|--------|
| **MS MARCO** | Web search, passage ranking | 500k queries, 8.8M passages | Uses MRR@10, NDCG@10 |
| **Natural Questions** | Wikipedia QA | 300k QA pairs | Recall@10 |
| **SQuAD** | Reading comprehension | 100k QA pairs on 500 docs | Recall@1, MRR |
| **MTEB (Massive Text Embedding Benchmark)** | Multi-task, including retrieval | 12 datasets across domains | NDCG, MAP, MRR |

### Production Monitoring

**Method 1: Implicit Feedback from User Clicks**

```python
def detect_retrieval_failure_from_clicks(user_session):
    """If user clicks through many results before finding answer, retrieval was poor."""
    
    # Track: how many results did user skip before finding relevant one?
    if len(user_session['clicked_results']) > 3:
        # User had to look past 3+ results → retrieval ranked poorly
        flag_as_retrieval_issue(session_id)
    
    if user_session['time_to_satisfaction'] > 60:
        # Took >60 seconds to find answer → likely retrieval failure
        flag_as_retrieval_issue(session_id)
```

**Method 2: Explicit Feedback and Rating**

```python
def track_user_satisfaction(query, retrieved_chunks, user_rating):
    """
    After showing retrieved results, ask: "Was this helpful?"
    1-star = unhelpful, 5-star = perfect
    """
    
    if user_rating <= 2:
        # Low satisfaction likely due to retrieval failure
        log_retrieval_failure(query, retrieved_chunks)
```

**Method 3: Answer Quality vs Retrieval**

```python
def infer_retrieval_quality_from_answer(question, answer, retrieved_chunks):
    """If answer is bad but retrieval looks reasonable, issue is elsewhere."""
    
    # Compute answer faithfulness (is it supported by chunks?)
    faithfulness = nli_model.predict([[" ".join(retrieved_chunks), answer]])
    
    if faithfulness < 0.5:
        # Answer is not faithful to retrieved context
        # → This is likely a retrieval issue (wrong chunks retrieved)
        log_retrieval_failure(question, retrieved_chunks, reason="low_faithfulness")
```

**Method 4: Query-Chunk Similarity Distribution**

```python
def monitor_retrieval_confidence(query, retrieved_chunks):
    """Low similarity scores across all results → retrieval struggling."""
    
    similarities = [
        embedding_model.similarity(query, chunk)
        for chunk in retrieved_chunks
    ]
    
    mean_similarity = statistics.mean(similarities)
    
    if mean_similarity < 0.6:  # threshold
        # All retrieved chunks have low similarity to query
        log_retrieval_failure(query, reason="low_confidence", score=mean_similarity)
```

### Production Dashboard Metrics

Track over time:

```python
monitoring_metrics = {
    'avg_recall@5': 0.75,  # Should be > 0.7
    'avg_recall@10': 0.85,  # Should be > 0.8
    'mrr': 0.68,  # Should be > 0.6
    'pct_zero_results': 0.05,  # % of queries returning 0 chunks (should be ~0%)
    'pct_low_similarity': 0.08,  # % of queries with mean similarity < 0.6 (alert if > 0.1)
    'user_satisfaction_rating': 3.8,  # Average 1-5 rating (should be > 3.5)
    'avg_query_response_time': 250,  # ms (benchmark for your system)
}
```

</details>

---

## Q5. What is the semantic gap problem, and why does it cause retrieval failure? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The semantic gap is the mismatch between how a query is phrased and how information is stored in documents. Even when both refer to the same concept, embedding models may fail to bridge the gap.

### Classic Examples

**Example 1: Acronyms vs Full Names**

```
Query: "What is JWT authentication?"
Document: "Bearer tokens using JSON Web Tokens are a modern approach..."

Embedding model confusion:
- "JWT" and "JSON Web Tokens" are synonymous
- But embeddings might treat them as distinct concepts
- Similarity could be moderate (~0.65) rather than high (~0.9)
```

**Example 2: Domain-Specific Terminology**

```
Query: "How do I set up sharding?"
Document: "Horizontal partitioning of data across multiple servers..."

A non-database expert wouldn't know these are synonymous.
A generic embedding model trained on web text may struggle too.
```

**Example 3: Granularity Mismatch**

```
Query: "Pricing for startups"
Document: "Our tiered pricing: Basic ($10/user/month), Professional ($30/user/month),
           Enterprise (custom). Recommended for growing companies and startups."

Query is high-level (startup = audience type).
Document is feature-focused (lists all tiers).
Embedding similarity: Moderate, may not rank at top.
```

### Root Causes

**1. Training Data Bias**

Embedding models are trained on general web corpora (Common Crawl, Wikipedia, etc.). They learn relationships that occur frequently in web text:

```
Frequent in training data:
  - "machine learning" near "neural networks"
  - "API" near "REST"
  
Rare in training data (especially for new domains):
  - "RAG" near "retrieval-augmented generation" (term coined ~2020)
  - "LLM prompt injection" near "security vulnerability"
  
Result: New/specialized concepts have poor embeddings.
```

**2. Vocabulary Coverage**

Embedding models have finite vocabularies (e.g., 250k tokens for BERT). Domain jargon may not be represented:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

# Well-covered terms
emb_good = model.encode("attention mechanism")  # Precise representation

# Domain-specific terms
emb_bad = model.encode("Byzantine fault tolerance")  # May tokenize as subwords
# Tokenized as ["By", "##zan", "##tine", " ", "fault", " ", "tolerance"]
# Leads to degraded embedding quality
```

**3. Asymmetric Similarity**

Short queries and long documents have different embedding distributions:

```
Query: "How to scale?"
Document: "Building Scalable Web Applications: Chapter 5 discusses horizontal
           scaling through database replication, load balancing, and caching
           strategies to support millions of concurrent users..."

Query embedding: Small, focused
Document embedding: Large, diffuse (averaged across many tokens)
Similarity may not be as high as expected.
```

### Quantifying the Semantic Gap

```python
def analyze_semantic_gap(query, document, embedding_model):
    """Measure the gap between query and document embeddings."""
    
    query_embedding = embedding_model.encode(query)
    document_embedding = embedding_model.encode(document)
    
    similarity = cosine_similarity([query_embedding], [document_embedding])[0][0]
    
    # Also compute keyword overlap (BM25-style)
    query_tokens = set(query.lower().split())
    doc_tokens = set(document.lower().split())
    keyword_overlap = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
    
    gap = {
        'dense_similarity': similarity,
        'sparse_overlap': keyword_overlap,
        'gap_size': keyword_overlap - similarity  # Should be ~0 if model is good
    }
    
    # If sparse >> dense, semantic gap is significant
    if gap['gap_size'] > 0.3:
        print(f"⚠️  Large semantic gap detected:")
        print(f"   Dense (embedding): {similarity:.2f}")
        print(f"   Sparse (BM25): {keyword_overlap:.2f}")
        print(f"   Gap: {gap['gap_size']:.2f}")
    
    return gap
```

### Mitigation Strategies

| Strategy | How It Works | Trade-off |
|----------|---|---|
| **Domain-Fine-Tuned Embeddings** | Fine-tune embedding model on domain data (medical, finance, etc.) | Requires labeled data, re-indexing |
| **Asymmetric Embeddings** | Use query-specific encoders (e.g., `query_encoder` vs `doc_encoder`) | Separate models, more compute |
| **Hybrid Search (Dense + Sparse)** | Combine embedding similarity with BM25 keyword matching | Complexity, tuning two systems |
| **Query Expansion** | Expand query with synonyms/related terms before retrieval | Extra preprocessing, latency |
| **Instruction-Tuned Models** | Use embeddings trained on query-document pairs (e.g., `text-embedding-3-large`, `e5-base-v2`) | Newer models may have fewer docs |

</details>

---

## Q6. What techniques detect and mitigate retrieval failure? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

### Detection Techniques

**1. Zero-Result Detection**

```python
def detect_zero_results(query, retrieved_chunks):
    """Simplest case: retrieval returned nothing."""
    if len(retrieved_chunks) == 0:
        log_alert("Zero retrieval results for query: " + query)
        return True  # Failure detected
    return False
```

**2. Similarity Threshold Monitoring**

```python
def detect_low_confidence_retrieval(query, retrieved_chunks, threshold=0.6):
    """All chunks have low similarity to query."""
    
    similarities = [
        embedding_model.similarity(query, chunk)
        for chunk in retrieved_chunks
    ]
    
    mean_similarity = statistics.mean(similarities)
    
    if mean_similarity < threshold:
        log_alert(f"Low retrieval confidence: {mean_similarity:.2f}")
        return True  # Likely failure
    
    return False
```

**3. User Feedback Integration**

```python
def detect_via_user_rating(query, retrieved_chunks, user_rating):
    """User explicitly rates retrieval quality."""
    
    if user_rating <= 2:  # On 1-5 scale
        log_retrieval_failure(query, retrieved_chunks)
        return True
    
    return False
```

### Mitigation Techniques

**Technique 1: Query Expansion**

Expand the original query with synonyms and related terms before retrieval:

```python
def query_expansion(query, expansion_method='synonym'):
    """Expand query to cover semantic variations."""
    
    if expansion_method == 'synonym':
        # Add synonyms manually or via WordNet
        synonyms = {
            'authenticate': ['login', 'auth', 'sign in', 'credential'],
            'scale': ['grow', 'expand', 'increase capacity'],
        }
        
        expanded = [query]
        for word in query.split():
            if word in synonyms:
                expanded.extend([query.replace(word, syn) for syn in synonyms[word]])
        
        return expanded
    
    elif expansion_method == 'llm':
        # Use LLM to generate variations
        prompt = f"""Generate 3 alternative phrasings of this query:
        "{query}"
        
        Return only the phrasings, one per line."""
        
        variations = llm(prompt).split('\n')
        return [query] + variations[:3]
    
    return [query]

def retrieve_with_expansion(query, retriever):
    """Retrieve using original and expanded queries."""
    
    expanded_queries = query_expansion(query, 'llm')
    all_chunks = []
    
    for q in expanded_queries:
        chunks = retriever.search(q, k=5)
        all_chunks.extend(chunks)
    
    # Deduplicate and re-rank
    unique_chunks = deduplicate_by_id(all_chunks)
    ranked = reranker.rank(query, unique_chunks, top_k=5)
    
    return ranked
```

**Technique 2: Hybrid Search (Dense + Sparse)**

Combine embedding-based and keyword-based retrieval:

```python
def hybrid_retrieval(query, dense_retriever, sparse_retriever, alpha=0.5):
    """Combine dense (embedding) and sparse (BM25) retrieval."""
    
    # Dense retrieval
    dense_results = dense_retriever.search(query, k=10)
    dense_scores = {chunk['id']: chunk['score'] for chunk in dense_results}
    
    # Sparse retrieval
    sparse_results = sparse_retriever.search(query, k=10)  # BM25
    sparse_scores = {chunk['id']: chunk['score'] for chunk in sparse_results}
    
    # Combine scores
    all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
    combined_scores = {}
    
    for chunk_id in all_ids:
        dense_score = dense_scores.get(chunk_id, 0)
        sparse_score = sparse_scores.get(chunk_id, 0)
        
        # Weighted combination
        combined_scores[chunk_id] = alpha * dense_score + (1 - alpha) * sparse_score
    
    # Re-rank by combined score
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [chunk for chunk_id, _ in ranked[:5]]
```

**Technique 3: Hypothetical Document Embedding (HyDE)**

Generate hypothetical documents from the query, then retrieve similar real documents:

```python
def hyde_retrieval(query, llm, embedding_model, retriever):
    """
    Generate hypothetical document that would answer the query,
    then retrieve similar real documents.
    """
    
    # Step 1: Generate hypothetical document
    hyde_prompt = f"""Generate a hypothetical document that would answer the following question.
    
    Question: {query}
    
    Hypothetical document:"""
    
    hypothetical_doc = llm(hyde_prompt)
    
    # Step 2: Retrieve real documents similar to hypothetical
    hypothetical_embedding = embedding_model.encode(hypothetical_doc)
    
    retrieved_chunks = retriever.search_by_embedding(
        hypothetical_embedding,
        k=5
    )
    
    # Step 3: Re-rank by relevance to original query
    reranked = reranker.rank(query, retrieved_chunks, top_k=5)
    
    return reranked
```

**Technique 4: Metadata Filtering and Faceted Search**

Use document metadata to narrow search space:

```python
def filtered_retrieval(query, retriever, filters=None):
    """Retrieve with metadata filters to reduce search space."""
    
    if filters is None:
        filters = {}
    
    # Example filters
    filters = {
        'product_id': 'product_x',  # Only search docs about Product X
        'date_range': ('2024-01-01', '2024-12-31'),  # Recent docs only
        'language': 'en',
    }
    
    results = retriever.search(query, k=10, filters=filters)
    
    return results[:5]
```

**Technique 5: Reranking**

Re-rank initial retrieval results using a more sophisticated model:

```python
def retrieve_and_rerank(query, initial_retriever, cross_encoder_reranker):
    """Retrieve more candidates, then rerank with cross-encoder."""
    
    # Step 1: Quick retrieval with dense embeddings (k=20)
    candidates = initial_retriever.search(query, k=20)
    
    # Step 2: Expensive reranking with cross-encoder (top-5)
    reranked = cross_encoder_reranker.rank(query, candidates, top_k=5)
    
    return reranked
```

</details>

---

## Q7. How do you implement hybrid retrieval and what are its trade-offs? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Hybrid retrieval combines dense embeddings (neural) and sparse (keyword) approaches to get the best of both worlds: semantic understanding + lexical precision.

### Architecture Comparison

```
Dense Retrieval Only (Embedding-based):
  Query: "How to scale a database?"
  ↓ Embed query & documents
  ↓ Cosine similarity search
  ✓ Semantic understanding
  ✗ May miss exact keyword matches ("scaling" vs "replica")
  
Sparse Retrieval Only (BM25):
  Query: "How to scale a database?"
  ↓ Tokenize and analyze term frequency
  ✓ Exact keyword matching
  ✗ No semantic understanding ("scaling" ≠ "enlargement" semantically)
  
Hybrid Retrieval:
  Query: "How to scale a database?"
  ↓ Retrieve with BOTH dense + sparse
  ↓ Combine scores (alpha * dense + (1-alpha) * sparse)
  ✓ Semantic + lexical
```

### Implementation

```python
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

class HybridRetriever:
    def __init__(self, documents, alpha=0.5, embedding_model='all-mpnet-base-v2'):
        """
        Initialize hybrid retriever.
        
        Args:
            documents: List of strings or dicts with 'text' key
            alpha: Weight for dense score (1-alpha for sparse)
            embedding_model: SentenceTransformer model name
        """
        self.alpha = alpha
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Extract text
        if isinstance(documents[0], str):
            self.documents = documents
            self.doc_texts = documents
        else:
            self.documents = documents
            self.doc_texts = [doc['text'] for doc in documents]
        
        # Initialize BM25
        tokenized_docs = [doc.split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Embed all documents (expensive, but one-time cost)
        self.doc_embeddings = self.embedding_model.encode(
            self.doc_texts,
            convert_to_tensor=True
        )
    
    def retrieve(self, query, k=5):
        """Retrieve top-k documents using hybrid method."""
        
        # Dense retrieval
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        dense_scores = (query_embedding @ self.doc_embeddings.T).cpu().numpy()
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
        
        # Sparse retrieval (BM25)
        sparse_scores_raw = self.bm25.get_scores(query.split())
        sparse_scores = (sparse_scores_raw - sparse_scores_raw.min()) / (sparse_scores_raw.max() - sparse_scores_raw.min() + 1e-8)
        
        # Combine
        combined_scores = self.alpha * dense_scores + (1 - self.alpha) * sparse_scores
        
        # Get top-k
        top_k_indices = np.argsort(combined_scores)[::-1][:k]
        
        results = [
            {
                'text': self.doc_texts[i],
                'index': i,
                'dense_score': float(dense_scores[i]),
                'sparse_score': float(sparse_scores[i]),
                'combined_score': float(combined_scores[i])
            }
            for i in top_k_indices
        ]
        
        return results

# Example usage
documents = [
    "Database scaling involves horizontal and vertical approaches.",
    "Horizontal scaling adds more servers; vertical scaling adds CPU/RAM.",
    "Replication creates copies of data across servers for redundancy.",
    "Sharding distributes data by key across multiple databases.",
    "Caching reduces database load by storing frequently accessed data."
]

retriever = HybridRetriever(documents, alpha=0.7)

query = "How do I replicate data across servers?"
results = retriever.retrieve(query, k=3)

for result in results:
    print(f"Score: {result['combined_score']:.3f} (dense={result['dense_score']:.3f}, sparse={result['sparse_score']:.3f})")
    print(f"Text: {result['text']}\n")
```

### Tuning Alpha Parameter

The `alpha` weight determines the balance:

```python
def analyze_alpha_impact(query, documents, true_relevant_indices):
    """Test different alpha values and measure recall."""
    
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        retriever = HybridRetriever(documents, alpha=alpha)
        results = retriever.retrieve(query, k=5)
        retrieved_indices = [r['index'] for r in results]
        
        recall = len(set(retrieved_indices) & set(true_relevant_indices)) / len(true_relevant_indices)
        
        print(f"Alpha={alpha}: Recall={recall:.2%} (0.0=sparse-only, 1.0=dense-only)")
```

**Recommended values:**

- **Alpha=1.0 (dense-only)**: For semantic matching, synonyms, paraphrasing
- **Alpha=0.7**: Balanced, works for most use cases
- **Alpha=0.5**: Equal weight to both
- **Alpha=0.3**: Emphasize exact keywords (technical docs, code)
- **Alpha=0.0 (sparse-only)**: When keywords are critical (product names, IDs)

### Trade-offs

| Aspect | Dense-Only | Sparse-Only | Hybrid |
|--------|---|---|---|
| **Semantic understanding** | Excellent | Poor | Good |
| **Exact keyword match** | Moderate | Excellent | Good |
| **Latency (search time)** | 50ms (vector similarity) | 5ms (inverted index) | 50ms (bottleneck: dense) |
| **Index size** | 10GB (embeddings) | 100MB (inverted index) | 10GB total |
| **Memory overhead** | High (store embeddings) | Low | High |
| **Works with synonyms** | Yes | No | Yes |
| **Domain-specific tuning** | Hard (requires fine-tuning) | Easy (stopwords, tokenizer) | Medium |
| **Freshness** | Must re-embed on update | Update inverted index immediately | Must re-embed, update index |

### Production Architecture

```
Input Query
  ├─ Dense Path
  │   ├─ Embed query
  │   ├─ Vector similarity search
  │   └─ Dense scores
  │
  ├─ Sparse Path
  │   ├─ Tokenize query
  │   ├─ BM25 score
  │   └─ Sparse scores
  │
  ├─ Combine Scores (alpha-weighted)
  │
  └─ Output: Top-5 results

Typical latency breakdown:
  - Query embedding: ~10ms (cached model)
  - Vector search: ~20ms
  - BM25 search: ~5ms
  - Combination + sorting: ~5ms
  Total: ~40ms
```

### Optimization: Two-Stage Retrieval

Use sparse (fast) for initial filtering, dense (accurate) for re-ranking:

```python
def two_stage_hybrid_retrieval(query, documents, k_sparse=50, k_dense=5):
    """Stage 1: Fast BM25 filter. Stage 2: Accurate dense re-rank."""
    
    # Stage 1: Quick BM25 to narrow down (50 candidates)
    bm25_results = bm25_retriever.search(query, k=k_sparse)
    
    # Stage 2: Dense re-rank on candidates (5 final results)
    reranked = dense_reranker.rank(query, bm25_results, k=k_dense)
    
    return reranked
    
    # Latency: 5ms (BM25) + 20ms (dense on 50 items) = 25ms vs 40ms for full hybrid
```

</details>

---

## Q8. How do you evaluate retrieval quality in production and establish SLOs? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Production retrieval quality requires continuous measurement against SLOs (Service Level Objectives):

### Offline Evaluation (Benchmark-Based)

Create a test set with ground truth:

```python
def build_evaluation_dataset(n_queries=1000):
    """
    Create a dataset with:
    - Questions/queries
    - Ground truth: which documents/chunks are relevant
    """
    
    # Manual curation (or use existing datasets)
    test_set = [
        {
            'query': 'How do I scale a database?',
            'relevant_doc_ids': [42, 105, 203]  # Documents that answer this
        },
        {
            'query': 'What is a transformer model?',
            'relevant_doc_ids': [500, 502]
        },
        # ... 1000 queries
    ]
    
    return test_set

def evaluate_baseline(retriever, test_set, k=5):
    """Evaluate on offline benchmark."""
    
    metrics = {
        'recall_at_k': [],
        'mrr': [],
        'ndcg': []
    }
    
    for item in test_set:
        query = item['query']
        relevant_ids = set(item['relevant_doc_ids'])
        
        # Retrieve
        results = retriever.search(query, k=k)
        retrieved_ids = {r['id'] for r in results}
        
        # Recall@k
        recall = len(retrieved_ids & relevant_ids) / len(relevant_ids)
        metrics['recall_at_k'].append(recall)
        
        # MRR
        for rank, result in enumerate(results, 1):
            if result['id'] in relevant_ids:
                metrics['mrr'].append(1 / rank)
                break
        else:
            metrics['mrr'].append(0)
        
        # NDCG (simplified)
        dcg = sum(
            (1 if result['id'] in relevant_ids else 0) / np.log2(rank + 1)
            for rank, result in enumerate(results, 1)
        )
        idcg = sum(1 / np.log2(i + 1) for i in range(1, min(len(relevant_ids) + 1, k + 1)))
        ndcg = dcg / idcg if idcg > 0 else 0
        metrics['ndcg'].append(ndcg)
    
    # Aggregate
    results = {
        'recall@5': np.mean(metrics['recall_at_k']),
        'mrr': np.mean(metrics['mrr']),
        'ndcg@5': np.mean(metrics['ndcg'])
    }
    
    return results

# Baseline metrics
baseline = evaluate_baseline(retriever, test_set, k=5)
print(f"Baseline Recall@5: {baseline['recall@5']:.2%}")
print(f"Baseline MRR: {baseline['mrr']:.3f}")
print(f"Baseline NDCG@5: {baseline['ndcg@5']:.3f}")
```

### Production Monitoring

**Metric 1: Query Success Rate**

```python
def track_query_success(query, retrieved_chunks, user_feedback):
    """
    Track: did this query return useful results?
    Inferred from user behavior + explicit feedback.
    """
    
    # Implicit signals
    time_to_satisfaction = measure_user_interaction_time(query)
    num_result_clicks = count_clicked_results(query)
    
    # Explicit signal
    user_rating = user_feedback.get('rating', None)  # 1-5 star
    
    success = (
        (time_to_satisfaction < 30) and  # Found answer quickly
        (num_result_clicks <= 2) and     # Didn't have to browse many results
        (user_rating >= 4)               # User was satisfied
    )
    
    return success

def monitor_success_rate(lookback_hours=24):
    """Track daily success rate."""
    
    recent_queries = get_recent_queries(hours=lookback_hours)
    success_count = sum(1 for q in recent_queries if q['success'])
    success_rate = success_count / len(recent_queries) if recent_queries else 0
    
    print(f"Success rate (24h): {success_rate:.2%}")
    
    if success_rate < 0.85:  # Alert if < 85% success
        alert("Retrieval quality degradation detected")
```

**Metric 2: No-Result Rate**

```python
def track_zero_results(query, retrieved_chunks):
    """Count queries that return zero results."""
    
    if len(retrieved_chunks) == 0:
        log_metric('zero_results', value=1)
    
def monitor_zero_result_rate(lookback_hours=24):
    """Alert if too many queries return nothing."""
    
    zero_result_count = sum_metric('zero_results', hours=lookback_hours)
    total_queries = get_query_count(hours=lookback_hours)
    
    zero_rate = zero_result_count / total_queries
    
    print(f"Zero-result rate (24h): {zero_rate:.2%}")
    
    if zero_rate > 0.05:  # Alert if > 5% of queries have no results
        alert("Retrieval index may be degraded")
```

**Metric 3: Similarity Score Distribution**

```python
def monitor_retrieval_confidence(query, retrieved_chunks):
    """Track distribution of retrieval confidence scores."""
    
    similarities = [chunk['similarity_score'] for chunk in retrieved_chunks]
    
    if len(similarities) > 0:
        mean_sim = np.mean(similarities)
        log_metric('mean_similarity_score', value=mean_sim)
        
        if mean_sim < 0.5:  # Threshold
            log_metric('low_confidence_retrieval', value=1)

def monitor_confidence_distribution(lookback_hours=24):
    """Alert if mean similarity dropping over time."""
    
    mean_similarity = get_metric_average('mean_similarity_score', hours=lookback_hours)
    
    if mean_similarity < 0.6:
        alert(f"Low retrieval confidence: {mean_similarity:.2f}")
```

### SLOs (Service Level Objectives)

Define targets for production retrieval:

```python
slos = {
    'availability': {
        'target': 0.9999,  # 99.99% uptime
        'window': '30d'
    },
    'latency': {
        'p50': 50,  # milliseconds
        'p95': 150,
        'p99': 500,
        'window': '5m'
    },
    'success_rate': {
        'target': 0.95,  # 95% of queries return useful results
        'window': '24h'
    },
    'zero_result_rate': {
        'threshold': 0.02,  # Alert if > 2% of queries have no results
        'window': '1h'
    },
    'retrieval_quality': {
        'recall@5': 0.85,
        'mrr': 0.75,
        'ndcg@5': 0.82,
        'window': 'weekly_evaluation'
    }
}

# Example SLO alert
if current_zero_result_rate > slos['zero_result_rate']['threshold']:
    alert_severity = 'critical'
    escalate_to_oncall()
```

### A/B Testing Retrieval Changes

```python
def ab_test_retriever_change(new_retriever, baseline_retriever, test_fraction=0.1):
    """
    Route 10% of traffic to new retriever, 90% to baseline.
    Measure quality differences.
    """
    
    baseline_metrics = {'recall': [], 'success': []}
    new_metrics = {'recall': [], 'success': []}
    
    for query in stream_incoming_queries():
        if random.random() < test_fraction:
            # Test: new retriever
            results = new_retriever.search(query, k=5)
            group = 'new'
            metrics = new_metrics
        else:
            # Control: baseline
            results = baseline_retriever.search(query, k=5)
            group = 'baseline'
            metrics = baseline_metrics
        
        # Measure outcomes
        recall = measure_recall(query, results)
        success = measure_success(query, results)
        
        metrics['recall'].append(recall)
        metrics['success'].append(success)
    
    # Statistical test
    t_stat, p_value = scipy.stats.ttest_ind(
        new_metrics['recall'],
        baseline_metrics['recall']
    )
    
    if p_value < 0.05 and np.mean(new_metrics['recall']) > np.mean(baseline_metrics['recall']):
        print(f"✓ New retriever is statistically better (p={p_value:.3f})")
        deploy_new_retriever()
    else:
        print(f"✗ No improvement (p={p_value:.3f})")
        rollback()
```

</details>

---

## Q9. What is the impact of chunking strategy on retrieval quality? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Chunking—how documents are split into retrievable units—is a critical design decision that directly affects retrieval quality, latency, and cost.

### Impact of Chunk Size

```python
def analyze_chunk_size_impact(document, embedding_model, query_samples):
    """Evaluate how chunk size affects retrieval."""
    
    chunk_sizes = [128, 256, 512, 1024, 2048]
    results = {}
    
    for chunk_size in chunk_sizes:
        chunks = split_document(document, size=chunk_size)
        chunk_embeddings = embedding_model.encode([c['text'] for c in chunks])
        
        retrieval_quality = 0
        for query in query_samples:
            query_embedding = embedding_model.encode(query)
            similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
            max_similarity = np.max(similarities)
            retrieval_quality += max_similarity
        
        avg_quality = retrieval_quality / len(query_samples)
        
        results[chunk_size] = {
            'avg_quality': avg_quality,
            'num_chunks': len(chunks),
            'avg_chunk_size': np.mean([len(c['text']) for c in chunks])
        }
    
    print("Chunk Size Analysis:")
    for size, metrics in results.items():
        print(f"  {size} chars: quality={metrics['avg_quality']:.3f}, chunks={metrics['num_chunks']}")
    
    # Trade-offs:
    # - Small (128): Many chunks, better granularity, higher cost, lower latency
    # - Medium (512): Sweet spot for most domains
    # - Large (2048): Few chunks, risk of burying relevant info, lower cost

# Output:
# Chunk Size Analysis:
#   128 chars: quality=0.745, chunks=342
#   256 chars: quality=0.758, chunks=171
#   512 chars: quality=0.762, chunks=86     ← Best overall
#  1024 chars: quality=0.751, chunks=43
#  2048 chars: quality=0.721, chunks=22
```

### Chunking Strategies

**Strategy 1: Fixed-Size Chunking**

```python
def fixed_size_chunking(document, chunk_size=512, overlap=100):
    """Split into fixed-size chunks with optional overlap."""
    
    chunks = []
    for i in range(0, len(document), chunk_size - overlap):
        chunk_text = document[i:i+chunk_size]
        chunks.append({
            'text': chunk_text,
            'start': i,
            'end': min(i+chunk_size, len(document))
        })
    
    return chunks

# Example
doc = "Databases scale horizontally... [1000 chars total]"
chunks = fixed_size_chunking(doc, chunk_size=256, overlap=50)
# Returns 5-6 chunks with 50-char overlap between consecutive chunks
```

**Strategy 2: Semantic Chunking**

Split at sentence/paragraph boundaries rather than arbitrary positions:

```python
def semantic_chunking(document, target_chunk_size=512, tokenizer='en_core_web_sm'):
    """Split at semantic boundaries (sentences, paragraphs)."""
    
    import spacy
    nlp = spacy.load(tokenizer)
    doc_nlp = nlp(document)
    
    chunks = []
    current_chunk = ""
    
    for sent in doc_nlp.sents:
        sentence = sent.text
        
        if len(current_chunk) + len(sentence) > target_chunk_size:
            # Start new chunk
            if current_chunk:
                chunks.append({'text': current_chunk})
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    
    if current_chunk:
        chunks.append({'text': current_chunk})
    
    return chunks

# Example
doc = "Replication ensures data durability. It maintains copies... [many sentences]"
chunks = semantic_chunking(doc, target_chunk_size=512)
# Each chunk ends at sentence boundary, maintains coherence
```

**Strategy 3: Hierarchical Chunking**

Create chunks at multiple granularities:

```python
def hierarchical_chunking(document):
    """Create nested chunks: summary + sections + paragraphs."""
    
    # Level 1: Summary of entire doc
    summary = generate_summary(document, max_tokens=100)
    chunks = [{'text': summary, 'level': 'document'}]
    
    # Level 2: Sections (split by headings)
    sections = split_by_heading(document)
    for section_title, section_text in sections:
        section_summary = generate_summary(section_text, max_tokens=50)
        chunks.append({
            'text': f"{section_title}\n{section_summary}",
            'level': 'section',
            'section': section_title
        })
        
        # Level 3: Paragraphs within section
        paragraphs = section_text.split('\n\n')
        for para in paragraphs[:10]:  # Limit to avoid explosion
            chunks.append({
                'text': para,
                'level': 'paragraph',
                'section': section_title
            })
    
    return chunks
```

### Impact on Retrieval Quality

| Chunking Strategy | Pros | Cons | Best For |
|---|---|---|---|
| **Fixed-size (256)** | Simple, fast indexing | May split mid-sentence | Large homogeneous corpora |
| **Fixed-size (512)** | Balanced, standard | Loses structure | General QA systems |
| **Fixed-size (1024)** | Fewer chunks, lower cost | May bury relevant info | Long-form documents (books, papers) |
| **Semantic** | Preserves coherence, high quality | Slower chunking, variable size | Well-structured docs (APIs, guides) |
| **Hierarchical** | Retrieves at right granularity, multi-level ranking | Complex indexing | Nested knowledge bases |

### Production Trade-offs

```python
# Cost-Quality Trade-off Example (1M documents, 500 chars avg each)

strategies = {
    'fixed_256': {
        'chunk_count': 2_000_000,
        'embedding_time': 30_000,  # seconds
        'embedding_cost': 500,  # dollars
        'retrieval_latency': 40,  # ms
        'retrieval_quality': 0.76
    },
    'fixed_512': {
        'chunk_count': 1_000_000,
        'embedding_time': 15_000,
        'embedding_cost': 250,
        'retrieval_latency': 35,
        'retrieval_quality': 0.79
    },
    'semantic': {
        'chunk_count': 1_200_000,
        'embedding_time': 20_000,
        'embedding_cost': 300,
        'retrieval_latency': 38,
        'retrieval_quality': 0.85
    }
}

# Conclusion: Semantic chunking best quality-to-cost ratio
```

### Monitoring Chunking Quality

```python
def evaluate_chunking_quality(retriever, test_queries):
    """Measure how chunking affects retrieval success."""
    
    metrics = {
        'avg_chunk_size': [],
        'recall': [],
        'time_to_answer': []
    }
    
    for query in test_queries:
        results = retriever.search(query, k=5)
        
        avg_size = np.mean([len(r['text']) for r in results])
        recall = measure_recall(query, results)
        
        metrics['avg_chunk_size'].append(avg_size)
        metrics['recall'].append(recall)
    
    print(f"Avg chunk size: {np.mean(metrics['avg_chunk_size']):.0f} chars")
    print(f"Recall@5: {np.mean(metrics['recall']):.2%}")
    
    # If recall is low, consider adjusting chunk size or strategy
```

</details>

---

## Q10. How do you balance cost, latency, and quality in retrieval systems? What are key trade-offs? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Retrieval system design is fundamentally about trade-offs: costs (indexing, inference, storage), latency (query response time), and quality (how well retrieval answers user queries).

### Cost Breakdown

```python
def calculate_retrieval_cost(corpus_size_gb, daily_queries, chunk_size=512):
    """Estimate annual cost of a retrieval system."""
    
    # Corpus analysis
    avg_doc_size = 5000  # characters
    avg_chunk_size = chunk_size
    num_documents = (corpus_size_gb * 1e9) / avg_doc_size
    num_chunks = num_documents * (avg_doc_size / avg_chunk_size)
    
    # Embedding cost (one-time + incremental)
    embedding_cost_per_1m = 0.02  # Arbitrary, depends on model/provider
    initial_embedding_cost = (num_chunks / 1e6) * embedding_cost_per_1m
    
    # Daily query cost
    avg_query_latency_ms = 50
    retrieval_calls_per_query = 1  # May be > 1 for reranking/expansion
    daily_embedding_cost = (daily_queries * retrieval_calls_per_query / 1e6) * embedding_cost_per_1m
    annual_embedding_cost = initial_embedding_cost + (daily_embedding_cost * 365)
    
    # Storage cost (vector DB)
    vector_db_storage_gb = num_chunks * 0.001  # Rough estimate
    annual_storage_cost = vector_db_storage_gb * 10  # $10/GB/year
    
    # Compute cost (infrastructure)
    annual_compute_cost = 5000  # Baseline for hosting
    
    total_annual_cost = annual_embedding_cost + annual_storage_cost + annual_compute_cost
    cost_per_query = total_annual_cost / (daily_queries * 365)
    
    return {
        'initial_embedding_cost': initial_embedding_cost,
        'annual_embedding_cost': annual_embedding_cost,
        'annual_storage_cost': annual_storage_cost,
        'annual_compute_cost': annual_compute_cost,
        'total_annual_cost': total_annual_cost,
        'cost_per_query': cost_per_query
    }

# Example
cost = calculate_retrieval_cost(corpus_size_gb=10, daily_queries=50000)
print(f"Annual cost: ${cost['total_annual_cost']:.2f}")
print(f"Cost per query: ${cost['cost_per_query']:.6f}")

# Output:
# Annual cost: $8,500
# Cost per query: $0.00047
```

### Latency Trade-offs

```python
latency_profiles = {
    'dense_only': {
        'query_embedding': 10,      # ms
        'vector_search': 30,
        'total': 40,
        'quality': 0.78,
        'cost': 'low'
    },
    'dense_reranked': {
        'query_embedding': 10,
        'vector_search': 30,
        'rerank_top_10': 50,        # Cross-encoder on 10 results
        'total': 90,
        'quality': 0.85,
        'cost': 'medium'
    },
    'hybrid': {
        'query_embedding': 10,
        'vector_search': 30,
        'bm25_search': 5,
        'combine_scores': 5,
        'total': 50,
        'quality': 0.82,
        'cost': 'medium'
    },
    'dense_reranked_hyd e': {
        'llm_hypothetical_doc': 200,  # Expensive LLM call
        'query_embedding': 10,
        'vector_search': 30,
        'rerank': 50,
        'total': 290,
        'quality': 0.88,
        'cost': 'high'
    }
}

# Selection logic:
# - P95 latency < 200ms? → dense_only
# - P95 latency < 500ms? → dense_reranked or hybrid
# - Offline batch processing? → dense_reranked_hyd e
```

### Quality-Cost Pareto Frontier

```python
import matplotlib.pyplot as plt

strategies = {
    'baseline_dense': {'quality': 0.78, 'cost': 100},
    'dense_reranked': {'quality': 0.85, 'cost': 180},
    'hybrid': {'quality': 0.82, 'cost': 140},
    'semantic_chunking': {'quality': 0.88, 'cost': 150},
    'full_stack': {'quality': 0.92, 'cost': 300},
}

# Plot
fig, ax = plt.subplots()
for name, metrics in strategies.items():
    ax.scatter(metrics['cost'], metrics['quality'], s=100, label=name)
    ax.annotate(name, (metrics['cost'], metrics['quality']))

ax.set_xlabel('Relative Cost')
ax.set_ylabel('Retrieval Quality (Recall@5)')
ax.set_title('Quality-Cost Trade-off')
ax.legend()
plt.show()

# The "Pareto frontier" includes: semantic_chunking, dense_reranked
# Off-frontier: full_stack (too expensive), baseline_dense (too low quality)
```

### Decision Matrix

Choose your strategy based on use case:

| Use Case | Latency Budget | Quality Target | Recommended Strategy |
|----------|---|---|---|
| **Chat assistant** | P95 < 100ms | Recall ≥ 0.75 | Dense-only with large k (retrieve 20, show 5) |
| **Search engine** | P95 < 200ms | Recall ≥ 0.85 | Dense + BM25 hybrid, rerank if time permits |
| **Knowledge base QA** | P95 < 500ms | Recall ≥ 0.90 | Semantic chunking + dense + reranker |
| **Batch processing** | No latency budget | Recall ≥ 0.95 | Query expansion + HyDE + dense + reranker + NLI |
| **Mission-critical** (medical, legal) | < 1s | Recall ≥ 0.98 | Multi-stage: sparse filter → dense → cross-encoder → NLI |

### A/B Testing Framework

```python
def run_retrieval_ab_test(baseline_strategy, new_strategy, duration_days=14):
    """Compare two retrieval strategies in production."""
    
    import numpy as np
    
    baseline_metrics = {'recall': [], 'latency': [], 'cost': []}
    new_metrics = {'recall': [], 'latency': [], 'cost': []}
    
    for day in range(duration_days):
        for query in stream_daily_queries(day):
            # Route 50/50
            if random.random() < 0.5:
                strategy = baseline_strategy
                metrics = baseline_metrics
            else:
                strategy = new_strategy
                metrics = new_metrics
            
            # Measure
            start = time.time()
            results = strategy.search(query)
            latency = (time.time() - start) * 1000  # ms
            cost = strategy.estimate_cost(query)
            recall = measure_recall(query, results)
            
            metrics['recall'].append(recall)
            metrics['latency'].append(latency)
            metrics['cost'].append(cost)
    
    # Statistical significance
    from scipy.stats import ttest_ind
    
    recall_t_stat, recall_p_value = ttest_ind(
        new_metrics['recall'],
        baseline_metrics['recall']
    )
    
    print(f"Recall improvement: {np.mean(new_metrics['recall']) - np.mean(baseline_metrics['recall']):.2%}")
    print(f"Statistical significance: p={recall_p_value:.4f}")
    print(f"Latency increase: {np.mean(new_metrics['latency']) - np.mean(baseline_metrics['latency']):.1f}ms")
    print(f"Cost increase: {(np.mean(new_metrics['cost']) - np.mean(baseline_metrics['cost'])) / np.mean(baseline_metrics['cost']):.1%}")
    
    if recall_p_value < 0.05:
        print("✓ New strategy is statistically better")
        return True
    else:
        print("✗ No significant improvement, keep baseline")
        return False
```

### Optimization Checklist

Deploy these in order of diminishing returns:

```
Priority 1 (Quick wins):
  □ Semantic chunking (often +5-10% quality, no extra cost)
  □ Reorder by relevance (no cost, +2-3% quality)
  □ Remove low-confidence results (filtering, no cost)

Priority 2 (Good ROI):
  □ Reranking (top-10) (+5-8% quality, +50ms latency)
  □ Hybrid (BM25 + dense) (+3-5% quality, minimal latency)
  □ Query expansion (LLM-based) (+4-7% quality, +100ms latency)

Priority 3 (Use only if needed):
  □ HyDE (hypothetical documents) (+5-10% quality, +200ms latency)
  □ Multi-modal retrieval (+10-15% quality, significant complexity)
  □ Custom fine-tuned embeddings (+5-8% quality, high up-front cost)
```

</details>

---
