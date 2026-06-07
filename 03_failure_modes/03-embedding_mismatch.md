# 03 — Embedding Mismatch

> The query and document embeddings are computed in different semantic spaces (domain drift, language variation, or asymmetric encoding), leading to poor retrieval despite semantic relevance.

---

## Q1. What is embedding mismatch and why does it cause retrieval failure? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Embedding mismatch occurs when a query and its relevant documents are embedded into different semantic spaces, causing low cosine similarity despite semantic relevance. The embedding model fails to recognize that they should be close.

**Root issue:**

Embeddings are learned from training data. If a general-purpose embedding model (trained on web text) is applied to a specialized domain (medical, financial, legal), it may not have learned the relationships between domain-specific terms.

| Scenario | Example | Symptom |
|----------|---------|---------|
| **Domain mismatch** | Query: "What is metformin?" (medical), Embedding trained on general web text | Similarity score is low despite document being highly relevant |
| **Language drift** | Query in formal English, Documents in colloquial English | Different token distributions → low similarity |
| **Vocabulary shift** | Query: "API authentication", Docs: "token-based authorization" | Synonyms not learned by model → similarity is 0.5 when should be 0.95 |
| **Asymmetric encoding** | Query is short (5 tokens), Document is long (500 tokens) | Embeddings average over different context window sizes |
| **New concepts** | Query mentions "retrieval-augmented generation" (term coined ~2020) | Pre-trained model (trained on 2019 data) never saw this term |

This is distinct from retrieval failure (wrong docs retrieved) because the **information exists and is relevant**, but the embedding model's representation of similarity is wrong.

</details>

---

## Q2. What are observable symptoms of embedding mismatch in production? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Symptoms of embedding mismatch can be detected through multiple signals:

| Symptom | Detection Method | Example |
|---------|---|---|
| **Low similarity scores across all results** | Monitor cosine similarity distribution | Query returns documents with max similarity=0.55 when good matches should be ~0.85 |
| **Manual audit reveals correct doc ranked low** | Periodically check: does ground-truth doc rank in top-5? | You search for "How to implement RAG", correct doc exists but ranked at position 23 |
| **Domain-specific queries fail but general queries succeed** | A/B by query type | Queries about "blockchain smart contracts" have low success, but "What is Python?" works fine |
| **Synonyms not recognized** | Test query expansion: do synonyms rank the same? | Query: "Web server scaling" vs "HTTP load balancing" → different results despite near-identical meaning |
| **Language variation causes divergence** | Test linguistic variants | Query: "How do I authenticate?" vs "How do I do authentication?" → Different rankings |
| **Performance degrades in new domain** | Recall@k drops after domain shift | Retrain on medical docs, recall drops from 0.90 to 0.65 |
| **User clicks don't match ranking** | Track: where do users click vs where results ranked? | Top result in retrieval ranking not clicked, user clicks result ranked 8th |

**Production signals:**

```python
def detect_embedding_mismatch_signals(query, retrieved_chunks, user_feedback):
    """Flag potential mismatch."""
    
    signals = {
        'low_max_similarity': np.max([c['similarity'] for c in retrieved_chunks]) < 0.6,
        'low_mean_similarity': np.mean([c['similarity'] for c in retrieved_chunks]) < 0.5,
        'user_clicked_lower_ranked': user_feedback.get('clicked_rank', 0) > 5,
        'user_explicit_failure': user_feedback.get('rating', 5) <= 2,
    }
    
    if sum(signals.values()) >= 2:
        log_mismatch_alert(query)
```

</details>

---

## Q3. What causes embedding mismatch? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Embedding mismatch stems from fundamental differences in how models represent meaning:

### 1. Domain Shift

Embedding models are trained on general-purpose corpora (web text, Wikipedia). When applied to specialized domains, the learned representations don't capture domain-specific semantics.

```
General-purpose (web) embedding:
  "bank" ≈ "financial institution, river edge, slope"
  Heavy weight on finance meaning

Domain: Riverbank management
  "bank" should mean "river edge"
  Embedding model's representation is misaligned

Result: Query "riverbank erosion" doesn't match doc about "bank stabilization"
```

**Quantifiable**: Fine-tuned embeddings on domain data outperform general models by 10-40% Recall@k on domain benchmarks.

### 2. Vocabulary and Tokenization Mismatch

Different models tokenize text differently:

```
Query (using BERT tokenizer):
  "retrieval-augmented generation"
  Tokens: ["retrieval", "-", "augmented", "generation"]

Document (using a different tokenizer):
  "retrieval augmented generation"  (no hyphen)
  Tokens: ["retrieval", "augmented", "generation"]

Different token sequences → Different embeddings
```

### 3. Asymmetric Query-Document Encoding

Most embedding models average token embeddings into a single vector. Short queries and long documents have different averaging distributions:

```
Short query (5 tokens):
  "How to scale?"
  Averaged over 5 tokens

Long document (500 tokens):
  "Scaling databases involves sharding, replication, caching... [continues]"
  Averaged over 500 tokens
  
The long document's embedding is noisier (averaged over many tokens).
Similarity calculation treats them equally despite asymmetry.
```

### 4. Language and Phrasing Variation

Same concept, different phrasings:

```
Query A: "How do I authenticate users?"
Query B: "How do I do user authentication?"
Query C: "What is the auth process?"

All mean the same thing, but token-level differences can cause embedding divergence.

Embedding similarity:
  sim(A, B) = 0.92 (syntactic overlap)
  sim(A, C) = 0.72 (lower, but should be high)
  sim(B, C) = 0.75
```

### 5. Temporal Drift

Embedding models trained on older data may not understand newer terminology:

```
Model trained: 2019
New terminology: "ChatGPT" (released Nov 2022), "RAG" (refined 2023)

Query (2024): "How does RAG differ from fine-tuning?"
Model sees: "RAG" = unknown subword tokens
Embedding is degraded.

Compare: Model trained in 2024 handles "RAG" naturally.
```

### 6. Cross-Lingual Mismatch

Multilingual models balance languages poorly; one language may be underrepresented:

```
Multilingual model trained on 80% English, 10% Spanish, 10% French

Spanish query: "¿Cómo autenticar usuarios?"
Spanish document: "Autenticación de usuarios mediante JWT..."

Model learned strong English associations, weaker Spanish.
Similarity: Lower than it should be for in-language matching.
```

### 7. Instruction-Tuning Asymmetry

Some embedding models expect a specific query format (e.g., "Represent the question:"):

```
Model: all-MiniLM-L6-v2 (no instruction format required)
Query: "How to scale a database?"
Doc: "Horizontal scaling adds more servers..."
Similarity: 0.70

Model: e5-base-v2 (instruction-tuned)
Query (correct format): "Represent the question: How to scale a database?"
Doc (correct format): "Represent the document: Horizontal scaling adds more servers..."
Similarity: 0.92

Using wrong format → Embedding mismatch.
```

</details>

---

## Q4. How do you detect embedding mismatch in a production system? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

### Detection Method 1: Embedding Quality Benchmarks (MTEB)

Use standard benchmarks to understand embedding model strengths/weaknesses:

```python
from mteb import MTEB

# MTEB: Massive Text Embedding Benchmark
# Evaluates embeddings on 50+ datasets across 8 task categories

model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)

# Evaluate on retrieval tasks
evaluation = MTEB(tasks=["TREC-COVID", "DBpedia", "SCIFACT"])
results = evaluation.run(model)

# Results structure
# {
#   'TREC-COVID': {'NDCG@10': 0.58, ...},
#   'DBpedia': {'NDCG@10': 0.45, ...},
#   'SCIFACT': {'NDCG@10': 0.67, ...}
# }

# Check if your domain matches these tasks
# If your domain ≠ MTEB task, expect lower performance
```

**Key MTEB benchmark domains:**

| Task | Domain | Top Model | Score |
|------|--------|-----------|-------|
| TREC-COVID | Medical/scientific retrieval | text-embedding-3-large | 0.74 |
| SCIFACT | Scientific fact verification | text-embedding-3-large | 0.72 |
| DBpedia | General knowledge retrieval | text-embedding-3-large | 0.69 |
| SciDocs | Scientific paper retrieval | text-embedding-3-large | 0.71 |
| MrTyDi | Multilingual (12 languages) | text-embedding-3-large | 0.53 |

**Interpretation:** If your domain is not on MTEB, your embedding model's performance is uncertain → high mismatch risk.

### Detection Method 2: Similarity Score Distribution Analysis

```python
def analyze_similarity_distribution(queries, documents, embedding_model):
    """Check if similarity distribution matches expected pattern."""
    
    similarities = []
    
    for query in queries:
        query_emb = embedding_model.encode(query)
        
        for doc in documents:
            doc_emb = embedding_model.encode(doc['text'])
            sim = cosine_similarity([query_emb], [doc_emb])[0][0]
            
            is_relevant = doc['is_relevant_to_query']
            similarities.append({
                'similarity': sim,
                'is_relevant': is_relevant
            })
    
    # Distribution analysis
    relevant_sims = [s['similarity'] for s in similarities if s['is_relevant']]
    irrelevant_sims = [s['similarity'] for s in similarities if not s['is_relevant']]
    
    print(f"Relevant docs:   mean={np.mean(relevant_sims):.3f}, std={np.std(relevant_sims):.3f}")
    print(f"Irrelevant docs: mean={np.mean(irrelevant_sims):.3f}, std={np.std(irrelevant_sims):.3f}")
    
    # Good separation: mean_relevant >> mean_irrelevant
    separation = np.mean(relevant_sims) - np.mean(irrelevant_sims)
    
    if separation < 0.2:
        print(f"⚠️  Poor separation: {separation:.3f} (should be > 0.3)")
        log_mismatch_alert("Embeddings not well-separated")
    
    return {
        'relevant_mean': np.mean(relevant_sims),
        'irrelevant_mean': np.mean(irrelevant_sims),
        'separation': separation
    }
```

### Detection Method 3: Synonym/Paraphrase Consistency

```python
def test_synonym_consistency(embedding_model):
    """Check if synonyms have high similarity."""
    
    synonym_pairs = [
        ("authenticate user", "user login"),
        ("database sharding", "horizontal partitioning"),
        ("API key", "access token"),
        ("machine learning", "artificial intelligence"),
    ]
    
    for query1, query2 in synonym_pairs:
        emb1 = embedding_model.encode(query1)
        emb2 = embedding_model.encode(query2)
        sim = cosine_similarity([emb1], [emb2])[0][0]
        
        print(f'"{query1}" vs "{query2}": {sim:.3f}')
        
        if sim < 0.85:
            print(f"  ⚠️  Low similarity for near-synonyms → mismatch risk")
```

**Example output:**
```
"authenticate user" vs "user login": 0.94          ✓
"database sharding" vs "horizontal partitioning": 0.68  ⚠️
"API key" vs "access token": 0.71                   ⚠️
```

### Detection Method 4: Domain-Specific Evaluation Set

Create a small labeled evaluation set in your domain:

```python
def create_domain_eval_set(domain_name, num_pairs=100):
    """
    Manually or semi-automatically create query-document pairs
    labeled with relevance.
    """
    
    eval_set = [
        {
            'query': 'How to set up JWT authentication?',
            'documents': [
                {'text': 'Bearer tokens and JSON Web Tokens...', 'relevant': True},
                {'text': 'Session-based auth uses cookies...', 'relevant': False},
            ]
        },
        # ... 100 query-document pairs
    ]
    
    # Evaluate embedding model on this set
    ndcg = evaluate_ndcg(embedding_model, eval_set)
    
    print(f"Domain-specific NDCG@5: {ndcg:.3f}")
    
    if ndcg < 0.75:
        print("⚠️  Below threshold for production (75%), consider fine-tuning")
    
    return eval_set

def evaluate_ndcg(embedding_model, eval_set):
    """Compute NDCG@5 on evaluation set."""
    
    ndcg_scores = []
    
    for item in eval_set:
        query = item['query']
        documents = item['documents']
        
        # Rank documents by embedding similarity
        query_emb = embedding_model.encode(query)
        ranked = sorted(
            documents,
            key=lambda d: cosine_similarity(
                [query_emb],
                [embedding_model.encode(d['text'])]
            )[0][0],
            reverse=True
        )
        
        # Compute NDCG
        dcg = sum(
            (1 if doc['relevant'] else 0) / np.log2(i+2)
            for i, doc in enumerate(ranked[:5])
        )
        idcg = sum(1 / np.log2(i+2) for i in range(min(5, sum(1 for d in documents if d['relevant']))))
        
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)
```

### Detection Method 5: A/B Testing Embedding Models

```python
def compare_embedding_models(queries, ground_truth, models_to_test):
    """Test multiple embedding models on your domain."""
    
    results = {}
    
    for model_name in models_to_test:
        model = SentenceTransformer(model_name)
        recall_at_5 = []
        
        for query, relevant_doc_ids in ground_truth.items():
            # Retrieve with model
            retrieved = model_based_retrieval(query, model, k=5)
            retrieved_ids = {d['id'] for d in retrieved}
            
            # Compute recall
            recall = len(retrieved_ids & set(relevant_doc_ids)) / len(relevant_doc_ids)
            recall_at_5.append(recall)
        
        results[model_name] = np.mean(recall_at_5)
    
    # Print ranking
    for model, recall in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: Recall@5 = {recall:.2%}")
    
    return results
```

</details>

---

## Q5. What causes embedding mismatch across different dimensions (domain, language, model architecture)? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Embedding mismatch has multiple independent causes across three dimensions:

### Dimension 1: Domain Shift

**Within-domain shift:** Vocabulary and concepts change over time within a single domain:

```
Year 2015: 
  "Deep learning" embedding ≈ "neural networks", "CNNs", "RNNs"

Year 2023:
  "Deep learning" embedding ≈ "transformers", "LLMs", "diffusion models"

Model trained in 2015 applied to 2023 data → Mismatch
```

**Cross-domain shift:** A model trained on one domain (web) applied to another (medical):

```
Web-trained embedding:
  "virus" ≈ "computer malware", "security threat"

Medical domain:
  "virus" ≈ "pathogen", "infection", "immune response"

Same word, completely different semantic neighbors → High mismatch
```

**Quantification:**

```python
from sklearn.metrics.pairwise import cosine_distances

def measure_domain_shift(model, domain_a_texts, domain_b_texts):
    """Compute distance between domain embeddings."""
    
    emb_a = model.encode(domain_a_texts)
    emb_b = model.encode(domain_b_texts)
    
    # Compute within-domain and cross-domain distances
    within_a = np.mean(cosine_distances(emb_a, emb_a))
    within_b = np.mean(cosine_distances(emb_b, emb_b))
    cross_domain = np.mean(cosine_distances(emb_a, emb_b))
    
    shift_magnitude = cross_domain - (within_a + within_b) / 2
    
    print(f"Domain shift magnitude: {shift_magnitude:.3f}")
    return shift_magnitude

# Example
domain_a = ["machine learning", "neural networks", "training data"]
domain_b = ["bacterial infection", "antibiotic resistance", "viral load"]

shift = measure_domain_shift(model, domain_a, domain_b)
# High shift → Model not suitable for both domains
```

### Dimension 2: Asymmetric Query vs Document Encoding

**The problem:**

Most embedding models pool token embeddings into a single vector via mean/cls. But queries and documents have different characteristics:

```python
# Simple mean-pooling
def encode_text(tokens):
    token_embeddings = [embed(token) for token in tokens]
    return np.mean(token_embeddings, axis=0)

query_tokens = ["How", "to", "scale", "database"]        # 4 tokens
query_emb = encode_text(query_tokens)                      # averaged over 4

doc_tokens = ["Horizontal scaling involves sharding, replication, caching, ..."]  # 50+ tokens
doc_emb = encode_text(doc_tokens)                          # averaged over 50+

# Different averaging contexts → Embedding mismatch
sim(query, doc) lower than it should be
```

**Solutions:**

```python
# Asymmetric encoding (e5-base, text-embedding-3)
model = SentenceTransformer('intfloat/e5-base-v2')

# Model includes instruction prompts
query_emb = model.encode("query: How to scale database")
doc_emb = model.encode("passage: Horizontal scaling...")

# Instructions guide model to handle asymmetry
sim(query, doc) is higher and more accurate
```

### Dimension 3: Language and Multilingual Mismatch

**Monolingual models in multilingual settings:**

```
Model: all-mpnet-base-v2 (trained heavily on English)

Query (Spanish): "¿Cómo escalar una base de datos?"
Document (Spanish): "El escalamiento horizontal distribuye datos..."

Model learned:
  - English embeddings: Dense, well-trained
  - Spanish embeddings: Sparse (less training data)
  
Result: Spanish query and doc have lower similarity than equivalent English pair
```

**Multilingual model imbalance:**

```
Multilingual model trained on:
  - 60% English
  - 20% Chinese
  - 15% Spanish
  - 5% other

Language-specific performance:
  - English NDCG@5: 0.82
  - Chinese NDCG@5: 0.65  ← Lower-resource language suffers
  - Spanish NDCG@5: 0.68  ← Imbalance causes mismatch

Query in Spanish, document in Chinese → Even worse alignment
```

**Cross-lingual mismatch:**

```
Query (English): "How to authenticate?"
Document (French): "Comment authentifier l'utilisateur?"

Cross-lingual model must bridge languages.
Quality degrades compared to within-language retrieval.

Similarity: 0.65 (cross-lingual)
vs. 0.90 (same language)
```

### Dimension 4: Model Architecture Mismatch

**Different model families:**

```
BERT-based (all-MiniLM-L6-v2):
  "database" embedding trained with masked LM objective
  
Contrastive (Sentence-Transformers):
  "database" embedding trained with InfoNCE loss
  
Different architectures → Different learned representations
→ Embeddings in different spaces
```

**In practice:**
```python
model_1 = SentenceTransformer('all-MiniLM-L6-v2')      # BERT-based
model_2 = SentenceTransformer('all-mpnet-base-v2')     # MPNet-based

query = "machine learning"
doc = "artificial intelligence"

sim_1 = cosine_similarity(
    [model_1.encode(query)],
    [model_1.encode(doc)]
)[0][0]  # 0.82

sim_2 = cosine_similarity(
    [model_2.encode(query)],
    [model_2.encode(doc)]
)[0][0]  # 0.91

# Different models give different results for same query-doc pair
```

### Summary: Multi-Dimensional Mismatch Space

```
                    Domain Shift
                        ↓
                    Language
                        ↓
Query-Doc Asymmetry → [Mismatch] ← Model Architecture
                        ↓
                    Temporal Drift
```

To mitigate, assess **all four dimensions** for your use case.

</details>

---

## Q6. What strategies mitigate embedding mismatch? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

### Strategy 1: Fine-Tuning Embeddings on Domain Data

Fine-tune a pre-trained embedding model on labeled (query, document) pairs from your domain:

```python
from sentence_transformers import SentenceTransformer, models, losses, InputExample
from torch.utils.data import DataLoader

# Step 1: Load pre-trained model
base_model = SentenceTransformer('all-mpnet-base-v2')

# Step 2: Prepare training data
training_data = [
    InputExample(
        texts=["How do I scale a database?", "Horizontal scaling distributes data..."],
        label=0.95  # High similarity (relevant pair)
    ),
    InputExample(
        texts=["How do I scale a database?", "Python tutorial for beginners"],
        label=0.1   # Low similarity (irrelevant pair)
    ),
    # ... 1000s of domain-specific examples
]

train_dataloader = DataLoader(training_data, shuffle=True, batch_size=16)

# Step 3: Fine-tune
train_loss = losses.CosineSimilarityLoss(model=base_model)
base_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100
)

# Step 4: Evaluate on domain test set
domain_test_set = [...]  # Query-doc pairs with relevance labels
metrics = evaluate_ndcg(base_model, domain_test_set)
print(f"Fine-tuned model NDCG@5: {metrics['ndcg']:.3f}")
```

**Cost-benefit:**

| Aspect | Cost | Benefit |
|--------|------|---------|
| Training data | 500-1000 labeled pairs | +10-30% Recall@k |
| Compute | 1-2 GPU hours | Domain-optimized |
| Ongoing | Needs retraining on new data | Stays relevant |

### Strategy 2: Asymmetric Embeddings

Use models designed for query-document asymmetry (e.g., e5-large-v2, text-embedding-3):

```python
from sentence_transformers import SentenceTransformer

# Instruction-tuned model handles asymmetry
model = SentenceTransformer('intfloat/e5-large-v2')

query_embedding = model.encode(
    "query: How to implement JWT authentication?",
    convert_to_tensor=True
)

doc_embedding = model.encode(
    "passage: Bearer tokens and JSON Web Tokens enable stateless authentication...",
    convert_to_tensor=True
)

similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
# Higher than non-instruction-tuned model due to asymmetric encoding

# Compared to non-asymmetric:
model_basic = SentenceTransformer('all-mpnet-base-v2')
sim_basic = 0.72
# Asymmetric: 0.88, improvement = +16%
```

**Instruction formats vary by model:**

```python
# e5 models
query_emb = model.encode("query: " + query_text)
doc_emb = model.encode("passage: " + doc_text)

# text-embedding-3 (OpenAI)
# Built-in asymmetry, no special prompt needed
query_emb = model.encode(query_text)
doc_emb = model.encode(doc_text)

# BGE models
query_emb = model.encode("Represent this query for searching: " + query_text)
doc_emb = model.encode("Represent this document for searching: " + doc_text)
```

### Strategy 3: Hybrid Query and Document Expansion

Expand queries and documents to reduce mismatch:

```python
def expand_for_mismatch_mitigation(text, text_type='query', llm_model=None):
    """Expand text with synonyms and related concepts."""
    
    if text_type == 'query':
        prompt = f"""Generate 2 alternative phrasings of this question that preserve meaning:
        "{text}"
        
        Return only the phrasings, one per line."""
    
    elif text_type == 'document':
        prompt = f"""Generate a 1-sentence summary and 2 key concept keywords for this text:
        "{text[:200]}..."
        
        Return in format:
        Summary: <summary>
        Keywords: <keyword1>, <keyword2>"""
    
    expansions = llm_model(prompt)
    return expansions

# Example
query = "How to authenticate users?"
expanded = expand_for_mismatch_mitigation(query, 'query')
# "How do I authenticate users?" + "User authentication methods" + "Login mechanisms"

# Embed both original and expanded, take max similarity
original_sim = compute_similarity(query, doc, model)
expanded_sims = [
    compute_similarity(expanded, doc, model)
    for expanded in expansions
]

best_similarity = max([original_sim] + expanded_sims)
```

### Strategy 4: Multi-Modal or Instruction-Tuned Models

Use models trained on diverse tasks/modalities:

```python
# MTEB-evaluated models (tested across 50+ tasks)
# Less likely to have domain-specific mismatch

models_ranked = [
    ('text-embedding-3-large', 0.86),      # Best overall
    ('text-embedding-3-small', 0.85),      
    ('bge-large-en-v1.5', 0.84),           
    ('e5-large-v2', 0.83),                 
    ('all-mpnet-base-v2', 0.77),           # Older, less robust
]

# For domain, check MTEB performance on similar tasks
# If similar task exists and score is high (>0.75), lower mismatch risk
```

### Strategy 5: Domain-Specific Pre-Training

If you have large unlabeled domain corpus, pre-train embeddings:

```python
from sentence_transformers import SentenceTransformer, models

# Step 1: Load unlabeled domain documents
domain_corpus = [
    "Database sharding distributes data across multiple servers...",
    "Horizontal scaling adds more database replicas...",
    # ... thousands of domain documents
]

# Step 2: Use contrastive loss with in-domain corpus
# Create synthetic pairs: anchor + positive (from same document) + negatives

# Step 3: Fine-tune on unlabeled data using self-supervised loss
model = SentenceTransformer('all-mpnet-base-v2')

# This is expensive but highly effective for specialized domains
# Improves recall@5 by 15-25% on that domain
```

### Strategy 6: Metadata-Guided Retrieval

Use document metadata to disambiguate and reduce mismatch:

```python
def metadata_aware_retrieval(query, embedding_model, metadata_filters=None):
    """Retrieve with embedding + metadata constraints."""
    
    query_emb = embedding_model.encode(query)
    
    # Retrieve from filtered subset
    candidates = db.search(
        query_emb,
        k=100,
        filters=metadata_filters  # e.g., {'domain': 'finance', 'year': 2024}
    )
    
    # Re-rank within filtered set
    # Avoids matching high similarity scores from unrelated domains
    return candidates[:5]

# Example
# Query: "What is the Fed rate?"
# Without metadata: Might return "Federal Investigation Bureau" docs
# With metadata {domain='finance'}: Returns only financial docs
```

### Comparison of Strategies

| Strategy | Effort | Time to Deploy | Quality Gain | Best For |
|----------|--------|---|---|---|
| **Switch to better model** | Low (just change model) | 1 day | +5-15% | Quick win, applicable to all domains |
| **Fine-tune on labeled data** | Medium (need 500-1000 pairs) | 1 week | +10-30% | Specialized domains with budget |
| **Use asymmetric model** | Low | 1 day | +5-10% | Query-heavy workloads |
| **Query expansion** | Low | 1 day | +3-8% | Complementary to other strategies |
| **Pre-train on corpus** | High (GPU-intensive) | 2-4 weeks | +15-25% | Large private corpus, long-term |
| **Metadata filtering** | Medium | 1 week | +5-10% | Heterogeneous knowledge bases |

</details>

---

## Q7. How do you choose and evaluate embedding models for your domain? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

### Evaluation Workflow

**Step 1: Identify Similar MTEB Benchmark Tasks**

```python
# MTEB benchmark includes 50+ datasets across domains
# Find tasks similar to your domain

your_domain = "medical QA"

mteb_similar_tasks = [
    ('TREC-COVID', 'medical retrieval'),
    ('SciFact', 'scientific fact verification'),
    ('NF-Corpus', 'scientific full text'),
]

# Check which models perform best on similar tasks
mteb_rankings = {
    'text-embedding-3-large': {'TREC-COVID': 0.74, 'SciFact': 0.72},
    'text-embedding-3-small': {'TREC-COVID': 0.71, 'SciFact': 0.69},
    'bge-large-en-v1.5': {'TREC-COVID': 0.70, 'SciFact': 0.68},
}

# Models ranking high on similar tasks likely good for your domain
```

**Step 2: Create Domain-Specific Evaluation Set**

```python
def create_evaluation_set(domain, num_queries=200):
    """
    Create labeled (query, document) pairs from your domain.
    
    Methods:
    1. Manual: Experts label 200 query-doc pairs
    2. Semi-automatic: Use existing QA datasets in domain + label gaps
    3. Synthetic: Use LLM to generate queries from docs
    """
    
    eval_set = []
    
    # Method 1: Manual labeling (highest quality)
    for doc in domain_documents:
        query = input(f"Write a query relevant to: {doc[:100]}...")
        eval_set.append({
            'query': query,
            'documents': [
                {'text': doc, 'relevant': True},
                {'text': random_other_doc, 'relevant': False},
                {'text': random_other_doc2, 'relevant': False},
            ]
        })
    
    return eval_set

# Example: Medical domain
eval_set_medical = [
    {
        'query': 'What are the contraindications for metformin?',
        'documents': [
            {'text': 'Metformin is contraindicated in renal failure...', 'relevant': True},
            {'text': 'Insulin is used for Type 1 diabetes...', 'relevant': False},
        ]
    },
    # ... 200 queries
]
```

**Step 3: Compute Metrics on Your Evaluation Set**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def evaluate_embedding_model(model_name, eval_set):
    """Evaluate model on your domain-specific eval set."""
    
    model = SentenceTransformer(model_name)
    
    ndcg_scores = []
    recall_scores = []
    
    for item in eval_set:
        query = item['query']
        documents = item['documents']
        
        # Encode query and documents
        query_emb = model.encode(query, convert_to_tensor=True)
        doc_embeddings = model.encode(
            [d['text'] for d in documents],
            convert_to_tensor=True
        )
        
        # Compute similarities
        similarities = (query_emb @ doc_embeddings.T).cpu().numpy()
        
        # Rank documents
        ranked_indices = np.argsort(-similarities[0])
        
        # Compute NDCG@5
        dcg = sum(
            (1 if documents[i]['relevant'] else 0) / np.log2(rank+2)
            for rank, i in enumerate(ranked_indices[:5])
        )
        idcg = sum(1 / np.log2(i+2) for i in range(min(5, sum(1 for d in documents if d['relevant']))))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
        
        # Compute Recall@5
        relevant_indices = {i for i, d in enumerate(documents) if d['relevant']}
        retrieved_indices = set(ranked_indices[:5])
        recall = len(relevant_indices & retrieved_indices) / len(relevant_indices) if relevant_indices else 0
        recall_scores.append(recall)
    
    return {
        'model': model_name,
        'ndcg@5': np.mean(ndcg_scores),
        'recall@5': np.mean(recall_scores),
        'ndcg_std': np.std(ndcg_scores),
    }

# Compare multiple models
models_to_evaluate = [
    'all-mpnet-base-v2',
    'intfloat/e5-base-v2',
    'intfloat/e5-large-v2',
    'BAAI/bge-large-en-v1.5',
    'text-embedding-3-small',
]

results = []
for model_name in models_to_evaluate:
    try:
        metrics = evaluate_embedding_model(model_name, eval_set_medical)
        results.append(metrics)
        print(f"{model_name}: NDCG@5={metrics['ndcg@5']:.3f}, Recall@5={metrics['recall@5']:.3f}")
    except Exception as e:
        print(f"{model_name}: Failed ({e})")

# Rank models
ranked = sorted(results, key=lambda x: x['ndcg@5'], reverse=True)
for rank, model in enumerate(ranked, 1):
    print(f"{rank}. {model['model']}: {model['ndcg@5']:.3f}")
```

**Step 4: Cost-Latency Analysis**

```python
def analyze_embedding_cost_latency(model_names, corpus_size_docs=100000):
    """Compare cost and latency of different models."""
    
    analysis = {}
    
    for model_name in model_names:
        model = SentenceTransformer(model_name)
        
        # Latency per embedding
        import time
        sample_texts = ["Sample document"] * 100
        start = time.time()
        model.encode(sample_texts)
        elapsed = time.time() - start
        latency_per_doc_ms = (elapsed / 100) * 1000
        
        # Model size (proxy for memory and cost)
        model_params = sum(p.numel() for p in model.parameters()) / 1e6  # millions
        
        # Cost estimates
        indexing_cost = (corpus_size_docs / 1000) * 0.01 * (model_params / 100)  # arbitrary units
        query_latency_ms = latency_per_doc_ms * 1.5  # queries often slower
        
        analysis[model_name] = {
            'latency_ms': query_latency_ms,
            'model_size_params_m': model_params,
            'relative_cost': indexing_cost,
        }
    
    return analysis

# Example output
analysis = analyze_embedding_cost_latency(models_to_evaluate)
for model, metrics in analysis.items():
    print(f"{model}: {metrics['latency_ms']:.1f}ms latency, {metrics['model_size_params_m']:.0f}M params")
```

### Decision Matrix

Choose based on your constraints:

```python
def recommend_model(quality_target, latency_target_ms, cost_budget):
    """Recommend model based on constraints."""
    
    candidates = {
        'text-embedding-3-large': {
            'quality': 0.90,  # Best
            'latency': 50,
            'cost': 'high',
        },
        'text-embedding-3-small': {
            'quality': 0.87,
            'latency': 20,
            'cost': 'low',
        },
        'intfloat/e5-large-v2': {
            'quality': 0.88,
            'latency': 80,  # Slower
            'cost': 'low',
        },
        'BAAI/bge-large-en-v1.5': {
            'quality': 0.85,
            'latency': 70,
            'cost': 'low',
        },
        'all-mpnet-base-v2': {
            'quality': 0.78,
            'latency': 15,
            'cost': 'very low',
        },
    }
    
    recommendations = []
    for model_name, metrics in candidates.items():
        if (metrics['quality'] >= quality_target and
            metrics['latency'] <= latency_target_ms):
            recommendations.append((model_name, metrics))
    
    # Rank by cost
    recommendations.sort(key=lambda x: x[1]['cost'])
    
    if recommendations:
        return recommendations[0][0]
    else:
        return None  # No model meets all constraints
```

### Final Recommendation Table

| Scenario | Recommended Models | Rationale |
|----------|---|---|
| **Production, high quality** | text-embedding-3-large, e5-large-v2 | MTEB #1 ranked, domain-specific fine-tuning possible |
| **Production, low latency** | text-embedding-3-small, all-mpnet-base-v2 | <20ms latency, reasonable quality |
| **Domain-specific (medical, finance)** | e5-large-v2 + fine-tuning | Fine-tune on labeled domain data for +15-25% improvement |
| **Multilingual** | multilingual-e5-large, text-embedding-3 | Support 50+ languages |
| **Open-source only** | bge-large-en-v1.5, e5-large-v2 | Best open models, no API dependency |
| **Very large scale (>1B docs)** | text-embedding-3-small, all-MiniLM | Balance indexing cost and quality |

</details>

---

## Q8. How do you implement fine-tuning of embeddings for your domain? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

### Full Fine-Tuning Workflow

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import ndcg_score

class DomainEmbeddingFinetuner:
    def __init__(self, base_model_name='all-mpnet-base-v2', domain_name='medical'):
        self.base_model = SentenceTransformer(base_model_name)
        self.domain_name = domain_name
    
    def prepare_training_data(self, query_doc_pairs_with_labels):
        """
        Convert labeled data to InputExample format.
        
        Input format:
          [{
            'query': 'How to treat diabetes?',
            'doc': 'Insulin therapy is...',
            'similarity': 0.95,  # 0-1, 1=highly relevant
          }, ...]
        """
        
        training_examples = []
        
        for pair in query_doc_pairs_with_labels:
            example = InputExample(
                texts=[pair['query'], pair['doc']],
                label=pair['similarity']
            )
            training_examples.append(example)
        
        return training_examples
    
    def finetune(self, training_data, val_data, epochs=1, batch_size=16):
        """Fine-tune the embedding model."""
        
        # Prepare data
        train_examples = self.prepare_training_data(training_data)
        
        # Create data loader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )
        
        # Define loss function (regression on similarity scores)
        train_loss = losses.CosineSimilarityLoss(
            model=self.base_model,
            tie_p=0.05  # Regularization
        )
        
        # Fine-tune
        self.base_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=len(train_examples) // (batch_size * 10),
            output_path=f'./checkpoints/{self.domain_name}-embeddings',
            save_best_model=True,
            evaluator=None,  # Optional: add dev set evaluator
        )
    
    def evaluate_on_test_set(self, test_data):
        """Evaluate fine-tuned model."""
        
        ndcg_scores = []
        
        for item in test_data:
            query = item['query']
            docs = item['documents']  # List of dicts with 'text' and 'relevant'
            
            # Encode
            query_emb = self.base_model.encode(query, convert_to_tensor=True)
            doc_embeddings = self.base_model.encode(
                [d['text'] for d in docs],
                convert_to_tensor=True
            )
            
            # Compute similarities
            similarities = (query_emb @ doc_embeddings.T).cpu().numpy()[0]
            
            # Create relevance labels
            y_true = [1 if d['relevant'] else 0 for d in docs]
            
            # Compute NDCG@5
            y_scores = similarities
            ndcg = ndcg_score([y_true], [y_scores], k=5)
            ndcg_scores.append(ndcg)
        
        return {
            'ndcg@5': np.mean(ndcg_scores),
            'ndcg_std': np.std(ndcg_scores),
        }

# Usage example
finetuner = DomainEmbeddingFinetuner(base_model_name='all-mpnet-base-v2', domain_name='medical')

# Prepare training data
training_pairs = [
    {'query': 'Metformin contraindications?', 'doc': 'Metformin is contraindicated in renal failure...', 'similarity': 0.98},
    {'query': 'Metformin contraindications?', 'doc': 'Insulin is used for Type 1 diabetes...', 'similarity': 0.05},
    # ... 500-1000 pairs
]

validation_pairs = [...]  # Separate set for monitoring

# Fine-tune
finetuner.finetune(training_pairs, validation_pairs, epochs=1, batch_size=32)

# Evaluate
test_data = [...]
metrics = finetuner.evaluate_on_test_set(test_data)
print(f"Fine-tuned NDCG@5: {metrics['ndcg@5']:.3f}")

# Compare to baseline
baseline_model = SentenceTransformer('all-mpnet-base-v2')
baseline_metrics = finetuner.evaluate_on_test_set_with_model(test_data, baseline_model)
print(f"Baseline NDCG@5: {baseline_metrics['ndcg@5']:.3f}")
print(f"Improvement: {(metrics['ndcg@5'] - baseline_metrics['ndcg@5']) / baseline_metrics['ndcg@5']:.1%}")
```

### Data Collection Strategies

**Strategy 1: Mining from Existing QA Logs**

```python
def mine_positive_pairs_from_logs(qa_logs, similarity_threshold=0.85):
    """
    Extract positive (query, doc) pairs from user interactions.
    
    Heuristic: If user spent time reading a doc after searching a query,
    it's likely relevant.
    """
    
    positive_pairs = []
    
    for log in qa_logs:
        query = log['query']
        clicked_doc_id = log['clicked_doc']  # User clicked this
        clicked_doc = fetch_doc(clicked_doc_id)
        
        dwell_time = log['dwell_time_seconds']
        
        # If user spent >30 seconds on doc, assume relevant
        if dwell_time > 30:
            positive_pairs.append({
                'query': query,
                'doc': clicked_doc['text'],
                'similarity': 0.95,  # High confidence
            })
    
    return positive_pairs
```

**Strategy 2: Weak Labeling**

```python
def weak_label_pairs(documents, embedding_model):
    """
    Use heuristics to automatically label (query, doc) pairs.
    
    Heuristic: If doc text contains several query terms, it's relevant.
    """
    
    pairs = []
    
    for doc in documents:
        # Generate synthetic query from doc (using LLM or extraction)
        query = generate_query_from_doc(doc)
        
        # If query → doc retrieval works well, doc is good for query
        # This creates positive (query, doc) pair
        
        pairs.append({
            'query': query,
            'doc': doc['text'],
            'similarity': 0.85,  # Medium-high confidence
        })
    
    return pairs
```

**Strategy 3: Expert Annotation**

```python
def create_expert_labeled_set(num_queries=200, domain='medical'):
    """
    Have domain experts label query-document relevance.
    
    High-quality but expensive approach.
    """
    
    pairs = []
    
    for _ in range(num_queries):
        query = get_random_query(domain)
        candidate_docs = retrieve_candidates(query, k=10)
        
        # Expert labels relevance: 0 (irrelevant), 0.5 (partial), 1 (highly relevant)
        for doc in candidate_docs:
            label = expert_label_relevance(query, doc)  # Human input
            
            pairs.append({
                'query': query,
                'doc': doc['text'],
                'similarity': float(label),
            })
    
    return pairs
```

### Monitoring Fine-Tuning Quality

```python
def monitor_finetuning_progress(finetuned_model, baseline_model, validation_set):
    """Track improvement over baseline during fine-tuning."""
    
    finetuned_ndcg = evaluate_model(finetuned_model, validation_set)
    baseline_ndcg = evaluate_model(baseline_model, validation_set)
    
    improvement = (finetuned_ndcg - baseline_ndcg) / baseline_ndcg
    
    print(f"Baseline NDCG@5: {baseline_ndcg:.3f}")
    print(f"Fine-tuned NDCG@5: {finetuned_ndcg:.3f}")
    print(f"Improvement: {improvement:.1%}")
    
    # Check for overfitting
    if improvement > 0.50:  # >50% gain seems too good
        print("⚠️  Possible overfitting, validation set may be leaking to training")
    
    return improvement
```

### Trade-offs

| Aspect | Cost | Benefit |
|--------|------|---------|
| **Data collection** | 10-50 hours (expert annotation) | Precise labels |
| **Fine-tuning** | 1-2 GPU hours | +10-30% domain-specific improvement |
| **Reindexing** | 2-4 hours (for large corpus) | Embeddings must be recomputed |
| **Maintenance** | Periodic retraining on new data | Keeps model fresh |

**When NOT to fine-tune:**
- Domain is very general (news, web search) → base model already good
- Training data is small (<100 labeled pairs) → risk of overfitting
- Latency is critical → fine-tuning may slow down inference

</details>

---

## Q9. How do you prevent embedding mismatch in multi-domain or evolving systems? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

### Challenge: Multi-Domain Systems

When a single embedding model serves multiple domains (e.g., product catalog for both medical and retail), mismatch risk is high:

```
Shared embedding model:
  - Medical domain: Low performance (0.65 NDCG@5)
  - Retail domain: Medium performance (0.75 NDCG@5)
  - Overall: Suboptimal for both

Domain-specific models:
  - Medical: 0.85 NDCG@5
  - Retail: 0.88 NDCG@5
  - But: 2x complexity, 2x cost
```

### Strategy 1: Domain-Aware Retrieval

Use metadata to route queries to appropriate embeddings:

```python
class DomainAwareRetriever:
    def __init__(self):
        self.embeddings = {
            'medical': SentenceTransformer('scifact-embeddings-finetuned'),
            'retail': SentenceTransformer('retail-product-embeddings-finetuned'),
            'default': SentenceTransformer('all-mpnet-base-v2'),
        }
    
    def retrieve(self, query, documents):
        """Detect domain from query, use appropriate embeddings."""
        
        # Detect domain
        domain = detect_domain(query)  # Rule-based or ML-based classifier
        
        # Select appropriate embedding model
        embedding_model = self.embeddings.get(domain, self.embeddings['default'])
        
        # Retrieve with domain-specific model
        query_emb = embedding_model.encode(query)
        results = vector_search(query_emb, documents, k=5)
        
        return results

def detect_domain(query):
    """Detect domain from query text."""
    
    # Simple rule-based approach
    medical_keywords = {'contraindications', 'diagnosis', 'treatment', 'metformin'}
    retail_keywords = {'price', 'shipping', 'product', 'buy'}
    
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in medical_keywords):
        return 'medical'
    elif any(kw in query_lower for kw in retail_keywords):
        return 'retail'
    else:
        return 'default'
```

### Strategy 2: Adapter Modules (Parameter-Efficient Fine-Tuning)

Instead of fine-tuning entire model for each domain, use lightweight adapters:

```python
from sentence_transformers import SentenceTransformer
import torch.nn as nn

class EmbeddingWithAdapters:
    def __init__(self, base_model_name='all-mpnet-base-v2', domains=['medical', 'retail']):
        self.base_model = SentenceTransformer(base_model_name)
        self.adapters = nn.ModuleDict({
            domain: self.create_adapter() for domain in domains
        })
    
    def create_adapter(self, input_dim=768, reduction=8):
        """Lightweight adapter: linear → ReLU → linear."""
        
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction),
            nn.ReLU(),
            nn.Linear(input_dim // reduction, input_dim)
        )
    
    def encode(self, texts, domain='default'):
        """Encode with domain-specific adapter."""
        
        # Get base embedding
        embeddings = self.base_model.encode(texts)
        
        # Apply adapter
        if domain in self.adapters:
            adapter = self.adapters[domain]
            adapted_embeddings = embeddings + adapter(embeddings)  # Residual connection
            return adapted_embeddings
        else:
            return embeddings

# Fine-tuning adapters (much cheaper than full fine-tuning)
# Each adapter: ~50k parameters vs ~200M for full model
```

### Strategy 3: Continuous Evaluation and Drift Detection

Monitor embedding quality over time:

```python
class EmbeddingQualityMonitor:
    def __init__(self, baseline_metrics_per_domain):
        self.baseline = baseline_metrics_per_domain  # e.g., {'medical': 0.85, 'retail': 0.82}
    
    def check_drift(self, domain, current_ndcg, window_hours=24):
        """Detect if embedding quality is degrading."""
        
        baseline = self.baseline[domain]
        
        # If current NDCG drops >5% from baseline, flag drift
        if current_ndcg < baseline * 0.95:
            print(f"⚠️  Embedding drift detected in {domain} domain")
            print(f"   Baseline: {baseline:.3f}, Current: {current_ndcg:.3f}")
            log_drift_alert(domain, baseline, current_ndcg)
            return True
        
        return False

def continuous_quality_monitoring(retriever, eval_sets_by_domain):
    """Continuously monitor embedding quality."""
    
    monitor = EmbeddingQualityMonitor(baseline_metrics_per_domain)
    
    while True:
        for domain, eval_set in eval_sets_by_domain.items():
            ndcg = evaluate_domain(retriever, domain, eval_set)
            
            monitor.check_drift(domain, ndcg)
            
            log_metric(f'embedding_quality_{domain}', ndcg)
        
        time.sleep(3600)  # Check hourly
```

### Strategy 4: Online Hard Negative Mining

Continuously improve embeddings by mining challenging examples:

```python
def online_hard_negative_mining(query, retrieval_results, ground_truth_relevant):
    """
    If retrieval returns incorrect result ranked high, that's a hard negative.
    Use it to fine-tune embeddings.
    """
    
    hard_negatives = []
    
    for result in retrieval_results[:5]:
        if result['id'] not in ground_truth_relevant:
            # This was ranked high but is actually irrelevant
            # It's a hard negative
            hard_negatives.append({
                'query': query,
                'doc': result['text'],
                'similarity': 0.0,  # Definitely irrelevant
            })
    
    # Use hard negatives to refine embeddings
    if hard_negatives:
        refine_embeddings_with_hard_negatives(hard_negatives)

def refine_embeddings_with_hard_negatives(hard_negatives, batch_size=32):
    """
    Periodically retrain embeddings with collected hard negatives.
    This prevents model from forgetting how to distinguish tricky cases.
    """
    
    # Collect over time (e.g., daily)
    hard_neg_batch = hard_negatives
    
    if len(hard_neg_batch) >= batch_size:
        # Fine-tune embedding model on hard negatives
        # (same fine-tuning process, but with hard examples)
        print(f"Refining embeddings with {len(hard_neg_batch)} hard negatives...")
        # finetuner.finetune(hard_neg_batch, epochs=1)
```

### Strategy 5: Temporal Adaptation

Handle concept drift as terminology evolves:

```python
def handle_temporal_drift(embedding_model, documents_over_time):
    """
    As new terminology emerges (e.g., "GenAI" ~2023), model may become stale.
    Periodically retrain on recent data.
    """
    
    # Group documents by era
    docs_2022 = [d for d in documents_over_time if d['year'] == 2022]
    docs_2023 = [d for d in documents_over_time if d['year'] == 2023]
    docs_2024 = [d for d in documents_over_time if d['year'] == 2024]
    
    # Test on recent data
    recent_ndcg = evaluate_model(embedding_model, docs_2024)
    old_ndcg = evaluate_model(embedding_model, docs_2022)
    
    if recent_ndcg < old_ndcg * 0.90:
        print(f"⚠️  Temporal drift: Performance on 2024 data ({recent_ndcg:.3f}) < 2022 ({old_ndcg:.3f})")
        print("   Retraining on recent data...")
        
        # Retrain on recent documents
        recent_data = docs_2023 + docs_2024
        finetuner.finetune(recent_data, epochs=1)
```

### Monitoring Checklist

```python
embedding_quality_slo = {
    'all_domains': {
        'minimum_ndcg@5': 0.75,
        'maximum_domain_variance': 0.10,  # Don't let some domains degrade
    },
    'per_domain': {
        'medical': {'minimum_ndcg@5': 0.82},
        'retail': {'minimum_ndcg@5': 0.80},
    },
    'drift_detection': {
        'check_frequency': '1h',
        'allowed_degradation': 0.05,  # 5% drop before alert
    },
    'retraining': {
        'frequency': 'quarterly',
        'trigger_on_drift': True,
    }
}
```

</details>

---

## Q10. What is the cost-performance-latency trade-off for embedding solutions, and how do you optimize? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Embedding systems involve a three-way trade-off: cost (inference + storage), quality (NDCG/Recall), and latency (query response time).

### Cost Breakdown

```python
def estimate_embedding_system_cost(
    corpus_size_gb=100,
    monthly_queries=1_000_000,
    embedding_model='all-mpnet-base-v2',
):
    """Estimate annual cost of embedding retrieval system."""
    
    # Document statistics
    avg_doc_chars = 5000
    num_docs = (corpus_size_gb * 1e9) / avg_doc_chars
    embedding_dim = 768  # typical
    
    # Embedding generation cost (one-time indexing + queries)
    embedding_generation_cost_per_call = 0.00001  # Assuming efficient inference
    
    # Initial indexing: embed all documents
    indexing_cost = num_docs * embedding_generation_cost_per_call
    
    # Query-time: embed every query
    query_embeddings_cost = monthly_queries * 12 * embedding_generation_cost_per_call
    
    total_embedding_cost = indexing_cost + query_embeddings_cost
    
    # Storage cost
    embedding_storage_gb = (num_docs * embedding_dim * 4 bytes) / 1e9  # float32
    vector_db_cost_per_gb_per_year = 10  # Pinecone, Weaviate, etc.
    annual_storage_cost = embedding_storage_gb * vector_db_cost_per_gb_per_year
    
    # Compute cost (GPU/CPU inference)
    annual_compute_cost = 50_000  # Rough: dedicated GPU + infrastructure
    
    # Total
    total_annual_cost = total_embedding_cost + annual_storage_cost + annual_compute_cost
    cost_per_query = total_annual_cost / (monthly_queries * 12)
    
    return {
        'embedding_cost': total_embedding_cost,
        'storage_cost': annual_storage_cost,
        'compute_cost': annual_compute_cost,
        'total_annual': total_annual_cost,
        'cost_per_query': cost_per_query,
        'cost_per_doc': total_annual_cost / num_docs,
    }

# Example: 100GB corpus, 1M monthly queries
cost = estimate_embedding_system_cost(corpus_size_gb=100, monthly_queries=1_000_000)
print(f"Annual cost: ${cost['total_annual']:.2f}")
print(f"Cost per query: ${cost['cost_per_query']:.6f}")
print(f"Cost per document: ${cost['cost_per_doc']:.6f}")

# Output (approximate):
# Annual cost: $125,000
# Cost per query: $0.00104
# Cost per document: $0.00125
```

### Model Selection: Cost vs. Quality

```python
# Embed models ranked by MTEB score (quality) and inference speed (latency)
models = {
    'all-MiniLM-L6-v2': {
        'quality': 0.76,
        'latency_ms': 10,
        'params_m': 22,
        'relative_cost': 1.0,      # Baseline
    },
    'all-mpnet-base-v2': {
        'quality': 0.81,
        'latency_ms': 50,
        'params_m': 109,
        'relative_cost': 1.5,
    },
    'intfloat/e5-small-v2': {
        'quality': 0.78,
        'latency_ms': 15,
        'params_m': 33,
        'relative_cost': 1.2,
    },
    'intfloat/e5-large-v2': {
        'quality': 0.88,
        'latency_ms': 80,
        'params_m': 335,
        'relative_cost': 3.0,
    },
    'text-embedding-3-small': {
        'quality': 0.87,
        'latency_ms': 40,  # Typical for API
        'params_m': 100,
        'relative_cost': 2.0,      # API pricing
    },
    'text-embedding-3-large': {
        'quality': 0.92,
        'latency_ms': 60,
        'params_m': 200,
        'relative_cost': 4.0,
    },
}

# Cost-quality Pareto frontier
def find_pareto_frontier(models):
    """Models where you can't improve quality without increasing cost."""
    
    pareto = []
    
    for name, metrics in models.items():
        is_dominated = False
        
        for other_name, other in models.items():
            if other_name == name:
                continue
            
            # Is other model strictly better (higher quality, lower cost)?
            if (other['quality'] > metrics['quality'] and
                other['relative_cost'] < metrics['relative_cost']):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto.append((name, metrics))
    
    return pareto

frontier = find_pareto_frontier(models)
print("Pareto frontier (optimal models):")
for name, metrics in sorted(frontier, key=lambda x: x[1]['relative_cost']):
    print(f"  {name}: quality={metrics['quality']:.2f}, cost={metrics['relative_cost']:.1f}x")

# Output:
# Pareto frontier (optimal models):
#   all-MiniLM-L6-v2: quality=0.76, cost=1.0x
#   intfloat/e5-small-v2: quality=0.78, cost=1.2x
#   all-mpnet-base-v2: quality=0.81, cost=1.5x
#   intfloat/e5-large-v2: quality=0.88, cost=3.0x
#   text-embedding-3-large: quality=0.92, cost=4.0x
```

### Optimization Strategies

**Strategy 1: Selective Fine-Tuning**

Instead of fine-tuning the largest model, fine-tune a smaller one:

```
Without fine-tuning:
  - all-mpnet-base-v2: NDCG=0.81, cost=1.5x
  
With fine-tuning on domain data:
  - all-MiniLM + fine-tuning: NDCG=0.85 (+4%), cost=1.0x + fine-tune overhead
  - all-mpnet + fine-tuning: NDCG=0.88 (+7%), cost=1.5x + fine-tune overhead
  
Result: Fine-tuned small model ≈ large model quality at lower cost
```

**Strategy 2: Caching and Pre-computation**

Cache embeddings for frequently queried documents:

```python
class CachedEmbeddingRetriever:
    def __init__(self, embedding_model):
        self.model = embedding_model
        self.embedding_cache = {}  # {doc_id: embedding}
    
    def retrieve(self, query, documents, use_cache=True):
        """Retrieve with optional embedding cache."""
        
        query_embedding = self.model.encode(query)  # Always compute query
        
        for doc in documents:
            doc_id = doc['id']
            
            # Check cache
            if doc_id in self.embedding_cache and use_cache:
                doc_embedding = self.embedding_cache[doc_id]
            else:
                doc_embedding = self.model.encode(doc['text'])
                self.embedding_cache[doc_id] = doc_embedding
            
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            # ...
```

**Strategy 3: Batch Processing for Queries**

For asynchronous systems, batch queries to amortize embedding cost:

```python
async def batch_retrieve(queries, batch_size=100):
    """Process queries in batches for efficiency."""
    
    all_results = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        
        # Batch embed (more efficient than single queries)
        query_embeddings = embedding_model.encode(batch, batch_size=batch_size)
        
        # Retrieve for each query
        for query, qe in zip(batch, query_embeddings):
            results = vector_search(qe, documents, k=5)
            all_results.append(results)
    
    return all_results

# Latency: 100 queries: ~200ms (batched) vs ~5000ms (sequential)
```

**Strategy 4: Dimension Reduction**

Use dimensionality reduction (PCA) to reduce embedding size:

```python
from sklearn.decomposition import PCA

# Original embeddings: 768 dimensions
full_embeddings = embedding_model.encode(documents)

# Reduce to 256 dimensions (67% smaller)
pca = PCA(n_components=256)
reduced_embeddings = pca.fit_transform(full_embeddings)

# Storage savings: 768 → 256 bytes per embedding (67% reduction)
# Quality loss: Typically <2% NDCG drop

# Trade-off: Tiny quality loss for significant storage/latency savings
```

**Strategy 5: Hybrid Retrieval with Sparse-First Filtering**

Use BM25 to pre-filter, then dense reranking on subset:

```
Traditional (all dense):
  Query → Embed query (10ms) → Dense search 1M docs (50ms) = 60ms

Hybrid with sparse pre-filter:
  Query → BM25 filter to 1000 (5ms) → Embed query (10ms) → 
  Dense search 1000 (20ms) → Rerank top-20 (10ms) = 45ms
  
Result: 25% latency reduction, same quality
```

### Decision Matrix

Choose based on your SLOs:

| SLO | Recommended Approach | Estimated Cost |
|-----|---|---|
| **Quality ≥ 0.85 NDCG, Latency < 50ms** | all-mpnet-base-v2 + hybrid retrieval | 1.5x |
| **Quality ≥ 0.90 NDCG, Latency < 100ms** | e5-large-v2 fine-tuned + caching | 3.5x |
| **Quality ≥ 0.92 NDCG, Cost-insensitive** | text-embedding-3-large + domain fine-tune | 4.0x |
| **Latency-critical (<20ms), Quality ≥ 0.75** | all-MiniLM-L6-v2 + sparse-first + cache | 1.0x |
| **Low-cost, best effort (< $10k/year)** | all-MiniLM + sparse retrieval only | 0.8x |

### ROI Analysis: Is Fine-Tuning Worth It?

```python
def calculate_finetuning_roi(baseline_quality, finetuned_quality, annual_queries, cost_per_quality_point=1000):
    """Calculate ROI of fine-tuning embeddings."""
    
    quality_improvement = (finetuned_quality - baseline_quality) * 100  # percentage points
    finetuning_cost = 5000  # One-time
    
    # Value of improvement (assuming each quality point is worth $X)
    improvement_value = quality_improvement * cost_per_quality_point
    
    roi = (improvement_value - finetuning_cost) / finetuning_cost
    
    print(f"Quality improvement: {quality_improvement:.1f}%")
    print(f"Finetuning cost: ${finetuning_cost:.0f}")
    print(f"Improvement value: ${improvement_value:.0f}")
    print(f"ROI: {roi:.0%}")
    
    return roi > 0  # Worth it if positive ROI

# Example: Improve from 0.78 to 0.86 NDCG
# 8 percentage points * $1000/point = $8000 benefit
# Cost: $5000
# ROI: (+$8000 - $5000) / $5000 = 60% → Worth it
```

</details>

---
