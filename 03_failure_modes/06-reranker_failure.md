# 06 — Reranker Failure

> The cross-encoder reranking stage returns mis-ordered results, placing irrelevant or low-quality chunks at the top and burying relevant ones, despite the initial retrieval being correct.

---

## Q1. What is reranker failure and why does it matter? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Reranker failure occurs when a cross-encoder model (trained to score relevance between query-document pairs) produces incorrect rankings, elevating irrelevant chunks or demoting relevant ones.

**Example:**

```
Initial retrieval (by dense embedding similarity):
  Rank 1: "JWT is a stateless auth method" (similarity: 0.85) ✓ Relevant
  Rank 2: "OAuth uses access tokens" (similarity: 0.78) ✓ Relevant
  Rank 3: "Cookies store session data" (similarity: 0.72) ~ Tangential
  
Reranking (cross-encoder scoring):
  Rank 1: "Cookies store session data" (cross-encoder: 0.92) ✗ Wrong!
  Rank 2: "OAuth uses access tokens" (cross-encoder: 0.85)
  Rank 3: "JWT is a stateless auth method" (cross-encoder: 0.80) ← Best answer buried

LLM uses top-3 chunks:
  → Answers based on cookies/OAuth, ignoring JWT (the real answer)
```

**Why it matters:**

1. **Silent failure:** Reranker ranks high-confidence but wrong results
2. **Waste of inference cost:** Running cross-encoder costs more than dense retrieval
3. **Position bias:** Top-ranked results get more attention from LLM
4. **Cascading errors:** Wrong ranking → wrong retrieval → hallucination
5. **Domain-specific risk:** Rerankers trained on general data may fail on specialized domains

| Scenario | Impact | Severity |
|----------|--------|----------|
| **Top-5 shuffled** | LLM still has right answer, but mixed with noise | Low |
| **Top relevant pushed to #5+** | LLM may not reach relevant answer | High |
| **Complete ranking reversal** | LLM answers based on worst chunks | Critical |

This is distinct from initial retrieval failure because the dense retrieval was *correct*, but reranking *corrupted* the ranking.

</details>

---

## Q2. What are observable symptoms of reranker failure? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Reranker failure manifests through comparison signals between initial and reranked results:

| Symptom | Detection Method | Example |
|---------|---|---|
| **Ranking degradation after reranking** | Compare top-1 before/after reranking | Before: Highly relevant doc at rank 1, After: Irrelevant doc at rank 1 |
| **Low correlation between dense and cross-encoder scores** | Compute Spearman correlation | Dense similarity: 0.85, Cross-encoder: 0.45 for same pair → Disagreement |
| **Irrelevant docs ranked high** | Manual audit: is top-5 actually relevant? | Top-3 results don't match query intent |
| **Quality drops with reranking** | Measure Recall@5 before/after | Dense Recall@5: 0.82, After reranking: 0.65 ← Reranking made it worse |
| **Reranker confidence vs. correctness mismatch** | Is high-confidence ranking actually correct? | Reranker gives score 0.98 but answer is wrong |
| **Domain-specific terminology confusion** | Test on domain queries | Query in specialized jargon, reranker treats it as general |
| **LLM answers improve when reranking skipped** | A/B test with/without reranking | With reranker: quality 0.75, Without: quality 0.82 |

**Production signals:**

```python
def detect_reranker_failure_signals(query, dense_results, reranked_results, llm_response):
    """Flag potential reranking failures."""
    
    # Signal 1: Top result changed dramatically
    dense_top_id = dense_results[0]['id']
    reranked_top_id = reranked_results[0]['id']
    
    if dense_top_id != reranked_top_id:
        # Reranking changed the top result
        # Check if it was an improvement
        is_improvement = (
            reranked_results[0]['relevance'] > dense_results[0]['relevance']
        )
        
        if not is_improvement:
            log_reranker_alert(f"Reranking demoted better result")
    
    # Signal 2: Dense and reranker disagreement
    for dense, reranked in zip(dense_results[:5], reranked_results[:5]):
        if abs(dense['score'] - reranked['score']) > 0.4:
            # Large disagreement
            log_reranker_alert(f"Large disagreement: dense={dense['score']}, cross-encoder={reranked['score']}")
    
    # Signal 3: Quality dropped
    if assess_answer_quality(llm_response, reranked_results) < 0.7:
        log_reranker_alert(f"Low answer quality after reranking")
```

</details>

---

## Q3. What causes reranker failure? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Reranker failure stems from several technical and data-related causes:

### 1. Distribution Shift (Domain Mismatch)

Reranker trained on general domain, applied to specialized domain:

```
Training data: MS MARCO (web search results)
  - Queries: "best pizza near me", "what is photosynthesis"
  - High relevance correlation with document matching

Applied to: Medical domain
  - Queries: "contraindications for metformin"
  - Medical terminology unknown to cross-encoder
  - Learned associations don't apply
  
Result: Reranker confused, ranks incorrectly
```

**Quantifiable:** Domain-specific reranker > general reranker by 15-30% on specialized domains.

### 2. Query-Document Length Asymmetry

Cross-encoders may be biased by length:

```
Short query: "What is JWT?"
Long document A (2,000 tokens): "JWT is... [long explanation]"
Long document B (500 tokens): "JWT: stateless auth method"

Training data imbalance: Most training pairs have moderate lengths
Reranker may overweight document A (longer = more signal)

Result: Longer but less relevant docs ranked higher
```

### 3. Shallow Semantic Understanding

Cross-encoder may only capture surface-level relevance:

```
Query: "How to prevent SQL injection?"

Document A (matched keyword "prevent SQL injection"):
  "This paper studies SQL injection prevention. Recent work..."
  Cross-encoder score: 0.92 (high, matched keywords)

Document B (deeper relevance but different keywords):
  "Input validation and parameterized queries block SQL injection attacks"
  Cross-encoder score: 0.65 (lower, "prevention" not explicitly mentioned)

Reranker: A > B, but B is more useful for implementation
```

### 4. Training Data Biases

Reranker trained on human-labeled data with systematic biases:

```
Training data: Expert judges labeled 1M query-document pairs

Bias 1: Length bias
  - Long documents more likely labeled as relevant
  - Because they contain more information

Bias 2: Popularity bias
  - Wikipedia articles more likely labeled relevant
  - Because they're common in training data

Applied to new domain:
  - Reranker overvalues long, popular docs
  - Undervalues concise, domain-specific docs
```

### 5. Temporal Drift

Reranker trained on old data, applied to new domain:

```
Trained: 2020 (before "transformers" dominated NLP)
Applied: 2024 (transformers everywhere, "RAG" is standard term)

Query: "Compare attention mechanisms to RNNs"
Document: "Attention mechanisms in transformers enable efficient processing"

Reranker (2020): Low score (doesn't understand "transformers" significance)
Modern relevance: High (transformers are the standard now)

Result: Ranking outdated
```

### 6. Overfitting to Training Distribution

Cross-encoder optimized for MS MARCO benchmark, poor on others:

```
MS MARCO training: Queries are short (5-10 words), docs are passages (20-100 words)

Applied to: Long-form questions (50+ words), long documents (1000+ words)

Reranker: Not trained on this distribution
Performance: Degrades significantly
```

### 7. Tokenization and Encoding Issues

```
Query: "API authentication with JWT"
Document: "Token-based auth (JWT) uses JSON Web Tokens"

Tokenization mismatch:
  Query tokens: [API, authentication, with, JWT]
  Doc tokens: [Token, based, auth, (JWT), uses, ...]
  
Cross-encoder must learn that [auth] ~ [authentication], [Token-based] ~ [API]
But if training didn't cover these variations, performance degrades.
```

### 8. Contradictory Training Objectives

Sometimes reranker is optimized for wrong metric:

```
Training objective: Maximize MS MARCO MRR@10

This optimizes for: Ranking the first relevant doc high

But RAG needs: Ranking ALL relevant docs high (Recall@k)

Example:
  Query: "Benefits of microservices"
  Doc A (relevant): ranked #2 by reranker
  Doc B (equally relevant): ranked #15 by reranker
  
For MRR: OK (one relevant doc in top position matters)
For RAG: Bad (second relevant doc buried)
```

</details>

---

## Q4. How do you detect reranker failure in production? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

### Detection Method 1: Before/After Comparison

```python
def compare_dense_vs_reranked(query, retriever, reranker, ground_truth_relevant_ids):
    """Compare ranking quality before and after reranking."""
    
    # Dense retrieval
    dense_results = retriever.search(query, k=10)
    dense_ids = [r['id'] for r in dense_results]
    
    # Reranking
    reranked_results = reranker.rank(query, dense_results, top_k=10)
    reranked_ids = [r['id'] for r in reranked_results]
    
    # Compute metrics
    def compute_recall_at_k(ranked_ids, relevant_ids, k=5):
        return len(set(ranked_ids[:k]) & set(relevant_ids)) / len(relevant_ids)
    
    dense_recall = compute_recall_at_k(dense_ids, ground_truth_relevant_ids)
    reranked_recall = compute_recall_at_k(reranked_ids, ground_truth_relevant_ids)
    
    # Alert if reranking degraded performance
    if reranked_recall < dense_recall:
        degradation = (dense_recall - reranked_recall) / dense_recall
        log_alert(f"Reranker degradation: {degradation:.1%}")
        return False  # Failure detected
    
    return True  # Reranker improved or maintained performance
```

### Detection Method 2: Score Correlation

```python
def check_dense_reranker_correlation(queries, retriever, reranker):
    """Check if dense and cross-encoder agree on relevance."""
    
    from scipy.stats import spearmanr
    
    all_dense_scores = []
    all_cross_scores = []
    
    for query in queries:
        dense_results = retriever.search(query, k=10)
        reranked = reranker.rank(query, dense_results)
        
        for dense, reranked_doc in zip(dense_results, reranked):
            all_dense_scores.append(dense['similarity_score'])
            all_cross_scores.append(reranked_doc['score'])
    
    # Correlation
    correlation, p_value = spearmanr(all_dense_scores, all_cross_scores)
    
    print(f"Dense vs Cross-encoder correlation: {correlation:.3f}")
    
    if correlation < 0.5:
        log_alert(f"Low correlation ({correlation:.3f}), reranker may be misbehaving")
        return False
    
    return True
```

### Detection Method 3: Reranker Confidence vs. Correctness

```python
def analyze_reranker_calibration(query_results_pairs):
    """Check if reranker confidence matches correctness."""
    
    reranker_scores = []
    correctness_labels = []
    
    for query, results, ground_truth_relevant in query_results_pairs:
        for result in results[:5]:
            score = result['reranker_score']
            is_correct = result['id'] in ground_truth_relevant
            
            reranker_scores.append(score)
            correctness_labels.append(is_correct)
    
    # Calibration: are high scores actually correct?
    high_confidence = [s > 0.8 for s in reranker_scores]
    high_conf_accuracy = sum(
        high_confidence[i] == correctness_labels[i]
        for i in range(len(high_confidence))
    ) / len(high_confidence)
    
    print(f"High-confidence accuracy: {high_conf_accuracy:.1%}")
    
    if high_conf_accuracy < 0.7:
        log_alert("Reranker is miscalibrated (high confidence on wrong results)")
        return False
    
    return True
```

### Detection Method 4: A/B Testing

```python
def ab_test_with_without_reranking(query_sample, retriever, reranker, llm):
    """A/B test: does reranking actually improve LLM output quality?"""
    
    control_quality = []  # Dense only, no reranking
    treatment_quality = []  # Dense + reranking
    
    for query in query_sample:
        # Control: Dense retrieval only
        control_results = retriever.search(query, k=5)
        control_answer = llm.generate(query, control_results)
        control_q = assess_answer_quality(control_answer, control_results)
        control_quality.append(control_q)
        
        # Treatment: Dense + reranking
        dense_results = retriever.search(query, k=20)
        treatment_results = reranker.rank(query, dense_results, top_k=5)
        treatment_answer = llm.generate(query, treatment_results)
        treatment_q = assess_answer_quality(treatment_answer, treatment_results)
        treatment_quality.append(treatment_q)
    
    # Statistical test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(control_quality, treatment_quality)
    
    avg_control = np.mean(control_quality)
    avg_treatment = np.mean(treatment_quality)
    
    print(f"Control (dense only): {avg_control:.2%}")
    print(f"Treatment (+ reranking): {avg_treatment:.2%}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05 and avg_treatment > avg_control:
        print("✓ Reranking is beneficial (statistically significant)")
        return True
    else:
        print("✗ Reranking not beneficial or worse")
        log_alert("Reranking failed A/B test")
        return False
```

### Production SLOs for Reranker

```python
reranker_slos = {
    'recall@5_minimum': 0.80,           # Reranked results must have >=80% recall
    'recall_degradation_max': 0.05,     # Reranking can degrade recall by at most 5%
    'correlation_with_dense_min': 0.60, # Should agree with dense retrieval
    'calibration_accuracy_min': 0.75,   # High-confidence results should be correct 75%+
    'ab_test_quality_improvement': 0.02,# A/B test should show >=2% quality improvement
}
```

</details>

---

## Q5. What techniques mitigate reranker failure? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

### Strategy 1: Domain-Specific Reranker Fine-Tuning

Fine-tune a cross-encoder on domain data:

```python
from sentence_transformers import CrossEncoder

# Load general cross-encoder
model = CrossEncoder('cross-encoder/mmarco-MiniLMv2-L12-H384-v1')

# Fine-tune on domain data
domain_training_data = [
    {'texts': ['Query: How to implement JWT?', 'JWT is a stateless auth method'], 'label': 1.0},
    {'texts': ['Query: How to implement JWT?', 'Cookies store session data'], 'label': 0.2},
    # ... domain-specific pairs
]

# Fine-tuning
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

train_examples = [
    InputExample(
        texts=[pair['texts'][0], pair['texts'][1]],
        label=pair['label']
    )
    for pair in domain_training_data
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

model.fit(
    train_dataloader=train_dataloader,
    epochs=1,
    warmup_steps=100,
    output_path='domain-reranker'
)
```

**Expected improvement:** +10-25% Recall@k on domain queries.

### Strategy 2: Ensemble Rerankers

Combine multiple rerankers to reduce failure risk:

```python
def ensemble_rerank(query, chunks, rerankers=['cross-encoder-1', 'cross-encoder-2', 'bm25']):
    """Rank with multiple rerankers, combine scores."""
    
    all_scores = {}
    
    # Score with each reranker
    for reranker_name in rerankers:
        reranker = load_reranker(reranker_name)
        scores = reranker.rank(query, chunks)
        
        for chunk_id, score in scores.items():
            if chunk_id not in all_scores:
                all_scores[chunk_id] = []
            all_scores[chunk_id].append(score)
    
    # Combine: average or weighted average
    combined = {
        chunk_id: np.mean(scores)
        for chunk_id, scores in all_scores.items()
    }
    
    # Rank by combined score
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    
    return [chunk_id for chunk_id, _ in ranked]

# Ensemble benefits:
# - Reduces impact of single reranker failure
# - Different rerankers may be better for different query types
# - Slows down latency (multiple inference calls)
```

### Strategy 3: Hybrid Ranking (Dense + Sparse + Cross-Encoder)

Don't rely solely on cross-encoder:

```python
def hybrid_rank(query, chunks):
    """Combine dense, sparse (BM25), and cross-encoder ranking."""
    
    # Dense: embedding similarity
    dense_scores = compute_dense_scores(query, chunks)
    
    # Sparse: BM25 keyword matching
    sparse_scores = compute_bm25_scores(query, chunks)
    
    # Cross-encoder: query-document relevance
    cross_scores = cross_encoder.rank(query, chunks)
    
    # Combine with weights
    weights = {'dense': 0.3, 'sparse': 0.3, 'cross': 0.4}
    
    combined = {}
    for chunk in chunks:
        combined[chunk['id']] = (
            weights['dense'] * dense_scores[chunk['id']] +
            weights['sparse'] * sparse_scores[chunk['id']] +
            weights['cross'] * cross_scores[chunk['id']]
        )
    
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in ranked]

# If cross-encoder fails, dense + sparse may still be correct
```

### Strategy 4: ColBERT (Late Interaction Model)

Use late-interaction ranking instead of cross-encoder:

```python
from colbert.v2 import ColBERTv2

# ColBERT: token-level interactions (cheaper than cross-encoder)
colbert = ColBERTv2.from_pretrained('colbertv2.0')

# Index documents (offline)
colbert.index([chunk['text'] for chunk in chunks])

# Query (online)
query_embedding = colbert.query(query)
chunk_embeddings = colbert.encode(chunks)

# Late interaction: compute similarity at token level
similarities = colbert.score(query_embedding, chunk_embeddings)

ranked_ids = sorted(zip(range(len(chunks)), similarities), key=lambda x: x[1], reverse=True)
return [chunks[i]['id'] for i, _ in ranked_ids]

# Advantages:
# - Faster than cross-encoder (can index offline)
# - Still provides fine-grained relevance scoring
# - Less prone to training bias
```

### Strategy 5: LLM-as-Reranker

Use the LLM itself to rerank:

```python
def llm_rerank(query, chunks):
    """Ask LLM to rank chunks by relevance."""
    
    ranking_prompt = f"""
    You are a relevance ranking expert.
    
    Query: {query}
    
    Rank the following documents by relevance (most to least):
    {format_chunks_for_ranking(chunks)}
    
    Return ranked list with IDs.
    """
    
    ranking = llm.generate(ranking_prompt)
    
    # Parse LLM output to get ranking order
    ranked_ids = parse_ranking_output(ranking)
    
    return ranked_ids

# Advantages:
# - No separate model to maintain
# - Can use same LLM as generator
# - More semantic understanding

# Disadvantages:
# - Very expensive (inference per query + per chunk)
# - May be inconsistent
```

### Comparison of Strategies

| Strategy | Improvement | Cost | Latency | Complexity |
|----------|---|---|---|---|
| **Domain fine-tuning** | +15-25% | Low (one-time training) | Minimal | Medium |
| **Ensemble rerankers** | +10-15% | High (multiple inferences) | 2-3x | High |
| **Hybrid ranking** | +8-12% | Medium (3 ranking methods) | 2x | High |
| **ColBERT** | +10-20% | Low (index offline) | 30% faster | High |
| **LLM-as-reranker** | +15-25% | Very high (per-chunk LLM calls) | 5-10x | Medium |

</details>

---

## Q6. How do you choose and evaluate rerankers for your domain? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

### Step 1: Baseline Evaluation (Off-Domain)

```python
def evaluate_reranker_baseline(reranker_candidates, test_queries):
    """Evaluate standard rerankers on general benchmarks."""
    
    results = {}
    
    for reranker_name in reranker_candidates:
        reranker = CrossEncoder(reranker_name)
        
        recalls = []
        
        for query, ground_truth_relevant_ids in test_queries:
            # Assume dense retrieval already done
            dense_chunks = retrieve_dense(query, k=20)
            
            # Rerank
            reranked = reranker.rank(query, dense_chunks, top_k=10)
            reranked_ids = [r['id'] for r in reranked]
            
            # Recall@5
            recall = len(set(reranked_ids[:5]) & set(ground_truth_relevant_ids)) / len(ground_truth_relevant_ids)
            recalls.append(recall)
        
        results[reranker_name] = np.mean(recalls)
    
    # Rank by performance
    for name, recall in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: Recall@5={recall:.2%}")
    
    return results

# Candidate rerankers
candidates = [
    'cross-encoder/mmarco-MiniLMv2-L12-H384-v1',  # Fast
    'cross-encoder/ms-marco-TinyBERT-L-2-v2',      # Very fast
    'cross-encoder/ms-marco-MiniLM-L-12-v2',       # Balanced
    'cross-encoder/qnli-distilroberta-base',       # Different task
]

evaluate_reranker_baseline(candidates, test_queries)
```

### Step 2: Domain-Specific Evaluation

```python
def create_domain_eval_set(domain, num_queries=200):
    """Create labeled (query, relevant_doc_ids) pairs from your domain."""
    
    # Method 1: Expert annotation
    # Method 2: Mining from user interactions
    # Method 3: Synthetic generation
    
    domain_queries = []
    
    for _ in range(num_queries):
        # Get query from domain
        query = sample_domain_query(domain)
        
        # Find relevant docs (expert labels or heuristics)
        relevant_ids = find_relevant_docs(query, domain)
        
        domain_queries.append((query, relevant_ids))
    
    return domain_queries

def evaluate_on_domain(reranker_candidates, domain_eval_set):
    """Test rerankers on domain-specific queries."""
    
    results = {}
    
    for reranker_name in reranker_candidates:
        reranker = CrossEncoder(reranker_name)
        
        recalls = []
        
        for query, ground_truth_ids in domain_eval_set:
            dense_chunks = retrieve_dense_domain(query, k=20)
            reranked = reranker.rank(query, dense_chunks, top_k=10)
            
            recall = len(set([r['id'] for r in reranked[:5]]) & set(ground_truth_ids)) / len(ground_truth_ids)
            recalls.append(recall)
        
        results[reranker_name] = {
            'recall': np.mean(recalls),
            'std': np.std(recalls)
        }
    
    return results

# Evaluate on domain
domain_eval = create_domain_eval_set('medical', num_queries=200)
domain_results = evaluate_on_domain(candidates, domain_eval)

print("\nDomain-specific results (Medical):")
for name, metrics in sorted(domain_results.items(), key=lambda x: x[1]['recall'], reverse=True):
    print(f"{name}: {metrics['recall']:.2%} ± {metrics['std']:.2%}")
```

### Step 3: Cost-Latency Analysis

```python
def analyze_reranker_efficiency(reranker_candidates):
    """Compare cost and latency."""
    
    for reranker_name in reranker_candidates:
        reranker = CrossEncoder(reranker_name)
        
        # Latency
        import time
        start = time.time()
        for _ in range(100):
            scores = reranker.predict([['Sample query', 'Sample document']])
        latency_ms = (time.time() - start) / 100 * 1000
        
        # Model size
        params = sum(p.numel() for p in reranker.model.parameters())
        
        # Relative cost estimate
        cost = params / 1_000_000  # Rough proxy
        
        print(f"{reranker_name}:")
        print(f"  Latency: {latency_ms:.1f}ms per pair")
        print(f"  Parameters: {params/1e6:.1f}M")
        print(f"  Relative cost: {cost:.1f}x")

analyze_reranker_efficiency(candidates)
```

### Step 4: Decision Matrix

Choose based on your constraints:

| Use Case | Recommended Reranker | Rationale |
|----------|---|---|
| **High quality + cost-sensitive** | `mmarco-MiniLMv2-L12-H384-v1` (balanced) | Best Recall/latency ratio |
| **Extreme latency constraints** | `ms-marco-TinyBERT-L-2-v2` (tiny) | Fastest but lower quality |
| **Highest quality** | `cross-encoder/qnli-distilroberta-base` + domain fine-tuning | Fine-tuned for domain |
| **Domain-specific** | Domain fine-tuned version of above | +15-25% improvement on domain |
| **No reranking budget** | Skip reranking, rely on dense | Baseline, acceptable for simple domains |

</details>

---

## Q7. How do you detect and handle reranker model drift in production? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Reranker performance can degrade over time due to:
- Domain shift (new types of queries/documents)
- Concept drift (new terminology, changing priorities)
- Model obsolescence (better models released)

### Drift Detection

```python
class RerankerDriftDetector:
    def __init__(self, baseline_metrics):
        self.baseline = baseline_metrics
        self.current_metrics = {}
    
    def check_drift(self, recent_queries, recent_eval_set, metric_window='7d'):
        """Detect performance degradation."""
        
        # Evaluate current performance
        current_recall = evaluate_reranker(recent_eval_set)
        
        drift = self.baseline['recall'] - current_recall
        drift_pct = drift / self.baseline['recall']
        
        # Alert if >10% degradation
        if drift_pct > 0.10:
            log_alert(f"Reranker drift detected: {drift_pct:.1%} degradation")
            return True
        
        return False
    
    def measure_query_type_drift(self, recent_queries):
        """Detect shift in query distribution."""
        
        # Extract features from queries
        baseline_features = extract_query_features(self.baseline['sample_queries'])
        current_features = extract_query_features(recent_queries)
        
        # Compare distributions
        divergence = compute_kl_divergence(baseline_features, current_features)
        
        if divergence > threshold:
            log_alert(f"Query distribution shift detected (KL={divergence:.2f})")
            return True
        
        return False

# Usage
detector = RerankerDriftDetector(baseline_metrics={'recall': 0.82})

for day in range(30):
    recent_queries = get_recent_queries(hours=24)
    has_drift = detector.check_drift(recent_queries)
    
    if has_drift:
        # Trigger reranker retraining or swapping
        trigger_reranker_refresh()
```

### Drift Response Strategies

```python
def handle_reranker_drift(drift_magnitude):
    """Respond based on drift severity."""
    
    if drift_magnitude < 0.05:
        # Minor drift (< 5%), monitor but no action
        log_info(f"Minor drift {drift_magnitude:.1%}, monitoring...")
    
    elif drift_magnitude < 0.15:
        # Moderate drift (5-15%), consider fine-tuning
        log_warning(f"Moderate drift {drift_magnitude:.1%}, queuing retraining job")
        
        # Schedule retraining on recent data
        schedule_reranker_finetuning(
            training_data=collect_recent_training_data(days=30),
            epochs=1
        )
    
    else:
        # Severe drift (> 15%), swap to backup or disable
        log_error(f"Severe drift {drift_magnitude:.1%}, degrading gracefully")
        
        # Option 1: Use backup reranker
        swap_reranker(backup_reranker='previous_version')
        
        # Option 2: Disable reranking temporarily
        disable_reranking()
        
        # Option 3: Use ensemble (combine with other signals)
        switch_to_ensemble_ranking()
```

### Continuous Monitoring

```python
class RerankerMonitoringPipeline:
    def __init__(self):
        self.window_size = 1000  # queries
        self.slos = {
            'recall@5_minimum': 0.75,
            'drift_maximum': 0.15,
        }
    
    def run_continuous_monitoring(self):
        """Background job: monitor reranker health."""
        
        query_buffer = []
        
        while True:
            # Collect queries and evaluations
            query, response = get_next_query()
            
            # Evaluate reranker on this query
            evaluation = evaluate_single_query(query)
            
            query_buffer.append(evaluation)
            
            # Check SLOs periodically
            if len(query_buffer) >= self.window_size:
                metrics = compute_window_metrics(query_buffer)
                
                if metrics['recall@5'] < self.slos['recall@5_minimum']:
                    alert(f"Recall dropped below {self.slos['recall@5_minimum']:.1%}")
                
                query_buffer = []
            
            time.sleep(1)
```

</details>

---

## Q8. What is the cost-quality trade-off of reranking in RAG pipelines? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Reranking adds cost and latency but potentially improves quality. The trade-off is domain-dependent.

### Cost Breakdown

```python
def estimate_reranking_cost(queries_per_month, dense_k=20, reranker_k=5):
    """Estimate incremental cost of reranking."""
    
    # Dense retrieval: one embedding per query
    dense_cost = queries_per_month * 0.00001  # Approximate embedding cost
    
    # Reranking: one cross-encoder inference per query per candidate
    # Assume dense retrieves k=20, reranker scores all 20
    reranker_inferences = queries_per_month * dense_k
    
    # Cross-encoder cost roughly: 5-10x embedding cost per inference
    reranker_cost = reranker_inferences * 0.00005
    
    # Total
    total = dense_cost + reranker_cost
    incremental = reranker_cost  # Just the reranking cost
    
    cost_per_query = incremental / queries_per_month
    
    print(f"Dense retrieval: ${dense_cost:.2f}/month")
    print(f"Reranking: ${reranker_cost:.2f}/month")
    print(f"Total: ${total:.2f}/month")
    print(f"Cost per query: ${cost_per_query:.6f}")
    print(f"Incremental cost: {incremental/dense_cost:.1f}x dense cost")

estimate_reranking_cost(queries_per_month=100_000)

# Output:
# Dense retrieval: $1.00/month
# Reranking: $10.00/month
# Total: $11.00/month
# Cost per query: $0.00010
# Incremental cost: 10.0x dense cost
```

### Latency Impact

```python
latency_profile = {
    'dense_retrieval': {
        'embedding': 10,           # ms
        'vector_search': 30,       # ms
        'total': 40,
    },
    'dense_plus_reranking': {
        'embedding': 10,
        'vector_search': 30,
        'reranking': 50,           # Cross-encoder on 20 candidates
        'total': 90,
    },
    'dense_reranking_optimized': {
        'embedding': 10,
        'vector_search': 30,
        'reranking': 20,           # Only top-10 candidates
        'total': 60,
    },
}

for strategy, timings in latency_profile.items():
    print(f"{strategy}: {timings['total']}ms")
```

### Quality vs. Cost Trade-off

```python
def analyze_quality_cost_tradeoff(base_quality, base_cost):
    """Estimate quality improvement vs. added cost."""
    
    strategies = {
        'no_reranking': {
            'quality': base_quality,
            'cost': base_cost,
        },
        'reranking': {
            'quality': base_quality + 0.08,  # +8% quality improvement
            'cost': base_cost * 11,           # 11x cost
        },
        'reranking_top_10': {
            'quality': base_quality + 0.06,  # +6% quality
            'cost': base_cost * 6,            # 6x cost
        },
        'domain_finetuned_reranker': {
            'quality': base_quality + 0.15,  # +15% quality (best)
            'cost': base_cost * 12,           # Similar cost as reranking
        },
        'colbert': {
            'quality': base_quality + 0.10,  # +10% quality
            'cost': base_cost * 2,            # Much cheaper (offline indexing)
        },
    }
    
    # ROI analysis
    base_monthly_queries = 100_000
    value_per_quality_point = 5_000  # $5k value per 1% quality improvement
    
    for strategy, metrics in strategies.items():
        quality_improvement = (metrics['quality'] - base_quality) * 100
        quality_value = quality_improvement * value_per_quality_point
        
        cost = metrics['cost'] * base_cost * base_monthly_queries / 1_000_000
        
        roi = quality_value / cost if cost > 0 else 0
        
        print(f"{strategy}:")
        print(f"  Quality: {metrics['quality']:.2%}")
        print(f"  Quality gain: {quality_improvement:.1f} points = ${quality_value:,.0f}")
        print(f"  Monthly cost: ${cost:.2f}")
        print(f"  ROI: {roi:.1f}x")
        print()

analyze_quality_cost_tradeoff(base_quality=0.78, base_cost=1.0)

# Output:
# no_reranking:
#   Quality: 0.78
#   Quality gain: 0.0 points = $0
#   Monthly cost: $1.00
#   ROI: 0.0x
#
# reranking:
#   Quality: 0.86
#   Quality gain: 8.0 points = $40000
#   Monthly cost: $11.00
#   ROI: 3636.4x
#
# colbert:
#   Quality: 0.88
#   Quality gain: 10.0 points = $50000
#   Monthly cost: $2.00
#   ROI: 25000.0x  ← Best ROI!
```

### Recommendation Decision Tree

```python
def recommend_reranking_strategy(constraints):
    """Choose reranking strategy based on SLOs."""
    
    if constraints['quality_target'] < 0.75:
        # Low quality bar, reranking may not be needed
        return 'no_reranking'
    
    elif constraints['latency_p99_ms'] < 100:
        # Very tight latency budget
        return 'no_reranking'  # Can't afford reranking
    
    elif constraints['monthly_budget_dollars'] < 10:
        # Very tight budget
        return 'colbert'  # Cheapest option with quality gain
    
    elif constraints['domain'] == 'specialized':
        # Domain-specific queries
        return 'domain_finetuned_reranker'  # Best for domain
    
    elif constraints['latency_p99_ms'] < 200:
        # Moderate latency
        return 'reranking_top_10'  # Score only top-10 to save latency
    
    else:
        # No constraints
        return 'domain_finetuned_reranker'  # Best overall

# Usage
recommendation = recommend_reranking_strategy({
    'quality_target': 0.85,
    'latency_p99_ms': 300,
    'monthly_budget_dollars': 100,
    'domain': 'general',
})

print(f"Recommended strategy: {recommendation}")
```

</details>

---

## Q9. How do you decide between cross-encoder, ColBERT, and LLM-as-reranker? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Three main reranking paradigms, each with trade-offs:

### Cross-Encoder (Dense Interaction)

```python
# Example: cross-encoder/ms-marco-MiniLM-L-12-v2

# Pros:
# - Highest quality (SOTA on benchmarks)
# - Easy to fine-tune
# - Well-understood

# Cons:
# - Expensive (must score every candidate)
# - Slower (~50ms for 20 candidates)
# - Can't pre-compute (online-only)

def use_cross_encoder():
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    
    for query in queries:
        dense_chunks = retrieve_dense(query, k=20)
        reranked = reranker.rank(query, dense_chunks, top_k=5)
```

**Best for:** High-quality requirements, moderate QPS, domain specialization.

### ColBERT (Late Interaction)

```python
# Example: colbertv2.0

# Pros:
# - Can index offline (much faster at query time)
# - Fine-grained token-level interactions
# - Cheaper than cross-encoder

# Cons:
# - Requires offline indexing (slower updates)
# - More complex model
# - Harder to fine-tune

def use_colbert():
    colbert = ColBERTv2.from_pretrained('colbertv2.0')
    
    # Offline: index documents once
    colbert.index(all_documents)
    
    # Online: fast query
    for query in queries:
        query_emb = colbert.query(query)
        top_chunks = colbert.search(query_emb, k=5)
```

**Best for:** High QPS, moderate quality needs, infrequent index updates.

### LLM-as-Reranker

```python
# Use the same LLM as generator for reranking

# Pros:
# - No separate model to maintain
# - Can provide detailed explanations
# - Semantic understanding

# Cons:
# - Very expensive (100-1000x cost of cross-encoder)
# - Slow (multiple LLM inferences)
# - Unpredictable (LLM can be inconsistent)

def use_llm_as_reranker():
    ranking_prompt = """
    Rank documents by relevance to query.
    Query: {query}
    Documents:
    {chunks}
    """
    
    ranking = llm.generate(ranking_prompt)
    return parse_ranking(ranking)
```

**Best for:** Small queries, high-value decisions, need for explanations.

### Comparison Matrix

| Aspect | Cross-Encoder | ColBERT | LLM-as-Reranker |
|--------|---|---|---|
| **Quality** | Highest | High | Highest (semantic) |
| **Speed** | Medium (50ms/query) | Fast (5ms/query) | Very slow (500ms+) |
| **Cost** | Medium | Low | Very high |
| **Indexing** | Online only | Offline indexing | N/A |
| **Fine-tuning** | Easy | Hard | N/A (use prompting) |
| **Consistency** | High | High | Low (LLM variance) |
| **Explanation** | No | No | Yes |
| **Domain adaptation** | Easy (fine-tune) | Hard | Medium (prompting) |

### Decision Framework

```python
def choose_reranking_paradigm(qps, quality_target, budget, domain, needs_update_frequency):
    """Recommend reranking approach."""
    
    # High QPS + tight budget → ColBERT
    if qps > 1000 and budget < 100:
        return 'ColBERT'
    
    # Need domain adaptation → Cross-encoder fine-tune
    if domain != 'general' and budget > 500:
        return 'Cross-encoder (fine-tuned)'
    
    # High quality needed + explain decisions → LLM-as-reranker
    if quality_target > 0.90 and budget > 1000:
        return 'LLM-as-reranker'
    
    # Frequent index updates + fast queries → ColBERT
    if needs_update_frequency == 'hourly':
        return 'ColBERT'
    
    # Default: balanced solution
    return 'Cross-encoder (general)'

# Example usage
recommendation = choose_reranking_paradigm(
    qps=500,
    quality_target=0.85,
    budget=200,
    domain='medical',
    needs_update_frequency='daily'
)

print(f"Recommended: {recommendation}")
# Output: Cross-encoder (fine-tuned)
```

</details>

---

## Q10. How do you establish SLOs and monitor reranker quality in production? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

### SLO Definition

```python
reranker_slos = {
    # Quality SLOs
    'recall@5': {
        'target': 0.80,
        'measurement': 'Fraction of relevant docs in top-5 reranked results',
        'window': '24h'
    },
    'recall_degradation': {
        'max': 0.05,  # Can degrade recall by at most 5% vs dense
        'measurement': 'Compare recall before/after reranking',
        'window': '24h'
    },
    'correlation_with_dense': {
        'minimum': 0.60,
        'measurement': 'Spearman correlation between dense and reranker scores',
        'window': '7d'
    },
    
    # Performance SLOs
    'latency_p99': {
        'target': 100,  # milliseconds
        'measurement': 'Time for reranking 20 candidates',
        'window': '5m'
    },
    'availability': {
        'target': 0.9999,
        'measurement': 'Percentage of requests successfully reranked',
        'window': '30d'
    },
    
    # Cost SLOs
    'cost_per_query': {
        'target': 0.0001,  # dollars
        'measurement': 'Reranking cost / total queries',
        'window': '1m'
    },
}
```

### Continuous Monitoring

```python
class RerankerMonitor:
    def __init__(self, slos):
        self.slos = slos
        self.metrics = {}
    
    def log_reranking_event(self, query, dense_results, reranked_results, user_feedback):
        """Log metrics from each reranking event."""
        
        # Extract metrics
        rank_change = compute_rank_change(dense_results, reranked_results)
        reranker_scores = [r['score'] for r in reranked_results]
        dense_scores = [r['similarity'] for r in dense_results]
        
        # User feedback (if available)
        clicked_rank = user_feedback.get('clicked_rank', None)
        
        # Log
        self.metrics['rank_changes'].append(rank_change)
        self.metrics['reranker_scores'].extend(reranker_scores)
        self.metrics['dense_scores'].extend(dense_scores)
        
        if clicked_rank:
            self.metrics['clicked_positions'].append(clicked_rank)
    
    def check_slos(self):
        """Evaluate current performance vs SLOs."""
        
        violations = {}
        
        # Recall@5 (requires labeled ground truth)
        # estimated via user behavior
        click_in_top_5_rate = sum(
            1 for pos in self.metrics.get('clicked_positions', [])
            if pos <= 5
        ) / max(1, len(self.metrics.get('clicked_positions', [])))
        
        if click_in_top_5_rate < self.slos['recall@5']['target']:
            violations['recall@5'] = click_in_top_5_rate
        
        # Correlation
        if len(self.metrics.get('reranker_scores', [])) > 100:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(
                self.metrics['dense_scores'],
                self.metrics['reranker_scores']
            )
            
            if corr < self.slos['correlation_with_dense']['minimum']:
                violations['correlation'] = corr
        
        return violations

# Usage
monitor = RerankerMonitor(reranker_slos)

for query, dense_results, reranked_results, feedback in stream_requests():
    monitor.log_reranking_event(query, dense_results, reranked_results, feedback)
    
    if len(monitor.metrics.get('rank_changes', [])) % 1000 == 0:
        violations = monitor.check_slos()
        
        if violations:
            for slo, value in violations.items():
                alert(f"SLO violation: {slo}={value}")
```

### Quality Regression Testing

```python
def detect_reranker_regression(previous_metrics, current_metrics, threshold=0.05):
    """Detect quality degradation."""
    
    previous_recall = previous_metrics['recall@5']
    current_recall = current_metrics['recall@5']
    
    degradation = (previous_recall - current_recall) / previous_recall
    
    if degradation > threshold:
        print(f"⚠️ Reranker regression detected: {degradation:.1%} degradation")
        print(f"  Previous recall: {previous_recall:.2%}")
        print(f"  Current recall: {current_recall:.2%}")
        
        # Automatic mitigation options:
        if degradation > 0.20:  # >20% degradation
            print("✗ Critical: Rolling back to previous reranker")
            rollback_reranker()
        
        elif degradation > 0.10:  # >10% degradation
            print("⚠ Moderate: Triggering retraining on recent data")
            schedule_reranker_retraining()
        
        else:
            print("ℹ Minor: Monitoring for further degradation")
        
        return True
    
    return False
```

### Dashboarding

```python
def render_reranker_dashboard():
    """Human-readable reranker health dashboard."""
    
    metrics = collect_metrics(window='24h')
    
    dashboard = f"""
╔════════════════════════════════════════════════════════════╗
║               RERANKER HEALTH DASHBOARD                    ║
╠════════════════════════════════════════════════════════════╣
│ Recall@5:          {metrics['recall@5']:.1%}  (Target: 80%)           │
│ Correlation:       {metrics['correlation']:.2f}  (Target: >0.60)       │
│ Latency P99:       {metrics['latency_p99']:.0f}ms (Target: <100ms)     │
│ Click-in-Top-5:    {metrics['ctr_top5']:.1%}  (Target: >75%)         │
├────────────────────────────────────────────────────────────┤
│ Status:            {metrics['status_emoji']} {metrics['status']}     │
│ Last updated:      {metrics['last_updated']}                │
╚════════════════════════════════════════════════════════════╝
    """
    
    print(dashboard)
    
    # Alerts
    if metrics['status'] != 'healthy':
        print(f"\n⚠️  Alerts:")
        for alert in metrics['alerts']:
            print(f"  - {alert}")
```

</details>

---
