# 11 — Adaptive RAG

> Dynamically selects no-retrieval, single-hop, or multi-hop strategy based on query complexity at runtime.

---

## Q1. What is Adaptive RAG and how does it differ from fixed-pipeline RAG? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Adaptive RAG is a retrieval strategy that dynamically routes queries to different retrieval depths based on an estimated complexity classifier, rather than applying a uniform pipeline to all queries.

**Fixed-pipeline RAG** executes the same retrieval strategy (e.g., always retrieve top-k, always do multi-hop) regardless of the query. **Adaptive RAG** uses a learned classifier to predict query complexity and routes accordingly:

- **No-retrieval path** — For simple queries (e.g., "What is X?") that the LLM can answer from parametric knowledge
- **Single-hop path** — For moderately complex queries requiring one retrieval step
- **Multi-hop path** — For complex reasoning queries requiring multiple retrieval and reasoning steps

This reduces latency and cost for simple queries while maintaining answer quality for hard questions. It is particularly effective when query complexity varies widely in production workloads.

</details>

---

## Q2. How does a query complexity classifier work in Adaptive RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A query complexity classifier predicts whether a given query is simple, moderate, or complex, using features like query length, linguistic markers, and semantic similarity to known simple/complex templates.

**Common approaches:**

1. **Keyword-based heuristics** — Flag queries with multi-hop keywords (e.g., "compare", "across") or multiple entities as complex.
2. **Supervised classifier** — Train a linear or neural model on labeled (query, complexity) pairs. Features can include:
   - Query length and token count
   - Presence of comparative/temporal/causal keywords
   - Named entity count
   - Embedding similarity to known simple vs. complex query templates
3. **LLM-as-judge** — Use a lightweight LLM or prompt to estimate complexity, balancing cost vs. accuracy.
4. **Confidence-based** — Let the LLM attempt to answer without retrieval and measure confidence. If confidence is below a threshold, escalate to retrieval.

**Training data** — Typically 500–2000 labeled (query, complexity, answer quality with/without retrieval) triplets from past user interactions or synthetic data.

The classifier runs *before* retrieval, so it must be fast (<10ms overhead to be practical).

</details>

---

## Q3. What are the three retrieval strategies in Adaptive RAG and when is each chosen? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The three strategies and their selection criteria are:

| Strategy | Trigger | Typical Queries | Benefit |
|---|---|---|---|
| **No-Retrieval** | Low complexity score (< 0.33) | "What is X?", "Define Y", factoid questions | Minimal latency, zero retrieval cost |
| **Single-Hop** | Moderate complexity (0.33–0.66) | "Where do X typically occur?", "Compare X and Y" | 1 retrieval round, faster than multi-hop |
| **Multi-Hop** | High complexity (> 0.66) | "How does X relate to Y in the context of Z?", reasoning chains | Multiple retrieval + reasoning steps, highest quality |

**Routing logic:**

1. Classifier produces a complexity score (0–1) for the input query.
2. Use threshold-based routing: if score < t1 → no-retrieval; if t1 ≤ score < t2 → single-hop; if score ≥ t2 → multi-hop.
3. Thresholds are tuned on a held-out validation set balancing latency, cost, and answer quality.

**End-to-end flow:**

```
User Query
    │
    ├─ Complexity Classifier
    │     │
    │     ├─ Score < 0.33 → Direct LLM generation (no retrieval)
    │     ├─ 0.33 ≤ Score < 0.66 → Retrieve 1x, then generate
    │     └─ Score ≥ 0.66 → Iterative multi-hop retrieval + generation
    │
    └─ Answer
```

</details>

---

## Q4. How is FLARE integrated into Adaptive RAG for uncertainty-triggered retrieval? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

FLARE (Forward-Looking Active Retrieval Augmented Generation) is a method that triggers retrieval *dynamically during generation*, based on the model's predictive uncertainty about upcoming tokens. Adaptive RAG can integrate FLARE to refine its initial routing decision or to escalate from no-retrieval to retrieval mid-generation.

**Integration approach:**

1. **Initial routing** — Complexity classifier routes to no-retrieval or single-hop as usual.
2. **During generation** — Monitor the LLM's confidence on each generated token.
3. **Uncertainty trigger** — If confidence drops below a threshold (e.g., next token probability < 0.5), pause generation and retrieve documents related to the low-confidence phrase.
4. **Augment context** — Append retrieved documents to the prompt and resume generation.

**Benefits:**

- Catches cases where the classifier underestimated complexity.
- Reduces unnecessary retrieval for simple questions (no-retrieval path is tried first).
- Enables late-stage correction if the LLM begins hallucinating.

**Example:** A query "Who won the 2024 World Cup?" is classified as simple. The LLM starts: "As of my training data..." but uncertainty spikes when generating the year. FLARE triggers retrieval of recent sports news and corrects the answer.

Combining Adaptive (upfront routing) + FLARE (runtime uncertainty) yields the best latency and quality balance.

</details>

---

## Q5. How do you train and evaluate a query complexity classifier for Adaptive RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Training data collection:**

1. Sample 500–2000 queries from your production logs or a representative dataset.
2. Manually annotate each query with a complexity label: Simple (0), Moderate (1), or Complex (2).
   - Simple: facts, definitions, single-entity questions.
   - Moderate: comparisons, aggregations, single-hop reasoning.
   - Complex: multi-hop reasoning, temporal reasoning, constraint satisfaction.
3. Optionally, also log the ground-truth retrieval depth required to answer each query well.

**Classifier architecture:**

A lightweight model such as:
- **Logistic regression** on handcrafted features (length, entity count, keyword TF-IDF).
- **Shallow neural net** (1–2 hidden layers) on query embeddings.
- **Fine-tuned small LM** (e.g., DistilBERT, MobileBERT) for more accuracy at higher cost.

For production, prefer simpler models with <10ms latency.

**Training procedure:**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(
    query_features, complexity_labels, test_size=0.2
)

clf = RandomForestClassifier(n_estimators=50, max_depth=5)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
```

**Evaluation metrics:**

- **Accuracy** — Overall classification correctness.
- **Per-class precision/recall** — Ensure the classifier does not systematically mis-label complex queries as simple (dangerous for quality).
- **Latency** — Measure classifier inference time in the serving pipeline.
- **End-to-end RAG quality** — Log F1/NDCG/answer-quality metrics binned by predicted complexity; verify that simpler queries routed to no-retrieval maintain acceptable quality.

Monitor classifier performance quarterly and retrain as query patterns shift.

</details>

---

## Q6. How does confidence-based no-retrieval skipping reduce latency and cost? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Confidence-based no-retrieval skipping leverages the LLM's token-level confidence estimates to decide, on a per-query basis, whether retrieval is necessary. Queries the model feels confident about skip retrieval entirely, cutting latency by ~500ms–2s and eliminating embedding/vector DB costs.

**Implementation:**

1. **First-pass generation without retrieval** — Feed the query directly to the LLM and collect output tokens with their probabilities.
2. **Aggregate confidence** — Compute a single confidence score from the token probabilities:
   ```
   confidence = exp(mean(log(p_i)))  # geometric mean of token probs
   ```
   Or use the minimum token probability as a conservative lower bound.
3. **Confidence threshold** — If confidence > threshold (e.g., 0.7), return the answer. Otherwise, retrieve and regenerate.
4. **Empirical threshold tuning** — On a validation set, measure F1/BLEU for each threshold and pick the one maximizing F1 subject to a latency constraint.

**Cost and latency impact:**

| Scenario | Latency (ms) | Retrieval Cost | Confidence Score |
|----------|--------------|-----------------|------------------|
| No-retrieval (direct) | 100 | $0 | High (>0.75) |
| With retrieval | 1500 | $0.05 | Low (<0.75) |
| Skip rate (% of queries) | — | 30–50% reduction | Depends on query distribution |

For a 30% skip rate, total inference cost drops ~15% and median latency improves ~300ms.

**Trade-offs:**

- **False positives** — Low confidence on a query the model can actually answer (requires retrieval unnecessarily). Mitigation: conservative threshold.
- **False negatives** — High confidence on a query the model cannot answer well (skips retrieval when needed). Mitigation: use a mixture of confidence signals (token probs + semantic uncertainty).

Confidence-based skipping pairs well with the Adaptive RAG classifier: the classifier provides a coarse routing decision, and confidence gates fine-grained skipping within each tier.

</details>

---

## Q7. How do you implement self-consistency scoring to select among multi-hop retrieval candidates? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Self-consistency scoring evaluates multiple candidate retrieval sequences (e.g., different intermediate query formulations, different retrieve-and-think steps) and selects the trajectory that produces the most consistent and coherent intermediate reasoning, without requiring ground-truth labels.

**Approach:**

1. **Candidate generation** — For a complex query, generate k different chains of retrieval + reasoning:
   - Retrieval sequence A: retrieve for subquery A1 → reason → retrieve for A2 → answer.
   - Retrieval sequence B: retrieve for subquery B1 → reason → retrieve for B2 → answer.
   - ... (e.g., k=3).

2. **Self-consistency metrics** — Score each trajectory by:
   - **Semantic coherence** — Measure how well consecutive reasoning steps align (embeddings of consecutive reasoning strings are near each other in latent space).
   - **Document relevance agreement** — Score how much the retrieved documents at each step reinforce each other (use citation overlap, entity overlap, or embedding similarity).
   - **Answer stability** — If you regenerate the answer from each trajectory, how similar are the final answers? (e.g., BLEU or semantic similarity).

3. **Voting / aggregation** — Pick the trajectory with the highest aggregate score, or ensemble the answers from all trajectories.

**Example implementation:**

```python
from sentence_transformers import CrossEncoderModel

def score_trajectory(reasoning_steps, retrieved_docs):
    coherence_score = 0.0
    for i in range(len(reasoning_steps) - 1):
        # Cross-encoder score between consecutive reasoning steps
        score = cross_encoder.predict(
            [[reasoning_steps[i], reasoning_steps[i+1]]]
        )[0]
        coherence_score += score
    
    doc_relevance = sum(
        cross_encoder.predict([[step, doc] for doc in retrieved_docs])
        for step in reasoning_steps
    ) / len(retrieved_docs)
    
    return coherence_score + doc_relevance

best_trajectory = max(
    trajectories, 
    key=lambda t: score_trajectory(t['steps'], t['docs'])
)
return best_trajectory['answer']
```

**Benefits:**

- Reduces hallucination in multi-hop reasoning without requiring an oracle.
- Adapts to new query types (no task-specific fine-tuning needed).

**Overhead:** Generating k trajectories multiplies compute cost by k; typically k=2–3 is practical.

</details>

---

## Q8. What are the latency vs. accuracy trade-offs of each adaptive strategy tier? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Strategy | Latency (p50) | Cost/Query | Answer Quality (F1) | When to Prefer |
|----------|---------------|-----------|---------------------|-----------------|
| No-Retrieval | ~50ms | $0.00 | 0.65–0.75 (varies by LLM) | Simple factoid queries; cost-sensitive |
| Single-Hop | ~600ms | $0.01–0.02 | 0.75–0.85 | Moderate complexity; balance of speed/quality |
| Multi-Hop | ~2000ms | $0.05–0.10 | 0.85–0.95 | Complex reasoning; quality-critical |

**Empirical trade-off curve:**

For a typical production workload:

```
F1
│
0.95 │                  Multi-Hop ●
     │
0.85 │        Single-Hop ●
     │
0.75 │ No-Retrieval ●
     │
0.65 │
     └──────────────────────────── Latency (ms)
       0    600   1200   2000
```

**Optimal operating point:**

- Query complexity distribution heavily skews simple → weighted average latency favors no-retrieval path.
- If 50% simple, 30% moderate, 20% complex → median latency ≈ 0.5×50 + 0.3×600 + 0.2×2000 ≈ 650ms.

**Strategies to optimize each tier:**

- **No-retrieval** — Use a smaller, faster LLM (e.g., Llama 7B vs. 70B). Add a re-ranker to catch low-confidence predictions.
- **Single-hop** — Cache embeddings of frequently-retrieved documents. Use approximate nearest neighbor search (HNSW, Annoy).
- **Multi-hop** — Prune intermediate retrieval results aggressively. Use speculative decoding to speed up verifier LLM.

**Cost breakdown for 1M queries/month with 50/30/20 distribution:**

- No-retrieval (500k): $0.
- Single-hop (300k): $0.015 × 300k ≈ $4,500.
- Multi-hop (200k): $0.075 × 200k ≈ $15,000.
- **Total:** ~$19,500/month.

The no-retrieval path is the biggest lever for cost reduction.

</details>

---

## Q9. How do you evaluate an Adaptive RAG system using the metrics from the original Adaptive-RAG paper? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The Adaptive-RAG paper proposes a set of metrics to jointly evaluate routing accuracy, retrieval efficiency, and final answer quality:

**1. Routing Accuracy (RA):**

Measure how often the classifier correctly routes a query to the optimal strategy:

```
RA = (# correctly routed queries) / (total queries)
```

A query is "correctly routed" if:
- Assigned to no-retrieval AND the LLM can answer it without retrieval (confidence > threshold or F1 match).
- Assigned to single-hop AND retrieval+generation beats no-retrieval but multi-hop yields only marginal gains.
- Assigned to multi-hop AND multi-hop significantly outperforms single-hop.

Define "correctly" empirically using a small ground-truth validation set with oracle labels.

**2. Retrieval Efficiency (RE):**

Measure the fraction of queries that skipped retrieval:

```
RE = (# no-retrieval queries) / (total queries)
```

Higher RE = more cost savings. A system with 40% RE avoids 40% of retrieval calls.

**3. Answer Quality (AQ) — Per-tier:**

Report F1, BLEU, or ROUGE for each strategy tier:

```
F1_no_ret = F1(answers on no-retrieval queries)
F1_single = F1(answers on single-hop queries)
F1_multi = F1(answers on multi-hop queries)
```

Ensure no-retrieval F1 is still acceptable (not degraded due to incorrect routing).

**4. Overall Quality vs. Cost:**

Define a joint metric:

```
AdaptiveQuality = (1 - α) × AQ_overall + α × (1 - normalized_cost)
```

where α ∈ [0, 1] trades off answer quality vs. cost. Adaptive-RAG should maximize this metric relative to a fixed pipeline (single-hop or multi-hop baseline).

**Evaluation protocol:**

1. Collect ~500 queries with ground-truth answers.
2. Bin queries by oracle complexity (simple, moderate, complex).
3. For each query, run all three strategies (no-ret, single-hop, multi-hop) and measure quality.
4. Train the classifier to predict oracle complexity.
5. Evaluate:
   - Routing Accuracy = % of queries the classifier routes to the oracle-optimal tier.
   - Retrieval Efficiency = % routed to no-retrieval.
   - Quality per tier = F1/BLEU for each routed subset.
   - Overall cost and latency.

**Example results:**

```
Routing Accuracy: 87%
Retrieval Efficiency: 45%
F1 (no-retrieval): 0.72
F1 (single-hop): 0.81
F1 (multi-hop): 0.91
Overall F1: 0.82
Cost per query: $0.035 (vs. $0.10 for always-multi-hop)
```

</details>

---

## Q10. Design a production Adaptive RAG deployment that handles query routing at 500 QPS. `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Architecture:**

```
User Query (500 QPS)
    │
    ├─ [Classification Service] (inference, batch size 32)
    │     │
    │     └─ Outputs: complexity score, confidence interval
    │
    ├─────────────────────────┬──────────────────────┬─────────────────┤
    │                        │                      │                 │
    ▼                        ▼                      ▼                 ▼
[Direct LLM]          [Retrieval SVC]       [Multi-Hop Orchestrator]
(No-Retrieval)        (Single-Hop)          (Multi-Step + Reasoning)
~100ms, $0/query      ~600ms, $0.015/q      ~2000ms, $0.075/q
    │                        │                      │                 │
    └────────────────────────┴──────────────────────┴─────────────────┘
                                    │
                                    ▼
                            [Response Aggregator]
                                    │
                                    ▼
                            [Client Response]
```

**Component design:**

1. **Classification Service:**
   - Model: Lightweight classifier (e.g., DistilBERT or logistic regression).
   - Batch size: 32 (allows 500 QPS with ~16ms latency per batch).
   - Container: GPU-optimized inference (NVIDIA Triton or vLLM).
   - Replicas: 2–3 for HA. Scaling rule: add replica if latency p99 > 50ms.

2. **Direct LLM Service (No-Retrieval Tier):**
   - Model: Smaller, faster LLM (e.g., Llama 7B) or a cached/quantized version of the main model.
   - Throughput: 500 tokens/sec × 32 batch size ≈ 16K tokens/sec. Use vLLM or TensorRT for fast batch inference.
   - Replicas: 1–2. Most queries route here, so this is the throughput bottleneck.

3. **Retrieval Service (Single-Hop Tier):**
   - Query embedding: Fast embedding model (e.g., BGE-small, <10ms).
   - Vector DB: Qdrant or Weaviate with approximate nearest neighbors (HNSW). Cache top-100 embeddings per day to reduce latency.
   - Batch retrieval: Retrieve for ~100 queries in parallel; typical response: <300ms.
   - Replicas: 2–3 to handle 30% of queries (~150 QPS).

4. **Multi-Hop Orchestrator (Complex Queries):**
   - Implement as a LangGraph workflow or custom agentic loop.
   - Parallel retrieval: Fan out multiple sub-queries to the Retrieval Service.
   - Result caching: Cache common sub-queries (e.g., "What is X?") across requests.
   - Replicas: 1–2 for the remaining ~100 QPS (20% of traffic).

**Load balancing:**

```
Load Balancer (nginx / AWS ALB)
    │
    ├─ Classify 500 QPS (round-robin to 2–3 classifiers)
    │
    └─ Route by predicted complexity:
        ├─ 50% → Direct LLM (2–3 replicas)
        ├─ 30% → Retrieval Service (2–3 replicas)
        └─ 20% → Multi-Hop (1–2 replicas)
```

**Latency SLO and monitoring:**

- **P50 latency target:** 400ms (30% on fast path, 70% blended).
- **P99 latency target:** 3000ms (multi-hop queries).
- **Metrics to log:**
  - Classifier latency, accuracy per bin.
  - Per-tier throughput, latency, cost/query.
  - Answer quality (F1) per tier.
  - Cache hit rate (for retrieval and multi-hop).

**Cost estimation (monthly, 1.3B queries at 500 QPS):**

- Classifier inference: $100 (GPU time, shared).
- Direct LLM: 650M queries × $0.0001/query (batch inference) ≈ $65K.
- Retrieval (embedding + vector DB): 390M queries × $0.015 ≈ $5.8K.
- Multi-hop (2–3 LLM calls per query): 260M queries × $0.06 ≈ $15.6K.
- **Total:** ~$87K/month (vs. $130K for always-multi-hop).

**Failure modes and recovery:**

- Classifier outage → default to single-hop for all queries (safe fallback).
- Retrieval timeout → escalate to multi-hop or return LLM-only answer with lower confidence flag.
- Multi-hop timeout (>5s) → return best-effort answer from intermediate steps.

</details>
