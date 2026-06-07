# System Design Principles: Building Production-Grade RAG

> The architectural principles that separate a RAG demo from a production-grade retrieval system.

---

## The Five Properties of a Production RAG System

Every production RAG system must optimize for five properties. Interviews test whether you understand the trade-offs between them.

**1. Reliability**
- The system must return a reasonable answer even when components fail.
- What breaks it: Embedding service down, vector DB timeout, LLM rate-limited, malformed retrieved documents
- How to design for it: Circuit breakers, graceful degradation, fallback strategies
- Example: If reranking times out, skip reranking and use dense retrieval top-5 directly

**2. Latency**
- P95 latency must be under SLA (typically 200–500ms for interactive use).
- What breaks it: Agentic RAG (multiple sequential calls), expensive reranking, slow embedding model
- How to design for it: Parallel retrieval calls, caching, model quantization, simpler architectures
- Example: For <300ms SLA, choose Adaptive RAG over Agentic RAG (5–10x slower)

**3. Freshness**
- Retrieved context must reflect recent data changes.
- What breaks it: Incremental indexing fails, documents are indexed but not retrieved immediately, old documents pollute results
- How to design for it: Incremental indexing with proper TTL, timestamp-based filtering, versioning
- Example: A news RAG must re-index daily; old articles should be deprioritized after 1 week

**4. Controllability**
- Operators must be able to control what the system retrieves and generates.
- What breaks it: No way to boost/suppress specific documents, no way to exclude sensitive results, LLM generates unpredictably
- How to design for it: Metadata filtering, namespace-based isolation, output constraints
- Example: For HIPAA-compliant RAG, retrieval must enforce user-level access control

**5. Observability**
- You must measure quality in production to catch regressions and failures.
- What breaks it: No metrics, metrics that don't correlate with quality, no alerts
- How to design for it: Comprehensive logging, end-to-end tracing, feedback collection
- Example: Track P50/P95 latency per query type, monitor RAGAS faithfulness on a sample of outputs

---

## Pipeline vs. Agent Architecture: The Core Trade-off

These are the two high-level organizational patterns for any RAG system. Every architecture in the taxonomy is an instance of one or the other.

### Pipeline Architecture

```
User Query
    │
    ├──► Embedding
    │     └──► Encoding
    │
    ├──► Retrieval (Fixed Strategy)
    │     └──► Top-k ANN Search
    │
    ├──► Reranking (Optional)
    │     └──► Cross-Encoder
    │
    └──► LLM Generation
          └──► Fixed Prompt Template
          
    Answer
```

**Strengths:**
- Fast (each step is a single function call; parallelizable)
- Deterministic (same query, same retrieval strategy)
- Debuggable (clear data flow; test each step independently)
- Cost-predictable (embedding cost + vector DB cost + LLM cost, all constant)

**Weaknesses:**
- Retrieval strategy is fixed (doesn't adapt to query complexity)
- No feedback loop (wrong retrieval not detected)
- Fails on complex multi-hop questions

**When to choose:** Most production systems start here. Naive, Advanced, Modular RAG are pipelines.

---

### Agent Architecture

```
User Query
    │
    ├──► Agent Decides: [Retrieve? Reason? Retrieve Again?]
    │
    ├──► Retrieve (Agent-Chosen Strategy)
    │     └──► Re-evaluate: Is Context Sufficient?
    │
    ├──► Generate (Partial Answer)
    │
    ├──► Agent Decides: [Done? Need More Context?]
    │
    └──► Loop Until Done
    
    Answer
```

**Strengths:**
- Adaptive (different queries get different strategies)
- Feedback loops (agent validates context before generation)
- Flexible (can handle multi-hop, meta-reasoning)
- Learns from errors (Self-RAG fine-tunes based on scores)

**Weaknesses:**
- Slow (multiple sequential LLM calls)
- Non-deterministic (agent may choose different paths for similar queries)
- Hard to debug (implicit decision logic in the LLM)
- Cost unpredictable (more calls for complex queries)

**When to choose:** When simple retrieval isn't enough. Agentic RAG, Self-RAG are agents.

| Dimension | Pipeline | Agent |
|-----------|----------|-------|
| Latency | <300ms (2–3 calls) | >500ms (4–8 calls) |
| Determinism | High | Low |
| Debuggability | High | Low |
| Cost predictability | High | Low |
| Failure recovery | Manual (circuits) | Implicit (agent decides) |
| Routing complexity | Hard-coded | Learned (LLM decides) |

### Incremental Migration: Pipeline → Agent

Don't rewrite your whole system. Add agent logic incrementally:

```python
# Start: Pure pipeline
def rag_v1(query: str) -> str:
    context = retrieve(query)
    return generate(query, context)

# v2: Add a validation step (Corrective RAG pattern)
def rag_v2(query: str) -> str:
    context = retrieve(query)
    if not is_sufficient(context):  # feedback
        context = retrieve(query, reranked=True)
    return generate(query, context)

# v3: Add agent logic (Agentic RAG pattern)
def rag_v3(query: str) -> str:
    agent_state = initialize_agent(query)
    while not agent_state.done:
        action = agent.decide(agent_state)
        if action == "retrieve":
            context = retrieve(query, agent_state.strategy)
            agent_state.update(context)
        elif action == "generate":
            answer = generate(query, agent_state.context)
            agent_state.mark_done(answer)
    return agent_state.answer
```

---

## Scalability Patterns

As your corpus grows, different bottlenecks emerge. These patterns address them.

### Pattern 1: Read-Path Optimization

Optimize the retrieval and ranking phases.

```
Query
  │
  ├──► Embedding Cache (check if embedding exists)
  │    └──► Miss → Embed online → Cache
  │
  ├──► Vector DB (partitioned index)
  │    └──► Partition 1  ┐
  │         Partition 2  ├──► Merge Top-k
  │         Partition 3  ┘
  │
  ├──► Reranker (GPU-accelerated, batched)
  │
  └──► Answer
```

**Techniques:**
- **Embedding Cache:** Store embeddings for frequent queries; re-use instead of re-computing
- **Index Partitioning:** Split vector index by document namespace; search only relevant partitions
- **Reranker Batching:** Collect top-50 from vector DB, send them all to reranker at once (amortize LLM overhead)
- **GPU Acceleration:** Run embedding and reranking on GPU

**Thresholds:**
- <10K documents: No optimization needed
- 10K–1M documents: Add embedding cache + index partitioning
- 1M+ documents: Partition by namespace; consider multi-step retrieval (sparse first, then dense)

---

### Pattern 2: Write-Path Optimization

Optimize indexing and updates.

```
New Document
  │
  ├──► Async Queue (batch documents)
  │
  ├──► Chunker (parallelize chunking)
  │
  ├──► Embedder (batch embed; share embedding service)
  │
  ├──► Vector DB Insert (batch insert)
  │
  └──► Index Ready (async; doesn't block user)
```

**Techniques:**
- **Async Indexing:** Don't wait for embedding to complete; put documents in a queue
- **Batch Embedding:** Collect N documents, embed them together (faster than one-by-one)
- **Incremental Updates:** Update only changed documents; full re-index is rare
- **TTL-Based Eviction:** Set expiration on old documents; they're automatically deprioritized

**Thresholds:**
- <100 docs/day: Sync indexing is fine
- 100–10K docs/day: Add async queue + batching
- 10K+ docs/day: Incremental indexing is mandatory; full re-index is too slow

---

### Pattern 3: Horizontal Scaling

Scale retrieval across multiple machines.

```
Query
  │
  ├──► Load Balancer
  │
  ├──► Retrieval Node 1 ┐
  │    Retrieval Node 2 ├──► Merge Results
  │    Retrieval Node 3 ┘
  │
  └──► Answer
```

**Techniques:**
- **Shard by Namespace:** Each retrieval node handles queries for a subset of namespaces
- **Replicate for Read Throughput:** Multiple nodes with same index; distribute queries across them
- **Geo-Distribution:** Place nodes near users to reduce latency

**Thresholds:**
- <100 QPS: Single machine
- 100–1K QPS: Replicate (2–3 read replicas)
- 1K+ QPS: Partition (shard by namespace)

---

## Failure Modes and Circuit Breakers

Production systems fail. Design for graceful degradation.

| Component | Failure Mode | Detection Signal | Mitigation | Circuit Breaker Pattern |
|---|---|---|---|---|
| Embedding Service | Timeout or error; embeddings are slow | Latency spike >500ms or error rate >5% | Fall back to BM25 (sparse-only retrieval) | If embedding fails 3x in 10s, skip embeddings for 30s |
| Vector DB | Out of memory or corrupted index | Query error; timeout | Use cached embeddings; skip reranking | If VectorDB fails 3x in 10s, return empty result + fallback |
| Reranker | Overloaded; queue backs up | Latency >5s per request | Skip reranking; return top-k from dense retrieval | If reranker latency >1s, disable reranking for 1 minute |
| LLM Service | Rate-limited or down | 429 or 503 errors | Return retrieved context as-is; don't generate | If LLM fails 5x, serve pre-generated summaries from cache |
| Chunker | Malformed document causes parsing error | Exception; incomplete chunks | Skip document; log error; alert | If chunker fails on >10% of batch, stop processing |

**Circuit Breaker Pattern in Python:**

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # "closed" = normal, "open" = failing, "half-open" = testing
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = "half-open"
            else:
                raise Exception(f"Circuit breaker open. Failing fast.")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise e

# Usage
embedding_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)

def embed_with_fallback(query: str) -> np.ndarray:
    try:
        return embedding_breaker.call(embed_model.encode, query)
    except:
        # Fallback: use BM25 instead
        return sparse_retrieval(query)
```

---

## Cost Architecture

Understanding cost drivers lets you optimize intelligently.

### Three Cost Centers in RAG

1. **Embedding Cost** (per-token, one-time at index-time + per-query at retrieval-time)
   - Embedding index-time: 10M documents × 500 tokens/doc × $0.02/1M tokens = $100
   - Embedding query-time (10K QPS × 50 tokens/query × $0.02/1M tokens): ~$10/day

2. **Vector Storage Cost** (per-GB-month)
   - 10M vectors × 1536 dimensions (OpenAI) × 4 bytes/dim = ~60 GB
   - Pinecone: ~$10/GB-month → $600/month
   - Self-hosted (Qdrant on disk): ~$5/month (storage cost only)

3. **LLM Inference Cost** (per-token, generation-time)
   - GPT-4 Turbo: $0.01 per 1K input tokens, $0.03 per 1K output tokens
   - At 10K QPS × 1000 input tokens (context + query) × $0.01/1K = $100/day input
   - At 10K QPS × 200 output tokens × $0.03/1K = $60/day output

### Cost Breakdown Example: 10M Docs, 10K QPS

| Cost Center | Cost per Day | Cost per Month | % of Total |
|---|---|---|---|
| Embedding (queries only) | $10 | $300 | 5% |
| Vector Storage | $20 | $600 | 10% |
| LLM Inference | $160 | $4800 | 85% |
| **Total** | **$190** | **$5700** | **100%** |

**Optimization Techniques:**

| Technique | Cost Component | Estimated Savings | Quality Trade-off |
|---|---|---|---|
| Semantic caching (don't re-compute for repeated queries) | Embedding | 20–30% | Minimal |
| Quantization (8-bit embeddings instead of 32-bit) | Vector storage | 75% | ~1% recall loss |
| Model tier routing (GPT-3.5 for simple queries, GPT-4 for complex) | LLM | 40–50% | Latency variance |
| Batch generation (generate 10 answers at once) | LLM | 5–10% (amortize overhead) | Latency increase |
| Smaller embedding model (BGE instead of OpenAI) | Embedding | 50% | Domain-specific; test first |

---

## Observability Stack

What to instrument and why.

### Retrieval Metrics

Track every retrieval call:

```
Query → [Embedding Time] → [VectorDB Latency] → [Top-k Results]
           ↓ Log                  ↓ Log              ↓ Log
        [Metric]              [Metric]           [Metric]
```

**Key Metrics:**
- **Latency:** P50, P95, P99 of embedding + retrieval + reranking
- **Throughput:** Queries per second processed
- **Cache Hit Rate:** % of queries served from embedding cache
- **Recall@k:** What fraction of relevant documents appear in top-k (requires labeled probe set)
- **Chunk Hit Rate:** How often the "right" chunk is in top-k

### Generation Metrics

Track LLM output quality:

**Offline (on a sample of queries):**
- **Faithfulness:** Does the answer use only retrieved context?
- **Answer Relevance:** Does it answer the question?
- **Context Precision:** Is retrieved context relevant?
- Run RAGAS on 1% of queries; alert if metrics drop >5%

**Online (continuous):**
- **Token Usage:** Tokens consumed per query (detect cost anomalies)
- **Refusal Rate:** How often does LLM refuse to answer?
- **User Feedback:** Thumbs up/down, dwell time on answer, follow-up queries

### System Metrics

Health of the system as a whole:

- **Error Rate:** % of queries that error (target: <0.1%)
- **Circuit Breaker State:** Which services are open/half-open
- **Queue Depth:** Documents waiting to be indexed (watch for backlog)
- **Resource Utilization:** CPU, GPU, memory on retrieval nodes

### Dashboard Mockup

```
┌─────────────────────────────────────────────────────────┐
│ RAG System Dashboard                                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Latency (P95): 287ms  ↑ 15% from baseline            │
│  QPS: 9,847           ↑ Normal                         │
│  Error Rate: 0.08%    ✓ Healthy                        │
│                                                         │
│  Retrieval Metrics:                                     │
│    Recall@5: 0.78     ↓ Alert (target: 0.85)          │
│    Precision@5: 0.92  ✓ Normal                         │
│    Cache Hit: 34%     ✓ Normal                         │
│                                                         │
│  Generation Metrics (sample):                           │
│    Faithfulness: 0.81 ↓ Alert (target: 0.90)          │
│    Relevance: 0.88    ✓ Normal                         │
│    User Feedback: 87% positive                         │
│                                                         │
│  Alerts:                                                │
│    ⚠ Recall dropped. Check embedding model quality    │
│    ⚠ Faithfulness below threshold. Check reranker     │
│    ⚠ P95 latency spiked. Check VectorDB load         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Instrumentation Example: OpenTelemetry

```python
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger import JaegerExporter

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

def retrieve_with_tracing(query: str) -> list:
    with tracer.start_as_current_span("retrieve") as span:
        span.set_attribute("query", query)
        
        # Embedding span
        with tracer.start_as_current_span("embed_query"):
            embedding = embed_model.encode(query)
        
        # VectorDB span
        with tracer.start_as_current_span("vector_search"):
            results = vector_db.search(embedding, k=10)
        
        span.set_attribute("num_results", len(results))
        return results

# Meter for metrics
retrieval_latency = meter.create_histogram("retrieval.latency_ms")

def tracked_retrieval(query: str):
    start = time.time()
    results = retrieve_with_tracing(query)
    latency_ms = (time.time() - start) * 1000
    retrieval_latency.record(latency_ms)
    return results
```

---

## The Canonical System Design Interview Answer Structure

When asked "Design a RAG system for...", use this template. Filling it in for any prompt gives you a complete, production-grade design.

### 1. Clarify Requirements
- **Functional:** What queries must the system answer? What retrieval source?
- **Non-functional:** Latency SLA? Accuracy target? Scale (QPS, corpus size)? Cost budget?

### 2. Data Flow Diagram
```
User Query
    ↓
[Which embedding model?]
    ↓
[Which vector database?]
    ↓
[Retrieval strategy: dense, hybrid, multi-hop?]
    ↓
[Reranking: yes or no?]
    ↓
[Which LLM? Which prompt template?]
    ↓
Answer
```

### 3. Component Selection
- **Embedding Model:** Domain-fit + latency constraints
- **Vector DB:** Managed vs. self-hosted, scale requirements
- **Retrieval Strategy:** Pipeline vs. agent, single-hop vs. multi-hop
- **Reranking:** Required for high precision?
- **LLM:** Model size vs. latency/cost trade-off

### 4. Scale Analysis
- How many vectors? → Vector DB size
- How many QPS? → Retrieval latency budget
- What's the cost per query? → Monthly cost projection

### 5. Failure Modes & Mitigation
- Embedding service down → Fallback to BM25
- Vector DB latency spike → Circuit breaker + cache
- LLM rate-limited → Queue or fallback to GPT-3.5

### 6. Evaluation Plan
- Retrieval metrics: recall@k on labeled probe set
- Generation metrics: RAGAS (faithfulness, relevance)
- User feedback: thumbs up/down on answers

### Example: Enterprise Document Search

**Requirements:**
- 10M internal documents, <300ms P95 latency, 1K QPS peak, handle multi-page PDFs
- Freshness: Daily updates acceptable
- Accuracy: Recall@5 >80%, no hallucinations

**Data Flow:**
```
User Query (natural language)
    ├──► Embedding (OpenAI text-embedding-3-small, 512 dims, $0.02/1M tokens)
    │
    ├──► Retrieval (Qdrant, HNSW index, partitioned by department)
    │    └──► Top-50 candidates
    │
    ├──► Reranking (BGE-reranker-large, cross-encoder)
    │    └──► Top-5 results
    │
    └──► LLM (GPT-4 Turbo, XML-delimited context to prevent injection)
         └──► Generated answer
```

**Component Selection:**
- **Embedding:** text-embedding-3-small (good for long documents, $0.02/1M)
- **Vector DB:** Qdrant (partitioned by department for access control)
- **Retrieval:** Dense + reranking (pipeline architecture for latency budget)
- **LLM:** GPT-4 Turbo ($0.01/1K input, $0.03/1K output)

**Scale Analysis:**
- 10M docs × 500 tokens/doc = 5B tokens indexed @ $0.02/1M = $100 (one-time)
- 1K QPS × 50 query tokens × $0.02/1M = $1/day in embedding costs
- 1K QPS × 1000 context tokens × $0.01/1K + 200 output × $0.03/1K = $10 + $6 = $16/day in LLM
- Monthly: ~$100 (storage) + $30 (embedding) + $480 (LLM) = $610

**Failure Modes:**
- Embedding timeout → Use BM25 directly on query keywords
- Reranker slow → Skip reranking, use top-10 from retrieval
- LLM rate-limited → Queue queries; serve most recent cached answers while queue drains

**Evaluation:**
- Labeled set: 100 internal queries with known relevant documents
- Retrieval: Recall@5 on labeled set (run weekly)
- Generation: RAGAS faithfulness on 1% of production queries (sample weekly)
- User feedback: Thumbs up/down on generated answers

