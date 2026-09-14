# 03 — Modular RAG

> Treats the RAG pipeline as a set of interchangeable modules — retriever, reranker, reader, memory, router.

---

## 🏗️ Architecture Flow, Components & Tools

### Architecture Flow

```
                        ┌──────────────────────────┐
                        │       Orchestrator        │
                        └────────────┬──────────────┘
                                     │
Query ──────────► Routing Module ───┤
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
     Retrieval Module A      Retrieval Module B      Memory Module
     (vector search)         (SQL / web / BM25)       (short + long term)
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     ▼
                             Fusion Module
                                     │
                                     ▼
                            Reranking Module
                                     │
                                     ▼
                            Generation Module → Answer
```

### Key Components

| Component | Responsibility |
|---|---|
| Router | Classifies the query and selects which retrieval module(s) to invoke |
| Retrieval Module(s) | Pluggable, swappable retrievers behind a common interface (vector, BM25, SQL, web) |
| Memory Module | Maintains short-term conversation state and long-term user/session knowledge |
| Fusion Module | Merges results from multiple retrieval modules (e.g., RRF) |
| Reranker Module | Reorders fused results by relevance before generation |
| Generator Module | Reader/LLM that synthesizes the grounded final answer |
| Orchestrator | Wires modules together, handles fallback/error recovery, and enables A/B swapping |

### Tools & Frameworks

| Category | Example Tools & Frameworks |
|---|---|
| Modular Query Pipelines | LlamaIndex modular query pipelines / `RouterQueryEngine` |
| Pipeline Composition | Haystack pipelines, LangChain LCEL, LangGraph |
| Memory | MemGPT, Zep |
| Fusion / Ensemble Retrieval | LangChain `EnsembleRetriever`, LlamaIndex `QueryFusionRetriever` |

---

## Q1. What is Modular RAG and why is it an improvement over fixed pipelines? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Modular RAG decomposes the RAG system into independent, swappable components:

- **Search module** — vector search, keyword search, web search
- **Memory module** — short-term (conversation) and long-term (knowledge base)
- **Fusion module** — merges results from multiple retrievers
- **Routing module** — decides which retriever to call for a given query
- **Reader/Generator module** — the LLM that synthesizes the answer

Unlike Naive or Advanced RAG's fixed pipelines, Modular RAG lets you **add, remove, or swap** components without redesigning the whole system. This maps well to production engineering where different use cases need different retrieval strategies.

</details>

---

## Q2. How does a routing module work in Modular RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A **routing module** decides which retrieval pathway to invoke based on the query type:

```
Query → Router
          ├── Structured query → SQL database retriever
          ├── Conceptual query → Vector store retriever
          ├── Recent events → Web search retriever
          └── Internal docs  → Private knowledge base retriever
```

Routing can be:
- **Rule-based** — keyword classifiers or regex patterns
- **LLM-based** — prompt the LLM to classify the query type
- **Learned** — a fine-tuned classifier

LlamaIndex's `RouterQueryEngine` and LangChain's `MultiRetrievalQAChain` are production implementations of this pattern.

</details>

---

## Q3. What is a fusion retriever and when would you use one? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A **fusion retriever** runs multiple retrievers in parallel and merges their results. For example:

1. Run dense vector search → top-10 chunks
2. Run BM25 keyword search → top-10 chunks
3. Run web search → top-5 results
4. Merge all 25 results using **Reciprocal Rank Fusion (RRF)**
5. Pass final top-5 to the LLM

**When to use it:**
- When no single retriever has full coverage
- For heterogeneous corpora (structured + unstructured data)
- When recall matters more than latency

**Trade-off:** Higher latency (parallel calls) and cost (more tokens passed to LLM).

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import Chroma

dense_retriever = Chroma(...).as_retriever(search_kwargs={"k": 10})
sparse_retriever = BM25Retriever.from_documents(docs, k=10)

fusion_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.6, 0.4],  # RRF-based merging
)
results = fusion_retriever.get_relevant_documents(query)
```

</details>

---

## Q4. How do you handle memory in a Modular RAG system for multi-turn conversations? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Two types of memory are needed:

| Memory Type | What it stores | Example |
|---|---|---|
| **Short-term** | Current conversation history | Last 5 user/assistant turns |
| **Long-term** | User preferences, past sessions | Summarized past interactions |

**Strategies:**

1. **Sliding window** — Keep the last N turns in the prompt.
2. **Summary memory** — Periodically summarize older turns with an LLM.
3. **Entity memory** — Extract and store key entities (names, preferences) as a mini knowledge base.
4. **Episodic memory** — Store past Q&A pairs in a vector DB; retrieve relevant past exchanges for context.

For production, tools like **MemGPT** or **Zep** implement long-term memory as a separate service.

</details>

---

## Q5. Compare Modular RAG to the LangChain and LlamaIndex frameworks. Which maps better? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Both frameworks implement modular RAG concepts but with different philosophies:

| Aspect | LangChain | LlamaIndex |
|---|---|---|
| **Focus** | General LLM orchestration | Data-centric RAG pipelines |
| **Routing** | `MultiRetrievalQAChain` | `RouterQueryEngine` |
| **Memory** | `ConversationBufferMemory` | `ChatMemoryBuffer` |
| **Fusion** | `EnsembleRetriever` | `QueryFusionRetriever` |
| **Best for** | Complex agentic workflows | Document-heavy RAG systems |

**LlamaIndex** maps more directly to Modular RAG's architecture since it treats data ingestion, indexing, and querying as first-class modular concerns. LangChain is more flexible but requires more manual wiring. In practice, many production systems use both.

</details>

---

## Q6. What is the "reader" module and how does it differ from a plain LLM call? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The **reader module** (also called the "generator") synthesizes an answer from retrieved chunks. It differs from a plain LLM call in three ways:

| Aspect | Plain LLM Call | Reader Module |
|--------|----------------|---------------|
| **Grounding** | Generates freely from training data | Must cite and ground in retrieved chunks |
| **Citation tracking** | No linkage to sources | Maps answer segments → source chunks |
| **Compression** | Passes all context verbatim | Can filter/compress context pre-generation |

**Reader module responsibilities:**

1. **Context filtering** — Drop irrelevant chunks before passing to LLM.
2. **Citation generation** — Mark which chunk(s) support each sentence.
3. **Answer synthesis** — Merge multiple chunks into a coherent answer.

**Example: Citation-aware reader**

```python
class CitationReader:
    def __init__(self, llm):
        self.llm = llm
    
    def generate(self, query, chunks):
        # Build a prompt that forces citation
        prompt = f"""Answer based ONLY on the chunks below.
Each answer sentence must cite its source [Chunk N].

Query: {query}

Chunks:
{chr(10).join(f'[Chunk {i}] {chunk.page_content}' for i, chunk in enumerate(chunks))}

Answer (with citations):"""
        
        answer = self.llm.invoke(prompt)
        # Parse answer to extract citations
        return answer  # e.g., "RAG improves QA [Chunk 0]..."
```

A grounded reader reduces hallucinations and enables end-to-end fact-checking.

</details>

---

## Q7. How do you implement a pluggable retriever interface for Modular RAG? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Define an abstract base retriever that all concrete retrievers implement. This allows swapping retrievers at runtime without changing the pipeline.

```python
from abc import ABC, abstractmethod
from typing import List, Dict

class BaseRetriever(ABC):
    """Abstract interface for all retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Args:
            query: User query string
            k: Number of results to return
        
        Returns:
            List of {"content": str, "source": str, "score": float}
        """
        pass


class VectorRetriever(BaseRetriever):
    """Dense vector similarity search."""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score)
            }
            for doc, score in results
        ]


class BM25Retriever(BaseRetriever):
    """Sparse keyword-based search."""
    
    def __init__(self, docs):
        self.bm25 = BM25Okapi([doc.split() for doc in docs])
        self.docs = docs
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        scores = self.bm25.get_scores(query.split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {
                "content": self.docs[i],
                "source": f"doc_{i}",
                "score": float(scores[i])
            }
            for i in top_indices
        ]


class WebSearchRetriever(BaseRetriever):
    """Search the web for recent information."""
    
    def __init__(self, api_key):
        self.api_key = api_key
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        # Call external web search API
        results = call_web_search_api(query, self.api_key, num_results=k)
        return [
            {
                "content": r["snippet"],
                "source": r["url"],
                "score": r["relevance_score"]
            }
            for r in results
        ]


# Usage: Swap retrievers at runtime
retriever_config = {
    "type": "vector",  # Could be "bm25", "web", "sql", etc.
}

if retriever_config["type"] == "vector":
    retriever = VectorRetriever(vectorstore)
elif retriever_config["type"] == "bm25":
    retriever = BM25Retriever(docs)
else:
    retriever = WebSearchRetriever(api_key)

# Same interface for all
results = retriever.retrieve("What is RAG?", k=5)
```

This pattern enables easy A/B testing: swap the retriever config and compare results without changing the QA pipeline.

</details>

---

## Q8. What is iterative retrieval and when does it outperform single-shot retrieval? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Iterative retrieval** runs multiple rounds of query-retrieve-refine cycles. After the LLM generates a partial answer, it identifies gaps and retrieves again to fill them.

```
┌─────────────────┐
│  Query: "Who    │
│  won the 2024   │
│  Nobel Prize?"  │
└────────┬────────┘
         │
         ▼
    ┌─────────────┐
    │ Retrieve 1  │ → "Prize categories include..."
    └────────┬────────────────────┐
             │                    │
             ▼                    │
       ┌──────────────────┐       │
       │ LLM: Generate    │       │
       │ partial answer   │       │
       │ "Physics winner  │       │
       │  is unknown..."  │       │
       └────────┬─────────┘       │
                │                │
      [Gap detected]             │
                │                │
                ▼                │
       ┌──────────────────┐      │
       │ New query:       │      │
       │ "2024 Nobel      │      │
       │ Physics winner"  │      │
       └────────┬─────────┘      │
                │                │
                ▼                │
       ┌──────────────────┐      │
       │ Retrieve 2       │──────┘
       │ → "The Physics   │
       │  prize goes to   │
       │  X, Y, Z..."     │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ LLM: Final       │
       │ answer with full │
       │ context          │
       └──────────────────┘
```

**When iterative outperforms single-shot:**

- **Complex multi-hop queries** — Finding the CEO of the company that acquired X.
- **Sparse knowledge bases** — A single retrieval pass misses relevant docs; second pass finds them.
- **Decomposable questions** — Q can be broken into sub-questions, each answered iteratively.

**Trade-off:** Multiple retrieval rounds increase latency. Typically iterative is 2-3x slower but improves recall 10-30%.

```python
class IterativeRetriever:
    def __init__(self, base_retriever, llm, max_iterations=3):
        self.retriever = base_retriever
        self.llm = llm
        self.max_iterations = max_iterations
    
    def retrieve_with_refinement(self, query):
        context = ""
        current_query = query
        
        for iteration in range(self.max_iterations):
            # Retrieve with current query
            results = self.retriever.retrieve(current_query, k=5)
            context += "\n".join([r["content"] for r in results])
            
            # Generate partial answer and detect gaps
            prompt = f"Query: {query}\nContext: {context}\n\nAnswer (note any gaps):"
            answer = self.llm.invoke(prompt)
            
            # Check if gaps detected
            if "[GAP]" not in answer or iteration == self.max_iterations - 1:
                return answer, context
            
            # Refine query based on gaps
            current_query = self.llm.invoke(f"What new query would fill this gap? {answer}")
        
        return answer, context
```

</details>

---

## Q9. How do you add observability (tracing + metrics) to a modular RAG pipeline? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Observability requires instrumenting each module to track latency, quality, and resource usage.

```python
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import time

# Set up tracing
jaeger_exporter = JaegerExporter(agent_host_name="localhost", agent_port=6831)
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(jaeger_exporter))
tracer = trace.get_tracer(__name__)

# Instrumented modular pipeline
class ObservableRAGPipeline:
    def __init__(self, retriever, reranker, reader):
        self.retriever = retriever
        self.reranker = reranker
        self.reader = reader
    
    def query(self, question: str) -> str:
        with tracer.start_as_current_span("rag_pipeline") as span:
            # 1. Retrieve
            with tracer.start_as_current_span("retrieval") as ret_span:
                start = time.time()
                chunks = self.retriever.retrieve(question, k=10)
                ret_span.set_attribute("num_chunks", len(chunks))
                ret_span.set_attribute("latency_ms", int((time.time() - start) * 1000))
            
            # 2. Rerank
            with tracer.start_as_current_span("reranking") as rer_span:
                start = time.time()
                top_chunks = self.reranker.rerank(question, chunks, k=5)
                rer_span.set_attribute("num_reranked", len(top_chunks))
                rer_span.set_attribute("latency_ms", int((time.time() - start) * 1000))
            
            # 3. Generate
            with tracer.start_as_current_span("generation") as gen_span:
                start = time.time()
                answer = self.reader.generate(question, top_chunks)
                gen_span.set_attribute("answer_length", len(answer))
                gen_span.set_attribute("latency_ms", int((time.time() - start) * 1000))
            
            return answer

# Key metrics to track
METRICS_TABLE = """
| Metric | Definition | Target |
|--------|-----------|--------|
| Retrieval P@k | Fraction of top-k results relevant | >85% |
| Reranking recall | Did reranker keep the best chunks? | >95% |
| Generation latency | E2E time from query to answer | <2s |
| Token efficiency | Avg tokens passed to LLM per query | <2000 |
| Hallucination rate | % answers contradicting context | <5% |
"""
```

**Visualization:** Use Jaeger or Grafana to view trace timelines and spot bottlenecks (e.g., "reranking is 50% of latency").

</details>

---

## Q10. How do you perform online A/B testing of individual RAG modules in production? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A/B test modules by splitting traffic and measuring quality metrics independently.

```
┌──────────────────────────────────────────────────────────────┐
│                    Query Traffic (100%)                       │
└───────────────────────┬──────────────────────────────────────┘
                        │
                ┌───────┴────────┐
                │ (traffic split)│
                ▼                ▼
            ┌─────────┐      ┌──────────┐
            │ Group A │      │ Group B  │
            │ (50%)   │      │ (50%)    │
            └────┬────┘      └────┬─────┘
                 │                │
            ┌────▼────┐       ┌───▼─────┐
            │ Control │       │ Variant │
            │Retriever│       │Retriever│
            │ (BM25)  │       │ (Hybrid)│
            └────┬────┘       └───┬─────┘
                 │                │
                 └────────┬───────┘
                          │
                    [Reranker (shared)]
                    [Reader (shared)]
                          │
                    ┌─────▼──────┐
                    │  Metrics   │
                    │  Collection│
                    └─────┬──────┘
                          │
            ┌─────────────┼──────────────┐
            ▼             ▼              ▼
       Context       Answer Quality   Latency
       Precision     (RAGAS)          (p50, p95)
       Recall        Hallucination    Cost per query
                     Rate
```

**Implementation:**

```python
import hashlib
import json
from enum import Enum

class ExperimentVariant(Enum):
    CONTROL = "control"
    VARIANT = "variant"

def assign_variant(user_id: str, experiment_id: str) -> ExperimentVariant:
    """Deterministic assignment based on user ID."""
    hash_val = int(hashlib.md5(f"{user_id}:{experiment_id}".encode()).hexdigest(), 16)
    return ExperimentVariant.CONTROL if hash_val % 2 == 0 else ExperimentVariant.VARIANT

class ExperimentPipeline:
    def __init__(self, control_retriever, variant_retriever, reranker, reader):
        self.control_ret = control_retriever
        self.variant_ret = variant_retriever
        self.reranker = reranker
        self.reader = reader
        self.metrics = []
    
    def query(self, user_id: str, question: str, experiment_id: str = "exp_001"):
        variant = assign_variant(user_id, experiment_id)
        start = time.time()
        
        # Retrieve with assigned variant
        if variant == ExperimentVariant.CONTROL:
            chunks = self.control_ret.retrieve(question, k=10)
            retriever_type = "bm25"
        else:
            chunks = self.variant_ret.retrieve(question, k=10)
            retriever_type = "hybrid"
        
        # Shared reranking + generation
        top_chunks = self.reranker.rerank(question, chunks, k=5)
        answer = self.reader.generate(question, top_chunks)
        latency = time.time() - start
        
        # Log metrics
        self.metrics.append({
            "user_id": user_id,
            "experiment_id": experiment_id,
            "variant": variant.value,
            "retriever": retriever_type,
            "latency_s": latency,
            "num_chunks": len(top_chunks),
            "answer": answer
            # Later: add RAGAS/context precision from offline eval
        })
        
        return answer

# Offline analysis (run daily)
def analyze_experiment(metrics_log: List[Dict]):
    control = [m for m in metrics_log if m["variant"] == "control"]
    variant = [m for m in metrics_log if m["variant"] == "variant"]
    
    print(f"Control (n={len(control)}):")
    print(f"  Avg latency: {sum(m['latency_s'] for m in control) / len(control):.2f}s")
    print(f"\nVariant (n={len(variant)}):")
    print(f"  Avg latency: {sum(m['latency_s'] for m in variant) / len(variant):.2f}s")
    
    # Statistical significance test (e.g., t-test)
    # Compare metrics: precision, recall, latency, cost, user satisfaction
```

**Best practices:**
- Use deterministic assignment (same user always sees same variant).
- Run for ≥1 week to capture temporal variance (weekday vs. weekend).
- Test one module at a time (otherwise can't isolate impact).
- Measure both positive (accuracy) and negative (latency, cost) metrics.
- Track guardrail metrics (hallucination rate) to ensure variant doesn't degrade quality.

</details>

---

## Q11. How do you model and minimize the routing overhead and module selection cost in a Modular RAG system with many active modules? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Routing cost breakdown:**

In Modular RAG, every query must pass through a router to decide which modules to invoke. With many modules (5–20), routing can become a bottleneck.

| Component | Latency | Cost |
|-----------|---------|------|
| **Router inference** (router LLM or classifier) | 200–500ms | $0.001–0.005 |
| **Module selection** (decision logic) | 10–50ms | negligible |
| **Parallel module dispatch** | 0ms (overlapped) | varies by module |
| **Module execution** | 500–3000ms | $0.01–0.10 (per module) |

**Cost optimization strategies:**

1. **Lightweight router** — Use a small classifier (logistic regression or DistilBERT) instead of querying an LLM:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   
   # Router trained on past queries → 5 ms latency, <$0.00001 cost
   router = RandomForestClassifier(n_estimators=10, max_depth=3)
   router.fit(query_embeddings, module_labels)
   
   selected_module = router.predict(query_embedding)[0]
   ```
   vs.
   ```python
   # LLM router → 500ms, $0.005 cost
   selected_module = llm.parse(f"Route to: {modules_description}")
   ```

2. **Multi-tier routing** — Route to a coarse tier first, then fine-grain:
   ```
   Query
     │
     ├─ Tier 1: Route to broad category (5 options)
     │     └─ 10ms, 1 LLM call
     │
     ├─ Tier 2: Within category, select specific module (3 options within category)
     │     └─ 50ms, 1 LLM call
     │
     └─ Total: 60ms vs. 500ms for flat routing
   ```

3. **Caching router decisions** — Cache (query, selected_modules) pairs:
   ```python
   def route_with_cache(query):
       cache_key = hash(query)
       if cache_key in routing_cache:
           return routing_cache[cache_key]
       
       modules = router(query)
       routing_cache[cache_key] = modules
       return modules
   ```
   - 30–50% cache hit rate → average routing latency drops 40%.

4. **Batch routing** — For offline/bulk queries, route in batches:
   ```python
   # Real-time: route 1 query at a time → 500ms
   # Batch: route 100 queries together → 500ms total (5ms per query)
   
   queries_batch = [q1, q2, ..., q100]
   router_outputs = llm.batch_route(queries_batch)
   ```

**Module selection cost:**

Beyond routing, selecting *which* modules to invoke adds overhead:

1. **Single module** — Router picks one module, invoke it.
   - Cost: $0.05 (if module is API call).
   - Latency: 200–500ms.

2. **Ensemble** — Router picks multiple modules (e.g., "use retriever + reranker + summarizer").
   - Cost: 3× $0.05 = $0.15.
   - Latency: 200–500ms (parallel execution).
   - Benefit: higher quality but higher cost.

3. **Adaptive selection** — Select modules based on query complexity:
   ```python
   def select_modules_adaptive(query):
       if is_simple_query(query):
           return ["simple_retriever"]  # $0.02
       elif is_moderate_query(query):
           return ["advanced_retriever", "reranker"]  # $0.08
       else:
           return ["multi_hop_agent", "reranker", "summarizer"]  # $0.20
   ```
   - Average cost: 0.3×0.02 + 0.5×0.08 + 0.2×0.20 = $0.08 (vs. always using full ensemble at $0.20).

**Example cost reduction:**

Baseline Modular RAG (10 modules, LLM router, always invoke top-3):
- Router: $0.005/query.
- Module invocation: 3×$0.05 = $0.15/query.
- Total: $0.155/query.
- Latency: 600ms (router) + 300ms (modules) = 900ms.

Optimized Modular RAG (lightweight router, adaptive selection):
- Router: $0.00001/query (forest classifier).
- Module invocation: adaptive (0.3×1 + 0.5×2 + 0.2×3) avg = 1.9 modules → $0.095/query.
- Total: $0.09/query (42% cost reduction).
- Latency: 10ms (router) + 300ms (modules) = 310ms (66% faster).

**Monitoring routing efficiency:**

Track:
- Router latency per query (target: <50ms).
- Cache hit rate (target: >40%).
- Cost per module invoked (flag modules with unexpectedly high cost).
- End-to-end latency by routing path.

</details>

---

## Q12. What are router manipulation and module substitution attacks in Modular RAG, and how do you defend against them? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Attack 1: Router manipulation**

An attacker crafts a query that fools the router into selecting a malicious module:

```
Query: "Show me [trigger_phrase] information"

Router trained on:
- "Show me medical information" → route to medical_module
- "Show me legal information" → route to legal_module

Attacker uses trigger phrase "[trigger_phrase]" that activates medical_module,
but the query actually asks for information the medical_module shouldn't answer.

Result: Confidential medical data is exposed by the wrong module.
```

**Attack 2: Module substitution**

An attacker replaces a legitimate module with a malicious one that looks identical:

```
Legitimate pipeline:
Query → Router → Trusted Retriever (module A) → Answer

Attacker substitutes module A with Trojan Retriever:
Query → Router → Trojan Retriever (looks identical, returns malicious docs) → Answer
```

**Attack 3: Cascade exploitation**

An attacker chains modules to exfiltrate data. Module A outputs partial information,
which becomes input to Module B, which leaks it further.

**Defences:**

**1. Router robustness and adversarial testing:**

Regularly test the router against adversarial queries:

```python
def test_router_robustness():
    adversarial_queries = [
        ("innocent query with trigger term", expected_safe_module),
        ("query designed to confuse router", expected_safe_module),
        ...
    ]
    
    for query, expected_module in adversarial_queries:
        predicted_module = router(query)
        
        if predicted_module != expected_module:
            # Router is vulnerable; retrain or adjust thresholds
            alert_security_team(query, predicted_module, expected_module)
```

Retrain the router with adversarial examples included in training data.

**2. Module authentication and integrity verification:**

Cryptographically sign all modules and verify signatures before execution:

```python
import hashlib
import hmac

# Sign module at deployment time
def sign_module(module_code, secret_key):
    signature = hmac.new(secret_key, module_code.encode(), hashlib.sha256).hexdigest()
    return signature

# Verify before execution
def verify_module(module_code, signature, secret_key):
    expected_sig = hmac.new(secret_key, module_code.encode(), hashlib.sha256).hexdigest()
    if signature != expected_sig:
        raise SecurityError("Module signature mismatch; possible substitution attack")
    return True

# At runtime
if not verify_module(loaded_module.code, loaded_module.signature, SECRET_KEY):
    raise SecurityError("Module failed integrity check")
```

**3. Module sandboxing:**

Execute each module in an isolated sandbox with restricted permissions:

```python
import subprocess

def execute_module_sandboxed(module_code, input_data):
    # Run module in a subprocess with:
    # - No network access (unless explicitly needed)
    # - No file system access (only read allowed documents)
    # - Limited memory (prevent resource exhaustion)
    # - Timeout (prevent infinite loops)
    
    process = subprocess.Popen(
        ["python", "-c", module_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,  # 5 second timeout
        env={"ALLOWED_MODULES": "retriever,reranker"},  # Whitelist
    )
    
    try:
        stdout, stderr = process.communicate(input=input_data, timeout=5)
        return stdout
    except subprocess.TimeoutExpired:
        process.kill()
        raise SecurityError("Module execution timeout")
```

**4. Output validation per module:**

Verify that each module's output is valid and safe:

```python
def validate_module_output(module_name, output):
    if module_name == "retriever":
        # Output should be list of documents
        assert isinstance(output, list)
        assert all(isinstance(doc, Document) for doc in output)
        # Check documents don't contain sensitive data
        for doc in output:
            if contains_pii(doc.text):
                raise SecurityError(f"Module {module_name} output contains PII")
    
    elif module_name == "reranker":
        # Output should be scores
        assert all(0 <= score <= 1 for score in output)
    
    return True
```

**5. Module allowlist and versioning:**

Maintain a whitelist of approved modules and versions:

```python
APPROVED_MODULES = {
    "retriever": ["v1.0", "v1.1", "v2.0"],
    "reranker": ["v1.0"],
    "summarizer": ["v1.2"]
}

def can_use_module(module_name, version):
    if module_name not in APPROVED_MODULES:
        raise SecurityError(f"Module {module_name} not in allowlist")
    
    if version not in APPROVED_MODULES[module_name]:
        raise SecurityError(f"Module {module_name} version {version} not approved")
    
    return True

# Before using a module
can_use_module(router_output_module, router_output_version)
```

**6. Cascade limit and information flow control:**

Limit how modules can chain and what data flows between them:

```python
MAX_CASCADE_DEPTH = 3

def execute_pipeline_controlled(query, modules_to_invoke):
    intermediate_results = {}
    
    for depth, module in enumerate(modules_to_invoke):
        if depth >= MAX_CASCADE_DEPTH:
            raise SecurityError("Cascade depth exceeded")
        
        # Module can only see its input, not outputs from other modules
        module_input = {
            "query": query,
            "previous_output": intermediate_results.get(module.input_module)
        }
        
        # Redact sensitive fields from previous output
        module_input = redact_sensitive_fields(module_input, module.allowed_fields)
        
        output = execute_module_sandboxed(module.code, module_input)
        intermediate_results[module.name] = output
    
    return intermediate_results
```

**7. Runtime monitoring and anomaly detection:**

Track module behavior and flag suspicious patterns:

```python
def monitor_module_behavior(module_name, input, output, execution_time):
    # Detect anomalies
    if execution_time > expected_latency[module_name] * 2:
        # Possible resource exhaustion attack
        alert_security_team(module_name, "Unusual latency")
    
    if len(output) > expected_output_size[module_name] * 10:
        # Module is outputting suspiciously much data
        alert_security_team(module_name, "Unusual output size")
    
    if output_contains_pii(output):
        # Module is leaking PII
        alert_security_team(module_name, "PII leak detected")
```

**Defence-in-depth strategy:**

1. Adversarial testing of router (frequent retraining).
2. Cryptographic signing and integrity verification.
3. Sandboxing and resource limits.
4. Output validation per module.
5. Allowlist and version control.
6. Cascade limits and data flow control.
7. Continuous monitoring and anomaly detection.

An attacker must defeat multiple layers, making successful attacks much harder.

</details>

---

## Q13. Walk through the Modular RAG architecture end-to-end. `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

```
Query ──► Routing Module ──► [Retrieval Module A, B, Memory Module] (parallel)
                                        │
                                  Fusion Module
                                        │
                                  Reranking Module
                                        │
                                  Generation Module → Answer
```

An Orchestrator wraps this whole flow, wiring modules together and handling fallback if a module fails or times out. The defining property visible in this diagram is that every box is a swappable interface, not a hard-coded implementation: the routing module can send a query to one retrieval module, several in parallel, or none at all (routing to memory only, for a conversational follow-up); which specific retrieval modules exist behind that router (vector, SQL, web, BM25) is a deployment-time configuration choice, not part of the architecture itself. This is what separates Modular RAG from Advanced RAG (#02), which wires a fixed set of enhancement stages in a fixed order.

</details>

---

## Q14. What is the research and practical origin of "Modular RAG" as a term? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

"Modular RAG" is most closely associated with Gao et al.'s survey work on RAG architectures (the same broad survey lineage — *Retrieval-Augmented Generation for Large Language Models: A Survey*, arXiv:2312.10997, 2023 — that provides much of this bank's own taxonomy framing for non-paper-specific architectures like Naive and Advanced RAG), which explicitly categorized RAG systems along a "Naive → Advanced → Modular" progression to describe increasing architectural flexibility rather than any single new algorithmic technique.

Unlike architectures with one specific originating paper and benchmark (DPR, RAPTOR, ColBERT), Modular RAG is a **structural pattern name** describing an engineering property — component swappability and composability — that emerged from practitioners building increasingly complex production RAG systems and needing a vocabulary for "a pipeline where the pieces aren't fixed," formalized retroactively by survey literature rather than introduced by one implementation.

</details>

---

## Q15. How does Modular RAG's orchestrator differ from Agentic RAG's (#04) LLM-driven orchestration? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Modular RAG's orchestrator is typically **deterministic, code-defined wiring**: a router module (which may itself be a small classifier or an LLM call) selects which modules to invoke based on query features, but the overall flow — router, then fusion, then rerank, then generate — is fixed at design time, and the "intelligence" is confined to the routing decision itself. Agentic RAG's (#04) orchestration is **LLM-driven at every step**: the LLM itself decides, dynamically and potentially differently each turn, whether to retrieve, which tool to call, whether to retrieve again, and when to stop — there's no fixed sequence of stages at all, only a loop the LLM navigates based on its own reasoning.

The practical distinction: Modular RAG's flexibility is in *which components exist and how they're wired* (a configuration choice made by engineers); Agentic RAG's flexibility is in *what happens at runtime, decided by the model itself*. A Modular RAG system with a sophisticated LLM-based router can start to resemble Agentic RAG, which is why the two are often described as points on a spectrum of increasing runtime flexibility rather than sharply distinct categories.

</details>

---

## Q16. What is the single distinctive mechanism that separates Modular RAG from Advanced RAG (#02)? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The distinctive mechanism is **componentization behind common interfaces, with routing deciding which components run per query**, replacing Advanced RAG's fixed sequence of named enhancement stages applied uniformly to every query. In Advanced RAG, every query passes through query rewriting, then hybrid retrieval, then reranking, then compression, in that order, every time. In Modular RAG, a routing module can send a finance query to a SQL retrieval module and a documentation query to a vector retrieval module, entirely skip reranking for a query the router judges unambiguous, or invoke the memory module alone for a conversational follow-up that needs no new retrieval at all.

This is precisely why Modular RAG is described as a generalization of Advanced RAG (#02 Q15) rather than a competing architecture: Advanced RAG's fixed pipeline is one valid configuration a Modular RAG router could always select, but Modular RAG additionally supports configurations Advanced RAG's fixed sequence cannot express at all — different pipelines for different query types within the same deployed system.

</details>

---

## Q17. What are the key design knobs when composing a Modular RAG pipeline? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Knob | Effect | Guidance |
|---|---|---|
| Router granularity (single classifier vs. cascade of checks) | Coarser routing is cheaper and simpler but less precise; finer-grained routing handles more query types correctly at more engineering cost | Start coarse (2-3 categories) and add routing categories only when evaluation shows a specific query type is being mis-routed |
| Module fallback ordering | Determines what happens when a preferred module fails, times out, or returns low-confidence results | Always define an explicit fallback chain (e.g., SQL module fails → fall back to vector search over a denormalized copy) rather than letting a module failure propagate as a user-facing error |
| Per-module timeout budget | Bounds how long the orchestrator waits for a slow module before proceeding without it | Set per-module budgets that sum to well under your end-to-end latency SLA, leaving headroom for fusion/rerank/generation |
| Parallel vs. sequential module invocation | Parallel invocation (as in this file's architecture diagram) minimizes latency when modules are independent; sequential is needed when one module's output feeds another's input | Default to parallel unless there's a genuine data dependency between modules |

Router granularity is the knob most teams get wrong in both directions — too coarse and query types that need genuinely different handling get routed identically (defeating the purpose of modularity); too fine and the routing layer itself becomes a maintenance burden with more edge cases than the modules it's routing between.

</details>

---

## Q18. How do you evaluate a Modular RAG system's routing accuracy separately from end-to-end accuracy? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

End-to-end accuracy alone can't tell you whether a wrong answer came from bad routing (query sent to the wrong module) or bad execution within a correctly-chosen module — these require different fixes, so they need separate metrics. Build a labeled set of (query, correct module or module combination) pairs, and measure **routing accuracy** directly: does the router send each query to the module(s) that can actually answer it, independent of whether that module's own retrieval and generation succeed.

Then, conditional on correct routing, measure each module's own retrieval/answer quality in isolation — this decomposition reveals whether a quality problem is systemic (a specific module underperforms regardless of how it's reached) or routing-specific (a module performs fine when reached directly but rarely gets routed to correctly). Track this per module and per query category over time, the same segmented-evaluation discipline used throughout this bank, since a router that was well-calibrated at launch can drift as query patterns evolve or new modules are added without a corresponding router update.

</details>

---

## Q19. What is the characteristic failure mode when modules have inconsistent output contracts? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Modularity's core promise — swap any retrieval module without touching the rest of the pipeline — breaks down if different modules return results in subtly incompatible shapes: one module's results carry a `score` field normalized 0-1, another's carry a raw BM25 score with no fixed range; one module tags results with `source_url`, another with `doc_id` and no URL at all. The fusion and reranking stages downstream (Q3 for fusion) assume a consistent contract, and a mismatch doesn't necessarily crash — it silently produces wrong fusion rankings (comparing incompatible score scales, exactly the RRF-motivating problem in #02 Q13) or missing citation metadata in the final answer.

**Detection:** this failure is easy to miss in testing if only one retrieval module is exercised per test case — it specifically requires a test that fuses results from *multiple* modules with genuinely different underlying implementations to surface. **Mitigation:** define a strict, versioned output schema every module must conform to (normalized score range, standard metadata field names) as part of the "pluggable retriever interface" contract (Q7), and validate every module's output against that schema at integration time — treating a schema violation as a hard error to catch during development, not a silent inconsistency to discover only when fusion rankings look subtly wrong in production.

</details>

---

## Q20. Design a Modular RAG system for a multi-domain enterprise assistant. `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** one assistant serving finance questions (needing SQL over structured financial data), documentation questions (needing vector search over a docs corpus), and current-events questions (needing live web search) — each domain has a genuinely different best-fit retrieval strategy.

```
1. Router: an LLM-based or fine-tuned classifier categorizes each query
   into finance / documentation / current-events / multi-domain,
   using query features (financial terminology, "latest"/"current"
   language per Agentic Web RAG's #31 routing cues, doc-specific
   jargon) -- coarse-grained (Q17), with a "multi-domain" catch-all
   category for queries that don't cleanly fit one bucket.

2. Modules behind a common interface (Q19): SQL retrieval module
   (Structured RAG's #12 text-to-SQL pattern) for finance; standard
   vector retrieval module for documentation; Agentic Web RAG's (#31)
   search-and-fetch module for current events -- each normalized to
   the same output contract (score, source metadata) before fusion.

3. Multi-domain handling: for queries the router flags as spanning
   multiple domains ("how does our latest pricing compare to
   competitor X's"), invoke multiple modules in parallel and fuse
   (Q3) rather than forcing a single-domain routing decision on a
   genuinely cross-domain query.

4. Fallback chain (Q17): if the SQL module times out or the query
   doesn't parse into valid SQL, fall back to a vector-indexed summary
   of the same financial data rather than surfacing an error --
   accepting reduced precision over a hard failure.

5. Observability (Q9): trace which module(s) each query was routed to
   and their individual latency/success, enabling per-domain debugging
   without needing to reproduce the full multi-module pipeline to
   diagnose a single domain's issue.
```

The key design choice is treating domain classification as the router's job and keeping each domain's retrieval strategy as an independently swappable module — this is exactly the scenario Modular RAG's architecture is built for, in contrast to Advanced RAG's single fixed pipeline, which has no way to send fundamentally different query types to fundamentally different retrieval mechanisms.

</details>

---

## Q21. A university IT helpdesk wants to merge three separate chatbots — password resets, WiFi setup, library hours — into one system. How would you structure it as Modular RAG? `[Basic]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The situation implies three existing, narrow bots, each already a small working pipeline over its own tiny document set, a small campus IT team, and a consolidation goal rather than a request for new capability. Query types are cleanly separable by topic and low in volume, which makes this a good fit for Modular RAG's simplest form rather than anything elaborate.

Treat each existing bot's retrieval logic as one module behind a common interface (Q7, Q16): a password-reset module, a WiFi-setup module, and a library-hours module, each keeping its own small index rather than being merged into one undifferentiated corpus. Add a lightweight router (Q17) with coarse, keyword- or intent-based classification into these three categories plus an "other" catch-all, since a full trained classifier is more machinery than three cleanly-separated topics need. Define an explicit fallback for "other" queries — a general campus-docs index, or a polite "I'm not sure, here's the help-desk contact" — rather than letting an unclassified query fall through silently.

The trade-off worth flagging: consolidating behind a shared router introduces a new failure point (misrouting) that didn't exist when the three bots were fully separate and independently simple. At this scale, logging routing decisions and reviewing misroutes weekly is enough oversight — a full routing-accuracy evaluation harness (Q18) can wait until query volume or category count grows.

</details>

---

## Q22. Design a telecom field-technician assistant whose retrieval and reranking modules must be swapped per region without redeploying the whole system. `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The hard constraint is zero-downtime, per-region module swaps at production scale: different regions run different equipment vendors, different regulatory documentation, and sometimes different languages, and a code redeploy to update one region's retriever is operationally unacceptable if it risks disrupting technicians in every other region simultaneously.

This requires treating the module interface contract (Q19) as a strict, versioned schema, and building a config-driven module registry keyed by region rather than a code-level module selection. A region's retriever or reranker becomes swappable via a configuration change — pointing that region's slot in the registry at a new, independently versioned and containerized module — with the fallback chain (Q17) automatically reverting to the last known-good module if a newly-swapped one fails health checks. Canary a new regional module in a single low-traffic region before wider rollout, rather than swapping everywhere at once. Keep each region's modules and data indexes strictly namespaced, since this is effectively a multi-tenant deployment and a region's retriever should never be able to reach another region's index even by misconfiguration.

What to monitor: per-region routing and module-health metrics (Q9), canary success/failure rates before a wider regional rollout, and rollback time when a swap goes wrong — the registry's value is only real if rollback is fast and automatic, not a manual incident-response process. The trade-off: a versioned, config-driven registry with canarying is substantially more operational tooling than a single fixed pipeline needs, but it's the only way to satisfy "swap without redeploy" as a hard constraint across many independently-evolving regions.

</details>

---

## Real-World Applications

| Application | Domain | Why Modular RAG Fits |
|---|---|---|
| Multi-tenant enterprise assistant (e.g., Glean, Microsoft Copilot for M365) | Enterprise | Different departments need different retrievers (SharePoint, Jira, Salesforce) — modular design lets each tenant swap retriever + reranker without touching generation |
| E-commerce conversational search | Retail | Product, review, and policy corpora require distinct retrieval strategies; a router module dispatches to the right index per query intent |
| Regulated industry knowledge platform (finance, pharma) | Compliance | Modules can be individually audited, versioned, and replaced as regulations change without rewriting the whole pipeline |
| Multilingual customer support | Global SaaS | Language-detection module gates to a locale-specific retriever and translation module, all without forking the pipeline |
| Research literature assistant | Academia / R&D | Abstract-only retrieval, full-text retrieval, and citation-graph retrieval are separate swappable modules depending on query depth |
