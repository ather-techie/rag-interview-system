# 03 — Modular RAG

> Treats the RAG pipeline as a set of interchangeable modules — retriever, reranker, reader, memory, router.

---

## Q1. What is Modular RAG and why is it an improvement over fixed pipelines? `[Basic]`

**Answer:**

Modular RAG decomposes the RAG system into independent, swappable components:

- **Search module** — vector search, keyword search, web search
- **Memory module** — short-term (conversation) and long-term (knowledge base)
- **Fusion module** — merges results from multiple retrievers
- **Routing module** — decides which retriever to call for a given query
- **Reader/Generator module** — the LLM that synthesizes the answer

Unlike Naive or Advanced RAG's fixed pipelines, Modular RAG lets you **add, remove, or swap** components without redesigning the whole system. This maps well to production engineering where different use cases need different retrieval strategies.

---

## Q2. How does a routing module work in Modular RAG? `[Intermediate]`

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

---

## Q3. What is a fusion retriever and when would you use one? `[Intermediate]`

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

---

## Q4. How do you handle memory in a Modular RAG system for multi-turn conversations? `[Advanced]`

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

---

## Q5. Compare Modular RAG to the LangChain and LlamaIndex frameworks. Which maps better? `[Advanced]`

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

---

## Q6. What is the "reader" module and how does it differ from a plain LLM call? `[Intermediate]`

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

---

## Q7. How do you implement a pluggable retriever interface for Modular RAG? `[Intermediate]`

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

---

## Q8. What is iterative retrieval and when does it outperform single-shot retrieval? `[Advanced]`

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

---

## Q9. How do you add observability (tracing + metrics) to a modular RAG pipeline? `[Advanced]`

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

---

## Q10. How do you perform online A/B testing of individual RAG modules in production? `[Advanced]`

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
