# RAG Quick Reference Cheatsheet

## RAG Types at a Glance

| # | Type | Key Mechanism | Best For | Avoid When |
|---|---|---|---|---|
| 1 | **Naive RAG** | Chunk → Embed → Top-k retrieval | Prototypes, small corpora | High precision needed |
| 2 | **Advanced RAG** | HyDE + Hybrid search + Reranking | Production Q&A | Latency is critical |
| 3 | **Modular RAG** | Swappable pipeline components | Custom enterprise pipelines | Simple single-source systems |
| 4 | **Agentic RAG** | LLM-controlled multi-step retrieval | Complex reasoning, tool use | Strict latency budgets |
| 5 | **Graph RAG** | Knowledge graph traversal | Relational data, multi-hop | Simple FAQ systems |
| 6 | **Corrective RAG** | Retrieval quality evaluation + fallback | Out-of-KB or stale corpora | All queries are well-covered |
| 7 | **Self-RAG** | LLM trained with reflection tokens | High-accuracy specialized domains | General chatbots, no fine-tuning |
| 8 | **Speculative RAG** | Drafter → Verifier pattern | Cost/quality optimization | Multi-hop queries |
| 9 | **Multi-modal RAG** | Cross-modal embeddings (CLIP) | Docs with images/tables | Text-only corpora |
| 10 | **Long-context RAG** | Full documents in context | Complex documents, small corpus | Large corpora, cost-sensitive |
| 11 | **Adaptive RAG** | Query classifier routes to no-retrieval / single-hop / multi-hop | Mixed-complexity query traffic, latency-sensitive | Homogeneous query distribution |
| 12 | **Structured RAG** | Text-to-SQL generation with schema linking | Relational databases, tabular data | Schema-free, unstructured data |
| 13 | **RAPTOR** | UMAP + GMM clustering → per-cluster LLM summaries → multi-level tree | Multi-hop questions, hierarchical docs | Frequently updated corpora (tree rebuild is expensive) |
| 14 | **Contextual RAG** | LLM prepends 1–2 sentence context prefix to each chunk before embedding | Dense corpora with ambiguous or short chunks | Token-budget-constrained indexing pipelines |
| 15 | **LightRAG** | Entity-relationship graph + dual local (entity) / global (community) retrieval | Queries mixing relational and semantic needs | Simple single-topic factoid Q&A |
| 16 | **RAFT** | Fine-tunes LLM generator on oracle docs + K distractor docs + CoT answers | Closed-domain high-stakes (medical, legal, internal KB) | Frequently changing corpus or when fine-tuning is off the table |
| 17 | **Cache-Augmented Generation (CAG)** | Precomputes KV cache for entire corpus; zero retrieval step at inference | Stable, bounded corpus that fits in context window | Large, dynamic, or multi-tenant corpora |
| 18 | **RAG-Fusion** | N query reformulations → N parallel retrievals → RRF merge → generation | Ambiguous queries, broad topic coverage | Hard latency budget (N× retrieval cost) |
| 19 | **Iterative / Multi-hop RAG** | Retrieve → reason → retrieve loops (IRCoT, Self-Ask, ITER-RETGEN) until a stopping criterion | Compositional multi-hop questions needing fact chaining | Simple single-fact queries; tight latency (sequential hops) |
| 20 | **HippoRAG** | Personalized PageRank over an LLM-built KG + synonym edges for single-step multi-hop | Entity-rich multi-hop on a stable, high-volume corpus | Entity-poor/abstract queries; fast-changing corpora |
| 21 | **Memory / Conversational RAG** | Tiered memory (working/summary/long-term) + history-aware query rewriting | Multi-turn assistants needing reference resolution and recall | Single-turn lookups; stateless-by-requirement domains |
| 22 | **HyDE** | Embed an LLM-generated hypothetical *answer* instead of the query | Zero-shot/unsupervised encoders, vocabulary-mismatched or cross-lingual queries | Fine-tuned in-domain retrievers; entity/factoid lookups |
| 23 | **FLARE** | Retrieve mid-generation when next-sentence token confidence < θ (forward-looking) | Long-form generation where information needs evolve | Short factoid answers; APIs without token logprobs |
| 24 | **KAG** | Logical-form-guided reasoning + KG↔text mutual indexing | Professional domains needing rule-following deduction + provenance (medical, legal, e-gov) | Open-domain/similarity-answerable Qs; fast-changing corpora |
| 25 | **GraphReader / GNN-RAG** | Agentic graph-of-notes traversal (GraphReader) / GNN-retrieved reasoning subgraphs (GNN-RAG) | Long-context multi-hop (GraphReader); KGQA over dense KGs (GNN-RAG) | Single-hop or similarity-answerable; no KG (GNN-RAG) |
| 26 | **REALM** *(training-time)* | Retriever learned end-to-end via masked-LM pre-training (latent-variable marginalization) | Research/learned-retrieval; updatable-corpus QA | When a frozen off-the-shelf retriever suffices |
| 27 | **RETRO** *(training-time)* | Chunked cross-attention over a trillion-token frozen datastore | Parameter-efficient LMs; large-scale knowledge LM | Most apps without trillion-scale datastore infra |
| 28 | **Atlas** *(training-time)* | Jointly-trained Contriever + FiD, attention-distillation | Few-shot knowledge tasks with scarce labels | When strong LLM + inference-time RAG already suffices |
| 29 | **Fusion-in-Decoder (FiD)** *(training-time)* | Encode each passage separately, fuse in the decoder | Generative reading/fusing of many retrieved passages | Single-passage reads; very tight decoder-latency budgets |

---

## Cost / Latency / Complexity by Architecture

Relative ratings for a typical mid-size deployment (●○○ low → ●●● high). Per-query cost assumes comparable answer quality targets.

| Type | Per-query cost | Latency | Build complexity | Ops complexity | Main cost driver |
|---|---|---|---|---|---|
| Naive RAG | ●○○ | ●○○ | ●○○ | ●○○ | LLM generation tokens |
| Advanced RAG | ●●○ | ●●○ | ●●○ | ●●○ | HyDE generation + reranker inference |
| Modular RAG | ●●○ | ●●○ | ●●● | ●●○ | Router + active modules |
| Agentic RAG | ●●● | ●●● | ●●● | ●●● | Multiple LLM calls per loop |
| Graph RAG | ●●○ | ●●○ | ●●● | ●●● | KG construction (index-time, LLM-heavy) |
| Corrective RAG | ●●○ | ●●○ | ●●○ | ●●○ | Retrieval evaluator LLM per query |
| Self-RAG | ●○○* | ●●○ | ●●● | ●●○ | *Up-front fine-tuning; cheap at inference |
| Speculative RAG | ●●○ | ●○○ | ●●● | ●●● | Parallel drafter GPUs + verifier scoring |
| Multi-modal RAG | ●●○ | ●●○ | ●●● | ●●○ | Image embedding at index time, vision LLM |
| Long-context RAG | ●●● | ●●● | ●○○ | ●○○ | Context tokens (linear in stuffed docs) |
| Adaptive RAG | ●○○ | ●○○ | ●●○ | ●●○ | Classifier (tiny); saves cost on easy queries |
| Structured RAG | ●●○ | ●●○ | ●●○ | ●●○ | Schema-in-prompt tokens + retry loops |
| RAPTOR | ●○○* | ●○○ | ●●● | ●●○ | *Up-front LLM summarization per cluster; cheap at query time |
| Contextual RAG | ●●○* | ●○○ | ●●○ | ●●○ | *One LLM call per chunk at index time (prompt caching cuts ~94%) |
| LightRAG | ●●○ | ●●○ | ●●● | ●●● | Entity/relationship extraction at build time; graph ops at query time |
| RAFT | ●○○* | ●○○ | ●●● | ●●○ | *Up-front fine-tuning cost; inference same as base model |
| CAG | ●●●* | ●○○ | ●○○ | ●●○ | *High cold-start KV cache load; zero retrieval latency per query |
| RAG-Fusion | ●●○ | ●●○ | ●●○ | ●●○ | N × (reformulation + retrieval); parallelizable |
| Iterative / Multi-hop RAG | ●●● | ●●● | ●●○ | ●●○ | One LLM reasoning call per hop (sequential, not parallelizable) |
| HippoRAG | ●○○* | ●○○ | ●●● | ●●● | *Up-front OpenIE over whole corpus; query-time retrieval is LLM-free PageRank |
| Memory / Conversational RAG | ●●○ | ●●○ | ●●○ | ●●● | Query-rewriting LLM call + memory store reads/writes per turn |
| HyDE | ●●○ | ●●○ | ●○○ | ●○○ | Extra LLM generation (the hypothetical) before retrieval |
| FLARE | ●●○ | ●●● | ●●○ | ●●○ | Re-generation of low-confidence sentences + interleaved retrievals |
| KAG | ●●○* | ●●○ | ●●● | ●●● | *Heavy KG build + extraction; multi-step parse/execute/compose per query |
| GraphReader | ●●● | ●●● | ●●○ | ●●○ | LLM call per exploration step (sequential agentic traversal) |
| GNN-RAG | ●○○* | ●○○ | ●●● | ●●● | *Up-front GNN training + KG upkeep; cheap GNN pass + one LLM call at query |
| REALM | ●○○ | ●○○ | ●●● | ●●● | *Heavy training (async index refresh); inference is RAG-like |
| RETRO | ●●○ | ●●○ | ●●● | ●●● | Per-chunk retrieval + huge frozen datastore storage/serving |
| Atlas | ●●○ | ●●○ | ●●● | ●●● | Joint training + index refresh; FiD encoder cost linear in passages |
| Fusion-in-Decoder | ●●○ | ●●○ | ●●○ | ●●○ | Decoder cross-attention over all passage tokens (grows with k) |

---

## Failure Modes: Symptom → Likely Cause → First Fix

| Symptom | Likely cause | First fix | Deep dive |
|---|---|---|---|
| Answer contradicts the retrieved docs | LLM ignores context (parametric memory wins) | Harden prompt ("answer ONLY from context"), measure faithfulness | [Hallucination](../03_failure_modes/01-hallucination_despite_context.md) |
| Relevant doc exists but never retrieved | Semantic gap between query and chunk wording | Hybrid search (BM25 + dense), query rewriting | [Retrieval Failure](../03_failure_modes/02-retrieval_failure.md) |
| Recall collapsed after a model/pipeline change | Query and docs embedded with different models/versions | Re-embed entire corpus with one model; version-stamp vectors | [Embedding Mismatch](../03_failure_modes/03-embedding_mismatch.md) |
| Confidently wrong answers about recent facts | Index lags source-of-truth updates | Incremental indexing + TTL/versioning; freshness alerts | [Stale Index](../03_failure_modes/04-stale_index_problem.md) |
| Answers degrade as k or doc size grows | Context overflow → truncation or lost-in-the-middle | Rerank then cut to top 3–5; compress context | [Context Overflow](../03_failure_modes/05-context_window_overflow.md) |
| Good docs retrieved but ranked below junk | Reranker domain mismatch or score miscalibration | Evaluate reranker on domain pairs; swap or fine-tune | [Reranker Failure](../03_failure_modes/06-reranker_failure.md) |
| Multi-turn answers degrade; wrong context retrieved after topic change or pronoun use | Conversation history poisons the retrieval query (coreference, implicit carry-over) | Query condensation: rewrite history + new turn into a standalone query before retrieval | [Conversational Context Drift](../03_failure_modes/07-conversational_context_drift.md) |

---

## Chunking Quick-Pick

| Content type | Strategy | Typical size | Notes |
|---|---|---|---|
| Uniform prose (articles, wikis) | Recursive splitting | 256–512 tokens, 10–20% overlap | Sensible default; respect paragraph boundaries |
| Long structured docs (manuals, contracts) | Parent-child | Child 200–300, parent 1,000+ | Retrieve on child precision, generate with parent context |
| Q&A / FAQ content | One Q&A pair per chunk | Natural unit | Never split an answer from its question |
| Code | Function/class boundary | Natural unit | Syntax-aware splitters; keep signatures with bodies |
| Tables | Whole table + caption per chunk | Natural unit | Serialize to markdown; never split rows from headers |
| Mixed/unknown | Semantic chunking | Variable | Embedding-similarity breakpoints; costs an embedding pass |
| Cross-chunk context critical (references span chunks) | Late Chunking | Variable | Embed full doc first → pool token embeddings into windows; requires token-level model (JinaAI v3, nomic-embed-text) |

Calibrate on your own data: build a small labeled probe set, sweep chunk size/overlap, measure Recall@5 — see [chunking_strategies.md](../01_concepts/chunking_strategies.md).

---

## Evaluation Metrics

| Metric | What it measures | Tool |
|---|---|---|
| Context Precision | Fraction of retrieved chunks that are relevant | RAGAS |
| Context Recall | Fraction of relevant info that was retrieved | RAGAS |
| Faithfulness | Does the answer only use the retrieved context? | RAGAS, TruLens |
| Answer Relevance | Does the answer actually address the question? | RAGAS |
| MRR | Rank of first relevant result | Custom |
| NDCG | Quality of ranked retrieval results | Custom |
| Latency P95 | 95th percentile end-to-end response time | Infrastructure |

Production-side evaluation (LLM-as-judge, online metrics, drift alerts): see [observability_and_evaluation_ops.md](../01_concepts/observability_and_evaluation_ops.md).

---

## Common Tools by Layer

### Embedding Models
- `text-embedding-3-small/large` (OpenAI)
- `BGE-large-en-v1.5` (BAAI, open-source)
- `E5-mistral-7b-instruct` (Microsoft, open-source)
- `Cohere Embed v3`
- `jina-embeddings-v3` (JinaAI — token-level output, required for Late Chunking)
- `nomic-embed-text` (open-source, token-level output, ColBERT / Late Chunking compatible)

### Vector Databases
- **Pinecone** — managed, production-grade
- **Weaviate** — hybrid search built-in
- **Qdrant** — fast, open-source
- **Chroma** — lightweight, local dev
- **FAISS** — library, not a server

### RAG Frameworks
- **LangChain** — general LLM orchestration
- **LlamaIndex** — data-centric RAG; has RAPTOR and Contextual Retrieval integrations
- **Haystack** — modular, open-source
- **LangGraph** — stateful agentic workflows
- **RAGatouille** — ColBERT / late-interaction retrieval (one-line index + retrieve)
- **LightRAG** — graph-based dual retrieval (pip install lightrag-hku)

### Evaluation Frameworks
- **RAGAS** — retrieval + generation metrics
- **TruLens** — LLM app evaluation
- **DeepEval** — unit-test style LLM eval
- **Arize Phoenix** — tracing + eval

### Rerankers
- `ms-marco-MiniLM-L-6-v2` (cross-encoder)
- `Cohere Rerank`
- `Jina Reranker v2`

---

## Decision Tree: Which RAG to Use?

```
Start
  │
  ├─ Is the primary data source a relational database or tabular store?
  │     └─ YES → Structured RAG
  │
  ├─ Is your corpus stable, bounded, and fits in a context window?
  │     └─ YES → Cache-Augmented Generation (CAG)  ← no retrieval step needed
  │
  ├─ Is your corpus relational / entity-heavy (graph-structured)?
  │     ├─ Large corpus, mandatory global summaries needed → Graph RAG (Microsoft)
  │     └─ Lower build cost, dual local+global retrieval → LightRAG
  │
  ├─ Does query complexity vary widely (trivial to multi-hop)?
  │     └─ YES → Adaptive RAG
  │
  ├─ Does your query require multiple retrieval steps or tool calls?
  │     └─ YES → Agentic RAG
  │
  ├─ Does your corpus include images/tables?
  │     └─ YES → Multi-modal RAG
  │
  ├─ Is your corpus small (<50 docs) and complex?
  │     └─ YES → Long-context RAG
  │
  ├─ Is your corpus hierarchical or multi-level (needs multi-hop summaries)?
  │     └─ YES → RAPTOR
  │
  ├─ Are chunks short / ambiguous and retrieval recall is low?
  │     └─ YES → Contextual RAG  ← LLM-generated context prefix per chunk
  │
  ├─ Are queries often ambiguous or benefit from multiple phrasings?
  │     └─ YES → RAG-Fusion  ← N reformulations + RRF merge
  │
  ├─ Is retrieval accuracy critical in a closed, specialized domain?
  │     ├─ Can fine-tune the LLM generator → RAFT
  │     └─ Cannot fine-tune → Self-RAG (if reflection tokens available)
  │
  ├─ Is your knowledge base potentially outdated/incomplete?
  │     └─ YES → Corrective RAG
  │
  ├─ Do you need cost/quality optimization at scale?
  │     └─ YES → Speculative RAG
  │
  ├─ Do you need a flexible, customizable pipeline?
  │     └─ YES → Modular RAG
  │
  ├─ Is this a production system needing good precision?
  │     └─ YES → Advanced RAG
  │
  └─ Prototyping or simple use case?
        └─ YES → Naive RAG
```

---

## Common Pitfalls Checklist

Run through this before any RAG interview (or production launch):

**Retrieval**
- [ ] Pure dense retrieval misses exact terms (IDs, SKUs, names) — hybrid search is the default answer
- [ ] Query and corpus MUST use the same embedding model and version — re-embed everything on model change
- [ ] Top-k similarity scores being "high" doesn't mean results are relevant — scores are not calibrated probabilities
- [ ] Highly selective metadata filters can degrade HNSW recall — know pre- vs. post-filtering trade-offs

**Chunking & Context**
- [ ] Bigger k is not better — irrelevant chunks dilute attention and increase hallucination
- [ ] Chunks that split mid-table or mid-sentence poison retrieval — validate chunk boundaries on real docs
- [ ] Lost-in-the-middle: order matters; put the strongest evidence first or last

**Generation**
- [ ] No citation/grounding mechanism = no way to audit answers — always attribute chunks
- [ ] Retrieved content is untrusted input — delimit it and defend against indirect prompt injection
- [ ] "I don't know" must be a designed behavior, not an accident — define the no-answer path

**Evaluation & Ops**
- [ ] No golden dataset = no way to know if a change helped — build one before optimizing
- [ ] Offline metrics passing ≠ production healthy — track online signals (regeneration rate, escalations)
- [ ] Index freshness needs an SLA and an alert — stale answers look identical to correct ones

**Security & Access**
- [ ] ACL filtering must happen in the retrieval layer, not after caching/reranking — filtered docs must never re-enter
- [ ] Multi-tenant: a metadata filter bug = cross-tenant data leak — test with cross-tenant probes in CI
- [ ] LLM-generated SQL needs read-only roles + allowlisted views, not just "injection-safe" prompts
