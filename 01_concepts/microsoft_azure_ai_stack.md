# The Microsoft / Azure AI Stack for RAG

> Which Microsoft service or open-source project fills each stage of a RAG pipeline, what you can run for free, and how to defend the choice in an interview.

---

## What is the "Microsoft stack" for RAG?

It is the set of Microsoft-built services and open-source projects that cover the RAG pipeline end to end: ingestion (Azure AI Document Intelligence), retrieval (Azure AI Search, Azure Cosmos DB, Azure Database for PostgreSQL), generation (Azure OpenAI, Phi models, Azure AI Foundry model catalog), orchestration (Semantic Kernel, Microsoft Agent Framework, AutoGen), safety (Azure AI Content Safety, Microsoft Presidio) and evaluation (Azure AI Foundry evaluators). It matters in interviews because many enterprise RAG roles are Azure-first, and the usual follow-up to "how would you build this?" is "on which platform, and why?"

> **Naming note.** Microsoft has renamed this product family several times (Azure OpenAI Service, Azure AI Studio, Azure AI Foundry, Microsoft Foundry). Concepts stay stable; names and SKUs change. Check the current Microsoft Learn docs before quoting a product name, tier, or limit.

---

## Pipeline Stage → Microsoft Service

```
                         ┌────────────────── Identity: Microsoft Entra ID / managed identity ──────────────────┐
                         │                                                                                    │
 Sources                 ▼                                                                                    │
 (Blob, SharePoint,  Ingest & parse          Index & retrieve              Generate               Safety     │
  OneLake, SQL)      ───────────────►   ───────────────────────►   ───────────────────►   ─────────────  │
                     Document            Azure AI Search               Azure OpenAI /        Content Safety │
                     Intelligence        (hybrid + semantic ranker)    Foundry models        Presidio (PII) │
                     + chunking          Cosmos DB / PostgreSQL        Phi (small / local)                  │
                                                                                                            │
 Orchestration: Semantic Kernel · Microsoft Agent Framework · AutoGen                                       │
 Evaluation & tracing: Azure AI Foundry evaluators · OpenTelemetry · Application Insights ◄─────────────────┘
```

| Stage | Microsoft option | Open-source / neutral alternative | Reference in this repo |
|---|---|---|---|
| Document parsing / OCR | Azure AI Document Intelligence | Unstructured, LlamaParse, Docling | [Document Ingestion](./document_ingestion_and_parsing.md) |
| Vector + keyword retrieval | Azure AI Search (vector, BM25, hybrid with RRF, semantic ranker) | Elasticsearch / OpenSearch, Qdrant, Weaviate | [Vector Databases](./vector_databases.md) |
| Operational DB with vectors | Azure Cosmos DB (DiskANN index), Azure Database for PostgreSQL (`pgvector`, `pg_diskann`) | Postgres + `pgvector`, MongoDB Atlas | [Vector Databases](./vector_databases.md) |
| Billion-scale ANN | DiskANN (Microsoft Research) | HNSW, ScaNN, FAISS | [Vector Databases](./vector_databases.md) |
| Embeddings | Azure OpenAI `text-embedding-3`, E5 (Microsoft Research) | BGE, Cohere, OpenAI direct | [Embeddings](./embeddings.md) |
| Reranking | Azure AI Search semantic ranker | Cohere Rerank, `bge-reranker` | [Reranking](./reranking.md) |
| Generation | Azure OpenAI, Phi via Foundry, Foundry model catalog | Anthropic, Gemini, Llama via vLLM / Ollama | [Cost Optimization](./cost_optimization.md) |
| Graph retrieval | `microsoft/graphrag` | LightRAG, Neo4j-based KG-RAG | [Knowledge Graph Construction](./knowledge_graph_construction.md) |
| Agent orchestration | Microsoft Agent Framework, Semantic Kernel, AutoGen | LangGraph, LlamaIndex, CrewAI | [Agentic Orchestration](./agentic_orchestration.md) |
| Safety / PII | Azure AI Content Safety, Azure AI Language PII detection, Microsoft Presidio | Llama Guard, spaCy + regex | [Prompt Injection Risks](./prompt_injection_risks.md) |
| Evaluation | Azure AI Foundry evaluators (groundedness, relevance, retrieval) | RAGAS, TruLens, DeepEval | [Evaluation Metrics](./evaluation_metrics.md) |
| Prompt compression | LLMLingua (Microsoft Research) | Selective Context | [Long-context RAG](../02_interview_bank/10-long-context-rag.md) |

---

## Running It Without an Azure Subscription

You can practise Microsoft-aligned RAG without paying for Azure:

| Need | Free option | Caveat |
|---|---|---|
| Local LLM for prototyping | **Foundry Local**, or **Phi** models through Ollama (both expose OpenAI-style local APIs) | Quality and speed depend on your hardware; small models behave differently from production models |
| Local embeddings | Open-source models such as E5 (Microsoft Research) via `sentence-transformers` or Ollama | You must re-embed if you later switch to a different production embedding model |
| Graph RAG | `microsoft/graphrag` against a local model (Ollama / Foundry Local) | Indexing makes many LLM calls; local models are slow and lower quality for extraction |
| Agent framework | Microsoft Agent Framework / Semantic Kernel (open source) | APIs are still evolving; pin versions |
| PII redaction | Microsoft Presidio (open source) | Rule/NER-based; tune for your data |

The repo's lab client ([`06_labs_py/ai_client.py`](../06_labs_py/ai_client.py)) already supports `AI_PROVIDER=azure` for readers who do have a subscription.

---

## Choosing Between Azure AI Search, Cosmos DB, and PostgreSQL

| If the main need is... | Prefer | Why |
|---|---|---|
| Hybrid search, semantic reranking, document-level security trimming, built-in ingestion | Azure AI Search | Purpose-built retrieval service with BM25 + vector fusion and a managed reranker |
| Vectors next to operational data, one datastore, low-latency point reads | Azure Cosmos DB with the DiskANN index | Avoids a second system and a sync pipeline |
| Existing Postgres estate, SQL joins with vector search | Azure Database for PostgreSQL with `pgvector` / `pg_diskann` | Reuses SQL skills, transactions, and tooling |
| Portability across clouds | `pgvector` or Qdrant on any host | Lowest lock-in |

The honest trade-off: Azure AI Search gives you the most retrieval features out of the box, at the cost of a separate service to run, pay for, and keep in sync with the source of truth.

---

## Interview Q&A

**Q: Walk through a RAG architecture on Azure and justify each service.** `[Intermediate]`

Documents land in Blob Storage or SharePoint; Azure AI Document Intelligence extracts text and layout (tables, headings); chunks are embedded with an Azure OpenAI embedding deployment and written to an Azure AI Search index with metadata and access-control fields. At query time the app runs a hybrid query (BM25 + vector, fused with RRF), optionally applies the semantic ranker, filters by the caller's identity, and sends the top chunks to an Azure OpenAI chat deployment. Content Safety screens inputs and outputs, Cosmos DB holds chat state, and Foundry evaluators plus Application Insights cover quality and tracing. Each choice is defensible on managed operations, Entra ID integration, and private networking; the cost is lock-in and per-service pricing.

---

**Q: When would you pick Azure AI Search over a self-hosted vector database?** `[Intermediate]`

Pick it when you need hybrid retrieval, a managed semantic reranker, built-in indexers and skillsets, and document-level security without operating infrastructure, especially if the organization is already on Azure and requires Entra ID and private endpoints. Pick a self-hosted store (Qdrant, `pgvector`, OpenSearch) when you need portability, tight cost control at high volume, or custom index behavior. Either way, measure recall and latency on your own corpus rather than trusting feature lists.

---

**Q: What is the Microsoft Agent Framework and how does it relate to Semantic Kernel and AutoGen?** `[Intermediate]`

Microsoft Agent Framework is Microsoft's consolidated open-source framework for building agents and multi-agent workflows in Python and .NET, positioned as the successor that brings together Semantic Kernel's enterprise-oriented building blocks and AutoGen's multi-agent patterns. For a RAG interview the useful framing is: retrieval is exposed as a tool the agent can call, the framework manages the tool-call loop and state, and you choose it over LangGraph mainly for Microsoft ecosystem fit and .NET support. Its status and APIs change quickly, so confirm the current docs before committing.

---

**Q: How do you do per-user security trimming in a RAG system on Azure?** `[Advanced]`

Store the access-control attributes (user or group IDs) as filterable fields on each indexed chunk, resolve the caller's identity and group memberships at query time (Microsoft Entra ID, ideally with an on-behalf-of flow so the search runs as the user), and apply them as a filter on the search request so unauthorized chunks are never retrieved. Trimming after retrieval or after generation is unsafe: the model may already have seen restricted text. Keep the ACL data in sync with the source system and test with users who should and should not see a document.

---

**Q: How would you evaluate a RAG system on Azure, and how does it map to RAGAS?** `[Intermediate]`

Azure AI Foundry provides built-in evaluators for groundedness, relevance, retrieval quality and response completeness that you can run over a golden dataset in CI. Groundedness corresponds to RAGAS faithfulness (see [Evaluation Metrics](./evaluation_metrics.md)); retrieval and relevance evaluators play the role of context precision/recall and answer relevancy. Whatever tool you use, the process is the same: a versioned golden set, LLM-as-judge plus retrieval metrics, a regression gate on pull requests, and production sampling for drift.

---

**Q: Your team cannot get an Azure subscription yet. How do you prototype Microsoft-stack RAG?** `[Basic]`

Run small models locally with Foundry Local or Phi through Ollama (GitHub Models, which used to offer a free hosted endpoint, was retired on July 30, 2026, so don't plan around hosted free tiers), and use open-source Microsoft projects (Semantic Kernel or Agent Framework, `microsoft/graphrag`, Presidio) for orchestration, graph retrieval and PII handling. Keep the model client behind a thin provider interface so moving to Azure OpenAI later is a configuration change, not a rewrite.

---

## Related

- [Vector Databases](./vector_databases.md) — DiskANN, Cosmos DB, and `pg_diskann` in depth
- [Agentic Orchestration](./agentic_orchestration.md) — framework selection including Microsoft Agent Framework
- [Document Ingestion & Parsing](./document_ingestion_and_parsing.md) — Azure AI Document Intelligence
- [Evaluation Metrics](./evaluation_metrics.md) — groundedness vs. faithfulness
- [Multi-Tenancy & Access Control](./multi_tenancy_access_control.md) — isolation and document ACL patterns
- [Graph RAG](../02_interview_bank/05-graph-rag.md) and [LazyGraphRAG](../02_interview_bank/47-lazygraphrag.md) — Microsoft Research approaches to graph retrieval
