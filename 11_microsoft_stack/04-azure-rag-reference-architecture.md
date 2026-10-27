# 04 — RAG Reference Architecture on Azure

> End-to-end RAG on the Microsoft stack: ingestion, Azure AI Search, Azure OpenAI / Foundry models, safety, identity, networking, cost, resilience and build-vs-buy trade-offs.

---

## 🏗️ Reference Architecture

```
[Ingestion - offline / event-driven]
Blob / OneLake / SharePoint ──► Document Intelligence (OCR + layout)
                                      │
                                      ▼
                           Chunker + enrichment ──► Embedding model (Azure OpenAI)
                                                          │
                                                          ▼
                                              Azure AI Search index
                                      (text + vector + metadata + ACL fields)
                                                          ▲
[Online - per request]                                    │
User ► Entra ID sign-in ► App API (Container Apps / App Service)
         │                    │  1. Content Safety / Prompt Shields (input)
         │                    │  2. Query rewrite ► hybrid + semantic ranker ──┘
         │                    │     (security filter from caller identity)
         │                    │  3. Prompt = instructions + top chunks ► Azure OpenAI / Foundry model
         │                    │  4. Content Safety (output) + citations
         │                    ▼
         │              Cosmos DB (chat history, feedback)
         └──────────────► App Insights / Azure Monitor (traces, metrics, eval logs)
```

| Component | Responsibility |
|---|---|
| Azure Blob / OneLake / SharePoint | Source-of-truth documents; change feed or indexer schedule drives re-ingestion |
| Azure AI Document Intelligence | OCR and layout extraction (tables, headings, reading order) before chunking |
| Azure AI Search | Hybrid (keyword + vector) retrieval, semantic ranker, filters, security trimming |
| Azure OpenAI / Foundry models | Embeddings and chat completion; deployment type controls cost, residency and throughput |
| Azure AI Content Safety | Harm categories and Prompt Shields on user input and retrieved documents |
| Azure Cosmos DB | Chat history, session state, feedback; optional vector index (DiskANN) |
| Container Apps / App Service / Functions / AKS | Hosts the orchestration API and front end |
| Microsoft Entra ID + managed identities | User sign-in, token exchange, passwordless service-to-service auth |
| Application Insights / Azure Monitor | Distributed traces, token and latency metrics, alerts |
| Bicep + azd | Repeatable infrastructure and deployment |

---

## Q1. Walk through the reference RAG pipeline on Azure and name the job of each service. `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Offline, documents land in Blob Storage, OneLake or SharePoint. Azure AI Document Intelligence extracts text and layout, a chunker splits the result, an embedding model (Azure OpenAI) vectorizes the chunks, and Azure AI Search stores text, vectors, metadata and access-control fields.

Online, the user signs in through Microsoft Entra ID. The app API checks the input (Content Safety / Prompt Shields), optionally rewrites the query, runs hybrid search with the semantic ranker plus a security filter, builds a prompt from the top chunks, calls an Azure OpenAI / Foundry chat model, checks the output, returns the answer with citations, and writes the turn to Cosmos DB. Everything emits telemetry to Application Insights.

Key point to say out loud in an interview: each box is replaceable. Search could be Cosmos DB or PostgreSQL, hosting could be AKS, the model could be a Foundry catalog model. What stays constant is the separation of ingestion, retrieval, generation, safety, identity and observability.

Reference: [RAG solution design and evaluation guide](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide)

</details>

---

## Q2. What is the `Azure-Samples/azure-search-openai-demo` repo and how should you use it in a real project? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

It is Microsoft's reference "chat over your data" application. Per its README it uses Azure OpenAI for the chat model, Azure AI Search for retrieval, Blob Storage for documents, Document Intelligence for layout extraction, Application Insights for tracing, and Azure Container Apps as the default host (App Service is an alternative). Optional pieces include Cosmos DB for persistent chat history, Entra login with document-level access control, multimodal support, and private-networking guidance. It deploys with `azd up` using Bicep templates.

How to use it: treat it as a working architecture to read and fork, not a product. It is valuable because it shows the integration details that are easy to get wrong (managed identity wiring, ACL trimming, citation rendering, prep scripts, IaC). Before production you still need your own evaluation set, content-safety policy, network isolation, capacity planning and CI/CD gates.

Check the repo for the current default model names and feature flags, since they change frequently: [github.com/Azure-Samples/azure-search-openai-demo](https://github.com/Azure-Samples/azure-search-openai-demo)

</details>

---

## Q3. Why put Azure AI Document Intelligence in front of chunking instead of using plain text extraction? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Plain text extractors break on scanned PDFs (no text layer), multi-column pages, and tables. Document Intelligence's layout model provides OCR plus structure: paragraphs, headings, reading order, tables and selection marks, and it can emit Markdown. That structure lets the chunker split on headings and keep a table intact instead of cutting it mid-row.

| Input | Plain extraction | Layout model |
|---|---|---|
| Scanned PDF | Empty or garbage | OCR text |
| Two-column report | Interleaved lines | Correct reading order |
| Financial table | Flattened cells | Table as Markdown / structured cells |

Trade-off: it is a paid per-page service and adds ingestion latency, so run it only for formats that need it (for example, skip it for born-digital text files) and cache results keyed by file hash. Alternatives include Azure AI Content Understanding, open-source parsers (Unstructured, Docling) or a multimodal LLM for image-heavy pages.

Reference: [Document Intelligence layout model](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout)

</details>

---

## Q4. Why use managed identities instead of API keys between the app, Azure AI Search and Azure OpenAI? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A managed identity gives the app a Microsoft Entra identity whose tokens are issued and rotated by the platform. You grant it narrowly scoped Azure RBAC roles on each resource, so there are no secrets to store, leak, or rotate.

Typical assignments for the app's identity:

- Azure AI Search: `Search Index Data Reader` (query), and for the ingestion job `Search Index Data Contributor`.
- Azure OpenAI / Foundry: a Cognitive Services OpenAI User style role.
- Blob Storage: `Storage Blob Data Reader`.
- Cosmos DB: a data-plane role assigned through Cosmos DB's own RBAC.

Benefits: per-resource least privilege, audit trail in Entra sign-in and activity logs, and the ability to disable key-based auth entirely (`disableLocalAuth`) as a policy. Use separate identities for the ingestion pipeline and the query API so a compromised web tier cannot rewrite the index. Caveat: a managed identity represents the application, not the end user, so it does not by itself solve per-user document security (see Q9).

Reference: [Managed identities overview](https://learn.microsoft.com/en-us/entra/identity/managed-identities-azure-resources/overview)

</details>

---

## Q5. Where do Azure AI Content Safety and Prompt Shields fit in a RAG pipeline? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Content Safety is a guardrail layer around the model, applied at three points:

1. **User input** - screen for harmful content and for **user prompt attacks** (jailbreak attempts) with Prompt Shields.
2. **Retrieved documents** - Prompt Shields can also scan grounding documents for **indirect (document) attacks**, which matters because RAG feeds untrusted text into the prompt.
3. **Model output** - screen for harm categories and, where configured, groundedness or protected-material checks.

Azure OpenAI also has built-in content filtering on deployments; Content Safety APIs are what you call explicitly when you need custom thresholds, blocklists, or checks on content that never touches an Azure OpenAI deployment (for example, documents at ingestion time).

Trade-offs: each call adds latency and cost, so run input checks in parallel with retrieval, and tune thresholds on your own traffic to balance false positives against missed attacks. Safety filters reduce risk; they do not replace least-privilege retrieval and not giving the model tools it should not have.

Reference: [Prompt Shields](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)

</details>

---

## Q6. How do you choose between Azure Container Apps, App Service, Functions and AKS to host a RAG API? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Host | Fits when | Watch out for |
|---|---|---|
| Container Apps | Default for containerized API; scale to zero, KEDA autoscale, VNet integration | Less control than Kubernetes |
| App Service | Team already runs web apps; simple deployment, deployment slots | Per-plan scaling, fewer container-native features |
| Functions | Event-driven ingestion (blob trigger, queue) and short tasks | Cold starts and execution limits for streaming chat |
| AKS | Platform team exists, need custom networking, GPUs, service mesh, self-hosted models | Highest operational burden |

Rule of thumb: Container Apps for the chat API, Functions or Container Apps jobs for ingestion, AKS only when you have a real reason (self-hosted open models on GPUs, existing cluster standards, strict multi-service networking). The demo repo defaults to Container Apps and documents App Service as an alternative.

Whatever you choose, keep the API stateless (history in Cosmos DB), stream responses with server-sent events, and size timeouts for slow model calls so the ingress does not cut off long generations.

</details>

---

## Q7. How would you design chat history and session state in Azure Cosmos DB for a RAG assistant? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Store one item per conversation turn (or one per conversation with an append-only message array if turns are small), and choose the partition key from the dominant access pattern: "load this user's conversation list and then one conversation". A common key is `/userId` (or `/tenantId` + hierarchical partition key with `userId`), which keeps a user's reads in one logical partition and cheap.

Design points:

- **Item contents**: conversation id, turn id, role, text, citations (doc ids and chunk ids, not whole chunks), model deployment name, token counts, latency, safety verdicts, feedback.
- **TTL**: set a retention TTL to meet privacy policy; chat logs often contain PII.
- **Context window**: send only the last N turns or a rolling summary to the model; store the full history for audit.
- **Identity**: use Cosmos DB data-plane RBAC with managed identity, not account keys.
- **Hot partitions**: a single very busy tenant on a tenant-only key can throttle; hierarchical partition keys mitigate this.

Alternatives: Redis for short-lived session state, PostgreSQL if the app is already relational. Cosmos DB is attractive for elastic scale, low-latency point reads, and multi-region writes.

Reference: [Partitioning in Azure Cosmos DB](https://learn.microsoft.com/en-us/azure/cosmos-db/partitioning-overview)

</details>

---

## Q8. When would you pick Azure AI Search, Cosmos DB DiskANN, or Azure Database for PostgreSQL (pgvector / pg_diskann) as the vector store? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Option | Strengths | Choose when |
|---|---|---|
| Azure AI Search | Hybrid search, semantic ranker, integrated vectorization, indexers, built-in document-level security, agentic retrieval | Document-centric RAG; default choice |
| Cosmos DB (NoSQL) with DiskANN vector index | Vectors co-located with operational JSON, partition-key scoping, global distribution, low-latency point reads | App data and vectors already in Cosmos DB; per-tenant or per-user partitioned vector search |
| PostgreSQL Flexible Server with pgvector / `pg_diskann` | SQL joins and transactions with vectors, familiar tooling, existing Postgres estate | Relational filtering and joins matter, team knows Postgres |

Decision drivers: (1) do you need hybrid + reranking out of the box (Search), (2) do vectors need transactional consistency with business data (Cosmos DB or PostgreSQL), (3) existing skills and estate, (4) scale and cost at your vector count. DiskANN keeps compressed vectors in memory and full vectors on SSD, which lowers memory cost at large scale (see `01_concepts/vector_databases.md`).

Do not decide on a benchmark blog. Load your own corpus and queries, measure recall at your latency target, and compare total cost. Non-Azure options (Pinecone, Qdrant, Weaviate, Elasticsearch/OpenSearch) are valid if you accept the extra operational and compliance surface.

Reference: [Vector search in Azure Cosmos DB for NoSQL](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search)

</details>

---

## Q9. How does per-user document security trimming work with Azure AI Search, and which approach would you choose? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Per Microsoft Learn ([document-level access control](https://learn.microsoft.com/en-us/azure/search/search-document-level-access-overview)), Azure AI Search offers several approaches:

| Approach | How it works | Status note |
|---|---|---|
| Security filters | Store user/group ids in a filterable string-collection field; app adds a filter such as `search.in(group_ids, 'g1,g2')` built from the caller's validated identity | Generally available, works with any API version and any identity system |
| Native ACL / RBAC scopes | Index stores ACL metadata; query carries the user's Entra token in the `x-ms-query-source-authorization` header and the service trims results | Preview as of late 2026, check current docs |
| SharePoint ACLs | SharePoint indexer extracts permissions into the index | Preview as of late 2026 |
| Purview sensitivity labels | Labels extracted and enforced with the user's token | Preview as of late 2026 |

Choice: for production today, security filters are the safe default because they are GA and identity-system agnostic; adopt native token-based trimming when it reaches GA and fits your sources, since it removes custom group-resolution code.

Non-negotiables regardless of approach: apply the filter inside the search query (before ranking and before the prompt), derive identity from a validated token and not from client-supplied fields, resolve nested groups carefully (token group claims overflow for users in many groups), and plan for permission sync lag between the source system and the index.

</details>

---

## Q10. How do you isolate a RAG system on a private network in Azure? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Goal: no data-plane traffic crosses the public internet and no PaaS service accepts public connections.

1. **Private endpoints** for Azure AI Search, Azure OpenAI / Foundry, Storage, Cosmos DB, Key Vault, Document Intelligence; set public network access to disabled on each.
2. **Private DNS zones** linked to the VNet so service hostnames resolve to private IPs (the classic failure is a missing DNS zone link).
3. **VNet-integrated hosting**: Container Apps environment or App Service with VNet integration; front door via Application Gateway or Front Door with WAF if users are external.
4. **Outbound control**: Azure Firewall or NAT gateway with allowlists; block everything else.
5. **Ingestion path**: indexers reach private data sources through shared private link (Search creates managed private endpoints to the sources) or trusted-service access; build agents need network line-of-sight too.
6. **Developer access**: Bastion, VPN, or a jump host; deployment pipelines need self-hosted runners inside the network.
7. **Identity still matters**: network isolation complements, and does not replace, RBAC and managed identity.

Trade-offs: more cost (private endpoints, firewall), harder local development, and harder troubleshooting. Define all of it in Bicep so it is reproducible; the demo repo has a private-networking guide. Monitor private endpoint connection states and DNS resolution failures.

Reference: [Private endpoints for Azure AI Search](https://learn.microsoft.com/en-us/azure/search/service-create-private-endpoint)

</details>

---

## Q11. Index-per-tenant versus filter-per-tenant in Azure AI Search: how do you decide? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Dimension | Filter-per-tenant (shared index) | Index-per-tenant |
|---|---|---|
| Isolation strength | Logical; one filter bug leaks data | Stronger boundary at the index level |
| Tenant count | Scales to many small tenants | Bounded by per-service index limits for your tier |
| Cost | Lowest; shared capacity | Higher; many small indexes waste capacity |
| Per-tenant schema/analyzer/embedding model | Not possible | Possible |
| Noisy neighbor | Shared resources | Still shared at the service level |
| Delete / export a tenant | Delete by filter, slower | Drop the index |
| Relevance | Shared statistics can skew BM25 | Per-tenant statistics |

Decision: many small tenants with the same schema, use a shared index with a mandatory `tenant_id` filter injected server-side (never by the client). Few large or regulated tenants, or tenants needing different keys, regions or embedding models, use index-per-tenant or even service-per-tenant. Hybrid is common: pooled for the long tail, dedicated for the top accounts.

Whatever you pick, add automated cross-tenant leakage tests to CI. Check current service limits for your tier before committing: [Azure AI Search service limits](https://learn.microsoft.com/en-us/azure/search/search-limits-quotas-capacity). See also `01_concepts/multi_tenancy_access_control.md`.

</details>

---

## Q12. What do you instrument for monitoring a RAG app with Application Insights, OpenTelemetry and Foundry tracing? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Use OpenTelemetry (via the Azure Monitor OpenTelemetry distro) so one trace follows a request across the API, Search call, model call and Cosmos DB write. Foundry tracing builds on OpenTelemetry and the GenAI semantic conventions, so model and agent spans can show up in the same pipeline.

| Signal | Why it matters |
|---|---|
| Per-stage latency (retrieve, rerank, generate, safety) | Find which stage blows the budget; track p50/p95/p99 |
| Time to first token | The user-perceived latency for streaming |
| Prompt and completion tokens, per deployment | Cost attribution and regression detection |
| Retrieval metadata: query, filter, top-k doc ids, scores | Debug wrong answers; spot empty retrievals |
| HTTP 429 / 5xx from Search and OpenAI | Capacity problems and retry storms |
| Safety filter hits, refusals | Policy drift and attack attempts |
| User feedback (thumbs) tied to trace id | Feeds the evaluation set |

Governance: prompts and completions can contain PII, so decide what to log, mask or sample, set retention, and restrict access to the Log Analytics workspace. Alert on error rate, p95 latency, 429 rate, token spend per day, and groundedness scores from sampled online evaluation.

Reference: [Application Insights OpenTelemetry](https://learn.microsoft.com/en-us/azure/azure-monitor/app/opentelemetry-enable)

</details>

---

## Q13. How do you handle 429s and transient failures from Azure OpenAI and Azure AI Search? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A 429 means you exceeded a token or request rate limit on that deployment (or the Search service is throttling). Handle it in layers:

1. **Retry with backoff**: honor the `Retry-After` header when present, otherwise exponential backoff with jitter, and cap attempts and total time. The Azure SDKs have built-in retry policies; do not stack your own on top without understanding the multiplier.
2. **Budget the request**: a retry loop that exceeds the user's latency budget is worse than a fast fallback. Set a per-request deadline.
3. **Circuit breaker**: after repeated failures, open the circuit and fail fast or route elsewhere instead of piling load onto a struggling endpoint.
4. **Spread load**: multiple deployments or regions behind an API Management AI gateway or app-level load balancer, with priority tiers (provisioned first, pay-as-you-go overflow).
5. **Reduce demand**: cap `max_tokens`, trim context, cache, and queue non-interactive work (batch or ingestion) separately from chat so it cannot starve users.
6. **Degrade gracefully**: fall back to a smaller model or return search results with citations if generation is unavailable.

Monitor 429 rate per deployment, retry counts, circuit state and queue depth. Treat quota as a capacity-planning input, not an error to be retried forever.

</details>

---

## Q14. How do you run RAG evaluation in CI with Foundry evaluators and use it as a regression gate? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Build a versioned evaluation dataset (questions, expected answers or reference passages, and tags for categories such as ACL-restricted, multi-hop, unanswerable). In CI, run the real pipeline against it and score outputs with evaluators from the Azure AI Evaluation SDK / Foundry: AI-assisted quality evaluators such as groundedness, relevance, retrieval and response completeness, plus safety evaluators, plus your own code-based checks (citation present, correct doc id retrieved).

Gate design:

- **Hard gates** (fail the build): ACL leakage tests, safety evaluator thresholds, retrieval hit rate on a golden subset not dropping beyond a tolerance versus the baseline.
- **Soft gates** (warn/report): LLM-judge scores, which are noisy; compare means with confidence intervals rather than single runs.
- **Triggers**: any change to prompts, chunking, index schema, model deployment version, or retrieval parameters.
- **Layers**: evaluate retrieval separately from generation so you know where a regression came from.

Caveats: LLM judges carry bias and cost tokens, so calibrate them against human labels on a sample and pin the judge model version. Then continue evaluating in production on sampled traces and feed failures back into the dataset.

Reference: [Evaluate generative AI apps with the Azure AI Evaluation SDK](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk)

</details>

---

## Q15. A regulated insurer wants a claims assistant where no customer can ever see another customer's PII. Design it on Azure. `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** per-customer isolation as a hard guarantee, auditability, PII minimization, regulatory retention, private networking, and acceptable latency for adjuster and customer use.

**Design:**

- **Identity**: users sign in with Entra ID (workforce) or Entra External ID (customers). The API validates the token and derives `customer_id` / `policy_ids` from claims or a server-side lookup; the client never supplies them.
- **Retrieval**: shared policy and procedure content in one index (no PII). Claim documents in an index filtered by a mandatory server-injected `customer_id` filter; for the strictest tier, dedicated index or service per business unit. Prefer structured claim data to be fetched by tool/API with the caller's permissions, not embedded.
- **PII minimization**: detect and redact or tokenize identifiers at ingestion (Azure AI Language PII detection or Presidio); store only what answers need.
- **Safety**: Prompt Shields on input and retrieved text, output screening, and a final deterministic check that the answer cites only documents from the allowed set.
- **Network and keys**: private endpoints everywhere, public access disabled, customer-managed keys where policy requires, managed identities.
- **State**: Cosmos DB partitioned by `customer_id`, TTL aligned to retention policy.

**Key trade-off:** physical isolation (index per customer) is the strongest guarantee but does not scale to millions of customers; logical filter isolation scales but moves the guarantee into code, so compensate with automated cross-tenant leakage tests, code review of the single retrieval wrapper, and no direct Search access from other code paths.

**Monitor:** filter-missing errors (fail closed), cross-customer citation violations (should be zero), PII detected in outputs, audit logs of who retrieved what, p95 latency, and 429 rates.

</details>

---

## Q16. An internal HR bot must answer from SharePoint, and each employee may only see documents they can open in SharePoint. How do you build it? `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** answers grounded in HR policy and case documents; permissions identical to SharePoint; permission changes reflected quickly; no bypass through the bot.

**Options, in order of how much custom code you own:**

1. **Do not build**: Microsoft 365 Copilot or a Copilot Studio agent grounded on SharePoint already enforces the user's permissions through Microsoft Graph. If the content lives only in M365 and the UX can be Teams/Copilot, this wins on time.
2. **Azure AI Search with SharePoint indexer ACLs** (preview as of late 2026, check current docs): the indexer extracts permission metadata; the query passes the user's Entra token in `x-ms-query-source-authorization` so the service trims results. See [SharePoint ACL ingestion](https://learn.microsoft.com/en-us/azure/search/search-indexer-sharepoint-access-control-lists).
3. **Custom security filter**: ingest SharePoint content plus permissions via Graph into a string-collection field (user and group object ids), and filter by the caller's id and group ids at query time.
4. **Retrieve via Graph** at query time with the user's delegated token (on-behalf-of flow) when freshness matters more than recall.

**Key trade-off:** an index copy is fast and supports hybrid search, but permissions go stale between syncs (inherited permission changes need explicit refresh); live Graph calls are always current but slower and weaker for ranking. For very sensitive documents, re-check access against SharePoint at answer time for the cited files.

**Monitor:** permission sync lag, indexer failures, count of documents with unresolved ACLs (fail closed), citation access re-check failures, and groundedness plus unanswerable-question rate.

</details>

---

## Q17. A support organization has 10 million documents and a p95 latency budget of 3 seconds for answers. How do you design for it? `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** p95 under 3 s end to end, high recall over a large corpus, frequent updates, moderate concurrency.

**Budget it per stage** (illustrative, measure your own): safety and query rewrite in parallel, retrieval a few hundred ms, rerank in the same order, then generation dominates. Use streaming so perceived latency is time to first token.

**Design:**

- **Index sizing**: estimate vectors times dimensions times bytes, plus overhead, and compare to the storage and vector quota of Search tiers; use partitions for size and replicas for QPS and availability. Reduce footprint with compression (scalar/binary quantization, truncated embedding dimensions) and keep full-precision vectors off the hot path where supported. Verify limits at [service limits](https://learn.microsoft.com/en-us/azure/search/search-limits-quotas-capacity).
- **Retrieval**: hybrid search with filters (product, language, version) to shrink the candidate set, semantic ranker on a bounded top-N, small final k (for example, 3 to 6 chunks).
- **Generation**: small, fast model for most questions; route hard ones to a larger model; cap output tokens; use prompt caching for the stable system prefix.
- **Caching**: cache embeddings of frequent queries and answers for FAQ-like traffic, invalidated on content version.
- **Ingestion**: incremental indexing with change detection, so the corpus never needs a full rebuild.

**Key trade-off:** each quality add-on (query rewrite, reranking, bigger model) costs latency; keep only the ones that move your evaluation metrics. If the corpus is dominated by one vector workload and cost is the issue, evaluate Cosmos DB DiskANN or PostgreSQL `pg_diskann` as alternatives (Q8).

**Monitor:** per-stage p95, time to first token, recall on the golden set, index fill percentage, replica CPU/throttling, and cache hit rate.

</details>

---

## Q18. Design a multi-region, highly available RAG service on Azure and say what fails over and how. `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** survive a regional outage with a defined RTO/RPO, keep data residency boundaries, and keep answers consistent enough.

**Design (active-active or active-passive):**

- **Entry**: Azure Front Door or Traffic Manager with health probes routing to regional app stacks.
- **App tier**: Container Apps (or equivalent) deployed identically in two paired or chosen regions via Bicep.
- **Azure AI Search**: no automatic cross-region index replica, so run a search service per region and populate both, either by running the ingestion pipeline to both targets or by pushing from a single pipeline. Replicas inside a region cover node failures and are required for the availability SLA (check the SLA page for counts).
- **Azure OpenAI / Foundry**: deployments in each region; an API Management or app-level gateway with priority and weighted routing and circuit breaker; Global or Data Zone deployment types can spread load within policy.
- **Cosmos DB**: multi-region replication with automatic failover (or multi-region writes for chat history).
- **Blob / OneLake**: geo-redundant storage for source documents.
- **Ingestion**: idempotent and replayable from the source of truth so a recovered region can be rebuilt.

**Key trade-off:** a second Search service roughly doubles index cost and introduces drift between regions; accept eventual consistency, measure index version lag, and decide whether a stale-but-available answer is acceptable. Active-passive is cheaper, with a slower and rehearsed failover.

**Monitor:** probe health per region, index document-count and version parity, cross-region failover events, 429 and 5xx per region, and run game-day failover tests, because untested failover does not work.

Reference: [Reliability in Azure AI Search](https://learn.microsoft.com/en-us/azure/reliability/reliability-azure-ai-search)

</details>

---

## Q19. A RAG app's monthly Azure bill doubled. Walk through how you would find and cut costs without hurting quality. `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Step 1: attribute spend.** Use Cost Management by resource, then token metrics per deployment and per feature from telemetry. Typical breakdown: model tokens (usually the largest and most variable), Search tier (fixed, driven by replicas and partitions), Document Intelligence pages, ingestion compute, private endpoints and logging.

**Step 2: apply levers, ordered by payoff and risk:**

| Lever | Detail | Risk |
|---|---|---|
| Model tiering | Route simple questions and rewrites to a small model, reserve the large one for hard cases | Misrouting hurts quality; evaluate the router |
| Context trimming | Fewer and smaller chunks, deduplicate, cap output tokens | Recall loss; verify with evals |
| Prompt caching and semantic caching | Reuse stable prefixes; cache FAQ answers | Staleness; invalidate on content version |
| Deployment type | Provisioned throughput (PTU) for steady baseline, pay-as-you-go for spikes; Batch for offline work | PTU is wasted if utilization is low |
| Search right-sizing | Replicas for actual QPS, quantization to shrink vector storage, right tier | Under-sizing causes throttling |
| Ingestion efficiency | Incremental indexing, hash-based skip, avoid re-OCR of unchanged files | Complexity |
| Logging | Sample prompts and completions, shorter retention | Less debugging data |

**Key trade-off:** every cut must pass the evaluation gate (Q14); otherwise you save money by silently degrading answers.

**Monitor:** cost per conversation, tokens per request, cache hit rate, PTU utilization, and quality scores on the same dashboard. Prices and SKUs change, so verify against [Azure pricing](https://azure.microsoft.com/en-us/pricing/) as of late 2026.

</details>

---

## Q20. When do you buy Microsoft 365 Copilot or Copilot Studio, and when do you build custom RAG on Azure? `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Scenario:** a business unit asks for "ChatGPT over our documents" and wants it in four weeks.

| Option | What it is | Best when |
|---|---|---|
| Microsoft 365 Copilot | Assistant inside M365 grounded on Graph content (SharePoint, mail, Teams) under existing permissions | Content lives in M365; users want it in Word, Teams, Outlook |
| Copilot Studio | Low-code agent builder; knowledge from SharePoint, public sites, connectors and, where supported, Azure AI Search | Departmental bots, Teams or web channel, low engineering capacity |
| Custom RAG on Azure (Foundry, AI Search, Azure OpenAI) | Full control over ingestion, retrieval, prompts, evaluation, UI, and hosting | Non-M365 data, specialized parsing, custom ranking, strict evaluation, embedding into your own product, tenant-specific data boundaries |

**Decision factors:** where the data lives, whether you need control over chunking, ranking and prompts, compliance requirements (some need your own network boundary and keys), per-user licensing versus consumption cost, required channels, and engineering capacity to own an evaluation and operations loop.

**Practical answer:** start with the managed product when it covers the requirement, and build custom only for a documented gap. Many organizations end up hybrid: Copilot Studio as the front end calling a custom retrieval API or an Azure AI Search knowledge source for the hard corpus.

**Monitor/decide on evidence:** pilot both against the same evaluation questions, compare answer quality, security behavior, and cost per active user. Check the current Copilot licensing and capability docs, as they change often.

</details>

---

## Q21. When should an Azure OpenAI RAG workload use provisioned throughput (PTU) instead of pay-as-you-go? `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Scenario:** a customer-facing assistant has a predictable daytime baseline, hard latency targets, and occasional spikes from marketing campaigns.

**Pay-as-you-go (Standard, Global, Data Zone):** billed per token, no commitment, shared capacity so latency and throughput vary and 429s are possible at quota.

**Provisioned (PTU):** you reserve model processing capacity, get more predictable latency and throughput, and pay for the reservation whether or not you use it. Reservations can reduce the hourly price if you commit.

**Decision approach:**

1. Measure sustained tokens per minute for input and output at p95 and convert to PTU using the Azure capacity calculator, not guesswork.
2. If baseline utilization would keep PTUs mostly busy, provision for the baseline only.
3. Send overflow to a pay-as-you-go deployment through a gateway with spillover or priority routing.
4. Keep batch and offline jobs on Batch deployments so they never consume interactive capacity.

**Trade-offs:** PTU is wasted when traffic is spiky or low; it needs capacity planning, model-version management, and regional availability checks. Pay-as-you-go is simple but exposes you to noisy-neighbor latency. Re-evaluate whenever model choice or prompt size changes, since both change the token profile.

**Monitor:** PTU utilization, 429 rate, time to first token, cost per 1,000 requests. Verify current offerings and rules at [provisioned throughput](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/provisioned-throughput) as of late 2026.

</details>

---

## Q22. A European bank must keep all customer data and prompts inside the EU. How do you design for data residency and compliance? `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Requirements:** storage, processing and logging of customer content stay in the EU; auditability; customer-controlled keys; no training on customer data.

**Design:**

- **Regions**: deploy every resource (Search, Cosmos DB, Storage, Document Intelligence, Content Safety, app hosting, Log Analytics) in EU regions.
- **Model deployment type**: Azure OpenAI offers Global, Data Zone and regional (Standard/Provisioned) deployments. Global deployments may process prompts in any region, so for strict residency choose a Data Zone (EU) or a regional deployment, and confirm with current docs what is stored versus processed where.
- **Abuse monitoring and logs**: understand the data, privacy and security documentation for Azure OpenAI, including retention of prompts for abuse monitoring and whether modified abuse monitoring is approved for your tenant.
- **Keys and access**: customer-managed keys in Key Vault or Managed HSM, private endpoints, managed identity, and Azure Policy to deny out-of-region resource creation.
- **Governance**: Microsoft Purview for classification, Defender for Cloud for posture, and an audit trail of retrieval and generation events.
- **Cross-region failover**: pair EU regions only; ensure the failover target is also in-boundary.

**Key trade-off:** residency constraints narrow model availability and may limit newest models or capacity, so test model availability in the EU zone before committing to a design.

**Monitor:** Azure Policy compliance, resources created outside allowed regions (should be zero), data flows in network logs, and key-vault access logs. Compliance claims (GDPR, DORA and others) are shared responsibility: Azure certifications do not make your application compliant by itself. Reference: [Data, privacy, and security for Azure OpenAI](https://learn.microsoft.com/en-us/legal/cognitive-services/openai/data-privacy)

</details>

---

## Q23. Which approach should a production RAG API on Azure use to authenticate to Azure AI Search and Azure OpenAI? `[Basic]` `[MCQ]`

- A. Admin API keys stored in app settings and rotated by hand
- B. A managed identity with narrowly scoped Azure RBAC roles on each resource
- C. A shared service principal secret kept in a team password manager
- D. Anonymous access restricted only by an IP allowlist on the services

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Correct: B.** A managed identity gets platform-issued, automatically rotated tokens, and RBAC lets you grant only what each component needs (for example, query-only on the index), with an Entra audit trail. Option A leaves long-lived, all-powerful secrets that can leak and be forgotten at rotation time. Option C still depends on a secret that humans copy around, and a shared principal blurs accountability across environments. Option D is not an authentication mechanism at all: IP allowlists limit where calls come from but anyone on that network, or any compromised host in the range, gets full access.

</details>

---

## Q24. Where must per-user document permissions be enforced so a restricted chunk can never reach the model prompt? `[Intermediate]` `[MCQ]`

- A. In the system prompt, instructing the model to ignore restricted material
- B. In the client app, by hiding citations the user is not allowed to open
- C. After generation, by scanning the answer for restricted document titles
- D. In the search query itself, using a filter built from the validated caller identity

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Correct: D.** Only a filter (or native token-based trimming) applied inside the retrieval call keeps unauthorized chunks from ever being candidates, so they cannot be ranked, summarized, or leaked. Option A is a soft constraint: the model has already seen the content, and prompt injection or a plain mistake can reveal it. Option B only hides the citation link while the answer text may still contain the restricted facts, and client code is untrusted anyway. Option C is a leaky heuristic: the content was already sent to the model, and a paraphrase does not contain the title, so scanning misses it.

</details>

---

## Q25. Azure OpenAI returns HTTP 429 with a `Retry-After` header during a traffic spike. What is the best first-line handling in the chat API? `[Intermediate]` `[MCQ]`

- A. Honor Retry-After with jittered backoff, bounded by the request deadline, and fail over to another deployment
- B. Retry immediately in a tight loop until the call succeeds, so users never see an error
- C. Increase the client timeout to several minutes so the platform queues the request
- D. Switch the whole app to a larger model so the quota limit no longer applies

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Correct: A.** A 429 is an explicit instruction to slow down; respecting Retry-After with jitter avoids synchronized retries, the deadline keeps the user experience bounded, and routing to a second deployment or region uses spare capacity. Option B creates a retry storm that deepens throttling for everyone and burns your latency budget. Option C hides the problem: the platform does not queue on your behalf, and users would stare at a spinner. Option D is wrong because quota is assigned per deployment and model, and larger models typically consume more tokens per request, which makes limits easier to hit.

</details>

---

## Q26. A SaaS vendor serves many thousands of small tenants with identical schemas on Azure AI Search. Which multi-tenancy pattern is the usual best starting point? `[Advanced]` `[MCQ]`

- A. One search service per tenant, to guarantee physical isolation for everyone
- B. One index per tenant, so each tenant has its own relevance statistics
- C. One shared index with a filterable tenant field enforced by the server-side retrieval layer
- D. One shared index with no tenant field, relying on the prompt to scope answers

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Correct: C.** With many small, same-schema tenants, a pooled index with a mandatory server-injected tenant filter keeps cost and operations manageable; add leakage tests and move large or regulated tenants to dedicated indexes or services. Option A offers the strongest isolation but multiplies fixed service cost per tenant and is not viable at this tenant count. Option B runs into per-service index limits for your tier and wastes capacity on tiny indexes, though it fits a smaller number of larger tenants. Option D has no data boundary at all, so one query can retrieve another tenant's content and the prompt cannot be trusted to prevent it.

</details>

---
