# 01 — Azure AI Search for RAG

> Azure AI Search as the retrieval layer of a RAG system: index design, vector/BM25/hybrid retrieval, semantic ranking, integrated vectorization, security trimming, scaling, and how it compares with open-source alternatives.

---

> **Accuracy note:** Product details (API versions, preview vs GA status, tier limits, prices) change often. Statements below were checked against Microsoft Learn in late 2026; treat specific names and numbers as "check current docs" and confirm on the linked pages before relying on them.

## Q1. What role does Azure AI Search play in an Azure-based RAG architecture? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Azure AI Search is the retrieval tier: it stores your chunked content (text, metadata, and embeddings) in an index and returns the best-matching chunks for a query. The LLM (typically Azure OpenAI) never talks to the index directly; your application, or an orchestration layer, calls the search service, takes the top results, and builds the grounded prompt.

```
Sources (Blob, SharePoint, SQL, Cosmos DB)
   │  indexer + skillset (crack, chunk, embed)
   ▼
Azure AI Search index  ◄── query: keyword + vector (+ semantic rerank) + filters
   │ top chunks + citations
   ▼
App / orchestrator ──► Azure OpenAI (generate grounded answer)
```

What it adds over a bare vector store: full-text BM25, hybrid fusion, a semantic reranker, filters/facets, indexers that pull from data sources, and enterprise security (Entra ID, RBAC, private endpoints). See the [RAG overview on Microsoft Learn](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview).

</details>

---

## Q2. Which query modes does Azure AI Search support for retrieval, and when would you pick each? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Mode | How it matches | Good at | Weak at |
|---|---|---|---|
| Full-text (BM25) | Tokenized terms, analyzers, inverted index | Exact terms, IDs, product codes, rare names | Paraphrases, synonyms, cross-language |
| Vector | Nearest neighbors in embedding space | Meaning-based and paraphrased queries | Exact identifiers, numbers, new jargon the embedder never saw |
| Hybrid | BM25 and vector run in parallel, fused with RRF | Most real RAG workloads | Slightly more tuning surface |
| Hybrid + semantic ranker | Hybrid, then a language-model L2 rerank of the top results | Best relevance on prose-heavy content | Extra latency and per-query billing |

Microsoft's own benchmark guidance is that hybrid plus semantic ranking gives the most relevant results ([vector relevance and ranking](https://learn.microsoft.com/en-us/azure/search/vector-search-ranking)); in an interview, say "start with hybrid + semantic ranker, then measure" and justify any simpler choice with data.

</details>

---

## Q3. What do the field attributes `searchable`, `filterable`, `facetable`, `sortable`, and `retrievable` control, and which matter most in a RAG index? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Each attribute decides which on-disk structures are built for a field, so they trade storage and indexing cost for capability.

| Attribute | Effect | RAG note |
|---|---|---|
| `searchable` | Included in full-text analysis (text) or vector search (vector fields) | Chunk text and the vector field must be searchable |
| `filterable` | Usable in OData `$filter` expressions | Needed for tenant, doc type, date, ACL group fields |
| `facetable` | Usable for facet counts | Useful for "narrow by source" UIs, rarely for the LLM path |
| `sortable` | Usable in `$orderby` | Rarely needed in RAG |
| `retrievable` | Returned in results | Set `false` on large vector fields to cut response size |

Practical tips: only mark fields you will actually filter or facet on, because each flag adds index overhead; keep an ID/parent key and a source URL field `retrievable` for citations; and remember a field with `retrievable: false` is not a security control. Reference: [create a search index](https://learn.microsoft.com/en-us/azure/search/search-how-to-create-search-index).

</details>

---

## Q4. What is the difference between HNSW and exhaustive KNN in Azure AI Search, and how do you choose? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

- **Exhaustive KNN** scans every vector and returns the exact nearest neighbors. It builds no graph, so it does not consume vector index size quota, but cost grows linearly with corpus size.
- **HNSW** builds a hierarchical navigable small world graph and does approximate nearest neighbor search: high recall, low latency, but the graph must sit in memory and counts against vector index size quota.

Tunable HNSW parameters are `m` (neighbors per node), `efConstruction` (index-time candidate list), `efSearch` (query-time candidate list) and the `metric` (`cosine`, `dotProduct`, `euclidean`). Docs note a default `efConstruction` of 400 with a range of 100 to 1,000.

Choose HNSW for most production corpora. Choose exhaustive KNN for small or medium datasets, when exactness matters more than speed, or to build ground truth when measuring HNSW recall. A field indexed for HNSW can still be queried exhaustively with `"exhaustive": true`; the reverse is not possible. Source: [vector relevance and ranking](https://learn.microsoft.com/en-us/azure/search/vector-search-ranking).

</details>

---

## Q5. What is the semantic ranker, and what can it not do? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The semantic ranker is a Microsoft-hosted language-model reranker (adapted from Bing) applied after initial retrieval. It takes the BM25 or RRF result set, re-scores it by meaning, and can return extractive captions (highlighted passages) and optionally answers. Scores in `@search.rerankerScore` range from 0 to 4.

Limits to know:

- It only reranks the top results of the first-stage ranking (docs say the top 50), so it cannot rescue a document that first-stage retrieval missed.
- Each document is summarized to a bounded input before scoring, so long chunks are truncated and field priority order in the semantic configuration matters.
- Captions and answers are verbatim text from your index; nothing is generated.
- It is a premium feature billed by usage, with a free monthly allowance.

Source: [semantic ranking overview](https://learn.microsoft.com/en-us/azure/search/semantic-search-overview).

</details>

---

## Q6. How do replicas and partitions differ in Azure AI Search, and what does each one scale? `[Basic]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Unit | What it adds | Use it for |
|---|---|---|
| Partition | Storage and indexing capacity; the index is sharded across partitions | Bigger corpora, more vectors, faster indexing |
| Replica | A full copy of the index that serves queries | Higher QPS, lower latency under load, high availability |

Billing in the dedicated model is by search units, replicas multiplied by partitions, so cost scales roughly linearly as you add either. A replica cannot help if you have run out of storage or vector quota; a partition will not raise QPS much. For availability SLAs you need multiple replicas (check the current SLA page for exact counts).

Tiers (Free, Basic, S1 to S3, S3 HD, L1/L2 storage-optimized) set per-partition size and object limits. As of late 2026 there is also a serverless pricing model in preview that removes replica/partition management; check current docs before depending on it. Source: [choose a tier](https://learn.microsoft.com/en-us/azure/search/search-sku-tier) and [capacity planning](https://learn.microsoft.com/en-us/azure/search/search-capacity-planning).

</details>

---

## Q7. How does hybrid search with Reciprocal Rank Fusion (RRF) work in Azure AI Search? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A hybrid request carries a `search` text string and one or more `vectorQueries`. The engine executes each as a separate query in parallel, producing independent ranked lists, then fuses them with RRF.

For every document in every list: `score = 1 / (rank + k)` with a small constant `k` (the docs cite 60). Scores for the same document are summed across lists and the fused list is sorted. Because RRF uses rank positions rather than raw scores, you never have to normalize BM25 (unbounded) against cosine-derived scores (about 0.33 to 1).

```json
{
  "search": "reset MFA for contractors",
  "vectorQueries": [
    { "kind": "text", "text": "reset MFA for contractors",
      "fields": "contentVector", "k": 50, "weight": 2.0 }
  ],
  "select": "id, title, chunk, url",
  "top": 10
}
```

Gotchas: the RRF constant `k` is unrelated to the `k` of nearest neighbors; `weight` on a vector query scales that list's contribution; a text query plus two vector queries over five fields is 11 parallel executions. The `kind: text` query relies on a vectorizer configured on the field. Source: [hybrid search scoring (RRF)](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking).

</details>

---

## Q8. What is integrated vectorization, and which components make it up? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Integrated vectorization moves chunking and embedding inside the indexing pipeline, and embedding of query text inside the query pipeline, so you do not run a separate ETL job.

**Indexing side:** a data source, an **indexer** (pulls, cracks documents, tracks changes), a **skillset** (a chunking skill such as Text Split, then an embedding skill such as the Azure OpenAI Embedding skill), and an index that receives the chunk rows (usually via index projections).

**Query side:** a **vectorizer** defined in the index and attached to a vector profile, so a `kind: text` vector query is embedded with the same model automatically.

```
Blob/SharePoint ─► indexer ─► skillset[ split ─► embed ] ─► index (chunks + vectors)
query text ─► vectorizer (same model) ─► vector query
```

Trade-offs: much less glue code and built-in change tracking, but less control over chunking logic, retries and throttling (retry/batching is built in and not configurable), and you share Azure OpenAI tokens-per-minute quota between indexing and queries. For custom parsing or exotic chunkers, many teams run their own pipeline and push documents. Source: [integrated vectorization](https://learn.microsoft.com/en-us/azure/search/vector-search-integrated-vectorization).

</details>

---

## Q9. How do you implement parent/child (small-to-big) chunking in Azure AI Search? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Goal: match on small, precise chunks but give the LLM the surrounding context.

Two common patterns:

1. **Index projections (one-to-many).** The skillset splits each document into chunks and projects each chunk as its own row into a chunk index, with a `parent_id` field linking back to the source. Optionally a second index holds parent-level metadata. When a chunk matches, you look up the parent (or neighbors) by key.
2. **Single index with context fields.** Store `chunk`, `chunk_order`, `parent_id` and perhaps `prev`/`next` text or a larger `parent_text` field (not searchable, retrievable). After retrieval, fetch neighbors with a filter like `parent_id eq 'x' and chunk_order ge 4 and chunk_order le 6`.

Design notes: store a stable key per chunk, keep `parent_id` filterable, and project ACL fields to every chunk row or security trimming silently fails. Chunk size guidance is corpus-specific: measure retrieval recall at a few sizes and overlaps rather than copying a default. See [chunk large documents](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents) and [index projections](https://learn.microsoft.com/en-us/azure/search/index-projections-concept-intro).

</details>

---

## Q10. What vector compression options exist in Azure AI Search, and how do oversampling and rescoring offset their quality loss? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Option | What it does | Docs-stated effect |
|---|---|---|
| Scalar quantization (int8) | Maps each float to an 8-bit bucket | About 4x smaller vector index |
| Binary quantization | One bit per dimension | Up to about 28x smaller index; works best on higher dimensions (1024+) and zero-centered embeddings |
| Truncated dimensions | Use fewer dimensions from Matryoshka-trained models such as text-embedding-3 | Smaller vectors; model must support it |
| Float16 storage (`Edm.Half`) | Half-precision fields | Smaller than float32 |
| Discard originals | Drop the full-precision copy kept for rescoring | Smaller storage, but no full-precision rescoring |

**Rescoring** re-ranks the compressed-search candidates using original (or, for binary, the binary dot product) vectors. **Oversampling** controls how many extra candidates are pulled before rescoring: effective candidates is roughly `k` times the oversampling factor, and only the top `k` are returned. Only HNSW supports it; exhaustive KNN scans everything anyway.

Process: create a new vector profile and field (you cannot change compression on an existing field), then measure recall/NDCG with and without oversampling on your own queries. Source: [vector quantization](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-quantization).

</details>

---

## Q11. Explain the main query-time knobs for vector and hybrid retrieval (`k`, `top`, `weight`, `oversampling`, `exhaustive`) and how they interact. `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Parameter | Where | Meaning |
|---|---|---|
| `k` | Each vector query | Number of nearest neighbors that vector query contributes to fusion |
| `top` | Request | Number of results returned from the final (fused) list |
| `weight` | Each vector query | Multiplier on that query's RRF contribution; default 1.0 |
| `oversampling` | Vector query on compressed fields | Candidate multiplier for rescoring |
| `exhaustive` | Vector query | Force brute-force KNN over an HNSW field, useful for recall tests |
| `efSearch` | Index (HNSW parameters) | Query-time candidate list size; higher is more accurate and slower |
| `filter` / `vectorFilterMode` | Request | Which filter strategy runs relative to vector search (pre- or post-filtering; check docs for the modes and defaults) |

Interaction rules of thumb: set `k` at least as large as what you want after fusion (for example 30 to 50) so the vector side can compete with BM25; use `top` for the number of chunks sent to the LLM; raise `efSearch` or use oversampling only after you measure a recall gap against exhaustive results. Do not confuse `k` with the RRF constant. A relevance threshold on vector or hybrid queries exists as a preview feature at the time of writing. Source: [vector query how-to](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-query).

</details>

---

## Q12. Where do scoring profiles fit in a RAG index, and where don't they help? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

A scoring profile adjusts the BM25 relevance score of text queries using field weights and functions (freshness, magnitude, distance, tag). That makes it good for boosting recent documents, authoritative sources, or title matches in the keyword portion of retrieval.

The Learn relevance page states that vector queries do not have scoring profile or term-boosting features. So in a hybrid query, a profile influences only the text ranking, and RRF then fuses it with vector ranks, which dilutes its effect. A boost you need to be strong should be applied differently:

- Filter instead of boost for hard rules ("only current policy versions").
- Add a recency or authority signal after retrieval in your own re-rank step, or via semantic configuration field choice.
- Use vector `weight` to emphasize semantic versus keyword evidence.

Interview point: treat scoring profiles as a keyword-side tool and verify with debug subscores whether they actually change final order. Source: [vector relevance and ranking](https://learn.microsoft.com/en-us/azure/search/vector-search-ranking) and [relevance overview](https://learn.microsoft.com/en-us/azure/search/search-relevance-overview).

</details>

---

## Q13. How do metadata filters interact with vector search, and what can go wrong? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Filters use OData expressions over `filterable` fields (`tenant eq 'a' and year ge 2024`). With vectors there are two strategies:

- **Pre-filter:** restrict the candidate set first, then do nearest-neighbor search among matches. You get `k` results that all satisfy the filter, which is what RAG usually wants.
- **Post-filter:** find nearest neighbors first, then filter. If the filter is selective, you can end up with fewer than `k` results, or none.

Azure AI Search exposes this through a vector filter mode setting (pre/post variants; confirm names and defaults in the [filters in vector queries](https://learn.microsoft.com/en-us/azure/search/vector-search-filters) article).

Failure modes to mention:

1. A very selective pre-filter on HNSW can reduce recall or latency predictability, so benchmark selective filters (per-tenant, per-user) explicitly.
2. Filters on non-`filterable` fields fail at query time.
3. High-cardinality `Collection(Edm.String)` fields used for ACLs are fine with `search.in`, but giant OR chains are slow.
4. Compare with Qdrant or Weaviate, which advertise filter-aware HNSW traversal; always test your own filter selectivity rather than assume parity.

</details>

---

## Q14. How do you debug and evaluate retrieval quality in Azure AI Search? `[Intermediate]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Evaluate retrieval separately from generation.

1. **Golden set:** 50 to 200 real queries with labeled relevant chunk or document IDs (mine from logs, have SMEs label, or bootstrap with an LLM and spot check).
2. **Metrics per configuration:** recall@k, MRR, NDCG@10 for each variant: BM25 only, vector only, hybrid, hybrid + semantic ranker, different chunk sizes, with and without quantization.
3. **Ground truth for ANN:** run the same vector query with `"exhaustive": true` and compare to HNSW results to measure recall loss from HNSW or compression.
4. **Per-query diagnosis:** pass the `debug` parameter (for example `vector`, `semantic`, or `all`) to see subscores and understand whether the text or vector side surfaced a document ([hybrid scoring docs](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)).
5. **End-to-end:** add faithfulness and answer relevance scoring (Azure AI Foundry evaluators, RAGAS, or similar) once retrieval is stable.
6. **Production signals:** log query, filters, returned IDs, reranker scores and thumbs up/down; alert on queries with a low top reranker score as a proxy for corpus gaps.

Change one variable at a time and keep the golden set fixed so regressions are attributable.

</details>

---

## Q15. What are the options for document-level security trimming in Azure AI Search, and how do you choose among them? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Microsoft Learn describes four approaches ([document-level access control](https://learn.microsoft.com/en-us/azure/search/search-document-level-access-overview)):

| Approach | Mechanism | Status noted in docs (late 2026) |
|---|---|---|
| Security filters | A `Collection(Edm.String)` field holds group/user IDs; app adds `group_ids/any(g: search.in(g, '...'))` to every query | Generally available, API-agnostic |
| Native ACL / RBAC scopes | Permission metadata indexed per document; caller's Entra token sent in `x-ms-query-source-authorization` header; engine trims | Preview |
| SharePoint ACLs | Indexer ingests SharePoint permissions, same token-based trimming | Preview |
| Purview sensitivity labels | Labels ingested and enforced against the caller's token | Preview |

Choosing: for custom identity systems or anything you need in production today, use security filters, with the group field `filterable` and `retrievable: false`. Use native trimming when the source is ADLS Gen2, SharePoint, or Purview-governed content and preview terms are acceptable.

Pitfalls: (1) the filter is only as trustworthy as your app; resolve group membership server-side from a validated token, never from client input. (2) Chunked indexes must carry the ACL on every chunk (index projections). (3) Permission changes in the source only appear after the next sync/indexer run, so there is an exposure window. (4) `retrievable: false` hides the field but is not an access control. Also consider separate indexes per tenant only when isolation requirements justify the object-count limits.

</details>

---

## Q16. A law firm wants a RAG assistant over matter documents in SharePoint, where each attorney may only see matters they are staffed on, and staffing changes daily. How do you design retrieval and security? `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Constraint:** ethical walls make a leak a compliance incident, and permissions change faster than a nightly crawl.

**Design:**

1. **Ingestion:** SharePoint indexer (or a custom pipeline using Microsoft Graph) into a chunk index through a skillset (Text Split + embedding). Project matter ID and permission fields onto every chunk.
2. **Authorization choice:** evaluate the SharePoint ACL ingestion preview, which uses the caller's Entra token. If preview status is unacceptable, use the security filter pattern: store the staffing group/matter IDs in a `filterable`, non-retrievable collection field and filter every query with `search.in`.
3. **Freshness of permissions:** do not rely on the nightly full indexer for access changes. Make staffing the source of truth in a small table; resolve the attorney's current matter list at query time and pass it as the filter. Then adding or removing a person takes effect immediately, with no re-index. Reserve re-indexing for content changes.
4. **Query path:** hybrid + semantic ranker, pre-filtered by matter list; log the filter applied for audit.
5. **Verification:** automated tests that run attorney A's queries and assert no chunk with matter outside A's list ever returns, including adversarial prompts and queries crafted to match another matter.
6. **Defense in depth:** private endpoints, managed identity, Entra RBAC (`Search Index Data Reader`) for the app, no direct user access to the index.

**Trade-off:** query-time matter lists can be long; `search.in` handles hundreds to thousands of values with sub-second filters per the docs, but if lists grow very large, model access as a smaller set of group IDs instead.

</details>

---

## Q17. Users complain that a support-docs RAG bot fails on exact error codes like `0x80070005` and part numbers, though paraphrased questions work well. The index is vector-only. Diagnose and fix. `[Intermediate]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Diagnosis:** dense embeddings capture meaning but tokenize opaque identifiers into subword pieces, so `0x80070005` is "close" to many similar codes. This is a lexical-match failure, not a model-quality failure. Confirm by running the failing queries as pure BM25 and checking that the right chunk ranks first; use `debug` subscores to see the two sides.

**Fix:**

1. Move to **hybrid** retrieval: keep the vector field, add `search` text over a `searchable` chunk field so BM25 catches the literal token. RRF merges both lists.
2. Check the **analyzer** on the text field. The standard analyzer may split `0x80070005` or part numbers with hyphens oddly; consider a custom analyzer or a keyword-style field for IDs, and mirror the ID into a separate `searchable` field you can boost with a scoring profile or `searchFields`.
3. Add the **semantic ranker** over the fused top results so a paraphrase-friendly model orders the final list.
4. Evaluate with a golden set containing both exact-code and paraphrase queries; tune vector `weight` if one side dominates.
5. Optionally normalize queries (extract code patterns with a regex and issue a filter/exact lookup first).

**Trade-off:** hybrid adds a text index and slightly more latency but is the standard remedy; do not fine-tune embeddings first.

</details>

---

## Q18. What is agentic retrieval in Azure AI Search, and when would you use it instead of a classic single-query RAG call? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

Agentic retrieval is a multi-query pipeline exposed through a **knowledge base** that references one or more **knowledge sources** (indexed content on your service or remote sources). Per the Learn overview, at low or medium reasoning effort an LLM plans focused subqueries from the question and chat history; the subqueries run in parallel (keyword, vector, or hybrid), each is semantically reranked, and results are merged into one grounding response with optional references and an activity log. At minimal effort the planning step is skipped.

**Use it when:** questions have several asks, depend on conversation context, contain typos, or span several sources.
**Stay classic when:** queries are simple single-fact lookups, latency is tight, or you want full control and predictable cost.

Cost and status notes (verify in current docs): classic semantic ranking is billed per query, agentic retrieval is billed in tokens plus Azure OpenAI tokens for planning; LLM-based planning and answer synthesis were preview features and tied to preview REST versions, while minimal-effort retrieval with GA knowledge source types was available on the stable API. Foundry IQ builds on the same knowledge-base concept. Source: [agentic retrieval overview](https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview).

Compare: this resembles query decomposition in LangChain/LlamaIndex, but hosted and with built-in reranking.

</details>

---

## Q19. How does Azure AI Search compare with pgvector, Elasticsearch/OpenSearch, Qdrant and Weaviate for RAG retrieval? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Option | Strengths | Watch-outs |
|---|---|---|
| Azure AI Search | Managed; hybrid + semantic ranker built in; indexers and skillsets; Entra/RBAC, private link; native Microsoft source connectors | Azure lock-in; tier and quota planning; semantic ranker is proprietary and billed; fewer low-level index knobs |
| pgvector (Postgres) | Vectors beside relational data and transactions; simple ops if you already run Postgres; SQL joins and filters | Hybrid needs `tsvector` plus your own fusion; scaling HNSW/IVF on very large corpora needs care |
| Elasticsearch / OpenSearch | Mature BM25, aggregations, hybrid/RRF support, large ecosystems, self-host or managed | Cluster ops and tuning; reranking models are separate add-ons |
| Qdrant | Fast purpose-built vector engine, rich payload filtering, quantization options, self-host or cloud | You build ingestion, text search depth and reranking around it |
| Weaviate | Vector DB with built-in hybrid (BM25 + vector), modules for vectorization, multi-tenancy features | Operational model and cost differ from managed Azure services |

How to decide: if the stack is already Azure, security/compliance matters, and you want fewer moving parts, Azure AI Search is usually the shortest path. Pick pgvector when the data already lives in Postgres and corpus size is moderate. Pick Elasticsearch/OpenSearch when you need heavy full-text analytics. Pick Qdrant or Weaviate for portability, cost control, or advanced filtering at scale. None of these is universally best: run your golden set against two candidates before committing. Capabilities change quickly, so check each project's current docs.

</details>

---

## Q20. How do you estimate capacity and cost for an index of 50 million 1536-dimension chunks on Azure AI Search? `[Advanced]` `[Scenario]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Step 1: raw vector math.** 1536 float32 dimensions is 6,144 bytes per vector. 50M vectors is about 307 GB before the HNSW graph, text index, and metadata. Quantization shrinks it: int8 scalar is about 4x (roughly 77 GB of vectors), binary far more, at some recall cost. Truncating dimensions on a Matryoshka-capable embedding model shrinks it further.

**Step 2: what counts against quota.** HNSW graphs live in memory and consume the *vector index size* quota; exhaustive KNN fields do not, but then you pay in query latency. Keeping full-precision originals for rescoring adds storage back. Look up the current vector quota per partition for your tier in [service limits](https://learn.microsoft.com/en-us/azure/search/search-limits-quotas-capacity) and [vector index size](https://learn.microsoft.com/en-us/azure/search/vector-search-index-size); do not rely on remembered numbers.

**Step 3: pick a tier and partitions.** Candidates are S-series for balanced latency or storage-optimized L-series for large, rarely updated indexes (higher query latency). Partitions = storage need divided by per-partition vector quota, with headroom for updates.

**Step 4: replicas** from peak QPS and HA requirements; load test, since latency depends on filters, hybrid, and semantic ranker.

**Step 5: other cost lines.** One-time embedding of 50M chunks (Azure OpenAI tokens), semantic ranker per-query billing, agentic retrieval tokens if used, and re-embedding cost if you change models.

**Levers:** quantization with oversampling, `retrievable: false` on vectors, fewer filterable/facetable fields, dedupe chunks, tier choice. Compare the total against self-hosted Qdrant or OpenSearch including ops time, not just infrastructure.

</details>

---

## Q21. What operational problems appear when ingesting large corpora with indexers, skillsets and embedding skills, and how do you handle them? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

| Problem | Why it happens | Mitigation |
|---|---|---|
| Throttled embeddings | Azure OpenAI tokens-per-minute quota is per model and subscription, shared by indexing and query vectorization | Separate deployments for indexing vs query, ideally separate subscriptions; run indexers on a schedule so throttled documents are retried |
| Deletions not reflected | Indexers need a deletion-detection policy (soft-delete column or blob metadata) | Configure change and deletion detection on the data source |
| Orphaned chunks | One source doc becomes many rows | Use index projections with a parent key and cascade deletes through the skillset pipeline |
| Model change | Vectors from different models are not comparable | Create a new vector field or index, re-embed, switch via index alias; never mix models in one field |
| Poison documents | Unparseable or oversized files fail the run | Set failure tolerance, review indexer status/errors, route bad files aside |
| Long runtimes | Large corpora on low tiers | Scale partitions during backfill, parallelize with multiple indexers, then scale back |
| Cost surprises | AI enrichment (OCR, extra skills) bills separately | Estimate per-document skill cost before the backfill |

Treat ingestion like any ETL system: idempotent keys, monitoring of success counts vs source counts, and a replayable path. Source: [integrated vectorization limitations](https://learn.microsoft.com/en-us/azure/search/vector-search-integrated-vectorization) and [indexer overview](https://learn.microsoft.com/en-us/azure/search/search-indexer-overview).

</details>

---

## Q22. Your hybrid + semantic ranker setup returns good top-3 results but the LLM keeps missing facts that live deep in long chunks. How do the semantic ranker's input limits and your chunk size interact? `[Advanced]`

<details>
<summary>💡 Show Answer</summary>

**Answer:**

The semantic ranker does not read your whole chunk. Per the Learn overview, only the top 50 first-stage results are reranked, and each document is summarized to a bounded input (up to 2,048 tokens for the generated summary string, built from title, keywords and content fields in the semantic configuration, with the title and keyword fields themselves capped at 128 tokens). Anything beyond the limit is trimmed, and field order in the configuration sets trimming priority.

Consequences:

- A chunk much longer than the summarization budget can be ranked on a partial view, so a fact near the tail may not influence its reranker score.
- Very large chunks dilute embeddings too, so vector recall also suffers.
- Captions are extractive snippets for display; they are not guaranteed to contain the fact the LLM needs, so do not send only captions to the model.

Fixes: use smaller chunks (a few hundred tokens up to roughly a page, tested on your corpus) with overlap; put the most descriptive field first in the semantic configuration; use parent/child retrieval so the ranker scores precise chunks while the LLM gets expanded context (Q9); raise `top` or `k` so more candidates reach the LLM; and evaluate with a golden set that includes "fact deep in the document" cases. Limits and token budgets can change, so re-check [semantic ranking overview](https://learn.microsoft.com/en-us/azure/search/semantic-search-overview).

</details>

---

## Q23. In an Azure AI Search hybrid query, how does Reciprocal Rank Fusion combine the BM25 result list with the vector result list? `[Intermediate]` `[MCQ]`

- A. It normalizes BM25 and cosine scores to 0-1 and adds them with equal weights
- B. It sums reciprocal-rank scores of the form 1/(rank + k) from each ranked list
- C. It keeps only documents appearing in both lists and averages their raw scores
- D. It trains a small model per index to learn the best score blend

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Correct: B.** RRF ignores raw scores and uses each document's position in each ranked list, adding `1/(rank + k)` contributions (docs cite k around 60) so documents ranked well in multiple lists rise. Option A describes score normalization, which RRF is designed to avoid because BM25 is unbounded while vector scores are not. Option C is an intersection strategy; RRF keeps documents that appear in only one list, just with a lower combined score. Option D is wrong because RRF is a fixed formula with no per-index training; the only tuning knobs are things like vector query weights.

</details>

---

## Q24. Which statement about the Azure AI Search semantic ranker is correct? `[Basic]` `[MCQ]`

- A. It scans the entire corpus with a language model to find semantically similar documents
- B. It generates new answer text that is not present in your indexed content
- C. It replaces the need for BM25 or vector retrieval as the first stage
- D. It reranks only the top first-stage results and extracts captions from existing text

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Correct: D.** The semantic ranker is a second-stage (L2) reranker over the BM25 or RRF results (docs state the top 50), and its captions and answers are extracted verbatim from indexed content. Option A is wrong because it cannot rerun the query over the whole corpus; recall is bounded by first-stage retrieval. Option B is wrong because there is no generative model in this step, so it never composes new text. Option C is wrong because it depends on first-stage retrieval to supply candidates and cannot replace it.

</details>

---

## Q25. You add a field to store the Entra group IDs allowed to read each chunk for security-filter trimming. Which field definition is most appropriate? `[Intermediate]` `[MCQ]`

- A. `Collection(Edm.String)`, filterable, not retrievable, populated on every chunk row
- B. `Edm.String` holding a comma-separated list, searchable, retrievable for debugging
- C. `Collection(Edm.String)`, facetable only, populated on the parent document
- D. `Edm.String`, sortable, populated once per source file in a separate index

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Correct: A.** The documented security-filter pattern uses a string collection that is `filterable` so queries can apply `group_ids/any(g: search.in(g, '...'))`, with `retrievable` set to false so the ACL list is not echoed in results. It must be present on every chunk row, otherwise chunks without it are not correctly trimmed. Option B makes access depend on full-text matching of a delimited string and exposes group lists in responses. Option C is wrong because facetable does not enable filtering, and trimming only the parent document leaves chunk rows unprotected. Option D is wrong because sortable has nothing to do with filtering and a separate per-file index does not constrain chunk results.

</details>

---

## Q26. A team compresses a 1536-dimension vector field with scalar quantization and sets `rescoreStorageMethod` to `discardOriginals`, then finds oversampling has no effect. What is the most likely reason? `[Advanced]` `[Scenario]` `[MCQ]`

- A. Oversampling works only with exhaustive KNN fields, not HNSW
- B. Oversampling is available only on the free tier of the service
- C. Scalar rescoring needs the full-precision vectors, which were discarded
- D. Scalar quantization cannot be combined with any query-time parameter

<details>
<summary>💡 Show Answer</summary>

**Answer:**

**Correct: C.** Per Microsoft Learn, oversampling for scalar-quantized vectors requires the original full-precision vectors to rescore candidates; with `discardOriginals` they are gone, so rescoring and oversampling cannot be used. (Binary quantization is the exception, since it can rescore using the binary dot product.) Option A is backwards: only HNSW supports rescoring, because exhaustive KNN scans everything anyway. Option B is wrong because oversampling is not tied to a pricing tier. Option D is wrong because `oversampling` is an ordinary query parameter on vector queries against compressed fields when the original vectors are kept.

</details>

---
